# Sampler-mean score baseline for REINFORCE under sampler mismatch

## What is wrong with the current loss

`loss_fn_rl` with `correction=None` is REINFORCE on the sampled token:

    L = -(1/N) sum_{b,t} A_b * log p(y_t | h_t),     y_t ~ q(. | h_t),

with h_t the prefix, p the trainer, q the sampler, and A_b the group-centred advantage.
Its gradient is -(1/N) sum A_b s_{y_t}, where s_v(h) = grad_theta log p(v|h).

Fix a prefix h and average over the next token y ~ q(.|h). Write A for the advantage of the
completion. Then

    E_q[A s_y | h] = E_q[A | h] s_bar(h) + Cov_q(A, s_y | h),
    s_bar(h) = sum_v q(v|h) s_v(h).

(The identity holds because E_q[s_y | h] = s_bar(h), so Cov_q(A, s_y | h) = E_q[A s_y | h] - E_q[A | h] s_bar(h).)

- On-policy (q = p): s_bar = sum_v p_v grad log p_v = grad sum_v p_v = 0. Only the covariance,
  which carries the reward signal, remains.
- Mismatched (q != p): s_bar != 0 in general. The term E_q[A|h] s_bar(h) is a drift. Its
  direction is set by the sampler's distribution, not by the reward. Its size is proportional
  to the expected advantage at that prefix. Groups with a positive mean advantage push the
  trainer toward whatever q over-weights, and negative groups push it away. The sampler is
  refreshed from the trainer at every step, so the bias feeds back into the next step's q. That
  is why the drift compounds rather than averaging out to zero.

Importance weights do not remove this term. A weight w(p/q) rescales the sampled-token score
and changes the magnitude of the drift, and token-level clipping truncates it, but the term is
not zero in expectation. This matches the observation that clipping slows the collapse and
that stronger clipping can make it worse.

## The fix: subtract the sampler-mean score

Replace each token's score s_y by s_y - s_bar(h), keeping the same advantage A:

    E_q[A (s_y - s_bar) | h] = E_q[A s_y | h] - E_q[A | h] s_bar(h) = Cov_q(A, s_y | h).

The drift is gone, and no importance ratio is used. On-policy, s_bar = 0, so the change is exactly
zero. The modified estimator is therefore unbiased for the covariance term, which is the only
part of the REINFORCE update that carries reward information.

Implementation, with sg = stop-gradient, per position t:

    loss_t = -A * ( log p(y_t|h_t) - sum_v sg[q(v|h_t)] log p(v|h_t) ),

whose gradient is -A (s_{y_t} - s_bar(h_t)). The sampled token keeps its own log p even when it
lies outside the sampler's head. Its expectation over y ~ q is exactly the covariance above.

### Only the top-k head of q is available

The sampler returns the top-K = 20 logprobs of q at each position (`sampler_topk_ids`,
`sampler_topk_logprobs`). Let H be that head, and let the tail of q be proportional to p there:

    q_v = rho p_v for v not in H,    rho = q_tail / p_tail,
    q_tail = 1 - sum_{v in H} q_v,   p_tail = 1 - sum_{v in H} p_v.

Since sum_v p_v grad log p_v = 0, the tail part of s_bar reduces to a head expression:

    s_bar = sum_v q_v s_v = sum_{v in H} q_v s_v + rho sum_{v not in H} p_v s_v
          = sum_{v in H} q_v s_v - rho sum_{v in H} p_v s_v
          = sum_{v in H} (q_v - rho p_v) s_v.

So the surrogate uses head coefficients c_v = sg[q_v - rho p_v] on log p_v for v in H:

    loss_t = -A * ( log p(y_t|h_t) - sum_{v in H} sg[q_v - rho p_v] log p(v|h_t) ).

The tail contributes no gradient beyond what the head already encodes. Because p_v <= p_tail,
rho p_v <= q_tail <= 1, so the coefficients stay bounded even when p_tail is small.

Checks (done in a scratch test, not a deliverable):
- With q exactly equal to a tail-proportional distribution, the surrogate gradient matches
  grad sum_v q_v log p_v to float64 precision (~1e-8).
- With q = p, all coefficients c_v are zero, so the baseline term has exactly zero gradient.

When the tail of q is not proportional to p, the residual error is
sum_{v not in H} (q_v - rho p_v) s_v. It is small when the head covers most of both
distributions, which is the regime of a top-20 sampler at LLM vocabulary sizes.

## Relation to importance corrections

The baseline assumes the per-token score is not reweighted by y. With an IS weight w(y) that
depends on the sampled token, the baseline term is -A w(y) s_bar, whose expectation under q is
-E_q[w A | h] s_bar(h) and is not zero. The baseline then adds a new drift of its own, which
does not cancel the original one. In a scratch test (mini setting, seed 0), TIS with the
baseline collapsed to eval reward 0.003, while the unmodified TIS loss reached 0.815. For this
reason the baseline is active only when `correction is None`, the default. With a correction the
loss is the existing one. A correction-aware baseline would need E_q[w A | h] in the
coefficient, which is not part of this fix.

## Configuration

No configuration changes are needed. The baseline is on by default (`correction=None`) when
`sampler_topk_ids` and `sampler_topk_logprobs` are present in the batch. `train_rl.py` provides
them with `logprobs_topk=20`. Without them the loss falls back to plain REINFORCE.
