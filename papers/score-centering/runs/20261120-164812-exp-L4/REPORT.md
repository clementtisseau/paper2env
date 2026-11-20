# Removing the sampler-mismatch drift from the REINFORCE loss

## What is wrong with the current loss

`loss_fn_rl` uses the sampled-token REINFORCE surrogate

    loss = -sum_t A * log p(y_t | h_t) / N,     y_t ~ q(. | h_t),

where q is the sampler, p the trainer, A the (group-centered) advantage of the
completion and h_t the prefix. Its gradient at one position, for a fixed
prefix h, is -A s_y with s_v = grad_theta log p(v|h).

Take the expectation over the next token y ~ q(.|h). Using E_q[s_y | h] = s_bar(h),

    E_q[A s_y | h] = E_q[A | h] s_bar(h) + Cov_q(A, s_y | h),
    s_bar(h) = sum_v q(v|h) grad log p(v|h).

On-policy (q = p), s_bar(h) = sum_v p_v grad log p_v = grad sum_v p_v = 0, so
only the covariance term, which is the true policy gradient, remains.

Under mismatch, s_bar(h) = sum_v (q_v - p_v) s_v is generally nonzero. The first
term then acts as a drift: it is proportional to the expected advantage at that
prefix, E_q[A|h], which carries no information about whether the reward improves.
The sign of E_q[A|h] changes with the prefix, so over a batch it does not cancel
once it is weighted by s_bar(h), which varies with h. The update therefore moves
the trainer's next-token distribution toward whatever the sampler favours at
prefixes where the completions happen to score well. This is a bias, not zero-mean
noise. Because the sampler is refreshed from the trainer at every step, the
mismatch (the fixed noise, quantization, or staleness) is re-applied each
step, and the drift accumulates over training.

Importance ratios w = p/q remove the bias only in expectation (E_q[(p/q) A s_y] =
E_p[A s_y]), with high variance. Clipping the ratio reintroduces a bias of the
form E_q[(w - clip(w)) A s_y], which is nonzero and grows with the clip
asymmetry, so clipping does not remove the drift.

## The modification

Replace each token's score s_y by s_y - s_bar(h), keeping the same advantage A.
The expectation of the new estimator is

    E_q[A (s_y - s_bar(h)) | h]
      = E_q[A s_y | h] - E_q[A | h] s_bar(h)
      = Cov_q(A, s_y | h),

because E_q[s_y | h] = s_bar(h) by definition. The drift term is removed exactly.
No importance ratio is involved. On-policy, s_bar = 0 and the estimator is
unchanged.

### Implementation with the top-k head

The trainer needs s_bar(h) under q, but q is only known through the sampler's
top-K logprobs (K = 20) at each position. Write H for the top-K set. Model the
tail of q as proportional to p outside H:

    q_v = rho p_v   for v not in H,     rho = q_tail / p_tail,
    q_tail = 1 - sum_{H} q_v,   p_tail = 1 - sum_{H} p_v.

Since sum_v p_v grad log p_v = 0, the tail contributes

    rho sum_{tail} p_v s_v = -rho sum_{H} p_v s_v,

so

    s_bar(h) ~= sum_{v in H} (q_v - rho p_v) s_v.

With sg = stop-gradient, the per-token surrogate is

    loss_t = -A ( log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p_v ),

whose gradient is -A (s_y - sum_H c_v s_v) with c_v = q_v - rho p_v. The sampled
token keeps its own log p even when it lies outside H. Its expectation over
y ~ q is Cov_q(A, s_y | h) under the tail model, because E_q[s_y] = sum_H q_v s_v
+ rho sum_tail p_v s_v = sum_H c_v s_v + rho sum_v p_v s_v = sum_H c_v s_v.
The last step uses sum_v p_v s_v = 0.

Numerics: p_tail is computed in log space from the trainer's full log-softmax
(logsumexp over the vocabulary outside H), which avoids the cancellation of
1 - sum_H p_v when H covers almost all of p. rho is set to 0 when H covers all of
p. The coefficients are stop-gradient, so the baseline adds no gradient path
through rho.

### Why this is stable under the mismatch cases in the task

- Weight noise, quantization, and quantized KV cache change q_v, and therefore
  change c_v. The drift is E_q[A|h] s_bar(h), so it is removed whatever the
  mismatch source is, as long as the sampler's top-K distribution is reported.
- Staleness is the same argument with q the stale policy.
- The correction is a control variate: on-policy it has zero expectation and
  changes no gradient.

## Checks

- With K = V (full head), the gradient of the surrogate equals -A (s_y -
  sum_v q_v s_v) exactly per position, verified numerically against the manual
  expression (difference 0.0).
- On-policy (q = p, K < V), the drift coefficients c_v vanish, and the gradient
  equals plain REINFORCE (difference ~3e-8).
- On the CPU stand-in (`dev/mini.py`, default setting, 600 steps, 3 seeds): the
  original loss ends at eval reward 0.72, 0.64, 0.70; the modified loss at 0.84,
  0.82, 0.79. At weight noise 0.6 (2 seeds) the original collapses to reward 0 on
  both seeds, and the modified loss keeps reward ~0.5 to 0.6. This is a
  qualitative check only. The mini sampler has a 12-token vocabulary, so the head
  covers it fully and the tail model is not exercised.

## Limits

- The tail model (q proportional to p outside H) is an approximation. The
  correction is exact when H covers the support of q.
- The estimator targets Cov_q, not the on-policy covariance Cov_p. The
  difference is a second-order term in (q - p) times the covariance, and it does
  not carry the drift.
- The estimator uses the sampler's reported top-K logprobs. Without them, the
  code falls back to the plain surrogate.

## Configuration

No configuration changes are needed. The modification is active under the
default arguments of `loss_fn_rl`.
