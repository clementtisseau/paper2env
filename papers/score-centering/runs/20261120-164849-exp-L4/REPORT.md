# Removing the sampler-mismatch drift from REINFORCE

## Setting and notation

Trainer policy p(v|h) = softmax(z(h))_v, parameters theta. Sampler distribution
q(v|h) (weight noise, quantized weights or KV cache, staleness). Completion
y_1..y_T is drawn from q, with sequence advantage A (group-centered). Let
s_v(h) = grad_theta log p(v|h) be the score of token v. The loss in
losses.py is

    loss = -(1/N) sum_b sum_t A_b w_bt log p(y_bt | h_bt),

with w = 1 when no IS correction is set. Its negative gradient is the
REINFORCE estimate g = sum_t A s_{y_t}(h_t).

## What is wrong with the current loss

Take one position, condition on the prefix h, and average over the next token
y ~ q(.|h). Because A depends on the whole completion, E_q[A s_y | h] splits as

    E_q[A s_y | h] = E_q[A | h] s_bar(h) + Cov_q(A, s_y | h),
    s_bar(h) = sum_v q(v|h) s_v(h).

The first term is the drift. On-policy it vanishes, since
s_bar_p(h) = sum_v p(v|h) grad log p(v|h) = grad sum_v p(v|h) = 0, so only the
covariance (the actual reward signal) is left. Under mismatch it does not
vanish. For a softmax policy, s_v = e_v - p, so

    s_bar(h) = q(.|h) - p(.|h)   (in logit space),

and the drift is E_q[A | h] (q(.|h) - p(.|h)). It is a systematic push of the
trainer's logits toward the sampler's distribution at every prefix whose
expected advantage is nonzero. The reward does not produce this term; it
comes from the mismatch alone. Group centering does not remove it: E[A | h]
is generally nonzero at a given prefix, because the prefix often already
determines whether the completion will be rewarded.

Because the sampler is refreshed from the trainer at every step, the
mismatch at step k shifts the sampling distribution at step k+1, and the
drift accumulates. This is the compounding degradation and collapse that
the task describes. It is a bias, not zero-mean noise, so averaging over
more rollouts does not remove it.

IS corrections and clipping act on the ratio p/q of the sampled token. They
do not touch E_q[A | h] s_bar(h): the drift is a property of the prefix
distribution, not of the single sampled token. Clipping the ratio makes the
weight one-sided, which adds a further bias, so stronger token-level clipping
can make things worse.

## The fix: subtract the expected score per token

Replace the score s_y by s_y - s_bar(h), keeping the same advantage A:

    loss_t = -A_t ( log p(y_t|h_t) - sum_v sg[q(v|h_t)] log p(v|h_t) ).

Its gradient is -A_t (s_{y_t} - s_bar(h_t)). Under the sampler, for each prefix,

    E_q[A (s_y - s_bar) | h] = E_q[A s_y | h] - s_bar(h) E_q[A | h]
                             = Cov_q(A, s_y | h).

The drift term is gone exactly. The modified update is the covariance between
advantage and score under the sampler, with no term that depends on q - p
through E[A | h] s_bar.

On-policy, q = p, the subtracted term has expectation
E_p[A | h] * s_bar_p(h) = 0, so the expected update is unchanged. The
modification adds no bias where the original loss was already unbiased. It
uses no importance ratios, so it adds no ratio-clipping bias either.

The per-token baseline s_bar(h) is a function of the prefix only, which is
why it is a valid subtraction: it does not depend on the sampled token y_t.
The advantage A_t is the usual sequence advantage, unchanged.

### Computing s_bar(h) from the top-K head

The loss needs the sampler's full next-token distribution q, but the sampler
returns only its top-K logprobs H (K = 20) per position. Model the tail of q as
proportional to the tail of p, with mass ratio

    rho = q_tail / p_tail = (1 - sum_{v in H} q_v) / (1 - sum_{v in H} p_v).

Then q_v = rho p_v for v outside H. Using sum_v p_v s_v = 0 over the whole
vocabulary, the tail contributes

    sum_{v not in H} rho p_v s_v = -rho sum_{v in H} p_v s_v,

so

    s_bar(h) ~= sum_{v in H} (q_v - rho p_v) s_v.

The surrogate is therefore

    loss_t = -A_t ( log p(y_t|h_t) - sum_{v in H} sg[q_v - rho p_v] log p_v ),

with sg the stop-gradient. The sampled token keeps its own log p(y_t|h_t)
even when it lies outside H; the gradient of the baseline at y_t is what
removes the drift. The coefficients q_v - rho p_v and rho are detached, so the
gradient is exactly -A_t (s_{y_t} - sum_{v in H} c_v s_v) with c_v = q_v - rho p_v.

Implementation: `score_baseline` in losses.py computes the sum over H with
the trainer's log-softmax, and `loss_fn_rl` applies it inside the same chunked
pass as the existing score. The head arrays `sampler_topk_ids` and
`sampler_topk_logprobs` are added to `sampler_keys`. With no head in the batch,
the loss falls back to the plain score.

## Derivation check

1. Full vocabulary (K = |V|): rho = 1 and the coefficients are exact. The
   expected gradient over y ~ q is zero for the baseline term, so the identity
   E_q[s_y - s_bar] = 0 holds exactly. I checked this numerically with a
   50-token softmax policy and q a perturbation of p: the centered expected
   gradient is about 1e-8, against 3.5e-2 for the uncentered loss.
2. Top-20 head on the same toy problem: the residual drops to about 2e-3,
   from the tail-proportionality approximation. This is still an order of
   magnitude below the uncentered drift.
3. On-policy (q = p) the coefficients are q_v - rho p_v = 0 for every v in H
   (rho = 1), so the surrogate reduces to the plain score and the update is
   identical to REINFORCE.

## Why this should stabilize training

The uncentered expected update is E_q[A s] = Cov_q(A, s) + E_q[A | h] s_bar(h).
The second term is an explicit, reward-independent bias toward q whenever q
differs from p. The modified update keeps only the covariance term, so the
sampler's mismatch no longer moves the trainer toward the sampler's
distribution. Because the sampler is refreshed from the trainer, removing the
drift also removes the feedback loop that makes the bias compound.
