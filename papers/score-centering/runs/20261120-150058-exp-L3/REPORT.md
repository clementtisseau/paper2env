# Report

## Problem
Fix a prefix h. The sampler draws y ~ q(.|h), the trainer is p, and s_v = grad log p(v|h). The REINFORCE
update has expectation

E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),   s_bar(h) = sum_v q(v|h) s_v.

On-policy (q = p), s_bar = grad sum_v p(v|h) = 0, so only the covariance, which is the reward signal, is left.
When q != p, s_bar(h) = sum_v (q - p)(v) s_v is nonzero (sum_v p s_v = 0). The first term is then a drift
E_q[A|h] * s_bar(h) that does not come from the reward. It is not zero-mean noise. It has a fixed
direction at each prefix, pushing log p up or down on the tokens q favours, with sign and size set by E[A|h].
Group-centering makes A zero-mean over the group, not conditional on h, so E[A|h] != 0 at most prefixes.
Because the sampler is refreshed from the trainer, the drift accumulates and can collapse training.
Per-token IS weights w(p/q) only reweight the samples. They do not remove s_bar, and clipping alters
the sampling distribution further, which adds a new bias.

## Fix
Use a per-position baseline that is itself a function of the score, namely the q-expected log-probability:

L = - sum_t w_t A ( log p(y_t|h_t) - sum_v q(v|h_t) log p(v|h_t) ),   with q and w detached.

The gradient is A (s_y - s_bar(h)). Since s_bar(h) is a function of h only, the term E_q[A s_bar | h] = E[A|h] s_bar
is exactly the drift, and subtracting it gives

E_q[A (s_y - s_bar) | h] = Cov_q(A, s_y | h).

The drift is removed exactly for any q, including int8/fp8, noise, a quantized KV cache and stale weights.
When q = p, s_bar = 0 and the loss is the ordinary REINFORCE loss. The baseline depends only on the
prefix and the next-token distribution, not on y, so it does not bias the covariance term. It costs one gather
of the top-k log-probs of p.

## Implementation
`losses.py` uses the sampler top-20 (`sampler_topk_ids` and `sampler_topk_logprobs`) to approximate the
sum over v. q is renormalised over the top-k, so the neglected tail mass is small. The baseline is only
approximate for the tail. If the keys are missing, the loss falls back to plain REINFORCE. It is active by
default, works with correction=None, and still composes with an IS weight w.

## Configuration
rl.is.level=None
