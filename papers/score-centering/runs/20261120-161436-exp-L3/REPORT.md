# Drift-free REINFORCE under sampler/trainer mismatch

## Problem
Rollouts y ~ q(.|h), trainer p. The REINFORCE gradient at prefix h is A * s_y, with s_v = grad log p(v|h). Then

E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.

On-policy, s_bar = grad sum_v p = 0, so only the covariance (the reward signal) remains. When q != p, s_bar(h) = sum_v (q - p)(v|h) grad log p(v|h) != 0 (it is the gradient of -KL-like cross-entropy E_q log p). The first term is a drift: it pushes p toward q (or away) in proportion to E_q[A|h], independent of the reward signal's covariance. Group-centering makes A zero mean only across the group, not per prefix, so E_q[A|h] != 0 at most prefixes. The sampler is refreshed from the trainer every step, so this bias is systematic and compounds rather than averaging out. Token-level IS weights w(p/q) only reweight the sampled tokens; they do not remove the mean-score term and (with clipping) add further bias, which is why they only slow the collapse.

## Fix
Subtract the q-expected score as a per-position control variate:

g = A * (s_y - s_bar(h)),  i.e. loss = -A * [ log p(y|h) - sum_v q(v|h) log p(v|h) ], with q detached.

Since s_bar is a function of h only (not of y), E_q[A (s_y - s_bar)|h] = E_q[A s_y|h] - E_q[A|h] s_bar = Cov_q(A, s_y|h). The drift is removed exactly for any q, and for q = p the correction is zero in expectation (s_bar = 0), so the on-policy gradient is unchanged. It is unbiased for the covariance term, and also acts as a baseline in score space (reducing variance). Gradient of sum_v q(v) log p(v) with q stop-gradiented is exactly sum_v q(v) s_v = s_bar.

## Implementation
q(.|h) is taken from the sampler's top-20 logprobs (sampler_topk_ids / sampler_topk_logprobs), renormalized over the 20 entries (the truncated tail mass is small); log p at those ids is gathered from the trainer's log-softmax in the same rematerialized chunk. The loss is active by default (no flag); it works with correction=None, and any optional IS weight w simply multiplies the centered score. If top-k keys are absent it falls back to plain REINFORCE. Residual bias comes only from the top-20 truncation.

No configuration changes are needed.
