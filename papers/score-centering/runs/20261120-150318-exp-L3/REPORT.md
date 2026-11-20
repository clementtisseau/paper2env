# Score-centered REINFORCE under sampler mismatch

## What is wrong
Fix a prefix h. Tokens y ~ q(.|h), advantage A, score s_v = grad_theta log p(v|h). The REINFORCE update has expectation

E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.

On-policy (q = p): s_bar = grad sum_v p(v|h) = 0, so only the covariance (the true reward signal) remains.
If q != p: s_bar(h) = sum_v (q_v - p_v) s_v != 0 (since sum_v p_v s_v = 0). The first term is a drift proportional to E_q[A|h], independent of the reward covariance. With group-centered rewards A is zero-mean over a group but not conditional on a prefix, so E[A|h] != 0 at most prefixes; the drift is a systematic bias, not zero-mean noise. Since the sampler is refreshed from the trainer each step, the bias is applied continually and compounds (it pushes p toward/away from q's mass in a way tied to the prefix-level success rate, which can drive entropy and policy collapse). Token-level IS weights w = p/q reweight the expectation to sum_v p_v A s_v, which removes the bias only in expectation with exact ratios; with clipping/dropping, w changes the weighting and s_bar of the weighted measure stays nonzero, so clipping only slows or worsens things.

## Fix
Replace the score s_y by the centered score s_y - s_bar(h), with s_bar computed under the sampler distribution from its returned top-k logprobs (q renormalized over the top-k, detached). Then

E_q[A (s_y - s_bar) | h] = E_q[A|h] s_bar + Cov(A, s_y|h) - E_q[A|h] s_bar = Cov_q(A, s_y | h)

(using E[A s_bar] = E[A] s_bar since s_bar is deterministic given h). The drift term is cancelled exactly for any q, leaving only the reward covariance, which is also the on-policy update's expectation. It is a baseline over actions (control variate with zero mean under q: E_q[s_y - s_bar] = 0), so it does not bias the covariance term. When q = p it is a no-op in expectation (s_bar = 0) and only reduces variance.

Implementation: the surrogate is  A * w * (log p(y|h) - sum_{v in topk} qhat_v log p(v|h)) with qhat detached; its gradient is A w (s_y - sum qhat_v s_v). Active by default whenever sampler_topk_* are in the batch (falls back to plain REINFORCE otherwise). Any IS correction, if given, multiplies as before. Approximation: top-20 truncation of q (renormalized) leaves a small residual; the tail mass is ignored.

## Configuration
rl.is.level=null
