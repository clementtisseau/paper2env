# Score centering of the REINFORCE loss

## Problem
Let h be a prefix, y ~ q(.|h) the sampled token, A the advantage, s_v = grad_theta log p(v|h). The REINFORCE gradient (per token, A*s_y) has expectation
E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.
On-policy (q=p), s_bar = grad sum_v p(v|h) = 0, so only the covariance (the true policy-gradient signal) remains. When q != p, s_bar = sum_v (q_v - p_v) s_v != 0 (since sum_v p_v s_v = 0). The first term is then a drift proportional to E_q[A|h], unrelated to reward covariance: it pushes up (A>0) or down (A<0) the tokens q favors over p at that prefix, i.e. systematically changes the entropy/sharpness of p. It is a bias, not zero-mean noise; because the sampler is refreshed from the trainer, it compounds. Importance weights w(p/q) rescale per-token terms but do not remove it (the weighted drift E_q[w A | h] sum q_v w_v s_v is still nonzero, and clipping/dropping tokens makes the weights depend on v, which makes it worse).

## Fix
Replace the score s_y by the centered score s_y - s_bar(h), with s_bar computed under the sampler distribution and treated as a constant weight (detached q):
loss_token = -A * ( log p(y|h) - sum_v q(v|h) log p(v|h) ), q stop-gradient.
Gradient: A (s_y - s_bar). Expectation over y~q: E[A|h] s_bar + Cov(A,s_y) - E[A|h] s_bar = Cov_q(A, s_y | h). The drift vanishes exactly for any q (the centering term has zero effect when q=p because s_bar=0, so on-policy the loss is unchanged); the reward-covariance signal is retained unchanged. It is a control variate that is exact for q, independent of A and of how q deviates from p (noise, quantization, KV cache, stale weights).

## Implementation
q is only known through the sampler top-20 logprobs. sum_v q_v log p_v uses the top-20 exactly (log p gathered from the full log-softmax). The leftover mass 1 - sum_topk q is distributed over the remaining tokens proportionally to p (detached weights), so the tail contributes resid/P_tail * sum_{v not in topk} p_v log p_v. This is computed chunkwise like the rest of the loss. If the top-k keys are absent, the loss falls back to plain REINFORCE. The optional IS correction still works and multiplies the centered score; it is off by default.

## Configuration
