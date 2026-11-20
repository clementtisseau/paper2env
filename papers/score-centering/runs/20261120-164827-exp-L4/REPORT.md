# Report

## Problem
Let q be the sampler, p the trained model, h a prefix, A the advantage, s_v = grad log p(v|h). REINFORCE's expected update at h is
E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.
On-policy (q=p): s_bar = grad sum_v p_v = 0, so only the covariance (the true policy gradient signal) remains. When q != p, s_bar(h) = sum_v (q_v - p_v) s_v != 0, giving a drift E_q[A|h] * s_bar(h) that is unrelated to reward covariance. It is a systematic bias (not zero mean), it pushes probability toward/away from tokens according to where q and p disagree weighted by the prefix's mean advantage, and since the sampler is refreshed from the trainer each step the bias compounds. IS ratios/clipping reweight tokens but do not remove this term (they change the weights, not the nonzero mean of the score under q) and clipping can add further bias.

## Fix
Replace s_y by s_y - s_bar(h), same advantage A. Then
E_q[A (s_y - s_bar) | h] = E_q[A|h] s_bar + Cov_q(A,s_y|h) - E_q[A|h] s_bar = Cov_q(A, s_y | h):
the drift cancels exactly; on-policy s_bar=0 so the loss is unchanged. No importance ratios are used (no variance blow-up).
Single backward pass surrogate (sg = stop-gradient):
loss_t = -A (log p(y_t|h) - sum_v sg[q_v] log p_v), gradient -A (s_y - s_bar).
With only the sampler top-k head H, model the tail as q_v = rho p_v for v not in H, rho = q_tail/p_tail, q_tail = 1 - sum_H q, p_tail = 1 - sum_H p. Because sum_v p_v s_v = 0, the tail's contribution to s_bar is rho * sum_{tail} p_v s_v = -rho sum_H p_v s_v, so
s_bar ≈ sum_H (q_v - rho p_v) s_v, giving
loss_t = -A (log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p_v).
The sampled token's own log p is kept even if outside H. Implementation: `losses.py` always applies this when `sampler_topk_ids/logprobs` are in the batch (rho clipped to [0,20], p_tail floored at 1e-6, non-finite q logprobs treated as 0 mass); falls back to plain REINFORCE if absent. It is active with correction=None; an optional IS correction weight multiplies the centered score as before.

## Configuration
