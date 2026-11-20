# Report

## Problem
Let q be the sampler and p the trained policy, h a prefix, s_v = grad log p(v|h), A the advantage. The loss -A log p(y) with y~q has expected gradient direction at h
E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.
On-policy, s_bar = grad sum_v p_v = 0, so only the covariance (the true reward signal) remains. When q != p, s_bar(h) != 0 and there is a drift E_q[A|h] * s_bar(h) unrelated to the reward: it pushes up/down the tokens by which q over-weights p, proportionally to the prefix's expected advantage. It is a systematic bias, not zero-mean noise; since the sampler is refreshed from the trainer each step, it compounds. Importance ratios/clipping reweight samples but do not make the expectation of A*s_y covariance-only (and clipping adds further bias), so they only slow the drift.

## Fix
Replace s_y by s_y - s_bar(h) with the same A. Its expectation is E_q[A (s_y - s_bar)] = Cov_q(A, s_y | h): drift removed exactly, no importance ratios, and the change is zero when q = p. Surrogate (sg = stop-gradient):
loss_t = -A ( log p(y_t|h) - sum_v sg[q_v] log p_v ), gradient -A (s_y - s_bar).

With only the sampler's top-k head H, model the tail as q_v = rho p_v for v not in H, rho = q_tail/p_tail, q_tail = 1 - sum_H q_v, p_tail = 1 - sum_H p_v. Then
s_bar = sum_H q_v s_v + rho sum_{tail} p_v s_v, and sum_tail p_v s_v = -sum_H p_v s_v (since sum_all p_v s_v = 0), so
s_bar = sum_H (q_v - rho p_v) s_v,
giving loss_t = -A ( log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p_v ). The sampled token's own log p is kept even if outside H. This is what losses.py implements, per token, using sampler_topk_ids / sampler_topk_logprobs (padding/invalid entries masked; rho clipped for numerical safety). It is active by default whenever those keys are in the batch; with no top-k available it falls back to plain REINFORCE. Any optional IS correction still multiplies as before, but none is needed.

## Configuration
