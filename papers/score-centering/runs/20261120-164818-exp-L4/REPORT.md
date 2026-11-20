# Drift-free REINFORCE under sampler/trainer mismatch

## Problem
Let p be the trained policy and q the sampler, s_v = grad log p(v|h). For a fixed prefix h and y ~ q,
E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.
On-policy, s_bar = grad sum_v p_v = 0 and only the covariance (the reward signal) remains. When q != p,
s_bar = sum_v (q_v - p_v) s_v != 0, so the update has a drift E_q[A|h] s_bar(h) unrelated to reward covariance.
It is a systematic bias, not zero-mean noise: with group-centered rewards E[A|h] is nonzero at individual prefixes
(good prefixes have A>0), so the drift systematically pushes p toward/away from where q differs from p. Since the sampler
is refreshed from the trainer every step, the bias compounds. Token-level IS weights/clipping only reweight s_y by a function
of the sampled token and do not make the expectation of the update drift-free (clipping even removes the samples that
carry the correction), so they slow but do not remove it.

## Fix
Replace s_y by s_y - s_bar(h), with the same advantage A. Then
E_q[A (s_y - s_bar) | h] = Cov_q(A, s_y | h): drift exactly removed; on-policy (q=p) s_bar=0 so the loss is unchanged. No importance ratios are used.

Single-backward surrogate (sg = stop-gradient):
loss_t = -A ( log p(y_t|h) - sum_v sg[q_v] log p_v ), gradient -A (s_y - s_bar).

With only the sampler's top-k head H, model the tail as q_v = rho p_v for v not in H, rho = q_tail/p_tail,
q_tail = 1 - sum_H q_v, p_tail = 1 - sum_H p_v. Then
s_bar = sum_H q_v s_v + rho sum_tail p_v s_v, and sum_all p_v s_v = 0 gives sum_tail p_v s_v = -sum_H p_v s_v, so
s_bar = sum_H (q_v - rho p_v) s_v, i.e.
loss_t = -A ( log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p_v ).
The sampled token keeps its own log p even if outside H. In the code rho is clipped to [0,10], p_tail floored at 1e-6,
invalid top-k slots are ignored. If the batch has no top-k keys the loss falls back to plain REINFORCE. Optional IS
corrections still multiply the sampled-token term only; the default is none.

## Configuration
rl.is.level=token
