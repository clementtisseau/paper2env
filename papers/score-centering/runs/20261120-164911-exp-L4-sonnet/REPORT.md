# Drift-free REINFORCE under sampler/trainer mismatch

## Problem
Let p be the trained policy, q the sampler, h a prefix, y ~ q(.|h), A the advantage, s_v = grad log p(v|h).
The REINFORCE update has expectation, at a fixed prefix,

E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.

On-policy (q=p): s_bar = grad sum_v p_v = grad 1 = 0, so only the covariance (the true reward signal) remains.
When q != p, s_bar(h) = sum_v (q_v - p_v) s_v != 0, and the first term is a drift proportional to E_q[A|h], unrelated to reward
information beyond the prefix value. It is a systematic bias: it pushes up/down the tokens that q over-weights relative to p
(sharpening or flattening depending on the sign of E[A|h]), and because the sampler is refreshed from the trainer each step the bias
compounds. Token-level IS weights/clipping only reweight y by p/q or mask tokens; they do not remove s_bar (and clipping changes
the weights asymmetrically, adding further bias), so degradation is slowed but not removed.

## Fix
Replace s_y by s_y - s_bar(h), keeping the same advantage A. Then
E_q[A (s_y - s_bar) | h] = E_q[A s_y] - E_q[A] s_bar = Cov_q(A, s_y | h):
the drift vanishes exactly, for any q, and on-policy the change is zero (s_bar=0). No importance ratios are used.

Single backward pass surrogate (sg = stop-gradient):
loss_t = -A ( log p(y_t|h) - sum_v sg[q(v|h)] log p(v|h) ),
with gradient -A (s_y - s_bar).

With only the sampler's top-k head H, model the tail as q_v = rho p_v for v not in H, rho = q_tail/p_tail,
q_tail = 1 - sum_H q_v, p_tail = 1 - sum_H p_v. Since sum_v p_v s_v = 0, sum_{tail} p_v s_v = -sum_H p_v s_v, so
s_bar = sum_H q_v s_v - rho sum_H p_v s_v = grad of sum_H sg[q_v - rho p_v] log p_v, giving
loss_t = -A ( log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p_v ).
The sampled token's own log p is kept even if outside H. If the tail model is wrong, the residual error is only in the (small) tail mass.
If top-k fields are absent the loss falls back to plain REINFORCE. An optional IS correction (if passed) multiplies the whole
per-token term as before; the default is none.

## Implementation
`losses.py` forwards `sampler_topk_ids/logprobs` to the chunked function, computes q_v, p_v on the head ids from the full
log-softmax, rho (clipped to [0,10], p_tail floored at 1e-6), and subtracts the detached-coefficient baseline from log p(y_t).
Invalid (non-finite) top-k entries are ignored. Default arguments activate it.

## Configuration
