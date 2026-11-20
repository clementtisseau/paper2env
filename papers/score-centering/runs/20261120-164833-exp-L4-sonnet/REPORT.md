# Drift-free REINFORCE under sampler/trainer mismatch

## Problem
Rollouts y ~ q(.|h), trainer p(.|h)=p_theta. Let s_v = grad log p(v|h), A the advantage. Plain REINFORCE (also with IS weights, which only reweight tokens) has expected per-prefix update

E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.

On-policy, s_bar = sum_v p_v grad log p_v = grad sum_v p_v = 0, so only the covariance (the true reward signal) remains. If q != p, s_bar = sum_v (q_v - p_v) s_v != 0, giving a drift E_q[A|h]*s_bar(h) that is unrelated to reward covariance: it pushes up/down the tokens in which q and p disagree in proportion to prefix-level advantage. It is a systematic bias, not zero-mean noise. Because the sampler is re-synced from the trainer each step, the update changes p, hence the mismatch, hence the drift, so it compounds. Token-level IS ratios/clipping only reweight A s_y per token and do not remove the term (E_q[w A s_y] still has a nonzero mean part; clipping adds further bias), so they delay but do not cure it.

## Fix
Use the score s_y - s_bar(h) with the same advantage:
E_q[A (s_y - s_bar) | h] = Cov_q(A, s_y | h) exactly (since E_q[s_y - s_bar]=0 and s_bar is constant given h). The drift is removed, and on-policy the change is zero (s_bar=0). No importance ratios are required (variance-free of ratios).

Surrogate with one backward pass (sg = stop-gradient):
loss_t = -A ( log p(y_t|h) - sum_v sg[q_v] log p_v ), gradient -A(s_y - s_bar).

## Top-k sampler head
Only the top-k set H of q is available. Model the tail as q_v = rho p_v for v not in H, rho = q_tail/p_tail, q_tail = 1 - sum_H q_v, p_tail = 1 - sum_H p_v. Then
s_bar = sum_H q_v s_v + rho sum_{tail} p_v s_v = sum_H q_v s_v - rho sum_H p_v s_v,
using sum_all p_v s_v = 0. Hence
loss_t = -A ( log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p_v ).
The sampled token's own log p is kept even if outside H. Implementation in `losses.py` reads `sampler_topk_ids/logprobs` (added to `sampler_keys`), is on by default, works with correction=None (and still composes with an optional IS weight, which multiplies the whole bracket). If the top-k keys are absent it falls back to plain REINFORCE. Padded/-inf entries are masked; rho is guarded when p_tail~0.

## Configuration
(none needed)
