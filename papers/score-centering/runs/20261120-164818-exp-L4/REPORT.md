# Report

## Problem
Let q be the sampler and p the trained model, h a prefix, y~q(.|h), s_v = grad log p(v|h), A the advantage. REINFORCE's expected update at h is

E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.

On-policy (q=p): s_bar = grad sum_v p_v = 0, leaving only the covariance (the true signal). When q != p, s_bar(h) != 0, so there is a drift E_q[A|h] * s_bar(h) that does not come from reward. It is a systematic bias (not zero-mean noise): it pushes p toward/away from q's mass pattern wherever the expected advantage at a prefix is nonzero. Since the sampler is refreshed from the trainer each step, the bias compounds. Per-token IS ratios/clipping reweight samples but do not remove the term (and with clipping change the weights asymmetrically, which can add bias).

## Fix
Use the score s_y - s_bar(h) with the same advantage. Its expectation is E_q[A s_y] - s_bar E_q[A] = Cov_q(A, s_y | h). The drift is exactly removed, with no importance ratios (no variance blow-up); on-policy s_bar=0 so the loss is unchanged.

Surrogate (sg = stop-gradient): loss_t = -A (log p(y_t|h) - sum_v sg[q_v] log p_v), gradient -A(s_y - s_bar).

Only the sampler top-k head H is available. Model q's tail as proportional to p: q_v = rho p_v for v not in H, rho = q_tail/p_tail, q_tail = 1 - sum_H q, p_tail = 1 - sum_H p. Then
s_bar = sum_H q_v s_v + rho sum_{tail} p_v s_v, and since sum_all p_v s_v = 0, sum_tail p_v s_v = -sum_H p_v s_v. Hence
s_bar = sum_H (q_v - rho p_v) s_v,
giving loss_t = -A (log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p_v). The sampled token's own log p is kept even if outside H. Padded/-inf top-k entries are masked; p_tail is floored at 1e-6.

Implementation: `loss_fn_rl` forwards `sampler_topk_ids/logprobs` and computes the baseline per chunk from the full log-softmax; active by default. If a correction is passed, its detached weight w multiplies the whole centered score. If top-k keys are absent it falls back to plain REINFORCE.

## Configuration
