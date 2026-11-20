# Report

## Problem
Let q be the sampler, p the trained model, h a prefix, y ~ q(.|h), s_v = grad_theta log p(v|h). The loss -A*log p(y) gives expected per-position update

E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.

On-policy (q=p) s_bar = grad sum_v p(v|h) = 0, so only the covariance (the true policy gradient) remains. When q != p, s_bar = sum_v (q_v - p_v) s_v != 0 (since sum_v p_v s_v = 0). The first term is a drift proportional to E_q[A|h], which is not reward signal. It is a systematic bias, not zero-mean noise: it pushes up/down the probability of the tokens where q and p disagree, according to the advantage of the prefix. Because the sampler is refreshed from the trainer each step, the drift accumulates and compounds. Group-centered advantages have zero mean per group, but not conditional on a prefix, so E_q[A|h] != 0. Token-level IS weights w(p/q) or clipping only reshape the weights: E_q[w A s_y] still contains w-weighted drift, so they slow but do not remove it (stronger clipping changes the weights unevenly, and can enlarge it).

## Fix
Use the score with its q-mean removed, per position:

 score_t = log p(y_t|h) - sum_v q(v|h) log p(v|h),  loss = -(sum A * w * score)/N.

Its gradient is s_y - s_bar(h) (the q-weights are detached; gradient flows through log p(v|h)). Then
E_q[A (s_y - s_bar) | h] = E_q[A s_y] - E_q[A] s_bar = Cov_q(A, s_y | h),
the drift is exactly cancelled for any q, and for q=p the extra term has zero gradient in expectation (s_bar=0), so the loss reduces to ordinary REINFORCE in expectation. This is a control-variate/baseline in token space: it is state-dependent but depends on y only through q's constant weights, so it adds no bias to the covariance term, and reduces variance too.

q(.|h) comes from the sampler's top-20 logprobs (sampler_topk_ids / sampler_topk_logprobs), renormalized over the top-20 (the tail mass is small for the sampler; the residual error is only from the tail). The correction is active by default, composes with an optional IS correction (w multiplies the whole centered score), and works with correction=None. Falls back to the plain score if top-k keys are absent.

## Configuration
