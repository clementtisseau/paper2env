# Report

## What is wrong
Fix a prefix h. The token y is drawn from the sampler q(.|h), A is the advantage, and s_v = grad log p(v|h). The REINFORCE update has expectation

E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.

On-policy (q = p), s_bar = grad sum_v p(v|h) = 0, so only the covariance, which is the reward signal, remains. When q != p, s_bar(h) = sum_v (q_v - p_v) s_v is not 0 (note sum_v p_v s_v = 0). The first term is then a drift proportional to E[A|h]. This is systematic, not zero-mean noise. It pushes the logits toward or away from q-weighted tokens at every prefix, whatever the reward. Because the sampler is refreshed from the trainer each step, the drift compounds. Token-level IS weights w(p/q) only reweight s_y and do not make s_bar vanish. They add their own bias and variance, and clipping zeroes more of the signal.

## Fix
Use the sampler's next-token distribution to subtract the drift exactly. The per-token surrogate is

L = - A * [ log p(y|h) - sum_v q(v|h) log p(v|h) ]  (the q-weights are detached).

Its gradient is A (s_y - s_bar(h)). Then

E_q[A (s_y - s_bar) | h] = E_q[A s_y] - E_q[A] s_bar = Cov_q(A, s_y | h).

This is the on-policy expected update, up to the choice of q versus p in the covariance, and the drift is removed for any q. It acts as a state-dependent, action-independent baseline on the score function: b(h) = s_bar(h) is the q-mean of the score, so it adds no variance of the sort IS weights do. It also needs no ratio clipping, so there is no extra bias. On-policy it has expectation zero, so it costs nothing.

Implementation: q(.|h) is taken from the sampler's top-20 logprobs (`sampler_topk_ids/logprobs`), renormalized over the 20 entries. The tail mass is ignored, which leaves a small residual drift. The residual is proportional to the tail mass, which is small for sampled LM distributions. The log p(v|h) at the top-k ids are gathered from the trainer's full log-softmax, so gradients flow through them. This is active by default with correction=None. If the keys are absent it falls back to plain REINFORCE. If a correction is passed, w multiplies both terms.

## Configuration
