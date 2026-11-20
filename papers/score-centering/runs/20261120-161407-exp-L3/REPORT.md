# Report

## Problem
Fix a prefix h. The sampler draws y ~ q(.|h), the trainer is p. With s_v = grad log p(v|h), REINFORCE's expected update at h is

E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) s_v.

On-policy, s_bar = grad sum_v p(v|h) = 0, so only the covariance (the reward signal) remains. When q != p, s_bar(h) = sum_v (q_v - p_v) s_v is nonzero, so a drift E_q[A|h] * s_bar(h) is added. It is not reward signal: it is a systematic push on the logits along s_bar, weighted by the mean advantage at the prefix. It is a bias, not zero-mean noise. Since the sampler is refreshed from the trainer each step, the bias compounds. With group-centered advantages A is zero-mean per prompt but not per prefix, so the drift does not cancel. IS weights w(p/q) only reweight samples. The drift term stays, scaled by w, and clipping or dropping tokens changes the weights and adds further bias. So the drift survives IS correction, and stronger clipping can worsen it.

## Fix
Subtract the q-mean score at every position, i.e. use a per-prefix baseline on the score, not on the reward:

loss = - sum_t A * w_t * ( log p(y_t|h_t) - b_t ),  b_t = sum_v q(v|h_t) log p(v|h_t), with q detached.

grad b_t = sum_v q_v s_v + sum_v log p_v grad q_v. The second term vanishes because q is detached, so grad b_t = s_bar. The update is then A (s_y - s_bar). Its expectation under q is E_q[A s_y] - E_q[A] s_bar = Cov_q(A, s_y | h). This is the drift-free covariance term for any q, and equals the on-policy gradient when q = p (s_bar = 0, so the correction is a no-op in expectation and only reduces variance). The drift is removed exactly, without depending on the ratio, so no clipping is needed to control it. The same centering applies under an IS weight w that is constant at the sampled token, up to the usual IS bias.

## Implementation
q is only available as the sampler's top-20 logprobs. We renormalise them over the top-k (softmax of sampler_topk_logprobs, detached) and take log p at those ids from the trainer's full log-softmax. This gives b_t. The tail mass is ignored (it is small for peaked LLM distributions). It is active by default whenever the batch has sampler_topk_* keys, and works with correction=None. If those keys are absent, the loss falls back to plain REINFORCE.

## Configuration
rl.is.level=token
