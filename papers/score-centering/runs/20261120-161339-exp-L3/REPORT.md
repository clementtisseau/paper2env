# Report

## What is wrong
Fix a prefix h. Tokens y ~ q(.|h), advantage A, score s_v = grad_theta log p(v|h). The REINFORCE update has expectation

E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),  s_bar(h) = sum_v q(v|h) grad log p(v|h).

On-policy (q=p), s_bar = grad sum_v p(v|h) = grad 1 = 0, so only the covariance (the true reward signal) remains. When q != p, s_bar = grad_theta sum_v q(v) log p(v) = -grad_theta CE(q, p) != 0 (it points toward making p match q, i.e. it is the gradient of -KL(q||p)). The first term is a drift of size E_q[A|h]*s_bar(h): at prefixes with positive mean advantage the model is pushed to *fit the sampler's distribution* (reinforcing q's idiosyncrasies, e.g. quantization errors), at prefixes with negative mean advantage it is pushed away from q. This is a systematic bias, not zero-mean noise. Because the sampler is refreshed from the trainer each step, the bias compounds. Token-level IS weights w(p/q) or clipping reweight/mask samples but do not make s_bar vanish (E_q[w s_y] = sum_v p(v) s_v = 0 only for exact unclipped weights, which have huge variance; clipping reintroduces a nonzero mean of w*s, and dropping tokens makes it worse).

## Fix
Subtract the drift per token, using the sampler's next-token distribution (top-20 logprobs, renormalised to sum 1):

loss = - A * [ log p(y|h) - sum_{v in top20} q~(v|h) log p(v|h) ]  (the q~ term is detached; gradient flows through log p).

The update direction is A*(s_y - s_bar(h)). With s_bar known (computed from q, independent of y) E_q[A (s_y - s_bar)|h] = Cov_q(A, s_y|h): the drift term vanishes exactly for any q, leaving the covariance, which equals the on-policy gradient signal. For q = p, s_bar = 0 so the loss is unchanged (unbiased, no cost). The subtracted term is a control variate with zero conditional correlation to the reward signal; it also lowers variance. The only approximation is truncation of q to its top-20 (tail mass ignored by renormalisation). It also composes with an optional IS weight w (multiplies both terms). It is active by default whenever the batch carries sampler_topk_* and needs no flag.

## Configuration
rl.is.level=null
