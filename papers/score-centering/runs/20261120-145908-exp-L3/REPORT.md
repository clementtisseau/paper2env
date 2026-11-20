# Report

## Problem
Fix a prefix h. The sampler draws y ~ q(.|h), the trainer is p(.|h) = p_theta. The loss is
-A * log p(y|h) (times a detached IS weight), so its expected gradient step is

E_q[A s_y | h] = E_q[A | h] * s_bar(h) + Cov_q(A, s_y | h),   s_bar(h) = sum_v q(v|h) grad log p(v|h).

On-policy q = p gives s_bar = grad sum_v p(v|h) = grad 1 = 0, so only the covariance (the true
policy-gradient signal) remains. With q != p, s_bar(h) = sum_v (q-p)(v) grad log p(v) != 0.
The first term is a drift: it is proportional to E_q[A | h] (positive prefixes get their sampled-mass
pushed up, i.e. the update raises log-likelihood of whatever q likes, regardless of reward), it is
systematic (not zero mean), and since the sampler is refreshed from the trainer each step it
compounds (e.g. drives entropy collapse or blow-up). With group-centered rewards A has zero mean
only per prompt, not per prefix, so E_q[A|h] != 0 at deeper prefixes. Per-token IS weights
w(p/q) reweight tokens but do not remove s_bar (E_q[w s] = sum_v p(v) s_v = 0 only for the
unclipped full ratio, which has huge variance; clipping/dropping destroys this identity and
re-introduces drift, which is why stronger clipping can be worse).

## Fix
Use the centered score. Replace log p(y|h) by

log p(y|h) - sum_v q(v|h) log p(v|h)

(q detached). Its gradient is s_y - s_bar(h), so
E_q[A (s_y - s_bar)|h] = E_q[A|h] s_bar + Cov_q(A,s_y|h) - E_q[A|h] s_bar = Cov_q(A, s_y | h).
The drift term cancels exactly, for any q, and the on-policy limit (s_bar = 0) is unchanged: the
modification is identical to REINFORCE when q = p, and it is a control variate (baseline per
prefix, depending only on h, not y) so it adds no bias to the covariance term and does not
increase variance in general (it removes the s_bar component).

The sampler only exposes its top-20 logprobs, so q is the top-20 distribution renormalized
(softmax of sampler_topk_logprobs); the neglected tail mass is small for the peaked LM
distributions, and the residual drift is only from that tail. Computation is done inside the
existing chunked log-softmax: gather log p at the top-k ids and take the q-weighted sum.
It is active by default whenever the batch carries sampler_topk_* keys, with correction=None or
any IS correction (the weight multiplies the centered score).

## Configuration
rl.is.level=null
