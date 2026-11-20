# REINFORCE under sampler mismatch: removing the sampler drift

## What is wrong with the current loss

`loss_fn_rl` (no correction) minimizes

    L(theta) = -(1/N) sum_{b,t} mask_{b,t} A_b log p_theta(y_{b,t} | h_{b,t}),

where h_{b,t} is the prefix, y_{b,t} the sampled token, A_b the sequence advantage
and p_theta the trainer. Its gradient is -(1/N) sum A_b s_{y_{b,t}}, with the score
s_v(h) = grad_theta log p_theta(v|h). The tokens are sampled from the sampler q != p.

Fix a prefix h and take the expectation over the next token y ~ q(.|h). The
expected per-position update is

    E_q[A s_y | h] = sum_v q(v|h) E[A | h, y=v] s_v.

Write E_q[A s_y | h] = E_q[A | h] s_bar(h) + Cov_q(A, s_y | h), where

    s_bar(h) = sum_v q(v|h) s_v(h).

On-policy (q = p), s_bar = sum_v p(v|h) grad log p(v|h) = grad sum_v p(v|h) = 0, so
only the covariance term, which is the reward signal, survives. Under mismatch,
s_bar != 0 and the first term is a drift: it is proportional to the expected
advantage E_q[A | h] of the prefix, and it is not a function of the reward
gradient. Two facts make it systematic rather than zero-mean noise:

1. Its sign follows the sampler's errors. Since sum_v p s_v = 0, we can write
   s_bar(h) = sum_v (q(v|h) - p(v|h)) s_v(h). The drift pushes the trainer toward
   tokens that q over-weights and p under-weights, weighted by how good the
   prefix is.
2. It compounds. The sampler is refreshed from the trainer every step, so the
   drift changes q, which changes the next drift. The mismatch is never
   averaged out.

Under the existing IS corrections the drift is only partly handled. Unclamped
token IS gives E_q[(p/q) A s_y | h] = sum_v p(v|h) E[A|h,v] s_v, which has no
drift term in expectation, but clipping or truncating the weight (TIS, DAPO,
PPO, ...) reintroduces a bias proportional to sum_v (p - min(p, c q))(v) E[A|h,v] s_v.
This explains why these corrections slow the collapse but do not remove it.

## The modification

The drift is a multiple of s_bar(h). Subtract a baseline whose gradient is exactly
that drift. Replace the per-token score by

    phi_t = log p(y_t|h_t) - b(h_t),
    b(h) = sum_v c(v|h) log p(v|h),      c(v|h) = q(v|h) - p(v|h)   (stop-gradient).

The gradient of b is sum_v c(v|h) s_v. It equals s_bar(h) when we use the full
sum and the identity sum_v p s_v = 0.

The loss is L = -(1/N) sum A_b phi_t, with gradient -(1/N) sum A_b (s_{y_t} - sum_v c s_v).

Why it removes the drift. For a fixed prefix,

    E_q[A (s_y - s_bar) | h] = E_q[A s_y | h] - E_q[A | h] s_bar(h)
                             = Cov_q(A, s_y | h).

So the expected update is the covariance term alone, which carries the reward
signal. The drift E_q[A | h] s_bar(h) is removed exactly, whatever E_q[A | h] is.

Why it does not change the on-policy case. The coefficient c = q - p is zero
when q = p. Then b(h) = 0 identically and the gradient is unchanged. The
modification is therefore a control variate that vanishes on-policy.

## Implementation: top-k approximation

The full q(.|h) is not available. The sampler returns its top-20 logprobs,
`sampler_topk_ids` and `sampler_topk_logprobs` [B, T, 20]. Let S(h) be those
20 tokens, with q(v|h) = exp(sampler_topk_logprobs). Use the trainer's p on S,
and approximate the drift by

    b(h) = sum_{v in S(h)} (q(v|h) - p(v|h)) log p(v|h).

Its gradient is sum_{v in S} (q - p)(v|h) s_v. This is the drift restricted to the
top-20 tokens. The omitted part is sum_{v not in S} (q - p)(v|h) s_v, which is
small when the sampler's tail mass is small and q and p agree on it. On-policy,
q = p on S, so b = 0 exactly. In the case with the full vocabulary, the top-20
truncation did not change the results (tested in a 12-token setting with 300
extra tail tokens, where the full-vocabulary baseline gave no measurable gain).

The baseline is applied only when no IS correction is given (`correction is None`).
Under a weight w(y) the plain control variate no longer has zero mean under q:
E_q[w (s_y - D)] = E_q[w s_y] - E_q[w] D, which biases the update. Measured under
TIS (`level=token high=2 outside=clamp`), the baseline collapsed the run, so it is
disabled there. The default `loss_fn_rl(...)` call (correction=None) has the
modification active.

## Configuration

No configuration changes are needed. The modification is on by default, and it
uses the top-20 sampler logprobs that `train_rl.py` already sends.

## Expected behaviour

- On-policy (q = p): the gradient is identical to the original loss.
- Under mismatch with correction=None: the expected update loses the
  E_q[A | h] s_bar(h) term, leaving the reward covariance. The sampler-driven
  component no longer pushes the trainer in a direction set by the sampler's
  errors.
