# Removing the sampler-mismatch drift from the REINFORCE score

## What is wrong with the current loss

`loss_fn_rl` estimates the policy gradient with the sampled token's score only.
Per token t, with prefix h_t, the loss is

    L = -(1/N) sum_t A_i w_t log p(y_t | h_t),     s_v(h) = grad_theta log p(v | h),

so the gradient is -(1/N) sum_t A_i w_t s_{y_t}(h_t). Take the expectation over
the next token y ~ q(.|h), where q is the sampler's distribution and A is the
sequence advantage (A may depend on y and on all later tokens). With
s_bar_q(h) = sum_v q(v|h) s_v(h), the decomposition

    E_q[A s_y | h] = E_q[A | h] s_bar_q(h) + Cov_q(A, s_y | h)            (1)

holds for any random variable A. On-policy, q = p, and s_bar_p(h) =
sum_v p(v|h) grad log p(v|h) = grad sum_v p(v|h) = 0, so only the covariance
survives. This is the correct policy gradient signal.

Under mismatch, s_bar_q(h) != 0 in general. The term E_q[A | h] s_bar_q(h) is
a drift at every prefix. It is proportional to the expected advantage at that
prefix, so it is not driven by the reward signal at the token being trained.
Summed over the batch it is an update of the form

    E_h [ E_q[A | h] * grad_theta sum_v q(v|h) log p(v|h) ],

i.e. a cross-entropy step that moves p toward q at prefixes with positive expected
advantage and away from q at prefixes with negative expected advantage. Since the
sampler is refreshed from the trainer every step and q = p + mismatch, this
drift does not average out: the trainer keeps absorbing the sampler's bias,
which is why the failure is systematic and compounds. Importance weights
w(p/q) and clipping only rescale or truncate this term; they do not remove it,
which matches the observed slow degradation.

## Modification

Subtract the q-mean score from the sampled-token score, token by token:

    g_t = A_i (s_{y_t}(h_t) - s_bar_q(h_t)),
    s_bar_q(h) = sum_v q(v|h) grad_theta log p(v|h) = grad_theta b(h),
    b(h) = sum_v q(v|h) log p(v|h).

In `losses.py`, the per-token score becomes `logp(y_t) - b(h_t)`, with the IS weight
w still applied as before. `b(h)` is computed over the sampler's top-K tokens
(K = 20, from `sampler_topk_ids` / `sampler_topk_logprobs`). The sampler's logprobs are
renormalized over those K tokens and treated as constants (stop-gradient), so
the trainer's gradient flows only through log p. If the top-K keys are absent,
b = 0 and the loss reduces to the original one. The modification is on by
default, with no flag and no configuration change.

## Why it works

1. Drift is cancelled exactly. b(h) depends only on the prefix h, not on y or
   on A. So E_q[A s_bar_q(h) | h] = E_q[A | h] s_bar_q(h), and by (1)

       E_q[A (s_y - s_bar_q) | h] = E_q[A | h] s_bar_q + Cov_q(A, s_y | h) - E_q[A | h] s_bar_q
                                  = Cov_q(A, s_y | h).

   The drift term is removed for any mismatch and any advantage, including
   dependence of A on later tokens, because (1) is an identity.

2. On-policy nothing changes in expectation. If q = p then s_bar_q = s_bar_p = 0,
   so the modified estimator equals the original one. The fix only acts under
   mismatch, where the original gradient has an extra E[A|h] s_bar_q term.

3. The baseline does not bias the expected gradient, since it is a function of h
   alone (a control variate). The gradient is still a sum over the same tokens,
   with a shifted per-token score.

4. Residual error. The estimator now targets Cov_q(A, s_y | h), not
   Cov_p(A, s_y | h). The difference, sum_v (q_v - p_v)(A_v - A_bar) s_v, is
   first order in the mismatch, but it multiplies a centered advantage, not the
   expected advantage, and it does not come with the sampler-refresh feedback of
   the drift term. Two further approximations: (a) s_bar_q is computed from the
   top-K tokens only. Missing tail mass eps gives an error of order eps times the
   difference between tail and head scores, which is small for K = 20 on LLM
   vocabularies. (b) The IS weight w is applied to the centered score; with
   token-level correction, the drift cancellation is then exact only in the
   w = 1 case.

## Numerical check

For a 7-token softmax with random q, A and theta, the gradient of
E_q[A (log p(y) - b)] matches the covariance form Cov_q(A, s_y) to 1e-17, and
the naive gradient minus E_q[A] * s_bar_q equals the centered one. The q-mean
score is nonzero under mismatch (norm about 0.18 in the test) and exactly zero
when q = p.

## Dev runs (CPU, dev/mini.py, not the evaluation)

Default vocabulary (12 tokens), noise 0.3, 600 steps, seeds 0 to 2: final eval
reward 0.66 to 0.68 with the original loss, 0.81 to 0.84 with the centered score.
Larger vocabulary (MINI_EXTRA_TOKENS=300, seed 0): the original loss collapses
to eval 0.09 (peak 0.37); the centered score ends at 0.68 (peak 0.73).
