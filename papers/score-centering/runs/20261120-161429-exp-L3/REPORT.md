# Removing the sampler-induced drift from the REINFORCE loss

## What is wrong with the current loss

Fix a prefix h (the tokens before position t) and let y be the next token. The
current loss, with correction=None, is

    L = -(1/N) sum_i A_i sum_t log p(y_it | h_it),

so its gradient at each position is -A * s_y, where s_v = grad_theta log p(v|h) is the
score of token v. Sampling y from the sampler q(.|h) instead of the trainer p(.|h), the
expected update at that position is

    E_q[A s_y | h] = E_q[A | h] * s_bar_q(h) + Cov_q(A, s_y | h),
    s_bar_q(h) = sum_v q(v|h) s_v.

On-policy (q = p) the first term vanishes, because s_bar_p(h) = grad sum_v p(v|h) = 0.
Under mismatch it does not: s_bar_q(h) = sum_v (q - p)(v|h) s_v is nonzero, and the
term E_q[A|h] s_bar_q(h) is a drift. Its coefficient is the expected advantage of the
prefix, so it is the gradient of a reward-weighted cross-entropy between sampler and
trainer,

    E_q[A|h] * s_bar_q(h) = -E_q[A|h] * grad_theta H(q(.|h), p(.|h)),

where H(q, p) = -sum_v q log p. The loss therefore pushes p(.|h) toward q at prefixes
with positive expected advantage and away from q at prefixes with negative expected
advantage. Nothing in this direction depends on the reward beyond the sign and size of
E_q[A|h], and it is a bias, not zero-mean noise. With a fixed weight perturbation the
bias points the same way every step. Each step the sampler is refreshed from the
trainer, so the error q - p that produced the drift is carried forward and accumulates
(the ratchet described in the task). Clipping and IS weights reduce the size of the
drift without removing it, because they only rescale the sampled-token term.

## Fix: subtract the sampler's expected score

Let D(h) be the expected score under the sampler, with per-token weight w(v|h) (see below):

    D(h) = sum_v q(v|h) w(v|h) s_v,   with w = 1 when no correction is used.

The loss becomes

    L = -(1/N) sum_i A_i sum_t [ w(y_it|h_it) log p(y_it|h_it) - sum_v q(v|h_it) w(v|h_it) log p(v|h_it) ],

so the per-token gradient is -A (w s_y - D). Only the sampled token y has its own
score in the first term. The second term depends on h only, not on y, through
q(.|h) and p(.|h).

Derivation. Write g_y = w(y|h) s_y. Then E_q[g_y | h] = sum_y q(y) w(y) s_y = D(h), and
therefore

    E_q[A g_y | h] = E_q[A | h] E_q[g_y | h] + Cov_q(A, g_y | h) = E_q[A | h] D(h) + Cov_q(A, g_y | h).

The subtracted term is E_q[A D(h) | h] = E_q[A | h] D(h). So

    E_q[A (g_y - D(h)) | h] = Cov_q(A, g_y | h).

The drift is removed exactly for any sampler distribution q, and the estimator is still
unbiased for the covariance. Three checks:

1. On-policy (q = p, w = 1): D = sum_v p s_v = 0, so the new loss has the same expected
   gradient as REINFORCE. Nothing changes where the original was already correct.
2. Under mismatch: the expected update is Cov_q(A, s_y | h). It has no component
   E_q[A|h] s_bar_q, so it contains only the reward signal as seen through the sampler's
   rollouts, not an advantage-weighted pull toward q.
3. The subtracted term is a function of h, so it adds variance only through the sampled
   A, not through a new source of noise.

The sampler's logprobs of the top K = 20 tokens (sampler_topk_ids, sampler_topk_logprobs,
returned by the sampler at every position) give q(v|h) for v in the top-K, so D is computed
from them. The trainer's p(v|h) for those same ids comes from the existing full-vocabulary
log-softmax, so the extra cost is a gather of [B, C, K]. The mass outside the top-K is
ignored. Its contribution to D is sum_{v not in top-K} q(v|h) w(v|h) s_v, which is small when
topk_mass = sum_{top-K} q is close to 1; topk_mass is logged.

## IS corrections

Corrections enter through the same weight w(v|h) = is_weight(p(v|h)/q(v|h)) for the sampled
token and for every top-K token v (for stat=tv, the TV distance is per token). Because the
centering uses the same w as the sampled-token term, the cancellation above holds for the
corrected estimator: with g_y = w(y) s_y, E_q[A(g_y - D)|h] = Cov_q(A, g_y | h). Using w only
on the sampled token and not in D would leave a drift of size E[A|h](sum_v q w s_v - sum_v q s_v)
and was observed to be worse than no centering at all.

Sequence-level corrections (seq_mean, seq_sum) multiply A by a weight that depends on the
whole sequence, so E_q[w_seq A | h] no longer factors out E[A|h], and the cancellation is
approximate there. Token-level corrections are exact in the sense above.

## Configuration

None. The default configuration (sampler.logprobs_topk=20 as in train_rl.py, rl.is.* unset)
activates the centering. The top-K keys are already passed to the loss by train_rl.py. If a
batch has no top-K keys, the loss reduces to the original REINFORCE expression.
