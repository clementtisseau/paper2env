# Drift-free REINFORCE under sampler mismatch

## What is wrong with the current loss

Fix a prefix h (the tokens before position t). The sampler draws the next token y from q(.|h); the trainer has p_theta(.|h). Write s_v(h) = grad_theta log p_theta(v|h) for the score of token v, and let A be the advantage of the whole completion. The loss gives the per-position gradient estimate

    g_t = -A * w_t * s_{y_t}        (w_t = 1 without a correction)

Its conditional expectation over y ~ q, given h, is

    E_q[A s_y | h] = E_q[A | h] * sbar_q(h) + Cov_q(A, s_y | h),
    sbar_q(h) = sum_v q(v|h) s_v(h).

(Proof: Cov_q(A, s_y|h) = E_q[A s_y|h] - E_q[A|h] E_q[s_y|h], and E_q[s_y|h] = sbar_q(h).)

On-policy, q = p, and sbar_p(h) = sum_v p(v|h) grad log p(v|h) = grad sum_v p(v|h) = 0. Only the covariance survives, and it is the policy-gradient term we want.

Under mismatch, sbar_q(h) = sum_v (q - p)(v|h) s_v(h) != 0. The expected update then contains the drift

    D_t = E_h[ E_q[A | h_t] * sbar_q(h_t) ],

which is not caused by the reward. Its direction is the gradient of sum_h V(h) * sum_v q(v|h) log p_theta(v|h) with q held fixed, where V(h) = E[A | h]. Ascending it pushes p_theta toward q at prefixes with positive expected advantage and away from q at prefixes with negative expected advantage. V(h) is not zero for individual prefixes even though the group-centered advantages average to zero over samples. So the drift is systematic, not zero-mean noise. The sampler is refreshed from the trainer at every step, so the sampler's error (noise, quantization, staleness) is re-injected every step and the drift keeps accumulating.

Why importance sampling and clipping only slow this down: with a token-level weight w(y|h) = clip(p/q), the expected update is

    E_q[w_y A s_y | h] = E[A|h] * sum_v m(v|h) s_v + Cov_q(A, w_y s_y | h),
    m(v|h) = q(v|h) w(v|h).

Without clipping, m = p and the drift vanishes on-policy. Clipping or dropping tokens changes m away from p, so the drift returns with coefficient E[A|h]. A stronger clip moves m further from p, which is why stronger token-level clipping can make things worse.

## The fix: a score-function control variate

Define a per-position baseline

    b_t(theta) = sum_v m(v|h_t) log p_theta(v|h_t),     m(v|h) = q(v|h) w(v|h),

with w = 1 when there is no correction, so m = q. Its gradient is grad b_t = sum_v m(v|h_t) s_v(h_t), and it depends on h_t only. The loss becomes

    L(theta) = -(1/N) sum_b A_b sum_t [ w_bt log p_theta(y_bt | h_bt) - b_bt ].

Since b_t is a function of h_t alone, E_q[A * grad b_t] = E_h[ E[A|h_t] * sum_v m s_v ]. Taking the expectation of the new per-position gradient, given h_t:

    E_q[ A (w_y s_y - sum_v m(v|h) s_v) | h ]
      = E_q[A w_y s_y | h] - E[A|h] * sum_v m(v|h) s_v
      = Cov_q(A, w_y s_y | h).

The drift term cancels exactly for any token-level weight w(y|h) that does not depend on A.

With no correction (w = 1, m = q), this reads E_q[A (s_y - sbar_q) | h] = Cov_q(A, s_y | h). This is the on-policy covariance term, evaluated under the sampler's distribution, with no E[A|h] sbar_q drift term left. On-policy (q = p), sbar_p = 0 and the baseline has zero gradient in expectation, so the estimator reduces to the original one.

Summing over t and sequences, the expected update is sum_t E_q[Cov_q(A, w s | h_t)], with no drift term.

### Top-k truncation

The sampler returns only its top K = 20 logprobs per position, so sum_v m s_v is computed over the top-K support. The residual is R_t = sum_{v not in top-K} m(v|h_t) s_v. With G = max_v ||s_v(h)|| and tail mass tau_t = sum_{v not in top-K} q(v|h_t), we have ||R_t|| <= G * max w * tau_t, so the drift that remains is O(tail mass). This is zero when the vocabulary has at most K tokens.

### Implementation (losses.py)

- The sampler's top-K ids and logprobs are forwarded to the loss.
- With lq_k the sampler logprob of top-K token v_k, lp_k = log p_theta(v_k|h), and w_k = w(v_k|h) from the same `is_weight` as the sampled-token term, the per-position term is w_y log p_theta(y|h) - sum_k exp(lq_k) w_k lp_k.
- The stop-gradient on the mass means the gradient flows only through log p_theta.
- Without a correction, w = 1 and the baseline is sum_k q_k log p_k.

The modification is on by default. No configuration changes are needed.

### Limits

- Exactness holds for token-level corrections whose weight does not depend on the sign of A (`sign=both`). For the sign-dependent modes (`ppo`, `pos`, `neg`) the baseline uses the same sign convention, but the identity is only approximate.
- Sequence-level corrections (`seq_mean`, `seq_sum`) apply a weight to the whole sequence, not to individual tokens. The baseline there is the uncorrected one (m = q), so drift from the sequence weight is not covered.
- This is an argument about the expected update, not a proof of stability. It is a first-order statement at each position.

## Evidence on the CPU proxy (dev/mini.py, 600 steps)

These runs are short and proxy-scale. They are not a substitute for the Qwen runs.

- Default mismatch (noise 0.3), last-10 mean reward: original 0.58, 0.61, 0.55 (seeds 0, 1, 2); new 0.85, 0.84, 0.85. Peaks: original about 0.67, new 0.93 to 0.96.
- Heavy tail (200 extra never-correct tokens, so tail mass is large): original collapses to 0.00 (peak 0.33) on seed 0 and to 0.09 (peak 0.32) on seed 1. New: 0.69, 0.77, 0.67 (seeds 0, 1, 2), with no collapse.
- TIS-clamped correction (`rl.is.high=2.0 rl.is.outside=clamp`): original 0.62, 0.58; new 0.65, 0.63 (seeds 0, 1).
- On-policy (noise 0): original and new reward curves are identical (0.56 to 0.63), as the derivation predicts.
- A first version used the uncorrected baseline sum_v q log p even with a correction. With TIS it collapsed (peak 0.42, last-10 mean 0.005). This is why the baseline must use m = q w.
- Noise 0.6 is too strong for either loss: peak 0.34 for both. The fix does not rescue an overwhelming mismatch.

## Configuration

No configuration changes are needed; the modification is on by default.
