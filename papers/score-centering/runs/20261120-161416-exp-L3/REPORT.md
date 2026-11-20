# Removing the mismatch drift from REINFORCE

## Summary

When the sampler q differs from the trainer p, the REINFORCE loss has a drift
term in its expected update. At every prefix, the score of the sampled token
has a nonzero mean under q, and that mean gets multiplied by the expected
advantage at the prefix. The drift does not come from the reward signal.
Because the sampler is copied from the trainer at every step, the bias builds up
over training. The fix subtracts, at every completion position, the score
expected under the sampler:

    per token:   A * ( w_y grad log p(y|h)  -  g(h) ),     g(h) = sum_v q(v|h) w_v grad log p(v|h)

Here w is the IS weight (w = 1 with no correction). g(h) depends only on the
prefix, so this is a baseline that the sampler makes necessary. It is computed
from the sampler's top-20 logprobs, with the rest of the vocabulary treated as
one aggregated tail event. On-policy g = 0, so the loss is exactly REINFORCE.
Under mismatch the expected update becomes a pure covariance between advantage
and token: if the sampled token carries no reward information, the expected
update is exactly zero.

## 1. What is wrong with the current loss

Notation: the prefix is h, the next token y ~ q(.|h), and the advantage of the
completion is A. The token score is s_v = grad_theta log p(v|h). The loss
`-(A * sum_t log p(y_t|h_t))` updates along A s_y at each position. Conditionally
on h:

    E_q[A s_y | h] = E_q[A|h] * sbar_q(h) + Cov_q(A, s_y | h),
    sbar_q(h)      = sum_v q(v|h) s_v.

**On-policy (q = p).** `sbar_p = sum_v p grad log p = grad sum_v p(v|h) = grad 1 = 0`.
Only the covariance remains. The update moves p only through how the
advantage varies with the chosen token. The level of the advantage at a prefix
(E[A|h]) has no effect, which is why REINFORCE needs no per-prefix baseline in
expectation.

**Mismatch (q != p).** `sbar_q = sum_v (q_v - p_v) s_v = grad_theta sum_v q_v log p_v = -grad_theta KL(q || p)`
(with q held fixed), which is not zero. So the update contains

    drift(h) = -E_q[A|h] * grad_theta KL(q(.|h) || p(.|h)).

The view in logit space makes this concrete. For softmax logits z at h,
`d log p_v / dz = e_v - p`, so the logit update at h is

    sum_v q_v Q_v (e_v - p) = q ⊙ Q - p * V_q,      Q_v = E_q[A | h, y=v],  V_q = E_q[A|h].

Assume the token carries no information (Q_v ≡ c). The update is then c (q - p),
not 0. The logits move towards the sampler when the prefix tends to succeed
(c > 0) and away from it when it tends to fail (c < 0).

Why this is a systematic bias and not noise:

* The sign of the drift is set by E_q[A|h], which is not zero at a given
  prefix, even with group-centered rewards. Centering makes the advantage
  average zero over a prompt's group. It does not make it zero at each prefix:
  prefixes after a mistake have E[A|h] < 0, and good prefixes have E[A|h] > 0.
  With uncentered rewards (plus_minus etc.), E[A|h] is not zero even on
  average.
* The direction q - p is not random. For a fixed perturbation (noise,
  quantization, KV-cache rounding, staleness), q is a deterministic function
  of p, and q - p has a consistent structure, typically a flatter distribution
  with extra mass on unlikely tokens. On failing prefixes the drift pushes p
  away from q along that structure. On succeeding prefixes it pulls p towards
  the sampler's errors. Both effects feed back, because the next sampler is
  q' = N(p'), so its mismatch with p' grows. This is the compounding collapse.
  Once the reward signal is weak, the drift is all that is left, so it decides
  where training ends up.

**Why IS corrections and clipping only slow it down.** With a detached weight
w_v = w(p_v/q_v), the expected update at h is

    E_q[A w_y s_y | h] = E_q[A|h] * g(h) + Cov_q(A, w_y s_y | h),   g(h) = sum_v q_v w_v s_v.

Exact token IS (w = p/q) gives `g = sum_v p_v s_v = 0`, but its variance is
unbounded, and practical methods truncate or drop the weights. Truncated weights
give `g = sum_v min(q_v c, p_v) s_v` (TIS), or the sum restricted to the band
(MIS/PPO-style drop). These are not zero: they differ from 0 by the probability
mass that clipping removed, where q and p disagree most. The tighter the band,
the more of the q weighting returns (in the limit w ≡ 1, g = sbar_q), which
explains why stronger token-level clipping can make things worse. Sequence-level
weights multiply A by a scalar and leave sbar_q unchanged.

## 2. The modification

Subtract the drift exactly, using a per-prefix control variate built from the
sampler's distribution:

    L = -(1/N) sum_i sg(A_i) sum_t [ w_{i,t} log p(y_t|h_t) - sum_v sg(q(v|h_t) w_v) log p(v|h_t) ]

Its gradient per token is `A (w_y s_y - g(h))`. With a sequence-level correction,
A is replaced by `A * w_seq` throughout, as in the original loss.

### Properties (derivation)

1. **Exact drift removal.** g(h) is a function of h alone and equals E_q[w_y s_y | h], so

       E_q[A (w_y s_y - g(h)) | h] = E_q[A w_y s_y | h] - E_q[A|h] E_q[w_y s_y | h] = Cov_q(A, w_y s_y | h).

   The term proportional to E_q[A|h] has gone. With the sampled A as its
   coefficient this is an unbiased removal (E_q[A g | h] = E_q[A|h] g).

2. **No update without signal.** If A is independent of y given h, the
   covariance is 0, so the expected update at h is 0 for any q. The original
   loss moves by c (q - p) in logit space in this case. The fixed point "no
   reward information means no movement", which on-policy REINFORCE has, is
   restored under any mismatch.

3. **Correct direction.** With w = 1, the expected logit update at h is

       sum_v q_v (Q_v - V_q)(e_v - p) = q ⊙ (Q - V_q) - p * sum_v q_v (Q_v - V_q) = q ⊙ (Q - V_q).

   Each logit moves by q_v (Q_v - V_q): tokens that are better than the
   sampler's average at h go up and worse ones go down. This is the on-policy
   update p ⊙ (Q - V_p) with p replaced by q in the weights. The remaining
   difference from the on-policy direction is
   `(q - p) ⊙ (Q - V)`. This is second order: it is the product of the mismatch
   with the advantage signal, and it disappears when either one does. It has
   the same sign structure as the true update, so it cannot drive the policy on
   its own.

4. **Unchanged on-policy.** When q = p and w = 1, g = grad sum_v p_v = 0, and the
   loss gradient is exactly the REINFORCE gradient (checked numerically: max
   gradient difference 7e-9).

5. **Compatible with every IS correction.** The q-weights become the
   "corrected masses" `q_v w_v`, which `log_mass` already computes (including
   clamp/drop bands, TV bands and sign-dependent weights). Whatever weight is
   used, the subtraction removes the E_q[A|h] g(h) term that clipping leaves
   behind.

### Computing g(h) from the sampler's top-k

The sampler returns its top-K (K = 20) logprobs, giving ids k ∈ K(h) and q_k.
The rest of the vocabulary has known total sampler mass
`Q_T = 1 - sum_{k∈K} q_k` but unknown per-token values. Split

    g(h) = sum_{k∈K} q_k w_k s_k + sum_{v∉K} q_v w_v s_v.

The head is computed exactly. The tail is treated as a single event
"y ∉ K", with trainer probability `P_T = 1 - sum_{k∈K} p_k` and sampler
probability Q_T. Its score is `grad log P_T`, and its corrected mass is m_T,
the `log_mass` of the aggregate ratio P_T/Q_T:

    g(h) ≈ sum_{k∈K} q_k w_k grad log p_k + m_T grad log P_T,    m_T = Q_T w(P_T/Q_T).

Justification: `grad log P_T = sum_{v∉K} (p_v/P_T) s_v`. The approximation is
therefore exact whenever q restricted to the tail is proportional to p
restricted to the tail. Its error, `sum_{v∉K} (q_v - Q_T p_v/P_T) s_v`, involves
only the shape of q inside the tail, which is low-mass by construction. Two
cases are exact:
* on-policy: `sum_k p_k grad log p_k + P_T grad log P_T = grad(sum_k p_k + P_T) = 0`;
* exact IS: `m_T = P_T`, and `P_T grad log P_T = grad P_T = sum_{v∉K} p_v s_v`.

The tail term matters. Mismatch typically moves mass into many unlikely tokens
(the tail), and the sampled token can itself lie in the tail. The `p_v`
log-probabilities of the top-k ids are gathered from the trainer's
full-vocabulary log-softmax that the loss already computes chunk by chunk, so
the cost is one [B, chunk, K] gather per chunk.

Numerical guards: P_T is computed as 1 - sum_k p_k in float32. The tail term
is dropped when P_T or Q_T is below 1e-6, where cancellation makes log P_T
unreliable and its weight is negligible. It is also dropped when the top-k ids
are not distinct, for example when the vocabulary is smaller than K. In that
case the head covers the whole distribution and the tail is empty. If the
batch has no top-k keys, the loss falls back to the original one.

## 3. Implementation (`losses.py`)

* `sampler_keys` now also forwards `sampler_topk_ids` and
  `sampler_topk_logprobs`.
* `expected_score_surrogate` returns, per position, the surrogate
  `sum_k sg(q_k w_k) log p_k + sg(m_T) log P_T`, whose gradient is g(h). The
  masses come from the existing `log_mass` evaluated with each token's
  (or the tail's) own ratio and the token-level correction.
* In `loss_fn_rl`, the per-sequence sum of this surrogate is subtracted from
  the per-sequence score sum, and the result is multiplied by the (detached,
  possibly sequence-weighted) advantage. Normalization (by the number of
  completion tokens), metrics and the signature are unchanged. The change is
  active with the default arguments (`correction=None`).

Sanity checks (CPU, `dev/mini.py`):
* With a full-vocabulary top-k, the baseline gradient matches
  `sum_v q_v grad log p_v` to 3e-7 relative error. With 40 vocabulary entries
  and K = 20, the tail approximation is within 1.2% relative error.
* Default mini setting (noise 0.3), 2 seeds: last-10 reward was 0.57 / 0.58
  with the old loss and 0.86 / 0.85 with the new one.
* With 40 extra tail tokens, the old loss collapsed to reward 0 by step
  50–100 on both seeds. The new loss reached a peak of 0.92 / 0.94 and stayed
  near 0.83–0.85 with no collapse.
* Group-centered rewards and TIS also run stably with the new term.

## Configuration

None. The modification is active with the default (no IS correction) and
also works with any `rl.is.*` correction.
