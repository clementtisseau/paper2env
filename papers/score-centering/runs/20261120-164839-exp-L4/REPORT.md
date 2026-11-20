# Sampler-centered REINFORCE: removing the mismatch drift

## Summary

The loss now replaces each token's score `s_y = ∇ log p(y|h)` with the
*sampler-centered* score `s_y − s̄(h)`, where `s̄(h) = Σ_v q(v|h) ∇ log p(v|h)`
is the mean score under the sampler's next-token distribution `q(·|h)`. The
advantage multiplying it stays the same. On-policy (`q = p`) we have `s̄ = 0`
identically, so the update does not change. Under a mismatch, the term removed is
exactly the part of the expected update that comes from the mismatch and not
from the reward. It is a deterministic function of the prefix, so it does not
average out across steps, and it is what makes training collapse. The fix
needs no importance ratios. It is on by default whenever the batch carries
the sampler's top-k logprobs, which `train_rl.py` always provides.

## 1. What is wrong with the current loss

Notation: `p = p_θ` is the trainer, `q` is the sampler (a perturbed, quantized
or stale copy of `p`), `h` is a prefix, `y ~ q(·|h)` is the next token, `A` is
the advantage of the whole completion, and `s_v(h) = ∇_θ log p(v|h)`.

The current loss is `L = −(1/N) Σ_{i,t} A_i log p(y_{i,t}|h_{i,t})`, so its
update direction is `Σ_t A s_{y_t}(h_t)`. Fix a position. Conditioning on the
prefix and taking the expectation over `y ~ q(·|h)` and over the continuation
(which is also sampled from `q`):

    E_q[A s_y | h] = Σ_v q(v|h) E_q[A | h, v] s_v(h)
                   = E_q[A|h] · s̄(h)  +  Cov_q(A, s_y | h),                   (1)

    s̄(h) = Σ_v q(v|h) s_v(h).

**On-policy.** `s̄(h) = Σ_v p(v|h) ∇ log p(v|h) = ∇ Σ_v p(v|h) = ∇ 1 = 0`.
Only the covariance term is left. It raises the log-probability of tokens
whose conditional advantage `E[A|h,v]` is above the prefix average
`E[A|h]` and lowers the others. This is the policy-gradient signal. Any
per-prefix baseline `b(h)` drops out, which is why REINFORCE tolerates
uncentered rewards such as `±1`.

**Mismatched.** `s̄(h) ≠ 0`. Because `s_v = ∇ log p(v|h)`,

    s̄(h) = ∇_θ Σ_v q(v|h) log p(v|h) = −∇_θ [ H(q) + KL(q(·|h) ‖ p(·|h)) ] = −∇_θ KL(q ‖ p)(h),

so the first term of (1) is

    drift(h) = −E_q[A|h] · ∇_θ KL(q(·|h) ‖ p_θ(·|h)).                        (2)

At every prefix with positive expected advantage the update pulls `p` toward
`q`. At every prefix with negative expected advantage it pushes `p` away from
`q`. This term does not depend on which token the reward favoured. It only
depends on the mismatch `q − p` and on the advantage averaged over the
prefix. Its properties:

* **It is not zero-mean noise.** Given `h`, (2) is deterministic. Averaging
  over many samples and steps does not cancel it, while the covariance
  signal is noisy. The ratio of bias to signal grows as the signal shrinks
  (for example, when the reward saturates and `Cov(A, s_y|h)` → 0 while
  `E[A|h]` stays non-zero at bad prefixes).
* **Group-centering does not remove it.** Centering makes `E[A] = 0` per
  prompt, not per prefix. Prefixes that already contain a mistake have
  `E_q[A|h] < 0`, and prefixes on the right track have `E_q[A|h] > 0`.
  With uncentered rewards (`±1`, `0/1`) the drift is present at every
  position. Centering only makes it smaller.
* **It compounds.** The sampler is rebuilt from the trainer at every step
  (`q_{k} = Q(p_{θ_k})`, with `Q` = add noise / quantize / delay). A drift
  that pushes `p` away from `Q(p)` is therefore applied again at every step
  in the same direction relative to the current weights. Unlike the
  covariance term, nothing pulls it back. As `p` moves away from the region
  where `Q(p) ≈ p` (for example, its logits become less robust to the
  perturbation), `KL(q‖p)` grows, `‖s̄‖` grows, and the drift grows with it.
  This is the positive-feedback loop behind the observed collapse: the
  KL between sampler and trainer rises, then the reward falls.

**Why IS corrections and clipping do not fix it.** A token-level correction
multiplies the score by a detached weight `w = w(p_y/q_y)`:

    E_q[w A s_y | h] = E_q[A|h] · Σ_v q_v w_v s_v  +  Cov-like term.

The drift vanishes only when `q_v w_v ∝ p_v` for every `v`, which happens
when `w = p/q` exactly, without truncation (then
`Σ_v q_v w_v s_v = Σ_v p_v s_v = 0`). Truncating, clamping or dropping
ratios (TIS, MIS, PPO/DAPO bands, DPPO) breaks `q_v w_v = p_v` precisely
on the tokens where the mismatch is largest. The remaining
`Σ_v q_v w_v s_v` still points along the mismatch, and a tighter band leaves
more of it. This is why stronger token-level clipping can make things worse.
Untruncated IS removes the drift at the token level but has unbounded
variance, and its sequence-level form is worse still. Sequence-level
weights rescale `A` but leave `s̄(h)` unchanged at every position.

## 2. The fix: subtract the sampler's mean score

Replace `s_y` by `s_y − s̄(h)` and keep the same `A`:

    g = Σ_t A · ( s_{y_t}(h_t) − s̄(h_t) ).

At a single position,

    E_q[A (s_y − s̄(h)) | h] = Σ_v q_v E[A|h,v] s_v − E_q[A|h] s̄(h)
                           = Cov_q(A, s_y | h).                              (3)

The drift (2) is removed exactly. Only the reward-driven covariance term
remains, at every prefix and for any advantage (centered or not). No
importance ratio appears, so there is no variance blow-up and nothing to
clip. Properties:

* **On-policy it is the identity.** `s̄(h) = 0` for `q = p`, so `g` equals
  the REINFORCE gradient sample by sample, not only in expectation.
* **It is a control variate in score space.** `E_q[s_y − s̄(h) | h] = 0` by
  construction, so the update has no component that depends on `E[A|h]`.
  Equivalently, it is REINFORCE with a perfect per-prefix baseline, applied
  to scores that are recentered for the distribution that actually
  generated `y`. In the on-policy case this centering is automatic. The
  usual baseline argument ("a baseline does not change the expected
  gradient") relies on `E_q[s_y|h] = 0`, which fails exactly when `q ≠ p`.
  The fix restores that identity.
* **The fixed point is right.** If the reward carries no information at a
  prefix (`E[A|h,v] = E[A|h]` for all `v`), the expected update there is 0
  regardless of `q − p`. The old loss would still have moved `p` along
  `∓∇KL(q‖p)` there. The feedback loop through the refreshed sampler
  therefore has nothing to amplify.
* **What remains of the mismatch** is the fact that `Cov_q` uses `q`'s
  token frequencies and the advantages of `q`'s continuations. This
  changes the relative weight of directions that the reward supports, so it
  is a second-order, signal-proportional effect, not a drift. For small
  mismatches, `Σ_v (q_v − p_v)(E[A|h,v] − E[A|h]) s_v` is `O(‖q−p‖ · signal)`.
  It goes to zero when the signal goes to zero.

### Single-backward-pass surrogate

With `sg` = stop-gradient, the per-token loss is

    ℓ_t = −A · ( log p(y_t|h_t) − Σ_v sg[q(v|h_t)] log p(v|h_t) ).

Its gradient is `−A (s_{y_t} − Σ_v q_v s_v) = −A (s_{y_t} − s̄(h_t))`. The
added term is a weighted sum of log-softmax entries at the same position,
so it costs one gather of K entries from the already computed
`log_softmax`, and it fits inside the existing rematerialized vocabulary
chunks.

### Only the top-k of q is known

The sampler returns its top-K (K = 20) logprobs per position: a head `H`
with `q_v` known for `v ∈ H`, and an unknown tail of mass
`q_T = 1 − Σ_{v∈H} q_v`. Model the tail as proportional to `p`:

    q_v ≈ ρ p_v  for v ∉ H,   ρ = q_T / p_T,   p_T = 1 − Σ_{v∈H} p_v.

This tail has the right total mass. Within the tail it has the trainer's
shape, which is the only shape available, and it is exact on-policy. Then,
using `Σ_{all v} p_v s_v = 0`, i.e. `Σ_{v∉H} p_v s_v = −Σ_{v∈H} p_v s_v`,

    s̄(h) = Σ_{v∈H} q_v s_v + ρ Σ_{v∉H} p_v s_v = Σ_{v∈H} (q_v − ρ p_v) s_v.

So the tail never has to be enumerated, and the surrogate is

    ℓ_t = −A · ( log p(y_t|h_t) − Σ_{v∈H} sg[q_v − ρ p_v] log p(v|h_t) ).

Checks:
* On-policy, `H` is `p`'s own top-K, `q_v = p_v`, and `ρ = 1`, so every
  coefficient is 0. The loss is unchanged exactly, including the top-K
  approximation.
* If the head covers everything (`q_T = 0`), then `ρ = 0` and the formula
  is the exact `Σ_v q_v s_v`.
* The sampled token keeps its own `log p(y_t|h_t)` even when `y_t ∉ H`.
  Only the subtracted mean uses the head.

Because `Σ_v p_v ∇log p_v = 0` holds exactly, the gradient of the head term
is `Σ_{v∈H} c_v (∇z_v − Σ_u p_u ∇z_u)` with logits `z`, and the tail logits
get gradient `−p_u Σ_{v∈H} c_v`. So the tail is still pushed as a block,
consistently with the model `q_tail ∝ p_tail`.

## 3. Implementation (`losses.py`)

* `loss_fn_rl` now forwards `sampler_topk_ids` / `sampler_topk_logprobs`
  (added to `sampler_keys`). Inside each vocabulary chunk,
  `sampler_mean_score_surrogate` gathers `log p` on the sampler's head ids
  and computes `q_v = exp(sampler logprob)`, `q_T = clip(1 − Σ_H q_v, 0, 1)`,
  `p_T = −expm1(logsumexp_H log p)`, `ρ = clip(q_T / max(p_T, 1e-6), 0, 100)`,
  and the detached coefficients `c_v = q_v − ρ p_v`. The per-token score
  used in the loss is `log p(y_t) − Σ_H c_v log p(v)`.
* This is active with the default arguments (`correction=None`). No flag
  or configuration is needed. If the top-k keys are absent (another caller
  without them), the loss falls back to plain REINFORCE.
* With a `correction`, the IS weight `w` multiplies the centered score, so
  the corrections still work. They are no longer needed to remove the
  drift, and the recommended setting is no correction (the default).
* Numerical guards: `p_T` is floored at 1e-6 (the float32 resolution of
  `1 − Σ_H p`), and `ρ` is capped at 100. Both only matter when the tails
  are negligible, where the tail term is negligible too. Padded positions
  after EOS (top-k logprobs of 0) are removed by the mask.
* New metrics: `q_tail` (sampler mass outside the top-k) and `tail_rho`.

Sanity checks done: with `q = p` the gradient matches the old loss to 1e-8.
With a perturbed `q` and a vocabulary of 30 / top-20, the top-k surrogate's
gradient matches the full-vocabulary `−A(s_y − Σ_v q_v s_v)` to about 3e-4
(relative to a gradient of size 0.1). The uncorrected loss differs from it
by 0.04.

## 4. Dev runs (CPU mini setting, for reference)

Default `dev/mini.py` setting (noise 0.3, ±1 rewards, 600 steps). These
numbers are final training reward, mean of the last 10 steps.
* seed 0: old 0.58 → new 0.84. seed 1: old 0.61 → new 0.86.
* 200 extra vocabulary tokens (top-20 no longer covers q), seed 2: the old
  loss collapses to 0.00 (eval 0.00 from step 100). The new loss reaches
  0.74 (eval rises steadily to about 0.72).

## Configuration

No configuration changes are needed. The modification is active by default,
and the recommended setting keeps the default `correction=None` (no `rl.is.*`
keys).
