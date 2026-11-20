# Drift-free REINFORCE under a sampler/trainer mismatch

## Summary

At every prefix, REINFORCE under a mismatched sampler q adds a term to the
expected update that has nothing to do with the reward signal. That term is
`E_q[A | h] * grad_theta[-CE(q(.|h), p_theta(.|h))]`: it distils the trainer toward the sampler
when the expected advantage at the prefix is positive and pushes it away from
the sampler when the expected advantage is negative. Because the sampler is
rebuilt from the trainer at every step through the same deterministic
distortion (fixed weight noise, quantization, stale weights), the term points
in a consistent direction from step to step. It does not average out. It
compounds.

The fix is to center the score at every position under the sampler's own
next-token distribution. The per-token objective becomes

    A * ( w_y log p(y|h)  -  sum_v m(v|h) log p(v|h) ),     m(v|h) = sg[ q(v|h) w(p(v|h)/q(v|h)) ]

where w is the detached IS weight of the active correction (w = 1 with
`correction=None`, so m = q). The second term depends only on the prefix,
not on the sampled token. Its gradient is exactly the drift direction, so the
expected update at each prefix becomes a pure covariance between advantage
and score. On-policy (q = p) the subtracted term has zero gradient, so the
estimator is identical to plain REINFORCE. `m` is built from the sampler's
top-20 logprobs, which the loss now forwards (`sampler_topk_ids`,
`sampler_topk_logprobs`). The remaining sampler mass is handled as one merged
"tail" token, and the approximation stays exact on-policy.

## 1. What is wrong with the current loss

Notation: prefix h; next token y ~ q(.|h) (the sampler); trainer p_theta;
score s_v(h) = grad_theta log p(v|h); A is the (group-centered) advantage of the
whole completion. The current loss is `-(1/N) sum_t A * w_t * log p(y_t|h_t)`.
Its gradient is a sum over positions of `A w_t s_{y_t}(h_t)`.

### 1.1 Per-position decomposition

Fix h and take expectations over y ~ q(.|h) and over the continuation that
determines A. Write `Abar_v(h) = E_q[A | h, y=v]` and `Abar(h) = E_q[A | h] = sum_v q(v|h) Abar_v(h)`.
For plain REINFORCE (w = 1):

    E_q[A s_y | h] = sum_v q(v|h) Abar_v(h) s_v(h)
                   = Abar(h) * sbar_q(h)  +  sum_v q(v|h) (Abar_v(h) - Abar(h)) s_v(h)
                   = Abar(h) * sbar_q(h)  +  Cov_q(A, s_y | h),

    sbar_q(h) = sum_v q(v|h) grad log p(v|h).

On-policy, `sbar_p(h) = sum_v p grad log p = grad sum_v p = grad 1 = 0`, so only
the covariance remains. That is the policy-gradient signal: it raises tokens
that are better than average at h and lowers the worse ones. When q != p,
`sbar_q(h) != 0` and the first term is a **drift**.

### 1.2 What the drift does

Write the logits at h as z, with p = softmax(z). Then
`d log p(v|h) / d z_u = 1[u=v] - p(u|h)`. In logit coordinates:

    sbar_q(h)  ->  sum_v q(v) (e_v - p) = q(.|h) - p(.|h)
    sbar_q(h)   =  grad_theta sum_v q(v|h) log p(v|h) = -grad_theta CE(q(.|h), p(.|h))

so the drift is

    Abar(h) * grad_theta [ -CE(q(.|h), p_theta(.|h)) ].

- If `Abar(h) > 0`, the trainer is distilled toward the sampler at h.
- If `Abar(h) < 0`, the trainer is anti-distilled. It is pushed away from q,
  i.e. extrapolated along p - q.

This holds even when every token at h is equally good (`Abar_v = Abar` for
all v). In that case the covariance vanishes, so there is no reward signal at
all, but the drift does not vanish. The update is "learn the sampler's error"
with a sign that depends on how well the rollout is going.

With group-centered rewards, `E[A] = 0` only at the prompt. At deeper prefixes
`Abar(h)` is far from zero: a prefix that already contains a mistake has
`Abar(h) < 0`, and a prefix on a good track has `Abar(h) > 0`. With
`plus_minus` rewards `Abar(h)` has the sign of (success rate - 1/2) everywhere.
The drift is therefore present at almost every position. Its size is
`|Abar(h)| * ||q - p||`. That does not shrink as the covariance signal shrinks,
and it is summed over all T positions of the completion.

### 1.3 Why it compounds

The sampler is `q = D(p_theta)`, where D is a fixed distortion: theta + delta
for a fixed delta, int8/fp8 rounding, KV quantization, or theta from k steps
ago. So `q - p ~ J_D(theta)` changes slowly and keeps the same direction over
training. The drift is therefore a systematic bias of the objective, not
zero-mean noise:

- Each step moves theta along `Abar * (-grad CE(D(theta), theta))`.
- The anti-distillation phase (`Abar < 0`, at failing prefixes or early in
  training) is a positive feedback. It moves p away from D(p). D is then
  re-applied to the new p, which moves q further away again, and so on.

Plain REINFORCE is not stable under this. KL(q||p) grows, the entropy
moves, and the reward eventually collapses.

### 1.4 Why IS corrections and clipping only slow it down

With a per-token detached weight w(r), r = p/q:

    E_q[A w s_y | h] = Abar(h) E_q[w s_y | h] + Cov_q(A, w s_y | h),

    drift = Abar(h) E_q[w s_y | h] = Abar(h) * sum_v m(v|h) s_v(h),   m(v|h) = q(v|h) w(p(v|h)/q(v|h)).

- With the exact ratio (w = r, no band), `m = p` and `sum_v p s_v = 0`, so the
  per-position drift disappears. But r is unbounded, the variance is huge, and
  the advantage of the rest of the sequence is still drawn from q.
- Every practical method truncates w (TIS, MIS, PPO/DAPO bands, DPPO, TOPR).
  Wherever w is clamped or dropped, `m != p`, and the residual drift is
  `Abar(h) * sum_v (m(v) - p(v)) s_v`, i.e. `Abar(h)(m - p)` in logit space.
- Tighter token-level clipping makes m deviate from p on more tokens. In the
  `drop` case m moves toward 0, and in the `clamp` case toward q. Either way the
  residual drift grows. That is why stronger token clipping can make training
  worse.
- Clipping and IS also only rescale the term `A * s_y` of the sampled token.
  They never see the other tokens of the distribution, so they cannot cancel
  the sum over the vocabulary that defines the drift. That needs the
  sampler's next-token distribution.

## 2. The modification

### 2.1 Score centering under the sampler

At each completion position t (prefix h_t), subtract a per-prefix control
variate from the per-token objective:

    L = -(1/N) sum_{b,t} mask * A_b * ( w_t log p(y_t|h_t)  -  B(h_t) ),
    B(h) = sum_v m(v|h) log p(v|h),     m = stop_grad(q * w(p/q))   (m = q when correction=None).

Gradient: `grad B(h) = sum_v m(v|h) s_v(h)`. This is exactly the drift
direction of section 1.4, because m is detached.

### 2.2 Expected update with the modification

For a fixed prefix h:

    E_q[ A (w_y s_y - grad B(h)) | h ]
      = E_q[A w_y s_y | h]  -  E_q[A|h] * sum_v q(v) w_v s_v
      = E_q[A w_y s_y | h]  -  E_q[A|h] * E_q[w_y s_y | h]
      = Cov_q( A, w_y s_y | h ).

The drift term is removed exactly, at every prefix, for any w (any
`ISCorrection`, including None). For `correction=None` this gives

    Cov_q(A, s_y | h) = sum_v q(v|h) (Abar_v(h) - Abar(h)) s_v(h),

and in logit coordinates `dz_u ∝ q(u|h) (Abar_u(h) - Abar(h))`. This is the
same form as the on-policy policy gradient: each token's logit moves in
proportion to how much better than average it is. The only differences are
that the token weights are the sampler's and the baseline is the sampler's
average. It has the property that plain REINFORCE loses under mismatch:

- **No reward signal, no update.** If all tokens at h are equally good
  (`Abar_v = Abar`), the expected update at h is exactly 0, whatever q is. The
  trainer is never pulled toward or away from the sampler's distortion. The
  fixed points of the learning dynamics are the same as on-policy, so the
  systematic bias that compounded through the sampler refresh is gone.

What remains of the mismatch is a reweighting: tokens weighted by q instead
of p, and prefixes visited with q's distribution. This changes how fast each
reward-improving direction is followed, but it does not add a direction that
is unrelated to reward. With q close to p the reweighting is first-order
small. The drift, by contrast, is first-order in ||q - p|| times
|E[A|h]|, with no reward contrast needed, and it accumulates.

### 2.3 Unchanged on-policy

If q = p (and w = 1, or w in-band), `grad B(h) = sum_v p grad log p = 0`, so
the gradient is exactly REINFORCE's, not just in expectation. The
modification costs nothing when there is no mismatch.

### 2.4 Effect on variance

`A * B(h)` uses no sampled-token quantity beyond A and the prefix. It
subtracts the conditional mean of `w_y s_y` given h. In logit space each
position's term goes from `A (w_y e_y - w_y p)` to `A (w_y e_y - m)`, and the
subtracted vector is bounded (`||m||_1 <= max w`). The variance therefore does
not increase meaningfully, and it removes a component that is coherent across
all positions and samples.

## 3. Using only the sampler's top-k

The loss only sees the sampler's top-K (K = 20) ids and logprobs at each
position. Let K(h) be those ids, `q_T = 1 - sum_{v in K} q(v)` the sampler's
leftover mass, and `P_T = 1 - sum_{v in K} p(v)` the trainer's mass on the
complement. The tail is treated as **one merged token** with sampler mass q_T
and trainer log-probability `log P_T`. This assumes q is proportional to p
inside the tail:

    B(h) = sum_{v in K} m(v) log p(v|h)  +  m_T log P_T(h),
    m(v) = sg(q(v) w(p(v)/q(v))),   m_T = sg(q_T w(P_T/q_T)).

Properties:

- **Exact drift on the head.** In logits, `dB/dz_u = m(u) - p(u)` for u in K and
  `m_T p(u)/P_T - p(u)` for u in the tail. The error compared with the exact
  `sum_v m(v) s_v` is confined to how q distributes its tail mass inside the
  tail, which is at most q_T.
- **Still exact on-policy.** If q = p:
  `grad B = sum_K grad p(v) + P_T grad log P_T = grad (sum_K p + P_T) = grad 1 = 0`.
- **Same weights as the correction.** m and m_T use `log_mass` of the active
  `ISCorrection` (band, clamp/drop, sign, TV statistic). That is the measure
  under which the residual drift of that correction lives (section 1.4). The
  modification therefore composes with every row of the README table. With
  sequence-level corrections the token-level w is 1, so m = q. The sequence
  weight multiplies A, so the expected update is `Cov_q(w_seq A, s_y | h)`,
  which is again drift-free.
- **Numerics.** `log P_T` is evaluated as `log(max(1 - sum_K p, 1e-6))`. When
  the tail is below 1e-6 its log is not differentiated, which bounds the
  gradient. `q_T` is clipped to [0, 1]. If the vocabulary is smaller than K, the
  head is the whole vocabulary and the tail term vanishes.
- **Fallback.** If a batch carries no top-k keys, the loss reduces to the
  original one.

## 4. Implementation (`losses.py`)

- `sampler_keys` now also forwards `sampler_topk_ids` and
  `sampler_topk_logprobs` into the chunked per-token computation.
- New function `sampler_baseline(logp_vocab, topk_ids, topk_logq, positive, c)`
  returns B(h) per position. It gathers `log p` on the sampler's top-k ids
  (with `take_vocab`, so it works with a TP-sharded vocabulary), builds the
  detached masses m with the existing `log_mass` (m = q for `correction=None`),
  and adds the merged-tail term.
- In `chunk_sums` the per-token score becomes `w * logp - B(h)`, and it is
  multiplied by A as before. The function is active with default arguments.
  Signature and metrics are unchanged.

A sanity check of the implementation: with q = p, `grad_z B` is around 1e-8. With q != p,
`grad_z B` matches `q - p` up to the tail-shape error.

On the CPU dev setting (`./run_dev.sh`, seed 0, 600 steps), training reward
over the last 10 steps went from 0.58 (plain REINFORCE, which plateaus and
slowly degrades) to 0.85. Eval reward rose steadily to about 0.83–0.86, against
about 0.66–0.71 for the baseline.
With 40 extra never-correct tokens (`MINI_EXTRA_TOKENS=40`, so the top-20
truncation and the tail term are exercised), plain REINFORCE collapsed to
reward 0 by step 100. The modified loss reached a last-10 reward of 0.85 and
an eval reward of 0.84. Combined with TIS (`high=2, clamp`) in the same setting,
it reached an eval reward of 0.89. Seed 1 of the dev setting gave a last-10
reward of 0.83 and no collapse.
