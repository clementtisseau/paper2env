# Sampler-centered REINFORCE: removing the mismatch drift

## 1. What is wrong with the current loss

`loss_fn_rl` minimises `-(1/N) sum_i A_i sum_t w_{i,t} log p(y_{i,t} | h_{i,t})`, with the tokens
`y ~ q` drawn by the sampler, `p = p_theta` the trainer, and `w` an optional detached IS weight
(`w = 1` without a correction). Its gradient is a sum of per-position terms `A * w * s_y`, where
`s_v(h) = grad_theta log p(v|h)` is the score of token `v` at prefix `h`.

Fix a prefix `h` and take the expectation over the next token `y ~ q(.|h)`. Write
`m(v|h) = q(v|h) w(v|h)` for the measure that actually draws *and* weights the sampled token
(`m = q` without correction). Then

    E[A w s_y | h] = sum_v m(v|h) E[A | h, v] s_v(h)
                   = E[A|h] * s_bar(h)  +  sum_v m(v|h) (E[A|h,v] - E[A|h]) s_v(h)

    with  s_bar(h) = sum_v m(v|h) grad_theta log p(v|h).

(For `w = 1` this is `E_q[A|h] s_bar(h) + Cov_q(A, s_y | h)`.)

* **On-policy** (`m = q = p`): `s_bar(h) = sum_v p grad log p = grad sum_v p(v|h) = grad 1 = 0`.
  Only the second term survives; it is non-zero only when the choice of `v` changes the expected
  advantage. This is the reward signal.
* **Mismatch** (`m != p`): `s_bar(h) = sum_v (m(v|h) - p(v|h)) s_v(h) != 0`. The first term,
  `E[A|h] s_bar(h)`, does not depend on *which* token was good: it is present at every prefix,
  even where all tokens lead to the same advantage. It is a drift.

What the drift does. For a softmax head with logits `z`, `d log p(v)/d z_u = 1[u=v] - p(u)`, so the
logit-space drift at `h` is

    E[A|h] * sum_v m(v) (e_v - p) = E[A|h] * (m - M p),    M = sum_v m(v) (= 1 for m = q),

i.e. the update pushes `p(.|h)` **towards `q` where the expected advantage is positive and away from
`q` where it is negative**, at every prefix, with a size proportional to `E[A|h]` and to the
mismatch. With group-centered advantages `E[A|h]` is zero only at the prompt (averaged over the
group); conditionally on any deeper prefix it is not (prefixes of good completions have
`E[A|h] > 0`, prefixes of failing ones `E[A|h] < 0`), so the drift is active along every rollout.
With `plus_minus` rewards it is active everywhere.

Why it compounds instead of averaging out. The drift is not zero-mean noise: for a fixed sampler
perturbation (weight noise, int8/fp8, KV quantisation) `q - p` has a consistent direction, and since
`q` is rebuilt from the updated trainer every step, the trainer is pushed along `q - p` (or `p - q`
on failing prefixes) step after step, and the perturbed sampler of the new trainer is again offset
in the same way. On failing prefixes (`E[A|h] < 0`) the update moves `p` *away* from `q`, which, since
the sampler mismatch is largest on low-probability tokens, inflates the tail / changes the
distribution in a reward-independent direction, increases `KL(q||p)`, which increases the drift:
a positive feedback loop that ends in collapse.

Why IS / clipping do not fix it. With a token weight `w = p/q` exactly, `m = p` and the drift
vanishes; but truncated / clipped / dropped weights (TIS, MIS, PPO, DAPO, DPPO ...) give `m = q w(p/q)`,
which equals `p` only inside the band, so `s_bar = sum_v (m - p) s_v` is non-zero exactly on the
tokens where the mismatch is largest. Clipping more strongly makes `m` *further* from `p` there,
which is why stronger token-level clipping can make it worse. Sequence-level weights rescale `A` but
leave the per-token measure `q` unchanged, so the drift is only rescaled. In addition, only the
sampled token's ratio is used, which cannot correct the contribution of the unsampled tokens to
`s_bar`.

## 2. The modification

Center the score of every token under the measure that drew it:

    s_y   ->   s_y - s_bar(h),         s_bar(h) = sum_v m(v|h) grad log p(v|h)  (m detached)

i.e. the per-token surrogate becomes

    L = -(1/N) sum_i A_i sum_t [ w_{i,t} log p(y_{i,t}|h_{i,t}) - sum_v sg(m(v|h_{i,t})) log p(v|h_{i,t}) ].

### Derivation: the drift is removed exactly, nothing else changes

Expected update at a prefix:

    E[A (w s_y - s_bar) | h] = E[A|h] s_bar + sum_v m (E[A|h,v] - E[A|h]) s_v - E[A|h] s_bar
                             = sum_v m(v|h) (E[A|h,v] - E[A|h]) s_v(h).

For `w = 1` this is exactly `Cov_q(A, s_y | h)`: the covariance term, the part of the update that
carries information about which token is better. Properties:

1. **On-policy it is the unchanged REINFORCE.** If `q = p`, `w = 1`: `s_bar = grad sum_v p = 0`
   identically (not only in expectation), so the gradient is bit-for-bit the original one.
   The modification only acts through the mismatch.
2. **No reward-independent motion.** If at a prefix all tokens lead to the same expected advantage
   (`E[A|h,v] = E[A|h]` for all v), the expected update is exactly 0, whatever `q` is. The original
   loss moves `p` towards/away from `q` there. So the trainer is no longer driven along `q - p`, and
   the feedback loop of section 1 is broken: the only systematic direction is the reward signal.
3. **It is a control variate along the action, not a bias.** `s_bar(h)` is a function of the prefix
   only; multiplying it by `A` and subtracting it changes the conditional mean exactly by the drift
   term and nothing else. In logit space the per-token update becomes
   `A (e_y - m)` (for `w=1`) instead of `A (e_y - p)`: the "baseline distribution" subtracted from
   the one-hot is the one the token was sampled from, so `E_q[e_y - q] = 0`: the per-prefix
   update has zero mean under the sampler, as on-policy REINFORCE has zero mean under `p`.
4. **It is an exact ascent direction, for the sampler's objective.** Work in logit space at one
   prefix, `p = softmax(z)`, `q = softmax(z')`, `a_v = E[A|h,v]`, `a_q = sum_v q_v a_v`. Since
   `d log p(v)/dz = e_v - p`, the centered per-token logit update is
   `A [(e_y - p) - sum_v q_v (e_v - p)] = A (e_y - q)`, with expectation

       E_q[A (e_y - q)] = q * (a - a_q)   (elementwise)  =  grad_{z'} sum_v softmax(z')_v a_v ,

   the exact on-policy policy gradient *of the sampler* at that prefix. The original loss gives
   `E_q[A (e_y - p)] = q * (a - a_q) + a_q (q - p)`: the same signal plus the drift `a_q (q - p)`,
   which moves `p` along `q - p` with a sign set by the average advantage and nothing else.
   For mismatches of the type studied (the sampler's logits are those of the trainer plus a
   perturbation, `z' = z + delta`, with `delta` changing slowly compared to `z`), a step
   `z <- z + eta q*(a - a_q)` moves `z'` by the same amount, so it is gradient ascent on the expected
   advantage of the deployed sampler: `dJ_q = eta sum_v q_v^2 (a_v - a_q)^2 >= 0`. Its inner product
   with the on-policy trainer gradient `p*(a - a_p)` is `sum_v p_v q_v (a_v - a_p)(a_v - a_q)`, which
   reduces to `sum_v p_v^2 (a_v - a_p)^2 >= 0` as `q -> p`. The remaining discrepancy (weighting the
   covariance by `q` instead of `p`) vanishes when the reward signal vanishes, so unlike the drift it
   cannot push `p` anywhere on its own.
5. **Variance.** Centering the score is the standard variance-reducing step (it is what the softmax
   gradient already does on-policy); it does not add variance.
6. **With a correction.** With token IS weight `w` the subtracted measure is `m = q w(p/q)`, so
   the same derivation holds for every point of the `rl.is.*` grid; for exact IS (`w = p/q`)
   `m = p` and the term is identically 0, recovering exact IS. With sequence-level corrections the
   sequence weight multiplies `A`, and the token-level centering under `q` is kept.

### Computing `s_bar` from the sampler's top-k

The sampler returns its top-k (k = 20) logprobs `log q(v|h), v in K(h)`. Since the coefficients are
detached, `s_bar(h) = grad_theta sum_v sg(m(v|h)) log p(v|h)`, which needs `m` on all of the
vocabulary. On `K` we have it. For the tail `v not in K` we use the exact identity
`sum_v p(v) s_v = 0`, i.e.

    sum_{v not in K} p(v) s_v = - sum_{v in K} p(v) s_v,

and assume `m` is proportional to `p` on the tail, with the tail mass `M_tail` of `m`
(`M_tail = 1 - sum_K q` for no correction). Then
`sum_{v not in K} m(v) s_v = (M_tail / P_tail) sum_{v not in K} p(v) s_v = -(M_tail/P_tail) sum_{v in K} p(v) s_v`,
with `P_tail = 1 - sum_K p`, and

    s_bar(h) ~= sum_{v in K} a_v s_v(h),     a_v = m(v|h) - (M_tail / P_tail) p(v|h).

So the loss only needs `log p` at the 20 top-k ids (one gather from the full log-softmax already
computed in each chunk). Properties of this estimator:

* exact if `q` is proportional to `p` outside the top-k (in particular exact when the vocabulary is
  smaller than k, as in `dev/mini.py`);
* exactly 0 on-policy (`m = p` gives `a_v = p_v - p_v = 0`), so property 1 holds exactly, not
  approximately; a plain renormalised top-k (`a_v = q_v / sum_K q`) would *introduce* a drift
  `-sum_{v not in K} p s_v` on-policy;
* the residual error is bounded by the tail mass of `|m - p|` outside the top-20, which for an LLM
  is small.

For numerical safety `P_tail` is floored at `1e-6` (`1 - sum p` loses precision in float32) and the
ratio `M_tail / P_tail` is capped at 10; invalid top-k entries (non-finite, positive, or out of the
completion mask) get coefficient 0. With a correction, `M_tail = (1 - sum_K q) w(P_tail / Q_tail)`
(the band applied to the aggregated tail ratio).

## 3. Implementation (`losses.py`)

* `loss_fn_rl` forwards `sampler_topk_ids` and `sampler_topk_logprobs` (added to `sampler_keys`).
  When they are in the batch, the centering is on by default, with `correction=None` or with any
  `ISCorrection`; when they are absent the loss is the original one.
* In each chunk: `logp_k = take_vocab(logp_vocab, topk_ids)`; `centering_coeffs` computes the
  detached `a_v` (`m_v = q_v w(p_v/q_v)` via the existing `log_mass`, so every `rl.is.*` setting gets
  the correct measure, minus the tail term); the per-token score becomes
  `w * log p(y) - sum_k a_k log p(v_k)`. Everything else (advantages, sequence weights,
  normalisation by the number of tokens) is unchanged.
* New metric `drift_mass = sum_k |a_k|`, the L1 size of the centering measure (about the TV
  distance between `m` and `p` on the top-k), which tracks the size of the removed drift.

No configuration changes are needed (it works with the default `correction=None`).

## 4. Sanity checks (dev/mini.py, CPU)

* On-policy (sampler = trainer) the gradient is identical to the original loss (max abs diff 0.0),
  with and without `rl.is.*` corrections.
* Default setting (noise 0.3, plus_minus): original loss plateaus at eval ~0.70; centered loss
  climbs steadily to ~0.83-0.86. With 40 extra unused tokens (vocab > top-20, so the tail term is
  exercised) the original loss collapses to reward 0 by step 100, the centered loss reaches ~0.8
  with no collapse; also stable with 200 extra tokens, and combined with TIS or PPO-style corrections.
