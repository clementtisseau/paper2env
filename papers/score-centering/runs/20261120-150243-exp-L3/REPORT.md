# Sampler-centered REINFORCE: removing the mismatch drift

## Summary

The REINFORCE loss `-A * log p(y_t | h_t)` assumes the sampled token comes
from the trained policy `p`. When it comes from a different sampler `q`, the
expected update at each prefix contains a **drift term**
`E_q[A | h] * sum_v q(v|h) grad log p(v|h)`. This term does not depend on
which token was good. It depends only on the average advantage at the prefix
and on the sampler error `q - p`. The fix subtracts that drift direction from
every score, using the sampler's top-20 logprobs plus a tail bucket:

    loss = -(1/N) sum_{b,t} A_b * [ w_t log p(y_t|h_t) - sum_{v in K_t} a_t(v) log p(v|h_t) ],

    a_t(v) = sg( m_t(v) - (m_T / P_T) * p(v|h_t) ),

Here `m(v) = q(v) w(v)` is the sampled mass of token `v` after the (optional)
IS correction. Without a correction, `m = q`. `K_t` is the sampler's top-k set,
and `P_T`, `m_T` are the trainer probability and the sampled mass of the tail
(the tokens outside `K_t`). The centering is always on (`loss_fn_rl` with
default arguments), works with `correction=None`, and composes with every
`rl.is.*` correction. When `q = p` it is exactly zero, so the on-policy loss
gradient does not change.

## 1. What is wrong with the current loss

Notation: `h` is a prefix (prompt plus completion tokens so far), `y ~ q(.|h)`
is the next token, `s_v = grad_theta log p(v|h)` is the score of token `v`, and
`A` is the advantage of the whole completion. Let

    Q(h,v) = E_q[A | h, y = v],      V(h) = E_q[A | h] = sum_v q(v|h) Q(h,v).

The loss gradient is `-(1/N) sum_t A s_{y_t}`. Its ascent direction at one
position, conditioned on the prefix, is

    E_q[A s_y | h] = sum_v q(v|h) Q(h,v) s_v
                   = V(h) * sbar_q(h)  +  sum_v q(v|h) (Q(h,v) - V(h)) s_v      (1)
                   = V(h) * sbar_q(h)  +  Cov_q(A, s_y | h),

    sbar_q(h) = sum_v q(v|h) s_v.

**On-policy** (`q = p`): `sbar_p = sum_v p grad log p = grad sum_v p = grad 1 = 0`.
Only the covariance term remains. It is the true policy gradient at `h`. It is
also invariant to any baseline: adding a constant `b` to `A` adds
`b * sbar = 0` to the update. Group-centering the rewards relies on exactly
this property.

**Mismatched** (`q != p`): `sbar_q(h) = sum_v (q(v|h) - p(v|h)) s_v != 0`.

Write it in logit space. For a softmax head with logits `z`,
`d log p(v)/d z_j = 1[v=j] - p_j`, so

    sbar_q(h) (wrt z) = sum_v q_v (e_v - p) = q(.|h) - p(.|h).               (2)

So the update contains `V(h) * (q - p)`.

- On prefixes where the completion tends to succeed (`V > 0`), it pulls the
  logits toward the sampler.
- On prefixes where it tends to fail (`V < 0`), it pushes them away from the
  sampler.

The reward signal plays no part in this, which has four consequences:

* **It does not vanish when there is no signal.** If every next token at `h` is
  equally good (`Q(h,v) = V(h)` for all `v`), the correct update is 0, but the
  loss still moves `p` by `V(h)(q - p)`. Equivalently, baselines are no longer
  free: `E_q[b s_y | h] = b * sbar_q(h)`. Group-centering subtracts a baseline,
  and per prefix that baseline turns into drift.
* **It is systematic, not noise.** `q - p` is the error of a deterministic
  transformation of the weights (fixed noise, quantization, KV quantization,
  staleness). Because `q = Phi(p)` is refreshed from the trainer every step,
  the error `q - p` is roughly the same at every step. The drift therefore adds
  up linearly over training instead of averaging out like zero-mean noise.
* **It grows the mismatch.** The gradient of `KL(q || p)` with respect to the
  logits is `p - q`. A drift step `dz = eta V (q - p)` changes the KL to first
  order by `<p - q, eta V (q - p)> = -eta V ||q - p||^2`, which is positive
  wherever `V < 0`. Failing prefixes are common: under `plus_minus` rewards and
  for hard prompts, `V < 0` on most of them. There the drift actively increases
  the trainer/sampler divergence, which enlarges `q - p`, which enlarges the
  drift. This positive feedback loop is the collapse.
* **It is unrelated to the variance of the gradient.** Larger batches do not
  remove a bias.

### Why IS corrections and clipping only slow it down

A detached weight `w(y)` turns (1) into `sum_v m(v) Q(h,v) s_v` with sampled
mass `m(v) = q(v) w(p(v)/q(v))`. With `M = sum_v m` and `V_m = sum m Q / M`:

    E_q[w A s_y | h] = V_m * sbar_m(h) + sum_v m(v)(Q - V_m) s_v,
    sbar_m(h) = sum_v m(v) s_v  =  m - M p   (in logit space).

* Exact IS (`m = p`) gives `sbar_m = 0`, but its weights `p/q` are unbounded,
  and every practical variant truncates them.
* Clamping (`TIS`), dropping (`MIS`, PPO-style bands, DPPO) and
  sign-dependent rules all set `m(v) != p(v)` on the tokens where the ratio
  leaves the band. The leftover drift is `V_m (m - M p)`, supported on those
  tokens. It is smaller than `V (q - p)`, so degradation is slower, but it is
  still systematic and still compounds.
* A **narrower** token band makes `m` differ from `p` on more tokens. Dropped
  tokens contribute `-V p_v` in full. This is why stronger token-level
  clipping can make things worse.
* Sequence-level weights rescale `A` per sequence. They cannot cancel a
  per-prefix term like `V(h) sbar(h)` and need heavy truncation anyway.

## 2. The fix: center every score by its drift under q

Define the per-prefix drift direction (detached coefficients, live score):

    d(h) = sum_v m(v|h) s_v,            m = q (no correction) or q * w.

Then use `A (w_y s_y - d(h))` instead of `A w_y s_y`. Its conditional
expectation is

    E_q[A (w_y s_y - d(h)) | h] = sum_v m(v) Q(h,v) s_v - V(h) sum_v m(v) s_v
                                = sum_v m(v) (Q(h,v) - V(h)) s_v.             (3)

Without a correction (`m = q`) this is exactly `Cov_q(A, s_y | h)`. The drift
term `V(h) sbar(h)` is gone. Its properties:

1. **Unchanged on-policy.** If `q = p`, then `d(h) = sum_v p s_v = 0`
   identically, for every sample and not only in expectation. The loss
   gradient is exactly the REINFORCE gradient. This is checked numerically
   against the original `losses.py`: the max gradient difference is
   `7e-9`.
2. **Baseline invariance is restored per prefix.** Replacing `A` by `A + b`
   adds `b (w_y s_y - d(h))`. Without a correction, its expectation under `q`
   is `b (sbar_q - sbar_q) = 0`. In particular the update is exactly zero in
   expectation at any prefix where the next token does not matter
   (`Q(h,.) = const`). It can no longer move the policy without reward
   information.
3. **The update is a reward-driven ascent direction.** In logit space, with
   `m = q`, (3) is

       dz_j = q_j (Q_j - V) - p_j sum_v q_v (Q_v - V) = q_j (Q_j - V),       (4)

   because `sum_v q_v (Q_v - V) = 0`. This is the on-policy update with the
   sampler's visit frequencies. It raises the logits of better-than-average
   tokens and lowers the worse ones, and it does nothing when all tokens are
   equal. Its first-order effect on the expected return under `p`,
   `sum_j p_j (Q_j - V_p) dz_j`, is
   `sum_j p_j q_j (Q_j - V_q)^2 + (V_q - V_p) sum_j p_j q_j (Q_j - V_q)`.
   The first part is nonnegative. The second part is proportional to
   `V_q - V_p = sum_v (q_v - p_v) Q_v`, an O(mismatch) quantity multiplied by
   the signal. In contrast, the removed drift is O(mismatch) multiplied by
   `V`, whether or not any signal is present.
4. **The feedback loop is broken.** The only term that systematically moved `p`
   along `+-(q - p)` was `V (q - p)`. Without it, the change in `KL(q||p)` comes
   only from the covariance term, which carries no fixed sign along `q - p`.
   The sampler error therefore stops being amplified step after step.
5. **Bounded and cheap.** In logit space `d(h) = m - M p`, so
   `||d||_1 <= 2M` (`M <= 1` without a correction). The coefficients are
   detached, so no new variance source appears. The extra cost is a gather
   of 20 logprobs per position, inside the existing chunked/rematerialized
   loop.

With a correction (`m = q w`), (3) removes the residual drift `V_m sbar_m`
that the clipping leaves. Any `rl.is.*` setting therefore also becomes
drift-free at the token level.

### Evaluating d(h) from the sampler's top-k

Only the sampler's top-k ids `K` and logprobs are available (k = 20). Split
the vocabulary into `K` and the tail `T`. Their sampler masses are
`q_v (v in K)` and `Q_T = 1 - sum_K q_v`, and their trainer masses are `p_v`
and `P_T = 1 - sum_K p_v`. Inside the tail, assume that the sampler is
proportional to the trainer: `q(v) ~= Q_T p(v) / P_T`. This is the
maximum-entropy completion consistent with `Q_T`, and it is exact when
`q = p`. Then

    sum_{v in T} q_v s_v = (Q_T / P_T) sum_{v in T} p_v grad log p_v
                         = (Q_T / P_T) grad P_T
                         = -(Q_T / P_T) sum_{v in K} p_v s_v,

because `grad P_T = -grad P_K`. So the whole drift folds into the top-k
coefficients:

    d(h) = sum_{v in K} a(v) s_v,      a(v) = m(v) - (m_T / P_T) p(v),

where `m_T` is the tail mass after the correction (`m_T = Q_T` without one).
In the loss this is the surrogate `sum_{v in K} sg(a(v)) log p(v|h)`, which is
subtracted from the sampled token's term.

- On-policy, `a(v) = p_v - (P_T/P_T) p_v = 0` exactly, including the tail.
- Off-policy, the only error is the deviation of `q` from proportionality to
  `p` inside the tail. Its logit-space size is at most twice the total
  variation between the two inside the tail, which is at most `2 Q_T`
  (small, since the top 20 cover almost all of the sampler's mass).
- The ratio `m_T / P_T` is capped at `1e3` to protect float32 when
  `P_T ~ 0 < Q_T`. Its gradient form stays bounded, because
  `m_T * grad log P_T` has logit-space norm `<= 2 m_T`.
- When the vocabulary is no larger than k, the tail is empty and the
  formula is exact.
- Sampler logprobs that sum to more than 1 (bf16 rounding, or garbage at
  non-completion positions, which are masked anyway) are renormalized.

The implementation was checked against a brute-force computation of
`sum_v q~(v) grad log p(v)` over the full vocabulary, with `q~` the top-k plus
proportional tail distribution. The max gradient difference is `2e-8`.

## 3. Implementation (`losses.py`)

* `drift_coefficients(logp_vocab, topk_ids, topk_logq, positive, c)` computes
  `a(v)` over the top-k ids. It reuses `log_mass`, so the sampled mass `m`
  follows whatever token-level correction `c` is active: `q` for none, and
  `q * w(p/q)` with the band, clamp/drop and sign rules otherwise. The tail
  bucket goes through the same `log_mass`.
* Inside `loss_fn_rl`, `sampler_topk_ids` / `sampler_topk_logprobs` are now
  forwarded (`sampler_keys`). The per-token score becomes
  `w * logp - sum_K sg(a) * logp_K`. Advantages, normalization and
  sequence-level corrections are unchanged. If the batch has no top-k
  logprobs, the loss falls back to the original one.
* A new metric `drift_l1` reports `sum_K |a(v)|` per token (0 on-policy). It
  measures how much drift was removed.

## Configuration

No configuration change is required. The centering is active by default with
`rl.is.level=null` (no IS correction), which is the recommended setting. It
also applies on top of any `rl.is.*` correction.
