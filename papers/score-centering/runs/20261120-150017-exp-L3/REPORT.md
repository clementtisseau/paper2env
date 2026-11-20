# Sampler-centered REINFORCE: removing the mismatch drift

## Summary

The loss now subtracts, at every completion position, the **sampler's mean score**
from the score of the sampled token:

    loss = -(1/N) sum_{b,t} A_b * [ w_t * log p(y_t|h_t)  -  sum_v sg(m(v|h_t)) * log p(v|h_t) ]

with `m = q` (the sampler's next-token distribution) when no correction is used,
and `m = q * w(p/q)` when a token-level IS correction `w` is used. `q` is read from
the sampler's top-20 logprobs (`sampler_topk_ids`, `sampler_topk_logprobs`) and
extended off the top-20 proportionally to `p`. The subtracted term is a control
variate that is identically zero on-policy and exactly cancels the
**drift** term `E_q[A|h] * s_bar(h)` that makes mismatched REINFORCE
unstable. It is active with default arguments (`correction=None`); no
configuration change is needed.

In logit space the change is one line: the per-token gradient w.r.t. the logits
`z(h)` of the trainer goes from `A (e_y - p)` to `A (e_y - q)`.

## 1. What is wrong with the current loss

Notation: prefix `h`, next token `y ~ q(.|h)` (sampler), trainer `p_theta(.|h)`,
score `s_v(h) = grad_theta log p(v|h)`, advantage `A` of the whole completion.
The current loss (no correction) has per-token gradient `A s_y(h)`. Decompose its
conditional expectation at a fixed prefix:

    E_q[A s_y | h] = E_q[A|h] s_bar_q(h) + Cov_q(A, s_y | h),
    s_bar_q(h)     = sum_v q(v|h) s_v(h).

**On-policy** (`q = p`): `s_bar_p = sum_v p grad log p = grad sum_v p = grad 1 = 0`.
Only the covariance survives: tokens are reinforced exactly as much as they
correlate with the advantage. A prefix where the advantage does not depend on
the next token produces zero expected update, whatever the level of `E[A|h]`.

**Mismatched** (`q != p`): `s_bar_q(h) != 0`. In logit space
(`grad_z log p(v) = e_v - p`):

    s_bar_q(h) = sum_v q(v) (e_v - p) = q(.|h) - p(.|h)        (in logit coordinates)

so the expected update contains the term

    D(h) = E_q[A|h] * (q(.|h) - p(.|h)) ,

which is the gradient of `E_q[A|h] * ( -KL(q||p) )` w.r.t. the logits. It does not
depend on which token was good: it is present even when `A` is independent of
`y`, i.e. when there is no reward signal at that position. It is a **drift**:

* at prefixes with `E_q[A|h] > 0` it pulls `p` toward `q`;
* at prefixes with `E_q[A|h] < 0` it pushes `p` *away* from `q`, i.e. it
  increases the mismatch itself.

Why this is a systematic bias and why it compounds:

* Group-centering makes `E[A] = 0` per prompt, but **not** `E[A|h]` per prefix:
  prefixes inside failing completions have `E[A|h] < 0`, those inside
  succeeding completions `E[A|h] > 0` (with `plus_minus` rewards, as in the
  mini setting, `E[A|h]` is non-zero even on average). Summed over a batch, the
  drift `sum_h E[A|h] (q - p)(h)` does not average out.
* The mismatch is a fixed transformation of the weights (fixed noise
  `delta`, quantization grid, KV quantization, staleness), so `q - p` points
  in nearly the same direction step after step: `q_theta - p_theta ~ J_theta delta`.
  The drift is therefore not zero-mean noise across steps; it adds up
  coherently (linearly in the number of steps), while the sampler is refreshed
  from the drifted trainer, so the next step starts from the drifted point
  and sees the same kind of bias again. In particular on failing prefixes,
  `p` is moved along `p - q`, which makes the trainer's logits move in a direction
  determined by the perturbation instead of the task; this accumulates into
  degraded or collapsed policies (entropy blow-up / garbage tokens with large
  `-log p`, which then get large `|A| s_y` gradients).

**Why IS corrections / clipping only slow it down.** With a token weight `w(r)`,
`r = p/q`, the expected update is

    E_q[w A s_y | h] = E_q[A|h] * sum_v q(v) w(v) s_v + Cov-type term.

The drift vanishes only if `q w = p` exactly for all `v`, i.e. untruncated IS
over the *whole vocabulary*. Any clamp/drop band (TIS, MIS, PPO, ...) makes
`q w != p` on the tokens outside the band, leaving a drift
`E_q[A|h] * sum_{v outside} (q w - p)(v) s_v`. Tighter token clipping puts more
tokens outside the band, which is why it can make things worse. Moreover
untruncated IS is unusable (variance), and the correction acts only on the
sampled token whereas the drift is a property of the whole next-token
distribution. Sequence-level weights rescale `A` but cannot cancel a per-prefix
term either.

## 2. The fix: subtract the sampler mean score (exact control variate)

Replace the score by its **sampler-centered** version

    s~_y(h) = s_y(h) - s_bar_q(h),         s_bar_q(h) = sum_v q(v|h) s_v(h)  (q detached).

Since `s_bar_q(h)` is a function of the prefix only (not of `y`),

    E_q[A s~_y | h] = E_q[A s_y | h] - E_q[A|h] s_bar_q(h) = Cov_q(A, s_y | h).

The drift term is removed exactly, for any `q`, at every prefix; what remains is
the covariance between the advantage and the token, i.e. the reward signal only.
Properties:

1. **On-policy it changes nothing**: `q = p` gives `s_bar = 0`, so the gradient is
   the original REINFORCE gradient (verified numerically to 1e-8).
2. **No signal, no update**: if `A` is independent of `y` at `h`, the expected
   update is zero, as on-policy. So the trainer is no longer pushed along the
   mismatch direction; the "ratchet" that compounds over refreshes is gone.
3. **Logit-space form**: the per-position update is `A (e_y - q)` instead of
   `A (e_y - p)`. Its `q`-expectation is
   `E_q[A e_y] - E_q[A] q = sum_v q(v) (E_q[A|h,v] - E_q[A|h]) e_v`: each token's
   logit moves by its own advantage relative to the sampler-average, the
   exact analogue of the on-policy update with `p` replaced by the distribution
   the data came from. Fixed points of this update (in expectation) are exactly
   the points where no token has a non-zero conditional advantage, as on-policy.
4. **Variance**: the control variate is a deterministic function of `h`
   and does not add sampling noise; it removes the `E[A|h] (q-p)` component,
   whose size is O(TV(q,p)).

It is a baseline in score space rather than in advantage space: the usual
`A - b(h)` baseline is unbiased only because `E_p[s_y] = 0`, which fails under
`q`; here we restore `E_q[s~_y | h] = 0` directly, so any advantage
(group-centered, plus/minus, ...) is admissible.

What remains (and is not drift): `Cov_q(A, s_y|h)` differs from `Cov_p` because
tokens are visited with `q`'s frequencies and continuations are sampled by `q`.
This is a reweighting of a genuine reward signal (each token's update still has
the sign of its advantage relative to the sampler mean), it vanishes when there
is no signal, and it is second order in the mismatch for the direction of
improvement. Sequence-level effects of the mismatch therefore enter only
through `A`'s estimate, not as a systematic pull.

### With an IS correction

With a token-level correction `w`, the expected update at `h` is
`E_q[A|h] * sum_v q(v) w(v) s_v + Cov_q(A, w s_y)`, so the matching
centering uses the mass `m(v) = q(v) w(p(v)/q(v))` (what `log_mass` computes):

    score = w_y s_y - sum_v m(v) s_v,      E_q[A * score | h] = Cov_q(A, w s_y | h).

For untruncated IS `m = p` and the subtracted term is `sum_v p s_v = 0` (nothing
changes, as it should); for clipped/dropped weights it removes exactly the drift
left by the band. For sequence-level corrections the per-token centering with
`m = q` is used and the whole centered score is multiplied by the sequence weight.

## 3. Estimating s_bar_q from the top-20 logprobs

The sampler only returns its top-K (K = 20) logprobs: `q(v)` for `v in K(h)`, with
mass `Q = sum_{v in K} q(v)`. Let `P = sum_{v in K} p(v)` and `P_tail = 1 - P`.
Off the top-K we extend `q` proportionally to `p` (the only information available
is that the tail has mass `1-Q`):

    q~(v) = q(v)                         v in K
    q~(v) = p(v) (1-Q) / P_tail          v not in K       (sum_v q~ = 1)

Then

    sum_v q~(v) s_v = sum_{v in K} q(v) s_v + (1-Q) * sum_{v not in K} (p(v)/P_tail) s_v
                    = sum_{v in K} q(v) grad log p(v)  +  (1-Q) grad log P_tail,

because `grad log P_tail = sum_{v not in K} p(v) s_v / P_tail`. So the surrogate is

    B(h) = sum_{v in K} sg(q(v)) log p(v|h)  +  sg(1-Q) * log P_tail(h),

`P_tail` being computed as `logsumexp` of the trainer's log-probs with the top-K
entries masked out (no `1 - P` cancellation). Properties: exact when the sampler's
tail is proportional to the trainer's tail; exactly zero gradient on-policy
(then `q~ = p`); and in logit space its gradient is `q~ - p`, so the
per-position update is `A (e_y - q~)`. With a token correction the tail ratio
`p/q~ = P_tail/(1-Q)` is constant over the tail, giving tail mass
`(1-Q) w(P_tail/(1-Q))`.

The surrogate enters as `B - sg(B)`, so the reported loss value is unchanged and
only the gradient is modified. The loss is still normalized by the number of
completion tokens. If the batch has no `sampler_topk_*` keys the loss falls back
to plain REINFORCE (nothing to center with).

## Implementation (losses.py)

* `sampler_keys` now forwards `sampler_topk_ids` and `sampler_topk_logprobs`.
* `sampler_mean_score(...)` builds `B(h)` per position (inside the rematerialized
  vocab chunks), `drop_vocab` masks the top-K ids to get `log P_tail`.
* `score = w * logp - B` replaces `w * logp`.
* New metrics: `drift_tv` = TV(q~, p) per token (size of the removed drift) and
  `tail_q` = sampler mass outside its top-20.

Sanity checks: on-policy gradients equal the original loss's to ~4e-8; under a
random mismatch the logit gradient equals `-A (e_y - q~)/N` to ~4e-9; finite
gradients with all corrections and when the top-K covers the whole vocabulary.
On the CPU dev setting (`./run_dev.sh`, noise 0.3), the original loss plateaued
around train reward 0.58 / eval 0.66-0.72 with growing sampler-trainer KL; the
centered loss rose steadily to train 0.85 / eval 0.83-0.86. With 40 extra
never-correct vocabulary tokens (`MINI_EXTRA_TOKENS=40`, so the top-20 does not
cover the vocabulary and the tail extrapolation is exercised), the original loss
collapsed to reward 0 within 100 steps, while the centered loss reached eval
0.80-0.86 (seed 0 and seed 1), and 0.895 eval combined with TIS (`high=2 clamp`).

## Configuration

No configuration change is needed (works with the default `correction=None`).
