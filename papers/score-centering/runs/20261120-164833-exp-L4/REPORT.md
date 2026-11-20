# Drift-free REINFORCE under a sampler/trainer mismatch

## Summary

The REINFORCE loss scores each sampled token with `s_y = grad log p(y|h)` and
weights it by the advantage `A`. That is unbiased only if `y ~ p`. When the
rollouts come from a different sampler `q`, the score is no longer zero-mean
at a fixed prefix. Every token then picks up a systematic **drift** term,
`E_q[A|h] * s_bar(h)`. This term does not depend on which token was good. It
pushes `p` toward (or away from) `q` in proportion to the prefix's expected
advantage, and it compounds because the sampler is re-synced to the trainer
at every step.

The fix centers the score under the sampler at each position:
`s_y -> s_y - s_bar(h)` with `s_bar(h) = sum_v q(v|h) grad log p(v|h)`. Its
conditional expectation is exactly the covariance `Cov_q(A, s_y | h)`, which
is the only term present on-policy. It uses no importance ratios, adds nothing
when `q = p`, and needs only the top-20 sampler logprobs that are already
returned. It is implemented in `losses.py` and is always on.

## 1. What is wrong with the current loss

Notation: prefix `h`, next token `y ~ q(.|h)`, completion advantage `A`
(a function of the whole completion, so of `y` and of what follows it), and
token score `s_v = grad_theta log p(v|h)`. The loss is
`L = -(1/N) sum_t A * log p(y_t|h_t)`, so its update direction at one
position is `A s_y`.

**Decomposition.** For a fixed prefix, with `mu(h) = E_q[A|h]`:

    E_q[A s_y | h] = sum_v q(v|h) E[A | h, v] s_v
                   = mu(h) * sum_v q(v|h) s_v  +  sum_v q(v|h) (E[A|h,v] - mu(h)) s_v
                   = mu(h) * s_bar(h)          +  Cov_q(A, s_y | h).

**On-policy** (`q = p`): `s_bar(h) = sum_v p_v grad log p_v = grad sum_v p_v = grad 1 = 0`.
Only the covariance remains. It moves probability toward the tokens whose
continuation has above-average advantage, which is the policy gradient. The
first term is also why any baseline that depends only on `h` is harmless
on-policy.

**Mismatched** (`q != p`): `s_bar(h) = sum_v (q_v - p_v) grad log p_v != 0`.
In logit space (`grad_z log p_v = e_v - p`) this is simply

    s_bar(h) = q(.|h) - p(.|h)          (gradient w.r.t. the logits z at h).

So the drift is `mu(h) (q - p)`. Gradient ascent with it moves `p` **toward
`q`** where the expected advantage is positive and **away from `q`** where it
is negative, independently of which tokens were actually good. Three
consequences follow:

* **It is a bias, not noise.** It survives averaging over any number of
  samples, because it is a conditional expectation.
* **It is not removed by centering rewards per group.** Group centering makes
  `E[A] ~ 0` over a prompt's completions, but not `E[A | h]` at each prefix.
  After a good or bad partial completion, `mu(h)` is far from 0. That is
  exactly where the model is confident, and where the drift is applied
  consistently across the rollouts.
* **It compounds.** With `mu(h) < 0` (frequent with `plus_minus`-style
  rewards, or on prefixes that already went wrong), the update pushes `p`
  away from `q`, which is `p` plus a fixed perturbation. The sampler is then
  rebuilt from the new `p`, and the perturbation (weight noise, quantization,
  staleness) maps it to a new `q` at the same systematic offset. The drift
  therefore points in a consistent direction step after step. In logit space,
  `-(q - p)` pushes up exactly the tokens that the sampler under-weights
  relative to the trainer. Typically these are tail tokens that quantization
  or noise squash or inflate. Mass leaks into them, entropy and the
  train/sampler KL grow, the mismatch itself grows, and training collapses.

**Why IS corrections and clipping do not fix it.** With a token IS weight
`w(r)`, `r = p_y/q_y`, the update is `w A s_y`:

    E_q[w A s_y | h] = mu_w(h) * sum_v q_v w_v s_v + Cov_q(...),

with `mu_w(h)` the advantage averaged under the reweighted measure. The drift
vanishes only if `q_v w_v = p_v` for **every** `v`, that is, untruncated IS.
That has unbounded variance (the ratios on tail tokens explode), and in
practice it cannot even be applied on the tail because `q_y` of rare tokens is
noisy. Any clamp or drop (TIS, MIS, PPO, ...) leaves `q_v w_v != p_v` exactly
on the tokens where `p/q` is far from 1, which are the mismatched tokens. The
residual drift is still `mu * sum_v (q_v w_v - p_v) s_v`. Clamping
`w <= c` replaces `p_v` by `c q_v` on tokens with `p_v > c q_v`, so the
residual stays aligned with `q - p` on those tokens. Tighter token clipping
puts more tokens out of band and can make the residual drift larger. It only
slows the degradation down. Sequence-level ratios do not help at the token
level either: they multiply `A` by a per-sequence scalar, which leaves the
`s_bar(h) != 0` factor untouched.

## 2. The fix: center the score under the sampler

Replace the score of the sampled token by its centered version, with the same
advantage:

    g_t = A * ( s_{y_t} - s_bar(h_t) ),      s_bar(h) = sum_v q(v|h) s_v.

**Unbiased for the covariance, at every prefix:**

    E_q[A (s_y - s_bar) | h] = E_q[A s_y | h] - mu(h) s_bar(h) = Cov_q(A, s_y | h).

The drift term is removed exactly, for any `q`, without ratios. Equivalently,
`s_y - s_bar` has zero mean under `q`, so a prefix-dependent advantage level
`mu(h)` produces no update. That is the property the on-policy score has under
`p`.

**On-policy it changes nothing:** `q = p` gives `s_bar = 0`, so `g_t` is the
original REINFORCE update. (Checked numerically: identical gradients.)

**What remains is the right signal.** In logit space
`s_y - s_bar = e_y - q`. That is the score of `log q` itself. So the update
raises the logit of the sampled token relative to the sampler's distribution,
in proportion to how much better than average its continuation was. The
covariance `Cov_q(A, s_y|h) = sum_v q_v (E[A|h,v] - mu(h)) s_v` depends on `q`
only through which tokens are visited and how they are weighted, never through
a systematic push along `q - p`. Its fixed point condition, "no token has an
above-average continuation", is the same as on-policy. The mismatch can only
change the weighting of real reward signal, a second-order effect, and not
create a direction of its own. In particular the update no longer contains a
component that grows `KL(q||p)`, which closes the feedback loop described
above.

**Variance.** `s_bar(h)` is the exact conditional mean of `s_y`, so subtracting
it is a control variate. The update stays `A` times a zero-mean, bounded
vector (`||e_y - q|| <= sqrt 2` in logit space). Unlike importance weights,
nothing can blow up.

### Single-backward surrogate

With `sg` the stop-gradient:

    loss_t = -A * ( log p(y_t|h) - sum_v sg[q(v|h)] log p(v|h) ),
    grad loss_t = -A * ( s_{y_t} - sum_v q_v s_v ) = -A (s_y - s_bar).

### Only the top-k head of q is known

The sampler returns its top-20 logprobs at each position (head `H`). Model
the unknown tail as proportional to `p`:
`q_v = rho p_v` for `v not in H`, with `rho = q_tail / p_tail`,
`q_tail = 1 - sum_{v in H} q_v`, `p_tail = 1 - sum_{v in H} p_v`. Then,
using `sum_{all v} p_v s_v = 0`,

    s_bar = sum_{v in H} q_v s_v + rho sum_{v not in H} p_v s_v
          = sum_{v in H} q_v s_v - rho sum_{v in H} p_v s_v
          = sum_{v in H} (q_v - rho p_v) s_v,

so

    loss_t = -A * ( log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p(v|h) ).

The sampled token keeps its own `log p(y_t|h)` even when it lies outside `H`.
In logit space the head approximation gives `s_bar = q - p` on `H` and
`(rho - 1) p_v` on each tail token. That is exact whenever `q`'s tail is
proportional to `p`'s, and its error is at most the tail mass of `q` that is
not proportional to `p`. When `H` covers the whole support, `q_tail = 0`,
`rho = 0`, and the exact formula is recovered. (Checked numerically against a
full-vocabulary centering: identical with a full head, and within 3e-3
relative error with a 20-of-50 head.)

**Numerics.** `p_tail` is clamped at `1e-12`, and `rho` is capped at `1e3`.
If the cap binds, the tail mass that is not modeled (at most `q_tail`) is
spread along `p` over the whole vocabulary, which is the zero-gradient
direction. So the cap can only remove drift, never add it. Head entries with
non-finite logprobs (padding) get zero weight, and ids are clipped into the
vocabulary range.

### Combination with IS corrections (optional)

With a correction, the update is `w_y A s_y`, whose drift is
`mu * sum_v m_v s_v` with the reweighted measure
`m_v = q_v w(p_v/q_v)` (= `exp(log_mass(...))` in the code). The same argument
gives

    loss_t = -A * ( w_y log p(y_t|h) - sum_v sg[m_v] log p(v|h) ),
    E_q[grad] = -Cov_q(A, w s | h),

with the head computed exactly and the tail (where `p/q = 1/rho`) given mass
`rho * w(1/rho) * p_v`. Sequence-level weights multiply `A` by a
per-sequence scalar `W`. The centered score still gives `Cov_q(W A, s_y | h)`.
With `correction=None`, `m = q` and the formulas above apply. The default and
recommended configuration is no correction: the centering alone removes the
drift, and the ratios only add variance.

## 3. Implementation (`losses.py`)

`loss_fn_rl(hidden, head, batch, correction=None, chunk=128)` keeps its
signature and chunked, rematerialized structure. It now also forwards
`sampler_topk_ids` / `sampler_topk_logprobs` (added to `sampler_keys`).
Inside each chunk it gathers `log p` at the head ids (`take_vocab`, so it
works with TP sharding) and forms `coef = sg(m_H - rho * w_tail * p_H)`. It
then uses `w_y log p_y - sum_H coef * log p_H` as the per-token score. The
modification is active by default. If the batch has no top-k head, the loss
reduces to the original one. Extra metrics: `q_tail` and `rho`.

## 4. Sanity checks (CPU dev setting, `dev/mini.py`, seed 0, 600 steps)

* Gradient with `q = p` is identical to the original loss. With a full head,
  it matches full-vocabulary centering to 7e-6.
* Default mismatch: original loss plateaus (last-10 reward 0.58, eval 0.66).
  Centered loss reaches 0.87 / 0.83.
* `MINI_EXTRA_TOKENS=200` (top-20 no longer covers the support): original
  loss collapses to 0 reward by step 25. Centered loss reaches 0.85 / 0.80.
  With TIS added, it reaches 0.63 / 0.87 and does not collapse.

## Configuration

No `rl.is.*` changes are needed (the defaults leave no correction).
