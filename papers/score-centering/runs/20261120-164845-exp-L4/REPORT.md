# Sampler-centered REINFORCE: removing the mismatch drift

## 1. What is wrong with the current loss

`loss_fn_rl` minimises `-A * log p(y_t | h_t)` for tokens `y_t ~ q(.|h_t)`, where `q` is the sampler and `p = p_theta` is the trainer.
Fix one position, with prefix `h`, and write `s_v = grad_theta log p(v|h)`. The expected update at that position is

    E_q[A s_y | h] = sum_v q(v|h) E[A | h, v] s_v
                   = E_q[A | h] * s_bar(h) + Cov_q(A, s_y | h),
    s_bar(h)       = sum_v q(v|h) s_v.

**On-policy (q = p).** `s_bar = sum_v p_v grad log p_v = grad sum_v p_v = grad 1 = 0`. Only the covariance term is left. It pushes up tokens whose continuations have above-average advantage, and that is the learning signal.

**Mismatched (q != p).** `s_bar != 0`. In logit space (`z` = logits at `h`), `grad_z log p_v = e_v - p`, so

    grad_z of the expected objective = E_q[A|h] * (q - p) + Cov_q(A, e_y | h).

The first term is a **drift**. It depends on the reward only through the scalar `E_q[A|h]` and not on which token was good. It is `E_q[A|h] * grad_theta sum_v q_v log p_v = -E_q[A|h] * grad_theta KL(q || p)`. So:

* at prefixes with `E_q[A|h] > 0`, it pulls `p` towards the sampler's errors `q`;
* at prefixes with `E_q[A|h] < 0`, it pushes `p` **away** from `q` by gradient ascent on `KL(q||p)`. It moves logits by `-(q - p)`: tokens that the sampler over-weights relative to `p` are pushed further down, and those it under-weights (often tail/junk tokens) are pushed up.

This term is a systematic bias and not zero-mean noise. It is large where the reward signal is weak. After an irrecoverable mistake every continuation fails, so `Cov_q(A, s_y | h) = 0` while `E[A|h] < 0`. At those prefixes the *entire* gradient is drift. Most tokens of failed rollouts are at such prefixes. With uncentered rewards (`plus_minus`), `|E[A|h]|` is O(1) everywhere. The sampler is refreshed from the trainer at every step (`q_{k+1} = p_{k+1} + perturbation`), so the drift acts again at every step on a fresh `q - p` of the same kind, and it accumulates. Pushing `p` away from `q` also raises `KL(q||p)`, which makes the next step's drift larger. That feedback loop produces the observed collapse.

**Why IS and clipping do not fix it.** A detached token weight `w = w(p_y/q_y)` turns the sampling measure into `m_v = q_v w(p_v/q_v)`. The drift becomes `E[A|h] * sum_v m_v s_v`, which vanishes only if `m ∝ p`, i.e. untruncated `w = p/q`. Truncation (TIS), dropping (MIS) and PPO/DPPO bands all restore `m_v = q_v` (or `0`) on exactly the tokens where `p` and `q` differ most. That is where the drift lives, so they only shrink it. Stronger token clipping makes `m` closer to `q` or zeroes out tokens. The drift can then grow, and dropped tokens also lose their covariance signal. Untruncated IS removes the bias but has unbounded variance. Sequence-level ratios have the same issue at the sequence level and much higher variance.

## 2. The modification: centre the score under the sampler

Replace each token's score `s_y` by `s_y - s_bar(h)`, with the same advantage `A`:

    E_q[A (s_y - s_bar) | h] = E_q[A s_y | h] - E_q[A|h] s_bar = Cov_q(A, s_y | h).

* The drift is removed exactly, for any `q`, at every prefix. No importance ratio is used, so no variance blows up.
* On-policy, `s_bar = 0` and the update is plain REINFORCE (verified numerically: identical gradients).
* The remaining term is the useful signal, measured under `q`. It raises tokens whose continuations do better than average under the sampler's own continuation distribution. Since `q ≈ p`, this is a small *variance-type* error and not a directional bias. In particular, at prefixes where all continuations fail, the update is now exactly zero in expectation instead of pure drift.
* In logit space the per-token gradient becomes `A (e_y - q)` instead of `A (e_y - p)`. The softmax baseline is taken under the distribution the token was actually drawn from, so `E_{y~q}[e_y - q] = 0`.

`s_bar(h)` depends on `theta` only through `log p`, and `q` is a constant, so the change fits in one backward pass as a surrogate:

    loss_t = -A * ( log p(y_t|h) - sum_v sg[q(v|h)] log p(v|h) ),
    grad   = -A * ( s_y - sum_v q_v s_v ) = -A * ( s_y - s_bar(h) ).

### Only the sampler's top-k head is available

The sampler returns the top-20 logprobs `H` per position. Model its tail as proportional to `p`: `q_v = rho p_v` for `v not in H`, with `rho = q_tail / p_tail`, `q_tail = 1 - sum_H q_v`, `p_tail = 1 - sum_H p_v`. Because `sum_{all v} p_v s_v = 0`,

    sum_{v not in H} rho p_v s_v = -rho sum_{v in H} p_v s_v,
    s_bar ≈ sum_{v in H} (q_v - rho p_v) s_v,

so

    loss_t = -A * ( log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p_v(h) ).

In logits the gradient is `-A (e_y - q_hat)`, where `q_hat = q` on `H` and `rho p` on the tail, and `q_hat` sums to 1. When `H` covers the whole support the formula is exact (verified numerically against the full-vocabulary surrogate: difference 0). With a truncated head the error is limited to how the tail's shape differs from `p`'s, weighted by `q_tail`, which is small for top-20. The sampled token's own `log p(y_t)` is kept even when `y_t` is outside `H`. Safeguards: `rho` is set to 0 if either tail is below `1e-6` (then nothing can be estimated about the tail and the head term alone is used), and it is capped at `1e3`. Coefficients are detached and masked outside completion tokens.

### With an IS correction (optional)

If a token-level `ISCorrection` is passed, the sampling measure is `m_v = q_v w(p_v/q_v)` (from the existing `log_mass`). The loss centres under that measure: head coefficients are `m_v`, and the tail uses `q = rho p`, so `p/q = 1/rho` and the tail measure is `rho w(1/rho) p_v`. This again subtracts `rho w(1/rho) p_v` on the head and adds nothing for the tail. Then `E_q[w A (s_y) - A sum_v m_v s_v] = Cov` under `m`, so the drift is removed for every point of the `rl.is.*` grid as well. Sequence-level corrections rescale `A` per sequence, and the token-level centering under `q` is kept unchanged. With `correction=None` (the default) the centering under `q` is active automatically whenever `sampler_topk_ids`/`sampler_topk_logprobs` are in the batch. They are now forwarded through `sampler_keys`.

## 3. Implementation (`losses.py`)

* `sampler_keys` now forwards `sampler_topk_ids` and `sampler_topk_logprobs`.
* `sampler_centering(...)` computes the detached coefficients `sg[q_v - rho p_v]` (or `sg[m_v - rho w(1/rho) p_v]` under a token correction) and gathers `log p` at the head ids. This runs inside the existing rematerialised chunks, so the extra memory is `[B, chunk, 20]`.
* The per-token score becomes `w * log p(y) - sum_H coef_v log p_v`, multiplied by the same advantage as before. The signature and return values are unchanged, plus a `centering_mass` metric.

Checks: identical gradient to the original loss on-policy, exact match with the full-vocabulary surrogate when the head covers the vocabulary, and finite results for all grid corrections. On `dev/mini.py`, the original PG plateaus/declines at eval ~0.70 (and collapses to 0 with `MINI_EXTRA_TOKENS=40`). The centered loss climbs steadily to ~0.83 (~0.82 with extra tokens, ~0.88 with TIS, no collapse).

No configuration changes are needed: the modification is active with the default `correction=None`.
