# Score-centred REINFORCE under sampler mismatch

## What is wrong with the current loss

With no correction, `loss_fn_rl` takes the gradient of

    L(θ) = -Σ_t A_t · log p_θ(y_t | h_t) / N,        y_t ~ q(·|h_t),

so its expected update at a token position, for a fixed prefix h, is (A is the
sequence advantage, s_v = ∇_θ log p_θ(v|h) the score of token v):

    E_q[A s_y | h] = E_q[A | h] · s̄_q(h) + Cov_q(A, s_y | h),
    s̄_q(h) = Σ_v q(v|h) s_v.

The on-policy case is harmless because s̄_p(h) = Σ_v p(v|h) ∇log p(v|h) = ∇ Σ_v p(v|h) = 0,
so only the covariance (the reward signal) survives. Under a mismatch q ≠ p,
s̄_q(h) ≠ 0 in general, and the first term

    D(h) = E_q[A | h] · s̄_q(h)

is a drift. It is not zero-mean noise: for a prefix h whose completions are on
average good (E_q[A|h] > 0) the update raises Σ_v q(v|h) log p(v|h), i.e. it
pulls the trainer toward the sampler's next-token distribution whatever the
reward says, and for bad prefixes it pushes away from it. Because the sampler
is re-synchronised from the trainer at each step, q itself moves with p, so
the drift is applied at every step, in a consistent direction, and compounds.
Group-centred advantages do not remove it: they make the average over a group
zero, not the conditional mean E_q[A | h] at each prefix.

Token-level clipping (TIS, PPO-style bands) only changes the weight on the
sampled token. The clipped estimator E_q[clip(p_y/q_y) s_y | h] differs from
Σ_v p_v s_v, and its difference is again a term of the form E[A]·(something
that is not zero), so clipping slows the drift but does not remove it. The
sampled-token form cannot see s̄_q, which requires the sampler's full
next-token distribution.

## The modification: centre the score under q

Replace the per-token score s_y by the centred score

    ŝ_y(h) = s_y − b(h),      b(h) = s̄_q(h) = Σ_v q(v|h) s_v,

so the loss becomes L'(θ) = −Σ_t A_t (log p(y_t|h_t) − Σ_v q(v|h_t) log p(v|h_t)) / N.
Its per-token expectation under q is

    E_q[A (s_y − s̄_q) | h] = E_q[A s_y | h] − E_q[A | h] s̄_q(h)
                            = Cov_q(A, s_y | h).

Two facts make this the fix:

1. **The drift is gone exactly.** The baseline b(h) depends on the prefix only,
   so it is a constant inside the conditional expectation. The term
   E_q[A|h] s̄_q(h) that appeared above is cancelled identically, for any
   advantage function A, including a biased one.
2. **On-policy the update is unchanged.** If q = p then s̄_q = s̄_p = 0 and
   b(h) has zero gradient, so L' has the same expected gradient as before. The
   centring adds no bias where the original estimator was correct.

The estimator is now a covariance with respect to the sampler, and a covariance
vanishes whenever the advantage does not depend on the sampled token. A mismatch
between p and q can therefore only enter through how q weights the tokens
whose advantage differs, not through a constant push on every prefix.

### Computing b(h) from the sampler

The sampler returns its top-K (K = 20) logprobs per position. Let S be that
top-K support, with tail mass ε = 1 − Σ_{v∈S} q(v|h). Use the renormalised
top-K distribution q̂(v|h) = q(v|h)/Σ_{u∈S} q(u|h) and

    b̂(h) = Σ_{v∈S} q̂(v|h) log p_θ(v|h).

Its gradient is ŝ̄ = Σ_{v∈S} q̂_v s_v. Writing E_S and E_tail for averages of s_v
over the top-K support and over the tail, the error is exactly

    ŝ̄ − s̄_q = ε · (E_S[s] − E_tail[s]),

so it is first order in the tail mass ε and vanishes when the sampler keeps all
its mass in the top-K. Gathering log p over the K ids needs only the trainer's
log-softmax, which the loss already computes.

### With an importance-sampling correction

If the per-token weight w_v = w(p_v/q_v) is applied, the same argument holds
with X_v = w_v s_v: the centred term X_y − E_q[X | h] has conditional expectation
Cov_q(A, X_y | h), which is again free of the drift E[A]·E_q[X]. Under the
un-clipped ratio E_q[w s] = Σ p_v s_v = 0, so the drift is already zero and the
centring is a no-op; under clipping, E_q[clip(w) s] ≠ 0 and the centring removes
the clipping-induced drift. The weight depends on the sign of A (for PPO-style
bands), so the baseline is computed separately for A ≥ 0 and A < 0 and selected
per sequence. This is implemented in `score_baseline` in `losses.py`.

## Implementation

In `losses.py`, `loss_fn_rl` adds `sampler_topk_ids` and `sampler_topk_logprobs`
to `sampler_keys`. In each chunk it computes `b(h)` over the top-K support
(`score_baseline`) and uses the score `w · (log p(y) − b(h))` in place of
`w · log p(y)`. It is active by default (`correction=None`) and needs no
configuration. If the top-K keys are absent, the baseline is zero and the loss
reduces to the original.

## Expected effect

The drift term is a bias in the direction of the sampler's own distribution,
weighted by the prefix's expected advantage. Removing it makes the update
depend on the reward only through covariances under q, so the systematic
push that compounds across sampler refreshes is removed. The remaining error
is O(|q − p|) in the covariance and O(ε) in the baseline, and neither
accumulates in a fixed direction.

Check on the dev proxy (`dev/mini.py`, sampler weights perturbed by 30% relative noise, 600 steps,
default correction=None, seeds 0–2): the original loss ends at eval reward about 0.67–0.69 and
the centred loss at about 0.81–0.84. This is one small setting, reported only as a sanity check.

## Configuration

No configuration changes are needed. The loss is active with default arguments.
