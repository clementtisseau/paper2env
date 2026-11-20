# Removing the sampler-induced drift from the REINFORCE loss

## What is wrong with the current loss

With `correction=None` (w = 1), `loss_fn_rl` is

    L(θ) = -(1/N) Σ_b Σ_t A_b · log p_θ(y_bt | h_bt),

where y_bt is the token drawn from the sampler q, h_bt the prefix, A_b the
(group-centred) advantage of completion b, and N the number of completion
tokens. Its gradient is -(1/N) Σ A_b s(y_bt | h_bt) with s(v|h) = ∇_θ log p(v|h).

Fix a position and prefix h. The next token y ~ q(·|h), and A is the advantage
of the whole completion. The expected per-token update is

    E_q[A s_y | h] = E_q[A | h] · s̄_q(h) + Cov_q(A, s_y | h),
    s̄_q(h) = Σ_v q(v|h) s(v|h).

(This is just E[XY] = E[X]E[Y] + Cov(X,Y) with X = A, Y = s_y, and E_q[s_y|h] = s̄_q.)

On-policy (q = p), s̄_p(h) = Σ_v p(v|h) ∇ log p(v|h) = ∇ Σ_v p(v|h) = 0, so only
the covariance remains. That covariance is the policy-gradient signal.

Under mismatch, s̄_q(h) ≠ 0 in general, and the current loss contains the drift
term E_q[A|h] · s̄_q(h). Three facts make this term harmful:

1. It is not reward signal. It is multiplied by E_q[A|h], the expected advantage
   of the prefix, which is generally nonzero. Group centring makes advantages
   sum to zero over a group of samples, not at every prefix.
2. It is a bias, not zero-mean noise. Averaging over many samples does not cancel it.
3. Its direction is known. Using s̄_q = -∇_θ H(q, p_θ) with H(q,p) = -Σ_v q_v log p_v
   (the cross-entropy of q under p at h), the update contains

       -E_q[A|h] · ∇_θ H(q(·|h), p_θ(·|h)).

   With E_q[A|h] > 0 the trainer is pushed toward the sampler's distribution at h,
   including the sampler's own perturbation (noise, quantisation, staleness). With
   E_q[A|h] < 0 it is pushed away from q. Because the sampler is refreshed from the
   trainer each step, the target q_t = p_t + δ_t moves with the trainer, so the
   push compounds over training.

Importance weights do not remove this. The per-token ratio w = p/q makes
E_q[w A s_y | h] = Σ_v p(v|h) E[A|h,v] s_v, which is the on-policy expression, so
an exact full-vocabulary ratio would cancel the drift. The sampled-token ratio
does not do that. Clipping it, as TIS or PPO do, replaces w = p/q by
min(max(p/q, lo), hi), which changes the expectation over v from Σ_v p_v to a
clipped mixture. That leaves a residual drift with a different coefficient, which
explains why clipping slows the degradation but does not remove it, and why
stronger clipping can make it worse.

## The modification

The drift is the only term that vanishes on-policy and whose coefficient is an
expected advantage, so we subtract exactly that term. For each position define

    G(h) = Σ_v (q(v|h) - p(v|h)) · log p(v|h)        (coefficients stop-gradient)

Its gradient is

    ∇G(h) = Σ_v (q_v - p_v) s_v = Σ_v q_v s_v - Σ_v p_v s_v = s̄_q(h) - 0 = s̄_q(h),

using Σ_v p_v s_v = 0 (exact, over the full vocabulary). The loss uses the per-token score

    ŝ_bt = log p(y_bt | h_bt) - G(h_bt),

so the gradient is -(1/N) Σ A_b (s(y_bt|h_bt) - s̄_q(h_bt)).

Expected update. Conditionally on h, with y ~ q,

    E_q[A (s_y - s̄_q) | h] = E_q[A s_y | h] - E_q[A | h] s̄_q(h) = Cov_q(A, s_y | h).

The drift term is gone and the remaining term is the covariance, the reward
signal under q.

On-policy. If q = p then G(h) ≡ 0 and the loss is the original one, bit for bit.

Implementation. G needs q over the vocabulary, but the sampler returns only its
top-K logprobs (K = 20 in train_rl.py). The code restricts the sum to the top-K
set 𝒦(h):

    G_K(h) = Σ_{v ∈ 𝒦(h)} (q_v - p_v) log p_v.

This is exact when K covers the vocabulary (the dev mini-model has 12 tokens, so
K = 12). In general the dropped tail term Σ_{v ∉ 𝒦} (q_v - p_v) s_v is the only
approximation. It is small when the sampler's tail mass is close to the trainer's,
which holds for small perturbations and quantisation. The approximation's
direction is no longer systematic. The loss still runs with a correction (w
multiplies the whole score), but the default configuration needs no
configuration change.

## Why this is the right fix, and its limits

- Exact in expectation for the drift: the subtracted term is a deterministic
  function of h times the sequence advantage, and its expectation is
  E_q[A|h] s̄_q(h), precisely the drift.
- The on-policy estimator is unchanged, so it introduces no new bias when the
  mismatch is zero.
- Residual bias: the signal is Cov_q(A, s_y) rather than Cov_p(A, s_y). This is an
  O(mismatch) change in the covariance, not a term that pushes p toward q in
  proportion to expected advantage.
- Variance: the subtracted term adds A·G(h) noise, which is small because
  |G| is O(mismatch).
- The top-K truncation drops the tail of the sampler's distribution.

## Checks

- Identity: a numerical check (JAX autodiff) confirmed that
  ∇_θ Σ_v (q_v - p_v) log p_v(θ) = Σ_v q_v s_v when Σ_v p_v s_v = 0.
- Dev runs (`dev/mini.py`, sampler = trainer + fixed relative weight noise,
  `plus_minus` rewards, 400 steps, CPU), last-20-step mean training reward.
  Baseline: current loss; modified: this loss. Seeds 0-2. Eval reward at the end is
  0.67-0.71 baseline vs 0.80-0.85 modified at noise 0.3.

  | noise | baseline (seeds 0-2) | modified (seeds 0-2) |
  | --- | --- | --- |
  | 0.3 | 0.58, 0.59, 0.57 | 0.83, 0.80, 0.78 |
  | 0.6 | 0.00, 0.00, 0.00 | 0.38, 0.38, 0.29 |

  With a 42-token vocabulary (K = 20 < V, so the tail is truncated), noise 0.6,
  seed 0: baseline 0.00, modified 0.40.

## Configuration

No configuration changes are needed. The modification is active under the default
arguments (`correction=None`). The trainer already sends `sampler_topk_ids` and
`sampler_topk_logprobs` (train_rl.py passes `logprobs_topk=20`), and `sampler_keys`
now forwards them.
