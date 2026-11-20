# Sampler-baselined REINFORCE under sampler/trainer mismatch

## 1. What is wrong with the current loss

Let p_θ be the trainer and q the sampler that generated the rollouts. At a
prefix h, let y be the next token and A the advantage of the completion. Write
Ā(h, y) = E[A | h, y] for the expected advantage given prefix and next token
(the completion's randomness after y is averaged out). The score of token v is
s_v(h) = ∇_θ log p_θ(v | h).

The current loss is `loss_t = -A · log p(y_t | h)` (times the IS weight w, if
any), so its per-token gradient is `-A s_y`. Its expectation over y ~ q is

    E_q[Ā s_y | h] = Σ_y q(y|h) Ā(h,y) s_y
                   = E_q[Ā | h] · s̄(h)  +  Cov_q(Ā, s_y | h),

with s̄(h) = Σ_v q(v|h) s_v(h). The identity is exact: it is the definition of
covariance with the mean term split off.

On-policy (q = p), s̄(h) = Σ_v p_v ∇ log p_v = ∇ Σ_v p_v = 0, so only the
covariance, which is the policy gradient, remains.

Under mismatch, s̄(h) ≠ 0 in general, and the first term is a drift:

    drift(h) = E_q[Ā | h] · s̄(h),    s̄(h) = ∇_θ Σ_v q(v|h) log p_θ(v|h).

Since q is fixed at a given step, s̄(h) is the gradient of the cross-entropy
between q and p at h, up to sign: s̄(h) = -∇_θ CE(q_h, p_h). The update therefore
moves p_h toward q_h at prefixes with positive expected advantage and away from
q_h where the expected advantage is negative. The reward signal plays no role in
this direction: it is set by the sampler's error, weighted by how rewarded that
prefix is. Because the next step's sampler is refreshed from the trainer, the
bias is fed back and compounds over training. Importance weights act on the sampled token only. Untruncated, w = p/q makes
the expectation exact but high-variance. Truncated or dropped weights (TIS, MIS,
PPO-style bands) make the effective weight q·w(p/q) differ from p on some tokens,
which leaves a residual bias of the same kind. The sampled-token weight never
touches the term Σ_v q_v s_v, which is where s̄ lives, so the drift is only
slowed, not removed.

## 2. The modification: a sampler-distribution baseline

Replace each token's score s_y with s_y - s̄(h), keeping the same advantage:

    loss_t = -w_t · A · ( log p(y_t | h) - Σ_v sg[q(v|h)] log p(v|h) ),

where sg is stop-gradient and w_t is the existing IS weight (1 with
`correction=None`). Its gradient is -w_t A (s_y - s̄(h)), using

    ∇ Σ_v sg[q_v] log p_v = Σ_v q_v s_v = s̄(h).

Expectation over y ~ q, with the same covariance identity:

    E_q[Ā (s_y - s̄) | h] = E_q[Ā s_y | h] - E_q[Ā | h] s̄(h)
                          = Cov_q(Ā, s_y | h).

So the drift term is removed exactly, and no importance ratio is involved.

Properties:
- On-policy (q = p): s̄(h) = 0, so the loss and its expected gradient equal the
  original ones. Only the variance changes.
- The baseline depends only on the prefix h and q, not on the sampled token,
  so it adds no bias relative to the covariance form. It does not depend on A.
- The remaining bias relative to the true policy gradient under p is
  Cov_q - Cov_p, which is proportional to the mismatch and vanishes when A does
  not correlate with the next token. It has no drift term and does not
  accumulate through E[A] at a prefix.

## 3. Using only the sampler's top-k head

The sampler returns the top-K (K = 20) logprobs at every position. The full
q(·|h) is not available, so model the tail of q as proportional to p:

    q_v = ρ p_v  for v outside the head H,   ρ = q_tail / p_tail,

where q_tail = 1 - Σ_{v∈H} q_v and p_tail = 1 - Σ_{v∈H} p_v (clamped below by
1e-6 for numerical safety).

Since Σ_all p_v s_v = 0, the tail gives Σ_{v∉H} q_v s_v = ρ Σ_{v∉H} p_v s_v =
-ρ Σ_{v∈H} p_v s_v. Hence

    s̄(h) = Σ_{v∈H} (q_v - ρ p_v) s_v   (exact under the tail model).

The surrogate becomes

    loss_t = -w_t A ( log p(y_t | h) - Σ_{v∈H} sg[q_v - ρ p_v] log p_v ).

The sampled token keeps its own log p even if it lies outside H. Its score is
not part of the baseline, which is what the covariance identity requires.

Implementation (losses.py): `sampler_baseline` computes the sum over H with
`take_vocab`, so it is TP-safe. `loss_fn_rl` forwards `sampler_topk_ids` and
`sampler_topk_logprobs` through `sampler_keys`, and it raises an error if they
are missing.

Checks run on random data: with q_tail ∝ p_tail, the gradient of the head
baseline equals the full-vocabulary E_q[∇ log p] to float32 precision (max
abs error 3.6e-7 against a scale of 0.59). On-policy the baseline and its
gradient are exactly 0.

## Configuration

None. The modification is active by default and needs no `rl.is.*` keys.
