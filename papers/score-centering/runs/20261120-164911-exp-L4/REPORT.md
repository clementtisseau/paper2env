# Sampler-centered REINFORCE: removing the mismatch drift

## 1. What is wrong with the current loss

The trainer has policy p_theta; rollouts come from a sampler q (noisy weights,
int8/fp8, quantized KV cache, stale weights). Per token the loss is

    loss = -(1/N) sum_{i,t} A_i * w_{i,t} * log p(y_t | h_t),

with w = 1 for plain PG, or a detached IS weight w(p/q). Its gradient is
-(1/N) sum A s_{y_t}, with score s_v(h) = grad_theta log p(v|h).

Fix a prefix h and take the expectation over the next token y ~ q(.|h).
Write A for the advantage of the completion (it depends on y and on what
follows). Then, exactly,

    E_q[A s_y | h] = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h),
    s_bar(h)       = sum_v q(v|h) grad log p(v|h).                         (1)

(Proof: E[A s] = E[A] E[s] + Cov(A, s), and E_q[s_y|h] = s_bar(h).)

* **On-policy (q = p):** s_bar(h) = sum_v p(v|h) grad p(v|h)/p(v|h)
  = grad sum_v p(v|h) = grad 1 = 0. Only the covariance survives: the update
  pushes up tokens that are correlated with higher-than-average advantage.
  This is the policy-gradient signal.
* **Mismatch (q != p):** s_bar(h) != 0. With the softmax parametrisation
  log p(v|h) = z_v - logsumexp(z), so d log p(v)/dz_u = 1[u=v] - p_u and

      d s_bar / dz_u = q_u - p_u.                                          (2)

  The first term of (1) is therefore E_q[A|h] * (q - p) in logit space. It
  moves p toward q wherever the expected advantage at h is positive and away
  from q (in the direction p - q) wherever it is negative. It is a drift: it
  does not depend on which token was good, only on the average advantage at
  the prefix and on the sampler/trainer gap. It is not zero-mean noise, so
  more samples do not average it out.

Why it compounds. Group-centering makes E[A] = 0 over a whole group, but not
conditionally on a prefix: E_q[A | h] is positive on prefixes that lead to
success and negative on prefixes that lead to failure (and in states that are
off the good path, which are exactly where the sampler and trainer tend to
disagree). On failing prefixes the drift pushes p away from q, i.e. it
*amplifies* the mismatch in the direction p - q; since the sampler is rebuilt
from the trainer every step (q_{k+1} = Noise(p_{k+1})), the perturbation is
re-applied on top of a policy that has already moved, and the gap and the
drift grow together. On successful prefixes it pulls p towards the sampler's
corrupted distribution (e.g. the noisy weights' tails), which also biases the
trained policy. The bias is first order in A and in (q - p) at every token
of every sequence, while the true signal Cov_q(A, s_y|h) is often small at
most positions (most tokens barely affect the outcome), so the drift
dominates per token.

Why IS weights / clipping do not remove it. With a token-level weight
w(r), r = p/q, the expected update is sum_v q_v w(r_v) E[A|h,v] s_v. With the
exact w = r this is E_p[A s], unbiased, but w = r has unbounded variance, and
A also depends on all later tokens (whose ratios a token-level weight
ignores), so the token-level correction is not the sequence-level IS
anyway. Once w is truncated or dropped (TIS, MIS, PPO, ...) the effective
sampling distribution is m_v proportional to q_v w(r_v), which is neither q
nor p, and the same decomposition gives a drift
E[A|h] * sum_v m_v s_v with sum_v m_v s_v != 0. Tighter token clipping
makes m deviate from p on more tokens, so the residual drift can grow. These
corrections reshape the drift; they do not cancel it.

## 2. The fix: center the score under the sampler

Replace every token's score s_y by s_y - s_bar(h), with the same advantage:

    g = -(1/N) sum_{i,t} A_i * ( s_{y_t}(h_t) - s_bar(h_t) ).              (3)

Conditionally on the prefix,

    E_q[A (s_y - s_bar) | h] = E_q[A s_y | h] - E_q[A|h] s_bar(h)
                             = Cov_q(A, s_y | h).                          (4)

So the drift term of (1) is removed exactly, for any q, any advantage and
any reward scheme, and what is left is the covariance between the
advantage and the score, the same quantity that drives on-policy
REINFORCE. On-policy s_bar = 0 and (3) reduces to ordinary REINFORCE: the
modification costs nothing when there is no mismatch.

Notes on the derivation:
* s_bar(h) depends only on the prefix h (and on q(.|h)), not on y, so
  subtracting it is a control variate on the score, the per-state
  analogue of a baseline: it changes neither the covariance term nor the
  on-policy gradient.
* In logit space at h the update is A (e_y - q), instead of A (e_y - p)
  for plain REINFORCE. Its q-expectation is sum_v q_v E[A|h,v](e_v - q),
  which is zero when E[A|h,v] does not depend on v. Uninformative
  positions therefore produce no expected update at all. Plain REINFORCE
  produces E[A|h](q - p) at those positions, the drift. Equivalently the
  fix replaces the trainer's normalizer gradient p by the distribution
  the tokens actually came from.
* No importance ratios are used, so there are no high-variance weights
  and nothing to clip. The bias that remains is only the second-order
  effect of following Cov_q instead of Cov_p (q is close to p), which
  does not have the systematic sign of the drift: it is no longer
  proportional to E[A|h] and does not push p along (q - p).
* It composes with an IS correction: with a token weight w the surrogate
  is w * (log p(y) - baseline). The default (correction=None) is the
  plain centered estimator.

## 3. Implementation with one backward pass and only the top-k head

Surrogate with stop-gradient sg:

    loss_t = -A * ( log p(y_t|h) - sum_v sg[q(v|h)] log p(v|h) ),

whose gradient is -A (s_y - s_bar(h)).

The sampler only returns its top-K (K = 20) logprobs: head H, with masses
q_v, v in H, and tail mass Q_T = 1 - sum_{v in H} q_v. Model q's tail as
proportional to p: q_v = rho p_v for v not in H, with
rho = Q_T / P_T and P_T = 1 - sum_{v in H} p_v. Then

    s_bar = sum_{v in H} q_v s_v + rho sum_{v not in H} p_v s_v
          = sum_{v in H} q_v s_v - rho sum_{v in H} p_v s_v
          = sum_{v in H} (q_v - rho p_v) s_v,

using sum_v p_v s_v = 0 (so the tail sum equals minus the head sum). The
whole full-vocabulary expectation is computed from the K head entries and
the trainer's own log-softmax, which the loss already has:

    loss_t = -A * ( log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p_v ).

The sampled token keeps its own log p(y_t|h) even when y_t is outside H.
Exactness check: for K = V (P_T = Q_T = 0, rho irrelevant) this is exactly
(3); in general it is exact whenever the sampler's tail is proportional to
the trainer's (verified numerically against the full-vocabulary gradient).

Numerical safeguards in `losses.py` (`sampler_baseline`):
* P_T is floored at 1e-6 and rho is capped at RHO_MAX = 4. When P_T is
  tiny the coefficient sum (Q_H - rho P_H) behaves like (Q_H - P_H)/P_T
  and would amplify sampler noise; capping rho only drops a part of a
  tail whose trainer mass is negligible anyway.
* Head entries with logprob -inf (e.g. padded vocab) get coefficient 0.
* The coefficients are detached; the surrogate's value is meaningless,
  only its gradient matters. A metric `tail_ratio` (mean rho) is logged.
* If the batch carries no `sampler_topk_*` keys the baseline is zero and
  the loss falls back to plain REINFORCE.

Everything runs inside the existing rematerialized chunk loop, so memory
stays [B, chunk, V] and the extra cost is one gather of K logits per
position.

## Configuration

No configuration changes are needed: the centered score is active with the
default arguments (correction=None).
