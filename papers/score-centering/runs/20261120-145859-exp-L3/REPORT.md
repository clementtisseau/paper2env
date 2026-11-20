# Removing the sampler-mismatch drift from REINFORCE

## 1. What is wrong with the current loss

The loss is `-(1/N) sum_i A_i sum_t w_{i,t} log p(y_{i,t} | h_{i,t})`, where the tokens y are drawn
from the sampler q, not from the trainer p. Fix one prefix h and write
s_v = grad_theta log p(v|h) for the score of token v. Without correction (w = 1), the expected update
at h is

    E_q[A s_y | h] = sum_v q(v|h) E[A | h, v] s_v
                   = E_q[A|h] * sbar(h)  +  Cov_q(A, s_y | h),
    sbar(h) = sum_v q(v|h) s_v.

* **On-policy (q = p):** sbar(h) = sum_v p grad log p = grad sum_v p(v|h) = grad 1 = 0. Only the
  covariance remains. It pushes up tokens whose continuations have a higher advantage than average.
  This is the reward signal.
* **Mismatch (q != p):** sbar(h) != 0. In logit space (z = logits at h, grad_z log p(v) = e_v - p) we get

      sbar(h) = sum_v q_v (e_v - p) = q(.|h) - p(.|h).

  So the loss adds a drift **E_q[A|h] (q - p)** at every prefix. It pulls p towards q where the
  expected advantage is positive and pushes p away from q where it is negative. This drift does
  not depend on which token was good. It depends only on the gap between the sampler and the
  trainer, and it is as large as the *mean* advantage at the prefix, which is not small.
  Group centering makes E[A] = 0 only at the prompt. After the first tokens, E[A|h] is far from 0,
  because a prefix that is already wrong has E[A|h] < 0.

The drift is a bias, not noise. Its direction (q - p) is the same systematic sampler error at every
step, because the sampler is rebuilt from the trainer each step with the same perturbation (fixed
noise, quantization or KV quantization). So the drift adds up linearly over steps, while
zero-mean noise would only grow like sqrt(steps). It is also sign-dependent. On prefixes with
negative mean advantage, p is pushed away from q, so the gap q - p grows, and the drift on the
next step grows with it. This feedback loop is what collapses training.

**Why IS and clipping do not fix it.** A detached token weight w(p/q) changes the update into
sum_v q_v w_v E[A|h,v] s_v. The drift part becomes

    sbar_c(h) = sum_v q(v|h) w(p_v/q_v) s_v = sum_v m_v (e_v - p),   m_v = q_v w(p_v/q_v).

This is zero only when m = p exactly, i.e. for untruncated w = p/q (naive IS). Its variance is
unbounded, and it still only fixes the current token. Any truncation, clamping or dropping
(TIS, MIS, PPO, DPPO...) makes m != p, and the drift comes back as E[A|h] (m - M p), where
M = sum_v m_v. Tighter token clipping moves m further from p, so it can make the bias *larger*.
Sequence-level weights scale A but leave sbar(h) = q - p unchanged.

## 2. The modification

Subtract, at every position, the expected score under the sampler. This is a per-prefix control
variate that does not depend on the sampled token:

    loss = -(1/N) sum_i A_i sum_t [ w_{i,t} log p(y_{i,t}|h_{i,t})  -  sum_v sg(m_v(h_{i,t})) log p(v|h_{i,t}) ]

with m_v = q(v|h) w(p_v/q_v) (and simply m_v = q(v|h) when `correction=None`, the default). The
per-token gradient becomes A (w_y s_y - sbar_c(h)), and its conditional expectation over y ~ q is

    E_q[A (w_y s_y - sbar_c) | h] = sum_v m_v E[A|h,v] s_v - E_q[A|h] sbar_c(h)

and for w = 1:

    E_q[A (s_y - sbar) | h] = E_q[A s_y|h] - E_q[A|h] sbar(h) = Cov_q(A, s_y | h).

The drift term is removed exactly, for any q, at every prefix. What remains is the covariance
between the advantage and the score, which is the actual learning signal. Only the covariance is
measured under q instead of p, which is a second-order effect. Sequence-level corrections still
compose: the per-sequence weight multiplies A in both terms, so the subtracted term is
E_q[A W|h] sbar(h), and that is exactly the drift for the reweighted advantage.

In logit space, the per-token update without correction is

    A (e_y - p)  ->  A (e_y - q).

For a fixed prefix this has mean 0 under y ~ q whenever A does not depend on y. So a prefix whose
continuations all have the same advantage gives no update at all, whatever the mismatch. This
holds for any q, so it covers weight noise, int8/fp8, KV quantization and staleness alike. It is
the same as using the *sampler's* distribution as the softmax baseline in the policy-gradient
logit update. On-policy (q = p) the subtracted term is grad sum_v p log p over a partition of the
vocabulary with p detached, which is identically 0. So the loss reduces exactly to the original
REINFORCE and keeps its unbiasedness.

Interpretation: the original loss implicitly assumes E_q[s_y|h] = 0. Under mismatch that
assumption is false, and the residual E_q[A|h] (q - p) acts like an extra advantage-weighted
"distill towards / away from the sampler" objective. The modification subtracts that residual.

### Using only the top-20 logprobs

The sampler returns q for its top-K (K = 20) tokens. The rest of the vocabulary is treated as one
merged outcome R, with q_R = 1 - sum_{v in K} q_v and p_R = 1 - sum_{v in K} p_v:

    sbar ~= sum_{v in K} m_v grad log p_v + m_R grad log p_R.

In logit space, the q-tail is replaced by q_R times the tail shape of p, i.e. the update is
A (e_y - q_K - q_R p_tail/p_R). Properties:

* It is exact on the K tokens and exact on the tail as a whole. The only error comes from the
  *shape* of q inside the tail, which is at most O(q_R * TV(q_tail, p_tail)). That error is tiny
  because top-20 holds almost all of the mass.
* On-policy it gives exactly zero, because {K, R} is a partition and
  sum_K p grad log p + p_R grad log p_R = grad(sum_K p + p_R) = 0. Renormalizing q over the top-K
  would leave a residual grad log P_K.
* p_R is floored at 1e-6 to avoid float32 cancellation.

When a correction is given, the masses are m_v = exp(log_mass(log p_v, log q_v, ...)): the same
band, clamp/drop and sign rules applied to every candidate token v. This way the subtracted term
equals the expected weighted score of that correction.

## 3. Implementation (`losses.py`)

* `sampler_keys` now also forwards `sampler_topk_ids` and `sampler_topk_logprobs`.
* `expected_score(...)` returns sum_K sg(m_v) log p_v + sg(m_R) log p_R per position. It is
  computed inside the existing rematerialized chunks from the same `logp_vocab`, using
  `take_vocab` so it is TP-safe.
* The per-token score is `w * logp - expected_score`. Everything else (advantages, sequence-level
  weights, normalization, metrics) is unchanged. The fix is active with default arguments and
  `correction=None`. It is skipped only if the batch has no top-k logprobs.
* Check: with q = p, the gradients equal those of the original loss (max abs diff 7e-9).

The mini dev run (noise 0.3, 600 steps) gives final eval reward 0.83 (rising steadily), against
0.66 (plateauing and declining) for the original loss.

## Configuration

No configuration changes are required. The correction works with the default `rl.is.*` settings
(i.e. no IS correction), and it composes with any IS correction.
