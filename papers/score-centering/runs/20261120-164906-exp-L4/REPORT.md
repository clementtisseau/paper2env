# Sampler-mismatch bias in REINFORCE, and a score-centering fix

## 1. What is wrong with the current loss

`loss_fn_rl` with `correction=None` is plain REINFORCE on sampler rollouts:

    L = -(1/N) sum_i sum_t A_i m_it log p(y_it | h_it),        (1)

where y ~ q (the sampler), h = prefix, A_i = advantage of completion i,
m = completion mask, N = number of masked tokens. Its gradient at a fixed
prefix h, with the next token drawn from q, is

    E_q[A s_y | h],    s_v := grad_theta log p(v|h).

Write E_q[A s_y | h] = E_q[A|h] s_bar(h) + Cov_q(A, s_y | h), where

    s_bar(h) := sum_v q(v|h) s_v.

The covariance term is the reward signal: it raises the tokens whose
completions earn more than average. The first term is a drift that
involves no reward difference between tokens.

On-policy (q = p), s_bar(h) = sum_v p_v s_v = grad sum_v p_v = 0, so the drift
vanishes. Under mismatch, s_bar(h) = sum_v (q_v - p_v) s_v is generically
nonzero, and the drift coefficient E_q[A|h] is the expected advantage of the
prefix. It is not zero: A depends on the whole completion, so
E_q[A|h] = V_q(h) - b, where V_q(h) is the value of h under the sampler and b
is the group mean. Group centering removes the mean over the sampled
rollouts, not the prefix-conditional mean. The drift is therefore
proportional to how good the prefix is, and it pushes the trainer's logits
along s_bar(h) regardless of which token was sampled. This is systematic, not
zero-mean noise. The trainer is then re-synced to the sampler, so the bias is
carried into the next step and compounds.

Importance-sampling weights do not remove this. A per-token ratio p/q corrects
only the current token's distribution, while the advantage depends on
continuations that are still drawn from q, so a bias from the mismatch of the
continuation distribution remains. Clipping or truncating the ratio also
introduces its own bias, which is why stronger token clipping can make things
worse.

## 2. The fix: subtract the q-expected score from each token

Replace each token's score s_y by s_y - s_bar(h), keeping the same advantage:

    E_q[A (s_y - s_bar) | h] = E_q[A s_y | h] - E_q[A|h] s_bar(h) = Cov_q(A, s_y | h).

The drift term is gone exactly, and the on-policy gradient is unchanged,
because s_bar = 0 when q = p. No importance ratios are used.

Since s_bar(h) = grad sum_v sg[q_v] log p_v (with sg = stop-gradient, q
held fixed), a single backward pass uses the surrogate

    loss_t = -A ( log p(y_t|h) - sum_v sg[q(v|h)] log p(v|h) ).        (2)

Its gradient is -A (s_y - s_bar(h)).

### Top-k head only

The sampler returns only its top-K logprobs H (K = 20 here). Model the sampler
tail as proportional to the trainer's:

    q_v = rho p_v for v not in H,     rho = q_tail / p_tail,
    q_tail = 1 - sum_{v in H} q_v,    p_tail = 1 - sum_{v in H} p_v.

Then, since sum_v p_v s_v = 0,

    sum_{v not in H} q_v s_v = rho sum_{v not in H} p_v s_v = -rho sum_{v in H} p_v s_v,

so

    s_bar(h) = sum_{v in H} (q_v - rho p_v) s_v.

The surrogate becomes

    loss_t = -A ( log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p_v ).   (3)

The sampled token keeps its own log p even when it lies outside H. The
expression is exact under the tail model, and I checked numerically (30-token
vocabulary, K = 5, arbitrary head q) that the full expected score
sum_v q_v grad log p_v equals the top-k surrogate gradient to 1e-16. The
tail-proportional assumption is the only approximation. Its error is
sum_{v not in H} (q_v - rho p_v) s_v, which is small when the tail mass is
small. Implementation: q_tail and p_tail are clipped to at least 1e-6 for
numerical safety.

### Properties

- On-policy, q = p gives rho = 1 and q_v - rho p_v = 0 for v in H, so (3) equals
  the original loss exactly. The fix changes nothing when there is no mismatch.
- It needs no configuration. It is active by default and works with
  `correction=None`. It multiplies with an IS weight if one is given, as the
  weight applies to the whole score term.
- Its cost is one gather of the trainer's log-probs at the K sampler ids, per
  position, in the same chunked pass that already computes the vocab
  log-softmax.

## 3. Implementation in losses.py

- `sampler_keys` now also forwards `sampler_topk_ids` and
  `sampler_topk_logprobs` (already present in the batch from `train_rl.py`).
- In `chunk_sums`, the per-token score is `w * (logp - baseline)`, where
  `baseline` computes (3) as `sum_H sg[q_v - rho p_v] * logp_v`. If the top-k
  keys are absent, the baseline is 0 and the loss reduces to the original.

## 4. Why this should help under mismatch

The old expected update at each prefix is Cov_q(A, s_y|h) + E_q[A|h] s_bar(h).
The new one is Cov_q(A, s_y|h) exactly (up to the tail-model error in (3)). The
only term that depends on the sampler mismatch, and that compounds through the
sampler refresh, is removed. What remains is the covariance between advantage
and token score, which is the same signal as on-policy REINFORCE, estimated
from rollouts drawn from q.

## Configuration

No configuration changes. The modification is on by default.
