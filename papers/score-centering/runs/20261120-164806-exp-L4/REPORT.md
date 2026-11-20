# Removing the mismatch drift from REINFORCE: a sampler-mean score baseline

## Summary

The current loss is REINFORCE, `-A * log p(y_t|h_t)`, with tokens drawn from a
sampler q that is not the trained model p. Fix one prefix h. In expectation over
the next token y ~ q(.|h), the update is a covariance term (the learning signal)
plus a **drift** `E_q[A|h] * s_bar(h)`, where
`s_bar(h) = sum_v q(v|h) grad log p(v|h)`. On-policy, `s_bar = 0`. Off-policy it
is not zero, and it moves p toward q at every prefix where the expected
advantage is positive and away from q where it is negative. It does this
whether or not the token choice affects the reward.

The modified loss subtracts this expectation from every token's score:

    loss_t = -A * ( log p(y_t|h_t) - sum_v sg[q~(v|h_t)] log p(v|h_t) )

Here q~ is the sampler's next-token distribution, rebuilt from its top-20
logprobs. Under q, the expected update is exactly `Cov_q(A, s_y | h)`, so the
drift is gone. On-policy the added term has zero gradient. The loss uses no
importance ratios. It is always on: `loss_fn_rl` turns it on whenever
`sampler_topk_ids` / `sampler_topk_logprobs` are in the batch, which
`train_rl.py` and `dev/mini.py` always provide. It works with `correction=None`
and also when combined with any `ISCorrection`.

## 1. What is wrong with the current loss

Notation. h is a prefix (prompt plus the completion tokens so far). y is the next
token. `s_v(h) = grad_theta log p(v|h)` is the score of token v. A is the
sequence advantage. A depends on y and on everything sampled after y.

The per-token REINFORCE term at h contributes `A * s_y(h)`. Condition on h and
average over y ~ q(.|h) and over the rest of the rollout (which is also drawn
from q):

    E_q[A s_y | h] = sum_v q(v|h) E_q[A | h, v] s_v(h)
                   = E_q[A|h] * s_bar(h) + sum_v q(v|h) (E_q[A|h,v] - E_q[A|h]) s_v(h)
                   = E_q[A|h] * s_bar(h) + Cov_q(A, s_y | h)                     (1)

    s_bar(h) = sum_v q(v|h) s_v(h).

**On-policy (q = p).** `s_bar(h) = sum_v p_v grad log p_v = grad sum_v p_v = grad 1 = 0`.
Only the covariance is left, and it is the policy gradient: it raises the
tokens whose continuation value `E[A|h,v]` is above the mean at h. This is why a
constant baseline (group centering) leaves REINFORCE unbiased: any function of
h times `s_y` has zero mean.

**Mismatched (q != p).** `s_bar(h) != 0`. To see what it does, take the gradient
with respect to the logits z(h) of the softmax at h:

    d log p(v|h) / dz_u = 1[u=v] - p_u     =>     s_bar(h) in logit space = q(.|h) - p(.|h).

So the drift term is `E_q[A|h] * (q - p)`. It is an update toward q when
`E_q[A|h] > 0`, and away from q when it is negative. It has these properties:

* **It is not noise.** It is the expectation of the update, so averaging over
  more samples does not shrink it. It is a systematic bias in the objective.
  Equivalently, the loss climbs `J(p) + sum_h E_q[A|h] * (-KL(q(.|h) || p(.|h)))`
  rather than J(p), up to terms of order (q - p)^2.
* **It is unrelated to reward.** It acts at every prefix with nonzero mean
  advantage, including prefixes where no token choice changes the outcome.
  There the covariance is zero and the drift is the whole update.
* **It is large where it matters most.** Group centering makes `E[A] = 0` only
  at the prompt. Deeper into a completion, `E[A|h]` is the prefix's value
  relative to its group, which is about ±1 for prefixes that are already
  committed to success or failure. With non-centered rewards (e.g.
  plus_minus) `E[A|h]` is nonzero everywhere. The drift has magnitude
  `|E[A|h]| * ||q - p||` at every token, while the covariance is concentrated on
  the few decision tokens.
* **It compounds.** The sampler is rebuilt from the trainer at every step
  (q = f(p), where f is noise, quantization or staleness). The drift therefore
  moves p in a direction set by f, and the next q = f(p) follows. On failing
  prefixes (A < 0) the drift pushes p away from q. This happens most on
  low-probability tokens, where quantization and noise change logits most in
  relative terms, so the drift inflates them. That makes the sampler produce
  more unusual tokens, which fail more, which makes A more negative. This
  feedback loop is the collapse we observe.

**Why IS corrections and clipping do not fix it.** Weighting by
`w = p_y/q_y` makes the expectation `sum_v p_v E[A|h,v] s_v`. With the exact
ratio the drift is gone, but the variance is large, and at the sequence level
it grows exponentially. Also, `E_q[A|h,v]` still comes from continuations
sampled by q, so the correction is only one token deep. Every practical
variant (TIS, MIS, PPO-style clipping, masking) replaces w by a bounded w~.
The leading term then becomes

    E_q[w~ A s_y | h] = E_q[A|h] * sum_v q_v w~_v s_v + Cov_q(...),

and `sum_v q_v w~_v s_v = sum_v min(p_v, c q_v) s_v != 0` as soon as clipping is
active. The drift is smaller but still a nonzero bias, so it still compounds,
only more slowly. Stronger token clipping can make it worse. Clipping is
asymmetric: it truncates exactly the tokens where p >> q, so `sum_v w~_v q_v s_v`
gets a systematic component toward the clipped tokens. Masking (w = 0)
removes the tokens' contribution entirely but keeps the others, so the
effective token distribution, and hence the drift direction, is neither p nor q.

## 2. The modification

Replace each token's score by its centered version, `s_y(h) - s_bar(h)`. Keep
the same advantage A:

    g = sum_t A * ( s_{y_t}(h_t) - s_bar(h_t) ).

**Unbiased covariance, no drift.** By (1) and `E_q[s_bar(h)|h] = s_bar(h)`:

    E_q[A (s_y - s_bar) | h] = E_q[A s_y | h] - E_q[A|h] s_bar(h) = Cov_q(A, s_y | h).     (2)

The drift is removed exactly, at every prefix, for any q. Nothing is
reweighted, so there are no ratios, no variance blow-up and no clipping
threshold. Centering also lowers variance: `s_y - s_bar` has mean 0 under q,
just as a baseline does.

**Zero change on-policy.** If q = p then `s_bar = 0` identically, so the new
loss has exactly the gradient of the old one. It never trades signal for
stability when there is no mismatch.

**What the remaining update learns.** In logit space the per-token direction
becomes `A * (e_y - q)` instead of `A * (e_y - p)`. Its mean under y ~ q is
`sum_v q_v (E[A|h,v] - E[A|h]) e_v`. It raises tokens with above-average
continuation value under the sampler, and lowers those below. When the reward
does not depend on the choice at h, the mean update at h is now zero. Before,
it was `E[A|h](q - p)`. The remaining effect of mismatch is that the
covariance is taken under q instead of p. That changes the size of the
learning signal by O(q - p), but not its sign or its fixed points: a token
with no effect on the reward gets zero expected update.

**Fixed points.** With the old loss, even a policy that already maximizes
reward has a nonzero expected update `sum_h d_q(h) E[A|h] (q - p)(h)`. So the
optimum is not a stationary point under mismatch, and p keeps drifting with
q = f(p) following it. With the centered loss, the expected update at any
prefix is a covariance, which vanishes wherever the advantage does not
depend on the choice. The stationary points are those of the reward again.
This is what stops the compounding.

### Surrogate for one backward pass

With sg = stop-gradient:

    loss_t = -A * ( log p(y_t|h) - sum_v sg[q(v|h)] log p(v|h) ),
    grad loss_t = -A * ( s_y - sum_v q_v s_v ) = -A * ( s_y - s_bar ).

The logit gradient of the bracket is `(e_y - p) - (q - p * sum_v q_v) = e_y - q`.

### Top-k head only

The sampler returns only its top-k (k = 20) logprobs `{q_v : v in H}`. Model the
tail as proportional to p:

    q~_v = q_v                      (v in H)
    q~_v = rho * p_v                (v not in H),     rho = q_tail / p_tail,
    q_tail = 1 - sum_{v in H} q_v,  p_tail = 1 - sum_{v in H} p_v.

This q~ is a proper distribution (`sum q~ = 1`), it agrees with q on the head,
and it puts the right total mass on the tail. Using
`sum_v p_v s_v = 0`, the tail part of `s_bar` becomes a head-only sum:

    sum_{v not in H} rho p_v s_v = -rho sum_{v in H} p_v s_v
    =>  s_bar~ = sum_{v in H} (q_v - rho p_v) s_v,

so

    loss_t = -A * ( log p(y_t|h) - sum_{v in H} sg[q_v - rho p_v] log p(v|h) ).

The weights `q_v - rho p_v`, rho and the masses are stop-gradiented. The
sampled token's own `log p(y_t|h)` is always kept, even when `y_t` is not in H.
The logit gradient of the bracket is exactly `e_y - q~`: on head logits it
is `e_y - q_v`, the same as with the full q, and on tail logits it is
`e_y - rho p_u`. The added term alone contributes `q~ - p`, which is
`q_v - p_v` on the head and `(rho - 1) p_u` on the tail. I checked numerically that autodiff through
`centered_baseline` gives `q~ - p` to 2e-7.

**Error from the tail model.** The only remaining drift is
`E_q[A|h] * sum_{v not in H} (q_v - rho p_v) s_v`. In logit space its L1 norm is
at most `2 * q_tail`, which is the sampler mass outside its top-20. For LLM
next-token distributions under the mismatches considered, this is usually
1e-3 or smaller. It is far below the full drift `|q - p|_1`, which the head
captures exactly. The tail's total mass is matched exactly, so the tail as a
whole is not pushed up or down (the logit gradient on the tail sums to
`q_tail - p_tail`, the same as with the exact q).

### Implementation details (`losses.py`)

* `sampler_keys` now also forwards `sampler_topk_ids` and
  `sampler_topk_logprobs` into the chunked loss. `centered_baseline(logp_vocab,
  ids, logprobs)` computes `sum_{v in H} sg[q_v - rho p_v] log p_v` per
  position. It gathers p at the head ids with `take_vocab`, so it works with TP
  sharding, and only [B, chunk, 20] extra memory is needed.
  The token score becomes `logp - centered_baseline(...)`.
* Numerics: head logprobs are cast to float32. Non-finite (padded) head
  entries get weight 0. Ids are clipped into range. q_tail and p_tail are
  clipped to [0, 1]. `rho = (q_tail + 1e-6) / (p_tail + 1e-6)`, so when both
  tails vanish rho tends to 1, which is the on-policy-consistent value. rho is
  capped at 1e3: the head weights involve `rho * p_v`, and float32
  cancellation error scales with rho. A cap matters only when p puts less
  than about 1e-3 of q's tail mass outside H. It then under-weights only that
  tail, and the error is bounded by q_tail.
* With an `ISCorrection` the token IS weight w multiplies the centered score
  (`w * (log p_y - baseline)`), and sequence-level weights multiply A as
  before. The default (`correction=None`) needs no weights at all.
* If the batch has no top-k head, the loss falls back to plain REINFORCE.
* Metrics are unchanged. The loss value is the surrogate, so its value (not its
  gradient) differs from the old one.

## Configuration

No configuration changes are needed. The modification is active with the
defaults and `correction=None`.
