# Sampler-centered REINFORCE: removing the mismatch drift

## Summary

The loss now uses the score **centered under the sampler** instead of the raw
score of the sampled token:

    loss = - (1/N) * sum_{i,t} A_i * w_{i,t} * [ log p(y_t | h_t) - sum_v sg(qhat(v | h_t)) log p(v | h_t) ]

Here `qhat(.|h)` is the sampler's next-token distribution, rebuilt from the
top-20 logprobs it returns. When an `rl.is.*` token correction is set, `qhat`
is reweighted by it (see below). `sg` is stop-gradient. On-policy (`q = p`)
the extra term has zero gradient and the loss is plain REINFORCE. Under
mismatch it removes the drift term exactly. The centering is always on and
needs no flag. It works with `correction=None` and with every point of the
`rl.is.*` grid. No configuration change is needed.

## 1. What is wrong with the current loss

Fix a prefix h. Write p = p_theta(.|h) for the trainer, q = q(.|h) for the
sampler, s_v = grad_theta log p(v|h) for the score of token v, and
a(v) = E[A | h, y = v] for the expected advantage after emitting v. REINFORCE
ascends A s_y with y ~ q. Its conditional expectation is

    g(h) = E_q[A s_y | h] = sum_v q(v) a(v) s_v
         = sum_v q(v) (a(v) - abar) s_v + abar * sum_v q(v) s_v
         = Cov_q(A, s_y | h) + abar(h) * sbar(h),

    abar(h) = E_q[A | h],    sbar(h) = sum_v q(v|h) grad log p(v|h).

**On-policy**, sbar = sum_v p grad log p = grad sum_v p(v) = grad 1 = 0. Only
the covariance remains: the update moves probability toward tokens with
above-average advantage and away from tokens with below-average advantage.
Any constant shift of the advantage at that prefix has no effect. This is
why baselines do not bias policy gradients.

**Under mismatch** sbar != 0. To see what it does, take the gradient with
respect to the logits z of the softmax at h. There grad_z log p(v) = e_v - p, so

    sbar(h) = sum_v q(v) (e_v - p) = q - p       (in logit space)

and the expected logit update is

    E_q[A (e_y - p)] = Cov_q(A, e_y) + abar(h) * (q - p).

The second term is a **drift** that has nothing to do with which token is
better. Its direction is the sampler/trainer discrepancy q - p, and its sign
and size come from the mean advantage at that prefix:

* If abar(h) > 0 (a prefix that is better than average), p is pulled toward
  q: the trainer copies the sampler's errors.
* If abar(h) < 0 (failing prefixes, which are most prefixes under
  `plus_minus` / `zero_minus` rewards and many under group centering), p is
  pushed **away** from q, so the trainer moves opposite to the sampler's
  perturbation.

The drift is not zero-mean noise, for three reasons. First, abar(h) is
correlated with h: group centering gives sum A = 0 over a group, but not
E[A | h] = 0 at each prefix. Second, q - p is a fixed function of the weights
(fixed weight noise, deterministic quantization error, KV quantization error,
or a stale copy). Third, the sampler is rebuilt from the trainer at every
step as q = Q(p_theta). So moving p away from q does not close the gap.
Because Q is applied to the new weights, the trainer keeps moving along
roughly the same direction, step after step. The bias adds up over training:
the logits drift and the entropy changes, which changes the mismatch
q - p = Q(p) - p, which feeds the drift again. The result is the slow
degradation and collapse that the task describes.

**Why IS and clipping do not fix it.** A token-level IS weight w(p/q) makes
the expectation sum_v q w(v) a(v) s_v. With exact IS (q w = p) this is the
on-policy gradient and sbar disappears. In practice the weight is clipped,
dropped, or one-sided (TIS, MIS, PPO, DAPO, TOPR, ...). Then the effective
measure m = q w differs from p, and the same decomposition gives a drift
abar_m * sum_v m(v) s_v = abar_m * (m - M p) in logit space, with M = sum m.
That is smaller than q - p but still nonzero and still systematic. Tighter
token clipping makes m differ more from p on exactly the tokens where the
mismatch is large, so it can increase this residual drift. Exact IS also has
unbounded variance on the tokens where q is much smaller than p. Seq-level
weights multiply A but leave sum_v q(v) s_v unchanged, so the drift stays.
Each method rescales the reward-dependent part. None of them subtracts the
part that is not reward-dependent.

## 2. The fix: center the score under the sampler

Replace s_y by the centered score s_y - sbar(h). The gradient term sbar(h)
depends only on the prefix, not on the sampled token y. So

    E_q[A (s_y - sbar(h)) | h] = sum_v q(v) a(v) s_v - abar(h) sbar(h)
                               = Cov_q(A, s_y | h).

The drift term disappears exactly, for every prefix and every value of
abar(h). The estimator uses the rollout's own advantage A, with no
per-prefix value estimate. It is unbiased for the covariance because sbar(h)
is deterministic given h, so E_q[A sbar(h) | h] = abar(h) sbar(h).

As a loss, sbar(h) = grad_theta sum_v sg(q(v|h)) log p(v|h): the gradient of
the trainer's negative cross-entropy to the detached sampler distribution.
The per-token objective is therefore

    A * [ log p(y|h) - sum_v sg(q(v|h)) log p(v|h) ].

In logit space its gradient is A (e_y - q) instead of A (e_y - p). REINFORCE
uses the *trainer's* mean as the baseline of the one-hot sample. That
baseline is only correct when the sample is drawn from p. The fix uses the
mean of the distribution the sample was actually drawn from, so
E_q[e_y - q] = 0 holds exactly.

Properties:

* **On-policy it is exactly REINFORCE.** If q = p, sum_v p grad log p = 0, so
  the added term has zero gradient.
* **A constant advantage gives zero update.** If a(v) = c at a prefix, the
  expected update is c * sum_v q(v)(s_v - sbar) = 0. With REINFORCE under
  mismatch it is c (q - p). The mismatch can no longer turn the mean
  advantage into motion.
* **Remaining error.** What is left is Cov_q(A, s_y) instead of Cov_p(A, s_y).
  The reward signal is measured under a slightly different distribution, but
  in the correct direction: tokens with above-average advantage are still
  pushed up. The difference is second order (q - p times the advantage
  spread). It does not compound, because it has no component along q - p
  that is independent of the reward. If wanted, IS can still correct it (§3).
* **Variance.** The added term does not depend on the sampled token. It
  replaces the implicit baseline p by q in logit space, so per-token variance
  barely changes (|e_y - q|^2 versus |e_y - p|^2).

### Top-k approximation of q

The sampler returns only its top-K (K = 20) logprobs. Split the vocabulary
into K and the tail. The tail has sampler mass q_T = 1 - sum_{v in K} q(v)
and trainer mass p_T = 1 - sum_{v in K} p(v). Assume q is proportional to p
inside the tail, q(v) = q_T p(v)/p_T for v not in K. Then the tail part of
sbar is

    sum_{v not in K} q(v) grad log p(v) = (q_T/p_T) grad sum_{v not in K} p(v) = q_T grad log p_T,

so

    sbar(h) = sum_{v in K} q(v) grad log p(v|h) + q_T grad log p_T(h).

This is exact when q and p have the same shape inside the tail. It also
preserves the key identity: if q = p, then sbar = grad(sum_K p + p_T) = 0
exactly, whatever the split. With K = 20 the tail mass is usually tiny for an
LLM, so the residual error is O(q_T * tail-shape mismatch). In float32,
log(1 - sum_K p) is imprecise when p_T is below about 1e-5. At such positions
the tail term is dropped and the top-K weights are renormalized.

### Combination with IS corrections (`rl.is.*`)

With a token-level correction w = w(p/q), the estimator A w_y s_y has
expectation sum_v m(v) a(v) s_v with m(v) = q(v) w(p(v)/q(v)). In the code,
m = `exp(log_mass(log p, log q, ...))`, computed on each top-K token and on
the tail. Center under the normalized measure mhat = m / sum m:

    A w_y (s_y - sum_v mhat(v) s_v)   has expectation
    sum_v m(v) a(v) (s_v - sbar_m),    sbar_m = sum_v mhat(v) s_v.

This is again zero when a(v) is constant, so it is drift-free for every
band, clamp/drop and sign rule. When the correction does nothing (w = 1),
mhat = q and we recover the formula above. If a correction drops all mass
at a position (sum m = 0), that position falls back to the plain score.

For sequence-level corrections (`seq_mean`, `seq_sum`), the weight W only
rescales A. Centering under q gives sum_v q(v) E[A W | h, v] (s_v - sbar),
which is also zero when E[A W | h, v] is constant in v.

## 3. Why this makes training stable

The instability comes from the term abar(h) (q - p) in the expected update.
That term is first order in the mismatch and has a consistent direction
across steps, because the sampler is a deterministic function of the current
trainer. Sampler-centering removes this term from the expected update
exactly, at every prefix and step, for any kind of mismatch (noise,
int8/fp8, KV quantization, staleness), since it only uses q as reported by
the sampler. What remains is the advantage-weighted covariance, which always
points toward higher reward under the distribution the data came from. The
update is still a valid ascent direction for the reward, and the
mismatch-driven feedback loop between q - p and the trainer's update is
broken. Unlike IS weights, the correction has no ratio p/q in it. It does
not blow up where q is much smaller than p, and it does not need clipping.

## Implementation notes (`losses.py`)

* `sampler_keys` now also forwards `sampler_topk_ids` and
  `sampler_topk_logprobs` into the chunked loss.
* `sampler_centering` gathers the trainer logprobs on the sampler's top-K ids
  (`take_vocab`, which is TP-safe). It builds the detached weights
  mhat (q, or q * w under a token correction) and the tail term, and returns
  the baseline sum_v sg(mhat_v) log p_v + sg(mhat_T) log p_T.
* The per-token score is `w * (logp - baseline)`. Everything else is
  unchanged: chunking and rematerialization, seq-level weights, the
  normalization by the number of tokens, and the metrics. One metric is
  added: `q_tail`, the tail weight that is used.
* If the batch has no top-k keys, the loss falls back to plain REINFORCE.
* A numerical check confirms that the logit gradient equals
  -A (e_y - qhat) / N exactly. Here qhat is the top-K q plus q_T times the
  tail shape of p.
