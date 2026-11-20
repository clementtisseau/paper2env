# Removing the sampler-mismatch drift from REINFORCE

## 1. What is wrong with the current loss

The loss is `-(1/N) sum_i A_i sum_t w_t log p(y_t | h_t)`, with tokens drawn from the
sampler q, not from the trained policy p. Its gradient per token is `A w s_y`, where
`s_v = grad_theta log p(v|h)` is the score of token v at prefix h.

Fix a prefix h and take the expectation over the next token y ~ q(.|h). A is the advantage
of the whole completion, so it depends on y (through the continuation) and on h. Then

    E_q[A s_y | h] = E_q[A | h] * s_bar(h) + Cov_q(A, s_y | h),
    s_bar(h)       = sum_v q(v|h) s_v.

**On-policy (q = p).** `s_bar(h) = sum_v p(v|h) grad log p(v|h) = grad sum_v p(v|h) = grad 1 = 0`.
Only the covariance term remains, and it is the policy gradient: it raises the tokens that lead
to above-average advantage *relative to the other tokens at this prefix*. The expected advantage at h
(`E[A|h]`, the part of the reward the token choice cannot explain) does not move the policy.

**Mismatch (q != p).** `s_bar(h) != 0`, so a second term `E_q[A|h] s_bar(h)` appears. It has
nothing to do with which token is better. For a softmax head with logits z(h),
`grad_z log p(v|h) = e_v - p(.|h)`, so

    s_bar(h) in logit space = sum_v q(v) (e_v - p) = q(.|h) - p(.|h).

The drift term moves the logits along `E_q[A|h] (q - p)`:
- at prefixes with `E[A|h] > 0` (good prefixes) it pulls p toward the sampler q, i.e. toward the
  perturbed/quantized model's errors;
- at prefixes with `E[A|h] < 0` (bad prefixes) it pushes p *away* from q, i.e. it amplifies
  the trainer/sampler difference in the opposite direction.

Group centering makes `E[A] = 0` only for the prompt as a whole (with `plus_minus` rewards,
not even that). Conditional on an intermediate prefix, `E[A|h]` is generically non-zero, and it is
strongly correlated with the prefix (early wrong steps give `E[A|h] < 0` for every later
token). So the term does not average out: it is a systematic bias, a deterministic
function of (p, q, h), not zero-mean noise. Because the sampler is copied from the
trainer at every step (`q_{k+1} = Perturb(p_{k+1})`), the mismatch `q - p` does not
shrink. The bias is re-applied every step in a direction tied to the current
perturbation, so it accumulates in the weights. This is the compounding drift and
eventual collapse.

**Why IS and clipping only slow it down.** With a token weight w(p/q) the drift term
becomes `E_q[A|h] s_bar_w(h)` with `s_bar_w(h) = sum_v q(v) w(v) s_v`. With exact,
unclipped `w = p/q`, `s_bar_w = sum_v p(v) s_v = 0`, so naive IS removes the drift but has
unbounded variance. Every bounded variant (TIS clamp, MIS/PPO band, dropping) makes
`q w != p` on some tokens, so `s_bar_w = sum_v (q w - p)(v) s_v != 0` again. The drift is
smaller but never zero. Tighter token clipping replaces p by q (or by 0) on more tokens. That
increases `||q w - p||` and can make the bias *larger* than with a looser band: dropping a
token removes it from the sum and moves the effective mass even further from p. Sequence-level
weights rescale A but leave `s_bar(h)` untouched at every position.

## 2. The modification: subtract the sampler-expected score at every position

Remove the drift term exactly by centering the score under the sampler distribution
at each position:

    g = A * ( w_y s_y - s_bar_w(h) ),        s_bar_w(h) = sum_v q(v|h) w(v) s_v

(with no correction, w = 1 and s_bar_w = s_bar). As a loss this is a detached
"sampler-expected log-likelihood" baseline subtracted from each token's log-prob:

    L = -(1/N) sum_i A_i sum_t [ w_t log p(y_t|h_t) - sum_v sg(q(v|h_t) w(v)) log p(v|h_t) ].

The baseline has gradient `s_bar_w(h_t)` because its coefficients are detached.

### Derivation that the drift is removed

`s_bar_w(h)` is a deterministic function of h (it does not depend on y), so

    E_q[A s_bar_w(h) | h] = E_q[A | h] s_bar_w(h).

Therefore

    E_q[ A (w_y s_y - s_bar_w(h)) | h ]
      = E_q[A w_y s_y | h] - E_q[A|h] s_bar_w(h)
      = E_q[A|h] E_q[w_y s_y|h] + Cov_q(A, w_y s_y | h) - E_q[A|h] s_bar_w(h)
      = Cov_q(A, w_y s_y | h)                       (since E_q[w_y s_y | h] = s_bar_w(h)).

At every prefix the expected update now has only the covariance term, as on-policy.
It moves the policy only toward tokens whose advantage is above the q-average at that
prefix. In logit space the update is `Cov_q(A, e_y - p) = Cov_q(A, e_y)`. This is the
q-covariance of the advantage with the token indicator: a reward signal, zero whenever A
does not depend on the token choice at h. The `E_q[A|h](q - p)` pull toward/away from the
sampler is gone, whatever the size or kind of mismatch (weight noise, int8/fp8, KV
quantization, staleness). Each of these just gives some q != p.

Properties:
- **No-op on-policy.** If q = p and w = 1, `s_bar = 0` and the gradient is exactly REINFORCE.
  Numerically, the gradients match to 1e-8.
- **Unbiased.** It needs no estimate of `E[A|h]`. The single-sample `A * s_bar_w(h)` is an unbiased
  estimate of the drift because s_bar_w does not depend on y.
- **Bounded.** In logits the extra term is `-A (q w - p)`, with `||q w - p||_1 <= 2` for any
  correction with `w <= 1/q` mass. Unlike exact IS it adds no variance blow-up.
- **Composes with any `rl.is.*` correction.** For a correction, the baseline uses the corrected
  mass `q(v) w(v)` (via the same `log_mass`), so it cancels exactly the drift that correction
  leaves (`sum_v (q w - p)(v) s_v`). For sequence-level corrections, the advantage
  is rescaled by w_seq and the centering applies per position unchanged.
- **The residual is a variance-type term, not a drift.** What remains is the difference
  between `Cov_q(A, s_y|h)` and `Cov_p(A, s_y|h)`. It is second-order: it is proportional to
  how much the reward signal itself differs between q- and p-continuations, and it vanishes
  when A does not depend on the token. It has no component along `E[A|h]`, so it does not
  accumulate from the large, prefix-correlated part of the advantage.

### Computing s_bar(h) from the sampler's top-20

The sampler returns only its top-K (K = 20) logprobs per position (`sampler_topk_ids`,
`sampler_topk_logprobs`, aligned with `sampling_token_logprobs`). Let `Q_K = sum_{k<=K} q_k`
and `P_K = sum_{k<=K} p(id_k)`. For the unreported tail, the best guess without further
information is that q has the trainer's shape:
`q(v) ~= p(v) (1 - Q_K)/(1 - P_K)` for v outside the top-K. Then

    s_bar(h) ~= sum_{k<=K} q_k s_{id_k} + (1-Q_K)/(1-P_K) * sum_{v not in top-K} p(v) s_v ,

and the tail sum is computed exactly from the full trainer softmax:
`sum_{v not in K} sg(p_v) log p_v = sum_v sg(p_v) log p_v - sum_{k<=K} sg(p_k) log p_k`.
For a correction, the tail ratio is the aggregate `r = (1-P_K)/(1-Q_K)` and the tail mass
is `log_mass` of that ratio. If q has all its mass in the top-K (the usual case for LMs), this is
exact. Otherwise the error is confined to the tail's *shape* `(q_tail - p_tail)`, whose
mass is already correct. The tail factor is capped at 100 to avoid fp32 cancellation
when `1 - P_K` is tiny. If the top-k keys are missing from the batch, the loss falls back
to the original REINFORCE.

## 3. Implementation (`losses.py`)

- `sampler_keys` now forwards `sampler_topk_ids` and `sampler_topk_logprobs`.
- `sampler_baseline(logp_vocab, ids, logq, positive, c)` returns, per position, the detached-
  coefficient baseline `sum_v sg(q_v w_v) log p_v` (top-K plus tail model above). It also returns
  the metric `drift_l1 = ||q_hat - p||_1`.
- In `loss_fn_rl` the per-token score is `w*logp - baseline`. The change is on by default
  (`correction=None`), needs no flags, and works with every `ISCorrection`.

Sanity check (CPU dev setting, noise 0.3, `plus_minus`, 600 steps). Original PG: final train
reward 0.58 / 0.60 and eval 0.67 / 0.70 (seeds 0 / 1). Centered loss with no correction: train
0.85 / 0.84 and eval 0.84 / 0.82, still improving at the end.

## Configuration

No configuration changes are needed. The modification is active with the default `rl.is.*`
settings (no correction).
