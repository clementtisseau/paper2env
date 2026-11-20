# Sampler-mismatch drift in REINFORCE, and a score baseline that removes it

## What is wrong with the current loss

`loss_fn_rl` (without a correction) minimizes `-sum_t A * log p(y_t | h_t) / N`,
so its gradient is `-A * s_{y_t}` with `s_v(h) = grad_theta log p(v | h)`.
The rollouts are drawn from the sampler q, not from the trainer p. Fix a prefix
h, draw the next token y ~ q(.|h), and let A be the advantage of the completion.
The expected update at that position is

    E_q[A s_y | h] = E_q[A | h] * s_bar(h) + Cov_q(A, s_y | h),
    s_bar(h) = sum_v q(v|h) grad log p(v|h).

The covariance term is the signal: it is what the reward says about the choice
of token. The first term is a drift. It is proportional to the expected
advantage at the prefix, E_q[A | h], which is generally nonzero at every
prefix (a prefix that already contains an error has negative expected
advantage, a prefix on a good path has positive expected advantage, even with
group-centered rewards), and it points along s_bar(h).

On-policy, q = p and s_bar(h) = sum_v p_v grad log p_v = grad sum_v p_v = 0, so
the drift vanishes. Under a sampler mismatch it does not:

    s_bar(h) = sum_v (q_v - p_v) s_v(h) + sum_v p_v s_v(h)
             = sum_v (q_v - p_v) s_v(h),

which is first order in the mismatch q - p. The drift therefore does not
average out over samples or over steps. It is a systematic term

    Delta theta  proportional to  sum_h d(h) E_q[A | h] * grad_theta sum_v q_v(h) log p_v(h),

that is, a cross-entropy pull of the trainer toward the sampler's
next-token distribution, weighted by expected advantage: it reinforces
the sampler's preferences at prefixes that look good and pushes against them
at prefixes that look bad. The sampler is refreshed from the trainer at every
step, so q is itself moved by this drift, and the bias compounds. This is the
mechanism behind the collapse. Importance ratios do not remove it. A weight
w(y) multiplies A s_y, and E_q[A w(y) s_y | h] still contains a drift term;
clipping only reweights the same biased sum, and it is largest on the tokens
where q and p disagree, which are the tokens that carry the drift.

## The fix

Replace the score of each token by its score minus the sampler's mean score at
that prefix, keeping the same advantage:

    s_y  ->  s_y - s_bar(h).

The expected update is then exact:

    E_q[A (s_y - s_bar) | h] = E_q[A s_y | h] - E_q[A | h] s_bar(h)
                             = Cov_q(A, s_y | h).

Here `s_bar(h)` depends only on h, so it can be pulled out of the expectation.
The drift is removed exactly, and what remains is the covariance, with no
importance ratio. The correction is a baseline that is independent of y, and
it multiplies the same advantage, so the estimator stays a valid
REINFORCE-type estimator.

On-policy the change is exactly zero, not just zero in expectation: q = p gives
rho = 1 and coefficients c_v = q_v - rho p_v = 0 on the head (see below), so the
loss is identical to plain REINFORCE.

### Single-backward surrogate

Let sg[.] be stop-gradient. Per token,

    loss_t = -A_t * ( log p(y_t | h_t) - sum_v sg[q_v(h_t)] log p_v(h_t) ),

whose gradient is `-A_t (s_{y_t} - s_bar(h_t))`. The sampled token keeps its own
`log p(y_t)`. Sum over tokens with the usual mask and divide by the token count.
With an IS correction, the weight w multiplies the whole bracket, which is
`w (log p(y) - baseline)`.

### Only the top-K head is available

The sampler returns its top-K (K = 20) logprobs per position, not the full q.
The tail of q is modeled as proportional to p: for v outside the head H,

    q_v = rho * p_v,    rho = q_T / p_T,

where q_T = 1 - sum_{v in H} q_v and p_T = 1 - sum_{v in H} p_v (the trainer's
tail mass, computed in log space from the full trainer vocabulary to avoid
cancellation). Since sum_v p_v s_v = 0 over the full vocabulary,
sum_{v in T} p_v s_v = -sum_{v in H} p_v s_v, and therefore

    s_bar(h) = sum_{v in H} q_v s_v + rho * sum_{v in T} p_v s_v
             = sum_{v in H} (q_v - rho p_v) s_v.

The tail contributes no gradient beyond this. The surrogate becomes

    loss_t = -A_t ( log p(y_t | h_t) - sum_{v in H} sg[q_v - rho p_v] log p_v(h_t) ),

with rho = 0 if the trainer's tail is empty (H is the whole vocabulary, where
the tail term is identically zero). I checked the identity numerically in
float64: the gradient of the head-based baseline equals sum_v q_v s_v over the
full vocabulary to 5e-17, and on-policy it is zero.

The sampler's top-K logprobs reach the loss batch as `sampler_topk_ids` and
`sampler_topk_logprobs` [B, T, K], aligned with the tokens exactly as
`sampling_token_logprobs` (position t holds the distribution of token t+1).
They are now listed in `sampler_keys`. If they are absent, the loss falls back
to plain REINFORCE, which is the only place the baseline is not applied.

## Why it stabilizes training

- The per-step expected update contains no term that depends on q except the
  covariance Cov_q(A, s_y | h). Its direction follows the reward, not the
  sampler's preferences, so a sampler refresh no longer injects a systematic
  push toward q.
- The bias has no first-order piece left. The remaining error comes from the
  tail model (an error only on the tail mass, of size about |q_T - rho p_T|)
  and from the difference Cov_q vs Cov_p, which is a variance-like effect and
  does not accumulate with a fixed sign.
- The correction uses no importance ratio, so it cannot blow up on rare tokens
  and does not need clipping. Its variance cost is small because the baseline
  term is O(|q - p|) in size.

## Configuration

No configuration changes are needed. The sampler's top-K keys are forwarded by
default, and the correction is active when `loss_fn_rl` is called with
`correction=None`. Use `rl.is.*` settings only as before if an IS correction is
also wanted (the baseline is combined with them, not replaced).

## Checks

- Identity: the head-based baseline gradient matches sum_v q_v grad log p_v over the
  full vocabulary (float64, random logits, random sampler head with
  tail-proportional mass), to 5e-17.
- Mini setup (`dev/mini.py`, sampler weights perturbed by 0.3 relative Gaussian
  noise, plus-minus rewards, 600 steps, 3 seeds): the original loss plateaus at
  eval reward about 0.66 to 0.73 (peak training reward 0.67 to 0.70) with the KL
  between sampler and trainer still rising. The corrected loss reaches eval reward
  about 0.80 to 0.86 (peak 0.93 to 0.96) on the same seeds. This is a proxy,
  not the Qwen3 setting; it was not run on a GPU or with TP sharding.
- `rl.is.level=token rl.is.high=2.0 rl.is.outside=clamp` and
  `rl.is.level=seq_mean ...` run without error with the baseline.
- Not verified: vocabulary-sharded (TP) logits. The head-mask scatter in
  `sampler_baseline` has not been exercised under vocabulary sharding.
