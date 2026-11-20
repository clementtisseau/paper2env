# Removing the sampler-induced drift from the REINFORCE loss

## What is wrong with the current loss

For a completion y_1..y_T drawn from the sampler q, the loss in `loss_fn_rl`
is (with no correction, w = 1)

    L = -(1/N) sum_i A_i sum_t log p(y_it | h_it),

where h_it is the prefix, A_i the group-centered advantage of completion i, and
N the number of completion tokens. Its gradient is an estimate of
-E_q[A s_y], with s_v(h) = grad_theta log p(v|h).

Fix a prefix h and take the expectation over the next token y ~ q(.|h). Since
E_q[s_y | h] = sum_v q(v|h) s_v =: s_bar(h), the identity

    E_q[A s_y | h] = E_q[A | h] s_bar(h) + Cov_q(A, s_y | h)

holds exactly. The second term is the reward signal, the covariance between
advantage and token choice. The first term is a drift: it is proportional to the
expected advantage at the prefix, and it points along s_bar(h).

On-policy (q = p), s_bar(h) = sum_v p(v|h) grad log p(v|h) = grad sum_v p(v|h) = 0,
so the drift vanishes. Under a sampler mismatch it does not. Two facts make it
harmful and not zero-mean noise:

1. s_bar(h) is the gradient of a cross-entropy. With q fixed,
   s_bar(h) = grad_theta sum_v q(v|h) log p(v|h) = -grad_theta H(q(.|h), p_theta(.|h)).
   The drift is therefore a supervised step that moves p(.|h) toward q(.|h),
   weighted by E_q[A | h]. Nothing in the reward asks for this step.
2. The advantage is group-centered only across a group, not conditionally on a
   prefix, so E_q[A | h] is generally nonzero for individual prefixes. Its sign
   changes from prefix to prefix, so the drift does not cancel in expectation.
   Because the sampler is refreshed from the trainer each step, the sampler's own
   error (q - p) is fed back into the next step's drift. This explains the
   compounding degradation.

Clipped or truncated token-level importance weights do not remove this term.
A clipped weight w(p/q) is a nonlinear function of y, so E_q[w(y) A s_y | h] is
not E_p[A s_y | h]. The clipping only changes which part of the drift survives.

## Modification

The drift can be subtracted exactly, because the sampler's distribution is known
at each position. The sampler returns its top-K logprobs (K = 20) at every
position, and the trainer has the full logits. Let K_h be the sampler's top-K
token set at h, and define

    Q = sum_{v in K_h} q(v|h),     P = sum_{v in K_h} p(v|h),
    rho = (1 - Q) / (1 - P),       c_v = q(v|h) - rho p(v|h)   for v in K_h,
    s_hat(h) = sum_{v in K_h} c_v s_v.

c_v is a stop-gradient weight. `s_hat` is a gradient estimate of s_bar(h), and
its gradient is a function of the trainer's logprobs of the top-K tokens only.
Replace the per-token score by a baseline-subtracted score:

    L = -(1/N) sum_i A_i sum_t [ log p(y_it|h_it) - b(h_it) ],
    b(h) = sum_{v in K_h} c_v log p(v|h),     grad b = s_hat(h).

Each token therefore contributes -A_i (s_y - s_hat(h)). The default
`loss_fn_rl` applies this whenever the batch has `sampler_topk_ids` and
`sampler_topk_logprobs`, which `train_rl.py` and `dev/mini.py` both supply, and
`sampler_keys` now forwards those two keys. With a correction the same baseline
multiplies the token weight w.

## Why it works

Fix h, and take the expectation over y ~ q(.|h) and over the advantage, which
depends on the whole completion. Since s_hat(h) depends only on the prefix, the
expectation of A_i s_hat(h_t) equals E[A | h_t] s_hat(h_t), so the drift is
removed in expectation even though the sequence's own advantage A_i is used:

    E_q[A (s_y - s_hat) | h] = Cov_q(A, s_y | h) + E_q[A | h] (s_bar(h) - s_hat(h)).

The error term is determined by the tail. Using sum_all_v p(v|h) s_v = 0, which
holds for any p, we have sum_{K_h} p_v s_v = -sum_{tail} p_v s_v, so

    s_bar - s_hat = sum_{tail} (q_v - rho p_v) s_v.

- If K_h is the full vocabulary, the tail is empty and s_hat = s_bar exactly,
  for any rho. This was checked numerically on a 30-token softmax: the expected
  gradient matched -Cov_q(A, s_y) to 6e-9. With top-8 of 30 tokens the residual
  was 1.6e-2 against a plain-PG error of 4.4e-2.
- The rho choice makes the tail weights sum to zero, sum_tail (q_v - rho p_v) = 0.
  The error then vanishes whenever the sampler's tail is proportional to the
  trainer's tail, and it is otherwise a within-tail variation of size about the
  tail mass.
- On-policy, q = p gives Q = P and rho = 1, so c_v = 0 on K_h and s_hat = 0.
  The modified loss is then identical to plain REINFORCE, so no bias is added
  where the sampler is correct. A naive choice that omits rho would add the
  on-policy bias -sum_tail p_v s_v, which is nonzero. This is why rho is needed.

Thus E_q[A (s_y - s_hat) | h] = Cov_q(A, s_y | h) up to a tail term, and it is
the covariance that on-policy REINFORCE estimates. The sampler's error no longer
enters the update through the expected advantage, so the feedback loop that
compounds the drift is cut at its source.

The cost is variance from the baseline term, which is bounded by |c_v| and the
top-K logprob range. rho is clipped to [0, 4] to guard against a near-full top-K
whose trainer tail is tiny. This is the only approximation beyond the tail error.

## Configuration

No configuration change is needed. The modification is on by default in
`loss_fn_rl`, and it is inactive only when the batch carries no top-K keys.

## Empirical check (dev/mini.py, CPU)

With the default mini setting (sampler noise 0.3, 600 steps, one seed), the
plain loss ends with eval reward 0.66 and the modified loss with 0.84 (peak
train reward 0.68 and 0.96). Under harsher mismatch, the plain loss collapses to
zero eval reward in every run I tried, while the modified loss keeps eval reward
around 0.5 to 0.8:

| setting | plain PG | modified |
| --- | --- | --- |
| noise 0.6, seeds 0, 1 | collapses to 0 | eval 0.6 and 0.5 |
| noise 0.3, 200 extra tokens | collapses to 0 | eval about 0.8 |
| noise 0.45, seed 2 | collapses to 0 | eval about 0.75 |

The sampler-trainer KL metric `kl_samp_train` rises under the modified loss
(about 2 to 10 at the end of harsh runs), while it is near zero for the plain
loss, which collapses. I did not diagnose this. It may mean the trainer moves
away from the sampler on sampled tokens while the reward holds.
