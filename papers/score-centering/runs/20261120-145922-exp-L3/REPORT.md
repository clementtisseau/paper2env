# Drift-free REINFORCE under sampler mismatch

## What is wrong with the current loss

`loss_fn_rl` with no correction minimises

    L = -(1/N) sum_{b,t} A_b * log p(y_bt | h_bt),

where h_bt is the prefix, y_bt the sampled token, A_b the (group-centred) advantage of sequence b,
and N the number of completion tokens. Its gradient is -(1/N) sum A_b s(y_bt | h_bt), with the score
s(v | h) = grad_theta log p(v | h).

The tokens are drawn from the sampler q, not from the trainer p. Fix a prefix h and take the
expectation over the next token y ~ q(.|h). Writing s_bar_q(h) = sum_v q(v|h) s(v|h),

    E_q[A s_y | h] = E_q[A | h] s_bar_q(h) + Cov_q(A, s_y | h).                     (1)

This follows from E_q[A s_y] = E_q[A (s_y - s_bar)] + E_q[A] s_bar, and the first term is Cov_q(A, s_y | h) because E_q[s_y - s_bar | h] = 0.

On-policy (q = p) the first term vanishes, since s_bar_p(h) = sum_v p(v|h) grad log p(v|h) = grad sum_v p(v|h) = 0.
Under mismatch it does not:

    s_bar_q(h) = grad_theta sum_v q(v|h) log p(v|h) = sum_v (q(v|h) - p(v|h)) s(v|h)   (exact, since sum_v p s = 0)

which is generally nonzero whenever q != p. The current loss therefore follows

    E_q[grad L | h] = -E_q[A | h] * s_bar_q(h) + (reward covariance term).

The first term is a drift, not a reward signal. Its sign and size are set by the prefix-conditional expected
advantage E_q[A | h], which is not zero even though the group-centred advantages average to zero over the
group. Where E_q[A | h] > 0 (prefixes that lead to reward), the update performs maximum likelihood on q, moving
p toward the sampler's distribution, including its over-represented tail tokens. Where E_q[A | h] < 0, it
pushes p away from q. Because the sampler is refreshed from the trainer at every step, p is pulled toward q,
the drift is recomputed against the new q, and the bias accumulates instead of averaging out.

Importance-sampling weights and clipping only rescale the terms `E_q[A s_y]`; they leave the drift in place
on the tokens that survive. Clipping to a band also removes the correction for the tokens it cuts, so
tightening the band leaves more of the drift, which matches the observation that stronger token-level
clipping can make things worse.

## The modification

Replace the per-token score s(y_t | h_t) by the score minus its sampler mean:

    loss = -(1/N) sum_{b,t} A_b * w_bt * ( log p(y_bt | h_bt) - b(h_bt) ),
    b(h) = sum_v pi(v|h) log p(v|h),     pi(v|h) = q(v|h) for v in top-20 / sum over top-20,

with the baseline b detached in its weight pi and not detached in log p. The gradient of b is
sum_v pi(v|h) s(v|h), an estimate of s_bar_q(h). The factor w_bt is the optional IS weight; it is 1 when
no correction is set. The modification is active with the default arguments. The sampler's top-20
logprobs (`sampler_topk_ids`, `sampler_topk_logprobs`) are now forwarded by `sampler_keys`.

**Derivation.** Let c(h) = s_bar_q(h). It depends on h only, not on y, so subtracting it from the score is a
baseline that does not depend on the sampled token. The expected update under q becomes

    E_q[A (s_y - c) | h] = E_q[A s_y | h] - E_q[A | h] c(h) = Cov_q(A, s_y | h).

The drift term of (1) is removed exactly. When q = p, c = 0 and the loss is unchanged in expectation, so the
modification does not alter the on-policy gradient.

**Approximation.** The sampler's full distribution is not available, only the top K = 20 entries. Writing
eps = 1 - sum_{v in top-K} q(v|h) for the tail mass, the exact c and its top-K estimate satisfy

    c(h) = (1 - eps) * sum_{top-K} pi(v) s(v) + sum_{tail} q(v) s(v),

so the top-K estimate differs from c by at most 2 * eps * max_v ||s(v|h)||. The residual scales with the
sampler's tail mass, which the training loop logs as `topk_mass` (= 1 - eps).

**What remains.** The remaining term is Cov_q(A, s_y | h), the covariance under the sampler's distribution
rather than the trainer's. Its difference from the on-policy covariance is first order in q - p, and it
carries reward information instead of a drift toward q. Standard IS weights can still be applied on top of it,
via the existing `rl.is.*` grid, to address that residual; the drift removal is independent of them.

## Checks

- Exact check at a single position with a 7-token vocabulary and a mismatched sampler: the original gradient
  minus the patched gradient equals E_q[A] s_bar_q to machine precision. The on-policy s_bar_p has norm 7e-17.
- Development runs (`dev/mini.py`, which has exactly 12 tokens so its top-20 is the full vocabulary and the
  approximation error is zero there):
  - Weight-noise 0.3, four seeds, 600 steps, final held-out eval reward: original 0.66, 0.66, 0.68, 0.66;
    patched 0.81, 0.77, 0.86, 0.81.
  - Weight-noise 0.5, two seeds: original collapses to 0 by about step 75 in both seeds; patched stays at
    0.66 and 0.73.
  These are small-model results and are not a test of the real Qwen3-0.6B setting.

## Configuration

No configuration changes are needed. The modification is active under the default settings with no correction, and it works with any correction in the existing grid.
