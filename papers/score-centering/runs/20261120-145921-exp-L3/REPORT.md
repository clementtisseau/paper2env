# Drift-free REINFORCE under sampler mismatch

## What is wrong with the current loss

`loss_fn_rl` (with `correction=None`) minimizes

    L = -(1/N) sum_b sum_t A_b log p(y_bt | h_bt),

where h_bt is the prefix, y_bt the sampled token, A_b the group-centred advantage of
completion b, and N the number of completion tokens. Its gradient is -A_b s(y_bt | h_bt),
with score s_v(h) = grad_theta log p(v|h).

The tokens are sampled from q (the sampler), not p (the trainer). Fix a prefix h and
write A_v = E[A | h, y = v] (the continuation is also drawn from q), and
Abar = E_q[A | h] = sum_v q(v|h) A_v. Then

    E_q[A s_y | h] = sum_v q_v A_v s_v
                   = sum_v q_v (A_v - Abar) s_v + Abar * sum_v q_v s_v
                   = Cov_q(A, s_y | h) + E_q[A | h] * s_bar_q(h),

with s_bar_q(h) = sum_v q(v|h) grad log p(v|h). The first identity uses
sum_v q_v (A_v - Abar) = 0, which gives Cov_q(A, s_y|h) = sum_v q_v (A_v - Abar) s_v.

On-policy (q = p) the drift vanishes: s_bar_p(h) = sum_v p_v grad log p_v = grad sum_v p_v = 0,
so E_p[A s_y | h] = Cov_p(A, s_y | h), the signal we want.

Under mismatch, s_bar_q(h) = sum_v (q_v - p_v) s_v is generically nonzero and first order
in the mismatch. The current loss therefore follows the expected gradient

    g_current = -E_h[ E_q[A|h] * s_bar_q(h) ]  -  E[Cov_q(A, s_y|h)],

whose first term is a drift that does not come from the reward signal. Its direction is
set by E_q[A|h], the expected advantage of the prefix. This is not zero mean. Group
centring makes the advantage average zero over a group, not conditionally on each prefix,
so E_q[A|h] is systematically positive on prefixes that lead to solutions and negative on
the others. The sampler is refreshed from the trainer at every step, so the same
sign-consistent drift is applied again and again. That explains the compounding
degradation and the collapse.

Token-level IS weights do not remove this term. A weight p_y/q_y on the sampled token
gives E_q[(p_y/q_y) A s_y | h] = sum_v p_v A_v s_v, but A_v still comes from continuations
drawn from q, so each later token contributes its own drift. Clipping changes the weights
and so reintroduces bias. This matches the observation that IS and clipping slow the
degradation without removing it.

## The modification

Subtract from each token's score a baseline that carries the q-expected score:

    c(h) = sum_v q(v|h) log p(v|h),      grad_theta c(h) = s_bar_q(h),

and use the loss

    L = -(1/N) sum_b sum_t w_bt A_b ( log p(y_bt|h_bt) - c(h_t) ),

where w_bt is the optional IS weight (w = 1 when `correction=None`). The baseline c is
computed from the sampler's top-20 logprobs `sampler_topk_ids` / `sampler_topk_logprobs`:

    c(h) = sum_{v in top20(q)} exp(sampler_topk_logprob_v) * log p(v|h),

with exp(sampler logprob) stop-gradiented but log p(v|h) not, so that c contributes its
gradient s_bar. The q weights are the sampler's unnormalized probabilities (no renormalization).

## Why it works

Expected update under q, with the full-vocabulary baseline, is

    E_q[ A (s_y - s_bar_q(h)) | h ] = E_q[A s_y|h] - E_q[A|h] s_bar_q(h) = Cov_q(A, s_y | h).

The drift term E_q[A|h] s_bar_q(h) is removed exactly. What remains is the covariance between
the advantage and the score under the distribution that actually generated the data. It
points toward raising the probability of tokens whose continuations earn advantage, which
is the reward signal.

Checks:

1. On-policy (q = p), s_bar_p = 0, so the baseline is zero in expectation and the
   estimator is unchanged in expectation: E_p[A(s_y - s_bar_p)|h] = Cov_p(A, s_y|h).
   Only the variance changes.
2. Under mismatch, the only remaining deviation from the on-policy target is
   Cov_q - Cov_p. Cov_q(A, s_y|h) = sum_v q_v (A_v - Abar_q) s_v has no multiplier
   E_q[A|h], so it does not push consistently in one direction for all prefixes, and it
   vanishes whenever the advantage carries no information about the next token.
3. The baseline must be differentiated through log p. If c were detached, the gradient
   would be unchanged and the drift would remain.

Truncation: with only the top-20 of q, the baseline misses the tail,
s_bar_q - s_bar_top = sum_{v not in top20} q_v s_v. The residual drift is
E_q[A|h] times that tail term, so it is bounded by the tail mass of q at that position.
With the full vocabulary the residual is zero. The metric `tail_mass_q` in `dev/mini.py` measures this.

Variance: the added term A*c(h) has gradient A*s_bar_q(h), which is bounded by |A| times the
score scale, so the variance increase is modest.

## Empirical sanity check (not required)

In `dev/mini.py` (CPU stand-in, noise 0.3, 600 steps, eval reward averaged over the last five
evaluations): the original loss reaches 0.69, 0.67 and 0.69 on seeds 0, 1 and 2. The
modified loss reaches 0.84, 0.82 and 0.82. With 200 extra vocabulary tokens (tail mass
large enough that top-20 matters), the original loss collapses to 0.0 and the modified loss
holds at 0.78.

## Configuration

No configuration changes are needed. The modification is active under the default
`correction=None`, and it needs the sampler's top-20 logprobs, which the trainer already
returns. If those keys are absent, the baseline is zero and the loss reduces to the original.
