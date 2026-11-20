# Sampler-centered REINFORCE: removing the mismatch drift

## Summary

`loss_fn_rl` now replaces the score of each sampled token, `grad log p(y|h)`, with the
**score centered under the sampler's next-token distribution**,

    g_t = A * ( grad log p(y_t|h_t) - s_bar_q(h_t) ),    s_bar_q(h) = sum_v q(v|h) grad log p(v|h).

This is on by default (`correction=None`) and needs no configuration. On-policy (q = p)
s_bar_q = 0 and the loss is exactly the old REINFORCE loss (the gradients match to the bit).
Under mismatch it removes a deterministic, reward-independent drift term that plain
REINFORCE adds at every prefix. `s_bar_q` is computed from the sampler's top-20 logprobs
(`sampler_topk_ids` / `sampler_topk_logprobs`, now in `sampler_keys`), with the rest of the
vocabulary treated as one "tail" outcome (details below).

## 1. What is wrong with the current loss

Notation: p = p_theta is the trainer, q the sampler. A rollout is y_1..y_T ~ q, h_t its prefix,
A its (group-centered) advantage, s_v(h) = grad_theta log p(v|h). The current loss
(no correction) gives the update  sum_t A * s_{y_t}(h_t).

Fix a prefix h and take the expectation over the next token y ~ q(.|h) and the rest of the
rollout. Write Q(h,v) = E_q[A | h, y=v] and V(h) = E_q[A | h] = sum_v q(v|h) Q(h,v). Then

    E_q[A s_y | h] = sum_v q(v|h) Q(h,v) s_v
                   = sum_v q(v|h) (Q(h,v) - V(h)) s_v  +  V(h) sum_v q(v|h) s_v
                   = Cov_q(A, s_y | h)                  +  V(h) * s_bar_q(h).            (1)

**On-policy** s_bar_p(h) = sum_v p(v|h) grad log p(v|h) = grad sum_v p(v|h) = grad 1 = 0, so only the
covariance survives: the update at h moves probability toward tokens whose continuation value
Q(h,v) is above V(h). That is the policy gradient.

**Mismatched** q != p, so

    s_bar_q(h) = sum_v (q(v|h) - p(v|h)) grad log p(v|h)
               = grad_theta sum_v q(v|h) log p_theta(v|h)          (q held fixed)
               = - grad_theta KL( q(.|h) || p_theta(.|h) ).                              (2)

So the current loss contains, at every prefix, the term

    drift(h) = - V(h) * grad_theta KL(q(.|h) || p_theta(.|h)).

Properties of this term:

* It does not depend on which token was chosen or on how good it was; it is non-zero even at a
  prefix where the next token is irrelevant to the reward (Q(h,v) constant in v), where the true
  gradient is zero. It is not reward signal.
* It is not zero-mean noise: given h it is a fixed vector, and it is summed over every token of
  every rollout. Under group centering V(h) is ~0 at the prompt but not at later prefixes: on a
  failing trajectory the prefixes have V(h) < 0, on a succeeding one V(h) > 0, and with
  plus/minus rewards V(h) has the sign of the success rate everywhere.
* Where V(h) < 0 the drift *increases* KL(q||p): it pushes p away from q, lowering p on the tokens
  q over-weights and, by normalisation, raising p on tokens q under-weights (the long tail of
  unlikely tokens). Where V(h) > 0 it pulls p toward the corrupted q, importing the sampler's
  error into the trainer.
* It compounds. The sampler is refreshed from the trainer every step, q = p_theta perturbed by a
  *fixed-type* error (same noise delta, same quantizer), so the drift points in a consistent
  direction step after step. Once p has drifted, KL(q||p) grows, which increases |s_bar_q| in (2),
  which increases the drift: a positive feedback loop. Low-probability tokens gain mass, rollouts
  degrade, V(h) becomes more negative, and the loop accelerates until the reward collapses.

### Why IS weights and clipping only slow it down

With a detached token weight w_v = w(p_v/q_v), the same decomposition gives

    E_q[w_y A s_y | h] = Cov_q(A, w_y s_y | h) + V(h) * sum_v q(v|h) w_v s_v(h).

With the exact ratio w_v = p_v/q_v the second factor is sum_v p_v s_v = 0, so the drift disappears,
but at the price of unbounded-variance weights. Every practical method bounds w (TIS clamps,
MIS/PPO/DAPO/DPPO drop or clip): wherever the bound is active, q_v w_v != p_v and
sum_v q_v w_v s_v != 0, so a drift of the same form returns, proportional to how much of the
distribution is clipped. Tighter token-level clipping makes q_v w_v deviate from p_v on more
tokens, hence can make the drift *larger*, as observed. Sequence-level weights multiply A by a
scalar per sequence and leave sum_v q_v s_v untouched at every prefix, so they cannot cancel it
either.

## 2. The fix: center the score under q

Since s_bar_q(h) depends only on the prefix (and theta), not on the sampled token, subtracting
it is a control variate:

    g(h, y) = A * ( s_y(h) - s_bar_q(h) ),
    E_q[g | h] = sum_v q(v|h) Q(h,v) s_v - V(h) s_bar_q(h) = Cov_q(A, s_y | h).               (3)

The drift term of (1) is removed exactly, for any q, at every prefix. What remains:

    E[update] = sum_t E_{h_t ~ q} [ sum_v q(v|h_t) (Q(h_t,v) - V(h_t)) grad log p(v|h_t) ].

* It is zero at any prefix where the next token does not affect the return, whatever the
  mismatch: the parameters only move where the reward says something. There is no
  reward-independent force along grad KL(q||p), so no feedback loop.
* It is the on-policy policy gradient with token weights q instead of p. For a small mismatch
  q_v = p_v (1 + eps_v) it equals the true policy gradient plus an O(eps) term that is itself
  multiplied by the advantage Q(h,v) - V(h); it vanishes as the advantages vanish
  (e.g. when the policy has converged at h), unlike the drift, whose size is set by V(h) and the
  mismatch and does not vanish.
* On-policy (q = p) s_bar_q = 0 and the update is exactly REINFORCE.
* It also reduces variance: the subtracted term is the q-mean of the score, i.e. the optimal
  "per-prefix baseline in score space" for the part of the variance carried by E[A|h].

Implementation as a loss: with detached coefficients a_v(h) = sg(q(v|h)),

    L = - (1/N) sum_{b,t} A_b * ( log p(y_t|h_t) - sum_v a_v(h_t) log p(v|h_t) ),

whose gradient is exactly - (1/N) sum A (s_y - s_bar_q).

### With an IS correction (optional)

If `correction` is a token-level `ISCorrection` with weight w_v, the consistent centering is the
q-mean of the *weighted* score:

    g = A * ( w_y s_y - sum_v q_v w_v s_v ),    E_q[g|h] = Cov_q(A, w_y s_y | h),

which again removes the drift exactly, for any clipping band, and reduces to zero centering when
w = p/q unclipped (since then sum_v q_v w_v s_v = sum_v p_v s_v = 0). Sequence-level corrections
multiply A by a per-sequence weight and keep the (unweighted) centering. None of this is
needed; the default (no correction) is the recommended setting.

## 3. Computing s_bar_q from the top-K list

The sampler only returns its top K = 20 logprobs per position: ids k in S (|S| = K), q_k for
k in S, and Q = sum_{k in S} q_k. Treat the vocabulary outside S as one outcome "tail" with
sampler mass 1 - Q and trainer mass 1 - P, where P = sum_{k in S} p_k. Its score is

    grad log(1 - P) = (1/(1-P)) sum_{v not in S} p_v s_v = - (1/(1-P)) sum_{k in S} p_k s_k

(using sum_v p_v s_v = 0). The mean score of this coarsened distribution is

    s_bar_q ~= sum_{k in S} q_k s_k + (1 - Q) grad log(1 - P)
            = sum_{k in S} ( q_k - rho p_k ) s_k,      rho = (1 - Q)/(1 - P).                 (4)

(4) is exact when q's tail is proportional to p's tail (q_v = rho p_v off S), which is the
natural assumption given that nothing about the tail of q is observed; it is exact for any q when
K covers the vocabulary; and it is identically zero when q = p, so on-policy the loss is
unchanged. The coefficients a_k = q_k - rho p_k are detached and multiply log p_k gathered at the
K ids, so the cost is one [B, chunk, K] gather per chunk. With a token correction,
a_k = q_k w_k - (1-Q) w_tail p_k / (1-P), with w_tail the weight at the tail ratio (1-P)/(1-Q).

Numerical safeguards (they do not change the method): -inf/padded sampler logprobs are clipped
so q_k = 0; P and Q are clipped to [0, 1]; the tail coefficient (1-Q) w_tail/(1-P) is capped at
100 (it multiplies sum_k p_k s_k = -sum_{tail} p_v s_v, which is O(1-P); the cap is active only
when the trainer's mass outside the top-K is more than 100 times smaller than the sampler's,
where float32 cancellation in sum_k p_k s_k would otherwise be amplified). If the batch has no top-K keys the loss falls back to plain REINFORCE.

## 4. Checks performed

(The method is active with `correction=None`; the Configuration section below only restates the
default, no IS correction.)

* Gradients equal the original loss exactly when q = p (with and without corrections).
* With q's tail proportional to p's, sum_y q(y) grad g(y) = 0 to 1e-8 (no drift), with and without
  TIS / naive IS weights.
* `dev/mini.py` (noise 0.3, plus/minus rewards, seed 0): original loss plateaus at 0.58 train
  reward; centered loss reaches 0.84. With 40 or 200 extra unlikely tokens (vocabulary larger than
  K = 20, so the tail approximation is exercised) the original loss collapses to ~0 reward while
  the centered loss trains stably to ~0.82-0.85.

## Configuration

rl.is.level=null
