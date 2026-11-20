# Removing the sampler-induced drift from REINFORCE

## 1. What goes wrong with the current loss

The loss is plain REINFORCE on rollouts drawn from the sampler q, optionally
with a detached importance weight:

    L = -(1/N) sum_{i,t} A_i * w_{i,t} * log p(y_t | h_t),        grad L = -(1/N) sum A w s_{y_t}

with p the trained model, h_t the prefix, A the (group-centered) advantage of the
whole completion and s_v(h) = grad_theta log p(v|h) the score of token v.

Fix one prefix h and take the expectation over the next token y ~ q(.|h)
(w = 1, no correction). Add and subtract the mean score:

    E_q[A s_y | h] = E_q[A | h] * s_bar(h)  +  Cov_q(A, s_y | h),
    s_bar(h)       = sum_v q(v|h) s_v(h)    = sum_v q(v|h) grad log p(v|h).

* **On-policy (q = p)**: s_bar(h) = sum_v p_v grad p_v / p_v = grad sum_v p_v = grad 1 = 0.
  Only the covariance survives: tokens are pushed up exactly to the extent
  that choosing them correlates with a higher return. This is the policy
  gradient.
* **Mismatched (q != p)**: s_bar(h) != 0. Written in logit space z (for the
  softmax head, s_v = e_v - p), s_bar = q - p: the first term moves the logits
  of every visited prefix by E_q[A|h] * (q(.|h) - p(.|h)).

The first term is a *drift*: its direction is fixed by the sampler/trainer
discrepancy, not by which tokens were good, and its size is the expected
advantage at the prefix, which is generically nonzero at almost every prefix
(group centering only makes the advantage zero-mean per prompt, not per
prefix: prefixes that lead to success have E[A|h] > 0 and those that lead to
failure E[A|h] < 0; with uncentered rewards it is nonzero everywhere). For
E[A|h] > 0 it pulls p toward q, for E[A|h] < 0 it pushes p *away* from q,
i.e. it amplifies the trainer/sampler difference along precisely those
directions in which the sampler deviates (e.g. toward the junk tokens that a
noisy or quantized sampler over-weights in its tail).

Why it compounds instead of averaging out: the sampler is rebuilt from the
trainer every step (q_{k+1} = Q(p_{k+1}) for a fixed corruption Q: noise,
int8/fp8, KV quantization, staleness). The discrepancy q - p is therefore a
deterministic function of the current parameters, so the drift is a
systematic vector field b(theta) = E_h[E_q[A|h] (s_bar(h))] that is added at
every step, with the same sign, and is not reduced by more samples. Summed
over training it moves theta along a direction that has nothing to do with
reward; as p moves, the corruption Q(p) moves with it and the drift is
re-applied, which is the observed slow degradation followed by collapse.

Why IS corrections and clipping do not fix it: with a token weight
w_v = w(p_v/q_v),

    E_q[A w_y s_y | h] = E_q[A|h] * sum_v q_v w_v s_v  +  Cov_q(A, w_y s_y | h).

Exact IS (w = p/q) makes sum_v q_v w_v s_v = sum_v p_v s_v = 0, but then
E_q[A|h] is still the sampler's conditional value, the weights at the
sequence level have huge variance, and any truncation/clipping (TIS, MIS,
PPO-style bands) breaks the identity: sum_v q_v w_v s_v = sum_v (q_v w_v - p_v) s_v
!= 0 again. Clipping only shrinks the drift coefficient; it does not cancel
it. Stronger token clipping can even make it larger, because it removes
weight from exactly the tokens where p and q disagree most, so
q_v w_v - p_v gets *further* from zero on those tokens.

## 2. The modification

Replace each token's score by its deviation from the sampler-expected score,
keeping the same advantage:

    g_t = A * ( s_{y_t}(h_t) - s_bar(h_t) ),       s_bar(h) = sum_v q(v|h) s_v(h).

Its conditional expectation is

    E_q[A (s_y - s_bar) | h] = E_q[A s_y | h] - E_q[A|h] s_bar(h) = Cov_q(A, s_y | h),

so the drift term is removed exactly at every prefix and only the
reward-correlated signal remains. Properties:

* **On-policy it is a no-op**: s_bar = sum_v p_v s_v = 0 identically (not just in
  expectation), so the gradient equals the original REINFORCE gradient.
* It is a *per-prefix control variate in the score*, not in the reward: s_bar(h)
  is a function of the prefix only, so it is a valid baseline for the token
  y ~ q(.|h). It uses no importance ratios, hence no extra variance and no
  clipping.
* What remains, Cov_q(A, s_y|h), still points uphill for the trainer: in logit
  space it is sum_v q_v (E_q[A|h,y=v] - E_q[A|h]) e_v, i.e. it raises the logit
  of every token whose continuation value is above the prefix average and
  lowers the others, weighted by how often the sampler visits them. A
  mismatched q now only reweights correct-sign pushes; it no longer adds a
  push of its own.
* In logit space the original update at a prefix is A (e_y - p); its mean is
  E[A|h](q - p) + Cov. The new update is A (e_y - q), with mean Cov only:
  the model is moved relative to what the sampler actually did, not relative
  to its own (different) distribution.

### Surrogate (one backward pass)

With sg = stop-gradient,

    loss_t = -A * ( log p(y_t|h_t) - sum_v sg[q(v|h_t)] log p(v|h_t) ),

whose gradient is -A (s_y - s_bar). The sum is over the vocabulary at every
position, computed in the same rematerialized chunks as the log-softmax.

### Only the top-k head of q is known

The sampler returns its top-20 logprobs (`sampler_topk_ids`,
`sampler_topk_logprobs`). Let H be that head, q_H = sum_{v in H} q_v,
p_H = sum_{v in H} p_v, q_T = 1 - q_H, p_T = 1 - p_H. Model the tail of q as
proportional to p:

    q_v ~= rho p_v  for v not in H,    rho = q_T / p_T.

(This is exact when the sampler's tail is a rescaled copy of the trainer's
tail; otherwise the error is bounded by the tail mass q_T, which top-20 makes
small.) Then, using
sum_v p_v s_v = 0, i.e. sum_{v not in H} p_v s_v = -sum_{v in H} p_v s_v,

    s_bar = sum_{v in H} q_v s_v + rho sum_{v not in H} p_v s_v
          = sum_{v in H} (q_v - rho p_v) s_v,

so the tail needs no gradient computation at all and

    loss_t = -A * ( log p(y_t|h_t) - sum_{v in H} sg[q_v - rho p_v] log p(v|h_t) ).

The sampled token keeps its own log p(y_t) even when y_t is outside H.
In logit space this baseline equals q_v on H and rho p_v on the tail; it sums
to 1, so the corrected update A(e_y - q_model) has exactly zero total logit
mass, as on-policy.
If the vocabulary has at most k tokens, H is everything, q_T = 0 and the
formula is exact. If p = q, the coefficients are p_v - p_v = 0 and the
loss is unchanged.

### With an IS correction (optional)

The default (`correction=None`) is the plain version above. If an
`rl.is.*` correction is passed, the same argument applies to the weighted
score w_y s_y: its drift term is E_q[A|h] sum_v q_v w_v s_v, so the baseline
coefficient m_v = q_v w(p_v/q_v) is used instead of q_v (this is exactly
`exp(log_mass(log p_v, log q_v))` in `losses.py`; for the tail, the aggregate
(q_T, p_T) is passed through the same weight, giving rho = m_T / p_T). This
keeps the expected update equal to Cov_q(A, w_y s_y | h) for any band or
clamp, so clipping no longer re-introduces a drift. Sequence-level
corrections multiply the advantage and use the plain q baseline per token.

## 3. Implementation (`losses.py`)

* `sampler_keys` now also forwards `sampler_topk_ids` and
  `sampler_topk_logprobs`; when they are present (always, in `train_rl.py`
  and `dev/mini.py`) the correction is active with default arguments.
* `sampler_score_baseline` gathers the trainer's log p on the sampler's head
  (`take_vocab`, TP-safe), builds the detached coefficients
  c_v = m_v - rho p_v (m_v = q_v without correction), and returns
  sum_{v in H} c_v log p_v, which is subtracted from each token's score
  before multiplying by the advantage. All mass quantities are detached.
* Numerical guards: rows outside the completion mask (where the sampler's
  head buffer holds stale values) get zero coefficients; q_T is clipped to
  [0, 1]; p_T is floored at 1e-6 so rho stays finite when the trainer puts
  essentially all mass on the head (the tail term is then O(q_T) anyway).
* New metrics: `q_head_mass` (mean sampler head mass) and `drift_mass`
  (mean L1 size of the baseline coefficients, i.e. how far q is from p on
  the head).
* Verified numerically: the gradient matches a full-vocabulary reference
  with the rho-tail model to 6e-8, and is identical to the original loss
  when q = p.

(For reference, on the CPU dev setting with uncentered rewards the original
loss plateaus/declines around 0.66-0.68 eval reward across 3 seeds while the
modified loss keeps improving to 0.81-0.86; with 40 extra junk tokens in the
vocabulary the original collapses to 0 reward (seed 0) while the modified loss
reaches ~0.81-0.86.)

## Configuration

No configuration changes are needed: the modification is active with the
default arguments and works with `correction=None`.
