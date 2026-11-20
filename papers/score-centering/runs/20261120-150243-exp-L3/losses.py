"""REINFORCE with importance-sampling corrections and a per-prefix score
centering under the sampler distribution (see REPORT.md).

Every correction is REINFORCE with a detached weight w(r) on the ratio
r = p/q of trainer (p) to sampler (q) probabilities, per token or per
sequence.

Score centering (always on). At each completion position with prefix h the
sampled token y ~ q(.|h). The REINFORCE term A * w(y) * grad log p(y|h) has
conditional expectation E_q[A|h] * d(h) + Cov_q(A, w s_y | h), with the drift
direction d(h) = sum_v q(v|h) w(v) grad log p(v|h). On-policy d = 0; under a
mismatched sampler it is not, and it biases every update toward (A > 0) or
away from (A < 0) the sampler. The loss subtracts d(h) from every score, so
the expected update is the covariance term only. d(h) is evaluated from the
sampler's top-k logprobs plus a tail bucket in which q is taken proportional
to p, which makes d(h) = 0 exactly when q = p.
"""
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

sg = jax.lax.stop_gradient

# advantage of a (solved, unsolved) rollout; group_centered also subtracts the group mean
REWARD_MODES = {"plus_minus": (1.0, -1.0), "plus_zero": (1.0, 0.0),
                "zero_minus": (0.0, -1.0), "group_centered": (1.0, -1.0)}


def compute_advantages(rewards, group_size, reward_mode):
    solved, unsolved = REWARD_MODES[reward_mode]
    advantages = np.where(np.asarray(rewards) > 0, solved, unsolved).astype(np.float32)
    if reward_mode == "group_centered":
        groups = advantages.reshape(-1, group_size)
        advantages = (groups - groups.mean(1, keepdims=True)).reshape(-1)
    return advantages


DUAL_CLIP = 3.0  # PPO's ratio bound for negative advantages (arXiv:1912.09729)


class ISCorrection(NamedTuple):
    """One point on the IS-correction grid. TIS, MIS, PPO, DAPO, GSPO, TOPR and
    DPPO are all spellings of it (table in README.md)."""
    level: str = "token"   # token | seq_mean (mean log r) | seq_sum (sum log r: trajectory ratio)
    stat: str = "ratio"    # band tests the ratio r, or the TV distance |q - p| (DPPO; high = delta)
    low: float = 0.0       # band [low, high] on r; 0 / inf leave a side open
    high: float = math.inf
    outside: str = "drop"  # clamp: w = nearest bound | drop: w = 0
    sign: str = "both"     # which advantages get corrected: both | neg | pos (others w = 1)
                           # | ppo: block only moves away from q; A>0: (0, high], A<0: [low, DUAL_CLIP]

    def check(self):
        choices = dict(level=("token", "seq_mean", "seq_sum"), stat=("ratio", "tv"), outside=("clamp", "drop"),
                       sign=("both", "neg", "pos", "ppo"))
        for field, allowed in choices.items():
            assert getattr(self, field) in allowed, f"rl.is.{field} must be one of {allowed}"
        assert self.stat == "ratio" or self.level != "seq_sum", "seq_sum requires stat=ratio"
        return self


def log_mass(log_p, log_q, positive, c, tv=None):
    """log(q * w(p/q)) under correction c (-inf where w = 0, log q for no
    correction). Log space keeps the common in-band case q * (p/q) = p exact
    and overflow-free. is_weight evaluates it at p = r, q = 1."""
    if c is None:
        return log_q
    lo = log_q + math.log(c.low) if c.low > 0 else -math.inf
    hi = log_q + math.log(c.high) if c.high < math.inf else math.inf
    if c.stat == "tv":
        in_band = tv <= c.high
        if c.sign == "ppo":
            in_band |= jnp.where(positive, log_p <= log_q, log_p >= log_q)
    elif c.sign == "ppo":
        in_band = jnp.where(positive, log_p <= hi, (log_p >= lo) & (log_p <= log_q + math.log(DUAL_CLIP)))
    else:
        in_band = (log_p >= lo) & (log_p <= hi)
    log_m = jnp.where(in_band, log_p, jnp.clip(log_p, lo, hi) if c.outside == "clamp" else -jnp.inf)
    if c.sign == "neg":
        log_m = jnp.where(positive, log_q, log_m)
    if c.sign == "pos":
        log_m = jnp.where(positive, log_m, log_q)
    return log_m


def is_weight(log_ratio, positive, c, tv=None):
    """Detached IS weight w(r), with r = exp(log_ratio) clipped to e^±20."""
    return sg(jnp.exp(log_mass(jnp.clip(log_ratio, -20.0, 20.0), 0.0, positive, c, tv)))


def take_vocab(logp, ids):
    """take_along_axis over a [B, C, V] vocab axis that may be TP-sharded (the
    gather drops that axis, so its output sharding must be explicit)."""
    spec = jax.typeof(logp).sharding.spec
    if spec[-1] is None:
        return jnp.take_along_axis(logp, ids, -1)
    b = jnp.arange(ids.shape[0])[:, None, None]
    c = jnp.arange(ids.shape[1])[None, :, None]
    return logp.at[b, c, ids].get(out_sharding=P(*spec[:-1], None))


# Cap on the tail ratio m_tail / p_tail (sampler-side tail mass over trainer
# tail mass); only reached when the trainer puts almost no mass outside the
# sampler's top-k while the sampler does.
TAIL_RATIO_MAX = 1e3


def drift_coefficients(logp_vocab, topk_ids, topk_logq, positive, c):
    """Detached coefficients a_v over the sampler's top-k ids such that
    sum_v a_v * grad log p(v|h) = d(h) = sum_v q(v|h) w(v) grad log p(v|h),
    the drift direction of the (corrected) REINFORCE score at prefix h.

    The top-k ids carry their exact sampler mass q_v (times w). The rest of
    the vocabulary is one bucket T with sampler mass Q_T = 1 - sum_k q_v,
    inside which q is taken proportional to p. Its contribution is
    m_T * grad log P_T with P_T = 1 - P_K, i.e. -(m_T / P_T) * grad P_K
    = -(m_T / P_T) sum_k p_v grad log p_v, so it folds into the top-k
    coefficients. With q = p and no correction, a_v = p_v - (P_T/P_T) p_v = 0:
    d(h) = grad P_K + grad P_T = grad 1 = 0 exactly, so the on-policy loss is
    unchanged.
    """
    topk_logq = jnp.where(jnp.isfinite(topk_logq), topk_logq, -jnp.inf).astype(jnp.float32)
    q_k = jnp.exp(topk_logq)
    q_sum = q_k.sum(-1, keepdims=True)
    # sampler logprobs may be rounded (bf16) or garbage outside the completion
    scale = jnp.where(q_sum > 1.0, 1.0 / jnp.maximum(q_sum, 1e-30), 1.0)
    topk_logq = topk_logq + jnp.log(scale)
    q_tail = jnp.clip(1.0 - q_sum * scale, 0.0, 1.0)[..., 0]
    logp_k = take_vocab(logp_vocab, topk_ids)
    log_p_k_sum = jax.nn.logsumexp(logp_k, axis=-1)
    p_tail = jnp.clip(-jnp.expm1(jnp.minimum(log_p_k_sum, 0.0)), 0.0, 1.0)
    log_p_tail = jnp.log(jnp.maximum(p_tail, 1e-30))
    log_q_tail = jnp.log(jnp.maximum(q_tail, 1e-30))
    pos = positive[..., None]
    tv_k = jnp.abs(q_k * scale - jnp.exp(logp_k))
    m_k = jnp.exp(log_mass(logp_k, topk_logq, pos, c, tv_k))
    m_tail = jnp.exp(log_mass(log_p_tail, log_q_tail, positive, c, jnp.abs(q_tail - p_tail)))
    m_tail = jnp.where((q_tail > 0) & (p_tail > 0), m_tail, 0.0)
    ratio = jnp.minimum(m_tail / jnp.maximum(p_tail, 1e-30), TAIL_RATIO_MAX)
    coef = m_k - ratio[..., None] * jnp.exp(logp_k)
    return sg(coef), logp_k


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts, with an optional ISCorrection.
    Returns (loss, metrics).

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled tokens,
    and sampler_topk_ids / sampler_topk_logprobs [B, T, k], the sampler's top-k
    next-token logprobs, used to center each score by its drift d(h) under q.
    """
    c = correction
    token_c = c if c and c.level == "token" else None
    seq_c = c if c and c.level != "token" else None
    targets = jnp.roll(batch["tokens"], -1, axis=1)
    mask = jnp.roll(batch["mask"], -1, axis=1).at[:, -1].set(False)
    positive = batch["advantages"][:, None] >= 0
    seq_len = targets.shape[1]
    chunk = next(n for n in range(min(chunk, seq_len), 0, -1) if seq_len % n == 0)
    sampler_keys = ("sampling_token_logprobs", "sampler_topk_ids", "sampler_topk_logprobs")
    per_token = {"hidden": hidden, "targets": targets, "mask": mask} | {k: batch[k] for k in sampler_keys if k in batch}

    def chunk_sums(start):
        b = {k: jax.lax.dynamic_slice_in_dim(v, start, chunk, axis=1) for k, v in per_token.items()}
        logp_vocab = jax.nn.log_softmax(head(b["hidden"]).astype(jnp.float32))
        logp = take_vocab(logp_vocab, b["targets"][..., None])[..., 0]
        logq = b["sampling_token_logprobs"]
        tv = sg(jnp.abs(jnp.exp(logq) - jnp.exp(logp)))
        w = is_weight(logp - logq, positive, token_c, tv)
        plogp = sg((jnp.exp(logp_vocab) * logp_vocab).sum(-1))  # policy entropy metric
        score = w * logp
        drift_norm = jnp.zeros_like(logp)
        if "sampler_topk_ids" in b:
            # subtract the drift d(h): the expected update is then Cov_q(A, w s_y | h)
            coef, logp_k = drift_coefficients(logp_vocab, b["sampler_topk_ids"], b["sampler_topk_logprobs"],
                                              jnp.broadcast_to(positive, logp.shape), token_c)
            score = score - (coef * logp_k).sum(-1)
            drift_norm = sg(jnp.abs(coef).sum(-1))
        terms = {"score": score, "drift_l1": drift_norm, "entropy": -plogp, "kl_samp_train": sg(logq - logp), "masked": w == 0, "tv": tv}
        return {k: (v * b["mask"]).sum(-1) for k, v in terms.items()}

    chunks = jax.lax.map(jax.checkpoint(chunk_sums), jnp.arange(0, seq_len, chunk))
    sums = {k: v.sum(0) for k, v in chunks.items()}  # per-sequence sums [B]

    n_tokens = jnp.maximum(mask.sum(), 1)
    advantages = batch["advantages"]
    frac_masked = sums["masked"].sum() / n_tokens
    if seq_c:
        seq_tokens = jnp.maximum(mask.sum(1), 1)
        log_ratio = -sums["kl_samp_train"]
        if seq_c.level == "seq_mean":
            log_ratio = log_ratio / seq_tokens
        w = is_weight(log_ratio, advantages >= 0, seq_c, sums["tv"] / seq_tokens)
        advantages = advantages * w
        frac_masked = (w == 0).mean()  # fraction of sequences
    loss = -(advantages * sums["score"]).sum() / n_tokens
    metrics = {k: sums[k].sum() / n_tokens for k in ("entropy", "kl_samp_train", "drift_l1") if k in sums}
    metrics["pol_entropy"] = metrics.pop("entropy")
    return loss, metrics | {"frac_masked": frac_masked}
