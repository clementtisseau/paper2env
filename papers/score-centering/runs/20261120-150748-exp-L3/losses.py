"""REINFORCE with importance-sampling corrections and a sampler-centered score.

Every correction is REINFORCE with a detached weight w(r) on the ratio
r = p/q of trainer (p) to sampler (q) probabilities, per token or per
sequence.

Always on: the per-token score grad log p(y|h) is centered under the measure
that actually draws y (the sampler q, times the IS weight if any),
    s_y  ->  s_y - sum_v m(v|h) grad log p(v|h),   m(v|h) = q(v|h) w(v|h).
On-policy (q = p, w = 1) the subtracted term is grad sum_v p(v|h) = 0, so
nothing changes. Under mismatch it removes the drift E[A|h] * s_bar(h), which
is a systematic, reward-independent bias of REINFORCE (see REPORT.md).
m is known on the sampler's top-k tokens; the tail outside the top-k is
taken proportional to p (exact when q is proportional to p there), which
keeps the on-policy case exactly zero.
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


TAIL_RATIO_MAX = 10.0  # cap on (sampler tail mass) / (trainer tail mass) outside the top-k
TAIL_MIN = 1e-6        # floor on the trainer tail mass (float32 cancellation in 1 - sum p)


def centering_coeffs(logp_k, logq_k, positive, c, valid):
    """Detached coefficients a_v on the sampler's top-k tokens such that
    sum_k a_v grad log p(v|h) estimates s_bar(h) = sum_v m(v|h) grad log p(v|h),
    with m = q * w(p/q) the measure that draws (and weights) the sampled token.

    Tail (v outside the top-k K): m is taken proportional to p there, with
    mass M_tail. As sum_v p(v) grad log p(v) = 0,
        sum_{v not in K} p(v) s_v = -sum_{v in K} p(v) s_v,
    so the tail adds -M_tail / P_tail * p(v) to each top-k coefficient.
    On-policy (m = p) every coefficient is exactly 0."""
    logp_k, logq_k = sg(logp_k), sg(logq_k)
    tv_k = jnp.abs(jnp.exp(logq_k) - jnp.exp(logp_k))
    m_k = jnp.where(valid, jnp.exp(log_mass(logp_k, logq_k, positive[..., None], c, tv_k)), 0.0)
    p_k = jnp.where(valid, jnp.exp(logp_k), 0.0)
    q_k = jnp.where(valid, jnp.exp(logq_k), 0.0)
    p_tail = jnp.clip(1.0 - p_k.sum(-1), 0.0, 1.0)
    q_tail = jnp.clip(1.0 - q_k.sum(-1), 0.0, 1.0)
    if c is None:
        m_tail = q_tail
    else:  # weight of the tail as a whole, at its aggregate ratio
        r_tail = jnp.log(jnp.maximum(p_tail, TAIL_MIN)) - jnp.log(jnp.maximum(q_tail, TAIL_MIN))
        tv_tail = jnp.abs(q_tail - p_tail)
        m_tail = q_tail * jnp.exp(log_mass(jnp.clip(r_tail, -20.0, 20.0), 0.0, positive, c, tv_tail))
    tail_ratio = jnp.where(p_tail > TAIL_MIN, jnp.minimum(m_tail / jnp.maximum(p_tail, TAIL_MIN), TAIL_RATIO_MAX), 0.0)
    return sg(m_k - tail_ratio[..., None] * p_k)


def take_vocab(logp, ids):
    """take_along_axis over a [B, C, V] vocab axis that may be TP-sharded (the
    gather drops that axis, so its output sharding must be explicit)."""
    spec = jax.typeof(logp).sharding.spec
    if spec[-1] is None:
        return jnp.take_along_axis(logp, ids, -1)
    b = jnp.arange(ids.shape[0])[:, None, None]
    c = jnp.arange(ids.shape[1])[None, :, None]
    return logp.at[b, c, ids].get(out_sharding=P(*spec[:-1], None))


def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):
    """REINFORCE loss on sampled rollouts, with an optional ISCorrection.
    Returns (loss, metrics).

    hidden [B, T, D] are pre-head activations and head maps them to logits.
    The head, log-softmax and all full-vocabulary float32 intermediates run in
    rematerialized chunks of `chunk` positions, so only [B, chunk, V] is live.

    batch: tokens, mask (completion tokens), advantages [B], and
    sampling_token_logprobs [B, T], the sampler's logprobs of the sampled tokens.
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
    center = "sampler_topk_ids" in batch and "sampler_topk_logprobs" in batch
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
        terms = {"entropy": -plogp, "kl_samp_train": sg(logq - logp), "masked": w == 0, "tv": tv}
        if center:  # subtract the sampler-measure mean score: removes the drift E[A|h] s_bar(h)
            ids = b["sampler_topk_ids"].astype(jnp.int32)
            logq_k = b["sampler_topk_logprobs"].astype(jnp.float32)
            valid = jnp.isfinite(logq_k) & (logq_k <= 0.0) & b["mask"][..., None]
            logq_k = jnp.where(valid, logq_k, -jnp.inf)
            logp_k = take_vocab(logp_vocab, ids)
            a_k = centering_coeffs(logp_k, logq_k, positive, token_c, valid)
            score = score - (a_k * jnp.where(valid, logp_k, 0.0)).sum(-1)
            terms["drift_mass"] = sg(jnp.abs(a_k).sum(-1))  # L1 size of the centering measure
        terms["score"] = score
        b_mask = b["mask"]
        return {k: (jnp.where(b_mask, v, 0) if v.dtype == jnp.bool_ else jnp.where(b_mask, v, 0.0)).sum(-1)
                for k, v in terms.items()}

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
    metrics = {k: sums[k].sum() / n_tokens for k in ("entropy", "kl_samp_train", "drift_mass") if k in sums}
    metrics["pol_entropy"] = metrics.pop("entropy")
    return loss, metrics | {"frac_masked": frac_masked}
