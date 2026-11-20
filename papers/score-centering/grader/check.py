"""Exact check of a submitted losses.py: is its expected update free of the
drift that a mismatched sampler adds, without breaking anything else?

A toy language model: vocabulary V = 5, completions of L = 3 tokens, so all
5^3 = 125 completions can be enumerated and every expectation is exact. The
trained model p is a table of logits indexed by prefix; the sampler q is the
same table plus noise, and the loss sees q only through its top-K logprobs,
as from an inference engine. The submission is called exactly as train_rl.py
calls it, with its default arguments: loss_fn_rl(hidden, head, batch).

The four checks, over three seeds (all must pass for "correct"):
  no_drift         constant advantage (+1 and -1), mild and severe noise:
                   the expected update is zero. Rejects REINFORCE, truncated
                   or masked importance sampling, and IS capped at a loose
                   bound (exact while no ratio exceeds the cap).
  no_drift_prefix  an advantage with zero mean that depends on the first d
                   tokens: the expected update of every later position is
                   zero. Rejects a global reward baseline (GRPO pushed to its
                   limit), which cancels the drift on average only.
  on_policy        q = p: the expected update equals REINFORCE's. Rejects
                   losses that are drift-free by learning nothing.
  bounded          severe noise: no single completion's gradient exceeds
                   10x REINFORCE's largest. Rejects unclipped IS.

    python check.py path/to/losses.py   # JSON on stdout, exit 0 if correct
"""
import importlib.util
import itertools
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

V, L, K = 5, 3, 3          # vocabulary, completion length, top-k size
SEEDS = (0, 1, 2)
MILD, SEVERE = 0.5, 3.0    # sampler logit noise scales
TOL = 1e-4                 # |E grad| relative to E|per-sample grad|
BOUND_FACTOR = 10.0


# ----------------------------------------------------------------------------
# Toy policy: the next-token logits are a table indexed by the prefix.
# hidden[t] is the one-hot of the prefix id at position t, head = hidden @ theta.
# ----------------------------------------------------------------------------

def prefixes():
    out = [()]
    for n in range(1, L):
        out += list(itertools.product(range(V), repeat=n))
    return {p: i for i, p in enumerate(out)}


PREFIX_ID = prefixes()
D = len(PREFIX_ID)
PREFIX_LEN = np.array([len(p) for p in PREFIX_ID])  # row of theta -> prefix length
COMPLETIONS = np.array(list(itertools.product(range(V), repeat=L)), np.int32)  # [N, L]
N = len(COMPLETIONS)
BOS = 0


def hidden_for(completion):
    """[T, D] with T = L + 1: position t (0 = prompt token) predicts token t+1."""
    h = np.zeros((L + 1, D), np.float32)
    for t in range(L):
        h[t, PREFIX_ID[tuple(completion[:t])]] = 1.0
    return h


HIDDEN = np.stack([hidden_for(c) for c in COMPLETIONS])  # [N, T, D]
TOKENS = np.concatenate([np.full((N, 1), BOS, np.int32), COMPLETIONS], 1)  # [N, T]
MASK = np.concatenate([np.zeros((N, 1), bool), np.ones((N, L), bool)], 1)


def table_logprobs(theta):
    """[N, T, V] next-token logprobs at every position of every completion."""
    return jax.nn.log_softmax(jnp.asarray(HIDDEN) @ theta)


def sequence_logprob(logp_vocab):
    """log probability of each completion, [N]."""
    tok = jnp.take_along_axis(logp_vocab[:, :L], jnp.asarray(COMPLETIONS)[..., None], -1)[..., 0]
    return tok.sum(-1)


def sampler_table(theta, scale, key):
    """Sampler logits per prefix: theta plus noise, with q outside its top-k
    made proportional to p (so a top-k estimate of an expectation under q can
    be exact)."""
    noise = scale * jax.random.normal(key, theta.shape)
    q_logits = theta + noise
    if scale > 0:
        logp = jax.nn.log_softmax(theta)
        logq = jax.nn.log_softmax(q_logits)
        order = jnp.argsort(-logq, -1)
        head = jnp.zeros_like(logq, bool).at[jnp.arange(D)[:, None], order[:, :K]].set(True)
        q, p = jnp.exp(logq), jnp.exp(logp)
        # head keeps the noisy sampler's shape; tail is proportional to p with
        # mass m, capped so that no tail entry outranks the smallest head entry
        q_head = jnp.where(head, q, 0)
        q_head = q_head / q_head.sum(-1, keepdims=True)
        p_tail = jnp.where(head, 0, p)
        p_tail = p_tail / p_tail.sum(-1, keepdims=True)
        a = jnp.where(head, q_head, jnp.inf).min(-1, keepdims=True)
        b = p_tail.max(-1, keepdims=True)
        m = jnp.minimum(jnp.where(head, 0, q).sum(-1, keepdims=True), 0.9 * a / (a + b))
        q = jnp.where(head, (1 - m) * q_head, m * p_tail)
        # keep the head the head: tail entries must not exceed the smallest head entry
        min_head = jnp.where(head, q, jnp.inf).min(-1, keepdims=True)
        assert bool((jnp.where(head, -jnp.inf, q) <= min_head + 1e-7).all()), "re-draw the sampler"
        q_logits = jnp.log(q)
    return q_logits


def make_batch(theta_q, advantages):
    """The batch train_rl.py builds: tokens, mask, advantages, the sampled
    tokens' logprobs under q, and q's top-K (ids and logprobs)."""
    logq_vocab = table_logprobs(theta_q)  # [N, T, V]
    targets = jnp.roll(jnp.asarray(TOKENS), -1, 1)
    tok = jnp.take_along_axis(logq_vocab, targets[..., None], -1)[..., 0]
    tok = jnp.where(jnp.roll(jnp.asarray(MASK), -1, 1).at[:, -1].set(False), tok, 0.0)
    top_lp, top_ids = jax.lax.top_k(logq_vocab, K)
    return {"tokens": jnp.asarray(TOKENS), "mask": jnp.asarray(MASK),
            "advantages": jnp.asarray(advantages, jnp.float32), "sampling_token_logprobs": tok,
            "sampler_topk_ids": top_ids, "sampler_topk_logprobs": top_lp}


def per_sample_grads(loss_fn, theta, theta_q, advantages):
    """[N, D, V] gradient of the loss on each completion alone (batch of 1)."""
    batch = make_batch(theta_q, advantages)

    def one(i):
        b = {k: v[i][None] for k, v in batch.items()}
        h = jnp.asarray(HIDDEN)[i][None]
        f = lambda th: loss_fn(h, lambda x: x @ th, b)[0]
        return jax.grad(f)(theta)

    try:
        return jax.vmap(one)(jnp.arange(N))
    except Exception:  # loss not vmappable: fall back to a loop
        return jnp.stack([one(i) for i in range(N)])


def q_weights(theta_q):
    """Probability of each completion under the sampler, [N]."""
    return jnp.exp(sequence_logprob(table_logprobs(theta_q)))


def expected(grads, weights):
    return jnp.tensordot(weights, grads, 1)


def reinforce(hidden, head, batch, *_, **__):
    """Plain REINFORCE, the reference point for on_policy and bounded."""
    logp = jax.nn.log_softmax(head(hidden).astype(jnp.float32))
    targets = jnp.roll(batch["tokens"], -1, 1)
    mask = jnp.roll(batch["mask"], -1, 1).at[:, -1].set(False)
    tok = (jnp.take_along_axis(logp, targets[..., None], -1)[..., 0] * mask).sum(-1)
    return -(batch["advantages"] * tok).sum() / jnp.maximum(mask.sum(), 1), {}


def drift(loss_fn, theta, theta_q, advantages, rows=None):
    """|expected update| relative to the expected per-sample gradient norm.
    rows: restrict to these rows of theta (prefixes of some length)."""
    w = q_weights(theta_q)
    g = per_sample_grads(loss_fn, theta, theta_q, advantages)
    if rows is not None:
        g = jnp.where(rows[None, :, None], g, 0.0)
    scale = (w * jnp.sqrt((g ** 2).sum((1, 2)))).sum()
    return float(jnp.linalg.norm(expected(g, w)) / jnp.maximum(scale, 1e-12))


def evaluate(loss_fn):
    """Run the four checks. loss_fn(hidden, head, batch) -> (loss, metrics)."""
    worst = {"no_drift": 0.0, "no_drift_prefix": 0.0, "on_policy": 0.0, "bounded": 0.0}
    for seed in SEEDS:
        k_theta, k_mild, k_severe, k_r = jax.random.split(jax.random.key(seed), 4)
        theta = jax.random.normal(k_theta, (D, V))
        mild = sampler_table(theta, MILD, k_mild)
        severe = sampler_table(theta, SEVERE, k_severe)
        r = jax.random.bernoulli(k_r, 0.4, (N,)).astype(jnp.float32)

        # no_drift: a constant advantage carries no signal, so any update is drift
        for theta_q in (mild, severe):
            for a in (1.0, -1.0):
                worst["no_drift"] = max(worst["no_drift"], drift(loss_fn, theta, theta_q, jnp.full(N, a)))

        # no_drift_prefix: advantage g(y_1..y_d), centered under q; after position d
        # it is a constant of the prefix, so the rows of theta used there must not move
        w = q_weights(mild)
        for depth in (1, 2):
            table = jax.random.normal(jax.random.fold_in(k_r, depth), (V,) * depth)
            adv = table[tuple(COMPLETIONS[:, i] for i in range(depth))]
            adv = adv - (w * adv).sum() / w.sum()
            rows = jnp.asarray(PREFIX_LEN >= depth)
            worst["no_drift_prefix"] = max(worst["no_drift_prefix"], drift(loss_fn, theta, mild, adv, rows))

        # on_policy: q = p, real rewards: same expected update as REINFORCE
        w_on = q_weights(theta)
        g_sub = expected(per_sample_grads(loss_fn, theta, theta, r), w_on)
        g_ref = expected(per_sample_grads(reinforce, theta, theta, r), w_on)
        worst["on_policy"] = max(worst["on_policy"],
                                 float(jnp.linalg.norm(g_sub - g_ref) / (jnp.linalg.norm(g_ref) + 1e-12)))

        # bounded: severe mismatch, largest single-completion gradient vs REINFORCE's
        g_sub = per_sample_grads(loss_fn, theta, severe, r * 2 - 1)
        g_ref = per_sample_grads(reinforce, theta, severe, r * 2 - 1)
        worst["bounded"] = max(worst["bounded"], float(jnp.sqrt((g_sub ** 2).sum((1, 2))).max()
                                                       / jnp.sqrt((g_ref ** 2).sum((1, 2))).max()))

    checks = {"no_drift": worst["no_drift"] < TOL, "no_drift_prefix": worst["no_drift_prefix"] < TOL,
              "on_policy": worst["on_policy"] < TOL, "bounded": worst["bounded"] < BOUND_FACTOR}
    finite = all(np.isfinite(v) for v in worst.values())
    return {"correct": finite and all(checks.values()), "checks": checks, "worst": worst}


def load_loss(path):
    """The submission's loss_fn_rl, called with its default arguments."""
    spec = importlib.util.spec_from_file_location("submission_losses", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda hidden, head, batch: module.loss_fn_rl(hidden, head, batch)


def check(path):
    """evaluate() on a losses.py file; a loss that fails to load or crashes is not correct."""
    try:
        return evaluate(load_loss(path))
    except Exception as e:
        return {"correct": False, "error": f"{type(e).__name__}: {e}"[:500]}


if __name__ == "__main__":
    result = check(sys.argv[1])
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["correct"] else 1)
