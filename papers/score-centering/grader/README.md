# Grader

Runs in a container the agent never sees. It puts the agent's `losses.py` in
a clean tree and computes three independent signals.

## `theory/check.py`: exact theory check (CPU, under a minute)

A toy autoregressive policy (vocabulary 5, length 3, logits indexed by
prefix) where every completion is enumerated, so expectations are exact. The
submitted loss is called exactly as `train_rl.py` calls it. Four checks:

- `no_drift`: with a constant advantage, the expected update under the
  sampler q is zero, under a mild and under a severe mismatch. Plain
  REINFORCE fails: E_q[∇log p] ≠ 0 when q ≠ p. Importance weights capped at a
  loose bound fail under the severe one, where ratios exceed the cap.
- `no_drift_prefix`: with an advantage of zero mean under q that depends on
  the first d tokens, the update at every position after d is zero. This
  rejects an exact global reward baseline (GRPO pushed to its limit), which
  cancels the average drift but not the drift at each prefix.
- `on_policy`: when q = p, the expected update equals REINFORCE's. Rejects
  losses that are drift-free by learning nothing.
- `bounded`: under a severe mismatch, no per-sample gradient exceeds 10x the
  worst REINFORCE one. Rejects unclipped importance sampling.

Each check runs with the sampler's full distribution and with its top-k only
(q built so that its tail is proportional to p, so a sound top-k estimate can
be exact). `passed` needs one scenario; `passed_topk` reports the top-k one,
which is what an inference engine provides. A third run calls the loss with
a TIS correction (`passed_with_tis`): a loss that centers the score under q
while the sampled token carries a weight w(p/q) fails there, because with
weights the centering must be under q·w. `alignment` (cosine with the true
on-policy gradient) is reported without a threshold.

Validated in `tests/test_theory.py`:

| Loss | `no_drift` | `no_drift_prefix` | other | verdict |
|---|---|---|---|---|
| Paper's method (alone, + TIS, + MIS) | 0 | 0 | | pass |
| Exact global baseline (GRPO limit) | 0 | 0.16 | | fail |
| REINFORCE | 0.21 | 0.16 | | fail |
| TIS | 0.027 | 0.024 | | fail |
| MIS | 0.13 | 0.017 | | fail |
| Unclipped IS | 0 | 0 | `bounded` ×85,000 | fail |
| IS capped at 10 | 0.21 (severe) | 0 | | fail |
| Zero loss | 0 | 0 | `on_policy` 1.0 | fail |

Two agent submissions are kept as regression fixtures
(`tests/fixtures/`): one that fails only with TIS, one that fails only top-k.

```bash
python papers/score-centering/grader/theory/check.py path/to/losses.py   # JSON, exit 0 if passed
```

## `empirical/mini_check.py`: CPU guard-rail (about 2 minutes)

Trains the `mini/` prototype (sorting digits, relative noise 0.3 on the
sampler's weights, ±1 advantages, 600 steps) with the submitted loss on three
held-out seeds, and normalizes the final reward between REINFORCE (0) and the
paper's method (1), cached in `baselines.json` (`--calibrate` recomputes
them). Passes at 0.5 or more. It applies the agent's `## Configuration`
(`rl.is.*`) and gives the loss the sampler's top-20, as the starter does.

This signal separates "removes the drift" from "does nothing"; it does
**not** separate the paper's method from TIS or MIS (see
[`../mini/RESULTS.md`](../mini/RESULTS.md)). On the toy's 12-token vocabulary
the top-20 is the whole distribution, so the guard-rail cannot punish a
naive top-k truncation either: that is the theory check's job. Validation,
mean of three seeds: uncorrected starter 0.67 → 0.00 (fail); starter + TIS
0.79 → 0.79 (pass); paper's method 0.83 → 1.00 (pass); zero loss 0.52 →
−0.89 (fail). `RUN_SLOW=1 python -m pytest papers/score-centering/tests/test_guard.py`
replays it.

## `judge/`: LLM judge

- `reference.md`: the reference explanation of the contribution.
- `rubric.md`: hidden rubric, six weighted criteria scored 0/1/2 (the
  per-prefix decomposition and the score-level correction weigh 3 each), seven
  near-misses (N1: sequence-level decomposition leading to a GRPO-style
  baseline, N2: expected score under p instead of q, …), an "alternative valid
  contribution" route to human review, and anti-manipulation rules.
- `prompt.md`, `schema.json`: the judge prompt and its JSON output.
- `config.json`: the criterion weights, near-miss ids, the minimum scores
  that count as `rediscovered` (C2 ≥ 1, C3 = 2, C6 ≥ 1), the criteria that
  must not come from the prompt for `discovered` (C2, C3), and the criterion a
  near-miss sets to 0 (C3).

The plumbing is shared ([`core/judge.py`](../../../core/judge.py)): it fills
the prompt in one pass over the template (agent text is never re-templated),
validates verdicts, scores them with `config.json` and aggregates several
judges.

The judge does not see the theory check's result, so the signals stay
independent. It has not been calibrated yet, and the runs in `runs/` were not
judged.
