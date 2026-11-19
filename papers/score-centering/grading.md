# How a run is graded

**A run succeeds when both hold:**

1. **The idea is found** (LLM judge). The judge reads the agent's report and
   its diff, against a hidden rubric. It must find:
   - the correction: each token's score minus its expected value under the
     sampler, with an argument that this removes the drift;
   - at least the drift term identified: under a mismatched sampler, the
     expected score is not zero, so the update gains a term that carries no
     reward signal. The full per-prefix decomposition is not required;
   - code that does what the report says.
2. **It is implemented correctly** (exact checks). The submitted loss passes
   the four checks below **using only the sampler's top-k logprobs**, which is
   all an inference engine returns.

The judge decides whether the idea was found; the checks only confirm that it
was implemented correctly. Neither replaces the other: a loss can pass the
checks on a toy problem for other reasons, and a right idea can be
implemented wrongly.

## The exact checks

On a toy policy (vocabulary 5, length 3) whose completions are all
enumerated, so expectations are exact. Each check rejects a plausible wrong
answer:

| Check | Requires | Rejects |
|---|---|---|
| `no_drift` | with a constant advantage, the expected update is zero, under a mild and a severe mismatch | REINFORCE; TIS and MIS (reduced, not removed); importance sampling capped at 10 |
| `no_drift_prefix` | zero expected update at every prefix, not only on average | a global reward baseline (GRPO pushed to its limit) |
| `on_policy` | when the sampler is exact, the update equals REINFORCE's | a loss that learns nothing |
| `bounded` | no per-sample gradient above 10× the worst REINFORCE one | unclipped importance sampling (×85,000) |

## Reported, not required

- **Correct with the full distribution only:** the idea is there, but the
  loss needs the sampler's whole distribution, which an engine does not give.
- **Composes with TIS:** still drift-free when combined with importance
  weights (the centering must then be under q·w).
- **CPU guard-rail:** a short training run, normalized between REINFORCE (0)
  and the paper's method (1). It confirms that training does not break, but
  cannot separate the method from importance sampling at this scale.

All of these are per run in [results_tables.md](results_tables.md).
Implementation of the checks and the judge: [grader/README.md](grader/README.md).
