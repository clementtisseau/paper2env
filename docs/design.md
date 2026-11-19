# Design

## The question

Can a model, given a real research codebase and only what was known before a
paper, find the paper's contribution? And how much of a hint does it need?

The paper here is *Score Centering Stabilizes Off-policy Reinforcement
Learning* (Marek & Ryabinin, arXiv:2609.20807, September 2026). When RL for
LLMs samples rollouts with an inference engine that differs slightly from the
trained model (quantization, kernels, stale weights), training drifts and
often collapses. The paper shows why and fixes it with one line of math:

- At a prefix h, the expected REINFORCE update under the sampler q splits
  into **E_q[A | h] · s̄(h)**, a drift along the sampler's expected score
  s̄(h) = Σ_v q(v|h) ∇log p(v|h), plus **Cov_q(A, s_y | h)**, the signal.
  On-policy s̄ = 0; off-policy it is not, and the drift pulls the trainer
  toward the sampler, which is rebuilt from the trainer every step, so it
  compounds.
- **Fix:** replace each token's score s_y by s_y − s̄(h), computed with a
  scalar surrogate and stop-gradients, and with a model of the tail when the
  engine returns only its top-k logprobs.

It is a good target for a rediscovery environment: the problem was widely
known and studied before the paper (importance-sampling corrections, clipping
variants, the "systematic bias, not noise" diagnosis), the contribution is a
small change to the loss with a crisp correctness criterion, and the paper is
recent enough that the models tested cannot have trained on it.

## What the agent gets

A Docker container (no network, 4 CPUs, 8 GB) holding a **starter**: the
paper's own repository with the method surgically removed
([`rlenv/sanitize.py`](../rlenv/sanitize.py)). It is a real JAX RL framework:
`train_rl.py` trains Qwen3-0.6B on Countdown with a deliberately mismatched
sampler, with every importance-sampling variant already implemented
(TIS, MIS, PPO, DAPO, GSPO, TOPR, DPPO). The agent must leave a modified
`losses.py` and a `REPORT.md` with its derivation.

In the `exp` mode it also gets `run_dev.sh`, which runs a CPU version of the
same setup (a small transformer sorting digits under the same kind of
mismatch, [`rlenv/mini/`](../rlenv/mini/)) in a few minutes, so it can see the
problem and test ideas. In the `noexp` mode it can only reason and call its
loss on inputs it builds.

## Hiding the answer

The hard part of a rediscovery environment is that the answer is everywhere:
in the code, the docs, the names, the history, and in the agent's own
context. What was done about each:

| Leak | Where | Fix |
|---|---|---|
| The method's code | `losses.py` branches, a `score_center` flag | Removed by exact, SHA-pinned replacements; the starter's loss equals the reference with the method off (tested on values, gradients and metrics for seven IS corrections). |
| Words | docstrings, config comments, README, names like `center_ids` | Rewritten; renamed to `sampler_topk_*`; neutral README; a grep audit fails the build on any revealing word. |
| Structure | a module (`models/topk.py`) that exists only for the method | Folded into the sampler as a private helper. |
| Data plumbing | a setting that enables the sampler's distribution | Removed: the sampler always returns its top-20 logprobs, as `logprobs=20` does in an inference engine, and nothing documents them for the loss. Saying what the batch holds is a hint of level L3. |
| History | git log, the paper's figure, sweeps, tests | Fresh git history; README, `assets/`, `sweeps/`, `tests/` dropped. |
| Prompt | "theoretical argument", "the IS corrections fail", the name of the useful config key | Moved into the hint levels or removed; tests check that no prompt says more than its level allows. |
| The agent's own context | the session's working directory and scratchpad path contained the repository's name | Runs are launched from a lab checkout with a neutral name; a probe subagent quotes its context before runs; all 45 transcripts were scanned afterwards. |
| The environment's own files | `.claude/` and the lab's `.gitignore` copied into the starter | Excluded (the audit caught the first). |

## Hint levels

| Level | The prompt adds |
|---|---|
| L0 | Only the problem, the deliverables and the tools. |
| L1 | What was known before the paper: a systematic bias, not noise; IS and clipping slow it down but do not remove it. |
| L2a | Analyze the expected update under the sampler, on- vs off-policy. |
| L2b | At one token position, conditionally on the prefix. |
| L3 | The decomposition drift + covariance, and that the batch holds the sampler's top-20. |
| L4 | The method itself (tests implementation only, like PaperBench). |

Levels are cumulative. The point of the ladder is a curve, not a pass rate:
at which level does each model start finding the contribution, and what does
it find before that?

## Grading

Three signals, independent by design ([`rlenv/grader/`](../rlenv/grader/)):

1. **Theory check (the main signal).** The contribution is a property: the
   loss's expected update has no drift at any prefix, is unchanged on-policy,
   and keeps per-sample gradients bounded. It is checked exactly, by
   enumerating every completion of a toy policy, so it rewards any correct
   fix (not only the paper's wording) and cannot be argued with. It runs with
   the full distribution, with the top-k only, and combined with TIS. Each
   check was introduced against a specific wrong answer: `no_drift_prefix`
   rejects an exact GRPO-style global baseline, `bounded` rejects unclipped
   importance sampling, the severe-mismatch drift check rejects loosely capped
   importance sampling, `on_policy` rejects losses that learn nothing.
2. **CPU guard-rail.** A real training run on held-out seeds, normalized
   between REINFORCE and the paper's method. It confirms that a loss does not
   break training. It cannot separate the method from TIS on a CPU-sized
   problem (see [lessons](lessons.md)).
3. **LLM judge** (built, not yet calibrated or run). A hidden rubric that
   weighs the per-prefix decomposition and the score-level correction most,
   lists seven near-misses that look like the method but are not, and
   separates what the agent discovered from what its prompt gave.

## Isolation and runners

The agent's code runs only in containers without network: its sandbox, and a
separate grader container with the submission mounted read-only. Two runners
share the starter, prompts and graders:

- an **Inspect AI task** for API models (limits on working time, tokens and
  messages; compaction for long runs);
- a **Claude Code subagent runner**, used for the runs here. A `PreToolUse`
  hook rewrites every subagent Bash call into `docker exec` in the run's
  container and denies every other tool, so the subagent never touches the
  host. It was tested with a subagent that tried to escape.
