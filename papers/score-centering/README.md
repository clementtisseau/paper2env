# score-centering

*Score Centering Stabilizes Off-policy Reinforcement Learning*, Martin Marek
and Max Ryabinin, [arXiv:2609.20807](https://arxiv.org/abs/2609.20807),
September 2026. Code: [`upstream/`](upstream/), untouched (MIT).

## The paper

When RL for LLMs samples rollouts with an inference engine that differs
slightly from the trained model (quantization, kernels, stale weights),
training drifts and often collapses. The paper shows why and fixes it in the
loss:

- At a prefix h, the expected REINFORCE update under the sampler q splits
  into **E_q[A | h] · s̄(h)**, a drift along the sampler's expected score
  s̄(h) = Σ_v q(v|h) ∇log p(v|h), plus **Cov_q(A, s_y | h)**, the signal.
  On-policy s̄ = 0; off-policy it is not, and the drift compounds because the
  sampler is rebuilt from the trainer every step.
- **Fix:** replace each token's score s_y by s_y − s̄(h), through the scalar
  surrogate −A·(log p(y) − Σ_v sg[q_v] log p(v)).
- **With only the top-k:** an engine returns q on its top k tokens, not on the
  vocabulary. The paper models q's tail as ρ·p, with ρ matching the tail mass.
  Then Σ_v q̂_v ∇log p(v) = Σ_{v∈top-k} (q_v − ρ p_v) ∇log p(v) + ρ·E_p[∇log p],
  and the last term is zero: the correction touches only the k logprobs the
  engine returned, at no cost over the whole vocabulary.

A good target: the problem was widely studied before the paper (importance
sampling, clipping, "a systematic bias, not noise"), the fix is a small change
to the loss with an exact correctness criterion.

## Results

27 runs, Claude Haiku / Sonnet / Opus 5.5 as Claude Code subagents, `exp`
mode, starter v2. Successful runs ([grading.md](grading.md)) / runs, with how
many found the idea (judge) and how many are correct (checks):

| Level | Haiku | Sonnet | Opus |
|---|---|---|---|
| L2b: per-prefix direction | 0/3 (idea 3, correct 0) | 0/3 (idea 0, correct 0) | 3/3 |
| L2a: direction | 0/3 (idea 3, correct 0) | 1/3 (idea 1, correct 1) | 3/3 |
| **L1: pre-paper knowledge only** | 1/3 (idea 2, correct 1) | 0/3 (idea 0, correct 0) | **3/3** |

Opus succeeds in every run. Haiku finds the centering but truncates it to the
sampler's top-k without the tail; Sonnet uses importance sampling or a reward
baseline. Every run: [tables](results_tables.md), [`runs/`](runs/). The 37
runs of starter v1 are kept in `runs/` but not counted.

## One episode

An Opus run at L1 ([full run](runs/20261119-103415-exp-L1/)).
The prompt describes the setting and adds only what was known before the paper:

> *The mismatch acts as a systematic bias in the training objective, not as
> zero-mean noise. [...] Importance-sampling corrections and clipping slow the
> degradation down but do not remove it.*

Fourteen minutes later, the agent's report:

> *Here the tokens come from the sampler `q != p`, and
> `E_{y~q}[grad log p(y)] = -grad KL(q || p) != 0`. Each token's update
> therefore picks up a term `V(s) * (q - p)`. [...] Center each sampled
> token's score under the distribution the token was actually drawn from.*

Judges: idea found, 3/3. Exact checks: drift-free with the sampler's top-k ✓.

## The environment

| Interface ([`paper.py`](paper.py)) | Value |
|---|---|
| Levels | `L0 L1 L2a L2b L3 L4` (default: `L2a L2b L3 L4`) |
| Modes | `exp`: the starter has `run_dev.sh` and `dev/mini.py` (a CPU run of the setting); `noexp`: without |
| Submission, report | `losses.py` (signature of `loss_fn_rl` fixed, change active by default), `REPORT.md` (derivation) |
| Grading | an LLM judge ([`grader/judge.md`](grader/judge.md)) and exact checks ([`grader/check.py`](grader/check.py)): [grading.md](grading.md) |

**What the agent gets.** A real JAX RL framework: `train_rl.py` trains
Qwen3-0.6B on Countdown with a deliberately mismatched sampler, with every
importance-sampling variant already implemented (TIS, MIS, PPO, DAPO, GSPO,
TOPR, DPPO), and the method removed. In `exp` mode, `run_dev.sh` runs the
same kind of mismatch on a tiny transformer sorting digits ([`mini/`](mini/))
in a few minutes.

### Hint levels

| Level | The prompt adds ([`prompts/`](prompts/)) |
|---|---|
| L0 | Only the problem (training collapses when the sampler differs from the trainer), the deliverables and the tools. |
| L1 | What was known before the paper: a systematic bias, not noise; IS corrections and clipping slow it down but do not remove it. |
| L2a | Analyze the expected update under the sampler, on- vs off-policy. |
| L2b | At one token position, conditionally on the prefix. |
| L3 | The decomposition E_q[A s_y \| h] = E_q[A\|h]·s̄(h) + Cov_q(A, s_y \| h), and that the batch holds the sampler's top-20 logprobs. |
| L4 | The method: subtract s̄(h), the scalar surrogate, the top-k tail model. |

[`tests/test_prompts.py`](tests/test_prompts.py) checks that no level says
more than it allows.

### Hiding the answer

[`sanitize_spec.py`](sanitize_spec.py) holds the edits; `core/sanitize.py`
applies and audits them.

| Leak | Where | Fix |
|---|---|---|
| The method's code | `losses.py` branches, a `score_center` flag | Removed by exact, SHA-pinned edits; the starter's loss equals the reference with the method off (tested on values, gradients and metrics for seven IS corrections). |
| Words | docstrings, config comments, README, names like `center_ids` | Rewritten; renamed to `sampler_topk_*`; neutral README; the audit fails the build on any revealing word. |
| Structure | `models/topk.py`, which exists only for the method | Folded into the sampler as a private helper. |
| Data plumbing | a setting that enables the sampler's distribution | Removed: the sampler always returns its top-20 logprobs, as `logprobs=20` does in an inference engine, and nothing documents them for the loss. Saying what the batch holds is an L3 hint. |
| History | git log, the paper's figure, sweeps, tests | Fresh history; `README.md`, `assets/`, `sweeps/`, `tests/` dropped. |
| Prompt | "theoretical argument", the name of the useful config key | Moved into the levels or removed. |
| The agent's context | the session's working directory contained the repository's name | Runs launched from a checkout with a neutral name; a probe subagent before runs; every transcript scanned afterwards. |

Kept on purpose: `log_mass` in `losses.py` (used by `is_weight`, a weak hint),
and the author's name in `LICENSE`, as the MIT license requires.

## Layout

| Path | Role |
|---|---|
| `paper.py` | What this paper declares to `core/` ([interface](../../docs/adding-a-paper.md)). |
| `sanitize_spec.py` | The edits that turn `upstream/` into the starter. |
| `upstream/` | The paper's code, untouched. |
| `starter_extra/` | Added to the `exp` starter: `run_dev.sh`, `dev/summarize.py`. |
| `starter/` | **Generated:** `exp/`, `noexp/`. |
| `prompts/` | `common.md`, one file per level, `tools_*.md` per mode. |
| `mini/` | CPU prototype of the setting; ships in the `exp` starter as `dev/mini.py`. |
| `grader/` | `check.py` (exact checks) and `judge.md` (the judge's prompt). |
| `docker/` | Agent (CPU) and grader images. |
| `tests/` | Starter equivalence, grader validation, prompt leaks. |
| `runs/` | Every run: prompts, transcripts, submission, report, `grade.json`. |
| `summarize_runs.py` | Writes `results_tables.md` from `runs/`. |
| `plot_results.py` | Writes `figures/success-{light,dark}.svg` from `runs/`. |
