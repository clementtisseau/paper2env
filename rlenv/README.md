# rlenv: technical reference

Everything that turns the paper's repository (`../upstream/`, untouched) into
an environment. The project overview and the results are in the top-level
[README](../README.md) and in [`docs/`](../docs/).

## Layout

| Path | Role |
|---|---|
| `sanitize.py` | Builds the agent's starter repository from `upstream/`: removes the method and every hint of it, then audits the result. Deterministic and replayable. |
| `starter/` | **Generated, not versioned.** What the agent receives in its container: `exp/` (with training tools) or `noexp/` (without). |
| `starter_extra/` | Added to the `exp` starter: `run_dev.sh` (a short training run on the visible setting) and `dev/summarize.py` (metrics table, text plot, collapse verdict). |
| `prompts/` | `common.md` (task, deliverables, tools, evaluation) plus cumulative hint levels, assembled by `compose.py`; one `tools_*.md` per experimentation mode. |
| `grader/` | Theory check, CPU guard-rail, LLM judge. See [`grader/README.md`](grader/README.md). |
| `mini/` | CPU prototype of the paper's setup (a tiny transformer sorting digits under a mismatched sampler). Ships in the `exp` starter as `dev/mini.py` and backs the guard-rail. |
| `docker/` | Agent sandbox image (CPU) and grader image. |
| `task.py` | Inspect AI task: one sample per level, a `react` agent (bash, python, editor) in the Docker sandbox, scorers `theory`, `guard`, `judge`. Needs an API key for the agent model. |
| `claude_agent/` | The same environment driven by a Claude Code subagent instead of an API model, confined to its container by a hook. This is how the runs in `runs/` were produced. |
| `shared.py` | Images, configuration parsing and grader calls shared by both runners. |
| `tests/` | Starter equivalence, grader validation, prompt leak tests, hook decisions. |

## Hint levels

Levels are cumulative: each one is the previous prompt plus one more piece.

| Level | Adds |
|---|---|
| L0 | Nothing: the problem (training collapses when the sampler differs from the trainer), the deliverables and the tools. |
| L1 | What was known before the paper: the mismatch acts as a systematic bias, not noise; importance-sampling corrections and clipping slow the degradation but do not remove it. |
| L2a | A direction: analyze the expected update under the sampler, on- vs off-policy. |
| L2b | Do it at a single token position, conditionally on the prefix. |
| L3 | The decomposition E_q[A s_y \| h] = E_q[A\|h]·s̄(h) + Cov_q(A, s_y \| h), and that the sampler's top-20 logprobs reach the loss batch. |
| L4 | The method: subtract s̄(h) from each token's score, the scalar surrogate with stop-gradients, the top-k tail model. |

## Running an evaluation

Two runners grade the same way (same starter, prompts and graders):

- **Claude Code subagent** (no API key, used for `runs/`): see
  [`claude_agent/README.md`](claude_agent/README.md).
- **Inspect AI** (API model):
  ```bash
  pip install -r rlenv/requirements-host.txt   # Docker is also required
  inspect eval rlenv/task.py --model <agent model> \
      -T levels=L1,L2a,L3 -T experiments=true -T hours=2 -T tokens=5000000 -T epochs=3
  ```
  A run ends when the agent calls `submit()` or hits a limit: `hours` (working
  time, API waits excluded; wall clock capped at 3x), `tokens` (total, the cost
  bound), `messages` (against loops). `compaction` summarizes the history near
  the context limit. `judges` takes one or more judge models (none: no judge).

The agent's code only ever runs in containers without network: its sandbox,
and the grader image (submission mounted read-only). On a network that
intercepts TLS, set `SC_RLENV_PIPCA=<CA bundle>` to build the images.

## Generating the starter

```bash
python rlenv/sanitize.py                    # rlenv/starter/exp/ (fresh git history)
python rlenv/sanitize.py --no-experiments   # rlenv/starter/noexp/
python -m pytest rlenv/tests                # starter == reference minus the method
```

`sanitize.py` copies `upstream/` without `README.md`, `assets/`, `sweeps/` and
`tests/`, removes the method from `losses.py` and `train_rl.py`, rewrites the
docstrings that mention it, renames `center_ids` / `center_logprobs` to
`sampler_topk_ids` / `sampler_topk_logprobs`, folds `models/topk.py` into the
sampler as a private helper, writes a neutral README, and greps the result for
revealing words. Every edit must match the reference exactly once (larger
spans are pinned by SHA-256): if the upstream code changes, the script fails
instead of letting the method through.

Decisions worth knowing:

- **The sampler's distribution.** The sampler always returns its top-20
  logprobs per position, like an inference engine's `logprobs=20`, and they
  always reach the loss batch. No setting is dedicated to them and nothing
  documents them for the loss: `losses.py` forwards only the sampled tokens'
  logprobs. The prompt says what the batch holds from L3 on.
- **`log_mass`** stays in `losses.py` (used by `is_weight`) with a neutral
  docstring. Its general form `log(q * w(p/q))` is a weak hint.
- **`LICENSE`** keeps the author's name, as the MIT license requires.

## Deliverables and scores

The agent leaves `losses.py` (signature of `loss_fn_rl` fixed, the change
active with default arguments) and `REPORT.md` (derivation, justification,
optional `## Configuration` with `rl.is.*` keys).

1. **Theory check** (CPU, exact): no drift in the expected update under the
   sampler, at every prefix; unchanged on-policy; bounded per-sample
   gradients. Reported with the full distribution, with the top-k only
   (`passed_topk`) and with a TIS correction (`passed_with_tis`).
2. **CPU guard-rail**: a training run on held-out seeds, normalized between
   REINFORCE (0) and the paper's method (1).
3. **LLM judge**: the report and diff against a hidden rubric with near-misses.
