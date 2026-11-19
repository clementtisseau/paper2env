# paper2env

**Can an agent rediscover a paper's key insight from what was known before it?**

A collection of rediscovery environments, each built from one research paper.
The agent gets the paper's codebase with the contribution removed, a prompt
limited to what was known before the paper (plus optional hints), and a
sandbox for experiments. It is graded on whether its fix *is* the
contribution, checked exactly, not on whether it reads like the paper.

> **Environments are built by hand, not generated.** Choosing what to hide,
> which property to check and which wrong answers to reject requires
> reading the paper closely. What carries over between papers is
> [listed below](#reusable-across-papers).

**One environment so far:** *Score Centering Stabilizes Off-policy RL*
(Marek & Ryabinin, [arXiv:2609.20807](https://arxiv.org/abs/2609.20807),
Sept. 2026). RL for LLMs drifts when the inference engine that samples
rollouts differs slightly from the trained model; the paper fixes it by
subtracting the sampler-expected score from each token's score. The models
tested predate the paper.

## Results

45 runs, Claude Haiku / Sonnet / Opus 5.5, hint levels L4 → L1. A run counts
as a rediscovery when its loss is **drift-free** under the exact theory
check.

| Level | Haiku | Sonnet | Opus |
|---|---|---|---|
| L3: decomposition given | 3/3 | 3/3 | 3/3 |
| L2b: per-prefix direction | 2/3 | 0/3 | 3/3 |
| L2a: direction | 2/3 | 0/3 | 3/3 |
| **L1: pre-paper knowledge only** | 1/6 | 2/6 | **5/5** |

- **Opus rediscovers the method from L1 in 5/5 runs**, including the top-k
  tail model; 4/5 also compose it correctly with importance weights.
- **Weaker models find the right diagnosis, then the familiar remedy**:
  importance sampling, unclipped or loosely capped.
- **More hints are not always better**: Sonnet scores lower at L2 than at L1.
- **The top-k is the hard part**: Haiku is drift-free with the full
  distribution in 9 runs, with only the top-k in 3.

→ [docs/results.md](docs/results.md) · every run with its transcript:
[`runs/`](runs/)

## How it works

```
upstream/ (paper's code) ──sanitize.py──► starter repo ──► agent container (no network)
                                                              ▲   │
                                       prompt: task + hint L0…L4   │
                                                                  ▼
                                                       losses.py + REPORT.md
                                                                  │
                                 ┌────────────────────────────────┼────────────────────┐
                                 ▼                                ▼                    ▼
                         theory check (main)              CPU guard-rail          LLM judge
                         exact, on a toy policy           doesn't break training  hidden rubric
```

| Piece | What it does |
|---|---|
| **Sanitizer** | Removes the method and every trace of it (code, words, names, history), then audits the result. [Leaks and fixes](docs/design.md#hiding-the-answer) |
| **Hint levels** | L0 (bare problem) → L4 (method given), so results are a curve, not a pass rate. |
| **Theory check** | Verifies the property itself (no drift at any prefix, unchanged on-policy, bounded gradients) by enumerating every completion of a toy policy. Each check rejects a specific plausible wrong answer. [Details](rlenv/grader/README.md) |
| **Isolation** | Agent code runs only in containers without network; grading in a separate container. |

## Running

**Requirements:** Docker (daemon running), Python 3.12, CPU only. On a
network that intercepts TLS, set `SC_RLENV_PIPCA=<CA bundle>` so the images
can build.

```bash
python rlenv/sanitize.py        # build the agent's starter and audit it
python -m pytest rlenv/tests    # validate starter, graders, prompts, hook
```

Both runners use the same starter, prompts and graders. Levels: `L0 L1 L2a
L2b L3 L4`. Mode: `exp` (the agent can run CPU training) or `noexp`.

### Option A: an API model, with Inspect AI

The agent loop is Inspect's `react` agent (bash, python, editor), with
enforced limits.

```bash
pip install -r rlenv/requirements-host.txt
export ANTHROPIC_API_KEY=...                  # or the provider of your model
inspect eval rlenv/task.py --model anthropic/claude-haiku-5-5 \
    -T levels=L1,L3 -T epochs=3 -T experiments=true \
    -T hours=2 -T tokens=5000000 -T messages=500
inspect view                                  # browse transcripts and scores
```

`epochs` = attempts per level (pass@k). Optional: `-T judges=<model>` adds
the LLM judge. Images are built on the first run.

### Option B: a Claude Code subagent, no API key

How every run in `runs/` was produced. The model under test is a subagent of
a Claude Code session, so it uses a Claude subscription instead of API
credits. A hook confines it: each of its Bash commands is rewritten into
`docker exec` in the run's container, and every other tool is denied.

1. **Open a Claude Code session at the repository root**, so it loads
   `.claude/settings.json` (the hook) and `.claude/agents/researcher.md`
   (a Bash-only agent). The checkout's directory name must say nothing about
   the method: the subagent sees it in its context.
2. **Paste this into the session** (choose the level, mode and model):

   ```text
   Run one paper2env episode. Run `python3 rlenv/claude_agent/run.py start --level L1 --hours 2`
   (add --no-experiments for the noexp mode). Then launch the `researcher` subagent with
   model opus, giving it as its prompt exactly the content of the printed agent_prompt.md
   and nothing else. When it has finished, run `python3 rlenv/claude_agent/run.py finish`
   and show me results.json.
   ```

3. **Update the tables:**
   `python scripts/summarize_runs.py > docs/results_tables.md`. New runs are
   picked up automatically.

Before a series of runs, check for leaks: ask the session to launch a
`researcher` subagent that only quotes the lines of its context mentioning a
directory, repository, branch, commit or project name. Details and limits
(no enforced budget, one run at a time):
[rlenv/claude_agent](rlenv/claude_agent/README.md).

## Reusable across papers

| Reusable as is | Rewritten for each paper |
|---|---|
| Claude Code runner and confinement hook (`rlenv/claude_agent/`) | The sanitizer's list of edits: what to remove and rewrite |
| Docker isolation, Inspect task, image tagging (`rlenv/docker/`, `task.py`, `shared.py`) | The property the theory check verifies, and the wrong answers it rejects |
| Sanitizer machinery: exact, SHA-pinned edits, grep audit, fresh history | The hint levels' content |
| Hint-ladder composition and leak tests (`rlenv/prompts/compose.py`) | The judge's reference, rubric and near-misses |
| Judge plumbing: prompt filling, verdict validation, aggregation | Any cheap proxy of the paper's experiments (here, a CPU prototype) |
| The leak checklist in [docs/design.md](docs/design.md) | |

When a second paper is added, each environment gets its own directory and
the reusable parts move into a shared package.

## Repository

| Path | Content |
|---|---|
| [`docs/`](docs/) | [design](docs/design.md) · [results](docs/results.md) · [lessons and limitations](docs/lessons.md) |
| [`rlenv/`](rlenv/) | the score-centering environment ([reference](rlenv/README.md)) |
| [`runs/`](runs/) | 45 runs: prompts, transcripts, submissions, reports, grading ([guide](runs/README.md)) |
| [`upstream/`](upstream/) | the paper's code, untouched (MIT, Martin Marek) |
| [`scripts/`](scripts/) | result tables from the runs |

## Status

**Done:** starter and audit, hint levels L0–L4, theory check and CPU
guard-rail validated on known answers, two runners, 45 runs.
**Not done:** LLM judge calibration and runs, GPU validation on Qwen3-0.6B,
`noexp` runs, more runs per cell. See [limitations](docs/lessons.md).

## Credits

The paper and `upstream/` are by Martin Marek and Max Ryabinin. This
repository builds an evaluation on their work and is not affiliated with
them.
