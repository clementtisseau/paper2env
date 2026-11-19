# paper2env

**Can an agent rediscover a paper's key insight from what was known before it?**

A collection of rediscovery environments, each built from one research paper.
The agent gets the paper's codebase with the contribution removed, a prompt
limited to what was known before the paper (plus optional hints), and a
sandbox for experiments. It is graded on whether its fix *is* the
contribution, checked exactly, not on whether it reads like the paper.

> **Environments are built by hand, not generated.** Choosing what to hide,
> which property to check and which wrong answers to reject requires reading
> the paper closely. What is shared between papers lives in [`core/`](core/);
> what is written for each paper lives in `papers/<name>/`
> ([what goes where](#what-is-shared-and-what-is-per-paper)).

## Papers

| Paper (`--paper`) | Agent must find | Runs | Headline |
|---|---|---|---|
| [`score-centering`](papers/score-centering/): *Score Centering Stabilizes Off-policy RL* (Marek & Ryabinin, [arXiv:2609.20807](https://arxiv.org/abs/2609.20807)) | Why RL for LLMs drifts when the sampler differs slightly from the trained model, and a loss with no drift | 45 (Haiku, Sonnet, Opus 5.5) | Opus rediscovers the method from pre-paper knowledge alone in 5/5 runs; weaker models find the diagnosis, then fall back on importance sampling. [Results](papers/score-centering/results.md) |

Adding a paper: [docs/adding-a-paper.md](docs/adding-a-paper.md).

## How an environment works

```
papers/<name>/upstream/ ──core/sanitize.py──► starter repo ──► agent container (no network)
     (paper's code)          + the paper's edits                  ▲   │
                                          prompt: task + hint level   │
                                                                      ▼
                                                        submission + report
                                                                      │
                                         the paper's graders, in a separate container
                                         (main: an exact check of the property)  + LLM judge
```

| Piece | Shared mechanism (`core/`) | Per paper |
|---|---|---|
| **Starter** | copy, pinned edits, grep audit, fresh git history | what to remove and rewrite, revealing words |
| **Hint levels** | cumulative composition, `exp`/`noexp` tools section | the levels' content, leak tests |
| **Graders** | images tagged by content, sandboxed runs, results | the property checked, the wrong answers it rejects |
| **LLM judge** | prompt filling, verdict validation, scoring, aggregation | reference, rubric, near-misses, weights |
| **Runners** | Inspect AI task; Claude Code subagent runner and hook | nothing |

Design and rationale: [docs/method.md](docs/method.md). What went wrong and
what it taught: [docs/lessons.md](docs/lessons.md).

## Running

**Requirements:** Docker (daemon running), Python 3.12, CPU only. On a
network that intercepts TLS, set `P2E_PIPCA=<CA bundle>` so the images can
build.

```bash
pip install -r requirements-host.txt
python -m pytest                                  # core, runner hook, every paper's tests
python core/sanitize.py --paper score-centering   # build and audit a starter (optional: runners do it)
```

Every command takes the paper by name (`--paper` or `-T paper=`). Both
runners use the paper's starter, prompts and graders, so they grade the same
way. Each paper lists its levels and modes in its README; for
`score-centering`: levels `L0 L1 L2a L2b L3 L4`, mode `exp` (the agent can
run CPU training) or `noexp`.

### Option A: an API model, with Inspect AI

The agent loop is Inspect's `react` agent (bash, python, editor), with
enforced limits.

```bash
export ANTHROPIC_API_KEY=...                  # or the provider of your model
inspect eval core/task.py --model anthropic/claude-haiku-5-5 \
    -T paper=score-centering -T levels=L1,L3 -T epochs=3 -T experiments=true \
    -T hours=2 -T tokens=5000000 -T messages=500
inspect view                                  # browse transcripts and scores
```

`levels` defaults to the paper's `DEFAULT_LEVELS`. `epochs` = attempts per
level (pass@k). `-T judges=<model>` adds the LLM judge. There is one scorer
per grader of the paper. Images are built on the first run.

### Option B: a Claude Code subagent, no API key

How every run in `papers/*/runs/` was produced. The model under test is a
subagent of a Claude Code session, so it uses a Claude subscription instead
of API credits. A hook confines it: each of its Bash commands is rewritten
into `docker exec` in the run's container, and every other tool is denied.

1. **Open a Claude Code session at the repository root**, so it loads
   `.claude/settings.json` (the hook) and `.claude/agents/researcher.md`
   (a Bash-only agent). The checkout's directory name must say nothing about
   the method: the subagent sees it in its context.
2. **Paste this into the session** (choose the paper, level, mode and model):

   ```text
   Run one paper2env episode. Run `python3 core/runner/run.py start --paper score-centering --level L1 --hours 2`
   (add --no-experiments for the noexp mode). Then launch the `researcher` subagent with
   model opus, giving it as its prompt exactly the content of the printed agent_prompt.md
   and nothing else. When it has finished, run `python3 core/runner/run.py finish`
   and show me results.json.
   ```

3. Runs land in `papers/<name>/runs/<run>/`. For score-centering, update the
   tables with
   `python papers/score-centering/summarize_runs.py > papers/score-centering/results_tables.md`.

Before a series of runs, check for leaks: ask the session to launch a
`researcher` subagent that only quotes the lines of its context mentioning a
directory, repository, branch, commit or project name. Details and limits
(no enforced budget, one run at a time): [core/runner](core/runner/README.md).

## What is shared and what is per paper

| `core/`: shared, reused as is | `papers/<name>/`: written for each paper |
|---|---|
| `paper.py`: loads `papers/<name>/paper.py`, the interface below | `paper.py`: name, levels, deliverables, graders, audit patterns |
| `sanitize.py`: copy, SHA-pinned edits, grep audit, fresh history | `sanitize_spec.py`: the edits that remove the contribution |
| `compose.py`: cumulative hint levels, tools section per mode | `prompts/`: common task, one file per level |
| `images.py`: content-tagged images, sandboxed grader runs | `docker/`: agent and grader Dockerfiles |
| `judge.py`: prompt filling, verdict validation and scoring | `grader/`: the checks, and the judge's reference, rubric and config |
| `task.py`: Inspect task, one scorer per grader | `upstream/`: the paper's code, untouched |
| `runner/`: Claude Code subagent runner and confinement hook | `tests/`, `runs/`, results and paper-specific lessons |

The interface a paper implements, and a recipe for a new one:
[docs/adding-a-paper.md](docs/adding-a-paper.md).

## Repository

| Path | Content |
|---|---|
| [`core/`](core/) | the shared machinery (above) |
| [`papers/`](papers/) | one directory per environment |
| [`docs/`](docs/) | [method](docs/method.md) · [adding a paper](docs/adding-a-paper.md) · [lessons](docs/lessons.md) |
| [`tests/`](tests/) | tests of `core/` (judge plumbing, runner hook) |

## Status

**score-centering:** starter and audit, levels L0–L4, theory check and CPU
guard-rail validated on known answers, both runners, 45 runs. **Not done:**
LLM judge calibration and runs, GPU validation, `noexp` runs, more runs per
cell, a second paper.

## Credits

Each paper's `upstream/` belongs to its authors, under its own license
(score-centering: MIT, Martin Marek and Max Ryabinin). This repository builds
evaluations on their work and is not affiliated with them.
