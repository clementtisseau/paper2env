# paper2env

**Can an agent rediscover a paper's key insight from what was known before it?**

paper2env turns a research paper into an evaluation environment. The agent
gets the paper's own codebase with the contribution surgically removed, a
prompt that says only what was known before the paper (plus optional hints),
and a sandbox where it can run experiments. Its fix is graded on whether it
*is* the contribution, checked exactly, not on whether it looks like the
paper.

The first environment is built on *Score Centering Stabilizes Off-policy
Reinforcement Learning* (Marek & Ryabinin, [arXiv:2609.20807](https://arxiv.org/abs/2609.20807),
September 2026): RL for LLMs drifts and collapses when the inference engine
that samples rollouts differs slightly from the model being trained, and the
paper fixes it by subtracting the sampler-expected score from each token's
score. The models tested predate the paper.

## Results

45 runs with Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5 as Claude Code
subagents, at hint levels L4 (method given) down to L1 (pre-paper knowledge
only). A run counts as a rediscovery when its loss is **drift-free**: the
exact theory check finds no drift at any prefix, no change on-policy, and
bounded gradients.

| Level | Haiku | Sonnet | Opus |
|---|---|---|---|
| L3 (decomposition given) | 3/3 | 3/3 | 3/3 |
| L2b (per-prefix direction) | 2/3 | 0/3 | 3/3 |
| L2a (direction) | 2/3 | 0/3 | 3/3 |
| **L1 (pre-paper knowledge)** | 1/6 | 2/6 | **5/5** |

- **Opus rediscovers the method from L1 in all five runs**: the per-prefix
  drift, why group centering does not remove it, the score correction, the
  top-k tail model, and in four runs the composition with importance weights.
- **Weaker models find the diagnosis but reach for the familiar remedy**:
  importance sampling, unclipped or loosely capped. The grader's checks were
  built to tell the two apart.
- **More hints are not always better**: Sonnet does worse at L2 than at L1;
  the L2 hint seems to steer it toward reweighting.
- **The top-k is the hard part**: inference engines return only their top-k
  logprobs, and truncating the sum is biased even on-policy. Haiku is
  drift-free with the full distribution in 9 runs, with the top-k in 3.

Details, per-run tables and caveats: [docs/results.md](docs/results.md). Every
run's prompt, transcript, submission, report and grading is in
[`runs/`](runs/).

## How it works

```
upstream/ (the paper's code, untouched)
   │  rlenv/sanitize.py: remove the method and every hint of it, audit, fresh git
   ▼
starter repo ──► agent container (no network)  ◄── prompt: task + hint level L0…L4
                       │  experiments on a CPU version of the setup (optional)
                       ▼
                 losses.py + REPORT.md
                       │
          ┌────────────┼──────────────────┐
          ▼            ▼                  ▼
   theory check   CPU guard-rail      LLM judge
   (exact, main)  (doesn't break      (hidden rubric,
                   training)           near-misses)
```

- **Hiding the answer** is most of the work: code branches, docstrings,
  names, a module that exists only for the method, a config setting, the git
  history, the prompt's wording, and the agent's own context (the session's
  directory name). Each leak and its fix: [docs/design.md](docs/design.md).
- **Hint levels** go from the bare problem (L0) to the method itself (L4), so
  results are a curve: where each model starts finding the contribution, and
  what it finds before that.
- **Grading checks the property, not the wording.** The contribution is that
  the loss's expected update has no drift under any sampler. The theory check
  verifies it exactly on a toy policy where every completion is enumerated,
  in three settings (full distribution, top-k only, combined with TIS). Each
  check was added to reject a specific plausible wrong answer, two of them
  found in agent submissions: [rlenv/grader](rlenv/grader/README.md).
- **Isolation.** Agent code runs only in containers without network. With the
  Claude Code runner, a hook rewrites every subagent command into the run's
  container and denies every other tool.

## Repository

| Path | What |
|---|---|
| [`docs/`](docs/) | [design](docs/design.md), [results](docs/results.md), [lessons and limitations](docs/lessons.md) |
| [`rlenv/`](rlenv/) | the environment: sanitizer, prompts, graders, CPU prototype, Docker, runners, tests ([reference](rlenv/README.md)) |
| [`runs/`](runs/) | all 45 runs with transcripts and grading ([guide](runs/README.md)) |
| [`upstream/`](upstream/) | the paper's reference implementation (MIT, Martin Marek), untouched |
| [`scripts/`](scripts/) | result tables from the runs |

## Running it

Requirements: Docker, Python 3.12. CPU only.

```bash
python rlenv/sanitize.py                 # build the agent's starter (and audit it)
python -m pytest rlenv/tests             # starter equivalence, grader validation, leak tests
```

With an API model (Inspect AI):

```bash
pip install -r rlenv/requirements-host.txt
inspect eval rlenv/task.py --model <provider/model> -T levels=L1,L3 -T epochs=3
```

With a Claude Code subagent (no API key; how `runs/` was produced), from a
checkout whose name says nothing about the method: see
[rlenv/claude_agent](rlenv/claude_agent/README.md).

## Status

Built: the starter and its audit, hint levels L0–L4, the theory check and
CPU guard-rail (validated on known answers), the LLM judge (built, not yet
calibrated or run), two runners, 45 runs. Not done: GPU validation on
Qwen3-0.6B, judge calibration, more runs per cell, runs without
experimentation (`noexp`). Limitations: [docs/lessons.md](docs/lessons.md).

## Credits

The paper and the code in `upstream/` are by Martin Marek and Max Ryabinin.
This repository builds an evaluation on top of their work and is not
affiliated with them.
