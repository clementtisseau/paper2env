# Method

How a paper becomes a rediscovery environment, independently of the paper.
Each paper's own choices are in its README
([score-centering](../papers/score-centering/README.md)).

## The question

Can a model, given a real research codebase and only what was known before a
paper, find the paper's contribution? How much of a hint does it need?

A paper is a good target when:

- the problem was known before the paper, so a prompt can state it without
  giving the answer;
- the contribution is a small change whose correctness can be checked, not
  only a benchmark number;
- the paper postdates the models tested;
- its code is public, so the agent works in a real codebase.

## Principles

- **Pre-paper knowledge only.** The prompt states the problem as it was known
  before the paper; hints are explicit and leveled (below).
- **Grade the contribution, not the wording.** An LLM judge decides whether
  the paper's idea is in the submission; exact checks decide whether it is
  implemented correctly. Any correct fix passes, whatever it is called.
- **A curve, not a pass rate.** Hint levels show how much help each model
  needs, and what it finds before that.
- **Isolation.** The agent's code only runs in containers without network.

## What the agent gets

A Docker container (no network, CPUs only) holding a **starter**: the paper's
own repository with the contribution surgically removed, so it equals the
reference with the method turned off. The agent leaves a modified file (the
paper's `SUBMISSION`) and a report.

Two modes: `exp`, where the starter also holds a cheap way to run the setting
(for score-centering, a CPU-sized version of the experiment), and `noexp`,
where the agent can only reason and call its code on inputs it builds.

## Hiding the answer

The answer is in the code, the docs, the names, the history, and the agent's
own context. Each paper goes through this checklist:

| Where the answer leaks | What `core/` gives | What the paper decides |
|---|---|---|
| The method's code | exact edits, each matching once; large spans pinned by SHA-256, so a changed upstream fails the build | what to remove; tests that the starter equals the reference with the method off |
| Words and names | a grep audit that fails the build | the revealing patterns, allowed exceptions, renames |
| Structure | — | modules or settings that exist only for the method |
| History, docs, figures | fresh git history; files left out by name | which files to drop or rewrite |
| The prompt | tests that each level says no more than it allows | the levels' wording |
| The agent's own context | — | a checkout with a neutral name; a probe of the agent's context before runs |
| The environment's own files | `.claude/`, caches and VCS files never copied | — |

## Hint levels

Cumulative prompts, from the bare problem to the method:

| Level | The prompt adds |
|---|---|
| L0 | The problem, the deliverables and the tools. |
| L1 | What was known before the paper. |
| L2… | A direction for the analysis, increasingly precise. |
| L3 | The key intermediate result. |
| L4 | The method itself (tests implementation only, like PaperBench). |

`core/compose.py` assembles them; each paper writes them and names its levels.

## Grading

Two questions, answered separately; depending on the paper, one or both are
graded:

1. **Is the idea there?** An LLM judge with one prompt: the method explained
   clearly, the near-misses that look like it (one label each), and the
   submission. It quotes the code that decides, says whether the idea is
   there, and which near-miss it is otherwise. Several judges, majority vote.
2. **Is it implemented correctly?** Exact checks of a property a correct
   submission must have, on a toy instance where every expectation can be
   computed exactly. Each check is written against a specific plausible wrong
   answer and tested on it.

Graders run in their own image, without network, with the submission mounted
read-only. Images are tagged by the content they copy in, so a changed check
is always the one that grades.

## Runners

Both use the paper's starter, prompts and graders:

- an **Inspect AI task** (`core/task.py`) for API models, with limits on
  working time, tokens and messages;
- a **Claude Code subagent runner** (`core/runner/`), which needs no API key:
  a hook rewrites every subagent Bash call into `docker exec` in the run's
  container and denies every other tool. Details:
  [core/runner](../core/runner/README.md).

## Lessons

From building score-centering; each holds for any paper.

- **Leaks come from where you do not look.** The method's name reached the
  agent through the orchestrating session's working directory, which Claude
  Code puts in every subagent's context. Fixed with a neutral checkout name,
  and checked by a probe subagent that quotes its own context. Tooling caught
  the rest: the grep audit stopped a build that copied `.claude/` into the
  starter, and a prompt test caught a section naming the one useful setting.
- **Each check exists because of a wrong answer.** The first drift check
  accepted an exact global reward baseline (it cancels the drift on average,
  not at each prefix), so `no_drift_prefix` was added; unclipped importance
  sampling is unbiased too, so `bounded` was added.
- **Read the runs that pass.** One "passing" run was importance sampling
  capped at 10: on the check's mild mismatch no ratio reached the cap, so it
  looked exact. The drift check now also runs under a severe mismatch, and
  every run was re-graded.
- **Separate the idea from its implementation.** Checks alone mix up two
  failures. With a judge, they come apart: most Haiku runs find the centering
  but drop the top-k tail, which the checks reject.
- **A cheap proxy reproduces the problem, not the separation.** On a CPU-sized
  model, the paper's method and importance sampling train equally well, so a
  short training run cannot grade the contribution; it was dropped.
- **Stale images grade silently.** Images rebuilt only when missing kept
  grading with an old check; they are now tagged by content.
