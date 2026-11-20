# Method

How a paper becomes a rediscovery environment here, independently of the
paper. Each paper's README describes its own choices
([score-centering](../papers/score-centering/README.md)).

## The question

Can a model, given a real research codebase and only what was known before a
paper, find the paper's contribution? How much of a hint does it need?

A paper is a good target when:

- the problem was known and studied before the paper, so a prompt can state
  it without giving the answer;
- the contribution is a small change with a **crisp correctness criterion**,
  a property that can be checked exactly rather than a benchmark number;
- the paper is recent enough that the models tested cannot have trained on it;
- its code is public, so the agent works in a real codebase.

## What the agent gets

A Docker container (no network, CPUs only by default) holding a **starter**:
the paper's own repository with the contribution surgically removed, so the
loss, the model or the algorithm equals the reference with the method turned
off. The agent must leave a modified file (the paper's `SUBMISSION`) and a
report (`REPORT`) with its derivation.

Two modes per paper: `exp`, where the starter also holds a cheap way to run
the setting (for instance a CPU-sized version of the paper's experiment) so
the agent can observe the problem and test ideas, and `noexp`, where it can
only reason and call its code on inputs it builds.

## Hiding the answer

The answer is everywhere: in the code, the docs, the names, the history, and
in the agent's own context. The checklist each paper goes through:

| Where the answer leaks | What `core/` gives | What the paper decides |
|---|---|---|
| The method's code | exact edits, each required to match once; large spans pinned by SHA-256, so a changed upstream fails the build instead of letting the method through | what to remove, and the starter's equivalence tests (values, gradients) |
| Words and names | a grep audit over the starter that fails the build | the revealing patterns, the allowed exceptions, renames |
| Structure | — | modules or settings that exist only for the method |
| History, docs, figures | fresh git history; files left out by name | which files to drop or rewrite |
| The prompt | tests that each level says no more than it allows | the levels' wording |
| The agent's own context | — (operational) | launch from a checkout with a neutral name; probe a subagent's context before runs |
| The environment's own files | `.claude/`, caches and VCS files never copied | — |

## Hint levels

A ladder of cumulative prompts, from the bare problem to the method itself,
so that results are a curve, not a pass rate: at which level does each model
start finding the contribution, and what does it find before that? The usual
shape:

| Level | The prompt adds |
|---|---|
| L0 | The problem, the deliverables and the tools. |
| L1 | What was known before the paper. |
| L2… | A direction for the analysis, increasingly precise. |
| L3 | The key intermediate result. |
| L4 | The method itself (tests implementation only, like PaperBench). |

`core/compose.py` assembles them; each paper writes the content and decides
its own level names.

## Grading

Two questions, answered separately: is the paper's idea there, and is it
implemented correctly? Depending on the paper, one or both are graded:

1. **Exact checks** answer the second. The contribution is stated as a
   property a correct submission must have, and checked exactly on a toy
   instance (for score-centering: no drift in the expected update, by
   enumerating every completion of a tiny policy). **Each check is introduced
   against a specific plausible wrong answer** and validated on it.
2. **An LLM judge** answers the first. One prompt: the method explained
   clearly, the near-misses that look like it with a label each, and the
   submission. It answers whether the idea is there, quoting the code, and
   which near-miss it is otherwise.

Graders run in their own image, without network, with the submission mounted
read-only. Images are tagged by the content they copy in, so a changed check
is always the one that grades.

## Isolation and runners

The agent's code only runs in containers without network: its sandbox and
the grader's. Two runners share the paper's starter, prompts and graders:

- an **Inspect AI task** (`core/task.py`) for API models, with limits on
  working time, tokens and messages, and compaction for long runs;
- a **Claude Code subagent runner** (`core/runner/`), which needs no API key.
  A `PreToolUse` hook rewrites every subagent Bash call into `docker exec` in
  the run's container and denies every other tool, so the subagent never
  touches the host. It was tested with a subagent that tried to escape.
