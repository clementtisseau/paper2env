# Running an environment with a Claude Code subagent

An alternative to `inspect eval` that needs no API key: the model under test
is a subagent of a Claude Code session, and the harness is Claude Code's
subagent harness. Works for any paper in `papers/`; all runs in
`papers/*/runs/` were produced this way.

## Isolation

- `hook.py` is a `PreToolUse` hook (registered in `.claude/settings.json`).
  While a run is active (`state/active.json`), every tool call made by a
  **subagent** (its hook input carries an `agent_id`) is intercepted:
  - Bash: the command is rewritten to
    `docker exec -w /workspace <container> bash -lc '<command>'`, so it runs
    in the container (no network, only the starter), whatever `&&`, `;` or
    `$(...)` it contains;
  - `SubagentHandback` (the final report): allowed;
  - any other tool (Read, Grep, Write, web, Agent…): denied.
  Every call is logged in `runs/<run>/agent_log.jsonl`.
- Calls from the main session are never touched.
- Tested with a real subagent that tried to read the repository, reach the
  network and use Read/Grep: it only ever reached the inside of its container.
  `tests/test_runner_hook.py` covers the hook's decisions.

## A run

1. `python3 core/runner/run.py start --paper <name> --level <level> [--no-experiments] [--hours 2]`
   builds the paper's starter and images, starts the container (no network,
   4 CPUs, 8 GB), activates the run and writes
   `papers/<name>/runs/<run>/agent_prompt.md` (harness preamble + level
   prompt).
2. The session launches the `researcher` subagent (`.claude/agents/`, Bash
   only) with the model under test and that prompt, nothing else.
3. `python3 core/runner/run.py finish` copies out the paper's submission and
   report, runs its graders in its grader image, writes `grade.json` (see
   `core/grade.py`) and the run's statistics in `meta.json`,
   removes the container and ends the run. It also copies the subagent's full
   transcript (messages, tool calls and outputs) into `runs/<run>/transcripts/`,
   with the requested model (`<id>.meta.json`) and the served one
   (`<id>.prefix.json`). Internal reasoning blocks are empty there: Claude
   Code does not keep them. `abort` ends a run without grading.

## Limits

- **The session's name leaks.** The subagent's context contains the session's
  working directory and scratchpad path, which contain the checkout's name.
  Runs must therefore be launched from a checkout whose name says nothing
  about the method. Before runs, a probe subagent quotes its context to check
  for leaks.
- No enforced time or token limit: the prompt's budget is an instruction.
  Time spent and commands run are recorded.
- No automatic LLM judge (it would go through the API).
- One run at a time.
