# Running with a Claude Code subagent

The model under test is a subagent of a Claude Code session, so a run needs
no API key. Works for any paper; every run in `papers/*/runs/` was made this
way.

## A run

Open a Claude Code session at the repository root (it loads the hook from
`.claude/settings.json` and the `researcher` agent from `.claude/agents/`),
then paste, filled in:

```text
Run one paper2env episode. Run `python3 core/runner/run.py start --paper <paper> --level <level> --hours 2`
(add --no-experiments for the noexp mode). Then launch the `researcher` subagent with
model <haiku|sonnet|opus>, giving it as its prompt exactly the content of the printed
agent_prompt.md and nothing else. When it has finished, run `python3 core/runner/run.py finish`
and show me grade.json.
```

1. `start` builds the paper's starter and images, starts the run's container
   (no network, 4 CPUs, 8 GB) and writes `runs/<run>/agent_prompt.md`: a short
   harness preamble plus the level's prompt.
2. The `researcher` subagent (Bash only) works with that prompt alone.
3. `finish` copies out the submission and the report, runs the paper's checks
   (`grade.json`), records time and commands in `meta.json`, copies the
   subagent's transcript into `runs/<run>/transcripts/`, and removes the
   container. `abort` ends a run without grading.

The judge is run separately, on finished runs ([grading](../../papers/score-centering/grading.md)).
On a network that intercepts TLS, set `P2E_PIPCA=<CA bundle>` so the images
build.

## Before a series of runs

- **Name the checkout neutrally.** The subagent's context holds the session's
  working directory and scratchpad path, so the checkout's name must say
  nothing about the method.
- **Probe for leaks.** Launch a `researcher` subagent that only quotes the
  lines of its context mentioning a directory, repository, branch, commit,
  session or project name, and check them.
- **One run at a time per session.** For parallel runs, use several sessions,
  each pushing to its own branch.

## Isolation

`hook.py` is a `PreToolUse` hook. While a run is active (`state/active.json`),
every tool call of a **subagent** (its hook input has an `agent_id`) is
handled there:

- Bash: rewritten to `docker exec -w /workspace <container> bash -lc '<command>'`,
  so the whole command runs in the container, whatever `&&`, `;` or `$(...)`
  it contains;
- `SubagentHandback` (the final report): allowed;
- anything else (Read, Write, web, Agent…): denied.

Every call is logged in `runs/<run>/agent_log.jsonl`; the main session's calls
are never touched. Tested with a subagent that tried to read the repository,
reach the network and use other tools; `tests/test_runner_hook.py` covers the
hook's decisions.

## Limits

- The time budget is stated in the prompt, not enforced; time and commands
  are recorded.
- The transcripts' internal reasoning blocks are empty: Claude Code does not
  keep them.
