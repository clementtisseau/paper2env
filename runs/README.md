# Runs

45 runs of the environment, all with the `exp` starter (the agent can run
CPU training) and a stated budget of 2 hours, driven by the Claude Code
subagent runner ([`rlenv/claude_agent/`](../rlenv/claude_agent/)). Tables and
per-run results: [`docs/results.md`](../docs/results.md).

`index.json` gives each run's model (as launched, and as served according to
the transcript), level and starter version (`v1`: the sampler's distribution
behind a setting that the L3 prompt names; `v2`: the sampler's top-20 always
in the loss batch, documented nowhere).

## A run directory

| File | Content |
|---|---|
| `task_prompt.md` | The level's prompt, as composed by `rlenv/prompts/compose.py`. |
| `agent_prompt.md` | What the subagent received: the harness preamble plus the task prompt, nothing else. |
| `transcripts/` | The subagent's full transcript (messages, tool calls and outputs). Internal reasoning blocks are empty: Claude Code does not keep them. Session links are masked. |
| `agent_log.jsonl` | Every tool call as the confinement hook saw it: the original command, the `docker exec` it was rewritten to, and the decision. |
| `losses.py`, `losses.diff` | The submitted loss and its diff against the starter. |
| `REPORT.md` | The agent's derivation and justification. |
| `results.json` | Grading recorded at the end of the run (theory check and CPU guard-rail, with the grader of that time). |
| `regraded.json` | The theory check re-run with the final grader, so every run is graded the same way. The tables use this one. |
| `meta.json` | Container, image, level, budget. |

## Worth reading

- **Full rediscovery from L1 (Opus):**
  [`20261118-223834-exp-L1`](20261118-223834-exp-L1/REPORT.md) and
  [`20261119-103415-exp-L1`](20261119-103415-exp-L1/REPORT.md) derive the
  per-prefix drift, explain why group centering does not remove it, center
  the score under the sampler, model the top-k tail, and fold importance
  weights into the centering measure.
- **Right diagnosis, familiar remedy (Sonnet):**
  [`20261118-222902-exp-L1`](20261118-222902-exp-L1/REPORT.md) finds that
  E_q[∇log p] = −∇KL(q‖p) turns every baseline into a drift, then fixes it
  with unclipped importance weights, which are unbiased but unbounded.
  [`20261119-095730-exp-L1`](20261119-095730-exp-L1/REPORT.md) does the same
  with weights capped at 10; it fooled the first version of the theory check
  (see [lessons](../docs/lessons.md)).
- **Correct with the full distribution, wrong with the top-k (Haiku, L3):**
  [`20261118-112015-exp-L3`](20261118-112015-exp-L3/REPORT.md) truncates the
  sum to the head without modeling the tail, which is biased even on-policy.
