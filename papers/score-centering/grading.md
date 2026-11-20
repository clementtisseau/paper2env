# How a run is graded

**A run succeeds when both hold:**

1. **The idea is found:** an LLM judge finds that the submitted `losses.py`
   centers each token's score under the sampler.
2. **It is implemented correctly:** exact checks confirm the loss is
   drift-free, using only the sampler's top-k logprobs.

Only `losses.py` is graded, called with its default arguments, as the task
asked. The report, its `## Configuration` section and how the loss handles an
importance-sampling correction are not graded.

The result is `runs/<run>/grade.json`: `correct` (1), `idea_found` (2),
`success` (both), with the details of each.

## 1. The judge: [`grader/judge.md`](grader/judge.md)

One prompt, with a clear account of the method, the near-misses that look
like it, and the submission (`losses.py` and its diff against the starter).
The judge answers with one JSON object:

```json
{"evidence": "<the lines of losses.py that decide>", "idea_found": true, "wrong_answer": null}
```

`wrong_answer` names what the loss does instead when the idea is missing:
`reward_baseline`, `expectation_under_p`, `wrong_coefficient`,
`sampled_tokens_only`, `importance_weights`, `kl_to_sampler`, `tuning` or
`other`. With several judges, `idea_found` is the majority.

## 2. The exact checks: [`grader/check.py`](grader/check.py)

A toy language model with 5 tokens and completions of 3 tokens: all 125
completions can be enumerated, so the loss's expected update is computed
exactly. The sampler q is the trained model p plus noise, seen only through
its top-3 logprobs. The loss is called exactly as `train_rl.py` calls it.
All four checks must pass, on three seeds:

| Check | Given | Requires | Rejects |
|---|---|---|---|
| `no_drift` | a constant advantage, mild and severe noise | expected update ≈ 0 | REINFORCE; truncated or masked IS; IS capped at 10 |
| `no_drift_prefix` | an advantage that depends on the first tokens, mean zero | ≈ 0 at every later position | a global reward baseline (GRPO at its limit) |
| `on_policy` | q = p | the same update as REINFORCE | a loss that learns nothing |
| `bounded` | severe noise | no gradient above 10× REINFORCE's largest | unclipped IS |

≈ 0 means below 10⁻⁴ of the gradients' size. Each rejection is tested in
[`tests/test_check.py`](tests/test_check.py), on the wrong answer itself.

The judge and the checks answer different questions: a loss can pass the
checks on a toy problem without the method, and the method can be
implemented wrongly.

## Running

```bash
python papers/score-centering/grader/check.py path/to/losses.py   # JSON, exit 0 if correct
python core/grade.py --paper score-centering [run ...]           # re-run the checks on runs
```

`check.py` alone runs the code on the host, for trusted files only;
`core/grade.py` runs it in the grader container
(no network, the file mounted read-only).
