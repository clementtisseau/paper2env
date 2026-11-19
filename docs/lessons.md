# Lessons and limitations

## Things that went wrong, and what they taught

- **Leaks come from where you do not look.** The method's name reached the
  agent not through the code but through the orchestrating session's working
  directory, which Claude Code puts in every subagent's context. The fix was
  operational (a neutral checkout name), and the check was a probe subagent
  asked to quote its own context. Other leaks were caught by tooling: the
  sanitizer's grep audit stopped a build when the environment's own
  `.claude/` directory was copied into the starter, and a prompt test caught
  a "Configuration" section that named the one setting the method needs.
- **Each grader check exists because of a specific wrong answer.** The first
  theory check (zero drift with a constant advantage) accepted an exact
  global reward baseline, which cancels the average drift but not the drift
  at each prefix; `no_drift_prefix` was added for it. Unclipped importance
  sampling is unbiased too, so `bounded` was added. Agent submissions later
  exposed two more blind spots, now reported separately: a loss that centers
  the score under q while the sampled token carries an IS weight (it
  over-corrects as much as REINFORCE drifts), and a top-k version that drops
  the tail and is biased even on-policy.
- **A grader can be fooled by its own test case.** While writing up the
  results, one "passing" run turned out to be token-level importance
  sampling capped at 10. On the check's mild mismatch no ratio reached 10, so
  the weights were exact and the estimator unbiased, and the `bounded` check
  tolerated exactly a factor of 10. The drift check now also runs under the
  severe mismatch, where ratios exceed any loose cap; the run fails there
  (drift 0.21). All runs were re-graded with the fixed check. Reading the
  reports of passing runs, not only the failing ones, is what found it.
- **A CPU proxy reproduces the problem, not the paper's separation.** On a
  tiny transformer, plain REINFORCE plateaus or collapses under a mismatched
  sampler, but the paper's method is not separable from TIS: the paper's
  regime (151k-token vocabulary, 512-token completions, drift summed over
  ~10⁵ tokens per update) does not fit on a CPU. A larger vocabulary did not
  help, because RL quickly crushes the tail. So the empirical signal became a
  guard-rail and the theory check became the main signal.
- **Stale images grade silently.** The grader image was first rebuilt only
  when missing, so a changed check would have kept grading with the old one.
  Images are now tagged by the content they copy in.
- **A judge needs a reference that names the near-misses.** Writing the
  rubric surfaced that the most natural wrong derivation (decompose at the
  sequence level, then subtract a reward baseline) leads to GRPO, which the
  paper explicitly argues is insufficient. The rubric scores that path as a
  near-miss rather than a partial success.

## Limitations of the results

- **Small samples.** Three runs per (level, model) cell in most cases. The
  tables show counts, not rates with confidence intervals.
- **One harness.** All runs use Claude Code's subagent harness with Bash
  only. Results measure "model in this harness"; a different harness (or the
  Inspect `react` agent) could change them.
- **No enforced budget.** The 2-hour budget is stated in the prompt but not
  enforced in the subagent runner; time and commands are recorded instead.
- **No judge yet.** The theory check proves a submission is drift-free, not
  that its report understands why; the reports were read by hand for the
  summary, and the LLM judge has not been run or calibrated.
- **Two starter versions.** 37 runs used the first starter (the sampler's
  distribution behind a setting the L3 prompt named); 8 L1 runs used the
  second (top-20 always in the batch, nothing documented). They are reported
  separately.
- **Contamination is a moving target.** The models tested predate the paper,
  but publishing this repository (rubric, reference, graders) means future
  models may have seen the answer. The method is also already shipped in
  several RL frameworks.
- **No GPU validation.** Nothing here has been trained on Qwen3-0.6B; the
  GPU agent image and the GPU grader are not built.
