# CPU prototype: results

Task: sort 5 digits (vocabulary 12, 12 tokens). A small transformer (2
layers, d = 64), 15 supervised warm-up steps (reward ≈ 0.56), then RL with
SGD (lr 0.2), 16 prompts × 8 completions per step. The sampler is the
trainer's weights plus a fixed relative Gaussian perturbation (as
`sampler.weight_noise_scale` in `train_rl.py`). Loss: the paper's
`loss_fn_rl`. A 600-step run takes 1 to 2 minutes on one CPU core.

Final evaluation reward (clean model, 512 fixed prompts).

## ±1 advantages (`plus_minus`), noise 0.3, 600 steps, 2 seeds

| Method | seed 0 | seed 1 |
|---|---|---|
| REINFORCE | 0.66 | 0.66 |
| TIS | 0.81 | 0.79 |
| MIS | 0.83 | 0.81 |
| Paper's method (SC) | 0.83 | 0.83 |
| SC + TIS | 0.86 | 0.85 |

## Group-centered advantages, 600 steps, seed 0

| Noise | REINFORCE | TIS | SC |
|---|---|---|---|
| 0.5 | 0.57 (peak 0.65, KL → 9) | 0.70 | 0.70 |
| 0.7 | 0.23 (collapse, KL → 23) | 0.64 | 0.49 |
| 1.0 | 0.22 | — | 0.37 |

## Group-centered advantages, 3,000 steps, seed 0

| Noise | REINFORCE | TIS | SC | SC + TIS |
|---|---|---|---|---|
| 0.2 | 0.93 | 0.98 | 0.96 | 0.97 |
| 0.3 | 0.90 | 0.94 | 0.94 | 0.94 |

## Large vocabulary

Distractor tokens that never appear in a correct answer
(`MINI_EXTRA_TOKENS`), to mimic the long tail of an LLM vocabulary.
Group-centered advantages, 600 steps, seed 0.

Vocabulary 512, 5-token answers, 30 warm-up steps (reward 0.73; tail mass
under q ≈ 4 %, falling to 0.4 % during RL):

| Noise | REINFORCE | TIS | SC |
|---|---|---|---|
| 0.1 | 0.91 | 0.92 | 0.92 |
| 0.3 | 0.82 | 0.88 | 0.86 |

Vocabulary 512, 10-token answers (`MINI_PROMPT=10`), 40 warm-up steps:

| Noise | REINFORCE | TIS | SC |
|---|---|---|---|
| 0.2 | 0.71 | 0.71 | 0.71 |
| 0.4 | 0.51 (falling) | 0.63 | 0.65 |

## Reading

- The drift is reproduced: REINFORCE plateaus or collapses, with the KL
  between sampler and trainer blowing up, while corrections hold.
- The paper's method clearly beats REINFORCE, but **not TIS or MIS**: the
  gaps (≤ 0.04) are within seed noise, and under a very large mismatch (KL of
  1 to 2 nats per token) TIS does better.
- A larger vocabulary does not separate them either: RL quickly crushes the
  tail, which removes the rare tokens' weight in the drift.
- The paper's regime (small mismatch, 151k-token vocabulary, 512-token
  completions, drift summed over ~10⁵ tokens per update) is out of reach on a
  CPU. So the prototype is used as a guard-rail ("removes the drift or not"),
  and the finer separation is left to the theory check and the judge.
