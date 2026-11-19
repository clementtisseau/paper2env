## Pass counts by level and model

Each cell: runs passing / runs. `theory` = drift-free with the full distribution or the top-k; `top-k` = drift-free with the sampler's top-k only (what an inference engine gives); `+TIS` = still drift-free when combined with truncated importance sampling.

### Starter v1, exp

| Level | Model | theory | top-k | +TIS | guard-rail | median minutes |
|---|---|---|---|---|---|---|
| L4 | Haiku | 1/1 | 1/1 | 0/1 | 1/1 | 7 |
| L3 | Haiku | 3/3 | 0/3 | 0/3 | 3/3 | 11 |
| L3 | Sonnet | 3/3 | 2/3 | 0/3 | 3/3 | 2 |
| L3 | Opus | 3/3 | 3/3 | 3/3 | 3/3 | 8 |
| L2b | Haiku | 2/3 | 0/3 | 0/3 | 3/3 | 18 |
| L2b | Sonnet | 0/3 | 0/3 | 0/3 | 2/3 | 2 |
| L2b | Opus | 3/3 | 2/3 | 3/3 | 3/3 | 12 |
| L2a | Haiku | 2/3 | 1/3 | 0/3 | 2/3 | 22 |
| L2a | Sonnet | 0/3 | 0/3 | 0/3 | 1/3 | 2 |
| L2a | Opus | 3/3 | 1/3 | 3/3 | 3/3 | 18 |
| L1 | Haiku | 0/3 | 0/3 | 0/3 | 2/3 | 18 |
| L1 | Sonnet | 2/3 | 0/3 | 1/3 | 2/3 | 4 |
| L1 | Opus | 3/3 | 3/3 | 3/3 | 3/3 | 16 |

### Starter v2, exp

| Level | Model | theory | top-k | +TIS | guard-rail | median minutes |
|---|---|---|---|---|---|---|
| L1 | Haiku | 1/3 | 1/3 | 0/3 | 2/3 | 28 |
| L1 | Sonnet | 0/3 | 0/3 | 0/3 | 3/3 | 6 |
| L1 | Opus | 2/2 | 2/2 | 1/2 | 2/2 | 10 |

## What the submitted losses do (theory check, best scenario)

| Level | Model | changes on-policy update | drift remains | drift under severe mismatch | drift-free | unbounded weights |
|---|---|---|---|---|---|---|
| L4 | Haiku | 0 | 0 | 0 | 1 | 0 |
| L3 | Haiku | 0 | 0 | 0 | 3 | 0 |
| L3 | Sonnet | 0 | 0 | 0 | 3 | 0 |
| L3 | Opus | 0 | 0 | 0 | 3 | 0 |
| L2b | Haiku | 0 | 0 | 1 | 2 | 0 |
| L2b | Sonnet | 0 | 0 | 2 | 0 | 1 |
| L2b | Opus | 0 | 0 | 0 | 3 | 0 |
| L2a | Haiku | 0 | 1 | 0 | 2 | 0 |
| L2a | Sonnet | 1 | 0 | 1 | 0 | 1 |
| L2a | Opus | 0 | 0 | 0 | 3 | 0 |
| L1 | Haiku | 2 | 3 | 0 | 1 | 0 |
| L1 | Sonnet | 0 | 2 | 2 | 2 | 0 |
| L1 | Opus | 0 | 0 | 0 | 5 | 0 |

## Every run

| Run | Starter | Mode | Level | Model | min | cmds | theory | top-k | +TIS | guard | mechanism |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [20261118-102641-exp-L4](runs/20261118-102641-exp-L4/) | v1 | exp | L4 | Haiku | 7 | 15 | ✓ | ✓ | ✗ | 1.00 | drift-free |
| [20261118-112015-exp-L3](runs/20261118-112015-exp-L3/) | v1 | exp | L3 | Haiku | 6 | 20 | ✓ | ✗ | ✗ | 1.00 | drift-free |
| [20261118-130727-exp-L3](runs/20261118-130727-exp-L3/) | v1 | exp | L3 | Haiku | 12 | 24 | ✓ | ✗ | ✗ | 1.00 | drift-free |
| [20261118-132121-exp-L3](runs/20261118-132121-exp-L3/) | v1 | exp | L3 | Haiku | 11 | 19 | ✓ | ✗ | ✗ | 0.97 | drift-free |
| [20261118-142128-exp-L3](runs/20261118-142128-exp-L3/) | v1 | exp | L3 | Sonnet | 2 | 4 | ✓ | ✗ | ✗ | 1.00 | drift-free |
| [20261118-142530-exp-L3](runs/20261118-142530-exp-L3/) | v1 | exp | L3 | Sonnet | 2 | 3 | ✓ | ✓ | ✗ | 1.00 | drift-free |
| [20261118-142918-exp-L3](runs/20261118-142918-exp-L3/) | v1 | exp | L3 | Sonnet | 1 | 5 | ✓ | ✓ | ✗ | 1.00 | drift-free |
| [20261118-145338-exp-L3](runs/20261118-145338-exp-L3/) | v1 | exp | L3 | Opus | 8 | 20 | ✓ | ✓ | ✓ | 0.89 | drift-free |
| [20261118-150417-exp-L3](runs/20261118-150417-exp-L3/) | v1 | exp | L3 | Opus | 8 | 20 | ✓ | ✓ | ✓ | 1.00 | drift-free |
| [20261118-151420-exp-L3](runs/20261118-151420-exp-L3/) | v1 | exp | L3 | Opus | 7 | 17 | ✓ | ✓ | ✓ | 0.90 | drift-free |
| [20261118-153038-exp-L2b](runs/20261118-153038-exp-L2b/) | v1 | exp | L2b | Haiku | 38 | 61 | ✗ | ✗ | ✗ | 1.12 | drift under severe mismatch |
| [20261118-161040-exp-L2b](runs/20261118-161040-exp-L2b/) | v1 | exp | L2b | Haiku | 11 | 15 | ✓ | ✗ | ✗ | 1.00 | drift-free |
| [20261118-162323-exp-L2b](runs/20261118-162323-exp-L2b/) | v1 | exp | L2b | Haiku | 18 | 27 | ✓ | ✗ | ✗ | 1.00 | drift-free |
| [20261118-164334-exp-L2b](runs/20261118-164334-exp-L2b/) | v1 | exp | L2b | Sonnet | 2 | 4 | ✗ | ✗ | ✗ | 0.90 | drift under severe mismatch |
| [20261118-164731-exp-L2b](runs/20261118-164731-exp-L2b/) | v1 | exp | L2b | Sonnet | 2 | 3 | ✗ | ✗ | ✗ | 0.37 | unbounded weights |
| [20261118-165114-exp-L2b](runs/20261118-165114-exp-L2b/) | v1 | exp | L2b | Sonnet | 2 | 3 | ✗ | ✗ | ✗ | 0.76 | drift under severe mismatch |
| [20261118-165509-exp-L2b](runs/20261118-165509-exp-L2b/) | v1 | exp | L2b | Opus | 12 | 23 | ✓ | ✗ | ✓ | 0.93 | drift-free |
| [20261118-170849-exp-L2b](runs/20261118-170849-exp-L2b/) | v1 | exp | L2b | Opus | 10 | 14 | ✓ | ✓ | ✓ | 0.88 | drift-free |
| [20261118-172123-exp-L2b](runs/20261118-172123-exp-L2b/) | v1 | exp | L2b | Opus | 17 | 19 | ✓ | ✓ | ✓ | 0.86 | drift-free |
| [20261118-180017-exp-L2a](runs/20261118-180017-exp-L2a/) | v1 | exp | L2a | Haiku | 15 | 19 | ✓ | ✓ | ✗ | 0.72 | drift-free |
| [20261118-181722-exp-L2a](runs/20261118-181722-exp-L2a/) | v1 | exp | L2a | Haiku | 38 | 21 | ✓ | ✗ | ✗ | 1.00 | drift-free |
| [20261118-185704-exp-L2a](runs/20261118-185704-exp-L2a/) | v1 | exp | L2a | Haiku | 22 | 22 | ✗ | ✗ | ✗ | 0.39 | drift remains |
| [20261118-192112-exp-L2a](runs/20261118-192112-exp-L2a/) | v1 | exp | L2a | Sonnet | 5 | 9 | ✗ | ✗ | ✗ | 0.37 | unbounded weights |
| [20261118-192800-exp-L2a](runs/20261118-192800-exp-L2a/) | v1 | exp | L2a | Sonnet | 2 | 3 | ✗ | ✗ | ✗ | 0.90 | drift under severe mismatch |
| [20261118-193142-exp-L2a](runs/20261118-193142-exp-L2a/) | v1 | exp | L2a | Sonnet | 1 | 5 | ✗ | ✗ | ✗ | -2.05 | changes on-policy update |
| [20261118-193441-exp-L2a](runs/20261118-193441-exp-L2a/) | v1 | exp | L2a | Opus | 18 | 24 | ✓ | ✓ | ✓ | 0.86 | drift-free |
| [20261118-195530-exp-L2a](runs/20261118-195530-exp-L2a/) | v1 | exp | L2a | Opus | 18 | 24 | ✓ | ✗ | ✓ | 0.93 | drift-free |
| [20261118-201507-exp-L2a](runs/20261118-201507-exp-L2a/) | v1 | exp | L2a | Opus | 18 | 27 | ✓ | ✗ | ✓ | 1.00 | drift-free |
| [20261118-211120-exp-L1](runs/20261118-211120-exp-L1/) | v1 | exp | L1 | Haiku | 28 | 22 | ✗ | ✗ | ✗ | 0.77 | drift remains |
| [20261118-214056-exp-L1](runs/20261118-214056-exp-L1/) | v1 | exp | L1 | Haiku | 18 | 24 | ✗ | ✗ | ✗ | 0.51 | changes on-policy update |
| [20261118-220047-exp-L1](runs/20261118-220047-exp-L1/) | v1 | exp | L1 | Haiku | 12 | 20 | ✗ | ✗ | ✗ | -0.40 | drift remains |
| [20261118-221758-exp-L1](runs/20261118-221758-exp-L1/) | v1 | exp | L1 | Sonnet | 4 | 8 | ✓ | ✗ | ✓ | 1.13 | drift-free |
| [20261118-222330-exp-L1](runs/20261118-222330-exp-L1/) | v1 | exp | L1 | Sonnet | 4 | 7 | ✓ | ✗ | ✗ | 1.00 | drift-free |
| [20261118-222902-exp-L1](runs/20261118-222902-exp-L1/) | v1 | exp | L1 | Sonnet | 8 | 9 | ✗ | ✗ | ✗ | 0.39 | drift remains |
| [20261118-223834-exp-L1](runs/20261118-223834-exp-L1/) | v1 | exp | L1 | Opus | 16 | 21 | ✓ | ✓ | ✓ | 0.86 | drift-free |
| [20261118-225630-exp-L1](runs/20261118-225630-exp-L1/) | v1 | exp | L1 | Opus | 11 | 18 | ✓ | ✓ | ✓ | 0.86 | drift-free |
| [20261118-230916-exp-L1](runs/20261118-230916-exp-L1/) | v1 | exp | L1 | Opus | 18 | 23 | ✓ | ✓ | ✓ | 0.85 | drift-free |
| [20261119-081907-exp-L1](runs/20261119-081907-exp-L1/) | v2 | exp | L1 | Haiku | 20 | 21 | ✗ | ✗ | ✗ | 0.97 | drift remains |
| [20261119-084106-exp-L1](runs/20261119-084106-exp-L1/) | v2 | exp | L1 | Haiku | 38 | 26 | ✓ | ✓ | ✗ | 0.91 | drift-free |
| [20261119-092039-exp-L1](runs/20261119-092039-exp-L1/) | v2 | exp | L1 | Haiku | 28 | 29 | ✗ | ✗ | ✗ | 0.49 | changes on-policy update |
| [20261119-095015-exp-L1](runs/20261119-095015-exp-L1/) | v2 | exp | L1 | Sonnet | 5 | 8 | ✗ | ✗ | ✗ | 1.03 | drift remains |
| [20261119-095730-exp-L1](runs/20261119-095730-exp-L1/) | v2 | exp | L1 | Sonnet | 19 | 13 | ✗ | ✗ | ✗ | 1.12 | drift under severe mismatch |
| [20261119-101754-exp-L1](runs/20261119-101754-exp-L1/) | v2 | exp | L1 | Sonnet | 6 | 8 | ✗ | ✗ | ✗ | 0.90 | drift under severe mismatch |
| [20261119-102609-exp-L1](runs/20261119-102609-exp-L1/) | v2 | exp | L1 | Opus | 6 | 20 | ✓ | ✓ | ✗ | 0.93 | drift-free |
| [20261119-103415-exp-L1](runs/20261119-103415-exp-L1/) | v2 | exp | L1 | Opus | 14 | 25 | ✓ | ✓ | ✓ | 0.95 | drift-free |
