## By level and model

Each cell: runs passing / runs. `correct`: the four exact checks pass with the sampler's top-k; `idea`: the judge finds the method in the loss; `success`: both, with its 95% Wilson interval.

### Starter v1, exp

| Level | Model | correct | idea | success | success, 95% CI | median minutes |
|---|---|---|---|---|---|---|
| L4 | Haiku | 1/1 | ? | ? | ? | 7 |
| L3 | Haiku | 0/3 | ? | ? | ? | 11 |
| L3 | Sonnet | 2/3 | ? | ? | ? | 2 |
| L3 | Opus | 3/3 | ? | ? | ? | 8 |
| L2b | Haiku | 0/3 | ? | ? | ? | 18 |
| L2b | Sonnet | 0/3 | ? | ? | ? | 2 |
| L2b | Opus | 2/3 | ? | ? | ? | 12 |
| L2a | Haiku | 1/3 | ? | ? | ? | 22 |
| L2a | Sonnet | 0/3 | ? | ? | ? | 2 |
| L2a | Opus | 1/3 | ? | ? | ? | 18 |
| L1 | Haiku | 0/3 | ? | ? | ? | 18 |
| L1 | Sonnet | 0/3 | ? | ? | ? | 4 |
| L1 | Opus | 3/3 | ? | ? | ? | 16 |

### Starter v2, exp

| Level | Model | correct | idea | success | success, 95% CI | median minutes |
|---|---|---|---|---|---|---|
| L2b | Haiku | 0/9 | 8/9 | 0/9 | 0–30% | 14 |
| L2b | Sonnet | 0/9 | 0/9 | 0/9 | 0–30% | 1 |
| L2b | Opus | 9/9 | 9/9 | 9/9 | 70–100% | 12 |
| L2a | Haiku | 1/9 | 8/9 | 1/9 | 2–44% | 12 |
| L2a | Sonnet | 3/9 | 3/9 | 3/9 | 12–65% | 4 |
| L2a | Opus | 9/9 | 9/9 | 9/9 | 70–100% | 12 |
| L1 | Haiku | 1/9 | 7/9 | 1/9 | 2–44% | 19 |
| L1 | Sonnet | 3/9 | 4/9 | 3/9 | 12–65% | 11 |
| L1 | Opus | 8/9 | 9/9 | 8/9 | 56–98% | 12 |

## Every run

| Run | Starter | Mode | Level | Model | min | cmds | correct | idea | success | judge: instead |
|---|---|---|---|---|---|---|---|---|---|---|
| [20261118-102641-exp-L4](runs/20261118-102641-exp-L4/) | v1 | exp | L4 | Haiku | 7 | 15 | ✓ | ? | ? |  |
| [20261118-112015-exp-L3](runs/20261118-112015-exp-L3/) | v1 | exp | L3 | Haiku | 6 | 20 | ✗ | ? | ? |  |
| [20261118-130727-exp-L3](runs/20261118-130727-exp-L3/) | v1 | exp | L3 | Haiku | 12 | 24 | ✗ | ? | ? |  |
| [20261118-132121-exp-L3](runs/20261118-132121-exp-L3/) | v1 | exp | L3 | Haiku | 11 | 19 | ✗ | ? | ? |  |
| [20261118-142128-exp-L3](runs/20261118-142128-exp-L3/) | v1 | exp | L3 | Sonnet | 2 | 4 | ✗ | ? | ? |  |
| [20261118-142530-exp-L3](runs/20261118-142530-exp-L3/) | v1 | exp | L3 | Sonnet | 2 | 3 | ✓ | ? | ? |  |
| [20261118-142918-exp-L3](runs/20261118-142918-exp-L3/) | v1 | exp | L3 | Sonnet | 1 | 5 | ✓ | ? | ? |  |
| [20261118-145338-exp-L3](runs/20261118-145338-exp-L3/) | v1 | exp | L3 | Opus | 8 | 20 | ✓ | ? | ? |  |
| [20261118-150417-exp-L3](runs/20261118-150417-exp-L3/) | v1 | exp | L3 | Opus | 8 | 20 | ✓ | ? | ? |  |
| [20261118-151420-exp-L3](runs/20261118-151420-exp-L3/) | v1 | exp | L3 | Opus | 7 | 17 | ✓ | ? | ? |  |
| [20261118-153038-exp-L2b](runs/20261118-153038-exp-L2b/) | v1 | exp | L2b | Haiku | 38 | 61 | ✗ | ? | ? |  |
| [20261118-161040-exp-L2b](runs/20261118-161040-exp-L2b/) | v1 | exp | L2b | Haiku | 11 | 15 | ✗ | ? | ? |  |
| [20261118-162323-exp-L2b](runs/20261118-162323-exp-L2b/) | v1 | exp | L2b | Haiku | 18 | 27 | ✗ | ? | ? |  |
| [20261118-164334-exp-L2b](runs/20261118-164334-exp-L2b/) | v1 | exp | L2b | Sonnet | 2 | 4 | ✗ | ? | ? |  |
| [20261118-164731-exp-L2b](runs/20261118-164731-exp-L2b/) | v1 | exp | L2b | Sonnet | 2 | 3 | ✗ | ? | ? |  |
| [20261118-165114-exp-L2b](runs/20261118-165114-exp-L2b/) | v1 | exp | L2b | Sonnet | 2 | 3 | ✗ | ? | ? |  |
| [20261118-165509-exp-L2b](runs/20261118-165509-exp-L2b/) | v1 | exp | L2b | Opus | 12 | 23 | ✗ | ? | ? |  |
| [20261118-170849-exp-L2b](runs/20261118-170849-exp-L2b/) | v1 | exp | L2b | Opus | 10 | 14 | ✓ | ? | ? |  |
| [20261118-172123-exp-L2b](runs/20261118-172123-exp-L2b/) | v1 | exp | L2b | Opus | 17 | 19 | ✓ | ? | ? |  |
| [20261118-180017-exp-L2a](runs/20261118-180017-exp-L2a/) | v1 | exp | L2a | Haiku | 15 | 19 | ✓ | ? | ? |  |
| [20261118-181722-exp-L2a](runs/20261118-181722-exp-L2a/) | v1 | exp | L2a | Haiku | 38 | 21 | ✗ | ? | ? |  |
| [20261118-185704-exp-L2a](runs/20261118-185704-exp-L2a/) | v1 | exp | L2a | Haiku | 22 | 22 | ✗ | ? | ? |  |
| [20261118-192112-exp-L2a](runs/20261118-192112-exp-L2a/) | v1 | exp | L2a | Sonnet | 5 | 9 | ✗ | ? | ? |  |
| [20261118-192800-exp-L2a](runs/20261118-192800-exp-L2a/) | v1 | exp | L2a | Sonnet | 2 | 3 | ✗ | ? | ? |  |
| [20261118-193142-exp-L2a](runs/20261118-193142-exp-L2a/) | v1 | exp | L2a | Sonnet | 1 | 5 | ✗ | ? | ? |  |
| [20261118-193441-exp-L2a](runs/20261118-193441-exp-L2a/) | v1 | exp | L2a | Opus | 18 | 24 | ✓ | ? | ? |  |
| [20261118-195530-exp-L2a](runs/20261118-195530-exp-L2a/) | v1 | exp | L2a | Opus | 18 | 24 | ✗ | ? | ? |  |
| [20261118-201507-exp-L2a](runs/20261118-201507-exp-L2a/) | v1 | exp | L2a | Opus | 18 | 27 | ✗ | ? | ? |  |
| [20261118-211120-exp-L1](runs/20261118-211120-exp-L1/) | v1 | exp | L1 | Haiku | 28 | 22 | ✗ | ? | ? |  |
| [20261118-214056-exp-L1](runs/20261118-214056-exp-L1/) | v1 | exp | L1 | Haiku | 18 | 24 | ✗ | ? | ? |  |
| [20261118-220047-exp-L1](runs/20261118-220047-exp-L1/) | v1 | exp | L1 | Haiku | 12 | 20 | ✗ | ? | ? |  |
| [20261118-221758-exp-L1](runs/20261118-221758-exp-L1/) | v1 | exp | L1 | Sonnet | 4 | 8 | ✗ | ? | ? |  |
| [20261118-222330-exp-L1](runs/20261118-222330-exp-L1/) | v1 | exp | L1 | Sonnet | 4 | 7 | ✗ | ? | ? |  |
| [20261118-222902-exp-L1](runs/20261118-222902-exp-L1/) | v1 | exp | L1 | Sonnet | 8 | 9 | ✗ | ? | ? |  |
| [20261118-223834-exp-L1](runs/20261118-223834-exp-L1/) | v1 | exp | L1 | Opus | 16 | 21 | ✓ | ? | ? |  |
| [20261118-225630-exp-L1](runs/20261118-225630-exp-L1/) | v1 | exp | L1 | Opus | 11 | 18 | ✓ | ? | ? |  |
| [20261118-230916-exp-L1](runs/20261118-230916-exp-L1/) | v1 | exp | L1 | Opus | 18 | 23 | ✓ | ? | ? |  |
| [20261119-081907-exp-L1](runs/20261119-081907-exp-L1/) | v2 | exp | L1 | Haiku | 20 | 21 | ✗ | ✓ | ✗ |  |
| [20261119-084106-exp-L1](runs/20261119-084106-exp-L1/) | v2 | exp | L1 | Haiku | 38 | 26 | ✓ | ✓ | ✓ |  |
| [20261119-092039-exp-L1](runs/20261119-092039-exp-L1/) | v2 | exp | L1 | Haiku | 28 | 29 | ✗ | ✗ | ✗ | importance_weights |
| [20261119-095015-exp-L1](runs/20261119-095015-exp-L1/) | v2 | exp | L1 | Sonnet | 5 | 8 | ✗ | ✗ | ✗ | importance_weights |
| [20261119-095730-exp-L1](runs/20261119-095730-exp-L1/) | v2 | exp | L1 | Sonnet | 19 | 13 | ✗ | ✗ | ✗ | importance_weights |
| [20261119-101754-exp-L1](runs/20261119-101754-exp-L1/) | v2 | exp | L1 | Sonnet | 6 | 8 | ✗ | ✗ | ✗ | importance_weights |
| [20261119-102609-exp-L1](runs/20261119-102609-exp-L1/) | v2 | exp | L1 | Opus | 6 | 20 | ✓ | ✓ | ✓ |  |
| [20261119-103415-exp-L1](runs/20261119-103415-exp-L1/) | v2 | exp | L1 | Opus | 14 | 25 | ✓ | ✓ | ✓ |  |
| [20261119-104955-exp-L1](runs/20261119-104955-exp-L1/) | v2 | exp | L1 | Opus | 10 | 15 | ✓ | ✓ | ✓ |  |
| [20261119-110224-exp-L2a](runs/20261119-110224-exp-L2a/) | v2 | exp | L2a | Haiku | 52 | 16 | ✗ | ✓ | ✗ |  |
| [20261119-115626-exp-L2a](runs/20261119-115626-exp-L2a/) | v2 | exp | L2a | Haiku | 9 | 19 | ✗ | ✓ | ✗ |  |
| [20261119-120713-exp-L2a](runs/20261119-120713-exp-L2a/) | v2 | exp | L2a | Haiku | 12 | 16 | ✗ | ✓ | ✗ |  |
| [20261119-122042-exp-L2a](runs/20261119-122042-exp-L2a/) | v2 | exp | L2a | Sonnet | 13 | 10 | ✗ | ✗ | ✗ | reward_baseline |
| [20261119-123601-exp-L2a](runs/20261119-123601-exp-L2a/) | v2 | exp | L2a | Sonnet | 1 | 4 | ✗ | ✗ | ✗ | reward_baseline |
| [20261119-123853-exp-L2a](runs/20261119-123853-exp-L2a/) | v2 | exp | L2a | Sonnet | 4 | 9 | ✓ | ✓ | ✓ |  |
| [20261119-124431-exp-L2a](runs/20261119-124431-exp-L2a/) | v2 | exp | L2a | Opus | 29 | 22 | ✓ | ✓ | ✓ |  |
| [20261119-131624-exp-L2a](runs/20261119-131624-exp-L2a/) | v2 | exp | L2a | Opus | 20 | 30 | ✓ | ✓ | ✓ |  |
| [20261119-133853-exp-L2a](runs/20261119-133853-exp-L2a/) | v2 | exp | L2a | Opus | 11 | 19 | ✓ | ✓ | ✓ |  |
| [20261119-135225-exp-L2b](runs/20261119-135225-exp-L2b/) | v2 | exp | L2b | Haiku | 7 | 17 | ✗ | ✓ | ✗ |  |
| [20261119-140145-exp-L2b](runs/20261119-140145-exp-L2b/) | v2 | exp | L2b | Haiku | 6 | 15 | ✗ | ✓ | ✗ |  |
| [20261119-141003-exp-L2b](runs/20261119-141003-exp-L2b/) | v2 | exp | L2b | Haiku | 29 | 33 | ✗ | ✓ | ✗ |  |
| [20261119-144130-exp-L2b](runs/20261119-144130-exp-L2b/) | v2 | exp | L2b | Sonnet | 1 | 3 | ✗ | ✗ | ✗ | importance_weights |
| [20261119-144418-exp-L2b](runs/20261119-144418-exp-L2b/) | v2 | exp | L2b | Sonnet | 1 | 6 | ✗ | ✗ | ✗ | importance_weights |
| [20261119-144726-exp-L2b](runs/20261119-144726-exp-L2b/) | v2 | exp | L2b | Sonnet | 2 | 3 | ✗ | ✗ | ✗ | importance_weights |
| [20261119-145052-exp-L2b](runs/20261119-145052-exp-L2b/) | v2 | exp | L2b | Opus | 47 | 20 | ✓ | ✓ | ✓ |  |
| [20261119-162635-exp-L2b](runs/20261119-162635-exp-L2b/) | v2 | exp | L2b | Opus | 12 | 16 | ✓ | ✓ | ✓ |  |
| [20261119-164144-exp-L2b](runs/20261119-164144-exp-L2b/) | v2 | exp | L2b | Opus | 7 | 15 | ✓ | ✓ | ✓ |  |
| [20261120-085258-exp-L1](runs/20261120-085258-exp-L1/) | v2 | exp | L1 | Opus | 12 | 14 | ✓ | ✓ | ✓ |  |
| [20261120-085344-exp-L1](runs/20261120-085344-exp-L1/) | v2 | exp | L1 | Haiku | 15 | 28 | ✗ | ✓ | ✗ |  |
| [20261120-085358-exp-L1](runs/20261120-085358-exp-L1/) | v2 | exp | L1 | Sonnet | 26 | 10 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-090650-exp-L1](runs/20261120-090650-exp-L1/) | v2 | exp | L1 | Opus | 6 | 14 | ✓ | ✓ | ✓ |  |
| [20261120-091044-exp-L1](runs/20261120-091044-exp-L1/) | v2 | exp | L1 | Haiku | 14 | 27 | ✗ | ✓ | ✗ |  |
| [20261120-091510-exp-L1](runs/20261120-091510-exp-L1/) | v2 | exp | L1 | Opus | 18 | 22 | ✓ | ✓ | ✓ |  |
| [20261120-092158-exp-L1](runs/20261120-092158-exp-L1/) | v2 | exp | L1 | Sonnet | 11 | 12 | ✓ | ✓ | ✓ |  |
| [20261120-092549-exp-L1](runs/20261120-092549-exp-L1/) | v2 | exp | L1 | Haiku | 19 | 23 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-093514-exp-L1](runs/20261120-093514-exp-L1/) | v2 | exp | L1 | Sonnet | 8 | 14 | ✓ | ✓ | ✓ |  |
| [20261120-093529-exp-L2a](runs/20261120-093529-exp-L2a/) | v2 | exp | L2a | Opus | 21 | 23 | ✓ | ✓ | ✓ |  |
| [20261120-094603-exp-L2a](runs/20261120-094603-exp-L2a/) | v2 | exp | L2a | Sonnet | 12 | 13 | ✗ | ✗ | ✗ | reward_baseline |
| [20261120-094610-exp-L2a](runs/20261120-094610-exp-L2a/) | v2 | exp | L2a | Haiku | 28 | 32 | ✓ | ✓ | ✓ |  |
| [20261120-095832-exp-L2a](runs/20261120-095832-exp-L2a/) | v2 | exp | L2a | Opus | 7 | 16 | ✓ | ✓ | ✓ |  |
| [20261120-095948-exp-L2a](runs/20261120-095948-exp-L2a/) | v2 | exp | L2a | Sonnet | 7 | 7 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-100749-exp-L2a](runs/20261120-100749-exp-L2a/) | v2 | exp | L2a | Opus | 12 | 17 | ✓ | ✓ | ✓ |  |
| [20261120-100847-exp-L2a](runs/20261120-100847-exp-L2a/) | v2 | exp | L2a | Sonnet | 3 | 10 | ✓ | ✓ | ✓ |  |
| [20261120-101418-exp-L2b](runs/20261120-101418-exp-L2b/) | v2 | exp | L2b | Sonnet | 1 | 3 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-101608-exp-L2a](runs/20261120-101608-exp-L2a/) | v2 | exp | L2a | Haiku | 16 | 18 | ✗ | ✓ | ✗ |  |
| [20261120-101731-exp-L2b](runs/20261120-101731-exp-L2b/) | v2 | exp | L2b | Sonnet | 1 | 3 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-102028-exp-L2b](runs/20261120-102028-exp-L2b/) | v2 | exp | L2b | Sonnet | 2 | 4 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-102139-exp-L2b](runs/20261120-102139-exp-L2b/) | v2 | exp | L2b | Opus | 9 | 25 | ✓ | ✓ | ✓ |  |
| [20261120-103212-exp-L2b](runs/20261120-103212-exp-L2b/) | v2 | exp | L2b | Opus | 8 | 18 | ✓ | ✓ | ✓ |  |
| [20261120-103324-exp-L2a](runs/20261120-103324-exp-L2a/) | v2 | exp | L2a | Haiku | 7 | 31 | ✗ | ✓ | ✗ |  |
| [20261120-104143-exp-L2b](runs/20261120-104143-exp-L2b/) | v2 | exp | L2b | Opus | 22 | 20 | ✓ | ✓ | ✓ |  |
| [20261120-104202-exp-L2b](runs/20261120-104202-exp-L2b/) | v2 | exp | L2b | Haiku | 10 | 27 | ✗ | ✓ | ✗ |  |
| [20261120-105325-exp-L2b](runs/20261120-105325-exp-L2b/) | v2 | exp | L2b | Haiku | 9 | 20 | ✗ | ✓ | ✗ |  |
| [20261120-110424-exp-L2b](runs/20261120-110424-exp-L2b/) | v2 | exp | L2b | Haiku | 16 | 19 | ✗ | ✓ | ✗ |  |
| [20261120-115601-exp-L1](runs/20261120-115601-exp-L1/) | v2 | exp | L1 | Opus | 10 | 17 | ✓ | ✓ | ✓ |  |
| [20261120-115628-exp-L1](runs/20261120-115628-exp-L1/) | v2 | exp | L1 | Sonnet | 11 | 13 | ✗ | ✓ | ✗ |  |
| [20261120-115646-exp-L1](runs/20261120-115646-exp-L1/) | v2 | exp | L1 | Haiku | 20 | 21 | ✗ | ✓ | ✗ |  |
| [20261120-120800-exp-L1](runs/20261120-120800-exp-L1/) | v2 | exp | L1 | Opus | 15 | 15 | ✗ | ✓ | ✗ |  |
| [20261120-120907-exp-L1](runs/20261120-120907-exp-L1/) | v2 | exp | L1 | Sonnet | 33 | 16 | ✓ | ✓ | ✓ |  |
| [20261120-121911-exp-L1](runs/20261120-121911-exp-L1/) | v2 | exp | L1 | Haiku | 12 | 22 | ✗ | ✓ | ✗ |  |
| [20261120-122426-exp-L1](runs/20261120-122426-exp-L1/) | v2 | exp | L1 | Opus | 14 | 22 | ✓ | ✓ | ✓ |  |
| [20261120-123307-exp-L1](runs/20261120-123307-exp-L1/) | v2 | exp | L1 | Haiku | 15 | 28 | ✗ | ✓ | ✗ |  |
| [20261120-124027-exp-L2a](runs/20261120-124027-exp-L2a/) | v2 | exp | L2a | Opus | 12 | 13 | ✓ | ✓ | ✓ |  |
| [20261120-124431-exp-L1](runs/20261120-124431-exp-L1/) | v2 | exp | L1 | Sonnet | 22 | 11 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-124945-exp-L2a](runs/20261120-124945-exp-L2a/) | v2 | exp | L2a | Haiku | 10 | 10 | ✗ | ✓ | ✗ |  |
| [20261120-125430-exp-L2a](runs/20261120-125430-exp-L2a/) | v2 | exp | L2a | Opus | 7 | 18 | ✓ | ✓ | ✓ |  |
| [20261120-130230-exp-L2a](runs/20261120-130230-exp-L2a/) | v2 | exp | L2a | Haiku | 20 | 20 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-130253-exp-L2a](runs/20261120-130253-exp-L2a/) | v2 | exp | L2a | Opus | 9 | 14 | ✓ | ✓ | ✓ |  |
| [20261120-130841-exp-L2a](runs/20261120-130841-exp-L2a/) | v2 | exp | L2a | Sonnet | 3 | 5 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-131308-exp-L2a](runs/20261120-131308-exp-L2a/) | v2 | exp | L2a | Sonnet | 2 | 4 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-131317-exp-L2b](runs/20261120-131317-exp-L2b/) | v2 | exp | L2b | Opus | 12 | 14 | ✓ | ✓ | ✓ |  |
| [20261120-131629-exp-L2a](runs/20261120-131629-exp-L2a/) | v2 | exp | L2a | Sonnet | 6 | 11 | ✓ | ✓ | ✓ |  |
| [20261120-132436-exp-L2b](runs/20261120-132436-exp-L2b/) | v2 | exp | L2b | Sonnet | 2 | 4 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-132501-exp-L2a](runs/20261120-132501-exp-L2a/) | v2 | exp | L2a | Haiku | 12 | 17 | ✗ | ✓ | ✗ |  |
| [20261120-132648-exp-L2b](runs/20261120-132648-exp-L2b/) | v2 | exp | L2b | Opus | 12 | 16 | ✓ | ✓ | ✓ |  |
| [20261120-132750-exp-L2b](runs/20261120-132750-exp-L2b/) | v2 | exp | L2b | Sonnet | 1 | 3 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-133020-exp-L2b](runs/20261120-133020-exp-L2b/) | v2 | exp | L2b | Sonnet | 2 | 5 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-133916-exp-L2b](runs/20261120-133916-exp-L2b/) | v2 | exp | L2b | Haiku | 16 | 26 | ✗ | ✗ | ✗ | importance_weights |
| [20261120-134030-exp-L2b](runs/20261120-134030-exp-L2b/) | v2 | exp | L2b | Opus | 9 | 17 | ✓ | ✓ | ✓ |  |
| [20261120-135655-exp-L2b](runs/20261120-135655-exp-L2b/) | v2 | exp | L2b | Haiku | 14 | 23 | ✗ | ✓ | ✗ |  |
| [20261120-141231-exp-L2b](runs/20261120-141231-exp-L2b/) | v2 | exp | L2b | Haiku | 14 | 17 | ✗ | ✓ | ✗ |  |
