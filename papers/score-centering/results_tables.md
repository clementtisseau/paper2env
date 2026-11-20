## By level and model

Each cell: runs passing / runs. `correct`: the four exact checks pass with the sampler's top-k; `idea`: the judge finds the method in the loss; `success`: both.

### Starter v1, exp

| Level | Model | correct | idea | success | median minutes |
|---|---|---|---|---|---|
| L4 | Haiku | 1/1 | ? | ? | 7 |
| L3 | Haiku | 0/3 | ? | ? | 11 |
| L3 | Sonnet | 2/3 | ? | ? | 2 |
| L3 | Opus | 3/3 | ? | ? | 8 |
| L2b | Haiku | 0/3 | ? | ? | 18 |
| L2b | Sonnet | 0/3 | ? | ? | 2 |
| L2b | Opus | 2/3 | ? | ? | 12 |
| L2a | Haiku | 1/3 | ? | ? | 22 |
| L2a | Sonnet | 0/3 | ? | ? | 2 |
| L2a | Opus | 1/3 | ? | ? | 18 |
| L1 | Haiku | 0/3 | ? | ? | 18 |
| L1 | Sonnet | 0/3 | ? | ? | 4 |
| L1 | Opus | 3/3 | ? | ? | 16 |

### Starter v2, exp

| Level | Model | correct | idea | success | median minutes |
|---|---|---|---|---|---|
| L2b | Haiku | 0/3 | ? | ? | 7 |
| L2b | Sonnet | 0/3 | ? | ? | 1 |
| L2b | Opus | 3/3 | ? | ? | 12 |
| L2a | Haiku | 0/3 | ? | ? | 12 |
| L2a | Sonnet | 1/3 | ? | ? | 4 |
| L2a | Opus | 3/3 | ? | ? | 20 |
| L1 | Haiku | 1/3 | ? | ? | 28 |
| L1 | Sonnet | 0/3 | ? | ? | 6 |
| L1 | Opus | 3/3 | ? | ? | 10 |

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
| [20261119-081907-exp-L1](runs/20261119-081907-exp-L1/) | v2 | exp | L1 | Haiku | 20 | 21 | ✗ | ? | ? |  |
| [20261119-084106-exp-L1](runs/20261119-084106-exp-L1/) | v2 | exp | L1 | Haiku | 38 | 26 | ✓ | ? | ? |  |
| [20261119-092039-exp-L1](runs/20261119-092039-exp-L1/) | v2 | exp | L1 | Haiku | 28 | 29 | ✗ | ? | ? |  |
| [20261119-095015-exp-L1](runs/20261119-095015-exp-L1/) | v2 | exp | L1 | Sonnet | 5 | 8 | ✗ | ? | ? |  |
| [20261119-095730-exp-L1](runs/20261119-095730-exp-L1/) | v2 | exp | L1 | Sonnet | 19 | 13 | ✗ | ? | ? |  |
| [20261119-101754-exp-L1](runs/20261119-101754-exp-L1/) | v2 | exp | L1 | Sonnet | 6 | 8 | ✗ | ? | ? |  |
| [20261119-102609-exp-L1](runs/20261119-102609-exp-L1/) | v2 | exp | L1 | Opus | 6 | 20 | ✓ | ? | ? |  |
| [20261119-103415-exp-L1](runs/20261119-103415-exp-L1/) | v2 | exp | L1 | Opus | 14 | 25 | ✓ | ? | ? |  |
| [20261119-104955-exp-L1](runs/20261119-104955-exp-L1/) | v2 | exp | L1 | Opus | 10 | 15 | ✓ | ? | ? |  |
| [20261119-110224-exp-L2a](runs/20261119-110224-exp-L2a/) | v2 | exp | L2a | Haiku | 52 | 16 | ✗ | ? | ? |  |
| [20261119-115626-exp-L2a](runs/20261119-115626-exp-L2a/) | v2 | exp | L2a | Haiku | 9 | 19 | ✗ | ? | ? |  |
| [20261119-120713-exp-L2a](runs/20261119-120713-exp-L2a/) | v2 | exp | L2a | Haiku | 12 | 16 | ✗ | ? | ? |  |
| [20261119-122042-exp-L2a](runs/20261119-122042-exp-L2a/) | v2 | exp | L2a | Sonnet | 13 | 10 | ✗ | ? | ? |  |
| [20261119-123601-exp-L2a](runs/20261119-123601-exp-L2a/) | v2 | exp | L2a | Sonnet | 1 | 4 | ✗ | ? | ? |  |
| [20261119-123853-exp-L2a](runs/20261119-123853-exp-L2a/) | v2 | exp | L2a | Sonnet | 4 | 9 | ✓ | ? | ? |  |
| [20261119-124431-exp-L2a](runs/20261119-124431-exp-L2a/) | v2 | exp | L2a | Opus | 29 | 22 | ✓ | ? | ? |  |
| [20261119-131624-exp-L2a](runs/20261119-131624-exp-L2a/) | v2 | exp | L2a | Opus | 20 | 30 | ✓ | ? | ? |  |
| [20261119-133853-exp-L2a](runs/20261119-133853-exp-L2a/) | v2 | exp | L2a | Opus | 11 | 19 | ✓ | ? | ? |  |
| [20261119-135225-exp-L2b](runs/20261119-135225-exp-L2b/) | v2 | exp | L2b | Haiku | 7 | 17 | ✗ | ? | ? |  |
| [20261119-140145-exp-L2b](runs/20261119-140145-exp-L2b/) | v2 | exp | L2b | Haiku | 6 | 15 | ✗ | ? | ? |  |
| [20261119-141003-exp-L2b](runs/20261119-141003-exp-L2b/) | v2 | exp | L2b | Haiku | 29 | 33 | ✗ | ? | ? |  |
| [20261119-144130-exp-L2b](runs/20261119-144130-exp-L2b/) | v2 | exp | L2b | Sonnet | 1 | 3 | ✗ | ? | ? |  |
| [20261119-144418-exp-L2b](runs/20261119-144418-exp-L2b/) | v2 | exp | L2b | Sonnet | 1 | 6 | ✗ | ? | ? |  |
| [20261119-144726-exp-L2b](runs/20261119-144726-exp-L2b/) | v2 | exp | L2b | Sonnet | 2 | 3 | ✗ | ? | ? |  |
| [20261119-145052-exp-L2b](runs/20261119-145052-exp-L2b/) | v2 | exp | L2b | Opus | 47 | 20 | ✓ | ? | ? |  |
| [20261119-162635-exp-L2b](runs/20261119-162635-exp-L2b/) | v2 | exp | L2b | Opus | 12 | 16 | ✓ | ? | ? |  |
| [20261119-164144-exp-L2b](runs/20261119-164144-exp-L2b/) | v2 | exp | L2b | Opus | 7 | 15 | ✓ | ? | ? |  |
