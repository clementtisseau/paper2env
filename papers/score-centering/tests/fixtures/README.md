`haiku_L4_losses.py`: the `losses.py` from the first Haiku run at L4
(`runs/20261118-102641-exp-L4/`). Correct without an IS correction, but it
centers the score under q even when the sampled token carries a weight
w(p/q): the regression case for the theory check's TIS scenario.

`haiku_L3_losses.py`: the `losses.py` from the first L3 run
(`runs/20261118-112015-exp-L3/`). Correct with the full distribution, but it
truncates the sum to the top-k head without modeling the tail: the
regression case for `passed_topk`.
