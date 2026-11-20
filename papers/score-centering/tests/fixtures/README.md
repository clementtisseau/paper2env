`haiku_L3_losses.py`: the `losses.py` from the first L3 run
(`runs/20261118-112015-exp-L3/`). It centers the score but truncates the sum
to the sampler's top-k head without modeling the tail, so it is biased even
on-policy: the regression case for `grader/check.py` with the top-k.
