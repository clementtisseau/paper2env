"""score-centering: what this environment declares to core/.

Paper: Score Centering Stabilizes Off-policy Reinforcement Learning
(Marek & Ryabinin, arXiv:2609.20807). The agent gets the paper's JAX RL
framework without the method, and must make RL stable when the sampler that
generates rollouts differs from the trained model. See README.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sanitize_spec  # noqa: E402

NAME = "score-centering"
TITLE = "Score Centering Stabilizes Off-policy Reinforcement Learning"
URL = "https://arxiv.org/abs/2609.20807"

# L0 = prompts/common.md alone; each later level adds prompts/<level>.md
LEVELS = ("L0", "L1", "L2a", "L2b", "L3", "L4")
DEFAULT_LEVELS = ("L2a", "L2b", "L3", "L4")

SUBMISSION = "losses.py"
REPORT = "REPORT.md"

# Scripts run in the grader image (WORKDIR = this directory's copy), on the
# submission mounted at /sub/losses.py. The judge's prompt is grader/judge.md.
GRADERS = [{"name": "check", "args": ["grader/check.py", "/sub/losses.py"], "timeout": 1800}]
# What docker/Dockerfile.grader copies in: the grader image is rebuilt when it changes.
GRADER_INPUTS = ["docker/Dockerfile.grader", "grader/check.py"]

AUDIT = sanitize_spec.AUDIT
ALLOW = sanitize_spec.ALLOW
AUDIT_EXEMPT = sanitize_spec.AUDIT_EXEMPT
write_starter = sanitize_spec.write_starter


# The labels grader/judge.md lets the judge give as wrong_answer (core/judge.py).
JUDGE_LABELS = ("reward_baseline", "expectation_under_p", "wrong_coefficient", "sampled_tokens_only",
                "importance_weights", "kl_to_sampler", "tuning", "other")
