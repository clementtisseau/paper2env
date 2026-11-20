"""grader/check.py accepts score centering and rejects every known wrong answer.

Each submission is a loss_fn(hidden, head, batch). The reference reads the
sampler top-k under its original names (center_ids / center_logprobs), so its
wrapper renames the starter's sampler_topk_* keys.
Run from the repository root: python -m pytest papers/score-centering/tests
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]  # the paper's directory
REPO = ROOT.parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = load(ROOT / "grader/check.py", "grader_check")
ref = load(ROOT / "upstream/losses.py", "ref_losses_check")
C = ref.ISCorrection
RENAME = {"sampler_topk_ids": "center_ids", "sampler_topk_logprobs": "center_logprobs"}


def reference(correction=None, sc=False):
    def loss(hidden, head, batch):
        batch = {RENAME.get(k, k): v for k, v in batch.items()}
        return ref.loss_fn_rl(hidden, head, batch, correction, sc)
    return loss


TIS, MIS = C(high=2.0, outside="clamp"), C(low=0.5, high=5.0)


class ExactGlobalBaseline:
    """GRPO taken to its limit: REINFORCE with the advantage centered by its
    exact mean under the sampler. A real loss cannot know that mean from one
    sample, so this oracle reads it from the check's batch builder."""

    def __init__(self):
        self.mean = None

    def __enter__(self):
        self.make_batch = check.make_batch

        def make_batch(theta_q, advantages):
            w = check.q_weights(theta_q)
            self.mean = (w * advantages).sum() / w.sum()
            return self.make_batch(theta_q, advantages)

        check.make_batch = make_batch
        return self

    def __exit__(self, *exc):
        check.make_batch = self.make_batch

    def __call__(self, hidden, head, batch):
        return check.reinforce(hidden, head, batch | {"advantages": batch["advantages"] - self.mean})


# name: (loss, expected correct, the check that must fail if not correct)
SUBMISSIONS = {
    "score_centering": (reference(None, True), True, None),
    # centering composed with importance weights (under q * w) stays drift-free
    "score_centering+tis": (reference(TIS, True), True, None),
    "score_centering+mis": (reference(MIS, True), True, None),
    "reinforce": (reference(), False, "no_drift"),
    "tis": (reference(TIS), False, "no_drift"),
    "mis": (reference(MIS), False, "no_drift"),
    "naive_is": (reference(C()), False, "bounded"),
    # exact while no ratio exceeds the cap (mild mismatch), drifts once they do
    "loosely_capped_is": (reference(C(high=10.0, outside="clamp")), False, "no_drift"),
    "zero_loss": (lambda h, head, b: (0.0 * head(h).sum(), {}), False, "on_policy"),
    "exact_global_baseline": (None, False, "no_drift_prefix"),
}


def evaluate(name):
    if name == "exact_global_baseline":
        with ExactGlobalBaseline() as loss:
            return check.evaluate(loss)
    return check.evaluate(SUBMISSIONS[name][0])


@pytest.fixture(scope="module")
def results():
    return {name: evaluate(name) for name in SUBMISSIONS}


@pytest.mark.parametrize("name", list(SUBMISSIONS))
def test_verdict(results, name):
    _, correct, failing = SUBMISSIONS[name]
    r = results[name]
    assert r["correct"] == correct, r
    if failing:
        assert not r["checks"][failing], r


def test_global_baseline_cancels_average_drift_only(results):
    """The exact global baseline passes no_drift (constant advantage: it
    subtracts it entirely) but not no_drift_prefix."""
    assert results["exact_global_baseline"]["checks"]["no_drift"]


def test_starter_fails(tmp_path):
    """The loss the agent starts from is not correct."""
    out = tmp_path / "starter"
    subprocess.run([sys.executable, str(REPO / "core/sanitize.py"), "--paper", "score-centering",
                    "--out", str(out), "--no-git"], check=True)
    r = check.check(out / "losses.py")
    assert not r["correct"] and not r["checks"]["no_drift"]


def test_top_k_head_only_fails():
    """A real submission (the losses.py of runs/20261118-112015-exp-L3) that
    truncates the sum to the top-k head without modeling the tail: biased
    even on-policy."""
    r = check.check(ROOT / "tests/fixtures/haiku_L3_losses.py")
    assert not r["correct"] and not r["checks"]["on_policy"]


def test_crash_is_not_correct(tmp_path):
    bad = tmp_path / "losses.py"
    bad.write_text("def loss_fn_rl(hidden, head, batch, correction=None, chunk=128):\n    raise ValueError('x')\n")
    r = check.check(bad)
    assert not r["correct"] and "ValueError" in r["error"]
