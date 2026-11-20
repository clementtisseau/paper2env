"""A run's grade: runs/<run>/grade.json.

    {"correct": bool,          every grader of the paper reports the submission correct
     "idea_found": bool|null,  the judge's majority verdict (null: not judged yet)
     "success": bool|null,     correct and idea_found (just correct for a paper without a judge)
     "graders": {name: output of the grader script},
     "judge": {"idea_found", "wrong_answer", "votes"} | null}

core/runner/run.py writes it at the end of a run; the judge's verdict is
added later with set_judge(). To re-grade runs after a grader changed:

    python core/grade.py --paper score-centering [run ...]
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import images  # noqa: E402
import judge as judge_lib  # noqa: E402
import paper as papers  # noqa: E402


def _write(paper, run_dir, graders, judge):
    correct = bool(graders) and all(g.get("correct") is True for g in graders.values())
    if judge_lib.has_judge(paper):
        idea_found = None if judge is None else judge["idea_found"]
        success = None if idea_found is None else correct and idea_found
    else:
        idea_found, success = None, correct
    grade = {"correct": correct, "idea_found": idea_found, "success": success, "graders": graders, "judge": judge}
    (run_dir / "grade.json").write_text(json.dumps(grade, indent=2) + "\n")
    return grade


def load(run_dir):
    path = Path(run_dir) / "grade.json"
    return json.loads(path.read_text()) if path.exists() else None


def grade(paper, run_dir):
    """Run the paper's graders on the run's submission and write grade.json,
    keeping a judge verdict already there."""
    run_dir = Path(run_dir)
    sub = run_dir / paper.SUBMISSION
    graders = images.run_graders(paper, sub.read_text()) if sub.exists() and sub.read_text() else {}
    old = load(run_dir)
    return _write(paper, run_dir, graders, old["judge"] if old else None)


def set_judge(paper, run_dir, answers):
    """Record the judges' answers (already validated) for a graded run."""
    run_dir = Path(run_dir)
    return _write(paper, run_dir, load(run_dir)["graders"], judge_lib.majority(answers))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Re-run a paper's graders on its runs and rewrite grade.json.")
    ap.add_argument("--paper", required=True, choices=papers.available())
    ap.add_argument("runs", nargs="*", help="run directory names (default: every run)")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    paper = papers.load(a.paper)
    root = paper.DIR / "runs"
    dirs = [root / r for r in a.runs] or sorted(d for d in root.iterdir() if (d / "meta.json").exists())
    with ThreadPoolExecutor(a.jobs) as ex:
        for d, g in zip(dirs, ex.map(lambda d: grade(paper, d), dirs)):
            print(d.name, "correct" if g["correct"] else "not correct", flush=True)
