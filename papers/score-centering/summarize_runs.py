"""Tables of all runs, from runs/index.json, each run's meta.json and grade.json.

index.json gives each run's model and starter version (v1: the first starter,
whose sampler distribution sat behind a setting the L3 prompt named; v2: the
current one, used for the results). Columns: `correct` (the exact checks,
grader/check.py), `idea` (the judge, grader/judge.md; "?" until judged) and
`success` (both), with its 95% Wilson interval.

    python papers/score-centering/summarize_runs.py > papers/score-centering/results_tables.md
"""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "runs"
LEVELS = ["L4", "L3", "L2b", "L2a", "L1", "L0"]
MODELS = ["Haiku", "Sonnet", "Opus"]


def rows():
    index = json.loads((RUNS / "index.json").read_text())
    out = []
    for run, info in sorted(index.items()):
        meta = json.loads((RUNS / run / "meta.json").read_text())
        grade = json.loads((RUNS / run / "grade.json").read_text())
        judge = grade["judge"] or {}
        out.append({"run": run, **info, "minutes": meta["minutes"], "commands": meta["agent_commands"],
                    "correct": grade["correct"], "idea": grade["idea_found"], "success": grade["success"],
                    "wrong_answer": judge.get("wrong_answer")})
    return out


def wilson(k, n, z=1.96):
    """95% Wilson score interval for k successes out of n runs."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def interval(rs, key):
    """The 95% interval of a count, as "lo–hi%"."""
    if any(r[key] is None for r in rs):
        return "?"
    lo, hi = wilson(sum(r[key] for r in rs), len(rs))
    return f"{100 * lo:.0f}–{100 * hi:.0f}%"


def count(rs, key):
    """passing / runs, or "?" while any run is not judged yet."""
    if any(r[key] is None for r in rs):
        return "?"
    return f"{sum(r[key] for r in rs)}/{len(rs)}"


def main():
    rs = rows()
    by = defaultdict(list)
    for r in rs:
        by[(r["era"], r["experiments"], r["level"], r["model"])].append(r)
    print("## By level and model\n")
    print("Each cell: runs passing / runs. `correct`: the four exact checks pass with the sampler's top-k; "
          "`idea`: the judge finds the method in the loss; `success`: both, with its 95% Wilson interval.\n")
    for era, experiments in sorted({(r["era"], r["experiments"]) for r in rs}, key=lambda g: (g[0], not g[1])):
        print(f"### Starter {era}, {'exp' if experiments else 'noexp'}\n")
        print("| Level | Model | correct | idea | success | success, 95% CI | median minutes |")
        print("|---|---|---|---|---|---|---|")
        for level in LEVELS:
            for model in MODELS:
                g = by.get((era, experiments, level, model))
                if g:
                    minutes = statistics.median(r["minutes"] for r in g)
                    print(f"| {level} | {model} | {count(g, 'correct')} | {count(g, 'idea')} | "
                          f"{count(g, 'success')} | {interval(g, 'success')} | {minutes:.0f} |")
        print()
    print("## Every run\n")
    print("| Run | Starter | Mode | Level | Model | min | cmds | correct | idea | success | judge: instead |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    mark = lambda b: "?" if b is None else ("✓" if b else "✗")
    for r in rs:
        mode = "exp" if r["experiments"] else "noexp"
        print(f"| [{r['run']}](runs/{r['run']}/) | {r['era']} | {mode} | {r['level']} | {r['model']} | "
              f"{r['minutes']:.0f} | {r['commands']} | {mark(r['correct'])} | {mark(r['idea'])} | "
              f"{mark(r['success'])} | {r['wrong_answer'] or ''} |")


if __name__ == "__main__":
    main()
