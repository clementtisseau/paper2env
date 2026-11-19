"""Tables of all runs, from runs/index.json and each run's grading.

Runs listed in runs/index.json take their model and starter version from
it; any other run directory is picked up too, with its model read from its
transcript and its level and mode from meta.json (starter "v2", the current
one). For each run it uses runs/<run>/regraded.json when present (the theory check
re-run with the current grader, so early runs get the top-k and TIS
scenarios too), otherwise the theory result recorded at the end of the run.
The guard-rail is the one recorded at the end of the run.

    python scripts/summarize_runs.py > docs/results_tables.md
"""
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"
LEVELS = ["L4", "L3", "L2b", "L2a", "L1", "L0"]
MODELS = ["Haiku", "Sonnet", "Opus"]
CHECKS = ("no_drift", "no_drift_prefix", "on_policy", "bounded")


def load(run):
    results = json.loads((RUNS / run / "results.json").read_text())
    regraded = RUNS / run / "regraded.json"
    theory = json.loads(regraded.read_text())["theory"] if regraded.exists() else results.get("theory", {})
    return results, theory


def mechanism(theory):
    """What the theory check says the submitted loss does, from its best scenario."""
    scenarios = [s for s in theory.get("scenarios", {}).values() if "checks" in s]
    if not scenarios:
        return "crashes"
    best = max(scenarios, key=lambda s: (s["passed"], sum(s["checks"].values())))
    c = best["checks"]
    if best["passed"]:
        return "drift-free"
    if not c["on_policy"]:
        return "changes on-policy update"
    if c["no_drift"] and c["no_drift_prefix"] and not c["bounded"]:
        return "unbounded weights"
    if c["no_drift"] and not c["no_drift_prefix"]:
        return "global baseline only"
    if c["no_drift_prefix"] and not c["no_drift"]:
        return "drift under severe mismatch"
    return "drift remains"


def run_info(run_dir):
    """Model, level, mode and starter version of a run missing from index.json."""
    meta = json.loads((run_dir / "meta.json").read_text())
    served = None
    for prefix in run_dir.glob("transcripts/*.prefix.json"):
        m = re.search(r'"model":"claude-(\w+)-', prefix.read_text())
        served = m.group(1).capitalize() if m else served
    return {"model": served or "?", "level": meta["level"], "experiments": meta["experiments"], "era": "v2"}


def rows():
    index = json.loads((RUNS / "index.json").read_text())
    for run_dir in sorted(p for p in RUNS.iterdir() if (p / "meta.json").exists()):
        index.setdefault(run_dir.name, run_info(run_dir))
    out = []
    for run, info in sorted(index.items()):
        results, theory = load(run)
        guard = results.get("guard", {})
        out.append({"run": run, **info, "minutes": results.get("minutes"), "commands": results.get("agent_commands"),
                    "theory": bool(theory.get("passed")), "topk": bool(theory.get("passed_topk")),
                    "tis": bool(theory.get("passed_with_tis")), "guard": guard.get("normalized"),
                    "guard_pass": bool(guard.get("passed")), "mechanism": mechanism(theory)})
    return out


def frac(xs):
    return f"{sum(xs)}/{len(xs)}" if xs else "–"


def main():
    rs = rows()
    by = defaultdict(list)
    for r in rs:
        by[(r["era"], r["experiments"], r["level"], r["model"])].append(r)
    groups = sorted({(r["era"], r["experiments"]) for r in rs}, key=lambda g: (g[0], not g[1]))
    print("## Pass counts by level and model\n")
    print("Each cell: runs passing / runs. `theory` = drift-free with the full distribution or the top-k; "
          "`top-k` = drift-free with the sampler's top-k only (what an inference engine gives); "
          "`+TIS` = still drift-free when combined with truncated importance sampling.\n")
    for era, experiments in groups:
        print(f"### Starter {era}, {'exp' if experiments else 'noexp'}\n")
        print("| Level | Model | theory | top-k | +TIS | guard-rail | median minutes |")
        print("|---|---|---|---|---|---|---|")
        for level in LEVELS:
            for model in MODELS:
                g = by.get((era, experiments, level, model))
                if not g:
                    continue
                minutes = statistics.median(r["minutes"] for r in g)
                print(f"| {level} | {model} | {frac([r['theory'] for r in g])} | {frac([r['topk'] for r in g])} | "
                      f"{frac([r['tis'] for r in g])} | {frac([r['guard_pass'] for r in g])} | {minutes:.0f} |")
        print()
    print("## What the submitted losses do (theory check, best scenario)\n")
    mechs = sorted({r["mechanism"] for r in rs})
    print("| Level | Model | " + " | ".join(mechs) + " |")
    print("|---|---|" + "---|" * len(mechs))
    for level in LEVELS:
        for model in MODELS:
            g = [r for r in rs if r["level"] == level and r["model"] == model]
            if g:
                print(f"| {level} | {model} | " + " | ".join(str(sum(r["mechanism"] == m for r in g)) for m in mechs) + " |")
    print("\n## Every run\n")
    print("| Run | Starter | Mode | Level | Model | min | cmds | theory | top-k | +TIS | guard | mechanism |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    yes = lambda b: "✓" if b else "✗"
    for r in rs:
        guard = "–" if r["guard"] is None else f"{r['guard']:.2f}"
        mode = "exp" if r["experiments"] else "noexp"
        print(f"| [{r['run']}](../runs/{r['run']}/) | {r['era']} | {mode} | {r['level']} | {r['model']} | {r['minutes']:.0f} | "
              f"{r['commands']} | {yes(r['theory'])} | {yes(r['topk'])} | {yes(r['tis'])} | {guard} | {r['mechanism']} |")


if __name__ == "__main__":
    main()
