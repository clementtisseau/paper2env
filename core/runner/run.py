"""Run a paper's environment with a Claude Code subagent instead of an API model.

A run has three steps; the middle one is done by the Claude Code session that
orchestrates (it launches the subagent with its Agent tool):

  1. python core/runner/run.py start --paper score-centering --level L3 [--no-experiments] [--hours 2]
     Builds the paper's starter and images, starts the agent's container (no
     network, 4 CPUs, 8 GB), marks the run active for hook.py, and writes the
     full prompt to give the subagent: papers/<paper>/runs/<run>/agent_prompt.md.
  2. The session launches the `researcher` subagent with that prompt. While the
     run is active, hook.py rewrites each of the subagent's Bash commands to run
     in the container and denies its other tools.
  3. python core/runner/run.py finish
     Copies the submission and the report out of the container, runs the
     paper's graders in its grader image, writes results.json, then stops the
     container and ends the run. It also copies the subagent's full transcript
     into runs/<run>/transcripts/ (see copy_transcripts).

`python core/runner/run.py abort` ends an active run without grading.
"""
import argparse
import difflib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import compose  # noqa: E402
import images  # noqa: E402
import paper as papers  # noqa: E402

STATE = HERE / "state" / "active.json"

# Harness preamble: what the subagent needs to know about how it works here.
PREAMBLE = """You are a research agent working on the task below.

How your environment works:
- Your only tool is Bash. Every command you run executes inside your own
  Linux workspace, a container whose working directory is /workspace. Other
  tools (file readers, editors, search, web) are disabled.
- Read and edit files with shell commands: cat, sed, heredocs, or python.
- Each Bash call starts a fresh shell in /workspace; `cd` does not persist
  between calls. Long commands may take a timeout of up to 600000 ms.
- When {submission} and {report} in /workspace are final, stop and reply with a
  one-line summary. Your work ends with that reply.

---

"""


def sh(*cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def start(args):
    if STATE.exists():
        sys.exit(f"a run is already active ({STATE}); finish or abort it first")
    paper = papers.load(args.paper)
    if args.level not in paper.LEVELS:
        sys.exit(f"{paper.NAME} levels: {', '.join(paper.LEVELS)}")
    experiments = not args.no_experiments
    starter, tag = images.ensure_images(paper, experiments, rebuild=args.rebuild)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{starter}-{args.level}"
    run_dir = paper.DIR / "runs" / run_id
    run_dir.mkdir(parents=True)
    container = f"p2e-run-{paper.NAME}-{run_id}"
    sh("docker", "run", "-d", "--name", container, "--network", "none", "--cpus", str(args.cpus),
       "--memory", "8g", "--init", tag, "tail", "-f", "/dev/null")
    budget = images.CPU_BUDGET.format(cpus=args.cpus, hours=args.hours)
    task_prompt = compose.compose(paper, args.level, budget, experiments=experiments)
    (run_dir / "task_prompt.md").write_text(task_prompt)
    preamble = PREAMBLE.format(submission=paper.SUBMISSION, report=paper.REPORT)
    (run_dir / "agent_prompt.md").write_text(preamble + task_prompt)
    meta = {"paper": paper.NAME, "run_id": run_id, "run_dir": str(run_dir), "container": container, "image": tag,
            "starter": starter, "level": args.level, "experiments": experiments, "hours": args.hours,
            "started": time.time(), "log": str(run_dir / "agent_log.jsonl")}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(meta))
    print(json.dumps({"run_dir": str(run_dir), "agent_prompt": str(run_dir / "agent_prompt.md")}, indent=2))


def read_from_container(container, name):
    r = sh("docker", "exec", container, "cat", f"/workspace/{name}", check=False)
    return r.stdout if r.returncode == 0 else ""


def end(meta):
    sh("docker", "rm", "-f", meta["container"], check=False)
    STATE.unlink(missing_ok=True)


TRANSCRIPTS = Path.home() / ".claude" / "projects"


def copy_transcripts(run_dir, log):
    """Copy the full transcript of each subagent that worked on this run (its
    messages, tool calls and tool outputs, from Claude Code's session store)
    into run_dir/transcripts/, with its metadata (requested and served model).
    Returns the number of transcripts copied."""
    if not log.exists():
        return 0
    ids = sorted({json.loads(line)["agent_id"] for line in open(log) if line.strip()})
    out = run_dir / "transcripts"
    copied = 0
    for agent_id in ids:
        for suffix, name in ((".jsonl", "transcript.jsonl"), (".meta.json", "meta.json"),
                             (".prefix.json", "prefix.json")):
            src = sorted(TRANSCRIPTS.glob(f"*/*/subagents/agent-{agent_id}{suffix}"))
            if src:
                out.mkdir(exist_ok=True)
                (out / f"{agent_id}.{name}").write_bytes(src[-1].read_bytes())
                copied += suffix == ".jsonl"
    return copied


def finish(args):
    if not STATE.exists():
        sys.exit("no active run")
    meta = json.loads(STATE.read_text())
    paper = papers.load(meta.get("paper", "score-centering"))
    run_dir = Path(meta["run_dir"])
    # end the run first, so no later subagent call reaches the container
    STATE.unlink()
    submission = read_from_container(meta["container"], paper.SUBMISSION)
    report = read_from_container(meta["container"], paper.REPORT)
    original = (papers.starter_dir(paper, meta["experiments"]) / paper.SUBMISSION).read_text()
    diff = "".join(difflib.unified_diff(original.splitlines(True), submission.splitlines(True),
                                        f"a/{paper.SUBMISSION}", f"b/{paper.SUBMISSION}"))
    (run_dir / paper.SUBMISSION).write_text(submission)
    (run_dir / paper.REPORT).write_text(report)
    (run_dir / f"{Path(paper.SUBMISSION).stem}.diff").write_text(diff)
    results = {"run_id": meta["run_id"], "level": meta["level"], "experiments": meta["experiments"],
               "minutes": round((time.time() - meta["started"]) / 60, 1),
               "submitted": bool(submission), "changed": bool(diff), "configuration": paper.configuration(report)}
    if submission:
        results.update(images.run_graders(paper, submission, report))
    log = Path(meta["log"])
    results["agent_commands"] = sum(1 for _ in open(log)) if log.exists() else 0
    results["transcripts"] = copy_transcripts(run_dir, log)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    end(meta)
    summary = {k: results.get(k) for k in ("run_id", "minutes", "submitted", "changed", "agent_commands")}
    for g in paper.GRADERS:
        if g["name"] in results:
            r = results[g["name"]]
            summary[g["name"]] = {k: v for k, v in r.items() if k.startswith("passed") or k == "normalized"}
    print(json.dumps(summary, indent=2))


def abort(args):
    if not STATE.exists():
        sys.exit("no active run")
    end(json.loads(STATE.read_text()))
    print("run aborted")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--paper", required=True, choices=papers.available())
    s.add_argument("--level", required=True)
    s.add_argument("--no-experiments", action="store_true")
    s.add_argument("--hours", type=float, default=2.0, help="time budget stated in the prompt (not enforced)")
    s.add_argument("--cpus", type=int, default=4)
    s.add_argument("--rebuild", action="store_true")
    sub.add_parser("finish")
    sub.add_parser("abort")
    args = ap.parse_args()
    {"start": start, "finish": finish, "abort": abort}[args.cmd](args)


if __name__ == "__main__":
    main()
