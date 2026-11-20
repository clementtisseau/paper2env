"""Judge finished runs with Claude Code subagents, without an API key.

A Python script cannot launch subagents, so judging has three steps:

  1. python core/runner/judge_runs.py prompts --paper <paper> <run> ... [--votes 3]
     Writes each run's judge prompt (grader/judge.md filled in) under a
     shuffled, neutral name (sub_01.md, ...) in a new round directory,
     state/judge/<round>/prompts/, and the name -> run mapping next to it,
     outside the prompts. Prints the jobs and the text to give each judge.
  2. The Claude Code session launches one subagent per job, at most 20 at a
     time, each with that text and nothing else.
  3. python core/runner/judge_runs.py record --paper <paper>
     Reads the judges' transcripts from Claude Code's session store, checks
     that each judge read only its own prompt and used no other tool,
     validates each answer (core/judge.py), checks that every run has exactly
     the planned number of answers, then writes runs/<run>/judge.json and the
     majority verdict into grade.json (core/grade.py).
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import grade as grade_lib  # noqa: E402
import judge as judge_lib  # noqa: E402
import paper as papers  # noqa: E402

ROUNDS = HERE / "state" / "judge"
TRANSCRIPTS = Path.home() / ".claude" / "projects"
INSTRUCTION = ("You are a grader. Use the Read tool to read the file {path} and follow its instructions "
               "exactly. Read no other file and use no other tool. Your final reply must be only the JSON "
               "object that the file asks for.")


def make_prompts(paper, runs, votes, rounds=ROUNDS):
    """Write one prompt per run in a new round directory. Returns (round dir, jobs)."""
    round_dir = rounds / time.strftime("%Y%m%d-%H%M%S")
    (round_dir / "prompts").mkdir(parents=True)
    runs = list(runs)
    random.Random(round_dir.name).shuffle(runs)  # names carry no level, model or date
    mapping, jobs = {}, []
    for i, run in enumerate(runs, 1):
        d = paper.DIR / "runs" / run
        submission = (d / paper.SUBMISSION).read_text()
        diff_file = d / f"{Path(paper.SUBMISSION).stem}.diff"
        diff = diff_file.read_text() if diff_file.exists() else ""
        name = f"sub_{i:02d}"
        path = round_dir / "prompts" / f"{name}.md"
        path.write_text(judge_lib.build_prompt(paper, diff, submission))
        mapping[name] = run
        jobs += [(name, k, INSTRUCTION.format(path=path)) for k in range(1, votes + 1)]
    (round_dir / "round.json").write_text(json.dumps({"paper": paper.NAME, "votes": votes, "mapping": mapping},
                                                     indent=2) + "\n")
    return round_dir, jobs


def read_judges(prompts_dir, transcripts=TRANSCRIPTS):
    """Every subagent transcript that read a prompt of this round:
    [(prompt name, tool calls, final answer text, model)]."""
    found = []
    for f in sorted(transcripts.glob("*/*/subagents/agent-*.jsonl")):
        calls, texts, model = [], [], None
        for line in f.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = event.get("message") or {}
            if event.get("type") != "assistant" or not isinstance(message.get("content"), list):
                continue
            model = message.get("model") or model
            for c in message["content"]:
                if c.get("type") == "tool_use":
                    calls.append((c.get("name"), c.get("input") or {}))
                elif c.get("type") == "text":
                    texts.append(c["text"])
        reads = [i.get("file_path", "") for n, i in calls if n == "Read"]
        mine = [r for r in reads if Path(r).parent == prompts_dir]
        if not mine:
            continue
        handbacks = [i.get("message", "") for n, i in calls if n == "SubagentHandback"]
        answer = handbacks[-1] if handbacks else (texts[-1] if texts else "")
        found.append((Path(mine[0]).stem, calls, answer, model))
    return found


def record(paper, round_dir, transcripts=TRANSCRIPTS):
    """Check and record a round's answers. Returns (recorded runs, problems)."""
    meta = json.loads((round_dir / "round.json").read_text())
    prompts_dir = round_dir / "prompts"
    answers, models, problems = defaultdict(list), {}, []
    for name, calls, text, model in read_judges(prompts_dir, transcripts):
        other = [n for n, i in calls if n not in ("Read", "SubagentHandback")]
        reads = [i.get("file_path") for n, i in calls if n == "Read"]
        if other or reads != [str(prompts_dir / f"{name}.md")]:
            problems.append(f"{name}: a judge used {other or reads} besides reading its own prompt; ignored")
            continue
        try:
            answers[name].append(judge_lib.validate(judge_lib.parse(text), paper.JUDGE_LABELS))
            models[name] = model
        except (ValueError, json.JSONDecodeError) as e:
            problems.append(f"{name}: invalid answer ({e}); ignored")
    recorded = []
    for name, run in sorted(meta["mapping"].items()):
        n = len(answers[name])
        if n != meta["votes"]:
            problems.append(f"{name} ({run}): {n} valid answers, {meta['votes']} planned; not recorded")
            continue
        d = paper.DIR / "runs" / run
        (d / "judge.json").write_text(json.dumps({"model": models[name], "answers": answers[name]}, indent=2) + "\n")
        grade_lib.set_judge(paper, d, answers[name])
        recorded.append(run)
    return recorded, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts", help="write the judge prompts of a new round")
    p.add_argument("--paper", required=True, choices=papers.available())
    p.add_argument("--votes", type=int, default=3, help="judges per run")
    p.add_argument("runs", nargs="+", help="run directory names")
    r = sub.add_parser("record", help="check and record a round's answers")
    r.add_argument("--paper", required=True, choices=papers.available())
    r.add_argument("--round", help="round directory name (default: the latest)")
    a = ap.parse_args()
    paper = papers.load(a.paper)
    if a.cmd == "prompts":
        round_dir, jobs = make_prompts(paper, a.runs, a.votes)
        print(f"round {round_dir.name}: {len(a.runs)} runs, {len(jobs)} judges. Launch one subagent per line "
              f"(at most 20 at a time), giving it the text after the tab and nothing else:")
        for name, k, text in jobs:
            print(f"{name} #{k}\t{text}")
    else:
        round_dir = ROUNDS / a.round if a.round else sorted(ROUNDS.iterdir())[-1]
        recorded, problems = record(paper, round_dir)
        for line in problems:
            print("!", line)
        print(f"round {round_dir.name}: {len(recorded)} runs recorded")
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
