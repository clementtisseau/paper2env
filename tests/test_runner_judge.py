"""core/runner/judge_runs.py on fake runs and fake judge transcripts: prompts under
neutral names, then only the answers of well-behaved judges are recorded."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core" / "runner"))
import judge_runs as runner_judge  # noqa: E402

LABELS = ("reward_baseline", "importance_weights", "other")
YES = {"evidence": "center = ...", "idea_found": True, "wrong_answer": None}
NO = {"evidence": "w = ...", "idea_found": False, "wrong_answer": "importance_weights"}


def fake_paper(tmp):
    d = tmp / "paper"
    (d / "grader").mkdir(parents=True)
    shutil.copy(ROOT / "papers/score-centering/grader/judge.md", d / "grader/judge.md")
    for run in ("run_a", "run_b"):
        r = d / "runs" / run
        r.mkdir(parents=True)
        (r / "losses.py").write_text(f"# {run}\n")
        (r / "grade.json").write_text(json.dumps({"correct": True, "idea_found": None, "success": None,
                                                  "graders": {"check": {"correct": True}}, "judge": None}))
    return SimpleNamespace(NAME="fake", DIR=d, SUBMISSION="losses.py", JUDGE_LABELS=LABELS)


def transcript(store, n, calls, answer):
    """One judge's transcript in Claude Code's layout: <project>/<session>/subagents/agent-<id>.jsonl."""
    content = [{"type": "tool_use", "name": name, "input": inp} for name, inp in calls]
    content.append({"type": "tool_use", "name": "SubagentHandback", "input": {"message": json.dumps(answer)}})
    f = store / "proj" / "session" / "subagents" / f"agent-{n}.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"type": "assistant", "message": {"model": "claude-opus-5-5", "content": content}}) + "\n")


def test_round(tmp_path):
    paper = fake_paper(tmp_path)
    round_dir, jobs = runner_judge.make_prompts(paper, ["run_a", "run_b"], votes=2, rounds=tmp_path / "rounds")
    mapping = json.loads((round_dir / "round.json").read_text())["mapping"]
    assert sorted(mapping) == ["sub_01", "sub_02"] and sorted(mapping.values()) == ["run_a", "run_b"]
    assert len(jobs) == 4 and all(str(round_dir / "prompts") in text for _, _, text in jobs)
    assert "run_a" not in "".join(p.read_text() for p in (round_dir / "prompts").iterdir()).split("# run_")[0]

    name_a = next(k for k, v in mapping.items() if v == "run_a")
    name_b = next(k for k, v in mapping.items() if v == "run_b")
    read = lambda name: ("Read", {"file_path": str(round_dir / "prompts" / f"{name}.md")})
    store = tmp_path / "store"
    transcript(store, 1, [read(name_a)], YES)
    transcript(store, 2, [read(name_a)], YES)
    transcript(store, 3, [read(name_b)], NO)                                       # run_b: one valid answer
    transcript(store, 4, [read(name_b), ("Read", {"file_path": "/etc/hosts"})], NO)  # read another file
    transcript(store, 5, [read(name_b)], {"idea_found": "maybe"})                  # invalid answer

    recorded, problems = runner_judge.record(paper, round_dir, transcripts=store)
    assert recorded == ["run_a"]
    assert len(problems) == 3 and any("run_b" in p and "1 valid" in p for p in problems)
    grade = json.loads((paper.DIR / "runs/run_a/grade.json").read_text())
    assert grade["idea_found"] is True and grade["success"] is True and grade["judge"]["votes"] == "2/2"
    assert json.loads((paper.DIR / "runs/run_a/judge.json").read_text())["answers"] == [YES, YES]
    assert json.loads((paper.DIR / "runs/run_b/grade.json").read_text())["idea_found"] is None
