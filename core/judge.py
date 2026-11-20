"""LLM judge plumbing: fill a paper's judge prompt, validate the answers, take
the majority. No model is called here.

A paper with a judge has papers/<name>/grader/judge.md, a prompt with two
placeholders, {{DIFF}} (the submission's diff against the starter) and
{{SUBMISSION}} (the submitted file), and declares in paper.py the labels its
judge may give as `wrong_answer` (JUDGE_LABELS). The judge answers with one
JSON object:

    {"evidence": str, "idea_found": bool, "wrong_answer": null | one of JUDGE_LABELS}

with wrong_answer null exactly when idea_found is true.
"""
import json
import re
from collections import Counter

FIELDS = ["evidence", "idea_found", "wrong_answer"]


def prompt_path(paper):
    return paper.DIR / "grader" / "judge.md"


def has_judge(paper):
    return prompt_path(paper).exists()


def build_prompt(paper, diff, submission):
    """The judge prompt for one submission. Agent text is inserted in one pass,
    so a "{{DIFF}}" inside the submission stays literal."""
    fields = {"DIFF": diff or "(no change)", "SUBMISSION": submission}
    parts = re.split(r"\{\{([A-Z_]+)\}\}", prompt_path(paper).read_text())
    names = parts[1::2]
    assert sorted(names) == sorted(fields), f"judge.md placeholders {names} != {sorted(fields)}"
    return "".join(fields[p] if i % 2 else p for i, p in enumerate(parts))


def parse(text):
    """The JSON object in a judge's reply (bare, or in a ```json fence)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


def validate(answer, labels):
    """Raise ValueError unless the answer has the expected shape."""
    def need(cond, msg):
        if not cond:
            raise ValueError(msg)
    need(isinstance(answer, dict) and list(answer) == FIELDS, f"fields must be {FIELDS}, in this order")
    need(isinstance(answer["evidence"], str), "evidence must be a string")
    need(isinstance(answer["idea_found"], bool), "idea_found must be true or false")
    if answer["idea_found"]:
        need(answer["wrong_answer"] is None, "wrong_answer must be null when idea_found is true")
    else:
        need(answer["wrong_answer"] in labels, f"wrong_answer must be one of {list(labels)}")
    return answer


def majority(answers):
    """Several judges on one submission: idea_found by strict majority, and the
    most common wrong_answer among the judges that said false."""
    found = sum(a["idea_found"] for a in answers)
    idea_found = found * 2 > len(answers)
    wrong = Counter(a["wrong_answer"] for a in answers if not a["idea_found"])
    return {"idea_found": idea_found,
            "wrong_answer": None if idea_found else wrong.most_common(1)[0][0],
            "votes": f"{found}/{len(answers)}"}
