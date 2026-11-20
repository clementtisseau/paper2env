"""Judge plumbing (core/judge.py), on score-centering's judge: the prompt is
filled verbatim, answers are validated, judges are combined by majority."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
import judge  # noqa: E402
import paper as papers  # noqa: E402

PAPER = papers.load("score-centering")
LABELS = PAPER.JUDGE_LABELS


def answer(found, wrong=None, evidence="x"):
    return {"evidence": evidence, "idea_found": found, "wrong_answer": wrong}


def answer_section():
    return judge.prompt_path(PAPER).read_text().split("## Your answer")[1]


def test_prompt_inserts_agent_text_verbatim():
    text = judge.build_prompt(PAPER, "DIFF {{SUBMISSION}}", "CODE")
    assert "DIFF {{SUBMISSION}}" in text and "CODE" in text  # never re-templated
    assert "{{" not in text.replace("DIFF {{SUBMISSION}}", "")


def test_prompt_lists_exactly_the_labels():
    """The labels quoted in judge.md's answer section are those of paper.py."""
    quoted = set(re.findall(r'"([a-z_]+)"', answer_section().split("Example")[0]))
    assert quoted - {"evidence", "idea_found", "wrong_answer", "unchanged"} == set(LABELS)


def test_examples_in_prompt_are_valid():
    examples = [judge.parse(line) for line in answer_section().splitlines() if line.startswith("{")]
    assert len(examples) == 2
    for e in examples:
        judge.validate(e, LABELS)


def test_parse_fenced_and_bare():
    assert judge.parse('Here:\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}
    assert judge.parse('prefix {"a": 1} suffix') == {"a": 1}


@pytest.mark.parametrize("bad", [
    answer(True, "reward_baseline"),                    # a label with idea_found true
    answer(False, None),                                # no label with idea_found false
    answer(False, "not_a_label"),
    answer("yes"),                                      # not a boolean
    {"idea_found": True, "evidence": "x", "wrong_answer": None},   # wrong order
    answer(True) | {"extra": 1},
])
def test_invalid_answers(bad):
    with pytest.raises(ValueError):
        judge.validate(bad, LABELS)


def test_valid_answers():
    judge.validate(answer(True), LABELS)
    judge.validate(answer(False, "importance_weights"), LABELS)


def test_majority():
    v = judge.majority([answer(True), answer(False, "tuning"), answer(True)])
    assert v == {"idea_found": True, "wrong_answer": None, "votes": "2/3"}
    v = judge.majority([answer(False, "tuning"), answer(False, "importance_weights"), answer(False, "tuning")])
    assert v == {"idea_found": False, "wrong_answer": "tuning", "votes": "0/3"}
