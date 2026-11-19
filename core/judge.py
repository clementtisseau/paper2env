"""Build a paper's judge prompt and turn judge verdicts into scores.

No model is called here: the caller sends build_prompt(...) to one or more
judge models and passes each JSON answer to score(...). What is
paper-specific lives in papers/<name>/grader/judge/:

- prompt.md     the template, with {{REFERENCE}} {{RUBRIC}} {{SCHEMA}}
                {{AGENT_PROMPT}} {{REPORT}} {{DIFF}}
- reference.md  the reference explanation of the contribution
- rubric.md     the hidden rubric, criteria and near-misses
- schema.json   the verdict's JSON schema (shown to the judge)
- config.json   {"weights": {criterion: weight}, "near_misses": [ids],
                 "rediscovered": {criterion: minimum score},
                 "discovery": [criteria that must not come from the prompt],
                 "near_miss_zeroes": criterion a near-miss sets to 0}
"""
import json
import re
from pathlib import Path


def judge_dir(paper):
    return paper.DIR / "grader" / "judge"


def config(paper):
    return json.loads((judge_dir(paper) / "config.json").read_text())


def build_prompt(paper, agent_prompt, report, diff):
    d = judge_dir(paper)
    fields = {
        "REFERENCE": (d / "reference.md").read_text(),
        "RUBRIC": (d / "rubric.md").read_text(),
        "SCHEMA": (d / "schema.json").read_text(),
        "AGENT_PROMPT": agent_prompt,
        "REPORT": report,
        "DIFF": diff,
    }
    # one pass over the template only: text inserted from the agent is never
    # scanned again, so a "{{DIFF}}" inside its report stays literal
    parts = re.split(r"\{\{([A-Z_]+)\}\}", (d / "prompt.md").read_text())
    names = parts[1::2]
    assert sorted(names) == sorted(fields), f"prompt.md placeholders {names} != {sorted(fields)}"
    return "".join(fields[p] if i % 2 else p for i, p in enumerate(parts))


def validate(verdict, cfg):
    """Raise ValueError if the verdict does not have the expected shape."""
    def need(cond, msg):
        if not cond:
            raise ValueError(msg)
    weights = cfg["weights"]
    need(isinstance(verdict, dict), "verdict must be an object")
    need(set(verdict) == {"criteria", "near_miss", "alternative_contribution", "summary"}, "bad top-level keys")
    criteria = verdict["criteria"]
    need(isinstance(criteria, dict) and set(criteria) == set(weights), f"criteria must be {sorted(weights)}")
    for name, c in criteria.items():
        need(isinstance(c, dict) and set(c) == {"score", "given_in_prompt", "evidence", "reasoning"}, f"{name}: bad keys")
        need(c["score"] in (0, 1, 2) and not isinstance(c["score"], bool), f"{name}: score must be 0, 1 or 2")
        need(isinstance(c["given_in_prompt"], bool), f"{name}: given_in_prompt must be a boolean")
        need(isinstance(c["evidence"], str) and isinstance(c["reasoning"], str), f"{name}: text fields")
    need(verdict["near_miss"] is None or verdict["near_miss"] in cfg["near_misses"],
         f"near_miss must be one of {cfg['near_misses']} or null")
    alt = verdict["alternative_contribution"]
    need(isinstance(alt, dict) and set(alt) == {"present", "description"} and isinstance(alt["present"], bool),
         "bad alternative_contribution")
    need(isinstance(verdict["summary"], str), "summary must be a string")


def score(verdict, cfg):
    """Weighted score in [0, 1], plus flags. `rediscovered`: every criterion of
    cfg["rediscovered"] reaches its minimum. `discovered` additionally requires
    that none of cfg["discovery"] was already given in the agent's prompt."""
    validate(verdict, cfg)
    c, weights = verdict["criteria"], cfg["weights"]
    zeroed = cfg.get("near_miss_zeroes")
    if verdict["near_miss"] is not None and zeroed and c[zeroed]["score"] != 0:
        raise ValueError(f"a near-miss requires {zeroed} = 0 (rubric)")
    rediscovered = all(c[k]["score"] >= v for k, v in cfg["rediscovered"].items())
    return {
        "score": sum(weights[k] * c[k]["score"] for k in weights) / (2 * sum(weights.values())),
        "rediscovered": rediscovered,
        "discovered": rediscovered and not any(c[k]["given_in_prompt"] for k in cfg["discovery"]),
        "near_miss": verdict["near_miss"],
        "needs_human_review": verdict["alternative_contribution"]["present"],
    }


def aggregate(verdicts, cfg):
    """Several judges: median weighted score, flags by majority, review if any."""
    scored = [score(v, cfg) for v in verdicts]
    values = sorted(s["score"] for s in scored)
    n = len(values)
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    majority = lambda key: sum(s[key] for s in scored) * 2 > n
    return {"score": median, "rediscovered": majority("rediscovered"), "discovered": majority("discovered"),
            "near_misses": [s["near_miss"] for s in scored],
            "needs_human_review": any(s["needs_human_review"] for s in scored), "judges": scored}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import paper as papers
    p = papers.load(sys.argv[1])
    print(json.dumps(aggregate([json.loads(Path(v).read_text()) for v in sys.argv[2:]], config(p)), indent=2))
