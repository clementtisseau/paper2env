"""Assemble the agent prompt for a paper and a hint level.

A paper's prompts/ holds common.md (task, deliverables, tools, evaluation,
with {{TOOLS}} and {{BUDGET}} placeholders), tools_experiments.md and
tools_no_experiments.md, and one file per level after the first. Levels are
cumulative: the first level is common.md alone, and each later level adds its
own file to everything before it. `experiments` picks the tools section and
must match the starter (built with or without the paper's training tools).

    python core/compose.py --paper score-centering L2a [--no-experiments]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper as papers  # noqa: E402

BUDGET = "Budget: {cpus} CPU cores, no GPU, and {hours:g} hours of working time."


def compose(paper, level, cpus=4, hours=2.0, experiments=True):
    budget = BUDGET.format(cpus=cpus, hours=float(hours))
    if level not in paper.LEVELS:
        raise ValueError(f"level must be one of {paper.LEVELS}")
    prompts = paper.DIR / "prompts"
    tools = (prompts / ("tools_experiments.md" if experiments else "tools_no_experiments.md")).read_text().rstrip()
    common = (prompts / "common.md").read_text().replace("{{TOOLS}}", tools).replace("{{BUDGET}}", budget)
    parts = [common.rstrip()]
    parts += [(prompts / f"{lvl}.md").read_text().rstrip() for lvl in paper.LEVELS[1:paper.LEVELS.index(level) + 1]]
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paper", required=True, choices=papers.available())
    ap.add_argument("level")
    ap.add_argument("--no-experiments", action="store_true")
    a = ap.parse_args()
    print(compose(papers.load(a.paper), a.level, experiments=not a.no_experiments), end="")
