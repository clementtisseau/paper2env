# Adding a paper

A paper is a directory `papers/<name>/` with a `paper.py` that declares what
differs from one paper to the next; `core/` reads only that module.
`papers/score-centering/` is the worked example: copy its structure, not its
content.

## What is shared, what you write

| Part | Shared, in `core/` | Written for each paper, in `papers/<name>/` |
|---|---|---|
| **Starter** | `sanitize.py`: copy, exact SHA-pinned edits, grep audit, fresh git history | `sanitize_spec.py`: what to remove, the revealing words |
| **Prompt** | `compose.py`: cumulative hint levels, tools section with or without experiments | `prompts/`: the task and each level |
| **Exact checks** | `images.py`, `grade.py`: content-tagged images, sandboxed runs, `grade.json` | `grader/check.py`: the property, the wrong answers it rejects |
| **LLM judge** | `judge.py`: prompt filling, answer validation, majority vote | `grader/judge.md`: the method and its near-misses |
| **Runners** | `task.py` (Inspect AI), `runner/` (Claude Code) | — |

## Layout

```
papers/<name>/
├── paper.py            the interface below
├── sanitize_spec.py    write_starter(): copy upstream/, apply the edits
├── upstream/           the paper's code, untouched (keep its license)
├── starter_extra/      files added to the exp starter (optional)
├── prompts/
│   ├── common.md       task and deliverables; {{TOOLS}} and {{BUDGET}} placeholders
│   ├── tools_experiments.md, tools_no_experiments.md
│   └── <level>.md      one per level after the first (levels are cumulative)
├── grader/
│   ├── check.py        exact checks: read the submission, print JSON with "correct"
│   └── judge.md        the judge's prompt, with {{DIFF}} and {{SUBMISSION}}
├── grading.md          what counts as a success
├── docker/             Dockerfile.agent-cpu (the starter), Dockerfile.grader (GRADER_INPUTS)
├── tests/              starter equivalence, check validation, prompt leaks
├── runs/               one directory per run
└── README.md           the paper, the environment, the results
```

Either grader is optional: a paper may have only checks, only a judge, or
both. Generated, not versioned: `starter/`, `docker/generated/`.

## `paper.py`

```python
NAME = "my-paper"                       # = the directory name
TITLE, URL = "...", "https://arxiv.org/abs/..."

LEVELS = ("L0", "L1", "L2", "L3", "L4") # the first is common.md alone
DEFAULT_LEVELS = ("L2", "L3", "L4")     # used when a runner is given no levels

SUBMISSION = "model.py"                 # the graded file, diffed against the starter
REPORT = "REPORT.md"

# Run in the grader image on the submission mounted at /sub/<SUBMISSION>;
# each prints JSON with a boolean "correct".
GRADERS = [{"name": "check", "args": ["grader/check.py", "/sub/model.py"], "timeout": 1800}]
GRADER_INPUTS = ["docker/Dockerfile.grader", "grader/check.py"]  # rebuild the image when they change

AUDIT = [r"method name", ...]           # regexes (case-insensitive) that fail the starter build
ALLOW = []                              # regexes of lines the audit tolerates
AUDIT_EXEMPT = {"LICENSE"}              # files not audited

def write_starter(out, experiments=True):
    """Write the starter into `out`; core.sanitize then audits and git-inits it."""

JUDGE_LABELS = ("reward_baseline", ..., "other")  # with judge.md: the wrong_answer labels
```

A missing attribute fails `core.paper.load(name)` with its name; the paper
appears in every runner's `--paper` choices once `paper.py` exists.

## Recipe

1. **Pick what to grade.** The idea the judge must recognize, and, if it can
   be checked, the property a correct submission must have on a toy
   instance. List the plausible wrong answers (familiar remedies,
   near-misses). This decides whether the paper is a good target.
2. **Vendor the code** into `upstream/`, untouched.
3. **Write the starter** with `core.sanitize` (`copy_tree`, `apply_edit`,
   `apply_span`). Test that it equals the reference with the method off, and
   iterate on `AUDIT` until `python core/sanitize.py --paper <name>` is clean.
4. **Write the prompts**: `common.md` states the problem as known before the
   paper; each level adds a hint. Test that no level says more than it should.
5. **Write the graders.** Checks: the reference passes, the starter fails,
   each wrong answer fails the check written for it. Judge: the method
   explained clearly, each near-miss with a label, the answer format. Then
   `grading.md`.
6. **Write the Dockerfiles** (build context: the paper's directory).
7. **Probe for leaks** with a subagent, then run
   `python core/runner/run.py start --paper <name> --level L0`.

`python -m pytest` runs every paper's tests.
