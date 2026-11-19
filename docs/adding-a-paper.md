# Adding a paper

A paper is a directory `papers/<name>/` with a `paper.py` that declares what
differs from one paper to the next. Everything in `core/` reads only that
module. `papers/score-centering/` is the worked example: copy its structure,
not its content.

## Layout

```
papers/<name>/
├── paper.py            the interface below
├── sanitize_spec.py    write_starter(): copy upstream/, apply the edits (optional split)
├── upstream/           the paper's code, untouched (keep its license)
├── starter_extra/      files added to the exp starter (optional)
├── prompts/
│   ├── common.md       task, deliverables, evaluation; {{TOOLS}} and {{BUDGET}} placeholders
│   ├── tools_experiments.md, tools_no_experiments.md
│   └── <level>.md      one per level after the first (levels are cumulative)
├── grader/
│   ├── …               grader scripts: read the submission, print JSON with "passed"
│   └── judge/          prompt.md, reference.md, rubric.md, schema.json, config.json
├── docker/
│   ├── Dockerfile.agent-cpu   COPY starter/${STARTER}/ /workspace/
│   └── Dockerfile.grader      copies GRADER_INPUTS
├── .dockerignore       keep runs/ and tests/ out of the build context
├── tests/              starter equivalence, grader validation, prompt leaks
├── runs/               one directory per run (written by core/runner)
└── README.md           the paper's choices, levels, results
```

Generated, not versioned: `starter/exp`, `starter/noexp`, `docker/generated/`.

## `paper.py`

```python
NAME = "my-paper"                       # = the directory name, used in image tags
TITLE, URL = "...", "https://arxiv.org/abs/..."

LEVELS = ("L0", "L1", "L2", "L3", "L4") # the first is common.md alone
DEFAULT_LEVELS = ("L2", "L3", "L4")     # used when a runner is given no levels

SUBMISSION = "model.py"                 # graded, and diffed against the starter
REPORT = "REPORT.md"

# Scripts run in the grader image (working directory = the paper's copy),
# on the submission mounted at /sub/<SUBMISSION>. Each prints JSON with at
# least "passed"; other "passed_*" keys are reported too. "config": True
# appends configuration(report) to the args.
GRADERS = [{"name": "property", "args": ["grader/check.py", "/sub/model.py"],
            "timeout": 1800, "config": False}]
GRADER_INPUTS = ["docker/Dockerfile.grader", "grader", "upstream/model.py"]

AUDIT = [r"method name", ...]           # regexes (case-insensitive) that fail the starter build
ALLOW = []                              # regexes of lines the audit tolerates
AUDIT_EXEMPT = {"LICENSE"}              # files not audited

def write_starter(out, experiments=True):
    """Write the starter into `out` (core.sanitize then audits and git-inits it)."""

def configuration(report):
    """key=value settings the agent may request in its report; [] if none."""
```

`core.paper.load(name)` fails with the list of missing attributes, and the
paper appears in every runner's `--paper` choices as soon as its `paper.py`
exists.

## Recipe

1. **Pick the property.** Write down, as a check on a toy instance, what a
   correct submission must satisfy. List the plausible wrong answers (the
   familiar remedies, the near-misses) and make sure each one fails a check.
   This is the hard part and decides whether the paper is a good target.
2. **Vendor the code** into `upstream/`, untouched.
3. **Write the starter.** With `core.sanitize.copy_tree`, `apply_edit` (exact
   match, once) and `apply_span` (SHA-pinned), remove the contribution. Test
   that the starter equals the reference with the method turned off. Iterate
   on `AUDIT` until `python core/sanitize.py --paper <name>` is clean for the
   right reasons.
4. **Write the prompts.** `common.md` states the problem as it was known
   before the paper. Add levels, and a test that no level says more than it
   should (see `papers/score-centering/tests/test_prompts.py`).
5. **Write the graders** and validate them on known answers: the reference
   passes, the starter fails, each wrong answer fails the check written for
   it.
6. **Write the judge's reference, rubric and config** (`config.json`:
   criterion weights, near-miss ids, the minimum scores that count as a
   rediscovery, the criteria that must not come from the prompt).
7. **Write the Dockerfiles** (build context = the paper's directory).
8. **Probe for leaks** with a subagent before the first runs, then run:
   `python core/runner/run.py start --paper <name> --level L0`.

`python -m pytest` runs every paper's tests.
