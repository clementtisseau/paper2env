"""Load a paper's environment definition: papers/<name>/paper.py.

A paper module declares everything that differs from one paper to the next;
the rest of core/ only reads these attributes. See docs/adding-a-paper.md.

Required:
    NAME, TITLE, URL               identity
    LEVELS                         hint levels, easiest last; the first has no file
                                   of its own (common.md alone)
    DEFAULT_LEVELS                 levels run when none are given
    SUBMISSION, REPORT             files the agent leaves in /workspace; SUBMISSION
                                   is graded and diffed against the starter
    GRADERS                        [{"name", "args", "timeout", "config"}]: scripts run
                                   in the grader image on /sub/<SUBMISSION>; "config"
                                   appends configuration(report) to the args
    GRADER_INPUTS                  paths (relative to the paper) copied into the grader
                                   image, so the image is rebuilt when they change
    AUDIT, ALLOW, AUDIT_EXEMPT     revealing patterns, allowed lines, exempt files
    write_starter(out, experiments)   write the starter into `out`
    configuration(report)          key=value settings the agent may request
Set by the loader:
    DIR                            the paper's directory
"""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPERS = REPO / "papers"
REQUIRED = ("NAME", "TITLE", "URL", "LEVELS", "DEFAULT_LEVELS", "SUBMISSION", "REPORT", "GRADERS",
            "GRADER_INPUTS", "AUDIT", "write_starter", "configuration")
_cache = {}


def available():
    return sorted(p.parent.name for p in PAPERS.glob("*/paper.py"))


def load(name):
    """The paper module for papers/<name>/, with DIR set."""
    if name not in _cache:
        path = PAPERS / name / "paper.py"
        if not path.exists():
            raise SystemExit(f"unknown paper {name!r}; available: {', '.join(available())}")
        spec = importlib.util.spec_from_file_location(f"paper_{name.replace('-', '_')}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        missing = [a for a in REQUIRED if not hasattr(module, a)]
        if missing:
            raise SystemExit(f"papers/{name}/paper.py lacks {', '.join(missing)}")
        module.DIR = path.parent
        _cache[name] = module
    return _cache[name]


def starter_name(experiments=True):
    return "exp" if experiments else "noexp"


def starter_dir(paper, experiments=True):
    return paper.DIR / "starter" / starter_name(experiments)
