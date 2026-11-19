"""Build a paper's starter: the repository the agent receives.

The engine is generic; what to remove is the paper's own business. A paper's
`write_starter(out, experiments)` copies its code and applies its edits, using
the helpers below; this module then audits the result and gives it a fresh
git history:

- `apply_edit`: an exact replacement that must match exactly once;
- `apply_span`: a replacement of a span pinned by its SHA-256, so a change in
  the paper's code makes the build fail instead of letting the method through;
- `copy_tree`: copy the paper's code, skipping what describes or tests it;
- the audit greps every file for the paper's revealing words and fails the
  build on any hit outside its allowlist.

    python core/sanitize.py --paper score-centering [--no-experiments] [--out DIR] [--no-git]
"""
import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper as papers  # noqa: E402

# Never copied into a starter, whatever the paper.
EXCLUDE_ANY = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".DS_Store", "outputs", "wandb", ".venv"}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def apply_edit(text, old, new, where):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{where}: expected exactly one match, found {n}:\n{old[:200]}")
    return text.replace(old, new)


def apply_span(text, start, end, digest, new, where):
    i = text.find(start)
    j = text.find(end, i + 1) if i >= 0 else -1
    if i < 0 or j < 0 or text.count(start) != 1:
        raise SystemExit(f"{where}: span markers not found exactly once: {start[:80]!r}")
    span = text[i:j + len(end)]
    if sha(span) != digest:
        raise SystemExit(f"{where}: span changed upstream (sha256 {sha(span)} != {digest}); review the edit")
    return text[:i] + new + text[j + len(end):]


def copy_tree(src, out, exclude_top=()):
    """Copy src to out, without EXCLUDE_ANY anywhere and exclude_top at the top."""
    src = Path(src).resolve()

    def ignore(directory, names):
        skip = {n for n in names if n in EXCLUDE_ANY}
        if Path(directory).resolve() == src:
            skip |= {n for n in names if n in exclude_top}
        return skip
    shutil.copytree(src, out, ignore=ignore)


def audit(out, patterns, allow=(), exempt=()):
    """Lines matching a revealing pattern and no allowed one."""
    hits = []
    for f in sorted(out.rglob("*")):
        if not f.is_file() or ".git" in f.parts or f.name in exempt:
            continue
        try:
            lines = f.read_text().splitlines()
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(lines, 1):
            if any(re.search(p, line, re.I) for p in patterns) and not any(re.search(a, line, re.I) for a in allow):
                hits.append(f"{f.relative_to(out)}:{n}: {line.strip()}")
    return hits


def git_init(out):
    """A fresh history: one commit, fixed author and date."""
    env = {"GIT_AUTHOR_NAME": "starter", "GIT_AUTHOR_EMAIL": "starter@localhost",
           "GIT_COMMITTER_NAME": "starter", "GIT_COMMITTER_EMAIL": "starter@localhost",
           "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}
    run = lambda *a: subprocess.run(["git", "-C", str(out), *a], check=True, capture_output=True,
                                    env={**env, "PATH": "/usr/bin:/bin:/usr/local/bin"})
    run("init", "-q", "-b", "main")
    run("add", "-A")
    run("commit", "-q", "-m", "Initial commit")


def build(paper, out=None, git=True, experiments=True):
    """Write, audit and git-init the paper's starter. Returns its directory."""
    name = papers.starter_name(experiments)
    out = Path(out or paper.DIR / "starter" / name).resolve()
    if out.exists():
        shutil.rmtree(out)
    paper.write_starter(out, experiments)
    hits = audit(out, paper.AUDIT, getattr(paper, "ALLOW", ()), getattr(paper, "AUDIT_EXEMPT", ()))
    if hits:
        raise SystemExit(f"{paper.NAME}/{name}: audit failed, possible leaks:\n" + "\n".join(hits))
    if git:
        git_init(out)
    n = sum(1 for f in out.rglob("*") if f.is_file() and ".git" not in f.parts)
    print(f"{paper.NAME}/{name}: starter written to {out} ({n} files), audit clean")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--paper", required=True, choices=papers.available())
    parser.add_argument("--no-experiments", action="store_true",
                        help="starter without the paper's training tools (the agent cannot run training)")
    parser.add_argument("--out", type=Path, help="output dir (default papers/<paper>/starter/exp or noexp)")
    parser.add_argument("--no-git", action="store_true", help="skip git init of the starter")
    args = parser.parse_args()
    build(papers.load(args.paper), args.out, git=not args.no_git, experiments=not args.no_experiments)


if __name__ == "__main__":
    main()
