"""Images and grading, shared by both runners (core/task.py for API models,
core/runner/run.py for a Claude Code subagent), so they grade the same way.

Each paper brings its own Dockerfiles (papers/<name>/docker/: an agent image
holding the starter, a grader image holding the grading scripts), built with
the paper's directory as context. Images are tagged by the content they copy
in, so they are rebuilt exactly when that content changes. Standard library
only.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper as papers  # noqa: E402
import sanitize  # noqa: E402

GRADER_IMAGE = "p2e-{paper}-grader:{digest}"
AGENT_IMAGE = "p2e-{paper}-agent:{starter}-{digest}"
WORKSPACE = "/workspace"
CPU_BUDGET = "Budget: {cpus} CPU cores, no GPU, and {hours:g} hours of working time."


def docker_build(tag, dockerfile, context, args=()):
    cmd = ["docker", "build", "-q", "-t", tag, "-f", str(dockerfile), *args]
    ca = os.environ.get("P2E_PIPCA") or os.environ.get("SC_RLENV_PIPCA")  # optional CA bundle (TLS interception)
    if ca:
        cmd += ["--secret", f"id=pipca,src={ca}"]
    subprocess.run([*cmd, str(context)], check=True, capture_output=True)


def have_image(tag):
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0


def content_digest(*paths, skip_docs=False):
    """Short hash of files' relative paths and contents (directories walked).
    skip_docs leaves out *.md files: right for a grader, whose docs do not
    change what it does, wrong for a starter, whose README the agent reads."""
    h = hashlib.sha256()
    for root in paths:
        # .git is skipped (it follows from the files, and its index holds
        # timestamps), and so are Python caches (not copied into images)
        skip = lambda f: ({".git", "__pycache__"} & set(f.relative_to(root).parts) or f.suffix == ".pyc"
                          or (skip_docs and f.suffix == ".md"))
        files = sorted(f for f in root.rglob("*") if f.is_file() and not skip(f)) if root.is_dir() else [root]
        for f in files:
            h.update(str(f.relative_to(root.parent)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:12]


def grader_image(paper):
    digest = content_digest(*(paper.DIR / p for p in paper.GRADER_INPUTS), skip_docs=True)
    return GRADER_IMAGE.format(paper=paper.NAME, digest=digest)


def ensure_images(paper, experiments, rebuild=False):
    """Build the paper's starter, grader image and agent image.
    Returns (starter name, agent image tag)."""
    if rebuild or not have_image(grader_image(paper)):
        docker_build(grader_image(paper), paper.DIR / "docker/Dockerfile.grader", paper.DIR)
    starter = papers.starter_name(experiments)
    sanitize.build(paper, experiments=experiments)
    dockerfile = paper.DIR / "docker/Dockerfile.agent-cpu"
    tag = AGENT_IMAGE.format(paper=paper.NAME, starter=starter,
                             digest=content_digest(papers.starter_dir(paper, experiments), dockerfile))
    if rebuild or not have_image(tag):
        docker_build(tag, dockerfile, paper.DIR, ["--build-arg", f"STARTER={starter}"])
    return starter, tag


def grader_command(paper, submission_dir, args):
    """docker run line for one of the paper's grader scripts; the submission
    directory is mounted read-only at /sub, without network."""
    return ["docker", "run", "--rm", "--network", "none", "--cpus", "4", "-v", f"{submission_dir}:/sub:ro",
            grader_image(paper), "python", "-I", *args]


def grader_args(paper, grader, report):
    return [*grader["args"], *(paper.configuration(report) if grader.get("config") else [])]


def parse_grader_output(out, err):
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"passed": False, "error": (err or out).decode(errors="replace")[-1500:]}


def run_grader_sync(paper, submission, args, timeout):
    """Run one grader script on the submission text (blocking)."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, paper.SUBMISSION).write_text(submission)
        os.chmod(tmp, 0o755)
        try:
            proc = subprocess.run(grader_command(paper, tmp, args), capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"grader timed out after {timeout}s"}
    return parse_grader_output(proc.stdout, proc.stderr)


def run_graders(paper, submission, report, only=None):
    """Every grader of the paper (or those named in `only`): {name: result}."""
    return {g["name"]: run_grader_sync(paper, submission, grader_args(paper, g, report), g["timeout"])
            for g in paper.GRADERS if only is None or g["name"] in only}
