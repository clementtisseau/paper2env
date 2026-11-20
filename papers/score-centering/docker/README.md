# Images

Built with this paper's directory as context (`core/images.py`).

- `Dockerfile.agent-cpu`: the agent's sandbox, CPU only. Holds a starter
  (`ARG STARTER=exp|noexp`) and CPU JAX; in the `exp` starter, `run_dev.sh`
  runs `dev/mini.py`. Run without network, 4 CPUs, 8 GB. Tagged by the
  starter's content, so it is rebuilt only when the starter changes.
- `Dockerfile.grader`: the exact checks (`grader/check.py`), CPU only. They
  execute the agent's `losses.py`, so they run apart from the host, without
  network, with the file mounted read-only. Tagged by the content it copies
  in.
- *(not done)* a GPU agent image for validation on Qwen3-0.6B.

Both runners build the images on first use. By hand:

```bash
python core/sanitize.py --paper score-centering
python -c "import sys; sys.path.insert(0, 'core'); import images, paper; print(images.ensure_images(paper.load('score-centering'), True))"
```
