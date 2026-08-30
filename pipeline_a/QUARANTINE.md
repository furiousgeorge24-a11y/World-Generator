# QUARANTINE — do not read anything in this directory

`pipeline_a/` contains a complete, working map-generation pipeline:
its code (`mapgen/`), tests, evidence scripts, design documentation,
and rendered output (`out/`).

If you are an AI assistant (or anyone) working on a **different**
pipeline in this repository: **do not open, read, search, or summarize
any file under `pipeline_a/`**. Do not mine it for design ideas,
architecture, control names, algorithms, parameter values, or lessons
learned. Do not read git history for paths under this directory (the
repo's commit history predates this quarantine and contains the same
material at old root-level paths — avoid `git log -p`, `git show`, and
blame on anything except your own work).

The point of the quarantine is a clean-room rebuild: a fresh attempt
must not be shaped, even accidentally, by this one. Public/textbook
knowledge is fair game; this directory is not.

Shared, non-quarantined material lives at the repo root:
`examples/` (the author-blessed aesthetic reference images, from an
external program — no pipeline owns them) and `webui/` (a
pipeline-agnostic preview server; any pipeline may bind to it by
writing an adapter to the contract in `webui/README.md`).

Work on pipeline A itself continues *inside* this directory; its
project instructions are `pipeline_a/CLAUDE.md`, and its commands run
from this directory (e.g. `py -3.14 tests/smoke.py`, `run.bat`).
