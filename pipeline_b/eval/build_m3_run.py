"""Build one immutable M3-family evaluation bundle.

Example:
    python eval/build_m3_run.py --run-dir out/m3_run1

The command accepts an existing run directory containing galleries, but
its ``eval`` child must not exist. It builds in a sibling temporary
directory and atomically publishes only after every stimulus, key, and
manifest has been written.
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from engine import VERSION
from engine.registry import effective_controls
from eval import m3_panels, m3_trials

OUT_ROOT = (ROOT / "out").resolve()
HISTORICAL_RUNS = {
    (ROOT / "out" / "m2").resolve(),
    (ROOT / "out" / "m3").resolve(),
}
BRIDGE_SOURCE = ROOT / "out" / "m3" / "eval"
GALLERY_NAMES = ("m3_gallery.png", "m3_instruments.png", "m3_pairs.png")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_tree_sha256():
    files = sorted((ROOT / "engine").rglob("*.py"))
    files += [
        ROOT / "eval" / "m2_trials.py",
        ROOT / "eval" / "m3_trials.py",
        ROOT / "eval" / "m3_panels.py",
        ROOT / "eval" / "build_m3_run.py",
        ROOT / "eval" / "score_run.py",
        ROOT / "eval" / "prompts" / "2afc_v2.md",
        ROOT / "eval" / "prompts" / "critique_v2.md",
    ]
    h = hashlib.sha256()
    for path in sorted(p for p in files if p.exists()):
        rel = path.relative_to(ROOT).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        data = path.read_bytes()
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def git_provenance():
    def run(*args):
        result = subprocess.run(
            ["git", *args], cwd=REPO, text=True,
            capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--short", "--", "pipeline_b")
    return {
        "head": run("rev-parse", "HEAD"),
        "pipeline_b_dirty": bool(status),
        "pipeline_b_status": status.splitlines() if status else [],
    }


def resolve_run_dir(raw):
    run_dir = Path(raw)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir = run_dir.resolve()
    try:
        run_dir.relative_to(OUT_ROOT)
    except ValueError as exc:
        raise ValueError("--run-dir must be inside pipeline_b/out") from exc
    if run_dir in HISTORICAL_RUNS:
        raise ValueError("refusing to modify a historical M2/M3 run")
    return run_dir


def stage_bridge(temp_dir):
    src_trials = BRIDGE_SOURCE / "trials"
    src_key = BRIDGE_SOURCE / "key" / "m3_trials_key.json"
    if not src_trials.is_dir() or not src_key.is_file():
        raise FileNotFoundError(
            "the preserved M3 bundle is required for bridge calibration")

    dst_trials = temp_dir / "bridge" / "m3_0_3_0" / "trials"
    dst_trials.mkdir(parents=True)
    copied = []
    for src in sorted(src_trials.glob("trial_*.png")):
        dst = dst_trials / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    if len(copied) != 12:
        raise RuntimeError(
            f"expected 12 archived M3 bridge trials, found {len(copied)}")
    shutil.copy2(ROOT / "eval" / "prompts" / "2afc_v2.md",
                 dst_trials / "instructions.md")
    shutil.copy2(src_key,
                 temp_dir / "key" / "bridge_m3_0_3_0_trials_key.json")
    return {
        "source_engine_version": "0.3.0-m3",
        "purpose": "new-judge/new-prompt regression bridge",
        "trial_count": len(copied),
        "stimulus_sha256": {
            path.name: sha256_file(path) for path in copied
        },
        "prompt_id": "2afc_v2",
        "known_result": (
            "historical v1 cohort detected all 8 candidate arms; "
            "v2 bridge must retain detection while reducing unsupported "
            "canon false-positive claims"),
    }


def build(run_dir):
    run_dir = resolve_run_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = run_dir / "eval"
    if eval_dir.exists():
        raise FileExistsError(
            f"evaluation bundle already exists and will not be overwritten: {eval_dir}")

    temp_dir = Path(tempfile.mkdtemp(prefix=".eval-build-", dir=run_dir))
    print(f"building unpublished bundle in {temp_dir}")
    try:
        trial_record = m3_trials.build(temp_dir)
        panel_record = m3_panels.build(temp_dir)
        bridge_record = stage_bridge(temp_dir)
        (temp_dir / "verdicts" / "trials").mkdir(parents=True)
        (temp_dir / "verdicts" / "panels").mkdir(parents=True)
        (temp_dir / "verdicts" / "bridge").mkdir(parents=True)

        gallery_hashes = {}
        for name in GALLERY_NAMES:
            path = run_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"missing Run 1 artifact: {path}")
            gallery_hashes[name] = sha256_file(path)

        canon_ids = sorted(set(m3_trials.REF_IDS) | set(m3_panels.CANON))
        canon_hashes = {
            f"ref{ref_id}.png": sha256_file(
                ROOT.parent / "examples" / f"ref{ref_id}.png")
            for ref_id in canon_ids
        }
        artifact_hashes = {
            path.relative_to(temp_dir).as_posix(): sha256_file(path)
            for path in sorted(temp_dir.rglob("*")) if path.is_file()
        }
        manifest = {
            "schema": 1,
            "harness_version": "m3-eval-v2",
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_directory": run_dir.relative_to(ROOT).as_posix(),
            "engine_version": VERSION,
            "effective_controls": effective_controls({}),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pillow": PILLOW_VERSION,
            },
            "git": git_provenance(),
            "pipeline_b_source_tree_sha256": source_tree_sha256(),
            "run_artifact_sha256": gallery_hashes,
            "canon_source_sha256": canon_hashes,
            "trials": trial_record,
            "panels": panel_record,
            "bridge": bridge_record,
            "judge_visible_directories": [
                "trials", "panels", "bridge/m3_0_3_0/trials"
            ],
            "hidden_directories": ["key", "verdicts"],
            "generated_artifact_sha256": artifact_hashes,
        }
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temp_dir.replace(eval_dir)
    except Exception:
        print(f"build failed; unpublished files retained at {temp_dir}",
              file=sys.stderr)
        raise
    print(f"published immutable evaluation bundle -> {eval_dir}")
    return eval_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", required=True,
        help="run directory under pipeline_b/out; its eval child must be new")
    args = parser.parse_args()
    build(args.run_dir)


if __name__ == "__main__":
    main()
