"""Build M3-family blind critique panels for an explicit run directory."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import VERSION
from eval.m2_trials import crop_metrics, ref_crop
from eval.m3_trials import candidate_png

ARCHIVED_OUT = (ROOT / "out" / "m3" / "eval").resolve()
PROMPT = ROOT / "eval" / "prompts" / "critique_v2.md"
CAND_SEEDS = [19, 40, 101]
CANON = [1, 6]
DUP_OF = 40
SIZE = 1024
RNG_SEED = 30082026


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_out(out):
    out = Path(out).resolve()
    if out == ARCHIVED_OUT:
        raise ValueError(
            "refusing to overwrite the archived out/m3/eval bundle")
    return out


def build(out):
    """Build panels into ``out`` and return run-manifest provenance."""
    out = _safe_out(out)
    panels = out / "panels"
    key_dir = out / "key"
    key_path = key_dir / "m3_panels_key.json"
    build_path = key_dir / "m3_panels_build.json"
    claimed = (panels, key_path, build_path)
    existing = [str(p) for p in claimed if p.exists()]
    if existing:
        raise FileExistsError(
            "panel build is fail-closed; existing targets: "
            + ", ".join(existing))

    panels.mkdir(parents=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    entries = [("cand", seed) for seed in CAND_SEEDS]
    entries += [("canon", ref_id) for ref_id in CANON]
    entries.append(("dup", DUP_OF))
    order = rng.permutation(len(entries))

    cache = {}
    key = []
    for slot, ei in enumerate(order):
        kind, ident = entries[ei]
        if kind in ("cand", "dup"):
            if ident not in cache:
                cache[ident] = candidate_png(ident)
            image = cache[ident]
            key.append({
                "panel": slot + 1,
                "kind": ("candidate" if kind == "cand"
                         else "duplicate_of_candidate"),
                "seed": ident,
            })
        else:
            image, (x, y) = ref_crop(rng, ident)
            key.append({
                "panel": slot + 1,
                "kind": "canon",
                "ref": f"ref{ident}@{x},{y}",
                "ref_crop_metrics": crop_metrics(image),
            })
        image.save(panels / f"panel_{slot + 1:02d}.png")

    key_path.write_text(json.dumps(key, indent=2) + "\n",
                        encoding="utf-8")
    prompt_bytes = PROMPT.read_bytes()
    (panels / "rubric.md").write_bytes(prompt_bytes)

    build_record = {
        "builder": "eval/m3_panels.py",
        "engine_version": VERSION,
        "prompt_id": "critique_v2",
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "rng_seed": RNG_SEED,
        "candidate_seeds": CAND_SEEDS,
        "canon_reference_ids": CANON,
        "duplicate_candidate_seed": DUP_OF,
        "panel_px": SIZE,
        "stimulus_sha256": {
            p.name: _sha256(p)
            for p in sorted(panels.glob("panel_*.png"))
        },
    }
    build_path.write_text(json.dumps(build_record, indent=2) + "\n",
                          encoding="utf-8")
    print(f"built {len(entries)} panels -> {panels}")
    return build_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", required=True,
        help="new evaluation directory; historical out/m3/eval is rejected")
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    build(out)


if __name__ == "__main__":
    main()
