"""Build M3-family blind 2AFC trials for an explicit run directory.

The historical ``out/m3/eval`` bundle contains judged 0.3.0-m3
evidence and is intentionally rejected as an output target. Run 1 is
built under ``out/m3_run1/eval`` by ``eval/build_m3_run.py``.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import VERSION
from engine.elevation import coarse_elevation
from engine.erosion import run_erosion
from engine.registry import make_config
from engine.render_map import render_map_view
from engine.surface import sample_map
from engine.tectonics import FRAME_KM, build_structure
from eval.m2_trials import REF_IDS, crop_metrics, pair, ref_crop

ARCHIVED_OUT = (ROOT / "out" / "m3" / "eval").resolve()
PROMPT = ROOT / "eval" / "prompts" / "2afc_v2.md"
CAND_SEEDS = [3, 7, 23, 31, 51, 63, 77, 88]
SIZE = 1024
N_CALIB = 4
RNG_SEED = 20260830


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


def candidate_png(seed):
    cfg = make_config({})
    s = build_structure(seed, cfg)
    ce = coarse_elevation(s, cfg, seed)
    er = run_erosion(s, ce, cfg, seed)
    m = sample_map(s, ce, er, cfg, seed, SIZE)
    return render_map_view(m, "hypsometric",
                           river_density=cfg.river_density)


def build(out):
    """Build trials into ``out`` and return run-manifest provenance."""
    out = _safe_out(out)
    trials = out / "trials"
    key_dir = out / "key"
    key_path = key_dir / "m3_trials_key.json"
    contact_path = out / "trials_contact.png"
    build_path = key_dir / "m3_trials_build.json"
    claimed = (trials, key_path, contact_path, build_path)
    existing = [str(p) for p in claimed if p.exists()]
    if existing:
        raise FileExistsError(
            "trial build is fail-closed; existing targets: "
            + ", ".join(existing))

    trials.mkdir(parents=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    cands = {seed: candidate_png(seed) for seed in CAND_SEEDS}
    print(f"rendered {len(cands)} candidates")

    key = []
    specs = []
    for i, seed in enumerate(CAND_SEEDS):
        specs.append(("cand", REF_IDS[i % len(REF_IDS)], seed))
    for _ in range(N_CALIB):
        r1, r2 = rng.choice(REF_IDS, 2, replace=False)
        specs.append(("calib", int(r1), int(r2)))
    order = rng.permutation(len(specs))

    for t, si in enumerate(order):
        kind, p1, p2 = specs[si]
        if kind == "cand":
            ref_im, (rx, ry) = ref_crop(rng, p1)
            cand_im = cands[p2]
            ref_side = "A" if rng.random() < 0.5 else "B"
            ims = ((ref_im, cand_im) if ref_side == "A"
                   else (cand_im, ref_im))
            key.append({
                "trial": t + 1,
                "kind": "ref_vs_cand",
                "ref": f"ref{p1}@{rx},{ry}",
                "ref_crop_metrics": crop_metrics(ref_im),
                "cand_seed": p2,
                "ref_side": ref_side,
            })
        else:
            im1, loc1 = ref_crop(rng, p1)
            im2, loc2 = ref_crop(rng, p2)
            ims = (im1, im2)
            key.append({
                "trial": t + 1,
                "kind": "calibration",
                "ref_A": f"ref{p1}@{loc1[0]},{loc1[1]}",
                "ref_B": f"ref{p2}@{loc2[0]},{loc2[1]}",
                "ref_A_crop_metrics": crop_metrics(im1),
                "ref_B_crop_metrics": crop_metrics(im2),
            })
        pair(*ims).save(trials / f"trial_{t + 1:02d}.png")

    key_path.write_text(json.dumps(key, indent=2) + "\n",
                        encoding="utf-8")
    prompt_bytes = PROMPT.read_bytes()
    (trials / "instructions.md").write_bytes(prompt_bytes)

    tiles = [
        Image.open(trials / f"trial_{t:02d}.png").resize((640, 336))
        for t in range(1, len(specs) + 1)
    ]
    sheet = Image.new("RGB", (2 * 648, ((len(tiles) + 1) // 2) * 344),
                      (12, 12, 14))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, (8 + (i % 2) * 648, 4 + (i // 2) * 344))
    sheet.save(contact_path)

    build_record = {
        "builder": "eval/m3_trials.py",
        "engine_version": VERSION,
        "prompt_id": "2afc_v2",
        "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "rng_seed": RNG_SEED,
        "candidate_seeds": CAND_SEEDS,
        "reference_ids": REF_IDS,
        "candidate_arms": len(CAND_SEEDS),
        "calibration_arms": N_CALIB,
        "tile_px": SIZE,
        "candidate_frame_km": FRAME_KM,
        "candidate_km_per_px": FRAME_KM / SIZE,
        "reference_km_per_px": None,
        "scale_matching": (
            "unresolved: reference screenshots lack physical-scale metadata"),
        "stimulus_sha256": {
            p.name: _sha256(p)
            for p in sorted(trials.glob("trial_*.png"))
        },
    }
    build_path.write_text(json.dumps(build_record, indent=2) + "\n",
                          encoding="utf-8")
    print(f"built {len(specs)} trials -> {trials}")
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
