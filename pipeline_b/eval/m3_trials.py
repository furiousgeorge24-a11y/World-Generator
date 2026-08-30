"""M3 2AFC imposter trials (EVAL.md layer 3, type 1) — same
construction as the M2 baseline run (comparability of the yardstick),
new candidate seeds through the M3 engine.
  out/m3/eval/trials/…  + key + contact sheet
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.elevation import coarse_elevation
from engine.erosion import run_erosion
from engine.registry import make_config
from engine.render_map import render_map_view
from engine.surface import sample_map
from engine.tectonics import build_structure
from eval.m2_trials import REF_IDS, bland, pair, ref_crop

OUT = ROOT / "out" / "m3" / "eval"
TRIALS = OUT / "trials"
KEY = OUT / "key"
CAND_SEEDS = [3, 7, 23, 31, 51, 63, 77, 88]
SIZE = 1024
N_CALIB = 4


def candidate_png(seed):
    cfg = make_config({})
    s = build_structure(seed, cfg)
    ce = coarse_elevation(s, cfg, seed)
    er = run_erosion(s, ce, cfg, seed)
    m = sample_map(s, ce, er, cfg, seed, SIZE)
    return render_map_view(m, "hypsometric",
                           river_density=cfg.river_density)


def main():
    TRIALS.mkdir(parents=True, exist_ok=True)
    KEY.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260830)

    cands = {sd: candidate_png(sd) for sd in CAND_SEEDS}
    print(f"rendered {len(cands)} candidates")

    key = []
    specs = []
    for i in range(len(CAND_SEEDS)):
        specs.append(("cand", REF_IDS[i % len(REF_IDS)], CAND_SEEDS[i]))
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
            ims = (ref_im, cand_im) if ref_side == "A" \
                else (cand_im, ref_im)
            key.append({"trial": t + 1, "kind": "ref_vs_cand",
                        "ref": f"ref{p1}@{rx},{ry}", "cand_seed": p2,
                        "ref_side": ref_side})
        else:
            im1, loc1 = ref_crop(rng, p1)
            im2, loc2 = ref_crop(rng, p2)
            ims = (im1, im2)
            key.append({"trial": t + 1, "kind": "calibration",
                        "ref_A": f"ref{p1}@{loc1[0]},{loc1[1]}",
                        "ref_B": f"ref{p2}@{loc2[0]},{loc2[1]}"})
        pair(*ims).save(TRIALS / f"trial_{t + 1:02d}.png")

    (KEY / "m3_trials_key.json").write_text(json.dumps(key, indent=1))
    src = (ROOT / "out" / "m2" / "eval" / "trials"
           / "instructions.md").read_text()
    (TRIALS / "instructions.md").write_text(src)

    tiles = [Image.open(TRIALS / f"trial_{t:02d}.png").resize((640, 336))
             for t in range(1, len(specs) + 1)]
    sheet = Image.new("RGB", (2 * 648, ((len(tiles) + 1) // 2) * 344),
                      (12, 12, 14))
    for i, tl in enumerate(tiles):
        sheet.paste(tl, (8 + (i % 2) * 648, 4 + (i // 2) * 344))
    sheet.save(OUT / "trials_contact.png")
    print(f"built {len(specs)} trials -> {TRIALS}")


if __name__ == "__main__":
    main()
