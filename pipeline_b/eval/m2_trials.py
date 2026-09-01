"""M2 baseline 2AFC imposter trials (EVAL.md layer 3, type 1).

Construction per the S4/S4b lessons codified in EVAL.md:
feature-targeted crops (bland-tile filter), palette preserved (the
candidate renders through its own stepped ramp), ref-vs-ref
calibration arms, seeded L/R shuffle, answer key stored OUTSIDE the
trials directory. Output:
  out/m2/eval/trials/trial_XX.png  + instructions.md   (judges see this)
  out/m2/eval/key/m2_trials_key.json                   (they never do)
  out/m2/eval/trials_contact.png                       (orchestrator QA)
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.elevation import coarse_elevation
from engine.erosion import run_erosion
from engine.registry import make_config
from engine.render_map import render_map_view
from engine.surface import sample_map
from engine.tectonics import build_structure

OUT = ROOT / "out" / "m2" / "eval"
TRIALS = OUT / "trials"
KEY = OUT / "key"
REFS = ROOT.parent / "examples"
REF_IDS = [1, 2, 6, 9, 10, 14]          # support 1024^2 crops (s4b)
CAND_SEEDS = [3, 7, 23, 31, 51, 63, 77, 88]
SIZE = 1024
N_CAND_TRIALS = 8
N_CALIB = 4


def candidate_png(seed):
    # tracks the current engine (the M2 run's own images are archived
    # under out/m2/eval/ and never regenerated)
    cfg = make_config({})
    s = build_structure(seed, cfg)
    ce = coarse_elevation(s, cfg, seed)
    er = run_erosion(s, ce, cfg, seed)
    m = sample_map(s, ce, er, cfg, seed, SIZE)
    return render_map_view(m, "hypsometric",
                           river_density=cfg.river_density)


def crop_metrics(im):
    """Return the calibrated S4b crop-quality measurements."""
    g = np.asarray(im.convert("L"), np.float64)
    gy, gx = np.gradient(g)
    grad = np.hypot(gy, gx).mean()
    edge_grad = (np.abs(np.diff(g, axis=0)).mean()
                 + np.abs(np.diff(g, axis=1)).mean())
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "gradient": float(grad),
        "edge_gradient": float(edge_grad),
        "near_black_fraction": float((g < 12.0).mean()),
    }


def bland(im):
    """True if a reference crop is not a valid formation stimulus.

    The near-black fraction comes from the repaired S4b yardstick. The
    standard-deviation threshold is retained. The gradient floor is
    calibrated below the accepted low-contrast ref10/ref14 crops; the
    3% near-black ceiling rejects the documented dim ref14 crop while a
    nearby judgeable ref14 crop remains a known positive. Candidate
    blandness is never filtered: it is legitimate evidence.
    """
    q = crop_metrics(im)
    return (q["mean"] < 20.0
            or q["std"] < 22.0
            or q["gradient"] < 1.2
            or q["near_black_fraction"] > 0.03)


def ref_crop(rng, rid):
    im = Image.open(REFS / f"ref{rid}.png").convert("RGB")
    w, h = im.size
    best = None
    best_score = -np.inf
    for _ in range(80):
        x = int(rng.integers(0, w - SIZE + 1))
        y = int(rng.integers(0, h - SIZE + 1))
        c = im.crop((x, y, x + SIZE, y + SIZE))
        q = crop_metrics(c)
        score = (q["std"] + 8.0 * q["edge_gradient"]
                 - 500.0 * max(0.0,
                               q["near_black_fraction"] - 0.03))
        if score > best_score:
            best = (c, (x, y), q)
            best_score = score
        if not bland(c):
            return c, (x, y)
    _, loc, q = best
    raise RuntimeError(
        f"ref{rid} produced no valid {SIZE}px crop after 80 attempts; "
        f"best at {loc} had metrics {q}")


def pair(a, b):
    gap = 24
    im = Image.new("RGB", (SIZE * 2 + gap * 3, SIZE + gap * 2 + 26),
                   (30, 30, 34))
    im.paste(a, (gap, gap + 26))
    im.paste(b, (SIZE + gap * 2, gap + 26))
    d = ImageDraw.Draw(im)
    d.text((gap + SIZE // 2, 8), "A", fill=(235, 235, 235))
    d.text((SIZE + gap * 2 + SIZE // 2, 8), "B", fill=(235, 235, 235))
    return im


def main():
    TRIALS.mkdir(parents=True, exist_ok=True)
    KEY.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260829)

    cands = {sd: candidate_png(sd) for sd in CAND_SEEDS}
    print(f"rendered {len(cands)} candidates")

    key = []
    specs = []
    ref_pool = list(REF_IDS)
    for i in range(N_CAND_TRIALS):
        rid = ref_pool[i % len(ref_pool)]
        specs.append(("cand", rid, CAND_SEEDS[i]))
    for i in range(N_CALIB):
        r1, r2 = rng.choice(REF_IDS, 2, replace=False)
        specs.append(("calib", int(r1), int(r2)))
    order = rng.permutation(len(specs))

    for t, si in enumerate(order):
        kind, p1, p2 = specs[si]
        if kind == "cand":
            ref_im, (rx, ry) = ref_crop(rng, p1)
            cand_im = cands[p2]
            ref_side = "A" if rng.random() < 0.5 else "B"
            ims = (ref_im, cand_im) if ref_side == "A" else (cand_im, ref_im)
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

    (KEY / "m2_trials_key.json").write_text(json.dumps(key, indent=1))

    (TRIALS / "instructions.md").write_text("""# Terrain map discrimination trials

Each `trial_XX.png` shows two terrain-map tiles, A (left) and B
(right). In every pair, at least one tile comes from a curated
reference set of terrain maps whose quality is considered excellent.
The other tile may come from the same reference set or from a
different map source.

For each trial, judge which tile is more plausibly from the excellent
reference set, purely on terrain formation quality.

Notes:
- The two sources use different colour palettes and different
  colour-band conventions. Hue or palette difference alone is NOT
  evidence; judge landform shapes, coherence of coasts/shelves/ranges,
  drainage and texture plausibility.
- Some reference-set images carry day/night darkening, projection
  stretching, or land running off the image edge; none of that is
  evidence either way.
- Tiny bright dots on high plateaus in reference-style images are
  floor lakes, not noise. Small volcanic islands can look like
  cross/star marks at pixel scale; that is not a stamp defect.
- If both tiles look reference-grade (or neither does), still pick the
  more plausible one and give low confidence.

Answer in JSON, one object per trial:
`{"trial": N, "pick": "A"|"B", "confidence": 1-5, "evidence": "one or
two sentences naming specific locations/features"}`
Wrap all objects in a JSON list. Do not consult anything outside the
trials directory.
""")

    tiles = [Image.open(TRIALS / f"trial_{t:02d}.png").resize((640, 336))
             for t in range(1, len(specs) + 1)]
    sheet = Image.new("RGB", (2 * 648, ((len(tiles) + 1) // 2) * 344),
                      (12, 12, 14))
    for i, tl in enumerate(tiles):
        sheet.paste(tl, (8 + (i % 2) * 648, 4 + (i // 2) * 344))
    sheet.save(OUT / "trials_contact.png")
    print(f"built {len(specs)} trials -> {TRIALS}")
    print(f"key -> {KEY / 'm2_trials_key.json'}")


if __name__ == "__main__":
    main()
