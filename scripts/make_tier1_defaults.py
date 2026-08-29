"""Tier-1 default retune evidence, behind the 2026-08-28 canon comparison.

    py -3.14 scripts/make_tier1_defaults.py

Sweeps the five authorized default candidates (plate scale, quantize,
lake threshold, deposition, plains grain) so the picks are made from
images, not arithmetic. Output: out/tier1_defaults/.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mapgen import pipeline, render  # noqa: E402

OUT = os.path.join("out", "tier1_defaults")
os.makedirs(OUT, exist_ok=True)

SEED = 5
BASE = {"cell_size_km": 8.0}


def gen(seed, over, size=512):
    return pipeline.generate(seed, dict(BASE, **over), size)


def save(w, stem):
    img = render.render_view(w, "hypsometric")
    render.save_png(img, os.path.join(OUT, stem + ".png"), w)
    return img


def sheet(name, tiles, cols, thumb=420):
    render.contact_sheet(tiles, cols=cols, thumb=thumb).save(
        os.path.join(OUT, name))
    print(name, "done")


def main():
    # plate scale ladder
    ts = []
    for n in (4, 6, 8, 10):
        ts.append((f"plate_count={n}",
                   save(gen(SEED, {"plate_count": n}), f"plates_{n:02d}")))
        print(f"plate_count {n} done")
    sheet("_plates.png", ts, 2)

    # variety de-risk: proposed count across seeds
    ts = []
    for s in (1, 3, 5, 7):
        ts.append((f"seed {s}, plate_count=6",
                   save(gen(s, {"plate_count": 6}), f"plates6_seed{s}")))
        print(f"plates6 seed {s} done")
    sheet("_plates6_seeds.png", ts, 2)

    # lake visibility threshold
    ts = []
    for d in (0.8, 3.0, 6.0, 10.0):
        ts.append((f"lake_min_depth_m={d:g}",
                   save(gen(SEED, {"lake_min_depth_m": d}), f"lakes_{d:g}")))
        print(f"lakes {d:g} done")
    sheet("_lakes.png", ts, 2)

    # deposition strength
    ts = []
    for v in (0.6, 0.8, 1.0):
        ts.append((f"deposition={v:g}",
                   save(gen(SEED, {"deposition": v}), f"dep_{v:g}")))
        print(f"deposition {v:g} done")
    sheet("_deposition.png", ts, 3, thumb=380)

    # plains grain
    ts = []
    for v in (0.5, 0.7, 0.9):
        ts.append((f"plains_grain={v:g}",
                   save(gen(SEED, {"plains_grain": v}), f"grain_{v:g}")))
        print(f"grain {v:g} done")
    sheet("_grain.png", ts, 3, thumb=380)

    # quantize is render-class: one world, four renders
    w = gen(3, {})
    ts = []
    for q in (0, 10, 12, 16):
        w.controls["render_quantize"] = q
        ts.append((f"quantize={q}" if q else "smooth", save(w, f"q_{q:02d}")))
    sheet("_quantize.png", ts, 2)

    print(f"tier1 evidence: {OUT}")


if __name__ == "__main__":
    main()
