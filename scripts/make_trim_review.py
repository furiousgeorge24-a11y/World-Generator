"""Trim-suspect comparison sheets (the ledger's pre-registered marginals).

    py -3.14 scripts/make_trim_review.py

For each suspect knob: ON (shipping default) vs OFF (=0) vs a
delta-elevation heat panel, cropped where the feature acts hardest, at
1024^2 / 4 km (several suspects were pre-registered invisible below
1024). Output: out/trim_review/_<knob>.png, one sheet per verdict.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from mapgen import pipeline, render  # noqa: E402
from mapgen.boundaries import _fft_gauss  # noqa: E402

OUT = os.path.join("out", "trim_review")
os.makedirs(OUT, exist_ok=True)

BASE = {"cell_size_km": 4.0}
SIZE = 1024
SEEDS = (3, 5)

# knob -> (delta cap in m, footprint field). The cap keeps chaotic
# knock-on flips (a lake spilling differently) from hijacking the crop;
# the footprint keeps the crop on the feature's own territory.
FEATURES = {
    "outer_rise": (400.0, lambda w: (
        np.abs(w["tect_rise"].astype(np.float64))
        + np.abs(w["tect_foreland"].astype(np.float64)))),
    "ridge_segmentation": (600.0, lambda w: (
        (w["elevation"].astype(np.float64) < -1500.0).astype(np.float64))),
    "backarc_basins": (900.0, lambda w: np.abs(
        w["tect_backarc"].astype(np.float64))),
    "failed_rifts": (300.0, lambda w: np.abs(
        w["tect_graben"].astype(np.float64))),
    "seafloor_fabric": (150.0, lambda w: (
        (w["elevation"].astype(np.float64) < -2800.0).astype(np.float64))),
}


def gen(seed, over):
    return pipeline.generate(seed, dict(BASE, **over), SIZE)


def heat(a):
    """|delta| -> orange-on-navy heat image."""
    a = np.maximum(a, 0.0)
    t = (a / max(float(a.max()), 1e-9)) ** 0.6
    rgb = np.stack([20 + 235 * t, 25 + 190 * t, 60 + 20 * (1 - t)],
                   axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def crop(img, r, c, half=170):
    r = int(np.clip(r, half, SIZE - half))
    c = int(np.clip(c, half, SIZE - half))
    t = img.crop((c - half, r - half, c + half, r + half))
    return t.resize((half * 4, half * 4), Image.NEAREST)


def main():
    on = {s: gen(s, {}) for s in SEEDS}
    on_img = {s: render.hypsometric(on[s]) for s in SEEDS}
    print("ON worlds done")

    for knob, (cap, footprint) in FEATURES.items():
        best = None              # (score, seed, off_world, delta, scoremap)
        for s in SEEDS:
            w_off = gen(s, {knob: 0.0})
            d = (on[s]["elevation"].astype(np.float64)
                 - w_off["elevation"].astype(np.float64))
            fp = footprint(on[s])
            fp = fp / max(float(fp.max()), 1e-9)
            sm = _fft_gauss(np.minimum(np.abs(d), cap) * fp, 6.0)
            score = float(sm.max())
            if best is None or score > best[0]:
                best = (score, s, w_off, d, sm)
            print(f"{knob}: seed {s} footprint-score = {score:,.1f}")
        _, s, w_off, d, sm = best
        r, c = np.unravel_index(int(np.argmax(sm)), sm.shape)
        tiles = [
            (f"{knob} ON (default) - seed {s}", crop(on_img[s], r, c)),
            (f"{knob} OFF (=0)", crop(render.hypsometric(w_off), r, c)),
            ("where it acts (|delta elevation|)", crop(heat(np.abs(d)), r, c)),
        ]
        render.contact_sheet(tiles, cols=3, thumb=440).save(
            os.path.join(OUT, f"_{knob}.png"))
        print(f"{knob}: sheet done (seed {s})")

    # axial valley has no knob (rides divergent painting) — visibility call
    s = max(SEEDS,
            key=lambda x: float(-on[x]["tect_axial"].astype(np.float64).min()))
    ax = -on[s]["tect_axial"].astype(np.float64)
    sm = _fft_gauss(ax, 6.0)
    r, c = np.unravel_index(int(np.argmax(sm)), sm.shape)
    tiles = [
        (f"slow-spread axial valley - seed {s} (no knob: visibility call)",
         crop(on_img[s], r, c)),
        ("tect_axial field (what to look for)", crop(heat(ax), r, c)),
    ]
    render.contact_sheet(tiles, cols=2, thumb=440).save(
        os.path.join(OUT, "_axial_valley.png"))
    print("axial sheet done")
    print(f"trim review: {OUT}")


if __name__ == "__main__":
    main()
