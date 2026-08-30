"""Remaining-ruling sheets, trim-review style (ON / OFF-or-old / delta).

    py -3.14 scripts/make_rulings_review.py

Covers what the k_review sitting still has to rule: era belts (open
q. 1 — the perf-expensive one), the plateau_tendency and
lowland_dissection defaults (0 / default / 1 ladders + delta), and the
five tier-1 pick confirmations as new-vs-old-default pairs. All at
1024^2 / 4 km on shipping defaults. Output: out/rulings_review/.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from mapgen import pipeline, render  # noqa: E402
from mapgen.boundaries import _fft_gauss  # noqa: E402

OUT = os.path.join("out", "rulings_review")
os.makedirs(OUT, exist_ok=True)

BASE = {"cell_size_km": 4.0}
SIZE = 1024


def gen(seed, over):
    return pipeline.generate(seed, dict(BASE, **over), SIZE)


def heat(a):
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


def spot(d, fp, cap):
    fp = fp / max(float(fp.max()), 1e-9)
    sm = _fft_gauss(np.minimum(np.abs(d), cap) * fp, 6.0)
    return np.unravel_index(int(np.argmax(sm)), sm.shape)


def sheet(name, tiles, cols):
    render.contact_sheet(tiles, cols=cols, thumb=440).save(
        os.path.join(OUT, name))
    print(name, "done")


def main():
    on = {s: gen(s, {}) for s in (3, 5)}
    img_on = {s: render.hypsometric(on[s]) for s in (3, 5)}
    print("default worlds done")

    # --- era belts (open q. 1): the expensive ruling --------------------
    best = None
    for s in (3, 5):
        off = gen(s, {"era_count": 1})
        d = (on[s]["elevation"].astype(np.float64)
             - off["elevation"].astype(np.float64))
        fp = np.abs(on[s]["tect_era_belt"].astype(np.float64))
        fpn = fp / max(float(fp.max()), 1e-9)
        score = float(_fft_gauss(np.minimum(np.abs(d), 900.0) * fpn,
                                 6.0).max())
        if best is None or score > best[0]:
            best = (score, s, off, d, fp)
        print(f"era: seed {s} scored")
    _, s, off, d, fp = best
    r, c = spot(d, fp, 900.0)
    sheet("_era_count.png", [
        (f"era_count=2 (default) - seed {s}", crop(img_on[s], r, c)),
        ("era_count=1 (ancient belts off)",
         crop(render.hypsometric(off), r, c)),
        ("where it acts (|delta elevation|)", crop(heat(np.abs(d)), r, c)),
    ], 3)

    # --- plateau_tendency default (0 / 0.5 / 1 + delta) -----------------
    p0 = gen(5, {"plateau_tendency": 0.0})
    p1 = gen(5, {"plateau_tendency": 1.0})
    d = (p1["elevation"].astype(np.float64)
         - p0["elevation"].astype(np.float64))
    fp = np.abs(p1["tect_plateau"].astype(np.float64))
    r, c = spot(d, fp, 1500.0)
    sheet("_plateau_tendency.png", [
        ("plateau_tendency=0 (peaks only)",
         crop(render.hypsometric(p0), r, c)),
        ("plateau_tendency=0.5 (default)", crop(img_on[5], r, c)),
        ("plateau_tendency=1 (max plateaus)",
         crop(render.hypsometric(p1), r, c)),
        ("where it acts (1 vs 0)", crop(heat(np.abs(d)), r, c)),
    ], 4)

    # --- lowland_dissection default (0 / 0.5 / 1 + delta) ---------------
    l0 = gen(5, {"lowland_dissection": 0.0})
    l1 = gen(5, {"lowland_dissection": 1.0})
    e5 = on[5]["elevation"].astype(np.float64)
    land5 = (e5 >= 0.0).astype(np.float64)
    # _fft_gauss is unnormalized: divide by the blur of ones to get a
    # true 0..1 neighborhood-land fraction before thresholding
    inter = (_fft_gauss(land5, 8.0)
             / np.maximum(_fft_gauss(np.ones_like(land5), 8.0), 1e-9))
    fp_int = (((e5 > 0.0) & (e5 < 300.0))
              & (inter > 0.85)).astype(np.float64)
    d = (l1["elevation"].astype(np.float64)
         - l0["elevation"].astype(np.float64)) * land5
    r, c = spot(d, fp_int, 150.0)
    sheet("_lowland_dissection.png", [
        ("lowland_dissection=0 (hard gate)",
         crop(render.hypsometric(l0), r, c)),
        ("lowland_dissection=0.5 (default)", crop(img_on[5], r, c)),
        ("lowland_dissection=1 (max plains carving)",
         crop(render.hypsometric(l1), r, c)),
        ("where it acts (1 vs 0)", crop(heat(np.abs(d)), r, c)),
    ], 4)

    # --- tier-1 confirmations: new default vs old -----------------------
    w10 = gen(5, {"plate_count": 10})
    sheet("_plate_count.png", [
        ("plate_count=6 (new default) - seed 5", img_on[5]),
        ("plate_count=10 (old default) - full recomposition, no delta",
         render.hypsometric(w10)),
    ], 2)

    on[5].controls["render_quantize"] = 0
    smooth = render.hypsometric(on[5])
    on[5].controls["render_quantize"] = 12
    sheet("_render_quantize.png", [
        ("render_quantize=12 (new default)", img_on[5]),
        ("render_quantize=0 (old default: smooth) - render-only", smooth),
    ], 2)

    # lake threshold is LATE/visibility-class: elevation is untouched by
    # design, so the comparison is on the rendered images
    w_lake = gen(5, {"lake_min_depth_m": 0.8})
    img_lake = render.hypsometric(w_lake)
    dd = np.abs(np.asarray(img_on[5], dtype=np.float64)
                - np.asarray(img_lake, dtype=np.float64)).mean(axis=2)
    r, c = np.unravel_index(int(np.argmax(_fft_gauss(dd, 5.0))), dd.shape)
    sheet("_lake_min_depth.png", [
        ("lake_min_depth_m=6 (new default) - seed 5",
         crop(img_on[5], r, c)),
        ("lake_min_depth_m=0.8 (old default)", crop(img_lake, r, c)),
        ("where lakes appear/vanish (render diff)", crop(heat(dd), r, c)),
    ], 3)

    for name, knob, new, old, cap in (
            ("_deposition.png", "deposition", 0.8, 0.6, 150.0),
            ("_plains_grain.png", "plains_grain", 0.7, 0.5, 100.0)):
        w_old = gen(5, {knob: old})
        d = (on[5]["elevation"].astype(np.float64)
             - w_old["elevation"].astype(np.float64)) * land5
        r, c = spot(d, fp_int, cap)
        sheet(name, [
            (f"{knob}={new:g} (new default) - seed 5",
             crop(img_on[5], r, c)),
            (f"{knob}={old:g} (old default)",
             crop(render.hypsometric(w_old), r, c)),
            ("where it acts (|delta elevation|)",
             crop(heat(np.abs(d)), r, c)),
        ], 3)

    print(f"rulings review: {OUT}")


if __name__ == "__main__":
    main()
