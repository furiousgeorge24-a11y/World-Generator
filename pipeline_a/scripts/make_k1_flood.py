"""Build the K1 drowned-datum gallery into out/k1_flood/.

    py -3.14 scripts/make_k1_flood.py

Sheets:
  _flood.png      3 seeds (rows) x flood_rise_m {0, 60, 120, 250} (cols) —
                  0 is the ablation (no lowstand ever existed)
  _planation.png  wave_planation {0, 0.6, 1.0} at default flood
  _coasts.png     zoomed coastal crops (auto-picked at max shelf
                  dissection) — the ria/drowned-valley evidence
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image  # noqa: E402

from mapgen import pipeline, render, report  # noqa: E402
from mapgen.boundaries import _fft_gauss  # noqa: E402

OUT = os.path.join("out", "k1_flood")
os.makedirs(OUT, exist_ok=True)

SEEDS = [1, 3, 5]
FLOODS = [0.0, 60.0, 120.0, 250.0]
BASE = {"cell_size_km": 8.0}


def main():
    grid = []
    for s in SEEDS:
        for f in FLOODS:
            ctl = dict(BASE, flood_rise_m=f)
            w = pipeline.generate(s, ctl, 512)
            img = render.hypsometric(w)
            stem = f"seed{s}_f{int(f):03d}"
            render.save_png(img, os.path.join(OUT, stem + ".png"), w)
            report.write(w, os.path.join(OUT, stem + ".json"))
            grid.append((f"s{s} flood={int(f)}m", img))
            print(f"seed {s} flood {f:g} done")
    render.contact_sheet(grid, cols=4, thumb=340).save(
        os.path.join(OUT, "_flood.png"))

    pl = []
    for p in (0.0, 0.6, 1.0):
        w = pipeline.generate(5, dict(BASE, wave_planation=p), 512)
        img = render.hypsometric(w)
        render.save_png(img, os.path.join(OUT, f"plan_{int(p * 10):02d}.png"),
                        w)
        pl.append((f"planation={p:g}", img))
        print(f"planation {p:g} done")
    render.contact_sheet(pl, cols=3, thumb=380).save(
        os.path.join(OUT, "_planation.png"))

    # coastal zooms: crops centered where drowned-shelf relief is densest
    w = pipeline.generate(5, dict(BASE, cell_size_km=6.0), 768)
    img = render.hypsometric(w)
    render.save_png(img, os.path.join(OUT, "coast_full.png"), w)
    e = w["elevation"].astype(np.float64)
    flood = float(w.controls["flood_rise_m"])
    shelf = (e > -flood) & (e < 0.0)
    gy, gx = np.gradient(e)
    rough = np.hypot(gy, gx) * shelf
    score = _fft_gauss(rough, 24.0)
    crops = []
    half = 140
    for _ in range(2):
        r, ccol = np.unravel_index(int(np.argmax(score)), score.shape)
        r = int(np.clip(r, half, 768 - half))
        ccol = int(np.clip(ccol, half, 768 - half))
        crops.append((r, ccol))
        score[max(r - 220, 0):r + 220, max(ccol - 220, 0):ccol + 220] = -1
    tiles = []
    for i, (r, ccol) in enumerate(crops):
        t = img.crop((ccol - half, r - half, ccol + half, r + half))
        t = t.resize((half * 4, half * 4), Image.NEAREST)
        t.save(os.path.join(OUT, f"coast_zoom{i + 1}.png"))
        tiles.append((f"zoom {i + 1} (2x)", t))
    render.contact_sheet(tiles, cols=2, thumb=560).save(
        os.path.join(OUT, "_coasts.png"))
    print(f"gallery: {OUT}")


if __name__ == "__main__":
    main()
