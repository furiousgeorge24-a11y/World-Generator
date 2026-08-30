"""Build the K3 mass-balance gallery into out/k3_carve/.

    py -3.14 scripts/make_k3_carve.py

Sheets:
  _defaults.png    3 seeds at K3 defaults (grain 0.5, deposition 0.6,
                   dissection 0.5, flood 250)
  _deposition.png  deposition {0, 0.3, 0.6, 0.9} — 0 is the ablation
  _grain.png       plains_grain {0, 0.5, 1}
  _dissection.png  lowland_dissection {0, 0.5, 1} — 0 = old hard gate
  _coasts.png      zoomed coastal crops: the ria / drowned-valley check
                   that K1 deferred to K3
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image  # noqa: E402

from mapgen import pipeline, render, report  # noqa: E402
from mapgen.boundaries import _fft_gauss  # noqa: E402

OUT = os.path.join("out", "k3_carve")
os.makedirs(OUT, exist_ok=True)

BASE = {"cell_size_km": 8.0}


def tile(seed, over, label, stem):
    w = pipeline.generate(seed, dict(BASE, **over), 512)
    img = render.hypsometric(w)
    render.save_png(img, os.path.join(OUT, stem + ".png"), w)
    report.write(w, os.path.join(OUT, stem + ".json"))
    print(f"{stem} done")
    return label, img


def main():
    sheet = [tile(s, {}, f"seed {s}", f"seed{s}") for s in (1, 3, 5)]
    render.contact_sheet(sheet, cols=3, thumb=380).save(
        os.path.join(OUT, "_defaults.png"))

    dep = [tile(5, {"deposition": v}, f"deposition={v:g}",
                f"dep_{int(v * 10):02d}") for v in (0.0, 0.3, 0.6, 0.9)]
    render.contact_sheet(dep, cols=4, thumb=340).save(
        os.path.join(OUT, "_deposition.png"))

    gr = [tile(3, {"plains_grain": v}, f"grain={v:g}",
               f"grain_{int(v * 10):02d}") for v in (0.0, 0.5, 1.0)]
    render.contact_sheet(gr, cols=3, thumb=380).save(
        os.path.join(OUT, "_grain.png"))

    ld = [tile(3, {"lowland_dissection": v}, f"dissection={v:g}",
               f"ld_{int(v * 10):02d}") for v in (0.0, 0.5, 1.0)]
    render.contact_sheet(ld, cols=3, thumb=380).save(
        os.path.join(OUT, "_dissection.png"))

    # coastal zooms at max drowned-shelf dissection
    w = pipeline.generate(5, dict(BASE, cell_size_km=6.0), 768)
    img = render.hypsometric(w)
    render.save_png(img, os.path.join(OUT, "coast_full.png"), w)
    e = w["elevation"].astype(np.float64)
    flood = float(w.controls["flood_rise_m"])
    shelf = (e > -flood) & (e < 0.0)
    gy, gx = np.gradient(e)
    score = _fft_gauss(np.hypot(gy, gx) * shelf, 24.0)
    tiles, half = [], 140
    for i in range(2):
        r, ccol = np.unravel_index(int(np.argmax(score)), score.shape)
        r = int(np.clip(r, half, 768 - half))
        ccol = int(np.clip(ccol, half, 768 - half))
        score[max(r - 220, 0):r + 220, max(ccol - 220, 0):ccol + 220] = -1
        t = img.crop((ccol - half, r - half, ccol + half, r + half))
        t = t.resize((half * 4, half * 4), Image.NEAREST)
        t.save(os.path.join(OUT, f"coast_zoom{i + 1}.png"))
        tiles.append((f"zoom {i + 1} (2x)", t))
    render.contact_sheet(tiles, cols=2, thumb=560).save(
        os.path.join(OUT, "_coasts.png"))
    print(f"gallery: {OUT}")


if __name__ == "__main__":
    main()
