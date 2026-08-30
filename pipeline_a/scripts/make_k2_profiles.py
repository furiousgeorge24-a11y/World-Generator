"""Build the K2 profile-model gallery into out/k2_profiles/.

    py -3.14 scripts/make_k2_profiles.py

Sheets:
  _belts.png    3 seeds at defaults — belt anatomy at map scale
  _plateau.png  plateau_tendency {0, 0.5, 1} on a collision-rich seed;
                0 is the ablation (peaks only, no rim-enclosed plateaus)
  _flexure.png  outer_rise {0, 0.6, 1.2}: outer rise + foreland together
  _anatomy.png  zoomed crops centered on the strongest belts — the
                rim / floor / apron / foreland cross-section check
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image  # noqa: E402

from mapgen import pipeline, render, report  # noqa: E402
from mapgen.boundaries import _fft_gauss  # noqa: E402

OUT = os.path.join("out", "k2_profiles")
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
    belts = [tile(s, {}, f"seed {s}", f"seed{s}") for s in (1, 3, 5)]
    render.contact_sheet(belts, cols=3, thumb=380).save(
        os.path.join(OUT, "_belts.png"))

    pt = [tile(3, {"plateau_tendency": v}, f"plateau={v:g}",
               f"pt_{int(v * 10):02d}") for v in (0.0, 0.5, 1.0)]
    render.contact_sheet(pt, cols=3, thumb=380).save(
        os.path.join(OUT, "_plateau.png"))

    fx = [tile(5, {"outer_rise": v}, f"flexure={v:g}",
               f"fx_{int(v * 10):02d}") for v in (0.0, 0.6, 1.0)]
    render.contact_sheet(fx, cols=3, thumb=380).save(
        os.path.join(OUT, "_flexure.png"))

    # anatomy zooms centered on the highest-relief belts
    w = pipeline.generate(3, dict(BASE, cell_size_km=6.0), 768)
    img = render.hypsometric(w)
    render.save_png(img, os.path.join(OUT, "anatomy_full.png"), w)
    e = w["elevation"].astype(np.float64)
    score = _fft_gauss(np.maximum(e - 1800.0, 0.0), 20.0)
    tiles, half = [], 150
    for i in range(2):
        r, ccol = np.unravel_index(int(np.argmax(score)), score.shape)
        r = int(np.clip(r, half, 768 - half))
        ccol = int(np.clip(ccol, half, 768 - half))
        score[max(r - 260, 0):r + 260, max(ccol - 260, 0):ccol + 260] = -1
        t = img.crop((ccol - half, r - half, ccol + half, r + half))
        t = t.resize((half * 4, half * 4), Image.NEAREST)
        t.save(os.path.join(OUT, f"anatomy_zoom{i + 1}.png"))
        tiles.append((f"belt zoom {i + 1} (2x)", t))
    render.contact_sheet(tiles, cols=2, thumb=600).save(
        os.path.join(OUT, "_anatomy.png"))
    print(f"gallery: {OUT}")


if __name__ == "__main__":
    main()
