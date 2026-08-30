"""Build the A1 gallery: crust-plate affinity sweep into out/a1_affinity/.

    py -3.14 scripts/make_a1_affinity.py

Sheets:
  _affinity.png  4 seeds x affinity {0, 0.65, 1.0} — same seed per row,
                 so only nucleus centers move (kernel shapes are drawn
                 from an unchanged stream); ablation pair is column 1 vs 2
  _plates.png    plates view for the same seeds (interiority shading):
                 the mosaic the continents anchored to, beside the result
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mapgen import pipeline, render, report  # noqa: E402

OUT = os.path.join("out", "a1_affinity")
os.makedirs(OUT, exist_ok=True)

SEEDS = [1, 3, 5, 7]
LEVELS = [0.0, 0.65, 1.0]
BASE = {"cell_size_km": 10.0}


def main():
    grid = []
    plates = []
    for s in SEEDS:
        for a in LEVELS:
            ctl = dict(BASE, crust_plate_affinity=a)
            w = pipeline.generate(s, ctl, 384)
            img = render.hypsometric(w)
            stem = f"seed{s}_a{int(a * 100):03d}"
            render.save_png(img, os.path.join(OUT, stem + ".png"), w)
            report.write(w, os.path.join(OUT, stem + ".json"))
            grid.append((f"s{s} a={a:g}", img))
            if a == LEVELS[-1]:
                pimg = render.render_view(w, "plates")
                render.save_png(pimg, os.path.join(OUT, f"seed{s}_plates.png"),
                                w)
                plates.append((f"s{s} plates", pimg))
            print(f"seed {s} a={a:g} done")
    render.contact_sheet(grid, cols=3, thumb=340).save(
        os.path.join(OUT, "_affinity.png"))
    render.contact_sheet(plates, cols=2, thumb=380).save(
        os.path.join(OUT, "_plates.png"))
    print(f"gallery: {OUT}")


if __name__ == "__main__":
    main()
