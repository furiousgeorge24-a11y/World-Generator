"""Build the KR ramp style sheets into out/kr_ramp/.

    py -3.14 scripts/make_kr_ramp.py

Sheets:
  _palettes.png  3 seeds (rows) x 4 palettes (cols) — the pick-a-default
                 sheet; each world generated once, re-rendered per palette
  _quantize.png  canon + canon-crisp x quantize {0, 14}: the banded look
Render-only run: worlds are generated once per seed and re-rendered.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mapgen import pipeline, render, report  # noqa: E402
from mapgen.render import PALETTES  # noqa: E402

OUT = os.path.join("out", "kr_ramp")
os.makedirs(OUT, exist_ok=True)

SEEDS = [1, 3, 5]
BASE = {"cell_size_km": 8.0}


def main():
    grid = []
    worlds = {}
    for s in SEEDS:
        w = pipeline.generate(s, dict(BASE), 512)
        worlds[s] = w
        report.write(w, os.path.join(OUT, f"seed{s}.json"))
        for p, (name, _, _) in sorted(PALETTES.items()):
            w.controls["render_palette"] = p
            img = render.hypsometric(w)
            render.save_png(img, os.path.join(OUT, f"seed{s}_{name}.png"), w)
            grid.append((f"s{s} {name}", img))
        print(f"seed {s} done")
    render.contact_sheet(grid, cols=4, thumb=340).save(
        os.path.join(OUT, "_palettes.png"))

    qgrid = []
    w = worlds[SEEDS[-1]]
    for p in (1, 3):
        for q in (0, 14):
            w.controls["render_palette"] = p
            w.controls["render_quantize"] = q
            img = render.hypsometric(w)
            name = PALETTES[p][0]
            render.save_png(img, os.path.join(OUT, f"q_{name}_{q}.png"), w)
            qgrid.append((f"{name} q={q}", img))
    render.contact_sheet(qgrid, cols=2, thumb=420).save(
        os.path.join(OUT, "_quantize.png"))
    print(f"style sheets: {OUT}")


if __name__ == "__main__":
    main()
