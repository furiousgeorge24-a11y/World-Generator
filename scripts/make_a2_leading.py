"""A2 leading-edge crust bias (`active_margin_bias`) — evidence galleries.

    py -3.14 scripts/make_a2_leading.py

out/a2_leading/: _audit.png (plates-view shift markers, bias 0 vs 1),
_sweep.png (bias ladder x seeds — the author's default pick), _seeds.png
(8 seeds at the provisional 0.5 default), _coasts.png (cordilleran-coast
zooms at bias 1, the configuration the knob promises).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from mapgen import pipeline, render, report  # noqa: E402
from mapgen.boundaries import _fft_gauss  # noqa: E402

OUT = os.path.join("out", "a2_leading")
os.makedirs(OUT, exist_ok=True)

BASE = {"cell_size_km": 8.0}


def gen(seed, over, size=512):
    return pipeline.generate(seed, dict(BASE, **over), size)


def save(w, stem, view="hypsometric"):
    img = render.render_view(w, view)
    render.save_png(img, os.path.join(OUT, stem + ".png"), w)
    return img


def main():
    # geometry audit: shift markers on the plates view
    ts = []
    for s in (3, 5):
        for b in (0.0, 1.0):
            w = gen(s, {"active_margin_bias": b})
            ts.append((f"s{s} bias={b:g} hypso",
                       save(w, f"audit_s{s}_b{b:g}")))
            ts.append((f"s{s} bias={b:g} plates",
                       save(w, f"audit_s{s}_b{b:g}_plates", "plates")))
            print(f"audit s{s} b{b:g} done")
    render.contact_sheet(ts, cols=4, thumb=380).save(
        os.path.join(OUT, "_audit.png"))

    # bias ladder (the default pick happens here)
    ts = []
    for s in (3, 5, 7):
        for b in (0.0, 0.35, 0.65, 1.0):
            w = gen(s, {"active_margin_bias": b})
            ts.append((f"seed {s}, bias={b:g}",
                       save(w, f"sweep_s{s}_b{int(b * 100):03d}")))
            print(f"sweep s{s} b{b:g} done")
    render.contact_sheet(ts, cols=4, thumb=340).save(
        os.path.join(OUT, "_sweep.png"))

    # variety at the provisional default
    ts = []
    for s in range(1, 9):
        w = gen(s, {})
        report.write(w, os.path.join(OUT, f"seed{s}.json"))
        ts.append((f"seed {s}", save(w, f"seed{s}")))
        print(f"seed {s} done")
    render.contact_sheet(ts, cols=4).save(os.path.join(OUT, "_seeds.png"))

    coasts()
    print(f"a2 galleries: {OUT}")


def coasts():
    """Cordilleran-coast zooms: best mountains-meet-ocean spot per seed
    at bias 1."""
    ts = []
    for s in (3, 5, 6):
        w = pipeline.generate(s, {"cell_size_km": 6.0,
                                  "active_margin_bias": 1.0}, 768)
        img = render.hypsometric(w)
        e = w["elevation"].astype(np.float64)
        cont = w["crust"].astype(np.float64)       # arcs sit on ocean floor
        high = _fft_gauss(np.maximum(e - 1200.0, 0.0) * cont, 14.0)
        sea = _fft_gauss((e < 0).astype(np.float64), 14.0)
        score = high * sea
        r, c = np.unravel_index(int(np.argmax(score)), score.shape)
        half = 150
        r = int(np.clip(r, half, score.shape[0] - half))
        c = int(np.clip(c, half, score.shape[1] - half))
        t = img.crop((c - half, r - half, c + half, r + half))
        ts.append((f"seed {s} coast", t.resize((600, 600), Image.NEAREST)))
        print(f"coast s{s} done")
    render.contact_sheet(ts, cols=3, thumb=420).save(
        os.path.join(OUT, "_coasts.png"))


if __name__ == "__main__":
    if "--cal" in sys.argv:      # quick calibration loop: audit + zooms
        ts = []
        for s in (3, 5):
            w = gen(s, {"active_margin_bias": 1.0})
            ts.append((f"s{s} hypso", save(w, f"audit_s{s}_b1")))
            ts.append((f"s{s} plates",
                       save(w, f"audit_s{s}_b1_plates", "plates")))
            print(f"cal audit s{s} done")
        render.contact_sheet(ts, cols=4, thumb=380).save(
            os.path.join(OUT, "_cal.png"))
        coasts()
    else:
        main()
