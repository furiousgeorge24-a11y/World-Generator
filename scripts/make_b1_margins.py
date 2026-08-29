"""B1 passive-margin bathymetry — evidence galleries.

    py -3.14 scripts/make_b1_margins.py

out/b1_margins/: _audit.png (B1 on/off pairs), _sweep.png
(margin_width_km ladder — the author's default pick), _seeds.png (8
seeds at defaults), _abl.png (per-mechanism knock-outs incl. the old
shelf_width redundancy check), _coasts.png (passive descent / trench
plunge / island pedestal zooms).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from mapgen import pipeline, render, report  # noqa: E402
from mapgen.boundaries import _fft_gauss  # noqa: E402

OUT = os.path.join("out", "b1_margins")
os.makedirs(OUT, exist_ok=True)

BASE = {"cell_size_km": 8.0}
OFF = {"margin_width_km": 0.0, "edifice_pedestal": 0.0, "rise_feed": 0.0}


def gen(seed, over, size=512):
    return pipeline.generate(seed, dict(BASE, **over), size)


def save(w, stem, view="hypsometric"):
    img = render.render_view(w, view)
    render.save_png(img, os.path.join(OUT, stem + ".png"), w)
    return img


def crop(w, img, score, half=150):
    r, c = np.unravel_index(int(np.argmax(score)), score.shape)
    r = int(np.clip(r, half, score.shape[0] - half))
    c = int(np.clip(c, half, score.shape[1] - half))
    t = img.crop((c - half, r - half, c + half, r + half))
    return t.resize((600, 600), Image.NEAREST)


def main():
    # on/off audit pairs
    ts = []
    for s in (3, 5, 6):
        for label, over in (("B1 on (defaults)", {}), ("B1 off", OFF)):
            w = gen(s, over)
            ts.append((f"seed {s}: {label}",
                       save(w, f"audit_s{s}_{'on' if not over else 'off'}")))
            print(f"audit s{s} {label} done")
    render.contact_sheet(ts, cols=2, thumb=430).save(
        os.path.join(OUT, "_audit.png"))

    # margin width ladder
    ts = []
    for s in (3, 6):
        for mw in (0.0, 120.0, 220.0, 350.0):
            w = gen(s, {"margin_width_km": mw})
            ts.append((f"seed {s}, margin_width={mw:g} km",
                       save(w, f"sweep_s{s}_mw{int(mw):03d}")))
            print(f"sweep s{s} mw{mw:g} done")
    render.contact_sheet(ts, cols=4, thumb=340).save(
        os.path.join(OUT, "_sweep.png"))

    # variety at defaults
    ts = []
    for s in range(1, 9):
        w = gen(s, {})
        report.write(w, os.path.join(OUT, f"seed{s}.json"))
        ts.append((f"seed {s}", save(w, f"seed{s}")))
        print(f"seed {s} done")
    render.contact_sheet(ts, cols=4).save(os.path.join(OUT, "_seeds.png"))

    # per-mechanism knock-outs (seed 5); shelf_width=0 tests whether the
    # old control is redundant under the taper (ledger trim question)
    ts = []
    for label, over in (
            ("baseline", {}),
            ("margin_width_km=0", {"margin_width_km": 0.0}),
            ("edifice_pedestal=0", {"edifice_pedestal": 0.0}),
            ("rise_feed=0", {"rise_feed": 0.0}),
            ("sediment_softening=0", {"sediment_softening": 0.0}),
            ("shelf_width=0", {"shelf_width": 0.0})):
        w = gen(5, over)
        ts.append((label, save(w, "abl_" + label.split("=")[0])))
        print(f"abl {label} done")
    render.contact_sheet(ts, cols=3, thumb=360).save(
        os.path.join(OUT, "_abl.png"))

    # zooms: the three promised behaviors, each from its best spot
    zt = []
    w = pipeline.generate(6, {"cell_size_km": 6.0}, 768)
    e = w["elevation"].astype(np.float64)
    mid = ((e < -500.0) & (e > -2600.0)).astype(np.float64)
    zt.append(("passive descent (seed 6)",
               crop(w, render.hypsometric(w), _fft_gauss(mid, 18.0))))
    print("zoom passive done")
    w = pipeline.generate(5, {"cell_size_km": 6.0}, 768)
    e = w["elevation"].astype(np.float64)
    lb = _fft_gauss((e >= 0).astype(np.float64), 12.0)
    zt.append(("trench margin (seed 5)",
               crop(w, render.hypsometric(w),
                    _fft_gauss(-w["tect_trench"].astype(np.float64) * lb,
                               16.0))))
    print("zoom trench done")
    w = pipeline.generate(3, {"cell_size_km": 6.0}, 768)
    e = w["elevation"].astype(np.float64)
    vol = w["volcanic"].astype(np.float64) * (e < 0.0)
    zt.append(("island pedestal (seed 3)",
               crop(w, render.hypsometric(w), _fft_gauss(vol, 12.0))))
    print("zoom pedestal done")
    render.contact_sheet(zt, cols=3, thumb=420).save(
        os.path.join(OUT, "_coasts.png"))

    print(f"b1 galleries: {OUT}")


if __name__ == "__main__":
    main()
