"""Tier 2 belt-and-basin anatomy — evidence galleries.

    py -3.14 scripts/make_tier2_anatomy.py

out/tier2_anatomy/: _audit.png (tier-2 on/off pairs), _plateau.png
(re-grounded plateau_tendency ladder — the author's default pick),
_rag.png (belt_raggedness ladder), _rift.png (rift_segmentation ladder),
_riftzoom.png (rift-arm zooms, laser vs en-echelon), _basins.png
(basin_fill ladder), _basinzoom.png (same interior crop across the
ladder), _crest.png (crest_zone zoom pair), _abl.png (per-mechanism
knock-outs incl. outer_rise=0 flexure ablation), _seeds.png (8 seeds at
defaults), _big.png + big_s*.png (1024^2 money shots).

Note: the tier-2 OFF composite is not bit-identical to 0.9.3 — the
outer-rise bulge trim, the graben depth retune and the thickened-zone
span formula are unconditional (authorized 2026-08-29); knock-outs
compare within-version.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from mapgen import pipeline, render, report  # noqa: E402
from mapgen.boundaries import _fft_gauss  # noqa: E402

OUT = os.path.join("out", "tier2_anatomy")
os.makedirs(OUT, exist_ok=True)

BASE = {"cell_size_km": 8.0}
OFF = {"belt_raggedness": 0.0, "rift_segmentation": 0.0,
       "basin_fill": 0.0, "crest_zone": 0.0}


def gen(seed, over, size=512):
    return pipeline.generate(seed, dict(BASE, **over), size)


def save(w, stem, view="hypsometric"):
    img = render.render_view(w, view)
    render.save_png(img, os.path.join(OUT, stem + ".png"), w)
    return img


def crop_at(img, r, c, shape, half=110):
    r = int(np.clip(r, half, shape[0] - half))
    c = int(np.clip(c, half, shape[1] - half))
    t = img.crop((c - half, r - half, c + half, r + half))
    return t.resize((560, 560), Image.NEAREST)


def best(score):
    return np.unravel_index(int(np.argmax(score)), score.shape)


def main():
    # tier-2 on/off audit pairs
    ts = []
    for s in (3, 5, 6):
        for label, over in (("tier 2 on (defaults)", {}),
                            ("tier 2 off (4 knobs at 0)", OFF)):
            w = gen(s, over)
            ts.append((f"seed {s}: {label}",
                       save(w, f"audit_s{s}_{'on' if not over else 'off'}")))
            print(f"audit s{s} done ({label})")
    render.contact_sheet(ts, cols=2, thumb=430).save(
        os.path.join(OUT, "_audit.png"))

    # re-grounded plateau ladder (default currently 0 — author pick)
    ts = []
    for s in (3, 5):
        for pt in (0.0, 0.35, 0.7, 1.0):
            w = gen(s, {"plateau_tendency": pt})
            ts.append((f"seed {s}, plateau_tendency={pt:g}",
                       save(w, f"plateau_s{s}_pt{int(pt * 100):03d}")))
            print(f"plateau s{s} pt{pt:g} done")
    render.contact_sheet(ts, cols=4, thumb=340).save(
        os.path.join(OUT, "_plateau.png"))

    # belt raggedness ladder
    ts = []
    for s in (5, 7):
        for rg in (0.0, 0.35, 0.7, 1.0):
            w = gen(s, {"belt_raggedness": rg})
            ts.append((f"seed {s}, belt_raggedness={rg:g}",
                       save(w, f"rag_s{s}_r{int(rg * 100):03d}")))
            print(f"rag s{s} r{rg:g} done")
    render.contact_sheet(ts, cols=4, thumb=340).save(
        os.path.join(OUT, "_rag.png"))

    # rift segmentation ladder + zooms (crop chosen on the seg=0 world's
    # graben so the same spot is compared across the ladder)
    ts = []
    zt = []
    for s in (5, 9):
        imgs = {}
        w0 = None
        for sg in (0.0, 0.35, 0.65, 1.0):
            w = gen(s, {"rift_segmentation": sg})
            img = save(w, f"rift_s{s}_sg{int(sg * 100):03d}")
            imgs[sg] = img
            ts.append((f"seed {s}, rift_segmentation={sg:g}", img))
            if sg == 0.0:
                w0 = w
            print(f"rift s{s} sg{sg:g} done")
        gsc = _fft_gauss(np.abs(w0["tect_graben"].astype(np.float64)), 10.0)
        r, c = best(gsc)
        for sg in (0.0, 0.65):
            zt.append((f"seed {s} zoom, seg={sg:g}",
                       crop_at(imgs[sg], r, c, w0.shape)))
    render.contact_sheet(ts, cols=4, thumb=340).save(
        os.path.join(OUT, "_rift.png"))
    render.contact_sheet(zt, cols=2, thumb=430).save(
        os.path.join(OUT, "_riftzoom.png"))

    # basin fill ladder + fixed interior crop across it (seed 3 carries
    # the mega-lake case)
    ts = []
    zt = []
    imgs = {}
    w0 = None
    for bf in (0.0, 0.35, 0.7, 1.0):
        w = gen(3, {"basin_fill": bf})
        img = save(w, f"basin_s3_bf{int(bf * 100):03d}")
        imgs[bf] = img
        ts.append((f"seed 3, basin_fill={bf:g}", img))
        if bf == 0.0:
            w0 = w
        print(f"basin bf{bf:g} done")
    lakes = (w0["lake_id"] > 0).astype(np.float64)
    r, c = best(_fft_gauss(lakes, 14.0))
    for bf in (0.0, 0.35, 0.7, 1.0):
        zt.append((f"seed 3 zoom, basin_fill={bf:g}",
                   crop_at(imgs[bf], r, c, w0.shape)))
    render.contact_sheet(ts, cols=4, thumb=340).save(
        os.path.join(OUT, "_basins.png"))
    render.contact_sheet(zt, cols=2, thumb=430).save(
        os.path.join(OUT, "_basinzoom.png"))

    # crest-zone zoom pair (summit regions vs ridgelines only)
    zt = []
    wz = pipeline.generate(7, {"cell_size_km": 6.0, "crest_zone": 1.0}, 768)
    zsc = _fft_gauss(wz["tect_crest_zone"].astype(np.float64), 10.0)
    r, c = best(zsc)
    zt.append(("crest_zone=1 (seed 7)",
               crop_at(render.hypsometric(wz), r, c, wz.shape, half=130)))
    wz0 = pipeline.generate(7, {"cell_size_km": 6.0, "crest_zone": 0.0}, 768)
    zt.append(("crest_zone=0 (same spot)",
               crop_at(render.hypsometric(wz0), r, c, wz0.shape, half=130)))
    print("crest zooms done")
    render.contact_sheet(zt, cols=2, thumb=430).save(
        os.path.join(OUT, "_crest.png"))

    # per-mechanism knock-outs (seed 5)
    ts = []
    for label, over in (
            ("baseline", {}),
            ("belt_raggedness=0", {"belt_raggedness": 0.0}),
            ("rift_segmentation=0", {"rift_segmentation": 0.0}),
            ("basin_fill=0", {"basin_fill": 0.0}),
            ("crest_zone=0", {"crest_zone": 0.0}),
            ("outer_rise=0 (flexure)", {"outer_rise": 0.0})):
        w = gen(5, over)
        ts.append((label, save(w, "abl_" + label.split("=")[0])))
        print(f"abl {label} done")
    render.contact_sheet(ts, cols=3, thumb=360).save(
        os.path.join(OUT, "_abl.png"))

    # variety at defaults
    ts = []
    for s in range(1, 9):
        w = gen(s, {})
        report.write(w, os.path.join(OUT, f"seed{s}.json"))
        ts.append((f"seed {s}", save(w, f"seed{s}")))
        print(f"seed {s} done")
    render.contact_sheet(ts, cols=4).save(os.path.join(OUT, "_seeds.png"))

    # 1024^2 money shots at 4 km
    ts = []
    for s in (3, 6):
        w = pipeline.generate(s, {"cell_size_km": 4.0}, 1024)
        ts.append((f"seed {s}, 1024^2 @ 4 km", save(w, f"big_s{s}")))
        print(f"big s{s} done")
    render.contact_sheet(ts, cols=2, thumb=500).save(
        os.path.join(OUT, "_big.png"))

    print(f"tier2 galleries: {OUT}")


if __name__ == "__main__":
    main()
