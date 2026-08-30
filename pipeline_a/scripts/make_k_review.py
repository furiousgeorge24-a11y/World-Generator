"""Build the combined formal review pack into out/k_review/.

    py -3.14 scripts/make_k_review.py          # sheets (~6 min)
    py -3.14 scripts/make_k_review.py --big    # 1024^2 + 2048^2 standalones

Replaces the separately pending M1/M2 formal reviews and closes the
K-series (milestones.md). Every ledger row with a pending observed-yield
column has a tile here; README.md in the pack maps sheets to verdicts.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image  # noqa: E402

from mapgen import pipeline, render, report  # noqa: E402
from mapgen.boundaries import _fft_gauss  # noqa: E402

OUT = os.path.join("out", "k_review")
os.makedirs(OUT, exist_ok=True)

SEED = 5
BASE = {"cell_size_km": 8.0}

ABL = {
    "_abl_tectonics": [
        ("baseline", {}),
        ("plate_anisotropy=0", {"plate_anisotropy": 0.0}),
        ("arc_curvature=0", {"arc_curvature": 0.0}),
        ("ridge_segmentation=0", {"ridge_segmentation": 0.0}),
        ("crest_sharpness=0", {"crest_sharpness": 0.0}),
        ("flexure=0 (outer_rise)", {"outer_rise": 0.0}),
        ("era_count=1", {"era_count": 1}),
        ("hotspot_count=0", {"hotspot_count": 0}),
        ("plateau_tendency=0", {"plateau_tendency": 0.0}),
        ("crust_plate_affinity=0", {"crust_plate_affinity": 0.0}),
        ("active_margin_bias=0", {"active_margin_bias": 0.0}),
    ],
    "_abl_flourishes": [
        ("baseline", {}),
        ("rift_maturity=0", {"rift_maturity": 0.0}),
        ("failed_rifts=0", {"failed_rifts": 0.0}),
        ("backarc_basins=0.55 (old default, now 0)", {"backarc_basins": 0.55}),
        ("seafloor_fabric=0", {"seafloor_fabric": 0.0}),
        ("province_relief=0", {"province_relief": 0.0}),
        ("tectonic_grain=0", {"tectonic_grain": 0.0}),
        ("coast_complexity=0", {"coast_complexity": 0.0}),
    ],
    "_abl_carve": [
        ("baseline", {}),
        ("erosion_strength=0", {"erosion_strength": 0.0}),
        ("hillslope_smoothing=0", {"hillslope_smoothing": 0.0}),
        ("lowland_dissection=0", {"lowland_dissection": 0.0}),
        ("deposition=0", {"deposition": 0.0}),
        ("plains_grain=0", {"plains_grain": 0.0}),
        ("volcano_youth=1", {"volcano_youth": 1.0}),
    ],
    "_abl_sea": [
        ("baseline", {}),
        ("flood_rise_m=0", {"flood_rise_m": 0.0}),
        ("wave_planation=0", {"wave_planation": 0.0}),
        ("sediment_softening=0", {"sediment_softening": 0.0}),
        ("fan_size=0", {"fan_size": 0.0}),
        ("canyon_depth=0", {"canyon_depth": 0.0}),
        ("shelf_width=0", {"shelf_width": 0.0}),
        ("ridge_swell=0", {"ridge_swell": 0.0}),
        ("lake_min_depth=0.8 (old default)", {"lake_min_depth_m": 0.8}),
        ("margin_width_km=0", {"margin_width_km": 0.0}),
        ("edifice_pedestal=0", {"edifice_pedestal": 0.0}),
        ("rise_feed=0", {"rise_feed": 0.0}),
    ],
}


def gen(seed, over, size=512):
    return pipeline.generate(seed, dict(BASE, **over), size)


def save(w, stem, view="hypsometric"):
    img = render.render_view(w, view)
    render.save_png(img, os.path.join(OUT, stem + ".png"), w)
    return img


def crop_at(img, score, taken, half=150):
    """Best-scoring crop, excluding neighborhoods already used so the
    canon tiles show distinct places."""
    s = score.copy()
    for r0, c0 in taken:
        s[max(r0 - 260, 0):r0 + 260, max(c0 - 260, 0):c0 + 260] = -np.inf
    r, c = np.unravel_index(int(np.argmax(s)), s.shape)
    taken.append((int(r), int(c)))
    r = int(np.clip(r, half, s.shape[0] - half))
    c = int(np.clip(c, half, s.shape[1] - half))
    t = img.crop((c - half, r - half, c + half, r + half))
    return t.resize((half * 4, half * 4), Image.NEAREST)


def main():
    # variety + drainage
    tiles, dr = [], []
    for s in range(1, 9):
        w = gen(s, {})
        report.write(w, os.path.join(OUT, f"seed{s}.json"))
        tiles.append((f"seed {s}", save(w, f"seed{s}")))
        if s <= 4:
            dr.append((f"seed {s}", save(w, f"seed{s}_drainage", "drainage")))
        print(f"seed {s} done")
    render.contact_sheet(tiles, cols=4).save(os.path.join(OUT, "_seeds.png"))
    render.contact_sheet(dr, cols=2, thumb=430).save(
        os.path.join(OUT, "_drainage.png"))

    # scale ladder
    sc = []
    for ck in (4.0, 8.0, 16.0, 32.0):
        w = gen(SEED, {"cell_size_km": ck})
        sc.append((f"{ck:g} km/cell = {int(ck * 512)} km",
                   save(w, f"scale_{int(ck)}km")))
        print(f"scale {ck:g} done")
    render.contact_sheet(sc, cols=2, thumb=420).save(
        os.path.join(OUT, "_scale.png"))

    # all views, one seed
    w = gen(3, {})
    vt = [(v, save(w, f"view_{v}", v))
          for v in ("hypsometric", "plates", "crust", "uplift", "margins",
                    "volcanic", "drainage")]
    render.contact_sheet(vt, cols=4, thumb=340).save(
        os.path.join(OUT, "_views.png"))
    print("views done")

    # ablation sheets (same seed everywhere; ablation = knob at zero)
    for sheet, specs in ABL.items():
        ts = []
        for label, over in specs:
            w = gen(SEED, over)
            stem = "abl_" + label.split("=")[0].split(" ")[0]
            ts.append((label, save(w, stem)))
            print(f"{sheet}: {label} done")
        render.contact_sheet(ts, cols=4, thumb=320).save(
            os.path.join(OUT, sheet + ".png"))

    # macro pair: the uncarved skeleton vs the full pipeline
    off = {"erosion_strength": 0.0, "sediment_softening": 0.0,
           "fan_size": 0.0, "canyon_depth": 0.0, "flood_rise_m": 0.0,
           "wave_planation": 0.0, "deposition": 0.0, "plains_grain": 0.0}
    pair = [("full pipeline", save(gen(SEED, {}), "macro_full")),
            ("skeleton (all M2+/K off)", save(gen(SEED, off), "macro_skel"))]
    render.contact_sheet(pair, cols=2, thumb=460).save(
        os.path.join(OUT, "_macro.png"))
    print("macro pair done")

    # chunky-grain decision sheet (open question 6)
    qt = []
    for q in (0, 8, 12, 16):
        w = gen(3, {"render_quantize": q})
        lbl = "smooth" if q == 0 else (
            f"quantize={q} (default)" if q == 12 else f"quantize={q}")
        qt.append((lbl, save(w, f"q_{q:02d}")))
    render.contact_sheet(qt, cols=2, thumb=420).save(
        os.path.join(OUT, "_quantize.png"))
    print("quantize sheet done")

    canon()
    print(f"review pack: {OUT}")


def canon():
    """Refs (top) vs ours (bottom); belt/plateau crops come from
    whichever review world carries the most of that feature (plate
    layouts move with defaults), crops mutually excluded per world."""
    pool = [pipeline.generate(s, {"cell_size_km": 6.0}, 768)
            for s in (3, 4, 5, 6)]
    ims = {id(W): render.hypsometric(W) for W in pool}
    taken = {id(W): [] for W in pool}
    used = set()

    def high(W):
        return np.maximum(W["elevation"].astype(np.float64) - 1800.0, 0.0)

    def plateau(W):
        return W["tect_plateau"].astype(np.float64)

    def coastal_trench(W):
        lb = _fft_gauss((W["elevation"] >= 0).astype(np.float64), 12.0)
        return -W["tect_trench"].astype(np.float64) * lb

    def crop(field, sigma):
        fresh = [W for W in pool if id(W) not in used] or pool
        W = max(fresh, key=lambda x: float(field(x).sum()))
        used.add(id(W))
        return crop_at(ims[id(W)], _fft_gauss(field(W), sigma),
                       taken[id(W)])

    belt = crop(high, 20.0)
    plat = crop(plateau, 16.0)
    tren = crop(coastal_trench, 16.0)
    full = render.hypsometric(gen(SEED, {}))
    ours = [("ours: composition", full), ("ours: belt", belt),
            ("ours: plateau", plat), ("ours: arc+trench coast", tren)]
    refs = []
    for rn in (13, 7, 14, 1):
        p = os.path.join("examples", f"ref{rn}.png")
        im = Image.open(p).convert("RGB")
        im.thumbnail((640, 640), Image.LANCZOS)
        refs.append((f"ref{rn}", im))
    render.contact_sheet(refs + ours, cols=4, thumb=340).save(
        os.path.join(OUT, "_canon.png"))
    print("canon sheet done")


def big():
    for size, ck in ((1024, 6.0), (2048, 4.0)):
        w = pipeline.generate(SEED, {"cell_size_km": ck}, size)
        save(w, f"big_{size}")
        report.write(w, os.path.join(OUT, f"big_{size}.json"))
        print(f"{size} done")


if __name__ == "__main__":
    if "--big" in sys.argv:
        big()
    elif "--canon" in sys.argv:
        canon()
    else:
        main()
