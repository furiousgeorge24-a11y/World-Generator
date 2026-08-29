"""Build the formal M2 review gallery set into out/m2_review/.

    py -3.14 scripts/make_m2_review.py

Sheets:
  _seeds.png     8 seeds, hypsometric, continental extent
  _drainage.png  same seeds, drainage view
  _scale.png     one seed across cell_size_km 4/8/16/32 (what a map means)
  _ablation.png  baseline vs each M2 feature at zero, incl. full-M1 pair
Individual abl_*.png files accompany the sheet. Every PNG carries
provenance; every world writes a report sidecar.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mapgen import pipeline, render, report  # noqa: E402

OUT = os.path.join("out", "m2_review")
os.makedirs(OUT, exist_ok=True)

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
BASE = {"cell_size_km": 10.0}


def world_png(seed, controls, size, stem, view="hypsometric"):
    w = pipeline.generate(seed, controls, size)
    img = render.render_view(w, view)
    render.save_png(img, os.path.join(OUT, stem + ".png"), w)
    report.write(w, os.path.join(OUT, stem + ".json"))
    return stem, img


def main():
    entries = []
    dr = []
    for s in SEEDS:
        w = pipeline.generate(s, BASE, 384)
        for view, bucket in (("hypsometric", entries), ("drainage", dr)):
            img = render.render_view(w, view)
            render.save_png(img, os.path.join(OUT, f"seed{s}_{view}.png"), w)
            bucket.append((f"s{s}", img))
        report.write(w, os.path.join(OUT, f"seed{s}.json"))
        print(f"seed {s} done")
    render.contact_sheet(entries, cols=4).save(os.path.join(OUT, "_seeds.png"))
    render.contact_sheet(dr, cols=4).save(os.path.join(OUT, "_drainage.png"))

    scale = []
    for ck in (4.0, 8.0, 16.0, 32.0):
        stem, img = world_png(5, {"cell_size_km": ck}, 384, f"scale_{int(ck)}km")
        scale.append((f"{int(ck)} km/cell ({int(ck * 384)} km)", img))
        print(f"scale {ck} done")
    render.contact_sheet(scale, cols=2, thumb=380).save(
        os.path.join(OUT, "_scale.png"))

    abl_specs = [
        ("baseline", {}),
        ("m1_skeleton", {"erosion_strength": 0.0, "sediment_softening": 0.0,
                         "fan_size": 0.0, "canyon_depth": 0.0}),
        ("erosion_off", {"erosion_strength": 0.0}),
        ("hillslope_off", {"hillslope_smoothing": 0.0}),
        ("volcano_youth_hi", {"volcano_youth": 1.0}),
        ("sediment_off", {"sediment_softening": 0.0}),
        ("fans_off", {"fan_size": 0.0}),
        ("canyons_off", {"canyon_depth": 0.0}),
        ("lakes_deep_only", {"lake_min_depth_m": 8.0}),
    ]
    abl = []
    for name, over in abl_specs:
        ctl = dict(BASE, cell_size_km=8.0)
        ctl.update(over)
        stem, img = world_png(5, ctl, 512, f"abl_{name}")
        abl.append((name, img))
        print(f"ablation {name} done")
    render.contact_sheet(abl, cols=3, thumb=340).save(
        os.path.join(OUT, "_ablation.png"))
    print(f"review pack: {OUT}")


if __name__ == "__main__":
    main()
