"""Build the formal M1 review gallery set into out/m1_review/.

    py -3.14 scripts/make_m1_review.py

Four sheets:
  _seeds.png     eight seeds at continental extent (the variety check)
  _scale.png     one seed at four cell sizes (what a map *means*)
  _style.png     render variants: smooth vs quantized ramps
  _ablation.png  same seed/extent, one feature zeroed per tile — the
                 value-ledger evidence (ablation = knob at zero)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mapgen import pipeline, render, report  # noqa: E402

OUT = os.path.join("out", "m1_review")
SEED = 5
BASE = {"cell_size_km": 8.0}
SIZE = 512

ABLATIONS = [
    ("baseline", {}),
    ("era_count=1", {"era_count": 1}),
    ("province_relief=0", {"province_relief": 0}),
    ("tectonic_grain=0", {"tectonic_grain": 0}),
    ("coast_complexity=0", {"coast_complexity": 0}),
    ("crest_sharpness=0", {"crest_sharpness": 0}),
    ("outer_rise=0", {"outer_rise": 0}),
    ("seafloor_fabric=0", {"seafloor_fabric": 0}),
    ("ridge_segmentation=0", {"ridge_segmentation": 0}),
    ("backarc_basins=0", {"backarc_basins": 0}),
    ("failed_rifts=0", {"failed_rifts": 0}),
    ("rift_maturity=0", {"rift_maturity": 0}),
    ("hotspot_count=0", {"hotspot_count": 0}),
]


def gen(seed, controls, size):
    world = pipeline.generate(seed, controls, size)
    return world, render.hypsometric(world)


def main():
    os.makedirs(OUT, exist_ok=True)

    entries = []
    for seed in range(1, 9):
        world, img = gen(seed, {"cell_size_km": 16.0}, 256)
        render.save_png(img, os.path.join(OUT, f"seed{seed}.png"), world)
        report.write(world, os.path.join(OUT, f"seed{seed}.json"))
        entries.append((f"seed {seed}", img))
        print(f"seeds: {seed}/8")
    render.contact_sheet(entries, cols=4).save(os.path.join(OUT, "_seeds.png"))

    entries = []
    for cell in (4.0, 8.0, 16.0, 32.0):
        world, img = gen(SEED, {"cell_size_km": cell}, 256)
        extent = int(256 * cell)
        entries.append((f"{cell:g} km/cell = {extent} km", img))
        print(f"scale: {cell:g} km/cell")
    render.contact_sheet(entries, cols=4).save(os.path.join(OUT, "_scale.png"))

    entries = []
    for q in (0, 8, 14, 22):
        world, img = gen(SEED, dict(BASE, render_quantize=q), SIZE)
        entries.append((f"quantize={q}" if q else "smooth ramp", img))
        print(f"style: quantize={q}")
    render.contact_sheet(entries, cols=2, thumb=420).save(
        os.path.join(OUT, "_style.png"))

    entries = []
    for label, over in ABLATIONS:
        world, img = gen(SEED, dict(BASE, **over), SIZE)
        render.save_png(img, os.path.join(OUT, f"abl_{label.split('=')[0]}.png"),
                        world)
        entries.append((label, img))
        print(f"ablation: {label}")
    render.contact_sheet(entries, cols=4, thumb=300).save(
        os.path.join(OUT, "_ablation.png"))

    print(f"review kit: {OUT}")


if __name__ == "__main__":
    main()
