"""M3 review gallery (§14).

Outputs to out/m3/:
- m3_gallery.png     — 12 seeds, hypsometric at 512, stats per tile
- m3_instruments.png — 3 seeds × (hypsometric, drainage, sediment, isobaths)
- m3_pairs.png       — same-seed ablations: erosion on/off, soil creep,
                       lowstand, erodibility range

Pass --out to preserve an existing judged bundle in a separate
directory (Run 1 uses out/m3_run1/).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.elevation import coarse_elevation
from engine.erosion import run_erosion
from engine.registry import make_config
from engine.render_map import render_map_view
from engine.surface import sample_map
from engine.tectonics import build_structure

OUT = ROOT / "out" / "m3"
PAD = 8
FOOT = 16


def gen(seed, size, **overrides):
    cfg = make_config(overrides)
    s = build_structure(seed, cfg)
    ce = coarse_elevation(s, cfg, seed)
    er = run_erosion(s, ce, cfg, seed)
    m = sample_map(s, ce, er, cfg, seed, size)
    return s, ce, er, m, cfg


def tile(im, caption, w=512):
    canvas = Image.new("RGB", (w, im.height + FOOT), (14, 14, 18))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((3, im.height + 2), caption, fill=(210, 210, 210))
    return canvas


def sheet(tiles, cols, path, title):
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    img = Image.new("RGB", (cols * w + (cols + 1) * PAD,
                            rows * h + (rows + 1) * PAD + 20), (10, 10, 12))
    d = ImageDraw.Draw(img)
    d.text((PAD, 4), title, fill=(230, 230, 230))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        img.paste(t, (PAD + c * (w + PAD), 20 + PAD + r * (h + PAD)))
    img.save(path)
    print(f"wrote {path}")


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=str(OUT),
        help="output directory (default: out/m3; use a new directory "
             "to preserve judged galleries)")
    args = parser.parse_args()
    OUT = Path(args.out)
    if not OUT.is_absolute():
        OUT = ROOT / OUT
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    seeds = [3, 7, 11, 19, 23, 31, 40, 51, 63, 77, 88, 101]

    tiles = []
    for sd in seeds:
        s, ce, er, m, cfg = gen(sd, 512)
        land = float((~m["water"]).mean())
        lakes = int((er["lake_depth"] > 0).sum())
        rd = cfg.river_density
        im = render_map_view(m, "hypsometric", river_density=rd)
        source_m3 = (float(er["ero"].sum())
                     * (er["e_km"] * 1000.0) ** 2)
        dep_frac = (float(er["sed"].sum())
                    * (er["e_km"] * 1000.0) ** 2
                    / max(source_m3, 1e-9))
        export_frac = (er["sediment_export_m3"]
                       / max(source_m3, 1e-9))
        cap = (f"seed {sd}  land {land:.2f}  peak {m['h'].max():.0f}m"
               f"  lakes {lakes}c  dep/export "
               f"{dep_frac:.2f}/{export_frac:.2f}")
        tiles.append(tile(im, cap))
        print(f"  seed {sd}: land={land:.2f} lakes={lakes}")
    sheet(tiles, 4, OUT / "m3_gallery.png",
          "M3 surface-process slice - 12 seeds, hypsometric, 512px "
          "(4096 km frame)")

    tiles = []
    for sd in (7, 23, 51):
        s, ce, er, m, cfg = gen(sd, 512)
        for view in ("hypsometric", "drainage", "sediment", "isobaths"):
            tiles.append(tile(render_map_view(m, view,
                                              cfg.river_density),
                              f"seed {sd}  {view}"))
    sheet(tiles, 4, OUT / "m3_instruments.png",
          "M3 instruments - flow, sediment, bathymetry vs the map")

    pairs = [
        ("erosion 20 Myr (default)", dict(), 23),
        ("erosion OFF (ablation)", dict(erosion_time=0.0), 23),
        ("soil creep 1 (default)", dict(), 7),
        ("soil creep OFF", dict(soil_creep=0.0), 7),
        ("lowstand 80 m (default)", dict(), 51),
        ("lowstand OFF", dict(lowstand_drop=0.0), 51),
        ("erodibility 0.5", dict(erodibility=0.5), 88),
        ("erodibility 2.0", dict(erodibility=2.0), 88),
    ]
    tiles = []
    for cap, ov, sd in pairs:
        _, _, er, m, cfg = gen(sd, 512, **ov)
        tiles.append(tile(render_map_view(m, "hypsometric",
                                          cfg.river_density),
                          f"seed {sd}  {cap}"))
    sheet(tiles, 2, OUT / "m3_pairs.png",
          "M3 same-seed pairs - ablations and control range")

    print(f"total {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
