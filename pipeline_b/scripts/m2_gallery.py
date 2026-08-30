"""M2 review gallery (§14: batch galleries, contact sheets first-class).

Outputs to out/m2/:
- m2_gallery.png     — 12 seeds, hypsometric at 512, stats footer per tile
- m2_instruments.png — 3 seeds × (hypsometric, isobaths, slope, crust)
- m2_pairs.png       — same-seed ablations: multi-lobe cratons on/off
                       (value-ledger queue), hydrosphere low/high,
                       orogeny on/off, detail on/off
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.elevation import coarse_elevation
from engine.registry import make_config
from engine.render_map import render_map_view
from engine.render_structure import render_view
from engine.surface import sample_map
from engine.tectonics import build_structure

OUT = ROOT / "out" / "m2"
OUT.mkdir(parents=True, exist_ok=True)
PAD = 8
FOOT = 16


def gen(seed, size, **overrides):
    cfg = make_config(overrides)
    if "multi_lobe" in overrides:
        cfg.multi_lobe = overrides["multi_lobe"]
    s = build_structure(seed, cfg)
    ce = coarse_elevation(s, cfg, seed)
    m = sample_map(s, ce, cfg, seed, size)
    return s, ce, m


def tile(im, caption, w):
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
    t0 = time.perf_counter()
    seeds = [3, 7, 11, 19, 23, 31, 40, 51, 63, 77, 88, 101]

    tiles = []
    for sd in seeds:
        s, ce, m = gen(sd, 512)
        land = float((~m["water"]).mean())
        ring = np.zeros_like(m["water"])
        ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
        rl = int(((~m["water"]) & ring).sum())
        im = render_map_view(m, "hypsometric")
        cap = (f"seed {sd}  land {land:.2f}  peak {m['h'].max():.0f}m"
               f"  L{ce['sea_level']:+.0f}  ring {rl}px")
        tiles.append(tile(im, cap, 512))
        print(f"  seed {sd}: land={land:.2f} ring={rl}")
    sheet(tiles, 4, OUT / "m2_gallery.png",
          "M2 end-to-end slice - 12 seeds, hypsometric, 512px "
          "(4096 km frame)")

    tiles = []
    for sd in (7, 31, 77):
        s, ce, m = gen(sd, 512)
        for view in ("hypsometric", "isobaths", "slope"):
            tiles.append(tile(render_map_view(m, view),
                              f"seed {sd}  {view}", 512))
        tiles.append(tile(render_view(s, "crust", 512).convert("RGB"),
                          f"seed {sd}  crust (cause field)", 512))
    sheet(tiles, 4, OUT / "m2_instruments.png",
          "M2 instruments - map vs cause fields")

    pairs = [
        ("lobes ON (default)", dict(), 31),
        ("lobes OFF (ablation)", dict(multi_lobe=False), 31),
        ("hydrosphere 4500 (lowstand)", dict(hydrosphere_depth=4500.0), 7),
        ("hydrosphere 5400 (highstand)", dict(hydrosphere_depth=5400.0), 7),
        ("orogeny 4000 (default)", dict(), 51),
        ("orogeny 0 (ablation)", dict(orogeny_height=0.0), 51),
        ("detail 1.0 (default)", dict(), 23),
        ("detail 0 (ablation)", dict(detail_amplitude=0.0), 23),
    ]
    tiles = []
    for cap, ov, sd in pairs:
        _, _, m = gen(sd, 512, **ov)
        tiles.append(tile(render_map_view(m, "hypsometric"),
                          f"seed {sd}  {cap}", 512))
    sheet(tiles, 2, OUT / "m2_pairs.png",
          "M2 same-seed pairs - ablations and control range")

    print(f"total {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
