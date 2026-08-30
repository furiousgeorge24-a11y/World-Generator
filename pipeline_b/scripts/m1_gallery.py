"""M1 exit gallery: structure views across seeds + tripwire stats.

    py -3.14 scripts\\m1_gallery.py [--seeds 12] [--tile 348]
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from engine.registry import make_config  # noqa: E402
from engine.render_structure import render_view  # noqa: E402
from engine.report import structure_report  # noqa: E402
from engine.tectonics import build_structure  # noqa: E402

GALLERY_VIEWS = ["crust", "margins", "belt_age", "boundaries"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--tile", type=int, default=348)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "out", "m1",
        "m1_gallery.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rows = []
    print(f"{'seed':>4} {'s':>5} {'cont%':>6} {'domains':>7} "
          f"{'ring':>5} {'hug':>6} {'plates':>6} {'act/pas':>9}")
    for seed in range(1, args.seeds + 1):
        t0 = time.perf_counter()
        s = build_structure(seed, make_config({}))
        el = time.perf_counter() - t0
        rep = structure_report(s, seed, {}, el)
        f = {x["name"]: x["value"] for x in rep["findings"]}
        print(f"{seed:>4} {el:>5.2f} {f['cont_fraction_in_frame']:>6.3f} "
              f"{f['structural_domains']:>7} {f['ring_cont_cells']:>5} "
              f"{f['frame_hug_occupancy']:>6.3f} {f['alive_plates']:>6} "
              f"{f['active_margin_cells']:>4}/{f['passive_margin_cells']}")
        tiles = [np.asarray(render_view(s, v, args.tile).convert("RGB"))
                 for v in GALLERY_VIEWS]
        strip = np.concatenate(tiles, axis=1)
        im = Image.fromarray(strip)
        d = ImageDraw.Draw(im)
        d.text((4, 2), f"seed {seed}  cont {f['cont_fraction_in_frame']:.2f}"
               f"  domains {f['structural_domains']}"
               f"  plates {f['alive_plates']}  {el*1000:.0f} ms",
               fill=(255, 255, 255))
        rows.append(np.asarray(im))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    print("gallery:", os.path.abspath(out))


if __name__ == "__main__":
    main()
