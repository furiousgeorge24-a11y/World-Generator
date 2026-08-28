"""Batch gallery CLI — the formal review vehicle (contract section 12).

    py -3.14 -m mapgen.batch --tag c0_smoke --seeds 1-6 --sizes 256
    py -3.14 -m mapgen.batch --tag t --seeds 1,5,9 --sizes 256,512 --set stub_land_bias=0.2
"""

import argparse
import json
import os

from . import VERSION, pipeline, render, report


def _parse_ints(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="mapgen batch gallery")
    ap.add_argument("--tag", required=True, help="output subdir under out/")
    ap.add_argument("--seeds", default="1-6")
    ap.add_argument("--sizes", default="256")
    ap.add_argument("--set", action="append", default=[], metavar="K=V",
                    help="control override, repeatable")
    ap.add_argument("--views", default="hypsometric",
                    help="comma list; one gallery sheet per view")
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()

    controls: dict = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        controls[k] = v

    outdir = os.path.join("out", args.tag)
    os.makedirs(outdir, exist_ok=True)
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    entries: dict[str, list] = {v: [] for v in views}
    n_maps = 0
    for size in _parse_ints(args.sizes):
        for seed in _parse_ints(args.seeds):
            world = pipeline.generate(seed, controls, size)
            stem = f"s{seed}_{size}"
            report.write(world, os.path.join(outdir, stem + ".json"))
            n_maps += 1
            for v in views:
                img = render.render_view(world, v)
                render.save_png(img, os.path.join(outdir, f"{stem}_{v}.png"),
                                world)
                entries[v].append((f"{stem} {v}", img))
            warns = sum(1 for f in world.findings if f.get("level") == "warn")
            print(f"{stem}: {sum(world.timings.values()):.2f}s"
                  + (f", {warns} warn" if warns else ""))
    for v in views:
        render.contact_sheet(entries[v], cols=args.cols).save(
            os.path.join(outdir, f"_sheet_{v}.png"))
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"version": VERSION, "tag": args.tag, "seeds": args.seeds,
                   "sizes": args.sizes, "views": views,
                   "overrides": controls}, f, indent=2)
    print(f"gallery: {outdir}  ({n_maps} maps x {len(views)} view(s))")


if __name__ == "__main__":
    main()
