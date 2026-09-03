"""Tile one view across a spread of seeds so the author can look at all of them.

    py -3.14 pipeline_c/tools/contact_sheet.py --view plates --pixels 512 \
        --scale 5 --out pipeline_c/out/plates_512.png

Nothing is drawn on the panels: no seed labels, no borders, no legends. The
gutter is black and the panels are the raw rasters at native resolution.
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import webui_adapter  # noqa: E402
from engine.history.plates import weak_mask  # noqa: E402
from engine.views import mask, scalar  # noqa: E402

# The twelve development seeds of `STATUS.md`, in that order.
DEVELOPMENT_SEEDS = (
    2075014389, 2477733044, 476149591, 151640007, 2697441485, 1504571935,
    548870008, 2157195430, 4108373596, 4287772760, 287488203, 1833546021,
)
COLUMNS = 4
ROWS = 3
GUTTER_PX = 4


def _tile(panels: list[Image.Image], columns: int) -> Image.Image:
    """Lay the panels out on a black gutter. Nothing is drawn on them."""
    width, height = panels[0].size
    columns = min(columns, len(panels))
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * width + (columns - 1) * GUTTER_PX,
         rows * height + (rows - 1) * GUTTER_PX),
        (0, 0, 0),
    )
    for index, panel in enumerate(panels):
        column, row = index % columns, index // columns
        sheet.paste(panel, (column * (width + GUTTER_PX),
                            row * (height + GUTTER_PX)))
    return sheet


def contact_sheet(seeds: list[int], view: str, pixels: int, scale: int) -> Image.Image:
    """Generate every seed, render `view`, and tile the panels four across."""
    panels = []
    for seed in seeds:
        started = time.perf_counter()
        world = webui_adapter.generate(seed, {"scale_km": scale}, pixels)
        elapsed = time.perf_counter() - started
        record = webui_adapter.report(world)
        panels.append(Image.open(BytesIO(webui_adapter.render_png(world, view))))
        print(f"seed {seed:>10d}  {elapsed:6.1f}s  "
              f"plate_count {record['plate_count']:>3d}  "
              f"weak_fraction_final {record['weak_fraction_final']:.6f}")
    return _tile(panels, COLUMNS)


def view_row(seed: int, views: list[str], pixels: int, scale: int) -> Image.Image:
    """Generate one world once and tile the named views in a single row.

    The world is generated once, so the panels are the same history read
    through different views rather than several runs of it.
    """
    started = time.perf_counter()
    world = webui_adapter.generate(seed, {"scale_km": scale}, pixels)
    elapsed = time.perf_counter() - started
    record = webui_adapter.report(world)
    print(f"seed {seed:>10d}  {elapsed:6.1f}s  "
          f"plate_count {record['plate_count']:>3d}  "
          f"weak_fraction_final {record['weak_fraction_final']:.6f}")
    panels = [Image.open(BytesIO(webui_adapter.render_png(world, view)))
              for view in views]
    print("views  " + ", ".join(views))
    return _tile(panels, len(panels))


def _panel(rgb: np.ndarray) -> Image.Image:
    """The same orientation `webui_adapter.render_png` delivers."""
    image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def early_sheet(seed: int, pixels: int, scale: int) -> Image.Image:
    """The early strength snapshots and their weak masks, two rows.

    The first kept epoch is a quarter of the way through the history, which is
    after the transient. These are the steps before it: the top row is
    strength, the bottom row the weak mask of the same step.
    """
    started = time.perf_counter()
    world = webui_adapter.generate(seed, {"scale_km": scale}, pixels)
    elapsed = time.perf_counter() - started
    snapshots = world.history.early
    print(f"seed {seed:>10d}  {elapsed:6.1f}s  "
          f"{len(snapshots)} early snapshots")
    for step, t_myr, strength in snapshots:
        print(f"  step {step:>3d}  t {t_myr:7.1f} Myr  "
              f"weak_fraction {float(np.mean(weak_mask(strength))):.6f}  "
              f"strength mean {float(strength.mean()):.6f}")
    panels = [_panel(scalar(strength)) for _, _, strength in snapshots]
    panels += [_panel(mask(weak_mask(strength))) for _, _, strength in snapshots]
    return _tile(panels, len(snapshots))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--view", default="plates",
                        help=f"one of {', '.join(webui_adapter.VIEWS)}")
    parser.add_argument("--views", default=None,
                        help="comma-separated views tiled in a row for one "
                             "seed; replaces --view and requires one seed")
    parser.add_argument("--early", action="store_true",
                        help="tile one seed's early strength snapshots and "
                             "their weak masks in two rows; requires one seed")
    parser.add_argument("--pixels", type=int, default=512)
    parser.add_argument("--scale", type=int, default=5)
    parser.add_argument("--seeds", default=None,
                        help="comma-separated seeds; defaults to the twelve")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    seeds = (
        [int(value) for value in args.seeds.split(",") if value.strip()]
        if args.seeds else list(DEVELOPMENT_SEEDS)
    )
    if not seeds:
        raise SystemExit("no seeds requested")

    if args.early:
        if args.views:
            raise SystemExit("--early and --views are separate sheets")
        if len(seeds) != 1:
            raise SystemExit("--early tiles one seed's snapshots; pass one seed")
        views = []
    elif args.views:
        views = [value.strip() for value in args.views.split(",") if value.strip()]
        unknown = sorted(set(views) - set(webui_adapter.VIEWS))
        if unknown:
            raise SystemExit(f"unknown view(s): {unknown}")
        if len(seeds) != 1:
            raise SystemExit("--views tiles one seed across views; pass one seed")
    else:
        views = []
        if args.view not in webui_adapter.VIEWS:
            raise SystemExit(f"unknown view: {args.view!r}")

    started = time.perf_counter()
    if args.early:
        sheet = early_sheet(seeds[0], args.pixels, args.scale)
    elif views:
        sheet = view_row(seeds[0], views, args.pixels, args.scale)
    else:
        sheet = contact_sheet(seeds, args.view, args.pixels, args.scale)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, format="PNG", optimize=True)
    if args.early:
        label = "early snapshots"
    else:
        label = f"{len(views)} views" if views else f"{len(seeds)} seeds"
    print(f"\n{label} in {time.perf_counter() - started:.1f}s")
    print(f"{sheet.width} x {sheet.height} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
