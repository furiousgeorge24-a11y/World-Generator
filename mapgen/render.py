"""Rendering: World -> PIL Image. Cheap by contract (section 2).

Hypsometric ramp approximating the author's reference: cyan shelf ->
navy abyss below sea level; green -> tan -> brown -> white above.
Nearest-neighbor aesthetic; no anti-aliasing.
"""

import json

import numpy as np
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

from .world import World

# (elevation_m, r, g, b) — ascending. Water uses e <= 0, land e >= 0.
_WATER = [
    (-10000, 5, 12, 35),
    (-7000, 10, 22, 60),
    (-4500, 18, 40, 100),
    (-2500, 30, 65, 140),
    (-1000, 45, 100, 175),
    (-200, 80, 150, 200),
    (-50, 120, 185, 210),
    (0, 150, 210, 220),
]
_LAND = [
    (0, 145, 175, 105),
    (250, 165, 190, 120),
    (700, 190, 205, 145),
    (1200, 205, 200, 150),
    (1800, 180, 150, 100),
    (2600, 135, 90, 60),
    (3400, 95, 60, 48),
    (4200, 130, 120, 115),
    (5000, 200, 200, 198),
    (6000, 250, 250, 250),
]


def _ramp(e: np.ndarray, stops: list[tuple]) -> np.ndarray:
    xs = np.array([s[0] for s in stops], dtype=np.float64)
    out = np.empty(e.shape + (3,), dtype=np.uint8)
    for ch in range(3):
        ys = np.array([s[1 + ch] for s in stops], dtype=np.float64)
        out[..., ch] = np.clip(np.interp(e, xs, ys), 0, 255).astype(np.uint8)
    return out


def _quantize(e: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
    step = (hi - lo) / n
    return lo + (np.clip(np.floor((e - lo) / step), 0, n - 1) + 0.5) * step


def hypsometric(world: World) -> Image.Image:
    e = world["elevation"].astype(np.float64)
    q = int(world.controls.get("render_quantize", 0))
    land = e >= 0.0
    ev = e.copy()
    if q > 0:  # quantize land and water separately: coastline stays exact
        ev[land] = _quantize(e[land], 0.0, 6000.0, q)
        ev[~land] = _quantize(e[~land], -8000.0, 0.0, q)
    rgb = np.where(land[..., None], _ramp(ev, _LAND), _ramp(ev, _WATER))
    return Image.fromarray(rgb, "RGB")


# --- view registry -----------------------------------------------------------
# Every layer a milestone adds ships a view here (name -> (renderer,
# required layers)). The webui's view selector and batch --views read this;
# nothing else needs editing when a view is added.

VIEWS: dict[str, tuple] = {
    "hypsometric": (hypsometric, ("elevation",)),
}


def available_views(world: World) -> list[str]:
    return [name for name, (_, req) in VIEWS.items()
            if all(r in world.layers for r in req)]


def render_view(world: World, name: str) -> Image.Image:
    fn, req = VIEWS.get(name, VIEWS["hypsometric"])
    if not all(r in world.layers for r in req):
        fn = hypsometric
    return fn(world)


def save_png(img: Image.Image, path: str, world: World) -> None:
    """Every PNG carries provenance (contract section 8)."""
    meta = PngInfo()
    meta.add_text("mapgen:version", world.version)
    meta.add_text("mapgen:seed", str(world.seed))
    meta.add_text("mapgen:size", f"{world.shape[1]}x{world.shape[0]}")
    meta.add_text("mapgen:controls", json.dumps(world.controls, sort_keys=True))
    img.save(path, format="PNG", pnginfo=meta)


def contact_sheet(entries: list[tuple[str, Image.Image]], cols: int = 3,
                  thumb: int = 320, pad: int = 6) -> Image.Image:
    """Labeled grid of renders — galleries are first-class (contract 12)."""
    rows = (len(entries) + cols - 1) // cols
    tiles = []
    for label, img in entries:
        t = img.copy()
        t.thumbnail((thumb, thumb), Image.NEAREST)
        ImageDraw.Draw(t).text((5, 4), label, fill=(255, 235, 80))
        tiles.append(t)
    cw = max(t.width for t in tiles)
    ch = max(t.height for t in tiles)
    sheet = Image.new("RGB", (cols * (cw + pad) + pad, rows * (ch + pad) + pad),
                      (24, 24, 28))
    for i, t in enumerate(tiles):
        r, cidx = divmod(i, cols)
        sheet.paste(t, (pad + cidx * (cw + pad), pad + r * (ch + pad)))
    return sheet
