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
# KR palettes. Stop placement is calibrated to measured hypsometry (median
# land ~250 m, summits ~4700 m, abyssal bulk -3000..-4600): the canon
# family spends its color budget densely near sea level (canon quality 7),
# compresses the mids into a long tan band, turns *darker* with height
# (maroon -> near-black violet) with sparse snow-grey only above ~4500 m,
# and flattens the deep-ocean gradient so the abyss reads calm.
_CLASSIC_WATER = [
    (-10000, 5, 12, 35),
    (-7000, 10, 22, 60),
    (-4500, 18, 40, 100),
    (-2500, 30, 65, 140),
    (-1000, 45, 100, 175),
    (-200, 80, 150, 200),
    (-50, 120, 185, 210),
    (0, 150, 210, 220),
]
_CLASSIC_LAND = [
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
_CANON_WATER = [
    (-9500, 4, 10, 24),
    (-7200, 8, 18, 40),
    (-5800, 14, 30, 60),
    (-4600, 21, 44, 84),
    (-2600, 27, 56, 102),      # -2600..-4600: near-flat gradient, calm abyss
    (-1500, 33, 70, 120),
    (-800, 40, 88, 141),
    (-400, 52, 116, 165),
    (-200, 72, 152, 189),
    (-120, 100, 184, 204),     # bright turquoise platform band
    (-50, 138, 205, 212),
    (0, 172, 219, 220),
]
_CANON_LAND = [
    (0, 96, 133, 86),          # lowest land: sage — valley floors read dark
    (70, 120, 151, 97),
    (160, 149, 167, 111),
    (300, 177, 183, 129),
    (500, 198, 192, 160),      # plains ceiling — 60% of land lives below here
    (900, 206, 196, 168),      # long compressed tan mid-band
    (1500, 195, 178, 148),
    (2100, 170, 142, 110),
    (2700, 137, 100, 62),
    (3200, 105, 68, 47),
    (3700, 76, 44, 38),
    (4100, 52, 30, 34),
    (4400, 38, 26, 38),        # near-black violet crest
    (4550, 130, 128, 138),     # sparse rock/snow transition (~top 0.05%)
    (4750, 232, 234, 240),
]
_SOFT_WATER = [
    (-9500, 7, 16, 36),
    (-7500, 12, 26, 52),
    (-6000, 18, 38, 72),
    (-4600, 26, 52, 96),
    (-3000, 30, 62, 110),
    (-1700, 36, 75, 125),
    (-900, 43, 92, 145),
    (-450, 55, 120, 168),
    (-200, 76, 155, 190),
    (-120, 104, 186, 205),
    (-50, 142, 207, 213),
    (0, 175, 220, 221),
]
_SOFT_LAND = [
    (0, 108, 144, 94),
    (80, 132, 160, 104),
    (180, 158, 173, 117),
    (320, 184, 188, 134),
    (550, 203, 196, 164),
    (1000, 207, 196, 167),
    (1600, 192, 172, 142),
    (2200, 164, 134, 105),
    (2800, 128, 92, 60),
    (3300, 100, 64, 48),
    (3800, 74, 46, 44),
    (4200, 92, 82, 84),
    (4600, 226, 228, 233),
]
_CRISP_WATER = [
    (-9500, 3, 8, 20),
    (-7200, 7, 15, 34),
    (-5800, 12, 26, 54),
    (-4600, 19, 40, 78),
    (-2400, 25, 52, 96),
    (-1300, 31, 66, 115),
    (-600, 38, 84, 136),
    (-260, 46, 104, 154),
    (-170, 60, 132, 176),      # hard shelf-break knee
    (-140, 88, 178, 201),
    (-60, 130, 204, 211),
    (0, 168, 220, 219),
]
_CRISP_LAND = [
    (0, 88, 128, 80),
    (60, 116, 149, 94),
    (140, 148, 167, 108),
    (260, 180, 185, 128),
    (420, 203, 195, 161),
    (800, 209, 197, 168),
    (1400, 196, 177, 145),
    (2000, 168, 138, 107),
    (2600, 132, 95, 58),
    (3100, 100, 63, 44),
    (3600, 70, 40, 36),
    (4050, 48, 28, 33),
    (4380, 36, 24, 36),
    (4550, 134, 132, 142),
    (4750, 236, 238, 243),
]

# render_palette control indexes this. name -> (water stops, land stops)
PALETTES: dict[int, tuple[str, list, list]] = {
    0: ("classic", _CLASSIC_WATER, _CLASSIC_LAND),
    1: ("canon", _CANON_WATER, _CANON_LAND),
    2: ("canon-soft", _SOFT_WATER, _SOFT_LAND),
    3: ("canon-crisp", _CRISP_WATER, _CRISP_LAND),
}


def _palette(world: World) -> tuple[list, list]:
    idx = int(world.controls.get("render_palette", 1))
    _, wstops, lstops = PALETTES.get(idx, PALETTES[1])
    return wstops, lstops


def _ramp(e: np.ndarray, stops: list[tuple]) -> np.ndarray:
    xs = np.array([s[0] for s in stops], dtype=np.float64)
    out = np.empty(e.shape + (3,), dtype=np.uint8)
    for ch in range(3):
        ys = np.array([s[1 + ch] for s in stops], dtype=np.float64)
        out[..., ch] = np.clip(np.interp(e, xs, ys), 0, 255).astype(np.uint8)
    return out


def _quantize(e: np.ndarray, top: float, n: int) -> np.ndarray:
    """Quantize magnitudes into n bands uniform in sqrt-space: dense near
    zero (plains, platform seas), sparse toward the extreme — the same
    budget philosophy as the canon ramp stops."""
    t = np.sqrt(np.clip(e, 0.0, top) / top)
    tq = (np.clip(np.floor(t * n), 0, n - 1) + 0.5) / n
    return top * tq * tq


def hypsometric(world: World) -> Image.Image:
    e = world["elevation"].astype(np.float64)
    wstops, lstops = _palette(world)
    q = int(world.controls.get("render_quantize", 0))
    land = e >= 0.0
    ev = e.copy()
    if q > 0:  # quantize land and water separately: coastline stays exact
        ev[land] = _quantize(e[land], float(lstops[-1][0]), q)
        ev[~land] = -_quantize(-e[~land], float(-wstops[0][0]), q)
    rgb = np.where(land[..., None], _ramp(ev, lstops), _ramp(ev, wstops))
    if "lake_id" in world.layers:               # lakes color by their depth
        lakes = world["lake_id"] > 0
        if lakes.any():
            depth = (world["lake_level"].astype(np.float64) - e)
            rgb[lakes] = _ramp(-np.clip(depth[lakes], 90.0, 300.0), wstops)
    return Image.fromarray(rgb, "RGB")


def view_drainage(world: World) -> Image.Image:
    """W1: log-scaled flow accumulation over a dimmed terrain base —
    the dendritic skeleton before any carving exists."""
    e = world["elevation"].astype(np.float64)
    acc = world["flow_acc"].astype(np.float64)
    land = e >= 0.0
    base = np.empty(e.shape + (3,), dtype=np.float64)
    t = np.clip(e / 4500.0, 0.0, 1.0)[..., None]
    base[land] = (np.array([44, 52, 40]) + t[land] * np.array([60, 48, 40]))
    tw = np.clip(-e / 6000.0, 0.0, 1.0)[..., None]
    base[~land] = (np.array([16, 22, 40]) - tw[~land] * np.array([8, 12, 22]))

    cell_area = world.cell_km ** 2
    a = np.log1p(acc / cell_area)
    a /= max(float(a.max()), 1e-9)
    ch = np.clip((a - 0.18) / 0.82, 0.0, 1.0) ** 1.6
    ch = np.where(land, ch, 0.0)[..., None]
    river = np.array([120, 190, 255], dtype=np.float64)
    rgb = base * (1.0 - ch) + river * ch
    if "lake_id" in world.layers:
        rgb[world["lake_id"] > 0] = (70, 140, 220)
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


# --- debug views -------------------------------------------------------------

def view_plates(world: World) -> Image.Image:
    """Plate partition: one hue per plate, boundaries dark, continental
    crust lightened, motion arrows at plate centroids."""
    import colorsys

    pid = world["plate_id"]
    n = int(pid.max()) + 1
    pal = np.array(
        [[int(255 * v) for v in colorsys.hsv_to_rgb((i * 0.618034) % 1.0,
                                                    0.55, 0.72)]
         for i in range(n)], dtype=np.uint8)
    rgb = pal[pid]
    if "plate_interior" in world.layers:  # A1: dark rims -> bright interiors
        sh = 0.55 + 0.45 * world["plate_interior"].astype(np.float64)
        rgb = (rgb * sh[..., None]).astype(np.uint8)

    crust = world["crust"].astype(bool)
    rgb = np.where(crust[..., None],
                   (rgb.astype(np.int16) * 5 + np.array([235, 230, 205]) * 3)
                   // 8, rgb).astype(np.uint8)

    edge = np.zeros(world.shape, dtype=bool)
    edge[:, 1:] |= pid[:, 1:] != pid[:, :-1]
    edge[1:, :] |= pid[1:, :] != pid[:-1, :]
    rgb[edge] = (rgb[edge] * 0.25).astype(np.uint8)

    img = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(img)
    meta = world.meta.get("plates", {})
    poles = np.array(meta.get("poles_km", []))
    omega = np.array(meta.get("omega", []))
    if poles.size:
        xkm, ykm = world.coords_km()
        speeds = np.hypot(world["plate_vx"], world["plate_vy"])
        ref = max(float(np.median(speeds)), 1e-9)
        alen = 0.07 * min(world.shape)
        for i in range(n):
            m = pid == i
            if not m.any():
                continue
            rows, cols = np.nonzero(m)
            cy, cx = float(rows.mean()), float(cols.mean())
            px = float(xkm[int(cy), int(cx)])
            py = float(ykm[int(cy), int(cx)])
            vx = -omega[i] * (py - poles[i, 1])
            vy = omega[i] * (px - poles[i, 0])
            s = alen / ref
            ex, ey = cx + vx * s, cy + vy * s
            draw.line([cx, cy, ex, ey], fill=(20, 20, 20), width=3)
            draw.line([cx, cy, ex, ey], fill=(255, 255, 255), width=1)
            draw.ellipse([ex - 2, ey - 2, ex + 2, ey + 2],
                         fill=(255, 255, 255))
    return img


def view_uplift(world: World) -> Image.Image:
    """Net tectonic forcing: blue = trench/graben, white = neutral,
    red = orogeny. The causal skeleton under the hypsometric view."""
    u = world["uplift"].astype(np.float64)
    lim = max(float(np.percentile(np.abs(u), 99.5)), 1.0)
    t = np.clip(u / lim, -1.0, 1.0)
    neg = np.array([40, 80, 200], dtype=np.float64)
    mid = np.array([240, 240, 238], dtype=np.float64)
    pos = np.array([190, 60, 40], dtype=np.float64)
    rgb = np.where(t[..., None] < 0,
                   mid + (-t[..., None]) * (neg - mid),
                   mid + t[..., None] * (pos - mid))
    edge = np.zeros(world.shape, dtype=bool)
    pid = world["plate_id"]
    edge[:, 1:] |= pid[:, 1:] != pid[:, :-1]
    edge[1:, :] |= pid[1:, :] != pid[:-1, :]
    rgb[edge] *= 0.55
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def view_volcanic(world: World) -> Image.Image:
    """Volcanism placement audit: dim hypsometry + red volcanic field
    (arc lines, rift lines, hotspot chains) — also the export flag."""
    base = np.asarray(hypsometric(world), dtype=np.float64) * 0.45
    v = world["volcanic"].astype(np.float64)[..., None]
    hot = np.array([255, 70, 30], dtype=np.float64)
    rgb = base * (1.0 - v) + hot * v
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def view_margins(world: World) -> Image.Image:
    """R1 memory layers: ocean shows crust age (light young -> dark old),
    land is neutral gray, coasts tinted by margin activity (red active,
    green passive)."""
    e = world["elevation"].astype(np.float64)
    age = world["crust_age"].astype(np.float64)
    act = world["margin_activity"].astype(np.float64)
    landm = e >= 0.0

    young = np.array([120, 170, 205], dtype=np.float64)
    old = np.array([14, 24, 52], dtype=np.float64)
    rgb = young + age[..., None] * (old - young)
    rgb[landm] = (95, 95, 92)

    coast = landm & ~(np.roll(landm, 1, 0) & np.roll(landm, -1, 0)
                      & np.roll(landm, 1, 1) & np.roll(landm, -1, 1))
    a = act[coast][..., None]
    rgb[coast] = (np.array([70, 200, 90]) * (1 - a)
                  + np.array([230, 60, 40]) * a)
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def view_crust(world: World) -> Image.Image:
    """Continental potential vs threshold: blues rising to the crust cut,
    greens above it."""
    pot = world["crust_potential"].astype(np.float64)
    thr = float(world.meta["crust"]["threshold"])
    p98 = max(float(world.meta["crust"]["p98"]), thr * 1.0001)
    below = np.clip(pot / max(thr, 1e-9), 0.0, 1.0)
    above = np.clip((pot - thr) / (p98 - thr), 0.0, 1.0)
    crust = world["crust"].astype(bool)

    lo = np.array([12, 20, 48], dtype=np.float64)
    hi = np.array([90, 150, 190], dtype=np.float64)
    g0 = np.array([110, 150, 90], dtype=np.float64)
    g1 = np.array([235, 220, 150], dtype=np.float64)
    rgb = lo + below[..., None] * (hi - lo)
    rgb[crust] = (g0 + above[crust][..., None] * (g1 - g0))
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


# --- view registry -----------------------------------------------------------
# Every layer a milestone adds ships a view here (name -> (renderer,
# required layers)). The webui's view selector and batch --views read this;
# nothing else needs editing when a view is added.

VIEWS: dict[str, tuple] = {
    "hypsometric": (hypsometric, ("elevation",)),
    "plates": (view_plates, ("plate_id", "crust")),
    "crust": (view_crust, ("crust_potential", "crust")),
    "uplift": (view_uplift, ("uplift", "plate_id")),
    "margins": (view_margins, ("elevation", "crust_age", "margin_activity")),
    "volcanic": (view_volcanic, ("elevation", "volcanic")),
    "drainage": (view_drainage, ("elevation", "flow_acc")),
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
