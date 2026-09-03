"""Array to pixels. Nothing is drawn on top of anything here.

Every function returns an `(n, n, 3)` uint8 image at native history
resolution. No text, no borders, no legends, no markers, and no resampling.
The vertical flip that puts north up belongs to the renderer, not here.
"""

from __future__ import annotations

import numpy as np

from eval.palette import CATEGORY_COLORS, categorical_rgb, scalar_rgb

REGIME_COLORS = {
    0: (226, 201, 79),    # shear
    1: (25, 187, 190),    # divergent
    2: (213, 73, 91),     # convergent
}


def categorical(labels: np.ndarray) -> np.ndarray:
    """Plate labels through the shared category table; -1 is black."""
    labels = np.asarray(labels)
    rgb = categorical_rgb(np.mod(labels, len(CATEGORY_COLORS)))
    rgb[labels < 0] = 0
    return rgb


def scalar(field: np.ndarray) -> np.ndarray:
    """A continuous field through the shared ramp, normalized to its own range."""
    return scalar_rgb(field)


def banded(field: np.ndarray, bands: int = 8) -> np.ndarray:
    """The same field quantized to `bands` equal-width levels.

    The contour companion `VIEWS.md` asks for: a ramp hides a gradient that
    changes slope, and a band edge does not.
    """
    if bands < 2:
        raise ValueError("bands must be at least two")
    values = np.asarray(field, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return scalar_rgb(np.zeros(values.shape, dtype=np.float64))
    level = np.floor((values - low) / (high - low) * bands)
    return scalar_rgb(np.clip(level, 0, bands - 1))


def mask(field: np.ndarray) -> np.ndarray:
    """White where true, black where false."""
    flags = np.asarray(field, dtype=bool)
    rgb = np.zeros(flags.shape + (3,), dtype=np.uint8)
    rgb[flags] = 255
    return rgb


def regime_rgb(regime: np.ndarray) -> np.ndarray:
    """Shear, divergent, and convergent on weak ground; black elsewhere."""
    values = np.asarray(regime)
    rgb = np.zeros(values.shape + (3,), dtype=np.uint8)
    for code, colour in REGIME_COLORS.items():
        rgb[values == code] = colour
    return rgb


def vector(v: np.ndarray) -> np.ndarray:
    """Direction as hue, magnitude as brightness. No arrow glyphs."""
    field = np.asarray(v, dtype=np.float64)
    magnitude = np.sqrt(field[0] ** 2 + field[1] ** 2)
    peak = float(magnitude.max())
    hue = np.mod(np.arctan2(field[1], field[0]) / (2.0 * np.pi), 1.0)
    value = 0.15 + 0.85 * (magnitude / peak if peak > 0.0 else magnitude)
    return hsv_to_rgb(hue, np.ones_like(hue), value)


def hsv_to_rgb(hue: np.ndarray, saturation: np.ndarray,
               value: np.ndarray) -> np.ndarray:
    """The six-sector conversion, vectorized over whole arrays."""
    sector = np.floor(hue * 6.0)
    fraction = hue * 6.0 - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))
    index = np.mod(sector.astype(np.int64), 6)
    channels = np.stack([
        np.select([index == 0, index == 1, index == 2,
                   index == 3, index == 4, index == 5],
                  [value, q, p, p, t, value]),
        np.select([index == 0, index == 1, index == 2,
                   index == 3, index == 4, index == 5],
                  [t, value, value, q, p, p]),
        np.select([index == 0, index == 1, index == 2,
                   index == 3, index == 4, index == 5],
                  [p, p, t, value, value, q]),
    ], axis=-1)
    return np.clip(np.rint(channels * 255.0), 0, 255).astype(np.uint8)


#: The longest arrow `arrows` draws, in history cells. The fastest body gets
#: this length and every other is scaled against it, so the picture is of the
#: *relative* motion of the bodies and not of an absolute speed.
ARROW_CELLS = 12

#: How far back the two barbs of the head lean from the shaft, in radians,
#: and how long they are as a share of the shaft.
ARROW_BARB_RADIANS = 2.6
ARROW_BARB_SHARE = 0.3


def arrows(rgb: np.ndarray, centroid: np.ndarray, motion: np.ndarray,
           *, length_cells: float = ARROW_CELLS) -> np.ndarray:
    """One arrow per body, drawn over `rgb`. The one view that draws on top.

    Every other function in this module returns a raster and nothing is drawn
    over it; `pieces_motion` is the exception `WORK_ORDER_C04_2.md` §5 asks
    for, because a body's velocity is three numbers and not a field, and a
    field view has nowhere to put it. The arrow is coloured the way the
    `velocity` view colours a cell — hue for direction, brightness for speed
    — so the two read alike, and its length carries the same speed again.

    `centroid` and `motion` are both `(2, N)`, x then y, in cell units and in
    km/Myr. Positions wrap. A body with no motion at all gets no arrow.
    """
    out = np.array(rgb, dtype=np.uint8, copy=True)
    centroid = np.asarray(centroid, dtype=np.float64)
    motion = np.asarray(motion, dtype=np.float64)
    if centroid.size == 0:
        return out
    rows = out.shape[0]
    columns = out.shape[1]
    speed = np.hypot(motion[0], motion[1])
    peak = float(speed.max())
    if peak <= 0.0:
        return out
    scale = float(length_cells) / peak
    hue = np.mod(np.arctan2(motion[1], motion[0]) / (2.0 * np.pi), 1.0)
    colour = hsv_to_rgb(hue, np.ones_like(hue), 0.15 + 0.85 * speed / peak)

    shaft = np.linspace(0.0, 1.0, max(2, int(round(2.0 * length_cells)) + 1))
    x = centroid[0][:, None] + motion[0][:, None] * scale * shaft[None, :]
    y = centroid[1][:, None] + motion[1][:, None] * scale * shaft[None, :]

    angle = np.arctan2(motion[1], motion[0])
    barb = ARROW_BARB_SHARE * scale * speed
    tip_x = centroid[0] + motion[0] * scale
    tip_y = centroid[1] + motion[1] * scale
    for lean in (ARROW_BARB_RADIANS, -ARROW_BARB_RADIANS):
        x = np.concatenate(
            (x, tip_x[:, None] + (barb * np.cos(angle + lean))[:, None]
             * shaft[None, :]), axis=1)
        y = np.concatenate(
            (y, tip_y[:, None] + (barb * np.sin(angle + lean))[:, None]
             * shaft[None, :]), axis=1)

    ix = np.mod(np.rint(x).astype(np.int64), columns)
    iy = np.mod(np.rint(y).astype(np.int64), rows)
    out[iy, ix] = np.repeat(colour[:, None, :], ix.shape[1], axis=1)
    return out


__all__ = [
    "ARROW_CELLS",
    "REGIME_COLORS",
    "arrows",
    "banded",
    "categorical",
    "hsv_to_rgb",
    "mask",
    "regime_rgb",
    "scalar",
    "vector",
]
