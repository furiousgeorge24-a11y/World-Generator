"""The C04.5 marker set, kept only so the two engines can be compared.

`WORK_ORDER_C04_6.md` §3.1 asks for the twelve-seed comparison row and §3.4
for seconds per world on both engines in the same session, and C04.5's report
did the same for C04.4 behind a private flag. This module is that flag's other
half: the three places the C04.5 seam set differed from the curve of §1, and
nothing else. `run_history(..., _c04_5=True)` reaches it; nothing in the
engine does.

It is temporary and is deleted before the build report, as C04.5 deleted its
own `_eight_directions` flag.
"""

from __future__ import annotations

import numpy as np

from .constants import SEAM_OPEN_STRENGTH, WEAK_THRESHOLD
from .markers import Markers, cells, healed_strength


def raster(markers: Markers, n: int) -> np.ndarray:
    """The C04.5 raster: the markers as points, with no segments drawn."""
    field = np.ones((n, n), dtype=np.float64)
    if markers.size == 0:
        return field
    rows, columns = cells(markers, n)
    flat = rows * n + columns
    order = np.lexsort((markers.s, flat))
    ordered_flat = flat[order]
    ordered_s = markers.s[order]
    first = np.empty(ordered_flat.size, dtype=bool)
    first[0] = True
    first[1:] = ordered_flat[1:] != ordered_flat[:-1]
    field.reshape(-1)[ordered_flat[first]] = ordered_s[first]
    return field


def create(markers: Markers, opened: np.ndarray,
           offsets: np.ndarray | None = None) -> Markers:
    """The C04.5 `create`: one marker per opened cell, at `p'` or the centre."""
    opened = np.asarray(opened, dtype=bool)
    if not opened.any():
        return markers
    rows, columns = np.nonzero(opened)
    x = columns.astype(np.float64)
    y = rows.astype(np.float64)
    if offsets is not None:
        offsets = np.asarray(offsets, dtype=np.float64)
        dx = offsets[0][rows, columns]
        dy = offsets[1][rows, columns]
        n = opened.shape[-1]
        x = np.mod(x + np.where(np.isnan(dx), 0.0, dx), n)
        y = np.mod(y + np.where(np.isnan(dy), 0.0, dy), n)
    fresh = np.full(rows.size, float(SEAM_OPEN_STRENGTH), dtype=np.float64)
    return Markers(x=np.concatenate((markers.x, x)),
                   y=np.concatenate((markers.y, y)),
                   s=np.concatenate((markers.s, fresh)))


def damage_and_heal(markers: Markers, excess: np.ndarray, heal_rate: float,
                    damage_rate: float, step_myr: float, n: int
                    ) -> tuple[Markers, int]:
    """The C04.5 removal: a marker leaves at `WEAK_THRESHOLD`."""
    if markers.size == 0:
        return markers, 0
    strength = healed_strength(markers, excess, heal_rate, damage_rate,
                               step_myr, n)
    keep = strength < WEAK_THRESHOLD
    removed = int(strength.size - keep.sum())
    return Markers(x=markers.x[keep], y=markers.y[keep],
                   s=strength[keep]), removed


__all__ = ["create", "damage_and_heal", "raster"]
