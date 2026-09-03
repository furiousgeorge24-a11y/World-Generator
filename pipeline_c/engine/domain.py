"""Periodic-grid operators.

Every field on the history grid wraps in both axes. Neighbour access is
`np.roll` or explicit modulo and nothing else: no padding, no edge modes, no
clamping. Nothing in this module knows what a field means.
"""

from __future__ import annotations

import numpy as np


def roll_x(a: np.ndarray, k: int) -> np.ndarray:
    """Shift along the column axis, wrapping."""
    return np.roll(a, k, axis=-1)


def roll_y(a: np.ndarray, k: int) -> np.ndarray:
    """Shift along the row axis, wrapping."""
    return np.roll(a, k, axis=-2)


def ddx(a: np.ndarray) -> np.ndarray:
    """Periodic central difference along x, per cell."""
    return (roll_x(a, -1) - roll_x(a, 1)) * 0.5


def ddy(a: np.ndarray) -> np.ndarray:
    """Periodic central difference along y, per cell."""
    return (roll_y(a, -1) - roll_y(a, 1)) * 0.5


def grad(a: np.ndarray) -> np.ndarray:
    """(2, n, n) gradient, component 0 = x, component 1 = y."""
    return np.stack((ddx(a), ddy(a)))


def perp_grad(a: np.ndarray) -> np.ndarray:
    """(2, n, n) gradient rotated a quarter turn: the divergence-free part."""
    return np.stack((-ddy(a), ddx(a)))


def div(v: np.ndarray) -> np.ndarray:
    """Divergence of a (2, n, n) vector field, per cell."""
    return ddx(v[0]) + ddy(v[1])


def sample_bilinear_periodic(a: np.ndarray, x: np.ndarray,
                             y: np.ndarray) -> np.ndarray:
    """Bilinear sample of `a` at fractional cell coordinates, wrapping.

    Cell centre `i` sits at coordinate `i`. `a` is `(n, n)` or `(k, n, n)`;
    `x` and `y` are float arrays of any common shape.
    """
    rows, columns = a.shape[-2], a.shape[-1]
    xw = np.mod(np.asarray(x, dtype=np.float64), columns)
    yw = np.mod(np.asarray(y, dtype=np.float64), rows)
    x0 = np.floor(xw)
    y0 = np.floor(yw)
    fx = xw - x0
    fy = yw - y0
    i0 = x0.astype(np.int64) % columns
    j0 = y0.astype(np.int64) % rows
    i1 = (i0 + 1) % columns
    j1 = (j0 + 1) % rows
    top = a[..., j0, i0] * (1.0 - fx) + a[..., j0, i1] * fx
    bottom = a[..., j1, i0] * (1.0 - fx) + a[..., j1, i1] * fx
    return top * (1.0 - fy) + bottom * fy


def sample_nearest_periodic(a: np.ndarray, x: np.ndarray,
                            y: np.ndarray) -> np.ndarray:
    """Nearest-cell sample of `a` at fractional cell coordinates, wrapping.

    The companion of `sample_bilinear_periodic`, with the same conventions:
    cell centre `i` sits at coordinate `i`, `a` is `(n, n)` or `(k, n, n)`,
    and `x` and `y` are float arrays of any common shape.

    It exists because a field with a one-cell discontinuity in it cannot be
    interpolated. Bilinear sampling of a one-cell line spreads a share of its
    value into the neighbour it leans toward, and a few steps of that turns
    the line into a ramp two or three cells across. Nearest-cell sampling
    moves whatever it carries by whole cells and leaves a one-cell feature
    one cell wide; the price is that a displacement below half a cell rounds
    to no displacement at all, which is why the caller carries the remainder
    forward rather than discarding it.
    """
    rows, columns = a.shape[-2], a.shape[-1]
    i = np.mod(np.rint(np.asarray(x, dtype=np.float64)).astype(np.int64),
               columns)
    j = np.mod(np.rint(np.asarray(y, dtype=np.float64)).astype(np.int64),
               rows)
    return a[..., j, i]


def _prolong_axis_bilinear(a: np.ndarray, axis: int) -> np.ndarray:
    """Double one axis by periodic linear interpolation, centres aligned."""
    lower = np.roll(a, 1, axis=axis)
    upper = np.roll(a, -1, axis=axis)
    even = 0.75 * a + 0.25 * lower
    odd = 0.75 * a + 0.25 * upper
    shape = list(a.shape)
    shape[axis] *= 2
    out = np.empty(shape, dtype=np.float64)
    index = [slice(None)] * a.ndim
    index[axis] = slice(0, None, 2)
    out[tuple(index)] = even
    index[axis] = slice(1, None, 2)
    out[tuple(index)] = odd
    return out


def prolong_bilinear(a: np.ndarray) -> np.ndarray:
    """Periodic bilinear lift of an `(..., m, m)` field to `(..., 2m, 2m)`.

    Cell centres are aligned: fine cell `i` sits at coarse coordinate
    `(i + 0.5) / 2 - 0.5`, so each fine value is three quarters of the coarse
    cell it lies in and a quarter of the coarse neighbour it leans toward.

    This is **not** the solver's `prolong`, which lifts a coarse *correction*
    inside a V-cycle and must stay piecewise constant to remain the adjoint of
    the 2 x 2 mean restriction. This one lifts a solved velocity field, where
    smoothness is the point and adjointness is irrelevant.
    """
    values = np.asarray(a, dtype=np.float64)
    if values.shape[-1] != values.shape[-2]:
        raise ValueError("prolong_bilinear expects a square grid")
    return _prolong_axis_bilinear(_prolong_axis_bilinear(values, -1), -2)


def tile2x2(a: np.ndarray) -> np.ndarray:
    """Repeat a `(n, n)` or `(n, n, 3)` field twice in each spatial axis."""
    if a.ndim == 2:
        return np.tile(a, (2, 2))
    if a.ndim == 3:
        return np.tile(a, (2, 2, 1))
    raise ValueError("tile2x2 expects a (n, n) or (n, n, 3) array")


__all__ = [
    "ddx",
    "ddy",
    "div",
    "grad",
    "perp_grad",
    "prolong_bilinear",
    "roll_x",
    "roll_y",
    "sample_bilinear_periodic",
    "sample_nearest_periodic",
    "tile2x2",
]
