"""Synthetic calibration panels for the single-image layer audit.

These are instruments, never candidates. Each one has a mechanism that is
known exactly because it is written here, which is what lets a judge be scored
instead of merely trusted. A batch whose formulaic controls go unflagged is
void, and a batch whose process controls are all called formulaic is void the
other way: an indiscriminate judge cannot clear anything.

Nothing in here is a design proposal. The process controls exist to prove the
judge can pass something; they are deliberately crude and must never be mined
as a source of engine mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .palette import categorical_rgb, scalar_rgb

FORMULAIC = "control_formulaic"
PROCESS = "control_process"


@dataclass(slots=True, frozen=True)
class Control:
    """One rendered calibration panel and the answer it is scored against."""

    control_id: str
    kind: str
    true_mechanism: str
    rgb: np.ndarray


def _torus_delta(coordinate: np.ndarray, origin: float, size: int) -> np.ndarray:
    delta = np.abs(coordinate - origin)
    return np.minimum(delta, size - delta)


def _point_distances(rng: np.random.Generator, size: int, count: int) -> np.ndarray:
    """Stacked wrapped Euclidean distances to `count` random points."""
    axis = np.arange(size, dtype=np.float64)
    points = rng.integers(0, size, size=(count, 2))
    stack = np.empty((count, size, size), dtype=np.float64)
    for index, (py, px) in enumerate(points):
        dy = _torus_delta(axis, float(py), size)[:, None]
        dx = _torus_delta(axis, float(px), size)[None, :]
        stack[index] = np.hypot(dy, dx)
    return stack


def _value_noise(rng: np.random.Generator, size: int, period: int) -> np.ndarray:
    """One octave of wrapping value noise with smoothstep interpolation."""
    lattice = rng.random((period, period))
    coordinate = np.arange(size, dtype=np.float64) * period / size
    cell = coordinate.astype(np.int64) % period
    frac = coordinate - np.floor(coordinate)
    smooth = frac * frac * (3.0 - 2.0 * frac)
    nxt = (cell + 1) % period

    top = (lattice[np.ix_(cell, cell)] * (1 - smooth)[None, :]
           + lattice[np.ix_(cell, nxt)] * smooth[None, :])
    bottom = (lattice[np.ix_(nxt, cell)] * (1 - smooth)[None, :]
              + lattice[np.ix_(nxt, nxt)] * smooth[None, :])
    return top * (1 - smooth)[:, None] + bottom * smooth[:, None]


def fractal_noise(rng: np.random.Generator, size: int, *, octaves: int = 6,
                  base_period: int = 4) -> np.ndarray:
    """Summed wrapping value-noise octaves. Seamless on the torus by design."""
    total = np.zeros((size, size), dtype=np.float64)
    amplitude = 1.0
    weight = 0.0
    for octave in range(octaves):
        period = base_period * 2**octave
        if period > size:
            break
        total += amplitude * _value_noise(rng, size, period)
        weight += amplitude
        amplitude *= 0.5
    return total / weight


def triangle_lattice(rng: np.random.Generator, size: int) -> np.ndarray:
    """Two superimposed triangle waves along integer lattice directions.

    This is the shape of the C02 resistance field, reproduced here as the
    reference example of a field that cannot be anything but a rigid plaid.
    """
    axis = np.arange(size, dtype=np.float64)
    y, x = axis[:, None], axis[None, :]
    field = np.zeros((size, size), dtype=np.float64)
    for _ in range(2):
        kx, ky = rng.integers(1, 4, size=2)
        wavelength = float(rng.integers(size // 12, size // 5))
        phase = rng.random() * wavelength
        projected = (kx * x + ky * y + phase) / wavelength
        field += np.abs(2.0 * (projected - np.floor(projected + 0.5)))
    return field


def radial_cost(rng: np.random.Generator, size: int) -> np.ndarray:
    """Distance to the nearest of several point sources. Constant curvature."""
    return _point_distances(rng, size, 5).min(axis=0)


def voronoi_cells(rng: np.random.Generator, size: int, classes: int = 7) -> np.ndarray:
    """Nearest-point class assignment: equant cells with straight contacts."""
    return _point_distances(rng, size, classes).argmin(axis=0).astype(np.int64)


def accretion_growth(rng: np.random.Generator, size: int, classes: int = 7) -> np.ndarray:
    """Stochastic competitive growth through a shared heterogeneous medium.

    Uniform-rate growth from points converges on the equidistance set, which is
    a Voronoi diagram — the formulaic control. The medium is what breaks that:
    one contrast-enhanced noise field slows every actor in the same places, so
    fronts channel and lobe instead of meeting halfway. Staggered starts and a
    wide vigour spread keep the actors unequal.

    Crude on purpose. It exists to prove a judge can pass something, and is not
    a proposal for how the engine should grow anything.
    """
    medium = fractal_noise(rng, size, octaves=5, base_period=3)
    medium = (medium - medium.min()) / max(float(np.ptp(medium)), 1e-12)
    medium = 0.06 + 0.94 * medium**2

    labels = np.full((size, size), -1, dtype=np.int64)
    germs = []
    for actor in range(classes):
        while True:
            y, x = rng.integers(0, size, size=2)
            if labels[y, x] == -1:
                germs.append((actor, int(y), int(x)))
                break
    vigour = rng.uniform(0.15, 1.0, size=classes)
    start = rng.integers(0, 40, size=classes)

    step = 0
    while (labels == -1).any():
        for actor, y, x in germs:
            if step == start[actor] and labels[y, x] == -1:
                labels[y, x] = actor
        # One draw per step, shared by every actor: the vigour multiplier still
        # separates them, and this is the loop's dominant cost.
        draw = rng.random((size, size))
        for actor in rng.permutation(classes):
            claimed = labels == actor
            if not claimed.any():
                continue
            front = np.zeros_like(claimed)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy or dx:
                        front |= np.roll(np.roll(claimed, dy, 0), dx, 1)
            take = front & (labels == -1) & (draw < vigour[actor] * medium)
            labels[take] = actor
        step += 1
    return labels


def build_controls(seed: int, size: int) -> list[Control]:
    """Render the full calibration set at panel resolution.

    Two formulaic and two process controls, one scalar and one categorical of
    each, so a batch of either panel type has calibration on both sides.
    """
    rng = np.random.default_rng(seed)
    return [
        Control("triangle_lattice", FORMULAIC, "periodic_waves",
                scalar_rgb(triangle_lattice(rng, size))),
        Control("radial_cost", FORMULAIC, "distance_or_cost_field",
                scalar_rgb(radial_cost(rng, size))),
        Control("voronoi_cells", FORMULAIC, "distance_or_cost_field",
                categorical_rgb(voronoi_cells(rng, size))),
        Control("fractal_noise", PROCESS, "filtered_noise",
                scalar_rgb(fractal_noise(rng, size))),
        Control("accretion_growth", PROCESS, "iterative_growth",
                categorical_rgb(accretion_growth(rng, size))),
    ]


__all__ = [
    "Control",
    "FORMULAIC",
    "PROCESS",
    "accretion_growth",
    "build_controls",
    "fractal_noise",
    "radial_cost",
    "triangle_lattice",
    "voronoi_cells",
]
