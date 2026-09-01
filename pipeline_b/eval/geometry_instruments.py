"""Shared morphology/geometry instruments for formation experiments.

Extracted from the sealed private ensembles (field_accretion_periodic /
periodic_convergence_formation, 2026-08-31) so future runs stop
re-embedding gate code inside sealed spikes. The sealed files remain
byte-identical; THIS module is the maintained implementation.

Calibration findings (tests/geometry_instrument_checks.py, 2026-08-31),
which govern how these numbers may be used:

1. **The analytic 11/45 rotation null is the wrong null for raster
   boundaries.** Thresholded, perfectly isotropic fBm masks produce
   long-ruler near-D4 fractions of ~0.70 (p ~ 0 under the analytic
   null): curved digital boundaries locally quantize into axis/diagonal
   runs regardless of formation. A D4 result is therefore only
   evidence of formation-caused grid lock when compared against a
   matched isotropic baseline measured on the same grid and component
   scale — never against the analytic null alone. (The sealed B16/B17
   D4 gates used the analytic null; their near-fractions, 0.447 and
   0.385, sit BELOW the isotropic-raster baseline.)
2. A hypothesized angle-dependent detection bias in the flat 0.70
   density rule was tested and REFUTED: the ±1-cell normal band holds
   both staircase rows on diagonals, and measured detection efficiency
   is 1.00 at 0/15/30/45 degrees in both modes. ``angle_fair`` is
   retained as a tested-equivalent option; the default reproduces the
   sealed semantics exactly (verified against a direct port).
3. The seed-blocked null has a power floor: with one ruler per block,
   k blocks can never beat p = (11/45)^k, so p<0.01 needs >= 4
   independent blocks even on maximally guilty input.
4. Finite (non-periodic) masks are first-class here: no torus unwrap
   is required, and Crofton perimeters pad instead of wrapping.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from engine.rng import fnv1a64

D4_TOLERANCE_DEGREES = 5.0
RULER_BAND_CELLS = 1.0
RULER_GAP_CELLS = 2.0
RULER_DENSITY_RATIO = 0.70
D4_RANDOMIZATION_TRIALS = 200_000
D4_RANDOMIZATION_SALT = "geometry-instruments-d4-null"

BLOB_COMPACTNESS_MIN = 0.72
BLOB_SOLIDITY_MIN = 0.90


# ---------------------------------------------------------------- masks

def label_components(mask, periodic=False):
    """Deterministic 4-connected components -> list of (ys, xs) index
    arrays, ordered by first cell in raster order."""
    mask = np.asarray(mask, bool)
    n0, n1 = mask.shape
    seen = np.zeros_like(mask)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        ys, xs = [], []
        dq = deque([(int(y0), int(x0))])
        seen[y0, x0] = True
        while dq:
            y, x = dq.popleft()
            ys.append(y)
            xs.append(x)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if periodic:
                    ny %= n0
                    nx %= n1
                elif not (0 <= ny < n0 and 0 <= nx < n1):
                    continue
                if mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    dq.append((ny, nx))
        out.append((np.asarray(ys, np.int64), np.asarray(xs, np.int64)))
    return out


def boundary_mask(mask, periodic=False):
    """Cells of ``mask`` with at least one 4-neighbor outside it."""
    mask = np.asarray(mask, bool)
    if periodic:
        interior = mask.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            interior &= np.roll(mask, (dy, dx), (0, 1))
    else:
        padded = np.pad(mask, 1, constant_values=False)
        interior = mask.copy()
        interior &= padded[2:, 1:-1]
        interior &= padded[:-2, 1:-1]
        interior &= padded[1:-1, 2:]
        interior &= padded[1:-1, :-2]
    return mask & ~interior


# ---------------------------------------------------------------- rulers

def max_ruler_run(by, bx, *, angle_fair=False,
                  band_cells=RULER_BAND_CELLS,
                  gap_cells=RULER_GAP_CELLS,
                  density_ratio=RULER_DENSITY_RATIO):
    """Longest boundary chord staying within ``band_cells`` of a straight
    line, scanned over integer degrees 0..179.

    ``by``/``bx`` are boundary-point coordinates in any consistent
    (already unwrapped, cell-unit) frame. Returns dict with ``cells``
    (tangent span), ``angle_degrees``, ``density``. Semantics match the
    sealed `_maximum_ruler_run` when ``angle_fair=False``.
    """
    y = np.asarray(by, np.float64)
    x = np.asarray(bx, np.float64)
    if y.size < 2:
        return {"cells": 0.0, "angle_degrees": None, "density": 0.0}
    best_len = 0.0
    best_angle = None
    best_density = 0.0
    for degrees in range(180):
        angle = np.radians(float(degrees))
        cosine = np.cos(angle)
        sine = np.sin(angle)
        t = x * cosine + y * sine
        nrm = -x * sine + y * cosine
        keys0 = np.rint(nrm)
        keys_l, ts_l = [], []
        for off in (-1.0, 0.0, 1.0):
            keys = keys0 + off
            ok = np.abs(nrm - keys) <= band_cells
            if ok.any():
                keys_l.append(keys[ok])
                ts_l.append(t[ok])
        if not keys_l:
            continue
        k_all = np.concatenate(keys_l)
        t_all = np.concatenate(ts_l)
        order = np.lexsort((t_all, k_all))
        k_s = k_all[order]
        t_s = t_all[order]
        new_seg = np.ones(k_s.size, bool)
        if k_s.size > 1:
            new_seg[1:] = ((k_s[1:] != k_s[:-1])
                           | (np.diff(t_s) > gap_cells))
        starts = np.flatnonzero(new_seg)
        ends = np.append(starts[1:], k_s.size) - 1
        spans = t_s[ends] - t_s[starts] + 1.0
        counts = (ends - starts + 1).astype(np.float64)
        density = counts / np.maximum(spans, 1.0)
        required = density_ratio
        if angle_fair:
            required = density_ratio * max(abs(cosine), abs(sine))
        valid = density >= required
        if not valid.any():
            continue
        idx = int(np.flatnonzero(valid)[np.argmax(spans[valid])])
        if spans[idx] > best_len:
            best_len = float(spans[idx])
            best_angle = float(degrees)
            best_density = float(density[idx])
    return {"cells": best_len, "angle_degrees": best_angle,
            "density": best_density}


def rulers_for_mask(mask, cell_km, min_run_km, *, periodic=False,
                    angle_fair=False, min_component_cells=8):
    """All component ruler records of at least ``min_run_km``."""
    mask = np.asarray(mask, bool)
    out = []
    for ys, xs in label_components(mask, periodic=periodic):
        if ys.size < min_component_cells:
            continue
        comp = np.zeros_like(mask)
        comp[ys, xs] = True
        if periodic:
            unwrapped = unwrap_periodic_component(
                comp, (int(ys[0]), int(xs[0])))
            if unwrapped["component_winds_torus"]:
                continue
            b = boundary_mask(comp, periodic=True)
            byy, bxx = np.nonzero(b)
            run = max_ruler_run(unwrapped["relative_y"][byy, bxx],
                                unwrapped["relative_x"][byy, bxx],
                                angle_fair=angle_fair)
        else:
            b = boundary_mask(comp, periodic=False)
            byy, bxx = np.nonzero(b)
            run = max_ruler_run(byy, bxx, angle_fair=angle_fair)
        if (run["angle_degrees"] is not None
                and run["cells"] * cell_km >= min_run_km):
            out.append({
                "length_km": float(run["cells"] * cell_km),
                "angle_degrees": run["angle_degrees"],
                "density": run["density"],
                "component_cells": int(ys.size),
            })
    return out


def unwrap_periodic_component(comp, pivot_yx):
    """Assign unwrapped relative coordinates from a pivot cell by BFS;
    detect components that wind around the torus."""
    comp = np.asarray(comp, bool)
    n0, n1 = comp.shape
    rel_y = np.zeros((n0, n1), np.int64)
    rel_x = np.zeros((n0, n1), np.int64)
    seen = np.zeros_like(comp)
    winding = set()
    py, px = pivot_yx
    seen[py, px] = True
    dq = deque([(py, px)])
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = (y + dy) % n0, (x + dx) % n1
            if not comp[ny, nx]:
                continue
            cy = rel_y[y, x] + dy
            cx = rel_x[y, x] + dx
            if seen[ny, nx]:
                wy, wx = cy - rel_y[ny, nx], cx - rel_x[ny, nx]
                if wy or wx:
                    winding.add((int(wy), int(wx)))
                continue
            seen[ny, nx] = True
            rel_y[ny, nx] = cy
            rel_x[ny, nx] = cx
            dq.append((ny, nx))
    return {"relative_y": rel_y, "relative_x": rel_x,
            "component_winds_torus": bool(winding),
            "winding_vectors_yx": sorted(winding)}


# ------------------------------------------------------------------- D4

def d4_distance_degrees(angle_degrees):
    folded = np.mod(np.asarray(angle_degrees, np.float64), 45.0)
    return np.minimum(folded, 45.0 - folded)


def seed_blocked_d4_test(rulers, *, trials=D4_RANDOMIZATION_TRIALS,
                         salt=D4_RANDOMIZATION_SALT,
                         tolerance=D4_TOLERANCE_DEGREES):
    """Sealed-equivalent seed-blocked rotation null: one shared integer
    rotation per block per trial preserves within-block dependence.
    ``rulers`` items need ``seed`` (block id) and ``angle_degrees``."""
    observed = sum(
        float(d4_distance_degrees(r["angle_degrees"])) <= tolerance
        for r in rulers)
    if not rulers:
        return {"observed_near_d4_count": 0, "ruler_count": 0,
                "near_fraction": 0.0,
                "randomization_upper_tail_p": 1.0, "trials": trials}
    grouped = {}
    for r in rulers:
        grouped.setdefault(r["seed"], []).append(
            float(r["angle_degrees"]))
    rng = np.random.default_rng(fnv1a64(salt))
    at_least = 0
    remaining = trials
    while remaining:
        count = min(5000, remaining)
        simulated = np.zeros(count, np.int64)
        for seed in sorted(grouped):
            rotations = rng.integers(0, 180, size=count)
            angles = np.asarray(grouped[seed], np.float64)
            rotated = np.mod(
                rotations[:, None].astype(np.float64)
                + angles[None, :], 180.0)
            simulated += np.count_nonzero(
                d4_distance_degrees(rotated) <= tolerance, axis=1)
        at_least += int(np.count_nonzero(simulated >= observed))
        remaining -= count
    return {
        "observed_near_d4_count": int(observed),
        "ruler_count": len(rulers),
        "near_fraction": float(observed / len(rulers)),
        "randomization_upper_tail_p":
            float((1.0 + at_least) / (trials + 1.0)),
        "trials": trials,
    }


# -------------------------------------------------------------- roundness

def crofton4(mask, periodic=False):
    value = np.asarray(mask, bool)
    if not periodic:
        value = np.pad(value, 1, constant_values=False)
    tx = np.count_nonzero(value != np.roll(value, 1, axis=1))
    ty = np.count_nonzero(value != np.roll(value, 1, axis=0))
    td1 = np.count_nonzero(value != np.roll(value, (1, 1), (0, 1)))
    td2 = np.count_nonzero(value != np.roll(value, (1, -1), (0, 1)))
    return float((np.pi / 8.0) * (tx + ty + (td1 + td2) / np.sqrt(2.0)))


def _convex_hull(points):
    unique = sorted(set(map(tuple, np.asarray(points, np.float64))))
    if len(unique) <= 1:
        return np.asarray(unique, np.float64)

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], np.float64)


def _polygon_area(hull):
    if hull.shape[0] < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1))
                           - np.dot(y, np.roll(x, -1))))


def component_roundness(ys, xs, shape):
    """Compactness (Crofton-4) + solidity for one finite component."""
    comp = np.zeros(shape, bool)
    comp[ys, xs] = True
    area = int(ys.size)
    perimeter = crofton4(comp, periodic=False)
    compactness = (None if perimeter <= 0.0
                   else float(4.0 * np.pi * area / perimeter ** 2))
    centers = np.column_stack((xs.astype(np.float64),
                               ys.astype(np.float64)))
    offsets = np.asarray(((-0.5, -0.5), (-0.5, 0.5),
                          (0.5, -0.5), (0.5, 0.5)), np.float64)
    corners = np.unique(
        (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 2),
        axis=0)
    hull_area = _polygon_area(_convex_hull(corners))
    solidity = float(area / hull_area) if hull_area > 0.0 else None
    rounded = bool(compactness is not None and solidity is not None
                   and compactness >= BLOB_COMPACTNESS_MIN
                   and solidity >= BLOB_SOLIDITY_MIN)
    return {"cells": area, "compactness": compactness,
            "solidity": solidity, "rounded": rounded}
