"""Stage: hydrology (M2/W1) — depression filling, flow routing, flow
accumulation, lakes. Reads elevation; does not modify it (the carve is W2).

Depression filling is classical Planchon-Darboux (init +inf, monotone
lowering via directional sweeps; pure numpy, no priority queue). Every
depression fills to its spill; deep ones surface as lakes (threshold is the author control
`lake_min_depth_m`). Endorheic basins are deferred to climate (M3), which
will re-evaluate lake levels causally.
"""

import numpy as np

from .world import World

_EPS = 1e-3          # m per step: drainable micro-slope across filled flats
_INF = 1.0e18

_OFF = ((-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1))
_DIST = (1.0, 1.0, 1.0, 1.0, 1.41421356, 1.41421356, 1.41421356, 1.41421356)


def _ring_mask(shape) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    return m


def _neighbor_min(F: np.ndarray) -> np.ndarray:
    m = np.full(F.shape, _INF)
    m[1:, :] = np.minimum(m[1:, :], F[:-1, :])
    m[:-1, :] = np.minimum(m[:-1, :], F[1:, :])
    m[:, 1:] = np.minimum(m[:, 1:], F[:, :-1])
    m[:, :-1] = np.minimum(m[:, :-1], F[:, 1:])
    m[1:, 1:] = np.minimum(m[1:, 1:], F[:-1, :-1])
    m[1:, :-1] = np.minimum(m[1:, :-1], F[:-1, 1:])
    m[:-1, 1:] = np.minimum(m[:-1, 1:], F[1:, :-1])
    m[:-1, :-1] = np.minimum(m[:-1, :-1], F[1:, 1:])
    return m


def _shift_l(row: np.ndarray) -> np.ndarray:
    return np.concatenate((row[1:], [_INF]))


def _shift_r(row: np.ndarray) -> np.ndarray:
    return np.concatenate(([_INF], row[:-1]))


def _sweep_axis(F: np.ndarray, E: np.ndarray) -> None:
    """One down + one up pass along axis 0 (rows), vectorized across
    columns. Propagates drainage the full map length in a single pass —
    the Planchon-Darboux trick that beats one-cell-per-iteration relax."""
    h = F.shape[0]
    for i in range(1, h):
        prev = F[i - 1]
        cand = np.minimum(prev, np.minimum(_shift_l(prev), _shift_r(prev)))
        F[i] = np.maximum(E[i], np.minimum(F[i], cand + _EPS))
    for i in range(h - 2, -1, -1):
        prev = F[i + 1]
        cand = np.minimum(prev, np.minimum(_shift_l(prev), _shift_r(prev)))
        F[i] = np.maximum(E[i], np.minimum(F[i], cand + _EPS))


def _relax(F: np.ndarray, E: np.ndarray, max_rounds: int, tol: float = 1e-4):
    ring = _ring_mask(F.shape)
    F[ring] = E[ring]
    rounds = 0
    Ft = F.T
    Et = np.ascontiguousarray(E.T)
    for rounds in range(1, max_rounds + 1):
        before = F.copy()
        _sweep_axis(F, E)               # down + up
        _sweep_axis(Ft, Et)             # left + right (transposed view)
        # one 8-neighbor polish for the diagonals the sweeps approximate
        F[:] = np.maximum(E, np.minimum(F, _neighbor_min(F) + _EPS))
        F[ring] = E[ring]
        if float(np.max(before - F)) < tol:
            break
    return F, rounds


def fill_depressions(e: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fill-to-spill surface F >= e, drainable to the border everywhere.

    Classical Planchon-Darboux: start at +inf (border ring = ground, the
    guaranteed-ocean outlet), lower monotonically via directional sweeps.
    The fixed point has no interior sinks by construction; extra sweep
    blocks run until that is actually reached (deterministic: fixed data,
    fixed order)."""
    E = e.astype(np.float64)
    F = np.full(E.shape, _INF)
    stats = {"iters": []}
    F, it = _relax(F, E, max_rounds=16)
    stats["iters"].append(it)
    interior = ~_ring_mask(E.shape)
    for _ in range(4):                     # insurance blocks, usually skipped
        if not np.any((_neighbor_min(F) >= F) & interior):
            break
        F, it = _relax(F, E, max_rounds=8)
        stats["iters"].append(it)
    return F, stats


def flow_directions(F: np.ndarray, cell_km: float):
    """D8 steepest descent on the filled surface. Returns (dir8, recv):
    dir8 int8 in 0..7 (-1 = outlet/sink), recv flat receiver index (-1)."""
    h, w = F.shape
    Fp = np.pad(F, 1, constant_values=_INF)
    best = np.full((h, w), 0.0)
    dir8 = np.full((h, w), -1, dtype=np.int8)
    for k, (dr, dc) in enumerate(_OFF):
        Fn = Fp[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        slope = (F - Fn) / (_DIST[k] * cell_km)
        take = slope > best
        best[take] = slope[take]
        dir8[take] = k
    dir8[_ring_mask(F.shape)] = -1
    rows, cols = np.mgrid[0:h, 0:w]
    recv = np.full((h, w), -1, dtype=np.int64)
    m = dir8 >= 0
    off = np.array(_OFF)
    dr = off[dir8[m], 0]
    dc = off[dir8[m], 1]
    recv[m] = (rows[m] + dr) * w + (cols[m] + dc)
    return dir8, recv


def flow_accumulation(F: np.ndarray, recv: np.ndarray,
                      cell_area_km2: float) -> np.ndarray:
    """Upstream area per cell (km^2), processed in descending fill order.
    Python-list inner loop: benchmarked as the numpy-only route (W1)."""
    hw = F.size
    order = np.argsort(F, axis=None, kind="stable")[::-1]
    acc = [cell_area_km2] * hw
    rl = recv.ravel().tolist()
    for i in order.tolist():
        j = rl[i]
        if j >= 0:
            acc[j] += acc[i]
    return np.asarray(acc, dtype=np.float64).reshape(F.shape)


def label_lakes(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components (4-conn) via union-find over lake cells only."""
    h, w = mask.shape
    parent = np.arange(h * w, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    flat = mask.ravel()
    right = np.flatnonzero(flat[:-1] & flat[1:])
    right = right[(right % w) != w - 1]
    down = np.flatnonzero(flat[:-w] & flat[w:])
    for a_idx, b_off in ((right, 1), (down, w)):
        for x in a_idx.tolist():
            ra, rb = find(x), find(x + b_off)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    ids = np.zeros(h * w, dtype=np.int32)
    lake_cells = np.flatnonzero(flat)
    roots = np.array([find(x) for x in lake_cells.tolist()], dtype=np.int64)
    uniq, inv = np.unique(roots, return_inverse=True)
    ids[lake_cells] = inv + 1
    return ids.reshape(h, w), len(uniq)


def stage_hydrology(world: World) -> None:
    c = world.controls
    e = world["elevation"].astype(np.float64)
    cell = world.cell_km

    F, fstats = fill_depressions(e)
    dir8, recv = flow_directions(F, cell)
    acc = flow_accumulation(F, recv, cell * cell)

    depth = F - e
    lake_mask = depth > float(c["lake_min_depth_m"])
    lake_mask &= e >= 0.0                     # submarine "depressions" are sea
    lake_id, n_lakes = label_lakes(lake_mask)

    world["filled_elevation"] = F.astype(np.float32)
    world["flow_dir"] = dir8
    world["flow_acc"] = acc.astype(np.float32)
    world["lake_id"] = lake_id
    world["lake_level"] = np.where(lake_mask, F, 0.0).astype(np.float32)

    # findings ----------------------------------------------------------
    land = e >= 0.0
    interior_sink = (dir8 == -1) & ~_ring_mask(e.shape)
    n_sink = int((interior_sink & land).sum())
    world.findings.append(
        {"check": "drainage", "level": "warn" if n_sink else "info",
         "unresolved_land_sinks": n_sink,
         "fill_iters": fstats["iters"],
         "msg": None if not n_sink else
         "some land cells found no downhill path after filling"})
    world.findings.append(
        {"check": "lakes", "level": "info", "count": n_lakes,
         "area_fraction": round(float(lake_mask.mean()), 5),
         "fill_volume_km3": round(float((depth[land]).sum())
                                  * cell * cell / 1e3, 1)})
    if land.any():
        world.findings.append(
            {"check": "rivers", "level": "info",
             "max_catchment_km2": round(float(acc[land].max()), 0)})
