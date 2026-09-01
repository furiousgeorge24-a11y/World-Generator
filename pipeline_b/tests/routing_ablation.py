"""Technical routing ablation for the pipeline_b surface-process stage.

This is a development instrument, not an official gallery or evaluator.  It
compares the shipped D8/MFD router with the experimental continuous-direction
router on known analytic surfaces and on the frozen standard seeds 19, 40,
and 101.  It writes nothing; the complete result is printed as JSON after the
human-readable checks.

The instrument deliberately measures two different things:

* ``flow_angle`` is the continuous downhill direction used by the diffuse
  drainage field.
* ``rcv`` is the one-receiver channel spine used by incision, sediment,
  concentrated lake inflow, and the rendered river graph.

Passing only the first is insufficient: a continuous vector hidden behind a
long compass-aligned channel spine would leave the visible/process artifact in
place.

Usage from the repository root::

    python pipeline_b/tests/routing_ablation.py
    python pipeline_b/tests/routing_ablation.py --baseline-only
    python pipeline_b/tests/routing_ablation.py --require-candidate
    python pipeline_b/tests/routing_ablation.py --output result.json

The candidate adapter accepts either a mapping or an object returned by one of
``routing_graph``, ``build_routing_graph``, or ``build_flow_graph`` in
``engine.erosion``.  See ``_candidate_graph`` for the intentionally small
experimental schema.  Public controls are not involved.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import engine.erosion as erosion  # noqa: E402
try:  # isolated internal spike; absence keeps the legacy calibration usable
    import engine.routing_experiment as routing_experiment  # noqa: E402
except ImportError:  # pragma: no cover - exercised before the spike exists
    routing_experiment = None
from engine.elevation import _chamfer_km, coarse_elevation  # noqa: E402
from engine.registry import make_config  # noqa: E402
from engine.tectonics import build_structure  # noqa: E402


PLANE_ANGLES_DEG = np.arange(0.0, 360.0, 5.0)
VALLEY_ANGLES_DEG = (17.0, 31.0, 68.0, 113.0)
REAL_SEEDS = (19, 40, 101)
FULL_PROCESS_ARMS = (
    ("B0", "legacy", "epsilon fill + D8 spine + Freeman MFD + uniform dx"),
    ("B1", "legacy_lengths", "B0 with selected-link physical lengths"),
    ("B2", "d8_flat", "exact fill/rank + local D8 + Freeman MFD"),
    ("C1", "ltd_mfd", "exact fill/rank + D8-LTD + Freeman MFD"),
    ("C2", "ltd_dinf", "exact fill/rank + D8-LTD + D-infinity"),
)
ANGLE_TOL_DEG = 1.0
WEIGHT_TOL = 1e-12
OUTPUT_ROOT = (ROOT / "out" / "routing_ablation").resolve()

_HEAD_CACHE: dict[int, tuple[Any, Any, Any]] = {}
_RUN_CACHE: dict[tuple[int, str], tuple[dict[str, Any], float]] = {}


@dataclass
class RoutingGraph:
    """Normalized view of either routing implementation."""

    name: str
    shape: tuple[int, int]
    filled_level: np.ndarray
    flat_mask: np.ndarray
    flat_rank: np.ndarray | None
    targets: np.ndarray
    weights: np.ndarray
    flow_angle: np.ndarray
    rcv: np.ndarray
    edge_len: np.ndarray
    batches: list[np.ndarray]
    main_donor: np.ndarray | None = None
    transverse_error: np.ndarray | None = None

    @property
    def n(self) -> int:
        return self.shape[0] * self.shape[1]


class Checks:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        (self.passed if ok else self.failed).append(name)
        mark = "ok" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))


def _get(raw: Any, names: tuple[str, ...], default: Any = None,
         required: bool = False) -> Any:
    for name in names:
        if isinstance(raw, dict) and name in raw:
            return raw[name]
        if hasattr(raw, name):
            return getattr(raw, name)
    if required:
        raise KeyError(f"candidate result lacks any of {names}")
    return default


def _as_batches(value: Any) -> list[np.ndarray]:
    return [np.asarray(b, dtype=np.int64).ravel() for b in value]


def _edge_geometry(rcv: np.ndarray, shape: tuple[int, int],
                   dx: float) -> tuple[np.ndarray, np.ndarray]:
    n = rcv.size
    src = np.arange(n, dtype=np.int64)
    sy, sx = np.divmod(src, shape[1])
    ry, rx = np.divmod(rcv, shape[1])
    ddx = rx - sx
    ddy = ry - sy
    terminal = rcv == src
    angle = np.arctan2(ddy, ddx).astype(np.float64)
    angle[terminal] = np.nan
    length = np.hypot(ddx, ddy).astype(np.float64) * float(dx)
    length[terminal] = 0.0
    return angle, length


def _weighted_angle(targets: np.ndarray, weights: np.ndarray,
                    shape: tuple[int, int]) -> np.ndarray:
    n = shape[0] * shape[1]
    src = np.arange(n, dtype=np.int64)
    sy, sx = np.divmod(src, shape[1])
    vx = np.zeros(n, dtype=np.float64)
    vy = np.zeros(n, dtype=np.float64)
    for k in range(targets.shape[0]):
        t = targets[k]
        w = weights[k]
        valid = (w > 0.0) & (t >= 0) & (t < n)
        ty, tx = np.divmod(np.where(valid, t, 0), shape[1])
        length = np.hypot(tx - sx, ty - sy)
        scale = np.divide(w, length, out=np.zeros_like(w), where=valid)
        vx += scale * (tx - sx)
        vy += scale * (ty - sy)
    angle = np.arctan2(vy, vx)
    angle[np.hypot(vx, vy) <= 0.0] = np.nan
    return angle


def _legacy_graph(surface: np.ndarray, dx: float = 1.0) -> RoutingGraph:
    filled = erosion.fill_depressions(np.asarray(surface, np.float64))
    rcv, targets, weights, flat = erosion.receivers(filled)
    batches = erosion.topo_batches(rcv, targets, weights, flat)

    # Normalize the current MFD fallback into the same explicit weighted-edge
    # representation expected of the candidate.
    targets = targets.copy()
    weights = weights.copy()
    wsum = weights.sum(axis=0)
    fallback = (wsum <= 0.0) & (rcv != np.arange(rcv.size))
    targets[0, fallback] = rcv[fallback]
    weights[0, fallback] = 1.0

    _, edge_len = _edge_geometry(rcv, surface.shape, dx)
    flow_angle = _weighted_angle(targets, weights, surface.shape)
    return RoutingGraph(
        name="legacy_d8_mfd",
        shape=surface.shape,
        filled_level=filled,
        flat_mask=np.asarray(flat, bool).reshape(surface.shape),
        flat_rank=None,
        targets=targets,
        weights=weights,
        flow_angle=flow_angle,
        rcv=np.asarray(rcv, np.int64),
        edge_len=edge_len,
        batches=_as_batches(batches),
    )


def _candidate_callable() -> Callable[..., Any] | None:
    modules = (() if routing_experiment is None
               else (routing_experiment,)) + (erosion,)
    for module in modules:
        for name in ("routing_graph", "build_routing_graph",
                     "build_flow_graph"):
            fn = getattr(module, name, None)
            if callable(fn):
                return fn
    return None


def _call_candidate(fn: Callable[..., Any], surface: np.ndarray,
                    dx: float) -> Any:
    params = inspect.signature(fn).parameters
    kwargs: dict[str, Any] = {}
    if "dx_km" in params:
        kwargs["dx_km"] = float(dx)
    elif "dx" in params:
        kwargs["dx"] = float(dx)
    if "mode" in params:
        kwargs["mode"] = "dinf_ltd"
    return fn(np.asarray(surface, np.float64), **kwargs)


def _candidate_graph(surface: np.ndarray, dx: float = 1.0) \
        -> RoutingGraph | None:
    fn = _candidate_callable()
    if fn is None:
        return None
    raw = _call_candidate(fn, surface, dx)
    shape = surface.shape
    n = surface.size

    rcv = np.asarray(_get(raw, ("rcv", "rcv_channel"), required=True),
                     np.int64).reshape(n)
    targets = np.asarray(_get(
        raw, ("targets", "targets_diffuse"), required=True), np.int64)
    weights = np.asarray(_get(
        raw, ("weights", "weights_diffuse"), required=True), np.float64)
    if targets.ndim != 2 or weights.shape != targets.shape:
        raise ValueError("candidate targets/weights must both have shape K x n")
    if targets.shape[1] != n and targets.shape[0] == n:
        targets = targets.T
        weights = weights.T
    if targets.shape[1] != n:
        raise ValueError("candidate targets/weights do not match surface")

    filled = np.asarray(_get(
        raw, ("filled_level", "fill_level", "filled"), required=True),
        np.float64).reshape(shape)
    flat = np.asarray(_get(
        raw, ("flat_mask", "flat"), np.zeros(shape, bool)), bool).reshape(shape)
    rank_raw = _get(raw, ("flat_rank", "rank"))
    rank = (None if rank_raw is None else
            np.asarray(rank_raw, np.int64).reshape(n))

    angle_raw = _get(raw, ("flow_angle", "angle", "direction"))
    flow_angle = (_weighted_angle(targets, weights, shape)
                  if angle_raw is None else
                  np.asarray(angle_raw, np.float64).reshape(n))
    edge_raw = _get(raw, ("edge_len", "edge_length", "edge_length_km"))
    _, measured_edge = _edge_geometry(rcv, shape, dx)
    edge_len = (measured_edge if edge_raw is None else
                np.asarray(edge_raw, np.float64).reshape(n))

    batches_raw = _get(raw, ("batches", "topo_batches", "topological_batches"))
    if batches_raw is None:
        batches = _topo_batches_union(rcv, targets, weights)
    else:
        batches = _as_batches(batches_raw)

    donor_raw = _get(raw, ("main_donor", "dominant_donor"))
    deviation_raw = _get(
        raw, ("cum_transverse_error", "transverse_error", "deviation"))
    return RoutingGraph(
        name="candidate_dinf_ltd",
        shape=shape,
        filled_level=filled,
        flat_mask=flat,
        flat_rank=rank,
        targets=targets,
        weights=weights,
        flow_angle=flow_angle,
        rcv=rcv,
        edge_len=edge_len,
        batches=batches,
        main_donor=(None if donor_raw is None else
                    np.asarray(donor_raw, np.int64).reshape(n)),
        transverse_error=(None if deviation_raw is None else
                          np.asarray(deviation_raw, np.float64).reshape(n)),
    )


def _topo_batches_union(rcv: np.ndarray, targets: np.ndarray,
                        weights: np.ndarray) -> list[np.ndarray]:
    """Independent Kahn order over diffuse edges plus the channel spine."""
    n = rcv.size
    indeg = np.zeros(n, np.int64)
    for k in range(targets.shape[0]):
        valid = (weights[k] > 0.0) & (targets[k] >= 0) & (targets[k] < n)
        np.add.at(indeg, targets[k, valid], 1)
    for i, r in enumerate(rcv):
        if r == i:
            continue
        represented = bool(np.any((targets[:, i] == r) &
                                  (weights[:, i] > 0.0)))
        if not represented:
            indeg[r] += 1

    frontier = np.flatnonzero(indeg == 0)
    batches: list[np.ndarray] = []
    seen = 0
    while frontier.size:
        batches.append(frontier)
        seen += frontier.size
        outs: list[np.ndarray] = []
        for k in range(targets.shape[0]):
            valid = ((weights[k, frontier] > 0.0)
                     & (targets[k, frontier] >= 0)
                     & (targets[k, frontier] < n))
            outs.append(targets[k, frontier[valid]])
        extra = []
        for i in frontier:
            r = int(rcv[i])
            if r == i:
                continue
            represented = bool(np.any((targets[:, i] == r) &
                                      (weights[:, i] > 0.0)))
            if not represented:
                extra.append(r)
        if extra:
            outs.append(np.asarray(extra, np.int64))
        tgt = np.concatenate(outs) if outs else np.empty(0, np.int64)
        if tgt.size:
            np.subtract.at(indeg, tgt, 1)
            cand = np.unique(tgt)
            frontier = cand[indeg[cand] == 0]
        else:
            frontier = np.empty(0, np.int64)
    if seen != n:
        raise AssertionError(f"routing graph is cyclic: covered {seen}/{n}")
    return batches


def _accumulate_channel(g: RoutingGraph, runoff: np.ndarray | None = None) \
        -> np.ndarray:
    q = (np.ones(g.n, np.float64) if runoff is None else
         np.asarray(runoff, np.float64).reshape(g.n).copy())
    for batch in g.batches:
        r = g.rcv[batch]
        move = r != batch
        np.add.at(q, r[move], q[batch[move]])
    return q


def _accumulate_diffuse(g: RoutingGraph, runoff: np.ndarray | None = None) \
        -> np.ndarray:
    q = (np.ones(g.n, np.float64) if runoff is None else
         np.asarray(runoff, np.float64).reshape(g.n).copy())
    for batch in g.batches:
        for k in range(g.targets.shape[0]):
            t = g.targets[k, batch]
            w = g.weights[k, batch]
            move = (w > 0.0) & (t >= 0) & (t < g.n)
            np.add.at(q, t[move], q[batch[move]] * w[move])
    return q


def _wrap_error_deg(got: np.ndarray, expected: np.ndarray | float) \
        -> np.ndarray:
    gd = np.rad2deg(got)
    return np.abs((gd - expected + 180.0) % 360.0 - 180.0)


def _harmonic(angle: np.ndarray, order: int,
              weights: np.ndarray | None = None) -> float:
    valid = np.isfinite(angle)
    if weights is not None:
        valid &= np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return float("nan")
    z = np.exp(1j * order * angle[valid])
    if weights is None:
        return float(np.abs(z.mean()))
    return float(np.abs(np.average(z, weights=weights[valid])))


def _plane_suite(builder: Callable[[np.ndarray, float], RoutingGraph]) \
        -> dict[str, float]:
    size = 41
    y, x = np.mgrid[0:size, 0:size]
    all_error: list[np.ndarray] = []
    all_direction: list[np.ndarray] = []
    all_spine_error: list[np.ndarray] = []
    all_spine_direction: list[np.ndarray] = []
    spine_cross_track: list[float] = []
    spine_endpoint_cross_track: list[float] = []
    spine_window_error: list[float] = []
    spine_turn_fraction: list[float] = []
    for degrees in PLANE_ANGLES_DEG:
        theta = np.deg2rad(degrees)
        surface = -(x * np.cos(theta) + y * np.sin(theta))
        graph = builder(surface, 1.0)
        core = np.zeros(surface.shape, bool)
        core[3:-3, 3:-3] = True
        idx = np.flatnonzero(core.ravel() & np.isfinite(graph.flow_angle))
        err = _wrap_error_deg(graph.flow_angle[idx], degrees)
        all_error.append(err)
        all_direction.append(graph.flow_angle[idx])
        spine_angle, _ = _edge_geometry(graph.rcv, graph.shape, 1.0)
        spine_idx = np.flatnonzero(core.ravel() & np.isfinite(spine_angle))
        all_spine_error.append(_wrap_error_deg(
            spine_angle[spine_idx], degrees))
        all_spine_direction.append(spine_angle[spine_idx])
        start = (size // 2) * size + size // 2
        path = _follow_path(graph, start)
        py, px = np.divmod(path, size)
        transverse = (-np.sin(theta) * (px - px[0])
                      + np.cos(theta) * (py - py[0]))
        spine_cross_track.append(float(np.max(np.abs(transverse))))
        spine_endpoint_cross_track.append(float(abs(transverse[-1])))
        window = min(8, path.size - 1)
        if window > 0:
            for j in range(path.size - window):
                vx = px[j + window] - px[j]
                vy = py[j + window] - py[j]
                bearing = np.arctan2(vy, vx)
                spine_window_error.append(float(
                    _wrap_error_deg(np.asarray([bearing]), degrees)[0]))
        if path.size > 2:
            ex = np.diff(px)
            ey = np.diff(py)
            codes = (ey + 1) * 3 + (ex + 1)
            spine_turn_fraction.append(float((codes[1:] != codes[:-1]).mean()))
    error = np.concatenate(all_error)
    direction = np.concatenate(all_direction)
    spine_error = np.concatenate(all_spine_error)
    spine_direction = np.concatenate(all_spine_direction)
    return {
        "continuous_mean_error_deg": float(error.mean()),
        "continuous_p95_error_deg": float(np.percentile(error, 95)),
        "continuous_max_error_deg": float(error.max()),
        "continuous_eightfold_harmonic": _harmonic(direction, 8),
        "spine_mean_error_deg": float(spine_error.mean()),
        "spine_p95_error_deg": float(np.percentile(spine_error, 95)),
        "spine_max_error_deg": float(spine_error.max()),
        "spine_eightfold_harmonic": _harmonic(spine_direction, 8),
        "spine_p95_max_cross_track_cells": float(
            np.percentile(spine_cross_track, 95)),
        "spine_max_cross_track_cells": float(np.max(spine_cross_track)),
        "spine_p95_endpoint_cross_track_cells": float(
            np.percentile(spine_endpoint_cross_track, 95)),
        "spine_window8_p95_bearing_error_deg": float(
            np.percentile(spine_window_error, 95)),
        "spine_mean_turn_fraction": float(np.mean(spine_turn_fraction)),
    }


def _follow_path(g: RoutingGraph, start: int) -> np.ndarray:
    out = []
    seen: set[int] = set()
    cur = int(start)
    for _ in range(g.n + 1):
        if cur in seen:
            raise AssertionError(f"cycle while following channel from {start}")
        seen.add(cur)
        out.append(cur)
        nxt = int(g.rcv[cur])
        if nxt == cur:
            break
        cur = nxt
    else:
        raise AssertionError(f"unterminated channel from {start}")
    return np.asarray(out, np.int64)


def _valley_suite(builder: Callable[[np.ndarray, float], RoutingGraph]) \
        -> dict[str, float]:
    size = 81
    yy, xx = np.mgrid[0:size, 0:size]
    x = xx - 0.5 * (size - 1)
    y = yy - 0.5 * (size - 1)
    direction_errors = []
    thalweg_errors = []
    path_rms = []
    path_max = []
    for degrees in VALLEY_ANGLES_DEG:
        theta = np.deg2rad(degrees)
        cs, sn = np.cos(theta), np.sin(theta)
        along = cs * x + sn * y
        cross = -sn * x + cs * y
        surface = -0.35 * along + 0.035 * cross ** 2
        graph = builder(surface, 1.0)

        # Analytic negative gradient of the rotated parabolic trough.
        vx = 0.35 * cs + 0.07 * cross * sn
        vy = 0.35 * sn - 0.07 * cross * cs
        expected = np.rad2deg(np.arctan2(vy, vx))
        core = np.zeros(surface.shape, bool)
        core[4:-4, 4:-4] = True
        use = core.ravel() & np.isfinite(graph.flow_angle)
        direction_errors.append(_wrap_error_deg(
            graph.flow_angle[use], expected.ravel()[use]))
        near = use & (np.abs(cross).ravel() <= 0.55)
        thalweg_errors.append(_wrap_error_deg(
            graph.flow_angle[near], expected.ravel()[near]))

        target_along = -0.32 * size
        score = np.abs(along - target_along) + 4.0 * np.abs(cross)
        start = int(np.argmin(score))
        path = _follow_path(graph, start)
        py, px = np.divmod(path, size)
        pcross = -sn * (px - 0.5 * (size - 1)) \
            + cs * (py - 0.5 * (size - 1))
        # Ignore the final four boundary-adjacent cells where the bounded
        # domain, not the analytic infinite trough, selects an outlet.
        interior = pcross[:-4] if pcross.size > 8 else pcross
        path_rms.append(float(np.sqrt(np.mean(interior ** 2))))
        path_max.append(float(np.max(np.abs(interior))))

    err = np.concatenate(direction_errors)
    thalweg = np.concatenate(thalweg_errors)
    return {
        "mean_direction_error_deg": float(err.mean()),
        "p95_direction_error_deg": float(np.percentile(err, 95)),
        "thalweg_mean_error_deg": float(thalweg.mean()),
        "thalweg_max_error_deg": float(thalweg.max()),
        "mean_channel_cross_track_cells": float(np.mean(path_rms)),
        "max_channel_cross_track_cells": float(np.max(path_max)),
    }


def _cone_suite(builder: Callable[[np.ndarray, float], RoutingGraph]) \
        -> dict[str, float]:
    size = 65
    yy, xx = np.mgrid[0:size, 0:size]
    x = xx - 0.5 * (size - 1)
    y = yy - 0.5 * (size - 1)
    radius = np.hypot(x, y)
    surface = -radius
    graph = builder(surface, 1.0)
    annulus = (radius >= 7.0) & (radius <= 25.0)
    use = annulus.ravel() & np.isfinite(graph.flow_angle)
    expected = np.rad2deg(np.arctan2(y, x)).ravel()[use]
    error = _wrap_error_deg(graph.flow_angle[use], expected)
    spine_angle, _ = _edge_geometry(graph.rcv, graph.shape, 1.0)
    spine_use = annulus.ravel() & np.isfinite(spine_angle)
    spine_expected = np.rad2deg(np.arctan2(y, x)).ravel()[spine_use]
    spine_error = _wrap_error_deg(spine_angle[spine_use], spine_expected)
    return {
        "continuous_mean_radial_error_deg": float(error.mean()),
        "continuous_p95_radial_error_deg": float(
            np.percentile(error, 95)),
        "continuous_eightfold_harmonic": _harmonic(
            graph.flow_angle[use], 8),
        "continuous_fourfold_harmonic": _harmonic(
            graph.flow_angle[use], 4),
        "spine_mean_radial_error_deg": float(spine_error.mean()),
        "spine_p95_radial_error_deg": float(
            np.percentile(spine_error, 95)),
        "spine_eightfold_harmonic": _harmonic(spine_angle[spine_use], 8),
    }


def _single_spill_surface(size: int = 65) \
        -> tuple[np.ndarray, np.ndarray, int]:
    """Closed bowl behind a 40 m spill and a unique outlet corridor."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = size // 2
    radius = np.hypot(xx - c, yy - c)
    surface = np.full((size, size), 100.0)
    basin = radius <= 14.0
    surface[basin] = 5.0 + 0.025 * radius[basin] ** 2

    # A one-cell outlet descends from the 40 m notch to the top boundary.
    notch_y = c - 15
    surface[notch_y, c] = 40.0
    surface[:notch_y + 1, c] = np.linspace(0.0, 40.0, notch_y + 1)
    outlet = c  # flat index (row 0, col c)
    return surface, basin, outlet


def _single_spill_suite(builder: Callable[[np.ndarray, float], RoutingGraph]) \
        -> dict[str, float | int | bool]:
    surface, basin, expected_outlet = _single_spill_surface()
    graph = builder(surface, 1.0)
    filled = graph.filled_level[basin]
    runoff = basin.astype(np.float64).ravel()
    q_channel = _accumulate_channel(graph, runoff)
    q_diffuse = _accumulate_diffuse(graph, runoff)
    terminal = graph.rcv == np.arange(graph.n)
    channel_terminals = np.flatnonzero(terminal & (q_channel > 1e-10))
    diffuse_terminals = np.flatnonzero(terminal & (q_diffuse > 1e-10))
    return {
        "basin_fill_mean": float(filled.mean()),
        "basin_fill_range": float(np.ptp(filled)),
        "basin_cells": int(basin.sum()),
        "channel_terminal_count": int(channel_terminals.size),
        "diffuse_terminal_count": int(diffuse_terminals.size),
        "channel_expected_outlet": bool(
            channel_terminals.size == 1 and
            int(channel_terminals[0]) == expected_outlet),
        "diffuse_expected_outlet": bool(
            diffuse_terminals.size == 1 and
            int(diffuse_terminals[0]) == expected_outlet),
        "channel_runoff_closure": float(
            q_channel[channel_terminals].sum() / max(runoff.sum(), 1.0)),
        "diffuse_runoff_closure": float(
            q_diffuse[diffuse_terminals].sum() / max(runoff.sum(), 1.0)),
        "flat_cells": int(graph.flat_mask.sum()),
    }


def _validate_graph(g: RoutingGraph, surface: np.ndarray) \
        -> dict[str, float | int | bool]:
    n = g.n
    src = np.arange(n)
    terminal = g.rcv == src
    valid_rcv = (g.rcv >= 0) & (g.rcv < n)
    wsum = g.weights.sum(axis=0)
    valid_slot = (g.targets >= 0) & (g.targets < n)
    bad_weight_slot = (g.weights > WEIGHT_TOL) & ~valid_slot
    nonnegative = bool(np.isfinite(g.weights).all()
                       and (g.weights >= -WEIGHT_TOL).all())

    represented = np.zeros(n, bool)
    for k in range(g.targets.shape[0]):
        represented |= ((g.targets[k] == g.rcv) &
                        (g.weights[k] > WEIGHT_TOL))
    off_flat = ~g.flat_mask.ravel() & ~terminal

    # Rebuild the union order independently. This is both the cycle check and
    # protection against a candidate-provided order omitting cells.
    independent = _topo_batches_union(g.rcv, g.targets, g.weights)
    covered = np.concatenate(independent) if independent else np.empty(0, int)

    q_channel = _accumulate_channel(g)
    q_diffuse = _accumulate_diffuse(g)
    channel_closure = float(q_channel[terminal].sum() / n)
    diffuse_closure = float(q_diffuse[terminal].sum() / n)

    _, measured = _edge_geometry(g.rcv, g.shape, 1.0)
    edge_scale = np.divide(g.edge_len, measured,
                           out=np.ones_like(g.edge_len), where=measured > 0)
    positive_edge = bool((g.edge_len[~terminal] > 0.0).all()
                         and (g.edge_len[terminal] == 0.0).all())

    level = g.filled_level.ravel()
    if g.flat_rank is None:
        rank = np.zeros(n, np.int64)
    else:
        rank = np.asarray(g.flat_rank, np.int64).reshape(n)

    def descends(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        lower = level[target] < level[source]
        ranked = (level[target] == level[source]) \
            & (rank[target] < rank[source])
        return lower | ranked

    moving = ~terminal
    channel_descent = bool(descends(src[moving], g.rcv[moving]).all())
    diffuse_descent = True
    for k in range(g.targets.shape[0]):
        active = g.weights[k] > WEIGHT_TOL
        if active.any() and not descends(src[active],
                                         g.targets[k, active]).all():
            diffuse_descent = False
            break
    return {
        "cells": n,
        "edge_slots": int(g.targets.shape[0]),
        "all_cells_topologically_covered": bool(
            covered.size == n and np.unique(covered).size == n),
        "receivers_in_range": bool(valid_rcv.all()),
        "weights_finite_nonnegative": nonnegative,
        "positive_weight_has_valid_target": bool(not bad_weight_slot.any()),
        "weight_sums_valid": bool(
            np.allclose(wsum[~terminal], 1.0, atol=WEIGHT_TOL, rtol=0.0)
            and np.all(np.abs(wsum[terminal]) <= WEIGHT_TOL)),
        "off_flat_channel_receiver_is_diffuse_bracket": bool(
            represented[off_flat].all()),
        "channel_runoff_closure": channel_closure,
        "diffuse_runoff_closure": diffuse_closure,
        "edge_lengths_positive": positive_edge,
        "edge_lengths_match_receiver_geometry": bool(
            np.allclose(g.edge_len, measured, rtol=0.0, atol=1e-12)),
        "edge_length_scale_min": float(edge_scale[~terminal].min())
        if (~terminal).any() else 1.0,
        "edge_length_scale_max": float(edge_scale[~terminal].max())
        if (~terminal).any() else 1.0,
        "filled_never_below_surface": bool(
            np.all(g.filled_level >= surface - 1e-12)),
        "channel_strict_composite_descent": channel_descent,
        "diffuse_strict_composite_descent": diffuse_descent,
    }


def _same_bearing_runs(g: RoutingGraph, selected: np.ndarray) \
        -> tuple[int, float]:
    """Longest selected channel run with one unchanged grid bearing."""
    sy, sx = np.divmod(np.arange(g.n), g.shape[1])
    ry, rx = np.divmod(g.rcv, g.shape[1])
    dx = rx - sx
    dy = ry - sy
    code = (dy + 1) * 3 + (dx + 1)
    run_edges = np.zeros(g.n, np.int64)
    run_length = np.zeros(g.n, np.float64)
    for batch in g.batches:
        for cell in batch:
            i = int(cell)
            r = int(g.rcv[i])
            if r == i or not selected[i] or not selected[r]:
                continue
            if code[i] == code[r]:
                cand_edges = run_edges[i] + 1
                cand_length = run_length[i] + g.edge_len[i]
            else:
                cand_edges = 1
                cand_length = g.edge_len[i]
            if cand_edges > run_edges[r] or (
                    cand_edges == run_edges[r]
                    and cand_length > run_length[r]):
                run_edges[r] = cand_edges
                run_length[r] = cand_length
    return int(run_edges.max(initial=0)), float(run_length.max(initial=0.0))


def _path_inflation(g: RoutingGraph, selected: np.ndarray) \
        -> tuple[float, float, int]:
    donors = np.zeros(g.n, np.int64)
    moving = selected & (g.rcv != np.arange(g.n))
    np.add.at(donors, g.rcv[moving], 1)
    heads = np.flatnonzero(moving & (donors == 0))
    ratios = []
    positive_lengths = g.edge_len[g.edge_len > 0.0]
    cell_length = (float(positive_lengths.min())
                   if positive_lengths.size else 1.0)
    for head in heads:
        path = [int(head)]
        cur = int(head)
        length = 0.0
        seen: set[int] = set()
        while cur not in seen and selected[cur]:
            seen.add(cur)
            nxt = int(g.rcv[cur])
            if nxt == cur:
                break
            length += float(g.edge_len[cur])
            path.append(nxt)
            cur = nxt
        if len(path) < 4:
            continue
        y0, x0 = divmod(path[0], g.shape[1])
        y1, x1 = divmod(path[-1], g.shape[1])
        direct = np.hypot(x1 - x0, y1 - y0) * cell_length
        if direct > 0:
            ratios.append(length / direct)
    if not ratios:
        return float("nan"), float("nan"), 0
    arr = np.asarray(ratios)
    return float(np.median(arr)), float(np.percentile(arr, 95)), len(ratios)


def _head(seed: int) -> tuple[Any, Any, Any]:
    if seed not in _HEAD_CACHE:
        cfg = make_config({})
        structure = build_structure(seed, cfg)
        coarse = coarse_elevation(structure, cfg, seed)
        _HEAD_CACHE[seed] = (cfg, structure, coarse)
    return _HEAD_CACHE[seed]


def _run_process(seed: int, mode: str) -> tuple[dict[str, Any], float]:
    key = (int(seed), str(mode))
    if key not in _RUN_CACHE:
        cfg, structure, coarse = _head(seed)
        t0 = time.perf_counter()
        result = erosion.run_erosion(
            structure, coarse, cfg, seed, _routing_mode=mode)
        _RUN_CACHE[key] = (result, time.perf_counter() - t0)
    return _RUN_CACHE[key]


def _real_seed_metrics(seed: int,
                       builder: Callable[[np.ndarray, float], RoutingGraph]) \
        -> dict[str, float | int | bool]:
    eroded, _ = _run_process(seed, "legacy")
    graph = builder(eroded["z"], float(eroded["e_km"]))
    runoff = np.exp(-_chamfer_km(eroded["z0"] < 0.0, eroded["e_km"])
                    / erosion.L_MOIST_KM)
    q = _accumulate_channel(graph, runoff)
    selected = ((q > 30.0) & (eroded["z"].ravel() >= 0.0)
                & (graph.rcv != np.arange(graph.n)))
    src = np.flatnonzero(selected)
    sy, sx = np.divmod(src, graph.shape[1])
    ry, rx = np.divmod(graph.rcv[src], graph.shape[1])
    dx = rx - sx
    dy = ry - sy
    edge_angle = np.arctan2(dy, dx)
    cardinal = (dx == 0) | (dy == 0)
    diagonal = (np.abs(dx) == 1) & (np.abs(dy) == 1)
    longest_edges, longest_km = _same_bearing_runs(graph, selected)
    inflation_med, inflation_p95, paths = _path_inflation(graph, selected)
    return {
        "process_cells": graph.n,
        "selected_channel_edges": int(src.size),
        "cardinal_edge_fraction": float(cardinal.mean()) if src.size else 0.0,
        "diagonal_edge_fraction": float(diagonal.mean()) if src.size else 0.0,
        "edge_fourfold_harmonic": _harmonic(edge_angle, 4, q[src]),
        "edge_eightfold_harmonic": _harmonic(edge_angle, 8, q[src]),
        "continuous_eightfold_harmonic": _harmonic(
            graph.flow_angle[src], 8, q[src]),
        "longest_same_bearing_edges": longest_edges,
        "longest_same_bearing_km": longest_km,
        "path_inflation_median": inflation_med,
        "path_inflation_p95": inflation_p95,
        "path_count": paths,
        "channel_runoff_closure": float(
            q[graph.rcv == np.arange(graph.n)].sum() / runoff.sum()),
    }


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for start in np.flatnonzero(mask.ravel()):
        y0, x0 = divmod(int(start), width)
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack = [(y0, x0)]
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append(y * width + x)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                yy, xx = y + dy, x + dx
                if (0 <= yy < height and 0 <= xx < width
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        components.append(np.asarray(cells, np.int64))
    return components


def _river_output_metrics(eroded: dict[str, Any]) -> dict[str, Any]:
    edges = eroded["river_edges"]
    count = int(edges["a8"].size)
    if count == 0:
        return {
            "river_edges": 0,
            "river_cardinal_fraction": 0.0,
            "river_diagonal_fraction": 0.0,
            "river_fourfold_harmonic": 0.0,
            "river_eightfold_harmonic": 0.0,
            "river_longest_same_bearing_edges": 0,
            "river_longest_same_bearing_km": 0.0,
            "river_turn_fraction": 0.0,
            "river_aba_alternation_fraction": 0.0,
            "river_window8_eightfold_harmonic": 0.0,
            "river_path_inflation_median": 0.0,
            "river_path_inflation_p95": 0.0,
        }

    n_e = int(eroded["n_e"])
    cell_km = float(eroded["e_km"])
    x0 = np.rint(edges["x0"] / cell_km - 0.5).astype(np.int64)
    y0 = np.rint(edges["y0"] / cell_km - 0.5).astype(np.int64)
    x1 = np.rint(edges["x1"] / cell_km - 0.5).astype(np.int64)
    y1 = np.rint(edges["y1"] / cell_km - 0.5).astype(np.int64)
    src = y0 * n_e + x0
    dst = y1 * n_e + x1
    dx = x1 - x0
    dy = y1 - y0
    bearing = np.arctan2(dy, dx)
    code = (dy + 1) * 3 + (dx + 1)
    length = cell_km * np.hypot(dx, dy)
    cardinal = (dx == 0) | (dy == 0)
    diagonal = (np.abs(dx) == 1) & (np.abs(dy) == 1)

    edge_at = np.full(n_e * n_e, -1, np.int64)
    edge_at[src] = np.arange(count)
    downstream_edge = edge_at[dst]
    connected = downstream_edge >= 0
    turns = np.zeros(count, bool)
    turns[connected] = code[connected] != code[downstream_edge[connected]]
    two_down = np.full(count, -1, np.int64)
    two_down[connected] = downstream_edge[connected]
    twice_connected = connected & (two_down >= 0)
    second = np.full(count, -1, np.int64)
    second[twice_connected] = downstream_edge[two_down[twice_connected]]
    aba_ok = second >= 0
    aba = np.zeros(count, bool)
    aba[aba_ok] = ((code[aba_ok] == code[second[aba_ok]])
                   & (code[aba_ok] != code[two_down[aba_ok]]))

    donor_count = np.zeros(count, np.int64)
    if connected.any():
        np.add.at(donor_count, downstream_edge[connected], 1)
    heads = np.flatnonzero(donor_count == 0)
    longest_edges = 0
    longest_km = 0.0
    path_inflation = []
    window_bearings = []
    for head in heads:
        path_edges = []
        seen: set[int] = set()
        cur = int(head)
        while cur >= 0 and cur not in seen:
            seen.add(cur)
            path_edges.append(cur)
            cur = int(downstream_edge[cur])
        if not path_edges:
            continue
        run_edges = 1
        run_km = float(length[path_edges[0]])
        best_edges = run_edges
        best_km = run_km
        for previous, current in zip(path_edges[:-1], path_edges[1:]):
            if code[current] == code[previous]:
                run_edges += 1
                run_km += float(length[current])
            else:
                run_edges = 1
                run_km = float(length[current])
            best_edges = max(best_edges, run_edges)
            best_km = max(best_km, run_km)
        longest_edges = max(longest_edges, best_edges)
        longest_km = max(longest_km, best_km)

        first = path_edges[0]
        last = path_edges[-1]
        direct = cell_km * np.hypot(x1[last] - x0[first],
                                    y1[last] - y0[first])
        traveled = float(length[path_edges].sum())
        if direct > 0.0 and len(path_edges) >= 3:
            path_inflation.append(traveled / direct)

        if len(path_edges) >= 8:
            for j in range(len(path_edges) - 7):
                first = path_edges[j]
                last = path_edges[j + 7]
                wx = x1[last] - x0[first]
                wy = y1[last] - y0[first]
                if wx != 0 or wy != 0:
                    window_bearings.append(np.arctan2(wy, wx))

    inflation = np.asarray(path_inflation, np.float64)
    windows = np.asarray(window_bearings, np.float64)
    return {
        "river_edges": count,
        "river_cardinal_fraction": float(cardinal.mean()),
        "river_diagonal_fraction": float(diagonal.mean()),
        "river_fourfold_harmonic": _harmonic(
            bearing, 4, np.asarray(edges["a8"], np.float64)),
        "river_eightfold_harmonic": _harmonic(
            bearing, 8, np.asarray(edges["a8"], np.float64)),
        "river_longest_same_bearing_edges": int(longest_edges),
        "river_longest_same_bearing_km": float(longest_km),
        "river_turn_fraction": float(turns[connected].mean())
        if connected.any() else 0.0,
        "river_aba_alternation_fraction": float(aba[aba_ok].mean())
        if aba_ok.any() else 0.0,
        "river_window8_eightfold_harmonic": _harmonic(windows, 8)
        if windows.size else 0.0,
        "river_path_inflation_median": float(np.median(inflation))
        if inflation.size else 0.0,
        "river_path_inflation_p95": float(np.percentile(inflation, 95))
        if inflation.size else 0.0,
    }


def _process_metrics(eroded: dict[str, Any], wall_s: float,
                     baseline_z: np.ndarray | None = None) -> dict[str, Any]:
    z = np.asarray(eroded["z"], np.float64)
    z0 = np.asarray(eroded["z0"], np.float64)
    e_km = float(eroded["e_km"])
    cell_area_m2 = (e_km * 1000.0) ** 2
    source_m3 = float(np.asarray(eroded["ero"]).sum()) * cell_area_m2
    deposited_m3 = float(np.asarray(eroded["sed"]).sum()) * cell_area_m2
    export_m3 = float(eroded["sediment_export_m3"])
    residual_m3 = float(eroded["sediment_terminal_residual_m3"])
    accounted = deposited_m3 + export_m3 + residual_m3
    closure = accounted / source_m3 if source_m3 > 0.0 else 1.0

    gy, gx = np.gradient(z, e_km)
    slope = np.hypot(gx, gy)
    land = z >= 0.0
    use = land & (slope > 1e-12)
    slope_angle = np.arctan2(gy[use], gx[use])
    slope_weight = slope[use]

    lake_mask = np.asarray(eroded["lake_depth"]) > 0.0
    lake_components = _components4(lake_mask)
    lake_level = np.asarray(eroded["lake_surf"]).ravel()
    lake_ranges = [float(np.ptp(lake_level[c])) for c in lake_components]
    largest_lake = max((c.size for c in lake_components), default=0)

    metrics: dict[str, Any] = {
        "wall_s": float(wall_s),
        "erosion_total_s": float(eroded["timings"]["erosion_total"]),
        "terrain_land_fraction": float(land.mean()),
        "terrain_min_m": float(z.min()),
        "terrain_max_m": float(z.max()),
        "terrain_rms_change_m": float(np.sqrt(np.mean((z - z0) ** 2))),
        "terrain_median_slope_m_per_km": float(np.median(slope[use]))
        if use.any() else 0.0,
        "terrain_p95_slope_m_per_km": float(np.percentile(slope[use], 95))
        if use.any() else 0.0,
        "terrain_gradient_fourfold_harmonic": _harmonic(
            slope_angle, 4, slope_weight),
        "terrain_gradient_eightfold_harmonic": _harmonic(
            slope_angle, 8, slope_weight),
        "erosion_volume_km3": source_m3 / 1e9,
        "sediment_deposited_fraction": deposited_m3 / source_m3
        if source_m3 > 0.0 else 0.0,
        "sediment_export_fraction": export_m3 / source_m3
        if source_m3 > 0.0 else 0.0,
        "sediment_terminal_fraction": residual_m3 / source_m3
        if source_m3 > 0.0 else 0.0,
        "sediment_closure": closure,
        "sediment_max_m": float(np.asarray(eroded["sed"]).max()),
        "lake_components": len(lake_components),
        "lake_cells": int(lake_mask.sum()),
        "largest_lake_km2": float(largest_lake * e_km * e_km),
        "lake_max_depth_m": float(np.asarray(eroded["lake_depth"]).max()),
        "lake_max_surface_range_m": max(lake_ranges, default=0.0),
    }
    for phase, seconds in eroded["timings"].items():
        metrics[f"timing_{phase}_s"] = float(seconds)
    metrics.update(_river_output_metrics(eroded))

    if baseline_z is not None:
        delta = z - baseline_z
        metrics["terrain_rms_delta_vs_B0_m"] = float(
            np.sqrt(np.mean(delta ** 2)))
        if np.array_equal(z, baseline_z):
            metrics["terrain_corr_vs_B0"] = 1.0
        else:
            metrics["terrain_corr_vs_B0"] = float(
                np.corrcoef(z.ravel(), baseline_z.ravel())[0, 1])
    return metrics


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _first_route_signature(eroded: dict[str, Any], mode: str) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    surface = np.asarray(eroded["z0"], np.float64)
    e_km = float(eroded["e_km"])
    runoff = np.exp(-_chamfer_km(surface < 0.0, e_km)
                    / erosion.L_MOIST_KM)
    if mode in ("legacy", "legacy_lengths"):
        filled = erosion.fill_depressions(surface)
        rcv, targets, weights, flat = erosion.receivers(filled)
        batches = erosion.topo_batches(rcv, targets, weights, flat)
        diffuse, channel = erosion.flow_accumulation(
            rcv, batches, surface.size, targets, weights, runoff)
    else:
        if routing_experiment is None:
            raise RuntimeError("experimental routing module unavailable")
        channel_mode = "dinf_d8" if mode == "d8_flat" else "dinf_ltd"
        graph = routing_experiment.routing_graph(
            surface, e_km, mode=channel_mode)
        if mode in ("d8_flat", "ltd_mfd"):
            targets, weights, batches = routing_experiment.freeman_graph(
                graph.filled_level, graph)
        else:
            targets, weights, batches = (
                graph.targets, graph.weights, graph.batches)
        rcv = graph.rcv
        diffuse = routing_experiment.accumulate_weighted(
            targets, weights, batches, runoff)
        channel = routing_experiment.accumulate_channel(
            rcv, batches, runoff)
    hashes = {
        "receiver_sha256": _hash_arrays(rcv),
        "diffuse_sha256": _hash_arrays(diffuse),
        "channel_sha256": _hash_arrays(channel),
    }
    return rcv, diffuse, channel, hashes


def _aggregate_full_process(seed_rows: dict[str, Any]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for arm, _, _ in FULL_PROCESS_ARMS:
        rows = [seed_rows[str(seed)]["arms"][arm] for seed in REAL_SEEDS]
        keys = sorted(set.intersection(*(set(row) for row in rows)))
        arm_agg: dict[str, Any] = {}
        for key in keys:
            values = [row[key] for row in rows]
            if (all(isinstance(v, (int, float, np.integer, np.floating))
                    and not isinstance(v, bool) for v in values)
                    and np.isfinite(np.asarray(values, float)).all()):
                arm_agg[f"median_{key}"] = float(np.median(values))
        aggregate[arm] = arm_agg
    return aggregate


def _run_full_process(checks: Checks) -> dict[str, Any]:
    print("\n== five-arm full-process ablation ==")
    result: dict[str, Any] = {
        "arm_definitions": {
            arm: {"routing_mode": mode, "description": description}
            for arm, mode, description in FULL_PROCESS_ARMS
        },
        "seeds": {},
    }
    for seed in REAL_SEEDS:
        print(f"\n  seed {seed}")
        raw: dict[str, dict[str, Any]] = {}
        walls: dict[str, float] = {}
        for arm, mode, _ in FULL_PROCESS_ARMS:
            eroded, wall_s = _run_process(seed, mode)
            raw[arm] = eroded
            walls[arm] = wall_s
        baseline_z = np.asarray(raw["B0"]["z"])
        arm_metrics: dict[str, Any] = {}
        for arm, _, _ in FULL_PROCESS_ARMS:
            metrics = _process_metrics(raw[arm], walls[arm], baseline_z)
            arm_metrics[arm] = metrics
            print(
                f"    {arm}: {metrics['wall_s']:.2f}s  "
                f"rivers={metrics['river_edges']}  "
                f"cardinal={metrics['river_cardinal_fraction']:.3f}  "
                f"same={metrics['river_longest_same_bearing_edges']}  "
                f"ero={metrics['erosion_volume_km3']:.1f} km3  "
                f"lakes={metrics['lake_components']}")
            checks.check(
                f"seed {seed} {arm}: sediment budget closes",
                abs(float(metrics["sediment_closure"]) - 1.0) <= 1e-9,
                f"closure={metrics['sediment_closure']:.12f}")
            checks.check(
                f"seed {seed} {arm}: no interior terminal sediment",
                float(metrics["sediment_terminal_fraction"]) <= 1e-12,
                f"fraction={metrics['sediment_terminal_fraction']:.3e}")
            checks.check(
                f"seed {seed} {arm}: lake components have flat surfaces",
                float(metrics["lake_max_surface_range_m"]) <= 1e-9,
                f"range={metrics['lake_max_surface_range_m']:.3e}")

        all_z0_same = all(np.array_equal(raw["B0"]["z0"], raw[arm]["z0"])
                          for arm, _, _ in FULL_PROCESS_ARMS)
        checks.check(f"seed {seed}: all arms share bit-identical z0",
                     all_z0_same)

        signatures = {}
        arrays = {}
        for arm, mode, _ in FULL_PROCESS_ARMS:
            rcv, diffuse, channel, hashes = _first_route_signature(
                raw["B0"], mode)
            arrays[arm] = (rcv, diffuse, channel)
            signatures[arm] = hashes
        b0_b1 = all(np.array_equal(a, b) for a, b in zip(
            arrays["B0"], arrays["B1"]))
        c1_c2_rcv = np.array_equal(arrays["C1"][0], arrays["C2"][0])
        c1_c2_channel_delta = float(np.max(np.abs(
            arrays["C1"][2] - arrays["C2"][2])))
        c1_c2_diffuse_rms_delta = float(np.sqrt(np.mean(
            (arrays["C1"][1] - arrays["C2"][1]) ** 2)))
        checks.check(f"seed {seed}: B0/B1 first route and accum identical",
                     b0_b1)
        checks.check(f"seed {seed}: C1/C2 first LTD receiver identical",
                     c1_c2_rcv)
        result["seeds"][str(seed)] = {
            "arms": arm_metrics,
            "first_route_hashes": signatures,
            "identity": {
                "all_arms_z0_identical": all_z0_same,
                "B0_B1_receiver_diffuse_channel_identical": b0_b1,
                "C1_C2_LTD_receiver_identical": c1_c2_rcv,
                "C1_C2_channel_max_abs_delta": c1_c2_channel_delta,
                "C1_C2_diffuse_rms_delta": c1_c2_diffuse_rms_delta,
            },
        }
    aggregate = _aggregate_full_process(result["seeds"])
    result["aggregate_medians"] = aggregate
    print("\n  aggregate medians")
    for arm, _, _ in FULL_PROCESS_ARMS:
        row = aggregate[arm]
        print(
            f"    {arm}: wall={row['median_wall_s']:.2f}s  "
            f"cardinal={row['median_river_cardinal_fraction']:.3f}  "
            f"same={row['median_river_longest_same_bearing_edges']:.0f}  "
            f"turn={row['median_river_turn_fraction']:.3f}  "
            f"erosion={row['median_erosion_volume_km3']:.0f} km3  "
            f"dep/export={row['median_sediment_deposited_fraction']:.3f}/"
            f"{row['median_sediment_export_fraction']:.3f}  "
            f"lakes={row['median_lake_components']:.0f}")
    return result


def _deterministic(builder: Callable[[np.ndarray, float], RoutingGraph]) -> bool:
    yy, xx = np.mgrid[0:37, 0:37]
    surface = -0.17 * xx - 0.11 * yy + 0.006 * (xx - yy) ** 2
    a = builder(surface, 2.0)
    b = builder(surface, 2.0)
    fields = ("filled_level", "flat_mask", "targets", "weights",
              "flow_angle", "rcv", "edge_len")
    return all(np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True)
               for f in fields)


def _run_arm(name: str,
             builder: Callable[[np.ndarray, float], RoutingGraph],
             candidate: bool, checks: Checks) -> dict[str, Any]:
    print(f"\n== {name}: synthetic direction and topology ==")
    t0 = time.perf_counter()
    plane = _plane_suite(builder)
    valley = _valley_suite(builder)
    cone = _cone_suite(builder)
    spill = _single_spill_suite(builder)
    probe_surface, _, _ = _single_spill_surface(49)
    graph = builder(probe_surface, 1.0)
    invariant = _validate_graph(graph, probe_surface)
    deterministic = _deterministic(builder)

    if candidate:
        checks.check(
            f"{name}: plane direction max <= {ANGLE_TOL_DEG:g} deg",
            plane["continuous_max_error_deg"] <= ANGLE_TOL_DEG,
            f"max={plane['continuous_max_error_deg']:.3f}")
        checks.check(
            f"{name}: plane aggregate eightfold harmonic <= 0.05",
            plane["continuous_eightfold_harmonic"] <= 0.05,
            f"h8={plane['continuous_eightfold_harmonic']:.3f}")
        checks.check(
            f"{name}: LTD plane spine stays within 1.5 cells",
            plane["spine_max_cross_track_cells"] <= 1.5,
            f"max={plane['spine_max_cross_track_cells']:.3f}")
        checks.check(
            f"{name}: cone eightfold direction harmonic <= 0.10",
            cone["continuous_eightfold_harmonic"] <= 0.10,
            f"h8={cone['continuous_eightfold_harmonic']:.3f}")
        checks.check(
            f"{name}: oblique thalweg mean error <= 2 deg",
            valley["thalweg_mean_error_deg"] <= 2.0,
            f"mean={valley['thalweg_mean_error_deg']:.3f}")
        checks.check(
            f"{name}: exact single-spill physical fill",
            abs(spill["basin_fill_mean"] - 40.0) <= 1e-9
            and spill["basin_fill_range"] <= 1e-9,
            f"mean/range={spill['basin_fill_mean']:.9f}/"
            f"{spill['basin_fill_range']:.3e}")
        checks.check(
            f"{name}: single-spill routes converge to unique outlet",
            bool(spill["channel_expected_outlet"])
            and bool(spill["diffuse_expected_outlet"]),
            f"terminal counts={spill['channel_terminal_count']}/"
            f"{spill['diffuse_terminal_count']}")
    else:
        # Known-positive calibration: the instrument must expose D8 snapping.
        checks.check(
            f"{name}: known D8 plane quantization detected",
            plane["spine_max_error_deg"] >= 15.0,
            f"max={plane['spine_max_error_deg']:.3f}")
        checks.check(
            f"{name}: known D8 plane eightfold lock detected",
            plane["spine_eightfold_harmonic"] >= 0.90,
            f"h8={plane['spine_eightfold_harmonic']:.3f}")
        checks.check(
            f"{name}: known D8 plane transverse drift detected",
            plane["spine_max_cross_track_cells"] >= 3.0,
            f"max={plane['spine_max_cross_track_cells']:.3f}")
        checks.check(
            f"{name}: known D8 cone eightfold lock detected",
            cone["spine_eightfold_harmonic"] >= 0.90,
            f"h8={cone['spine_eightfold_harmonic']:.3f}")

    checks.check(f"{name}: all graph cells topologically covered",
                 bool(invariant["all_cells_topologically_covered"]))
    checks.check(f"{name}: valid finite normalized edge weights",
                 bool(invariant["weights_finite_nonnegative"])
                 and bool(invariant["positive_weight_has_valid_target"])
                 and bool(invariant["weight_sums_valid"]))
    checks.check(f"{name}: channel receiver uses D-inf bracket off flats",
                 bool(invariant["off_flat_channel_receiver_is_diffuse_bracket"]))
    checks.check(f"{name}: channel/diffuse edges descend composite key",
                 bool(invariant["channel_strict_composite_descent"])
                 and bool(invariant["diffuse_strict_composite_descent"]))
    checks.check(f"{name}: selected edge lengths match receiver geometry",
                 bool(invariant["edge_lengths_positive"])
                 and bool(invariant["edge_lengths_match_receiver_geometry"]))
    checks.check(f"{name}: exact channel runoff closure",
                 abs(float(invariant["channel_runoff_closure"]) - 1.0)
                 <= 1e-12,
                 f"closure={invariant['channel_runoff_closure']:.12f}")
    checks.check(f"{name}: exact diffuse runoff closure",
                 abs(float(invariant["diffuse_runoff_closure"]) - 1.0)
                 <= 1e-12,
                 f"closure={invariant['diffuse_runoff_closure']:.12f}")
    checks.check(f"{name}: deterministic graph fields", deterministic)

    print(f"\n== {name}: real-seed route-only metrics ==")
    real: dict[str, Any] = {}
    for seed in REAL_SEEDS:
        metrics = _real_seed_metrics(seed, builder)
        real[str(seed)] = metrics
        print(
            f"  seed {seed}: cardinal={metrics['cardinal_edge_fraction']:.3f} "
            f"same-bearing={metrics['longest_same_bearing_edges']} edges/"
            f"{metrics['longest_same_bearing_km']:.0f} km "
            f"continuous-h8={metrics['continuous_eightfold_harmonic']:.3f} "
            f"inflation-p95={metrics['path_inflation_p95']:.3f}")
    return {
        "synthetic": {
            "planes": plane,
            "oblique_valleys": valley,
            "cone": cone,
            "single_spill_bowl": spill,
            "invariants": invariant,
            "deterministic": deterministic,
        },
        "real_seed_route_only": real,
        "elapsed_s": time.perf_counter() - t0,
    }


def _resolve_output(value: str) -> Path:
    requested = Path(value)
    path = (requested.resolve() if requested.is_absolute()
            else (OUTPUT_ROOT / requested).resolve())
    try:
        path.relative_to(OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"output must stay under {OUTPUT_ROOT}, got {path}") from exc
    if path.suffix.lower() != ".json":
        raise ValueError("routing-ablation output must be a .json file")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="run only the shipped D8/MFD known-positive calibration")
    parser.add_argument(
        "--require-candidate", action="store_true",
        help="fail if the experimental routing_graph API is unavailable")
    parser.add_argument(
        "--skip-full-process", action="store_true",
        help="run only the route-kernel synthetics and fixed-surface metrics")
    parser.add_argument(
        "--output", metavar="JSON",
        help="also persist JSON under pipeline_b/out/routing_ablation; "
             "relative paths are resolved beneath that directory")
    args = parser.parse_args()

    checks = Checks()
    result: dict[str, Any] = {
        "instrument": "pipeline_b-routing-ablation-v1",
        "official_evaluation": False,
        "public_controls_changed": False,
        "real_seeds": list(REAL_SEEDS),
        "arms": {},
    }

    result["arms"]["legacy"] = _run_arm(
        "legacy", _legacy_graph, False, checks)
    candidate_available = _candidate_callable() is not None
    result["candidate_available"] = candidate_available
    if not args.baseline_only and candidate_available:
        candidate_builder = _candidate_graph

        def build(surface: np.ndarray, dx: float) -> RoutingGraph:
            graph = candidate_builder(surface, dx)
            assert graph is not None
            return graph

        result["arms"]["candidate"] = _run_arm(
            "candidate", build, True, checks)
    elif not args.baseline_only:
        print("\n[candidate unavailable] engine.erosion exposes no "
              "experimental routing_graph/build_routing_graph/"
              "build_flow_graph callable")
        if args.require_candidate:
            checks.check("candidate router is available", False)

    if (not args.baseline_only and not args.skip_full_process
            and candidate_available):
        result["full_process"] = _run_full_process(checks)

    result["checks"] = {
        "passed": len(checks.passed),
        "failed": len(checks.failed),
        "failures": checks.failed,
    }
    print("\n== machine-readable result ==")
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(encoded)
    if args.output:
        output = _resolve_output(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
        print(f"wrote {output}")
    if checks.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
