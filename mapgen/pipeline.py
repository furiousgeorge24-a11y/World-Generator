"""The generation spine. Stages consume the World and add layers.

C0 ships a single stub stage so the harness (registry, RNG keying, render,
report, batch, webui) is real before any terrain science exists. The stub
and its controls die in C1.
"""

import time

import numpy as np

from . import registry
from .noise import fbm
from .rng import salts_for
from .world import World

_MAX_OCTAVES = 6


def _stage_stub_relief(world: World) -> None:
    """STUB: fBm elevation. Exists only to exercise the harness."""
    c = world.controls
    xkm, ykm = world.coords_km()
    scale = c["stub_feature_scale_km"]
    # Cap octaves so the finest lattice stays >= ~2 cells: coarse structure
    # is resolution-invariant; only sub-cell filigree drops out at preview
    # sizes (contract section 6).
    fit = int(np.floor(np.log2(max(scale / (2.0 * world.cell_km), 2.0))))
    n_oct = max(1, min(_MAX_OCTAVES, fit))
    salts = salts_for(world.seed, "stub_relief", _MAX_OCTAVES)[:n_oct]
    e = fbm(xkm, ykm, scale, salts)
    world["elevation"] = ((e + c["stub_land_bias"]) * c["stub_relief_amp_m"]).astype(
        np.float32
    )


STAGES: list[tuple[str, object]] = [
    ("stub_relief", _stage_stub_relief),
]


def _findings_pass(world: World) -> None:
    """Post-run checks (contract section 8). Findings report; never raise."""
    e = world["elevation"]
    land = e >= 0.0
    world.findings.append(
        {"check": "land_fraction", "level": "info",
         "value": round(float(land.mean()), 4)}
    )
    world.findings.append(
        {"check": "elevation_range_m", "level": "info",
         "value": [round(float(e.min()), 1), round(float(e.max()), 1)]}
    )
    if land.any():
        rows, cols = np.nonzero(land)
        h, w = world.shape
        dist = np.minimum(np.minimum(rows, h - 1 - rows),
                          np.minimum(cols, w - 1 - cols))
        d = int(dist.min())
        world.findings.append(
            {"check": "border_ring", "level": "info" if d >= 1 else "warn",
             "min_land_border_dist_cells": d, "ok": d >= 1,
             "msg": None if d >= 1 else
             "land touches the frame (expected until the C1 border stack)"}
        )
    else:
        world.findings.append(
            {"check": "border_ring", "level": "info",
             "min_land_border_dist_cells": None, "ok": True,
             "msg": "no land on map"}
        )


def generate(seed: int, controls: dict | None = None,
             size: int | tuple[int, int] = 256) -> World:
    """generate(seed, controls, size) -> World. Never raises for in-range
    input; problems become findings (contract section 8)."""
    shape = (size, size) if isinstance(size, int) else (size[0], size[1])
    values, findings = registry.resolve(controls)
    world = World(seed, shape, values)
    world.findings.extend(findings)
    for name, fn in STAGES:
        t0 = time.perf_counter()
        fn(world)
        world.timings[name] = round(time.perf_counter() - t0, 4)
    _findings_pass(world)
    return world
