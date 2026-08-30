"""The generation spine. Stages consume the World and add layers.

C1: plates (partition + Euler motion) -> crust (nuclei + border stack) ->
proto-elevation (dies in C2 when the real two-regime elevation lands).
"""

import time

import numpy as np

from . import registry
from .boundaries import stage_boundaries
from .crust import stage_crust
from .eras import stage_eras
from .erosion import stage_erosion
from .hydrology import stage_hydrology
from .relief import stage_relief
from .sediment import stage_sediment
from .tectonics import stage_plates
from .world import World


def stage_sealevel(world: World) -> None:
    """Tail stage: the late-class sea-level trim. Applying it after the
    expensive head is what makes the slider near-instant in the webui."""
    sl = float(world.controls["sea_level_m"])
    if sl != 0.0:
        world["elevation"] = (world["elevation"].astype(np.float64)
                              - sl).astype(np.float32)


# Head: everything expensive. Tail: cheap stages a late-class control can
# re-run against a cached head (contract section 9).
HEAD_STAGES: list[tuple[str, object]] = [
    ("plates", stage_plates),
    ("crust", stage_crust),
    ("boundaries", stage_boundaries),
    ("eras", stage_eras),
    ("relief", stage_relief),
    ("erosion", stage_erosion),
    ("sediment", stage_sediment),
]
TAIL_STAGES: list[tuple[str, object]] = [
    ("sealevel", stage_sealevel),
    ("hydrology", stage_hydrology),
]
STAGES = HEAD_STAGES + TAIL_STAGES


def _findings_pass(world: World) -> None:
    """Post-run checks (contract section 8). Findings report; never raise."""
    e = world["elevation"]
    land = e >= 0.0
    lf = float(land.mean())
    world.findings.append(
        {"check": "land_fraction", "level": "info", "value": round(lf, 4)}
    )
    target = float(world.controls.get("land_fraction", lf))
    if abs(lf - target) > 0.08:
        world.findings.append(
            {"check": "land_fraction_target", "level": "warn",
             "msg": f"achieved {lf:.3f} vs target {target:.3f}"})
    world.findings.append(
        {"check": "elevation_range_m", "level": "info",
         "value": [round(float(e.min()), 1), round(float(e.max()), 1)]}
    )
    # roughness grammar (design.md): land is rough, the deep floor smooth.
    ef = e.astype(np.float64)
    gy, gx = np.gradient(ef)
    g = np.hypot(gx, gy) / max(world.cell_km, 1e-9)   # m per km
    zones = {"land": land, "shelf_band": (ef < 0) & (ef >= -250.0),
             "deep": ef < -1500.0}
    # median, not mean: sparse steep features (trench walls, seamounts)
    # are legitimate in the deep; uniform *texture* roughness is not.
    rough = {k: round(float(np.median(g[m])), 1) for k, m in zones.items()
             if m.any()}
    ratio = (round(rough["deep"] / rough["land"], 3)
             if rough.get("deep") is not None and rough.get("land", 0) > 0
             else None)
    world.findings.append(
        {"check": "roughness", "level":
         "warn" if ratio is not None and ratio > 0.8 else "info",
         "median_grad_m_per_km": rough, "deep_vs_land": ratio,
         "msg": None if ratio is None or ratio <= 0.8 else
         "deep floor texture as rough as land (dendrites-in-the-abyss?)"})
    if land.any():
        rows, cols = np.nonzero(land)
        h, w = world.shape
        dist = np.minimum(np.minimum(rows, h - 1 - rows),
                          np.minimum(cols, w - 1 - cols))
        d = int(dist.min())
        world.findings.append(
            {"check": "border_ring", "level": "info" if d >= 1 else "warn",
             "min_land_border_dist_cells": d, "ok": d >= 1,
             "msg": None if d >= 1 else "land touches the frame"}
        )
    else:
        world.findings.append(
            {"check": "border_ring", "level": "info",
             "min_land_border_dist_cells": None, "ok": True,
             "msg": "no land on map"}
        )


def _run(world: World, stages) -> None:
    for name, fn in stages:
        t0 = time.perf_counter()
        fn(world)
        world.timings[name] = round(time.perf_counter() - t0, 4)


def generate(seed: int, controls: dict | None = None,
             size: int | tuple[int, int] = 256) -> World:
    """generate(seed, controls, size) -> World. Never raises for in-range
    input; problems become findings (contract section 8)."""
    world = generate_head(seed, controls, size)
    run_tail(world)
    return world


def generate_head(seed: int, controls: dict | None = None,
                  size: int | tuple[int, int] = 256) -> World:
    """Run the expensive head only (through sediment). Pair with
    run_tail(); cache/clone between them for late-class recompute."""
    shape = (size, size) if isinstance(size, int) else (size[0], size[1])
    values, findings = registry.resolve(controls)
    world = World(seed, shape, values)
    world.findings.extend(findings)
    _run(world, HEAD_STAGES)
    return world


def run_tail(world: World) -> None:
    _run(world, TAIL_STAGES)
    _findings_pass(world)


def clone_world(world: World) -> World:
    """Deep-enough copy for tail re-runs against a cached head."""
    import copy

    w = World(world.seed, world.shape, dict(world.controls))
    w.layers = {k: v.copy() for k, v in world.layers.items()}
    w.meta = copy.deepcopy(world.meta)
    w.findings = [dict(f) for f in world.findings]
    w.timings = dict(world.timings)
    return w
