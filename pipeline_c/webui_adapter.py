"""Generate-and-view adapter for the shared WebUI.

Pick a seed, generate, look at the result. There is no review packet,
baseline, delta, snapshot, or approval workflow here: the engine is
deterministic, so any world can be rebuilt from its seed, its resolution, and
its scale, which together are its identity.

What this generates is the *kinematic history* stage — a mantle drive, a
lithosphere strength field, the velocity solved from the two, and the plates
and boundaries that emerge from where strain localizes. It is not a map.
There is no crust, elevation, water, coastline, island, or land in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from io import BytesIO
import time

import numpy as np
from PIL import Image

from engine import VERSION
from engine.domain import tile2x2
from engine.geometry import WorldGeometry
from engine.history.constants import (
    DEFAULT_SIZE,
    HISTORY_MYR,
    SCALE_DEFAULT,
    SCALE_MAX,
    SCALE_MIN,
    STAGE_ID,
    SUPPORTED_SIZES,
)
from engine.history.kinematics import History, run_history
from engine.history.plates import (
    REGIME_CONVERGENT,
    REGIME_DIVERGENT,
    REGIME_SHEAR,
    boundary_mask,
    label_plates,
    plate_areas,
    regime,
    weak_mask,
)
from engine.views import (
    arrows,
    banded,
    categorical,
    mask,
    regime_rgb,
    scalar,
    vector,
)

VIEWS = (
    "plates",
    "boundaries",
    "regime",
    "strength",
    "strength_banded",
    "velocity",
    "strain_rate",
    "power",
    "stress",
    "intact_strength",
    "mismatch",
    "pieces_motion",
    "strain_rate_banded",
    "drive",
    "drive_phi",
    "drive_psi",
    "strength_initial",
    "boundaries_t25",
    "boundaries_t50",
    "boundaries_t75",
    "strength_t25",
    "strength_t50",
    "strength_t75",
    "plates_tiled",
)

_VIEW_PURPOSE = {
    "plates": (
        "Connected regions of strong lithosphere at the end of the history. "
        "A plate is read off the strength field; it is an input to nothing."
    ),
    "boundaries": (
        "Weak cells and the strong cells that touch a different plate. This "
        "is where the design predicts curvature, segmentation, and offsets."
    ),
    "regime": (
        "Divergent, convergent, or shear on each weak cell, from the local "
        "divergence and strain rate, so regime can vary along a boundary. "
        "Both fields are block-constant over 2 x 2 kinematic cells, so the "
        "class can only change on a solve-cell edge."
    ),
    "strength": "Lithosphere integrity at the end of the history.",
    "strength_banded": (
        "The same field in eight bands, so a gradient that changes slope "
        "shows as a band edge instead of hiding in a ramp."
    ),
    "velocity": (
        "Solved lithosphere velocity: hue is direction, brightness is speed. "
        "Grid-locked flow would show as banded hue along the axes."
    ),
    "strain_rate": (
        "Second invariant of the strain-rate tensor. It is computed from the "
        "solver's edge fluxes on the solve grid, half the kinematic grid per "
        "side, and lifted piecewise constant, so the view reads as 2 x 2 "
        "blocks: that is where damage is resolved."
    ),
    "power": (
        "Dissipated power, stiffness times the square of the strain rate. It "
        "is formed on the solve grid beside the strain rate and lifted the "
        "same way, so it too reads as 2 x 2 blocks."
    ),
    "stress": (
        "Magnitude of the stress tensor at the end, stiffness times the "
        "strain rate, interpolated to the kinematic grid rather than lifted "
        "in blocks because the seam rules of `DESIGN.md` §3.6 choose cells "
        "and directions from it."),
    "intact_strength": (
        "The stress an intact cell carries before it cracks, under the seam "
        "formulation. Flat black here: production runs the sheet's damage "
        "law, which has no intact strength."),
    "mismatch": (
        "The drag a piece failed to match, `D - u`, zero on seam cells: what "
        "the internal-stress solve is forced with at `seams = 2`. Flat black "
        "under the sheet and under `seams = 1`, which solve the drive "
        "itself."),
    "pieces_motion": (
        "The rigid bodies the last step's solve moved, categorical, with each "
        "body's velocity drawn as one arrow from its centroid in the colours "
        "the `velocity` view uses. No arrows under the sheet or `seams = 1`, "
        "which have no rigid bodies."),
    "strain_rate_banded": (
        "The same field in eight bands. It too is block-constant over 2 x 2 "
        "kinematic cells, because the strain that drives damage is resolved "
        "on the solve grid."
    ),
    "drive": "Basal traction from the mantle at the last epoch.",
    "drive_phi": "The curl-free potential the gradient part comes from.",
    "drive_psi": "The rotational potential the perpendicular part comes from.",
    "strength_initial": "The strength field before any history ran.",
    "boundaries_t25": (
        "Weak lithosphere a quarter of the way through; plate contacts are "
        "labelled at the final epoch only."
    ),
    "boundaries_t50": (
        "Weak lithosphere half way through; plate contacts are labelled at "
        "the final epoch only."
    ),
    "boundaries_t75": (
        "Weak lithosphere three quarters of the way through; plate contacts "
        "are labelled at the final epoch only."
    ),
    "strength_t25": "Strength a quarter of the way through.",
    "strength_t50": "Strength half way through.",
    "strength_t75": "Strength three quarters of the way through.",
    "plates_tiled": (
        "The plates view repeated two by two at native resolution, so the "
        "wrap point sits mid-image and a seam could not hide."
    ),
}

_EPOCH_SUFFIX = ("_t25", "_t50", "_t75")


@dataclass(slots=True)
class EpochViews:
    """The plate reading of one epoch. Only the final epoch gets one."""

    labels: np.ndarray
    weak: np.ndarray
    boundary: np.ndarray
    regime: np.ndarray


@dataclass(slots=True)
class World:
    seed: int
    world_id: str
    pixels: int
    scale_km: int
    geometry: WorldGeometry
    history: History
    final: EpochViews
    epoch_weak: list[np.ndarray]
    drive_field: np.ndarray
    drive_phi: np.ndarray
    drive_psi: np.ndarray
    elapsed_s: float
    plate_percent: list[float] = dataclass_field(default_factory=list)


def meta() -> dict:
    return {
        "name": "pipeline_c land-origin lab",
        "version": VERSION,
        "ready": True,
        "stage": STAGE_ID,
        "status": (
            "Kinematic history only: emergent plates and boundaries over a "
            "periodic parent world. No crust, elevation, water, coastline, "
            "island, or land."
        ),
        "controls": [
            {
                "name": "scale_km",
                "ctype": "int",
                "default": SCALE_DEFAULT,
                "lo": SCALE_MIN,
                "hi": SCALE_MAX,
                "tier": "primary",
                "invalidates": "full",
                "promise": (
                    "Kilometres per delivered pixel. World geometry, not a "
                    "formation control: it sizes the simulated planet and is "
                    "never swept."
                ),
            }
        ],
        "default_size": DEFAULT_SIZE,
        "supported_sizes": list(SUPPORTED_SIZES),
        "views": list(VIEWS),
        "view_purposes": dict(_VIEW_PURPOSE),
    }


def _normalized_scale(controls: dict | None) -> int:
    if controls is None:
        return SCALE_DEFAULT
    if not isinstance(controls, dict):
        raise ValueError("controls must be a mapping")
    unknown = sorted(set(controls) - {"scale_km"})
    if unknown:
        raise ValueError(f"unknown control(s): {unknown}")
    if "scale_km" not in controls:
        return SCALE_DEFAULT
    value = controls["scale_km"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("scale_km must be an integer")
    if not SCALE_MIN <= value <= SCALE_MAX:
        raise ValueError(f"scale_km must be between {SCALE_MIN} and {SCALE_MAX}")
    return value


def _normalized_size(size: int | None) -> int:
    if size is None:
        return DEFAULT_SIZE
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError("size must be an integer")
    if size not in SUPPORTED_SIZES:
        raise ValueError(f"size must be one of {SUPPORTED_SIZES}")
    return size


def generate(seed: int, controls: dict | None = None, size: int | None = None,
             *, _steps: int | None = None) -> World:
    """Run one kinematic history. Deterministic and self-contained."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    scale_km = _normalized_scale(controls)
    pixels = _normalized_size(size)
    geometry = WorldGeometry(seed, pixels, scale_km)

    started = time.perf_counter()
    history = run_history(geometry, steps=_steps)
    # Labels, contacts, and regimes are read off the final epoch only. The
    # earlier epochs keep their weak mask, which is what their views show.
    epoch_weak = [weak_mask(epoch.strength) for epoch in history.epochs]
    last = history.epochs[-1]
    labels = label_plates(last.strength)
    final = EpochViews(
        labels=labels,
        weak=epoch_weak[-1],
        boundary=boundary_mask(labels, epoch_weak[-1]),
        regime=regime(last.divergence, last.strain_rate, epoch_weak[-1]),
    )
    phi, psi = history.drive.potentials(HISTORY_MYR)
    elapsed = time.perf_counter() - started

    cells = geometry.history_n**2
    areas = plate_areas(final.labels)
    percent = [round(100.0 * float(area) / cells, 4)
               for area in areas if area >= 0.01 * cells]

    return World(
        seed=seed,
        world_id=geometry.world_id,
        pixels=pixels,
        scale_km=scale_km,
        geometry=geometry,
        history=history,
        final=final,
        epoch_weak=epoch_weak,
        drive_field=history.drive.field(HISTORY_MYR),
        drive_phi=phi,
        drive_psi=psi,
        elapsed_s=round(elapsed, 3),
        plate_percent=percent,
    )


def views(world: World) -> list[str]:
    return list(VIEWS)


def _epoch_rgb(world: World, stem: str, index: int) -> np.ndarray:
    if stem == "boundaries":
        return mask(world.epoch_weak[index])
    if stem == "strength":
        return scalar(world.history.epochs[index].strength)
    raise ValueError(f"unknown view: {stem}{_EPOCH_SUFFIX[index]!r}")


def _rgb(world: World, view: str) -> np.ndarray:
    for index, suffix in enumerate(_EPOCH_SUFFIX):
        if view.endswith(suffix):
            return _epoch_rgb(world, view[: -len(suffix)], index)

    final = world.history.epochs[-1]
    plate = world.final
    if view == "plates":
        return categorical(plate.labels)
    if view == "plates_tiled":
        return tile2x2(categorical(plate.labels))
    if view == "boundaries":
        return mask(plate.boundary)
    if view == "regime":
        return regime_rgb(plate.regime)
    if view == "strength":
        return scalar(final.strength)
    if view == "strength_banded":
        return banded(final.strength)
    if view == "velocity":
        return vector(final.velocity)
    if view == "strain_rate":
        return scalar(final.strain_rate)
    if view == "power":
        return scalar(final.power)
    if view == "stress":
        return scalar(final.stress)
    if view == "intact_strength":
        return scalar(world.history.sigma_c_field)
    if view == "mismatch":
        return scalar(final.mismatch)
    if view == "pieces_motion":
        if final.piece_labels is None:
            return categorical(plate.labels)
        return arrows(categorical(final.piece_labels), final.piece_centroid,
                      final.piece_velocity)
    if view == "strain_rate_banded":
        return banded(final.strain_rate)
    if view == "drive":
        return vector(world.drive_field)
    if view == "drive_phi":
        return scalar(world.drive_phi)
    if view == "drive_psi":
        return scalar(world.drive_psi)
    if view == "strength_initial":
        return scalar(world.history.strength_initial)
    raise ValueError(f"unknown view: {view!r}")


def render_png(world: World, view: str) -> bytes:
    """Native-resolution PNG. Never resized, never drawn on."""
    if view not in VIEWS:
        raise ValueError(f"unknown view: {view!r}")
    image = Image.fromarray(np.ascontiguousarray(_rgb(world, view)), mode="RGB")
    # Row zero is the parent minimum-y row; display north up.
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _regime_share(regime_field: np.ndarray, weak: np.ndarray) -> dict:
    """Share of the weak cells the local strain calls each regime."""
    total = int(np.count_nonzero(weak))
    if not total:
        return {"divergent": 0.0, "convergent": 0.0, "shear": 0.0}
    return {
        "divergent": round(
            float(np.count_nonzero(regime_field == REGIME_DIVERGENT)) / total, 6),
        "convergent": round(
            float(np.count_nonzero(regime_field == REGIME_CONVERGENT)) / total, 6),
        "shear": round(
            float(np.count_nonzero(regime_field == REGIME_SHEAR)) / total, 6),
    }


def report(world: World) -> dict:
    history = world.history
    geometry = world.geometry
    final = history.epochs[-1]
    speed = np.sqrt(final.velocity[0] ** 2 + final.velocity[1] ** 2)
    steps = history.steps
    return {
        "seed": int(world.seed),
        "pixels": int(world.pixels),
        "scale_km": int(world.scale_km),
        "window_km": int(geometry.window_km),
        "parent_km": int(geometry.parent_km),
        "history_n": int(geometry.history_n),
        "cell_km": int(geometry.cell_km),
        "steps": int(steps),
        "step_myr": float(history.step_myr),
        "history_myr": float(HISTORY_MYR),
        "generation_seconds": float(world.elapsed_s),
        "world_id": world.world_id,
        "stage": STAGE_ID,
        "plate_count": len(world.plate_percent),
        "plate_area_percent": [float(value) for value in world.plate_percent],
        "weak_fraction_final": float(history.weak_fraction[-1]),
        "yield_strain_per_myr": float(history.yield_strain_per_myr),
        "boundary_cell_fraction": float(np.mean(world.final.boundary)),
        "regime_share": _regime_share(world.final.regime, world.final.weak),
        "weak_fraction_by_epoch": [
            float(history.weak_fraction[round(fraction * steps) - 1])
            for fraction in (0.25, 0.5, 0.75, 1.0)
        ],
        "solver_cycles_mean": float(np.mean(history.solver_cycles)),
        "solver_cycles_max": int(max(history.solver_cycles)),
        "solver_residual_max": float(max(history.solver_residual)),
        "velocity_rms_km_per_myr": float(np.sqrt(np.mean(speed**2))),
        "contains": "emergent plate labels, boundaries, and kinematic fields only",
        "does_not_contain": (
            "crust, elevation, water, coastline, islands, land, or a map"),
    }


__all__ = [
    "VIEWS",
    "EpochViews",
    "World",
    "generate",
    "meta",
    "render_png",
    "report",
    "views",
]
