"""The exploration lab: the same engine, on its own port, with dials.

A second adapter for the shared WebUI shell, served on port 5003 by
`explore.bat`. It shares the engine with `webui_adapter.py` and changes
nothing there. Every generation runs several seeds at once and every view is a
contact sheet of them, because a setting that works on one world and not on
the next has not been found.

**These dials are development instruments.** They are not author controls,
they never appear in the production adapter, and whatever is found with them
is frozen back into `constants.py` afterwards. The percentile yield in
particular is a calibration convenience that is not to survive into
production; `run_history` says so in its docstring.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from io import BytesIO
import multiprocessing
import os
import time

import numpy as np
from PIL import Image

from engine import VERSION
from engine.history.constants import (
    DEFAULT_SIZE,
    SCALE_DEFAULT,
    SCALE_MAX,
    SCALE_MIN,
    STAGE_ID,
    SUPPORTED_SIZES,
    WEAK_THRESHOLD,
)
from engine.history.kinematics import HistoryParams
from engine.views import (
    arrows,
    categorical,
    mask,
    regime_rgb,
    scalar,
    vector,
)
from eval.palette import CATEGORY_COLORS
from explore_worker import EARLY_MYR, run_one_world

SEED_MODULUS = 2**32

#: Worlds per generation, and workers in the pool: the same number, so one
#: generation is one round of the pool.
MAX_SEEDS = 8

#: Sheet layout. Panels are the raw rasters at native history resolution with
#: nothing drawn on them; the gutter is black.
SHEET_COLUMNS = 4
SHEET_GUTTER_PX = 4

#: The trajectory strip: one filled column per step, `STRIP_PX` tall.
STRIP_PX = 64
STRIP_GUTTER_PX = 2
STRIP_COLUMN_RGB = tuple(int(value) for value in CATEGORY_COLORS[2])
STRIP_LINE_RGB = tuple(int(value) for value in CATEGORY_COLORS[0])

VIEWS = (
    "plates",
    "boundaries",
    "weak_t16",
    "weak_t32",
    "weak_t64",
    "weak_t25",
    "weak_t50",
    "weak_t75",
    "strength",
    "strength_t25",
    "strength_t50",
    "strength_t75",
    "regime",
    "velocity",
    "strain_rate",
    "power",
    "stress",
    "intact_strength",
    "mismatch",
    "pieces_motion",
    "trajectory",
    "drive",
)

_VIEW_PURPOSE = {
    "plates": "Connected regions of strong lithosphere at the end, per seed.",
    "boundaries": "Weak cells plus strong cells touching a different plate.",
    "weak_t16": "Weak lithosphere at 16 Myr: what fails first, and in what shape.",
    "weak_t32": "Weak lithosphere at 32 Myr.",
    "weak_t64": "Weak lithosphere at 64 Myr.",
    "weak_t25": "Weak lithosphere a quarter of the way through.",
    "weak_t50": "Weak lithosphere half way through.",
    "weak_t75": "Weak lithosphere three quarters of the way through.",
    "strength": "Lithosphere integrity at the end.",
    "strength_t25": "Strength a quarter of the way through.",
    "strength_t50": "Strength half way through.",
    "strength_t75": "Strength three quarters of the way through.",
    "regime": "Divergent, convergent, or shear on each weak cell at the end.",
    "velocity": "Solved velocity: hue is direction, brightness is speed.",
    "strain_rate": (
        "Second invariant of the strain rate, block-constant over 2 x 2 "
        "kinematic cells because that is where damage is resolved."),
    "power": (
        "Dissipated power, stiffness times the square of the strain rate, "
        "block-constant over 2 x 2 kinematic cells because that is where "
        "damage is resolved."),
    "stress": (
        "Magnitude of the stress tensor at the end, stiffness times the "
        "strain rate, interpolated to the kinematic grid rather than lifted "
        "in blocks because the seam rules choose cells and directions from "
        "it."),
    "intact_strength": (
        "The stress an intact cell carries before it cracks: a percentile of "
        "the first step's stress, scaled cell by cell by the strength noise. "
        "Flat black under the sheet's damage law, which has no intact "
        "strength."),
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
    "trajectory": (
        "Weak fraction against time, one filled column per step, one strip "
        "per seed. A flat strip is a settled regime; a rising one is not."),
    "drive": "Basal traction from the mantle at the end of the history.",
}

#: The dials, in the order `WORK_ORDER_C03_5.md` §3.1 lists them. `kind` is
#: the Python type the adapter enforces; `field` names the `HistoryParams`
#: field a dial sets, or is `None` for the two that are not history params.
_DIALS = (
    ("scale_km", int, SCALE_DEFAULT, SCALE_MIN, SCALE_MAX, "advanced", None,
     "Kilometres per delivered pixel. World geometry, not a formation "
     "control: it sizes the simulated planet and is never swept."),
    ("seeds_per_view", int, MAX_SEEDS, 1, MAX_SEEDS, "primary", None,
     "Worlds per generation, seeds `seed` to `seed + n - 1`, shown side by "
     "side."),
    ("stiffness_fraction", float, 0.125, 0.05, 2.0, "primary",
     "stiffness_fraction",
     "Fraction of the world over which a plate holds together. Below a "
     "plate's size the plate deforms internally."),
    ("yield_percentile", float, 12.0, 1.0, 40.0, "primary", "yield_percentile",
     "Percent of the initial strain field above yield. What breaks first."),
    ("heal_time_myr", float, 100.0, 10.0, 1000.0, "primary", "heal_time_myr",
     "Time for a fault to seal once it stops moving."),
    # The floor is the search's own lower bound since `WORK_ORDER_C04_5.md`
    # §3, so a cell the search drew can be typed into the lab as it stands.
    ("damage_time_myr", float, 5.0, 0.5, 100.0, "primary", "damage_time_myr",
     "Time for intact rock at twice yield to fail."),
    ("work_damage", int, 0, 0, 1, "primary", "work_damage",
     "0: damage by strain-rate excess. 1: damage by dissipated-work excess, "
     "stress times strain rate. Above the same threshold the work law fails "
     "a cell faster. Under seams, 0 is the slip-rate law that keeps a "
     "slipping fault weak; 1 is the work law, under which an open fault "
     "heals."),
    ("seams", int, 2, 0, 2, "primary", "seams",
     "0: the sheet, diffuse damage wherever strain exceeds yield. 1: seams, "
     "damage only on a seam, at its tip, or at a nucleation site; boundaries "
     "one cell wide by construction. 2: rigid pieces; stress is the integral "
     "of the unmatched drag over a piece; seams on markers."),
    ("crack_speed_km_per_myr", float, 40.0, 0.0, 400.0, "primary",
     "crack_speed_km_per_myr",
     "How fast a crack tip runs. A rift propagates at tens of kilometres per "
     "million years."),
    ("nucleations_per_step", int, 2, 0, 20, "primary", "nucleations_per_step",
     "New cracks per step at the highest-stress intact cells away from "
     "existing seams."),
    ("toughness_fraction", float, 1.0, 0.05, 1.0, "primary",
     "toughness_fraction",
     "Fracture toughness as a fraction of intact strength. Cracks propagate "
     "at this fraction of the stress it takes to nucleate one, for a crack "
     "one cell long; longer cracks propagate at less."),
    ("strength_spread", float, 0.1, 0.0, 0.3, "primary", "strength_spread",
     "Initial heterogeneity of the lithosphere. Soft spots concentrate "
     "strain and may seed failure."),
    ("strength_exponent", int, 4, 2, 6, "advanced", "strength_exponent",
     "How steeply stiffness falls with damage."),
    ("drive_wavelength_km", float, 5120.0, 640.0, 40960.0, "primary",
     "drive_wavelength_km",
     "Coarsest mantle wavelength in kilometres, the same at every resolution "
     "and scale. It sets how many mantle cells the world holds and so how "
     "many plates can form; 5,120 km is two cells across the default "
     "1024-px world."),
    ("drive_shear", float, 0.5, 0.0, 1.0, "advanced", "drive_shear",
     "Rotational drive relative to pushing drive."),
    ("history_myr", int, 300, 100, 600, "advanced", "history_myr",
     "How long the history runs."),
    ("max_cycles", int, 40, 10, 100, "advanced", "max_cycles",
     "Solver effort per step; the report shows the worst residual."),
    ("solve_divisor", int, 2, 1, 2, "advanced", "solve_divisor",
     "Kinematic cells per solve cell. 2 solves velocity on half the grid and "
     "lifts strain back in 2 x 2 blocks, so a zone cannot be narrower than "
     "two cells. 1 solves on the full grid at about six times the cost."),
)

_DIAL_BY_NAME = {dial[0]: dial for dial in _DIALS}

#: A world is `stable` for the screen when it has plates, a weak set that is
#: neither vanishing nor most of the world, and a weak set that is not still
#: growing at the end. It is a screening number for the person at the dials.
STABLE_PLATE_COUNT = (3, 8)
STABLE_WEAK_FINAL = (0.02, 0.25)
STABLE_PEAK_RATIO = 1.5

_POOL: ProcessPoolExecutor | None = None
_POOL_REFUSED = False


@dataclass(slots=True)
class Bundle:
    """One generation: several worlds under one setting of the dials."""

    seed: int
    pixels: int
    scale_km: int
    controls: dict
    params: HistoryParams
    worlds: list[dict]
    parallel: bool
    elapsed_s: float


def meta() -> dict:
    return {
        "name": "pipeline_c exploration lab",
        "version": VERSION,
        "ready": True,
        "stage": STAGE_ID,
        "status": (
            "Development dials on the kinematic history, not author controls: "
            "every generation runs eight seeds at the chosen setting and every "
            "view is a contact sheet of them. The lab defaults to `seams = 2`, "
            "the block model, because that is what it exists to look at; the "
            "engine's own default is 0 and production is the sheet."
        ),
        "default_size": DEFAULT_SIZE,
        "supported_sizes": list(SUPPORTED_SIZES),
        "controls": [
            {
                "name": name,
                "ctype": "int" if kind is int else "float",
                "default": default,
                "lo": low,
                "hi": high,
                "tier": tier,
                "invalidates": "full",
                "promise": promise,
            }
            for name, kind, default, low, high, tier, _field, promise in _DIALS
        ],
        "views": list(VIEWS),
        "view_purposes": dict(_VIEW_PURPOSE),
    }


def _normalized_controls(controls: dict | None) -> dict:
    """Every dial, defaulted, type-checked, and range-checked."""
    if controls is None:
        controls = {}
    if not isinstance(controls, dict):
        raise ValueError("controls must be a mapping")
    unknown = sorted(set(controls) - set(_DIAL_BY_NAME))
    if unknown:
        raise ValueError(f"unknown control(s): {unknown}")
    values: dict = {}
    for name, kind, default, low, high, _tier, _field, _promise in _DIALS:
        if name not in controls:
            values[name] = default
            continue
        value = controls[name]
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a number, not a bool")
        if kind is int:
            if not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        elif not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        else:
            value = float(value)
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        values[name] = value
    return values


def _params_of(controls: dict) -> HistoryParams:
    return HistoryParams(**{
        dial[6]: controls[dial[0]] for dial in _DIALS if dial[6] is not None
    })


def _normalized_size(size: int | None) -> int:
    if size is None:
        return DEFAULT_SIZE
    if isinstance(size, bool) or not isinstance(size, int):
        raise ValueError("size must be an integer")
    if size not in SUPPORTED_SIZES:
        raise ValueError(f"size must be one of {SUPPORTED_SIZES}")
    return size


def cap_blas_threads() -> None:
    """One BLAS thread per worker, set before any pool is created.

    The block model solves a dense system every step, and OpenBLAS would
    give each of the eight workers a thread per logical core for it,
    spinning between calls: eight workers times twenty-four threads on
    thirty-two cores starved the search of 2026-09-03 to 3.4 cells a minute
    against 14 with the cap. A spawned child inherits the environment at
    its start, before it imports NumPy, so setting it in the parent before
    the pool exists is what reaches the children. Every pool that runs
    worlds calls this: the lab's here, the search server's in
    `search_server.py`, and the launchers set the same variables so the
    server process itself is covered whatever path creates its pool.
    """
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def _pool() -> ProcessPoolExecutor | None:
    """The one pool, created on the first generate and kept for the process.

    A pool and the shell's reloader do not mix: the reloader restarts this
    process on any edit and would leak the children, so with it on the lab
    runs sequentially and says so in the report. `explore.bat` sets
    `WEBUI_RELOAD=0`.
    """
    global _POOL, _POOL_REFUSED
    if _POOL is not None:
        return _POOL
    if _POOL_REFUSED:
        return None
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        _POOL_REFUSED = True
        return None
    cap_blas_threads()
    try:
        _POOL = ProcessPoolExecutor(
            max_workers=MAX_SEEDS,
            mp_context=multiprocessing.get_context("spawn"))
    except OSError:
        _POOL_REFUSED = True
        return None
    return _POOL


def _seeds(seed: int, count: int) -> list[int]:
    return [(seed + offset) % SEED_MODULUS for offset in range(count)]


def generate(seed: int, controls: dict | None = None, size: int | None = None,
             *, _parallel: bool | None = None) -> Bundle:
    """Run `seeds_per_view` worlds under one setting of the dials.

    `_parallel` forces the pool on or off; the shell never passes it and the
    determinism test uses it to compare the two paths.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not 0 <= seed < SEED_MODULUS:
        raise ValueError("seed must fit in a uint32")
    values = _normalized_controls(controls)
    pixels = _normalized_size(size)
    params = _params_of(values)
    record = params.to_record()
    seeds = _seeds(seed, values["seeds_per_view"])
    scale_km = values["scale_km"]

    started = time.perf_counter()
    pool = None if _parallel is False else _pool()
    parallel = False
    worlds: list[dict] = []
    if pool is not None:
        try:
            futures = [pool.submit(run_one_world, one, pixels, scale_km, record)
                       for one in seeds]
            worlds = [future.result() for future in futures]
            parallel = True
        except (BrokenProcessPool, OSError):
            worlds = []
    if not worlds:
        worlds = [run_one_world(one, pixels, scale_km, record) for one in seeds]
    elapsed = time.perf_counter() - started

    return Bundle(seed=seed, pixels=pixels, scale_km=scale_km, controls=values,
                  params=params, worlds=worlds, parallel=parallel,
                  elapsed_s=round(elapsed, 3))


def views(bundle: Bundle) -> list[str]:
    return list(VIEWS)


def _panel(rgb: np.ndarray) -> Image.Image:
    """One raster, in the orientation the production renderer delivers."""
    image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    # Row zero is the parent minimum-y row; display north up.
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def _tile(panels: list[Image.Image]) -> Image.Image:
    """Panels in reading order, up to four across, on a black gutter."""
    width, height = panels[0].size
    columns = min(SHEET_COLUMNS, len(panels))
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * width + (columns - 1) * SHEET_GUTTER_PX,
         rows * height + (rows - 1) * SHEET_GUTTER_PX),
        (0, 0, 0),
    )
    for index, panel in enumerate(panels):
        column, row = index % columns, index // columns
        sheet.paste(panel, (column * (width + SHEET_GUTTER_PX),
                            row * (height + SHEET_GUTTER_PX)))
    return sheet


def _world_rgb(world: dict, view: str) -> np.ndarray:
    if view == "plates":
        return categorical(world["labels"])
    if view == "boundaries":
        return mask(world["boundary"])
    for index, t_myr in enumerate(EARLY_MYR):
        if view == f"weak_t{int(t_myr)}":
            return mask(world["weak_early"][index])
    for index, suffix in enumerate(("_t25", "_t50", "_t75")):
        if view == f"weak{suffix}":
            return mask(world["weak_epochs"][index])
        if view == f"strength{suffix}":
            return scalar(world["strength_epochs"][index])
    if view == "strength":
        return scalar(world["strength_final"])
    if view == "regime":
        return regime_rgb(world["regime"])
    if view == "velocity":
        return vector(world["velocity"])
    if view == "strain_rate":
        return scalar(world["strain_rate"])
    if view == "power":
        return scalar(world["power"])
    if view == "stress":
        return scalar(world["stress"])
    if view == "intact_strength":
        return scalar(world["intact_strength"])
    if view == "mismatch":
        return scalar(world["mismatch"])
    if view == "pieces_motion":
        labels = world["piece_labels"]
        if labels is None:
            return categorical(world["labels"])
        return arrows(categorical(labels), world["piece_centroid"],
                      world["piece_motion"])
    if view == "drive":
        return vector(world["drive"])
    raise ValueError(f"unknown view: {view!r}")


def _trajectory_sheet(bundle: Bundle) -> Image.Image:
    """Weak fraction against time, one strip per world, stacked.

    A filled column per step whose height is the weak fraction, on black, with
    a one-pixel line at the half mark. No axes and no text: it is here to
    separate a settled regime from a slow collapse at a glance.
    """
    strips = []
    for world in bundle.worlds:
        fractions = world["weak_fraction"]
        strip = np.zeros((STRIP_PX, len(fractions), 3), dtype=np.uint8)
        for column, fraction in enumerate(fractions):
            height = int(round(min(max(float(fraction), 0.0), 1.0) * STRIP_PX))
            if height:
                strip[STRIP_PX - height:, column] = STRIP_COLUMN_RGB
        strip[STRIP_PX // 2, :] = STRIP_LINE_RGB
        strips.append(strip)

    width = max(strip.shape[1] for strip in strips)
    height = (len(strips) * STRIP_PX
              + (len(strips) - 1) * STRIP_GUTTER_PX)
    sheet = np.zeros((height, width, 3), dtype=np.uint8)
    for index, strip in enumerate(strips):
        top = index * (STRIP_PX + STRIP_GUTTER_PX)
        sheet[top:top + STRIP_PX, :strip.shape[1]] = strip
    return Image.fromarray(sheet, mode="RGB")


def sheet(bundle: Bundle, view: str) -> Image.Image:
    """The contact sheet for one view. Nothing is drawn on the panels."""
    if view not in VIEWS:
        raise ValueError(f"unknown view: {view!r}")
    if view == "trajectory":
        return _trajectory_sheet(bundle)
    return _tile([_panel(_world_rgb(world, view)) for world in bundle.worlds])


def render_png(bundle: Bundle, view: str) -> bytes:
    buffer = BytesIO()
    sheet(bundle, view).save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _is_stable(world: dict) -> bool:
    """The screen of §3.4. Not a gate and not an approval."""
    plate_count = len(world["plate_percent"])
    weak_final = world["weak_final"]
    if not STABLE_PLATE_COUNT[0] <= plate_count <= STABLE_PLATE_COUNT[1]:
        return False
    if not STABLE_WEAK_FINAL[0] <= weak_final <= STABLE_WEAK_FINAL[1]:
        return False
    if weak_final <= 0.0:
        return False
    return world["weak_peak"] / weak_final < STABLE_PEAK_RATIO


def report(bundle: Bundle) -> dict:
    worlds = [
        {
            "seed": world["seed"],
            "plate_count": len(world["plate_percent"]),
            "plate_area_percent": list(world["plate_percent"]),
            "weak_final": round(world["weak_final"], 6),
            "weak_peak": round(world["weak_peak"], 6),
            "weak_peak_myr": world["weak_peak_myr"],
            "weak_at_100_myr": round(world["weak_at_report_myr"], 6),
            "strength_mean_strong": round(world["strength_mean_strong"], 6),
            "solver_cycles_mean": round(
                float(np.mean(world["solver_cycles"])), 3),
            "solver_residual_max": float(max(world["solver_residual"])),
            "exhausted_steps": world["exhausted_steps"],
            "seconds": world["seconds"],
        }
        for world in bundle.worlds
    ]
    plate_counts = [row["plate_count"] for row in worlds]
    return {
        "dials": dict(bundle.controls),
        "pixels": bundle.pixels,
        "history_n": bundle.worlds[0]["history_n"],
        "steps": bundle.worlds[0]["steps"],
        "yield_strain_per_myr": [world["yield_strain_per_myr"]
                                 for world in bundle.worlds],
        "parallel": bundle.parallel,
        "generation_seconds": bundle.elapsed_s,
        "worlds": worlds,
        "summary": {
            "plate_count_min": min(plate_counts),
            "plate_count_max": max(plate_counts),
            "weak_final_mean": round(
                float(np.mean([row["weak_final"] for row in worlds])), 6),
            "stable_count": sum(1 for world in bundle.worlds
                                if _is_stable(world)),
        },
        "stable_count_note": (
            f"stable_count screens each world for {STABLE_PLATE_COUNT[0]} to "
            f"{STABLE_PLATE_COUNT[1]} plates, a final weak fraction between "
            f"{STABLE_WEAK_FINAL[0]} and {STABLE_WEAK_FINAL[1]}, and a peak "
            f"under {STABLE_PEAK_RATIO} times the final. It is a screening "
            "number for the person at the dials, not a gate and not an "
            "approval."
        ),
        "dials_note": (
            "These are development instruments, not author controls. The "
            "percentile yield is a calibration convenience and does not "
            "survive into production."
        ),
        "weak_threshold": WEAK_THRESHOLD,
        "contains": "emergent plate labels, boundaries, and kinematic fields only",
        "does_not_contain": (
            "crust, elevation, water, coastline, islands, land, or a map"),
    }


__all__ = [
    "MAX_SEEDS",
    "VIEWS",
    "Bundle",
    "generate",
    "meta",
    "render_png",
    "report",
    "sheet",
    "views",
]
