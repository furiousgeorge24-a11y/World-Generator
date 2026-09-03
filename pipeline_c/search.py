"""The regime search: sample the dial space, measure, screen, keep candidates.

Two sweeps over two dials found no stable regime and there are seven more
dials. This module searches them. It is a library: importable without Flask,
deterministic given a search seed, and it adds nothing to the engine. Every
world is one task on the exploration lab's own process-pool worker.

Three things it will not do.

- **The screen is physics, not aesthetics.** Every term is a measured property
  of a plate regime: how much of the lithosphere failed, whether the failed
  set stopped moving, how many coherent regions are left, whether the failed
  set is one network or scattered, and how thin it is. Nothing in it says how
  a field looks, and nothing in it is a comparison to any image. Under the
  seam formulation the width term is satisfied by construction, so the
  question the search is left with is plate count and settling.
- **An unconverged cell is not a result.** A cell whose worst solver residual
  exceeds tolerance is marked invalid and never scored: its velocity fields
  were not solved, so its plate counts and weak fractions are readings off an
  unfinished iterate.
- **A passing cell is a candidate, not an approval.** Eight seeds screen and
  twelve confirm; what comes out the far end is a setting for the author to
  look at in the exploration lab.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import threading
import time

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

import explore_adapter  # noqa: E402
from engine.geometry import WorldGeometry  # noqa: E402
from engine.history.kinematics import HistoryParams  # noqa: E402
from engine.history.plates import (  # noqa: E402
    NEIGHBOURS_4,
    label_components,
    weak_mask,
)
from explore_worker import run_one_world  # noqa: E402

#: The twelve development seeds of `STATUS.md`, in order. Stage 3 runs on all
#: of them; a cell that passes there is a finding.
DEVELOPMENT_SEEDS = (
    2075014389, 2477733044, 476149591, 151640007, 2697441485, 1504571935,
    548870008, 2157195430, 4108373596, 4287772760, 287488203, 1833546021,
)

#: The seed the earlier sweeps and audits used, and the default first seed of
#: a stage-1 or stage-2 cell.
BASE_SEED = 4287772760

SEED_MODULUS = 2**32

#: Sheets kept for every cell. Stage-3 cells and findings keep every view.
#: `stress` joined the two under `WORK_ORDER_C04.md` §7.3: under the seam
#: formulation the stress field is what decides where a crack starts and
#: which way a tip runs, so it is the sheet to look at beside the plates when
#: a cell did not do what was expected.
CELL_SHEETS = ("plates", "stress", "trajectory")

#: Every view the exploration lab can draw, which is what a stage-3 cell gets.
ALL_SHEETS = tuple(explore_adapter.VIEWS)

#: A per-term normalized violation is clamped here so a cell whose weak set
#: vanished (peak ratio divided by nothing) still sorts against the rest
#: instead of poisoning the ordering with an infinity.
VIOLATION_CAP = 1e3
# Finite stand-in for "the weak set healed away entirely"; see world_metrics.
PEAK_RATIO_CAP = 1e6

#: Cells submitted to the pool at once, so results arrive continuously rather
#: than in stage-sized lumps. It is the control run's value, so pressing Start
#: with no edits reproduces the run this space is paired against.
CELL_WINDOW = 14


# --------------------------------------------------------------------------
# The screen, the space, and the stages
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Screen:
    """What "a plate regime" means, as six measured terms plus two gates.

    Six terms decide whether one world passes. `residual_max` is not a term:
    it decides whether the world was solved at all. `pass_fraction` is not a
    term either: it decides how many of a cell's worlds must pass.
    """

    weak_min: float = 0.02
    weak_max: float = 0.25
    peak_ratio_max: float = 1.5
    flat_window_myr: float = 100.0
    flat_tolerance: float = 0.03
    plates_min: int = 3
    plates_max: int = 8
    network_share_min: float = 0.5
    edge_fraction_min: float = 0.5
    residual_max: float = 1e-3
    pass_fraction: float = 1.0


@dataclass(frozen=True, slots=True)
class Space:
    """The dial ranges the search samples, and the settings it holds fixed."""

    # The corner. Run `20260902T170740Z-s3` sampled the whole space at the
    # work law: of 1460 cells, seven had a weak fraction of 0.30 or less and
    # three or more plates, and those seven share a setting the run's medians
    # do not — healing in about 27 Myr against a median of 95, damage in
    # about 1.4 Myr against 6.6, a yield percentile near 9 against 3.4, the
    # coarsest drive — one mantle cell across the parent, 10,240 km at the
    # run's own size — in five of seven, and stiffness exponent 2 in five of
    # seven. Their edge fraction reaches 0.25 against the run's in-band
    # median of 0.20. These ranges are that corner, opened downward on both
    # times to the engine's own floors, because the question the corner is
    # sampled to answer is whether zone width keeps narrowing as damage and
    # healing both get faster, or bottoms out near 0.25. Restore the full
    # space from `SEARCH.md` to search broadly again.
    stiffness_fraction_lo: float = 0.08
    stiffness_fraction_hi: float = 0.5
    yield_percentile_lo: float = 2.0
    yield_percentile_hi: float = 15.0
    #: 20, not 5: C04.4 measured that a cut loop reopens when one seam
    #: cell heals, and at the sheet's corner value of 10 Myr a locked cell
    #: seals in two steps. A plate boundary is a suture that persists.
    heal_time_myr_lo: float = 20.0
    #: Healing is the one range this order widened, to 200 Myr. The corner's
    #: 60 was the sheet's question — how fast a diffuse zone must seal to stay
    #: narrow — and under the seam formulation the question is a different
    #: one: how long a fault that has stopped slipping takes to close, which
    #: is what decides whether a crack is still there when the next one
    #: reaches it. Nothing else in the corner moved.
    heal_time_myr_hi: float = 200.0
    damage_time_myr_lo: float = 0.5
    damage_time_myr_hi: float = 5.0
    strength_exponent_set: tuple[int, ...] = (2, 3)
    strength_spread_lo: float = 0.0
    strength_spread_hi: float = 0.1
    #: The coarsest mantle wavelength in kilometres, sampled log-uniform. The
    #: corner's old set `{1, 2}` was a node count at 1024 px and 5 km/px, so
    #: it named the parent and half the parent: 10,240 km and 5,120 km.
    #: 2560, not 5120: at 512 px the parent is 5,120 km, so the corner's
    #: range held at most one mantle cell across the world. Opening it to
    #: two cells lets the search see worlds with more than one drive cell.
    drive_wavelength_km_lo: float = 2560.0
    drive_wavelength_km_hi: float = 10240.0
    drive_shear_lo: float = 0.0
    drive_shear_hi: float = 1.0
    #: How fast a crack tip runs, in kilometres per million years, sampled
    #: log-uniform. Read only under the seam formulation.
    crack_speed_km_per_myr_lo: float = 10.0
    crack_speed_km_per_myr_hi: float = 200.0
    #: New cracks per step, sampled uniformly from this set.
    nucleations_per_step_set: tuple[int, ...] = (1, 2, 4)
    #: Fracture toughness as a fraction of the intact strength, sampled
    #: log-uniform. `WORK_ORDER_C04.md` §2.4 wrote the tip threshold with no
    #: coefficient, which fixed this at 1.0; `WORK_ORDER_C04_4.md` §2 made it
    #: a dial, and the range opens it downward by a decade. Read only under
    #: the seam formulation.
    toughness_fraction_lo: float = 0.1
    toughness_fraction_hi: float = 1.0

    #: 512 px on the full-grid solve, not 1024 on the half grid: C04.4
    #: measured that the stress concentration at a crack tip appears only
    #: when the solve grid is the kinematic grid, and that at 1024 px the
    #: full-grid solve costs 58 s a world while at 512 px it costs what the
    #: 1024-px half-grid search cost. Since C03.10 a 512-px world is the
    #: same physics on a smaller parent. `max_cycles` 80 because six of
    #: forty probe cells were invalid at 40 on the full grid.
    pixels: int = 512
    scale_km: int = 5
    history_myr: float = 300.0
    max_cycles: int = 80
    #: Which damage law every cell of the run uses. Fixed, never sampled:
    #: the Latin hypercube is then identical for a given search seed, so a
    #: run at 1 samples exactly the cells a run at 0 sampled, cell for cell,
    #: and the two runs are an ablation pair. The default is **0**, the
    #: slip-rate law, because under `seams` that is the law that keeps a
    #: slipping fault weak; at 1 an open seam dissipates almost nothing, so
    #: it heals shut, which is what C04 measured. The engine's own default is
    #: 0 too and production is untouched.
    work_damage: int = 0
    #: Kinematic cells per solve cell, fixed and never sampled for the same
    #: reason `work_damage` is: the Latin hypercube is then identical for a
    #: given search seed, so a run at 1 visits exactly the cells a run at 2
    #: visited. 2 solves the velocity on half the kinematic grid and lifts
    #: strain back in 2 x 2 blocks, so a zone cannot be narrower than two
    #: kinematic cells; 1 solves on the full grid at about six times the cost.
    solve_divisor: int = 1
    #: Which damage rule every cell of the run uses, fixed and never sampled
    #: for the same reason `work_damage` and `solve_divisor` are: the Latin
    #: hypercube is then identical for a given search seed, so a run at 1
    #: samples exactly the cells a run at 2 sampled and the two are an
    #: ablation pair. The default is **2**, the block model of `DESIGN.md`
    #: §3.6's last paragraph: pieces are rigid bodies, the stress the seam
    #: rules read is the integral of the drag a piece failed to match, and
    #: seams are carried on markers that cannot duplicate. 1 is the same seam
    #: rules on the sheet's velocity solve, which C04.1 measured; 0 is the
    #: sheet. The engine's own default is 0 and production is untouched.
    seams: int = 2
    base_seed: int = BASE_SEED

    def bounds(self, name: str) -> tuple[float, float]:
        return (float(getattr(self, f"{name}_lo")),
                float(getattr(self, f"{name}_hi")))

    def values(self, name: str) -> tuple[int, ...]:
        return tuple(int(value) for value in getattr(self, f"{name}_set"))


@dataclass(frozen=True, slots=True)
class Stages:
    """How many cells each stage runs, and on how many seeds."""

    stage1_cells: int = 200
    stage1_seeds: int = 4
    #: 20 and 8, not 10 and 5: under the block model the probe's best cells
    #: sit on a gradient (plate count moves with the dials), so stage 2's
    #: local refinement is worth more cells than it was on the sheet.
    stage2_top: int = 20
    stage2_perturbations: int = 8
    stage2_seeds: int = 8
    stage3_top: int = 3


@dataclass(frozen=True, slots=True)
class SearchConfig:
    screen: Screen = Screen()
    space: Space = Space()
    stages: Stages = Stages()
    #: The control run's search seed, so the default space draws the same
    #: Latin hypercube the control drew and the two runs pair cell for cell.
    #: A seed the search has not drawn on before, so the corner is a
    #: fresh sample rather than a redraw of a run already on disk.
    #: 12: seed 11 was spent on the five 40-cell probes of C04 – C04.4.
    #: Round `r` of a run draws from `search_seed + r`, so a run that
    #: restarted blindly has spent a range of seeds, not one.
    #: 17: the overnight run of 2026-09-03 spent 12 – 15 and the afternoon
    #: run, started at 13, spent 13 – 16 and re-sampled three of its rounds.
    search_seed: int = 17
    window: int = CELL_WINDOW

    def to_json(self) -> dict:
        return {
            "screen": asdict(self.screen),
            "space": {key: (list(value) if isinstance(value, tuple) else value)
                      for key, value in asdict(self.space).items()},
            "stages": asdict(self.stages),
            "search_seed": int(self.search_seed),
            "window": int(self.window),
            "development_seeds": list(DEVELOPMENT_SEEDS),
        }


#: The dials the search samples, in the order §2.3 lists them, which is also
#: the column order of the Latin hypercube. `kind` says how a range is
#: sampled: `log` spreads samples evenly in the logarithm, `linear` evenly in
#: the value, `set` picks uniformly from a finite set.
DIALS: tuple[tuple[str, str], ...] = (
    ("stiffness_fraction", "log"),
    ("yield_percentile", "log"),
    ("heal_time_myr", "log"),
    ("damage_time_myr", "log"),
    ("strength_exponent", "set"),
    ("strength_spread", "linear"),
    ("drive_wavelength_km", "log"),
    ("drive_shear", "linear"),
    ("crack_speed_km_per_myr", "log"),
    ("nucleations_per_step", "set"),
    # Appended by `WORK_ORDER_C04_4.md` §2. A Latin hypercube draws its
    # columns in order from one generator, so a column added at the end
    # leaves every earlier column's draw untouched.
    ("toughness_fraction", "log"),
)

DIAL_NAMES = tuple(name for name, _kind in DIALS)
CONTINUOUS_DIALS = tuple((name, kind) for name, kind in DIALS if kind != "set")
SET_DIALS = tuple(name for name, kind in DIALS if kind == "set")


def _from_unit(space: Space, name: str, kind: str, u: float):
    if kind == "set":
        values = space.values(name)
        index = min(int(u * len(values)), len(values) - 1)
        return int(values[index])
    lo, hi = space.bounds(name)
    if kind == "log":
        if lo <= 0.0:
            raise ValueError(f"{name} is sampled log-uniform and needs lo > 0")
        return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))
    return float(lo + u * (hi - lo))


def latin_hypercube(space: Space, count: int,
                    rng: np.random.Generator) -> list[dict]:
    """`count` samples, one per stratum per dial, in the dial order of §2.3.

    A Latin hypercube, not a uniform draw: each dial's range is cut into
    `count` equal strata (equal in the logarithm for a log dial) and each
    stratum is used exactly once, so a hundred cells cannot leave a decade of
    heal time unvisited the way independent draws can.
    """
    if count < 1:
        raise ValueError("count must be positive")
    unit = np.empty((count, len(DIALS)), dtype=np.float64)
    for column in range(len(DIALS)):
        order = rng.permutation(count)
        unit[:, column] = (order + rng.random(count)) / count
    return [
        {name: _from_unit(space, name, kind, float(unit[row, column]))
         for column, (name, kind) in enumerate(DIALS)}
        for row in range(count)
    ]


def perturb(dials: dict, space: Space, rng: np.random.Generator,
            *, scale: float = 0.1,
            discrete_probability: float = 0.2) -> dict:
    """One Gaussian step of `scale` of each range's width, clipped to it.

    Continuous dials move by a normal draw of a tenth of their range's width,
    taken in the logarithm for a log dial so a step near the bottom of the
    range is proportionally the same size as one near the top. A discrete dial
    is resampled outright with probability `discrete_probability`.
    """
    out = dict(dials)
    for name, kind in CONTINUOUS_DIALS:
        lo, hi = space.bounds(name)
        if kind == "log":
            width = math.log(hi) - math.log(lo)
            moved = math.log(float(dials[name])) + rng.normal(0.0, scale * width)
            value = math.exp(moved)
        else:
            width = hi - lo
            value = float(dials[name]) + rng.normal(0.0, scale * width)
        out[name] = float(min(max(value, lo), hi))
    for name in SET_DIALS:
        values = space.values(name)
        if rng.random() < discrete_probability:
            out[name] = int(values[int(rng.integers(len(values)))])
        else:
            out[name] = int(dials[name])
    return out


#: What a dial that a cell predates is filled with on read: the engine's own
#: default, which is the setting the run that wrote the cell was using,
#: because the dial did not exist and the engine's default is what ran.
LEGACY_DIAL_DEFAULTS = {
    "seams": int(HistoryParams().seams),
    "crack_speed_km_per_myr": float(HistoryParams().crack_speed_km_per_myr),
    "nucleations_per_step": int(HistoryParams().nucleations_per_step),
    "toughness_fraction": float(HistoryParams().toughness_fraction),
}


def modernize_dials(dials: dict, pixels: int, scale_km: int) -> dict:
    """A cell's dials as the current engine reads them.

    Two translations, both read-time and neither a validator.

    **`drive_nodes`.** Every run written before `WORK_ORDER_C03_10.md`
    recorded a count of the coarsest mantle wavelength's cycles across the
    parent world. The engine now takes that wavelength in kilometres, so a
    legacy dial is read back through the geometry of the run that wrote it:
    `parent_km / drive_nodes`. At 1024 px and 5 km/px the parent is 10,240 km
    and the conversion is exact, so a rerun reproduces the logged metrics.

    **The seam dials.** Every run written before `WORK_ORDER_C04.md` has no
    `seams`, `crack_speed_km_per_myr` or `nucleations_per_step` in it, and
    every run written before `WORK_ORDER_C04_4.md` has no
    `toughness_fraction`. They are filled with the engine's own defaults,
    which is what those runs ran on — 1.0 for the toughness, the constant the
    tip rule carried implicitly before it was a dial — so a legacy cell stays
    runnable and pairs by dial value against a new one. `seams` itself is a
    fixed setting of the space and not a sampled dial, so `params_of` takes
    it from the space; it is filled here so the record of a cell is complete
    however it is read.

    A dial set that already carries everything is returned unchanged.
    """
    missing = [name for name in LEGACY_DIAL_DEFAULTS if name not in dials]
    legacy_drive = "drive_nodes" in dials and "drive_wavelength_km" not in dials
    if not missing and not legacy_drive:
        return dials
    out = {name: value for name, value in dials.items()
           if not (legacy_drive and name == "drive_nodes")}
    if legacy_drive:
        nodes = float(dials["drive_nodes"])
        if nodes <= 0.0:
            raise ValueError("drive_nodes must be positive to convert")
        parent_km = WorldGeometry(0, int(pixels), int(scale_km)).parent_km
        out["drive_wavelength_km"] = float(parent_km) / nodes
    for name in missing:
        out[name] = LEGACY_DIAL_DEFAULTS[name]
    return out


def params_of(dials: dict, space: Space) -> HistoryParams:
    """The `HistoryParams` a cell's dials and the space's fixed settings make.

    The dials are modernized first, through the space's own geometry, so a
    cell logged before the drive moved into kilometres is still runnable.
    """
    dials = modernize_dials(dials, space.pixels, space.scale_km)
    return HistoryParams(
        stiffness_fraction=float(dials["stiffness_fraction"]),
        yield_percentile=float(dials["yield_percentile"]),
        heal_time_myr=float(dials["heal_time_myr"]),
        damage_time_myr=float(dials["damage_time_myr"]),
        strength_exponent=int(dials["strength_exponent"]),
        strength_spread=float(dials["strength_spread"]),
        drive_wavelength_km=float(dials["drive_wavelength_km"]),
        drive_shear=float(dials["drive_shear"]),
        crack_speed_km_per_myr=float(dials["crack_speed_km_per_myr"]),
        nucleations_per_step=int(dials["nucleations_per_step"]),
        toughness_fraction=float(dials["toughness_fraction"]),
        work_damage=int(space.work_damage),
        seams=int(space.seams),
        history_myr=float(space.history_myr),
        max_cycles=int(space.max_cycles),
        solve_divisor=int(space.solve_divisor),
    )


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def network_share(weak: np.ndarray) -> float:
    """The largest 8-connected component of the weak set, as a share of it.

    One long branching network of failure gives a share near one; the same
    number of weak cells scattered over the sheet gives a small one. Zero when
    nothing is weak.
    """
    weak = np.asarray(weak, dtype=bool)
    total = int(weak.sum())
    if total == 0:
        return 0.0
    labels = label_components(weak, 8)
    # `label_components` renumbers by area, largest first, so 0 is the largest.
    return float(int((labels == 0).sum()) / total)


def edge_fraction(weak: np.ndarray) -> float:
    """Weak cells with a strong 4-neighbour, as a share of all weak cells.

    A line of weak cells `w` wide has about `2 / w` of its cells on an edge,
    so this reads as the reciprocal half-width of the weak set: 1.0 for a
    one-cell line, 0.5 for four cells across, and small for a filled region.
    Zero when nothing is weak.
    """
    weak = np.asarray(weak, dtype=bool)
    total = int(weak.sum())
    if total == 0:
        return 0.0
    strong = ~weak
    touching = np.zeros_like(weak)
    for offset in NEIGHBOURS_4:
        touching |= np.roll(strong, offset, axis=(-2, -1))
    return float(int((weak & touching).sum()) / total)


def world_metrics(world: dict, screen: Screen) -> dict:
    """The seven numbers of §2.1, read off one worker result."""
    fractions = [float(value) for value in world["weak_fraction"]]
    step_myr = float(world["step_myr"])
    weak_final = fractions[-1]
    weak_peak = max(fractions)
    back = int(round(screen.flat_window_myr / step_myr)) if step_myr > 0 else 0
    index = max(0, len(fractions) - 1 - back)
    weak = weak_mask(world["strength_final"])
    # Every metric must be finite: JSON has no infinity, and the page parses
    # the whole cells response at once, so one non-finite number blanks the
    # gallery. A weak set that peaked and then healed to nothing is capped
    # rather than infinite; the screen's violation clamp treats both alike.
    if weak_final > 0.0:
        peak_ratio = min(weak_peak / weak_final, PEAK_RATIO_CAP)
    elif weak_peak > 0.0:
        peak_ratio = PEAK_RATIO_CAP
    else:
        peak_ratio = 1.0
    return {
        "seed": int(world["seed"]),
        "weak_final": weak_final,
        "weak_peak": weak_peak,
        "weak_drift": abs(weak_final - fractions[index]),
        "peak_ratio": peak_ratio,
        "plate_count": len(world["plate_percent"]),
        "network_share": network_share(weak),
        "edge_fraction": edge_fraction(weak),
        "residual_max": float(max(world["solver_residual"])),
        "seconds": float(world["seconds"]),
    }


def term_bounds(screen: Screen) -> tuple[tuple[str, float, float], ...]:
    """The six terms as closed intervals, in report order."""
    return (
        ("weak_final", screen.weak_min, screen.weak_max),
        ("peak_ratio", 1.0, screen.peak_ratio_max),
        ("weak_drift", 0.0, screen.flat_tolerance),
        ("plate_count", float(screen.plates_min), float(screen.plates_max)),
        ("network_share", screen.network_share_min, 1.0),
        ("edge_fraction", screen.edge_fraction_min, 1.0),
    )


def screen_world(metrics: dict, screen: Screen) -> dict:
    """Per-term pass and normalized violation for one world.

    A violation is how far outside its interval a value sits, divided by the
    interval's width, so the six terms are commensurable and can be summed.
    Inside the interval it is zero.
    """
    terms = {}
    total = 0.0
    for name, lo, hi in term_bounds(screen):
        value = float(metrics[name])
        width = hi - lo
        if width <= 0.0:
            width = max(abs(hi), 1.0)
        if math.isnan(value):
            outside = VIOLATION_CAP * width
        else:
            outside = max(lo - value, value - hi, 0.0)
        violation = min(outside / width, VIOLATION_CAP)
        terms[name] = {"value": value, "lo": lo, "hi": hi,
                       "ok": violation <= 0.0, "violation": violation}
        total += violation
    return {"terms": terms,
            "passed": all(term["ok"] for term in terms.values()),
            "violation": total}


def screen_cell(metrics_list: list[dict], screen: Screen) -> dict:
    """Invalid, passed, and the soft score, for one cell's worlds.

    Invalid comes first and is absolute: one world above the residual
    tolerance and the cell has no result to screen. Otherwise the cell passes
    when at least `pass_fraction` of its worlds pass every term. The soft
    score is the mean per-world sum of violations; it orders cells that do not
    pass and contains no aesthetic term.
    """
    verdicts = [screen_world(metrics, screen) for metrics in metrics_list]
    invalid = any(metrics["residual_max"] > screen.residual_max
                  for metrics in metrics_list)
    passed_count = sum(1 for verdict in verdicts if verdict["passed"])
    share = passed_count / len(verdicts) if verdicts else 0.0
    soft = (sum(verdict["violation"] for verdict in verdicts) / len(verdicts)
            if verdicts else float("inf"))
    return {
        "verdicts": verdicts,
        "invalid": invalid,
        "pass_count": passed_count,
        "passed": (not invalid) and share >= screen.pass_fraction - 1e-12,
        "soft_score": soft,
    }


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


class SequentialExecutor:
    """An executor that runs each task on the calling thread.

    The tests and any Flask-free use of this module take this instead of a
    process pool: the search only ever asks an executor for `submit` and a
    future, so the two are interchangeable and the arithmetic is identical.
    """

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:               # noqa: BLE001 - re-raised
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True) -> None:
        return None


def run_id_for(search_seed: int, *, when: float | None = None) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ",
                          time.gmtime(time.time() if when is None else when))
    return f"{stamp}-s{int(search_seed)}"


def runs_root() -> Path:
    return _HERE / "out" / "search"


class SearchRun:
    """One search: stages, cells, sheets on disk, and a stop that is polite.

    The caller owns the executor. `run()` blocks until the search finds a
    stopped; the server calls it on a background thread.
    """

    def __init__(self, config: SearchConfig, out_dir: Path, executor,
                 *, on_cell=None) -> None:
        self.config = config
        self.out_dir = Path(out_dir)
        self.executor = executor
        self.on_cell = on_cell
        self.run_id = self.out_dir.name
        self.cells: list[dict] = []
        self.stage = "starting"
        self.round = 0
        #: Every stage-3 passer so far, in the order found. A finding does
        #: not stop the run; the author asked for the search to go on.
        self.findings: list[dict] = []
        self.error: str | None = None
        self.started_s = time.perf_counter()
        #: The clock stops when the run does, so a finished run keeps the
        #: elapsed time and rate it ended with instead of decaying on screen.
        self.stopped_s: float | None = None
        self.finished = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._next_index = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "config.json").write_text(
            json.dumps(self.config.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    # -- control -----------------------------------------------------------

    def stop(self) -> None:
        """Ask the run to stop once the cells already in flight finish."""
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def status(self) -> dict:
        with self._lock:
            cells = list(self.cells)
        done = len(cells)
        passers = sum(1 for cell in cells if cell["passed"])
        invalid = sum(1 for cell in cells if cell["invalid"])
        scores = [cell["soft_score"] for cell in cells if not cell["invalid"]]
        end = self.stopped_s if self.stopped_s is not None else time.perf_counter()
        elapsed = end - self.started_s
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "round": self.round,
            "search_seed": self.config.search_seed + self.round,
            "cells_done": done,
            "passers": passers,
            "invalid": invalid,
            "cells_per_minute": round(done / elapsed * 60.0, 3) if elapsed else 0.0,
            "best_soft_score": (round(min(scores), 6) if scores else None),
            "elapsed_s": round(elapsed, 1),
            "running": not self.finished,
            "stopping": self._stop.is_set(),
            "finding": (self.findings[-1]["id"] if self.findings else None),
            "findings": len(self.findings),
            "error": self.error,
        }

    # -- one cell ----------------------------------------------------------

    def _cell_dir(self, cell_id: str) -> Path:
        return self.out_dir / "cells" / cell_id

    def _write_sheets(self, cell_id: str, dials: dict, worlds: list[dict],
                      sheets: tuple[str, ...]) -> list[str]:
        bundle = explore_adapter.Bundle(
            seed=int(worlds[0]["seed"]),
            pixels=int(self.config.space.pixels),
            scale_km=int(self.config.space.scale_km),
            controls=dict(dials),
            params=params_of(dials, self.config.space),
            worlds=worlds,
            parallel=True,
            elapsed_s=0.0,
        )
        directory = self._cell_dir(cell_id)
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        for view in sheets:
            (directory / f"{view}.png").write_bytes(
                explore_adapter.render_png(bundle, view))
            written.append(view)
        return written

    def _finish_cell(self, stage: int, dials: dict, seeds: list[int],
                     worlds: list[dict], seconds: float) -> dict:
        screen = self.config.screen
        metrics = [world_metrics(world, screen) for world in worlds]
        verdict = screen_cell(metrics, screen)
        index = self._next_index
        self._next_index += 1
        cell_id = f"c{index:05d}"
        sheets = tuple(explore_adapter.VIEWS) if stage == 3 else CELL_SHEETS
        written = self._write_sheets(cell_id, dials, worlds, sheets)
        cell = {
            "id": cell_id,
            "index": index,
            "stage": stage,
            "round": self.round,
            "dials": {name: dials[name] for name in DIAL_NAMES},
            "seeds": [int(seed) for seed in seeds],
            "worlds": [
                {**metric,
                 "passed": verdict["verdicts"][position]["passed"],
                 "terms": {name: term["ok"] for name, term
                           in verdict["verdicts"][position]["terms"].items()}}
                for position, metric in enumerate(metrics)
            ],
            "terms": {
                name: all(v["terms"][name]["ok"] for v in verdict["verdicts"])
                for name, _lo, _hi in term_bounds(screen)
            },
            "pass_count": verdict["pass_count"],
            "passed": verdict["passed"],
            "invalid": verdict["invalid"],
            "soft_score": (None if math.isinf(verdict["soft_score"])
                           else round(verdict["soft_score"], 6)),
            "seconds": round(seconds, 3),
            "world_seconds_mean": round(
                sum(metric["seconds"] for metric in metrics) / len(metrics), 3),
            "sheets": written,
            "finding": bool(stage == 3 and verdict["passed"]),
        }
        with self._lock:
            self.cells.append(cell)
            with (self.out_dir / "cells.jsonl").open("a", encoding="utf-8") as fh:
                # allow_nan=False: a non-finite number is a bug here, and it
                # is better to fail the write than to poison the gallery.
                fh.write(json.dumps(cell, sort_keys=True, allow_nan=False) + "\n")
        if cell["finding"]:
            self.findings.append(cell)
        if self.on_cell is not None:
            self.on_cell(cell)
        return cell

    # -- a batch of cells --------------------------------------------------

    def _run_cells(self, stage: int, dial_sets: list[dict],
                   seeds: list[int]) -> list[dict]:
        """Submit cells in a rolling window so results arrive continuously."""
        space = self.config.space
        queue = deque(dial_sets)
        inflight: deque = deque()
        done: list[dict] = []
        window = max(1, int(self.config.window))
        while queue or inflight:
            if self._stop.is_set():
                queue.clear()
            while queue and len(inflight) < window:
                dials = queue.popleft()
                record = params_of(dials, space).to_record()
                futures = [
                    self.executor.submit(run_one_world, int(seed),
                                         int(space.pixels), int(space.scale_km),
                                         record)
                    for seed in seeds
                ]
                inflight.append((dials, futures, time.perf_counter()))
            if not inflight:
                break
            dials, futures, started = inflight.popleft()
            worlds = [future.result() for future in futures]
            done.append(self._finish_cell(
                stage, dials, seeds, worlds, time.perf_counter() - started))
        return done

    # -- the stages --------------------------------------------------------

    def _seeds(self, count: int) -> list[int]:
        base = int(self.config.space.base_seed)
        return [(base + offset) % SEED_MODULUS for offset in range(count)]

    @staticmethod
    def _by_soft(cells: list[dict]) -> list[dict]:
        scored = [cell for cell in cells
                  if not cell["invalid"] and cell["soft_score"] is not None]
        return sorted(scored, key=lambda cell: (cell["soft_score"],
                                                cell["index"]))

    def run(self) -> None:
        try:
            self._run()
        except BaseException as exc:                # noqa: BLE001 - reported
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.stopped_s = time.perf_counter()
            self.finished = True
            if self.stage != "error":
                self.stage = "stopped" if self._stop.is_set() else "done"
            if self.error is not None:
                self.stage = "error"

    def _run(self) -> None:
        stages = self.config.stages
        while not self._stop.is_set():
            rng = np.random.default_rng(self.config.search_seed + self.round)

            self.stage = "stage1"
            first = self._run_cells(
                1, latin_hypercube(self.config.space, stages.stage1_cells, rng),
                self._seeds(stages.stage1_seeds))
            if self._stop.is_set():
                return

            self.stage = "stage2"
            candidates = [cell for cell in first if cell["passed"]]
            seen = {cell["id"] for cell in candidates}
            for cell in self._by_soft(first)[:stages.stage2_top]:
                if cell["id"] not in seen:
                    candidates.append(cell)
                    seen.add(cell["id"])
            refine: list[dict] = []
            for cell in candidates:
                refine.append(dict(cell["dials"]))
                for _ in range(stages.stage2_perturbations):
                    refine.append(perturb(cell["dials"], self.config.space, rng))
            second = self._run_cells(2, refine, self._seeds(stages.stage2_seeds))
            if self._stop.is_set():
                return

            self.stage = "stage3"
            confirm = [cell for cell in second if cell["passed"]]
            if not confirm:
                confirm = self._by_soft(second)[:stages.stage3_top]
            third = self._run_cells(
                3, [dict(cell["dials"]) for cell in confirm],
                list(DEVELOPMENT_SEEDS))
            # A finding is recorded, pinned by the page, and the run goes
            # on: more findings are more candidates for the author.
            if self._stop.is_set():
                return
            self.round += 1


# --------------------------------------------------------------------------
# Reading a finished run back off disk
# --------------------------------------------------------------------------


def load_cells(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "cells.jsonl"
    if not path.exists():
        return []
    cells = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cells.append(json.loads(line))
    return cells


def list_runs(root: Path | None = None) -> list[dict]:
    """Previous runs on disk, newest first."""
    root = runs_root() if root is None else Path(root)
    if not root.exists():
        return []
    rows = []
    for directory in sorted(root.iterdir(), reverse=True):
        if not directory.is_dir():
            continue
        config_path = directory / "config.json"
        cells = load_cells(directory)
        rows.append({
            "run_id": directory.name,
            "cells": len(cells),
            "passers": sum(1 for cell in cells if cell["passed"]),
            "invalid": sum(1 for cell in cells if cell["invalid"]),
            "findings": sum(1 for cell in cells if cell.get("finding")),
            "has_config": config_path.exists(),
        })
    return rows


__all__ = [
    "ALL_SHEETS",
    "BASE_SEED",
    "CELL_SHEETS",
    "CELL_WINDOW",
    "DEVELOPMENT_SEEDS",
    "DIALS",
    "DIAL_NAMES",
    "Screen",
    "SearchConfig",
    "SearchRun",
    "SequentialExecutor",
    "Space",
    "Stages",
    "edge_fraction",
    "latin_hypercube",
    "list_runs",
    "load_cells",
    "modernize_dials",
    "network_share",
    "params_of",
    "perturb",
    "run_id_for",
    "runs_root",
    "screen_cell",
    "screen_world",
    "term_bounds",
    "world_metrics",
]
