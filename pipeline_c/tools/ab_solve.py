"""Rerun a search run's cells on both solve grids and compare them cell for cell.

`SOLVE_GRID_DIVISOR = 2` solves the velocity on half the kinematic grid and
lifts the strain back as 2 x 2 blocks, so damage happens in units of one solve
cell and a weak zone cannot be narrower than two kinematic cells. This tool
takes cells from a finished search run, reruns each of them on its own seeds at
`solve_divisor = 2` and then at `1`, and reports what moved.

    py -3.14 pipeline_c/tools/ab_solve.py <run_id> [--cells 20] [--pixels 1024]
                                          [--max-cycles 80] [--tag NAME]
                                          [--divisors 2 1]

`--divisors` names the variants to run, in report order; giving one runs one
variant and the paired sections say so rather than inventing a partner.

The page goes to stdout and to `out/ab_solve_<run_id>.md`; the sheets go to
`out/ab_solve/<run_id>/`. It describes nothing and proposes nothing: every row
is a count, a median, a share or a second.

**The determinism gate.** When the rerun is at the run's own pixel size, the
`solve_divisor = 2` variant must reproduce every metric the run logged for
every world to 1e-9. It is the same engine on the same seeds at the same dials,
so anything else is a bug in this tool or a change in the engine, and the tool
stops rather than compare against a moved baseline. At any other pixel size the
worlds are different worlds and the gate does not apply.

**Legacy dials are modernized on read.** A run written before
`WORK_ORDER_C03_10.md` recorded `drive_nodes`, a fraction of its own parent
world. `search.modernize_dials` turns that into the wavelength in kilometres
it meant at the run's own resolution and scale, so the cells stay rerunnable
and a rerun at the run's own size reproduces its logged metrics exactly.

**Width is measured on the field, not on a PNG.** The final weak mask comes
from the world's own strength field. For k = 2 … 8 the tool reports the share
of weak cells that lie inside at least one fully weak k x k square on the
torus, and the share of weak cells whose even-aligned 2 x 2 block is entirely
weak. A zone whose width is quantized to the solve cell has the share at k = 3
equal to the share at k = 4, and its alignment share is near one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

_PIPELINE_C = Path(__file__).resolve().parents[1]
if str(_PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_C))

import explore_adapter  # noqa: E402
import search  # noqa: E402
from engine.history.plates import weak_mask  # noqa: E402
from explore_worker import run_one_world  # noqa: E402

#: The two variants, in the order they are run and reported.
DIVISORS = (2, 1)

#: Cells taken by their count of worlds that fail on `edge_fraction` alone,
#: before the rest of the selection is filled in by soft score.
TOP_BY_EDGE = 12

#: Square sizes the width shares are read at.
WIDTH_K = (2, 3, 4, 5, 6, 7, 8)

#: How far apart the shares at k = 3 and k = 4 must be for a world to count as
#: having left the even quantization. It is the number `WORK_ORDER_C03_9.md`
#: §0's first prediction names, and it is reported, not applied to anything.
ODD_WIDTH_TOLERANCE = 0.05

#: Metrics compared between the two variants, in report order. The six screen
#: terms, which is every number the screen reads.
PAIRED_METRICS = ("weak_final", "peak_ratio", "weak_drift", "plate_count",
                  "network_share", "edge_fraction")

#: Metrics the determinism gate checks against the run's own record, and the
#: tolerance it checks them to. `seconds` is a clock and is not one of them.
GATE_METRICS = ("weak_final", "weak_peak", "weak_drift", "peak_ratio",
                "plate_count", "network_share", "edge_fraction",
                "residual_max")
GATE_TOLERANCE = 1e-9

#: A paired difference smaller than this counts as neither a rise nor a fall.
SAME_TOL = 1e-12

#: The cycle budget the search ran on. Worlds that need more than this at a
#: finer grid are counted, because a rerun at the search's own budget would
#: have marked them invalid.
OLD_BUDGET = 40

#: Cells whose `strain_rate` and `power` sheets are written as well as their
#: `plates` sheet.
FIELD_SHEET_CELLS = 3

#: What is kept of a worker's result once the world has crossed back into this
#: process: the numbers every measurement reads, plus the labels the `plates`
#: sheet draws. A world at 1024 px carries about six megabytes of fields and
#: two variants of a hundred and fifty worlds are held at once, so the rest is
#: dropped as it arrives rather than at the end.
KEPT_FIELDS = (
    "seed", "history_n", "steps", "step_myr", "history_myr",
    "yield_strain_per_myr", "yield_power", "labels", "strength_final",
    "weak_fraction", "plate_percent", "solver_cycles", "solver_residual",
    "exhausted_steps", "seconds",
)

#: Kept as well for the cells whose `strain_rate` and `power` sheets are drawn.
FIELD_SHEET_FIELDS = ("strain_rate", "power")


class DeterminismError(RuntimeError):
    """The `solve_divisor = 2` rerun did not reproduce the run's own record."""


# --------------------------------------------------------------------------
# The run on disk
# --------------------------------------------------------------------------


def runs_root() -> Path:
    return _PIPELINE_C / "out" / "search"


def resolve_run(name: str) -> Path:
    """A run id under `out/search`, or a path to a run directory."""
    candidate = Path(name)
    if candidate.is_dir() and (candidate / "cells.jsonl").exists():
        return candidate
    directory = runs_root() / name
    if not (directory / "cells.jsonl").exists():
        raise FileNotFoundError(f"no run with cells.jsonl at {directory}")
    return directory


def load_run(directory: Path) -> tuple[list[dict], dict]:
    directory = Path(directory)
    cells = [
        json.loads(line)
        for line in (directory / "cells.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    config_path = directory / "config.json"
    config = (json.loads(config_path.read_text(encoding="utf-8"))
              if config_path.exists() else {})
    return cells, config


def screen_of(config: dict) -> search.Screen:
    """The run's own screen, defaulted for anything it did not record."""
    fields = search.Screen.__dataclass_fields__
    values = {name: value for name, value in config.get("screen", {}).items()
              if name in fields}
    return search.Screen(**values)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


def edge_only_failures(cell: dict, screen: search.Screen) -> int:
    """Worlds of this cell that fail the screen on `edge_fraction` alone.

    Read off the per-world `terms` the run wrote, which are that run's own
    screen applied at the time it ran.
    """
    count = 0
    for world in cell["worlds"]:
        terms = world.get("terms", {})
        if not terms or terms.get("edge_fraction", True):
            continue
        if all(ok for name, ok in terms.items() if name != "edge_fraction"):
            count += 1
    return count


def soft_key(cell: dict) -> float:
    score = cell.get("soft_score")
    return float("inf") if score is None else float(score)


def select_cells(cells: list[dict], count: int, screen: search.Screen,
                 *, top_by_edge: int = TOP_BY_EDGE) -> list[dict]:
    """The selection rule of `WORK_ORDER_C03_9.md` §3.1.

    Rank by the count of worlds failing on `edge_fraction` alone, ties by soft
    score; take the top `top_by_edge`; then add the best cells by soft score
    that are not already taken until there are `count`. Invalid cells are never
    taken: their velocity fields were not solved.
    """
    if count < 1:
        raise ValueError("count must be positive")
    usable = [cell for cell in cells if not cell.get("invalid")]
    ranked = sorted(
        usable,
        key=lambda cell: (-edge_only_failures(cell, screen), soft_key(cell),
                          cell["index"]))
    chosen = ranked[:min(top_by_edge, count)]
    taken = {cell["id"] for cell in chosen}
    by_soft = sorted(
        (cell for cell in usable if cell.get("soft_score") is not None),
        key=lambda cell: (soft_key(cell), cell["index"]))
    for cell in by_soft:
        if len(chosen) >= count:
            break
        if cell["id"] not in taken:
            chosen.append(cell)
            taken.add(cell["id"])
    return chosen


# --------------------------------------------------------------------------
# Width, on the field
# --------------------------------------------------------------------------


def all_weak_squares(weak: np.ndarray, k: int) -> np.ndarray:
    """`out[i, j]` is true when the k x k square with corner (i, j) is weak.

    On the torus, so a square may wrap. Built by folding the mask into itself
    with rolls, which is exact and costs `2 (k - 1)` array operations.
    """
    if k < 1:
        raise ValueError("k must be positive")
    weak = np.asarray(weak, dtype=bool)
    rows = weak.copy()
    for offset in range(1, k):
        rows &= np.roll(weak, -offset, axis=-1)
    out = rows.copy()
    for offset in range(1, k):
        out &= np.roll(rows, -offset, axis=-2)
    return out


def covered_by_square(weak: np.ndarray, k: int) -> np.ndarray:
    """Cells that lie inside at least one fully weak k x k square."""
    squares = all_weak_squares(weak, k)
    covered = np.zeros_like(squares)
    for down in range(k):
        shifted = np.roll(squares, down, axis=-2)
        for right in range(k):
            covered |= np.roll(shifted, right, axis=-1)
    return covered


def aligned_blocks(weak: np.ndarray) -> np.ndarray:
    """Cells whose even-aligned 2 x 2 block is entirely weak.

    The blocks the solve grid at divisor 2 damages in: rows and columns
    `2i, 2i + 1`. A grid with an odd side has no such tiling and gives an
    all-false mask.
    """
    weak = np.asarray(weak, dtype=bool)
    n = weak.shape[-1]
    if weak.shape[-2] != n or n % 2:
        return np.zeros_like(weak)
    blocks = weak.reshape(n // 2, 2, n // 2, 2).all(axis=(1, 3))
    return np.repeat(np.repeat(blocks, 2, axis=-1), 2, axis=-2)


def width_shares(weak: np.ndarray) -> dict | None:
    """Share of the weak set at each width, and the block-alignment share.

    `None` when nothing is weak, because a share of an empty set is not a
    number. Every value is a share of the weak set, so it falls with k.
    """
    weak = np.asarray(weak, dtype=bool)
    total = int(weak.sum())
    if total == 0:
        return None
    shares = {f"k{k}": float(int((covered_by_square(weak, k) & weak).sum())
                             / total)
              for k in WIDTH_K}
    shares["aligned"] = float(int((aligned_blocks(weak) & weak).sum()) / total)
    return shares


# --------------------------------------------------------------------------
# The rerun
# --------------------------------------------------------------------------


def run_geometry(config: dict) -> tuple[int, int]:
    """The pixels and scale the run itself used, defaulted to production's."""
    space = config.get("space", {})
    return int(space.get("pixels", 1024)), int(space.get("scale_km", 5))


def dials_of(cell: dict, config: dict) -> dict:
    """One cell's dials, modernized through the geometry of its own run.

    A legacy `drive_nodes` meant `parent / nodes` at the size the run ran at,
    which is not the size this rerun may be at, so the conversion uses the
    run's own pixels and scale and never the rerun's.
    """
    pixels, scale_km = run_geometry(config)
    return search.modernize_dials(cell["dials"], pixels, scale_km)


def params_record(cell: dict, config: dict, divisor: int, max_cycles: int,
                  ) -> dict:
    """`HistoryParams.to_record()` for one cell at one divisor.

    Every sampled dial comes from the cell; `work_damage` and `history_myr`
    come from the run's own space, so the rerun is the same physics; the
    divisor and the cycle budget come from this tool's flags. The run's own
    pixels and scale go in too, because they are what a legacy `drive_nodes`
    is converted through.
    """
    space = config.get("space", {})
    pixels, scale_km = run_geometry(config)
    return search.params_of(
        cell["dials"],
        search.Space(
            pixels=pixels,
            scale_km=scale_km,
            work_damage=int(space.get("work_damage", 0)),
            seams=int(space.get("seams", 0)),
            history_myr=float(space.get("history_myr", 300.0)),
            max_cycles=int(max_cycles),
            solve_divisor=int(divisor),
        ),
    ).to_record()


def prune(world: dict, fields: tuple[str, ...]) -> dict:
    """What this tool keeps of one worker result. See `KEPT_FIELDS`."""
    return {name: world[name] for name in fields if name in world}


def run_variant(selection: list[dict], config: dict, divisor: int, *,
                pixels: int, scale_km: int, max_cycles: int,
                executor=None) -> tuple[list[list[dict]], float, bool]:
    """Every world of every selected cell at one divisor.

    Returns the worlds cell by cell, the wall seconds, and whether the pool
    was used. The pool is `explore_adapter`'s own eight-worker spawn pool, the
    one the lab and the search run on. Each world is pruned to `KEPT_FIELDS`
    as it arrives, so the fields of three hundred worlds do not have to be
    held at once.
    """
    pool = explore_adapter._pool() if executor is None else executor
    parallel = pool is not None
    started = time.perf_counter()
    tasks: list[list] = []
    for cell in selection:
        record = params_record(cell, config, divisor, max_cycles)
        row = []
        for seed in cell["seeds"]:
            arguments = (int(seed), int(pixels), int(scale_km), record)
            row.append(pool.submit(run_one_world, *arguments) if parallel
                       else arguments)
        tasks.append(row)
    worlds: list[list[dict]] = []
    for position, row in enumerate(tasks):
        fields = KEPT_FIELDS
        if position < FIELD_SHEET_CELLS:
            fields = KEPT_FIELDS + FIELD_SHEET_FIELDS
        worlds.append([
            prune(item.result() if parallel else run_one_world(*item), fields)
            for item in row])
    return worlds, time.perf_counter() - started, parallel


def check_determinism(selection: list[dict], worlds: list[list[dict]],
                      screen: search.Screen,
                      tolerance: float = GATE_TOLERANCE) -> dict:
    """Every rerun world must reproduce the run's own record. Raises if not.

    Returns `{"checked": worlds compared, "maxima": {metric: largest absolute
    difference seen}}`, so the gate can be shown as the numbers it passed on
    rather than only as the word "passed". A metric no world logged is absent
    from `maxima`.
    """
    checked = 0
    maxima: dict[str, float] = {}
    for cell, rerun in zip(selection, worlds):
        logged = cell["worlds"]
        if len(logged) != len(rerun):
            raise DeterminismError(
                f"{cell['id']}: the run logged {len(logged)} worlds and the "
                f"rerun produced {len(rerun)}")
        for before, world in zip(logged, rerun):
            metrics = search.world_metrics(world, screen)
            if int(metrics["seed"]) != int(before["seed"]):
                raise DeterminismError(
                    f"{cell['id']}: seed {before['seed']} came back as "
                    f"{metrics['seed']}")
            for name in GATE_METRICS:
                if name not in before:
                    continue
                gap = abs(float(metrics[name]) - float(before[name]))
                maxima[name] = max(maxima.get(name, 0.0), gap)
                if gap > tolerance:
                    raise DeterminismError(
                        f"{cell['id']} seed {before['seed']}: {name} was "
                        f"{before[name]!r} in the run and {metrics[name]!r} "
                        f"in the rerun, a difference of {gap:.3e} against a "
                        f"tolerance of {tolerance:.0e}")
            checked += 1
    return {"checked": checked, "maxima": maxima}


def measure(world: dict, screen: search.Screen) -> dict:
    """One world's metrics, screen verdict, solver effort and width shares."""
    metrics = search.world_metrics(world, screen)
    verdict = search.screen_world(metrics, screen)
    cycles = [int(value) for value in world["solver_cycles"]]
    weak = weak_mask(world["strength_final"])
    return {
        **metrics,
        "passed": bool(verdict["passed"]),
        "violation": float(verdict["violation"]),
        "terms": {name: bool(term["ok"])
                  for name, term in verdict["terms"].items()},
        "invalid": bool(metrics["residual_max"] > screen.residual_max),
        "cycles_mean": float(np.mean(cycles)) if cycles else 0.0,
        "cycles_max": max(cycles) if cycles else 0,
        "exhausted_steps": int(world.get("exhausted_steps", 0)),
        "weak_cells": int(weak.sum()),
        "width": width_shares(weak),
    }


# --------------------------------------------------------------------------
# Sheets
# --------------------------------------------------------------------------


def bundle_of(cell: dict, worlds: list[dict], config: dict, divisor: int,
              *, pixels: int, scale_km: int, max_cycles: int
              ) -> explore_adapter.Bundle:
    from engine.history.kinematics import HistoryParams
    record = params_record(cell, config, divisor, max_cycles)
    return explore_adapter.Bundle(
        seed=int(worlds[0]["seed"]),
        pixels=int(pixels),
        scale_km=int(scale_km),
        controls=dials_of(cell, config),
        params=HistoryParams(**record),
        worlds=worlds,
        parallel=True,
        elapsed_s=0.0,
    )


def write_sheets(directory: Path, selection: list[dict],
                 results: dict[int, list[list[dict]]], config: dict, *,
                 pixels: int, scale_km: int, max_cycles: int,
                 divisors: tuple[int, ...] = DIVISORS) -> list[str]:
    """`plates` for every cell, `strain_rate` and `power` for the first few."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for position, cell in enumerate(selection):
        views = ["plates"]
        if position < FIELD_SHEET_CELLS:
            views += ["strain_rate", "power"]
        for divisor in divisors:
            bundle = bundle_of(cell, results[divisor][position], config,
                               divisor, pixels=pixels, scale_km=scale_km,
                               max_cycles=max_cycles)
            for view in views:
                name = f"{cell['id']}_d{divisor}_{view}.png"
                (directory / name).write_bytes(
                    explore_adapter.render_png(bundle, view))
                written.append(name)
    return written


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def median(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def tally(differences, tolerance: float = SAME_TOL) -> tuple[int, int, int]:
    """Counts of rising, falling and unchanged."""
    differences = np.asarray(list(differences), dtype=np.float64)
    if differences.size == 0:
        return 0, 0, 0
    return (int(np.sum(differences > tolerance)),
            int(np.sum(differences < -tolerance)),
            int(np.sum(np.abs(differences) <= tolerance)))


def _fmt(value, places: int = 4) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    if np.isnan(value):
        return "n/a"
    return f"{value:.{places}f}"


def mean_width(rows: list[dict]) -> tuple[dict, int]:
    """Width shares averaged over the worlds that have a weak set."""
    kept = [row["width"] for row in rows if row["width"] is not None]
    if not kept:
        return {}, 0
    keys = [f"k{k}" for k in WIDTH_K] + ["aligned"]
    return ({key: float(np.mean([one[key] for one in kept])) for key in keys},
            len(kept))


def odd_width_count(rows: list[dict],
                    tolerance: float = ODD_WIDTH_TOLERANCE) -> tuple[int, int]:
    """Worlds whose share at k = 3 differs from the share at k = 4."""
    kept = [row["width"] for row in rows if row["width"] is not None]
    return (sum(1 for one in kept if abs(one["k3"] - one["k4"]) > tolerance),
            len(kept))


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------


def build_report(run_id: str, selection: list[dict], config: dict,
                 screen: search.Screen,
                 measured: dict[int, list[list[dict]]],
                 wall: dict[int, float], parallel: dict[int, bool], *,
                 pixels: int, scale_km: int, max_cycles: int,
                 gate: str, sheet_dir: Path,
                 divisors: tuple[int, ...] = DIVISORS,
                 gate_maxima: dict | None = None) -> str:
    lines: list[str] = []
    add = lines.append

    flat = {divisor: [row for rows in measured[divisor] for row in rows]
            for divisor in divisors}
    worlds = len(flat[divisors[0]])
    paired = len(divisors) == 2

    add(f"# A/B on the solve grid: `{run_id}` at {pixels} px")
    add("")
    variants = "at " + " and then ".join(f"`solve_divisor = {divisor}`"
                                         for divisor in divisors)
    add(f"{len(selection)} cells of `{run_id}`, each on its own seeds, rerun "
        f"{variants}. {worlds} worlds per variant "
        f"at {pixels} px, {scale_km} km per pixel, "
        f"`history_myr` {float(config.get('space', {}).get('history_myr', 300.0)):g}, "
        f"`max_cycles` {max_cycles}, `work_damage` "
        f"{int(config.get('space', {}).get('work_damage', 0))}, `seams` "
        f"{int(config.get('space', {}).get('seams', 0))}. Every number "
        "below is a count, a median, a share or a second.")
    add("")
    add(f"Determinism gate: {gate}")
    add("")
    if gate_maxima:
        add("| metric | maximum absolute difference | tolerance |")
        add("|---|---|---|")
        for name in GATE_METRICS:
            if name in gate_maxima:
                add(f"| `{name}` | {gate_maxima[name]:.3e} | "
                    f"{GATE_TOLERANCE:.0e} |")
        add("")

    # -- 1. the selection ------------------------------------------------
    add("## 1. The selection")
    add("")
    add("Ranked by the count of worlds that fail the run's screen on "
        f"`edge_fraction` alone, ties by soft score, top {TOP_BY_EDGE}; then "
        "the best remaining cells by soft score. `drive km` is the coarsest "
        "mantle wavelength, converted from the run's own `drive_nodes` where "
        "the run predates the kilometre dial.")
    add("")
    add("| cell | stage | seeds | edge-only failures | soft score | "
        "stiffness | yield % | heal Myr | damage Myr | exponent | spread | "
        "drive km | shear | crack km/Myr | nuclei |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in selection:
        dials = dials_of(cell, config)
        add(f"| `{cell['id']}` | {cell['stage']} | {len(cell['seeds'])} | "
            f"{edge_only_failures(cell, screen)} | "
            f"{_fmt(cell.get('soft_score'), 4)} | "
            f"{dials['stiffness_fraction']:.4f} | "
            f"{dials['yield_percentile']:.3f} | "
            f"{dials['heal_time_myr']:.3f} | "
            f"{dials['damage_time_myr']:.3f} | "
            f"{int(dials['strength_exponent'])} | "
            f"{dials['strength_spread']:.4f} | "
            f"{dials['drive_wavelength_km']:.1f} | "
            f"{dials['drive_shear']:.4f} | "
            f"{dials['crack_speed_km_per_myr']:.1f} | "
            f"{int(dials['nucleations_per_step'])} |")
    add("")

    # -- 2. world by world -----------------------------------------------
    add("## 2. World by world, every variant")
    add("")
    add("`ef` is `edge_fraction`, `wf` `weak_final`, `pc` `plate_count`, "
        "`wd` `weak_drift`, `ns` `network_share`. `pass` is all six terms of "
        "the run's screen. `cycles` is the mean and the worst step of the "
        "solve; `resid` is the worst relative residual over the history.")
    add("")
    columns = ["cell", "seed"]
    for label in ("ef", "wf", "pc", "wd", "ns", "pass", "cycles", "resid"):
        columns += [f"{label} d{divisor}" for divisor in divisors]
    add("| " + " | ".join(columns) + " |")
    add("|" + "---|" * len(columns))
    for position, cell in enumerate(selection):
        rows = [measured[divisor][position] for divisor in divisors]
        for index in range(len(rows[0])):
            here = [row[index] for row in rows]
            values = [f"`{cell['id']}`", str(int(here[0]["seed"]))]
            values += [_fmt(row["edge_fraction"]) for row in here]
            values += [_fmt(row["weak_final"]) for row in here]
            values += [str(int(row["plate_count"])) for row in here]
            values += [_fmt(row["weak_drift"]) for row in here]
            values += [_fmt(row["network_share"]) for row in here]
            values += ["yes" if row["passed"] else "no" for row in here]
            values += [f"{row['cycles_mean']:.1f} / {row['cycles_max']}"
                       for row in here]
            values += [f"{row['residual_max']:.2e}" for row in here]
            add("| " + " | ".join(values) + " |")
    add("")

    # -- 3. paired differences -------------------------------------------
    if paired:
        first, second = divisors
        add(f"## 3. Paired differences, divisor {second} minus divisor {first}")
        add("")
        add("One pair per world: the same cell, the same seed, the same "
            "dials.")
        add("")
        add("| metric | median difference | rising | falling | unchanged |")
        add("|---|---|---|---|---|")
        for metric in PAIRED_METRICS:
            differences = [float(b[metric]) - float(a[metric])
                           for a, b in zip(flat[first], flat[second])]
            rising, falling, same = tally(differences)
            add(f"| `{metric}` | {_fmt(median(differences), 6)} | {rising} | "
                f"{falling} | {same} |")
        add("")
    else:
        add("## 3. Paired differences")
        add("")
        add(f"One variant was run, `solve_divisor = {divisors[0]}`, so there "
            "is no pair to difference.")
        add("")

    # -- 4. the screen ----------------------------------------------------
    add("## 4. The screen at each divisor")
    add("")
    add("| | " + " | ".join(f"divisor {divisor}" for divisor in divisors)
        + " |")
    add("|" + "---|" * (len(divisors) + 1))
    add("| worlds | " + " | ".join(str(len(flat[d])) for d in divisors) + " |")
    add("| worlds passing all six terms | "
        + " | ".join(str(sum(1 for row in flat[d] if row["passed"]))
                     for d in divisors) + " |")
    add("| invalid worlds, residual above tolerance | "
        + " | ".join(str(sum(1 for row in flat[d] if row["invalid"]))
                     for d in divisors) + " |")
    add("| worlds with a step at the cycle budget | "
        + " | ".join(str(sum(1 for row in flat[d]
                             if row["exhausted_steps"] > 0))
                     for d in divisors) + " |")
    add("| worst residual over all worlds | "
        + " | ".join(f"{max(row['residual_max'] for row in flat[d]):.2e}"
                     for d in divisors) + " |")
    add("| worst step cycle count | "
        + " | ".join(str(max(row["cycles_max"] for row in flat[d]))
                     for d in divisors) + " |")
    add(f"| worlds with a step above {OLD_BUDGET} cycles | "
        + " | ".join(str(sum(1 for row in flat[d]
                             if row["cycles_max"] > OLD_BUDGET))
                     for d in divisors) + " |")
    for name, _lo, _hi in search.term_bounds(screen):
        add(f"| worlds passing `{name}` | "
            + " | ".join(str(sum(1 for row in flat[d] if row["terms"][name]))
                         for d in divisors) + " |")
    add("")
    for divisor in divisors:
        passers = [(cell["id"], row)
                   for position, cell in enumerate(selection)
                   for row in measured[divisor][position] if row["passed"]]
        add(f"Worlds passing all six at divisor {divisor}: "
            + (", ".join(f"`{cell_id}` seed {int(row['seed'])}"
                         for cell_id, row in passers) if passers else "none")
            + ".")
        add("")

    # -- 5. width ---------------------------------------------------------
    add("## 5. Width of the final weak set")
    add("")
    add("Share of the weak set covered by at least one fully weak k x k "
        "square on the torus, averaged over the worlds with a weak set, and "
        "the share whose even-aligned 2 x 2 block is entirely weak. Read off "
        "the strength field, not off a sheet.")
    add("")
    add("| | " + " | ".join(f"k = {k}" for k in WIDTH_K)
        + " | aligned 2 x 2 | worlds |")
    add("|" + "---|" * (len(WIDTH_K) + 3))
    for divisor in divisors:
        shares, count = mean_width(flat[divisor])
        add(f"| divisor {divisor} | "
            + " | ".join(_fmt(shares.get(f"k{k}")) for k in WIDTH_K)
            + f" | {_fmt(shares.get('aligned'))} | {count} |")
    add("")
    add("| | share k = 3 minus share k = 4 | worlds with a gap above "
        f"{ODD_WIDTH_TOLERANCE} | worlds measured |")
    add("|---|---|---|---|")
    for divisor in divisors:
        shares, _count = mean_width(flat[divisor])
        odd, measured_count = odd_width_count(flat[divisor])
        gap = (shares.get("k3", float("nan")) - shares.get("k4", float("nan"))
               if shares else float("nan"))
        add(f"| divisor {divisor} | {_fmt(gap)} | {odd} | {measured_count} |")
    add("")
    add("| | share k = 5 minus share k = 6 | worlds with a gap above "
        f"{ODD_WIDTH_TOLERANCE} |")
    add("|---|---|---|")
    for divisor in divisors:
        kept = [row["width"] for row in flat[divisor]
                if row["width"] is not None]
        gaps = [one["k5"] - one["k6"] for one in kept]
        add(f"| divisor {divisor} | {_fmt(median(gaps))} | "
            f"{sum(1 for gap in gaps if abs(gap) > ODD_WIDTH_TOLERANCE)} |")
    add("")

    # -- 6. cost ----------------------------------------------------------
    add("## 6. Cost")
    add("")
    add("| | " + " | ".join(f"divisor {divisor}" for divisor in divisors)
        + " |" + (" ratio |" if paired else ""))
    add("|" + "---|" * (len(divisors) + 1 + (1 if paired else 0)))
    seconds = {d: [float(row["seconds"]) for row in flat[d]] for d in divisors}
    for label, function in (("seconds per world, min", min),
                            ("seconds per world, mean",
                             lambda values: float(np.mean(values))),
                            ("seconds per world, max", max)):
        values = [function(seconds[d]) for d in divisors]
        row = f"| {label} | " + " | ".join(_fmt(one, 3) for one in values) + " |"
        if paired:
            ratio = values[1] / values[0] if values[0] else float("nan")
            row += f" {_fmt(ratio, 2)} |"
        add(row)
    for label, values in (("summed world seconds",
                           [sum(seconds[d]) for d in divisors]),
                          ("wall seconds on the pool",
                           [wall[d] for d in divisors])):
        row = f"| {label} | " + " | ".join(_fmt(one, 1) for one in values) + " |"
        if paired:
            ratio = values[1] / values[0] if values[0] else float("nan")
            row += f" {_fmt(ratio, 2)} |"
        add(row)
    add("")
    add("Pool used: "
        + ", ".join(f"divisor {d} {'yes' if parallel[d] else 'no'}"
                    for d in divisors)
        + f". {explore_adapter.MAX_SEEDS} workers, spawn context.")
    add("")
    add(f"Sheets: `{sheet_dir.as_posix()}`, `plates` for every variant of "
        f"every cell and `strain_rate` and `power` for the first "
        f"{FIELD_SHEET_CELLS}.")
    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The command
# --------------------------------------------------------------------------


def output_path(run_id: str, tag: str = "") -> Path:
    suffix = f"_{tag}" if tag else ""
    return _PIPELINE_C / "out" / f"ab_solve_{run_id}{suffix}.md"


def sheet_dir_for(run_id: str, tag: str = "") -> Path:
    suffix = f"_{tag}" if tag else ""
    return _PIPELINE_C / "out" / "ab_solve" / f"{run_id}{suffix}"


def run(run_name: str, *, cells: int = 20, pixels: int = 1024,
        max_cycles: int = 80, tag: str = "", executor=None,
        write: bool = True,
        divisors: tuple[int, ...] = DIVISORS) -> str:
    directory = resolve_run(run_name)
    run_id = directory.name
    all_cells, config = load_run(directory)
    screen = screen_of(config)
    space = config.get("space", {})
    scale_km = int(space.get("scale_km", 5))
    divisors = tuple(int(divisor) for divisor in divisors)
    if not divisors:
        raise ValueError("at least one divisor must be run")
    selection = select_cells(all_cells, cells, screen)

    results: dict[int, list[list[dict]]] = {}
    wall: dict[int, float] = {}
    parallel: dict[int, bool] = {}
    gate = ""
    gate_maxima: dict = {}
    for divisor in divisors:
        worlds, seconds, used = run_variant(
            selection, config, divisor, pixels=pixels, scale_km=scale_km,
            max_cycles=max_cycles, executor=executor)
        results[divisor] = worlds
        wall[divisor] = seconds
        parallel[divisor] = used
        if divisor == 2:
            logged_pixels = int(space.get("pixels", 0))
            if logged_pixels == int(pixels):
                outcome = check_determinism(selection, worlds, screen)
                gate_maxima = outcome["maxima"]
                gate = (f"passed. {outcome['checked']} worlds at "
                        f"`solve_divisor = 2` reproduce the run's logged "
                        f"{', '.join(f'`{name}`' for name in GATE_METRICS)} "
                        f"to {GATE_TOLERANCE:.0e}.")
            else:
                gate = (f"not applicable. The run ran at {logged_pixels} px "
                        f"and this rerun is at {pixels} px, so the parent is "
                        "a different size and the worlds are different "
                        "worlds. Pairing here is between the divisors "
                        "within this run, by cell and seed.")
    if not gate:
        gate = ("not applicable. `solve_divisor = 2` was not among the "
                "variants run, and the gate is the divisor-2 rerun against "
                "the run's own record.")

    measured = {divisor: [[measure(world, screen) for world in row]
                          for row in results[divisor]]
                for divisor in divisors}

    sheet_dir = sheet_dir_for(run_id, tag)
    if write:
        write_sheets(sheet_dir, selection, results, config, pixels=pixels,
                     scale_km=scale_km, max_cycles=max_cycles,
                     divisors=divisors)

    text = build_report(run_id, selection, config, screen, measured,
                        wall, parallel, pixels=pixels, scale_km=scale_km,
                        max_cycles=max_cycles, gate=gate,
                        sheet_dir=sheet_dir, divisors=divisors,
                        gate_maxima=gate_maxima)
    if write:
        destination = output_path(run_id, tag)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
        sys.stderr.write(f"written to {destination}\n")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rerun a search run's cells on both solve grids.")
    parser.add_argument("run", help="run id under out/search, or a directory")
    parser.add_argument("--cells", type=int, default=20,
                        help="how many cells to rerun (default 20)")
    parser.add_argument("--pixels", type=int, default=1024,
                        help="delivered pixels per axis (default 1024)")
    parser.add_argument("--max-cycles", type=int, default=80,
                        help="solver effort per step (default 80)")
    parser.add_argument("--tag", default="",
                        help="suffix for the page and the sheet directory, so "
                             "a second run at another size does not overwrite "
                             "the first")
    parser.add_argument("--divisors", type=int, nargs="+", default=None,
                        metavar="D",
                        help="solve divisors to run, in report order "
                             f"(default {' '.join(str(d) for d in DIVISORS)}); "
                             "one value runs one variant and the paired "
                             "sections say so")
    args = parser.parse_args(argv)
    divisors = tuple(args.divisors) if args.divisors else DIVISORS
    text = run(args.run, cells=args.cells, pixels=args.pixels,
               max_cycles=args.max_cycles, tag=args.tag, divisors=divisors)
    sys.stdout.write(text + "\n")
    return 0


__all__ = [
    "DIVISORS",
    "GATE_METRICS",
    "TOP_BY_EDGE",
    "WIDTH_K",
    "DeterminismError",
    "aligned_blocks",
    "all_weak_squares",
    "build_report",
    "check_determinism",
    "covered_by_square",
    "dials_of",
    "edge_only_failures",
    "load_run",
    "main",
    "measure",
    "mean_width",
    "odd_width_count",
    "output_path",
    "params_record",
    "resolve_run",
    "run",
    "run_geometry",
    "run_variant",
    "screen_of",
    "select_cells",
    "sheet_dir_for",
    "width_shares",
    "write_sheets",
]


if __name__ == "__main__":
    raise SystemExit(main())
