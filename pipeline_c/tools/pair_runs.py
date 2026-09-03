"""Pair two search runs cell for cell and report what moved.

`work_damage` is a fixed setting of the search space, not a sampled dial, so
two runs at the same `search_seed` over the same space draw exactly the same
Latin hypercube. A run at `work_damage = 1` therefore visits the same stage-1
cells as a run at `0`, cell for cell, and the two are an ablation pair: every
difference between them is the law and nothing else.

This tool reads both runs off disk, pairs their stage-1 cells by dial values,
and writes one markdown page of paired differences, rank correlations, band
statistics, time to failure and throughput.

    py -3.14 pipeline_c/tools/pair_runs.py <control_run_id> <treatment_run_id>

The page goes to stdout and to `out/pair_<control>_<treatment>.md`. It
describes nothing and proposes nothing: every row is a count, a median or a
share.

**Legacy dials are modernized on read.** A `drive_nodes` recorded before
`WORK_ORDER_C03_10.md` becomes the wavelength in kilometres it meant at the
run's own resolution and scale, on both sides, so the pairing is by physical
dial values rather than by whichever spelling a run used.

**On the trajectory.** `cells.jsonl` keeps each world's final, peak and drift
but not its weak-fraction trajectory, so time to half is read off the
`trajectory.png` sheet every cell carries, whose columns are the per-step weak
fraction quantized to 64 levels. Both runs are decoded by the same routine, so
the paired difference is a comparison of like with like; the resolution is one
sixty-fourth of the sheet, and heights of 31 and 32 are indistinguishable
because the half-mark line is drawn over row 32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

_PIPELINE_C = Path(__file__).resolve().parents[1]
if str(_PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_C))

import search  # noqa: E402

#: Dial values are compared at this many decimal places, as the order says.
DIAL_PLACES = 12

#: Metrics reported as paired differences, in report order.
PAIRED_METRICS = ("weak_final", "edge_fraction", "plate_count",
                  "network_share", "weak_peak")

#: A paired difference smaller than this counts as neither a rise nor a fall.
SAME_TOL = 1e-6

#: The trajectory sheet's geometry, from `explore_adapter`: one strip per
#: world, `STRIP_PX` tall, `STRIP_GUTTER_PX` of black between strips, one
#: filled column per step in `STRIP_COLUMN_RGB`, and a one-pixel line at the
#: half mark in `STRIP_LINE_RGB` drawn over whatever is there.
STRIP_PX = 64
STRIP_GUTTER_PX = 2
STRIP_COLUMN_RGB = (242, 157, 74)
STRIP_LINE_RGB = (213, 73, 91)


# --------------------------------------------------------------------------
# Loading
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


class Run:
    """One search run read off disk."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.run_id = self.dir.name
        config_path = self.dir / "config.json"
        self.config = (json.loads(config_path.read_text(encoding="utf-8"))
                       if config_path.exists() else {})
        self.screen = self.config.get("screen", {})
        self.space = self.config.get("space", {})
        pixels = int(self.space.get("pixels", 1024))
        scale_km = int(self.space.get("scale_km", 5))
        self.cells = [
            json.loads(line)
            for line in (self.dir / "cells.jsonl").read_text(
                encoding="utf-8").splitlines()
            if line.strip()
        ]
        # A run written before `WORK_ORDER_C03_10.md` recorded `drive_nodes`,
        # a fraction of its own parent world. Both sides are read through the
        # same conversion, at each run's own geometry, so a pre-order run and
        # a post-order run at the same physical wavelength still pair by dial
        # values.
        for cell in self.cells:
            cell["dials"] = search.modernize_dials(cell["dials"], pixels,
                                                   scale_km)

    # -- slices --------------------------------------------------------

    @property
    def stage1(self) -> list[dict]:
        return [cell for cell in self.cells if cell["stage"] == 1]

    @property
    def worlds(self) -> list[dict]:
        return [world for cell in self.cells for world in cell["worlds"]]

    @property
    def invalid(self) -> int:
        return sum(1 for cell in self.cells if cell["invalid"])

    @property
    def passers(self) -> int:
        return sum(1 for cell in self.cells if cell["passed"])

    @property
    def findings(self) -> int:
        return sum(1 for cell in self.cells if cell.get("finding"))

    @property
    def work_damage(self) -> int:
        return int(self.space.get("work_damage", 0))

    @property
    def seams(self) -> int:
        return int(self.space.get("seams", 0))

    @property
    def crack_speed_range(self) -> tuple[float, float]:
        return (float(self.space.get("crack_speed_km_per_myr_lo", 0.0)),
                float(self.space.get("crack_speed_km_per_myr_hi", 0.0)))

    @property
    def nucleations_set(self) -> list[int]:
        return [int(value) for value
                in self.space.get("nucleations_per_step_set", [])]

    def elapsed_s(self) -> float:
        """Wall time, as the run directory records it.

        `config.json` is written when the run starts and `cells.jsonl` when
        its last cell lands, so the gap between the two file times is the
        run's wall clock. It is the only elapsed time on disk.
        """
        config_path = self.dir / "config.json"
        cells_path = self.dir / "cells.jsonl"
        if not config_path.exists():
            return 0.0
        return max(0.0, cells_path.stat().st_mtime - config_path.stat().st_mtime)


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------


def dial_key(dials: dict) -> tuple:
    """A cell's dials, rounded, in a fixed order, as a hashable key."""
    return tuple((name, round(float(dials[name]), DIAL_PLACES))
                 for name in sorted(dials))


def pair_cells(control: list[dict],
               treatment: list[dict]) -> tuple[list[tuple[dict, dict]],
                                               list[dict], list[dict]]:
    """Pair by dial values, first come first served.

    Returns the pairs in control order, then the control cells that found no
    partner, then the treatment cells that found none. Nothing is dropped
    silently: every cell of both runs is in exactly one of the three lists.
    """
    buckets: dict[tuple, list[dict]] = {}
    for cell in treatment:
        buckets.setdefault(dial_key(cell["dials"]), []).append(cell)
    pairs: list[tuple[dict, dict]] = []
    lonely_control: list[dict] = []
    for cell in control:
        bucket = buckets.get(dial_key(cell["dials"]))
        if bucket:
            pairs.append((cell, bucket.pop(0)))
        else:
            lonely_control.append(cell)
    lonely_treatment = [cell for bucket in buckets.values() for cell in bucket]
    lonely_treatment.sort(key=lambda cell: cell["index"])
    return pairs, lonely_control, lonely_treatment


def cell_mean(cell: dict, metric: str) -> float:
    return float(np.mean([float(world[metric]) for world in cell["worlds"]]))


# --------------------------------------------------------------------------
# Statistics, written by hand from numpy
# --------------------------------------------------------------------------


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, ties sharing the mean of the ranks they span."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    ordered = values[order]
    start = 0
    while start < ordered.size:
        stop = start
        while stop + 1 < ordered.size and ordered[stop + 1] == ordered[start]:
            stop += 1
        if stop > start:
            ranks[order[start:stop + 1]] = (start + stop + 2) / 2.0
        start = stop + 1
    return ranks


def spearman(x, y) -> float:
    """Spearman's rank correlation: Pearson on average ranks. No SciPy."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("spearman needs two equally long sequences")
    if x.size < 2:
        return float("nan")
    rx = average_ranks(x) - (x.size + 1) / 2.0
    ry = average_ranks(y) - (y.size + 1) / 2.0
    denominator = float(np.sqrt((rx @ rx) * (ry @ ry)))
    if denominator == 0.0:
        return float("nan")
    return float((rx @ ry) / denominator)


def median(values) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


def tally(differences) -> tuple[int, int, int]:
    """Counts of rising, falling and unchanged, at `SAME_TOL`."""
    differences = np.asarray(list(differences), dtype=np.float64)
    if differences.size == 0:
        return 0, 0, 0
    same = int(np.sum(np.abs(differences) <= SAME_TOL))
    rising = int(np.sum(differences > SAME_TOL))
    falling = int(np.sum(differences < -SAME_TOL))
    return rising, falling, same


# --------------------------------------------------------------------------
# The trajectory sheet
# --------------------------------------------------------------------------


def decode_trajectory(path: Path, worlds: int) -> list[list[int]] | None:
    """Per-world, per-step column heights read off a `trajectory.png`.

    A height is the weak fraction in sixty-fourths. Returns `None` when the
    sheet is missing or is not the shape this run's cells should have.
    """
    path = Path(path)
    if not path.exists():
        return None
    array = np.asarray(Image.open(path).convert("RGB"))
    height, width, _ = array.shape
    expected = worlds * STRIP_PX + (worlds - 1) * STRIP_GUTTER_PX
    if height != expected or width < 1:
        return None
    column_rgb = np.array(STRIP_COLUMN_RGB, dtype=np.uint8)
    out: list[list[int]] = []
    for index in range(worlds):
        top = index * (STRIP_PX + STRIP_GUTTER_PX)
        strip = array[top:top + STRIP_PX]
        filled = np.all(strip == column_rgb, axis=-1)
        heights: list[int] = []
        for column in range(width):
            rows = np.flatnonzero(filled[:, column])
            heights.append(0 if rows.size == 0 else STRIP_PX - int(rows[0]))
        out.append(heights)
    return out


def half_step(heights: list[int]) -> float | None:
    """The first step, counting from one, that reaches half the final height.

    `None` when the trajectory ends at zero, where half of the final value is
    not a threshold anything can cross.
    """
    if not heights:
        return None
    final = heights[-1]
    if final <= 0:
        return None
    threshold = final / 2.0
    for index, value in enumerate(heights):
        if value >= threshold:
            return float(index + 1)
    return float(len(heights))


def cell_half_step(run: Run, cell: dict) -> float | None:
    """Mean over a cell's worlds of the first step reaching half the final."""
    heights = decode_trajectory(
        run.dir / "cells" / cell["id"] / "trajectory.png", len(cell["worlds"]))
    if heights is None:
        return None
    steps = [half_step(one) for one in heights]
    kept = [value for value in steps if value is not None]
    if not kept:
        return None
    return float(np.mean(kept))


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def band_stats(run: Run) -> dict:
    """Worlds whose final weak fraction is inside the screen's band."""
    low = float(run.screen.get("weak_min", 0.02))
    high = float(run.screen.get("weak_max", 0.25))
    inside = [world for world in run.worlds
              if low <= float(world["weak_final"]) <= high]
    if not inside:
        return {"low": low, "high": high, "worlds": 0, "total": len(run.worlds),
                "p50": float("nan"), "p90": float("nan"), "max": float("nan"),
                "one_plate": float("nan"), "two_or_more": float("nan")}
    edges = np.array([float(world["edge_fraction"]) for world in inside])
    plates = np.array([int(world["plate_count"]) for world in inside])
    return {
        "low": low,
        "high": high,
        "worlds": len(inside),
        "total": len(run.worlds),
        "p50": float(np.percentile(edges, 50)),
        "p90": float(np.percentile(edges, 90)),
        "max": float(edges.max()),
        "one_plate": float(np.mean(plates == 1)),
        "two_or_more": float(np.mean(plates >= 2)),
    }


def _fmt(value: float, places: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{places}f}"


def build_report(control: Run, treatment: Run) -> str:
    pairs, lonely_control, lonely_treatment = pair_cells(
        control.stage1, treatment.stage1)

    lines: list[str] = []
    add = lines.append

    add(f"# Paired runs: `{control.run_id}` against `{treatment.run_id}`")
    add("")
    add(f"Control `{control.run_id}` ran at `work_damage = "
        f"{control.work_damage}`; treatment `{treatment.run_id}` at "
        f"`work_damage = {treatment.work_damage}`. Stage-1 cells are paired "
        f"by dial values equal to {DIAL_PLACES} decimal places over all "
        "dials. Every number below is a count, a median or a share.")
    add("")

    # -- 1. cell counts -------------------------------------------------
    add("## 1. Cells")
    add("")
    add("| | control | treatment |")
    add("|---|---|---|")
    add(f"| run id | `{control.run_id}` | `{treatment.run_id}` |")
    add(f"| `work_damage` | {control.work_damage} | {treatment.work_damage} |")
    add(f"| `seams` | {control.seams} | {treatment.seams} |")
    add(f"| `crack_speed_km_per_myr` | "
        f"{control.crack_speed_range[0]:g} - {control.crack_speed_range[1]:g} | "
        f"{treatment.crack_speed_range[0]:g} - "
        f"{treatment.crack_speed_range[1]:g} |")
    add(f"| `nucleations_per_step` | {control.nucleations_set} | "
        f"{treatment.nucleations_set} |")
    add(f"| cells, all stages | {len(control.cells)} | {len(treatment.cells)} |")
    add(f"| stage-1 cells | {len(control.stage1)} | {len(treatment.stage1)} |")
    add(f"| stage-2 cells | "
        f"{sum(1 for c in control.cells if c['stage'] == 2)} | "
        f"{sum(1 for c in treatment.cells if c['stage'] == 2)} |")
    add(f"| stage-3 cells | "
        f"{sum(1 for c in control.cells if c['stage'] == 3)} | "
        f"{sum(1 for c in treatment.cells if c['stage'] == 3)} |")
    add(f"| worlds, all stages | {len(control.worlds)} | "
        f"{len(treatment.worlds)} |")
    add(f"| invalid cells | {control.invalid} | {treatment.invalid} |")
    add(f"| stage-1 cells paired | {len(pairs)} | {len(pairs)} |")
    add(f"| stage-1 cells unpaired | {len(lonely_control)} | "
        f"{len(lonely_treatment)} |")
    add("")
    if lonely_control or lonely_treatment:
        add("Unpaired stage-1 cells, by id: control "
            + (", ".join(f"`{cell['id']}`" for cell in lonely_control[:40])
               or "none")
            + "; treatment "
            + (", ".join(f"`{cell['id']}`" for cell in lonely_treatment[:40])
               or "none")
            + ".")
        add("")

    # -- 2. paired differences -----------------------------------------
    add("## 2. Paired differences, treatment minus control, over cell means")
    add("")
    add("| metric | median difference | rising | falling | within 1e-6 |")
    add("|---|---|---|---|---|")
    for metric in PAIRED_METRICS:
        differences = [cell_mean(after, metric) - cell_mean(before, metric)
                       for before, after in pairs]
        rising, falling, same = tally(differences)
        add(f"| `{metric}` | {_fmt(median(differences), 6)} | {rising} | "
            f"{falling} | {same} |")
    add("")

    # -- 3. rank correlation -------------------------------------------
    add("## 3. Rank correlation of `edge_fraction` with `weak_final`")
    add("")
    add("Spearman over stage-1 cell means, each run on its own cells.")
    add("")
    add("| run | stage-1 cells | rank correlation |")
    add("|---|---|---|")
    for run in (control, treatment):
        cells = run.stage1
        add(f"| `{run.run_id}` | {len(cells)} | "
            + _fmt(spearman([cell_mean(cell, "edge_fraction")
                             for cell in cells],
                            [cell_mean(cell, "weak_final")
                             for cell in cells]), 4)
            + " |")
    add("")

    # -- 4. band statistics --------------------------------------------
    control_band = band_stats(control)
    treatment_band = band_stats(treatment)
    add("## 4. Worlds inside the screen's weak band")
    add("")
    add(f"Every world of every stage whose final weak fraction is in "
        f"[{control_band['low']}, {control_band['high']}].")
    add("")
    add("| | control | treatment |")
    add("|---|---|---|")
    add(f"| band worlds | {control_band['worlds']} of "
        f"{control_band['total']} | {treatment_band['worlds']} of "
        f"{treatment_band['total']} |")
    add(f"| `edge_fraction` p50 | {_fmt(control_band['p50'])} | "
        f"{_fmt(treatment_band['p50'])} |")
    add(f"| `edge_fraction` p90 | {_fmt(control_band['p90'])} | "
        f"{_fmt(treatment_band['p90'])} |")
    add(f"| `edge_fraction` max | {_fmt(control_band['max'])} | "
        f"{_fmt(treatment_band['max'])} |")
    add(f"| share with one plate | {_fmt(control_band['one_plate'])} | "
        f"{_fmt(treatment_band['one_plate'])} |")
    add(f"| share with two or more | {_fmt(control_band['two_or_more'])} | "
        f"{_fmt(treatment_band['two_or_more'])} |")
    add("")

    # -- 5. time to half -----------------------------------------------
    add("## 5. Time to half the final weak fraction")
    add("")
    add("Pairs whose control `weak_final` cell mean is at least 0.1. The step "
        "is read off each cell's `trajectory.png`, whose columns are the "
        "per-step weak fraction in sixty-fourths, and averaged over the "
        "cell's worlds.")
    add("")
    subset = [(before, after) for before, after in pairs
              if cell_mean(before, "weak_final") >= 0.1]
    differences = []
    missing = 0
    for before, after in subset:
        first = cell_half_step(control, before)
        second = cell_half_step(treatment, after)
        if first is None or second is None:
            missing += 1
            continue
        differences.append(second - first)
    earlier, later, same = tally(differences)
    # `tally` counts a rise; earlier is a fall.
    earlier, later = later, earlier
    add("| | value |")
    add("|---|---|")
    add(f"| pairs with control mean `weak_final` >= 0.1 | {len(subset)} |")
    add(f"| pairs measured | {len(differences)} |")
    add(f"| pairs without a readable trajectory | {missing} |")
    add(f"| median difference in steps, treatment minus control | "
        f"{_fmt(median(differences), 3)} |")
    add(f"| earlier under the treatment | {earlier} |")
    add(f"| later under the treatment | {later} |")
    add(f"| equal | {same} |")
    add("")

    # -- 6. passers, findings, throughput -------------------------------
    add("## 6. Passers, findings and throughput")
    add("")
    add("| | control | treatment |")
    add("|---|---|---|")
    add(f"| passing cells | {control.passers} | {treatment.passers} |")
    add(f"| findings | {control.findings} | {treatment.findings} |")
    rows = []
    for run in (control, treatment):
        elapsed = run.elapsed_s()
        rows.append((
            elapsed,
            len(run.cells) / elapsed * 60.0 if elapsed > 0 else float("nan"),
            len(run.worlds) / elapsed if elapsed > 0 else float("nan"),
        ))
    add(f"| wall seconds | {_fmt(rows[0][0], 1)} | {_fmt(rows[1][0], 1)} |")
    add(f"| cells per minute | {_fmt(rows[0][1], 2)} | {_fmt(rows[1][1], 2)} |")
    add(f"| worlds per second | {_fmt(rows[0][2], 3)} | "
        f"{_fmt(rows[1][2], 3)} |")
    add("")
    add("Wall seconds is the gap between `config.json`, written when the run "
        "starts, and the last line of `cells.jsonl`.")
    add("")
    return "\n".join(lines)


def output_path(control: Run, treatment: Run) -> Path:
    return (_PIPELINE_C / "out"
            / f"pair_{control.run_id}_{treatment.run_id}.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pair two regime-search runs and report what moved.")
    parser.add_argument("control", help="control run id, or a run directory")
    parser.add_argument("treatment",
                        help="treatment run id, or a run directory")
    parser.add_argument("--out", default=None,
                        help="where to write the page; the default is "
                             "out/pair_<control>_<treatment>.md")
    args = parser.parse_args(argv)

    control = Run(resolve_run(args.control))
    treatment = Run(resolve_run(args.treatment))
    text = build_report(control, treatment)
    destination = (Path(args.out) if args.out
                   else output_path(control, treatment))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")
    sys.stderr.write(f"written to {destination}\n")
    return 0


__all__ = [
    "DIAL_PLACES",
    "PAIRED_METRICS",
    "Run",
    "average_ranks",
    "band_stats",
    "build_report",
    "cell_half_step",
    "cell_mean",
    "decode_trajectory",
    "dial_key",
    "half_step",
    "median",
    "output_path",
    "pair_cells",
    "resolve_run",
    "spearman",
    "tally",
]


if __name__ == "__main__":
    raise SystemExit(main())
