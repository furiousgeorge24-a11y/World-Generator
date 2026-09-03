"""One pass over a grid of two dials, so the author starts with a map.

    py -3.14 pipeline_c/tools/explore_sweep.py --out pipeline_c/out

Eight worlds per cell, in the exploration lab's process pool, every other dial
at its default. This is a map of the space, not a search: it runs once, it
proposes nothing, and `stable_count` is a screening number for the person at
the dials rather than a gate or an approval.

A cell whose mean worst solver residual is not below `MG_TOL` is **not a
result**. The velocity fields it reports were not solved, and every number
downstream of them — plate count, weak fraction, the screen — is a reading off
an unfinished iterate. Such rows are marked `unconverged` and are excluded
from the best-cell choice.

The rows default to C03.6's: the stiff half of the C03.5 grid, rerun after the
solver change. `--label` and `--prefix` set the naming for any other pass.

`--prior` reads an earlier pass's CSV and joins its converged rows outside
this pass's stiffness rows into the pool the best cell is chosen from, so a
rerun of part of a grid can still name the best cell of the whole grid. If the
winner comes from the prior pass, its cell is regenerated here — with this
code, not the code that wrote the CSV — so the sheets show what the engine
does now, and the report says both numbers.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import explore_adapter  # noqa: E402
from engine.history.constants import MG_TOL  # noqa: E402

SEED = 4287772760
PIXELS = 1024
SCALE_KM = 5
SEEDS_PER_VIEW = 8
MAX_CYCLES = 40

STIFFNESS = (0.25, 0.5, 1.0, 2.0)
YIELD_PERCENTILE = (3.0, 6.0, 12.0, 20.0, 30.0)

#: The sheets written for the best cell.
BEST_VIEWS = ("plates", "weak_t32", "trajectory")

COLUMNS = (
    "stiffness_fraction",
    "yield_percentile",
    "stable_count",
    "weak_final_mean",
    "plate_count_mean",
    "solver_residual_max_mean",
    "exhausted_steps_mean",
    "seconds_per_world_mean",
    "converged",
)


def run_cell(stiffness: float, percentile: float) -> tuple[dict, object]:
    controls = {
        "scale_km": SCALE_KM,
        "seeds_per_view": SEEDS_PER_VIEW,
        "stiffness_fraction": stiffness,
        "yield_percentile": percentile,
        "max_cycles": MAX_CYCLES,
    }
    bundle = explore_adapter.generate(SEED, controls, PIXELS)
    record = explore_adapter.report(bundle)
    worlds = record["worlds"]
    residual = float(np.mean([w["solver_residual_max"] for w in worlds]))
    row = {
        "stiffness_fraction": stiffness,
        "yield_percentile": percentile,
        "stable_count": record["summary"]["stable_count"],
        "weak_final_mean": round(
            float(np.mean([w["weak_final"] for w in worlds])), 6),
        "plate_count_mean": round(
            float(np.mean([w["plate_count"] for w in worlds])), 3),
        "solver_residual_max_mean": residual,
        "exhausted_steps_mean": round(
            float(np.mean([w["exhausted_steps"] for w in worlds])), 3),
        "seconds_per_world_mean": round(
            float(np.mean([w["seconds"] for w in worlds])), 2),
        "converged": residual < MG_TOL,
    }
    return row, bundle


def read_prior(source: Path) -> list[dict]:
    """The converged rows of an earlier pass, outside this pass's rows.

    Only `stiffness_fraction`, `yield_percentile`, `stable_count` and the
    residual are read; the rest of that pass's columns describe the code that
    wrote them, not this one.
    """
    rows: list[dict] = []
    with source.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            stiffness = float(record["stiffness_fraction"])
            if stiffness in STIFFNESS:
                continue
            residual = float(record["solver_residual_max_mean"])
            if residual >= MG_TOL:
                continue
            rows.append({
                "stiffness_fraction": stiffness,
                "yield_percentile": float(record["yield_percentile"]),
                "stable_count": int(record["stable_count"]),
                "weak_final_mean": float(record["weak_final_mean"]),
                "plate_count_mean": float(record["plate_count_mean"]),
                "solver_residual_max_mean": residual,
                "converged": True,
                "prior": source.name,
            })
    return rows


def write_csv(rows: list[dict], target: Path) -> None:
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows: list[dict], target: Path, best: dict | None,
                   wall_s: float, parallel: bool, sheets: list[str],
                   label: str) -> None:
    lines = [
        f"# {label} sweep — the stiff rows, after the solver change",
        "",
        f"One pass over `stiffness_fraction` x `yield_percentile`, seeds "
        f"`{SEED}` to `{SEED + SEEDS_PER_VIEW - 1}`, {PIXELS} px, scale "
        f"{SCALE_KM}, `max_cycles` {MAX_CYCLES}, every other dial at its "
        f"default. {len(rows)} cells, {SEEDS_PER_VIEW} worlds each, "
        f"{wall_s / 60.0:.1f} minutes in the pool (`parallel: {parallel}`).",
        "",
        "A cell whose mean worst residual is not below "
        f"`MG_TOL` = {MG_TOL:g} is marked **unconverged**. Its velocity fields "
        "were not solved, so its plate counts, weak fractions and screen are "
        "readings off an unfinished iterate rather than results.",
        "",
        "`stable_count` counts the worlds of a cell with 3 to 8 plates above "
        "1 % of the parent, a final weak fraction between 0.02 and 0.25, and "
        "a peak weak fraction under 1.5 times the final. It is a screening "
        "number for the person at the dials: not a gate, not an approval, and "
        "not a statement about how any of these fields look.",
        "",
        "| stiffness | yield % | stable of 8 | mean weak_final | mean plates "
        "| mean residual max | mean exhausted steps | mean s/world | solved |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stiffness_fraction']} | {row['yield_percentile']:g} | "
            f"{row['stable_count']} | {row['weak_final_mean']:.6f} | "
            f"{row['plate_count_mean']:.3f} | "
            f"{row['solver_residual_max_mean']:.3e} | "
            f"{row['exhausted_steps_mean']:.3f} | "
            f"{row['seconds_per_world_mean']:.2f} | "
            f"{'yes' if row['converged'] else '**unconverged**'} |")
    lines += [
        "",
        "## `stable_count` as a grid",
        "",
        "An unconverged cell is written as `-`, because it has no result to "
        "screen.",
        "",
        "| stiffness \\ yield % | " + " | ".join(
            f"{value:g}" for value in YIELD_PERCENTILE) + " |",
        "|---|" + "---|" * len(YIELD_PERCENTILE),
    ]
    by_cell = {(row["stiffness_fraction"], row["yield_percentile"]): row
               for row in rows}
    for stiffness in STIFFNESS:
        cells = []
        for percentile in YIELD_PERCENTILE:
            row = by_cell[(stiffness, percentile)]
            cells.append(str(row["stable_count"]) if row["converged"] else "-")
        lines.append(f"| {stiffness} | {' | '.join(cells)} |")
    lines += ["", "## Highest `stable_count`", ""]
    if best is None:
        lines.append(
            "No cell of this pass converged, so this pass names no best cell.")
    else:
        source = (f"carried over from `{best['prior']}`" if "prior" in best
                  else "from this pass")
        lines.append(
            f"`stiffness_fraction` {best['stiffness_fraction']}, "
            f"`yield_percentile` {best['yield_percentile']:g}, {source}: "
            f"{best['stable_count']} of {SEEDS_PER_VIEW} worlds pass the "
            f"screen, mean weak_final {best['weak_final_mean']:.6f}, mean "
            f"plate count {best['plate_count_mean']:.3f}. Ties are broken "
            "toward the lowest stiffness, then the lowest yield percentile.")
        if "regenerated" in best:
            now = best["regenerated"]
            lines += [
                "",
                "The sheets below were regenerated with this run's solver, "
                f"which gives that cell {now['stable_count']} of "
                f"{SEEDS_PER_VIEW} on the screen, mean weak_final "
                f"{now['weak_final_mean']:.6f}, mean plate count "
                f"{now['plate_count_mean']:.3f}, mean worst residual "
                f"{now['solver_residual_max_mean']:.3e}, mean exhausted steps "
                f"{now['exhausted_steps_mean']:.3f}.",
            ]
    if sheets:
        lines += ["", "Its sheets:", ""]
        lines += [f"- {path}" for path in sheets]
    lines.append("")
    target.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "out"))
    parser.add_argument("--label", default="C03.6")
    parser.add_argument("--prefix", default="c03_6")
    parser.add_argument("--prior", default=None,
                        help="CSV of an earlier pass whose converged rows "
                             "outside these stiffness rows join the pool")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    bundles: dict[tuple[float, float], object] = {}
    started = time.perf_counter()
    parallel = True
    for stiffness in STIFFNESS:
        for percentile in YIELD_PERCENTILE:
            cell_started = time.perf_counter()
            row, bundle = run_cell(stiffness, percentile)
            parallel = parallel and bundle.parallel
            rows.append(row)
            bundles[(stiffness, percentile)] = bundle
            print(f"stiffness {stiffness:<6g} yield {percentile:<5g} "
                  f"stable {row['stable_count']}/{SEEDS_PER_VIEW}  "
                  f"weak_final {row['weak_final_mean']:.4f}  "
                  f"plates {row['plate_count_mean']:.2f}  "
                  f"residual {row['solver_residual_max_mean']:.2e}  "
                  f"{'solved' if row['converged'] else 'UNCONVERGED'}  "
                  f"{time.perf_counter() - cell_started:5.1f} s", flush=True)
    wall = time.perf_counter() - started

    pool = [row for row in rows if row["converged"]]
    prior = read_prior(Path(args.prior)) if args.prior else []
    pool += prior
    best = None
    sheets: list[str] = []
    if pool:
        best = min(pool, key=lambda row: (-row["stable_count"],
                                          row["stiffness_fraction"],
                                          row["yield_percentile"]))
        key = (best["stiffness_fraction"], best["yield_percentile"])
        if key in bundles:
            bundle = bundles[key]
        else:
            print(f"best cell {key} comes from the prior pass; regenerating it "
                  f"with this code", flush=True)
            best_now, bundle = run_cell(*key)
            print(f"  regenerated: stable {best_now['stable_count']}, "
                  f"weak_final {best_now['weak_final_mean']:.6f}, plates "
                  f"{best_now['plate_count_mean']:.3f}, residual "
                  f"{best_now['solver_residual_max_mean']:.3e}", flush=True)
            best = dict(best, regenerated=best_now)
        for view in BEST_VIEWS:
            target = out / f"{args.prefix}_best_{view}.png"
            image = explore_adapter.sheet(bundle, view)
            image.save(target, format="PNG", optimize=True)
            sheets.append(
                f"`{target.as_posix()}` — {image.width} x {image.height}")
            print(f"{view}: {image.width} x {image.height} -> {target}")

    write_csv(rows, out / f"{args.prefix}_sweep.csv")
    write_markdown(rows, out / f"{args.prefix}_sweep.md", best, wall, parallel,
                   sheets, args.label)
    print(f"\n{len(rows)} cells in {wall / 60.0:.1f} minutes, "
          f"parallel {parallel}, "
          f"{sum(1 for row in rows if row['converged'])} of {len(rows)} "
          f"converged")
    if best is not None:
        print(f"best converged cell: stiffness {best['stiffness_fraction']} "
              f"yield {best['yield_percentile']:g} "
              f"stable {best['stable_count']}/{SEEDS_PER_VIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
