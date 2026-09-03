"""The 512-px rerun of `WORK_ORDER_C03_10.md` §5.2.

`WORK_ORDER_C03_9.md` reran twenty cells of `20260902T183110Z-s2` at 512 px
and three of those worlds passed the whole screen, where none passed at
1024 px on the same dials. The reason was that the mantle drive's coarsest
wavelength was a fraction of the parent world, so at 512 px it was 2,560 km
against 5,120 km at 1024 px: the plates shrank and the boundaries did not.
C03.10 put that wavelength in kilometres. This tool reruns the same twenty
cells at 512 px, on their own seeds, with the dials modernized, and sets the
result beside the two pages already on disk.

    py -3.14 pipeline_c/tools/c03_10_512.py

The page goes to stdout and to `out/c03_10_512_rerun.md`. It describes
nothing and proposes nothing: every number is a count, a mean or a share.

**Where the comparison numbers come from.** The 1024-px and old 512-px
columns are read out of the section-2 tables of
`out/ab_solve_20260902T183110Z-s2.md` and
`out/ab_solve_20260902T183110Z-s2_512px.md`. They cannot be recomputed: the
old 512-px numbers are the output of the engine before this order and that
engine no longer exists in the tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

_PIPELINE_C = Path(__file__).resolve().parents[1]
if str(_PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_C))

import search  # noqa: E402
from tools import ab_solve  # noqa: E402

RUN_ID = "20260902T183110Z-s2"
CELLS = 20
PIXELS = 512
MAX_CYCLES = 80

#: The two pages the rerun is set beside, and the label each gets.
PAGE_1024 = "ab_solve_20260902T183110Z-s2.md"
PAGE_512_OLD = "ab_solve_20260902T183110Z-s2_512px.md"

#: The metrics compared per cell, as `(key in the parsed table, label)`.
COMPARED = (("pc", "plate count"), ("wf", "weak fraction"),
            ("ef", "edge fraction"))


def out_dir() -> Path:
    return _PIPELINE_C / "out"


# --------------------------------------------------------------------------
# Reading a page already on disk
# --------------------------------------------------------------------------


def parse_world_table(text: str) -> dict[tuple[str, int], dict[str, float]]:
    """The per-world rows of an `ab_solve` page's section 2.

    Keyed by `(cell id, seed)`; each value maps a column label such as
    `ef d2` to its number. Columns that are not numbers — the pass verdicts,
    the cycle counts — are skipped, because nothing here reads them.
    """
    rows: dict[tuple[str, int], dict[str, float]] = {}
    lines = text.splitlines()
    header: list[str] | None = None
    for line in lines:
        if not line.startswith("|"):
            header = None
            continue
        cells = [part.strip() for part in line.strip().strip("|").split("|")]
        if cells[:2] == ["cell", "seed"]:
            header = cells
            continue
        if header is None or len(cells) != len(header):
            continue
        if set(cells[0]) <= set("-"):
            continue
        cell_id = cells[0].strip("`")
        try:
            seed = int(cells[1])
        except ValueError:
            continue
        values: dict[str, float] = {}
        for name, raw in zip(header[2:], cells[2:]):
            try:
                values[name] = float(raw)
            except ValueError:
                continue
        rows[(cell_id, seed)] = values
    return rows


def parse_passers(text: str) -> dict[int, list[tuple[str, int]]]:
    """The `Worlds passing all six at divisor N:` lines of a page's section 4."""
    out: dict[int, list[tuple[str, int]]] = {}
    for line in text.splitlines():
        marker = "Worlds passing all six at divisor "
        if not line.startswith(marker):
            continue
        head, _, tail = line[len(marker):].partition(":")
        divisor = int(head.strip())
        found: list[tuple[str, int]] = []
        for chunk in tail.strip().rstrip(".").split(","):
            chunk = chunk.strip()
            if not chunk or chunk == "none":
                continue
            cell_id, _, seed = chunk.partition(" seed ")
            found.append((cell_id.strip().strip("`"), int(seed)))
        out[divisor] = found
    return out


# --------------------------------------------------------------------------
# The rerun
# --------------------------------------------------------------------------


def cell_mean(rows: list[dict], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows]))


def logged_mean(table: dict, cell: dict, column: str) -> float:
    """The mean over a cell's seeds of one column of a page already on disk."""
    values = [table[(cell["id"], int(seed))][column]
              for seed in cell["seeds"]
              if (cell["id"], int(seed)) in table
              and column in table[(cell["id"], int(seed))]]
    return float(np.mean(values)) if values else float("nan")


def _fmt(value: float, places: int = 4) -> str:
    value = float(value)
    if np.isnan(value):
        return "n/a"
    return f"{value:.{places}f}"


def build_page(selection: list[dict], config: dict, screen: search.Screen,
               measured: dict[int, list[list[dict]]],
               wall: dict[int, float],
               old_1024: dict, old_512: dict,
               old_passers: dict[int, list[tuple[str, int]]],
               divisors: tuple[int, ...]) -> str:
    lines: list[str] = []
    add = lines.append

    flat = {divisor: [row for rows in measured[divisor] for row in rows]
            for divisor in divisors}
    worlds = len(flat[divisors[0]])

    add(f"# `{RUN_ID}` at 512 px with the drive in kilometres")
    add("")
    add(f"The twenty-cell selection of `WORK_ORDER_C03_9.md` §3.1, each cell "
        f"on its own seeds, rerun at {PIXELS} px with the dials modernized: "
        f"the coarsest mantle wavelength is now 5,120 km at every resolution "
        f"rather than half whatever parent the resolution produced. "
        f"{worlds} worlds per variant, 5 km per pixel, `history_myr` "
        f"{float(config.get('space', {}).get('history_myr', 300.0)):g}, "
        f"`max_cycles` {MAX_CYCLES}, `work_damage` "
        f"{int(config.get('space', {}).get('work_damage', 0))}. Every number "
        "below is a count, a mean, a share or a second.")
    add("")
    add("A world at 512 px is a different world from the same seed at "
        "1024 px — half the parent, a quarter of the area — so the columns "
        "below are paired by dials and by seed, never by world. The 1024-px "
        f"and old 512-px columns are read off `{PAGE_1024}` and "
        f"`{PAGE_512_OLD}`; the old 512-px numbers came from the engine "
        "before this order and cannot be recomputed.")
    add("")

    # -- 1. per cell -------------------------------------------------------
    add("## 1. Per cell, at the three readings")
    add("")
    add("Means over each cell's own worlds. `1024` is the divisor-2 column of "
        f"`{PAGE_1024}`; `512 old` is the divisor-2 column of "
        f"`{PAGE_512_OLD}`, where the drive was 2,560 km; `512 new` is this "
        "rerun at divisor 2, where it is 5,120 km.")
    add("")
    header = ["cell", "seeds", "drive km"]
    for _key, label in COMPARED:
        header += [f"{label} 1024", f"{label} 512 old", f"{label} 512 new"]
    add("| " + " | ".join(header) + " |")
    add("|" + "---|" * len(header))
    for position, cell in enumerate(selection):
        dials = ab_solve.dials_of(cell, config)
        rows = measured[2][position]
        values = [f"`{cell['id']}`", str(len(cell["seeds"])),
                  f"{dials['drive_wavelength_km']:.1f}"]
        for key, _label in COMPARED:
            metric = {"pc": "plate_count", "wf": "weak_final",
                      "ef": "edge_fraction"}[key]
            places = 2 if key == "pc" else 4
            values.append(_fmt(logged_mean(old_1024, cell, f"{key} d2"),
                               places))
            values.append(_fmt(logged_mean(old_512, cell, f"{key} d2"),
                               places))
            values.append(_fmt(cell_mean(rows, metric), places))
        add("| " + " | ".join(values) + " |")
    add("")

    # -- 2. over all worlds ------------------------------------------------
    add("## 2. Over all worlds of the selection")
    add("")
    add("| metric | 1024 | 512 old | 512 new |")
    add("|---|---|---|---|")
    for key, label in COMPARED:
        metric = {"pc": "plate_count", "wf": "weak_final",
                  "ef": "edge_fraction"}[key]
        places = 2 if key == "pc" else 4
        big = [table[f"{key} d2"] for table in old_1024.values()
               if f"{key} d2" in table]
        small = [table[f"{key} d2"] for table in old_512.values()
                 if f"{key} d2" in table]
        new = [float(row[metric]) for row in flat[2]]
        add(f"| mean {label} | {_fmt(np.mean(big), places)} | "
            f"{_fmt(np.mean(small), places)} | {_fmt(np.mean(new), places)} |")
    add("")
    add("| | 1024 | 512 old | 512 new |")
    add("|---|---|---|---|")
    add("| worlds | "
        + f"{len([1 for t in old_1024.values() if 'pc d2' in t])} | "
        + f"{len([1 for t in old_512.values() if 'pc d2' in t])} | "
        + f"{len(flat[2])} |")
    add("")

    # -- 3. distance to the 1024-px reading --------------------------------
    add("## 3. Distance from the 1024-px reading")
    add("")
    add("Per cell, the absolute difference between the 512-px mean and the "
        "1024-px mean of the same cell, then the mean of that over the "
        "twenty cells. Smaller is closer; nothing here says closer is "
        "better.")
    add("")
    add("| metric | mean gap to 1024, old 512 | mean gap to 1024, new 512 "
        "| cells closer under the new drive |")
    add("|---|---|---|---|")
    for key, label in COMPARED:
        metric = {"pc": "plate_count", "wf": "weak_final",
                  "ef": "edge_fraction"}[key]
        old_gaps: list[float] = []
        new_gaps: list[float] = []
        for position, cell in enumerate(selection):
            big = logged_mean(old_1024, cell, f"{key} d2")
            small = logged_mean(old_512, cell, f"{key} d2")
            here = cell_mean(measured[2][position], metric)
            old_gaps.append(abs(big - small))
            new_gaps.append(abs(big - here))
        closer = sum(1 for a, b in zip(old_gaps, new_gaps) if b < a)
        places = 3 if key == "pc" else 4
        add(f"| {label} | {_fmt(np.mean(old_gaps), places)} | "
            f"{_fmt(np.mean(new_gaps), places)} | {closer} of "
            f"{len(selection)} |")
    add("")

    # -- 4. the screen -----------------------------------------------------
    add("## 4. The screen, and the three passes")
    add("")
    add("| | " + " | ".join(f"512 new, divisor {d}" for d in divisors) + " |")
    add("|" + "---|" * (len(divisors) + 1))
    add("| worlds | " + " | ".join(str(len(flat[d])) for d in divisors) + " |")
    add("| worlds passing all six terms | "
        + " | ".join(str(sum(1 for row in flat[d] if row["passed"]))
                     for d in divisors) + " |")
    add("| invalid worlds, residual above tolerance | "
        + " | ".join(str(sum(1 for row in flat[d] if row["invalid"]))
                     for d in divisors) + " |")
    for name, _lo, _hi in search.term_bounds(screen):
        add(f"| worlds passing `{name}` | "
            + " | ".join(str(sum(1 for row in flat[d] if row["terms"][name]))
                         for d in divisors) + " |")
    add("")
    for divisor in divisors:
        passers = [(cell["id"], int(row["seed"]))
                   for position, cell in enumerate(selection)
                   for row in measured[divisor][position] if row["passed"]]
        add(f"Worlds passing all six at divisor {divisor}, this rerun: "
            + (", ".join(f"`{cell_id}` seed {seed}"
                         for cell_id, seed in passers) if passers else "none")
            + ".")
        add("")
    add("The three worlds that passed at 512 px before this order, and what "
        "the same cell and seed did in this rerun:")
    add("")
    add("| divisor | cell | seed | passed before | passed now | terms failed "
        "now |")
    add("|---|---|---|---|---|---|")
    for divisor in sorted(old_passers):
        for cell_id, seed in old_passers[divisor]:
            found = None
            if divisor in measured:
                for position, cell in enumerate(selection):
                    if cell["id"] != cell_id:
                        continue
                    for row in measured[divisor][position]:
                        if int(row["seed"]) == seed:
                            found = row
            if found is None:
                add(f"| {divisor} | `{cell_id}` | {seed} | yes | not rerun | "
                    "n/a |")
                continue
            failed = [name for name, ok in found["terms"].items() if not ok]
            add(f"| {divisor} | `{cell_id}` | {seed} | yes | "
                f"{'yes' if found['passed'] else 'no'} | "
                + (", ".join(f"`{name}`" for name in failed) if failed
                   else "none") + " |")
    add("")

    # -- 5. cost -----------------------------------------------------------
    add("## 5. Cost")
    add("")
    add("| | " + " | ".join(f"divisor {d}" for d in divisors) + " |")
    add("|" + "---|" * (len(divisors) + 1))
    add("| summed world seconds | "
        + " | ".join(f"{sum(float(row['seconds']) for row in flat[d]):.1f}"
                     for d in divisors) + " |")
    add("| wall seconds on the pool | "
        + " | ".join(f"{wall[d]:.1f}" for d in divisors) + " |")
    add("")
    return "\n".join(lines)


def run(*, cells: int = CELLS, pixels: int = PIXELS,
        max_cycles: int = MAX_CYCLES,
        divisors: tuple[int, ...] = (2, 1),
        write: bool = True, executor=None) -> str:
    directory = ab_solve.resolve_run(RUN_ID)
    all_cells, config = ab_solve.load_run(directory)
    screen = ab_solve.screen_of(config)
    scale_km = int(config.get("space", {}).get("scale_km", 5))
    selection = ab_solve.select_cells(all_cells, cells, screen)

    results: dict[int, list[list[dict]]] = {}
    wall: dict[int, float] = {}
    for divisor in divisors:
        worlds, seconds, _used = ab_solve.run_variant(
            selection, config, divisor, pixels=pixels, scale_km=scale_km,
            max_cycles=max_cycles, executor=executor)
        results[divisor] = worlds
        wall[divisor] = seconds
    measured = {divisor: [[ab_solve.measure(world, screen) for world in row]
                          for row in results[divisor]]
                for divisor in divisors}

    old_1024_text = (out_dir() / PAGE_1024).read_text(encoding="utf-8")
    old_512_text = (out_dir() / PAGE_512_OLD).read_text(encoding="utf-8")
    text = build_page(selection, config, screen, measured, wall,
                      parse_world_table(old_1024_text),
                      parse_world_table(old_512_text),
                      parse_passers(old_512_text), divisors)
    if write:
        destination = out_dir() / "c03_10_512_rerun.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
        sys.stderr.write(f"written to {destination}\n")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--divisors", type=int, nargs="+", default=[2, 1],
                        metavar="D",
                        help="solve divisors to run (default 2 1); the order "
                             "asks for 2, and 1 is run as well because two of "
                             "the three passes it asks about were divisor-1 "
                             "passes")
    args = parser.parse_args(argv)
    sys.stdout.write(run(divisors=tuple(args.divisors)) + "\n")
    return 0


__all__ = ["RUN_ID", "build_page", "main", "parse_passers",
           "parse_world_table", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
