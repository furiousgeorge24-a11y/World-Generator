"""Gates on the run-pairing tool of `WORK_ORDER_C03_8.md` §4.

Two synthetic runs are written to a temporary directory with differences
chosen so every number the tool reports can be worked out by hand: three
pairs, one unpaired cell on each side, a rank correlation of exactly -1 in
one run and exactly 0.6 in the other, and two trajectory sheets whose half
steps are known.

Nothing here runs the engine.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

import search  # noqa: E402
from tools import pair_runs  # noqa: E402

def dials_for(index: int) -> dict:
    """A dial set that is a function of one integer, so runs can share them."""
    return {
        "stiffness_fraction": 0.1 + 0.01 * index,
        "yield_percentile": 5.0 + index,
        "heal_time_myr": 100.0 + index,
        "damage_time_myr": 5.0 + index,
        "strength_exponent": 3,
        "strength_spread": 0.05,
        "drive_nodes": 2,
        "drive_shear": 0.5,
    }


def world(seed: int, *, weak_final: float, edge_fraction: float,
          plate_count: int = 1, network_share: float = 0.9,
          weak_peak: float | None = None) -> dict:
    return {
        "seed": seed,
        "weak_final": weak_final,
        "weak_peak": weak_final if weak_peak is None else weak_peak,
        "weak_drift": 0.0,
        "peak_ratio": 1.0,
        "plate_count": plate_count,
        "network_share": network_share,
        "edge_fraction": edge_fraction,
        "residual_max": 5e-4,
        "seconds": 1.0,
        "passed": False,
        "terms": {},
    }


def cell(index: int, dial_index: int, worlds: list[dict], *,
         stage: int = 1, invalid: bool = False, passed: bool = False,
         finding: bool = False) -> dict:
    return {
        "id": f"c{index:05d}",
        "index": index,
        "stage": stage,
        "round": 0,
        "dials": dials_for(dial_index),
        "seeds": [one["seed"] for one in worlds],
        "worlds": worlds,
        "terms": {},
        "pass_count": len(worlds) if passed else 0,
        "passed": passed,
        "invalid": invalid,
        "soft_score": 1.0,
        "seconds": 1.0,
        "world_seconds_mean": 1.0,
        "sheets": ["plates", "trajectory"],
        "finding": finding,
    }


def write_trajectory(path: Path, trajectories: list[list[float]]) -> None:
    """A `trajectory.png` in the sheet's own convention.

    One strip per world, 64 rows tall, two rows of black between strips, one
    filled column per step whose height is the weak fraction, and the half
    mark drawn over row 32 whatever is under it.
    """
    strip_px = pair_runs.STRIP_PX
    gutter = pair_runs.STRIP_GUTTER_PX
    width = max(len(one) for one in trajectories)
    height = len(trajectories) * strip_px + (len(trajectories) - 1) * gutter
    sheet = np.zeros((height, width, 3), dtype=np.uint8)
    for index, fractions in enumerate(trajectories):
        top = index * (strip_px + gutter)
        for column, fraction in enumerate(fractions):
            filled = int(round(min(max(fraction, 0.0), 1.0) * strip_px))
            if filled:
                sheet[top + strip_px - filled:top + strip_px, column] = \
                    pair_runs.STRIP_COLUMN_RGB
        sheet[top + strip_px // 2, :] = pair_runs.STRIP_LINE_RGB
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet, mode="RGB").save(path, format="PNG")


def write_run(root: Path, run_id: str, cells: list[dict], *,
              work_damage: int,
              trajectories: dict[str, list[list[float]]] | None = None) -> Path:
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({
        "screen": {"weak_min": 0.02, "weak_max": 0.25},
        "space": {"work_damage": work_damage},
        "search_seed": 1,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (directory / "cells.jsonl").open("w", encoding="utf-8") as handle:
        for one in cells:
            handle.write(json.dumps(one, sort_keys=True) + "\n")
    for cell_id, rows in (trajectories or {}).items():
        write_trajectory(directory / "cells" / cell_id / "trajectory.png", rows)
    return directory


class Statistics(unittest.TestCase):
    def test_average_ranks_share_a_tie(self) -> None:
        ranks = pair_runs.average_ranks(np.array([10.0, 20.0, 20.0, 30.0]))
        self.assertEqual(list(ranks), [1.0, 2.5, 2.5, 4.0])

    def test_spearman_is_minus_one_on_a_reversed_order(self) -> None:
        self.assertAlmostEqual(
            pair_runs.spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]),
            -1.0, places=12)

    def test_spearman_on_a_hand_computed_case(self) -> None:
        # Ranks 1,2,3,4 against 2,1,4,3: sum of squared rank differences is 4,
        # so rho = 1 - 6*4 / (4 * 15) = 0.6.
        self.assertAlmostEqual(
            pair_runs.spearman([1.0, 2.0, 3.0, 4.0], [20.0, 10.0, 40.0, 30.0]),
            0.6, places=12)

    def test_the_tally_counts_rises_falls_and_ties(self) -> None:
        self.assertEqual(
            pair_runs.tally([0.5, -0.5, 0.0, 1e-9, 2.0]), (2, 1, 2))


class TheTrajectorySheet(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="c03_8_pair_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_sheet_round_trips_to_column_heights(self) -> None:
        path = self.root / "trajectory.png"
        write_trajectory(path, [[0.0, 0.25, 0.75], [0.375, 0.375, 0.375]])
        heights = pair_runs.decode_trajectory(path, 2)
        self.assertEqual(heights, [[0, 16, 48], [24, 24, 24]])

    def test_the_half_mark_line_hides_one_level(self) -> None:
        # The sheet paints its half-mark line over row 32 whatever is under
        # it, so a column filled to exactly 32 is indistinguishable from one
        # filled to 31. Both read back as 31; every other height is exact.
        path = self.root / "line.png"
        write_trajectory(path, [[31 / 64, 32 / 64, 33 / 64]])
        self.assertEqual(pair_runs.decode_trajectory(path, 1),
                         [[31, 31, 33]])

    def test_half_is_the_first_step_that_reaches_it(self) -> None:
        # Final 24; half is 12, first reached at the third step.
        self.assertEqual(pair_runs.half_step([2, 6, 12, 20, 24]), 3.0)
        self.assertEqual(pair_runs.half_step([24, 24, 24]), 1.0)
        self.assertIsNone(pair_runs.half_step([10, 4, 0]))
        self.assertIsNone(pair_runs.half_step([]))

    def test_a_missing_sheet_is_reported_not_guessed(self) -> None:
        self.assertIsNone(
            pair_runs.decode_trajectory(self.root / "nothing.png", 2))


class ThePairing(unittest.TestCase):
    """Three pairs, one unpaired cell on each side, known differences."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="c03_8_pair_"))

        # Control stage-1 cells 0..3 on dial sets 0..3. The cell means of
        # `edge_fraction` and `weak_final` run in exactly opposite order, so
        # the rank correlation is -1.
        control_cells = [
            cell(0, 0, [world(1, weak_final=0.10, edge_fraction=0.40)]),
            cell(1, 1, [world(1, weak_final=0.20, edge_fraction=0.30)]),
            cell(2, 2, [world(1, weak_final=0.30, edge_fraction=0.20)]),
            cell(3, 3, [world(1, weak_final=0.40, edge_fraction=0.10)]),
            # A stage-2 cell, which is never paired.
            cell(4, 9, [world(1, weak_final=0.05, edge_fraction=0.55)],
                 stage=2),
        ]
        # Treatment keeps dial sets 0, 1, 2 and swaps 3 for 4, so dial set 3
        # is unpaired on the control side and 4 on the treatment side.
        treatment_cells = [
            cell(0, 0, [world(1, weak_final=0.20, edge_fraction=0.40,
                              plate_count=2)]),
            cell(1, 1, [world(1, weak_final=0.30, edge_fraction=0.30)]),
            cell(2, 2, [world(1, weak_final=0.30, edge_fraction=0.10)]),
            cell(3, 4, [world(1, weak_final=0.40, edge_fraction=0.20)],
                 invalid=True),
        ]
        self.control_dir = write_run(
            self.root, "20200101T000000Z-s1", control_cells, work_damage=0,
            trajectories={
                "c00000": [[0.01, 0.02, 0.05, 0.09, 0.10]],
                "c00001": [[0.02, 0.05, 0.10, 0.16, 0.20]],
                "c00002": [[0.05, 0.10, 0.20, 0.28, 0.30]],
                "c00003": [[0.10, 0.20, 0.30, 0.36, 0.40]],
            })
        self.treatment_dir = write_run(
            self.root, "20200101T010000Z-s1", treatment_cells, work_damage=1,
            trajectories={
                "c00000": [[0.10, 0.16, 0.20, 0.20, 0.20]],
                "c00001": [[0.16, 0.24, 0.28, 0.30, 0.30]],
                "c00002": [[0.02, 0.05, 0.16, 0.28, 0.30]],
                "c00003": [[0.20, 0.30, 0.36, 0.40, 0.40]],
            })
        self.control = pair_runs.Run(self.control_dir)
        self.treatment = pair_runs.Run(self.treatment_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_the_runs_carry_their_law(self) -> None:
        self.assertEqual(self.control.work_damage, 0)
        self.assertEqual(self.treatment.work_damage, 1)

    def test_a_legacy_run_is_modernized_on_read(self) -> None:
        # Every run on disk records `drive_nodes`, a fraction of its own
        # parent world. `Run` hands back the wavelength that meant at the
        # run's own geometry — 1024 px and 5 km/px by default, a 10,240 km
        # parent — so both sides pair by physical dial values.
        raw = json.loads((self.control_dir / "cells.jsonl").read_text(
            encoding="utf-8").splitlines()[0])
        self.assertEqual(raw["dials"]["drive_nodes"], 2)
        dials = self.control.cells[0]["dials"]
        self.assertEqual(dials["drive_wavelength_km"], 5120.0)
        self.assertNotIn("drive_nodes", dials)
        self.assertEqual(
            search.modernize_dials(raw["dials"], 1024, 5)
            ["drive_wavelength_km"], 5120.0)
        # And the key the pairing is built on carries the new name.
        self.assertIn("drive_wavelength_km",
                      dict(pair_runs.dial_key(dials)))

    def test_only_stage_one_cells_are_paired(self) -> None:
        pairs, lonely_control, lonely_treatment = pair_runs.pair_cells(
            self.control.stage1, self.treatment.stage1)
        self.assertEqual(len(pairs), 3)
        self.assertEqual([one["id"] for one in lonely_control], ["c00003"])
        self.assertEqual([one["id"] for one in lonely_treatment], ["c00003"])
        # Every stage-1 cell of both runs is in exactly one of the three
        # lists: nothing is dropped.
        self.assertEqual(len(pairs) + len(lonely_control),
                         len(self.control.stage1))
        self.assertEqual(len(pairs) + len(lonely_treatment),
                         len(self.treatment.stage1))
        self.assertEqual(len(self.control.stage1), 4)

    def test_a_cell_whose_dials_differ_in_the_twelfth_place_is_unpaired(self) -> None:
        moved = [dict(one) for one in self.treatment.stage1]
        moved[0] = dict(moved[0])
        dials = dict(moved[0]["dials"])
        dials["heal_time_myr"] = dials["heal_time_myr"] + 1e-9
        moved[0]["dials"] = dials
        pairs, lonely_control, lonely_treatment = pair_runs.pair_cells(
            self.control.stage1, moved)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(lonely_control), 2)
        self.assertEqual(len(lonely_treatment), 2)

    def test_the_report_counts_the_cells_and_the_unpaired(self) -> None:
        text = pair_runs.build_report(self.control, self.treatment)
        self.assertIn("| stage-1 cells paired | 3 | 3 |", text)
        self.assertIn("| stage-1 cells unpaired | 1 | 1 |", text)
        self.assertIn("| invalid cells | 0 | 1 |", text)
        self.assertIn("Unpaired stage-1 cells", text)
        # The unpaired cells are named, not merely counted away.
        self.assertIn("control `c00003`", text)
        self.assertIn("treatment `c00003`", text)

    def test_the_paired_medians_are_the_hand_computed_ones(self) -> None:
        pairs, _, _ = pair_runs.pair_cells(self.control.stage1,
                                           self.treatment.stage1)
        weak = [pair_runs.cell_mean(after, "weak_final")
                - pair_runs.cell_mean(before, "weak_final")
                for before, after in pairs]
        # 0.20-0.10, 0.30-0.20, 0.30-0.30
        self.assertAlmostEqual(pair_runs.median(weak), 0.1, places=12)
        self.assertEqual(pair_runs.tally(weak), (2, 0, 1))
        edge = [pair_runs.cell_mean(after, "edge_fraction")
                - pair_runs.cell_mean(before, "edge_fraction")
                for before, after in pairs]
        # 0.40-0.40, 0.30-0.30, 0.10-0.20
        self.assertAlmostEqual(pair_runs.median(edge), 0.0, places=12)
        self.assertEqual(pair_runs.tally(edge), (0, 1, 2))
        text = pair_runs.build_report(self.control, self.treatment)
        self.assertIn("| `weak_final` | 0.100000 | 2 | 0 | 1 |", text)
        self.assertIn("| `edge_fraction` | 0.000000 | 0 | 1 | 2 |", text)

    def test_the_rank_correlations_are_the_hand_computed_ones(self) -> None:
        cells = self.control.stage1
        self.assertAlmostEqual(
            pair_runs.spearman(
                [pair_runs.cell_mean(one, "edge_fraction") for one in cells],
                [pair_runs.cell_mean(one, "weak_final") for one in cells]),
            -1.0, places=12)
        text = pair_runs.build_report(self.control, self.treatment)
        self.assertIn("| `20200101T000000Z-s1` | 4 | -1.0000 |", text)

    def test_the_band_statistics_read_the_screen_off_the_config(self) -> None:
        stats = pair_runs.band_stats(self.control)
        # Every world with a final weak fraction in [0.02, 0.25]: 0.10, 0.20
        # from stage 1 and 0.05 from the stage-2 cell.
        self.assertEqual(stats["worlds"], 3)
        self.assertEqual(stats["total"], 5)
        self.assertAlmostEqual(stats["p50"], 0.40, places=12)
        self.assertAlmostEqual(stats["max"], 0.55, places=12)
        self.assertAlmostEqual(stats["one_plate"], 1.0, places=12)
        self.assertAlmostEqual(stats["two_or_more"], 0.0, places=12)
        treated = pair_runs.band_stats(self.treatment)
        self.assertEqual(treated["worlds"], 1)
        self.assertAlmostEqual(treated["two_or_more"], 1.0, places=12)

    def test_time_to_half_is_read_off_the_sheets(self) -> None:
        pairs, _, _ = pair_runs.pair_cells(self.control.stage1,
                                           self.treatment.stage1)
        subset = [(before, after) for before, after in pairs
                  if pair_runs.cell_mean(before, "weak_final") >= 0.1]
        self.assertEqual(len(subset), 3)
        control_steps = [pair_runs.cell_half_step(self.control, before)
                         for before, _ in subset]
        treatment_steps = [pair_runs.cell_half_step(self.treatment, after)
                           for _, after in subset]
        # Control c00000 is 0.01, 0.02, 0.05, 0.09, 0.10 of the sheet, which
        # is 1, 1, 3, 6, 6 sixty-fourths. The final is 6, half of it is 3, and
        # the third step is the first to reach 3.
        self.assertEqual(control_steps, [3.0, 4.0, 3.0])
        self.assertEqual(treatment_steps, [2.0, 1.0, 3.0])
        text = pair_runs.build_report(self.control, self.treatment)
        self.assertIn("| pairs with control mean `weak_final` >= 0.1 | 3 |",
                      text)
        self.assertIn("| median difference in steps, treatment minus control "
                      "| -1.000 |", text)
        self.assertIn("| earlier under the treatment | 2 |", text)
        self.assertIn("| later under the treatment | 0 |", text)
        self.assertIn("| equal | 1 |", text)

    def test_the_report_writes_a_file_and_prints_it(self) -> None:
        destination = self.root / "pair.md"
        printed, noise = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(printed),                 contextlib.redirect_stderr(noise):
            code = pair_runs.main([str(self.control_dir),
                                   str(self.treatment_dir),
                                   "--out", str(destination)])
        self.assertEqual(code, 0)
        self.assertIn("# Paired runs:", printed.getvalue())
        self.assertIn(str(destination), noise.getvalue())
        text = destination.read_text(encoding="utf-8")
        self.assertIn("# Paired runs:", text)
        self.assertIn("## 6. Passers, findings and throughput", text)

    def test_the_default_output_path_names_both_runs(self) -> None:
        path = pair_runs.output_path(self.control, self.treatment)
        self.assertEqual(
            path.name,
            "pair_20200101T000000Z-s1_20200101T010000Z-s1.md")


if __name__ == "__main__":
    unittest.main()
