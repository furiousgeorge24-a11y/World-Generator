"""Gates on the A/B solve tool of `WORK_ORDER_C03_9.md` §3.

Three things are worth holding: that the selection rule picks the cells the
order describes and in the order it describes them, that the width shares read
off a hand-built mask are the numbers a hand count gives, and that the
determinism gate fires when a rerun world does not reproduce the run's own
record. None of them runs the engine.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

import search  # noqa: E402
from tools import ab_solve  # noqa: E402

TERMS = tuple(name for name, _lo, _hi in search.term_bounds(search.Screen()))


def world_record(seed: int, failing: tuple[str, ...] = ()) -> dict:
    """One logged world, passing every term but the named ones."""
    return {
        "seed": seed,
        "edge_fraction": 0.3,
        "weak_final": 0.15,
        "weak_peak": 0.18,
        "weak_drift": 0.01,
        "peak_ratio": 1.2,
        "plate_count": 4,
        "network_share": 0.8,
        "residual_max": 5e-4,
        "seconds": 3.0,
        "passed": not failing,
        "terms": {name: name not in failing for name in TERMS},
    }


def cell_record(index: int, *, stage: int = 1, seeds: int = 4,
                soft: float | None = 1.0, invalid: bool = False,
                failing: tuple[tuple[str, ...], ...] = ()) -> dict:
    """One logged cell. `failing` names the failed terms of each world."""
    if not failing:
        failing = ((),) * seeds
    worlds = [world_record(1000 + index * 100 + position, failing[position])
              for position in range(seeds)]
    return {
        "id": f"c{index:05d}",
        "index": index,
        "stage": stage,
        "round": 0,
        "dials": {
            "stiffness_fraction": 0.1 + index * 0.01,
            "yield_percentile": 5.0,
            "heal_time_myr": 10.0,
            "damage_time_myr": 1.0,
            "strength_exponent": 2,
            "strength_spread": 0.05,
            "drive_nodes": 1,
            "drive_shear": 0.5,
        },
        "seeds": [world["seed"] for world in worlds],
        "worlds": worlds,
        "soft_score": soft,
        "invalid": invalid,
        "passed": False,
        "pass_count": 0,
        "sheets": ["plates", "trajectory"],
        "finding": False,
    }


class TheSelectionRule(unittest.TestCase):
    """Edge-only failures first, ties by soft score, then soft score."""

    def setUp(self) -> None:
        edge = ("edge_fraction",)
        both = ("edge_fraction", "plate_count")
        self.cells = [
            # two edge-only failures, poor soft score
            cell_record(0, soft=9.0, failing=(edge, edge, (), ())),
            # one edge-only failure, best soft score of the three
            cell_record(1, soft=0.1, failing=(edge, (), (), ())),
            # two edge-only failures, better soft score than cell 0
            cell_record(2, soft=2.0, failing=(edge, edge, (), ())),
            # fails on two terms, so not an edge-only failure at all
            cell_record(3, soft=0.2, failing=(both, both, both, both)),
            # nothing fails
            cell_record(4, soft=0.3),
        ]
        self.screen = search.Screen()

    def test_edge_only_failures_are_counted_per_world(self) -> None:
        counts = [ab_solve.edge_only_failures(cell, self.screen)
                  for cell in self.cells]
        self.assertEqual(counts, [2, 1, 2, 0, 0])

    def test_the_top_cells_are_ranked_by_that_count_then_soft_score(self) -> None:
        chosen = ab_solve.select_cells(self.cells, 3, self.screen,
                                       top_by_edge=3)
        self.assertEqual([cell["id"] for cell in chosen],
                         ["c00002", "c00000", "c00001"])

    def test_the_rest_is_filled_in_by_soft_score(self) -> None:
        chosen = ab_solve.select_cells(self.cells, 5, self.screen,
                                       top_by_edge=2)
        # The top two by edge-only count, then the best soft scores that are
        # not already taken: c00001 (0.1), c00003 (0.2), c00004 (0.3).
        self.assertEqual([cell["id"] for cell in chosen],
                         ["c00002", "c00000", "c00001", "c00003", "c00004"])

    def test_no_cell_is_taken_twice(self) -> None:
        chosen = ab_solve.select_cells(self.cells, 5, self.screen)
        self.assertEqual(len({cell["id"] for cell in chosen}), len(chosen))
        self.assertEqual(len(chosen), 5)

    def test_an_invalid_cell_is_never_taken(self) -> None:
        cells = self.cells + [cell_record(5, soft=None, invalid=True)]
        chosen = ab_solve.select_cells(cells, 6, self.screen)
        self.assertNotIn("c00005", [cell["id"] for cell in chosen])
        self.assertEqual(len(chosen), 5)

    def test_it_reads_a_run_off_disk(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw) / "20260101T000000Z-s1"
            directory.mkdir()
            with (directory / "cells.jsonl").open("w", encoding="utf-8") as fh:
                for cell in self.cells:
                    fh.write(json.dumps(cell, sort_keys=True) + "\n")
            (directory / "config.json").write_text(
                json.dumps(search.SearchConfig().to_json()), encoding="utf-8")
            self.assertEqual(ab_solve.resolve_run(str(directory)), directory)
            cells, config = ab_solve.load_run(directory)
            self.assertEqual(len(cells), len(self.cells))
            self.assertEqual(ab_solve.screen_of(config), search.Screen())


class TheWidthShares(unittest.TestCase):
    """Hand-built masks on a 16-cell torus, with hand-counted answers."""

    N = 16

    def line(self, first_row: int, width: int) -> np.ndarray:
        weak = np.zeros((self.N, self.N), dtype=bool)
        for offset in range(width):
            weak[(first_row + offset) % self.N, :] = True
        return weak

    def test_a_two_wide_line_is_two_wide_and_not_three(self) -> None:
        shares = ab_solve.width_shares(self.line(0, 2))
        self.assertEqual(shares["k2"], 1.0)
        self.assertEqual(shares["k3"], 0.0)
        self.assertEqual(shares["k4"], 0.0)

    def test_a_three_wide_line_is_three_wide_and_not_four(self) -> None:
        shares = ab_solve.width_shares(self.line(0, 3))
        self.assertEqual(shares["k2"], 1.0)
        self.assertEqual(shares["k3"], 1.0)
        self.assertEqual(shares["k4"], 0.0)

    def test_a_six_wide_line_reaches_six_and_not_seven(self) -> None:
        shares = ab_solve.width_shares(self.line(0, 6))
        self.assertEqual(shares["k6"], 1.0)
        self.assertEqual(shares["k7"], 0.0)

    def test_a_block_aligned_two_wide_line_is_aligned(self) -> None:
        self.assertEqual(ab_solve.width_shares(self.line(0, 2))["aligned"],
                         1.0)
        self.assertEqual(ab_solve.width_shares(self.line(4, 2))["aligned"],
                         1.0)

    def test_a_line_shifted_by_one_cell_is_not_aligned(self) -> None:
        self.assertEqual(ab_solve.width_shares(self.line(1, 2))["aligned"],
                         0.0)

    def test_a_single_cell_is_not_two_wide(self) -> None:
        weak = np.zeros((self.N, self.N), dtype=bool)
        weak[3, 3] = True
        shares = ab_solve.width_shares(weak)
        self.assertEqual(shares["k2"], 0.0)
        self.assertEqual(shares["aligned"], 0.0)

    def test_a_square_is_covered_in_part(self) -> None:
        # A 4 x 4 block: every cell is in some fully weak 2 x 2 square, and
        # every cell is in the one 4 x 4 square, but no 5 x 5 square is weak.
        weak = np.zeros((self.N, self.N), dtype=bool)
        weak[2:6, 2:6] = True
        shares = ab_solve.width_shares(weak)
        self.assertEqual(shares["k4"], 1.0)
        self.assertEqual(shares["k5"], 0.0)
        self.assertEqual(shares["aligned"], 1.0)

    def test_the_squares_wrap_on_the_torus(self) -> None:
        weak = self.line(self.N - 1, 2)          # rows 15 and 0
        self.assertEqual(ab_solve.width_shares(weak)["k2"], 1.0)
        self.assertEqual(ab_solve.width_shares(weak)["aligned"], 0.0)

    def test_an_empty_mask_has_no_shares(self) -> None:
        self.assertIsNone(
            ab_solve.width_shares(np.zeros((self.N, self.N), dtype=bool)))

    def test_the_shares_never_rise_with_k(self) -> None:
        rng = np.random.default_rng(3)
        weak = rng.random((self.N, self.N)) < 0.4
        shares = ab_solve.width_shares(weak)
        values = [shares[f"k{k}"] for k in ab_solve.WIDTH_K]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_odd_width_counts_the_gap_between_three_and_four(self) -> None:
        rows = [{"width": {"k3": 0.9, "k4": 0.1}},
                {"width": {"k3": 0.5, "k4": 0.5}},
                {"width": None}]
        self.assertEqual(ab_solve.odd_width_count(rows), (1, 2))


class StubWorld:
    """A worker result shaped like `explore_worker.run_one_world` returns."""

    @staticmethod
    def make(seed: int, weak_cells: int, n: int = 8) -> dict:
        strength = np.ones((n, n), dtype=np.float64) * 0.9
        flat = strength.reshape(-1)
        flat[:weak_cells] = 0.1
        fractions = [0.05, 0.1, weak_cells / (n * n)]
        return {
            "seed": seed,
            "step_myr": 4.0,
            "strength_final": strength,
            "weak_fraction": fractions,
            "plate_percent": [50.0, 30.0],
            "solver_cycles": [4, 5, 6],
            "solver_residual": [1e-4, 2e-4, 3e-4],
            "exhausted_steps": 0,
            "seconds": 2.0,
        }


class TheDeterminismGate(unittest.TestCase):
    """The divisor-2 rerun must reproduce the run's own logged metrics."""

    def setUp(self) -> None:
        self.screen = search.Screen()
        self.worlds = [StubWorld.make(11, 6), StubWorld.make(12, 9)]
        logged = [search.world_metrics(world, self.screen)
                  for world in self.worlds]
        self.cell = {
            "id": "c00000",
            "index": 0,
            "seeds": [11, 12],
            "worlds": [dict(row) for row in logged],
        }

    def test_a_faithful_rerun_passes(self) -> None:
        outcome = ab_solve.check_determinism([self.cell], [self.worlds],
                                             self.screen)
        self.assertEqual(outcome["checked"], 2)
        self.assertEqual(max(outcome["maxima"].values()), 0.0)
        for name in ab_solve.GATE_METRICS:
            self.assertIn(name, outcome["maxima"])

    def test_the_maxima_carry_the_largest_difference_seen(self) -> None:
        before = float(self.cell["worlds"][1]["network_share"])
        self.cell["worlds"][1]["network_share"] = before + 3e-13
        moved = self.cell["worlds"][1]["network_share"] - before
        outcome = ab_solve.check_determinism([self.cell], [self.worlds],
                                             self.screen)
        self.assertEqual(outcome["maxima"]["network_share"], moved)
        self.assertGreater(moved, 0.0)
        self.assertLess(moved, ab_solve.GATE_TOLERANCE)
        # The metrics that did not move are still reported, at zero.
        self.assertEqual(outcome["maxima"]["edge_fraction"], 0.0)

    def test_a_metric_off_by_more_than_the_tolerance_raises(self) -> None:
        self.cell["worlds"][1]["edge_fraction"] += 1e-6
        with self.assertRaises(ab_solve.DeterminismError) as caught:
            ab_solve.check_determinism([self.cell], [self.worlds], self.screen)
        self.assertIn("edge_fraction", str(caught.exception))
        self.assertIn("seed 12", str(caught.exception))

    def test_a_metric_inside_the_tolerance_passes(self) -> None:
        self.cell["worlds"][0]["network_share"] += 1e-12
        ab_solve.check_determinism([self.cell], [self.worlds], self.screen)

    def test_a_seed_out_of_order_raises(self) -> None:
        swapped = [self.worlds[1], self.worlds[0]]
        with self.assertRaises(ab_solve.DeterminismError) as caught:
            ab_solve.check_determinism([self.cell], [swapped], self.screen)
        self.assertIn("seed", str(caught.exception))

    def test_a_missing_world_raises(self) -> None:
        with self.assertRaises(ab_solve.DeterminismError) as caught:
            ab_solve.check_determinism([self.cell], [self.worlds[:1]],
                                       self.screen)
        self.assertIn("logged 2 worlds", str(caught.exception))


class TheMeasurement(unittest.TestCase):
    def test_it_carries_the_screen_verdict_and_the_solver_effort(self) -> None:
        row = ab_solve.measure(StubWorld.make(11, 6), search.Screen())
        self.assertIn("passed", row)
        self.assertIn("terms", row)
        self.assertEqual(row["cycles_max"], 6)
        self.assertAlmostEqual(row["cycles_mean"], 5.0, places=12)
        self.assertFalse(row["invalid"])
        self.assertEqual(row["weak_cells"], 6)
        self.assertIsNotNone(row["width"])

    def test_a_world_above_tolerance_is_invalid(self) -> None:
        world = StubWorld.make(11, 6)
        world["solver_residual"] = [1e-4, 2e-2]
        self.assertTrue(ab_solve.measure(world, search.Screen())["invalid"])

    def test_the_mean_width_skips_worlds_with_no_weak_set(self) -> None:
        rows = [ab_solve.measure(StubWorld.make(11, 6), search.Screen()),
                ab_solve.measure(StubWorld.make(12, 0), search.Screen())]
        shares, count = ab_solve.mean_width(rows)
        self.assertEqual(count, 1)
        self.assertIn("aligned", shares)


class TheParamsRecord(unittest.TestCase):
    def test_the_divisor_and_the_budget_come_from_the_flags(self) -> None:
        cell = cell_record(0)
        config = {"space": {"work_damage": 1, "history_myr": 300.0,
                            "pixels": 1024, "scale_km": 5}}
        for divisor in (1, 2):
            record = ab_solve.params_record(cell, config, divisor, 80)
            with self.subTest(divisor=divisor):
                self.assertEqual(record["solve_divisor"], divisor)
                self.assertEqual(record["max_cycles"], 80)
                self.assertEqual(record["work_damage"], 1)
                self.assertEqual(record["history_myr"], 300.0)
                for name in cell["dials"]:
                    if name == "drive_nodes":
                        continue
                    self.assertAlmostEqual(record[name], cell["dials"][name],
                                           places=12)

    def test_a_legacy_run_is_modernized_on_read(self) -> None:
        # `cell_record` writes the `drive_nodes` every run on disk carries.
        # At 1024 px and 5 km/px the parent is 10,240 km, so a node count of
        # one is a 10,240 km wavelength, and the record the rerun runs on
        # carries that and no node count at all.
        cell = cell_record(0)
        config = {"space": {"pixels": 1024, "scale_km": 5}}
        self.assertEqual(cell["dials"]["drive_nodes"], 1)
        dials = ab_solve.dials_of(cell, config)
        self.assertEqual(dials["drive_wavelength_km"], 10240.0)
        self.assertNotIn("drive_nodes", dials)
        record = ab_solve.params_record(cell, config, 2, 80)
        self.assertEqual(record["drive_wavelength_km"], 10240.0)
        self.assertNotIn("drive_nodes", record)

    def test_the_conversion_uses_the_run_geometry_not_the_rerun(self) -> None:
        # A 1024-px run reread for a 512-px rerun keeps the 1024-px meaning
        # of its node count, which is the whole point of the order.
        cell = cell_record(0)
        self.assertEqual(
            ab_solve.run_geometry({"space": {"pixels": 1024, "scale_km": 5}}),
            (1024, 5))
        dials = ab_solve.dials_of(
            cell, {"space": {"pixels": 1024, "scale_km": 5}})
        self.assertEqual(dials["drive_wavelength_km"], 10240.0)

    def test_the_two_variants_differ_only_in_the_divisor(self) -> None:
        cell = cell_record(0)
        config = {"space": {"work_damage": 1}}
        one = ab_solve.params_record(cell, config, 1, 80)
        two = ab_solve.params_record(cell, config, 2, 80)
        self.assertEqual({name: value for name, value in one.items()
                          if name != "solve_divisor"},
                         {name: value for name, value in two.items()
                          if name != "solve_divisor"})


class ThePaths(unittest.TestCase):
    def test_the_page_and_the_sheets_take_the_tag(self) -> None:
        self.assertEqual(ab_solve.output_path("r").name, "ab_solve_r.md")
        self.assertEqual(ab_solve.output_path("r", "512px").name,
                         "ab_solve_r_512px.md")
        self.assertEqual(ab_solve.sheet_dir_for("r").name, "r")
        self.assertEqual(ab_solve.sheet_dir_for("r", "512px").name, "r_512px")


if __name__ == "__main__":
    unittest.main()
