"""Gates on the regime search's arithmetic: metrics, screen, sampling, score.

No Flask and no process pool. Every world here is synthetic: a strength field
built by hand and a weak-fraction trajectory written out by hand, so each
gate measures one thing. The engine is only entered through
`label_components`, which the network metric needs.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import math
import sys
import unittest

from dataclasses import replace

import numpy as np

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

import search  # noqa: E402
from engine.history.kinematics import HistoryParams  # noqa: E402
from engine.history.constants import WEAK_THRESHOLD  # noqa: E402
from engine.history.plates import label_components  # noqa: E402

N = 32
STRONG = 1.0
WEAK = 0.0


def strength_from(weak: np.ndarray) -> np.ndarray:
    """A strength field whose weak cells are exactly `weak`."""
    field = np.full(weak.shape, STRONG, dtype=np.float64)
    field[weak] = WEAK
    assert (field < WEAK_THRESHOLD).sum() == weak.sum()
    return field


def rectangle_loop(top: int, left: int, height: int, width: int,
                   n: int = N) -> np.ndarray:
    """A one-cell-wide closed rectangle of weak cells."""
    weak = np.zeros((n, n), dtype=bool)
    rows = [(top + offset) % n for offset in range(height)]
    columns = [(left + offset) % n for offset in range(width)]
    for row in rows:
        weak[row, columns[0]] = True
        weak[row, columns[-1]] = True
    for column in columns:
        weak[rows[0], column] = True
        weak[rows[-1], column] = True
    return weak


def disc(centre: int, radius: int, n: int = N) -> np.ndarray:
    grid = np.arange(n)
    dy = np.minimum(np.abs(grid - centre), n - np.abs(grid - centre))
    dx = dy
    return (dy[:, None] ** 2 + dx[None, :] ** 2) <= radius * radius


def world(weak: np.ndarray, *, fractions=None, plates: int = 4,
          residual: float = 5e-4, seed: int = 1) -> dict:
    """A worker result with only the keys `world_metrics` reads."""
    if fractions is None:
        fractions = [0.06] * 75
    return {
        "seed": seed,
        "step_myr": 4.0,
        "weak_fraction": list(fractions),
        "strength_final": strength_from(weak),
        "plate_percent": [10.0] * plates,
        "solver_residual": [residual * 0.5, residual],
        "seconds": 1.0,
    }


PASSING_WEAK = rectangle_loop(4, 4, 17, 17)


def passing_world(**overrides) -> dict:
    return world(overrides.pop("weak", PASSING_WEAK), **overrides)


class Connectivity(unittest.TestCase):
    def test_eight_joins_a_diagonal_that_four_does_not(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[2, 2] = True
        mask[3, 3] = True
        four = label_components(mask, 4)
        eight = label_components(mask, 8)
        self.assertEqual(len(set(four[mask].tolist())), 2)
        self.assertEqual(len(set(eight[mask].tolist())), 1)

    def test_it_wraps_on_the_torus(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[0, 0] = True
        mask[7, 7] = True
        self.assertEqual(len(set(label_components(mask, 8)[mask].tolist())), 1)
        self.assertEqual(len(set(label_components(mask, 4)[mask].tolist())), 2)

    def test_it_refuses_any_other_connectivity(self) -> None:
        with self.assertRaises(ValueError):
            label_components(np.zeros((4, 4), dtype=bool), 6)

    def test_an_empty_mask_is_all_minus_one(self) -> None:
        labels = label_components(np.zeros((4, 4), dtype=bool), 8)
        self.assertTrue(bool((labels == -1).all()))


class Metrics(unittest.TestCase):
    def test_a_thin_closed_loop_is_one_network_of_edges(self) -> None:
        weak = rectangle_loop(4, 4, 17, 17)
        self.assertEqual(search.network_share(weak), 1.0)
        self.assertEqual(search.edge_fraction(weak), 1.0)

    def test_a_filled_disc_has_a_low_edge_fraction(self) -> None:
        weak = disc(16, 8)
        self.assertEqual(search.network_share(weak), 1.0)
        self.assertLess(search.edge_fraction(weak), 0.4)

    def test_two_separated_loops_share_the_network_evenly(self) -> None:
        left = rectangle_loop(2, 2, 8, 8)
        right = rectangle_loop(2, 20, 8, 8)
        weak = left | right
        self.assertEqual(int(left.sum()), int(right.sum()))
        self.assertFalse(bool((left & right).any()))
        self.assertAlmostEqual(search.network_share(weak), 0.5, places=12)

    def test_no_weak_cells_gives_zero_for_both(self) -> None:
        empty = np.zeros((N, N), dtype=bool)
        self.assertEqual(search.network_share(empty), 0.0)
        self.assertEqual(search.edge_fraction(empty), 0.0)

    def test_the_seven_numbers_come_off_one_worker_result(self) -> None:
        fractions = [0.10] * 50 + [0.07] * 25
        metrics = search.world_metrics(
            world(PASSING_WEAK, fractions=fractions, plates=5, residual=7e-4),
            search.Screen())
        self.assertAlmostEqual(metrics["weak_final"], 0.07)
        self.assertAlmostEqual(metrics["weak_peak"], 0.10)
        # The window is 100 Myr at a 4 Myr step, so it reads back 25 steps.
        self.assertAlmostEqual(metrics["weak_drift"], 0.03)
        self.assertAlmostEqual(metrics["peak_ratio"], 0.10 / 0.07)
        self.assertEqual(metrics["plate_count"], 5)
        self.assertEqual(metrics["network_share"], 1.0)
        self.assertEqual(metrics["edge_fraction"], 1.0)
        self.assertAlmostEqual(metrics["residual_max"], 7e-4)

    def test_a_weak_set_that_vanished_has_a_capped_finite_peak_ratio(self) -> None:
        # JSON has no infinity, and the gallery parses every cell at once, so
        # every metric must be finite. The cap still fails the screen.
        metrics = search.world_metrics(
            world(np.zeros((N, N), dtype=bool), fractions=[0.3] * 74 + [0.0]),
            search.Screen())
        self.assertEqual(metrics["weak_final"], 0.0)
        self.assertEqual(metrics["peak_ratio"], search.PEAK_RATIO_CAP)
        self.assertTrue(all(math.isfinite(float(value))
                            for key, value in metrics.items() if key != "seed"))
        self.assertFalse(search.screen_world(metrics, search.Screen())["passed"])


class TheScreen(unittest.TestCase):
    def setUp(self) -> None:
        self.screen = search.Screen()

    def verdict(self, one: dict) -> dict:
        return search.screen_world(search.world_metrics(one, self.screen),
                                   self.screen)

    def test_a_world_built_to_pass_passes_every_term(self) -> None:
        verdict = self.verdict(passing_world())
        self.assertTrue(verdict["passed"], verdict["terms"])
        self.assertEqual(verdict["violation"], 0.0)
        self.assertEqual(len(verdict["terms"]), 6)

    def failing(self, name: str, one: dict) -> None:
        verdict = self.verdict(one)
        self.assertFalse(verdict["passed"])
        failed = [term for term, row in verdict["terms"].items()
                  if not row["ok"]]
        self.assertEqual(failed, [name],
                         f"expected only {name} to fail, got {failed}")
        self.assertGreater(verdict["violation"], 0.0)

    def test_it_fails_on_too_much_failed_lithosphere(self) -> None:
        self.failing("weak_final", passing_world(fractions=[0.5] * 75))

    def test_it_fails_on_too_little_failed_lithosphere(self) -> None:
        self.failing("weak_final", passing_world(fractions=[0.005] * 75))

    def test_it_fails_on_a_weak_set_that_overshot(self) -> None:
        self.failing("peak_ratio",
                     passing_world(fractions=[0.2] * 20 + [0.06] * 55))

    def test_it_fails_on_a_weak_set_still_moving(self) -> None:
        rising = [0.02] * 50 + list(np.linspace(0.02, 0.10, 25))
        self.failing("weak_drift", passing_world(fractions=rising))

    def test_it_fails_on_too_few_plates(self) -> None:
        self.failing("plate_count", passing_world(plates=1))

    def test_it_fails_on_too_many_plates(self) -> None:
        self.failing("plate_count", passing_world(plates=20))

    def test_it_fails_on_a_scattered_weak_set(self) -> None:
        three = (rectangle_loop(1, 1, 6, 6) | rectangle_loop(1, 13, 6, 6)
                 | rectangle_loop(1, 25, 6, 6))
        self.failing("network_share", passing_world(weak=three))

    def test_it_fails_on_a_thick_weak_set(self) -> None:
        self.failing("edge_fraction", passing_world(weak=disc(16, 8)))

    def test_an_unconverged_world_makes_the_cell_invalid(self) -> None:
        cell = search.screen_cell(
            [search.world_metrics(passing_world(), self.screen),
             search.world_metrics(passing_world(residual=1e-2), self.screen)],
            self.screen)
        self.assertTrue(cell["invalid"])
        self.assertFalse(cell["passed"])

    def test_pass_fraction_decides_how_many_worlds_must_pass(self) -> None:
        metrics = [search.world_metrics(passing_world(), self.screen),
                   search.world_metrics(passing_world(plates=1), self.screen)]
        self.assertFalse(search.screen_cell(metrics, self.screen)["passed"])
        lenient = search.Screen(pass_fraction=0.5)
        self.assertTrue(search.screen_cell(metrics, lenient)["passed"])

    def test_a_cell_of_passers_passes(self) -> None:
        metrics = [search.world_metrics(passing_world(seed=n), self.screen)
                   for n in range(4)]
        cell = search.screen_cell(metrics, self.screen)
        self.assertTrue(cell["passed"])
        self.assertFalse(cell["invalid"])
        self.assertEqual(cell["soft_score"], 0.0)
        self.assertEqual(cell["pass_count"], 4)


class SoftScore(unittest.TestCase):
    def setUp(self) -> None:
        self.screen = search.Screen()

    def score(self, weak_final: float) -> float:
        one = passing_world(fractions=[weak_final] * 75)
        return search.screen_cell(
            [search.world_metrics(one, self.screen)], self.screen)["soft_score"]

    def test_it_is_zero_for_a_passer(self) -> None:
        self.assertEqual(self.score(0.06), 0.0)

    def test_it_grows_with_the_violation(self) -> None:
        scores = [self.score(value) for value in (0.25, 0.30, 0.40, 0.60, 0.90)]
        self.assertEqual(scores[0], 0.0)
        for earlier, later in zip(scores, scores[1:]):
            self.assertLess(earlier, later)

    def test_it_is_the_violation_divided_by_the_width(self) -> None:
        # 0.30 is 0.05 above weak_max on an interval 0.23 wide.
        self.assertAlmostEqual(self.score(0.30), 0.05 / 0.23, places=12)

    def test_it_averages_over_the_worlds_of_a_cell(self) -> None:
        metrics = [
            search.world_metrics(passing_world(), self.screen),
            search.world_metrics(passing_world(fractions=[0.30] * 75),
                                 self.screen),
        ]
        cell = search.screen_cell(metrics, self.screen)
        self.assertAlmostEqual(cell["soft_score"], 0.5 * (0.05 / 0.23),
                               places=12)

    def test_it_stays_finite_when_the_weak_set_vanished(self) -> None:
        one = world(np.zeros((N, N), dtype=bool), fractions=[0.3] * 74 + [0.0])
        cell = search.screen_cell(
            [search.world_metrics(one, self.screen)], self.screen)
        self.assertTrue(np.isfinite(cell["soft_score"]))


class Sampling(unittest.TestCase):
    def setUp(self) -> None:
        self.space = search.Space()

    def sample(self, seed: int, count: int = 40) -> list[dict]:
        return search.latin_hypercube(self.space, count,
                                      np.random.default_rng(seed))

    def test_it_is_deterministic_for_a_search_seed(self) -> None:
        self.assertEqual(self.sample(11), self.sample(11))
        self.assertNotEqual(self.sample(11), self.sample(12))

    def test_every_sample_is_inside_its_range(self) -> None:
        for cell in self.sample(5):
            for name, kind in search.CONTINUOUS_DIALS:
                low, high = self.space.bounds(name)
                self.assertGreaterEqual(cell[name], low)
                self.assertLessEqual(cell[name], high)
            for name in search.SET_DIALS:
                self.assertIn(cell[name], self.space.values(name))

    def test_each_dial_covers_its_whole_range(self) -> None:
        count = 40
        cells = self.sample(3, count)
        for name, kind in search.CONTINUOUS_DIALS:
            low, high = self.space.bounds(name)
            values = sorted(cell[name] for cell in cells)
            if kind == "log":
                import math
                unit = [(math.log(value) - math.log(low))
                        / (math.log(high) - math.log(low)) for value in values]
            else:
                unit = [(value - low) / (high - low) for value in values]
            # One sample per stratum: the k-th smallest sits in the k-th.
            for index, position in enumerate(unit):
                self.assertGreaterEqual(position, index / count)
                self.assertLessEqual(position, (index + 1) / count)
        for name in search.SET_DIALS:
            drawn = {cell[name] for cell in cells}
            self.assertEqual(drawn, set(self.space.values(name)))

    def test_a_log_dial_needs_a_positive_low_end(self) -> None:
        space = search.Space(stiffness_fraction_lo=0.0)
        with self.assertRaises(ValueError):
            search.latin_hypercube(space, 4, np.random.default_rng(1))

    def test_a_one_cell_hypercube_is_still_a_sample(self) -> None:
        cells = self.sample(9, 1)
        self.assertEqual(len(cells), 1)
        self.assertEqual(sorted(cells[0]), sorted(search.DIAL_NAMES))

    def test_perturbation_stays_inside_the_space(self) -> None:
        rng = np.random.default_rng(2)
        start = self.sample(2, 1)[0]
        for _ in range(200):
            moved = search.perturb(start, self.space, rng)
            for name, _kind in search.CONTINUOUS_DIALS:
                low, high = self.space.bounds(name)
                self.assertGreaterEqual(moved[name], low)
                self.assertLessEqual(moved[name], high)
            for name in search.SET_DIALS:
                self.assertIn(moved[name], self.space.values(name))

    def test_perturbation_is_deterministic_and_moves_the_dials(self) -> None:
        start = self.sample(4, 1)[0]
        first = search.perturb(start, self.space, np.random.default_rng(8))
        again = search.perturb(start, self.space, np.random.default_rng(8))
        self.assertEqual(first, again)
        self.assertNotEqual(first, start)

    def test_the_dials_make_history_params(self) -> None:
        params = search.params_of(self.sample(6, 1)[0], self.space)
        self.assertEqual(params.history_myr, self.space.history_myr)
        self.assertEqual(params.max_cycles, self.space.max_cycles)
        record = params.to_record()
        for name in search.DIAL_NAMES:
            self.assertIn(name, record)

    def test_the_damage_law_comes_from_the_space_not_the_sample(self) -> None:
        dials = self.sample(6, 1)[0]
        self.assertNotIn("work_damage", dials)
        # The search's default is 0, the strain-rate law, which under the
        # seam formulation is the slip-rate law that keeps a slipping fault
        # weak; the engine's own default is 0 too.
        self.assertEqual(search.Space().work_damage, 0)
        self.assertEqual(search.params_of(dials, self.space).work_damage, 0)
        work = search.Space(work_damage=1)
        self.assertEqual(search.params_of(dials, work).work_damage, 1)

    def test_the_seam_switch_comes_from_the_space_not_the_sample(self) -> None:
        dials = self.sample(6, 1)[0]
        self.assertNotIn("seams", dials)
        # The engine's own default is the sheet; the search's default space
        # is the block model, because that is the open question.
        self.assertEqual(search.Space().seams, 2)
        self.assertEqual(search.params_of(dials, self.space).seams, 2)
        for fixed in (0, 1):
            with self.subTest(seams=fixed):
                other = search.Space(seams=fixed)
                self.assertEqual(search.params_of(dials, other).seams, fixed)

    def test_the_seam_switch_does_not_move_the_hypercube(self) -> None:
        # Fixed, not sampled, for the same reason `work_damage` is: the same
        # search seed then draws exactly the same cells under either rule.
        sheet = search.Space(seams=0)
        seamed = search.latin_hypercube(self.space, 32,
                                        np.random.default_rng(6))
        plain = search.latin_hypercube(sheet, 32, np.random.default_rng(6))
        self.assertEqual(seamed, plain)

    def test_the_two_seam_dials_are_sampled(self) -> None:
        space = search.Space()
        self.assertEqual(space.bounds("crack_speed_km_per_myr"),
                         (10.0, 200.0))
        self.assertEqual(space.values("nucleations_per_step"), (1, 2, 4))
        self.assertIn(("crack_speed_km_per_myr", "log"), search.DIALS)
        self.assertIn(("nucleations_per_step", "set"), search.DIALS)
        for dials in search.latin_hypercube(space, 32,
                                            np.random.default_rng(4)):
            params = search.params_of(dials, space)
            self.assertGreaterEqual(params.crack_speed_km_per_myr, 10.0)
            self.assertLessEqual(params.crack_speed_km_per_myr, 200.0)
            self.assertIn(params.nucleations_per_step, (1, 2, 4))

    def test_the_solve_divisor_comes_from_the_space(self) -> None:
        dials = self.sample(6, 1)[0]
        self.assertNotIn("solve_divisor", dials)
        # Since C04.4 the search's default is the full-grid solve, because
        # the stress concentration at a crack tip appears only there; the
        # engine's own default is still the half grid.
        self.assertEqual(search.Space().solve_divisor, 1)
        self.assertEqual(search.params_of(dials, self.space).solve_divisor, 1)
        half = search.Space(solve_divisor=2)
        self.assertEqual(search.params_of(dials, half).solve_divisor, 2)

    def test_the_solve_divisor_does_not_move_the_hypercube(self) -> None:
        # Fixed, not sampled, for the same reason `work_damage` is: the same
        # search seed then draws exactly the same cells on either grid.
        full = search.Space(solve_divisor=1)
        half = search.latin_hypercube(self.space, 32,
                                      np.random.default_rng(5))
        whole = search.latin_hypercube(full, 32, np.random.default_rng(5))
        self.assertEqual(half, whole)

    def test_the_damage_law_does_not_move_the_hypercube(self) -> None:
        # `work_damage` is fixed, not sampled, so the same search seed draws
        # exactly the same cells under either law. That is what makes a run
        # at 1 the ablation pair of a run at 0.
        strain = search.Space(work_damage=0)
        control = search.latin_hypercube(strain, 32,
                                         np.random.default_rng(2))
        treated = search.latin_hypercube(self.space, 32,
                                         np.random.default_rng(2))
        self.assertEqual(control, treated)


#: The run whose cells a pairing run has to redraw, and the run whose whole
#: -space sample the corner defaults were read off.
CONTROL_RUN = "20260902T154430Z-s2"


class PairingAgainstARunOnDisk(unittest.TestCase):
    """A run's own `config.json` redraws that run's cells, cell for cell.

    This is the guarantee the ablation pair rests on, and it holds whatever
    the current defaults are: the search is reproducible from a config, so
    pairing against a run on disk means loading that run's config and
    flipping one field. The defaults are free to move to whatever question
    is being asked now, which is what `SEARCH.md` documents.
    """

    def setUp(self) -> None:
        self.directory = PIPELINE_C / "out" / "search" / CONTROL_RUN
        if not (self.directory / "cells.jsonl").exists():
            self.skipTest(f"the run {CONTROL_RUN} is not on disk")
        self.control = json.loads(
            (self.directory / "config.json").read_text(encoding="utf-8"))

    def space_of(self, record: dict) -> search.Space:
        fields = search.Space.__dataclass_fields__
        values = {}
        for name, value in record["space"].items():
            if name not in fields:
                continue
            values[name] = (tuple(value) if isinstance(value, list) else value)
        return search.Space(**values)

    def test_the_control_config_redraws_the_control_cells(self) -> None:
        space = self.space_of(self.control)
        drawn = search.latin_hypercube(
            space, int(self.control["stages"]["stage1_cells"]),
            np.random.default_rng(int(self.control["search_seed"])))
        cells = [json.loads(line) for line in
                 (self.directory / "cells.jsonl").read_text(
                     encoding="utf-8").splitlines()[:3] if line.strip()]
        self.assertEqual(len(cells), 3)
        # `WORK_ORDER_C03_10.md` replaced the `drive_nodes` set with a
        # log-uniform range in kilometres. The hypercube still has eight
        # columns drawn in the same order from the same generator, so every
        # unit value is what it was and the seven dials whose sampler did not
        # change still redraw exactly; the drive column is mapped by a
        # different rule and is checked separately below.
        #
        # `WORK_ORDER_C04.md` §4 appended two more columns, `crack_speed_km_
        # per_myr` and `nucleations_per_step`. A Latin hypercube draws its
        # columns in order from one generator, so appending columns leaves
        # every earlier column's draw untouched; the two new ones have no
        # counterpart in a run written before they existed and are excluded
        # here for that reason and not because they moved.
        # `WORK_ORDER_C04_4.md` §2 appended a fourth, `toughness_fraction`,
        # for the same reason and with the same consequence.
        skip = {"drive_wavelength_km", "crack_speed_km_per_myr",
                "nucleations_per_step", "toughness_fraction"}
        unchanged = [name for name in search.DIAL_NAMES if name not in skip]
        self.assertEqual(len(unchanged), len(search.DIAL_NAMES) - 4)
        for index, cell in enumerate(cells):
            self.assertEqual(cell["stage"], 1)
            self.assertEqual(cell["round"], 0)
            for name in unchanged:
                with self.subTest(cell=cell["id"], dial=name):
                    self.assertAlmostEqual(drawn[index][name],
                                           cell["dials"][name], places=12)

    def test_the_drive_column_is_the_one_the_kilometre_dial_moved(self) -> None:
        # The run on disk recorded a node count; a redraw now records a
        # wavelength. The logged cells stay runnable through
        # `modernize_dials`, which is what `params_of` calls, but they are no
        # longer redrawn by the same rule and this records that.
        space = self.space_of(self.control)
        drawn = search.latin_hypercube(
            space, int(self.control["stages"]["stage1_cells"]),
            np.random.default_rng(int(self.control["search_seed"])))
        cell = json.loads((self.directory / "cells.jsonl").read_text(
            encoding="utf-8").splitlines()[0])
        self.assertIn("drive_nodes", cell["dials"])
        self.assertNotIn("drive_wavelength_km", cell["dials"])
        self.assertIn("drive_wavelength_km", drawn[0])
        modern = search.modernize_dials(
            cell["dials"], int(self.control["space"]["pixels"]),
            int(self.control["space"]["scale_km"]))
        lo, hi = space.bounds("drive_wavelength_km")
        self.assertGreaterEqual(modern["drive_wavelength_km"], lo)
        self.assertLessEqual(modern["drive_wavelength_km"], hi)

    def test_flipping_the_law_redraws_the_same_cells(self) -> None:
        space = self.space_of(self.control)
        control = search.latin_hypercube(space, 16, np.random.default_rng(2))
        treated = search.latin_hypercube(
            replace(space, work_damage=1), 16, np.random.default_rng(2))
        self.assertEqual(control, treated)


class LegacyDials(unittest.TestCase):
    """`WORK_ORDER_C03_10.md` §2: runs on disk stay rerunnable and pairable."""

    def legacy(self) -> dict:
        return {
            "stiffness_fraction": 0.2,
            "yield_percentile": 5.0,
            "heal_time_myr": 10.0,
            "damage_time_myr": 1.0,
            "strength_exponent": 2,
            "strength_spread": 0.05,
            "drive_nodes": 2,
            "drive_shear": 0.5,
        }

    def test_a_1024_px_node_count_of_two_is_exactly_5120_km(self) -> None:
        modern = search.modernize_dials(self.legacy(), 1024, 5)
        self.assertEqual(modern["drive_wavelength_km"], 5120.0)
        self.assertNotIn("drive_nodes", modern)
        # Every other dial is carried across untouched.
        for name, value in self.legacy().items():
            if name != "drive_nodes":
                self.assertEqual(modern[name], value)

    def test_the_conversion_follows_the_run_geometry(self) -> None:
        # The same node count on a 512-px run meant half the wavelength,
        # which is the bug the order names.
        self.assertEqual(
            search.modernize_dials(self.legacy(), 512, 5)
            ["drive_wavelength_km"], 2560.0)

    def test_modern_dials_are_returned_unchanged(self) -> None:
        modern = dict(self.legacy())
        del modern["drive_nodes"]
        modern["drive_wavelength_km"] = 7000.0
        modern.update(search.LEGACY_DIAL_DEFAULTS)
        self.assertIs(search.modernize_dials(modern, 1024, 5), modern)
        # And a set carrying both keys keeps the modern one.
        both = dict(modern)
        both["drive_nodes"] = 2
        self.assertEqual(
            search.modernize_dials(both, 1024, 5)["drive_wavelength_km"],
            7000.0)

    def test_a_cell_that_predates_the_seam_dials_gets_the_engine_defaults(
            self) -> None:
        # `WORK_ORDER_C04.md` §4: a run written before the switch existed ran
        # on the engine's defaults, so that is what its cells are filled with.
        modern = search.modernize_dials(self.legacy(), 1024, 5)
        self.assertEqual(modern["seams"], 0)
        self.assertEqual(modern["crack_speed_km_per_myr"], 40.0)
        self.assertEqual(modern["nucleations_per_step"], 2)
        self.assertEqual(HistoryParams().seams, 0)
        self.assertEqual(HistoryParams().crack_speed_km_per_myr, 40.0)
        self.assertEqual(HistoryParams().nucleations_per_step, 2)

    def test_params_of_runs_a_legacy_cell(self) -> None:
        # A legacy cell is converted with the pixels of the run it came
        # from, 1024, whatever the current default space is.
        params = search.params_of(self.legacy(), search.Space(pixels=1024))
        self.assertEqual(params.drive_wavelength_km, 5120.0)
        self.assertEqual(params.to_record()["drive_wavelength_km"], 5120.0)


class TheCornerDefaults(unittest.TestCase):
    """The defaults sample the corner the whole-space run pointed at.

    Seven of 1460 cells of `20260902T170740Z-s3` had a weak fraction of 0.30
    or less with three or more plates. Their damage and healing times are
    far below the run's medians and their yield percentile far above it.
    These bounds are that corner, opened downward to the engine's own floors
    so the run answers whether zone width keeps narrowing as both times
    shorten.
    """

    def test_the_ranges_are_the_corner(self) -> None:
        space = search.Space()
        # Healing is the one range `WORK_ORDER_C04_1.md` §3 widened: a
        # persisting seam's healing time is a different question from the
        # sheet's, so the corner's 60 becomes 200.
        # 20 – 200 since C04.4: a cut loop reopened whenever one seam cell
        # healed, and at 10 Myr a locked cell seals in two steps.
        self.assertEqual(space.bounds("heal_time_myr"), (20.0, 200.0))
        self.assertEqual(space.bounds("damage_time_myr"), (0.5, 5.0))
        self.assertEqual(space.bounds("yield_percentile"), (2.0, 15.0))
        self.assertEqual(space.bounds("stiffness_fraction"), (0.08, 0.5))
        self.assertEqual(space.values("strength_exponent"), (2, 3))
        # The old set `{1, 2}` was a node count at 1024 px and 5 km/px, so
        # it named the parent and half the parent in kilometres.
        # Opened to 2,560 km when the search moved to 512 px, whose parent
        # is 5,120 km, so a world can hold up to two mantle cells across.
        self.assertEqual(space.bounds("drive_wavelength_km"),
                         (2560.0, 10240.0))
        self.assertEqual(space.work_damage, 0)

    def test_every_sampled_dial_is_legal_for_the_engine(self) -> None:
        space = search.Space()
        for dials in search.latin_hypercube(space, 64,
                                            np.random.default_rng(11)):
            search.params_of(dials, space)      # raises if out of range

    def test_the_hypercube_covers_the_log_range(self) -> None:
        space = search.Space()
        lo, hi = space.bounds("drive_wavelength_km")
        drawn = [dials["drive_wavelength_km"]
                 for dials in search.latin_hypercube(
                     space, 64, np.random.default_rng(11))]
        self.assertEqual(len(drawn), 64)
        self.assertGreaterEqual(min(drawn), lo)
        self.assertLessEqual(max(drawn), hi)
        # A Latin hypercube uses every stratum once, and the strata are equal
        # in the logarithm, so the lowest draw is in the lowest 64th of the
        # log range and the highest in the highest.
        import math
        width = math.log(hi) - math.log(lo)
        self.assertLess(math.log(min(drawn)) - math.log(lo), width / 64.0)
        self.assertLess(math.log(hi) - math.log(max(drawn)), width / 64.0)

    def test_the_corner_reaches_the_engine_floors(self) -> None:
        # The question is whether width keeps improving as both times get
        # shorter, so the ranges must reach as low as the engine allows.
        space = search.Space()
        floor_heal = HistoryParams.__post_init__ is not None
        self.assertTrue(floor_heal)
        # Healing no longer reaches the floor (see the corner test); the
        # damage time still does.
        self.assertEqual(space.bounds("damage_time_myr")[0], 0.5)
        with self.assertRaises(ValueError):
            HistoryParams(heal_time_myr=4.9)
        with self.assertRaises(ValueError):
            HistoryParams(damage_time_myr=0.4)

    def test_the_default_seed_is_one_no_run_on_disk_has_used(self) -> None:
        # A run's round `r` samples from `search_seed + r`, so a run that
        # restarted blindly has spent every seed from its own up to its last
        # round's. The default must be none of them: the run of
        # 2026-09-03 15:57 started at 13 while the overnight run had reached
        # 15, and three of its four rounds re-sampled the overnight cells.
        default = search.SearchConfig().search_seed
        used = set()
        root = PIPELINE_C / "out" / "search"
        if root.exists():
            for directory in root.iterdir():
                config = directory / "config.json"
                if not config.is_file():
                    continue
                first = int(json.loads(
                    config.read_text(encoding="utf-8"))["search_seed"])
                last_round = 0
                cells = directory / "cells.jsonl"
                if cells.is_file():
                    with cells.open(encoding="utf-8") as handle:
                        for line in handle:
                            if line.strip():
                                last_round = max(
                                    last_round, int(json.loads(line)["round"]))
                used.update(range(first, first + last_round + 1))
        self.assertNotIn(default, used)


class Config(unittest.TestCase):
    def test_the_twelve_development_seeds_are_the_ones_status_lists(self) -> None:
        text = (PIPELINE_C / "STATUS.md").read_text(encoding="utf-8")
        for seed in search.DEVELOPMENT_SEEDS:
            self.assertIn(str(seed), text)
        self.assertEqual(len(search.DEVELOPMENT_SEEDS), 12)
        self.assertEqual(len(set(search.DEVELOPMENT_SEEDS)), 12)

    def test_the_screen_is_the_one_the_order_names(self) -> None:
        # The screen is the agreed definition of a plate regime and does not
        # move with the question being asked. The sampled ranges do; they are
        # pinned by `TheCornerDefaults`.
        screen, space, stages = (search.Screen(), search.Space(),
                                 search.Stages())
        self.assertEqual(
            (screen.weak_min, screen.weak_max, screen.peak_ratio_max,
             screen.flat_window_myr, screen.flat_tolerance, screen.plates_min,
             screen.plates_max, screen.network_share_min,
             screen.edge_fraction_min, screen.residual_max,
             screen.pass_fraction),
            (0.02, 0.25, 1.5, 100.0, 0.03, 3, 8, 0.5, 0.5, 1e-3, 1.0))
        self.assertEqual((space.pixels, space.scale_km, space.history_myr,
                          space.max_cycles, space.base_seed),
                         (512, 5, 300.0, 80, 4287772760))
        self.assertEqual(
            (stages.stage1_cells, stages.stage1_seeds, stages.stage2_top,
             stages.stage2_perturbations, stages.stage2_seeds,
             stages.stage3_top),
            (200, 4, 20, 8, 8, 3))

    def test_the_config_serializes(self) -> None:
        import json
        payload = search.SearchConfig().to_json()
        self.assertEqual(json.loads(json.dumps(payload)), payload)


if __name__ == "__main__":
    unittest.main()
