"""Gates on the seam formulation of `WORK_ORDER_C04.md` and `DESIGN.md` §3.6.

`seams` is one integer on `HistoryParams`. At `0` the sheet damages every cell
whose load exceeds the yield, which is what the engine has always done and
what production runs; at `1` damage happens on a seam, at a seam's tip, or at
a nucleation site, and nowhere else.

`WORK_ORDER_C04_1.md` then made `work_damage` mean something under `seams`:
at `1` the seam damages by the work its slip dissipates, which is the law C04
ran and which is pinned here bit for bit; at `0`, the default, it damages by
its slip rate, so a fault that is slipping stays weak.

The first gate is the one that matters for production: at `seams = 0` the
history is byte-identical to the pre-order engine, checked here on a dial set
well away from every default. `test_regression_c03_1` and
`test_work_damage` cover the production default and the C03.8 dial set.

Everything after that is the rule itself, on hand-built fields where the
answer can be worked out by hand, then one short run under each law.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path
import sys
import unittest

import numpy as np

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

import search  # noqa: E402
from engine.domain import sample_nearest_periodic  # noqa: E402
from engine.geometry import WorldGeometry  # noqa: E402
from engine.history.constants import (  # noqa: E402
    INTACT_SPREAD_CLIP,
    MG_TOL,
    SEAM_OPEN_STRENGTH,
    STRENGTH_MIN,
    WEAK_THRESHOLD,
)
from engine.history.kinematics import (  # noqa: E402
    HistoryParams,
    run_history,
    strength_noise,
)
from engine.history.plates import label_plates  # noqa: E402
from engine.history.seams import (  # noqa: E402
    DIRECTIONS,
    advect_nearest,
    crack_lengths,
    damage_excess,
    intact_strength_field,
    nucleate,
    seam_mask,
    tip_pass,
    tips,
    traction_magnitude,
)

#: The world the byte-identity gate runs on. Any world does: the pre-order
#: engine here is the C03.10 engine this order started from, not something
#: older whose noise bands moved.
GATE_WORLD = WorldGeometry(11, 512, 10)

#: A dial set away from every default and from `test_work_damage`'s, so the
#: gate is not a second copy of an existing regression.
OFF_DEFAULT = dict(
    stiffness_fraction=0.25,
    yield_percentile=8.0,
    heal_time_myr=25.0,
    damage_time_myr=2.0,
    work_damage=1,
    strength_exponent=2,
    strength_spread=0.07,
    drive_wavelength_km=5120.0,
    drive_shear=0.4,
)

#: SHA-256 over the four kept epochs' strength and velocity fields, in order,
#: of `run_history(GATE_WORLD, params=HistoryParams(**OFF_DEFAULT), steps=8)`.
#:
#: Recorded 2026-09-02 **before this order's edits**, against a copy of the
#: engine rebuilt by undoing every one of them: `pipeline_c`'s C03 engine is
#: not committed, so the pre-order engine could not be read out of git and
#: was reconstructed instead, one literal reversal per edit, and the copy
#: differed from the working tree in exactly the lines this order added.
PRE_ORDER_SHA256 = (
    "500b547543e6445265fa055a671fa0bcd113d0f64a6f47399619af6ca707a572")

#: The dials `WORK_ORDER_C04.md` §7.1 runs the twelve seeds at: the
#: production defaults with the corner search's centre for the dials the
#: corner moved.
CORNER_CENTRE = dict(
    heal_time_myr=10.0,
    damage_time_myr=1.5,
    yield_percentile=6.0,
    stiffness_fraction=0.3,
    strength_exponent=2,
    drive_wavelength_km=5120.0,
    drive_shear=0.6,
    strength_spread=0.03,
)

N = 16


def uniform_tensor(sxx: float, syy: float, sxy: float, n: int = N):
    """Three constant tensor components on an `n x n` grid."""
    ones = np.ones((n, n), dtype=np.float64)
    return sxx * ones, syy * ones, sxy * ones


def intact(n: int = N) -> np.ndarray:
    return np.ones((n, n), dtype=np.float64)


def digest(history) -> str:
    sha = hashlib.sha256()
    for epoch in history.epochs:
        sha.update(epoch.strength.tobytes())
        sha.update(epoch.velocity.tobytes())
    return sha.hexdigest()


class TheSwitch(unittest.TestCase):
    def test_the_default_is_the_sheet(self) -> None:
        self.assertEqual(HistoryParams().seams, 0)
        self.assertEqual(HistoryParams().to_record()["seams"], 0)
        self.assertEqual(HistoryParams(seams=1).to_record()["seams"], 1)

    def test_the_new_dials_are_recorded_whatever_the_rule(self) -> None:
        record = HistoryParams().to_record()
        self.assertEqual(record["crack_speed_km_per_myr"], 40.0)
        self.assertEqual(record["nucleations_per_step"], 2)
        # The default is the constant the tip rule carried before the dial
        # existed, so every output at the default is what it was.
        self.assertEqual(record["toughness_fraction"], 1.0)
        self.assertIsInstance(record["toughness_fraction"], float)
        self.assertEqual(
            HistoryParams(toughness_fraction=0.5)
            .to_record()["toughness_fraction"], 0.5)
        # The engine's default `work_damage` is 0 under both rules, which
        # under the seam rule is the slip-rate law.
        self.assertEqual(HistoryParams(seams=1).to_record()["work_damage"], 0)
        self.assertEqual(
            HistoryParams(seams=1, work_damage=1).to_record()["work_damage"], 1)

    def test_out_of_range_settings_are_refused(self) -> None:
        for kwargs in ({"seams": 3}, {"seams": -1}, {"seams": True},
                       {"seams": 1.0},
                       {"crack_speed_km_per_myr": -1.0},
                       {"crack_speed_km_per_myr": 401.0},
                       {"nucleations_per_step": 21},
                       {"nucleations_per_step": -1},
                       {"nucleations_per_step": 2.0},
                       {"nucleations_per_step": True},
                       # `WORK_ORDER_C04_4.md` §2: the toughness is a
                       # fraction of the intact strength, so 0 is not a
                       # material and 1.5 is a rock tougher than it is
                       # strong.
                       {"toughness_fraction": 0.0},
                       {"toughness_fraction": 1.5},
                       {"toughness_fraction": 0.04},
                       {"toughness_fraction": True}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                HistoryParams(**kwargs)

    def test_the_constants_are_what_the_order_names(self) -> None:
        self.assertEqual(SEAM_OPEN_STRENGTH, STRENGTH_MIN)
        self.assertEqual(INTACT_SPREAD_CLIP, (0.2, 2.0))


class TheProductionPath(unittest.TestCase):
    """`seams = 0` must not have moved anything by one bit."""

    def test_a_non_default_dial_set_is_byte_identical_to_the_old_engine(
            self) -> None:
        history = run_history(GATE_WORLD, params=HistoryParams(**OFF_DEFAULT),
                              steps=8)
        self.assertEqual(digest(history), PRE_ORDER_SHA256)

    def test_the_seam_rule_at_the_same_dials_is_a_different_history(
            self) -> None:
        for rule in (1, 2):
            with self.subTest(seams=rule):
                seamed = run_history(
                    GATE_WORLD,
                    params=HistoryParams(seams=rule, **OFF_DEFAULT), steps=8)
                self.assertNotEqual(digest(seamed), PRE_ORDER_SHA256)

    def test_the_two_seam_rules_are_different_histories(self) -> None:
        sheet_solve = run_history(
            GATE_WORLD, params=HistoryParams(seams=1, **OFF_DEFAULT), steps=8)
        blocks = run_history(
            GATE_WORLD, params=HistoryParams(seams=2, **OFF_DEFAULT), steps=8)
        self.assertNotEqual(digest(sheet_solve), digest(blocks))


class TheDamageLaw(unittest.TestCase):
    """`work_damage` under `seams`, which is the one change C04.1 makes.

    C04 damaged a seam by the work its slip dissipates. A fully open seam has
    stiffness `KAPPA0 * STRENGTH_MIN ** exponent`, carries almost no traction
    and dissipates almost nothing however fast it slips, so its damage rate is
    near zero and healing shuts it. The slip-rate law is the ordinary one for
    a fault: while it slips it stays weak.
    """

    #: A seam cell and an intact cell, both carrying three times the yield
    #: strain rate, and a power field sitting exactly at its own yield so the
    #: work law finds no excess in it. The two laws therefore disagree on
    #: every cell of this field, which is what the tests below read.
    YIELD_STRAIN = 0.25
    POWER_YIELD = 7.0

    def fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        strength = intact()
        strength[5, 5] = SEAM_OPEN_STRENGTH
        strain_rate = np.full((N, N), 3.0 * self.YIELD_STRAIN)
        power = np.full((N, N), self.POWER_YIELD)
        return strength, strain_rate, power

    def excess(self, work_damage: int) -> np.ndarray:
        strength, strain_rate, power = self.fields()
        return damage_excess(strength, power, self.POWER_YIELD, strain_rate,
                             self.YIELD_STRAIN, work_damage)

    def test_a_slipping_seam_damages_at_the_squared_excess(self) -> None:
        # Three times yield is an excess of two, and the law is the square,
        # so the seam cell loses strength at `damage_rate * 4` per Myr.
        excess = self.excess(0)
        self.assertAlmostEqual(float(excess[5, 5]), 2.0, places=12)
        params = HistoryParams(seams=1, **CORNER_CENTRE)
        self.assertAlmostEqual(params.damage_rate * float(excess[5, 5]) ** 2,
                               params.damage_rate * 4.0, places=12)

    def test_an_intact_cell_at_the_same_strain_rate_loses_none(self) -> None:
        excess = self.excess(0)
        self.assertEqual(float(excess[0, 0]), 0.0)
        # Exactly one cell of the field is a seam, and it is the only one
        # with any damage at all.
        self.assertEqual(int(np.count_nonzero(excess)), 1)

    def test_the_work_law_reads_the_power_and_not_the_strain_rate(
            self) -> None:
        # The same field, the same seam cell, `work_damage = 1`: the power is
        # exactly at its yield, so there is no excess anywhere.
        self.assertEqual(float(np.abs(self.excess(1)).max()), 0.0)
        # And at three times the power yield the work law finds the same
        # excess of two the slip-rate law found in the strain rate.
        strength, _strain, power = self.fields()
        excess = damage_excess(strength, 3.0 * power, self.POWER_YIELD,
                               np.zeros((N, N)), self.YIELD_STRAIN, 1)
        self.assertAlmostEqual(float(excess[5, 5]), 2.0, places=12)
        self.assertEqual(int(np.count_nonzero(excess)), 1)

    def test_the_open_seam_that_slips_does_not_heal_and_the_one_that_does_not_does(
            self) -> None:
        """The integrator, at the corner's healing time and the 4 Myr step."""
        params = HistoryParams(seams=1, **CORNER_CENTRE)
        heal, damage, step = params.heal_rate, params.damage_rate, 4.0

        def after(strength: float, excess: float, steps: int) -> float:
            rate = damage * excess * excess
            total = heal + rate
            equilibrium = heal / total
            for _ in range(steps):
                strength = float(np.clip(
                    equilibrium + (strength - equilibrium)
                    * math.exp(-total * step), STRENGTH_MIN, 1.0))
            return strength

        # Slipping at three times yield: the equilibrium is below the floor,
        # so the cell sits at `STRENGTH_MIN` and is still a seam at 300 Myr.
        self.assertEqual(after(SEAM_OPEN_STRENGTH, 2.0, 75), STRENGTH_MIN)
        # Not slipping — which is what the work law reads on an open seam,
        # because an open seam dissipates almost nothing — and it is intact
        # again after two steps, which is eight million years.
        self.assertLess(after(SEAM_OPEN_STRENGTH, 0.0, 1), WEAK_THRESHOLD)
        self.assertGreater(after(SEAM_OPEN_STRENGTH, 0.0, 2), WEAK_THRESHOLD)


class Tips(unittest.TestCase):
    def test_a_straight_line_has_two_tips(self) -> None:
        seam = np.zeros((N, N), dtype=bool)
        seam[8, 4:9] = True
        found = tips(seam)
        self.assertEqual(int(found.sum()), 2)
        self.assertTrue(found[8, 4])
        self.assertTrue(found[8, 8])

    def test_a_loop_has_none(self) -> None:
        # A row that wraps the torus is a closed loop, and it is the loop the
        # formulation is watching for: a pair of them cuts the sheet in two.
        seam = np.zeros((N, N), dtype=bool)
        seam[8, :] = True
        self.assertEqual(int(tips(seam).sum()), 0)
        # And so is a square ring, which closes without using the wrap.
        ring = np.zeros((N, N), dtype=bool)
        ring[4:10, 4] = True
        ring[4:10, 9] = True
        ring[4, 4:10] = True
        ring[9, 4:10] = True
        self.assertEqual(int(tips(ring).sum()), 0)

    def test_an_isolated_cell_is_a_tip_of_length_one(self) -> None:
        seam = np.zeros((N, N), dtype=bool)
        seam[5, 5] = True
        self.assertTrue(tips(seam)[5, 5])
        self.assertEqual(int(tips(seam).sum()), 1)
        self.assertEqual(int(crack_lengths(seam)[5, 5]), 1)

    def test_crack_length_is_the_component_size(self) -> None:
        seam = np.zeros((N, N), dtype=bool)
        seam[2, 2:6] = True          # four cells
        seam[12, 12] = True          # one cell, elsewhere
        lengths = crack_lengths(seam)
        self.assertEqual(int(lengths[2, 2]), 4)
        self.assertEqual(int(lengths[2, 5]), 4)
        self.assertEqual(int(lengths[12, 12]), 1)
        self.assertEqual(int(lengths[0, 0]), 0)

    def test_an_empty_mask_has_no_tips_and_no_lengths(self) -> None:
        seam = np.zeros((N, N), dtype=bool)
        self.assertEqual(int(tips(seam).sum()), 0)
        self.assertEqual(int(crack_lengths(seam).sum()), 0)


class TheTipRule(unittest.TestCase):
    """A uniform tension along x, so the answer can be worked out by hand.

    With `sxx = 1` and everything else zero, the traction on the face of a
    seam running along `(dy, dx)` is `|dy| / |d|`: 1 for the two vertical
    directions, `1 / sqrt(2)` for the four diagonals, 0 for the two
    horizontal ones. A crack under tension therefore runs across the pull,
    which is what a crack does.
    """

    def test_the_traction_is_the_normal_component(self) -> None:
        sxx, syy, sxy = uniform_tensor(1.0, 0.0, 0.0)
        for dy, dx in DIRECTIONS:
            with self.subTest(direction=(dy, dx)):
                expected = abs(dy) / math.hypot(dx, dy)
                magnitude = traction_magnitude(sxx, syy, sxy, dy, dx)
                self.assertAlmostEqual(float(magnitude[0, 0]), expected,
                                       places=12)

    def test_the_tip_takes_the_direction_carrying_the_most_traction(
            self) -> None:
        strength = intact()
        strength[5, 5] = SEAM_OPEN_STRENGTH
        sxx, syy, sxy = uniform_tensor(1.0, 0.0, 0.0)
        moved, tip_count, advances = tip_pass(
            strength, sxx, syy, sxy, np.full((N, N), 0.5))
        self.assertEqual((tip_count, advances), (1, 1))
        opened = np.argwhere(seam_mask(moved) & ~seam_mask(strength))
        self.assertEqual(opened.tolist(), [[4, 5]])
        # A vertical direction, and the first of the two in `DIRECTIONS`:
        # the tie between (-1, 0) and (1, 0) is broken by that fixed order.
        self.assertEqual(DIRECTIONS.index((-1, 0)),
                         min(DIRECTIONS.index((-1, 0)),
                             DIRECTIONS.index((1, 0))))

    def test_shear_turns_the_crack(self) -> None:
        # Pure shear: `sxy = 1`, the rest zero. `|t|` is then
        # `sqrt(ny**2 + nx**2) = 1` in every direction, so every candidate
        # ties and the fixed order decides — the first direction in
        # `DIRECTIONS`, which is (0, -1).
        strength = intact()
        strength[5, 5] = SEAM_OPEN_STRENGTH
        sxx, syy, sxy = uniform_tensor(0.0, 0.0, 1.0)
        moved, _tips, advances = tip_pass(strength, sxx, syy, sxy,
                                          np.full((N, N), 0.5))
        self.assertEqual(advances, 1)
        opened = np.argwhere(seam_mask(moved) & ~seam_mask(strength))
        self.assertEqual(opened.tolist(), [[5, 4]])
        self.assertEqual(DIRECTIONS[0], (0, -1))

    def test_a_strength_above_the_traction_refuses_every_direction(
            self) -> None:
        strength = intact()
        strength[5, 5] = SEAM_OPEN_STRENGTH
        sxx, syy, sxy = uniform_tensor(1.0, 0.0, 0.0)
        moved, tip_count, advances = tip_pass(
            strength, sxx, syy, sxy, np.full((N, N), 1.5))
        self.assertEqual((tip_count, advances), (1, 0))
        self.assertEqual(moved.tobytes(), strength.tobytes())

    def test_the_griffith_threshold_at_four_cells_is_half_the_one_at_one(
            self) -> None:
        sxx, syy, sxy = uniform_tensor(1.0, 0.0, 0.0)
        sigma_c = np.full((N, N), 1.5)
        # The rule is `|t| >= sigma_c / sqrt(L)`, so the four-cell threshold
        # is exactly half the one-cell threshold.
        self.assertAlmostEqual(float(sigma_c[0, 0] / math.sqrt(4)),
                               float(sigma_c[0, 0] / math.sqrt(1)) / 2.0,
                               places=12)
        # One cell: the threshold is 1.5 and the best traction is 1.0.
        one = intact()
        one[5, 5] = SEAM_OPEN_STRENGTH
        self.assertEqual(tip_pass(one, sxx, syy, sxy, sigma_c)[2], 0)
        # Four cells in a line: the threshold is 0.75, the vertical traction
        # is still 1.0, and both tips run on. The diagonals, at 0.707, still
        # do not qualify, so the crack stays straight.
        four = intact()
        four[8:12, 5] = SEAM_OPEN_STRENGTH
        moved, tip_count, advances = tip_pass(four, sxx, syy, sxy, sigma_c)
        self.assertEqual((tip_count, advances), (2, 2))
        opened = np.argwhere(seam_mask(moved) & ~seam_mask(four))
        self.assertEqual(sorted(opened.tolist()), [[7, 5], [12, 5]])

    def test_an_intact_sheet_has_no_tips_and_no_advances(self) -> None:
        sxx, syy, sxy = uniform_tensor(1.0, 0.0, 0.0)
        moved, tip_count, advances = tip_pass(
            intact(), sxx, syy, sxy, np.full((N, N), 0.1))
        self.assertEqual((tip_count, advances), (0, 0))
        self.assertEqual(moved.tobytes(), intact().tobytes())


class TheToughnessDial(unittest.TestCase):
    """The toughness scales the tip threshold and nothing else.

    The same uniform tension along x as `TheTipRule`: the best traction any
    direction carries is exactly 1.0, so a threshold above 1.0 refuses every
    candidate and one below it opens the vertical neighbour.
    """

    def setUp(self) -> None:
        self.sxx, self.syy, self.sxy = uniform_tensor(1.0, 0.0, 0.0)
        self.sigma_c = np.full((N, N), 1.5)
        self.strength = intact()
        self.strength[5, 5] = SEAM_OPEN_STRENGTH

    def test_the_default_is_the_threshold_the_rule_already_had(self) -> None:
        without = tip_pass(self.strength, self.sxx, self.syy, self.sxy,
                           self.sigma_c)
        with_one = tip_pass(self.strength, self.sxx, self.syy, self.sxy,
                            self.sigma_c, 1.0)
        self.assertEqual(without[0].tobytes(), with_one[0].tobytes())
        self.assertEqual(without[1:], with_one[1:])

    def test_at_half_a_tip_advances_where_at_one_it_did_not(self) -> None:
        """The one thing the dial is for, on a crack of one cell.

        `L` is 1, so `sqrt(L)` is 1 and the threshold is the toughness times
        the intact strength: 1.5 at 1.0, which a traction of 1.0 cannot
        reach, and 0.75 at 0.5, which it passes.
        """
        held, tip_count, advances = tip_pass(
            self.strength, self.sxx, self.syy, self.sxy, self.sigma_c, 1.0)
        self.assertEqual((tip_count, advances), (1, 0))
        self.assertEqual(held.tobytes(), self.strength.tobytes())

        moved, tip_count, advances = tip_pass(
            self.strength, self.sxx, self.syy, self.sxy, self.sigma_c, 0.5)
        self.assertEqual((tip_count, advances), (1, 1))
        opened = np.argwhere(seam_mask(moved) & ~seam_mask(self.strength))
        self.assertEqual(opened.tolist(), [[4, 5]])

    def test_the_threshold_is_the_fraction_times_the_old_one(self) -> None:
        # The traction is 1.0 and the intact strength 1.5, so the value at
        # which a one-cell tip starts advancing is exactly 1 / 1.5.
        edge = 1.0 / 1.5
        above = tip_pass(self.strength, self.sxx, self.syy, self.sxy,
                         self.sigma_c, edge * 1.001)[2]
        below = tip_pass(self.strength, self.sxx, self.syy, self.sxy,
                         self.sigma_c, edge * 0.999)[2]
        self.assertEqual((above, below), (0, 1))

    def test_the_crack_length_still_divides_the_threshold(self) -> None:
        # Four cells in a line: the threshold is 0.75 at toughness 1.0 and
        # 0.375 at 0.5, and the vertical traction of 1.0 passes both, so both
        # tips run either way. The dial multiplies the length rule; it does
        # not replace it.
        four = intact()
        four[8:12, 5] = SEAM_OPEN_STRENGTH
        for toughness in (1.0, 0.5):
            with self.subTest(toughness=toughness):
                moved, tip_count, advances = tip_pass(
                    four, self.sxx, self.syy, self.sxy, self.sigma_c,
                    toughness)
                self.assertEqual((tip_count, advances), (2, 2))
                opened = np.argwhere(seam_mask(moved) & ~seam_mask(four))
                self.assertEqual(sorted(opened.tolist()), [[7, 5], [12, 5]])

    def test_nucleation_still_needs_the_full_intact_strength(self) -> None:
        """The dial is the propagation threshold and not the nucleation one."""
        # A stress of 1.0 against an intact strength of 1.5 opens nothing,
        # and `nucleate` has no toughness to be told otherwise.
        smag = np.full((N, N), 1.0)
        _strength, count = nucleate(intact(), smag, self.sigma_c, 4)
        self.assertEqual(count, 0)
        self.assertNotIn(
            "toughness",
            " ".join(inspect.signature(nucleate).parameters))


class Nucleation(unittest.TestCase):
    def field(self) -> tuple[np.ndarray, np.ndarray]:
        strength = intact()
        strength[5, 5] = SEAM_OPEN_STRENGTH
        smag = np.zeros((N, N), dtype=np.float64)
        return strength, smag

    def test_a_cell_beside_a_seam_is_never_a_nucleus(self) -> None:
        strength, smag = self.field()
        # The highest stress in the world, right beside the seam.
        smag[5, 6] = 100.0
        smag[12, 12] = 2.0
        moved, count = nucleate(strength, smag, np.ones((N, N)), 4)
        self.assertEqual(count, 1)
        opened = np.argwhere(seam_mask(moved) & ~seam_mask(strength))
        self.assertEqual(opened.tolist(), [[12, 12]])
        # Diagonal neighbours are excluded too: the neighbourhood is the
        # eight, not the four.
        strength, smag = self.field()
        smag[4, 4] = 100.0
        self.assertEqual(nucleate(strength, smag, np.ones((N, N)), 4)[1], 0)

    def test_below_the_intact_strength_nothing_nucleates(self) -> None:
        strength, smag = self.field()
        smag[12, 12] = 0.99
        self.assertEqual(nucleate(strength, smag, np.ones((N, N)), 4)[1], 0)
        smag[12, 12] = 1.0
        self.assertEqual(nucleate(strength, smag, np.ones((N, N)), 4)[1], 1)

    def test_the_cap_is_respected_and_zero_means_none(self) -> None:
        strength, smag = self.field()
        smag[10, :] = 5.0
        self.assertEqual(nucleate(strength, smag, np.ones((N, N)), 3)[1], 3)
        self.assertEqual(nucleate(strength, smag, np.ones((N, N)), 0)[1], 0)

    def test_the_order_is_by_ratio_and_not_by_stress(self) -> None:
        strength, smag = self.field()
        sigma_c = np.ones((N, N), dtype=np.float64)
        smag[12, 12] = 4.0
        sigma_c[12, 12] = 2.0            # ratio 2.0
        smag[13, 13] = 3.0
        sigma_c[13, 13] = 1.0            # ratio 3.0, lower stress
        moved, count = nucleate(strength, smag, sigma_c, 1)
        self.assertEqual(count, 1)
        opened = np.argwhere(seam_mask(moved) & ~seam_mask(strength))
        self.assertEqual(opened.tolist(), [[13, 13]])

    def test_a_tie_breaks_by_row_then_column(self) -> None:
        strength, smag = self.field()
        for cell in ((12, 3), (12, 1), (10, 9)):
            smag[cell] = 2.0
        moved, count = nucleate(strength, smag, np.ones((N, N)), 2)
        self.assertEqual(count, 2)
        opened = sorted(
            np.argwhere(seam_mask(moved) & ~seam_mask(strength)).tolist())
        # Row 10 before row 12, then column 1 before column 3.
        self.assertEqual(opened, [[10, 9], [12, 1]])

    def test_the_intact_strength_field_is_the_noise_clipped(self) -> None:
        noise = np.array([[-100.0, 0.0], [0.5, 100.0]])
        field = intact_strength_field(2.0, noise, 0.1, INTACT_SPREAD_CLIP)
        self.assertAlmostEqual(float(field[0, 0]), 2.0 * 0.2)
        self.assertAlmostEqual(float(field[0, 1]), 2.0)
        self.assertAlmostEqual(float(field[1, 0]), 2.0 * 1.05)
        self.assertAlmostEqual(float(field[1, 1]), 2.0 * 2.0)


class HealingMerges(unittest.TestCase):
    """A seam that seals is not a boundary any more, and nothing says so.

    `label_plates` reads the strength field, so a seam healed back to the
    threshold stops separating the two sides without a line of code.
    """

    def test_a_healed_seam_merges_two_plates_into_one(self) -> None:
        # Two full rows cut the torus into two bands; one alone does not,
        # because a single loop that wraps once leaves the surface connected.
        strength = intact(32)
        strength[8, :] = SEAM_OPEN_STRENGTH
        strength[24, :] = SEAM_OPEN_STRENGTH
        labels = label_plates(strength)
        self.assertEqual(int(labels.max()) + 1, 2)

        healed = strength.copy()
        healed[24, :] = WEAK_THRESHOLD          # at the threshold is intact
        self.assertFalse(bool(seam_mask(healed)[24, 0]))
        self.assertEqual(int(label_plates(healed).max()) + 1, 1)


class NearestCellAdvection(unittest.TestCase):
    def test_the_sampler_takes_the_nearest_cell(self) -> None:
        field = np.arange(16, dtype=np.float64).reshape(4, 4)
        columns = np.arange(4, dtype=np.float64)[None, :]
        rows = np.arange(4, dtype=np.float64)[:, None]
        # A displacement of 0.3 rounds to nothing at all, which is the whole
        # reason the caller carries the remainder.
        same = sample_nearest_periodic(field, columns - 0.3, rows)
        self.assertTrue(np.array_equal(same, field))
        # And it wraps.
        shifted = sample_nearest_periodic(field, columns - 1.0, rows)
        self.assertTrue(np.array_equal(shifted, np.roll(field, 1, axis=1)))

    def test_a_one_cell_seam_stays_one_cell_and_moves_at_the_right_speed(
            self) -> None:
        n = 64
        strength = np.ones((n, n), dtype=np.float64)
        strength[:, 10] = SEAM_OPEN_STRENGTH
        columns = np.arange(n, dtype=np.float64)[None, :]
        rows = np.arange(n, dtype=np.float64)[:, None]
        # A uniform velocity of 0.3 cells per step in +x: the departure point
        # is 0.3 cells behind, so the displacement is -0.3.
        displacement = np.zeros((2, n, n), dtype=np.float64)
        displacement[0] = -0.3
        offset = np.zeros((2, n, n), dtype=np.float64)
        for _ in range(75):
            strength, offset = advect_nearest(strength, displacement, offset,
                                              columns, rows)

        seam = seam_mask(strength)
        widths = seam.sum(axis=1)
        self.assertTrue(bool(np.all(widths == 1)),
                        f"seam widths {sorted(set(widths.tolist()))}")
        column = int(np.argmax(seam[0]))
        self.assertIn((column - 10) % n, (22, 23))
        # Every row moved together: the sheet moved as a body.
        self.assertEqual(len({int(np.argmax(row)) for row in seam}), 1)
        # And the arrears never grew past half a cell.
        self.assertLessEqual(float(np.abs(offset).max()), 0.5)

    def test_bilinear_advection_is_what_would_widen_it(self) -> None:
        # Not a property of the seam formulation but the reason it does not
        # interpolate: the same seam under the sampler the sheet uses spreads
        # into a ramp several cells across, and the ramp is so shallow that
        # after ten steps no cell in it is below the weak threshold any more.
        from engine.domain import sample_bilinear_periodic
        n = 64
        strength = np.ones((n, n), dtype=np.float64)
        strength[:, 10] = SEAM_OPEN_STRENGTH
        columns = np.arange(n, dtype=np.float64)[None, :]
        rows = np.arange(n, dtype=np.float64)[:, None]
        for _ in range(10):
            strength = sample_bilinear_periodic(strength, columns - 0.3, rows)
        spread = int((np.abs(strength[0] - 1.0) > 1e-9).sum())
        self.assertGreaterEqual(spread, 3)
        self.assertGreater(float(strength[0].min()), SEAM_OPEN_STRENGTH)
        self.assertEqual(int(seam_mask(strength)[0].sum()), 0)


class ShortRunChecks:
    """One 128-px history end to end, at the dials §7.1 runs the seeds at.

    A mixin, not a `TestCase`, so the checks that hold under either damage law
    are written once and run twice: `WORK_ORDER_C04_1.md` makes `work_damage`
    mean something under `seams`, so there are two runs to make and both have
    to start intact, converge, and be deterministic. What each law does to the
    seam set is checked in its own class below.
    """

    #: Which law and which rule the run is made under. The subclass sets
    #: both; `SEAMS` is 1 for the sheet's velocity solve and 2 for the block
    #: model of `WORK_ORDER_C04_2.md`.
    WORK_DAMAGE = 0
    SEAMS = 1
    LABEL = ""

    #: How long the run is. 300 Myr is the engine's own history and what the
    #: two `seams = 1` lines have always run; the block model runs 120 Myr
    #: since `WORK_ORDER_C04_6.md`, for the reason its subclass gives.
    HISTORY_MYR = 300.0

    @classmethod
    def setUpClass(cls) -> None:
        cls.params = HistoryParams(seams=cls.SEAMS,
                                   work_damage=cls.WORK_DAMAGE,
                                   history_myr=cls.HISTORY_MYR,
                                   **CORNER_CENTRE)
        cls.world = WorldGeometry(7, 128, 5)
        cls.history = run_history(cls.world, params=cls.params)

    def test_it_starts_from_an_intact_sheet(self) -> None:
        self.assertTrue(bool(np.all(self.history.strength_initial == 1.0)))
        self.assertEqual(int(seam_mask(self.history.strength_initial).sum()), 0)

    def test_the_intact_strength_is_the_noise_around_sigma_c(self) -> None:
        expected = intact_strength_field(
            self.history.sigma_c, strength_noise(self.world),
            self.params.strength_spread, INTACT_SPREAD_CLIP)
        self.assertGreater(self.history.sigma_c, 0.0)
        self.assertTrue(np.array_equal(self.history.sigma_c_field, expected))

    def test_it_is_deterministic(self) -> None:
        again = run_history(self.world, params=self.params)
        for a, b in zip(self.history.epochs, again.epochs):
            with self.subTest(t_myr=a.t_myr):
                self.assertEqual(a.strength.tobytes(), b.strength.tobytes())
                self.assertEqual(a.velocity.tobytes(), b.velocity.tobytes())
                self.assertEqual(a.stress.tobytes(), b.stress.tobytes())
        self.assertEqual(self.history.tip_count, again.tip_count)
        self.assertEqual(self.history.advance_count, again.advance_count)
        self.assertEqual(self.history.nucleation_count,
                         again.nucleation_count)

    def test_every_step_converged(self) -> None:
        worst = max(self.history.solver_residual)
        print()
        print(f"  seam run at 128 px, {self.LABEL} law: worst "
              f"residual {worst:.3e} against MG_TOL {MG_TOL:g}")
        self.assertLess(worst, MG_TOL)

    def test_the_seam_set_is_on_the_record(self) -> None:
        weak = seam_mask(self.history.epochs[-1].strength)
        print(f"  {self.LABEL} law: weak fraction {float(weak.mean()):.5f}, "
              f"edge fraction {search.edge_fraction(weak):.4f}, "
              f"{sum(self.history.nucleation_count)} nuclei and "
              f"{sum(self.history.advance_count)} advances over the run")
        self.assertEqual(len(self.history.seam_fraction), self.history.steps)
        self.assertEqual(self.history.seam_fraction,
                         self.history.weak_fraction)
        for name in ("tip_count", "nucleation_count", "advance_count"):
            with self.subTest(name=name):
                self.assertEqual(len(getattr(self.history, name)),
                                 self.history.steps)
        self.assertEqual(self.history.power_yield, self.history.yield_power)
        self.assertLessEqual(max(self.history.nucleation_count),
                             self.params.nucleations_per_step)


class AShortRunAtTheWorkLaw(ShortRunChecks, unittest.TestCase):
    """`work_damage = 1`: the run C04 made, pinned so C04.1 did not move it."""

    WORK_DAMAGE = 1
    SEAMS = 1
    LABEL = "work"

    #: The weak fraction of the final epoch of this run, measured on this
    #: tree **before** C04.1's edit, when `seams` did not consult
    #: `work_damage` and always damaged by dissipated work. The one mechanism
    #: change must leave this law exactly where it was.
    C04_WEAK_FINAL = 0.004150390625

    def test_the_c04_numbers_are_unchanged(self) -> None:
        weak = seam_mask(self.history.epochs[-1].strength)
        self.assertEqual(float(weak.mean()), self.C04_WEAK_FINAL)

    def test_the_seams_are_one_cell_wide(self) -> None:
        weak = seam_mask(self.history.epochs[-1].strength)
        self.assertGreaterEqual(search.edge_fraction(weak), 0.85)


class AShortRunAtTheBlockModel(ShortRunChecks, unittest.TestCase):
    """`seams = 2`: the same rules on rigid pieces and marker seams.

    The mixin's checks all hold here too — an intact start, the intact
    strength read off step 1, determinism, convergence, and the seam
    trajectory on the record — because `seams = 2` changes the velocity the
    rules act on, not the rules. What is checked below is the block model's
    own record: every step has a piece count and a residual, the rigid solve
    balances to rounding, and the marker set is what the seam raster is made
    of, so the seam set can never be wider than the markers in it.
    """

    WORK_DAMAGE = 0
    SEAMS = 2
    LABEL = "block model"

    #: **120 Myr, not 300.** `WORK_ORDER_C04_6.md` §1.4 subdivides every edge
    #: the move stretches past `SEGMENT_MAX_CELLS`, and on this world the
    #: vertex count grows by about 1.25 per step from the step the first
    #: pieces separate, so a 300 Myr run at 128 px reaches tens of millions of
    #: markers and does not finish. 120 Myr is 30 steps of the same 4 Myr and
    #: is the longest end-to-end block-model run this suite can carry; the
    #: growth itself is measured in `out/C04_6_BUILD_REPORT.md`.
    HISTORY_MYR = 120.0

    def test_the_block_record_is_one_entry_per_step(self) -> None:
        for name in ("piece_count", "largest_piece_share",
                     "force_residual_max", "torque_residual_max",
                     "wrapping_pieces", "marker_count", "gaps_closed",
                     "meetings", "subdivisions", "suture_markers",
                     "reactivations", "sample_count"):
            with self.subTest(name=name):
                self.assertEqual(len(getattr(self.history, name)),
                                 self.history.steps)

    def test_the_rigid_solve_balances(self) -> None:
        worst = max(self.history.force_residual_max)
        turn = max(self.history.torque_residual_max)
        print(f"  block model at 128 px: worst force residual {worst:.3e}, "
              f"worst torque residual {turn:.3e}")
        # The drag sum a residual is measured against is at least the cell
        # count times the drive, thousands; 1e-6 is a rounding error on it.
        self.assertLess(worst, 1e-6)
        self.assertLess(turn, 1e-6)

    def test_the_seam_set_is_exactly_the_cells_the_markers_hold(self) -> None:
        weak = seam_mask(self.history.epochs[-1].strength)
        print(f"  block model: weak fraction {float(weak.mean()):.5f}, "
              f"edge fraction {search.edge_fraction(weak):.4f}, "
              f"{self.history.marker_count[-1]} markers, "
              f"{self.history.piece_count[-1]} pieces, "
              f"largest {self.history.largest_piece_share[-1]:.4f}")
        # A cell is weak only if the curve drew a sample into it, so the weak
        # count can never exceed the sample count: the network cannot be
        # duplicated. Before `WORK_ORDER_C04_6.md` the bound was the marker
        # count, because a marker was the only thing that drew; the curve
        # draws the segments between markers as well, so a chain of `k`
        # markers can hold more than `k` cells.
        self.assertLessEqual(int(weak.sum()), self.history.sample_count[-1])
        self.assertGreater(self.history.marker_count[-1], 0)

    #: Re-pinned by `WORK_ORDER_C04_6.md` §1, whose curve moves every number
    #: that depends on what the raster draws, and at 120 Myr rather than 300
    #: for the reason `HISTORY_MYR` gives. The C04.5 line these replace —
    #: weak fraction 0.09185791015625, 1725 markers, 20 pieces, largest
    #: 0.8968505859375, 150 nuclei, 1883 advances over 300 Myr — was measured
    #: on the point raster and is what the removed `_c04_5` path ran.
    C04_6_WEAK_FINAL = 0.0648193359375
    C04_6_MARKERS = 1323
    C04_6_PIECES = 10
    C04_6_LARGEST = 0.93719482421875
    C04_6_NUCLEI = 60
    C04_6_ADVANCES = 808
    C04_6_MEETINGS = 23
    C04_6_SUBDIVISIONS = 515

    def test_the_pinned_line_is_the_one_this_order_measured(self) -> None:
        """The 128-px run at the default toughness, number for number."""
        self.assertEqual(self.params.toughness_fraction, 1.0)
        weak = seam_mask(self.history.epochs[-1].strength)
        self.assertEqual(float(weak.mean()), self.C04_6_WEAK_FINAL)
        self.assertEqual(self.history.marker_count[-1], self.C04_6_MARKERS)
        self.assertEqual(self.history.piece_count[-1], self.C04_6_PIECES)
        self.assertEqual(float(self.history.largest_piece_share[-1]),
                         self.C04_6_LARGEST)
        self.assertEqual(sum(self.history.nucleation_count),
                         self.C04_6_NUCLEI)
        self.assertEqual(sum(self.history.advance_count),
                         self.C04_6_ADVANCES)
        self.assertEqual(sum(self.history.meetings), self.C04_6_MEETINGS)
        self.assertEqual(sum(self.history.subdivisions),
                         self.C04_6_SUBDIVISIONS)
        # Setting it explicitly is the same run, bit for bit.
        again = run_history(
            self.world,
            params=HistoryParams(seams=2, work_damage=0,
                                 toughness_fraction=1.0,
                                 history_myr=self.HISTORY_MYR,
                                 **CORNER_CENTRE))
        self.assertEqual(again.epochs[-1].strength.tobytes(),
                         self.history.epochs[-1].strength.tobytes())

    def test_the_construction_closes_its_own_gaps(self) -> None:
        """`gap_cells` is what a set of points leaves behind and a curve does
        not: the cells between two linked markers are drawn wherever the two
        ends go. Over this run the tip rule closes one such cell."""
        self.assertLessEqual(sum(self.history.gaps_closed), 1)

    def test_a_lower_toughness_advances_more_tips(self) -> None:
        # Eight steps is enough to separate them and cheap enough to run in
        # the suite.
        half = run_history(
            self.world,
            params=HistoryParams(seams=2, work_damage=0,
                                 toughness_fraction=0.5,
                                 history_myr=self.HISTORY_MYR,
                                 **CORNER_CENTRE),
            steps=8)
        base = run_history(
            self.world,
            params=HistoryParams(seams=2, work_damage=0,
                                 history_myr=self.HISTORY_MYR,
                                 **CORNER_CENTRE),
            steps=8)
        self.assertGreater(sum(half.advance_count), sum(base.advance_count))

    def test_the_step_one_calibration_is_the_sheet_s(self) -> None:
        """At step 1 there are no seams and the one piece stands still.

        The mismatch is then the drive itself and the solve is the sheet's,
        so the three numbers read off it are the ones `seams = 1` reads.
        """
        other = run_history(self.world,
                            params=HistoryParams(seams=1, **CORNER_CENTRE),
                            steps=4)
        self.assertAlmostEqual(self.history.sigma_c, other.sigma_c, places=9)
        self.assertAlmostEqual(self.history.yield_strain_per_myr,
                               other.yield_strain_per_myr, places=12)
        self.assertAlmostEqual(self.history.yield_power, other.yield_power,
                               places=9)

    def test_the_pieces_are_on_the_epoch_for_the_view(self) -> None:
        final = self.history.epochs[-1]
        self.assertIsNotNone(final.piece_labels)
        self.assertEqual(final.piece_labels.shape, final.strength.shape)
        self.assertEqual(final.piece_centroid.shape[0], 2)
        self.assertEqual(final.piece_velocity.shape,
                         final.piece_centroid.shape)
        self.assertEqual(final.mismatch.shape, final.strength.shape)
        # The mismatch is zero wherever a seam is, by construction.
        self.assertEqual(
            float(final.mismatch[final.piece_labels < 0].max(initial=0.0)),
            0.0)


class AShortRunAtTheSlipRateLaw(ShortRunChecks, unittest.TestCase):
    """`work_damage = 0`, the engine's default and C04.1's law."""

    WORK_DAMAGE = 0
    SEAMS = 1
    LABEL = "slip-rate"

    def test_the_seam_set_outlives_the_work_law_s(self) -> None:
        """The one change, end to end: a slipping seam does not heal shut."""
        weak = seam_mask(self.history.epochs[-1].strength)
        self.assertGreater(float(weak.mean()),
                           AShortRunAtTheWorkLaw.C04_WEAK_FINAL)

    def test_the_sheet_at_the_same_dials_keeps_its_own_start(self) -> None:
        sheet = run_history(self.world,
                            params=HistoryParams(**CORNER_CENTRE), steps=4)
        self.assertFalse(bool(np.all(sheet.strength_initial == 1.0)))
        self.assertEqual(sheet.sigma_c, 0.0)
        self.assertEqual(sum(sheet.tip_count), 0)
        self.assertEqual(sum(sheet.nucleation_count), 0)


if __name__ == "__main__":
    unittest.main()
