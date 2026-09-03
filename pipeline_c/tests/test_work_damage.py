"""Gates on the work-based damage law of `WORK_ORDER_C03_8.md`.

`work_damage` is one integer on `HistoryParams`. At `0` the sheet damages on
the excess of the strain rate over its own percentile, which is what the
engine did before this order; at `1` it damages on the excess of the
dissipated power over the same percentile of the power field. Everything
after the excess — the square, the exact integrator, healing, the floor, the
advection — is the same arithmetic either way.

The first gate is the one that matters for production: at `work_damage = 0`
the history is byte-identical to the pre-order engine. `test_regression_c03_1`
covers the production default; this covers a dial set well away from it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

from engine.geometry import WorldGeometry  # noqa: E402
from engine.history.constants import (  # noqa: E402
    SOLVE_GRID_DIVISOR,
    STRENGTH_WAVELENGTH_KM,
)
from engine.noise import band_cycles  # noqa: E402
from engine.history.drive import build_drive  # noqa: E402
from engine.history.kinematics import (  # noqa: E402
    HistoryParams,
    run_history,
    solve_n_for,
    to_kinematic_blocks,
    to_solve_grid,
)
from engine.history.solver import (  # noqa: E402
    effective_gradients,
    kappa0_for,
    solve,
)

WORLD = WorldGeometry(7, 128, 5)

#: The world the byte-identity gate runs on: 512 delivered pixels at 10 km
#: per pixel, whose parent is 10,240 km on a 128-cell grid.
#:
#: **Why not `WORLD`.** `WORK_ORDER_C03_10.md` put the two noise bands in
#: kilometres. `STRENGTH_WAVELENGTH_KM` is 1,280 km, which is the parent over
#: eight on a 10,240 km parent and the parent over four on the 5,120 km parent
#: of a 128-px world, so the initial strength field at `WORLD` is a different
#: field afterwards — deliberately, since that order's whole point is that a
#: smaller world holds fewer mantle and strength cells of the same size rather
#: than the same number of smaller ones. The gate therefore moved to a world
#: where the order changes nothing: at a 10,240 km parent the strength band is
#: 8.0 cycles and the drive band below is 3.0 cycles, exactly the integers the
#: pre-order engine passed, so `_radial_envelope` builds the identical arrays
#: and the whole history is bit-for-bit what the old engine produced there.
GATE_WORLD = WorldGeometry(7, 512, 10)

#: A dial set well away from every default, so the byte-identity gate is not
#: a second copy of the production regression.
OFF_DEFAULT = dict(
    stiffness_fraction=0.3,
    yield_percentile=5.0,
    heal_time_myr=40.0,
    damage_time_myr=3.0,
    strength_exponent=3,
    strength_spread=0.05,
    #: Exactly the parent over three on `GATE_WORLD`, and `10240.0 / 3.0`
    #: divides back into 10,240 as exactly 3.0 in floating point, so this is
    #: the band the pre-order engine built from `drive_nodes = 3`.
    drive_wavelength_km=10240.0 / 3.0,
    drive_shear=0.25,
)

#: SHA-256 over the four kept epochs' strength fields, in order, of
#: `run_history(GATE_WORLD, params=HistoryParams(**OFF_DEFAULT), steps=8)`.
#:
#: Recorded 2026-09-02 for `WORK_ORDER_C03_10.md`, on a world and a dial set
#: this order leaves bit-identical for the reason `GATE_WORLD` gives: the two
#: envelopes are the same arrays the node counts built, and nothing else in
#: the loop moved, so this is the value the pre-`WORK_ORDER_C03_8.md` engine
#: produces there. The 128-px anchor it replaces, recorded the same day
#: against that engine directly, was
#: `c7bc454c8dae438f24d1e2ce9bf5f0f064d448860c3af09f82e9d7f925b5d5c5`, and it
#: cannot survive a change that is a change of world at every size but one.
PRE_ORDER_STRENGTH_SHA256 = (
    "edd7edc4c1dbf1ca6d49b6f2b4487c82879fb458aadf8df84b905340bd03caf0")


def strength_digest(history) -> str:
    digest = hashlib.sha256()
    for epoch in history.epochs:
        digest.update(epoch.strength.tobytes())
    return digest.hexdigest()


def first_step_solve(params: HistoryParams):
    """Step one, recomputed outside the loop, on the solve grid.

    The same calls in the same order as `run_history`, so what comes back is
    bit-for-bit what the loop formed on its first step.
    """
    n = WORLD.history_n
    solve_n = solve_n_for(n)
    drive = build_drive(WORLD, wavelength_km=params.drive_wavelength_km,
                        rot_ratio=params.drive_shear,
                        history_myr=params.history_myr)
    from engine.history.kinematics import initial_strength
    strength = initial_strength(WORLD, params.strength_spread)
    kappa = kappa0_for(n, params.stiffness_fraction) \
        * strength**params.strength_exponent
    kappa_s, traction_s = to_solve_grid(kappa, drive.field(0.0), solve_n)
    solved, _cycles, _residual = solve(traction_s, kappa_s, u0=None,
                                       max_cycles=params.max_cycles)
    g_x, g_y = effective_gradients(solved, kappa_s)
    cell_s_km = WORLD.cell_km * SOLVE_GRID_DIVISOR
    exx = g_x[0] / cell_s_km
    eyy = g_y[1] / cell_s_km
    exy = 0.5 * (g_y[0] + g_x[1]) / cell_s_km
    strain_rate_s = np.sqrt(exx * exx + eyy * eyy + 2.0 * exy * exy)
    return kappa_s, strain_rate_s


class TheDial(unittest.TestCase):
    def test_the_default_is_the_strain_rate_law(self) -> None:
        self.assertEqual(HistoryParams().work_damage, 0)
        self.assertEqual(HistoryParams().to_record()["work_damage"], 0)
        self.assertEqual(
            HistoryParams(work_damage=1).to_record()["work_damage"], 1)

    def test_it_refuses_two_and_a_bool(self) -> None:
        with self.assertRaises(ValueError):
            HistoryParams(work_damage=2)
        with self.assertRaises(ValueError):
            HistoryParams(work_damage=True)
        with self.assertRaises(ValueError):
            HistoryParams(work_damage=-1)
        with self.assertRaises(ValueError):
            HistoryParams(work_damage=0.0)


class TheProductionPath(unittest.TestCase):
    """`work_damage = 0` must not have moved anything by one bit."""

    def test_a_non_default_dial_set_is_byte_identical_to_the_old_engine(self) -> None:
        history = run_history(GATE_WORLD, params=HistoryParams(**OFF_DEFAULT),
                              steps=8)
        self.assertEqual(strength_digest(history), PRE_ORDER_STRENGTH_SHA256)

    def test_the_gate_world_is_one_the_kilometre_bands_do_not_move(self) -> None:
        # The claim `GATE_WORLD` rests on, checked rather than asserted: on a
        # 10,240 km parent the two bands in kilometres are the two integer
        # cycle counts the pre-order engine used, so the envelopes are the
        # same arrays and the history is the old one.
        self.assertEqual(GATE_WORLD.parent_km, 10240)
        self.assertEqual(
            band_cycles(GATE_WORLD, wavelength_km=STRENGTH_WAVELENGTH_KM), 8.0)
        self.assertEqual(
            band_cycles(GATE_WORLD,
                        wavelength_km=OFF_DEFAULT["drive_wavelength_km"]), 3.0)

    def test_the_work_law_at_the_same_dials_is_a_different_history(self) -> None:
        work = run_history(
            GATE_WORLD, params=HistoryParams(work_damage=1, **OFF_DEFAULT),
            steps=8)
        self.assertNotEqual(strength_digest(work), PRE_ORDER_STRENGTH_SHA256)


class ThePowerField(unittest.TestCase):
    def setUp(self) -> None:
        self.params = HistoryParams()
        self.history = run_history(WORLD, params=self.params, steps=4)
        # `epoch_steps(4)` is [1, 2, 3, 4], so the first kept epoch is step 1,
        # which is the step the two yields are read on.
        self.first = self.history.epochs[0]

    def test_power_is_the_stiffness_times_the_squared_strain_rate(self) -> None:
        kappa_s, strain_rate_s = first_step_solve(self.params)
        expected = kappa_s * strain_rate_s * strain_rate_s
        lifted = to_kinematic_blocks(expected, WORLD.history_n)
        self.assertEqual(self.first.power.shape, self.first.strain_rate.shape)
        self.assertTrue(np.allclose(self.first.power, lifted,
                                    rtol=1e-12, atol=0.0))

    def test_power_is_never_negative(self) -> None:
        for epoch in self.history.epochs:
            with self.subTest(t_myr=epoch.t_myr):
                self.assertTrue(bool(np.all(epoch.power >= 0.0)))

    def test_the_power_yield_is_that_percentile_of_the_power_field(self) -> None:
        kappa_s, strain_rate_s = first_step_solve(self.params)
        power_s = kappa_s * strain_rate_s * strain_rate_s
        expected = float(np.percentile(
            power_s, 100.0 - self.params.yield_percentile, method="linear"))
        self.assertEqual(self.history.yield_power, expected)
        self.assertGreater(self.history.yield_power, 0.0)

    def test_both_yields_are_read_whatever_the_law(self) -> None:
        work = run_history(WORLD, params=HistoryParams(work_damage=1), steps=4)
        self.assertEqual(work.yield_power, self.history.yield_power)
        self.assertEqual(work.yield_strain_per_myr,
                         self.history.yield_strain_per_myr)


class TheTwoLawsAtUniformStiffness(unittest.TestCase):
    """At `strength_spread = 0` the sheet starts uniform, so the stiffness is
    one number and the power is a monotone function of the strain rate. The
    same percentile of the two fields therefore falls between the same pair of
    neighbouring order statistics, and the two laws select the same cells."""

    def test_the_sets_above_threshold_are_the_same(self) -> None:
        history = run_history(WORLD, params=HistoryParams(strength_spread=0.0),
                              steps=4)
        first = history.epochs[0]
        by_strain = first.strain_rate > history.yield_strain_per_myr
        by_power = first.power > history.yield_power
        self.assertGreater(int(by_strain.sum()), 0)
        self.assertTrue(bool(np.array_equal(by_strain, by_power)))

    def test_the_two_runs_exceed_on_the_same_share_of_the_first_step(self) -> None:
        flat = dict(strength_spread=0.0)
        strain = run_history(WORLD, params=HistoryParams(**flat), steps=4)
        work = run_history(WORLD, params=HistoryParams(work_damage=1, **flat),
                           steps=4)
        self.assertEqual(strain.exceed_fraction[0], work.exceed_fraction[0])
        # The same cells are above threshold; how far above differs, because
        # the power excess is not the strain excess, so the strength fields
        # part company on that very first step.
        self.assertNotEqual(strain.epochs[0].strength.tobytes(),
                            work.epochs[0].strength.tobytes())

    def test_a_heterogeneous_sheet_separates_the_two_laws(self) -> None:
        history = run_history(WORLD, params=HistoryParams(), steps=4)
        first = history.epochs[0]
        by_strain = first.strain_rate > history.yield_strain_per_myr
        by_power = first.power > history.yield_power
        # The same count, because both are the same percentile; different
        # cells, because the stiffness is no longer one number.
        self.assertEqual(int(by_strain.sum()), int(by_power.sum()))
        self.assertFalse(bool(np.array_equal(by_strain, by_power)))


if __name__ == "__main__":
    unittest.main()
