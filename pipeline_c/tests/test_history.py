"""Gates on the drive field and the history loop."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import unittest

import numpy as np

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

from engine.geometry import WorldGeometry  # noqa: E402
from engine.history.constants import (  # noqa: E402
    DAMAGE_RATE,
    SOLVE_GRID_DIVISOR,
    DRIVE_KEYFRAMES,
    DRIVE_RMS_KM_PER_MYR,
    DRIVE_ROT_RATIO,
    DRIVE_WAVELENGTH_KM,
    HEAL_RATE,
    HISTORY_MYR,
    HOMOG_LENGTH_FRACTION,
    MG_MAX_CYCLES,
    MG_TOL,
    STEP_MYR,
    STRENGTH_EXPONENT,
    STRENGTH_INIT_SPREAD,
    STRENGTH_MIN,
)
from engine.domain import grad, perp_grad  # noqa: E402
from engine.history.drive import build_drive  # noqa: E402
from engine.history.kinematics import (  # noqa: E402
    DEFAULT_PARAMS,
    EARLY_SNAPSHOT_STEPS,
    HistoryParams,
    default_steps,
    epoch_steps,
    initial_strength,
    run_history,
    solve_n_for,
    to_solve_grid,
)
from engine.history.solver import (  # noqa: E402
    effective_gradients,
    kappa0_for,
    solve,
)
from tools.spectrum import axis_to_diagonal  # noqa: E402
import webui_adapter  # noqa: E402

AUDIT_SEED = 4287772760


def step_one_strain_on_the_solve_grid(
        geometry: WorldGeometry,
        params: HistoryParams = DEFAULT_PARAMS) -> np.ndarray:
    """The first step's strain field, written out so the test reads no code.

    This is the engine's own first step: the initial strength, the drive at
    `t = 0`, both coarsened to the solve grid, solved, and the strain built
    from the solver's edge fluxes. It is the field the yield percentile is
    read off.
    """
    strength = initial_strength(geometry)
    kappa = (kappa0_for(geometry.history_n, params.stiffness_fraction)
             * strength ** params.strength_exponent)
    traction = build_drive(geometry,
                           wavelength_km=params.drive_wavelength_km,
                           rot_ratio=params.drive_shear,
                           history_myr=params.history_myr).field(0.0)
    kappa_s, traction_s = to_solve_grid(
        kappa, traction, solve_n_for(geometry.history_n))
    solved = solve(traction_s, kappa_s, max_cycles=params.max_cycles)[0]
    g_x, g_y = effective_gradients(solved, kappa_s)
    cell_s_km = geometry.cell_km * SOLVE_GRID_DIVISOR
    exx = g_x[0] / cell_s_km
    eyy = g_y[1] / cell_s_km
    exy = 0.5 * (g_y[0] + g_x[1]) / cell_s_km
    return np.sqrt(exx * exx + eyy * eyy + 2.0 * exy * exy)


def one_step(strength: float, strain_rate: float, yield_strain: float,
             dt: float) -> float:
    """The engine's law, applied to a single cell."""
    excess = max(strain_rate / yield_strain - 1.0, 0.0)
    rate = DAMAGE_RATE * excess * excess
    total = HEAL_RATE + rate
    equilibrium = HEAL_RATE / total
    return min(max(equilibrium + (strength - equilibrium) * math.exp(-total * dt),
                   STRENGTH_MIN), 1.0)

FLOOR_WORLD = WorldGeometry(7, 128, 5)


def rms_speed(field: np.ndarray) -> float:
    return float(np.sqrt(np.mean(field[0] ** 2 + field[1] ** 2)))


def seam_and_interior(field: np.ndarray) -> tuple[float, float]:
    seam = max(float(np.abs(field[..., :, 0] - field[..., :, -1]).max()),
               float(np.abs(field[..., 0, :] - field[..., -1, :]).max()))
    interior = max(float(np.abs(np.diff(field, axis=-1)).max()),
                   float(np.abs(np.diff(field, axis=-2)).max()))
    return seam, interior


class DriveField(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = WorldGeometry(4287772760, 512, 5)
        cls.drive = build_drive(cls.geometry)

    def test_speed_is_normalized_at_the_first_keyframe(self) -> None:
        self.assertAlmostEqual(rms_speed(self.drive.field(0.0)),
                               DRIVE_RMS_KM_PER_MYR, delta=1e-9)

    def test_the_field_has_no_seam(self) -> None:
        for t_myr in (0.0, HISTORY_MYR / 3.0, HISTORY_MYR):
            with self.subTest(t_myr=t_myr):
                seam, interior = seam_and_interior(self.drive.field(t_myr))
                self.assertLessEqual(seam, interior)

    def test_a_keyframe_time_gives_that_keyframe_exactly(self) -> None:
        span = HISTORY_MYR / (DRIVE_KEYFRAMES - 1)
        for keyframe in range(DRIVE_KEYFRAMES):
            with self.subTest(keyframe=keyframe):
                alone = self.drive.scale * (
                    grad(self.drive.phi[keyframe])
                    + DRIVE_ROT_RATIO * perp_grad(self.drive.psi[keyframe]))
                blended = self.drive.field(keyframe * span)
                self.assertLess(float(np.abs(blended - alone).max()), 1e-9)

    def test_the_field_between_keyframes_is_neither_of_them(self) -> None:
        span = HISTORY_MYR / (DRIVE_KEYFRAMES - 1)
        middle = self.drive.field(span * 0.5)
        for keyframe in (0, 1):
            self.assertGreater(
                float(np.abs(middle - self.drive.field(keyframe * span)).max()),
                1e-6)

    def test_time_is_clamped_to_the_history(self) -> None:
        self.assertTrue(np.array_equal(self.drive.field(-10.0),
                                       self.drive.field(0.0)))
        self.assertTrue(np.array_equal(self.drive.field(HISTORY_MYR + 10.0),
                                       self.drive.field(HISTORY_MYR)))

    def test_the_drive_is_deterministic(self) -> None:
        again = build_drive(self.geometry)
        self.assertEqual(again.phi.tobytes(), self.drive.phi.tobytes())
        self.assertEqual(again.psi.tobytes(), self.drive.psi.tobytes())
        self.assertEqual(again.scale, self.drive.scale)

    def test_a_different_world_drives_differently(self) -> None:
        other = build_drive(WorldGeometry(1, 512, 5))
        self.assertFalse(np.array_equal(other.phi, self.drive.phi))


class ShortRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.history = run_history(FLOOR_WORLD, steps=8)

    #: Strength is clipped to `[STRENGTH_MIN, 1]` and then advected. The
    #: advection is a bilinear sample, so a cell whose four sources all sit
    #: exactly on the floor can land one unit in the last place below it. That
    #: is rounding in the interpolation, not a cell below the floor.
    FLOOR_SLACK = 1e-12

    def test_strength_stays_in_range(self) -> None:
        for epoch in self.history.epochs:
            with self.subTest(t_myr=epoch.t_myr):
                self.assertGreaterEqual(float(epoch.strength.min()),
                                        STRENGTH_MIN - self.FLOOR_SLACK)
                self.assertLessEqual(float(epoch.strength.max()), 1.0)

    def test_one_record_per_step(self) -> None:
        self.assertEqual(len(self.history.weak_fraction), 8)
        self.assertEqual(len(self.history.solver_cycles), 8)
        self.assertEqual(len(self.history.solver_residual), 8)

    def test_every_solve_either_converged_or_spent_its_budget(self) -> None:
        # Run 4 already reported 67 of 75 steps at 1024 px stopping at
        # `MG_MAX_CYCLES` rather than at `MG_TOL`; the isotropic noise puts
        # power at every wavelength up to the Nyquist, so the floor world now
        # does the same. What the solver promises is that a step stops at one
        # of the two, and reports the residual it reached.
        worst = max(self.history.solver_residual)
        print(f"\n  floor world, {self.history.steps} steps: worst residual "
              f"{worst:.3e} against MG_TOL {MG_TOL:g}, cycles "
              f"{min(self.history.solver_cycles)}-"
              f"{max(self.history.solver_cycles)} of {MG_MAX_CYCLES}")
        for cycles, residual in zip(self.history.solver_cycles,
                                    self.history.solver_residual):
            with self.subTest(cycles=cycles):
                self.assertTrue(residual < MG_TOL or cycles == MG_MAX_CYCLES)
        self.assertLess(worst, 10.0 * MG_TOL)

    def test_four_epochs_at_the_declared_steps(self) -> None:
        self.assertEqual(epoch_steps(8), [2, 4, 6, 8])
        self.assertEqual(len(self.history.epochs), 4)
        step_myr = self.history.step_myr
        self.assertEqual([round(e.t_myr / step_myr) for e in self.history.epochs],
                         [2, 4, 6, 8])
        self.assertAlmostEqual(self.history.epochs[-1].t_myr, HISTORY_MYR, places=9)

    def test_a_short_run_still_spans_the_whole_drive_schedule(self) -> None:
        self.assertAlmostEqual(self.history.step_myr * self.history.steps,
                               HISTORY_MYR, places=9)

    def test_a_step_count_that_is_not_a_multiple_of_four_is_refused(self) -> None:
        for steps in (5, 3, 0, -4):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                run_history(FLOOR_WORLD, steps=steps)

    def test_the_default_step_count_follows_the_constants(self) -> None:
        self.assertEqual(default_steps(), 75)
        self.assertEqual(epoch_steps(75), [19, 38, 56, 75])


class YieldStrain(unittest.TestCase):
    """The yield is a percentile of the first step's own strain field.

    A stiffer sheet has smaller strains everywhere, so a yield written as a
    fraction of the drive's characteristic strain means something different at
    every stiffness. Reading it off the first step's distribution makes the
    dial mean one thing throughout the exploration. It is a calibration
    convenience and it is not to survive into production; the docstring of
    `run_history` says so.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.geometry = WorldGeometry(AUDIT_SEED, 1024, 5)
        cls.strain = step_one_strain_on_the_solve_grid(cls.geometry)

    def test_the_default_yield_is_the_declared_percentile(self) -> None:
        history = run_history(self.geometry, steps=4)
        expected = float(np.percentile(self.strain, 100.0 - 12.0,
                                       method="linear"))
        self.assertAlmostEqual(history.yield_strain_per_myr, expected, places=12)

    def test_percentile_fifty_is_the_median_of_the_step_one_strain(self) -> None:
        params = HistoryParams(yield_percentile=50.0)
        history = run_history(self.geometry, params=params, steps=4)
        self.assertAlmostEqual(history.yield_strain_per_myr,
                               float(np.median(self.strain)), places=12)

    def test_the_step_one_exceedance_is_the_percentile(self) -> None:
        # What the percentile buys: the share of the field above yield on the
        # first step is the dial itself, at any stiffness.
        for percentile in (3.0, 12.0, 30.0):
            for stiffness in (0.125, 1.0):
                params = HistoryParams(yield_percentile=percentile,
                                       stiffness_fraction=stiffness)
                history = run_history(self.geometry, params=params, steps=4)
                with self.subTest(percentile=percentile, stiffness=stiffness):
                    self.assertAlmostEqual(100.0 * history.exceed_fraction[0],
                                           percentile, delta=0.5)

    def test_the_default_yield_on_the_audit_seed_is_pinned(self) -> None:
        # Run 4 measured 0.019640 per Myr on this world, with 12.7 % of the
        # first strain field above it. The percentile derivation reproduces
        # that share by construction; the value moved because section 2
        # replaced the noise, so this is a different first strain field, not a
        # different rule.
        history = run_history(self.geometry, steps=4)
        print(f"\n  default-dial yield on seed {AUDIT_SEED} at 1024 px: "
              f"{history.yield_strain_per_myr:.6f} per Myr, against run 4's "
              f"0.019640")
        self.assertAlmostEqual(history.yield_strain_per_myr, 0.010748,
                               delta=1e-6)

    def test_the_report_carries_the_yield(self) -> None:
        world = webui_adapter.generate(1, {"scale_km": 5}, 1024, _steps=4)
        record = webui_adapter.report(world)
        self.assertIn("yield_strain_per_myr", record)
        self.assertNotIn("strain_ref_per_myr", record)
        self.assertGreater(record["yield_strain_per_myr"], 0.0)


class Params(unittest.TestCase):
    """`HistoryParams` is a record of what `constants.py` already holds."""

    def test_the_defaults_are_the_constants(self) -> None:
        params = HistoryParams()
        self.assertEqual(params.stiffness_fraction, HOMOG_LENGTH_FRACTION)
        self.assertEqual(params.yield_percentile, 12.0)
        self.assertAlmostEqual(params.heal_time_myr, 1.0 / HEAL_RATE, places=12)
        self.assertAlmostEqual(params.damage_time_myr, 1.0 / DAMAGE_RATE,
                               places=12)
        self.assertEqual(params.work_damage, 0)
        self.assertEqual(params.strength_exponent, STRENGTH_EXPONENT)
        self.assertEqual(params.strength_spread, STRENGTH_INIT_SPREAD)
        self.assertEqual(params.drive_wavelength_km, DRIVE_WAVELENGTH_KM)
        self.assertEqual(params.drive_shear, DRIVE_ROT_RATIO)
        self.assertEqual(params.history_myr, HISTORY_MYR)
        self.assertEqual(params.max_cycles, MG_MAX_CYCLES)
        self.assertEqual(params.solve_divisor, SOLVE_GRID_DIVISOR)
        self.assertAlmostEqual(params.heal_rate, HEAL_RATE, places=12)
        self.assertAlmostEqual(params.damage_rate, DAMAGE_RATE, places=12)
        self.assertEqual(params.steps, round(HISTORY_MYR / STEP_MYR))
        self.assertEqual(params, DEFAULT_PARAMS)

    def test_every_range_is_enforced(self) -> None:
        for name, below, above in (
            ("stiffness_fraction", 0.019, 4.001),
            ("yield_percentile", 0.4, 50.1),
            ("heal_time_myr", 4.9, 2001.0),
            ("damage_time_myr", 0.49, 201.0),
            ("work_damage", -1, 2),
            ("strength_exponent", 0, 9),
            ("strength_spread", -0.001, 0.301),
            ("drive_wavelength_km", 99.0, 100001.0),
            ("drive_shear", -0.001, 2.001),
            ("history_myr", 49.0, 1001.0),
            ("max_cycles", 4, 201),
            ("solve_divisor", 0, 3),
        ):
            for value in (below, above):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        HistoryParams(**{name: value})

    def test_the_drive_wavelength_refuses_zero_a_negative_and_true(self) -> None:
        # `WORK_ORDER_C03_10.md` §4. It is a length, so it is a number and
        # not an integer count, but it is still bounded and still not a bool.
        for value in (0, 0.0, -5120.0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                HistoryParams(drive_wavelength_km=value)
        self.assertEqual(HistoryParams(drive_wavelength_km=1234.5)
                         .drive_wavelength_km, 1234.5)

    def test_the_record_carries_the_wavelength_and_not_the_node_count(self) -> None:
        record = HistoryParams().to_record()
        self.assertEqual(record["drive_wavelength_km"], DRIVE_WAVELENGTH_KM)
        self.assertNotIn("drive_nodes", record)
        self.assertIsInstance(record["drive_wavelength_km"], float)

    def test_the_integer_dials_refuse_a_float_or_a_bool(self) -> None:
        for name in ("work_damage", "strength_exponent",
                     "max_cycles", "solve_divisor"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                HistoryParams(**{name: 4.5})
        with self.assertRaises(ValueError):
            HistoryParams(stiffness_fraction=True)

    def test_the_spread_dial_scales_the_initial_heterogeneity(self) -> None:
        default = initial_strength(FLOOR_WORLD)
        same = initial_strength(FLOOR_WORLD, STRENGTH_INIT_SPREAD)
        self.assertEqual(default.tobytes(), same.tobytes())
        flat = initial_strength(FLOOR_WORLD, 0.0)
        # Every cell is the same number; `std` of a constant field is not
        # exactly zero in floating point, but its range is.
        self.assertEqual(float(flat.max()), float(flat.min()))
        # The noise field is unchanged; only its amplitude moves. Measured
        # well below the default, where the clip at 1.0 never bites.
        small = initial_strength(FLOOR_WORLD, 0.02)
        smaller = initial_strength(FLOOR_WORLD, 0.01)
        self.assertLess(float(small.max()), 1.0)
        self.assertAlmostEqual(float(smaller.std()) / float(small.std()), 0.5,
                               places=9)
        self.assertLess(float(smaller.std()), float(default.std()))

    def test_the_spread_dial_reaches_the_history(self) -> None:
        flat = run_history(FLOOR_WORLD, params=HistoryParams(strength_spread=0.0),
                           steps=4)
        spread = run_history(FLOOR_WORLD, params=HistoryParams(), steps=4)
        self.assertEqual(float(flat.strength_initial.max()),
                         float(flat.strength_initial.min()))
        self.assertNotEqual(flat.epochs[-1].strength.tobytes(),
                            spread.epochs[-1].strength.tobytes())

    def test_run_history_refuses_anything_but_params(self) -> None:
        with self.assertRaises(TypeError):
            run_history(FLOOR_WORLD, params={"stiffness_fraction": 0.5}, steps=4)

    def test_default_params_reproduce_no_params_byte_for_byte(self) -> None:
        without = run_history(FLOOR_WORLD, steps=8)
        with_defaults = run_history(FLOOR_WORLD, params=HistoryParams(), steps=8)
        for a, b in zip(without.epochs, with_defaults.epochs):
            self.assertEqual(a.strength.tobytes(), b.strength.tobytes())
            self.assertEqual(a.velocity.tobytes(), b.velocity.tobytes())
            self.assertEqual(a.strain_rate.tobytes(), b.strain_rate.tobytes())
        self.assertEqual(without.weak_fraction, with_defaults.weak_fraction)
        self.assertEqual(without.solver_cycles, with_defaults.solver_cycles)
        self.assertEqual(without.yield_strain_per_myr,
                         with_defaults.yield_strain_per_myr)

    def test_a_dial_that_moves_changes_the_history(self) -> None:
        base = run_history(FLOOR_WORLD, steps=8)
        for params in (HistoryParams(stiffness_fraction=0.5),
                       HistoryParams(yield_percentile=30.0),
                       HistoryParams(heal_time_myr=20.0),
                       HistoryParams(damage_time_myr=50.0),
                       HistoryParams(strength_exponent=2),
                       HistoryParams(drive_wavelength_km=2560.0),
                       HistoryParams(drive_shear=0.0)):
            with self.subTest(params=params):
                other = run_history(FLOOR_WORLD, params=params, steps=8)
                self.assertNotEqual(base.epochs[-1].strength.tobytes(),
                                    other.epochs[-1].strength.tobytes())

    def test_the_history_length_sets_the_step_count(self) -> None:
        history = run_history(FLOOR_WORLD,
                              params=HistoryParams(history_myr=100.0))
        self.assertEqual(history.steps, 25)
        self.assertAlmostEqual(history.step_myr, STEP_MYR, places=9)
        self.assertAlmostEqual(history.epochs[-1].t_myr, 100.0, places=9)
        self.assertEqual(default_steps(100.0), 25)


class InitialStrengthIsotropy(unittest.TestCase):
    """The field damage starts from must not prefer the world's axes.

    It enters the stiffness at the fourth power, so an axis bias here is
    amplified into every velocity and every zone. The lattice noise this
    replaced measured 1.777 on the audit seed in every run's audit.
    """

    def test_the_twelve_development_seeds_are_within_the_bound(self) -> None:
        seeds = (2075014389, 2477733044, 476149591, 151640007, 2697441485,
                 1504571935, 548870008, 2157195430, 4108373596, 4287772760,
                 287488203, 1833546021)
        values = [axis_to_diagonal(initial_strength(WorldGeometry(seed, 1024, 5)))
                  for seed in seeds]
        print(f"\n  initial strength isotropy over twelve seeds: "
              f"min {min(values):.4f} max {max(values):.4f} "
              f"mean {float(np.mean(values)):.4f}")
        for seed, value in zip(seeds, values):
            with self.subTest(seed=seed):
                self.assertLess(value, 1.15)
                self.assertGreater(value, 1.0 / 1.15)


class YieldLaw(unittest.TestCase):
    """Below yield a cell heals; above it a cell fails."""

    #: A representative yield. The law depends only on the ratio of strain to
    #: yield, so the value has only to be positive; this is the order of the
    #: yields the audit world produces.
    YIELD = 0.02

    def setUp(self) -> None:
        self.yield_strain = self.YIELD

    def test_a_cell_below_yield_gains_strength_in_one_step(self) -> None:
        after = one_step(0.3, 0.9 * self.yield_strain, self.yield_strain, 4.0)
        print(f"\n  below yield: 0.300000 -> {after:.6f} over one 4 Myr step")
        self.assertGreater(after, 0.3)

    def test_a_cell_below_yield_recovers_over_four_hundred_myr(self) -> None:
        strength = 0.3
        for _ in range(100):
            strength = one_step(strength, 0.9 * self.yield_strain,
                                self.yield_strain, 4.0)
        print(f"  below yield: 0.300000 -> {strength:.6f} over 400 Myr")
        self.assertGreater(strength, 0.98)

    def test_a_cell_at_three_times_yield_loses_strength_in_one_step(self) -> None:
        after = one_step(1.0, 3.0 * self.yield_strain, self.yield_strain, 4.0)
        print(f"  three times yield: 1.000000 -> {after:.6f} over one 4 Myr step")
        self.assertLess(after, 1.0)

    def test_exactly_at_yield_there_is_no_damage(self) -> None:
        after = one_step(0.3, self.yield_strain, self.yield_strain, 4.0)
        self.assertGreater(after, 0.3)


class Trajectory(unittest.TestCase):
    """The per-step record the build report reads."""

    def test_one_entry_per_step_in_every_series(self) -> None:
        history = run_history(FLOOR_WORLD, steps=8)
        for series in (history.weak_fraction, history.exceed_fraction,
                       history.strength_mean, history.strength_min,
                       history.strain_rate_mean, history.strain_rate_max):
            self.assertEqual(len(series), 8)


class StrainIsResolvedOnTheSolveGrid(unittest.TestCase):
    """Damage is driven by the strain of the cell that carries the stress.

    The strain comes from the solver's edge fluxes on the solve grid and is
    lifted piecewise constant, so on the kinematic grid every 2 x 2 block
    holds one value. A bilinear lift would put intermediate strain into the
    strong cells beside a failed one, which is the halo this discretization
    removes.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.history = run_history(FLOOR_WORLD, steps=8)

    def test_the_strain_field_is_on_the_kinematic_grid(self) -> None:
        n = FLOOR_WORLD.history_n
        for epoch in self.history.epochs:
            with self.subTest(t_myr=epoch.t_myr):
                self.assertEqual(epoch.strain_rate.shape, (n, n))
                self.assertEqual(epoch.divergence.shape, (n, n))

    def test_every_block_of_the_strain_field_holds_one_value(self) -> None:
        size = SOLVE_GRID_DIVISOR
        for epoch in self.history.epochs:
            for name, field in (("strain_rate", epoch.strain_rate),
                                ("divergence", epoch.divergence)):
                with self.subTest(t_myr=epoch.t_myr, field=name):
                    n = field.shape[0]
                    blocks = field.reshape(n // size, size, n // size, size)
                    spread = float(np.abs(
                        blocks - blocks[:, :1, :, :1]).max())
                    self.assertEqual(spread, 0.0)

    def test_the_field_is_not_constant_over_the_whole_grid(self) -> None:
        # Block-constant, not constant: the blocks differ from each other.
        for epoch in self.history.epochs:
            with self.subTest(t_myr=epoch.t_myr):
                self.assertGreater(float(np.ptp(epoch.strain_rate)), 0.0)


class EarlySnapshots(unittest.TestCase):
    """Report-only strength snapshots from before the first kept epoch."""

    def test_one_snapshot_per_declared_step(self) -> None:
        history = run_history(FLOOR_WORLD, steps=20)
        self.assertEqual([step for step, _, _ in history.early],
                         list(EARLY_SNAPSHOT_STEPS))
        for step, t_myr, strength in history.early:
            with self.subTest(step=step):
                self.assertAlmostEqual(t_myr, step * history.step_myr, places=9)
                self.assertEqual(strength.shape,
                                 (FLOOR_WORLD.history_n,) * 2)

    def test_a_run_shorter_than_the_last_snapshot_keeps_what_it_reached(self) -> None:
        history = run_history(FLOOR_WORLD, steps=8)
        self.assertEqual([step for step, _, _ in history.early], [2, 4, 8])


class StrengthMoves(unittest.TestCase):
    def test_the_field_is_not_the_one_it_started_from(self) -> None:
        history = run_history(FLOOR_WORLD, steps=4)
        start = initial_strength(FLOOR_WORLD)
        self.assertTrue(np.array_equal(history.strength_initial, start))
        self.assertFalse(np.array_equal(history.epochs[-1].strength, start))


class FullLengthDeterminism(unittest.TestCase):
    def test_two_full_runs_are_byte_identical(self) -> None:
        started = time.perf_counter()
        first = run_history(FLOOR_WORLD)
        second = run_history(FLOOR_WORLD)
        elapsed = time.perf_counter() - started
        print(f"\n  full-length determinism: {elapsed:.1f}s "
              f"for two {FLOOR_WORLD.history_n}-cell runs of {first.steps} steps")
        self.assertEqual(len(first.epochs), 4)
        for a, b in zip(first.epochs, second.epochs):
            self.assertTrue(np.array_equal(a.strength, b.strength))
            self.assertTrue(np.array_equal(a.velocity, b.velocity))
            self.assertEqual(a.strength.tobytes(), b.strength.tobytes())
            self.assertEqual(a.velocity.tobytes(), b.velocity.tobytes())
        self.assertEqual(first.weak_fraction, second.weak_fraction)
        self.assertEqual(first.solver_cycles, second.solver_cycles)


if __name__ == "__main__":
    unittest.main()
