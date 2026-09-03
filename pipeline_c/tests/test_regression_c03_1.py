"""Gates that separate "faster" from "different".

C03.1 changed the discretization, the solve grid, the solver tolerance, the
step length, and the labelling algorithm. None of those is a mechanism, so the
fields they produce must be the ones C03 produced. These tests measure how far
each change moved the answer.

The fourth check of `WORK_ORDER_C03_1.md` §5 asserted the pre-fix outcome — no
localization, one plate per world — and was deleted by `WORK_ORDER_C03_2.md`
§3 because it pinned an error rather than a behaviour. The third check was
rewritten by `WORK_ORDER_C03_3.md` §3 for the excess-squared damage law, and
again by `WORK_ORDER_C03_5.md` §1.2, which replaced the constant yield
fraction with a percentile of the first step's strain.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

from engine.geometry import WorldGeometry  # noqa: E402
from engine.noise import periodic_noise  # noqa: E402
from engine.sampler import StageSampler  # noqa: E402
from engine.history.constants import (  # noqa: E402
    DAMAGE_RATE,
    HEAL_RATE,
    MG_TOL,
    STAGE_ID,
    STAGE_VERSION,
    STRENGTH_EXPONENT,
    STRENGTH_MIN,
)
from engine.history.drive import build_drive  # noqa: E402
from engine.history.kinematics import (  # noqa: E402
    initial_strength,
    run_history,
    solve_n_for,
    to_kinematic_grid,
    to_solve_grid,
)
from engine.history.solver import kappa0_for, solve  # noqa: E402

SEED = 4287772760


def rms(field: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(field) ** 2)))


def rms_speed(velocity: np.ndarray) -> float:
    return float(np.sqrt(np.mean(velocity[0] ** 2 + velocity[1] ** 2)))


def first_step_problem(pixels: int = 1024) -> tuple[np.ndarray, np.ndarray, int]:
    """The drive and coefficients of step one, on the kinematic grid."""
    geometry = WorldGeometry(SEED, pixels, 5)
    strength = initial_strength(geometry)
    kappa = kappa0_for(geometry.history_n) * strength**STRENGTH_EXPONENT
    traction = build_drive(geometry).field(0.0)
    return traction, kappa, geometry.history_n


class HalfGridSolve(unittest.TestCase):
    def test_the_coarser_solve_gives_the_same_velocity(self) -> None:
        traction, kappa, n = first_step_problem()
        full = solve(traction, kappa)[0]

        kappa_s, traction_s = to_solve_grid(kappa, traction, solve_n_for(n))
        half = to_kinematic_grid(solve(traction_s, kappa_s)[0], n)

        difference = rms(full - half) / rms_speed(full)
        print(f"\n  half-grid solve: RMS difference {difference:.4%} of RMS speed")
        self.assertLess(difference, 0.05)


class SolverTolerance(unittest.TestCase):
    def test_a_tenth_of_a_percent_residual_is_the_same_velocity(self) -> None:
        traction, kappa, n = first_step_problem()
        kappa_s, traction_s = to_solve_grid(kappa, traction, solve_n_for(n))
        tight = solve(traction_s, kappa_s, tol=1e-5)[0]
        loose = solve(traction_s, kappa_s, tol=MG_TOL)[0]

        difference = rms(tight - loose) / rms_speed(tight)
        print(f"  tolerance {MG_TOL:g} against 1e-5: RMS difference "
              f"{difference:.4%} of RMS speed")
        self.assertLess(difference, 0.01)


class DamageIntegrator(unittest.TestCase):
    """The exact integrator against the explicit one it replaced.

    The strain field carries the magnitudes the C03 build measured over a full
    run: mean 0.017 and maximum 0.06 per Myr. Both fields come from the
    engine's own noise so the case is fixed rather than drawn. The rate is the
    excess-squared law of `WORK_ORDER_C03_3.md` §1.3, with the yield read off
    the same field at the engine's default percentile.
    """

    #: The order's §3 asked for the pre-C03.3 bound of 1e-3. The excess-squared
    #: law reaches a damage rate twenty times the old law's on the same field,
    #: so the two integrators separate by 4.3e-3 rather than 2.1e-4 and the old
    #: bound cannot hold. The bound records what the new law actually does.
    BOUND = 1e-2

    #: `0.4 * 2 * pi * 40 / (5120 / 2)`, the constant yield this world had
    #: before `WORK_ORDER_C03_5.md` §1.2 replaced the fraction with a
    #: percentile. This case is a regression check on the two integrators over
    #: one fixed field, so it keeps the operating point C03.1 compared them at
    #: rather than following the yield rule around.
    YIELD_STRAIN = 0.4 * (2.0 * math.pi * 40.0 / (5120.0 / 2.0))

    def fields(self) -> tuple[np.ndarray, np.ndarray, float]:
        geometry = WorldGeometry(SEED, 512, 5)
        sampler = StageSampler(geometry.world_id, STAGE_ID, STAGE_VERSION,
                               "regression-c03-1")
        raw_strength = periodic_noise(sampler, geometry, channel=0,
                                      nodes_coarsest=8, octaves=5)
        raw_strain = np.abs(periodic_noise(sampler, geometry, channel=1,
                                           nodes_coarsest=8, octaves=5))
        strength = np.clip(0.7 + 0.3 * raw_strength, STRENGTH_MIN, 1.0)
        strain = raw_strain * (0.017 / float(raw_strain.mean()))
        strain *= 0.06 / float(strain.max())
        strain += 0.017 - float(strain.mean())
        strain = np.clip(strain, 0.0, 0.06)
        return strength, strain, self.YIELD_STRAIN

    def test_one_exact_step_matches_one_euler_step(self) -> None:
        strength, strain, yield_strain = self.fields()
        self.assertLess(abs(float(strain.mean()) - 0.017), 5e-3)
        self.assertLess(abs(float(strain.max()) - 0.06), 5e-3)
        exceeding = float(np.mean(strain > yield_strain))
        self.assertGreater(exceeding, 0.0)

        dt = 2.0
        excess = np.maximum(strain / yield_strain - 1.0, 0.0)
        rate = DAMAGE_RATE * excess * excess
        total = HEAL_RATE + rate
        equilibrium = HEAL_RATE / total
        exact = np.clip(
            equilibrium + (strength - equilibrium) * np.exp(-total * dt),
            STRENGTH_MIN, 1.0)
        euler = np.clip(
            strength + dt * (HEAL_RATE * (1.0 - strength) - rate * strength),
            STRENGTH_MIN, 1.0)

        difference = float(np.abs(exact - euler).max())
        print(f"  damage integrator at dt = 2: maximum difference "
              f"{difference:.3e} in strength, over a field {exceeding:.2%} of "
              f"which exceeds the yield")
        self.assertLess(difference, self.BOUND)


class Determinism(unittest.TestCase):
    def test_two_full_runs_at_512_are_byte_identical(self) -> None:
        geometry = WorldGeometry(7, 512, 5)
        first = run_history(geometry)
        second = run_history(geometry)
        self.assertEqual(len(first.epochs), 4)
        for a, b in zip(first.epochs, second.epochs):
            with self.subTest(t_myr=a.t_myr):
                self.assertEqual(a.strength.tobytes(), b.strength.tobytes())
                self.assertEqual(a.velocity.tobytes(), b.velocity.tobytes())
        self.assertEqual(first.weak_fraction, second.weak_fraction)
        self.assertEqual(first.solver_cycles, second.solver_cycles)
        print("  determinism: two full runs of seed 7 at 512 px are "
              "byte-identical in strength and velocity at every epoch")


if __name__ == "__main__":
    unittest.main()
