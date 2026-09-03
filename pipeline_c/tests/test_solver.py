"""Gates on the velocity solve.

The operator must not prefer an axis, must respect a weak barrier, and must
converge inside the declared cycle budget on a coefficient contrast as sharp
as the history can produce.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

from engine.domain import ddx, ddy  # noqa: E402
from engine.geometry import WorldGeometry  # noqa: E402
from engine.noise import periodic_noise  # noqa: E402
from engine.sampler import StageSampler  # noqa: E402
from engine.history.constants import (  # noqa: E402
    MG_MAX_CYCLES,
    MG_TOL,
    STAGE_ID,
    STAGE_VERSION,
    STRENGTH_MIN,
)
from engine.history.solver import (  # noqa: E402
    apply_A,
    build_levels,
    diagonal,
    edge_coefficients,
    effective_gradients,
    kappa0_for,
    prolong,
    restrict,
    restrict_kappa,
    solve,
)


def noise_pair(n: int, channel: int = 0) -> np.ndarray:
    """A periodic (2, n, n) right-hand side of the right size."""
    geometry = WorldGeometry(21, 2 * n if 2 * n >= 256 else 256, 5)
    sampler = StageSampler(geometry.world_id, STAGE_ID, STAGE_VERSION, "solver-test")
    a = periodic_noise(sampler, geometry, channel=channel,
                       nodes_coarsest=4, octaves=3)[:n, :n]
    b = periodic_noise(sampler, geometry, channel=channel + 1,
                       nodes_coarsest=4, octaves=3)[:n, :n]
    return np.stack((a, b))


def stiff_network_problem() -> tuple[np.ndarray, np.ndarray]:
    """The stiffest sheet the dials reach, cut by a weak network.

    `stiffness_fraction` at its ceiling of 2.0 on the 128 cell grid the
    velocity is solved on at the default resolution, so the homogenization
    length is twice the parent and the coefficient contrast is
    `STRENGTH_MIN ** -4`. The network is three rows, two columns, and one
    diagonal, each two cells wide and periodic: the diagonal is the part a
    coarse grid cannot keep, because it never lies along a coarse cell edge.
    """
    n = 128
    kappa0 = kappa0_for(n, 2.0)
    kappa = np.full((n, n), kappa0)
    weak = np.zeros((n, n), dtype=bool)
    for row in (20, 70, 110):
        weak[row:row + 2] = True
    for column in (30, 90):
        weak[:, column:column + 2] = True
    line = np.arange(n)
    weak[line, line] = True
    weak[line, (line + 1) % n] = True
    kappa[weak] = kappa0 * STRENGTH_MIN**4
    drive = noise_pair(n, channel=20)
    return drive / float(np.sqrt(np.mean(drive**2))), kappa


def barrier_problem() -> tuple[np.ndarray, np.ndarray]:
    n = 256
    kappa0 = kappa0_for(n)
    kappa = np.full((n, n), kappa0)
    kappa[100:102] = kappa0 * STRENGTH_MIN**4
    kappa[200:202] = kappa0 * STRENGTH_MIN**4
    drive = np.zeros((2, n, n))
    drive[0] = -1.0
    drive[0][102:200] = 1.0
    return drive, kappa


class Operator(unittest.TestCase):
    def test_edge_coefficients_are_the_harmonic_mean(self) -> None:
        kappa = np.asarray([[1.0, 4.0], [9.0, 16.0]])
        k_east, k_north = edge_coefficients(kappa)
        self.assertAlmostEqual(k_east[0, 0], 2 * 1 * 4 / 5)
        self.assertAlmostEqual(k_north[0, 0], 2 * 1 * 9 / 10)

    def test_diagonal_is_one_plus_the_edges(self) -> None:
        kappa = np.abs(noise_pair(32)[0]) + 0.5
        k_east, k_north = edge_coefficients(kappa)
        expected = (1.0 + k_east + np.roll(k_east, 1, axis=1)
                    + k_north + np.roll(k_north, 1, axis=0))
        self.assertTrue(np.allclose(diagonal(kappa), expected))

    def test_operator_is_symmetric(self) -> None:
        n = 32
        kappa = np.abs(noise_pair(n)[0]) + 0.5
        a = noise_pair(n, channel=4)
        b = noise_pair(n, channel=6)
        self.assertAlmostEqual(float(np.sum(apply_A(a, kappa) * b)),
                               float(np.sum(a * apply_A(b, kappa))), places=9)

    def test_transfers_are_an_adjoint_pair(self) -> None:
        # `restrict` is `prolong` transposed, up to the factor four. That is
        # what makes one V-cycle a usable conjugate gradient preconditioner.
        fine = noise_pair(32, channel=8)
        coarse = restrict(noise_pair(32, channel=10))
        left = float(np.sum(restrict(fine) * coarse))
        right = float(np.sum(fine * prolong(coarse))) / 4.0
        self.assertAlmostEqual(left, right, places=9)

    def test_kappa_restriction_is_harmonic_and_quartered(self) -> None:
        kappa = np.asarray([[1.0, 2.0], [4.0, 8.0]])
        expected = 1.0 / (1.0 + 0.5 + 0.25 + 0.125)
        self.assertAlmostEqual(float(restrict_kappa(kappa)[0, 0]), expected)

    def test_a_thin_weak_line_survives_coarsening(self) -> None:
        kappa = np.full((64, 64), 1000.0)
        kappa[30:32] = 1e-3
        coarse = restrict_kappa(kappa)
        self.assertLess(coarse[15, 0], 1e-3)
        self.assertGreater(coarse[0, 0], 100.0)


class EffectiveGradients(unittest.TestCase):
    """The gradient a cell carries, from the edge fluxes rather than from `u`.

    Across a stiffness contrast the flux is continuous and the gradient is
    not. A central difference on `u` gives the strong cell beside a weak one a
    share of the velocity jump; dividing the cell's own mean edge flux by its
    own kappa does not.
    """

    def test_uniform_kappa_is_the_central_difference(self) -> None:
        n = 64
        u = noise_pair(n, channel=14)
        for value in (1.0, 1000.0, kappa0_for(n)):
            with self.subTest(kappa=value):
                kappa = np.full((n, n), value)
                g_x, g_y = effective_gradients(u, kappa)
                self.assertLess(float(np.abs(g_x - ddx(u)).max()), 1e-12)
                self.assertLess(float(np.abs(g_y - ddy(u)).max()), 1e-12)

    def test_a_weak_row_carries_the_jump_and_its_neighbours_do_not(self) -> None:
        n = 64
        kappa = np.full((n, n), 1000.0)
        kappa[32] = 1000.0 * STRENGTH_MIN**4
        u = np.zeros((2, n, n))
        u[0][33:] = 1.0
        u[0][:32] = -1.0
        _, g_y = effective_gradients(u, kappa)

        row = g_y[0]
        self.assertGreater(abs(float(row[32].min())), 1.0)
        self.assertLess(float(np.abs(row[31]).max()), 1e-3)
        self.assertLess(float(np.abs(row[33]).max()), 1e-3)

        # Rows 0 and 63 carry the second jump a step function on a torus must
        # have: it is on strong ground and reads as a full central difference.
        # "Elsewhere" is every other row.
        elsewhere = np.concatenate((row[1:31], row[34:63]))
        self.assertEqual(float(np.abs(elsewhere).max()), 0.0)
        self.assertAlmostEqual(float(row[0].min()), -1.0, places=12)
        self.assertAlmostEqual(float(row[63].min()), -1.0, places=12)

    def test_transposing_the_problem_transposes_the_gradients(self) -> None:
        n = 64
        kappa = kappa0_for(n) * (0.2 + 0.8 * np.abs(noise_pair(n)[0]))
        u = noise_pair(n, channel=16)
        g_x, g_y = effective_gradients(u, kappa)

        kappa_t = kappa.T.copy()
        u_t = np.stack((u[1].T, u[0].T))
        g_x_t, g_y_t = effective_gradients(u_t, kappa_t)

        self.assertLess(
            float(np.abs(g_x_t - np.stack((g_y[1].T, g_y[0].T))).max()), 1e-12)
        self.assertLess(
            float(np.abs(g_y_t - np.stack((g_x[1].T, g_x[0].T))).max()), 1e-12)


class ConstantCoefficient(unittest.TestCase):
    def test_matches_the_fourier_solution(self) -> None:
        n = 128
        drive = noise_pair(n)
        kappa = np.full((n, n), 100.0)
        u, cycles, residual = solve(drive, kappa)
        self.assertLess(cycles, MG_MAX_CYCLES)
        self.assertLess(residual, MG_TOL)

        wave = np.fft.fftfreq(n) * 2.0 * np.pi
        kx, ky = np.meshgrid(wave, wave, indexing="xy")
        symbol = 1.0 + 100.0 * (4.0 - 2.0 * np.cos(kx) - 2.0 * np.cos(ky))
        exact = np.stack([
            np.real(np.fft.ifft2(np.fft.fft2(drive[c]) / symbol)) for c in (0, 1)])
        error = np.linalg.norm(u - exact) / np.linalg.norm(exact)
        self.assertLess(error, 1e-4)


class Barrier(unittest.TestCase):
    def test_a_weak_line_lets_velocity_jump(self) -> None:
        drive, kappa = barrier_problem()
        u, cycles, residual = solve(drive, kappa)
        self.assertLessEqual(cycles, MG_MAX_CYCLES)
        self.assertLess(residual, MG_TOL)
        self.assertAlmostEqual(float(u[0][120:181].mean()), 1.0, delta=0.05)
        self.assertAlmostEqual(float(u[0][220:256].mean()), -1.0, delta=0.05)
        self.assertAlmostEqual(float(u[0][0:81].mean()), -1.0, delta=0.05)

    def test_warm_start_costs_at_most_one_cycle(self) -> None:
        drive, kappa = barrier_problem()
        u, _, _ = solve(drive, kappa)
        _, cycles, residual = solve(drive, kappa, u0=u)
        self.assertLessEqual(cycles, 1)
        self.assertLess(residual, MG_TOL)


class StiffNetwork(unittest.TestCase):
    """The regime the exploration dials reach at the top of the stiffness range.

    A budget of forty cycles is what the lab runs with, so this is the gate the
    solver has to clear before any sweep above the default stiffness reports a
    velocity field rather than an unfinished one.
    """

    CYCLE_BUDGET = 40

    def test_converges_inside_the_cycle_budget(self) -> None:
        drive, kappa = stiff_network_problem()
        _, cycles, residual = solve(drive, kappa,
                                    max_cycles=self.CYCLE_BUDGET)
        self.assertLessEqual(cycles, self.CYCLE_BUDGET)
        self.assertLess(residual, MG_TOL)

    def test_warm_start_costs_at_most_two_cycles(self) -> None:
        drive, kappa = stiff_network_problem()
        u, _, _ = solve(drive, kappa, max_cycles=self.CYCLE_BUDGET)
        _, cycles, residual = solve(drive, kappa, u0=u,
                                    max_cycles=self.CYCLE_BUDGET)
        self.assertLessEqual(cycles, 2)
        self.assertLess(residual, MG_TOL)


class LatticeSymmetry(unittest.TestCase):
    def setUp(self) -> None:
        n = 128
        self.kappa = kappa0_for(n) * (0.2 + 0.8 * np.abs(noise_pair(n)[0]))
        self.drive = noise_pair(n, channel=12)
        self.solution = solve(self.drive, self.kappa)[0]

    def test_the_operator_does_not_prefer_an_axis(self) -> None:
        kappa_t = self.kappa.T.copy()
        drive_t = np.stack((self.drive[1].T, self.drive[0].T))
        u_t = solve(drive_t, kappa_t)[0]
        expected = np.stack((self.solution[1].T, self.solution[0].T))
        self.assertLess(float(np.abs(u_t - expected).max()), 1e-9)

    def test_a_half_turn_maps_to_a_half_turn(self) -> None:
        kappa_r = self.kappa[::-1, ::-1].copy()
        drive_r = -self.drive[:, ::-1, ::-1].copy()
        u_r = solve(drive_r, kappa_r)[0]
        expected = -self.solution[:, ::-1, ::-1]
        self.assertLess(float(np.abs(u_r - expected).max()), 1e-9)


class Levels(unittest.TestCase):
    def test_hierarchy_reaches_the_coarsest_grid(self) -> None:
        kappa = np.full((256, 256), 4.0)
        sizes = [level.n for level in build_levels(kappa)]
        self.assertEqual(sizes, [256, 128, 64, 32, 16, 8])

    def test_a_non_power_of_two_grid_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            build_levels(np.ones((96, 96)))

    def test_a_zero_drive_solves_to_zero(self) -> None:
        kappa = np.full((64, 64), 10.0)
        u, cycles, residual = solve(np.zeros((2, 64, 64)), kappa)
        self.assertEqual(cycles, 0)
        self.assertEqual(residual, 0.0)
        self.assertFalse(u.any())


if __name__ == "__main__":
    unittest.main()
