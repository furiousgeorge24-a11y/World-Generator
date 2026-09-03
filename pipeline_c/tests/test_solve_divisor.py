"""Gates on the solve divisor of `WORK_ORDER_C03_9.md`.

`SOLVE_GRID_DIVISOR` used to be a module constant read straight out of
`constants.py` by `run_history`. It is now `HistoryParams.solve_divisor`, an
integer in {1, 2} whose default is that constant, so the production path is
unmoved and a run can be repeated on the full kinematic grid.

At divisor 2 the velocity is solved on half the grid, strain is lifted back in
2 x 2 blocks, and damage happens in units of one solve cell. At divisor 1 the
solve grid is the kinematic grid and every transfer is the identity. These
tests hold both, and hold that the stiffness length in kilometres is the same
number either way.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

from engine.geometry import WorldGeometry  # noqa: E402
from engine.history.constants import (  # noqa: E402
    MG_TOL,
    SOLVE_GRID_DIVISOR,
)
from engine.history.drive import build_drive  # noqa: E402
from engine.history.kinematics import (  # noqa: E402
    HistoryParams,
    initial_strength,
    run_history,
    solve_n_for,
    to_kinematic_blocks,
    to_kinematic_grid,
    to_solve_grid,
)
from engine.history.solver import kappa0_for, restrict_kappa  # noqa: E402

WORLD = WorldGeometry(7, 128, 5)


def kappa_field(params: HistoryParams = HistoryParams()) -> np.ndarray:
    """The stiffness the first step solves through, on the kinematic grid."""
    strength = initial_strength(WORLD, params.strength_spread)
    return (kappa0_for(WORLD.history_n, params.stiffness_fraction)
            * strength ** params.strength_exponent)


class TheDial(unittest.TestCase):
    def test_the_default_is_the_constant(self) -> None:
        self.assertEqual(HistoryParams().solve_divisor, SOLVE_GRID_DIVISOR)
        self.assertEqual(SOLVE_GRID_DIVISOR, 2)

    def test_it_is_recorded(self) -> None:
        self.assertEqual(HistoryParams().to_record()["solve_divisor"], 2)
        self.assertEqual(
            HistoryParams(solve_divisor=1).to_record()["solve_divisor"], 1)

    def test_three_zero_and_a_bool_are_refused(self) -> None:
        for value in (3, 0, -1, True, 1.0, 2.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                HistoryParams(solve_divisor=value)


class TheSolveGridSize(unittest.TestCase):
    def test_divisor_one_is_the_kinematic_grid(self) -> None:
        for n in (128, 256, 512):
            with self.subTest(n=n):
                self.assertEqual(solve_n_for(n, 1), n)

    def test_divisor_two_is_half_of_it(self) -> None:
        for n in (128, 256, 512):
            with self.subTest(n=n):
                self.assertEqual(solve_n_for(n, 2), n // 2)

    def test_the_default_argument_is_the_constant(self) -> None:
        self.assertEqual(solve_n_for(256), solve_n_for(256, SOLVE_GRID_DIVISOR))

    def test_an_odd_grid_is_refused_at_divisor_two(self) -> None:
        with self.assertRaises(ValueError):
            solve_n_for(127, 2)


class TheTransfersAtDivisorOne(unittest.TestCase):
    """No restriction, no interpolation, no lift: every transfer is exact."""

    def setUp(self) -> None:
        self.n = WORLD.history_n
        self.kappa = kappa_field()
        self.traction = build_drive(WORLD).field(0.0)

    def test_to_solve_grid_is_the_identity(self) -> None:
        kappa, traction = to_solve_grid(self.kappa, self.traction,
                                        solve_n_for(self.n, 1))
        self.assertIs(kappa, self.kappa)
        self.assertIs(traction, self.traction)

    def test_to_kinematic_grid_is_the_identity(self) -> None:
        velocity = np.arange(2 * self.n * self.n, dtype=np.float64).reshape(
            2, self.n, self.n)
        self.assertIs(to_kinematic_grid(velocity, self.n), velocity)

    def test_to_kinematic_blocks_is_the_identity(self) -> None:
        field = np.arange(self.n * self.n, dtype=np.float64).reshape(
            self.n, self.n)
        self.assertIs(to_kinematic_blocks(field, self.n), field)

    def test_the_block_factor_comes_from_the_shapes(self) -> None:
        # Not from `SOLVE_GRID_DIVISOR`: a field a quarter of the grid's size
        # is lifted by four, which the constant would get wrong.
        coarse = np.arange(16, dtype=np.float64).reshape(4, 4)
        lifted = to_kinematic_blocks(coarse, 16)
        self.assertEqual(lifted.shape, (16, 16))
        blocks = lifted.reshape(4, 4, 4, 4)
        self.assertEqual(float(np.abs(blocks - blocks[:, :1, :, :1]).max()),
                         0.0)


class TheStiffnessLength(unittest.TestCase):
    """`kappa0_for` is untouched, and the two grids carry the same length.

    Its length is in kinematic cells, and `restrict_kappa` carries a quarter
    per coarsening, which is exactly the factor a cell of twice the size needs.
    So `kappa` on the solve grid at divisor 2 is a quarter of the harmonic
    2 x 2 mean of `kappa` at divisor 1 on the same strength field, and in
    solve-cell units the homogenization length is the same number of
    kilometres.
    """

    def test_the_solve_grid_kappa_is_a_quarter_of_the_block_mean(self) -> None:
        kappa = kappa_field()
        n = kappa.shape[-1]
        blocks = kappa.reshape(n // 2, 2, n // 2, 2)
        harmonic = 4.0 / (1.0 / blocks).sum(axis=(1, 3))
        coarse, _traction = to_solve_grid(
            kappa, np.zeros((2, n, n)), solve_n_for(n, 2))
        self.assertTrue(np.allclose(coarse, harmonic / 4.0,
                                    rtol=1e-12, atol=0.0))
        self.assertTrue(np.allclose(restrict_kappa(kappa), harmonic / 4.0,
                                    rtol=1e-12, atol=0.0))

    def test_the_homogenization_length_in_cells_scales_with_the_grid(self) -> None:
        # kappa0 is (fraction * n)^2 in kinematic cells squared. A quarter of
        # it is (fraction * n / 2)^2, which is the same fraction of a grid of
        # half as many cells of twice the size: the same length in kilometres.
        fraction = 0.125
        n = 256
        fine = kappa0_for(n, fraction)
        coarse = kappa0_for(n // 2, fraction)
        self.assertAlmostEqual(coarse, fine / 4.0, places=9)


class ARunAtDivisorOne(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.params = HistoryParams(solve_divisor=1)
        cls.history = run_history(WORLD, params=cls.params, steps=8)

    def test_it_converges_on_every_step(self) -> None:
        self.assertLess(max(self.history.solver_residual), MG_TOL)

    def test_it_is_deterministic(self) -> None:
        again = run_history(WORLD, params=self.params, steps=8)
        for first, second in zip(self.history.epochs, again.epochs):
            self.assertEqual(first.strength.tobytes(),
                             second.strength.tobytes())
            self.assertEqual(first.velocity.tobytes(),
                             second.velocity.tobytes())
            self.assertEqual(first.strain_rate.tobytes(),
                             second.strain_rate.tobytes())

    def test_the_fields_are_on_the_kinematic_grid(self) -> None:
        n = WORLD.history_n
        for epoch in self.history.epochs:
            with self.subTest(t_myr=epoch.t_myr):
                self.assertEqual(epoch.strain_rate.shape, (n, n))
                self.assertEqual(epoch.velocity.shape, (2, n, n))

    def test_the_strain_field_is_not_block_constant(self) -> None:
        # At divisor 2 every even-aligned 2 x 2 block holds one value, which
        # `test_history` gates. At divisor 1 the strain is resolved per cell,
        # so the blocks are no longer flat.
        n = WORLD.history_n
        field = self.history.epochs[-1].strain_rate
        blocks = field.reshape(n // 2, 2, n // 2, 2)
        self.assertGreater(float(np.abs(blocks - blocks[:, :1, :, :1]).max()),
                           0.0)

    def test_it_is_a_different_history_from_divisor_two(self) -> None:
        two = run_history(WORLD, params=HistoryParams(solve_divisor=2),
                          steps=8)
        self.assertNotEqual(self.history.epochs[-1].strength.tobytes(),
                            two.epochs[-1].strength.tobytes())
        self.assertLess(max(two.solver_residual), MG_TOL)


if __name__ == "__main__":
    unittest.main()
