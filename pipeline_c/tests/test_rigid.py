"""Gates on the rigid-body motion of `WORK_ORDER_C04_2.md` §1.

Pieces are rigid bodies. Each has three unknowns — two of velocity and one of
rotation — and the equations are the sheet's own, summed over the piece: basal
drag over its cells against the seam tractions on its boundary. No constant
enters that the sheet did not already have, so every number below is worked
out from the drive, the cell count and `kappa(S)`.

Every case here is a hand-built mask small enough to check by hand.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

from engine.history.constants import (  # noqa: E402
    STRENGTH_EXPONENT,
    STRENGTH_MIN,
    WEAK_THRESHOLD,
)
from engine.history.plates import label_plates  # noqa: E402
from engine.history.rigid import (  # noqa: E402
    assemble,
    mismatch,
    net_load,
    piece_frames,
    piece_state,
    rigid_motion,
    seam_frame,
    seam_slip_rate,
    seam_traction,
    velocity_field,
)
from engine.history.solver import (  # noqa: E402
    effective_gradients,
    kappa0_for,
    solve,
)

#: A stiffness at the scale the production grid uses, so the seam couplings
#: below are the ones a real run sees rather than a toy's.
KAPPA0 = kappa0_for(512, 0.3)


def intact(n: int) -> np.ndarray:
    return np.ones((n, n), dtype=np.float64)


def uniform_drive(n: int, dx: float, dy: float) -> np.ndarray:
    drive = np.empty((2, n, n), dtype=np.float64)
    drive[0] = dx
    drive[1] = dy
    return drive


def solved(strength: np.ndarray, drive: np.ndarray, *,
           exponent: int = STRENGTH_EXPONENT, kappa0: float = KAPPA0):
    labels = label_plates(strength)
    pieces = piece_frames(labels)
    frame = seam_frame(pieces, strength, kappa0, exponent)
    velocity, omega, force, torque = rigid_motion(pieces, frame, drive)
    return pieces, frame, velocity, omega, force, torque


class OnePiece(unittest.TestCase):
    """A body with nothing to push against moves at the drive."""

    def block(self, n: int = 16) -> np.ndarray:
        """One square piece with a moat of seam cells around it.

        The moat's cells all have the same piece on every side they touch, so
        they link no pair and transmit nothing; the block is on its own.
        """
        strength = np.full((n, n), STRENGTH_MIN, dtype=np.float64)
        strength[3:11, 3:11] = 1.0
        return strength

    def test_a_uniform_drive_translates_it_and_does_not_turn_it(self) -> None:
        drive = uniform_drive(16, 2.5, -1.25)
        pieces, _frame, velocity, omega, force, torque = solved(
            self.block(), drive)
        self.assertEqual(pieces.count, 1)
        self.assertFalse(bool(pieces.wrapping[0]))
        self.assertAlmostEqual(float(velocity[0][0]), 2.5, places=12)
        self.assertAlmostEqual(float(velocity[1][0]), -1.25, places=12)
        self.assertAlmostEqual(float(omega[0]), 0.0, places=12)
        self.assertLess(force, 1e-8)
        self.assertLess(torque, 1e-8)

    def test_a_pure_rotation_gives_the_field_s_own_angular_rate(self) -> None:
        """A disc under `D = omega0 x r` turns at `omega0` and does not drift.

        The drive sums to zero over a centred disc, so the force balance puts
        `v` at zero; the torque balance is `omega0 I - omega I = 0`, so the
        rate is the field's, whatever the shape's inertia.
        """
        n = 32
        rate = 0.0375
        strength = np.full((n, n), STRENGTH_MIN, dtype=np.float64)
        rows, columns = np.indices((n, n))
        centre = 15.5
        # Radius six: twelve cells across, well inside the half-world at
        # which the minimal image folds and `piece_frames` pins the rotation.
        disc = (rows - centre) ** 2 + (columns - centre) ** 2 <= 6.0 ** 2
        strength[disc] = 1.0
        drive = np.zeros((2, n, n), dtype=np.float64)
        drive[0] = -rate * (rows - centre)
        drive[1] = rate * (columns - centre)

        pieces, _frame, velocity, omega, force, torque = solved(strength, drive)
        self.assertEqual(pieces.count, 1)
        self.assertFalse(bool(pieces.folded[0]))
        self.assertAlmostEqual(float(pieces.centroid[0][0]), centre, places=9)
        self.assertAlmostEqual(float(pieces.centroid[1][0]), centre, places=9)
        self.assertAlmostEqual(float(omega[0]), rate, places=6)
        self.assertAlmostEqual(float(velocity[0][0]), 0.0, places=9)
        self.assertAlmostEqual(float(velocity[1][0]), 0.0, places=9)
        self.assertLess(force, 1e-8)
        self.assertLess(torque, 1e-8)


class TwoPiecesAcrossASeam(unittest.TestCase):
    """How much a seam couples depends only on `kappa(S)`, steeply.

    Two bands of seven columns each, separated by one seam column on each
    side of the torus, driven in opposite directions. Each band has 112 cells
    of drag and each seam column contributes 16 links, so the balance is
    `v = m D / (m + 2 L c)` with `m = 112`, `L = 32` and `c = kappa(S) / 2`.
    At `S = 0.05` and the production exponent that coupling is a rounding
    error against the drag; at `S = 0.49` it is `(0.49 / 0.05) ** 4`, some
    nine thousand times larger, and the two bands can barely move apart.
    """

    N = 16

    def bands(self, seam_strength: float) -> np.ndarray:
        strength = intact(self.N)
        strength[:, 7] = seam_strength
        strength[:, 15] = seam_strength
        return strength

    def opposed(self) -> np.ndarray:
        drive = np.zeros((2, self.N, self.N), dtype=np.float64)
        drive[0, :, 0:7] = 4.0
        drive[0, :, 8:15] = -4.0
        return drive

    def moved(self, seam_strength: float):
        strength = self.bands(seam_strength)
        pieces, _frame, velocity, omega, force, torque = solved(
            strength, self.opposed())
        self.assertEqual(pieces.count, 2)
        self.assertLess(force, 1e-6)
        self.assertLess(torque, 1e-6)
        return pieces, velocity, omega

    def test_an_open_seam_lets_them_move_almost_independently(self) -> None:
        pieces, velocity, _omega = self.moved(STRENGTH_MIN)
        # Label 0 is the larger, or the lower flat index on a tie; both bands
        # are 112 cells, so read the drive off each piece's own cells.
        for piece in range(2):
            own = float(np.mean(self.opposed()[0][pieces.labels == piece]))
            with self.subTest(piece=piece):
                self.assertAlmostEqual(
                    float(velocity[0][piece]) / own, 1.0, delta=0.05)
                self.assertAlmostEqual(float(velocity[1][piece]), 0.0,
                                       places=9)

    def test_a_nearly_intact_seam_carries_them_together(self) -> None:
        _pieces, velocity, _omega = self.moved(0.49)
        # Opposite drives, so "together" is the gap between them: it is under
        # one per cent of the drive that is pulling them apart.
        gap = float(np.hypot(velocity[0][0] - velocity[0][1],
                             velocity[1][0] - velocity[1][1]))
        self.assertLess(gap, 0.01 * 4.0)
        # And each is far below its own drive, which is the same statement.
        self.assertLess(abs(float(velocity[0][0])), 0.01 * 4.0)

    def test_the_seam_is_still_a_seam_at_the_threshold(self) -> None:
        self.assertLess(0.49, WEAK_THRESHOLD)
        self.assertEqual(int((label_plates(self.bands(0.49)) < 0).sum()),
                         2 * self.N)


class WrappingPieces(unittest.TestCase):
    def test_a_piece_that_wraps_has_no_rotation_and_is_counted(self) -> None:
        n = 16
        strength = intact(n)
        strength[8, :] = STRENGTH_MIN        # one seam row: still one piece
        pieces = piece_frames(label_plates(strength))
        self.assertEqual(pieces.count, 1)
        self.assertTrue(bool(pieces.wrapping[0]))

        drive = np.zeros((2, n, n), dtype=np.float64)
        rows, columns = np.indices((n, n))
        drive[0] = -0.05 * (rows - 7.5)
        drive[1] = 0.05 * (columns - 7.5)
        _pieces, _frame, _velocity, omega, force, _torque = solved(
            strength, drive)
        self.assertEqual(float(omega[0]), 0.0)
        self.assertLess(force, 1e-8)

    def test_a_band_that_wraps_one_axis_only_still_wraps(self) -> None:
        n = 16
        strength = intact(n)
        strength[6, :] = STRENGTH_MIN
        strength[12, :] = STRENGTH_MIN
        pieces = piece_frames(label_plates(strength))
        self.assertEqual(pieces.count, 2)
        self.assertTrue(bool(pieces.wrapping.all()))

    def test_a_piece_more_than_half_the_world_across_is_folded_and_pinned(
            self) -> None:
        """The minimal image folds at `n / 2`, and a folded frame is not one.

        A piece that reaches more than half the torus across an axis without
        covering it has cells whose offset from the reference comes back with
        the wrong sign, so its centroid and its inertia are not a planar
        body's. It is pinned exactly as a wrapping piece is, which makes
        every one of those quantities inert.
        """
        n = 16
        strength = np.full((n, n), STRENGTH_MIN, dtype=np.float64)
        strength[6:9, 0:12] = 1.0        # twelve columns of sixteen
        pieces = piece_frames(label_plates(strength))
        self.assertEqual(pieces.count, 1)
        self.assertFalse(bool(pieces.wrapping[0]))
        self.assertTrue(bool(pieces.folded[0]))
        self.assertTrue(bool(pieces.pinned[0]))
        drive = np.zeros((2, n, n), dtype=np.float64)
        rows, columns = np.indices((n, n))
        drive[0] = -0.05 * (rows - 7.5)
        drive[1] = 0.05 * (columns - 7.5)
        _pieces, _frame, _velocity, omega, force, _torque = solved(
            strength, drive)
        self.assertEqual(float(omega[0]), 0.0)
        self.assertLess(force, 1e-8)

    def test_a_single_cell_piece_has_no_rotation_either(self) -> None:
        n = 16
        strength = np.full((n, n), STRENGTH_MIN, dtype=np.float64)
        strength[5, 5] = 1.0
        pieces = piece_frames(label_plates(strength))
        self.assertEqual(pieces.count, 1)
        self.assertEqual(float(pieces.inertia[0]), 0.0)
        matrix, right = assemble(pieces, seam_frame(pieces, strength, KAPPA0,
                                                    STRENGTH_EXPONENT),
                                 uniform_drive(n, 1.0, 0.0))
        self.assertEqual(float(matrix[2, 2]), 1.0)
        self.assertEqual(float(right[2]), 0.0)


class TheMismatch(unittest.TestCase):
    """What the internal-stress solve is forced with, and what it carries."""

    def test_an_uncoupled_piece_carries_no_net_force_or_torque(self) -> None:
        n = 16
        strength = np.full((n, n), STRENGTH_MIN, dtype=np.float64)
        strength[3:11, 3:11] = 1.0
        rows, columns = np.indices((n, n))
        drive = np.zeros((2, n, n), dtype=np.float64)
        drive[0] = 3.0 - 0.2 * rows
        drive[1] = 0.1 * columns - 1.0

        state = piece_state(strength, drive, KAPPA0, STRENGTH_EXPONENT, 40.0)
        force, torque = net_load(state["labels"], state["mismatch"],
                                 state["pieces"])
        self.assertLess(float(np.abs(force).max()), 1e-8)
        self.assertLess(float(np.abs(torque).max()), 1e-8)

    def test_a_coupled_piece_carries_minus_the_traction_on_it(self) -> None:
        """The §1.2 balance itself: drag mismatch plus seam traction is zero.

        A piece a seam links to another does not have zero net mismatch — it
        has minus what that seam transmits, which is the same equation. Over
        the whole world the tractions cancel in pairs and the net force of
        the mismatch is zero, so it carries no rigid motion of the world.
        """
        n = 16
        strength = intact(n)
        strength[:, 7] = STRENGTH_MIN
        strength[:, 15] = STRENGTH_MIN
        drive = np.zeros((2, n, n), dtype=np.float64)
        drive[0, :, 0:7] = 4.0
        drive[0, :, 8:15] = -4.0

        state = piece_state(strength, drive, KAPPA0, STRENGTH_EXPONENT, 40.0)
        pieces = state["pieces"]
        force, torque = net_load(state["labels"], state["mismatch"], pieces)
        seam_force, seam_torque = seam_traction(
            pieces, state["frame"], state["piece_velocity"], state["omega"])
        self.assertGreater(float(np.abs(seam_force).max()), 0.0)
        self.assertLess(float(np.abs(force + seam_force).max()), 1e-8)
        self.assertLess(float(np.abs(torque + seam_torque).max()), 1e-8)
        self.assertLess(float(np.abs(force.sum(axis=1)).max()), 1e-8)

    def test_the_mismatch_is_zero_on_every_seam_cell(self) -> None:
        n = 16
        strength = intact(n)
        strength[:, 7] = STRENGTH_MIN
        drive = uniform_drive(n, 1.0, -2.0)
        state = piece_state(strength, drive, KAPPA0, STRENGTH_EXPONENT, 40.0)
        seam = state["labels"] < 0
        self.assertTrue(bool(np.all(state["mismatch"][:, seam] == 0.0)))


class TheAssembledSystem(unittest.TestCase):
    def mask(self) -> np.ndarray:
        """Three pieces: two blocks and an L, all inside one seam field."""
        strength = np.full((20, 20), STRENGTH_MIN, dtype=np.float64)
        strength[2:8, 2:8] = 1.0
        strength[2:8, 12:18] = 1.0
        strength[12:18, 2:8] = 1.0
        strength[12:15, 8:12] = 1.0        # the L's arm, joining nothing
        return strength

    def test_it_is_symmetric_and_it_solves(self) -> None:
        strength = self.mask()
        labels = label_plates(strength)
        pieces = piece_frames(labels)
        self.assertEqual(pieces.count, 3)
        frame = seam_frame(pieces, strength, KAPPA0, STRENGTH_EXPONENT)
        drive = np.zeros((2, 20, 20), dtype=np.float64)
        rows, columns = np.indices((20, 20))
        drive[0] = np.sin(rows / 3.0)
        drive[1] = np.cos(columns / 5.0)

        matrix, right = assemble(pieces, frame, drive)
        self.assertEqual(matrix.shape, (9, 9))
        self.assertTrue(np.array_equal(matrix, matrix.T),
                        f"asymmetry {np.abs(matrix - matrix.T).max():.3e}")
        velocity, omega, force, torque = rigid_motion(pieces, frame, drive)
        self.assertEqual(velocity.shape, (2, 3))
        self.assertEqual(omega.shape, (3,))
        self.assertLess(force, 1e-8)
        self.assertLess(torque, 1e-8)
        # The drag diagonal is the cell count, which is the one term that
        # cannot vanish: the system is never singular for a real piece.
        for piece in range(3):
            self.assertLessEqual(float(matrix[3 * piece, 3 * piece]),
                                 -float(pieces.cells[piece]))

    def test_two_pieces_that_touch_a_seam_cell_are_coupled(self) -> None:
        strength = intact(16)
        strength[:, 7] = STRENGTH_MIN
        strength[:, 15] = STRENGTH_MIN
        pieces = piece_frames(label_plates(strength))
        frame = seam_frame(pieces, strength, KAPPA0, STRENGTH_EXPONENT)
        # Every seam cell of a straight seam sees one piece on each side and
        # links exactly one pair.
        self.assertEqual(frame.size, 32)
        self.assertTrue(bool(np.all(frame.first.sum(axis=0) == 2)))
        self.assertEqual(int(frame.pair.sum()), 32)
        matrix, _right = assemble(pieces, frame, uniform_drive(16, 1.0, 0.0))
        self.assertGreater(float(matrix[0, 3]), 0.0)
        self.assertTrue(np.array_equal(matrix, matrix.T))

    def test_a_seam_cell_with_one_piece_around_it_transmits_nothing(
            self) -> None:
        strength = np.full((16, 16), STRENGTH_MIN, dtype=np.float64)
        strength[3:11, 3:11] = 1.0
        pieces = piece_frames(label_plates(strength))
        frame = seam_frame(pieces, strength, KAPPA0, STRENGTH_EXPONENT)
        self.assertEqual(int(frame.pair.sum()), 0)


class TheVelocityField(unittest.TestCase):
    def test_a_seam_cell_takes_the_mean_of_the_pieces_it_touches(self) -> None:
        n = 16
        strength = intact(n)
        strength[:, 7] = STRENGTH_MIN
        strength[:, 15] = STRENGTH_MIN
        labels = label_plates(strength)
        pieces = piece_frames(labels)
        frame = seam_frame(pieces, strength, KAPPA0, STRENGTH_EXPONENT)
        velocity = np.array([[1.0, -3.0], [0.0, 0.0]])
        omega = np.zeros(2)
        field = velocity_field(pieces, frame, velocity, omega, None, None)
        self.assertAlmostEqual(float(field[0][4, 7]), -1.0, places=12)
        # And an intact cell has its own piece's velocity exactly.
        for piece in range(2):
            here = np.argwhere(labels == piece)[0]
            self.assertAlmostEqual(float(field[0][here[0], here[1]]),
                                   float(velocity[0][piece]), places=12)

    def test_a_seam_cell_with_no_piece_takes_the_previous_step_s_seams(
            self) -> None:
        n = 8
        strength = np.full((n, n), STRENGTH_MIN, dtype=np.float64)
        strength[0, 0] = 1.0
        labels = label_plates(strength)
        pieces = piece_frames(labels)
        frame = seam_frame(pieces, strength, KAPPA0, STRENGTH_EXPONENT)
        previous_u = np.zeros((2, n, n), dtype=np.float64)
        previous_u[0] = 5.0
        previous_seam = np.zeros((n, n), dtype=bool)
        previous_seam[4, 4] = True
        field = velocity_field(pieces, frame, np.zeros((2, 1)), np.zeros(1),
                               previous_u, previous_seam)
        # (3, 3) has one seam neighbour in the previous step, at (4, 4).
        self.assertAlmostEqual(float(field[0][3, 3]), 5.0, places=12)
        # (0, 4) has none, so it takes zero.
        self.assertEqual(float(field[0][0, 4]), 0.0)

    def test_the_slip_rate_is_the_jump_across_the_seam(self) -> None:
        n = 16
        strength = intact(n)
        strength[:, 7] = STRENGTH_MIN
        strength[:, 15] = STRENGTH_MIN
        drive = np.zeros((2, n, n), dtype=np.float64)
        drive[0, :, 0:7] = 4.0
        drive[0, :, 8:15] = -4.0
        state = piece_state(strength, drive, KAPPA0, STRENGTH_EXPONENT, 40.0)
        slip = state["slip_rate"]
        jump = abs(float(state["piece_velocity"][0][0]
                         - state["piece_velocity"][0][1]))
        self.assertAlmostEqual(float(slip[3, 7]), jump / 40.0, places=12)
        # Intact cells are rigid: no strain rate and therefore no damage.
        self.assertEqual(float(slip[state["labels"] >= 0].max()), 0.0)


class WholeGridCases(unittest.TestCase):
    def test_an_intact_sheet_is_one_wrapping_piece_at_the_drive_s_mean(
            self) -> None:
        n = 16
        drive = np.zeros((2, n, n), dtype=np.float64)
        rows, columns = np.indices((n, n))
        drive[0] = np.sin(2.0 * np.pi * columns / n)
        drive[1] = np.cos(2.0 * np.pi * rows / n)
        state = piece_state(intact(n), drive, KAPPA0, STRENGTH_EXPONENT, 40.0)
        self.assertEqual(state["piece_count"], 1)
        self.assertEqual(state["wrapping_pieces"], 1)
        self.assertEqual(state["largest_piece_share"], 1.0)
        # A drive with zero mean leaves the one piece standing still, which
        # is why step 1's mismatch is the drive and its solve is the sheet's.
        self.assertLess(float(np.abs(state["velocity"]).max()), 1e-12)
        self.assertLess(float(np.abs(state["mismatch"] - drive).max()), 1e-12)

    def test_a_world_with_no_intact_cell_stands_still(self) -> None:
        n = 8
        state = piece_state(np.full((n, n), STRENGTH_MIN), uniform_drive(n, 1.0, 1.0),
                            KAPPA0, STRENGTH_EXPONENT, 40.0)
        self.assertEqual(state["piece_count"], 0)
        self.assertEqual(float(np.abs(state["velocity"]).max()), 0.0)
        self.assertEqual(float(np.abs(state["mismatch"]).max()), 0.0)


class AnInternalCrack(unittest.TestCase):
    """A crack that has not cut a piece still slips: `WORK_ORDER_C04_3.md` §1.

    The rigid jump across a crack inside one piece is exactly zero — the same
    piece is on both sides of it, so it links no pair — and under C04.2 that
    left it with no damage and nothing opposing its healing. The elastic part
    is not zero: the internal-stress solve is forced by the mismatch, so its
    solution is the non-rigid part of the velocity, and its strain-rate
    invariant at a crack cell is the rate that cell's faces displace relative
    to each other under the load the piece carries.

    Every field below is built with the same functions `run_history` uses, at
    `solve_divisor` 1 so the solve grid is the kinematic grid and the block
    lift is the identity.
    """

    N = 32
    CELL_KM = 40.0
    #: The percentile the yield is read at, the C04 §7.1 dial.
    YIELD_PERCENTILE = 12.0

    def kappa0(self) -> float:
        return kappa0_for(self.N, 0.3)

    def shear_drive(self) -> np.ndarray:
        """Opposed flow either side of the crack line, with zero mean.

        Zero mean so the one piece the world holds stands still and the
        mismatch is the whole drive, which is the situation §1 describes.
        """
        drive = np.zeros((2, self.N, self.N), dtype=np.float64)
        rows = np.arange(self.N, dtype=np.float64)[:, None]
        drive[0] = np.where(rows < self.N // 2, 4.0, -4.0)
        drive[0] -= drive[0].mean()
        return drive

    def fields(self, strength: np.ndarray, drive: np.ndarray):
        """`(state, strain_rate)`: the block model's step, up to the damage."""
        kappa0 = self.kappa0()
        state = piece_state(strength, drive, kappa0, STRENGTH_EXPONENT,
                            self.CELL_KM)
        kappa = kappa0 * strength ** STRENGTH_EXPONENT
        solved, _cycles, _residual = solve(state["mismatch"], kappa)
        g_x, g_y = effective_gradients(solved, kappa)
        exx = g_x[0] / self.CELL_KM
        eyy = g_y[1] / self.CELL_KM
        exy = 0.5 * (g_y[0] + g_x[1]) / self.CELL_KM
        return state, np.sqrt(exx * exx + eyy * eyy + 2.0 * exy * exy)

    def cracked(self) -> np.ndarray:
        """One piece with a straight internal crack of twenty cells."""
        strength = intact(self.N)
        strength[self.N // 2, 6:26] = STRENGTH_MIN
        return strength

    def test_a_crack_inside_a_piece_slips_above_the_yield(self) -> None:
        strength = self.cracked()
        state, rate = self.fields(strength, self.shear_drive())
        # One piece, and its rigid jump is zero everywhere: the crack links
        # no pair, which is exactly the C04.2 failure this order addresses.
        self.assertEqual(state["piece_count"], 1)
        self.assertEqual(float(state["slip_rate"].max()), 0.0)

        slip = seam_slip_rate(state["slip_rate"], rate, strength)
        crack = strength < WEAK_THRESHOLD
        self.assertEqual(int(crack.sum()), 20)
        # The yield read from the same field, at the same percentile the run
        # reads it at.
        yield_rate = float(np.percentile(rate, 100.0 - self.YIELD_PERCENTILE,
                                         method="linear"))
        self.assertGreater(yield_rate, 0.0)
        self.assertGreater(float(slip[crack].min()), yield_rate)
        # And nothing beside the crack slips at all: an intact cell is rigid.
        self.assertEqual(float(np.abs(slip[~crack]).max()), 0.0)

    def test_the_same_crack_set_intact_has_no_seam_slip_anywhere(self) -> None:
        strength = intact(self.N)
        state, rate = self.fields(strength, self.shear_drive())
        self.assertEqual(state["piece_count"], 1)
        slip = seam_slip_rate(state["slip_rate"], rate, strength)
        self.assertGreater(float(rate.max()), 0.0)
        self.assertEqual(float(np.abs(slip).max()), 0.0)

    def test_a_seam_between_two_pieces_slips_by_both_parts(self) -> None:
        """The rigid jump is kept, not replaced: the two parts add."""
        n = 16
        strength = intact(n)
        strength[:, 7] = STRENGTH_MIN
        strength[:, 15] = STRENGTH_MIN
        drive = np.zeros((2, n, n), dtype=np.float64)
        drive[0, :, 0:7] = 4.0
        drive[0, :, 8:15] = -4.0
        state = piece_state(strength, drive, KAPPA0, STRENGTH_EXPONENT, 40.0)
        rate = np.full((n, n), 0.25, dtype=np.float64)
        slip = seam_slip_rate(state["slip_rate"], rate, strength)
        seam = strength < WEAK_THRESHOLD
        self.assertGreater(float(state["slip_rate"][3, 7]), 0.0)
        self.assertAlmostEqual(float(slip[3, 7]),
                               float(state["slip_rate"][3, 7]) + 0.25,
                               places=12)
        self.assertEqual(float(np.abs(slip[~seam]).max()), 0.0)


class MinimalImage(unittest.TestCase):
    def test_offsets_are_the_short_way_round(self) -> None:
        from engine.history.rigid import minimal_image
        n = 16
        self.assertEqual(float(minimal_image(1, n)), 1.0)
        self.assertEqual(float(minimal_image(15, n)), -1.0)
        self.assertEqual(float(minimal_image(-15, n)), 1.0)
        self.assertEqual(float(minimal_image(8, n)), -8.0)

    def test_a_piece_across_the_wrap_has_a_sensible_centroid(self) -> None:
        n = 16
        strength = np.full((n, n), STRENGTH_MIN, dtype=np.float64)
        strength[7:9, 15] = 1.0
        strength[7:9, 0] = 1.0
        pieces = piece_frames(label_plates(strength))
        self.assertEqual(pieces.count, 1)
        # Four cells straddling the seam at column 15/0: the centroid sits
        # between them, at 15.5 or -0.5 depending on the reference cell.
        self.assertAlmostEqual(float(pieces.centroid[0][0]) % n, 15.5,
                               places=9)
        self.assertAlmostEqual(float(pieces.centroid[1][0]), 7.5, places=9)
        self.assertFalse(bool(pieces.wrapping[0]))


if __name__ == "__main__":
    unittest.main()
