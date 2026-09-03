"""Gates on the marker seams of `WORK_ORDER_C04_2.md` §4.

The one property that matters is that a marker cannot be written twice. C04's
nearest-cell advection of the strength raster duplicated seam cells wherever
the velocity jumped across a seam, which is everywhere the network is, and
C04.1 measured `edge_fraction` at 0.43 to 0.62 because of it. A marker moves;
it is not resampled; there is nothing to duplicate.

The rest is bookkeeping the loop depends on: the raster is the minimum over a
cell's markers, a healed marker leaves, and the gap a junction opens is closed
by the next tip pass.

Every marker set here carries no edges, so the raster draws points alone and
these gates read exactly what they read before `WORK_ORDER_C04_6.md`. What
that order changed — the segments between linked markers, and the removal
threshold `damage_and_heal` now uses — is gated in `test_curve.py`.
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
    SEAM_OPEN_STRENGTH,
    STRENGTH_MIN,
    SUTURE_STRENGTH,
    WEAK_THRESHOLD,
)
from engine.history.markers import (  # noqa: E402
    Markers,
    cells,
    create,
    damage_and_heal,
    empty,
    gap_cells,
    move,
    opened_cells,
    raster,
)
from engine.history.seams import seam_mask, tip_pass, tips  # noqa: E402

N = 64


def straight_seam(column: int = 10, n: int = N) -> Markers:
    """One marker per row, all in one column: a seam one cell wide."""
    rows = np.arange(n, dtype=np.float64)
    return Markers(x=np.full(n, float(column)), y=rows,
                   s=np.full(n, float(SEAM_OPEN_STRENGTH)))


class TheRaster(unittest.TestCase):
    def test_an_empty_marker_set_is_an_intact_sheet(self) -> None:
        field = raster(empty(), 8)
        self.assertTrue(bool(np.all(field == 1.0)))
        self.assertEqual(int(seam_mask(field).sum()), 0)

    def test_two_markers_in_one_cell_give_the_cell_the_lower_strength(
            self) -> None:
        markers = Markers(x=np.array([3.0, 3.2, 5.0]),
                          y=np.array([4.0, 3.8, 5.0]),
                          s=np.array([0.4, 0.11, 0.3]))
        field = raster(markers, 8)
        # The first two round to the same cell, (4, 3); the lower wins.
        self.assertAlmostEqual(float(field[4, 3]), 0.11, places=12)
        self.assertAlmostEqual(float(field[5, 5]), 0.3, places=12)
        self.assertEqual(int((field < 1.0).sum()), 2)

    def test_the_order_the_markers_arrive_in_does_not_change_the_raster(
            self) -> None:
        markers = Markers(x=np.array([3.0, 3.2, 3.4, 5.0]),
                          y=np.array([4.0, 3.8, 4.2, 5.0]),
                          s=np.array([0.4, 0.11, 0.22, 0.3]))
        reversed_markers = Markers(x=markers.x[::-1].copy(),
                                   y=markers.y[::-1].copy(),
                                   s=markers.s[::-1].copy())
        self.assertTrue(np.array_equal(raster(markers, 8),
                                       raster(reversed_markers, 8)))

    def test_a_marker_wraps_into_the_grid(self) -> None:
        markers = Markers(x=np.array([7.6]), y=np.array([0.0]),
                          s=np.array([0.2]))
        field = raster(markers, 8)
        self.assertAlmostEqual(float(field[0, 0]), 0.2, places=12)


class Motion(unittest.TestCase):
    def test_a_straight_seam_stays_one_cell_wide_and_moves_at_the_speed(
            self) -> None:
        """0.3 cells a step for 75 steps: 22.5 cells, and never two wide."""
        markers = straight_seam()
        velocity = np.zeros((2, N, N), dtype=np.float64)
        # 0.3 cells per step: the mover divides by `cell_km` and multiplies
        # by the step, so 0.3 cells is 0.3 * cell_km / step in km/Myr.
        velocity[0] = 0.3 * 40.0 / 4.0
        for step in range(75):
            markers = move(markers, velocity, 4.0, 40.0, N)
            field = raster(markers, N)
            seam = seam_mask(field)
            widths = seam.sum(axis=1)
            with self.subTest(step=step):
                self.assertTrue(bool(np.all(widths == 1)),
                                f"widths {sorted(set(widths.tolist()))}")
                self.assertEqual(markers.size, N)
        column = int(np.argmax(seam_mask(raster(markers, N))[0]))
        self.assertIn((column - 10) % N, (22, 23))
        # Every row moved together: the seam moved as a line.
        self.assertEqual(
            len({int(np.argmax(row)) for row in seam_mask(raster(markers, N))}),
            1)

    def test_a_velocity_that_jumps_across_the_seam_does_not_duplicate_it(
            self) -> None:
        """The case that broke the raster: a whole cell of jump per step.

        The seam cell's own velocity is what a marker moves at, so a jump
        across the seam moves the two sides apart and moves the seam once.
        Nothing resamples, so nothing is written twice: the marker count is
        constant and the seam is never two cells wide.
        """
        markers = straight_seam()
        velocity = np.zeros((2, N, N), dtype=np.float64)
        # A jump of one cell per step across column 10, and the seam itself
        # taking the mean of the two sides, which is what `velocity_field`
        # gives a seam cell between two pieces.
        velocity[0, :, :10] = -0.5 * 40.0 / 4.0
        velocity[0, :, 10] = 0.0
        velocity[0, :, 11:] = 0.5 * 40.0 / 4.0
        for step in range(75):
            markers = move(markers, velocity, 4.0, 40.0, N)
            seam = seam_mask(raster(markers, N))
            with self.subTest(step=step):
                self.assertEqual(markers.size, N)
                self.assertEqual(int(seam.sum()), N)
                self.assertTrue(bool(np.all(seam.sum(axis=1) == 1)))

    def test_two_markers_may_land_in_one_cell_and_the_count_still_holds(
            self) -> None:
        """Marker motion can thin the network; it can never thicken it."""
        markers = Markers(x=np.array([4.0, 5.0]), y=np.array([4.0, 4.0]),
                          s=np.full(2, float(SEAM_OPEN_STRENGTH)))
        velocity = np.zeros((2, 8, 8), dtype=np.float64)
        velocity[0, :, 5] = -1.0 * 40.0 / 4.0     # the right one steps left
        markers = move(markers, velocity, 4.0, 40.0, 8)
        self.assertEqual(markers.size, 2)
        self.assertEqual(int(seam_mask(raster(markers, 8)).sum()), 1)


class DamageAndHealing(unittest.TestCase):
    HEAL = 1.0 / 10.0
    DAMAGE = 1.0 / 1.5

    def test_a_marker_healed_past_the_threshold_leaves_the_cell_intact(
            self) -> None:
        """Two steps at no slip take it over `WEAK_THRESHOLD`.

        Since `WORK_ORDER_C04_6.md` §1.4 the marker itself stays until
        `SUTURE_STRENGTH`; what leaves at the weak threshold is the *cell*,
        which is intact again from the step the strength crosses it.
        """
        markers = Markers(x=np.array([3.0]), y=np.array([3.0]),
                          s=np.array([float(SEAM_OPEN_STRENGTH)]))
        excess = np.zeros((8, 8), dtype=np.float64)
        markers, removed, reactivated = damage_and_heal(
            markers, excess, self.HEAL, self.DAMAGE, 4.0, 8)
        self.assertEqual((markers.size, removed, reactivated), (1, 0, 0))
        self.assertLess(float(markers.s[0]), WEAK_THRESHOLD)
        self.assertTrue(bool(seam_mask(raster(markers, 8))[3, 3]))
        markers, removed, _reactivated = damage_and_heal(
            markers, excess, self.HEAL, self.DAMAGE, 4.0, 8)
        self.assertEqual((markers.size, removed), (1, 0))
        self.assertGreater(float(markers.s[0]), WEAK_THRESHOLD)
        self.assertLess(float(markers.s[0]), SUTURE_STRENGTH)
        self.assertFalse(bool(seam_mask(raster(markers, 8))[3, 3]))
        # And it leaves for good once it reaches the suture strength.
        for _ in range(10):
            markers, removed, _reactivated = damage_and_heal(
                markers, excess, self.HEAL, self.DAMAGE, 4.0, 8)
            if markers.size == 0:
                break
        self.assertEqual((markers.size, removed), (0, 1))
        self.assertTrue(bool(np.all(raster(markers, 8) == 1.0)))

    def test_a_slipping_marker_sits_at_the_floor(self) -> None:
        markers = Markers(x=np.array([3.0]), y=np.array([3.0]),
                          s=np.array([float(SEAM_OPEN_STRENGTH)]))
        excess = np.zeros((8, 8), dtype=np.float64)
        excess[3, 3] = 2.0                     # three times yield
        for _ in range(75):
            markers, removed, _reactivated = damage_and_heal(
                markers, excess, self.HEAL, self.DAMAGE, 4.0, 8)
            self.assertEqual(removed, 0)
        self.assertEqual(float(markers.s[0]), STRENGTH_MIN)

    def test_two_markers_in_one_cell_see_the_same_rate(self) -> None:
        markers = Markers(x=np.array([3.0, 3.2]), y=np.array([3.0, 2.8]),
                          s=np.array([0.2, 0.3]))
        excess = np.zeros((8, 8), dtype=np.float64)
        excess[3, 3] = 2.0
        moved, _removed, _reactivated = damage_and_heal(
            markers, excess, self.HEAL, self.DAMAGE, 4.0, 8)
        # Same cell, same rate: both fall to the same equilibrium floor.
        self.assertEqual(float(moved.s[0]), float(moved.s[1]))

    def test_a_marker_whose_cell_is_held_by_another_leaves_the_cell_weak(
            self) -> None:
        markers = Markers(x=np.array([3.0, 3.0]), y=np.array([3.0, 3.0]),
                          s=np.array([0.49, 0.05]))
        excess = np.zeros((8, 8), dtype=np.float64)
        excess[3, 3] = 2.0
        moved, removed, _reactivated = damage_and_heal(
            markers, excess, self.HEAL, self.DAMAGE, 4.0, 8)
        self.assertEqual((moved.size, removed), (2, 0))
        self.assertTrue(bool(seam_mask(raster(moved, 8))[3, 3]))


class Creation(unittest.TestCase):
    def test_a_cell_opened_twice_in_one_step_gains_one_marker(self) -> None:
        opened = np.zeros((8, 8), dtype=bool)
        opened[2, 2] = True
        opened[5, 5] = True
        markers = create(empty(), opened)
        self.assertEqual(markers.size, 2)
        self.assertEqual(sorted(zip(markers.y.tolist(), markers.x.tolist())),
                         [(2.0, 2.0), (5.0, 5.0)])
        self.assertTrue(bool(np.all(markers.s == SEAM_OPEN_STRENGTH)))

    def test_opened_cells_reads_a_pass_off_the_raster(self) -> None:
        before = np.ones((8, 8), dtype=np.float64)
        after = before.copy()
        after[3, 4] = SEAM_OPEN_STRENGTH
        opened = opened_cells(before, after)
        self.assertEqual(np.argwhere(opened).tolist(), [[3, 4]])


class Gaps(unittest.TestCase):
    """A junction opens a hole; the tip rule on either side closes it."""

    def loaded(self, n: int):
        """Uniform tension along x, under which a crack runs along y.

        `traction_magnitude` for the direction `(dy, dx)` is `|dy| / |d|`, so
        the two vertical directions carry 1 and the two horizontal ones 0: a
        crack under tension runs across the pull.
        """
        ones = np.ones((n, n), dtype=np.float64)
        return ones, np.zeros((n, n)), np.zeros((n, n))

    def test_a_gap_between_two_loaded_segments_is_closed_by_one_tip_pass(
            self) -> None:
        n = 16
        # Two collinear segments in column 5 with one intact cell between.
        markers = Markers(
            x=np.full(8, 5.0),
            y=np.array([2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0]),
            s=np.full(8, float(SEAM_OPEN_STRENGTH)))
        field = raster(markers, n)
        self.assertFalse(bool(seam_mask(field)[6, 5]))
        # Both cells beside the gap are tips.
        self.assertTrue(bool(tips(seam_mask(field))[5, 5]))
        self.assertTrue(bool(tips(seam_mask(field))[7, 5]))

        sxx, syy, sxy = self.loaded(n)
        moved, tip_count, advances = tip_pass(field, sxx, syy, sxy,
                                              np.full((n, n), 0.5))
        self.assertGreaterEqual(advances, 1)
        opened = opened_cells(field, moved)
        self.assertTrue(bool(opened[6, 5]), "the gap was not closed")
        markers = create(markers, opened)
        self.assertTrue(bool(seam_mask(raster(markers, n))[6, 5]))
        # Four tips: the two facing the gap and the two outer ends.
        self.assertEqual(tip_count, 4)

    def test_a_gap_is_a_hole_in_the_network_and_not_a_retreating_tip(
            self) -> None:
        n = 16
        previous = np.zeros((n, n), dtype=bool)
        previous[2:9, 5] = True
        current = previous.copy()
        current[6, 5] = False        # a hole
        current[8, 5] = False        # the end of the line
        found = gap_cells(previous, current)
        self.assertEqual(np.argwhere(found).tolist(), [[6, 5]])

    def test_no_gap_where_the_network_did_not_lose_a_cell(self) -> None:
        n = 8
        seam = np.zeros((n, n), dtype=bool)
        seam[2:6, 3] = True
        self.assertEqual(int(gap_cells(seam, seam).sum()), 0)


class Cells(unittest.TestCase):
    def test_a_marker_belongs_to_the_nearest_cell(self) -> None:
        markers = Markers(x=np.array([0.4, 0.6, 7.9, -0.4]),
                          y=np.zeros(4), s=np.zeros(4))
        _rows, columns = cells(markers, 8)
        self.assertEqual(columns.tolist(), [0, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
