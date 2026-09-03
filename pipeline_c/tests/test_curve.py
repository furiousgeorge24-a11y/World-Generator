"""Gates on the seam curve of `WORK_ORDER_C04_6.md` §1.

C04.2 made a seam a set of markers so it could not be duplicated. C04.5 §2
then measured what a set of points costs: of the cells through which a cut
piece rejoined a larger one, 99.65 % were **vacated** rather than healed. Two
markers in one cell translate together, cross a cell boundary, round into two
different cells, and the cell they shared holds nothing — so it is intact
while the seam through it is still slipping at twenty times the yield.

A segment cannot be vacated. The markers of a crack are now linked in order,
the raster draws the segments between them as well as the markers themselves,
and wherever a segment's two ends go the cells between them are drawn. The
tests below are what that has to mean: a segment rasters to an 8-connected
path one cell wide, C04.5's own traced event no longer opens its cell, a
crack that reaches another links to it and stops, a vertex that leaves is
spliced over rather than breaking the curve, a stretched edge is subdivided,
and every operation that drops a marker reindexes the edge list.
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

from engine.history.constants import (  # noqa: E402
    MEETING_RADIUS_CELLS,
    SEAM_OPEN_STRENGTH,
    SEGMENT_MAX_CELLS,
    SUTURE_STRENGTH,
    WEAK_THRESHOLD,
)
from engine.history.markers import (  # noqa: E402
    Markers,
    add,
    advance_tips,
    canonical_edges,
    chain_labels,
    damage_and_heal,
    degrees,
    empty,
    raster,
    remove,
    segment_lengths,
    segments_per_cell,
    subdivide,
)
from engine.history.seams import seam_mask  # noqa: E402

N = 64


def edge_between(ax: float, ay: float, bx: float, by: float,
                 s: float = SEAM_OPEN_STRENGTH) -> Markers:
    """Two markers and the one edge between them."""
    return Markers(x=np.array([ax, bx]), y=np.array([ay, by]),
                   s=np.full(2, float(s)),
                   a=np.array([0], dtype=np.int64),
                   b=np.array([1], dtype=np.int64))


def drawn(markers: Markers, n: int = N) -> np.ndarray:
    return seam_mask(raster(markers, n))


def eight_connected(mask: np.ndarray) -> bool:
    """Every drawn cell reaches every other through 8-neighbour steps."""
    from engine.history.plates import label_components
    labels = label_components(mask, 8)
    return int(labels.max()) == 0


def one_cell_wide(mask: np.ndarray) -> bool:
    """No 2 x 2 block of the torus is drawn in full."""
    block = (mask & np.roll(mask, -1, axis=0) & np.roll(mask, -1, axis=1)
             & np.roll(np.roll(mask, -1, axis=0), -1, axis=1))
    return not bool(block.any())


class TheRaster(unittest.TestCase):
    def test_an_empty_marker_set_has_no_edges_and_is_an_intact_sheet(
            self) -> None:
        markers = empty()
        self.assertEqual((markers.size, markers.edge_count), (0, 0))
        self.assertTrue(bool(np.all(raster(markers, 8) == 1.0)))

    def test_a_segment_rasters_to_an_eight_connected_path_one_cell_wide(
            self) -> None:
        markers = edge_between(10.0, 10.0, 27.3, 20.0)
        mask = drawn(markers)
        self.assertTrue(bool(mask[10, 10]), "the first end cell is not drawn")
        self.assertTrue(bool(mask[20, 27]), "the last end cell is not drawn")
        self.assertTrue(eight_connected(mask), "the path is broken")
        self.assertTrue(one_cell_wide(mask), "the path is two cells wide")
        # It is a line and not a smear: a staircase between the two ends
        # cannot draw more cells than the two spans plus one.
        self.assertLessEqual(int(mask.sum()), 18 + 11 + 1)

    def test_the_same_segment_across_the_wrap_does_too(self) -> None:
        # The same displacement, started so that both coordinates wrap.
        markers = edge_between(58.0, 58.0, (58.0 + 17.3) % N,
                               (58.0 + 10.0) % N)
        mask = drawn(markers)
        self.assertTrue(bool(mask[58, 58]))
        self.assertTrue(bool(mask[4, 11]))
        self.assertTrue(eight_connected(mask))
        self.assertTrue(one_cell_wide(mask))
        self.assertLessEqual(int(mask.sum()), 18 + 11 + 1)

    def test_a_cell_takes_the_lowest_strength_drawn_into_it(self) -> None:
        markers = Markers(x=np.array([3.0, 3.2, 5.0]),
                          y=np.array([4.0, 3.8, 5.0]),
                          s=np.array([0.4, 0.11, 0.3]))
        field = raster(markers, 8)
        self.assertAlmostEqual(float(field[4, 3]), 0.11, places=12)
        self.assertAlmostEqual(float(field[5, 5]), 0.3, places=12)

    def test_the_strength_along_a_segment_is_interpolated(self) -> None:
        """One end open, the other sutured: the middle is between them.

        The strength rises linearly from 0.1 to 0.9 over four cells at a
        sample every half cell, and a cell takes the **minimum** over the
        samples that land in it. `np.rint` sends a half to the even integer,
        so cell 12 receives the samples at 11.5, 12.0 and 12.5 and reads the
        lowest of them, 0.4, while cell 13 receives only the sample at 13.0
        and reads 0.7.
        """
        markers = edge_between(10.0, 10.0, 14.0, 10.0)
        markers = Markers(x=markers.x, y=markers.y,
                          s=np.array([0.1, 0.9]), a=markers.a, b=markers.b)
        field = raster(markers, N)
        self.assertAlmostEqual(float(field[10, 10]), 0.1, places=12)
        self.assertAlmostEqual(float(field[10, 11]), 0.3, places=12)
        self.assertAlmostEqual(float(field[10, 12]), 0.4, places=12)
        self.assertAlmostEqual(float(field[10, 13]), 0.7, places=12)
        self.assertAlmostEqual(float(field[10, 14]), 0.8, places=12)
        # A cell a segment covers at or above the weak threshold is intact.
        self.assertFalse(bool(seam_mask(field)[10, 13]))
        self.assertTrue(bool(seam_mask(field)[10, 11]))


class TheEventC04FiveTraced(unittest.TestCase):
    """`C04_5_BUILD_REPORT.md` §6, as a unit test.

    Two markers 0.88 of a cell apart inside cell (99, 99), both carried by the
    same velocity of `(-0.269, +0.509)` cells over one step, round into cells
    (99, 98) and (100, 99). Under the C04.5 point raster the cell they shared
    held nothing and a piece of 892 cells rejoined through it. Linked, the
    segment between them still draws it.
    """

    NN = 128
    A = (98.5012, 98.6648)
    B = (99.3770, 99.1475)
    STEP = (-0.269, 0.509)

    def test_the_two_markers_start_in_the_one_cell(self) -> None:
        markers = edge_between(*self.A, *self.B)
        self.assertTrue(bool(drawn(markers, self.NN)[99, 99]))

    def test_the_cell_is_still_drawn_after_the_move(self) -> None:
        markers = edge_between(self.A[0] + self.STEP[0],
                               self.A[1] + self.STEP[1],
                               self.B[0] + self.STEP[0],
                               self.B[1] + self.STEP[1])
        # The two ends have parted into two cells, as C04.5 measured.
        self.assertEqual(
            (int(round(markers.y[0])), int(round(markers.x[0]))), (99, 98))
        self.assertEqual(
            (int(round(markers.y[1])), int(round(markers.x[1]))), (100, 99))
        # And the cell between them is drawn by the segment.
        self.assertTrue(bool(drawn(markers, self.NN)[99, 99]),
                        "the segment vacated the cell it spans")

    def test_the_c04_5_point_raster_is_what_left_it(self) -> None:
        """The same two markers with no edge: the cell goes intact."""
        loose = Markers(x=np.array([self.A[0] + self.STEP[0],
                                    self.B[0] + self.STEP[0]]),
                        y=np.array([self.A[1] + self.STEP[1],
                                    self.B[1] + self.STEP[1]]),
                        s=np.full(2, float(SEAM_OPEN_STRENGTH)))
        self.assertFalse(bool(drawn(loose, self.NN)[99, 99]))


class Meeting(unittest.TestCase):
    """A crack that reaches another links to it and stops."""

    def loaded_along_x(self, n: int = N):
        """Uniform tension across y, under which a crack runs along x."""
        ones = np.ones((n, n), dtype=np.float64)
        zero = np.zeros((n, n), dtype=np.float64)
        return zero, ones, zero

    def test_an_advance_into_another_chain_gains_a_second_edge(self) -> None:
        # One chain of two markers running along x, and a lone marker two
        # cells ahead of its tip, which is a chain of its own.
        markers = Markers(x=np.array([30.0, 31.0, 33.0]),
                          y=np.full(3, 32.0),
                          s=np.full(3, float(SEAM_OPEN_STRENGTH)),
                          a=np.array([0], dtype=np.int64),
                          b=np.array([1], dtype=np.int64))
        sxx, syy, sxy = self.loaded_along_x()
        sigma_c = np.full((N, N), 0.5)
        grown, tip_count, advances, meetings, _degenerate = advance_tips(
            markers, raster(markers, N), sxx, syy, sxy, sigma_c)
        self.assertEqual(meetings, 1)
        # One of the markers the pass made carries two edges: one back to its
        # own tip and one to the chain it met.
        degree = degrees(grown)
        made = [index for index in range(markers.size, grown.size)
                if int(degree[index]) == 2]
        self.assertEqual(len(made), 1)
        # It is not a tip, so this crack stops here.
        self.assertNotIn(made[0], np.nonzero(degree <= 1)[0].tolist())
        # It is the advance from the tip that ran towards the lone marker.
        self.assertAlmostEqual(float(grown.x[made[0]]), 32.0, places=12)
        # And the two chains are now one.
        labels = chain_labels(grown)
        self.assertEqual(int(labels[0]), int(labels[2]))
        self.assertEqual((tip_count, advances), (3, 3))

    def test_a_marker_of_the_same_chain_is_not_a_meeting(self) -> None:
        markers = Markers(x=np.array([30.0, 31.0, 33.0]),
                          y=np.full(3, 32.0),
                          s=np.full(3, float(SEAM_OPEN_STRENGTH)),
                          a=np.array([0, 1], dtype=np.int64),
                          b=np.array([1, 2], dtype=np.int64))
        sxx, syy, sxy = self.loaded_along_x()
        _grown, _tips, _advances, meetings, _degenerate = advance_tips(
            markers, raster(markers, N), sxx, syy, sxy,
            np.full((N, N), 0.5))
        self.assertEqual(meetings, 0)

    def test_the_radius_is_the_constant_the_order_names(self) -> None:
        self.assertEqual(MEETING_RADIUS_CELLS, 1.5)
        self.assertEqual(SEGMENT_MAX_CELLS, 1.5)
        self.assertEqual(SUTURE_STRENGTH, 0.9)


class Removal(unittest.TestCase):
    def chain(self, count: int, n: int = N) -> Markers:
        x = 10.0 + np.arange(count, dtype=np.float64)
        return Markers(x=x, y=np.full(count, 20.0),
                       s=np.full(count, float(SEAM_OPEN_STRENGTH)),
                       a=np.arange(count - 1, dtype=np.int64),
                       b=np.arange(1, count, dtype=np.int64))

    def test_a_degree_two_marker_is_replaced_by_one_edge(self) -> None:
        markers = self.chain(5)
        before = drawn(markers)
        keep = np.ones(5, dtype=bool)
        keep[2] = False
        after = remove(markers, keep)
        self.assertEqual(after.size, 4)
        self.assertEqual(after.edge_count, 3)
        # The two neighbours of the marker that left are now linked: they
        # were 1 and 3 and are 1 and 2 after the reindex.
        self.assertIn((1, 2), list(zip(after.a.tolist(), after.b.tolist())))
        # And the raster between them is unbroken.
        self.assertTrue(bool(drawn(after)[20, 12]))
        self.assertTrue(np.array_equal(before, drawn(after)))

    def test_a_degree_three_marker_drops_three_edges(self) -> None:
        # A star: marker 0 linked to 1, 2 and 3.
        markers = Markers(x=np.array([20.0, 21.0, 19.0, 20.0]),
                          y=np.array([20.0, 20.0, 20.0, 21.0]),
                          s=np.full(4, float(SEAM_OPEN_STRENGTH)),
                          a=np.array([0, 0, 0], dtype=np.int64),
                          b=np.array([1, 2, 3], dtype=np.int64))
        self.assertEqual(int(degrees(markers)[0]), 3)
        keep = np.array([False, True, True, True])
        after = remove(markers, keep)
        self.assertEqual((after.size, after.edge_count), (3, 0))

    def test_edges_reindex_under_a_random_removal(self) -> None:
        rng = np.random.default_rng(4287772760)
        for trial in range(20):
            count = int(rng.integers(6, 40))
            markers = Markers(
                x=rng.uniform(0.0, N, count), y=rng.uniform(0.0, N, count),
                s=np.full(count, float(SEAM_OPEN_STRENGTH)))
            pairs = rng.integers(0, count, size=(2, count))
            a, b = canonical_edges(pairs[0], pairs[1], count)
            markers = Markers(x=markers.x, y=markers.y, s=markers.s,
                              a=a, b=b)
            keep = rng.random(count) > 0.35
            if not keep.any():
                continue
            # The identity of every marker, so a survivor can be recognized
            # after the reindex.
            before = {i: (float(markers.x[i]), float(markers.y[i]))
                      for i in range(count)}
            survivors = [i for i in range(count) if keep[i]]
            expected = {
                tuple(sorted((survivors.index(int(u)),
                              survivors.index(int(v)))))
                for u, v in zip(markers.a.tolist(), markers.b.tolist())
                if keep[u] and keep[v]}
            after = remove(markers, keep)
            with self.subTest(trial=trial):
                self.assertEqual(after.size, len(survivors))
                for new, old in enumerate(survivors):
                    self.assertEqual(
                        (float(after.x[new]), float(after.y[new])),
                        before[old])
                got = set(zip(after.a.tolist(), after.b.tolist()))
                # Every edge that joined two survivors is still there, on the
                # same two markers; the extra ones are the degree-2 splices.
                self.assertTrue(expected <= got)
                self.assertTrue(bool(np.all(after.a < after.size)))
                self.assertTrue(bool(np.all(after.b < after.size)))
                self.assertTrue(bool(np.all(after.a != after.b)))
                packed = after.a * after.size + after.b
                self.assertEqual(packed.size, np.unique(packed).size)

    def test_a_marker_leaves_at_the_suture_strength_and_not_before(
            self) -> None:
        markers = Markers(x=np.array([3.0]), y=np.array([3.0]),
                          s=np.array([0.7]))
        excess = np.zeros((8, 8), dtype=np.float64)
        # 0.7 is above the weak threshold and below the suture strength: a
        # remembered vertex on an intact cell, which C04.5 would have removed.
        survivors, removed, reactivated = damage_and_heal(
            markers, excess, 1.0 / 10.0, 1.0 / 1.5, 4.0, 8)
        self.assertEqual((survivors.size, removed, reactivated), (1, 0, 0))
        self.assertFalse(bool(seam_mask(raster(survivors, 8))[3, 3]))
        # Healing on to the suture strength does remove it.
        for _ in range(20):
            survivors, removed, _re = damage_and_heal(
                survivors, excess, 1.0 / 10.0, 1.0 / 1.5, 4.0, 8)
            if survivors.size == 0:
                break
        self.assertEqual((survivors.size, removed), (0, 1))

    def test_a_marker_that_crosses_back_below_the_threshold_reactivates(
            self) -> None:
        markers = Markers(x=np.array([3.0]), y=np.array([3.0]),
                          s=np.array([0.6]))
        excess = np.zeros((8, 8), dtype=np.float64)
        excess[3, 3] = 3.0
        _survivors, removed, reactivated = damage_and_heal(
            markers, excess, 1.0 / 10.0, 1.0 / 1.5, 4.0, 8)
        self.assertEqual((removed, reactivated), (0, 1))


class Subdivision(unittest.TestCase):
    def test_an_edge_stretched_past_the_bound_is_split_once(self) -> None:
        markers = Markers(x=np.array([10.0, 12.0]), y=np.full(2, 20.0),
                          s=np.array([0.1, 0.3]),
                          a=np.array([0], dtype=np.int64),
                          b=np.array([1], dtype=np.int64))
        split, count = subdivide(markers, N)
        self.assertEqual(count, 1)
        self.assertEqual((split.size, split.edge_count), (3, 2))
        self.assertAlmostEqual(float(split.x[2]), 11.0, places=12)
        self.assertAlmostEqual(float(split.y[2]), 20.0, places=12)
        self.assertAlmostEqual(float(split.s[2]), 0.2, places=12)
        # Two edges, each from an end to the midpoint.
        self.assertEqual(sorted(zip(split.a.tolist(), split.b.tolist())),
                         [(0, 2), (1, 2)])
        self.assertTrue(bool(np.all(segment_lengths(split, N) <= 1.0 + 1e-12)))

    def test_an_edge_inside_the_bound_is_left_alone(self) -> None:
        markers = Markers(x=np.array([10.0, 11.4]), y=np.full(2, 20.0),
                          s=np.full(2, 0.1),
                          a=np.array([0], dtype=np.int64),
                          b=np.array([1], dtype=np.int64))
        split, count = subdivide(markers, N)
        self.assertEqual(count, 0)
        self.assertEqual(split.size, 2)

    def test_the_midpoint_is_the_minimal_image_one(self) -> None:
        markers = Markers(x=np.array([63.0, 2.0]), y=np.full(2, 20.0),
                          s=np.full(2, 0.1),
                          a=np.array([0], dtype=np.int64),
                          b=np.array([1], dtype=np.int64))
        split, count = subdivide(markers, N)
        self.assertEqual(count, 1)
        # Across the wrap the two are three cells apart, not sixty-one.
        self.assertAlmostEqual(float(split.x[2]) % N, 0.5, places=12)


class TheEdgeList(unittest.TestCase):
    def test_it_carries_no_duplicates_and_no_self_edges(self) -> None:
        a, b = canonical_edges(np.array([1, 2, 1, 3, 3]),
                               np.array([2, 1, 2, 3, 4]), 5)
        self.assertEqual(list(zip(a.tolist(), b.tolist())), [(1, 2), (3, 4)])

    def test_appending_markers_does_not_move_an_index(self) -> None:
        markers = Markers(x=np.array([1.0, 2.0]), y=np.zeros(2),
                          s=np.full(2, 0.1),
                          a=np.array([0], dtype=np.int64),
                          b=np.array([1], dtype=np.int64))
        grown = add(markers, np.array([3.0]), np.array([0.0]),
                    np.array([0.1]), np.array([1]))
        self.assertEqual(sorted(zip(grown.a.tolist(), grown.b.tolist())),
                         [(0, 1), (1, 2)])

    def test_a_chain_is_a_connected_component(self) -> None:
        markers = Markers(x=np.arange(6.0), y=np.zeros(6),
                          s=np.full(6, 0.1),
                          a=np.array([0, 1, 3], dtype=np.int64),
                          b=np.array([1, 2, 4], dtype=np.int64))
        labels = chain_labels(markers)
        self.assertEqual(labels.tolist(), [0, 0, 0, 3, 3, 5])


class Tangling(unittest.TestCase):
    def test_one_segment_per_cell_on_a_straight_chain(self) -> None:
        x = 10.0 + np.arange(8, dtype=np.float64)
        markers = Markers(x=x, y=np.full(8, 20.0), s=np.full(8, 0.1),
                          a=np.arange(7, dtype=np.int64),
                          b=np.arange(1, 8, dtype=np.int64))
        counts = segments_per_cell(markers, N)
        covered = counts[counts > 0]
        # Consecutive segments share their end cells, so the mean is under 2.
        self.assertLess(float(covered.mean()), 2.0)
        self.assertEqual(int(counts.max()), 2)


class TheWeakSetIsWhatTheCurveDraws(unittest.TestCase):
    def test_the_weak_cells_may_outnumber_the_markers(self) -> None:
        """The curve is a line, not a point set: it draws between vertices."""
        markers = Markers(x=np.array([10.0, 11.4, 12.8]),
                          y=np.full(3, 20.0), s=np.full(3, 0.1),
                          a=np.array([0, 1], dtype=np.int64),
                          b=np.array([1, 2], dtype=np.int64))
        mask = drawn(markers)
        self.assertGreaterEqual(int(mask.sum()), markers.size)
        self.assertTrue(eight_connected(mask))

    def test_a_segment_at_every_angle_is_one_cell_wide(self) -> None:
        for degrees_ in range(0, 180, 3):
            angle = math.radians(degrees_)
            markers = edge_between(32.0, 32.0,
                                   32.0 + 17.0 * math.cos(angle),
                                   32.0 + 17.0 * math.sin(angle))
            mask = drawn(markers)
            with self.subTest(degrees=degrees_):
                self.assertTrue(eight_connected(mask))
                self.assertTrue(one_cell_wide(mask))

    def test_the_threshold_is_still_the_weak_one(self) -> None:
        self.assertEqual(WEAK_THRESHOLD, 0.5)


if __name__ == "__main__":
    unittest.main()
