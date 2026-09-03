"""Gates on the continuous tip direction of `WORK_ORDER_C04_5.md` §1.

The eight-direction rule of `WORK_ORDER_C04.md` §2.4 scores a tip's eight
neighbours and steps into the best of them. Under a stress field that varies
over hundreds of cells the winner is whichever of the eight lies nearest the
principal axis, and it wins again at the next advance, so a crack loaded at
20 degrees runs at 0: nothing in that rule can alternate two lattice
directions to make an angle between them, because it has no memory of how far
the tip has been pushed off its line.

`markers.advance_tips` gives it one. The tip is the end vertex of its chain
and its position is that vertex's own — a float since C04.2 — the direction is
read off the stress as a continuous vector, and the advance walks from that
position, so the marker it creates lands where the walk reached and the next
advance starts from there. The tests below are the three things that has to
mean: a crack loaded between two lattice directions runs between them, a crack
loaded along one runs straight along it, and a marker sits where the walk left
it.

`WORK_ORDER_C04_5.md` §1 read the tip off the raster and averaged the markers
of its cell; `WORK_ORDER_C04_6.md` §1.3 reads it off the curve instead, and
the direction rule below is C04.5's unchanged. `tip_pass` itself is untouched
and is pinned by `test_seams.py`; `seams = 1` runs it still.
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

from engine.history.constants import SEAM_OPEN_STRENGTH  # noqa: E402
from engine.history.markers import (  # noqa: E402
    Markers,
    advance_tips,
    degrees,
    raster,
)
from engine.history.seams import principal_normal, seam_mask  # noqa: E402

N = 64


def loaded_at(degrees: float, n: int = N):
    """A uniform stress whose seam direction is `degrees` from the x axis.

    A seam along `d` carries the traction on its own face, so the normal is
    `d` turned a quarter turn and the rule runs the crack along the
    eigenvector of the *normal*. Building the tensor as a uniaxial tension of
    magnitude one along `n` therefore makes `d` the direction the rule must
    pick, and by construction the largest absolute eigenvalue is 1 against a
    second of 0, so the choice is not a degenerate one.
    """
    angle = math.radians(degrees)
    dx, dy = math.cos(angle), math.sin(angle)
    nx, ny = -dy, dx
    ones = np.ones((n, n), dtype=np.float64)
    return (nx * nx * ones, ny * ny * ones, nx * ny * ones)


def marker_chain(markers: Markers) -> list[int]:
    """The one chain of the curve, walked from one of its two end vertices.

    The chord and the bends are read off the **curve** and not off the
    raster: half-cell sampling draws a staircase whose corners are three
    mutually 8-adjacent cells, so a walk over the drawn cells has a choice at
    every corner and is not a path. The vertices have no such ambiguity.
    """
    degree = degrees(markers)
    ends = np.nonzero(degree == 1)[0].tolist()
    if len(ends) != 2:
        raise AssertionError(f"not one unbranched chain: {len(ends)} ends")
    neighbours: dict[int, list[int]] = {i: [] for i in range(markers.size)}
    for u, v in zip(markers.a.tolist(), markers.b.tolist()):
        neighbours[u].append(v)
        neighbours[v].append(u)
    chain = [ends[0]]
    previous = -1
    while True:
        step = [i for i in neighbours[chain[-1]] if i != previous]
        if not step:
            break
        previous = chain[-1]
        chain.append(step[0])
    return chain


def nucleus(row: int = 32, column: int = 32) -> Markers:
    """One marker of degree 0 at a cell centre: a fresh nucleus."""
    return Markers(x=np.array([float(column)]), y=np.array([float(row)]),
                   s=np.array([float(SEAM_OPEN_STRENGTH)]))


def ordered_chain(seam: np.ndarray) -> list[tuple[int, int]]:
    """The one 8-connected seam component, walked from an end."""
    seam = np.asarray(seam, dtype=bool)
    n = seam.shape[0]
    cells = [tuple(cell) for cell in np.argwhere(seam).tolist()]
    neighbours = {
        cell: [((cell[0] + dy) % n, (cell[1] + dx) % n)
               for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and seam[(cell[0] + dy) % n, (cell[1] + dx) % n]]
        for cell in cells
    }
    ends = [cell for cell in cells if len(neighbours[cell]) <= 1]
    if len(ends) != 2:
        raise AssertionError(f"not one unbranched chain: {len(ends)} ends")
    chain = [ends[0]]
    previous = None
    while True:
        step = [cell for cell in neighbours[chain[-1]] if cell != previous]
        if not step:
            break
        previous = chain[-1]
        chain.append(step[0])
    return chain


class TheDirection(unittest.TestCase):
    def test_the_normal_is_the_eigenvector_of_the_largest_eigenvalue(
            self) -> None:
        for degrees in (0.0, 20.0, 45.0, 70.0, 135.0):
            with self.subTest(degrees=degrees):
                sxx, syy, sxy = loaded_at(degrees, 1)
                nx, ny, larger, smaller = principal_normal(sxx, syy, sxy)
                # The tensor is a unit tension along the normal.
                self.assertAlmostEqual(float(larger[0, 0]), 1.0, places=12)
                self.assertAlmostEqual(float(smaller[0, 0]), 0.0, places=12)
                angle = math.radians(degrees)
                want = (-math.sin(angle), math.cos(angle))
                got = (float(nx[0, 0]), float(ny[0, 0]))
                # Either sign of an eigenvector is an eigenvector.
                self.assertAlmostEqual(abs(got[0] * want[0] + got[1] * want[1]),
                                       1.0, places=9)

    def test_an_isotropic_tensor_is_answered_deterministically(self) -> None:
        ones = np.ones((1, 1), dtype=np.float64)
        nx, ny, larger, smaller = principal_normal(ones, ones, 0.0 * ones)
        self.assertEqual((float(nx[0, 0]), float(ny[0, 0])), (1.0, 0.0))
        self.assertEqual(float(larger[0, 0]), float(smaller[0, 0]))


class AChainOfAdvances(unittest.TestCase):
    """Forty advances from one nucleus under a uniform stress.

    `sigma_c` is 0.5 against a best traction of 1.0, so every candidate
    qualifies and the only question is where the rule walks. The chain is
    grown one pass at a time exactly as `run_history` grows it: the raster is
    the markers', a pass opens a cell, the marker for that cell goes at the
    point the walk reached, and the next pass starts from it.
    """

    def grow(self, degrees: float, advances: int = 40,
             start: tuple[int, int] = (32, 32)):
        """The chain in path order, and the markers that carry it.

        A nucleus is a tip at both ends once it is two cells long, so the
        crack grows both ways and the cells are not opened in path order;
        the chain is walked off the raster at the end instead.
        """
        markers = nucleus(*start)
        sxx, syy, sxy = loaded_at(degrees)
        sigma_c = np.full((N, N), 0.5)
        opened_total = 0
        for _ in range(advances):
            grown, _tips, opened, _meetings, _degenerate = advance_tips(
                markers, raster(markers, N), sxx, syy, sxy, sigma_c)
            if opened == 0:
                break
            opened_total += opened
            markers = grown
            if opened_total >= advances:
                break
        order = marker_chain(markers)
        cells = [(int(round(float(markers.y[i]))),
                  int(round(float(markers.x[i])))) for i in order]
        squeezed = [cells[0]]
        for cell in cells[1:]:
            if cell != squeezed[-1]:
                squeezed.append(cell)
        return squeezed, markers

    def bends(self, chain) -> int:
        """Steps whose offset differs from the one before it."""
        steps = [((b[0] - a[0] + 32) % 64 - 32, (b[1] - a[1] + 32) % 64 - 32)
                 for a, b in zip(chain, chain[1:])]
        return sum(1 for a, b in zip(steps, steps[1:]) if a != b)

    def chord_degrees(self, chain) -> float:
        (y0, x0), (y1, x1) = chain[0], chain[-1]
        return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0

    def test_a_crack_at_twenty_degrees_runs_at_twenty_degrees(self) -> None:
        chain, _markers = self.grow(20.0)
        self.assertGreaterEqual(len(chain), 30)
        self.assertAlmostEqual(self.chord_degrees(chain), 20.0, delta=3.0)
        self.assertGreaterEqual(self.bends(chain), 8)

    def test_a_crack_along_the_lattice_runs_straight(self) -> None:
        for degrees_ in (0.0, 45.0):
            with self.subTest(degrees=degrees_):
                chain, _markers = self.grow(degrees_)
                self.assertGreaterEqual(len(chain), 30)
                self.assertAlmostEqual(self.chord_degrees(chain), degrees_,
                                       delta=3.0)
                self.assertEqual(self.bends(chain), 0)

    def test_the_eight_direction_rule_cannot_do_the_same(self) -> None:
        """The measurement that made this order: the old rule locks at 0.

        Grown the same way through `tip_pass`, a crack loaded at 20 degrees
        runs along the nearest lattice direction and never leaves it.
        """
        from engine.history.seams import tip_pass

        strength = np.ones((N, N), dtype=np.float64)
        strength[32, 32] = SEAM_OPEN_STRENGTH
        sxx, syy, sxy = loaded_at(20.0)
        sigma_c = np.full((N, N), 0.5)
        total = 0
        for _ in range(40):
            moved, _count, opened = tip_pass(strength, sxx, syy, sxy, sigma_c)
            if opened == 0:
                break
            total += opened
            strength = moved
            if total >= 40:
                break
        chain = ordered_chain(seam_mask(strength))
        self.assertGreaterEqual(len(chain), 41)
        self.assertEqual(self.bends(chain), 0)
        self.assertAlmostEqual(self.chord_degrees(chain), 0.0, delta=1e-9)

class TheMarkerAnAdvanceCreates(unittest.TestCase):
    def test_it_sits_at_the_point_reached_and_inside_the_cell_it_opened(
            self) -> None:
        markers = nucleus()
        sxx, syy, sxy = loaded_at(20.0)
        grown, _tips, opened, _meetings, _degenerate = advance_tips(
            markers, raster(markers, N), sxx, syy, sxy, np.full((N, N), 0.5))
        self.assertEqual(opened, 1)
        self.assertEqual(grown.size, 2)
        # The walk is one cell long from the nucleus at (32, 32) along the
        # 20-degree direction, so it lands at (32 + cos 20, 32 + sin 20) =
        # (32.940, 32.342), whose nearest cell is (32, 33).
        x = float(grown.x[1])
        y = float(grown.y[1])
        self.assertAlmostEqual(x, 32.0 + math.cos(math.radians(20.0)),
                               places=12)
        self.assertAlmostEqual(y, 32.0 + math.sin(math.radians(20.0)),
                               places=12)
        cell = (int(round(y)), int(round(x)))
        self.assertEqual(cell, (32, 33))
        # Not the centre, and inside the cell it opened.
        self.assertNotEqual((y, x), (float(cell[0]), float(cell[1])))
        self.assertTrue(bool(seam_mask(raster(grown, N))[cell]))

    def test_the_advance_carries_one_edge_back_to_its_tip(self) -> None:
        markers = nucleus()
        sxx, syy, sxy = loaded_at(20.0)
        grown, _tips, _opened, meetings, _degenerate = advance_tips(
            markers, raster(markers, N), sxx, syy, sxy, np.full((N, N), 0.5))
        self.assertEqual(meetings, 0)
        self.assertEqual(list(zip(grown.a.tolist(), grown.b.tolist())),
                         [(0, 1)])


class ThePassItself(unittest.TestCase):
    def test_an_intact_sheet_has_no_tips_and_no_advances(self) -> None:
        sxx, syy, sxy = loaded_at(20.0)
        markers = Markers(x=np.zeros(0), y=np.zeros(0), s=np.zeros(0))
        result = advance_tips(markers, np.ones((N, N)), sxx, syy, sxy,
                              np.full((N, N), 0.5))
        self.assertEqual(result[1:], (0, 0, 0, 0))

    def test_a_strength_above_the_traction_refuses_the_candidate(self) -> None:
        markers = nucleus()
        strength = raster(markers, N)
        sxx, syy, sxy = loaded_at(20.0)
        held = advance_tips(markers, strength, sxx, syy, sxy,
                            np.full((N, N), 1.5))
        self.assertEqual(held[1:3], (1, 0))
        self.assertEqual(held[0].size, 1)
        # The threshold is the toughness times the intact strength at a
        # crack one cell long, exactly as the eight-direction rule has it.
        opened = advance_tips(markers, strength, sxx, syy, sxy,
                              np.full((N, N), 1.5), 0.5)
        self.assertEqual(opened[2], 1)

    def test_the_candidate_is_always_one_of_the_eight_neighbours(self) -> None:
        sigma_c = np.full((N, N), 0.5)
        markers = nucleus()
        strength = raster(markers, N)
        for degrees in range(0, 180, 7):
            with self.subTest(degrees=degrees):
                sxx, syy, sxy = loaded_at(float(degrees))
                grown, _tips, opened, _meetings, _degenerate = advance_tips(
                    markers, strength, sxx, syy, sxy, sigma_c)
                self.assertEqual(opened, 1)
                y = int(round(float(grown.y[1])))
                x = int(round(float(grown.x[1])))
                self.assertLessEqual(abs(y - 32), 1)
                self.assertLessEqual(abs(x - 32), 1)

    def test_a_tip_whose_neighbours_are_all_seams_stands_still(self) -> None:
        # A closed ring of eight markers around one at the centre. The ring
        # has degree 2 everywhere and holds no tip; the centre is the one
        # tip, and its eight neighbours are all seams, so it has no intact
        # neighbour to read a tensor from and nowhere to go.
        ring = [(31, 31), (31, 32), (31, 33), (32, 33),
                (33, 33), (33, 32), (33, 31), (32, 31)]
        rows = [row for row, _column in ring] + [32]
        columns = [column for _row, column in ring] + [32]
        markers = Markers(
            x=np.array(columns, dtype=np.float64),
            y=np.array(rows, dtype=np.float64),
            s=np.full(9, float(SEAM_OPEN_STRENGTH)),
            a=np.arange(8, dtype=np.int64),
            b=np.roll(np.arange(8, dtype=np.int64), -1))
        self.assertEqual(int((degrees(markers) <= 1).sum()), 1)
        sxx, syy, sxy = loaded_at(20.0)
        result = advance_tips(markers, raster(markers, N), sxx, syy, sxy,
                              np.full((N, N), 0.5))
        self.assertEqual(result[1:3], (1, 0))

    def test_the_crack_length_is_the_markers_in_the_chain(self) -> None:
        """`L` divides the threshold, so a longer chain advances where a
        shorter one cannot. Four markers in a chain give `sqrt(4) = 2`."""
        x = 30.0 + np.arange(4, dtype=np.float64)
        chain = Markers(x=x, y=np.full(4, 32.0),
                        s=np.full(4, float(SEAM_OPEN_STRENGTH)),
                        a=np.arange(3, dtype=np.int64),
                        b=np.arange(1, 4, dtype=np.int64))
        one = nucleus(32, 30)
        sxx, syy, sxy = loaded_at(0.0)
        # A strength between the one-cell threshold and the four-cell one.
        sigma_c = np.full((N, N), 1.6)
        self.assertEqual(advance_tips(one, raster(one, N), sxx, syy, sxy,
                                      sigma_c)[2], 0)
        self.assertGreater(advance_tips(chain, raster(chain, N), sxx, syy,
                                        sxy, sigma_c)[2], 0)


if __name__ == "__main__":
    unittest.main()
