"""Gates on reading plates and boundaries off a strength field."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

from engine.history.constants import WEAK_THRESHOLD  # noqa: E402
from engine.history.plates import (  # noqa: E402
    REGIME_CONVERGENT,
    REGIME_DIVERGENT,
    REGIME_NONE,
    REGIME_SHEAR,
    boundary_mask,
    label_plates,
    plate_areas,
    regime,
    weak_mask,
)

N = 128
STRONG = 0.9
WEAK = 0.1


def strong_field() -> np.ndarray:
    return np.full((N, N), STRONG)


def _label_plates_reference(strength: np.ndarray) -> np.ndarray:
    """The C03 labelling, kept here to compare the C03.1 one against.

    Neighbour minima and nothing else, repeated until the field stops moving.
    Correct and slow: it moves a label one cell per round.
    """
    weak = weak_mask(strength)
    strong = ~weak
    n = strength.shape[0]
    labels = np.arange(n * n, dtype=np.int64).reshape(n, n)
    sentinel = n * n
    labels = np.where(strong, labels, sentinel)

    shifts = ((-1, -1), (1, -1), (-1, -2), (1, -2))
    while True:
        previous = labels.copy()
        for shift, axis in shifts:
            rolled = np.roll(labels, shift, axis=axis)
            both = strong & np.roll(strong, shift, axis=axis)
            labels = np.where(both, np.minimum(labels, rolled), labels)
        if np.array_equal(labels, previous):
            break

    result = np.full((n, n), -1, dtype=np.int32)
    if not strong.any():
        return result
    roots, inverse, counts = np.unique(
        labels[strong], return_inverse=True, return_counts=True)
    order = np.lexsort((roots, -counts))
    rank = np.empty(order.size, dtype=np.int32)
    rank[order] = np.arange(order.size, dtype=np.int32)
    result[strong] = rank[inverse]
    return result


def winding_corridor() -> np.ndarray:
    """One strong corridor that snakes across the whole grid, weak elsewhere.

    Sixteen horizontal strips two cells deep, joined alternately at the left
    and the right end, so the corridor is a single component whose diameter is
    the length of the whole snake rather than the width of the grid.
    """
    strength = np.full((N, N), WEAK)
    for band in range(N // 8):
        row = 8 * band
        strength[row:row + 2, 1:N - 1] = STRONG
        column = 1 if band % 2 else N - 2
        rows = np.arange(row, row + 10) % N
        strength[rows, column] = STRONG
    return strength


class Labelling(unittest.TestCase):
    def test_two_bands_give_two_plates_that_wrap(self) -> None:
        strength = strong_field()
        strength[10:12] = WEAK
        strength[100:102] = WEAK
        labels = label_plates(strength)
        areas = plate_areas(labels)
        self.assertEqual(len(areas), 2)
        self.assertEqual(list(areas), [88 * N, 36 * N])
        # Every plate spans the columns, so both touch the wrap edge.
        self.assertTrue(np.array_equal(labels[:, 0], labels[:, -1]))
        # The larger plate is rows 12..99; the smaller wraps across row 0.
        self.assertEqual(set(np.unique(labels[12:100])), {0})
        self.assertEqual(set(np.unique(labels[102:])), {1})
        self.assertEqual(set(np.unique(labels[:10])), {1})
        self.assertEqual(set(np.unique(labels[10:12])), {-1})

    def test_one_band_gives_one_plate_on_the_torus(self) -> None:
        strength = strong_field()
        strength[40:42] = WEAK
        labels = label_plates(strength)
        self.assertEqual(len(plate_areas(labels)), 1)
        self.assertEqual(int(plate_areas(labels)[0]), (N - 2) * N)

    def test_an_isolated_weak_square_leaves_one_plate(self) -> None:
        strength = strong_field()
        strength[60:70, 60:70] = WEAK
        labels = label_plates(strength)
        self.assertEqual(len(plate_areas(labels)), 1)
        self.assertTrue((labels[60:70, 60:70] == -1).all())
        self.assertEqual(int(plate_areas(labels)[0]), N * N - 100)

    def test_a_wholly_strong_world_is_one_plate(self) -> None:
        labels = label_plates(strong_field())
        self.assertEqual(list(plate_areas(labels)), [N * N])
        self.assertFalse((labels == -1).any())

    def test_labels_are_ordered_by_area(self) -> None:
        strength = strong_field()
        strength[10:12] = WEAK
        strength[30:32] = WEAK
        strength[100:102] = WEAK
        areas = plate_areas(label_plates(strength))
        self.assertEqual(len(areas), 3)
        self.assertTrue(all(areas[i] >= areas[i + 1] for i in range(len(areas) - 1)))

    def test_labelling_is_deterministic(self) -> None:
        strength = strong_field()
        strength[10:12] = WEAK
        strength[100:102] = WEAK
        self.assertEqual(label_plates(strength).tobytes(),
                         label_plates(strength).tobytes())

    def test_a_long_snake_agrees_with_the_old_implementation(self) -> None:
        strength = winding_corridor()
        labels = label_plates(strength)
        self.assertTrue(np.array_equal(labels, _label_plates_reference(strength)))
        self.assertEqual(len(plate_areas(labels)), 1)
        self.assertGreater(int(plate_areas(labels)[0]), 8 * N)

    def test_weak_mask_uses_the_declared_threshold(self) -> None:
        strength = np.asarray([[WEAK_THRESHOLD - 1e-9, WEAK_THRESHOLD]])
        self.assertEqual(list(weak_mask(strength)[0]), [True, False])


class Boundaries(unittest.TestCase):
    def test_boundary_is_the_weak_rows_and_their_strong_neighbours(self) -> None:
        strength = strong_field()
        strength[10:12] = WEAK
        strength[100:102] = WEAK
        weak = weak_mask(strength)
        boundary = boundary_mask(label_plates(strength), weak)
        marked = {int(row) for row in np.unique(np.nonzero(boundary)[0])}
        self.assertEqual(marked, {9, 10, 11, 12, 99, 100, 101, 102})
        for row in sorted(marked):
            self.assertTrue(boundary[row].all())


class Regimes(unittest.TestCase):
    def test_sign_of_divergence_selects_the_regime(self) -> None:
        strain = np.full((4, 4), 2.0)
        weak = np.ones((4, 4), dtype=bool)
        self.assertTrue((regime(strain, strain, weak) == REGIME_DIVERGENT).all())
        self.assertTrue((regime(-strain, strain, weak) == REGIME_CONVERGENT).all())
        self.assertTrue(
            (regime(np.zeros((4, 4)), strain, weak) == REGIME_SHEAR).all())

    def test_strong_cells_have_no_regime(self) -> None:
        strain = np.full((4, 4), 2.0)
        weak = np.zeros((4, 4), dtype=bool)
        self.assertTrue((regime(strain, strain, weak) == REGIME_NONE).all())

    def test_a_shallow_ratio_reads_as_shear(self) -> None:
        strain = np.full((4, 4), 1.0)
        weak = np.ones((4, 4), dtype=bool)
        self.assertTrue((regime(strain * 0.2, strain, weak) == REGIME_SHEAR).all())
        self.assertTrue(
            (regime(strain * -0.2, strain, weak) == REGIME_SHEAR).all())

    def test_zero_strain_does_not_divide_by_zero(self) -> None:
        zero = np.zeros((4, 4))
        weak = np.ones((4, 4), dtype=bool)
        self.assertTrue((regime(zero, zero, weak) == REGIME_SHEAR).all())


if __name__ == "__main__":
    unittest.main()
