"""Gates on world identity and the integer geometry fields live on."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

from engine.geometry import WorldGeometry  # noqa: E402
from engine.history.constants import (  # noqa: E402
    CELL_PX,
    DEFAULT_SIZE,
    SCALE_DEFAULT,
    SUPPORTED_SIZES,
)

# pixels, scale, history_n, cell_km, parent_km, window_km, window_cells
EXPECTED = (
    (1024, 5, 256, 40, 10240, 5120, 128),
    (512, 5, 128, 40, 5120, 2560, 64),
    (128, 5, 128, 40, 5120, 640, 16),
    (2048, 5, 512, 40, 20480, 10240, 256),
    (1024, 20, 256, 160, 40960, 20480, 128),
    (2048, 20, 512, 160, 81920, 40960, 256),
)


class DerivedGeometry(unittest.TestCase):
    def test_derived_values_are_exact(self) -> None:
        for pixels, scale, n, cell, parent, window, cells in EXPECTED:
            with self.subTest(pixels=pixels, scale=scale):
                geometry = WorldGeometry(7, pixels, scale)
                self.assertEqual(geometry.history_n, n)
                self.assertEqual(geometry.cell_km, cell)
                self.assertEqual(geometry.parent_km, parent)
                self.assertEqual(geometry.window_km, window)
                self.assertEqual(geometry.window_cells, cells)
                self.assertEqual(geometry.cell_m, cell * 1000)
                self.assertEqual(geometry.parent_m, parent * 1000)

    def test_every_derived_value_is_a_plain_int(self) -> None:
        geometry = WorldGeometry(7, 1024, 5)
        for name in ("history_n", "cell_km", "parent_km", "window_km",
                     "window_cells", "cell_m", "parent_m"):
            with self.subTest(name):
                self.assertIs(type(getattr(geometry, name)), int)

    def test_cell_centre_is_exact_and_inside_its_cell(self) -> None:
        geometry = WorldGeometry(7, 1024, 5)
        cell_m = geometry.cell_m
        self.assertEqual(cell_m % 2, 0)
        for index in (0, 1, 127, geometry.history_n - 1):
            with self.subTest(index=index):
                centre = geometry.cell_centre_m(index)
                self.assertIs(type(centre), int)
                self.assertEqual(centre, cell_m * index + cell_m // 2)
                self.assertLess(cell_m * index, centre)
                self.assertLess(centre, cell_m * (index + 1))

    def test_parent_is_twice_the_window_above_the_floor(self) -> None:
        for pixels in (1024, 2048):
            for scale in (5, 20):
                with self.subTest(pixels=pixels, scale=scale):
                    geometry = WorldGeometry(7, pixels, scale)
                    self.assertEqual(geometry.parent_km, 2 * geometry.window_km)

    def test_defaults_are_in_the_supported_sets(self) -> None:
        self.assertIn(DEFAULT_SIZE, SUPPORTED_SIZES)
        self.assertEqual(CELL_PX, 8)
        geometry = WorldGeometry(0, DEFAULT_SIZE, SCALE_DEFAULT)
        self.assertEqual(geometry.history_n, 256)


class Validation(unittest.TestCase):
    def test_unsupported_resolution_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            WorldGeometry(7, 1000, 5)

    def test_out_of_range_scale_is_rejected(self) -> None:
        for scale in (4, 21, 0, -5):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                WorldGeometry(7, 1024, scale)

    def test_non_integer_scale_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            WorldGeometry(7, 1024, 5.0)

    def test_bools_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            WorldGeometry(True, 1024, 5)

    def test_out_of_range_seed_is_rejected(self) -> None:
        for seed in (-1, 2**32):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                WorldGeometry(seed, 1024, 5)
        WorldGeometry(0, 1024, 5)
        WorldGeometry(2**32 - 1, 1024, 5)


class WorldIdentity(unittest.TestCase):
    def test_identity_is_stable_across_calls(self) -> None:
        geometry = WorldGeometry(4287772760, 1024, 5)
        self.assertEqual(geometry.world_id, geometry.world_id)
        self.assertEqual(len(geometry.world_id), 64)

    def test_resolution_and_scale_are_part_of_the_world(self) -> None:
        seed = 4287772760
        base = WorldGeometry(seed, 1024, 5).world_id
        self.assertNotEqual(base, WorldGeometry(seed, 512, 5).world_id)
        self.assertNotEqual(base, WorldGeometry(seed, 1024, 6).world_id)
        self.assertNotEqual(base, WorldGeometry(seed + 1, 1024, 5).world_id)

    def test_record_carries_every_field_and_property(self) -> None:
        geometry = WorldGeometry(7, 512, 5)
        record = geometry.to_record()
        self.assertEqual(sorted(record), [
            "cell_km", "cell_m", "history_n", "parent_km", "parent_m",
            "pixels", "scale_km", "seed", "window_cells", "window_km",
            "world_id",
        ])
        self.assertEqual(record["world_id"], geometry.world_id)
        self.assertEqual(record["history_n"], 128)


if __name__ == "__main__":
    unittest.main()
