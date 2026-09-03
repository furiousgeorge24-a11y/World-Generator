"""Gates on the WebUI adapter: its declared interface, and what it refuses."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
import unittest

from PIL import Image

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

import webui_adapter  # noqa: E402
from engine.history.constants import (  # noqa: E402
    DEFAULT_SIZE,
    SCALE_DEFAULT,
    SCALE_MAX,
    SCALE_MIN,
    STAGE_ID,
    SUPPORTED_SIZES,
)

FLOOR_PIXELS = 128
FLOOR_VIEW_PX = 128


class Meta(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = webui_adapter.meta()

    def test_declares_the_keys_the_shell_reads(self) -> None:
        for key in ("name", "version", "ready", "stage", "status", "controls",
                    "default_size", "supported_sizes", "views", "view_purposes"):
            with self.subTest(key=key):
                self.assertIn(key, self.meta)
        self.assertEqual(self.meta["name"], "pipeline_c land-origin lab")
        self.assertTrue(self.meta["ready"])
        self.assertEqual(self.meta["stage"], STAGE_ID)

    def test_scale_is_the_only_control(self) -> None:
        controls = self.meta["controls"]
        self.assertEqual(len(controls), 1)
        control = controls[0]
        self.assertEqual(control["name"], "scale_km")
        self.assertEqual(control["ctype"], "int")
        self.assertEqual(control["default"], SCALE_DEFAULT)
        self.assertEqual(control["lo"], SCALE_MIN)
        self.assertEqual(control["hi"], SCALE_MAX)
        self.assertEqual(control["tier"], "primary")
        self.assertEqual(control["invalidates"], "full")
        self.assertIn("never swept", control["promise"])

    def test_sizes_and_views_match_the_engine(self) -> None:
        self.assertEqual(self.meta["supported_sizes"], list(SUPPORTED_SIZES))
        self.assertEqual(self.meta["default_size"], DEFAULT_SIZE)
        self.assertEqual(self.meta["views"], list(webui_adapter.VIEWS))
        self.assertEqual(sorted(self.meta["view_purposes"]),
                         sorted(webui_adapter.VIEWS))
        self.assertEqual(webui_adapter.VIEWS[0], "plates")

    def test_plates_are_labelled_at_the_final_epoch_only(self) -> None:
        self.assertEqual(len(webui_adapter.VIEWS), 24)
        for suffix in ("_t25", "_t50", "_t75"):
            with self.subTest(suffix=suffix):
                self.assertNotIn(f"plates{suffix}", webui_adapter.VIEWS)
                self.assertIn(f"boundaries{suffix}", webui_adapter.VIEWS)
                self.assertIn("final epoch only",
                              self.meta["view_purposes"][f"boundaries{suffix}"])

    def test_the_seam_views_are_offered_and_render(self) -> None:
        # `WORK_ORDER_C04.md` §3: every new layer gets a view, in both
        # adapters, immediately after `power`.
        views = list(webui_adapter.VIEWS)
        self.assertEqual(views[views.index("power") + 1], "stress")
        self.assertEqual(views[views.index("power") + 2], "intact_strength")
        # `WORK_ORDER_C04_2.md` §5: the block model's two, after them.
        self.assertEqual(views[views.index("power") + 3], "mismatch")
        self.assertEqual(views[views.index("power") + 4], "pieces_motion")
        for view in ("stress", "intact_strength", "mismatch",
                     "pieces_motion"):
            with self.subTest(view=view):
                self.assertIn(view, self.meta["view_purposes"])

    def test_there_is_no_elevation_view(self) -> None:
        # A `hypsometric` view would be a placeholder; nothing here is height.
        self.assertNotIn("hypsometric", webui_adapter.VIEWS)
        self.assertIn("No crust", self.meta["status"])


class Validation(unittest.TestCase):
    def test_bad_controls_are_refused(self) -> None:
        for controls in ({"scale_km": 4}, {"scale_km": 21}, {"scale_km": True},
                         {"scale_km": 5.5}, {"unknown": 1},
                         {"scale_km": 5, "unknown": 1}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                webui_adapter.generate(1, controls, FLOOR_PIXELS, _steps=4)

    def test_bad_sizes_are_refused(self) -> None:
        for size in (1000, 64, 0, 129):
            with self.subTest(size=size), self.assertRaises(ValueError):
                webui_adapter.generate(1, None, size, _steps=4)

    def test_bad_seeds_are_refused(self) -> None:
        for seed in (-1, 2**32, True, 1.0):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                webui_adapter.generate(seed, None, FLOOR_PIXELS, _steps=4)

    def test_a_step_count_that_breaks_the_epochs_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            webui_adapter.generate(1, None, FLOOR_PIXELS, _steps=5)


class Rendering(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world = webui_adapter.generate(1, None, FLOOR_PIXELS, _steps=8)

    def test_the_world_carries_what_the_audit_runner_reads(self) -> None:
        self.assertEqual(self.world.seed, 1)
        self.assertEqual(self.world.pixels, FLOOR_PIXELS)
        self.assertEqual(self.world.scale_km, SCALE_DEFAULT)
        self.assertEqual(len(self.world.world_id), 64)

    def test_views_are_the_declared_list(self) -> None:
        self.assertEqual(webui_adapter.views(self.world), list(webui_adapter.VIEWS))

    def test_every_view_renders_at_native_resolution(self) -> None:
        for view in webui_adapter.VIEWS:
            with self.subTest(view=view):
                image = Image.open(BytesIO(webui_adapter.render_png(self.world, view)))
                expected = 2 * FLOOR_VIEW_PX if view == "plates_tiled" else FLOOR_VIEW_PX
                self.assertEqual(image.size, (expected, expected))
                self.assertEqual(image.convert("RGB").mode, "RGB")

    def test_an_unknown_view_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            webui_adapter.render_png(self.world, "hypsometric")

    def test_the_report_is_json_and_complete(self) -> None:
        record = webui_adapter.report(self.world)
        json.dumps(record)
        for key in ("seed", "pixels", "scale_km", "window_km", "parent_km",
                    "history_n", "cell_km", "steps", "step_myr", "history_myr",
                    "generation_seconds", "world_id", "stage", "plate_count",
                    "plate_area_percent", "weak_fraction_final",
                    "weak_fraction_by_epoch", "solver_cycles_mean",
                    "solver_cycles_max", "solver_residual_max",
                    "velocity_rms_km_per_myr", "contains", "does_not_contain"):
            with self.subTest(key=key):
                self.assertIn(key, record)
        self.assertEqual(len(record["weak_fraction_by_epoch"]), 4)
        self.assertNotIn("land", record["contains"])


class Determinism(unittest.TestCase):
    def test_the_same_inputs_give_the_same_pixels(self) -> None:
        first = webui_adapter.generate(9, {"scale_km": 5}, FLOOR_PIXELS, _steps=4)
        second = webui_adapter.generate(9, {"scale_km": 5}, FLOOR_PIXELS, _steps=4)
        for view in ("plates", "strength"):
            with self.subTest(view=view):
                self.assertEqual(webui_adapter.render_png(first, view),
                                 webui_adapter.render_png(second, view))

    def test_scale_changes_the_world(self) -> None:
        base = webui_adapter.generate(9, {"scale_km": 5}, FLOOR_PIXELS, _steps=4)
        other = webui_adapter.generate(9, {"scale_km": 6}, FLOOR_PIXELS, _steps=4)
        self.assertNotEqual(base.world_id, other.world_id)
        self.assertNotEqual(webui_adapter.render_png(base, "strength_initial"),
                            webui_adapter.render_png(other, "strength_initial"))


if __name__ == "__main__":
    unittest.main()
