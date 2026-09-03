"""Gates on the exploration lab: its dials, its sheets, and its two paths.

The lab is a development instrument on its own port. These tests check that
it declares what the shared shell reads, that every view comes back at the
size the sheet layout implies, that the report carries the keys the person at
the dials reads, and that the process pool and the sequential fallback produce
the same bundle byte for byte.
"""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

import explore_adapter  # noqa: E402
import webui_adapter  # noqa: E402
from engine.history.constants import (  # noqa: E402
    DEFAULT_SIZE,
    STAGE_ID,
    SUPPORTED_SIZES,
)
from engine.history.kinematics import HistoryParams  # noqa: E402

FLOOR_PIXELS = 128
FLOOR_VIEW_PX = 128
SEED = 4287772760

#: Two worlds is the smallest bundle that still has a gutter and two strips.
#: **120 Myr, not the lab's 300.** The lab defaults to `seams = 2`, and
#: `WORK_ORDER_C04_6.md` §1.4 subdivides every edge the move stretches past
#: `SEGMENT_MAX_CELLS`, so the vertex count grows by about 1.25 per step from
#: the step the first pieces separate and a 300 Myr world at 128 px does not
#: finish. 120 Myr is 30 steps of the same 4 Myr and exercises every path
#: these gates read; the growth itself is measured in
#: `out/C04_6_BUILD_REPORT.md`.
TWO = {"seeds_per_view": 2, "history_myr": 120}


class Meta(unittest.TestCase):
    def setUp(self) -> None:
        self.meta = explore_adapter.meta()

    def test_declares_the_keys_the_shell_reads(self) -> None:
        for key in ("name", "version", "ready", "stage", "status", "controls",
                    "default_size", "supported_sizes", "views"):
            with self.subTest(key=key):
                self.assertIn(key, self.meta)
        self.assertEqual(self.meta["name"], "pipeline_c exploration lab")
        self.assertTrue(self.meta["ready"])
        self.assertEqual(self.meta["stage"], STAGE_ID)
        self.assertEqual(self.meta["default_size"], DEFAULT_SIZE)
        self.assertEqual(self.meta["supported_sizes"], list(SUPPORTED_SIZES))

    def test_the_status_says_what_the_dials_are(self) -> None:
        status = self.meta["status"]
        self.assertIn("not author controls", status)
        self.assertIn("eight seeds", status)

    def test_it_is_a_different_backend_from_production(self) -> None:
        self.assertNotEqual(self.meta["name"], webui_adapter.meta()["name"])
        self.assertEqual(len(webui_adapter.meta()["controls"]), 1)

    def test_the_dials_are_declared_in_order(self) -> None:
        names = [control["name"] for control in self.meta["controls"]]
        self.assertEqual(names, [
            "scale_km", "seeds_per_view", "stiffness_fraction",
            "yield_percentile", "heal_time_myr", "damage_time_myr",
            "work_damage", "seams", "crack_speed_km_per_myr",
            "nucleations_per_step", "toughness_fraction",
            "strength_spread", "strength_exponent",
            "drive_wavelength_km",
            "drive_shear", "history_myr", "max_cycles", "solve_divisor",
        ])
        for control in self.meta["controls"]:
            with self.subTest(name=control["name"]):
                self.assertEqual(control["invalidates"], "full")
                self.assertIn(control["ctype"], ("int", "float"))
                self.assertIn(control["tier"], ("primary", "advanced"))
                self.assertTrue(control["promise"].strip())
                self.assertLessEqual(control["lo"], control["default"])
                self.assertLessEqual(control["default"], control["hi"])

    def test_the_lab_defaults_to_the_seam_formulation(self) -> None:
        # `WORK_ORDER_C04.md` §4 and `WORK_ORDER_C04_2.md` §6: the lab
        # exists to look at the newest formulation, so its default is 2, the
        # block model. The engine's own default is 0 and production is the
        # sheet.
        dials = {control["name"]: control for control in self.meta["controls"]}
        self.assertEqual(dials["seams"]["default"], 2)
        self.assertEqual(dials["seams"]["lo"], 0)
        self.assertEqual(dials["seams"]["hi"], 2)
        self.assertEqual(HistoryParams().seams, 0)
        self.assertEqual(dials["crack_speed_km_per_myr"]["default"], 40.0)
        self.assertEqual(dials["nucleations_per_step"]["default"], 2)
        self.assertIn("one cell wide", dials["seams"]["promise"])
        self.assertIn("rigid pieces", dials["seams"]["promise"])

    def test_the_seam_views_are_offered_after_power(self) -> None:
        views = list(self.meta["views"])
        self.assertEqual(views[views.index("power") + 1], "stress")
        self.assertEqual(views[views.index("power") + 2], "intact_strength")
        # `WORK_ORDER_C04_2.md` §5: the block model's two, after them.
        self.assertEqual(views[views.index("power") + 3], "mismatch")
        self.assertEqual(views[views.index("power") + 4], "pieces_motion")
        for view in ("stress", "intact_strength", "mismatch",
                     "pieces_motion"):
            with self.subTest(view=view):
                self.assertIn(view, self.meta["view_purposes"])

    def test_plates_is_first_and_there_is_no_hypsometric(self) -> None:
        self.assertEqual(self.meta["views"][0], "plates")
        self.assertNotIn("hypsometric", self.meta["views"])
        self.assertIn("trajectory", self.meta["views"])

    def test_the_defaults_that_are_history_params_are_in_range(self) -> None:
        defaults = {control["name"]: control["default"]
                    for control in self.meta["controls"]}
        HistoryParams(
            stiffness_fraction=defaults["stiffness_fraction"],
            yield_percentile=defaults["yield_percentile"],
            heal_time_myr=defaults["heal_time_myr"],
            damage_time_myr=defaults["damage_time_myr"],
            strength_exponent=defaults["strength_exponent"],
            drive_wavelength_km=defaults["drive_wavelength_km"],
            drive_shear=defaults["drive_shear"],
            history_myr=defaults["history_myr"],
            max_cycles=defaults["max_cycles"],
        )


class Validation(unittest.TestCase):
    def test_bad_controls_are_refused(self) -> None:
        for controls in ({"unknown": 1}, {"scale_km": 4}, {"scale_km": 21},
                         {"scale_km": 5.5}, {"seeds_per_view": 0},
                         {"seeds_per_view": 9}, {"stiffness_fraction": 0.04},
                         {"stiffness_fraction": 2.1}, {"yield_percentile": 0.5},
                         {"yield_percentile": 41}, {"heal_time_myr": 9},
                         {"damage_time_myr": 101}, {"strength_exponent": 7},
                         {"strength_exponent": 4.0},
                         {"drive_wavelength_km": 639.0},
                         {"drive_wavelength_km": 40961.0},
                         {"drive_shear": 1.5}, {"history_myr": 99},
                         {"max_cycles": 101}, {"max_cycles": True}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                explore_adapter.generate(1, controls, FLOOR_PIXELS)

    def test_bad_sizes_and_seeds_are_refused(self) -> None:
        for size in (1000, 64, 0, 129):
            with self.subTest(size=size), self.assertRaises(ValueError):
                explore_adapter.generate(1, TWO, size)
        for seed in (-1, 2**32, True, 1.0):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                explore_adapter.generate(seed, TWO, FLOOR_PIXELS)

    def test_the_seeds_run_on_from_the_one_asked_for(self) -> None:
        bundle = explore_adapter.generate(2**32 - 2,
                                          dict(TWO, seeds_per_view=3),
                                          FLOOR_PIXELS, _parallel=False)
        self.assertEqual([world["seed"] for world in bundle.worlds],
                         [2**32 - 2, 2**32 - 1, 0])


class Sheets(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = explore_adapter.generate(SEED, TWO, FLOOR_PIXELS,
                                              _parallel=False)

    def test_views_are_the_declared_list(self) -> None:
        self.assertEqual(explore_adapter.views(self.bundle),
                         list(explore_adapter.VIEWS))

    def test_every_view_renders_at_the_sheet_size(self) -> None:
        steps = self.bundle.worlds[0]["steps"]
        field = (2 * FLOOR_VIEW_PX + explore_adapter.SHEET_GUTTER_PX,
                 FLOOR_VIEW_PX)
        strip = (steps, 2 * explore_adapter.STRIP_PX
                 + explore_adapter.STRIP_GUTTER_PX)
        for view in explore_adapter.VIEWS:
            with self.subTest(view=view):
                image = Image.open(
                    BytesIO(explore_adapter.render_png(self.bundle, view)))
                self.assertEqual(image.size,
                                 strip if view == "trajectory" else field)
                self.assertEqual(image.convert("RGB").mode, "RGB")

    def test_one_seed_is_one_panel_with_no_gutter(self) -> None:
        bundle = explore_adapter.generate(SEED,
                                          dict(TWO, seeds_per_view=1),
                                          FLOOR_PIXELS, _parallel=False)
        image = Image.open(BytesIO(explore_adapter.render_png(bundle, "plates")))
        self.assertEqual(image.size, (FLOOR_VIEW_PX, FLOOR_VIEW_PX))

    def test_the_trajectory_strip_carries_the_half_line(self) -> None:
        sheet = np.asarray(explore_adapter.sheet(self.bundle, "trajectory"))
        row = sheet[explore_adapter.STRIP_PX // 2]
        self.assertTrue(np.all(row == np.asarray(
            explore_adapter.STRIP_LINE_RGB, dtype=np.uint8)))

    def test_an_unknown_view_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            explore_adapter.render_png(self.bundle, "hypsometric")

    def test_the_report_is_json_and_complete(self) -> None:
        record = explore_adapter.report(self.bundle)
        json.dumps(record)
        for key in ("dials", "yield_strain_per_myr", "parallel",
                    "generation_seconds", "worlds", "summary",
                    "stable_count_note", "dials_note"):
            with self.subTest(key=key):
                self.assertIn(key, record)
        self.assertEqual(len(record["worlds"]), 2)
        self.assertEqual(len(record["yield_strain_per_myr"]), 2)
        for world in record["worlds"]:
            for key in ("seed", "plate_count", "plate_area_percent",
                        "weak_final", "weak_peak", "weak_peak_myr",
                        "weak_at_100_myr", "strength_mean_strong",
                        "solver_cycles_mean", "solver_residual_max",
                        "exhausted_steps"):
                with self.subTest(key=key):
                    self.assertIn(key, world)
        for key in ("plate_count_min", "plate_count_max", "weak_final_mean",
                    "stable_count"):
            with self.subTest(key=key):
                self.assertIn(key, record["summary"])
        self.assertIn("not a gate", record["stable_count_note"])
        self.assertIn("not author controls", record["dials_note"])
        self.assertEqual(sorted(record["dials"]),
                         sorted(control["name"]
                                for control in explore_adapter.meta()["controls"]))

    def test_the_report_says_nothing_about_land(self) -> None:
        record = explore_adapter.report(self.bundle)
        self.assertNotIn("land", record["contains"])


class Determinism(unittest.TestCase):
    def test_two_generates_give_the_same_pixels(self) -> None:
        first = explore_adapter.generate(SEED, TWO, FLOOR_PIXELS,
                                         _parallel=False)
        second = explore_adapter.generate(SEED, TWO, FLOOR_PIXELS,
                                          _parallel=False)
        for view in explore_adapter.VIEWS:
            with self.subTest(view=view):
                self.assertEqual(explore_adapter.render_png(first, view),
                                 explore_adapter.render_png(second, view))

    def test_a_dial_that_moves_changes_the_sheet(self) -> None:
        base = explore_adapter.generate(SEED, TWO, FLOOR_PIXELS,
                                        _parallel=False)
        other = explore_adapter.generate(
            SEED, dict(TWO, yield_percentile=30.0), FLOOR_PIXELS,
            _parallel=False)
        self.assertNotEqual(explore_adapter.render_png(base, "strength"),
                            explore_adapter.render_png(other, "strength"))


class PoolEqualsSequential(unittest.TestCase):
    """Determinism must not depend on which path ran.

    The pool is created on the first parallel generate and kept for the life
    of the process. If it cannot be created the lab falls back to sequential
    and says so in the report; this test then compares sequential with
    sequential, which is still worth running, and prints which path it got.
    """

    def test_a_two_world_bundle_is_the_same_either_way(self) -> None:
        parallel = explore_adapter.generate(SEED, TWO, FLOOR_PIXELS)
        sequential = explore_adapter.generate(SEED, TWO, FLOOR_PIXELS,
                                              _parallel=False)
        print(f"\n  pool path available: {parallel.parallel}")
        self.assertFalse(sequential.parallel)
        for left, right in zip(parallel.worlds, sequential.worlds):
            for key, value in left.items():
                if key == "seconds":
                    continue
                with self.subTest(key=key):
                    other = right[key]
                    if isinstance(value, np.ndarray):
                        self.assertEqual(value.tobytes(), other.tobytes())
                    elif isinstance(value, list) and value and isinstance(
                            value[0], np.ndarray):
                        for a, b in zip(value, other):
                            self.assertEqual(a.tobytes(), b.tobytes())
                    else:
                        self.assertEqual(value, other)
        for view in explore_adapter.VIEWS:
            with self.subTest(view=view):
                self.assertEqual(explore_adapter.render_png(parallel, view),
                                 explore_adapter.render_png(sequential, view))


if __name__ == "__main__":
    unittest.main()


class ThePoolCapsBlasThreads(unittest.TestCase):
    """Workers get one BLAS thread each.

    The block model solves a dense system every step. Left to itself
    OpenBLAS gives every worker a thread per logical core and spins them
    between calls, and eight workers times twenty-four threads on
    thirty-two cores starved the run of 2026-09-03 to 3.4 cells a minute
    against 14 with the cap: the same 32 worlds took 442 s of wall uncapped
    and 34 s capped. The cap is set in the parent's environment before the
    pool is created, which is what a spawned child inherits.
    """

    def test_the_cap_is_in_the_environment_once_the_pool_exists(self) -> None:
        import os
        pool = explore_adapter._pool()
        if pool is None:
            self.skipTest("no pool in this process")
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                     "MKL_NUM_THREADS"):
            self.assertIn(name, os.environ)
