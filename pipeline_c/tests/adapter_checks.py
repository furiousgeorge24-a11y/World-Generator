"""Checks for the generate-and-view adapter and the quarantine boundary."""

from __future__ import annotations

import ast
from io import BytesIO
from pathlib import Path
import re
import sys
import unittest

import numpy as np
from PIL import Image


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import webui_adapter as adapter  # noqa: E402
from engine import registry  # noqa: E402


SEED = 20260901


class AdapterSurface(unittest.TestCase):
    def test_meta_advertises_only_working_controls(self):
        metadata = adapter.meta()
        self.assertIs(metadata["ready"], True)
        self.assertEqual(metadata["default_size"], 1024)
        self.assertEqual(metadata["supported_sizes"], [512, 1024])
        # Nothing may be advertised that is not implemented; the two author
        # controls return here when C11/C13 give them a causal stage.
        self.assertEqual(metadata["controls"], [])
        self.assertEqual(metadata["views"], list(adapter.VIEWS))

    def test_generate_returns_every_declared_view(self):
        world = adapter.generate(SEED, {}, 1024)
        self.assertEqual(adapter.views(world), list(adapter.VIEWS))
        for name in adapter.views(world):
            blob = adapter.render_png(world, name)
            self.assertTrue(blob.startswith(b"\x89PNG\r\n\x1a\n"))
            with Image.open(BytesIO(blob)) as image:
                self.assertEqual(image.size, (1024, 1024))

    def test_generation_is_deterministic_for_a_seed(self):
        first = adapter.generate(SEED, {}, 1024)
        second = adapter.generate(SEED, {}, 1024)
        self.assertEqual(first.affiliation_sha256, second.affiliation_sha256)
        self.assertEqual(
            adapter.render_png(first, "affiliation"),
            adapter.render_png(second, "affiliation"),
        )

    def test_different_seeds_produce_different_worlds(self):
        first = adapter.generate(SEED, {}, 1024)
        second = adapter.generate(SEED + 1, {}, 1024)
        self.assertNotEqual(first.affiliation_sha256, second.affiliation_sha256)

    def test_affiliation_covers_the_lattice_with_seven_actors(self):
        world = adapter.generate(SEED, {}, 1024)
        self.assertEqual(world.affiliation.shape, (1024, 1024))
        self.assertEqual(sorted(np.unique(world.affiliation)), list(range(7)))
        self.assertAlmostEqual(sum(world.shares), 100.0, places=3)

    def test_size_changes_only_the_rendered_raster(self):
        world = adapter.generate(SEED, {}, 512)
        with Image.open(BytesIO(adapter.render_png(world, "affiliation"))) as image:
            self.assertEqual(image.size, (512, 512))
        # The canonical lattice is fixed; size is a render setting only.
        self.assertEqual(world.affiliation.shape, (1024, 1024))

    def test_invalid_inputs_are_rejected_rather_than_clamped(self):
        with self.assertRaises(ValueError):
            adapter.generate(-1, {}, 1024)
        with self.assertRaises(ValueError):
            adapter.generate(SEED, {}, 777)
        with self.assertRaises(ValueError):
            registry.normalize_controls({"not_a_control": 1})
        world = adapter.generate(SEED, {}, 1024)
        with self.assertRaises(ValueError):
            adapter.render_png(world, "no_such_view")

    def test_report_states_what_the_stage_does_not_contain(self):
        report = adapter.report(adapter.generate(SEED, {}, 1024))
        self.assertEqual(report["seed"], SEED)
        self.assertEqual(report["actors"], 7)
        self.assertIn(
            report["layout_family"],
            {"scatter", "belt", "dual_focus", "arc_void"},
        )
        for absent in ("elevation", "water", "coastline", "land"):
            self.assertIn(absent, report["does_not_contain"])


class Quarantine(unittest.TestCase):
    def test_no_python_file_reaches_another_pipeline(self):
        for path in PIPELINE_ROOT.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
            self.assertFalse(
                [
                    name
                    for name in imported
                    if name.startswith("pipeline_")
                    and not name.startswith("pipeline_c")
                ],
                str(path),
            )
            foreign = {
                match
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                for match in re.findall(r"\bpipeline_[A-Za-z0-9_]+", node.value)
                if match != "pipeline_c"
            }
            self.assertFalse(foreign, str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
