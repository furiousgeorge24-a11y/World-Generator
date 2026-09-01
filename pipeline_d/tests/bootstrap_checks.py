"""Run 1 checks: frozen controls, fail-closed adapter, and quarantine."""

from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path
import re
import sys
import unittest


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from engine import EngineUnavailableError, VERSION  # noqa: E402
from engine import registry  # noqa: E402


class BootstrapChecks(unittest.TestCase):
    def test_version_and_readiness_are_explicit(self):
        self.assertEqual(VERSION, "0.1.0-bootstrap")
        metadata = registry.meta()
        self.assertIs(metadata["ready"], False)
        self.assertTrue(metadata["status"].strip())

    def test_only_frozen_controls_are_exposed(self):
        controls = registry.meta()["controls"]
        self.assertEqual(
            [control["name"] for control in controls],
            ["target_land_percent", "landmass_fragmentation"],
        )
        expected = {
            "target_land_percent": (35.0, 0.0, 70.0),
            "landmass_fragmentation": (0.5, 0.0, 1.0),
        }
        for control in controls:
            self.assertEqual(
                (control["default"], control["lo"], control["hi"]),
                expected[control["name"]],
            )
            self.assertEqual(control["invalidates"], "full")
            self.assertEqual(control["tier"], "primary")
            self.assertTrue(control["promise"].strip())

    def test_normalization_is_complete_and_strict(self):
        self.assertEqual(
            registry.normalize_controls(),
            {"target_land_percent": 35.0, "landmass_fragmentation": 0.5},
        )
        self.assertEqual(
            registry.normalize_controls(
                {"target_land_percent": "70", "landmass_fragmentation": 0}
            ),
            {"target_land_percent": 70.0, "landmass_fragmentation": 0.0},
        )
        bad = (
            [],
            {1: 0.5},
            {"target_land_percent": -0.01},
            {"target_land_percent": 70.01},
            {"landmass_fragmentation": math.nan},
            {"landmass_fragmentation": True},
            {"not_a_control": 1},
        )
        for controls in bad:
            with self.subTest(controls=controls), self.assertRaises(
                (TypeError, ValueError)
            ):
                registry.normalize_controls(controls)

    def test_required_adapter_surface_fails_closed(self):
        adapter = importlib.import_module("webui_adapter")
        self.assertIs(adapter.meta()["ready"], False)
        calls = (
            lambda: adapter.generate(7, {}, 256),
            lambda: adapter.views(None),
            lambda: adapter.render_png(None, "hypsometric"),
            lambda: adapter.report(None),
        )
        before = {
            path.relative_to(PIPELINE_ROOT)
            for path in PIPELINE_ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        for call in calls:
            with self.assertRaisesRegex(EngineUnavailableError, "no land-origin"):
                call()
        after = {
            path.relative_to(PIPELINE_ROOT)
            for path in PIPELINE_ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        self.assertEqual(after, before)

    def test_engine_directory_contains_no_generator(self):
        modules = {
            path.name
            for path in (PIPELINE_ROOT / "engine").glob("*.py")
        }
        self.assertEqual(modules, {"__init__.py", "registry.py"})

    def test_python_runtime_has_no_cross_pipeline_dependency(self):
        for path in PIPELINE_ROOT.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
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
            foreign_runtime_names = {
                match
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                for match in re.findall(r"\bpipeline_[A-Za-z0-9_]+", node.value)
                if match != "pipeline_c"
            }
            self.assertFalse(foreign_runtime_names, str(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
