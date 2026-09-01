"""Focused checks for causal, frame-independent border acceptance.

The historical contour measurements remain useful diagnostics, but they
cannot veto a crop whose generated outer ring is water.  Full experiment
promotion is deliberately separate: a recovered border does not excuse a
failed nested/shifted process-domain independence check.

Run from ``pipeline_b`` with::

    py -3.14 tests/causal_border_acceptance_checks.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
FIXTURE_PATH = (
    ROOT / "tests" / "fixtures" / "recovered_seed11_causal_border.json"
)
sys.path.insert(0, str(ROOT))

from spikes.causal_border_acceptance import (  # noqa: E402
    evaluate_causal_border,
    evaluate_promotion,
)


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class CausalBorderAcceptanceChecks(unittest.TestCase):
    def test_all_water_outer_ring_passes_at_any_negative_depth(self):
        for depth_m in (-0.001, -1.0, -5250.0):
            elevation_m = np.full((4, 6), depth_m, dtype=np.float64)
            result = evaluate_causal_border(elevation_m < 0.0)
            self.assertTrue(result["passed"])
            self.assertEqual(result["outer_ring_non_water_cell_count"], 0)

    def test_one_land_cell_on_outer_ring_fails(self):
        water = np.ones((4, 6), dtype=bool)
        water[0, 3] = False

        result = evaluate_causal_border(water)

        self.assertFalse(result["passed"])
        self.assertEqual(result["outer_ring_non_water_cell_count"], 1)
        self.assertEqual(
            result["outer_ring_non_water_coordinates_row_column"], [[0, 3]])

    def test_interior_land_is_allowed(self):
        water = np.ones((5, 7), dtype=bool)
        water[1:-1, 1:-1] = False

        result = evaluate_causal_border(water)

        self.assertTrue(result["passed"])
        self.assertEqual(result["outer_ring_non_water_cell_count"], 0)

    def test_contour_diagnostic_cannot_change_causal_acceptance(self):
        water = np.ones((4, 6), dtype=bool)
        passing_diagnostic = {"passed": True, "max_parallel_span_km": 0.0}
        failing_diagnostic = {
            "passed": False,
            "max_parallel_span_km": 1081.0343058376836,
        }

        passing = evaluate_causal_border(
            water, contour_diagnostic=passing_diagnostic)
        failing = evaluate_causal_border(
            water, contour_diagnostic=failing_diagnostic)

        self.assertTrue(passing["passed"])
        self.assertTrue(failing["passed"])
        self.assertEqual(
            passing["outer_ring_non_water_cell_count"],
            failing["outer_ring_non_water_cell_count"],
        )
        self.assertEqual(
            failing["contour_diagnostic"], failing_diagnostic)
        self.assertFalse(failing["contour_diagnostic_affects_passed"])

    def test_rectangular_perimeter_counts_each_cell_once(self):
        for shape, expected_count in (
            ((1, 1), 1),
            ((1, 5), 5),
            ((4, 1), 4),
            ((2, 2), 4),
            ((3, 5), 12),
        ):
            with self.subTest(shape=shape):
                result = evaluate_causal_border(
                    np.ones(shape, dtype=bool))
                self.assertEqual(
                    result["outer_ring_cell_count"], expected_count)
                self.assertEqual(
                    result["outer_ring_water_cell_count"], expected_count)
                self.assertEqual(
                    result["outer_ring_non_water_cell_count"], 0)

    def test_non_2d_and_empty_masks_are_rejected(self):
        invalid = (
            np.ones(5, dtype=bool),
            np.ones((2, 3, 4), dtype=bool),
            np.ones((0, 4), dtype=bool),
            np.ones((4, 0), dtype=bool),
        )
        for mask in invalid:
            with self.subTest(shape=mask.shape):
                with self.assertRaises(ValueError):
                    evaluate_causal_border(mask)

    def test_numeric_elevation_cannot_be_mistaken_for_a_water_mask(self):
        with self.assertRaises(TypeError):
            evaluate_causal_border(np.full((3, 5), -1.0))

    def test_promotion_requires_all_three_independent_facts(self):
        for border, frame_independent, domain_independent, expected in (
            (True, True, True, True),
            (False, True, True, False),
            (True, False, True, False),
            (True, True, False, False),
        ):
            with self.subTest(
                border=border,
                frame_independent=frame_independent,
                domain_independent=domain_independent,
            ):
                result = evaluate_promotion(
                    border_passed=border,
                    frame_independent_generation=frame_independent,
                    process_domain_independent=domain_independent,
                )
                self.assertEqual(result["passed"], expected)

    def test_recovered_seed11_border_passes_but_promotion_does_not(self):
        fixture = _fixture()
        candidate = fixture["candidate"]
        contour = fixture["obsolete_contour_diagnostic"]
        generation = fixture["generation_causality"]
        process = fixture["process_domain_independence"]
        expected = fixture["expected"]

        border_flags = [
            item["ocean_edge_band_passed"]
            for item in candidate["rendered_border"].values()
        ]
        self.assertEqual(
            all(border_flags), expected["border_evidence_passed"])
        land_evidence_passed = (
            0.20 <= candidate["structural_land_fraction"] < 0.50)
        self.assertEqual(
            land_evidence_passed, expected["land_evidence_passed"])
        self.assertFalse(contour["visible_contour_gate_passed"])
        self.assertFalse(contour["gating"])
        self.assertEqual(
            not contour["gating"],
            expected["contour_diagnostic_is_non_gating"],
        )
        self.assertEqual(
            generation["frame_independent_generation"],
            expected["frame_independent_generation"],
        )

        # The fixed report records aggregate outer-ring evidence rather than
        # a million-cell mask.  This compact representative mask has the same
        # causal facts: unique water perimeter and permitted interior land.
        water = np.ones((3, 5), dtype=bool)
        water[1, 1:-1] = False
        border = evaluate_causal_border(
            water, contour_diagnostic=contour)
        self.assertEqual(border["passed"], expected["border_evidence_passed"])

        convergence_flags = list(process["render_convergence"].values())
        process_independent = all((
            process["nested_small_vs_large_passed"],
            process["shifted_vs_large_passed"],
            *convergence_flags,
        ))
        self.assertEqual(
            process_independent, expected["process_domain_independent"])
        promotion = evaluate_promotion(
            border_passed=border["passed"],
            frame_independent_generation=(
                generation["frame_independent_generation"]
                and not generation["terrain_edited_tapered_or_masked_to_frame"]
            ),
            process_domain_independent=process_independent,
        )
        self.assertEqual(
            promotion["passed"], expected["full_promotion_passed"])
        self.assertTrue(promotion["border_passed"])
        self.assertTrue(promotion["frame_independent_generation"])
        self.assertFalse(promotion["process_domain_independent"])

    def test_fixture_preserves_source_report_hash(self):
        fixture = _fixture()
        source = fixture["source_report"]
        expected_sha256 = (
            "9fcb7741f42b5399ead3931c93164ff9dd50f87f2c57c4c448b31bf18d82c12d"
        )
        self.assertEqual(source["sha256"], expected_sha256)

        # The regression fixture is self-contained when historical ``out``
        # artifacts are absent.  When its source report is available, guard
        # the provenance link against accidental drift as well.
        source_path = WORKSPACE / source["path"]
        if source_path.exists():
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
            self.assertEqual(digest, expected_sha256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
