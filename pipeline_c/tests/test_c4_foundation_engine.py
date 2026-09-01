"""Independent C4 gates for the geography-free parent-world foundation."""

from __future__ import annotations

import ast
from fractions import Fraction
import inspect
from pathlib import Path
import random
import struct
import sys
import unittest


PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

from engine.foundation import (  # noqa: E402
    COHORT_MANIFEST,
    COHORT_MANIFEST_SHA256,
    DEVELOPMENT_ANALYSIS_RECT,
    NUMERICAL_RECT,
    PARENT_RECT,
    REGISTERED_PROBES_M,
    SUPPORTED_SIZES,
    PhysicalGrid,
    PhysicalRect,
    SampleAddress,
    StageSampler,
    ValidationAccessError,
    build_development_cohort_state,
    build_formation_context_with_poison_audit,
    build_foundation_state,
    build_identity_bundle,
    changed_cache_keys,
    compute_cache_keys,
    derive_seed,
    exact_nested_ratio,
    production_dependency_graph,
    run_foundation_audits,
    seed_for_execution,
)
from engine.foundation._util import FoundationRecordError  # noqa: E402
from engine.foundation.cohorts import (  # noqa: E402
    DEBUG_SEEDS,
    DEVELOPMENT_SEEDS,
    VALIDATION_SEEDS,
)
from engine.foundation.constants import (  # noqa: E402
    FOUNDATION_STAGE_ID,
    FOUNDATION_STAGE_VERSION,
    KEY_SCHEDULE_ID,
    REGISTRATION_FIELD_ID,
    REGISTRATION_PROCESS_ID,
)


class C4FrozenGeometryAndCohortGates(unittest.TestCase):
    def test_b02_frozen_geometry_and_grid_convention_are_exact(self) -> None:
        self.assertEqual(
            (PARENT_RECT.min_x_m, PARENT_RECT.min_y_m,
             PARENT_RECT.width_m, PARENT_RECT.height_m),
            (0, 0, 40_960_000, 40_960_000),
        )
        self.assertEqual(
            (NUMERICAL_RECT.min_x_m, NUMERICAL_RECT.min_y_m,
             NUMERICAL_RECT.width_m, NUMERICAL_RECT.height_m),
            (-5_120_000, -5_120_000, 51_200_000, 51_200_000),
        )
        self.assertEqual(
            (DEVELOPMENT_ANALYSIS_RECT.min_x_m,
             DEVELOPMENT_ANALYSIS_RECT.min_y_m,
             DEVELOPMENT_ANALYSIS_RECT.width_m,
             DEVELOPMENT_ANALYSIS_RECT.height_m),
            (15_360_000, 15_360_000, 10_240_000, 10_240_000),
        )
        self.assertEqual(SUPPORTED_SIZES, (512, 1024))
        self.assertEqual(
            PARENT_RECT.area_m2,
            16 * DEVELOPMENT_ANALYSIS_RECT.area_m2,
        )
        self.assertEqual(
            (
                PARENT_RECT.min_x_m - NUMERICAL_RECT.min_x_m,
                PARENT_RECT.min_y_m - NUMERICAL_RECT.min_y_m,
                NUMERICAL_RECT.max_x_m - PARENT_RECT.max_x_m,
                NUMERICAL_RECT.max_y_m - PARENT_RECT.max_y_m,
            ),
            (5_120_000,) * 4,
        )
        self.assertTrue(PARENT_RECT.contains_point(0, 0))
        self.assertFalse(PARENT_RECT.contains_point(PARENT_RECT.max_x_m, 0))
        self.assertFalse(PARENT_RECT.contains_point(0, PARENT_RECT.max_y_m))

        coarse = PhysicalGrid(DEVELOPMENT_ANALYSIS_RECT, 512, 512)
        fine = PhysicalGrid(DEVELOPMENT_ANALYSIS_RECT, 1024, 1024)
        self.assertEqual((coarse.cell_width_m, coarse.cell_height_m), (20_000, 20_000))
        self.assertEqual((fine.cell_width_m, fine.cell_height_m), (10_000, 10_000))
        self.assertEqual(exact_nested_ratio(coarse, fine), (2, 2))
        self.assertEqual(
            coarse.cell_center_m(0, 0),
            (Fraction(15_370_000), Fraction(25_590_000)),
        )
        self.assertEqual(
            fine.cell_center_m(0, 0),
            (Fraction(15_365_000), Fraction(25_595_000)),
        )
        self.assertNotEqual(coarse.cell_center_m(0, 0), fine.cell_center_m(0, 0))

        # A coarse cell is exactly the union of the corresponding 2 x 2 fine cells.
        for column, row in ((0, 0), (137, 291), (511, 511)):
            coarse_bounds = coarse.cell_bounds_m(column, row)
            fine_bounds = [
                fine.cell_bounds_m(2 * column + dx, 2 * row + dy)
                for dy in (0, 1)
                for dx in (0, 1)
            ]
            self.assertEqual(min(item.min_x_m for item in fine_bounds), coarse_bounds.min_x_m)
            self.assertEqual(min(item.min_y_m for item in fine_bounds), coarse_bounds.min_y_m)
            self.assertEqual(max(item.max_x_m for item in fine_bounds), coarse_bounds.max_x_m)
            self.assertEqual(max(item.max_y_m for item in fine_bounds), coarse_bounds.max_y_m)

        rectangle_capable = PhysicalGrid(PhysicalRect(10, 20, 600, 200), 3, 1)
        self.assertEqual((rectangle_capable.width_px, rectangle_capable.height_px), (3, 1))

    def test_b02_invalid_or_ambiguous_geometry_fails_closed(self) -> None:
        for args in (
            (0, 0, 0, 1),
            (0, 0, 1, -1),
            (True, 0, 1, 1),
            (2**63 - 1, 0, 2, 1),
        ):
            with self.subTest(args=args), self.assertRaises(FoundationRecordError):
                PhysicalRect(*args)
        with self.assertRaises(FoundationRecordError):
            PhysicalGrid(PhysicalRect(0, 0, 10, 10), 3, 1)
        record = PARENT_RECT.to_record()
        record["max_x_m"] += 1
        with self.assertRaises(FoundationRecordError):
            PhysicalRect.from_record(record)

    def test_b08_cohort_identities_derivation_and_hash_are_literal(self) -> None:
        expected_debug = (2849919200, 3534786579, 3413470326, 1140583516)
        expected_development = (
            2075014389, 2477733044, 476149591, 151640007,
            2697441485, 1504571935, 548870008, 2157195430,
            4108373596, 4287772760, 287488203, 1833546021,
        )
        expected_validation = (
            2791121701, 3130115100, 1455726917, 1973562563,
            1338344000, 1420467597, 4162438652, 1437289707,
            4007147443, 1674537276, 3118573802, 3603308068,
            3071127819, 829628780, 1240803007, 2251729814,
            3107823350, 3992594243, 3806252049, 3159806290,
            1664116445, 3232083246, 3563470347, 3886709965,
            1749459072, 2995727652, 4009585040, 1292709806,
            3305252046, 2145001681, 1578577612, 3129225758,
        )
        self.assertEqual(DEBUG_SEEDS, expected_debug)
        self.assertEqual(DEVELOPMENT_SEEDS, expected_development)
        self.assertEqual(VALIDATION_SEEDS, expected_validation)
        for role, values in (
            ("debug", expected_debug),
            ("development", expected_development),
            ("validation", expected_validation),
        ):
            self.assertEqual(
                tuple(derive_seed(role, index) for index in range(len(values))),
                values,
            )
        self.assertEqual(len(set(expected_debug + expected_development + expected_validation)), 48)
        self.assertEqual(
            COHORT_MANIFEST_SHA256,
            "a97323bceead5f55a6870354256f4279dfd4ca6d939df2728876d7ef3da4382a",
        )
        self.assertEqual(COHORT_MANIFEST.sha256, COHORT_MANIFEST_SHA256)

    def test_b08_validation_guard_fires_before_index_access_or_state_build(self) -> None:
        class PoisonIndex:
            def __getattribute__(self, name):
                raise AssertionError(f"validation index was accessed: {name}")

            def __index__(self):
                raise AssertionError("validation index was converted")

            def __int__(self):
                raise AssertionError("validation index was converted")

        for call in (
            lambda: seed_for_execution("validation", PoisonIndex()),
            lambda: build_foundation_state("validation", PoisonIndex()),
        ):
            with self.assertRaises(ValidationAccessError) as raised:
                call()
            self.assertEqual(
                raised.exception.code,
                "VALIDATION_COHORT_SEALED_UNTIL_C15",
            )

    def test_b08_development_cohort_is_complete_ordered_and_unique(self) -> None:
        cohort = build_development_cohort_state()
        self.assertEqual(len(cohort.states), 12)
        self.assertEqual(
            tuple(state.execution_seed.seed for state in cohort.states),
            DEVELOPMENT_SEEDS,
        )
        self.assertEqual(
            tuple(state.execution_seed.index for state in cohort.states),
            tuple(range(12)),
        )
        self.assertEqual(len({state.canonical_sha256 for state in cohort.states}), 12)


class C4PrfAndResolutionGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.state = build_foundation_state("debug", 0)

    def test_b04_prf_frozen_golden_vector_and_float_conversion(self) -> None:
        sampler = self.state.sampler
        x_m, y_m = REGISTERED_PROBES_M[0]
        self.assertEqual(
            self.state.identities.world_id,
            "6fec2784811c2d4f2f33cd09bae963803039920759e35f90097a0f8f2ee72b9f",
        )
        self.assertEqual(
            sampler.stage_key_sha256,
            "5409736c4bac8eed5a25575385e970633f7835d105d89558c011446fc70e7216",
        )
        self.assertEqual(
            sampler.digest_hex(x_m, y_m),
            "31bd0cce9b2ecfe7c3e142f216ba60403c271491bf4cdae9e703d04208d32269",
        )
        self.assertEqual(sampler.uint64(x_m, y_m), 3584034959963115495)
        self.assertEqual(sampler.unit_float(x_m, y_m), 0.19429092449280028)
        self.assertEqual(
            sampler.unit_float(x_m, y_m),
            (sampler.uint64(x_m, y_m) >> 11) / float(2**53),
        )
        address = sampler.address(x_m, y_m)
        suffix = struct.pack(">q", x_m) + struct.pack(">q", y_m)
        suffix += struct.pack(">I", 0) + struct.pack(">Q", 0)
        self.assertTrue(address.canonical_bytes().endswith(suffix))
        self.assertEqual(address.to_record()["key_schedule_id"], KEY_SCHEDULE_ID)

    def test_b04_prf_is_stateless_order_independent_and_domain_separated(self) -> None:
        sampler = self.state.sampler
        forward = {
            point: sampler.digest_hex(*point) for point in REGISTERED_PROBES_M
        }
        reverse = {
            point: sampler.digest_hex(*point) for point in reversed(REGISTERED_PROBES_M)
        }
        random.seed(998877)
        shuffled = list(REGISTERED_PROBES_M)
        random.shuffle(shuffled)
        interleaved = {}
        for point in shuffled:
            sampler.digest_hex(point[0] + 123, point[1] - 456, channel=9, index=3)
            interleaved[point] = sampler.digest_hex(*point)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward, interleaved)

        x_m, y_m = REGISTERED_PROBES_M[0]
        variants = {
            sampler.digest(x_m, y_m),
            sampler.digest(x_m + 1, y_m),
            sampler.digest(x_m, y_m + 1),
            sampler.digest(x_m, y_m, channel=1),
            sampler.digest(x_m, y_m, index=1),
            StageSampler(
                sampler.world_id,
                "alternate-stage.v1",
                sampler.stage_version,
                sampler.process_id,
            ).digest(x_m, y_m),
            StageSampler(
                sampler.world_id,
                sampler.stage_id,
                "2",
                sampler.process_id,
            ).digest(x_m, y_m),
            StageSampler(
                sampler.world_id,
                sampler.stage_id,
                sampler.stage_version,
                "alternate-process",
            ).digest(x_m, y_m),
            build_foundation_state("debug", 1).sampler.digest(x_m, y_m),
        }
        self.assertEqual(len(variants), 9)

    def test_b04_prf_rejects_noninteger_and_out_of_range_addresses(self) -> None:
        sampler = self.state.sampler
        for x_m, y_m, channel, index in (
            (1.0, 2, 0, 0),
            (True, 2, 0, 0),
            (2**63, 2, 0, 0),
            (1, -(2**63) - 1, 0, 0),
            (1, 2, -1, 0),
            (1, 2, 0, 2**64),
        ):
            with self.subTest((x_m, y_m, channel, index)), self.assertRaises(
                FoundationRecordError
            ):
                sampler.address(x_m, y_m, channel=channel, index=index)

    def test_b05_shared_physical_probes_match_without_cell_center_claim(self) -> None:
        coarse = self.state.sampling_grids[512]
        fine = self.state.sampling_grids[1024]
        self.assertEqual(coarse.rectangle, fine.rectangle)
        self.assertNotEqual(coarse.cell_center_m(0, 0), fine.cell_center_m(0, 0))
        self.assertEqual(len(REGISTERED_PROBES_M), 9)
        self.assertTrue(
            all(DEVELOPMENT_ANALYSIS_RECT.contains_point(*point)
                for point in REGISTERED_PROBES_M)
        )
        by_resolution = {
            size: tuple(self.state.sampler.digest_hex(*point)
                        for point in REGISTERED_PROBES_M)
            for size in SUPPORTED_SIZES
        }
        self.assertEqual(by_resolution[512], by_resolution[1024])


class C4FrameIdentityAndCacheGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.state = build_foundation_state("debug", 0)

    def test_b03_identity_dependencies_are_exact(self) -> None:
        fixed = self.state.identities
        other_seed = build_identity_bundle(DEBUG_SEEDS[1])
        self.assertEqual(other_seed.parent_geometry_id, fixed.parent_geometry_id)
        self.assertNotEqual(other_seed.world_id, fixed.world_id)
        self.assertNotEqual(other_seed.parent_domain_id, fixed.parent_domain_id)
        self.assertNotEqual(other_seed.numerical_domain_id, fixed.numerical_domain_id)

        alternate_numerical_rect = PhysicalRect(
            -6_400_000, -3_840_000, 53_760_000, 51_200_000
        )
        numerical = build_identity_bundle(
            self.state.execution_seed.seed,
            numerical_rectangle=alternate_numerical_rect,
        )
        self.assertEqual(numerical.world_id, fixed.world_id)
        self.assertEqual(numerical.parent_domain_id, fixed.parent_domain_id)
        self.assertNotEqual(numerical.numerical_domain_id, fixed.numerical_domain_id)
        self.assertEqual(
            numerical.development_analysis_window_id,
            fixed.development_analysis_window_id,
        )

        moved = build_identity_bundle(
            self.state.execution_seed.seed,
            analysis_rectangle=DEVELOPMENT_ANALYSIS_RECT.translated(1_280_000, 0),
        )
        absent = build_identity_bundle(
            self.state.execution_seed.seed,
            analysis_rectangle=None,
        )
        for observer in (moved, absent):
            self.assertEqual(observer.world_id, fixed.world_id)
            self.assertEqual(observer.parent_domain_id, fixed.parent_domain_id)
            self.assertEqual(observer.numerical_domain_id, fixed.numerical_domain_id)
        self.assertNotEqual(
            moved.development_analysis_window_id,
            fixed.development_analysis_window_id,
        )
        self.assertIsNone(absent.development_analysis_window_id)

    def test_b07_formation_context_is_exact_and_poison_state_is_untouched(self) -> None:
        expected_keys = {
            "consuming_stage_id", "consuming_stage_version",
            "coordinate_system_id", "foundation_stage_id",
            "foundation_stage_version", "numerical_boundary_policy",
            "numerical_domain_id", "numerical_extent", "parent_domain_id",
            "parent_geometry_id", "rng_key_schedule_id", "schema_id",
            "schema_version", "seed", "units", "upstream_sha256s", "world_id",
        }
        self.assertEqual(set(self.state.formation_context.to_record()), expected_keys)
        self.assertNotIn("development_analysis_window_id", expected_keys)
        self.assertNotIn("target_land_percent", expected_keys)
        self.assertNotIn("landmass_fragmentation", expected_keys)

        class Poison:
            def __getattribute__(self, name):
                raise AssertionError(f"poison was accessed: {name}")

            def __iter__(self):
                raise AssertionError("poison was iterated")

            def __bool__(self):
                raise AssertionError("poison truth value was read")

            def __hash__(self):
                raise AssertionError("poison was hashed")

            def __repr__(self):
                raise AssertionError("poison was represented")

        first = build_formation_context_with_poison_audit(
            self.state,
            observer_state=Poison(),
            frame_state=Poison(),
            selection_state=Poison(),
        )
        second = build_formation_context_with_poison_audit(
            self.state,
            observer_state=Poison(),
            frame_state=Poison(),
            selection_state=Poison(),
        )
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.sha256, self.state.formation_context.sha256)

    def test_b07_dependency_and_import_layers_are_one_way(self) -> None:
        graph = production_dependency_graph()
        by_id = {node.stage_id: node for node in graph}
        formation = by_id[FOUNDATION_STAGE_ID]
        self.assertEqual(formation.stage_kind, "formation")
        self.assertEqual(formation.frame_access, "none")
        self.assertEqual(formation.depends_on, ())
        self.assertTrue(all(
            node.stage_kind != "formation" or node.frame_access == "none"
            for node in graph
        ))

        foundation_root = PIPELINE_C / "engine" / "foundation"
        forbidden_import_tokens = (
            "foundation_review", "lab", "webui", "selection", "delivery"
        )
        for path in foundation_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            bad = sorted(
                name for name in imports
                if any(token in name.casefold() for token in forbidden_import_tokens)
            )
            self.assertEqual(bad, [], str(path))

    def test_b10_cache_invalidation_sets_are_exact(self) -> None:
        base = self.state.cache_keys

        numerical_rect = PhysicalRect(
            -6_400_000, -3_840_000, 53_760_000, 51_200_000
        )
        numerical_ids = build_identity_bundle(
            self.state.execution_seed.seed,
            numerical_rectangle=numerical_rect,
        )
        numerical = compute_cache_keys(
            numerical_ids,
            self.state.sampler,
            self.state.sampling_grids,
            consuming_stage_id=FOUNDATION_STAGE_ID,
            field_id=REGISTRATION_FIELD_ID,
        )
        self.assertEqual(
            set(changed_cache_keys(base, numerical)),
            {"numerical_domain_key", "evidence_key", "render_key"},
        )

        moved_rect = DEVELOPMENT_ANALYSIS_RECT.translated(1_280_000, 0)
        moved_ids = build_identity_bundle(
            self.state.execution_seed.seed, analysis_rectangle=moved_rect
        )
        moved_grids = {
            size: PhysicalGrid(moved_rect, size, size) for size in SUPPORTED_SIZES
        }
        moved = compute_cache_keys(
            moved_ids,
            self.state.sampler,
            moved_grids,
            consuming_stage_id=FOUNDATION_STAGE_ID,
            field_id=REGISTRATION_FIELD_ID,
        )
        self.assertEqual(
            set(changed_cache_keys(base, moved)),
            {
                "observer_window_key", "sampling_grid_keys.512",
                "sampling_grid_keys.1024", "resolution_audit_key",
                "evidence_key", "render_key",
            },
        )

        resolution = compute_cache_keys(
            self.state.identities,
            self.state.sampler,
            {1024: self.state.sampling_grids[1024]},
            consuming_stage_id=FOUNDATION_STAGE_ID,
            field_id=REGISTRATION_FIELD_ID,
        )
        self.assertEqual(
            set(changed_cache_keys(base, resolution)),
            {
                "sampling_grid_keys.512", "resolution_audit_key",
                "evidence_key", "render_key",
            },
        )

        rendered = compute_cache_keys(
            self.state.identities,
            self.state.sampler,
            self.state.sampling_grids,
            consuming_stage_id=FOUNDATION_STAGE_ID,
            field_id=REGISTRATION_FIELD_ID,
            render_settings={"overlay_opacity": 0.5},
        )
        self.assertEqual(
            set(changed_cache_keys(base, rendered)),
            {"render_key", "render_settings"},
        )

    def test_b10_author_controls_are_not_accepted_as_c4_cache_inputs(self) -> None:
        signature = inspect.signature(compute_cache_keys)
        self.assertNotIn("target_land_percent", signature.parameters)
        self.assertNotIn("landmass_fragmentation", signature.parameters)
        self.assertNotIn("development_analysis_window", inspect.signature(StageSampler).parameters)

    def test_b03_to_b10_all_independent_core_audits_pass(self) -> None:
        results = run_foundation_audits(self.state)
        self.assertEqual(
            set(results),
            {
                "c4.address-prf.v1",
                "c4.cache-invalidation.v1",
                "c4.cohorts.v1",
                "c4.geometry.v1",
                "c4.honest-semantics.v1",
                "c4.identity-dependencies.v1",
                "c4.numerical-overlap.v1",
                "c4.observer-isolation.v1",
                "c4.physical-registration.v1",
            },
        )
        failures = {
            audit_id: result.to_record()
            for audit_id, result in results.items()
            if not result.passes
        }
        self.assertEqual(failures, {})
        record = self.state.to_record()
        self.assertEqual(record["material_latent_fields"], [])
        self.assertIs(record["geography_evidence"], False)
        self.assertIs(record["generator_ready"], False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
