"""Focused C02 engine contracts using one cached development world.

The complete twelve-member cohort is exercised separately.  This module pays
for one normal formation and one maximally reversed replay, then shares those
immutable states across all tests.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np


PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

from engine.foundation import (  # noqa: E402
    DEVELOPMENT_ANALYSIS_RECT,
    PARENT_RECT,
    PhysicalGrid,
    REGISTERED_PROBES_M,
    build_foundation_state,
)
from engine.tectonic_fabric_c02.constants import (  # noqa: E402
    ATTEMPT_ID,
    CANONICAL_CELL_M,
    CANONICAL_SIZE,
    CASE_ID,
    COMPARISON_FAMILY_ID,
    CROWDING_BONUS_PER_CELL,
    CROWDING_TARGET_DISTANCE_M,
    DISPLAY_LABEL,
    EVIDENCE_KIND,
    FAMILY_MINIMUM_SEPARATION_M,
    FROZEN_DEVELOPMENT_FAMILY_IDS,
    GERM_CELL_COUNT,
    MAX_ACTOR_AREA_PERCENT,
    MAX_COMPACTNESS_PENALTY,
    MAX_CONTACT_PAIR_COUNT,
    MAX_HIERARCHY,
    MIN_ACTOR_AREA_PERCENT,
    MIN_ACTOR_MEAN_ENDPOINT_AGREEMENT,
    MIN_BELT_ASPECT_RATIO,
    MIN_CONTACT_PAIR_COUNT,
    MIN_EROSION_RETENTION,
    MIN_HIERARCHY_DENOMINATOR,
    MIN_HIERARCHY_NUMERATOR,
    MIN_LARGEST_ACTOR_PERCENT,
    MIN_NUCLEUS_NEIGHBOR_CV,
    MIN_TOTAL_MEAN_ENDPOINT_AGREEMENT,
    PARENT_SIDE_M,
    PRIMARY_ACTOR_COUNT,
    REPRESENTATION_ID,
    ROADMAP_RUN,
    STAGE_ID,
    STAGE_VERSION,
)
from engine.tectonic_fabric_c02.cache import (  # noqa: E402
    changed_fabric_cache_keys,
    compute_fabric_cache_keys,
)
from engine.tectonic_fabric_c02.construction import (  # noqa: E402
    build_tectonic_fabric_state,
)
from engine.tectonic_fabric_c02.context import (  # noqa: E402
    build_fabric_context_with_poison_audit,
)
from engine.tectonic_fabric_c02.growth import (  # noqa: E402
    clear_growth_caches,
    resistance_array,
)
from engine.tectonic_fabric_c02.layout import (  # noqa: E402
    signed_periodic_delta,
    triangle_wave,
)
from engine.tectonic_fabric_c02.metrics import (  # noqa: E402
    measure_census,
    measure_morphology,
)
from engine.tectonic_fabric_c02.observation import (  # noqa: E402
    observe_analysis,
    observe_canonical_affiliation,
    observe_parent_census,
)
from engine.tectonic_fabric_c02.records import (  # noqa: E402
    actor_lineage_id,
    frozen_c02_spec,
)
from engine.tectonic_fabric_c02.topology import (  # noqa: E402
    canonical_cell_indices,
    endpoint_agreement_count,
    owner_slot,
    owner_slots,
    strict_all_eight,
)


class _Poison:
    def __getattribute__(self, name: str):
        raise AssertionError(f"poison value was accessed: {name}")

    def __iter__(self):
        raise AssertionError("poison value was iterated")

    def __bool__(self):
        raise AssertionError("poison truth value was read")


def _toroidal_cardinal_neighbors(
    children: np.ndarray, parents: np.ndarray
) -> np.ndarray:
    child_rows, child_columns = divmod(children, CANONICAL_SIZE)
    parent_rows, parent_columns = divmod(parents, CANONICAL_SIZE)
    row_distance = np.minimum(
        np.abs(child_rows - parent_rows),
        CANONICAL_SIZE - np.abs(child_rows - parent_rows),
    )
    column_distance = np.minimum(
        np.abs(child_columns - parent_columns),
        CANONICAL_SIZE - np.abs(child_columns - parent_columns),
    )
    return row_distance + column_distance == 1


class C5C02RepresentativeEngineGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.foundation = build_foundation_state("development", 0)
        arguments = (
            cls.foundation.execution_seed,
            cls.foundation.identities,
            cls.foundation.spec.parent_rectangle,
        )
        cls.state = build_tectonic_fabric_state(*arguments)
        cls.reversed_state = build_tectonic_fabric_state(
            *arguments,
            candidate_order="reverse",
            source_order="reverse",
            neighbor_order="reverse",
        )
        cls.canonical = observe_canonical_affiliation(cls.state)
        cls.parent = observe_parent_census(cls.state)
        cls.analysis_512 = observe_analysis(cls.state, 512)
        cls.analysis_1024 = observe_analysis(cls.state, 1024)
        cls.census = measure_census(cls.state, cls.canonical)
        cls.morphology = measure_morphology(
            cls.state, cls.canonical, cls.census
        )

    def test_e01_exact_identity_specification_and_scope(self) -> None:
        self.assertEqual(
            (
                ROADMAP_RUN,
                ATTEMPT_ID,
                STAGE_ID,
                STAGE_VERSION,
                COMPARISON_FAMILY_ID,
                EVIDENCE_KIND,
                CASE_ID,
                REPRESENTATION_ID,
            ),
            (
                "C5",
                "C02",
                "tectonic_fabric.v2",
                "2",
                "c5-initial-tectonic-fabric-v1",
                "engine_tectonic_fabric",
                "c5-c02-development-cohort-v1",
                "connected-competitive-growth-affiliation-v2",
            ),
        )
        spec = frozen_c02_spec()
        self.assertTrue(spec.is_frozen_c02)
        self.assertEqual(spec.actor_count, 7)
        self.assertEqual(spec.canonical_size, 1024)
        self.assertEqual(spec.canonical_cell_m, 40_000)
        self.assertEqual(spec.candidates_per_actor, 512)
        self.assertEqual(spec.germ_half_steps, 16)
        record = self.state.to_record()
        self.assertIs(record["ready"], False)
        self.assertEqual(record["attempt_id"], "C02")
        self.assertNotIn("target_land_percent", str(record))
        self.assertNotIn("landmass_fragmentation", str(record))
        self.assertIn("NO KINEMATICS", DISPLAY_LABEL)

    def test_e02_context_ignores_all_observer_render_and_author_poison(self) -> None:
        context = build_fabric_context_with_poison_audit(
            self.foundation.execution_seed.seed,
            self.foundation.identities,
            PARENT_RECT,
            numerical_state=_Poison(),
            observer_state=_Poison(),
            frame_state=_Poison(),
            control_state=_Poison(),
            render_state=_Poison(),
            target_state=_Poison(),
            fragmentation_state=_Poison(),
        )
        self.assertEqual(context, self.state.context)
        self.assertNotIn("window", str(context.to_record()).casefold())
        self.assertNotIn("observer", str(context.to_record()).casefold())

    def test_e03_family_layout_nuclei_germs_and_lineages_are_exact(self) -> None:
        state = self.state
        self.assertEqual(state.family_id, FROZEN_DEVELOPMENT_FAMILY_IDS[0])
        self.assertEqual(tuple(actor.slot for actor in state.actors), tuple(range(7)))
        self.assertEqual(sorted(actor.tie_rank for actor in state.actors), list(range(7)))
        self.assertEqual(
            len({(actor.nucleus_x_m, actor.nucleus_y_m) for actor in state.actors}),
            7,
        )
        all_germs = [cell for actor in state.actors for cell in actor.germ_flat_indices]
        self.assertEqual(len(all_germs), 7 * GERM_CELL_COUNT)
        self.assertEqual(len(set(all_germs)), len(all_germs))
        minimum_separation = FAMILY_MINIMUM_SEPARATION_M[state.family_id]
        for actor in state.actors:
            self.assertGreaterEqual(actor.nearest_nucleus_distance_m, minimum_separation)
            self.assertIn(actor.global_axis, {0, 1})
            self.assertIn(actor.preferred_sign, {-1, 1})
            self.assertEqual(len(actor.germ_flat_indices), GERM_CELL_COUNT)
            for left, right in zip(
                actor.germ_flat_indices, actor.germ_flat_indices[1:]
            ):
                self.assertTrue(
                    bool(
                        _toroidal_cardinal_neighbors(
                            np.asarray([right]), np.asarray([left])
                        )[0]
                    )
                )
            expected_bonus = CROWDING_BONUS_PER_CELL * (
                max(
                    0,
                    CROWDING_TARGET_DISTANCE_M
                    - actor.nearest_nucleus_distance_m,
                )
                // CANONICAL_CELL_M
            )
            self.assertEqual(actor.crowding_bonus, expected_bonus)
            self.assertEqual(actor.initial_arrival, -expected_bonus)
            self.assertEqual(
                actor.lineage_id,
                actor_lineage_id(state.context.world_id, actor.slot),
            )

    def test_e04_reversed_candidate_source_and_neighbor_orders_replay_exactly(self) -> None:
        first = self.state
        replay = self.reversed_state
        self.assertEqual(first.to_record(), replay.to_record())
        self.assertEqual(first.affiliation_bytes, replay.affiliation_bytes)
        self.assertEqual(first.arrival_times_bytes, replay.arrival_times_bytes)
        self.assertEqual(first.parent_indices_bytes, replay.parent_indices_bytes)
        self.assertEqual(first.source_mask_packed_bytes, replay.source_mask_packed_bytes)
        self.assertEqual(first.canonical_sha256, replay.canonical_sha256)

    def test_e05_growth_certificate_and_parent_induction_are_independent(self) -> None:
        state = self.state
        self.assertTrue(state.certificate.passes)
        owners = state.slots_array().ravel()
        arrivals = state.arrival_array().ravel()
        parents = state.parent_array().ravel().astype(np.int64, copy=False)
        sources = state.source_mask_array().ravel()
        indices = np.arange(owners.size, dtype=np.int64)
        self.assertEqual(set(np.unique(owners).tolist()), set(range(7)))
        self.assertTrue(np.all(parents[sources] == indices[sources]))
        children = indices[~sources]
        predecessors = parents[~sources]
        self.assertTrue(np.all(predecessors >= 0))
        self.assertTrue(np.all(predecessors < owners.size))
        self.assertTrue(np.all(owners[children] == owners[predecessors]))
        self.assertTrue(np.all(arrivals[predecessors] < arrivals[children]))
        self.assertTrue(
            np.all(_toroidal_cardinal_neighbors(children, predecessors))
        )
        self.assertEqual(self.census.toroidal_component_counts, (1,) * 7)
        self.assertTrue(self.census.contact_graph_connected)

    def test_e06_registered_physical_readout_is_scalar_vector_and_traversal_stable(self) -> None:
        probe_x = np.asarray([item[0] for item in REGISTERED_PROBES_M], dtype=np.int64)
        probe_y = np.asarray([item[1] for item in REGISTERED_PROBES_M], dtype=np.int64)
        scalar = [owner_slot(self.state, x, y) for x, y in REGISTERED_PROBES_M]
        vector = owner_slots(self.state, probe_x, probe_y).tolist()
        self.assertEqual(scalar, vector)
        self.assertEqual(
            owner_slot(self.state, -1, -1),
            owner_slot(self.state, PARENT_SIDE_M - 1, PARENT_SIDE_M - 1),
        )
        self.assertEqual(canonical_cell_indices(0, 0), (0, 0))
        self.assertEqual(canonical_cell_indices(-1, -1), (1023, 1023))
        reverse_parent = observe_parent_census(
            self.state, traversal="reverse", chunk_rows=37
        )
        reverse_analysis = observe_analysis(
            self.state, 512, traversal="reverse", chunk_rows=29
        )
        self.assertEqual(reverse_parent.actor_slots_bytes, self.parent.actor_slots_bytes)
        self.assertEqual(
            reverse_analysis.actor_slots_bytes,
            self.analysis_512.actor_slots_bytes,
        )

    def test_e07_observations_are_independent_reads_of_one_canonical_field(self) -> None:
        self.assertEqual(
            self.canonical.actor_slots_bytes, self.state.affiliation_bytes
        )
        self.assertTrue(np.all(
            self.canonical.strict_all_eight_array()
            == (self.canonical.endpoint_agreement_array() == 8)
        ))
        for x_m, y_m in REGISTERED_PROBES_M:
            count = endpoint_agreement_count(self.state, x_m, y_m)
            self.assertEqual(strict_all_eight(self.state, x_m, y_m), count == 8)
        for observation in (
            self.parent,
            self.analysis_512,
            self.analysis_1024,
        ):
            grid = observation.grid
            x_m = (
                grid.rectangle.min_x_m
                + np.arange(grid.width_px, dtype=np.int64) * grid.cell_width_m
                + grid.cell_width_m // 2
            )
            y_m = (
                grid.rectangle.max_y_m
                - np.arange(grid.height_px, dtype=np.int64) * grid.cell_height_m
                - grid.cell_height_m // 2
            )
            independent = owner_slots(
                self.state, x_m[np.newaxis, :], y_m[:, np.newaxis]
            )
            self.assertTrue(np.array_equal(observation.slots_array(), independent))
            self.assertEqual(observation.source_state_sha256, self.state.canonical_sha256)

    def test_e08_representative_world_passes_frozen_hierarchy_shape_and_stability(self) -> None:
        census = self.census
        morphology = self.morphology
        total = census.total_cell_count
        self.assertTrue(all(
            MIN_ACTOR_AREA_PERCENT * total <= 100 * count
            <= MAX_ACTOR_AREA_PERCENT * total
            for count in census.actor_cell_counts
        ))
        self.assertGreaterEqual(
            100 * max(census.actor_cell_counts),
            MIN_LARGEST_ACTOR_PERCENT * total,
        )
        self.assertGreaterEqual(
            max(census.actor_cell_counts) * MIN_HIERARCHY_DENOMINATOR,
            min(census.actor_cell_counts) * MIN_HIERARCHY_NUMERATOR,
        )
        self.assertLessEqual(census.hierarchy_ratio, MAX_HIERARCHY)
        self.assertGreaterEqual(len(census.observed_contact_pairs), MIN_CONTACT_PAIR_COUNT)
        self.assertLessEqual(len(census.observed_contact_pairs), MAX_CONTACT_PAIR_COUNT)
        self.assertGreaterEqual(
            morphology.nucleus_nearest_neighbor_cv,
            MIN_NUCLEUS_NEIGHBOR_CV,
        )
        self.assertGreaterEqual(
            max(item.aspect_ratio for item in morphology.actor_metrics),
            MIN_BELT_ASPECT_RATIO,
        )
        self.assertTrue(all(
            item.compactness_penalty <= MAX_COMPACTNESS_PENALTY
            and item.erosion_retention >= MIN_EROSION_RETENTION
            and item.mean_endpoint_agreement >= MIN_ACTOR_MEAN_ENDPOINT_AGREEMENT
            for item in morphology.actor_metrics
        ))
        self.assertGreaterEqual(
            morphology.total_mean_endpoint_agreement,
            MIN_TOTAL_MEAN_ENDPOINT_AGREEMENT,
        )

    def test_e09_resistance_cache_is_exactly_warm_cold_invariant(self) -> None:
        world_id = self.state.context.world_id
        clear_growth_caches()
        cold = resistance_array(world_id).tobytes(order="C")
        warm = resistance_array(world_id).tobytes(order="C")
        self.assertEqual(cold, warm)
        self.assertEqual(
            hashlib.sha256(cold).hexdigest(),
            self.state.certificate.resistance_sha256,
        )

    def test_e10_frozen_integer_periodic_primitives_cover_half_period_ties(self) -> None:
        self.assertEqual(signed_periodic_delta(0), 0)
        self.assertEqual(signed_periodic_delta(PARENT_SIDE_M // 2), -PARENT_SIDE_M // 2)
        self.assertEqual(signed_periodic_delta(PARENT_SIDE_M), 0)
        self.assertEqual(triangle_wave(0), -PARENT_SIDE_M)
        self.assertEqual(triangle_wave(PARENT_SIDE_M // 2), PARENT_SIDE_M)
        self.assertEqual(triangle_wave(PARENT_SIDE_M), -PARENT_SIDE_M)

    def test_e11_observer_resolution_and_render_cache_changes_stay_downstream(self) -> None:
        grids = {
            "analysis_512": self.analysis_512.grid,
            "analysis_1024": self.analysis_1024.grid,
            "parent_census": self.parent.grid,
        }
        base = compute_fabric_cache_keys(self.state, grids)
        rendered = compute_fabric_cache_keys(
            self.state,
            grids,
            render_settings={"tectonic_overlay_opacity": 0.5},
        )
        reduced = compute_fabric_cache_keys(
            self.state,
            {key: value for key, value in grids.items() if key != "analysis_1024"},
        )
        moved = compute_fabric_cache_keys(
            self.state,
            {
                **grids,
                "analysis_512": PhysicalGrid(
                    DEVELOPMENT_ANALYSIS_RECT.translated(1_280_000, 0),
                    512,
                    512,
                ),
            },
        )
        semantic = set(base.semantic_record())
        render_changes = set(changed_fabric_cache_keys(base, rendered))
        resolution_changes = set(changed_fabric_cache_keys(base, reduced))
        observer_changes = set(changed_fabric_cache_keys(base, moved))
        self.assertEqual(render_changes, {"render_key", "render_settings"})
        self.assertFalse(semantic & resolution_changes)
        self.assertFalse(semantic & observer_changes)
        self.assertIn("observation_keys.analysis_1024", resolution_changes)
        self.assertIn("observation_keys.analysis_512", observer_changes)

    def test_e12_resistance_and_arrival_diagnostics_close_exact_engine_arrays(self) -> None:
        controls = self.state.resistance_controls
        preview = controls.preview_array()
        self.assertEqual(preview.shape, (128, 128))
        self.assertEqual(preview.dtype, np.dtype("<u2"))
        self.assertFalse(preview.flags.writeable)
        self.assertEqual(
            controls.to_record()["preview"]["sha256"],
            hashlib.sha256(controls.preview_bytes).hexdigest(),
        )
        self.assertEqual(
            controls.resistance_sha256,
            self.state.certificate.resistance_sha256,
        )

        summary = self.state.arrival_summary
        arrivals = self.state.arrival_array().ravel()
        owners = self.state.slots_array().ravel()
        self.assertEqual(summary.total_cell_count, arrivals.size)
        self.assertEqual(summary.minimum_arrival, int(arrivals.min()))
        self.assertEqual(summary.maximum_arrival, int(arrivals.max()))
        self.assertEqual(summary.arrival_sum, int(arrivals.sum(dtype=np.int64)))
        self.assertEqual(
            summary.arrival_times_sha256,
            hashlib.sha256(self.state.arrival_times_bytes).hexdigest(),
        )
        for actor in summary.actor_summaries:
            values = arrivals[owners == actor.slot]
            self.assertEqual(actor.cell_count, values.size)
            self.assertEqual(actor.minimum_arrival, int(values.min()))
            self.assertEqual(actor.maximum_arrival, int(values.max()))
            self.assertEqual(actor.arrival_sum, int(values.sum(dtype=np.int64)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
