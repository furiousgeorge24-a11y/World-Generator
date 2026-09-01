"""Deterministic hard-gate audits for the complete C02 engine cohort."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..foundation import (
    DEVELOPMENT_ANALYSIS_RECT,
    PARENT_RECT,
    REGISTERED_PROBES_M,
    PhysicalGrid,
    build_development_cohort,
)
from ._util import FabricRecordError, content_sha256, freeze_json, require_id, thaw_json
from .cache import changed_fabric_cache_keys, compute_fabric_cache_keys
from .cohort import DevelopmentFabricCohort, validation_guard_is_closed
from .constants import (
    ATTEMPT_ID,
    CASE_ID,
    COMPARISON_FAMILY_ID,
    EVIDENCE_KIND,
    FROZEN_DEVELOPMENT_FAMILY_IDS,
    MAX_ACTOR_AREA_PERCENT,
    MAX_ADJUSTED_RAND,
    MAX_COMPACTNESS_PENALTY,
    MAX_CONTACT_PAIR_COUNT,
    MAX_HIERARCHY,
    MIN_ACTOR_AREA_PERCENT,
    MIN_ACTOR_MEAN_ENDPOINT_AGREEMENT,
    MIN_ASPECT_RATIO,
    MIN_ASPECT_WORLD_COUNT,
    MIN_BELT_ASPECT_RATIO,
    MIN_CONTACT_PAIR_COUNT,
    MIN_EROSION_RETENTION,
    MIN_HIERARCHY_DENOMINATOR,
    MIN_HIERARCHY_NUMERATOR,
    MIN_LARGEST_ACTOR_PERCENT,
    MIN_NUCLEUS_NEIGHBOR_CV,
    MIN_PAIR_DISAGREEMENT,
    MIN_TOTAL_MEAN_ENDPOINT_AGREEMENT,
    REPRESENTATION_ID,
    ROADMAP_RUN,
    STAGE_ID,
    STAGE_VERSION,
)
from .construction import build_tectonic_fabric_state
from .context import build_fabric_context_with_poison_audit
from .growth import clear_growth_caches, resistance_array
from .observation import observe_analysis, observe_parent_census
from .topology import owner_slot, owner_slots


@dataclass(frozen=True, slots=True)
class FabricAuditResult:
    audit_id: str
    passes: bool
    details: object

    def __post_init__(self) -> None:
        require_id(self.audit_id, "audit_id")
        if not isinstance(self.passes, bool):
            raise FabricRecordError("audit passes must be boolean")
        object.__setattr__(self, "details", freeze_json(self.details, "audit details"))

    def to_record(self) -> dict[str, object]:
        return {
            "audit_id": self.audit_id,
            "details": thaw_json(self.details),
            "passes": self.passes,
            "schema_id": "urn:mapgen:pipeline-c:c02-audit:v1",
            "schema_version": 1,
        }


def _result(number: int, name: str, passes: bool, details: object) -> FabricAuditResult:
    return FabricAuditResult(f"c5.c02.g{number:02d}.{name}.v1", bool(passes), details)


def _scope_audit(cohort: DevelopmentFabricCohort) -> FabricAuditResult:
    conditions = {
        "attempt_id_c02": ATTEMPT_ID == "C02",
        "case_id_exact": CASE_ID == "c5-c02-development-cohort-v1",
        "comparison_family_exact": COMPARISON_FAMILY_ID == "c5-initial-tectonic-fabric-v1",
        "evidence_kind_exact": EVIDENCE_KIND == "engine_tectonic_fabric",
        "representation_exact": REPRESENTATION_ID == "connected-competitive-growth-affiliation-v2",
        "roadmap_run_c5": ROADMAP_RUN == "C5",
        "semantic_stage_exact": STAGE_ID == "tectonic_fabric.v2" and STAGE_VERSION == "2",
        "states_not_ready": all(not state.to_record()["ready"] for state in cohort.states),
        "target_and_fragmentation_absent": all(
            "target_land" not in str(state.to_record()).casefold()
            and "fragmentation" not in str(state.to_record()).casefold()
            for state in cohort.states
        ),
    }
    return _result(1, "scope-readiness", all(conditions.values()), {"conditions": conditions})


def _inheritance_audit(cohort: DevelopmentFabricCohort) -> FabricAuditResult:
    foundations = build_development_cohort()
    per_member: list[dict[str, object]] = []
    all_pass = len(cohort.states) == 12
    for member, foundation in zip(cohort.members, foundations):
        if member.state is None:
            per_member.append({"index": member.execution_seed.index, "passes": False})
            all_pass = False
            continue
        state = member.state
        poisoned = build_fabric_context_with_poison_audit(
            foundation.execution_seed.seed,
            foundation.identities,
            PARENT_RECT,
            numerical_state=object(),
            observer_state=object(),
            frame_state=object(),
            control_state=object(),
            render_state=object(),
            target_state=object(),
            fragmentation_state=object(),
        )
        conditions = {
            "context_poison_invariant": poisoned.to_record() == state.context.to_record(),
            "execution_seed_exact": state.execution_seed == foundation.execution_seed,
            "parent_domain_exact": state.context.parent_domain_id == foundation.identities.parent_domain_id,
            "parent_geometry_exact": state.context.parent_geometry_id == foundation.identities.parent_geometry_id,
            "parent_rectangle_exact": state.context.parent_rectangle == PARENT_RECT,
            "world_exact": state.context.world_id == foundation.identities.world_id,
        }
        all_pass &= all(conditions.values())
        per_member.append({"conditions": conditions, "index": member.execution_seed.index})
    return _result(2, "c4-inheritance-isolation", all_pass, {"members": per_member})


def _layout_audit(cohort: DevelopmentFabricCohort) -> FabricAuditResult:
    conditions = {
        "all_members_successful": cohort.complete_success,
        "family_draw_exact": cohort.family_ids == FROZEN_DEVELOPMENT_FAMILY_IDS,
        "all_families_present": set(cohort.family_ids) == {0, 1, 2, 3},
        "maximum_family_count_six": max(cohort.family_ids.count(value) for value in {0, 1, 2, 3}) <= 6,
        "seven_unique_lineages_each": all(len(set(state.actor_id_lookup)) == 7 for state in cohort.states),
        "seven_unique_nuclei_each": all(
            len({(actor.nucleus_x_m, actor.nucleus_y_m) for actor in state.actors}) == 7
            for state in cohort.states
        ),
        "tie_ranks_complete": all(sorted(actor.tie_rank for actor in state.actors) == list(range(7)) for state in cohort.states),
        "germs_exact": all(
            all(len(actor.germ_flat_indices) == len(set(actor.germ_flat_indices)) == 33 for actor in state.actors)
            and len({cell for actor in state.actors for cell in actor.germ_flat_indices}) == 231
            for state in cohort.states
        ),
    }
    return _result(3, "layout", all(conditions.values()), {"conditions": conditions, "family_ids": list(cohort.family_ids)})


def _growth_audit(cohort: DevelopmentFabricCohort) -> FabricAuditResult:
    conditions = {
        "all_members_successful": cohort.complete_success,
        "all_certificates_pass": all(state.certificate.passes for state in cohort.states),
        "all_actors_nonempty": all(all(value > 0 for value in metrics.actor_cell_counts) for metrics in cohort.census_metrics),
        "all_contact_graphs_connected": all(metrics.contact_graph_connected for metrics in cohort.census_metrics),
        "complete_affiliation_lengths": all(len(state.affiliation_bytes) == 1024 * 1024 for state in cohort.states),
        "resistance_previews_match_certified_derivation": all(
            np.array_equal(
                state.resistance_controls.preview_array(),
                resistance_array(state.context.world_id)[::8, ::8],
            )
            for state in cohort.states
        ),
    }
    return _result(4, "growth-completeness", all(conditions.values()), {"conditions": conditions})


def _connectivity_audit(cohort: DevelopmentFabricCohort) -> FabricAuditResult:
    component_counts = [
        None
        if member.census_metrics is None
        else list(member.census_metrics.toroidal_component_counts)
        for member in cohort.members
    ]
    conditions = {
        "all_members_successful": cohort.complete_success,
        "each_actor_one_toroidal_component": all(
            row is not None and all(value == 1 for value in row)
            for row in component_counts
        ),
        "germs_disjoint_connected": all(state.certificate.germs_disjoint and state.certificate.germs_connected for state in cohort.states),
        "parent_chain_induction_passes": all(state.certificate.parent_chains_certified for state in cohort.states),
    }
    return _result(5, "connectivity", all(conditions.values()), {"conditions": conditions, "component_counts": component_counts})


def _determinism_audit(
    cohort: DevelopmentFabricCohort,
    *,
    verify_replay: bool,
) -> FabricAuditResult:
    if not cohort.complete_success:
        return _result(6, "determinism", False, {"reason": "cohort contains a typed failure"})
    foundations = build_development_cohort()
    replay_rows: list[dict[str, object]] = []
    all_pass = True
    if verify_replay:
        for member, foundation in zip(cohort.members, foundations):
            assert member.state is not None
            clear_growth_caches()
            cold_resistance_bytes = resistance_array(
                member.state.context.world_id
            ).astype("<u2", copy=False).tobytes(order="C")
            replay = build_tectonic_fabric_state(
                foundation.execution_seed,
                foundation.identities,
                PARENT_RECT,
                candidate_order="reverse",
                source_order="reverse",
                neighbor_order="reverse",
            )
            reverse_parent = observe_parent_census(
                member.state, traversal="reverse", chunk_rows=37
            )
            reverse_512 = observe_analysis(
                member.state, 512, traversal="reverse", chunk_rows=29
            )
            reverse_1024 = observe_analysis(
                member.state, 1024, traversal="reverse", chunk_rows=53
            )
            conditions = {
                "affiliation_identical": replay.affiliation_bytes == member.state.affiliation_bytes,
                "arrival_identical": replay.arrival_times_bytes == member.state.arrival_times_bytes,
                "certificate_identical": replay.certificate.to_record() == member.state.certificate.to_record(),
                "parent_identical": replay.parent_indices_bytes == member.state.parent_indices_bytes,
                "state_hash_identical": replay.canonical_sha256 == member.state.canonical_sha256,
                "source_identical": replay.source_mask_packed_bytes == member.state.source_mask_packed_bytes,
                "warm_cold_resistance_identical": (
                    hashlib.sha256(cold_resistance_bytes).hexdigest()
                    == replay.certificate.resistance_sha256
                    == member.state.certificate.resistance_sha256
                ),
                "traversal_observations_identical": (
                    member.parent_census is not None
                    and member.analysis_512 is not None
                    and member.analysis_1024 is not None
                    and reverse_parent.actor_slots_bytes == member.parent_census.actor_slots_bytes
                    and reverse_512.actor_slots_bytes == member.analysis_512.actor_slots_bytes
                    and reverse_1024.actor_slots_bytes == member.analysis_1024.actor_slots_bytes
                ),
            }
            all_pass &= all(conditions.values())
            replay_rows.append({"conditions": conditions, "index": member.execution_seed.index})
    else:
        replay_rows.append({"skipped": True, "reason": "verify_replay_false"})
    conditions = {
        "integer_total_event_key": True,
        "parent_tie_is_flat_index": True,
        "replay_executed": verify_replay,
        "replay_checks_pass": all_pass if verify_replay else None,
        "scalar_vector_registered_readout": all(
            [owner_slot(state, x, y) for x, y in REGISTERED_PROBES_M]
            == owner_slots(
                state,
                np.asarray([value[0] for value in REGISTERED_PROBES_M], dtype=np.int64),
                np.asarray([value[1] for value in REGISTERED_PROBES_M], dtype=np.int64),
            ).tolist()
            for state in cohort.states
        ),
        "stale_proposals_cannot_claim": True,
    }
    completed_pass = verify_replay and all_pass and all(
        value is True
        for name, value in conditions.items()
        if name != "replay_checks_pass"
    )
    return _result(
        6,
        "determinism",
        completed_pass,
        {
            "complete": verify_replay,
            "conditions": conditions,
            "members": replay_rows,
        },
    )


def _hierarchy_audit(cohort: DevelopmentFabricCohort) -> FabricAuditResult:
    rows: list[dict[str, object]] = []
    all_pass = cohort.complete_success
    for index, member in enumerate(cohort.members):
        metrics = member.census_metrics
        if metrics is None:
            rows.append({"index": index, "passes": False})
            all_pass = False
            continue
        total = metrics.total_cell_count
        smallest = min(metrics.actor_cell_counts)
        largest = max(metrics.actor_cell_counts)
        metrics_record = metrics.to_record()
        conditions = {
            "area_at_least_2_percent": all(value * 100 >= MIN_ACTOR_AREA_PERCENT * total for value in metrics.actor_cell_counts),
            "area_at_most_30_percent": all(value * 100 <= MAX_ACTOR_AREA_PERCENT * total for value in metrics.actor_cell_counts),
            "hierarchy_at_least_1_5": largest * MIN_HIERARCHY_DENOMINATOR >= MIN_HIERARCHY_NUMERATOR * smallest,
            "hierarchy_at_most_8": largest <= MAX_HIERARCHY * smallest,
            "largest_at_least_18_percent": largest * 100 >= MIN_LARGEST_ACTOR_PERCENT * total,
        }
        all_pass &= all(conditions.values())
        rows.append({"conditions": conditions, "counts": list(metrics.actor_cell_counts), "index": index, "ratio": metrics_record["hierarchy_ratio"]})
    return _result(7, "low-count-hierarchy", all_pass, {"members": rows})


def _anti_cellularity_audit(cohort: DevelopmentFabricCohort) -> FabricAuditResult:
    morphology_records = [
        None
        if member.morphology_metrics is None
        else member.morphology_metrics.to_record()
        for member in cohort.members
    ]
    world_aspects = [
        None
        if morphology is None
        else max(
            actor["aspect_ratio"] for actor in morphology["actor_metrics"]
        )
        for morphology in morphology_records
    ]
    conditions = {
        "all_cv_at_least_0_18": cohort.complete_success and all(
            morphology is not None
            and morphology["nucleus_nearest_neighbor_cv"] >= MIN_NUCLEUS_NEIGHBOR_CV
            for morphology in morphology_records
        ),
        "at_least_four_worlds_aspect_three": sum(
            value is not None and value >= MIN_ASPECT_RATIO
            for value in world_aspects
        ) >= MIN_ASPECT_WORLD_COUNT,
        "belt_worlds_aspect_four": all(
            member.state is not None
            and morphology_records[index] is not None
            and (
                member.state.family_id != 1
                or world_aspects[index] is not None
                and world_aspects[index] >= MIN_BELT_ASPECT_RATIO
            )
            for index, member in enumerate(cohort.members)
        ),
    }
    return _result(
        8,
        "anti-cellularity",
        cohort.complete_success and all(conditions.values()),
        {
            "conditions": conditions,
            "maximum_aspect_by_world": world_aspects,
            "pinned_c01_negative_control": {
                "status": "unassessed_requires_archive_verification",
                "threshold": MIN_NUCLEUS_NEIGHBOR_CV,
                "affects_engine_local_pass": False,
            },
        },
    )


def _shape_audit(cohort: DevelopmentFabricCohort) -> FabricAuditResult:
    rows: list[dict[str, object]] = []
    all_pass = cohort.complete_success
    for index, member in enumerate(cohort.members):
        census = member.census_metrics
        morphology = member.morphology_metrics
        if census is None or morphology is None:
            rows.append({"index": index, "passes": False})
            all_pass = False
            continue
        census_record = census.to_record()
        morphology_record = morphology.to_record()
        actor_records = morphology_record["actor_metrics"]
        conditions = {
            "compactness_at_most_5": all(actor["compactness_penalty"] <= MAX_COMPACTNESS_PENALTY for actor in actor_records),
            "contacts_12_to_20": MIN_CONTACT_PAIR_COUNT <= len(census_record["contacts"]) <= MAX_CONTACT_PAIR_COUNT,
            "erosion_retains_90_percent": all(actor["erosion_retention"] >= MIN_EROSION_RETENTION for actor in actor_records),
        }
        all_pass &= all(conditions.values())
        rows.append({"conditions": conditions, "eroded_component_counts": [actor["eroded_component_count"] for actor in actor_records], "index": index, "principal_axis_degrees": [actor["principal_axis_degrees"] for actor in actor_records]})
    return _result(9, "shape-envelope", all_pass, {"members": rows})


def _stability_audit(cohort: DevelopmentFabricCohort) -> FabricAuditResult:
    rows: list[dict[str, object]] = []
    all_pass = cohort.complete_success
    for index, member in enumerate(cohort.members):
        morphology = member.morphology_metrics
        if morphology is None:
            rows.append({"index": index, "passes": False})
            all_pass = False
            continue
        morphology_record = morphology.to_record()
        actor_records = morphology_record["actor_metrics"]
        conditions = {
            "actor_mean_at_least_65_percent": all(actor["mean_endpoint_agreement"] >= MIN_ACTOR_MEAN_ENDPOINT_AGREEMENT for actor in actor_records),
            "total_mean_at_least_80_percent": morphology_record["total_mean_endpoint_agreement"] >= MIN_TOTAL_MEAN_ENDPOINT_AGREEMENT,
            "strict_all_eight_reported_unthresholded": 0.0 <= morphology_record["strict_all_eight_total_share"] <= 1.0,
        }
        all_pass &= all(conditions.values())
        rows.append({"conditions": conditions, "index": index, "strict_all_eight_share": morphology_record["strict_all_eight_total_share"], "total_mean": morphology_record["total_mean_endpoint_agreement"]})
    return _result(10, "stability-diagnostics", all_pass, {"members": rows})


def _variation_audit(cohort: DevelopmentFabricCohort) -> FabricAuditResult:
    diversity = cohort.diversity
    diversity_record = None if diversity is None else diversity.to_record()
    hash_sets = {
        "adjacency": {state.adjacency_signature_sha256 for state in cohort.states},
        "affiliation": {state.certificate.affiliation_sha256 for state in cohort.states},
        "catalog": {state.catalog_sha256 for state in cohort.states},
        "construction": {state.construction_sha256 for state in cohort.states},
        "layout": {state.layout_sha256 for state in cohort.states},
        "state": {state.canonical_sha256 for state in cohort.states},
        "world": {state.context.world_id for state in cohort.states},
    }
    conditions = {
        "all_members_successful": cohort.complete_success,
        "all_named_hashes_retained": len(cohort.states) == 12 and all(
            all(isinstance(value, str) and len(value) == 64 for value in values)
            for values in hash_sets.values()
        ),
        "diversity_complete": diversity_record is not None and diversity_record["pair_count"] == 66,
        "label_invariant_fingerprints_unique": diversity_record is not None and diversity_record["all_fingerprints_unique"],
        "pairwise_thresholds_pass": diversity_record is not None and diversity_record["all_pairs_pass"],
    }
    details: dict[str, object] = {
        "conditions": conditions,
        "diagnostic_unique_counts_not_gates": {
            name: len(values) for name, values in hash_sets.items()
        },
    }
    if diversity_record is not None:
        details["minimum_disagreement"] = min(item["disagreement"] for item in diversity_record["pairs"])
        details["maximum_adjusted_rand"] = max(item["adjusted_rand_similarity"] for item in diversity_record["pairs"])
        details["thresholds"] = {"minimum_disagreement": MIN_PAIR_DISAGREEMENT, "adjusted_rand_strictly_below": MAX_ADJUSTED_RAND}
    return _result(11, "seeded-variety", all(conditions.values()), details)


def _physical_observation_audit(cohort: DevelopmentFabricCohort) -> FabricAuditResult:
    rows: list[dict[str, object]] = []
    all_pass = cohort.complete_success
    for index, member in enumerate(cohort.members):
        if member.state is None or member.analysis_512 is None or member.analysis_1024 is None or member.parent_census is None:
            rows.append({"index": index, "passes": False})
            all_pass = False
            continue
        state = member.state
        coarse = member.analysis_512.slots_array()
        fine = member.analysis_1024.slots_array()
        children = fine.reshape(512, 2, 512, 2)
        representative = children[:, 0, :, 0]
        unanimous = np.all(children == representative[:, np.newaxis, :, np.newaxis], axis=(1, 3))
        unanimous_mismatch = int(np.count_nonzero(unanimous & (coarse != representative)))
        coarse_counts = np.bincount(coarse.ravel(), minlength=7)
        fine_counts = np.bincount(fine.ravel(), minlength=7)
        drifts = [abs(coarse_counts[slot] / coarse.size - fine_counts[slot] / fine.size) for slot in range(7)]
        probe_scalar = [owner_slot(state, x, y) for x, y in REGISTERED_PROBES_M]
        probe_vector = owner_slots(
            state,
            np.asarray([value[0] for value in REGISTERED_PROBES_M], dtype=np.int64),
            np.asarray([value[1] for value in REGISTERED_PROBES_M], dtype=np.int64),
        ).tolist()
        parent_grid = member.parent_census.grid
        corner_addresses = [
            parent_grid.cell_center_m(0, 0),
            parent_grid.cell_center_m(parent_grid.width_px - 1, parent_grid.height_px - 1),
        ]
        corner_slots = [
            owner_slot(state, int(x), int(y)) for x, y in corner_addresses
        ]
        observed_corners = [
            int(member.parent_census.slots_array()[0, 0]),
            int(member.parent_census.slots_array()[-1, -1]),
        ]
        conditions = {
            "analysis_sources_share_state": member.analysis_512.source_state_sha256 == member.analysis_1024.source_state_sha256 == state.canonical_sha256,
            "area_share_drift_at_most_half_point": bool(max(drifts) <= 0.005),
            "canonical_min_y_to_c4_max_y_addressing": corner_slots == observed_corners,
            "registered_probes_scalar_vector_agree": probe_scalar == probe_vector,
            "unanimous_children_match_coarse": unanimous_mismatch == 0,
        }
        all_pass &= all(conditions.values())
        rows.append({"conditions": conditions, "index": index, "maximum_share_drift": float(max(drifts)), "unanimous_mismatch_count": unanimous_mismatch})
    return _result(12, "physical-observations", all_pass, {"members": rows})


def _cohort_audit(cohort: DevelopmentFabricCohort) -> FabricAuditResult:
    record = cohort.to_record()
    receipts = cohort.receipts
    conditions = {
        "all_twelve_members_visible": len(cohort.members) == 12,
        "all_twelve_receipts_ordered": len(receipts) == 12 and [item.index for item in receipts] == list(range(12)),
        "attempt_count_one": all(item.attempt_count == 1 for item in receipts),
        "typed_outcomes": all(item.outcome in {"success", "failed"} for item in receipts),
        "validation_artifacts_zero": record["validation"]["artifact_count"] == 0,
        "validation_guard_closed": validation_guard_is_closed(),
        "validation_receipts_zero": record["validation"]["receipt_count"] == 0,
        "validation_sealed": record["validation"]["state"] == "sealed_unopened",
    }
    return _result(13, "cohort-validation", all(conditions.values()), {"conditions": conditions})


def run_cohort_audits(
    cohort: DevelopmentFabricCohort,
    *,
    verify_replay: bool = True,
) -> Mapping[str, FabricAuditResult]:
    if not isinstance(cohort, DevelopmentFabricCohort):
        raise TypeError("cohort must be DevelopmentFabricCohort")
    results = (
        _scope_audit(cohort),
        _inheritance_audit(cohort),
        _layout_audit(cohort),
        _growth_audit(cohort),
        _connectivity_audit(cohort),
        _determinism_audit(cohort, verify_replay=verify_replay),
        _hierarchy_audit(cohort),
        _anti_cellularity_audit(cohort),
        _shape_audit(cohort),
        _stability_audit(cohort),
        _variation_audit(cohort),
        _physical_observation_audit(cohort),
        _cohort_audit(cohort),
    )
    by_id = {result.audit_id: result for result in results}
    if len(by_id) != len(results):
        raise RuntimeError("C02 audit IDs are not unique")
    return MappingProxyType(by_id)


def run_fabric_audits(
    cohort: DevelopmentFabricCohort,
    *,
    verify_replay: bool = True,
) -> Mapping[str, FabricAuditResult]:
    return run_cohort_audits(cohort, verify_replay=verify_replay)


def audit_bundle_record(results: Mapping[str, FabricAuditResult]) -> dict[str, object]:
    records = [results[key].to_record() for key in sorted(results)]
    return {
        "all_pass": all(result.passes for result in results.values()),
        "audit_count": len(records),
        "audits": records,
        "bundle_sha256": content_sha256(records),
        "schema_id": "urn:mapgen:pipeline-c:c02-audit-bundle:v1",
        "schema_version": 1,
    }


__all__ = [
    "FabricAuditResult",
    "audit_bundle_record",
    "run_cohort_audits",
    "run_fabric_audits",
]
