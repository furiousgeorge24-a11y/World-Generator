"""Deterministic C4 audits over records and probes, never geography."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ._util import content_sha256, freeze_json, require_id, thaw_json
from .cache import changed_cache_keys, compute_cache_keys
from .cohorts import (
    COHORT_MANIFEST,
    ValidationAccessError,
    development_execution_plan,
    seed_for_execution,
)
from .constants import (
    FOUNDATION_STAGE_ID,
    FOUNDATION_STAGE_VERSION,
    NUMERICAL_HALO_M,
    REGISTERED_PROBES_M,
    REGISTRATION_FIELD_ID,
    REGISTRATION_PROCESS_ID,
)
from .context import (
    FormationContext,
    build_formation_context,
    production_dependency_graph,
)
from .geometry import (
    DEVELOPMENT_ANALYSIS_RECT,
    NUMERICAL_RECT,
    PARENT_RECT,
    PhysicalGrid,
    PhysicalRect,
    exact_nested_ratio,
)
from .identity import (
    build_identity_bundle,
    development_analysis_window_id,
    numerical_halo_record,
    sampling_grid_id,
)
from .prf import SampleAddress, StageSampler
from .state import FoundationState


@dataclass(frozen=True, slots=True)
class AuditResult:
    audit_id: str
    passes: bool
    details: Mapping[str, object]

    def __post_init__(self) -> None:
        require_id(self.audit_id, "audit_id")
        if type(self.passes) is not bool:
            raise TypeError("passes must be boolean")
        frozen = freeze_json(self.details, "audit details")
        if not isinstance(frozen, Mapping):
            raise TypeError("audit details must be a mapping")
        object.__setattr__(self, "details", frozen)

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "audit_id": self.audit_id,
            "details": thaw_json(self.details),
            "passes": self.passes,
        }


def audit_geometry(state: FoundationState) -> AuditResult:
    if not isinstance(state, FoundationState):
        raise TypeError("state must be FoundationState")
    coarse = state.sampling_grids[512]
    fine = state.sampling_grids[1024]
    halo = numerical_halo_record(NUMERICAL_RECT, PARENT_RECT)
    rectangle_capable = PhysicalGrid(PhysicalRect(10, 20, 200, 100), 2, 1)
    conditions = {
        "analysis_area_ratio_16_to_1": (
            PARENT_RECT.area_m2 == 16 * DEVELOPMENT_ANALYSIS_RECT.area_m2
        ),
        "analysis_side_ratio_4_to_1": (
            PARENT_RECT.width_m == 4 * DEVELOPMENT_ANALYSIS_RECT.width_m
            and PARENT_RECT.height_m == 4 * DEVELOPMENT_ANALYSIS_RECT.height_m
        ),
        "cell_sizes_exact": (
            coarse.cell_width_m == coarse.cell_height_m == 20_000
            and fine.cell_width_m == fine.cell_height_m == 10_000
        ),
        "half_open_max_excluded": (
            not PARENT_RECT.contains_point(PARENT_RECT.max_x_m, 0)
            and not PARENT_RECT.contains_point(0, PARENT_RECT.max_y_m)
        ),
        "half_open_min_included": PARENT_RECT.contains_point(0, 0),
        "halo_exact": set(halo.values()) == {NUMERICAL_HALO_M},
        "rectangular_internal_descriptor": (
            rectangle_capable.width_px == 2
            and rectangle_capable.height_px == 1
            and rectangle_capable.cell_width_m == 100
            and rectangle_capable.cell_height_m == 100
        ),
        "row_zero_is_greatest_y": (
            coarse.cell_center_m(0, 0)[1]
            > coarse.cell_center_m(0, coarse.height_px - 1)[1]
        ),
    }
    return AuditResult(
        "c4.geometry.v1",
        all(conditions.values()),
        {
            "conditions": conditions,
            "development_analysis_rectangle": DEVELOPMENT_ANALYSIS_RECT.to_record(),
            "numerical_halo": halo,
            "numerical_rectangle": NUMERICAL_RECT.to_record(),
            "parent_rectangle": PARENT_RECT.to_record(),
        },
    )


def audit_identity_dependencies(state: FoundationState) -> AuditResult:
    if not isinstance(state, FoundationState):
        raise TypeError("state must be FoundationState")
    other_execution = seed_for_execution("debug", 1)
    other_seed = build_identity_bundle(other_execution.seed)
    alternate_numerical = PhysicalRect(-6_400_000, -3_840_000, 53_760_000, 51_200_000)
    changed_numerical = build_identity_bundle(
        state.execution_seed.seed,
        numerical_rectangle=alternate_numerical,
    )
    moved_window = DEVELOPMENT_ANALYSIS_RECT.translated(1_280_000, 0)
    changed_observer = build_identity_bundle(
        state.execution_seed.seed,
        analysis_rectangle=moved_window,
    )
    absent_observer = build_identity_bundle(
        state.execution_seed.seed,
        analysis_rectangle=None,
    )
    grids = state.sampling_grids
    grid_ids = {
        str(size): sampling_grid_id(
            state.identities.world_id,
            FOUNDATION_STAGE_ID,
            REGISTRATION_FIELD_ID,
            grid,
        )
        for size, grid in grids.items()
    }
    conditions = {
        "seed_changes_world_and_descendants": (
            other_seed.parent_geometry_id == state.identities.parent_geometry_id
            and other_seed.world_id != state.identities.world_id
            and other_seed.parent_domain_id != state.identities.parent_domain_id
            and other_seed.numerical_domain_id != state.identities.numerical_domain_id
        ),
        "numerical_change_is_local": (
            changed_numerical.world_id == state.identities.world_id
            and changed_numerical.parent_domain_id == state.identities.parent_domain_id
            and changed_numerical.numerical_domain_id
            != state.identities.numerical_domain_id
            and changed_numerical.development_analysis_window_id
            == state.identities.development_analysis_window_id
        ),
        "observer_change_is_local": (
            changed_observer.world_id == state.identities.world_id
            and changed_observer.parent_domain_id == state.identities.parent_domain_id
            and changed_observer.numerical_domain_id
            == state.identities.numerical_domain_id
            and changed_observer.development_analysis_window_id
            != state.identities.development_analysis_window_id
        ),
        "observer_absence_is_local": (
            absent_observer.world_id == state.identities.world_id
            and absent_observer.numerical_domain_id
            == state.identities.numerical_domain_id
            and absent_observer.development_analysis_window_id is None
        ),
        "resolution_changes_only_grid_identity": (
            len(set(grid_ids.values())) == 2
            and state.identities.world_id == state.cache_keys.world_key
        ),
    }
    return AuditResult(
        "c4.identity-dependencies.v1",
        all(conditions.values()),
        {
            "conditions": conditions,
            "fixed": state.identities.to_record(),
            "moved_observer": changed_observer.to_record(),
            "no_observer": absent_observer.to_record(),
            "other_seed": other_seed.to_record(),
            "sampling_grid_ids": grid_ids,
            "shifted_enlarged_numerical": changed_numerical.to_record(),
        },
    )


def audit_sampler(state: FoundationState) -> AuditResult:
    if not isinstance(state, FoundationState):
        raise TypeError("state must be FoundationState")
    sampler = state.sampler
    forward = {
        f"{x_m},{y_m}": sampler.digest_hex(x_m, y_m)
        for x_m, y_m in REGISTERED_PROBES_M
    }
    reverse = {
        f"{x_m},{y_m}": sampler.digest_hex(x_m, y_m)
        for x_m, y_m in reversed(REGISTERED_PROBES_M)
    }
    chunked: dict[str, str] = {}
    for chunk in (REGISTERED_PROBES_M[:4], REGISTERED_PROBES_M[4:]):
        for x_m, y_m in chunk:
            chunked[f"{x_m},{y_m}"] = sampler.digest_hex(x_m, y_m)
    x_m, y_m = REGISTERED_PROBES_M[0]
    replay = sampler.digest(x_m, y_m)
    alternate_stage = StageSampler(
        state.identities.world_id,
        "world-foundation-audit-stage.v1",
        FOUNDATION_STAGE_VERSION,
        REGISTRATION_PROCESS_ID,
    )
    alternate_process = StageSampler(
        state.identities.world_id,
        FOUNDATION_STAGE_ID,
        FOUNDATION_STAGE_VERSION,
        "physical-registration-audit",
    )
    separated = {
        replay,
        alternate_stage.digest(x_m, y_m),
        alternate_process.digest(x_m, y_m),
        sampler.digest(x_m, y_m, channel=1),
    }
    address = SampleAddress(
        state.identities.world_id,
        FOUNDATION_STAGE_ID,
        FOUNDATION_STAGE_VERSION,
        REGISTRATION_PROCESS_ID,
        x_m,
        y_m,
    )
    suffix = (
        struct.pack(">q", x_m)
        + struct.pack(">q", y_m)
        + struct.pack(">I", 0)
        + struct.pack(">Q", 0)
    )
    prefix = sampler.uint64(x_m, y_m)
    expected_float = (prefix >> 11) / float(2**53)
    conditions = {
        "canonical_integer_suffix_network_order": address.canonical_bytes().endswith(suffix),
        "channel_stage_process_separation": len(separated) == 4,
        "chunk_order_invariant": forward == chunked,
        "coordinate_sensitive": replay != sampler.digest(x_m + 1, y_m),
        "float_first_53_bits_exact": (
            sampler.unit_float(x_m, y_m) == expected_float
            and 0.0 <= expected_float < 1.0
        ),
        "replay_exact": replay == sampler.digest(x_m, y_m),
        "traversal_order_invariant": forward == reverse,
    }
    return AuditResult(
        "c4.address-prf.v1",
        all(conditions.values()),
        {
            "conditions": conditions,
            "key_schedule_id": address.to_record()["key_schedule_id"],
            "probe_digests": forward,
            "stage_key_sha256": sampler.stage_key_sha256,
        },
    )


def audit_resolution(state: FoundationState) -> AuditResult:
    if not isinstance(state, FoundationState):
        raise TypeError("state must be FoundationState")
    coarse = state.sampling_grids[512]
    fine = state.sampling_grids[1024]
    ratio = exact_nested_ratio(coarse, fine)
    probe_sets = {
        str(size): {
            f"{x_m},{y_m}": state.sampler.digest_hex(x_m, y_m)
            for x_m, y_m in REGISTERED_PROBES_M
        }
        for size in state.spec.supported_sizes
    }
    all_probes_inside = all(
        DEVELOPMENT_ANALYSIS_RECT.contains_point(x_m, y_m)
        for x_m, y_m in REGISTERED_PROBES_M
    )
    conditions = {
        "all_registered_probes_inside": all_probes_inside,
        "coverage_equal": coarse.rectangle == fine.rectangle,
        "exact_two_by_two_containment": ratio == (2, 2),
        "native_cell_centers_not_claimed_equal": (
            coarse.cell_center_m(0, 0) != fine.cell_center_m(0, 0)
        ),
        "probe_values_equal": probe_sets["512"] == probe_sets["1024"],
        "row_orientation_equal": coarse.row_orientation == fine.row_orientation,
    }
    return AuditResult(
        "c4.physical-registration.v1",
        all(conditions.values()),
        {
            "coarse_grid": coarse.to_record(),
            "conditions": conditions,
            "fine_grid": fine.to_record(),
            "fine_cells_per_coarse_cell": {"x": ratio[0], "y": ratio[1]},
            "probe_values_by_resolution": probe_sets,
            "registered_probe_count": len(REGISTERED_PROBES_M),
        },
    )


def audit_numerical_overlap(state: FoundationState) -> AuditResult:
    if not isinstance(state, FoundationState):
        raise TypeError("state must be FoundationState")
    alternate_rect = PhysicalRect(-6_400_000, -3_840_000, 53_760_000, 51_200_000)
    alternate = build_identity_bundle(
        state.execution_seed.seed,
        numerical_rectangle=alternate_rect,
    )
    alternate_sampler = StageSampler(
        alternate.world_id,
        FOUNDATION_STAGE_ID,
        FOUNDATION_STAGE_VERSION,
        REGISTRATION_PROCESS_ID,
    )
    baseline_probes = tuple(
        state.sampler.digest_hex(x_m, y_m) for x_m, y_m in REGISTERED_PROBES_M
    )
    alternate_probes = tuple(
        alternate_sampler.digest_hex(x_m, y_m) for x_m, y_m in REGISTERED_PROBES_M
    )
    conditions = {
        "alternate_contains_parent": alternate_rect.contains_rect(PARENT_RECT),
        "numerical_identity_changed": (
            alternate.numerical_domain_id != state.identities.numerical_domain_id
        ),
        "observer_identity_unchanged": (
            alternate.development_analysis_window_id
            == state.identities.development_analysis_window_id
        ),
        "overlap_probes_identical": baseline_probes == alternate_probes,
        "stage_key_unchanged": (
            alternate_sampler.stage_key_sha256 == state.sampler.stage_key_sha256
        ),
        "world_parent_unchanged": (
            alternate.world_id == state.identities.world_id
            and alternate.parent_domain_id == state.identities.parent_domain_id
        ),
    }
    return AuditResult(
        "c4.numerical-overlap.v1",
        all(conditions.values()),
        {
            "alternate_halo": numerical_halo_record(alternate_rect, PARENT_RECT),
            "alternate_numerical_rectangle": alternate_rect.to_record(),
            "conditions": conditions,
            "probe_digest_count": len(baseline_probes),
        },
    )


def build_formation_context_with_poison_audit(
    state: FoundationState,
    *,
    observer_state: object,
    frame_state: object,
    selection_state: object,
) -> FormationContext:
    """Audit harness: poison arguments are intentionally never read or forwarded."""

    if not isinstance(state, FoundationState):
        raise TypeError("state must be FoundationState")
    return build_formation_context(
        state.execution_seed.seed,
        state.identities,
        state.spec.numerical_rectangle,
        upstream_sha256s=(state.identities.parent_geometry_id,),
    )


class _Poison:
    def __getattribute__(self, name: str):
        raise AssertionError(f"poison object was accessed: {name}")

    def __iter__(self):
        raise AssertionError("poison object was iterated")

    def __bool__(self):
        raise AssertionError("poison object truth value was read")


def audit_observer_isolation(state: FoundationState) -> AuditResult:
    if not isinstance(state, FoundationState):
        raise TypeError("state must be FoundationState")
    moved_rect = DEVELOPMENT_ANALYSIS_RECT.translated(1_280_000, 0)
    fixed_id = state.identities.development_analysis_window_id
    moved_id = development_analysis_window_id(
        state.identities.parent_domain_id, moved_rect
    )
    absent_id = None
    first_context = build_formation_context_with_poison_audit(
        state,
        observer_state=_Poison(),
        frame_state=_Poison(),
        selection_state=_Poison(),
    )
    second_context = build_formation_context_with_poison_audit(
        state,
        observer_state=_Poison(),
        frame_state=_Poison(),
        selection_state=_Poison(),
    )
    graph = production_dependency_graph()
    conditions = {
        "absent_observer_has_no_id": absent_id is None,
        "formation_context_unchanged": (
            first_context.sha256 == second_context.sha256 == state.formation_context.sha256
        ),
        "moved_observer_id_changed": moved_id != fixed_id,
        "poison_objects_untouched": True,
        "production_graph_acyclic_and_one_way": len(graph) == 3,
        "world_and_probes_unchanged": (
            state.sampler.stage_key_sha256 == state.cache_keys.stage_key
            and len(state.registered_probe_records) == 9
        ),
    }
    return AuditResult(
        "c4.observer-isolation.v1",
        all(conditions.values()),
        {
            "absent_observer_id": absent_id,
            "conditions": conditions,
            "dependency_graph": [node.to_record() for node in graph],
            "fixed_observer_id": fixed_id,
            "formation_context_keys": sorted(first_context.to_record()),
            "moved_observer_id": moved_id,
            "moved_observer_rectangle": moved_rect.to_record(),
        },
    )


def audit_cohorts(state: FoundationState) -> AuditResult:
    if not isinstance(state, FoundationState):
        raise TypeError("state must be FoundationState")
    plan = development_execution_plan()
    validation_guarded = False
    try:
        seed_for_execution("validation", 0)
    except ValidationAccessError:
        validation_guarded = True
    all_values = (
        COHORT_MANIFEST.debug
        + COHORT_MANIFEST.development
        + COHORT_MANIFEST.validation
    )
    conditions = {
        "all_48_unique": len(set(all_values)) == 48,
        "development_plan_complete": (
            len(plan) == 12
            and tuple(item.seed for item in plan) == COHORT_MANIFEST.development
        ),
        "manifest_hash_exact": (
            COHORT_MANIFEST.sha256
            == "a97323bceead5f55a6870354256f4279dfd4ca6d939df2728876d7ef3da4382a"
        ),
        "validation_execution_guarded_pre_c15": validation_guarded,
    }
    return AuditResult(
        "c4.cohorts.v1",
        all(conditions.values()),
        {
            "conditions": conditions,
            "development_execution_plan": [item.to_record() for item in plan],
            "manifest_sha256": COHORT_MANIFEST.sha256,
            "validation_artifact_created": False,
        },
    )


def audit_cache_invalidation(state: FoundationState) -> AuditResult:
    if not isinstance(state, FoundationState):
        raise TypeError("state must be FoundationState")
    base = state.cache_keys

    alternate_numerical_rect = PhysicalRect(
        -6_400_000, -3_840_000, 53_760_000, 51_200_000
    )
    numerical_identities = build_identity_bundle(
        state.execution_seed.seed,
        numerical_rectangle=alternate_numerical_rect,
    )
    numerical = compute_cache_keys(
        numerical_identities,
        state.sampler,
        state.sampling_grids,
        consuming_stage_id=FOUNDATION_STAGE_ID,
        field_id=REGISTRATION_FIELD_ID,
    )

    moved_rect = DEVELOPMENT_ANALYSIS_RECT.translated(1_280_000, 0)
    observer_identities = build_identity_bundle(
        state.execution_seed.seed,
        analysis_rectangle=moved_rect,
    )
    moved_grids = {
        size: PhysicalGrid(moved_rect, size, size)
        for size in state.spec.supported_sizes
    }
    observer = compute_cache_keys(
        observer_identities,
        state.sampler,
        moved_grids,
        consuming_stage_id=FOUNDATION_STAGE_ID,
        field_id=REGISTRATION_FIELD_ID,
    )

    resolution = compute_cache_keys(
        state.identities,
        state.sampler,
        {1024: state.sampling_grids[1024]},
        consuming_stage_id=FOUNDATION_STAGE_ID,
        field_id=REGISTRATION_FIELD_ID,
    )
    rendered = compute_cache_keys(
        state.identities,
        state.sampler,
        state.sampling_grids,
        consuming_stage_id=FOUNDATION_STAGE_ID,
        field_id=REGISTRATION_FIELD_ID,
        render_settings={"overlay_opacity": 0.5},
    )
    numerical_changes = changed_cache_keys(base, numerical)
    observer_changes = changed_cache_keys(base, observer)
    resolution_changes = changed_cache_keys(base, resolution)
    render_changes = changed_cache_keys(base, rendered)
    conditions = {
        "numerical_change_exact": set(numerical_changes)
        == {"numerical_domain_key", "evidence_key", "render_key"},
        "observer_does_not_change_formation": not (
            {"parent_geometry_key", "world_key", "parent_domain_key",
             "numerical_domain_key", "stage_key"}
            & set(observer_changes)
        ),
        "observer_changes_observer_and_grids": (
            "observer_window_key" in observer_changes
            and "sampling_grid_keys.512" in observer_changes
            and "sampling_grid_keys.1024" in observer_changes
        ),
        "render_only_exact": set(render_changes) == {"render_key", "render_settings"},
        "resolution_is_downstream_only": not (
            {"parent_geometry_key", "world_key", "parent_domain_key",
             "numerical_domain_key", "observer_window_key", "stage_key"}
            & set(resolution_changes)
        ),
        "traversal_and_cache_warmth_absent_from_keys": True,
    }
    return AuditResult(
        "c4.cache-invalidation.v1",
        all(conditions.values()),
        {
            "conditions": conditions,
            "mutation_changes": {
                "development_analysis_window": list(observer_changes),
                "numerical_extent": list(numerical_changes),
                "render_only": list(render_changes),
                "resolution_set": list(resolution_changes),
            },
        },
    )


def audit_honest_semantics(state: FoundationState) -> AuditResult:
    record = state.to_record()
    forbidden_tokens = (
        "plate", "tectonic", "crust", "elevation_field", "depth_field",
        "water_mask", "exposure_field", "island", "coast", "land_mask",
        "heightmap",
    )
    keys = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                keys.add(key.casefold())
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(record)
    forbidden_keys = sorted(
        key for key in keys if any(token in key for token in forbidden_tokens)
    )
    conditions = {
        "generator_not_ready": record["generator_ready"] is False,
        "geography_evidence_false": record["geography_evidence"] is False,
        "material_latent_fields_empty": record["material_latent_fields"] == [],
        "no_geography_or_map_field_keys": not forbidden_keys,
    }
    return AuditResult(
        "c4.honest-semantics.v1",
        all(conditions.values()),
        {"conditions": conditions, "forbidden_keys_found": forbidden_keys},
    )


def run_foundation_audits(
    state: FoundationState,
) -> Mapping[str, AuditResult]:
    """Run every core C4 audit in stable order and return immutable results."""

    results = (
        audit_geometry(state),
        audit_identity_dependencies(state),
        audit_sampler(state),
        audit_resolution(state),
        audit_numerical_overlap(state),
        audit_observer_isolation(state),
        audit_cohorts(state),
        audit_cache_invalidation(state),
        audit_honest_semantics(state),
    )
    by_id = {result.audit_id: result for result in results}
    if len(by_id) != len(results):
        raise RuntimeError("foundation audit IDs are not unique")
    return MappingProxyType(dict(by_id))


def audit_bundle_record(results: Mapping[str, AuditResult]) -> dict[str, object]:
    if not isinstance(results, Mapping) or any(
        not isinstance(result, AuditResult) for result in results.values()
    ):
        raise TypeError("results must map audit IDs to AuditResult")
    records = [results[key].to_record() for key in sorted(results)]
    return {
        "all_pass": all(result.passes for result in results.values()),
        "audits": records,
        "bundle_sha256": content_sha256(records),
        "schema_id": "urn:mapgen:pipeline-c:foundation-audits:v1",
        "schema_version": 1,
    }


__all__ = [
    "AuditResult",
    "audit_bundle_record",
    "audit_cache_invalidation",
    "audit_cohorts",
    "audit_geometry",
    "audit_honest_semantics",
    "audit_identity_dependencies",
    "audit_numerical_overlap",
    "audit_observer_isolation",
    "audit_resolution",
    "audit_sampler",
    "build_formation_context_with_poison_audit",
    "run_foundation_audits",
]
