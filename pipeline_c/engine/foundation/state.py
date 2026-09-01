"""Canonical, geography-free C4 foundation state construction."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ._util import (
    FoundationRecordError,
    content_sha256,
    freeze_json,
    require_hash,
    thaw_json,
)
from .cache import FoundationCacheKeys, compute_cache_keys
from .cohorts import (
    COHORT_MANIFEST_SHA256,
    ExecutionSeed,
    development_execution_plan,
    seed_for_execution,
)
from .constants import (
    ATTEMPT_ID,
    ANALYSIS_ORIENTATION_DEGREES,
    CANONICAL_UNITS,
    COMPARISON_FAMILY_ID,
    COORDINATE_SYSTEM_ID,
    DEFAULT_SIZE,
    DISPLAY_LABEL,
    EVIDENCE_KIND,
    FOUNDATION_SCHEMA_ID,
    FOUNDATION_STAGE_ID,
    FOUNDATION_STAGE_VERSION,
    NUMERICAL_BOUNDARY_POLICY,
    NUMERICAL_HALO_M,
    OBSERVER_PURPOSE,
    OBSERVER_VERSION,
    PARENT_DOMAIN_POLICY,
    RECTANGLE_SEMANTICS,
    REGISTRATION_FIELD_ID,
    REGISTRATION_PROCESS_ID,
    ROADMAP_RUN,
    SUPPORTED_SIZES,
    REGISTERED_PROBES_M,
)
from .context import FormationContext, build_formation_context
from .geometry import (
    DEVELOPMENT_ANALYSIS_RECT,
    NUMERICAL_RECT,
    PARENT_RECT,
    PhysicalGrid,
    PhysicalRect,
)
from .identity import (
    IdentityBundle,
    build_identity_bundle,
    frozen_geometry_is_exact,
    numerical_halo_record,
)
from .prf import StageSampler


@dataclass(frozen=True, slots=True)
class FoundationSpec:
    parent_rectangle: PhysicalRect = PARENT_RECT
    numerical_rectangle: PhysicalRect = NUMERICAL_RECT
    development_analysis_rectangle: PhysicalRect = DEVELOPMENT_ANALYSIS_RECT
    supported_sizes: tuple[int, ...] = SUPPORTED_SIZES
    default_size: int = DEFAULT_SIZE

    def __post_init__(self) -> None:
        for name in (
            "parent_rectangle",
            "numerical_rectangle",
            "development_analysis_rectangle",
        ):
            if not isinstance(getattr(self, name), PhysicalRect):
                raise FoundationRecordError(f"{name} must be PhysicalRect")
        if not self.numerical_rectangle.contains_rect(self.parent_rectangle):
            raise FoundationRecordError("numerical extent must contain parent extent")
        if not self.parent_rectangle.contains_rect(self.development_analysis_rectangle):
            raise FoundationRecordError("analysis window must lie within parent extent")
        if (
            not isinstance(self.supported_sizes, tuple)
            or not self.supported_sizes
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size < 1
                for size in self.supported_sizes
            )
            or tuple(sorted(set(self.supported_sizes))) != self.supported_sizes
        ):
            raise FoundationRecordError("supported_sizes must be a sorted unique tuple")
        if self.default_size not in self.supported_sizes:
            raise FoundationRecordError("default_size must be supported")
        for size in self.supported_sizes:
            PhysicalGrid(self.development_analysis_rectangle, size, size)

    @property
    def is_frozen_c4(self) -> bool:
        return (
            self.parent_rectangle == PARENT_RECT
            and self.numerical_rectangle == NUMERICAL_RECT
            and self.development_analysis_rectangle == DEVELOPMENT_ANALYSIS_RECT
            and self.supported_sizes == SUPPORTED_SIZES
            and self.default_size == DEFAULT_SIZE
            and frozen_geometry_is_exact()
        )

    def to_record(self) -> dict[str, object]:
        return {
            "default_size": self.default_size,
            "coordinate_system_id": COORDINATE_SYSTEM_ID,
            "development_analysis_rectangle": (
                self.development_analysis_rectangle.to_record()
            ),
            "development_analysis_orientation_degrees": (
                ANALYSIS_ORIENTATION_DEGREES
            ),
            "numerical_boundary_policy": NUMERICAL_BOUNDARY_POLICY,
            "numerical_halo": numerical_halo_record(
                self.numerical_rectangle, self.parent_rectangle
            ),
            "numerical_rectangle": self.numerical_rectangle.to_record(),
            "observer_purpose": OBSERVER_PURPOSE,
            "observer_version": OBSERVER_VERSION,
            "parent_domain_policy": PARENT_DOMAIN_POLICY,
            "parent_rectangle": self.parent_rectangle.to_record(),
            "rectangle_semantics": RECTANGLE_SEMANTICS,
            "supported_sizes": list(self.supported_sizes),
            "units": CANONICAL_UNITS,
        }


_FROZEN_C4_SPEC = FoundationSpec()


def frozen_c4_spec() -> FoundationSpec:
    return _FROZEN_C4_SPEC


@dataclass(frozen=True, slots=True)
class FoundationState:
    execution_seed: ExecutionSeed
    spec: FoundationSpec
    identities: IdentityBundle
    sampler: StageSampler
    sampling_grids: Mapping[int, PhysicalGrid]
    registered_probe_records: tuple[object, ...]
    formation_context: FormationContext
    cache_keys: FoundationCacheKeys
    material_latent_fields: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.execution_seed, ExecutionSeed):
            raise FoundationRecordError("execution_seed must be ExecutionSeed")
        if self.execution_seed.role not in {"debug", "development"}:
            raise FoundationRecordError("C4 state cannot consume validation seeds")
        if not isinstance(self.spec, FoundationSpec) or not self.spec.is_frozen_c4:
            raise FoundationRecordError("C00 state must use the frozen C4 specification")
        if not isinstance(self.identities, IdentityBundle):
            raise FoundationRecordError("identities must be IdentityBundle")
        expected = build_identity_bundle(
            self.execution_seed.seed,
            parent_rectangle=self.spec.parent_rectangle,
            numerical_rectangle=self.spec.numerical_rectangle,
            analysis_rectangle=self.spec.development_analysis_rectangle,
        )
        if self.identities != expected:
            raise FoundationRecordError("foundation identities do not match seed/spec")
        if not isinstance(self.sampler, StageSampler):
            raise FoundationRecordError("sampler must be StageSampler")
        if (
            self.sampler.world_id != self.identities.world_id
            or self.sampler.stage_id != FOUNDATION_STAGE_ID
            or self.sampler.stage_version != FOUNDATION_STAGE_VERSION
            or self.sampler.process_id != REGISTRATION_PROCESS_ID
        ):
            raise FoundationRecordError("sampler does not match foundation identity")
        if not isinstance(self.sampling_grids, Mapping):
            raise FoundationRecordError("sampling_grids must be a mapping")
        grids = dict(self.sampling_grids)
        if tuple(sorted(grids)) != self.spec.supported_sizes:
            raise FoundationRecordError("sampling grids do not cover supported sizes")
        for size, grid in grids.items():
            if grid != PhysicalGrid(self.spec.development_analysis_rectangle, size, size):
                raise FoundationRecordError("sampling grid differs from frozen geometry")
        object.__setattr__(
            self, "sampling_grids", MappingProxyType(dict(sorted(grids.items())))
        )
        probes = freeze_json(self.registered_probe_records, "registered_probe_records")
        if not isinstance(probes, tuple) or len(probes) != len(REGISTERED_PROBES_M):
            raise FoundationRecordError("all nine registered probes must be retained")
        object.__setattr__(self, "registered_probe_records", probes)
        expected_probes = freeze_json(
            tuple(
                self.sampler.probe_record(x_m, y_m)
                for x_m, y_m in REGISTERED_PROBES_M
            ),
            "expected_probe_records",
        )
        if probes != expected_probes:
            raise FoundationRecordError("registered probes do not match the address PRF")
        if not isinstance(self.formation_context, FormationContext):
            raise FoundationRecordError("formation_context must be FormationContext")
        expected_context = build_formation_context(
            self.execution_seed.seed,
            self.identities,
            self.spec.numerical_rectangle,
            upstream_sha256s=(self.identities.parent_geometry_id,),
        )
        if self.formation_context != expected_context:
            raise FoundationRecordError("formation context differs from exact allowlist")
        if not isinstance(self.cache_keys, FoundationCacheKeys):
            raise FoundationRecordError("cache_keys must be FoundationCacheKeys")
        expected_cache = compute_cache_keys(
            self.identities,
            self.sampler,
            grids,
            consuming_stage_id=FOUNDATION_STAGE_ID,
            field_id=REGISTRATION_FIELD_ID,
        )
        if self.cache_keys != expected_cache:
            raise FoundationRecordError("cache keys differ from canonical dependencies")
        if self.material_latent_fields != ():
            raise FoundationRecordError("C4 material_latent_fields must be empty")

    @property
    def canonical_sha256(self) -> str:
        return content_sha256(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "attempt_id": ATTEMPT_ID,
            "cache_keys": self.cache_keys.causal_record(),
            "comparison_family_id": COMPARISON_FAMILY_ID,
            "cohort_manifest_sha256": COHORT_MANIFEST_SHA256,
            "coordinate_system_id": COORDINATE_SYSTEM_ID,
            "display_label": DISPLAY_LABEL,
            "evidence_eligible": True,
            "evidence_kind": EVIDENCE_KIND,
            "execution_seed": self.execution_seed.to_record(),
            "formation_context": self.formation_context.to_record(),
            "formation_context_sha256": self.formation_context.sha256,
            "foundation_spec": self.spec.to_record(),
            "generator_ready": False,
            "geography_evidence": False,
            "identities": self.identities.to_record(),
            "material_latent_fields": [],
            "registered_physical_probes": thaw_json(self.registered_probe_records),
            "roadmap_run": ROADMAP_RUN,
            "sampling_grids": {
                str(size): grid.to_record()
                for size, grid in self.sampling_grids.items()
            },
            "schema_id": FOUNDATION_SCHEMA_ID,
            "schema_version": 1,
            "stage_id": FOUNDATION_STAGE_ID,
            "stage_key_sha256": self.sampler.stage_key_sha256,
            "stage_version": FOUNDATION_STAGE_VERSION,
            "units": CANONICAL_UNITS,
        }


@dataclass(frozen=True, slots=True)
class DevelopmentCohortState:
    """Exact all-seed C00 payload; construction rejects omitted/reordered seeds."""

    states: tuple[FoundationState, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.states, tuple) or len(self.states) != 12:
            raise FoundationRecordError(
                "C00 development cohort must contain exactly 12 foundation states"
            )
        if any(not isinstance(state, FoundationState) for state in self.states):
            raise FoundationRecordError("development cohort contains a non-state")
        plan = development_execution_plan()
        actual = tuple(state.execution_seed for state in self.states)
        if actual != plan:
            raise FoundationRecordError(
                "development cohort states must follow the complete frozen manifest order"
            )
        if len({state.canonical_sha256 for state in self.states}) != len(self.states):
            raise FoundationRecordError("development cohort state identities must be unique")

    @property
    def canonical_sha256(self) -> str:
        return content_sha256(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "attempt_id": ATTEMPT_ID,
            "cohort_manifest_sha256": COHORT_MANIFEST_SHA256,
            "roadmap_run": ROADMAP_RUN,
            "schema_id": "urn:mapgen:pipeline-c:development-cohort-state:v1",
            "schema_version": 1,
            "state_count": len(self.states),
            "state_sha256s": [state.canonical_sha256 for state in self.states],
            "states": [state.to_record() for state in self.states],
        }


def build_foundation_state(
    role: str,
    index: int,
    *,
    spec: FoundationSpec | None = None,
) -> FoundationState:
    spec = frozen_c4_spec() if spec is None else spec
    if not isinstance(spec, FoundationSpec) or not spec.is_frozen_c4:
        raise FoundationRecordError("C00 execution requires the exact frozen C4 spec")
    execution = seed_for_execution(role, index, roadmap_run=ROADMAP_RUN)
    identities = build_identity_bundle(
        execution.seed,
        parent_rectangle=spec.parent_rectangle,
        numerical_rectangle=spec.numerical_rectangle,
        analysis_rectangle=spec.development_analysis_rectangle,
    )
    sampler = StageSampler(
        identities.world_id,
        FOUNDATION_STAGE_ID,
        FOUNDATION_STAGE_VERSION,
        REGISTRATION_PROCESS_ID,
    )
    grids = {
        size: PhysicalGrid(spec.development_analysis_rectangle, size, size)
        for size in spec.supported_sizes
    }
    probes = tuple(
        sampler.probe_record(x_m, y_m)
        for x_m, y_m in REGISTERED_PROBES_M
    )
    context = build_formation_context(
        execution.seed,
        identities,
        spec.numerical_rectangle,
        upstream_sha256s=(identities.parent_geometry_id,),
    )
    cache = compute_cache_keys(
        identities,
        sampler,
        grids,
        consuming_stage_id=FOUNDATION_STAGE_ID,
        field_id=REGISTRATION_FIELD_ID,
    )
    return FoundationState(
        execution_seed=execution,
        spec=spec,
        identities=identities,
        sampler=sampler,
        sampling_grids=grids,
        registered_probe_records=probes,
        formation_context=context,
        cache_keys=cache,
    )


def build_development_cohort() -> tuple[FoundationState, ...]:
    """Execute every frozen development identity exactly once in manifest order."""

    plan = development_execution_plan()
    return tuple(build_foundation_state(item.role, item.index) for item in plan)


def build_development_cohort_state() -> DevelopmentCohortState:
    return DevelopmentCohortState(build_development_cohort())


__all__ = [
    "FoundationSpec",
    "FoundationState",
    "DevelopmentCohortState",
    "build_development_cohort",
    "build_development_cohort_state",
    "build_foundation_state",
    "frozen_c4_spec",
]
