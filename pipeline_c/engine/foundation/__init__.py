"""Production C4 parent-world foundation, with no geology or land.

The package exposes immutable physical-domain records, content identities,
the stateless address PRF, cohort guards, an exact formation allowlist, cache
keys, canonical state construction, and deterministic audits.  It deliberately
contains no latent raster, map, tectonic state, exposure, or delivery logic.
"""

from .audit import (
    AuditResult,
    audit_bundle_record,
    build_formation_context_with_poison_audit,
    run_foundation_audits,
)
from .cache import FoundationCacheKeys, changed_cache_keys, compute_cache_keys
from .cohorts import (
    COHORT_MANIFEST,
    COHORT_MANIFEST_SHA256,
    CohortManifest,
    ExecutionSeed,
    ValidationAccessError,
    debug_execution_plan,
    derive_seed,
    development_execution_plan,
    seed_for_execution,
)
from .constants import (
    ATTEMPT_ID,
    COMPARISON_FAMILY_ID,
    DEFAULT_SIZE,
    DISPLAY_LABEL,
    EVIDENCE_KIND,
    FOUNDATION_STAGE_ID,
    FOUNDATION_STAGE_VERSION,
    REGISTERED_PROBES_M,
    ROADMAP_RUN,
    SUPPORTED_SIZES,
)
from .context import (
    DependencyNode,
    FormationContext,
    FormationIsolationError,
    build_formation_context,
    production_dependency_graph,
    validate_dependency_graph,
)
from .geometry import (
    DEVELOPMENT_ANALYSIS_RECT,
    NUMERICAL_RECT,
    PARENT_RECT,
    PhysicalGrid,
    PhysicalRect,
    analysis_grid,
    exact_nested_ratio,
)
from .identity import (
    IdentityBundle,
    build_identity_bundle,
    development_analysis_window_id,
    numerical_domain_id,
    numerical_halo_record,
    parent_domain_id,
    parent_geometry_id,
    sampling_grid_id,
    world_id,
)
from .prf import SampleAddress, StageSampler
from .state import (
    DevelopmentCohortState,
    FoundationSpec,
    FoundationState,
    build_development_cohort,
    build_development_cohort_state,
    build_foundation_state,
    frozen_c4_spec,
)


__all__ = [
    "ATTEMPT_ID",
    "AuditResult",
    "COHORT_MANIFEST",
    "COHORT_MANIFEST_SHA256",
    "COMPARISON_FAMILY_ID",
    "CohortManifest",
    "DEFAULT_SIZE",
    "DEVELOPMENT_ANALYSIS_RECT",
    "DevelopmentCohortState",
    "DISPLAY_LABEL",
    "DependencyNode",
    "EVIDENCE_KIND",
    "ExecutionSeed",
    "FOUNDATION_STAGE_ID",
    "FOUNDATION_STAGE_VERSION",
    "FormationContext",
    "FormationIsolationError",
    "FoundationCacheKeys",
    "FoundationSpec",
    "FoundationState",
    "IdentityBundle",
    "NUMERICAL_RECT",
    "PARENT_RECT",
    "PhysicalGrid",
    "PhysicalRect",
    "REGISTERED_PROBES_M",
    "ROADMAP_RUN",
    "SUPPORTED_SIZES",
    "SampleAddress",
    "StageSampler",
    "ValidationAccessError",
    "analysis_grid",
    "audit_bundle_record",
    "build_development_cohort",
    "build_development_cohort_state",
    "build_formation_context",
    "build_formation_context_with_poison_audit",
    "build_foundation_state",
    "build_identity_bundle",
    "changed_cache_keys",
    "compute_cache_keys",
    "debug_execution_plan",
    "derive_seed",
    "development_analysis_window_id",
    "development_execution_plan",
    "exact_nested_ratio",
    "frozen_c4_spec",
    "numerical_domain_id",
    "numerical_halo_record",
    "parent_domain_id",
    "parent_geometry_id",
    "production_dependency_graph",
    "run_foundation_audits",
    "sampling_grid_id",
    "seed_for_execution",
    "validate_dependency_graph",
    "world_id",
]
