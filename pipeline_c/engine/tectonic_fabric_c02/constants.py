"""Frozen constants for C02 / Run C5 connected macro affiliation."""

from __future__ import annotations


ROADMAP_RUN = "C5"
ATTEMPT_ID = "C02"
STAGE_ID = "tectonic_fabric.v2"
STAGE_VERSION = "2"
COMPARISON_FAMILY_ID = "c5-initial-tectonic-fabric-v1"
EVIDENCE_KIND = "engine_tectonic_fabric"
CASE_ID = "c5-c02-development-cohort-v1"
DISPLAY_LABEL = (
    "CONNECTED MACRO AFFILIATION — NO KINEMATICS, ELEVATION, WATER, OR LAND"
)

TOPOLOGY_ID = "flat-parent-torus-v1"
REPRESENTATION_ID = "connected-competitive-growth-affiliation-v2"
LINEAGE_SCHEMA_ID = "actor-lineage-v1"
COORDINATE_MAPPING = "uint64-multiply-high-v1"
OWNER_READOUT_POLICY = "canonical-containing-cell-floor-modulo-v1"
EVENT_TIE_POLICY = "arrival-rank-destination-actor-parent-v1"

PRIMARY_ACTOR_COUNT = 7
CANONICAL_SIZE = 1024
CANONICAL_CELL_M = 40_000
PARENT_SIDE_M = 40_960_000
CANDIDATES_PER_ACTOR = 512
GERM_HALF_STEPS = 16
GERM_CELL_COUNT = 33
GERM_ENDPOINT_SPAN_M = 1_280_000
STABILITY_CARDINAL_OFFSET_M = 1_280_000
STABILITY_DIAGONAL_OFFSET_M = 905_097
PARENT_CENSUS_SIZE = 512
SUPPORTED_OBSERVATION_SIZES = (512, 1024)

FAMILY_NAMES = ("scatter", "belt", "dual_focus", "arc_void")
FAMILY_MINIMUM_SEPARATION_M = (
    3_686_400,
    2_662_400,
    2_867_200,
    2_457_600,
)

LAYOUT_PROCESS_ID = "growth-layout-controls"
NUCLEUS_PROCESS_ID = "growth-nucleus-candidates"
DIRECTION_PROCESS_ID = "growth-directions"
TIE_ORDER_PROCESS_ID = "growth-tie-order"
RESISTANCE_PROCESS_ID = "growth-resistance"

RESISTANCE_MODES = ((1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2))
RESISTANCE_BASE = 32
RESISTANCE_AMPLITUDES = (160, 96)
RESISTANCE_PREVIEW_SIZE = 128
STEP_BASE_COST = 256
PARALLEL_DIRECTIONAL_COST = 192
PERPENDICULAR_DIRECTIONAL_COST = 896
BACKWARD_SIGN_PENALTY = 128

CROWDING_TARGET_DISTANCE_M = 5_120_000
CROWDING_BONUS_PER_CELL = 128

MIN_ACTOR_AREA_PERCENT = 2
MAX_ACTOR_AREA_PERCENT = 30
MIN_LARGEST_ACTOR_PERCENT = 18
MIN_HIERARCHY_NUMERATOR = 3
MIN_HIERARCHY_DENOMINATOR = 2
MAX_HIERARCHY = 8
MIN_NUCLEUS_NEIGHBOR_CV = 0.18
MIN_ASPECT_RATIO = 3.0
MIN_BELT_ASPECT_RATIO = 4.0
MIN_ASPECT_WORLD_COUNT = 4
MIN_CONTACT_PAIR_COUNT = 12
MAX_CONTACT_PAIR_COUNT = 20
MAX_COMPACTNESS_PENALTY = 5.0
MIN_EROSION_RETENTION = 0.90
MIN_TOTAL_MEAN_ENDPOINT_AGREEMENT = 0.80
MIN_ACTOR_MEAN_ENDPOINT_AGREEMENT = 0.65
MIN_PAIR_DISAGREEMENT = 0.15
MAX_ADJUSTED_RAND = 0.98

FROZEN_DEVELOPMENT_FAMILY_IDS = (1, 2, 3, 0, 3, 3, 0, 1, 0, 2, 0, 0)

FABRIC_SPEC_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-fabric-spec:v1"
FABRIC_CONTEXT_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-fabric-context:v1"
LAYOUT_CONTROL_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-layout-controls:v1"
RESISTANCE_CONTROL_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-resistance-controls:v1"
ARRIVAL_SUMMARY_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-arrival-summary:v1"
ACTOR_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-primary-actor:v1"
CERTIFICATE_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-growth-certificate:v1"
FABRIC_STATE_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-fabric-state:v1"
OBSERVATION_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-observation:v1"
CANONICAL_OBSERVATION_SCHEMA_ID = (
    "urn:mapgen:pipeline-c:c02-canonical-affiliation:v1"
)
CENSUS_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-census-metrics:v1"
MORPHOLOGY_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-morphology-metrics:v1"
DIVERSITY_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-diversity:v1"
COHORT_MEMBER_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-cohort-member:v1"
COHORT_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-development-cohort:v1"
CACHE_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-cache-keys:v1"
AUDIT_SCHEMA_ID = "urn:mapgen:pipeline-c:c02-audit:v1"


__all__ = [name for name in globals() if name.isupper()]
