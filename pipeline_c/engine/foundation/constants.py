"""Frozen constants for C00 / Run C4."""

from __future__ import annotations


ROADMAP_RUN = "C4"
ATTEMPT_ID = "C00"
FOUNDATION_STAGE_ID = "world_foundation.v1"
FOUNDATION_STAGE_VERSION = "1"
COMPARISON_FAMILY_ID = "c4-world-foundation-v1"
DISPLAY_LABEL = "WORLD FOUNDATION — NO GEOLOGY OR LAND"
EVIDENCE_KIND = "engine_foundation"

COORDINATE_SYSTEM_ID = "pipeline-c-planar-metres-v1"
CANONICAL_UNITS = "m"
DISPLAY_UNITS = "km"
SAMPLE_LOCATION = "cell_center"
ROW_ORIENTATION = "row_zero_greatest_y"
RECTANGLE_SEMANTICS = "half_open"

KEY_SCHEDULE_ID = "pipeline-c-sha256-address-prf-v1"
REGISTRATION_PROCESS_ID = "physical-registration"
REGISTRATION_FIELD_ID = "registered-physical-probes.v1"
NUMERICAL_BOUNDARY_POLICY = "extent_only_no_formation_boundary_v1"
PARENT_DOMAIN_POLICY = "finite_parent_extent_v1"
OBSERVER_PURPOSE = "c4-development-analysis"
OBSERVER_VERSION = "1"

PARENT_MIN_X_M = 0
PARENT_MIN_Y_M = 0
PARENT_WIDTH_M = 40_960_000
PARENT_HEIGHT_M = 40_960_000

NUMERICAL_MIN_X_M = -5_120_000
NUMERICAL_MIN_Y_M = -5_120_000
NUMERICAL_WIDTH_M = 51_200_000
NUMERICAL_HEIGHT_M = 51_200_000
NUMERICAL_HALO_M = 5_120_000

ANALYSIS_MIN_X_M = 15_360_000
ANALYSIS_MIN_Y_M = 15_360_000
ANALYSIS_WIDTH_M = 10_240_000
ANALYSIS_HEIGHT_M = 10_240_000
ANALYSIS_ORIENTATION_DEGREES = 0

SUPPORTED_SIZES = (512, 1024)
DEFAULT_SIZE = 1024
REGISTERED_PROBE_AXIS_M = (17_920_000, 20_480_000, 23_040_000)
REGISTERED_PROBES_M = tuple(
    (x_m, y_m)
    for y_m in REGISTERED_PROBE_AXIS_M
    for x_m in REGISTERED_PROBE_AXIS_M
)

FOUNDATION_SCHEMA_ID = "urn:mapgen:pipeline-c:foundation-state:v1"
GEOMETRY_SCHEMA_ID = "urn:mapgen:pipeline-c:physical-geometry:v1"
GRID_SCHEMA_ID = "urn:mapgen:pipeline-c:physical-grid:v1"
FORMATION_CONTEXT_SCHEMA_ID = "urn:mapgen:pipeline-c:formation-context:v1"
COHORT_SCHEMA_ID = "urn:mapgen:pipeline-c:cohort:v1"


__all__ = [name for name in globals() if name.isupper()]
