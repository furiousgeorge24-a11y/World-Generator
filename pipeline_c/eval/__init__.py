"""Engine-independent evaluation infrastructure for pipeline_c."""

from .metrics import (
    FRAGMENTATION_MAX,
    LAND_TARGET_MAX_PERCENT,
    LAND_TARGET_MIN_PERCENT,
    LAND_TOLERANCE_PERCENTAGE_POINTS,
    fragmentation_metrics,
    land_fraction,
    land_percent,
    monotonic_land_percentages,
    outer_ring_is_water,
    target_interval_percent,
    target_within_tolerance_percent,
)

__all__ = [
    "FRAGMENTATION_MAX",
    "LAND_TARGET_MAX_PERCENT",
    "LAND_TARGET_MIN_PERCENT",
    "LAND_TOLERANCE_PERCENTAGE_POINTS",
    "fragmentation_metrics",
    "land_fraction",
    "land_percent",
    "monotonic_land_percentages",
    "outer_ring_is_water",
    "target_interval_percent",
    "target_within_tolerance_percent",
]
