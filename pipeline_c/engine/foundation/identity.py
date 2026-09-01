"""Content-derived identity graph for the C4 world foundation."""

from __future__ import annotations

from dataclasses import dataclass

from ._util import (
    FoundationRecordError,
    content_sha256,
    require_hash,
    require_id,
    require_int,
)
from .constants import (
    ANALYSIS_ORIENTATION_DEGREES,
    CANONICAL_UNITS,
    COORDINATE_SYSTEM_ID,
    FOUNDATION_STAGE_ID,
    FOUNDATION_STAGE_VERSION,
    GEOMETRY_SCHEMA_ID,
    KEY_SCHEDULE_ID,
    NUMERICAL_BOUNDARY_POLICY,
    NUMERICAL_HALO_M,
    OBSERVER_PURPOSE,
    OBSERVER_VERSION,
    PARENT_DOMAIN_POLICY,
    RECTANGLE_SEMANTICS,
)
from .geometry import (
    DEVELOPMENT_ANALYSIS_RECT,
    NUMERICAL_RECT,
    PARENT_RECT,
    PhysicalGrid,
    PhysicalRect,
)


def _digest(schema_id: str, **payload: object) -> str:
    require_id(schema_id, "schema_id")
    return content_sha256(
        {
            "payload": payload,
            "schema_id": schema_id,
            "schema_version": 1,
        }
    )


def parent_geometry_record(
    rectangle: PhysicalRect = PARENT_RECT,
) -> dict[str, object]:
    if not isinstance(rectangle, PhysicalRect):
        raise TypeError("rectangle must be PhysicalRect")
    return {
        "coordinate_system_id": COORDINATE_SYSTEM_ID,
        "rectangle": rectangle.to_record(),
        "rectangle_semantics": RECTANGLE_SEMANTICS,
        "schema_id": GEOMETRY_SCHEMA_ID,
        "schema_version": 1,
        "units": CANONICAL_UNITS,
    }


def parent_geometry_id(rectangle: PhysicalRect = PARENT_RECT) -> str:
    return content_sha256(parent_geometry_record(rectangle))


def world_id(seed: int, geometry_id: str) -> str:
    require_int(seed, "seed", minimum=0, maximum=2**32 - 1)
    require_hash(geometry_id, "parent_geometry_id")
    return _digest(
        "pipeline-c-world-identity-v1",
        foundation_stage_id=FOUNDATION_STAGE_ID,
        foundation_stage_version=FOUNDATION_STAGE_VERSION,
        parent_geometry_id=geometry_id,
        rng_key_schedule_id=KEY_SCHEDULE_ID,
        seed=seed,
    )


def parent_domain_id(world_identity: str, geometry_id: str) -> str:
    require_hash(world_identity, "world_id")
    require_hash(geometry_id, "parent_geometry_id")
    return _digest(
        "pipeline-c-parent-domain-identity-v1",
        parent_domain_policy=PARENT_DOMAIN_POLICY,
        parent_geometry_id=geometry_id,
        world_id=world_identity,
    )


def numerical_halo_record(
    numerical_rectangle: PhysicalRect,
    parent_rectangle: PhysicalRect = PARENT_RECT,
) -> dict[str, int]:
    if not isinstance(numerical_rectangle, PhysicalRect):
        raise TypeError("numerical_rectangle must be PhysicalRect")
    if not isinstance(parent_rectangle, PhysicalRect):
        raise TypeError("parent_rectangle must be PhysicalRect")
    if not numerical_rectangle.contains_rect(parent_rectangle):
        raise FoundationRecordError("numerical extent must contain the parent extent")
    return {
        "bottom_m": parent_rectangle.min_y_m - numerical_rectangle.min_y_m,
        "left_m": parent_rectangle.min_x_m - numerical_rectangle.min_x_m,
        "right_m": numerical_rectangle.max_x_m - parent_rectangle.max_x_m,
        "top_m": numerical_rectangle.max_y_m - parent_rectangle.max_y_m,
    }


def numerical_domain_id(
    parent_identity: str,
    numerical_rectangle: PhysicalRect = NUMERICAL_RECT,
    *,
    parent_rectangle: PhysicalRect = PARENT_RECT,
    policy: str = NUMERICAL_BOUNDARY_POLICY,
) -> str:
    require_hash(parent_identity, "parent_domain_id")
    require_id(policy, "numerical_domain_policy")
    halo = numerical_halo_record(numerical_rectangle, parent_rectangle)
    return _digest(
        "pipeline-c-numerical-domain-identity-v1",
        halo=halo,
        numerical_domain_policy=policy,
        numerical_rectangle=numerical_rectangle.to_record(),
        parent_domain_id=parent_identity,
    )


def development_analysis_window_id(
    parent_identity: str,
    rectangle: PhysicalRect = DEVELOPMENT_ANALYSIS_RECT,
    *,
    orientation_degrees: int = ANALYSIS_ORIENTATION_DEGREES,
    purpose: str = OBSERVER_PURPOSE,
    observer_version: str = OBSERVER_VERSION,
) -> str:
    require_hash(parent_identity, "parent_domain_id")
    if not isinstance(rectangle, PhysicalRect):
        raise TypeError("rectangle must be PhysicalRect")
    if not PARENT_RECT.contains_rect(rectangle):
        raise FoundationRecordError("development analysis window must lie in parent")
    require_int(orientation_degrees, "orientation_degrees", minimum=-360, maximum=360)
    require_id(purpose, "observer purpose")
    require_id(observer_version, "observer version")
    return _digest(
        "pipeline-c-development-analysis-window-identity-v1",
        observer_purpose=purpose,
        observer_version=observer_version,
        orientation_degrees=orientation_degrees,
        parent_domain_id=parent_identity,
        rectangle=rectangle.to_record(),
    )


def sampling_grid_id(
    world_identity: str,
    consuming_stage_id: str,
    field_id: str,
    grid: PhysicalGrid,
) -> str:
    require_hash(world_identity, "world_id")
    require_id(consuming_stage_id, "consuming_stage_id")
    require_id(field_id, "field_id")
    if not isinstance(grid, PhysicalGrid):
        raise TypeError("grid must be PhysicalGrid")
    return _digest(
        "pipeline-c-sampling-grid-identity-v1",
        consuming_stage_id=consuming_stage_id,
        field_id=field_id,
        grid=grid.to_record(),
        world_id=world_identity,
    )


@dataclass(frozen=True, slots=True)
class IdentityBundle:
    parent_geometry_id: str
    world_id: str
    parent_domain_id: str
    numerical_domain_id: str
    development_analysis_window_id: str | None

    def __post_init__(self) -> None:
        for name in (
            "parent_geometry_id",
            "world_id",
            "parent_domain_id",
            "numerical_domain_id",
        ):
            require_hash(getattr(self, name), name)
        if self.development_analysis_window_id is not None:
            require_hash(
                self.development_analysis_window_id,
                "development_analysis_window_id",
            )

    def to_record(self) -> dict[str, object]:
        return {
            "development_analysis_window_id": self.development_analysis_window_id,
            "numerical_domain_id": self.numerical_domain_id,
            "parent_domain_id": self.parent_domain_id,
            "parent_geometry_id": self.parent_geometry_id,
            "world_id": self.world_id,
        }


def build_identity_bundle(
    seed: int,
    *,
    parent_rectangle: PhysicalRect = PARENT_RECT,
    numerical_rectangle: PhysicalRect = NUMERICAL_RECT,
    analysis_rectangle: PhysicalRect | None = DEVELOPMENT_ANALYSIS_RECT,
) -> IdentityBundle:
    geometry = parent_geometry_id(parent_rectangle)
    world = world_id(seed, geometry)
    parent = parent_domain_id(world, geometry)
    numerical = numerical_domain_id(
        parent,
        numerical_rectangle,
        parent_rectangle=parent_rectangle,
    )
    observer = (
        None
        if analysis_rectangle is None
        else development_analysis_window_id(parent, analysis_rectangle)
    )
    return IdentityBundle(geometry, world, parent, numerical, observer)


def frozen_geometry_is_exact() -> bool:
    halo = numerical_halo_record(NUMERICAL_RECT, PARENT_RECT)
    return (
        halo == {
            "bottom_m": NUMERICAL_HALO_M,
            "left_m": NUMERICAL_HALO_M,
            "right_m": NUMERICAL_HALO_M,
            "top_m": NUMERICAL_HALO_M,
        }
        and PARENT_RECT.contains_rect(DEVELOPMENT_ANALYSIS_RECT)
        and PARENT_RECT.width_m == 4 * DEVELOPMENT_ANALYSIS_RECT.width_m
        and PARENT_RECT.height_m == 4 * DEVELOPMENT_ANALYSIS_RECT.height_m
    )


__all__ = [
    "IdentityBundle",
    "build_identity_bundle",
    "development_analysis_window_id",
    "frozen_geometry_is_exact",
    "numerical_domain_id",
    "numerical_halo_record",
    "parent_domain_id",
    "parent_geometry_id",
    "parent_geometry_record",
    "sampling_grid_id",
    "world_id",
]
