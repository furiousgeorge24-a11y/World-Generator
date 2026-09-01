"""Exact observer-free formation input allowlist for C02."""

from __future__ import annotations

from dataclasses import dataclass

from ..foundation import (
    FOUNDATION_STAGE_ID,
    FOUNDATION_STAGE_VERSION,
    IdentityBundle,
    PARENT_RECT,
    PhysicalRect,
    parent_domain_id,
    parent_geometry_id,
    world_id,
)
from ._util import FabricRecordError, content_sha256, require_hash, require_int
from .constants import FABRIC_CONTEXT_SCHEMA_ID, STAGE_ID, STAGE_VERSION


class FabricIsolationError(FabricRecordError):
    """Formation was offered an inconsistent C4 physical identity."""


@dataclass(frozen=True, slots=True)
class FabricFormationContext:
    seed: int
    world_id: str
    parent_geometry_id: str
    parent_domain_id: str
    parent_rectangle: PhysicalRect
    consuming_stage_id: str = STAGE_ID
    consuming_stage_version: str = STAGE_VERSION
    foundation_stage_id: str = FOUNDATION_STAGE_ID
    foundation_stage_version: str = FOUNDATION_STAGE_VERSION
    coordinate_system_id: str = "pipeline-c-planar-metres-v1"
    rng_key_schedule_id: str = "pipeline-c-sha256-address-prf-v1"
    units: str = "m"
    topology: str = "flat_torus_xy"
    schema_id: str = FABRIC_CONTEXT_SCHEMA_ID
    schema_version: int = 1

    def __post_init__(self) -> None:
        require_int(self.seed, "seed", minimum=0, maximum=2**32 - 1)
        for name in ("world_id", "parent_geometry_id", "parent_domain_id"):
            require_hash(getattr(self, name), name)
        if self.parent_rectangle != PARENT_RECT:
            raise FabricIsolationError("C02 requires the exact C4 parent rectangle")
        if self.consuming_stage_id != STAGE_ID or self.consuming_stage_version != STAGE_VERSION:
            raise FabricIsolationError("C02 consuming-stage identity is inconsistent")
        if (
            self.foundation_stage_id != FOUNDATION_STAGE_ID
            or self.foundation_stage_version != FOUNDATION_STAGE_VERSION
        ):
            raise FabricIsolationError("C02 foundation-stage identity is inconsistent")
        if (
            self.coordinate_system_id != "pipeline-c-planar-metres-v1"
            or self.rng_key_schedule_id != "pipeline-c-sha256-address-prf-v1"
            or self.units != "m"
            or self.topology != "flat_torus_xy"
        ):
            raise FabricIsolationError("C02 requires integer-metre flat-torus context")
        if self.schema_id != FABRIC_CONTEXT_SCHEMA_ID or self.schema_version != 1:
            raise FabricIsolationError("unsupported C02 context schema")

    def to_record(self) -> dict[str, object]:
        return {
            "consuming_stage_id": self.consuming_stage_id,
            "consuming_stage_version": self.consuming_stage_version,
            "coordinate_system_id": self.coordinate_system_id,
            "foundation_stage_id": self.foundation_stage_id,
            "foundation_stage_version": self.foundation_stage_version,
            "parent_domain_id": self.parent_domain_id,
            "parent_geometry_id": self.parent_geometry_id,
            "parent_rectangle": self.parent_rectangle.to_record(),
            "rng_key_schedule_id": self.rng_key_schedule_id,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "topology": self.topology,
            "units": self.units,
            "world_id": self.world_id,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_record())


def build_fabric_context(
    seed: int,
    identities: IdentityBundle,
    parent_rectangle: PhysicalRect,
) -> FabricFormationContext:
    require_int(seed, "seed", minimum=0, maximum=2**32 - 1)
    if not isinstance(identities, IdentityBundle):
        raise TypeError("identities must be IdentityBundle")
    if not isinstance(parent_rectangle, PhysicalRect):
        raise TypeError("parent_rectangle must be PhysicalRect")
    if parent_rectangle != PARENT_RECT:
        raise FabricIsolationError("C02 requires the exact frozen C4 parent rectangle")
    expected_geometry = parent_geometry_id(parent_rectangle)
    expected_world = world_id(seed, expected_geometry)
    expected_parent = parent_domain_id(expected_world, expected_geometry)
    if (
        identities.parent_geometry_id != expected_geometry
        or identities.world_id != expected_world
        or identities.parent_domain_id != expected_parent
    ):
        raise FabricIsolationError(
            "seed, parent rectangle, and C4 world/parent identities are inconsistent"
        )
    return FabricFormationContext(
        seed=seed,
        world_id=identities.world_id,
        parent_geometry_id=identities.parent_geometry_id,
        parent_domain_id=identities.parent_domain_id,
        parent_rectangle=parent_rectangle,
    )


def build_fabric_context_with_poison_audit(
    seed: int,
    identities: IdentityBundle,
    parent_rectangle: PhysicalRect,
    *,
    numerical_state: object,
    observer_state: object,
    frame_state: object,
    control_state: object,
    render_state: object,
    target_state: object = None,
    fragmentation_state: object = None,
) -> FabricFormationContext:
    """Poison values are deliberately accepted by the audit harness and ignored."""

    return build_fabric_context(seed, identities, parent_rectangle)


__all__ = [
    "FabricFormationContext",
    "FabricIsolationError",
    "build_fabric_context",
    "build_fabric_context_with_poison_audit",
]
