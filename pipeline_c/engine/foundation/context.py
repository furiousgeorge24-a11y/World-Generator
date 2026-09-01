"""Exact formation allowlist and one-way C4 dependency graph."""

from __future__ import annotations

from dataclasses import dataclass

from ._util import (
    FoundationRecordError,
    content_sha256,
    require_hash,
    require_id,
    require_int,
    require_text,
)
from .constants import (
    CANONICAL_UNITS,
    COORDINATE_SYSTEM_ID,
    FORMATION_CONTEXT_SCHEMA_ID,
    FOUNDATION_STAGE_ID,
    FOUNDATION_STAGE_VERSION,
    KEY_SCHEDULE_ID,
    NUMERICAL_BOUNDARY_POLICY,
)
from .geometry import PhysicalRect
from .identity import IdentityBundle


class FormationIsolationError(FoundationRecordError):
    """Formation could receive observer, frame, selection, or undeclared state."""


@dataclass(frozen=True, slots=True)
class FormationContext:
    """The complete and exact C4 formation input allowlist."""

    seed: int
    world_id: str
    parent_geometry_id: str
    parent_domain_id: str
    numerical_domain_id: str
    numerical_extent: PhysicalRect
    consuming_stage_id: str
    consuming_stage_version: str
    upstream_sha256s: tuple[str, ...]
    coordinate_system_id: str = COORDINATE_SYSTEM_ID
    units: str = CANONICAL_UNITS
    foundation_stage_id: str = FOUNDATION_STAGE_ID
    foundation_stage_version: str = FOUNDATION_STAGE_VERSION
    rng_key_schedule_id: str = KEY_SCHEDULE_ID
    numerical_boundary_policy: str = NUMERICAL_BOUNDARY_POLICY
    schema_id: str = FORMATION_CONTEXT_SCHEMA_ID
    schema_version: int = 1

    def __post_init__(self) -> None:
        require_int(self.seed, "seed", minimum=0, maximum=2**32 - 1)
        for name in (
            "world_id",
            "parent_geometry_id",
            "parent_domain_id",
            "numerical_domain_id",
        ):
            require_hash(getattr(self, name), name)
        if not isinstance(self.numerical_extent, PhysicalRect):
            raise FormationIsolationError("numerical_extent must be PhysicalRect")
        for name in (
            "consuming_stage_id",
            "consuming_stage_version",
            "coordinate_system_id",
            "foundation_stage_id",
            "foundation_stage_version",
            "rng_key_schedule_id",
            "numerical_boundary_policy",
        ):
            require_id(getattr(self, name), name)
        require_text(self.schema_id, "schema_id")
        if self.coordinate_system_id != COORDINATE_SYSTEM_ID:
            raise FormationIsolationError("unsupported formation coordinate system")
        if self.units != CANONICAL_UNITS:
            raise FormationIsolationError("formation context must use integer metres")
        if self.foundation_stage_id != FOUNDATION_STAGE_ID:
            raise FormationIsolationError("formation context has wrong foundation stage")
        if self.foundation_stage_version != FOUNDATION_STAGE_VERSION:
            raise FormationIsolationError("formation context has wrong foundation version")
        if self.rng_key_schedule_id != KEY_SCHEDULE_ID:
            raise FormationIsolationError("formation context has wrong RNG schedule")
        if self.numerical_boundary_policy != NUMERICAL_BOUNDARY_POLICY:
            raise FormationIsolationError("formation context has wrong numerical policy")
        if self.schema_id != FORMATION_CONTEXT_SCHEMA_ID or self.schema_version != 1:
            raise FormationIsolationError("unsupported formation-context schema")
        if not isinstance(self.upstream_sha256s, tuple):
            raise FormationIsolationError("upstream_sha256s must be a tuple")
        for digest in self.upstream_sha256s:
            require_hash(digest, "upstream_sha256")
        if tuple(sorted(set(self.upstream_sha256s))) != self.upstream_sha256s:
            raise FormationIsolationError("upstream hashes must be sorted and unique")

    def to_record(self) -> dict[str, object]:
        return {
            "consuming_stage_id": self.consuming_stage_id,
            "consuming_stage_version": self.consuming_stage_version,
            "coordinate_system_id": self.coordinate_system_id,
            "foundation_stage_id": self.foundation_stage_id,
            "foundation_stage_version": self.foundation_stage_version,
            "numerical_boundary_policy": self.numerical_boundary_policy,
            "numerical_domain_id": self.numerical_domain_id,
            "numerical_extent": self.numerical_extent.to_record(),
            "parent_domain_id": self.parent_domain_id,
            "parent_geometry_id": self.parent_geometry_id,
            "rng_key_schedule_id": self.rng_key_schedule_id,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "units": self.units,
            "upstream_sha256s": list(self.upstream_sha256s),
            "world_id": self.world_id,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> "FormationContext":
        keys = {
            "consuming_stage_id", "consuming_stage_version",
            "coordinate_system_id", "foundation_stage_id",
            "foundation_stage_version", "numerical_boundary_policy",
            "numerical_domain_id", "numerical_extent", "parent_domain_id",
            "parent_geometry_id", "rng_key_schedule_id", "schema_id",
            "schema_version", "seed", "units", "upstream_sha256s", "world_id",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise FormationIsolationError(
                "formation-context record has unexpected or missing keys"
            )
        upstream = value["upstream_sha256s"]
        if not isinstance(upstream, list):
            raise FormationIsolationError("upstream_sha256s must be an array")
        kwargs = dict(value)
        kwargs["upstream_sha256s"] = tuple(upstream)
        kwargs["numerical_extent"] = PhysicalRect.from_record(value["numerical_extent"])
        return cls(**kwargs)


def build_formation_context(
    seed: int,
    identities: IdentityBundle,
    numerical_extent: PhysicalRect,
    *,
    consuming_stage_id: str = FOUNDATION_STAGE_ID,
    consuming_stage_version: str = FOUNDATION_STAGE_VERSION,
    upstream_sha256s: tuple[str, ...] = (),
) -> FormationContext:
    """Build context without accepting an observer/frame/selection argument."""

    if not isinstance(identities, IdentityBundle):
        raise TypeError("identities must be IdentityBundle")
    return FormationContext(
        seed=seed,
        world_id=identities.world_id,
        parent_geometry_id=identities.parent_geometry_id,
        parent_domain_id=identities.parent_domain_id,
        numerical_domain_id=identities.numerical_domain_id,
        numerical_extent=numerical_extent,
        consuming_stage_id=consuming_stage_id,
        consuming_stage_version=consuming_stage_version,
        upstream_sha256s=upstream_sha256s,
    )


@dataclass(frozen=True, slots=True)
class DependencyNode:
    stage_id: str
    stage_kind: str
    depends_on: tuple[str, ...]
    input_records: tuple[str, ...]
    output_records: tuple[str, ...]
    frame_access: str = "none"

    def __post_init__(self) -> None:
        require_id(self.stage_id, "stage_id")
        if self.stage_kind not in {"formation", "observer", "review"}:
            raise FormationIsolationError("unsupported dependency-node kind")
        if self.frame_access not in {"none", "observation_only"}:
            raise FormationIsolationError("unsupported frame access")
        if self.stage_kind == "formation" and self.frame_access != "none":
            raise FormationIsolationError("formation cannot observe a frame")
        for name in ("depends_on", "input_records", "output_records"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(set(values)) != len(values):
                raise FormationIsolationError(f"{name} must be a unique tuple")
            for item in values:
                require_id(item, f"{name} item")
        if not self.output_records:
            raise FormationIsolationError("dependency nodes must declare outputs")

    def to_record(self) -> dict[str, object]:
        return {
            "depends_on": list(self.depends_on),
            "frame_access": self.frame_access,
            "input_records": list(self.input_records),
            "output_records": list(self.output_records),
            "stage_id": self.stage_id,
            "stage_kind": self.stage_kind,
        }


def validate_dependency_graph(nodes: tuple[DependencyNode, ...]) -> tuple[DependencyNode, ...]:
    if not isinstance(nodes, tuple) or not nodes:
        raise FormationIsolationError("dependency graph must be a non-empty tuple")
    if any(not isinstance(node, DependencyNode) for node in nodes):
        raise FormationIsolationError("dependency graph contains a non-node")
    by_id = {node.stage_id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise FormationIsolationError("dependency stage IDs must be unique")
    output_owner: dict[str, str] = {}
    for node in nodes:
        missing = set(node.depends_on) - set(by_id)
        if missing:
            raise FormationIsolationError(
                f"stage {node.stage_id} has missing dependencies {sorted(missing)}"
            )
        for output in node.output_records:
            if output in output_owner:
                raise FormationIsolationError(f"record {output} has multiple producers")
            output_owner[output] = node.stage_id

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage_id: str) -> set[str]:
        if stage_id in visiting:
            raise FormationIsolationError("dependency graph contains a cycle")
        if stage_id in visited:
            return set()
        visiting.add(stage_id)
        ancestors: set[str] = set()
        for parent in by_id[stage_id].depends_on:
            ancestors.add(parent)
            ancestors.update(visit(parent))
        visiting.remove(stage_id)
        visited.add(stage_id)
        return ancestors

    all_ancestors: dict[str, set[str]] = {}
    for stage_id in sorted(by_id):
        visited.clear()
        all_ancestors[stage_id] = visit(stage_id)
    for node in nodes:
        ancestors = all_ancestors[node.stage_id]
        if node.stage_kind == "formation":
            feedback = sorted(
                stage_id
                for stage_id in ancestors
                if by_id[stage_id].stage_kind in {"observer", "review"}
                or by_id[stage_id].frame_access != "none"
            )
            if feedback:
                raise FormationIsolationError(
                    f"formation stage has observer/review feedback {feedback}"
                )
        available = {
            record
            for ancestor in ancestors
            for record in by_id[ancestor].output_records
        }
        missing_inputs = set(node.input_records) - available
        if missing_inputs:
            raise FormationIsolationError(
                f"stage {node.stage_id} lacks producers for {sorted(missing_inputs)}"
            )
    return nodes


def production_dependency_graph() -> tuple[DependencyNode, ...]:
    return validate_dependency_graph(
        (
            DependencyNode(
                stage_id=FOUNDATION_STAGE_ID,
                stage_kind="formation",
                depends_on=(),
                input_records=(),
                output_records=("foundation-state.v1",),
                frame_access="none",
            ),
            DependencyNode(
                stage_id="development-analysis-observer.v1",
                stage_kind="observer",
                depends_on=(FOUNDATION_STAGE_ID,),
                input_records=("foundation-state.v1",),
                output_records=("development-analysis-observation.v1",),
                frame_access="observation_only",
            ),
            DependencyNode(
                stage_id="foundation-review.v1",
                stage_kind="review",
                depends_on=(FOUNDATION_STAGE_ID, "development-analysis-observer.v1"),
                input_records=(
                    "foundation-state.v1",
                    "development-analysis-observation.v1",
                ),
                output_records=("foundation-review-record.v1",),
                frame_access="observation_only",
            ),
        )
    )


__all__ = [
    "DependencyNode",
    "FormationContext",
    "FormationIsolationError",
    "build_formation_context",
    "production_dependency_graph",
    "validate_dependency_graph",
]
