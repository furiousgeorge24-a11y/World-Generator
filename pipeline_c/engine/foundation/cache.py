"""Canonical C4 cache keys and invalidation comparison."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ._util import content_sha256, freeze_json, require_hash, thaw_json
from .geometry import PhysicalGrid
from .identity import IdentityBundle, sampling_grid_id
from .prf import StageSampler


@dataclass(frozen=True, slots=True)
class FoundationCacheKeys:
    parent_geometry_key: str
    world_key: str
    parent_domain_key: str
    numerical_domain_key: str
    observer_window_key: str | None
    stage_key: str
    sampling_grid_keys: Mapping[str, str]
    resolution_audit_key: str
    evidence_key: str
    render_key: str
    render_settings: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in (
            "parent_geometry_key", "world_key", "parent_domain_key",
            "numerical_domain_key", "stage_key", "resolution_audit_key",
            "evidence_key", "render_key",
        ):
            require_hash(getattr(self, name), name)
        if self.observer_window_key is not None:
            require_hash(self.observer_window_key, "observer_window_key")
        if not isinstance(self.sampling_grid_keys, Mapping):
            raise TypeError("sampling_grid_keys must be a mapping")
        grids: dict[str, str] = {}
        for size, digest in self.sampling_grid_keys.items():
            if not isinstance(size, str) or not size.isdigit():
                raise ValueError("sampling-grid key names must be decimal sizes")
            require_hash(digest, f"sampling_grid_keys[{size}]")
            grids[size] = digest
        object.__setattr__(
            self, "sampling_grid_keys", MappingProxyType(dict(sorted(grids.items())))
        )
        settings = freeze_json(self.render_settings, "render_settings")
        if not isinstance(settings, Mapping):
            raise TypeError("render_settings must be a mapping")
        object.__setattr__(self, "render_settings", settings)

    def to_record(self) -> dict[str, object]:
        return {
            "evidence_key": self.evidence_key,
            "numerical_domain_key": self.numerical_domain_key,
            "observer_window_key": self.observer_window_key,
            "parent_domain_key": self.parent_domain_key,
            "parent_geometry_key": self.parent_geometry_key,
            "render_key": self.render_key,
            "render_settings": thaw_json(self.render_settings),
            "resolution_audit_key": self.resolution_audit_key,
            "sampling_grid_keys": dict(self.sampling_grid_keys),
            "stage_key": self.stage_key,
            "world_key": self.world_key,
        }

    def causal_record(self) -> dict[str, object]:
        """Return cache identities that may enter canonical foundation state."""

        return {
            "evidence_key": self.evidence_key,
            "numerical_domain_key": self.numerical_domain_key,
            "observer_window_key": self.observer_window_key,
            "parent_domain_key": self.parent_domain_key,
            "parent_geometry_key": self.parent_geometry_key,
            "resolution_audit_key": self.resolution_audit_key,
            "sampling_grid_keys": dict(self.sampling_grid_keys),
            "stage_key": self.stage_key,
            "world_key": self.world_key,
        }


def compute_cache_keys(
    identities: IdentityBundle,
    sampler: StageSampler,
    grids: Mapping[int, PhysicalGrid],
    *,
    consuming_stage_id: str,
    field_id: str,
    render_settings: Mapping[str, object] | None = None,
) -> FoundationCacheKeys:
    if not isinstance(identities, IdentityBundle):
        raise TypeError("identities must be IdentityBundle")
    if not isinstance(sampler, StageSampler):
        raise TypeError("sampler must be StageSampler")
    if sampler.world_id != identities.world_id:
        raise ValueError("sampler world does not match identity bundle")
    grid_keys: dict[str, str] = {}
    grid_records: dict[str, object] = {}
    for size, grid in sorted(grids.items()):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("grid mapping keys must be positive integer sizes")
        if not isinstance(grid, PhysicalGrid):
            raise TypeError("grid mapping values must be PhysicalGrid")
        key = sampling_grid_id(
            identities.world_id, consuming_stage_id, field_id, grid
        )
        grid_keys[str(size)] = key
        grid_records[str(size)] = grid.to_record()
    resolution_audit_key = content_sha256(
        {
            "grid_keys": grid_keys,
            "grid_records": grid_records,
            "schema_id": "urn:mapgen:pipeline-c:resolution-audit-key:v1",
            "schema_version": 1,
        }
    )
    evidence_key = content_sha256(
        {
            "identities": identities.to_record(),
            "resolution_audit_key": resolution_audit_key,
            "sampling_grid_keys": grid_keys,
            "schema_id": "urn:mapgen:pipeline-c:foundation-evidence-key:v1",
            "schema_version": 1,
            "stage_key": sampler.stage_key_sha256,
        }
    )
    normalized_render = {} if render_settings is None else render_settings
    frozen_render = freeze_json(normalized_render, "render_settings")
    if not isinstance(frozen_render, Mapping):
        raise TypeError("render_settings must be a mapping")
    render_key = content_sha256(
        {
            "evidence_key": evidence_key,
            "render_settings": thaw_json(frozen_render),
            "schema_id": "urn:mapgen:pipeline-c:foundation-render-key:v1",
            "schema_version": 1,
        }
    )
    return FoundationCacheKeys(
        parent_geometry_key=identities.parent_geometry_id,
        world_key=identities.world_id,
        parent_domain_key=identities.parent_domain_id,
        numerical_domain_key=identities.numerical_domain_id,
        observer_window_key=identities.development_analysis_window_id,
        stage_key=sampler.stage_key_sha256,
        sampling_grid_keys=grid_keys,
        resolution_audit_key=resolution_audit_key,
        evidence_key=evidence_key,
        render_key=render_key,
        render_settings=normalized_render,
    )


def changed_cache_keys(
    baseline: FoundationCacheKeys, current: FoundationCacheKeys
) -> tuple[str, ...]:
    if not isinstance(baseline, FoundationCacheKeys) or not isinstance(
        current, FoundationCacheKeys
    ):
        raise TypeError("baseline and current must be FoundationCacheKeys")
    changed: list[str] = []
    for name in (
        "parent_geometry_key", "world_key", "parent_domain_key",
        "numerical_domain_key", "observer_window_key", "stage_key",
        "resolution_audit_key", "evidence_key", "render_key",
    ):
        if getattr(baseline, name) != getattr(current, name):
            changed.append(name)
    sizes = sorted(
        set(baseline.sampling_grid_keys) | set(current.sampling_grid_keys),
        key=int,
    )
    changed.extend(
        f"sampling_grid_keys.{size}"
        for size in sizes
        if baseline.sampling_grid_keys.get(size) != current.sampling_grid_keys.get(size)
    )
    if baseline.render_settings != current.render_settings:
        changed.append("render_settings")
    return tuple(changed)


__all__ = ["FoundationCacheKeys", "changed_cache_keys", "compute_cache_keys"]
