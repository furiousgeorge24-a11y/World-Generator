"""Causal C02 cache DAG and exact invalidation comparison."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..foundation import PARENT_RECT, PhysicalGrid, StageSampler
from ._util import (
    FabricRecordError,
    content_sha256,
    freeze_json,
    require_hash,
    thaw_json,
)
from .constants import (
    CACHE_SCHEMA_ID,
    NUCLEUS_PROCESS_ID,
    PARENT_CENSUS_SIZE,
    STAGE_ID,
    STAGE_VERSION,
)
from .records import TectonicFabricState


@dataclass(frozen=True, slots=True)
class FabricCacheKeys:
    fabric_context_key: str
    specification_key: str
    layout_controls_key: str
    nucleus_candidates_key: str
    actor_layout_key: str
    actor_catalog_key: str
    resistance_key: str
    construction_key: str
    affiliation_key: str
    certificate_key: str
    parent_census_key: str
    metrics_key: str
    observation_keys: Mapping[str, str]
    evidence_key: str
    render_key: str
    render_settings: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in (
            "fabric_context_key",
            "specification_key",
            "layout_controls_key",
            "nucleus_candidates_key",
            "actor_layout_key",
            "actor_catalog_key",
            "resistance_key",
            "construction_key",
            "affiliation_key",
            "certificate_key",
            "parent_census_key",
            "metrics_key",
            "evidence_key",
            "render_key",
        ):
            require_hash(getattr(self, name), name)
        if not isinstance(self.observation_keys, Mapping):
            raise FabricRecordError("observation_keys must be a mapping")
        observations: dict[str, str] = {}
        for name, digest in self.observation_keys.items():
            if not isinstance(name, str) or not name:
                raise FabricRecordError("observation cache names must be non-empty")
            require_hash(digest, f"observation_keys[{name}]")
            observations[name] = digest
        object.__setattr__(
            self,
            "observation_keys",
            MappingProxyType(dict(sorted(observations.items()))),
        )
        settings = freeze_json(self.render_settings, "render_settings")
        if not isinstance(settings, Mapping):
            raise FabricRecordError("render_settings must be a mapping")
        object.__setattr__(self, "render_settings", settings)

    def to_record(self) -> dict[str, object]:
        return {
            "actor_catalog_key": self.actor_catalog_key,
            "actor_layout_key": self.actor_layout_key,
            "affiliation_key": self.affiliation_key,
            "certificate_key": self.certificate_key,
            "construction_key": self.construction_key,
            "evidence_key": self.evidence_key,
            "fabric_context_key": self.fabric_context_key,
            "layout_controls_key": self.layout_controls_key,
            "metrics_key": self.metrics_key,
            "nucleus_candidates_key": self.nucleus_candidates_key,
            "observation_keys": dict(self.observation_keys),
            "parent_census_key": self.parent_census_key,
            "render_key": self.render_key,
            "render_settings": thaw_json(self.render_settings),
            "resistance_key": self.resistance_key,
            "schema_id": CACHE_SCHEMA_ID,
            "schema_version": 1,
            "specification_key": self.specification_key,
        }

    def semantic_record(self) -> dict[str, object]:
        return {
            "actor_catalog_key": self.actor_catalog_key,
            "actor_layout_key": self.actor_layout_key,
            "affiliation_key": self.affiliation_key,
            "certificate_key": self.certificate_key,
            "construction_key": self.construction_key,
            "fabric_context_key": self.fabric_context_key,
            "layout_controls_key": self.layout_controls_key,
            "nucleus_candidates_key": self.nucleus_candidates_key,
            "resistance_key": self.resistance_key,
            "specification_key": self.specification_key,
        }


def compute_fabric_cache_keys(
    state: TectonicFabricState,
    observation_grids: Mapping[str, PhysicalGrid],
    *,
    render_settings: Mapping[str, object] | None = None,
) -> FabricCacheKeys:
    if not isinstance(state, TectonicFabricState):
        raise TypeError("state must be TectonicFabricState")
    if not isinstance(observation_grids, Mapping):
        raise TypeError("observation_grids must be a mapping")
    candidate_sampler = StageSampler(
        state.context.world_id, STAGE_ID, STAGE_VERSION, NUCLEUS_PROCESS_ID
    )
    candidate_key = content_sha256(
        {
            "context_sha256": state.context.sha256,
            "coordinate_mapping": state.spec.coordinate_mapping,
            "count_per_actor": state.spec.candidates_per_actor,
            "process_id": NUCLEUS_PROCESS_ID,
            "schema_id": "urn:mapgen:pipeline-c:c02-nucleus-candidate-cache:v1",
            "schema_version": 1,
            "stage_key_sha256": candidate_sampler.stage_key_sha256,
        }
    )
    parent_grid = PhysicalGrid(PARENT_RECT, PARENT_CENSUS_SIZE, PARENT_CENSUS_SIZE)
    parent_census = content_sha256(
        {
            "affiliation_sha256": state.certificate.affiliation_sha256,
            "grid": parent_grid.to_record(),
            "readout_policy": state.spec.owner_readout_policy,
            "schema_id": "urn:mapgen:pipeline-c:c02-parent-census-cache:v1",
            "schema_version": 1,
        }
    )
    observations: dict[str, str] = {}
    for name, grid in sorted(observation_grids.items()):
        if not isinstance(name, str) or not name:
            raise FabricRecordError("observation grid names must be non-empty")
        if not isinstance(grid, PhysicalGrid):
            raise TypeError("observation grid values must be PhysicalGrid")
        observations[name] = content_sha256(
            {
                "affiliation_sha256": state.certificate.affiliation_sha256,
                "grid": grid.to_record(),
                "readout_policy": state.spec.owner_readout_policy,
                "schema_id": "urn:mapgen:pipeline-c:c02-observation-cache:v1",
                "schema_version": 1,
            }
        )
    metrics = content_sha256(
        {
            "adjacency_signature_sha256": state.adjacency_signature_sha256,
            "canonical_affiliation_sha256": state.certificate.affiliation_sha256,
            "definitions_schema": "urn:mapgen:pipeline-c:c02-morphology-metrics:v1",
            "schema_id": "urn:mapgen:pipeline-c:c02-metrics-cache:v1",
            "schema_version": 1,
        }
    )
    evidence = content_sha256(
        {
            "metrics_key": metrics,
            "observation_keys": observations,
            "parent_census_key": parent_census,
            "state_sha256": state.canonical_sha256,
            "schema_id": "urn:mapgen:pipeline-c:c02-evidence-cache:v1",
            "schema_version": 1,
        }
    )
    normalized_render = {} if render_settings is None else render_settings
    frozen_render = freeze_json(normalized_render, "render_settings")
    render = content_sha256(
        {
            "evidence_key": evidence,
            "render_settings": thaw_json(frozen_render),
            "schema_id": "urn:mapgen:pipeline-c:c02-render-cache:v1",
            "schema_version": 1,
        }
    )
    return FabricCacheKeys(
        fabric_context_key=state.context.sha256,
        specification_key=state.spec.formation_sha256,
        layout_controls_key=state.controls.sha256,
        nucleus_candidates_key=candidate_key,
        actor_layout_key=state.layout_sha256,
        actor_catalog_key=state.catalog_sha256,
        resistance_key=state.resistance_controls.sha256,
        construction_key=state.construction_sha256,
        affiliation_key=state.certificate.affiliation_sha256,
        certificate_key=state.certificate.canonical_sha256,
        parent_census_key=parent_census,
        metrics_key=metrics,
        observation_keys=observations,
        evidence_key=evidence,
        render_key=render,
        render_settings=normalized_render,
    )


def changed_fabric_cache_keys(
    baseline: FabricCacheKeys,
    current: FabricCacheKeys,
) -> tuple[str, ...]:
    if not isinstance(baseline, FabricCacheKeys) or not isinstance(current, FabricCacheKeys):
        raise TypeError("baseline and current must be FabricCacheKeys")
    names = (
        "fabric_context_key",
        "specification_key",
        "layout_controls_key",
        "nucleus_candidates_key",
        "actor_layout_key",
        "actor_catalog_key",
        "resistance_key",
        "construction_key",
        "affiliation_key",
        "certificate_key",
        "parent_census_key",
        "metrics_key",
        "evidence_key",
        "render_key",
    )
    changed = [name for name in names if getattr(baseline, name) != getattr(current, name)]
    for name in sorted(set(baseline.observation_keys) | set(current.observation_keys)):
        if baseline.observation_keys.get(name) != current.observation_keys.get(name):
            changed.append(f"observation_keys.{name}")
    if baseline.render_settings != current.render_settings:
        changed.append("render_settings")
    return tuple(changed)


__all__ = [
    "FabricCacheKeys",
    "changed_fabric_cache_keys",
    "compute_fabric_cache_keys",
]
