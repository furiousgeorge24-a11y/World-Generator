"""C02 state construction from the accepted C4 world identity."""

from __future__ import annotations

from ..foundation import ExecutionSeed, IdentityBundle, PhysicalRect
from ._util import FabricRecordError, content_sha256
from .constants import REPRESENTATION_ID, TOPOLOGY_ID
from .context import build_fabric_context
from .growth import grow_affiliation
from .layout import build_primary_actor_layout, toroidal_squared_distance_m2
from .records import TectonicFabricState, frozen_c02_spec


def build_tectonic_fabric_state(
    execution_seed: ExecutionSeed,
    identities: IdentityBundle,
    parent_rectangle: PhysicalRect,
    *,
    candidate_order: str = "forward",
    source_order: str = "forward",
    neighbor_order: str = "canonical",
) -> TectonicFabricState:
    """Execute exactly one frozen C02 attempt for a debug/development identity."""

    if not isinstance(execution_seed, ExecutionSeed):
        raise TypeError("execution_seed must be ExecutionSeed")
    if execution_seed.role not in {"debug", "development"}:
        raise FabricRecordError("C02 cannot consume validation seeds")
    if not isinstance(identities, IdentityBundle):
        raise TypeError("identities must be IdentityBundle")
    if not isinstance(parent_rectangle, PhysicalRect):
        raise TypeError("parent_rectangle must be PhysicalRect")
    context = build_fabric_context(
        execution_seed.seed, identities, parent_rectangle
    )
    spec = frozen_c02_spec()
    controls, actors = build_primary_actor_layout(
        context.world_id, candidate_order=candidate_order
    )
    growth = grow_affiliation(
        context.world_id,
        actors,
        source_order=source_order,
        neighbor_order=neighbor_order,
    )
    actor_records = [actor.to_record() for actor in actors]
    layout_sha256 = content_sha256(
        {
            "actors": actor_records,
            "context_sha256": context.sha256,
            "controls": controls.to_record(),
            "schema_id": "urn:mapgen:pipeline-c:c02-layout-identity:v1",
            "schema_version": 1,
            "spec_formation_sha256": spec.formation_sha256,
        }
    )
    catalog_sha256 = content_sha256(
        {
            "actor_lineage_ids": [actor.lineage_id for actor in actors],
            "schema_id": "urn:mapgen:pipeline-c:c02-actor-catalog:v1",
            "schema_version": 1,
            "world_id": context.world_id,
        }
    )
    construction_sha256 = content_sha256(
        {
            "adjacency_signature_sha256": growth.adjacency_signature_sha256,
            "certificate": growth.certificate.to_record(),
            "layout_sha256": layout_sha256,
            "resistance_controls": growth.resistance_controls.to_record(),
            "arrival_summary": growth.arrival_summary.to_record(),
            "schema_id": "urn:mapgen:pipeline-c:c02-growth-construction:v1",
            "schema_version": 1,
        }
    )
    partition_sha256 = content_sha256(
        {
            "affiliation_sha256": growth.certificate.affiliation_sha256,
            "actor_id_lookup": [actor.lineage_id for actor in actors],
            "representation_id": REPRESENTATION_ID,
            "schema_id": "urn:mapgen:pipeline-c:c02-partition-identity:v1",
            "schema_version": 1,
            "topology_id": TOPOLOGY_ID,
        }
    )
    return TectonicFabricState(
        execution_seed=execution_seed,
        context=context,
        spec=spec,
        controls=controls,
        resistance_controls=growth.resistance_controls,
        arrival_summary=growth.arrival_summary,
        actors=actors,
        affiliation_bytes=growth.affiliation_bytes,
        arrival_times_bytes=growth.arrival_times_bytes,
        parent_indices_bytes=growth.parent_indices_bytes,
        source_mask_packed_bytes=growth.source_mask_packed_bytes,
        certificate=growth.certificate,
        layout_sha256=layout_sha256,
        catalog_sha256=catalog_sha256,
        construction_sha256=construction_sha256,
        partition_sha256=partition_sha256,
        adjacency_signature_sha256=growth.adjacency_signature_sha256,
    )


__all__ = [
    "build_tectonic_fabric_state",
    "toroidal_squared_distance_m2",
]
