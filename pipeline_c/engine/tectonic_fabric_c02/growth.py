"""Exact deterministic claim-once competitive growth for C02."""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..foundation import StageSampler
from ._util import FabricFormationError, content_sha256, require_hash
from .constants import (
    BACKWARD_SIGN_PENALTY,
    CANONICAL_CELL_M,
    CANONICAL_SIZE,
    PARALLEL_DIRECTIONAL_COST,
    PARENT_SIDE_M,
    PERPENDICULAR_DIRECTIONAL_COST,
    PRIMARY_ACTOR_COUNT,
    RESISTANCE_AMPLITUDES,
    RESISTANCE_BASE,
    RESISTANCE_MODES,
    RESISTANCE_PROCESS_ID,
    RESISTANCE_PREVIEW_SIZE,
    STAGE_ID,
    STAGE_VERSION,
    STEP_BASE_COST,
)
from .layout import multiply_high_axis
from .records import (
    ActorArrivalSummary,
    ArrivalSummaryRecord,
    GrowthCertificate,
    PrimaryActorRecord,
    ResistanceControlRecord,
)


_CELL_COUNT = CANONICAL_SIZE * CANONICAL_SIZE
# Canonical rows increase toward physical north (minimum-y row is zero), so
# physical east, south, west, north is +column, -row, -column, +row.
_CANONICAL_DIRECTIONS = ((0, 1), (-1, 0), (0, -1), (1, 0))


def _triangle_array(values: np.ndarray) -> np.ndarray:
    positions = values % PARENT_SIDE_M
    return PARENT_SIDE_M - 4 * np.abs(positions - PARENT_SIDE_M // 2)


def _resistance_parameters(
    world_id: str,
) -> tuple[tuple[int, int], tuple[int, int], int, int]:
    require_hash(world_id, "world_id")
    sampler = StageSampler(world_id, STAGE_ID, STAGE_VERSION, RESISTANCE_PROCESS_ID)
    mode_1 = RESISTANCE_MODES[
        sampler.uint64(0, 0, channel=0, index=0) % len(RESISTANCE_MODES)
    ]
    mode_2 = RESISTANCE_MODES[
        sampler.uint64(0, 0, channel=1, index=0) % len(RESISTANCE_MODES)
    ]
    phase_1 = multiply_high_axis(sampler.uint64(0, 0, channel=2, index=0))
    phase_2 = multiply_high_axis(sampler.uint64(0, 0, channel=3, index=0))
    return mode_1, mode_2, phase_1, phase_2


@lru_cache(maxsize=16)
def _resistance_bytes(world_id: str) -> bytes:
    mode_1, mode_2, phase_1, phase_2 = _resistance_parameters(world_id)
    coordinate = (
        np.arange(CANONICAL_SIZE, dtype=np.int64) * CANONICAL_CELL_M
        + CANONICAL_CELL_M // 2
    )
    x_m = coordinate[np.newaxis, :]
    y_m = coordinate[:, np.newaxis]
    first = _triangle_array(
        mode_1[0] * x_m + mode_1[1] * y_m + phase_1
    )
    second = _triangle_array(
        mode_2[0] * x_m + mode_2[1] * y_m + phase_2
    )
    resistance = (
        RESISTANCE_BASE
        + ((first + PARENT_SIDE_M) * RESISTANCE_AMPLITUDES[0])
        // (2 * PARENT_SIDE_M)
        + ((second + PARENT_SIDE_M) * RESISTANCE_AMPLITUDES[1])
        // (2 * PARENT_SIDE_M)
    ).astype("<u2", copy=False)
    if int(resistance.min()) < 32 or int(resistance.max()) > 288:
        raise RuntimeError("frozen resistance escaped its exact integer range")
    return resistance.tobytes(order="C")


def derive_resistance_controls(world_id: str) -> ResistanceControlRecord:
    mode_1, mode_2, phase_1, phase_2 = _resistance_parameters(world_id)
    data = _resistance_bytes(world_id)
    return ResistanceControlRecord(
        world_id=world_id,
        first_mode=mode_1,
        second_mode=mode_2,
        first_phase_m=phase_1,
        second_phase_m=phase_2,
        resistance_sha256=hashlib.sha256(data).hexdigest(),
        preview_bytes=resistance_array(world_id)[::8, ::8]
        .astype("<u2", copy=False)
        .tobytes(order="C"),
    )


def resistance_array(world_id: str) -> np.ndarray:
    result = np.frombuffer(_resistance_bytes(world_id), dtype="<u2").reshape(
        CANONICAL_SIZE, CANONICAL_SIZE
    )
    result.flags.writeable = False
    return result


def clear_growth_caches() -> None:
    _resistance_bytes.cache_clear()


@dataclass(frozen=True, slots=True)
class GrowthResult:
    affiliation_bytes: bytes
    arrival_times_bytes: bytes
    parent_indices_bytes: bytes
    source_mask_packed_bytes: bytes
    certificate: GrowthCertificate
    resistance_controls: ResistanceControlRecord
    arrival_summary: ArrivalSummaryRecord
    adjacency_signature_sha256: str


def _adjacency_signature(owner: np.ndarray) -> str:
    edge_counts: dict[tuple[int, int], int] = {}
    for other in (np.roll(owner, -1, axis=0), np.roll(owner, -1, axis=1)):
        changed = owner != other
        first = owner[changed].astype(np.int16, copy=False)
        second = other[changed].astype(np.int16, copy=False)
        low = np.minimum(first, second)
        high = np.maximum(first, second)
        codes = low * PRIMARY_ACTOR_COUNT + high
        counts = np.bincount(codes, minlength=PRIMARY_ACTOR_COUNT**2)
        for code in np.flatnonzero(counts):
            pair = (int(code) // PRIMARY_ACTOR_COUNT, int(code) % PRIMARY_ACTOR_COUNT)
            edge_counts[pair] = edge_counts.get(pair, 0) + int(counts[code])
    return content_sha256(
        {
            "cardinal_contact_edges": [
                {"edge_count": edge_counts[pair], "slots": list(pair)}
                for pair in sorted(edge_counts)
            ],
            "schema_id": "urn:mapgen:pipeline-c:c02-adjacency-signature:v1",
            "schema_version": 1,
        }
    )


def grow_affiliation(
    world_id: str,
    actors: tuple[PrimaryActorRecord, ...],
    *,
    source_order: str = "forward",
    neighbor_order: str = "canonical",
) -> GrowthResult:
    """Run the frozen blocked growth once; no fallback or post-growth repair."""

    require_hash(world_id, "world_id")
    if source_order not in {"forward", "reverse"}:
        raise ValueError("source_order must be forward or reverse")
    if neighbor_order not in {"canonical", "reverse"}:
        raise ValueError("neighbor_order must be canonical or reverse")
    if (
        not isinstance(actors, tuple)
        or len(actors) != PRIMARY_ACTOR_COUNT
        or tuple(actor.slot for actor in actors) != tuple(range(PRIMARY_ACTOR_COUNT))
        or any(actor.world_id != world_id for actor in actors)
    ):
        raise TypeError("actors must be the seven ordered records for world_id")

    resistance_bytes = _resistance_bytes(world_id)
    resistance_controls = derive_resistance_controls(world_id)
    resistance = np.frombuffer(resistance_bytes, dtype="<u2")
    owner = np.full(_CELL_COUNT, -1, dtype=np.int8)
    arrival = np.full(_CELL_COUNT, np.iinfo(np.int64).max, dtype=np.int64)
    parent = np.full(_CELL_COUNT, -1, dtype=np.int32)
    source = np.zeros(_CELL_COUNT, dtype=np.bool_)
    for actor in actors:
        for cell in actor.germ_flat_indices:
            if owner[cell] >= 0:
                raise FabricFormationError(
                    "C02_GERM_OVERLAP",
                    "source cells overlap at growth initialization",
                    cell=cell,
                    first_slot=int(owner[cell]),
                    second_slot=actor.slot,
                )
            owner[cell] = actor.slot
            arrival[cell] = actor.initial_arrival
            parent[cell] = cell
            source[cell] = True

    maximum_int64 = np.iinfo(np.int64).max
    best_arrival = np.full(_CELL_COUNT, maximum_int64, dtype=np.int64)
    best_rank = np.full(_CELL_COUNT, 127, dtype=np.int8)
    best_actor = np.full(_CELL_COUNT, 127, dtype=np.int8)
    best_parent = np.full(_CELL_COUNT, np.iinfo(np.int32).max, dtype=np.int32)
    heap: list[tuple[int, int, int, int, int]] = []
    directions = _CANONICAL_DIRECTIONS
    if neighbor_order == "reverse":
        directions = tuple(reversed(directions))

    def propose(cell: int, actor: PrimaryActorRecord, current_arrival: int) -> None:
        row, column = divmod(cell, CANONICAL_SIZE)
        for delta_row, delta_column in directions:
            next_row = (row + delta_row) % CANONICAL_SIZE
            next_column = (column + delta_column) % CANONICAL_SIZE
            destination = next_row * CANONICAL_SIZE + next_column
            if owner[destination] >= 0:
                continue
            parallel = (
                (actor.global_axis == 0 and delta_column != 0)
                or (actor.global_axis == 1 and delta_row != 0)
            )
            directional = (
                PARALLEL_DIRECTIONAL_COST
                if parallel
                else PERPENDICULAR_DIRECTIONAL_COST
            )
            step_sign = delta_column if actor.global_axis == 0 else delta_row
            sign_penalty = (
                BACKWARD_SIGN_PENALTY
                if parallel and step_sign != actor.preferred_sign
                else 0
            )
            proposed_arrival = (
                int(current_arrival)
                + STEP_BASE_COST
                + int(resistance[destination])
                + directional
                + sign_penalty
            )
            candidate = (
                proposed_arrival,
                actor.tie_rank,
                destination,
                actor.slot,
                cell,
            )
            current = (
                int(best_arrival[destination]),
                int(best_rank[destination]),
                destination,
                int(best_actor[destination]),
                int(best_parent[destination]),
            )
            if candidate < current:
                best_arrival[destination] = proposed_arrival
                best_rank[destination] = actor.tie_rank
                best_actor[destination] = actor.slot
                best_parent[destination] = cell
                heapq.heappush(heap, candidate)

    actor_indices = range(PRIMARY_ACTOR_COUNT)
    if source_order == "reverse":
        actor_indices = reversed(tuple(actor_indices))
    for actor_index in actor_indices:
        actor = actors[actor_index]
        source_cells: object = actor.germ_flat_indices
        if source_order == "reverse":
            source_cells = reversed(actor.germ_flat_indices)
        for cell in source_cells:
            propose(cell, actor, int(arrival[cell]))

    claimed = int(source.sum())
    while heap:
        event = heapq.heappop(heap)
        proposed_arrival, rank, destination, actor_slot, parent_cell = event
        if owner[destination] >= 0:
            continue
        if (
            proposed_arrival != int(best_arrival[destination])
            or rank != int(best_rank[destination])
            or actor_slot != int(best_actor[destination])
            or parent_cell != int(best_parent[destination])
        ):
            continue
        owner[destination] = actor_slot
        arrival[destination] = proposed_arrival
        parent[destination] = parent_cell
        claimed += 1
        propose(destination, actors[actor_slot], proposed_arrival)

    if claimed != _CELL_COUNT or np.any(owner < 0):
        raise FabricFormationError(
            "C02_INCOMPLETE_COVERAGE",
            "competitive growth did not cover the canonical torus",
            claimed_cell_count=claimed,
            expected_cell_count=_CELL_COUNT,
        )

    non_source_indices = np.flatnonzero(~source)
    parent_values = parent[non_source_indices]
    parent_valid = parent_values >= 0
    same_owner = parent_valid & (
        owner[parent_values.clip(min=0)] == owner[non_source_indices]
    )
    earlier = parent_valid & (
        arrival[parent_values.clip(min=0)] < arrival[non_source_indices]
    )
    rows = non_source_indices // CANONICAL_SIZE
    columns = non_source_indices % CANONICAL_SIZE
    parent_rows = parent_values.clip(min=0) // CANONICAL_SIZE
    parent_columns = parent_values.clip(min=0) % CANONICAL_SIZE
    row_delta = (parent_rows - rows) % CANONICAL_SIZE
    column_delta = (parent_columns - columns) % CANONICAL_SIZE
    cardinal = parent_valid & (
        ((row_delta == 0) & ((column_delta == 1) | (column_delta == CANONICAL_SIZE - 1)))
        | ((column_delta == 0) & ((row_delta == 1) | (row_delta == CANONICAL_SIZE - 1)))
    )
    parent_chains_certified = bool(
        np.all(parent_valid) and np.all(same_owner) and np.all(earlier) and np.all(cardinal)
    )
    if not parent_chains_certified:
        raise FabricFormationError(
            "C02_PARENT_CERTIFICATE_FAILED",
            "one or more growth parents violate the frozen induction certificate",
            cardinal_count=int(cardinal.sum()),
            earlier_count=int(earlier.sum()),
            same_owner_count=int(same_owner.sum()),
        )
    if int(arrival.min()) < np.iinfo(np.int32).min or int(arrival.max()) > np.iinfo(np.int32).max:
        raise FabricFormationError(
            "C02_ARRIVAL_OVERFLOW",
            "arrival time cannot be serialized as canonical int32",
            maximum=int(arrival.max()),
            minimum=int(arrival.min()),
        )

    affiliation_bytes = owner.astype(np.uint8, copy=False).tobytes(order="C")
    arrival_bytes = arrival.astype("<i4", copy=False).tobytes(order="C")
    parent_bytes = parent.astype("<u4", copy=False).tobytes(order="C")
    source_bytes = np.packbits(source, bitorder="little").tobytes(order="C")
    certificate = GrowthCertificate(
        affiliation_sha256=hashlib.sha256(affiliation_bytes).hexdigest(),
        arrival_times_sha256=hashlib.sha256(arrival_bytes).hexdigest(),
        parent_indices_sha256=hashlib.sha256(parent_bytes).hexdigest(),
        source_mask_sha256=hashlib.sha256(source_bytes).hexdigest(),
        resistance_sha256=hashlib.sha256(resistance_bytes).hexdigest(),
        covered_cell_count=claimed,
        source_cell_count=int(source.sum()),
        non_source_parent_count=len(non_source_indices),
        same_owner_parent_count=int(same_owner.sum()),
        earlier_parent_count=int(earlier.sum()),
        cardinal_parent_count=int(cardinal.sum()),
        germ_count=PRIMARY_ACTOR_COUNT,
        germs_disjoint=True,
        germs_connected=True,
        complete_coverage=True,
        parent_chains_certified=True,
    )
    actor_arrivals = tuple(
        ActorArrivalSummary(
            slot=actor.slot,
            cell_count=int(np.count_nonzero(owner == actor.slot)),
            minimum_arrival=int(arrival[owner == actor.slot].min()),
            maximum_arrival=int(arrival[owner == actor.slot].max()),
            arrival_sum=int(arrival[owner == actor.slot].sum(dtype=np.int64)),
        )
        for actor in actors
    )
    arrival_summary = ArrivalSummaryRecord(
        arrival_times_sha256=certificate.arrival_times_sha256,
        total_cell_count=_CELL_COUNT,
        minimum_arrival=int(arrival.min()),
        maximum_arrival=int(arrival.max()),
        arrival_sum=int(arrival.sum(dtype=np.int64)),
        actor_summaries=actor_arrivals,
    )
    return GrowthResult(
        affiliation_bytes=affiliation_bytes,
        arrival_times_bytes=arrival_bytes,
        parent_indices_bytes=parent_bytes,
        source_mask_packed_bytes=source_bytes,
        certificate=certificate,
        resistance_controls=resistance_controls,
        arrival_summary=arrival_summary,
        adjacency_signature_sha256=_adjacency_signature(
            owner.reshape(CANONICAL_SIZE, CANONICAL_SIZE)
        ),
    )


__all__ = [
    "GrowthResult",
    "clear_growth_caches",
    "derive_resistance_controls",
    "grow_affiliation",
    "resistance_array",
]
