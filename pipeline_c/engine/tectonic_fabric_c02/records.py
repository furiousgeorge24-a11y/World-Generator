"""Immutable specification, actor, certificate, and C02 state records."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass

import numpy as np

from ..foundation import ExecutionSeed, PARENT_RECT
from ._util import (
    FabricRecordError,
    content_sha256,
    require_hash,
    require_id,
    require_int,
)
from .constants import (
    ACTOR_SCHEMA_ID,
    ARRIVAL_SUMMARY_SCHEMA_ID,
    ATTEMPT_ID,
    BACKWARD_SIGN_PENALTY,
    CANONICAL_CELL_M,
    CANONICAL_SIZE,
    CANDIDATES_PER_ACTOR,
    CASE_ID,
    CERTIFICATE_SCHEMA_ID,
    COMPARISON_FAMILY_ID,
    COORDINATE_MAPPING,
    CROWDING_BONUS_PER_CELL,
    CROWDING_TARGET_DISTANCE_M,
    DIRECTION_PROCESS_ID,
    DISPLAY_LABEL,
    EVENT_TIE_POLICY,
    EVIDENCE_KIND,
    FABRIC_SPEC_SCHEMA_ID,
    FABRIC_STATE_SCHEMA_ID,
    FAMILY_MINIMUM_SEPARATION_M,
    FAMILY_NAMES,
    GERM_CELL_COUNT,
    GERM_ENDPOINT_SPAN_M,
    GERM_HALF_STEPS,
    LAYOUT_CONTROL_SCHEMA_ID,
    LAYOUT_PROCESS_ID,
    LINEAGE_SCHEMA_ID,
    NUCLEUS_PROCESS_ID,
    OWNER_READOUT_POLICY,
    PARALLEL_DIRECTIONAL_COST,
    PARENT_CENSUS_SIZE,
    PARENT_SIDE_M,
    PERPENDICULAR_DIRECTIONAL_COST,
    PRIMARY_ACTOR_COUNT,
    REPRESENTATION_ID,
    RESISTANCE_AMPLITUDES,
    RESISTANCE_BASE,
    RESISTANCE_MODES,
    RESISTANCE_PROCESS_ID,
    RESISTANCE_CONTROL_SCHEMA_ID,
    RESISTANCE_PREVIEW_SIZE,
    ROADMAP_RUN,
    STAGE_ID,
    STAGE_VERSION,
    STEP_BASE_COST,
    SUPPORTED_OBSERVATION_SIZES,
    TIE_ORDER_PROCESS_ID,
    TOPOLOGY_ID,
)
from .context import FabricFormationContext


_CELL_COUNT = CANONICAL_SIZE * CANONICAL_SIZE
_SOURCE_COUNT = PRIMARY_ACTOR_COUNT * GERM_CELL_COUNT


@dataclass(frozen=True, slots=True)
class TectonicFabricSpec:
    actor_count: int = PRIMARY_ACTOR_COUNT
    canonical_size: int = CANONICAL_SIZE
    canonical_cell_m: int = CANONICAL_CELL_M
    candidates_per_actor: int = CANDIDATES_PER_ACTOR
    germ_half_steps: int = GERM_HALF_STEPS
    parent_census_size: int = PARENT_CENSUS_SIZE
    supported_observation_sizes: tuple[int, ...] = SUPPORTED_OBSERVATION_SIZES
    topology_id: str = TOPOLOGY_ID
    representation_id: str = REPRESENTATION_ID
    coordinate_mapping: str = COORDINATE_MAPPING
    owner_readout_policy: str = OWNER_READOUT_POLICY
    event_tie_policy: str = EVENT_TIE_POLICY

    def __post_init__(self) -> None:
        for name, value in (
            ("actor_count", self.actor_count),
            ("canonical_size", self.canonical_size),
            ("canonical_cell_m", self.canonical_cell_m),
            ("candidates_per_actor", self.candidates_per_actor),
            ("germ_half_steps", self.germ_half_steps),
            ("parent_census_size", self.parent_census_size),
        ):
            require_int(value, name, minimum=1)
        if self.supported_observation_sizes != SUPPORTED_OBSERVATION_SIZES:
            raise FabricRecordError("C02 observation sizes differ from the freeze")
        if not self.is_frozen_c02:
            raise FabricRecordError("C02 requires the exact frozen specification")

    @property
    def is_frozen_c02(self) -> bool:
        return (
            self.actor_count == PRIMARY_ACTOR_COUNT
            and self.canonical_size == CANONICAL_SIZE
            and self.canonical_cell_m == CANONICAL_CELL_M
            and self.candidates_per_actor == CANDIDATES_PER_ACTOR
            and self.germ_half_steps == GERM_HALF_STEPS
            and self.parent_census_size == PARENT_CENSUS_SIZE
            and self.supported_observation_sizes == SUPPORTED_OBSERVATION_SIZES
            and self.topology_id == TOPOLOGY_ID
            and self.representation_id == REPRESENTATION_ID
            and self.coordinate_mapping == COORDINATE_MAPPING
            and self.owner_readout_policy == OWNER_READOUT_POLICY
            and self.event_tie_policy == EVENT_TIE_POLICY
            and PARENT_RECT.min_x_m == PARENT_RECT.min_y_m == 0
            and PARENT_RECT.width_m == PARENT_RECT.height_m == PARENT_SIDE_M
            and self.canonical_size * self.canonical_cell_m == PARENT_SIDE_M
        )

    def to_record(self) -> dict[str, object]:
        return {
            "actor_count": self.actor_count,
            "canonical_lattice": {
                "cell_height_m": self.canonical_cell_m,
                "cell_width_m": self.canonical_cell_m,
                "height_cells": self.canonical_size,
                "row_zero": "minimum_y",
                "width_cells": self.canonical_size,
            },
            "crowding_head_start": {
                "bonus_per_deficit_cell": CROWDING_BONUS_PER_CELL,
                "distance_target_m": CROWDING_TARGET_DISTANCE_M,
            },
            "event_tie_policy": self.event_tie_policy,
            "families": [
                {
                    "family_id": family_id,
                    "minimum_nucleus_separation_m": (
                        FAMILY_MINIMUM_SEPARATION_M[family_id]
                    ),
                    "name": name,
                }
                for family_id, name in enumerate(FAMILY_NAMES)
            ],
            "germ": {
                "cell_count": GERM_CELL_COUNT,
                "endpoint_span_m": GERM_ENDPOINT_SPAN_M,
                "half_steps": self.germ_half_steps,
            },
            "growth_cost": {
                "backward_sign_penalty": BACKWARD_SIGN_PENALTY,
                "base": STEP_BASE_COST,
                "parallel_directional": PARALLEL_DIRECTIONAL_COST,
                "perpendicular_directional": PERPENDICULAR_DIRECTIONAL_COST,
                "resistance_amplitudes": list(RESISTANCE_AMPLITUDES),
                "resistance_base": RESISTANCE_BASE,
                "resistance_modes": [list(value) for value in RESISTANCE_MODES],
            },
            "nucleus_candidates": {
                "coordinate_mapping": self.coordinate_mapping,
                "count_per_actor": self.candidates_per_actor,
                "process_id": NUCLEUS_PROCESS_ID,
                "selection": "first-region-and-separation-eligible-by-priority-index",
            },
            "owner_readout_policy": self.owner_readout_policy,
            "parent_census_size": self.parent_census_size,
            "processes": {
                "direction": DIRECTION_PROCESS_ID,
                "layout": LAYOUT_PROCESS_ID,
                "resistance": RESISTANCE_PROCESS_ID,
                "tie_order": TIE_ORDER_PROCESS_ID,
            },
            "representation_id": self.representation_id,
            "schema_id": FABRIC_SPEC_SCHEMA_ID,
            "schema_version": 1,
            "supported_observation_sizes": list(self.supported_observation_sizes),
            "topology_id": self.topology_id,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_record())

    def formation_record(self) -> dict[str, object]:
        result = self.to_record()
        result.pop("parent_census_size")
        result.pop("supported_observation_sizes")
        return result

    @property
    def formation_sha256(self) -> str:
        return content_sha256(self.formation_record())


_FROZEN_C02_SPEC = TectonicFabricSpec()


def frozen_c02_spec() -> TectonicFabricSpec:
    return _FROZEN_C02_SPEC


def frozen_c5_spec() -> TectonicFabricSpec:
    """Compatibility spelling for callers that name the roadmap run."""

    return _FROZEN_C02_SPEC


def actor_lineage_id(world_id: str, slot: int) -> str:
    require_hash(world_id, "world_id")
    require_int(slot, "slot", minimum=0, maximum=PRIMARY_ACTOR_COUNT - 1)
    return content_sha256(
        {
            "actor_slot": slot,
            "lineage_schema_id": LINEAGE_SCHEMA_ID,
            "schema_version": 1,
            "world_id": world_id,
        }
    )


def actor_state_id(actor_record_without_id: dict[str, object]) -> str:
    if not isinstance(actor_record_without_id, dict):
        raise TypeError("actor identity material must be a dict")
    return content_sha256(
        {
            "actor": actor_record_without_id,
            "representation_id": REPRESENTATION_ID,
            "schema_id": "urn:mapgen:pipeline-c:c02-actor-state-identity:v1",
            "schema_version": 1,
            "topology_id": TOPOLOGY_ID,
        }
    )


@dataclass(frozen=True, slots=True)
class LayoutControlRecord:
    world_id: str
    family_id: int
    origin_x_m: int
    origin_y_m: int
    orientation_quarter_turns: int
    phase_m: int

    def __post_init__(self) -> None:
        require_hash(self.world_id, "world_id")
        require_int(self.family_id, "family_id", minimum=0, maximum=3)
        for name in ("origin_x_m", "origin_y_m", "phase_m"):
            require_int(getattr(self, name), name, minimum=0, maximum=PARENT_SIDE_M - 1)
        require_int(
            self.orientation_quarter_turns,
            "orientation_quarter_turns",
            minimum=0,
            maximum=3,
        )

    @property
    def family_name(self) -> str:
        return FAMILY_NAMES[self.family_id]

    def to_record(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "family_name": self.family_name,
            "orientation_quarter_turns": self.orientation_quarter_turns,
            "origin_x_m": self.origin_x_m,
            "origin_y_m": self.origin_y_m,
            "phase_m": self.phase_m,
            "process_id": LAYOUT_PROCESS_ID,
            "schema_id": LAYOUT_CONTROL_SCHEMA_ID,
            "schema_version": 1,
            "world_id": self.world_id,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_record())


@dataclass(frozen=True, slots=True)
class ResistanceControlRecord:
    world_id: str
    first_mode: tuple[int, int]
    second_mode: tuple[int, int]
    first_phase_m: int
    second_phase_m: int
    resistance_sha256: str
    preview_bytes: bytes

    def __post_init__(self) -> None:
        require_hash(self.world_id, "world_id")
        if self.first_mode not in RESISTANCE_MODES or self.second_mode not in RESISTANCE_MODES:
            raise FabricRecordError("resistance mode differs from the frozen six-mode set")
        for name in ("first_phase_m", "second_phase_m"):
            require_int(getattr(self, name), name, minimum=0, maximum=PARENT_SIDE_M - 1)
        require_hash(self.resistance_sha256, "resistance_sha256")
        expected = 2 * RESISTANCE_PREVIEW_SIZE * RESISTANCE_PREVIEW_SIZE
        if not isinstance(self.preview_bytes, bytes) or len(self.preview_bytes) != expected:
            raise FabricRecordError("resistance preview has the wrong byte length")

    def preview_array(self) -> np.ndarray:
        result = np.frombuffer(self.preview_bytes, dtype="<u2").reshape(
            RESISTANCE_PREVIEW_SIZE, RESISTANCE_PREVIEW_SIZE
        )
        result.flags.writeable = False
        return result

    def to_record(self, *, include_data: bool = False) -> dict[str, object]:
        preview = {
            "byte_length": len(self.preview_bytes),
            "dtype": "uint16_le",
            "row_zero": "minimum_y",
            "sampling": "every_eighth_canonical_cell_from_row_column_zero",
            "sha256": hashlib.sha256(self.preview_bytes).hexdigest(),
            "shape": [RESISTANCE_PREVIEW_SIZE, RESISTANCE_PREVIEW_SIZE],
        }
        if include_data:
            preview["base64"] = base64.b64encode(self.preview_bytes).decode("ascii")
        return {
            "first_mode": list(self.first_mode),
            "first_phase_m": self.first_phase_m,
            "process_id": RESISTANCE_PROCESS_ID,
            "resistance_sha256": self.resistance_sha256,
            "preview": preview,
            "schema_id": RESISTANCE_CONTROL_SCHEMA_ID,
            "schema_version": 1,
            "second_mode": list(self.second_mode),
            "second_phase_m": self.second_phase_m,
            "world_id": self.world_id,
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_record())


@dataclass(frozen=True, slots=True)
class ActorArrivalSummary:
    slot: int
    cell_count: int
    minimum_arrival: int
    maximum_arrival: int
    arrival_sum: int

    def __post_init__(self) -> None:
        require_int(self.slot, "slot", minimum=0, maximum=PRIMARY_ACTOR_COUNT - 1)
        require_int(self.cell_count, "cell_count", minimum=1)
        require_int(self.minimum_arrival, "minimum_arrival")
        require_int(self.maximum_arrival, "maximum_arrival")
        require_int(self.arrival_sum, "arrival_sum")
        if self.minimum_arrival > self.maximum_arrival:
            raise FabricRecordError("arrival summary minimum exceeds maximum")

    def to_record(self) -> dict[str, object]:
        return {
            "actor_slot": self.slot,
            "arrival_mean_fraction": {
                "denominator": self.cell_count,
                "numerator": self.arrival_sum,
            },
            "arrival_sum": self.arrival_sum,
            "cell_count": self.cell_count,
            "maximum_arrival": self.maximum_arrival,
            "minimum_arrival": self.minimum_arrival,
        }


@dataclass(frozen=True, slots=True)
class ArrivalSummaryRecord:
    arrival_times_sha256: str
    total_cell_count: int
    minimum_arrival: int
    maximum_arrival: int
    arrival_sum: int
    actor_summaries: tuple[ActorArrivalSummary, ...]

    def __post_init__(self) -> None:
        require_hash(self.arrival_times_sha256, "arrival_times_sha256")
        require_int(self.total_cell_count, "total_cell_count", minimum=1)
        require_int(self.minimum_arrival, "minimum_arrival")
        require_int(self.maximum_arrival, "maximum_arrival")
        require_int(self.arrival_sum, "arrival_sum")
        if self.minimum_arrival > self.maximum_arrival:
            raise FabricRecordError("global arrival minimum exceeds maximum")
        if (
            not isinstance(self.actor_summaries, tuple)
            or tuple(value.slot for value in self.actor_summaries)
            != tuple(range(PRIMARY_ACTOR_COUNT))
            or sum(value.cell_count for value in self.actor_summaries)
            != self.total_cell_count
            or sum(value.arrival_sum for value in self.actor_summaries)
            != self.arrival_sum
            or min(value.minimum_arrival for value in self.actor_summaries)
            != self.minimum_arrival
            or max(value.maximum_arrival for value in self.actor_summaries)
            != self.maximum_arrival
        ):
            raise FabricRecordError("per-actor arrival summaries do not close globally")

    def to_record(self) -> dict[str, object]:
        return {
            "actor_summaries": [value.to_record() for value in self.actor_summaries],
            "arrival_mean_fraction": {
                "denominator": self.total_cell_count,
                "numerator": self.arrival_sum,
            },
            "arrival_sum": self.arrival_sum,
            "arrival_times_sha256": self.arrival_times_sha256,
            "maximum_arrival": self.maximum_arrival,
            "minimum_arrival": self.minimum_arrival,
            "schema_id": ARRIVAL_SUMMARY_SCHEMA_ID,
            "schema_version": 1,
            "total_cell_count": self.total_cell_count,
        }

    @property
    def canonical_sha256(self) -> str:
        return content_sha256(self.to_record())


@dataclass(frozen=True, slots=True)
class PrimaryActorRecord:
    world_id: str
    slot: int
    lineage_id: str
    actor_state_id: str
    family_id: int
    nucleus_x_m: int
    nucleus_y_m: int
    selected_candidate_index: int
    selected_global_candidate_index: int
    candidate_priority: int
    region_eligible_candidate_count: int
    nearest_prior_nucleus_distance_squared_m2: int | None
    nearest_nucleus_distance_m: int
    global_axis: int
    preferred_sign: int
    tie_rank: int
    germ_flat_indices: tuple[int, ...]
    crowding_bonus: int

    def __post_init__(self) -> None:
        require_hash(self.world_id, "world_id")
        require_int(self.slot, "slot", minimum=0, maximum=PRIMARY_ACTOR_COUNT - 1)
        require_hash(self.lineage_id, "lineage_id")
        require_hash(self.actor_state_id, "actor_state_id")
        require_int(self.family_id, "family_id", minimum=0, maximum=3)
        for name in ("nucleus_x_m", "nucleus_y_m"):
            require_int(getattr(self, name), name, minimum=0, maximum=PARENT_SIDE_M - 1)
        require_int(
            self.selected_candidate_index,
            "selected_candidate_index",
            minimum=0,
            maximum=CANDIDATES_PER_ACTOR - 1,
        )
        if self.selected_global_candidate_index != (
            self.slot * CANDIDATES_PER_ACTOR + self.selected_candidate_index
        ):
            raise FabricRecordError("global candidate index is inconsistent")
        require_int(self.candidate_priority, "candidate_priority", minimum=0, maximum=2**64 - 1)
        require_int(self.region_eligible_candidate_count, "region_eligible_candidate_count", minimum=1, maximum=CANDIDATES_PER_ACTOR)
        if self.slot == 0:
            if self.nearest_prior_nucleus_distance_squared_m2 is not None:
                raise FabricRecordError("slot zero cannot have a prior-nucleus distance")
        else:
            require_int(
                self.nearest_prior_nucleus_distance_squared_m2,
                "nearest_prior_nucleus_distance_squared_m2",
                minimum=1,
            )
        require_int(self.nearest_nucleus_distance_m, "nearest_nucleus_distance_m", minimum=1)
        require_int(self.global_axis, "global_axis", minimum=0, maximum=1)
        if self.preferred_sign not in {-1, 1}:
            raise FabricRecordError("preferred_sign must be -1 or 1")
        require_int(self.tie_rank, "tie_rank", minimum=0, maximum=PRIMARY_ACTOR_COUNT - 1)
        if (
            not isinstance(self.germ_flat_indices, tuple)
            or len(self.germ_flat_indices) != GERM_CELL_COUNT
            or len(set(self.germ_flat_indices)) != GERM_CELL_COUNT
        ):
            raise FabricRecordError("germ must contain 33 distinct cells")
        for value in self.germ_flat_indices:
            require_int(value, "germ cell", minimum=0, maximum=_CELL_COUNT - 1)
        require_int(self.crowding_bonus, "crowding_bonus", minimum=0)
        if self.crowding_bonus % CROWDING_BONUS_PER_CELL:
            raise FabricRecordError("crowding bonus is not an exact frozen increment")
        if self.lineage_id != actor_lineage_id(self.world_id, self.slot):
            raise FabricRecordError("actor lineage differs from world/slot lineage")
        material = self.identity_record()
        if self.actor_state_id != actor_state_id(material):
            raise FabricRecordError("actor state identity is inconsistent")

    @property
    def axis_name(self) -> str:
        return "x" if self.global_axis == 0 else "y"

    @property
    def initial_arrival(self) -> int:
        return -self.crowding_bonus

    def identity_record(self) -> dict[str, object]:
        return {
            "candidate_priority": self.candidate_priority,
            "crowding_bonus": self.crowding_bonus,
            "family_id": self.family_id,
            "germ_flat_indices": list(self.germ_flat_indices),
            "global_axis": self.global_axis,
            "lineage_id": self.lineage_id,
            "nearest_nucleus_distance_m": self.nearest_nucleus_distance_m,
            "nucleus_x_m": self.nucleus_x_m,
            "nucleus_y_m": self.nucleus_y_m,
            "preferred_sign": self.preferred_sign,
            "selected_candidate_index": self.selected_candidate_index,
            "slot": self.slot,
            "tie_rank": self.tie_rank,
            "world_id": self.world_id,
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self.identity_record(),
            "actor_state_id": self.actor_state_id,
            "axis_name": self.axis_name,
            "initial_arrival": self.initial_arrival,
            "nearest_prior_nucleus_distance_squared_m2": (
                self.nearest_prior_nucleus_distance_squared_m2
            ),
            "region_eligible_candidate_count": self.region_eligible_candidate_count,
            "schema_id": ACTOR_SCHEMA_ID,
            "schema_version": 1,
            "selected_global_candidate_index": self.selected_global_candidate_index,
        }


@dataclass(frozen=True, slots=True)
class GrowthCertificate:
    affiliation_sha256: str
    arrival_times_sha256: str
    parent_indices_sha256: str
    source_mask_sha256: str
    resistance_sha256: str
    covered_cell_count: int
    source_cell_count: int
    non_source_parent_count: int
    same_owner_parent_count: int
    earlier_parent_count: int
    cardinal_parent_count: int
    germ_count: int
    germs_disjoint: bool
    germs_connected: bool
    complete_coverage: bool
    parent_chains_certified: bool

    def __post_init__(self) -> None:
        for name in (
            "affiliation_sha256",
            "arrival_times_sha256",
            "parent_indices_sha256",
            "source_mask_sha256",
            "resistance_sha256",
        ):
            require_hash(getattr(self, name), name)
        require_int(self.covered_cell_count, "covered_cell_count", minimum=0, maximum=_CELL_COUNT)
        require_int(self.source_cell_count, "source_cell_count", minimum=0, maximum=_CELL_COUNT)
        for name in (
            "non_source_parent_count",
            "same_owner_parent_count",
            "earlier_parent_count",
            "cardinal_parent_count",
        ):
            require_int(getattr(self, name), name, minimum=0, maximum=_CELL_COUNT)
        require_int(self.germ_count, "germ_count", minimum=0, maximum=PRIMARY_ACTOR_COUNT)
        for name in (
            "germs_disjoint",
            "germs_connected",
            "complete_coverage",
            "parent_chains_certified",
        ):
            if not isinstance(getattr(self, name), bool):
                raise FabricRecordError(f"{name} must be boolean")

    @property
    def passes(self) -> bool:
        non_sources = _CELL_COUNT - _SOURCE_COUNT
        return (
            self.covered_cell_count == _CELL_COUNT
            and self.source_cell_count == _SOURCE_COUNT
            and self.non_source_parent_count == non_sources
            and self.same_owner_parent_count == non_sources
            and self.earlier_parent_count == non_sources
            and self.cardinal_parent_count == non_sources
            and self.germ_count == PRIMARY_ACTOR_COUNT
            and self.germs_disjoint
            and self.germs_connected
            and self.complete_coverage
            and self.parent_chains_certified
        )

    def to_record(self) -> dict[str, object]:
        return {
            "array_sha256": {
                "affiliation": self.affiliation_sha256,
                "arrival_times": self.arrival_times_sha256,
                "parent_indices": self.parent_indices_sha256,
                "resistance": self.resistance_sha256,
                "source_mask": self.source_mask_sha256,
            },
            "cardinal_parent_count": self.cardinal_parent_count,
            "complete_coverage": self.complete_coverage,
            "covered_cell_count": self.covered_cell_count,
            "earlier_parent_count": self.earlier_parent_count,
            "germ_count": self.germ_count,
            "germs_connected": self.germs_connected,
            "germs_disjoint": self.germs_disjoint,
            "non_source_parent_count": self.non_source_parent_count,
            "parent_chains_certified": self.parent_chains_certified,
            "passes": self.passes,
            "same_owner_parent_count": self.same_owner_parent_count,
            "schema_id": CERTIFICATE_SCHEMA_ID,
            "schema_version": 1,
            "source_cell_count": self.source_cell_count,
        }

    @property
    def canonical_sha256(self) -> str:
        return content_sha256(self.to_record())


def _array_descriptor(data: bytes, dtype: str, shape: tuple[int, int]) -> dict[str, object]:
    return {
        "byte_length": len(data),
        "dtype": dtype,
        "layout": "row_major_c",
        "sha256": hashlib.sha256(data).hexdigest(),
        "shape": list(shape),
    }


@dataclass(frozen=True, slots=True)
class TectonicFabricState:
    execution_seed: ExecutionSeed
    context: FabricFormationContext
    spec: TectonicFabricSpec
    controls: LayoutControlRecord
    resistance_controls: ResistanceControlRecord
    arrival_summary: ArrivalSummaryRecord
    actors: tuple[PrimaryActorRecord, ...]
    affiliation_bytes: bytes
    arrival_times_bytes: bytes
    parent_indices_bytes: bytes
    source_mask_packed_bytes: bytes
    certificate: GrowthCertificate
    layout_sha256: str
    catalog_sha256: str
    construction_sha256: str
    partition_sha256: str
    adjacency_signature_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.execution_seed, ExecutionSeed):
            raise FabricRecordError("execution_seed must be ExecutionSeed")
        if self.execution_seed.role not in {"debug", "development"}:
            raise FabricRecordError("C02 cannot consume validation seeds")
        if not isinstance(self.context, FabricFormationContext):
            raise FabricRecordError("context must be FabricFormationContext")
        if self.context.seed != self.execution_seed.seed:
            raise FabricRecordError("context seed differs from execution seed")
        if self.spec != _FROZEN_C02_SPEC:
            raise FabricRecordError("state spec differs from frozen C02 spec")
        if not isinstance(self.controls, LayoutControlRecord):
            raise FabricRecordError("controls must be LayoutControlRecord")
        if self.controls.world_id != self.context.world_id:
            raise FabricRecordError("layout controls belong to another world")
        if (
            not isinstance(self.resistance_controls, ResistanceControlRecord)
            or self.resistance_controls.world_id != self.context.world_id
            or self.resistance_controls.resistance_sha256
            != self.certificate.resistance_sha256
        ):
            raise FabricRecordError("resistance controls differ from the certified field")
        if (
            not isinstance(self.arrival_summary, ArrivalSummaryRecord)
            or self.arrival_summary.arrival_times_sha256
            != self.certificate.arrival_times_sha256
        ):
            raise FabricRecordError("arrival summary differs from the certified array")
        if (
            not isinstance(self.actors, tuple)
            or len(self.actors) != PRIMARY_ACTOR_COUNT
            or tuple(actor.slot for actor in self.actors) != tuple(range(PRIMARY_ACTOR_COUNT))
            or any(actor.world_id != self.context.world_id for actor in self.actors)
            or any(actor.family_id != self.controls.family_id for actor in self.actors)
        ):
            raise FabricRecordError("state requires seven ordered actors from one world/family")
        if len({actor.lineage_id for actor in self.actors}) != PRIMARY_ACTOR_COUNT:
            raise FabricRecordError("actor lineages must be unique")
        if sorted(actor.tie_rank for actor in self.actors) != list(range(PRIMARY_ACTOR_COUNT)):
            raise FabricRecordError("tie ranks must be a complete permutation")
        expected_source_bytes = (_CELL_COUNT + 7) // 8
        if not isinstance(self.affiliation_bytes, bytes) or len(self.affiliation_bytes) != _CELL_COUNT:
            raise FabricRecordError("affiliation bytes have the wrong length")
        if not isinstance(self.arrival_times_bytes, bytes) or len(self.arrival_times_bytes) != 4 * _CELL_COUNT:
            raise FabricRecordError("arrival bytes have the wrong length")
        if not isinstance(self.parent_indices_bytes, bytes) or len(self.parent_indices_bytes) != 4 * _CELL_COUNT:
            raise FabricRecordError("parent bytes have the wrong length")
        if not isinstance(self.source_mask_packed_bytes, bytes) or len(self.source_mask_packed_bytes) != expected_source_bytes:
            raise FabricRecordError("packed source mask has the wrong length")
        if np.any(np.frombuffer(self.affiliation_bytes, dtype=np.uint8) >= PRIMARY_ACTOR_COUNT):
            raise FabricRecordError("affiliation contains an invalid owner")
        if int(self.source_mask_array().sum()) != _SOURCE_COUNT:
            raise FabricRecordError("source mask does not contain exactly 231 cells")
        if not isinstance(self.certificate, GrowthCertificate) or not self.certificate.passes:
            raise FabricRecordError("growth certificate does not pass")
        hashes = {
            "affiliation": hashlib.sha256(self.affiliation_bytes).hexdigest(),
            "arrival": hashlib.sha256(self.arrival_times_bytes).hexdigest(),
            "parent": hashlib.sha256(self.parent_indices_bytes).hexdigest(),
            "source": hashlib.sha256(self.source_mask_packed_bytes).hexdigest(),
        }
        if (
            hashes["affiliation"] != self.certificate.affiliation_sha256
            or hashes["arrival"] != self.certificate.arrival_times_sha256
            or hashes["parent"] != self.certificate.parent_indices_sha256
            or hashes["source"] != self.certificate.source_mask_sha256
        ):
            raise FabricRecordError("certificate array hashes differ from state bytes")
        owners = np.frombuffer(self.affiliation_bytes, dtype=np.uint8)
        arrivals = np.frombuffer(self.arrival_times_bytes, dtype="<i4")
        if (
            self.arrival_summary.total_cell_count != _CELL_COUNT
            or self.arrival_summary.minimum_arrival != int(arrivals.min())
            or self.arrival_summary.maximum_arrival != int(arrivals.max())
            or self.arrival_summary.arrival_sum != int(arrivals.sum(dtype=np.int64))
        ):
            raise FabricRecordError("global arrival summary does not match array bytes")
        for summary in self.arrival_summary.actor_summaries:
            values = arrivals[owners == summary.slot]
            if (
                summary.cell_count != len(values)
                or summary.minimum_arrival != int(values.min())
                or summary.maximum_arrival != int(values.max())
                or summary.arrival_sum != int(values.sum(dtype=np.int64))
            ):
                raise FabricRecordError(
                    "per-actor arrival summary does not match array bytes"
                )
        for name in (
            "layout_sha256",
            "catalog_sha256",
            "construction_sha256",
            "partition_sha256",
            "adjacency_signature_sha256",
        ):
            require_hash(getattr(self, name), name)

    @property
    def actor_id_lookup(self) -> tuple[str, ...]:
        return tuple(actor.lineage_id for actor in self.actors)

    @property
    def family_id(self) -> int:
        return self.controls.family_id

    @property
    def family_name(self) -> str:
        return self.controls.family_name

    @property
    def affiliation_sha256(self) -> str:
        return self.certificate.affiliation_sha256

    @property
    def arrival_times_sha256(self) -> str:
        return self.certificate.arrival_times_sha256

    @property
    def parent_indices_sha256(self) -> str:
        return self.certificate.parent_indices_sha256

    @property
    def source_mask_sha256(self) -> str:
        return self.certificate.source_mask_sha256

    def slots_array(self) -> np.ndarray:
        result = np.frombuffer(self.affiliation_bytes, dtype=np.uint8).reshape(
            CANONICAL_SIZE, CANONICAL_SIZE
        )
        result.flags.writeable = False
        return result

    def arrival_array(self) -> np.ndarray:
        result = np.frombuffer(self.arrival_times_bytes, dtype="<i4").reshape(
            CANONICAL_SIZE, CANONICAL_SIZE
        )
        result.flags.writeable = False
        return result

    def parent_array(self) -> np.ndarray:
        result = np.frombuffer(self.parent_indices_bytes, dtype="<u4").reshape(
            CANONICAL_SIZE, CANONICAL_SIZE
        )
        result.flags.writeable = False
        return result

    def source_mask_array(self) -> np.ndarray:
        values = np.unpackbits(
            np.frombuffer(self.source_mask_packed_bytes, dtype=np.uint8),
            bitorder="little",
        )[:_CELL_COUNT].reshape(CANONICAL_SIZE, CANONICAL_SIZE).astype(np.bool_)
        values.flags.writeable = False
        return values

    def to_record(self, *, include_arrays: bool = False) -> dict[str, object]:
        affiliation = _array_descriptor(
            self.affiliation_bytes, "uint8", (CANONICAL_SIZE, CANONICAL_SIZE)
        )
        arrival = _array_descriptor(
            self.arrival_times_bytes, "int32_le", (CANONICAL_SIZE, CANONICAL_SIZE)
        )
        parent = _array_descriptor(
            self.parent_indices_bytes, "uint32_le", (CANONICAL_SIZE, CANONICAL_SIZE)
        )
        source = {
            "bit_count": _CELL_COUNT,
            "bit_order": "little",
            "byte_length": len(self.source_mask_packed_bytes),
            "dtype": "packed_boolean",
            "sha256": hashlib.sha256(self.source_mask_packed_bytes).hexdigest(),
            "shape": [CANONICAL_SIZE, CANONICAL_SIZE],
        }
        if include_arrays:
            affiliation["base64"] = base64.b64encode(self.affiliation_bytes).decode("ascii")
            arrival["base64"] = base64.b64encode(self.arrival_times_bytes).decode("ascii")
            parent["base64"] = base64.b64encode(self.parent_indices_bytes).decode("ascii")
            source["base64"] = base64.b64encode(self.source_mask_packed_bytes).decode("ascii")
        return {
            "actors": [actor.to_record() for actor in self.actors],
            "adjacency_signature_sha256": self.adjacency_signature_sha256,
            "arrays": {
                "affiliation": affiliation,
                "arrival_times": arrival,
                "parent_indices": parent,
                "source_mask": source,
            },
            "arrival_summary": self.arrival_summary.to_record(),
            "attempt_id": ATTEMPT_ID,
            "catalog_sha256": self.catalog_sha256,
            "case_id": CASE_ID,
            "certificate": self.certificate.to_record(),
            "comparison_family_id": COMPARISON_FAMILY_ID,
            "construction_sha256": self.construction_sha256,
            "context": self.context.to_record(),
            "display_label": DISPLAY_LABEL,
            "evidence_kind": EVIDENCE_KIND,
            "execution_seed": self.execution_seed.to_record(),
            "layout_controls": self.controls.to_record(),
            "layout_sha256": self.layout_sha256,
            "omissions": [
                "contact_geometry",
                "kinematics",
                "geology",
                "elevation",
                "water",
                "coast",
                "land",
                "delivered_window",
                "map",
            ],
            "partition_sha256": self.partition_sha256,
            "ready": False,
            "resistance_controls": self.resistance_controls.to_record(
                include_data=include_arrays
            ),
            "roadmap_run": ROADMAP_RUN,
            "schema_id": FABRIC_STATE_SCHEMA_ID,
            "schema_version": 1,
            "spec": self.spec.to_record(),
            "stage_id": STAGE_ID,
            "stage_version": STAGE_VERSION,
        }

    @property
    def canonical_sha256(self) -> str:
        return content_sha256(self.to_record())


__all__ = [
    "GrowthCertificate",
    "ActorArrivalSummary",
    "ArrivalSummaryRecord",
    "LayoutControlRecord",
    "PrimaryActorRecord",
    "ResistanceControlRecord",
    "TectonicFabricSpec",
    "TectonicFabricState",
    "actor_lineage_id",
    "actor_state_id",
    "frozen_c02_spec",
    "frozen_c5_spec",
]
