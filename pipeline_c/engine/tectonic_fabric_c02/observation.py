"""Lossless canonical and registered observer-only C02 readouts."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass

import numpy as np

from ..foundation import (
    DEVELOPMENT_ANALYSIS_RECT,
    PARENT_RECT,
    PhysicalGrid,
    analysis_grid,
)
from ._util import FabricRecordError, content_sha256, require_hash, require_int
from .constants import (
    CANONICAL_CELL_M,
    CANONICAL_OBSERVATION_SCHEMA_ID,
    CANONICAL_SIZE,
    OBSERVATION_SCHEMA_ID,
    PARENT_CENSUS_SIZE,
)
from .records import TectonicFabricState
from .topology import owner_slots


def _data_record(data: bytes, dtype: str) -> dict[str, object]:
    return {
        "byte_length": len(data),
        "dtype": dtype,
        "layout": "row_major_c",
        "sha256": hashlib.sha256(data).hexdigest(),
    }


@dataclass(frozen=True, slots=True)
class CanonicalAffiliationObservation:
    source_state_sha256: str
    actor_id_lookup: tuple[str, ...]
    actor_slots_bytes: bytes
    endpoint_agreement_count_bytes: bytes
    strict_all_eight_bytes: bytes

    def __post_init__(self) -> None:
        require_hash(self.source_state_sha256, "source_state_sha256")
        if (
            not isinstance(self.actor_id_lookup, tuple)
            or len(self.actor_id_lookup) != 7
            or len(set(self.actor_id_lookup)) != 7
        ):
            raise FabricRecordError("actor lookup must contain seven unique IDs")
        for value in self.actor_id_lookup:
            require_hash(value, "actor lookup item")
        expected = CANONICAL_SIZE * CANONICAL_SIZE
        for name in (
            "actor_slots_bytes",
            "endpoint_agreement_count_bytes",
            "strict_all_eight_bytes",
        ):
            value = getattr(self, name)
            if not isinstance(value, bytes) or len(value) != expected:
                raise FabricRecordError(f"{name} has the wrong length")
        owners = np.frombuffer(self.actor_slots_bytes, dtype=np.uint8)
        agreement = np.frombuffer(
            self.endpoint_agreement_count_bytes, dtype=np.uint8
        )
        strict = np.frombuffer(self.strict_all_eight_bytes, dtype=np.uint8)
        if np.any(owners >= 7) or np.any(agreement > 8) or np.any(strict > 1):
            raise FabricRecordError("canonical observation contains invalid values")
        if np.any(strict != (agreement == 8)):
            raise FabricRecordError("strict mask differs from all-eight agreement")

    @property
    def width_px(self) -> int:
        return CANONICAL_SIZE

    @property
    def height_px(self) -> int:
        return CANONICAL_SIZE

    def slots_array(self) -> np.ndarray:
        result = np.frombuffer(self.actor_slots_bytes, dtype=np.uint8).reshape(
            CANONICAL_SIZE, CANONICAL_SIZE
        )
        result.flags.writeable = False
        return result

    def endpoint_agreement_array(self) -> np.ndarray:
        result = np.frombuffer(
            self.endpoint_agreement_count_bytes, dtype=np.uint8
        ).reshape(CANONICAL_SIZE, CANONICAL_SIZE)
        result.flags.writeable = False
        return result

    def strict_all_eight_array(self) -> np.ndarray:
        result = np.frombuffer(
            self.strict_all_eight_bytes, dtype=np.uint8
        ).reshape(CANONICAL_SIZE, CANONICAL_SIZE)
        result.flags.writeable = False
        return result

    def to_record(self, *, include_data: bool = False) -> dict[str, object]:
        slots = _data_record(self.actor_slots_bytes, "uint8")
        agreement = _data_record(
            self.endpoint_agreement_count_bytes, "uint8_0_to_8"
        )
        strict = _data_record(self.strict_all_eight_bytes, "uint8_boolean")
        if include_data:
            slots["base64"] = base64.b64encode(self.actor_slots_bytes).decode("ascii")
            agreement["base64"] = base64.b64encode(
                self.endpoint_agreement_count_bytes
            ).decode("ascii")
            strict["base64"] = base64.b64encode(
                self.strict_all_eight_bytes
            ).decode("ascii")
        return {
            "actor_id_lookup": list(self.actor_id_lookup),
            "actor_slots": slots,
            "canonical_lattice": {
                "cell_m": CANONICAL_CELL_M,
                "height_cells": CANONICAL_SIZE,
                "row_zero": "minimum_y",
                "width_cells": CANONICAL_SIZE,
            },
            "endpoint_agreement_count": agreement,
            "observation_kind": "canonical_affiliation",
            "schema_id": CANONICAL_OBSERVATION_SCHEMA_ID,
            "schema_version": 1,
            "source_state_sha256": self.source_state_sha256,
            "stability_semantics": {
                "cardinal_offset_m": 1_280_000,
                "diagonal_component_offset_m": 905_097,
                "diagonal_containing_cell_center_distance_m_approx": 1_301_076,
                "mean_is_not_strict": True,
                "sample_addresses": "canonical_cell_centers",
            },
            "strict_all_eight": strict,
        }

    @property
    def canonical_sha256(self) -> str:
        return content_sha256(self.to_record())


@dataclass(frozen=True, slots=True)
class FabricObservation:
    observation_kind: str
    source_state_sha256: str
    grid: PhysicalGrid
    actor_id_lookup: tuple[str, ...]
    actor_slots_bytes: bytes

    def __post_init__(self) -> None:
        if self.observation_kind not in {"parent_census", "development_analysis"}:
            raise FabricRecordError("unsupported C02 observation kind")
        require_hash(self.source_state_sha256, "source_state_sha256")
        if not isinstance(self.grid, PhysicalGrid):
            raise FabricRecordError("grid must be PhysicalGrid")
        if (
            not isinstance(self.actor_id_lookup, tuple)
            or len(self.actor_id_lookup) != 7
            or len(set(self.actor_id_lookup)) != 7
        ):
            raise FabricRecordError("actor lookup must contain seven unique IDs")
        for value in self.actor_id_lookup:
            require_hash(value, "actor lookup item")
        expected = self.grid.width_px * self.grid.height_px
        if not isinstance(self.actor_slots_bytes, bytes) or len(self.actor_slots_bytes) != expected:
            raise FabricRecordError("actor slot bytes have the wrong length")
        if np.any(np.frombuffer(self.actor_slots_bytes, dtype=np.uint8) >= 7):
            raise FabricRecordError("observation contains an invalid actor slot")
        if self.observation_kind == "parent_census":
            if self.grid != PhysicalGrid(PARENT_RECT, PARENT_CENSUS_SIZE, PARENT_CENSUS_SIZE):
                raise FabricRecordError("parent census has the wrong registered grid")
        elif self.grid.rectangle != DEVELOPMENT_ANALYSIS_RECT:
            raise FabricRecordError("analysis observation has the wrong rectangle")

    @property
    def width_px(self) -> int:
        return self.grid.width_px

    @property
    def height_px(self) -> int:
        return self.grid.height_px

    @property
    def actor_slots_sha256(self) -> str:
        return hashlib.sha256(self.actor_slots_bytes).hexdigest()

    def slots_array(self) -> np.ndarray:
        result = np.frombuffer(self.actor_slots_bytes, dtype=np.uint8).reshape(
            self.height_px, self.width_px
        )
        result.flags.writeable = False
        return result

    def to_record(self, *, include_data: bool = False) -> dict[str, object]:
        slots = _data_record(self.actor_slots_bytes, "uint8")
        if include_data:
            slots["base64"] = base64.b64encode(self.actor_slots_bytes).decode("ascii")
        return {
            "actor_id_lookup": list(self.actor_id_lookup),
            "actor_slots": slots,
            "grid": self.grid.to_record(),
            "observation_kind": self.observation_kind,
            "schema_id": OBSERVATION_SCHEMA_ID,
            "schema_version": 1,
            "source_state_sha256": self.source_state_sha256,
        }

    @property
    def canonical_sha256(self) -> str:
        return content_sha256(self.to_record())


def observe_canonical_affiliation(
    state: TectonicFabricState,
) -> CanonicalAffiliationObservation:
    if not isinstance(state, TectonicFabricState):
        raise TypeError("state must be TectonicFabricState")
    owners = state.slots_array()
    agreement = np.zeros(owners.shape, dtype=np.uint8)
    offsets = (
        (0, 32),
        (0, -32),
        (32, 0),
        (-32, 0),
        (23, 23),
        (23, -23),
        (-23, 23),
        (-23, -23),
    )
    for delta_row, delta_column in offsets:
        agreement += owners == np.roll(
            owners, shift=(-delta_row, -delta_column), axis=(0, 1)
        )
    strict = (agreement == 8).astype(np.uint8)
    return CanonicalAffiliationObservation(
        source_state_sha256=state.canonical_sha256,
        actor_id_lookup=state.actor_id_lookup,
        actor_slots_bytes=state.affiliation_bytes,
        endpoint_agreement_count_bytes=agreement.tobytes(order="C"),
        strict_all_eight_bytes=strict.tobytes(order="C"),
    )


def observe_grid(
    state: TectonicFabricState,
    grid: PhysicalGrid,
    *,
    observation_kind: str,
    traversal: str = "forward",
    chunk_rows: int = 64,
) -> FabricObservation:
    if not isinstance(state, TectonicFabricState):
        raise TypeError("state must be TectonicFabricState")
    if not isinstance(grid, PhysicalGrid):
        raise TypeError("grid must be PhysicalGrid")
    if traversal not in {"forward", "reverse"}:
        raise FabricRecordError("traversal must be forward or reverse")
    require_int(chunk_rows, "chunk_rows", minimum=1)
    if grid.cell_width_m % 2 or grid.cell_height_m % 2:
        raise FabricRecordError("C02 observations require integer-metre centers")
    x_m = (
        grid.rectangle.min_x_m
        + np.arange(grid.width_px, dtype=np.int64) * grid.cell_width_m
        + grid.cell_width_m // 2
    )
    y_m = (
        grid.rectangle.max_y_m
        - np.arange(grid.height_px, dtype=np.int64) * grid.cell_height_m
        - grid.cell_height_m // 2
    )
    slots = np.empty((grid.height_px, grid.width_px), dtype=np.uint8)
    starts = list(range(0, grid.height_px, chunk_rows))
    if traversal == "reverse":
        starts.reverse()
    for start in starts:
        stop = min(grid.height_px, start + chunk_rows)
        slots[start:stop] = owner_slots(
            state,
            x_m[np.newaxis, :],
            y_m[start:stop, np.newaxis],
        )
    return FabricObservation(
        observation_kind=observation_kind,
        source_state_sha256=state.canonical_sha256,
        grid=grid,
        actor_id_lookup=state.actor_id_lookup,
        actor_slots_bytes=slots.tobytes(order="C"),
    )


def observe_parent_census(
    state: TectonicFabricState,
    *,
    traversal: str = "forward",
    chunk_rows: int = 64,
) -> FabricObservation:
    return observe_grid(
        state,
        PhysicalGrid(PARENT_RECT, PARENT_CENSUS_SIZE, PARENT_CENSUS_SIZE),
        observation_kind="parent_census",
        traversal=traversal,
        chunk_rows=chunk_rows,
    )


def observe_analysis(
    state: TectonicFabricState,
    size: int,
    *,
    traversal: str = "forward",
    chunk_rows: int = 64,
) -> FabricObservation:
    return observe_grid(
        state,
        analysis_grid(size),
        observation_kind="development_analysis",
        traversal=traversal,
        chunk_rows=chunk_rows,
    )


__all__ = [
    "CanonicalAffiliationObservation",
    "FabricObservation",
    "observe_analysis",
    "observe_canonical_affiliation",
    "observe_grid",
    "observe_parent_census",
]
