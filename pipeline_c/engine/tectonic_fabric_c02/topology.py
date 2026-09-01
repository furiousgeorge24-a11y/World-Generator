"""Containing-cell physical readout of the fixed C02 affiliation lattice."""

from __future__ import annotations

import numpy as np

from ._util import FabricRecordError, require_int
from .constants import (
    CANONICAL_CELL_M,
    CANONICAL_SIZE,
    PARENT_SIDE_M,
    STABILITY_CARDINAL_OFFSET_M,
    STABILITY_DIAGONAL_OFFSET_M,
)
from .records import TectonicFabricState


STABILITY_OFFSETS_M = (
    (STABILITY_CARDINAL_OFFSET_M, 0),
    (-STABILITY_CARDINAL_OFFSET_M, 0),
    (0, STABILITY_CARDINAL_OFFSET_M),
    (0, -STABILITY_CARDINAL_OFFSET_M),
    (STABILITY_DIAGONAL_OFFSET_M, STABILITY_DIAGONAL_OFFSET_M),
    (STABILITY_DIAGONAL_OFFSET_M, -STABILITY_DIAGONAL_OFFSET_M),
    (-STABILITY_DIAGONAL_OFFSET_M, STABILITY_DIAGONAL_OFFSET_M),
    (-STABILITY_DIAGONAL_OFFSET_M, -STABILITY_DIAGONAL_OFFSET_M),
)


def canonical_cell_indices(x_m: int, y_m: int) -> tuple[int, int]:
    """Return canonical minimum-y row and column for an integer-metre address."""

    require_int(x_m, "x_m")
    require_int(y_m, "y_m")
    column = (x_m % PARENT_SIDE_M) // CANONICAL_CELL_M
    row = (y_m % PARENT_SIDE_M) // CANONICAL_CELL_M
    return row, column


def canonical_flat_index(x_m: int, y_m: int) -> int:
    row, column = canonical_cell_indices(x_m, y_m)
    return row * CANONICAL_SIZE + column


def owner_slot(state: TectonicFabricState, x_m: int, y_m: int) -> int:
    if not isinstance(state, TectonicFabricState):
        raise TypeError("state must be TectonicFabricState")
    return state.affiliation_bytes[canonical_flat_index(x_m, y_m)]


def _integer_array(value: object, label: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind not in {"i", "u"}:
        raise FabricRecordError(f"{label} must contain integer metres")
    return result.astype(np.int64, copy=False)


def owner_slots(
    state: TectonicFabricState,
    x_m: object,
    y_m: object,
) -> np.ndarray:
    if not isinstance(state, TectonicFabricState):
        raise TypeError("state must be TectonicFabricState")
    x = _integer_array(x_m, "x_m")
    y = _integer_array(y_m, "y_m")
    try:
        x, y = np.broadcast_arrays(x, y)
    except ValueError as exc:
        raise FabricRecordError("x_m and y_m are not broadcast-compatible") from exc
    columns = (x % PARENT_SIDE_M) // CANONICAL_CELL_M
    rows = (y % PARENT_SIDE_M) // CANONICAL_CELL_M
    flat = rows * CANONICAL_SIZE + columns
    owners = np.frombuffer(state.affiliation_bytes, dtype=np.uint8)[flat]
    return owners.astype(np.uint8, copy=False)


def evaluate_slots_and_stability(
    state: TectonicFabricState,
    x_m: object,
    y_m: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return owner, count of eight agreeing endpoints, and strict-all-eight."""

    x = _integer_array(x_m, "x_m")
    y = _integer_array(y_m, "y_m")
    try:
        x, y = np.broadcast_arrays(x, y)
    except ValueError as exc:
        raise FabricRecordError("x_m and y_m are not broadcast-compatible") from exc
    owners = owner_slots(state, x, y)
    agreement = np.zeros(owners.shape, dtype=np.uint8)
    for dx_m, dy_m in STABILITY_OFFSETS_M:
        agreement += owner_slots(state, x + dx_m, y + dy_m) == owners
    strict = agreement == len(STABILITY_OFFSETS_M)
    return owners, agreement, strict


def endpoint_agreement_count(
    state: TectonicFabricState,
    x_m: int,
    y_m: int,
) -> int:
    require_int(x_m, "x_m")
    require_int(y_m, "y_m")
    owner = owner_slot(state, x_m, y_m)
    return sum(
        owner_slot(state, x_m + dx_m, y_m + dy_m) == owner
        for dx_m, dy_m in STABILITY_OFFSETS_M
    )


def strict_all_eight(state: TectonicFabricState, x_m: int, y_m: int) -> bool:
    return endpoint_agreement_count(state, x_m, y_m) == 8


__all__ = [
    "STABILITY_OFFSETS_M",
    "canonical_cell_indices",
    "canonical_flat_index",
    "endpoint_agreement_count",
    "evaluate_slots_and_stability",
    "owner_slot",
    "owner_slots",
    "strict_all_eight",
]
