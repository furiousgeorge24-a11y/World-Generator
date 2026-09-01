"""Causal acceptance policy for a delivered map border.

Border water is a property of the final categorical water mask: every unique
cell on the delivered frame's outer ring must be water.  Water depth, a wider
water collar, clearance for hypothetical later relief, and alignment of
coastlines or other contours do not affect that decision.

Visual contour measurements are retained only as diagnostics.  A naturally
generated contour may parallel the delivered frame without being caused by
it, so contour shape is not an acceptance gate.

Border validity also does not establish causal independence.  The caller
must separately certify both frame-independent terrain generation and
independence from numerical-rim/local-process boundaries.
``evaluate_promotion`` combines those explicit facts without attempting to
infer causality from the appearance of the border.
"""

from __future__ import annotations

import json
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


__all__ = ["evaluate_causal_border", "evaluate_promotion"]


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = (
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
)


def _rectangular_bool_mask(water_mask: ArrayLike) -> NDArray[np.bool_]:
    """Return a validated, non-empty two-dimensional boolean array."""

    try:
        mask = np.asarray(water_mask)
    except ValueError as exc:
        raise ValueError("water_mask must be a rectangular 2-D mask") from exc
    if mask.ndim != 2:
        raise ValueError("water_mask must be two-dimensional")
    if mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("water_mask dimensions must both be non-zero")
    if mask.dtype != np.dtype(bool):
        raise TypeError("water_mask must contain boolean water classifications")
    return mask


def _outer_ring_coordinates(rows: int, columns: int) -> list[tuple[int, int]]:
    """Enumerate each outer-ring coordinate exactly once, clockwise."""

    if rows == 1:
        return [(0, column) for column in range(columns)]
    if columns == 1:
        return [(row, 0) for row in range(rows)]

    coordinates = [(0, column) for column in range(columns)]
    coordinates.extend((row, columns - 1) for row in range(1, rows))
    coordinates.extend(
        (rows - 1, column) for column in range(columns - 2, -1, -1)
    )
    coordinates.extend((row, 0) for row in range(rows - 2, 0, -1))
    return coordinates


def _json_clone(value: JsonValue | None) -> JsonValue | None:
    """Copy a diagnostic while enforcing ordinary JSON compatibility."""

    try:
        return cast(JsonValue | None, json.loads(json.dumps(value, allow_nan=False)))
    except (TypeError, ValueError) as exc:
        raise TypeError("contour_diagnostic must be JSON-serializable") from exc


def evaluate_causal_border(
    water_mask: ArrayLike,
    *,
    contour_diagnostic: JsonValue | None = None,
) -> dict[str, JsonValue]:
    """Evaluate exact outer-ring water without using depth or contour shape.

    ``water_mask`` must be a non-empty rectangular 2-D boolean mask, where
    ``True`` means water on the final delivered surface.  Coordinates are
    reported as ``[row, column]`` pairs.  The contour payload is JSON-cloned
    for safe reporting and cannot change ``border_passed`` or ``passed``.
    """

    mask = _rectangular_bool_mask(water_mask)
    rows, columns = (int(value) for value in mask.shape)
    ring_coordinates = _outer_ring_coordinates(rows, columns)
    non_water_coordinates = [
        [row, column]
        for row, column in ring_coordinates
        if not bool(mask[row, column])
    ]
    ring_count = len(ring_coordinates)
    non_water_count = len(non_water_coordinates)
    water_count = ring_count - non_water_count
    border_passed = non_water_count == 0

    return {
        "policy": "causal-border-acceptance-v1",
        "border_passed": border_passed,
        "passed": border_passed,
        "shape_rows_columns": [rows, columns],
        "outer_ring_cell_count": ring_count,
        "outer_ring_water_cell_count": water_count,
        "outer_ring_non_water_cell_count": non_water_count,
        "outer_ring_non_water_coordinates_row_column": non_water_coordinates,
        "contour_diagnostic": _json_clone(contour_diagnostic),
        "contour_diagnostic_affects_passed": False,
    }


def evaluate_promotion(
    *,
    border_passed: bool,
    frame_independent_generation: bool,
    process_domain_independent: bool,
) -> dict[str, JsonValue]:
    """Combine border water with separate causal certificates.

    ``frame_independent_generation`` certifies that terrain-forming processes
    did not consume the delivered border, its distance field, or a crop mask.
    ``process_domain_independent`` is supplied by finite-rim, process-halo,
    and nested/shifted-domain evidence.  Neither is inferred from the water
    mask or contour geometry.
    """

    if type(border_passed) is not bool:
        raise TypeError("border_passed must be bool")
    if type(frame_independent_generation) is not bool:
        raise TypeError("frame_independent_generation must be bool")
    if type(process_domain_independent) is not bool:
        raise TypeError("process_domain_independent must be bool")

    promotion_passed = (
        border_passed
        and frame_independent_generation
        and process_domain_independent
    )
    failed_requirements: list[str] = []
    if not border_passed:
        failed_requirements.append("exact_outer_ring_water")
    if not frame_independent_generation:
        failed_requirements.append("frame_independent_terrain_generation")
    if not process_domain_independent:
        failed_requirements.append("numerical_and_process_domain_independence")

    return {
        "policy": "causal-border-acceptance-v1",
        "border_passed": border_passed,
        "frame_independent_generation": frame_independent_generation,
        "process_domain_independent": process_domain_independent,
        "promotion_passed": promotion_passed,
        "passed": promotion_passed,
        "failed_requirements": failed_requirements,
    }
