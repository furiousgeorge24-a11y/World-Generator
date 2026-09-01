"""Deterministic land-origin measurements.

These functions consume already-produced masks or scalar records. They do
not generate, alter, select, or repair geography. A truthy mask cell means
land and a false cell means water.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence

LAND_TARGET_MIN_PERCENT = 0.0
LAND_TARGET_MAX_PERCENT = 70.0
LAND_TOLERANCE_PERCENTAGE_POINTS = 10.0
FRAGMENTATION_MAX = 1.0


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def target_interval_percent(target_land_percent: float) -> tuple[float, float]:
    """Return the accepted realized-land percentage interval.

    The tolerance is ten percentage points, clipped only to the physical
    0..100% range. Thus target 0 accepts 0..10 and target 70 accepts 60..80.
    The 70% limit is a request ceiling, not an output ceiling.
    """
    target = _finite_number("target_land_percent", target_land_percent)
    if not LAND_TARGET_MIN_PERCENT <= target <= LAND_TARGET_MAX_PERCENT:
        raise ValueError(
            "target_land_percent must be in "
            f"[{LAND_TARGET_MIN_PERCENT}, {LAND_TARGET_MAX_PERCENT}]")
    return max(0.0, target - LAND_TOLERANCE_PERCENTAGE_POINTS), min(
        100.0, target + LAND_TOLERANCE_PERCENTAGE_POINTS)


def target_within_tolerance_percent(
    target_land_percent: float, realized_land_percent: float
) -> bool:
    realized = _finite_number("realized_land_percent", realized_land_percent)
    if not 0.0 <= realized <= 100.0:
        raise ValueError("realized_land_percent must be in [0, 100]")
    low, high = target_interval_percent(target_land_percent)
    return low - 1e-12 <= realized <= high + 1e-12


def _mask_rows(mask: Iterable[Iterable[object]]) -> list[list[bool]]:
    rows = [list(row) for row in mask]
    if not rows or not rows[0]:
        raise ValueError("mask must be a non-empty rectangular grid")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("mask must be rectangular")
    normalized: list[list[bool]] = []
    for row in rows:
        out_row = []
        for value in row:
            if value is True or value == 1:
                out_row.append(True)
            elif value is False or value == 0:
                out_row.append(False)
            else:
                raise ValueError("mask cells must be boolean or 0/1")
        normalized.append(out_row)
    return normalized


def land_fraction(mask: Iterable[Iterable[object]]) -> float:
    rows = _mask_rows(mask)
    land = sum(sum(row) for row in rows)
    return land / (len(rows) * len(rows[0]))


def land_percent(mask: Iterable[Iterable[object]]) -> float:
    return 100.0 * land_fraction(mask)


def outer_ring_is_water(mask: Iterable[Iterable[object]]) -> bool:
    rows = _mask_rows(mask)
    if any(rows[0]) or any(rows[-1]):
        return False
    return all(not row[0] and not row[-1] for row in rows)


def component_areas(
    mask: Iterable[Iterable[object]], *, connectivity: int = 8
) -> list[int]:
    """Return land-component areas in descending order.

    Eight-connectivity is the declared fragmentation diagnostic convention:
    it avoids counting diagonal raster contact as two invented islands. Raw
    component count remains diagnostic and is never an author-facing island
    target.
    """
    rows = _mask_rows(mask)
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    height, width = len(rows), len(rows[0])
    seen = [[False] * width for _ in range(height)]
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        neighbors += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    areas: list[int] = []
    for y in range(height):
        for x in range(width):
            if not rows[y][x] or seen[y][x]:
                continue
            seen[y][x] = True
            queue = deque([(y, x)])
            area = 0
            while queue:
                cy, cx = queue.popleft()
                area += 1
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if not (0 <= ny < height and 0 <= nx < width):
                        continue
                    if rows[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        queue.append((ny, nx))
            areas.append(area)
    return sorted(areas, reverse=True)


def fragmentation_metrics(
    mask: Iterable[Iterable[object]], *, connectivity: int = 8
) -> dict[str, object]:
    """Describe fragmentation without turning component count into a target."""
    areas = component_areas(mask, connectivity=connectivity)
    total = sum(areas)
    if total == 0:
        return {
            "applicable": False,
            "connectivity": connectivity,
            "land_cells": 0,
            "component_count": 0,
            "component_areas_cells": [],
            "largest_component_land_share": None,
            "inverse_simpson_effective_components": None,
            "component_area_entropy": None,
        }
    shares = [area / total for area in areas]
    inverse_simpson = 1.0 / sum(share * share for share in shares)
    entropy = -sum(share * math.log(share) for share in shares)
    return {
        "applicable": True,
        "connectivity": connectivity,
        "land_cells": total,
        "component_count": len(areas),
        "component_areas_cells": areas,
        "largest_component_land_share": shares[0],
        "inverse_simpson_effective_components": inverse_simpson,
        "component_area_entropy": entropy,
    }


def monotonic_land_percentages(
    points: Sequence[tuple[float, float, int]],
) -> dict[str, object]:
    """Check a same-family increasing-target sweep for backwards motion.

    Each point is ``(requested_percent, realized_percent, cell_count)``.
    Requested targets
    must be strictly increasing. A reversal no larger than one final-mask
    cell is treated as measurement equality, not as controller motion.
    """
    if not points:
        raise ValueError("at least one sweep point is required")
    checked: list[tuple[float, float, int]] = []
    for target, realized, cell_count in points:
        target = _finite_number("target", target)
        realized = _finite_number("realized", realized)
        if not LAND_TARGET_MIN_PERCENT <= target <= LAND_TARGET_MAX_PERCENT:
            raise ValueError("sweep target is outside the request range")
        if not 0.0 <= realized <= 100.0:
            raise ValueError("sweep realization is outside [0, 100]")
        if isinstance(cell_count, bool) or not isinstance(cell_count, int):
            raise TypeError("cell_count must be an integer")
        if cell_count <= 0:
            raise ValueError("cell_count must be positive")
        if checked and target <= checked[-1][0]:
            raise ValueError("sweep targets must be strictly increasing")
        checked.append((target, realized, cell_count))

    violations = []
    for previous, current in zip(checked, checked[1:]):
        tolerance = 100.0 * max(1.0 / previous[2], 1.0 / current[2])
        if current[1] + tolerance + 1e-12 < previous[1]:
            violations.append({
                "from_target": previous[0],
                "to_target": current[0],
                "from_realized": previous[1],
                "to_realized": current[1],
                "one_cell_tolerance": tolerance,
            })
    return {"passes": not violations, "violations": violations}
