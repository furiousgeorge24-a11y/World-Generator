"""Exact seeded C02 family layout, nuclei, directions, and connected germs."""

from __future__ import annotations

import math

from ..foundation import StageSampler
from ._util import FabricFormationError, require_hash
from .constants import (
    CANDIDATES_PER_ACTOR,
    CANONICAL_CELL_M,
    CANONICAL_SIZE,
    CROWDING_BONUS_PER_CELL,
    CROWDING_TARGET_DISTANCE_M,
    DIRECTION_PROCESS_ID,
    FAMILY_MINIMUM_SEPARATION_M,
    GERM_HALF_STEPS,
    LAYOUT_PROCESS_ID,
    NUCLEUS_PROCESS_ID,
    PARENT_SIDE_M,
    PRIMARY_ACTOR_COUNT,
    STAGE_ID,
    STAGE_VERSION,
    TIE_ORDER_PROCESS_ID,
)
from .records import (
    LayoutControlRecord,
    PrimaryActorRecord,
    actor_lineage_id,
    actor_state_id,
)


def multiply_high_axis(value: int, extent_m: int = PARENT_SIDE_M) -> int:
    """Map one uint64 to ``[0, extent)`` without floating point."""

    return (int(value) * extent_m) >> 64


def signed_periodic_delta(value: int, period_m: int = PARENT_SIDE_M) -> int:
    return ((int(value) + period_m // 2) % period_m) - period_m // 2


def triangle_wave(value: int, period_m: int = PARENT_SIDE_M) -> int:
    position = int(value) % period_m
    return period_m - 4 * abs(position - period_m // 2)


def toroidal_squared_distance_m2(
    x1_m: int,
    y1_m: int,
    x2_m: int,
    y2_m: int,
    *,
    period_x_m: int = PARENT_SIDE_M,
    period_y_m: int = PARENT_SIDE_M,
) -> int:
    dx = abs(int(x1_m) - int(x2_m)) % period_x_m
    dy = abs(int(y1_m) - int(y2_m)) % period_y_m
    dx = min(dx, period_x_m - dx)
    dy = min(dy, period_y_m - dy)
    return dx * dx + dy * dy


def local_coordinates(
    x_m: int,
    y_m: int,
    controls: LayoutControlRecord,
) -> tuple[int, int]:
    dx = signed_periodic_delta(x_m - controls.origin_x_m)
    dy = signed_periodic_delta(y_m - controls.origin_y_m)
    orientation = controls.orientation_quarter_turns
    if orientation == 0:
        return dx, dy
    if orientation == 1:
        return dy, -dx
    if orientation == 2:
        return -dx, -dy
    return -dy, dx


def _family_eligible(
    family_id: int,
    slot: int,
    u_m: int,
    v_m: int,
    phase_m: int,
) -> bool:
    side = PARENT_SIDE_M
    if family_id == 0:
        return True
    if family_id == 1:
        bend = (8 * triangle_wave(u_m + phase_m)) // 100
        dv = signed_periodic_delta(v_m - bend)
        if slot <= 3:
            return 10 * abs(dv) <= side
        if slot == 4:
            return 16 * side <= 100 * dv <= 42 * side
        if slot == 5:
            return -42 * side <= 100 * dv <= -16 * side
        return 16 * side <= 100 * abs(dv)
    if family_id == 2:
        first_u = -(18 * side) // 100
        second_u = (18 * side) // 100
        d1 = signed_periodic_delta(u_m - first_u) ** 2 + signed_periodic_delta(v_m) ** 2
        d2 = signed_periodic_delta(u_m - second_u) ** 2 + signed_periodic_delta(v_m) ** 2
        if slot <= 2:
            return 10_000 * d1 <= (16 * side) ** 2
        if slot <= 4:
            return 10_000 * d2 <= (16 * side) ** 2
        return 10_000 * d1 >= (22 * side) ** 2 and 10_000 * d2 >= (22 * side) ** 2
    radius_squared = u_m * u_m + v_m * v_m
    if slot <= 2:
        return (
            u_m >= 0
            and (18 * side) ** 2 <= 10_000 * radius_squared
            and 10_000 * radius_squared <= (30 * side) ** 2
        )
    return 10_000 * radius_squared >= (24 * side) ** 2


def derive_layout_controls(world_id: str) -> LayoutControlRecord:
    require_hash(world_id, "world_id")
    sampler = StageSampler(world_id, STAGE_ID, STAGE_VERSION, LAYOUT_PROCESS_ID)
    return LayoutControlRecord(
        world_id=world_id,
        family_id=int(sampler.uint64(0, 0, channel=0, index=0) % 4),
        origin_x_m=multiply_high_axis(sampler.uint64(0, 0, channel=1, index=0)),
        origin_y_m=multiply_high_axis(sampler.uint64(0, 0, channel=2, index=0)),
        orientation_quarter_turns=int(
            sampler.uint64(0, 0, channel=3, index=0) % 4
        ),
        phase_m=multiply_high_axis(sampler.uint64(0, 0, channel=4, index=0)),
    )


def _local_axis_to_global(local_axis: int, orientation: int) -> int:
    return (local_axis + (orientation & 1)) & 1


def _germ_cells(x_m: int, y_m: int, global_axis: int) -> tuple[int, ...]:
    column = x_m // CANONICAL_CELL_M
    row = y_m // CANONICAL_CELL_M
    cells: list[int] = []
    for offset in range(-GERM_HALF_STEPS, GERM_HALF_STEPS + 1):
        next_row = (row + (offset if global_axis == 1 else 0)) % CANONICAL_SIZE
        next_column = (
            column + (offset if global_axis == 0 else 0)
        ) % CANONICAL_SIZE
        cells.append(next_row * CANONICAL_SIZE + next_column)
    return tuple(cells)


def _germ_is_toroidally_connected(cells: tuple[int, ...]) -> bool:
    remaining = set(cells)
    if not remaining:
        return False
    pending = [remaining.pop()]
    while pending:
        cell = pending.pop()
        row, column = divmod(cell, CANONICAL_SIZE)
        for next_row, next_column in (
            (row, (column + 1) % CANONICAL_SIZE),
            ((row - 1) % CANONICAL_SIZE, column),
            (row, (column - 1) % CANONICAL_SIZE),
            ((row + 1) % CANONICAL_SIZE, column),
        ):
            neighbor = next_row * CANONICAL_SIZE + next_column
            if neighbor in remaining:
                remaining.remove(neighbor)
                pending.append(neighbor)
    return not remaining


def build_primary_actor_layout(
    world_id: str,
    *,
    candidate_order: str = "forward",
) -> tuple[LayoutControlRecord, tuple[PrimaryActorRecord, ...]]:
    """Derive all seven actors without observing any generated ownership."""

    require_hash(world_id, "world_id")
    if candidate_order not in {"forward", "reverse"}:
        raise ValueError("candidate_order must be forward or reverse")
    controls = derive_layout_controls(world_id)
    sampler = StageSampler(world_id, STAGE_ID, STAGE_VERSION, NUCLEUS_PROCESS_ID)
    selected: list[dict[str, int | None]] = []
    for slot in range(PRIMARY_ACTOR_COUNT):
        indices = range(CANDIDATES_PER_ACTOR)
        if candidate_order == "reverse":
            indices = reversed(tuple(indices))
        pool: list[tuple[int, int, int, int]] = []
        for candidate_index in indices:
            global_index = slot * CANDIDATES_PER_ACTOR + candidate_index
            x_m = multiply_high_axis(
                sampler.uint64(0, 0, channel=0, index=global_index)
            )
            y_m = multiply_high_axis(
                sampler.uint64(0, 0, channel=1, index=global_index)
            )
            priority = sampler.uint64(0, 0, channel=2, index=global_index)
            u_m, v_m = local_coordinates(x_m, y_m, controls)
            if _family_eligible(
                controls.family_id, slot, u_m, v_m, controls.phase_m
            ):
                pool.append((priority, candidate_index, x_m, y_m))
        pool.sort(key=lambda item: (item[0], item[1]))
        minimum_squared = FAMILY_MINIMUM_SEPARATION_M[controls.family_id] ** 2
        chosen: tuple[int, int, int, int] | None = None
        nearest_prior: int | None = None
        for candidate in pool:
            distances = [
                toroidal_squared_distance_m2(
                    candidate[2],
                    candidate[3],
                    int(prior["x_m"]),
                    int(prior["y_m"]),
                )
                for prior in selected
            ]
            if not distances or min(distances) >= minimum_squared:
                chosen = candidate
                nearest_prior = min(distances) if distances else None
                break
        if chosen is None:
            raise FabricFormationError(
                "C02_NUCLEUS_EXHAUSTED",
                "no frozen nucleus candidate satisfies region and separation",
                family_id=controls.family_id,
                region_eligible_count=len(pool),
                slot=slot,
            )
        selected.append(
            {
                "priority": chosen[0],
                "candidate_index": chosen[1],
                "x_m": chosen[2],
                "y_m": chosen[3],
                "eligible_count": len(pool),
                "nearest_prior": nearest_prior,
            }
        )

    nearest_distances: list[int] = []
    for slot, current in enumerate(selected):
        nearest_squared = min(
            toroidal_squared_distance_m2(
                int(current["x_m"]),
                int(current["y_m"]),
                int(other["x_m"]),
                int(other["y_m"]),
            )
            for other_slot, other in enumerate(selected)
            if other_slot != slot
        )
        nearest_distances.append(math.isqrt(nearest_squared))

    direction_sampler = StageSampler(
        world_id, STAGE_ID, STAGE_VERSION, DIRECTION_PROCESS_ID
    )
    tie_sampler = StageSampler(
        world_id, STAGE_ID, STAGE_VERSION, TIE_ORDER_PROCESS_ID
    )
    tie_order = sorted(
        range(PRIMARY_ACTOR_COUNT),
        key=lambda slot: (
            tie_sampler.uint64(0, 0, channel=0, index=slot),
            slot,
        ),
    )
    tie_ranks = {slot: rank for rank, slot in enumerate(tie_order)}
    axes: list[int] = []
    signs: list[int] = []
    for slot, item in enumerate(selected):
        random_axis = int(
            direction_sampler.uint64(0, 0, channel=0, index=slot) % 2
        )
        preferred_sign = (
            1 if direction_sampler.uint64(0, 0, channel=1, index=slot) % 2 else -1
        )
        u_m, v_m = local_coordinates(
            int(item["x_m"]), int(item["y_m"]), controls
        )
        local_axis: int | None = None
        if controls.family_id == 1 and slot <= 3:
            local_axis = 0
        elif controls.family_id == 2 and slot <= 4:
            focus_u = (
                -(18 * PARENT_SIDE_M) // 100
                if slot <= 2
                else (18 * PARENT_SIDE_M) // 100
            )
            du = signed_periodic_delta(u_m - focus_u)
            dv = signed_periodic_delta(v_m)
            local_axis = 1 if abs(du) >= abs(dv) else 0
        elif controls.family_id == 3 and slot <= 2:
            local_axis = 1 if abs(u_m) >= abs(v_m) else 0
        axes.append(
            random_axis
            if local_axis is None
            else _local_axis_to_global(
                local_axis, controls.orientation_quarter_turns
            )
        )
        signs.append(preferred_sign)

    germ_sets: list[tuple[int, ...]] = []
    occupied: dict[int, int] = {}
    for slot, item in enumerate(selected):
        germ = _germ_cells(int(item["x_m"]), int(item["y_m"]), axes[slot])
        if (
            len(germ) != 33
            or len(set(germ)) != 33
            or not _germ_is_toroidally_connected(germ)
        ):
            raise FabricFormationError(
                "C02_GERM_INVALID",
                "a frozen germ is not an exact connected 33-cell source",
                slot=slot,
            )
        for cell in germ:
            if cell in occupied:
                raise FabricFormationError(
                    "C02_GERM_OVERLAP",
                    "two frozen germs overlap",
                    cell=cell,
                    first_slot=occupied[cell],
                    second_slot=slot,
                )
            occupied[cell] = slot
        germ_sets.append(germ)

    actors: list[PrimaryActorRecord] = []
    for slot, item in enumerate(selected):
        bonus = CROWDING_BONUS_PER_CELL * (
            max(0, CROWDING_TARGET_DISTANCE_M - nearest_distances[slot])
            // CANONICAL_CELL_M
        )
        lineage = actor_lineage_id(world_id, slot)
        identity = {
            "candidate_priority": int(item["priority"]),
            "crowding_bonus": bonus,
            "family_id": controls.family_id,
            "germ_flat_indices": list(germ_sets[slot]),
            "global_axis": axes[slot],
            "lineage_id": lineage,
            "nearest_nucleus_distance_m": nearest_distances[slot],
            "nucleus_x_m": int(item["x_m"]),
            "nucleus_y_m": int(item["y_m"]),
            "preferred_sign": signs[slot],
            "selected_candidate_index": int(item["candidate_index"]),
            "slot": slot,
            "tie_rank": tie_ranks[slot],
            "world_id": world_id,
        }
        actors.append(
            PrimaryActorRecord(
                world_id=world_id,
                slot=slot,
                lineage_id=lineage,
                actor_state_id=actor_state_id(identity),
                family_id=controls.family_id,
                nucleus_x_m=int(item["x_m"]),
                nucleus_y_m=int(item["y_m"]),
                selected_candidate_index=int(item["candidate_index"]),
                selected_global_candidate_index=(
                    slot * CANDIDATES_PER_ACTOR + int(item["candidate_index"])
                ),
                candidate_priority=int(item["priority"]),
                region_eligible_candidate_count=int(item["eligible_count"]),
                nearest_prior_nucleus_distance_squared_m2=(
                    None
                    if item["nearest_prior"] is None
                    else int(item["nearest_prior"])
                ),
                nearest_nucleus_distance_m=nearest_distances[slot],
                global_axis=axes[slot],
                preferred_sign=signs[slot],
                tie_rank=tie_ranks[slot],
                germ_flat_indices=germ_sets[slot],
                crowding_bonus=bonus,
            )
        )
    return controls, tuple(actors)


__all__ = [
    "build_primary_actor_layout",
    "derive_layout_controls",
    "local_coordinates",
    "multiply_high_axis",
    "signed_periodic_delta",
    "toroidal_squared_distance_m2",
    "triangle_wave",
]
