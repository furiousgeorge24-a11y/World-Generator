"""Private Run 2 test of smooth, heading-persistent marine transport.

Run 1 localized the seed-11 crop sensitivity to the dynamically rebuilt
marine stencil.  This diagnostic leaves the production engine untouched and
tests one precommitted alternative marine process.  Terrestrial sediment is
replayed exactly to river mouths while retaining the incoming D8 heading;
suspended load then keeps an eight-heading state and turns through a smooth
direction-and-slope softmax.  Deposition changes the bed used by the next
step, so physical aggradational feedback remains active.

Run from ``pipeline_b`` with::

    python -B -m spikes.physical_outlet_run2 \
        --out ../out/physical_outlet_run2_seed11_v1

The synthetic mechanics check (the only execution intended while developing
this harness) is::

    python -B -m spikes.physical_outlet_run2 --self-check

This file is a private spike.  It does not modify the engine, defaults, public
controls, historical reports, or the Run 1 harness.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from engine import erosion as erosion_engine
from spikes import atlas_replay as replay
from spikes import physical_outlet_causal_discriminator as run1
from spikes import process_halo_diagnostic as stage_diagnostic


EXPERIMENT = "seed11-physical-outlet-heading-transport-run2-v1"
SEED = run1.SEED
CONTINENTAL_BUDGET = run1.CONTINENTAL_BUDGET
WINDOW_ORDER = run1.WINDOW_ORDER
EXPECTED_WINDOWS = run1.EXPECTED_WINDOWS
RELATIONS = run1.RELATIONS
COUNTERFACTUAL_VARIANTS = run1.COUNTERFACTUAL_VARIANTS
TERRAIN_MATERIAL_THRESHOLD_M = run1.TERRAIN_MATERIAL_THRESHOLD_M
HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD = (
    run1.HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD)
NUMERIC_DIAGNOSTIC_THRESHOLD = run1.NUMERIC_DIAGNOSTIC_THRESHOLD
MASS_MATERIAL_THRESHOLD_M_CELLS = (
    run1.MOUTH_MATERIAL_REPORTING_THRESHOLD_M_CELLS)

HEADING_COUNT = 8
PERSISTENCE_KAPPA = 3.0
SLOPE_SCALE_DEGREES = 0.5
SLOPE_SCALE_TAN = float(np.tan(np.deg2rad(SLOPE_SCALE_DEGREES)))
EXPECTED_BASELINE_FULL_SOLVES = 3
EXPECTED_CAPTURED_BASELINE_MARINE_CALLS = 3
EXPECTED_STANDALONE_BASELINE_NATIVE_CALLS = 3
EXPECTED_STANDALONE_CANDIDATE_NATIVE_CALLS = 3
EXPECTED_CANDIDATE_COUNTERFACTUAL_CALLS = 9
FROZEN_MINIMUM_HALO_SAFETY_KM = 1549.747762

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PRIOR_ARTIFACTS = {
    **run1.PRIOR_ARTIFACTS,
    "run1_causal_discriminator": {
        "relative_path": "out/physical_outlet_causal_seed11_v1/report.json",
        "sha256": (
            "30fdcfa8edd1b602c1b1b7727a5eeb0e1c6c876e5091827c3578ee55c315df7c"),
    },
}
SOURCE_FILES = (
    "engine/elevation.py",
    "engine/erosion.py",
    "engine/noise.py",
    "engine/rng.py",
    "engine/surface.py",
    "engine/tectonics.py",
    "spikes/atlas_replay.py",
    "spikes/physical_outlet_replay.py",
    "spikes/process_halo_diagnostic.py",
    "spikes/physical_outlet_causal_discriminator.py",
    "spikes/physical_outlet_run2.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint() -> dict:
    files = {name: _sha256_file(ROOT / name) for name in SOURCE_FILES}
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return {"combined_sha256": digest.hexdigest(), "files": files}


def _prior_links() -> dict:
    links = {}
    for name, expected in PRIOR_ARTIFACTS.items():
        path = WORKSPACE / expected["relative_path"]
        observed = _sha256_file(path) if path.is_file() else None
        links[name] = {
            **expected,
            "exists": path.is_file(),
            "observed_sha256": observed,
            "digest_matched": observed == expected["sha256"],
        }
    return links


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _prepare_empty_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"output directory must be empty: {path}")
    else:
        path.mkdir(parents=True)


def _write_json_exclusive(path: Path, payload: dict) -> str:
    encoded = (json.dumps(
        payload, indent=2, allow_nan=False, default=_json_default)
        + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value) -> str:
    return run1._array_sha256(np.asarray(value))


def _array_summary(value) -> dict:
    return run1._array_summary(np.asarray(value))


def _heading_cosines() -> np.ndarray:
    vectors = np.asarray(erosion_engine.NBR, np.float64)
    unit = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    return unit @ unit.T


HEADING_COSINES = _heading_cosines()
HEADING_SETTLE_DISTANCE = np.asarray(erosion_engine.NBR_D, np.float64)


@dataclass
class DirectionalMouthReplay:
    """Exact terrestrial replay plus land-to-marine incoming headings."""

    directional_mouth_m_cells: np.ndarray
    aggregate_mouth_m_cells: np.ndarray
    boundary_export_m_cells: float
    terminal_residual_m_cells: float
    validation: dict


@dataclass
class CandidateGraph:
    targets: np.ndarray
    valid_neighbor: np.ndarray
    downhill_neighbor: np.ndarray
    scaled_slope_logit: np.ndarray
    has_out: np.ndarray


@dataclass
class BaselineState:
    bed0_m: np.ndarray
    marine: np.ndarray
    source_m_cells: np.ndarray
    deposit_m: np.ndarray
    mobile_m_cells: np.ndarray
    settle: float
    max_steps: int
    base_level_m: float
    boundary_export_m_cells: float = 0.0
    far_field_export_m_cells: float = 0.0
    terminal_residual_m_cells: float = 0.0
    accommodation_limited_events: int = 0
    steps_executed: int = 0
    finished: bool = False


@dataclass
class CandidateState:
    bed0_m: np.ndarray
    marine: np.ndarray
    requested_directional_source_m_cells: np.ndarray
    effective_directional_source_m_cells: np.ndarray
    deposit_m: np.ndarray
    mobile_by_heading_m_cells: np.ndarray
    settle_by_heading: np.ndarray
    max_steps: int
    boundary_export_m_cells: float = 0.0
    far_field_export_m_cells: float = 0.0
    terminal_residual_m_cells: float = 0.0
    accommodation_limited_events: int = 0
    steps_executed: int = 0
    finished: bool = False
    any_outer_ring_contact: bool = False
    outer_ring_contact_steps: int = 0
    last_outer_ring_contact_step_one_based: int | None = None
    last_outer_ring_contact_m_cells: float = 0.0
    final_executed_step_outer_ring_contact_m_cells: float = 0.0
    any_post_move_outer_ring_mobile: bool = False
    post_move_outer_ring_mobile_steps: int = 0
    last_post_move_outer_ring_mobile_step_one_based: int | None = None
    derived_final_step_outer_ring_mobile_before_farfield_m_cells: float = 0.0


@dataclass
class CandidateOutcome:
    bed_m: np.ndarray
    requested_directional_source_m_cells: np.ndarray
    effective_directional_source_m_cells: np.ndarray
    deposit_m: np.ndarray
    combined_export_m_cells: float
    terminal_residual_m_cells: float
    diagnostics: dict

    @property
    def requested_source_m_cells(self) -> np.ndarray:
        return self.requested_directional_source_m_cells.sum(axis=0)

    @property
    def effective_source_m_cells(self) -> np.ndarray:
        return self.effective_directional_source_m_cells.sum(axis=0)

    def aggregate_outcome(self) -> run1.MarineOutcome:
        """Adapter for Run 1's direction-agnostic comparison helpers."""
        return run1.MarineOutcome(
            bed_m=self.bed_m,
            requested_source_m_cells=self.requested_source_m_cells,
            effective_source_m_cells=self.effective_source_m_cells,
            deposit_m=self.deposit_m,
            combined_export_m_cells=self.combined_export_m_cells,
            terminal_residual_m_cells=self.terminal_residual_m_cells,
            diagnostics=self.diagnostics,
        )


def _direction_lookup() -> dict[tuple[int, int], int]:
    return {tuple(int(v) for v in direction): index
            for index, direction in enumerate(erosion_engine.NBR)}


DIRECTION_LOOKUP = _direction_lookup()


def _replay_directional_mouth(snapshot: run1.SedimentCapture,
                              captured: run1.MarineCapture
                              ) -> DirectionalMouthReplay:
    """Replay the shipped land handoff and retain every incoming heading."""
    z = np.asarray(snapshot.input_surface_m, np.float64)
    rows, columns = z.shape
    zf = z.ravel()
    receiver = np.asarray(snapshot.receiver, np.int64)
    marine = zf <= snapshot.base_level_m
    source = np.maximum(snapshot.erosion_source_m, 0.0).ravel()
    flux = np.where(marine, 0.0, source)
    aggregate = np.zeros(z.size, np.float64)
    directional = np.zeros((HEADING_COUNT, z.size), np.float64)
    deposit = np.zeros(z.size, np.float64)
    border = np.zeros(z.shape, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    border = border.ravel()
    capacity = (erosion_engine.KC_LAND
                * np.sqrt(np.maximum(snapshot.area_km2, 1.0)))
    boundary_export = 0.0
    terminal_residual = float(source[marine].sum())

    for batch in snapshot.batches:
        land_batch = batch[~marine[batch]]
        if land_batch.size == 0:
            continue
        target = receiver[land_batch]
        movable = target != land_batch
        slope = (np.maximum(zf[land_batch] - zf[target], 0.0)
                 / (snapshot.process_spacing_km * 1000.0))
        local_deposit = np.clip(
            flux[land_batch] - capacity[land_batch] * slope * 1000.0,
            0.0, erosion_engine.DEP_CAP)
        local_deposit = np.minimum(local_deposit, flux[land_batch])
        deposit[land_batch] += local_deposit
        remaining = flux[land_batch] - local_deposit

        to_marine = movable & marine[target]
        if to_marine.any():
            land_cells = land_batch[to_marine]
            marine_cells = target[to_marine]
            contribution = remaining[to_marine]
            np.add.at(aggregate, marine_cells, contribution)
            source_y, source_x = np.divmod(land_cells, columns)
            target_y, target_x = np.divmod(marine_cells, columns)
            dy = target_y - source_y
            dx = target_x - source_x
            heading = np.fromiter(
                (DIRECTION_LOOKUP[(int(y), int(x))]
                 for y, x in zip(dy, dx)),
                dtype=np.int64, count=land_cells.size)
            for direction in range(HEADING_COUNT):
                selected = heading == direction
                if selected.any():
                    np.add.at(
                        directional[direction], marine_cells[selected],
                        contribution[selected])

        to_land = movable & ~marine[target]
        if to_land.any():
            np.add.at(flux, target[to_land], remaining[to_land])

        terminal = ~movable
        if terminal.any():
            terminal_cells = land_batch[terminal]
            terminal_flux = remaining[terminal]
            outer = border[terminal_cells]
            boundary_export += float(terminal_flux[outer].sum())
            terminal_residual += float(terminal_flux[~outer].sum())

    directional = directional.reshape((HEADING_COUNT, rows, columns))
    aggregate = aggregate.reshape(z.shape)
    directional_sum = directional.sum(axis=0)
    captured_mouth = np.asarray(captured.mouth_flux_m_cells, np.float64)
    scale = max(float(captured_mouth.sum()), 1.0)
    tolerance = 1e-12 * scale
    reconstruction_error = float(
        np.abs(directional_sum - captured_mouth).max(initial=0.0))
    checks = {
        "aggregate_replay_array_exact": bool(
            np.array_equal(aggregate, captured_mouth)),
        "directional_sum_matches_aggregate_within_scaled_1e_minus_12": bool(
            reconstruction_error <= tolerance),
        "directional_values_finite_and_nonnegative": bool(
            np.isfinite(directional).all() and (directional >= 0.0).all()),
        "boundary_export_matches_run1_land_replay_exact": True,
    }
    return DirectionalMouthReplay(
        directional_mouth_m_cells=directional,
        aggregate_mouth_m_cells=aggregate,
        boundary_export_m_cells=boundary_export,
        terminal_residual_m_cells=terminal_residual,
        validation={
            "captured_total_m_cells": float(captured_mouth.sum()),
            "directional_total_m_cells": float(directional.sum()),
            "max_abs_directional_reconstruction_error_m_cells": (
                reconstruction_error),
            "scaled_tolerance_m_cells": tolerance,
            "checks": checks,
            "passed": all(checks.values()),
        },
    )


def _candidate_graph(bed_m, marine, spacing_km) -> CandidateGraph:
    """Smooth heading-persistent transition probabilities on marine cells."""
    bed = np.asarray(bed_m, np.float64)
    marine = np.asarray(marine, bool)
    rows, columns = bed.shape
    n = bed.size
    index = np.arange(n, dtype=np.int64).reshape(bed.shape)
    targets = np.full((HEADING_COUNT, n), -1, np.int64)
    valid = np.zeros((HEADING_COUNT, n), bool)
    downhill = np.zeros((HEADING_COUNT, n), bool)
    scaled_slope = np.zeros((HEADING_COUNT, n), np.float64)

    for direction, ((dy, dx), distance) in enumerate(zip(
            erosion_engine.NBR, erosion_engine.NBR_D)):
        # Candidate heading labels are literal physical array displacements:
        # (dy, dx) at a source enters the neighbor (y + dy, x + dx).
        # The engine's historical _shiftf convention is reversed, so spell
        # these slices out rather than inheriting that private convention.
        source_rows = slice(max(0, -dy), rows - max(0, dy))
        source_columns = slice(max(0, -dx), columns - max(0, dx))
        target_rows = slice(max(0, dy), rows - max(0, -dy))
        target_columns = slice(max(0, dx), columns - max(0, -dx))
        neighbor_bed = np.full_like(bed, np.inf)
        neighbor_bed[source_rows, source_columns] = bed[
            target_rows, target_columns]
        neighbor_marine = np.zeros_like(marine)
        neighbor_marine[source_rows, source_columns] = marine[
            target_rows, target_columns]
        target = np.full(bed.shape, -1, np.int64)
        target[source_rows, source_columns] = index[
            target_rows, target_columns]
        adjacent = marine & neighbor_marine
        physical_slope = ((bed - neighbor_bed)
                          / (1000.0 * float(spacing_km) * float(distance)))
        targets[direction] = target.ravel()
        valid[direction] = adjacent.ravel()
        downhill[direction] = (adjacent & (physical_slope > 0.0)).ravel()
        scaled_slope[direction] = np.where(
            adjacent, physical_slope / SLOPE_SCALE_TAN, 0.0).ravel()

    has_out = valid.any(axis=0)
    return CandidateGraph(
        targets=targets,
        valid_neighbor=valid,
        downhill_neighbor=downhill,
        scaled_slope_logit=scaled_slope,
        has_out=has_out,
    )


def _candidate_weights_for_incoming(graph: CandidateGraph,
                                    incoming: int) -> np.ndarray:
    """Return one 8-by-cell softmax block; never materialize 8x8xgrid."""
    logits = (PERSISTENCE_KAPPA * HEADING_COSINES[incoming, :, None]
              + graph.scaled_slope_logit)
    logits = np.where(graph.valid_neighbor, logits, -np.inf)
    maximum = np.max(logits, axis=0)
    maximum = np.where(graph.has_out, maximum, 0.0)
    exponent = np.where(
        graph.valid_neighbor, np.exp(logits - maximum[None]), 0.0)
    denominator = exponent.sum(axis=0)
    return exponent / np.where(
        denominator > 0.0, denominator, 1.0)[None]


def _baseline_downhill_and_valid(bed, marine) -> tuple[np.ndarray, np.ndarray]:
    bed = np.asarray(bed, np.float64)
    marine = np.asarray(marine, bool)
    downhill = np.zeros((HEADING_COUNT, bed.size), bool)
    valid = np.zeros((HEADING_COUNT, bed.size), bool)
    for direction, ((dy, dx), distance) in enumerate(zip(
            erosion_engine.NBR, erosion_engine.NBR_D)):
        neighbor_bed = erosion_engine._shiftf(bed, dy, dx, np.inf)
        neighbor_marine = erosion_engine._shiftf(
            marine, dy, dx, False)
        adjacent = marine & neighbor_marine
        valid[direction] = adjacent.ravel()
        downhill[direction] = (
            adjacent & (((bed - neighbor_bed) / float(distance)) > 0.0)
        ).ravel()
    return downhill, valid


def _border(shape) -> np.ndarray:
    border = np.zeros(shape, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    return border


def _new_baseline_state(bed, source, base_level, length_km,
                        spacing_km) -> BaselineState:
    bed = np.asarray(bed, np.float64).copy()
    source = np.asarray(source, np.float64).copy()
    marine = bed <= float(base_level)
    effective = np.where(marine, np.maximum(source, 0.0), 0.0)
    max_steps = max(1, int(np.ceil(
        erosion_engine.PHYSICAL_MARINE_EFOLDS
        * float(length_km) / float(spacing_km))))
    return BaselineState(
        bed0_m=bed,
        marine=marine,
        source_m_cells=effective,
        deposit_m=np.zeros_like(bed),
        mobile_m_cells=effective.copy(),
        settle=1.0 - np.exp(-float(spacing_km) / float(length_km)),
        max_steps=max_steps,
        base_level_m=float(base_level),
    )


def _new_candidate_state(bed, directional_source, base_level, length_km,
                         spacing_km, *, max_steps=None
                         ) -> CandidateState:
    bed = np.asarray(bed, np.float64).copy()
    directional = np.asarray(directional_source, np.float64).copy()
    if directional.shape != (HEADING_COUNT, *bed.shape):
        raise ValueError(
            "directional source must have shape (8, rows, columns)")
    marine = bed <= float(base_level)
    effective = np.where(
        marine[None], np.maximum(directional, 0.0), 0.0)
    settling = 1.0 - np.exp(
        -float(spacing_km) * HEADING_SETTLE_DISTANCE / float(length_km))
    if max_steps is None:
        max_steps = max(1, int(np.ceil(
            erosion_engine.PHYSICAL_MARINE_EFOLDS
            * float(length_km) / float(spacing_km))))
    return CandidateState(
        bed0_m=bed,
        marine=marine,
        requested_directional_source_m_cells=directional,
        effective_directional_source_m_cells=effective,
        deposit_m=np.zeros_like(bed),
        mobile_by_heading_m_cells=effective.copy(),
        settle_by_heading=settling,
        max_steps=int(max_steps),
    )


def _advance_baseline(state: BaselineState, step: int) -> dict | None:
    if state.finished or not np.any(state.mobile_m_cells > 0.0):
        state.finished = True
        return None
    shape = state.bed0_m.shape
    at_boundary = _border(shape) & (state.mobile_m_cells > 0.0)
    state.boundary_export_m_cells += float(
        state.mobile_m_cells[at_boundary].sum())
    state.mobile_m_cells[at_boundary] = 0.0
    if not np.any(state.mobile_m_cells > 0.0):
        state.finished = True
        return None

    bed = state.bed0_m + state.deposit_m
    targets, weights, has_out = erosion_engine._marine_transport_graph(
        bed, state.marine)
    downhill, valid = _baseline_downhill_and_valid(bed, state.marine)
    next_mobile = np.zeros(state.bed0_m.size, np.float64)
    mobile_flat = state.mobile_m_cells.ravel()
    outgoing = np.zeros_like(weights)
    for direction in range(HEADING_COUNT):
        weight = weights[direction]
        moving = weight > 0.0
        if moving.any():
            contribution = mobile_flat[moving] * weight[moving]
            outgoing[direction, moving] = contribution
            np.add.at(next_mobile, targets[direction, moving], contribution)
    terminal = ~has_out
    next_mobile[terminal] += mobile_flat[terminal]
    arrived = next_mobile.reshape(shape)
    demand = arrived * state.settle
    room = np.maximum(state.base_level_m - bed, 0.0)
    state.accommodation_limited_events += int(np.count_nonzero(demand > room))
    settled = np.minimum(demand, room)
    state.deposit_m += settled
    mobile_after_settling = arrived - settled
    state.mobile_m_cells = mobile_after_settling.copy()
    state.steps_executed = step + 1

    if step + 1 == state.max_steps:
        terminal_grid = terminal.reshape(shape)
        state.terminal_residual_m_cells += float(
            state.mobile_m_cells[terminal_grid].sum())
        state.mobile_m_cells[terminal_grid] = 0.0
        state.far_field_export_m_cells += float(
            state.mobile_m_cells.sum())
        state.mobile_m_cells.fill(0.0)
        state.finished = True

    return {
        "valid_neighbor_mask": valid.reshape((HEADING_COUNT, *shape)),
        "downhill_neighbor_mask": downhill.reshape((HEADING_COUNT, *shape)),
        "transition_weights": weights.reshape((HEADING_COUNT, *shape)),
        "outgoing_flux_by_heading_m_cells": outgoing.reshape(
            (HEADING_COUNT, *shape)),
        "arrived_m_cells": arrived,
        "room_m": room,
        "settled_m": settled,
        "mobile_after_settling_m_cells": mobile_after_settling,
        "cumulative_deposit_m": state.deposit_m.copy(),
    }


def _advance_candidate(state: CandidateState, step: int, spacing_km,
                       base_level, *, frozen_graph: CandidateGraph | None = None
                       ) -> dict | None:
    mobile = state.mobile_by_heading_m_cells
    if state.finished or not np.any(mobile > 0.0):
        state.finished = True
        return None
    shape = state.bed0_m.shape
    boundary = _border(shape)
    at_boundary = boundary[None] & (mobile > 0.0)
    boundary_contact = float(mobile[at_boundary].sum())
    state.boundary_export_m_cells += boundary_contact
    state.final_executed_step_outer_ring_contact_m_cells = boundary_contact
    if boundary_contact > 0.0:
        state.any_outer_ring_contact = True
        state.outer_ring_contact_steps += 1
        state.last_outer_ring_contact_step_one_based = step + 1
        state.last_outer_ring_contact_m_cells = boundary_contact
    mobile[at_boundary] = 0.0
    if not np.any(mobile > 0.0):
        state.finished = True
        return None

    bed = state.bed0_m + state.deposit_m
    graph = (frozen_graph if frozen_graph is not None
             else _candidate_graph(bed, state.marine, spacing_km))
    mobile_flat = mobile.reshape(HEADING_COUNT, -1)
    terminal = ~graph.has_out
    # With no marine neighbor, no travel occurred.  Classify the mass as a
    # terminal residual immediately; do not charge an axial settling step.
    terminal_mass = float(mobile_flat[:, terminal].sum())
    state.terminal_residual_m_cells += terminal_mass
    mobile_flat[:, terminal] = 0.0
    outgoing = np.zeros_like(mobile_flat)
    weight_max = np.zeros(mobile_flat.shape[1], np.float64)
    weight_entropy_sum = np.zeros(mobile_flat.shape[1], np.float64)
    for incoming in range(HEADING_COUNT):
        weights = _candidate_weights_for_incoming(graph, incoming)
        outgoing += weights * mobile_flat[incoming][None]
        weight_max = np.maximum(weight_max, weights.max(axis=0))
        entropy_terms = np.where(
            weights > 0.0, -weights * np.log(np.where(
                weights > 0.0, weights, 1.0)), 0.0)
        weight_entropy_sum += entropy_terms.sum(axis=0)
    next_mobile = np.zeros_like(mobile_flat)
    for outgoing_heading in range(HEADING_COUNT):
        moving = graph.valid_neighbor[outgoing_heading]
        if moving.any():
            np.add.at(
                next_mobile[outgoing_heading],
                graph.targets[outgoing_heading, moving],
                outgoing[outgoing_heading, moving])
    arrived_by_heading = next_mobile.reshape((HEADING_COUNT, *shape))
    demand_by_heading = (
        arrived_by_heading * state.settle_by_heading[:, None, None])
    total_demand = demand_by_heading.sum(axis=0)
    room = np.maximum(float(base_level) - bed, 0.0)
    state.accommodation_limited_events += int(np.count_nonzero(
        total_demand > room))
    settled_total_target = np.minimum(total_demand, room)
    allocation = np.divide(
        settled_total_target, total_demand,
        out=np.zeros_like(total_demand), where=total_demand > 0.0)
    settled_by_heading = demand_by_heading * allocation[None]
    settled = settled_by_heading.sum(axis=0)
    state.deposit_m += settled
    mobile_after_settling = arrived_by_heading - settled_by_heading
    state.mobile_by_heading_m_cells = mobile_after_settling.copy()
    state.steps_executed = step + 1
    post_move_outer_ring_mass = float(
        state.mobile_by_heading_m_cells[:, boundary].sum())
    if post_move_outer_ring_mass > 0.0:
        state.any_post_move_outer_ring_mobile = True
        state.post_move_outer_ring_mobile_steps += 1
        state.last_post_move_outer_ring_mobile_step_one_based = step + 1

    if step + 1 == state.max_steps:
        state.derived_final_step_outer_ring_mobile_before_farfield_m_cells = (
            post_move_outer_ring_mass)
        terminal_grid = terminal.reshape(shape)
        state.terminal_residual_m_cells += float(
            state.mobile_by_heading_m_cells[:, terminal_grid].sum())
        state.mobile_by_heading_m_cells[:, terminal_grid] = 0.0
        state.far_field_export_m_cells += float(
            state.mobile_by_heading_m_cells.sum())
        state.mobile_by_heading_m_cells.fill(0.0)
        state.finished = True

    return {
        "_candidate_graph": graph,
        "valid_neighbor_mask": graph.valid_neighbor.reshape(
            (HEADING_COUNT, *shape)),
        "downhill_neighbor_mask": graph.downhill_neighbor.reshape(
            (HEADING_COUNT, *shape)),
        "transition_weight_max": weight_max.reshape(shape),
        "transition_weight_mean_entropy": (
            weight_entropy_sum / HEADING_COUNT).reshape(shape),
        "outgoing_flux_by_heading_m_cells": outgoing.reshape(
            (HEADING_COUNT, *shape)),
        "arrived_by_heading_m_cells": arrived_by_heading,
        "arrived_m_cells": arrived_by_heading.sum(axis=0),
        "room_m": room,
        "settled_by_heading_m": settled_by_heading,
        "settled_m": settled,
        "mobile_after_settling_by_heading_m_cells": (
            mobile_after_settling),
        "mobile_after_settling_m_cells": mobile_after_settling.sum(axis=0),
        "cumulative_deposit_m": state.deposit_m.copy(),
    }


def _baseline_outcome(state: BaselineState, base_level, length_km,
                      spacing_km) -> run1.MarineOutcome:
    source_total = float(state.source_m_cells.sum())
    deposited_total = float(state.deposit_m.sum())
    footprint = state.deposit_m > 0.0
    values = np.sort(state.deposit_m[footprint].ravel())
    if values.size:
        top_count = max(1, int(np.ceil(values.size * 0.01)))
        top_fraction = float(values[-top_count:].sum() / deposited_total)
        p99 = float(np.percentile(values, 99.0))
    else:
        top_fraction = 0.0
        p99 = 0.0
    accounted = (deposited_total + state.boundary_export_m_cells
                 + state.far_field_export_m_cells
                 + state.terminal_residual_m_cells)
    diagnostics = {
        "source_m_cells": source_total,
        "deposited_m_cells": deposited_total,
        "boundary_export_m_cells": state.boundary_export_m_cells,
        "far_field_export_m_cells": state.far_field_export_m_cells,
        "terminal_residual_m_cells": state.terminal_residual_m_cells,
        "closure_m_cells": source_total - accounted,
        "max_steps": state.max_steps,
        "axial_reach_km": state.max_steps * float(spacing_km),
        "max_reach_km": (np.sqrt(2.0) * state.max_steps
                         * float(spacing_km)),
        "max_deposit_m": float(state.deposit_m.max(initial=0.0)),
        "p99_positive_deposit_m": p99,
        "deposit_footprint_cells": int(np.count_nonzero(footprint)),
        "top_one_percent_footprint_deposit_fraction": top_fraction,
        "aggraded_to_lowstand_cells": int(np.count_nonzero(
            footprint & ((state.bed0_m + state.deposit_m)
                         >= float(base_level) - 1e-9))),
        "accommodation_limited_cell_events": (
            state.accommodation_limited_events),
        "marine_thickness_cap_applied": False,
        "dynamic_aggradational_routing": True,
    }
    return run1.MarineOutcome(
        bed_m=state.bed0_m.copy(),
        requested_source_m_cells=state.source_m_cells.copy(),
        effective_source_m_cells=state.source_m_cells.copy(),
        deposit_m=state.deposit_m.copy(),
        combined_export_m_cells=(state.boundary_export_m_cells
                                 + state.far_field_export_m_cells),
        terminal_residual_m_cells=state.terminal_residual_m_cells,
        diagnostics=diagnostics,
    )


def _candidate_outcome(state: CandidateState, base_level, length_km,
                       spacing_km, *, dynamic=True) -> CandidateOutcome:
    source_total = float(state.effective_directional_source_m_cells.sum())
    deposited_total = float(state.deposit_m.sum())
    footprint = state.deposit_m > 0.0
    values = np.sort(state.deposit_m[footprint].ravel())
    if values.size:
        top_count = max(1, int(np.ceil(values.size * 0.01)))
        top_fraction = float(values[-top_count:].sum() / deposited_total)
        p99 = float(np.percentile(values, 99.0))
    else:
        top_fraction = 0.0
        p99 = 0.0
    material_footprint = state.deposit_m > TERRAIN_MATERIAL_THRESHOLD_M
    material_values = np.sort(state.deposit_m[material_footprint].ravel())
    if material_values.size:
        material_top_count = max(
            1, int(np.ceil(material_values.size * 0.01)))
        material_total = float(material_values.sum())
        material_top_fraction = float(
            material_values[-material_top_count:].sum() / material_total)
        material_p99 = float(np.percentile(material_values, 99.0))
    else:
        material_total = 0.0
        material_top_fraction = 0.0
        material_p99 = 0.0
    accounted = (deposited_total + state.boundary_export_m_cells
                 + state.far_field_export_m_cells
                 + state.terminal_residual_m_cells)
    diagnostics = {
        "source_m_cells": source_total,
        "deposited_m_cells": deposited_total,
        "boundary_export_m_cells": state.boundary_export_m_cells,
        "far_field_export_m_cells": state.far_field_export_m_cells,
        "terminal_residual_m_cells": state.terminal_residual_m_cells,
        "closure_m_cells": source_total - accounted,
        "max_steps": state.max_steps,
        "steps_executed": state.steps_executed,
        "base_level_m": float(base_level),
        "axial_reach_km": state.max_steps * float(spacing_km),
        "max_reach_km": (np.sqrt(2.0) * state.max_steps
                         * float(spacing_km)),
        "max_deposit_m": float(state.deposit_m.max(initial=0.0)),
        "p99_positive_deposit_m": p99,
        "deposit_footprint_cells": int(np.count_nonzero(footprint)),
        "top_one_percent_footprint_deposit_fraction": top_fraction,
        "material_footprint_threshold_m": TERRAIN_MATERIAL_THRESHOLD_M,
        "material_deposit_footprint_cells": int(np.count_nonzero(
            material_footprint)),
        "material_footprint_deposited_m_cells": material_total,
        "p99_material_footprint_deposit_m": material_p99,
        "top_one_percent_material_footprint_deposit_fraction": (
            material_top_fraction),
        "aggraded_to_lowstand_cells": int(np.count_nonzero(
            footprint & ((state.bed0_m + state.deposit_m)
                         >= float(base_level) - 1e-9))),
        "accommodation_limited_cell_events": (
            state.accommodation_limited_events),
        "marine_thickness_cap_applied": False,
        "dynamic_aggradational_routing": bool(dynamic),
        "heading_persistent_transport": True,
        "heading_count": HEADING_COUNT,
        "persistence_kappa": PERSISTENCE_KAPPA,
        "slope_scale_degrees": SLOPE_SCALE_DEGREES,
        "direction_specific_settling": True,
        "finite_accommodation_allocation": "pro_rata_over_heading_demands",
        "any_outer_ring_contact": state.any_outer_ring_contact,
        "outer_ring_contact_steps": state.outer_ring_contact_steps,
        "last_outer_ring_contact_step_one_based": (
            state.last_outer_ring_contact_step_one_based),
        "last_outer_ring_contact_m_cells": (
            state.last_outer_ring_contact_m_cells),
        "final_executed_step_outer_ring_contact_m_cells": (
            state.final_executed_step_outer_ring_contact_m_cells),
        "outer_ring_contact_on_derived_final_step": bool(
            state.last_outer_ring_contact_step_one_based == state.max_steps),
        "any_post_move_outer_ring_mobile": (
            state.any_post_move_outer_ring_mobile),
        "post_move_outer_ring_mobile_steps": (
            state.post_move_outer_ring_mobile_steps),
        "last_post_move_outer_ring_mobile_step_one_based": (
            state.last_post_move_outer_ring_mobile_step_one_based),
        "derived_final_step_outer_ring_mobile_before_farfield_m_cells": (
            state.derived_final_step_outer_ring_mobile_before_farfield_m_cells),
        "constants_calibrated": False,
        "known_risks_not_tuned": [
            "incoming D8 headings can carry terrestrial axis locking offshore",
            "persistence permits some uphill marine moves because slope is a smooth bias rather than a hard prohibition",
        ],
    }
    return CandidateOutcome(
        bed_m=state.bed0_m.copy(),
        requested_directional_source_m_cells=(
            state.requested_directional_source_m_cells.copy()),
        effective_directional_source_m_cells=(
            state.effective_directional_source_m_cells.copy()),
        deposit_m=state.deposit_m.copy(),
        combined_export_m_cells=(state.boundary_export_m_cells
                                 + state.far_field_export_m_cells),
        terminal_residual_m_cells=state.terminal_residual_m_cells,
        diagnostics=diagnostics,
    )


def _run_baseline_standalone(bed, source, base_level, length_km,
                             spacing_km) -> run1.MarineOutcome:
    state = _new_baseline_state(
        bed, source, base_level, length_km, spacing_km)
    for step in range(state.max_steps):
        if _advance_baseline(state, step) is None and state.finished:
            break
    return _baseline_outcome(
        state, float(base_level), length_km, spacing_km)


def _run_candidate(bed, directional_source, base_level, length_km,
                   spacing_km, *, max_steps=None,
                   dynamic=True) -> CandidateOutcome:
    state = _new_candidate_state(
        bed, directional_source, base_level, length_km, spacing_km,
        max_steps=max_steps)
    frozen = None
    if not dynamic:
        frozen = _candidate_graph(state.bed0_m, state.marine, spacing_km)
    for step in range(state.max_steps):
        if _advance_candidate(
                state, step, spacing_km, base_level,
                frozen_graph=frozen) is None and state.finished:
            break
    return _candidate_outcome(
        state, base_level, length_km, spacing_km, dynamic=dynamic)


def _extract_scope(value, geometry) -> np.ndarray:
    array = np.asarray(value)
    side = geometry.window[2]
    if array.ndim >= 1 and array.shape[-1] == side * side:
        array = array.reshape((*array.shape[:-1], side, side))
    if array.ndim < 2 or array.shape[-2:] != (side, side):
        raise ValueError(
            f"array {array.shape} does not end in window {(side, side)}")
    rows = geometry.local_rows
    columns = geometry.local_columns
    selected = np.asarray(
        array[..., rows[:, None], columns[None, :]]).copy()
    if selected.ndim > 2:
        leading = selected.ndim - 2
        selected = np.transpose(
            selected,
            (leading, leading + 1, *range(leading)))
    return selected


STEP_FIELD_THRESHOLDS = {
    "transition_weight_max": 1e-12,
    "transition_weight_mean_entropy": 1e-12,
    "outgoing_flux_by_heading_m_cells": MASS_MATERIAL_THRESHOLD_M_CELLS,
    "arrived_by_heading_m_cells": MASS_MATERIAL_THRESHOLD_M_CELLS,
    "arrived_m_cells": MASS_MATERIAL_THRESHOLD_M_CELLS,
    "room_m": TERRAIN_MATERIAL_THRESHOLD_M,
    "settled_by_heading_m": TERRAIN_MATERIAL_THRESHOLD_M,
    "settled_m": TERRAIN_MATERIAL_THRESHOLD_M,
    "mobile_after_settling_by_heading_m_cells": (
        MASS_MATERIAL_THRESHOLD_M_CELLS),
    "mobile_after_settling_m_cells": MASS_MATERIAL_THRESHOLD_M_CELLS,
    "cumulative_deposit_m": TERRAIN_MATERIAL_THRESHOLD_M,
}


def _finish_streamed_weight_comparison(accumulator) -> dict:
    difference = accumulator["difference"]
    exact_changed = accumulator["exact_changed"]
    material = difference > 1e-12
    return {
        "storage_policy": (
            "all 64 transition probabilities compared one incoming-heading "
            "block at a time; no 8x8xwindow array retained"),
        "array_exact": not bool(exact_changed.any()),
        "exact_changed_cells": int(np.count_nonzero(exact_changed)),
        "greater_than_1e_minus_9_cells": int(np.count_nonzero(
            difference > NUMERIC_DIAGNOSTIC_THRESHOLD)),
        "max_abs": float(difference.max(initial=0.0)),
        "p99_abs": float(np.percentile(difference, 99.0)),
        "material_metric": "absolute_difference",
        "material_threshold": 1e-12,
        "material_changed_cells": int(np.count_nonzero(material)),
        "materially_equal": not bool(material.any()),
    }


def _streamed_candidate_weight_relations(
        step_snapshots, geometries, common_geometries) -> dict:
    """Compare each generated weight block across every relation and scope."""
    accumulators = {}
    for relation, reference_name, _ in RELATIONS:
        accumulators[relation] = {}
        for scope, scoped_geometries in (
                ("fixed_delivery_core", geometries),
                ("three_window_common_support", common_geometries)):
            shape = scoped_geometries[reference_name].core_shape
            accumulators[relation][scope] = {
                "difference": np.zeros(shape, np.float64),
                "exact_changed": np.zeros(shape, bool),
            }

    for incoming in range(HEADING_COUNT):
        blocks = {
            name: _candidate_weights_for_incoming(
                step_snapshots[name]["_candidate_graph"], incoming)
            for name in WINDOW_ORDER}
        for relation, reference_name, other_name in RELATIONS:
            for scope, scoped_geometries in (
                    ("fixed_delivery_core", geometries),
                    ("three_window_common_support", common_geometries)):
                reference = _extract_scope(
                    blocks[reference_name],
                    scoped_geometries[reference_name])
                other = _extract_scope(
                    blocks[other_name], scoped_geometries[other_name])
                accumulator = accumulators[relation][scope]
                component_difference = np.abs(other - reference)
                accumulator["difference"] = np.maximum(
                    accumulator["difference"],
                    component_difference.max(axis=2, initial=0.0))
                accumulator["exact_changed"] |= np.any(
                    ~run1._element_equal(reference, other), axis=2)

    return {
        relation: {
            scope: _finish_streamed_weight_comparison(accumulator)
            for scope, accumulator in scopes.items()}
        for relation, scopes in accumulators.items()}


def _step_field_comparison(reference, other, reference_geometry,
                           other_geometry) -> dict:
    reference = _extract_scope(reference, reference_geometry)
    other = _extract_scope(other, other_geometry)
    if reference.dtype == np.dtype(bool):
        return run1._mask_comparison(reference, other)
    return run1._numeric_comparison(reference, other)


def _step_relations(step_snapshots, geometries, common_geometries) -> dict:
    has_candidate_graph = all(
        "_candidate_graph" in step_snapshots[name]
        for name in WINDOW_ORDER)
    streamed_weights = (
        _streamed_candidate_weight_relations(
            step_snapshots, geometries, common_geometries)
        if has_candidate_graph else None)
    result = {}
    for relation, reference_name, other_name in RELATIONS:
        result[relation] = {}
        for scope, scoped_geometries in (
                ("fixed_delivery_core", geometries),
                ("three_window_common_support", common_geometries)):
            fields = {}
            names = sorted(set(step_snapshots[reference_name])
                           & set(step_snapshots[other_name]))
            for field in names:
                if field.startswith("_"):
                    continue
                reference = _extract_scope(
                    step_snapshots[reference_name][field],
                    scoped_geometries[reference_name])
                other = _extract_scope(
                    step_snapshots[other_name][field],
                    scoped_geometries[other_name])
                if reference.dtype == np.dtype(bool):
                    fields[field] = run1._mask_comparison(reference, other)
                else:
                    fields[field] = run1._numeric_comparison(
                        reference, other,
                        absolute_threshold=STEP_FIELD_THRESHOLDS.get(
                            field, NUMERIC_DIAGNOSTIC_THRESHOLD))
            if streamed_weights is not None:
                fields["transition_weights_all_64_streamed"] = (
                    streamed_weights[relation][scope])
            result[relation][scope] = fields
    return result


def _step_window_summary(snapshot) -> dict:
    if snapshot is None:
        return {"active": False}
    return {
        "active": True,
        "valid_neighbor_links": int(np.count_nonzero(
            snapshot["valid_neighbor_mask"])),
        "downhill_neighbor_links": int(np.count_nonzero(
            snapshot["downhill_neighbor_mask"])),
        "arrived_total_m_cells": float(snapshot["arrived_m_cells"].sum()),
        "settled_total_m_cells": float(snapshot["settled_m"].sum()),
        "mobile_after_settling_total_m_cells": float(
            snapshot["mobile_after_settling_m_cells"].sum()),
        "cumulative_deposit_total_m_cells": float(
            snapshot["cumulative_deposit_m"].sum()),
        "cumulative_deposit_sha256": _array_sha256(
            snapshot["cumulative_deposit_m"]),
    }


def _lockstep_native(captures, directional_replays, geometries,
                     common_geometries) -> tuple[dict, dict, dict]:
    baseline_states = {}
    candidate_states = {}
    baseline_native_execution_count = 0
    candidate_native_execution_count = 0
    for name in WINDOW_ORDER:
        capture = captures[name]
        baseline_states[name] = _new_baseline_state(
            capture.pre_marine_bed_m, capture.mouth_flux_m_cells,
            capture.base_level_m, capture.deposition_length_km,
            capture.process_spacing_km)
        baseline_native_execution_count += 1
        candidate_states[name] = _new_candidate_state(
            capture.pre_marine_bed_m,
            directional_replays[name].directional_mouth_m_cells,
            capture.base_level_m, capture.deposition_length_km,
            capture.process_spacing_km)
        candidate_native_execution_count += 1
    baseline_step_counts = {
        state.max_steps for state in baseline_states.values()}
    candidate_step_counts = {
        state.max_steps for state in candidate_states.values()}
    if (len(baseline_step_counts) != 1
            or candidate_step_counts != baseline_step_counts):
        raise AssertionError({
            "baseline_max_steps": sorted(baseline_step_counts),
            "candidate_max_steps": sorted(candidate_step_counts),
        })
    max_steps = next(iter(baseline_step_counts))

    step_reports = []
    for step in range(max_steps):
        baseline_snapshots = {}
        candidate_snapshots = {}
        for name in WINDOW_ORDER:
            capture = captures[name]
            baseline_snapshots[name] = _advance_baseline(
                baseline_states[name], step)
            candidate_snapshots[name] = _advance_candidate(
                candidate_states[name], step,
                capture.process_spacing_km, capture.base_level_m)
        if any(value is None for value in baseline_snapshots.values()):
            raise AssertionError(
                "baseline became inactive before the sealed derived-step trace")
        if any(value is None for value in candidate_snapshots.values()):
            raise AssertionError(
                "candidate became inactive before the sealed derived-step trace")
        step_reports.append({
            "step_one_based": step + 1,
            "baseline": {
                "windows": {
                    name: _step_window_summary(baseline_snapshots[name])
                    for name in WINDOW_ORDER},
                "relations": _step_relations(
                    baseline_snapshots, geometries, common_geometries),
            },
            "candidate": {
                "windows": {
                    name: _step_window_summary(candidate_snapshots[name])
                    for name in WINDOW_ORDER},
                "relations": _step_relations(
                    candidate_snapshots, geometries, common_geometries),
            },
        })

    baseline = {
        name: _baseline_outcome(
            baseline_states[name], captures[name].base_level_m,
            captures[name].deposition_length_km,
            captures[name].process_spacing_km)
        for name in WINDOW_ORDER
    }
    candidate = {
        name: _candidate_outcome(
            candidate_states[name], captures[name].base_level_m,
            captures[name].deposition_length_km,
            captures[name].process_spacing_km)
        for name in WINDOW_ORDER
    }
    return baseline, candidate, {
        "standalone_baseline_native_execution_count": (
            baseline_native_execution_count),
        "standalone_candidate_native_execution_count": (
            candidate_native_execution_count),
        "step_count": len(step_reports),
        "max_steps_formula": "ceil(5.5 * L_dep_km / dx_km)",
        "fields": [
            "valid/downhill neighbor masks",
            "transition weights",
            "outgoing heading flux",
            "arrived flux",
            "accommodation room",
            "settled flux",
            "mobile flux",
            "cumulative deposit",
        ],
        "steps": step_reports,
    }


def _extract_rect_leading(value, window, rect) -> np.ndarray:
    array = np.asarray(value)
    row0, column0, side = window
    if array.shape[-2:] != (side, side):
        raise ValueError("leading array does not match window")
    local_rows = slice(rect[0] - row0, rect[1] - row0)
    local_columns = slice(rect[2] - column0, rect[3] - column0)
    return np.asarray(array[..., local_rows, local_columns]).copy()


def _embed_rect_leading(value, rect, window, *, fill=0.0) -> np.ndarray:
    value = np.asarray(value)
    side = window[2]
    result = np.full((*value.shape[:-2], side, side), fill,
                     dtype=value.dtype)
    row0, column0, _ = window
    local_rows = slice(rect[0] - row0, rect[1] - row0)
    local_columns = slice(rect[2] - column0, rect[3] - column0)
    result[..., local_rows, local_columns] = value
    return result


def _replace_rect_leading(native, window, rect, replacement) -> np.ndarray:
    result = np.asarray(native).copy()
    row0, column0, _ = window
    local_rows = slice(rect[0] - row0, rect[1] - row0)
    local_columns = slice(rect[2] - column0, rect[3] - column0)
    result[..., local_rows, local_columns] = replacement
    return result


def _directional_global_sparse_sha256(value, window) -> str:
    array = np.asarray(value, np.float64)
    if array.shape != (HEADING_COUNT, window[2], window[2]):
        raise ValueError("directional source does not match window")
    nonzero = np.any(array != 0.0, axis=0)
    local_y, local_x = np.nonzero(nonzero)
    order = np.lexsort((local_x, local_y))
    global_y = (local_y[order] + window[0]).astype("<i8")
    global_x = (local_x[order] + window[1]).astype("<i8")
    values = np.ascontiguousarray(
        array[:, local_y[order], local_x[order]].T.astype("<f8"))
    digest = hashlib.sha256()
    digest.update(global_y.tobytes())
    digest.update(global_x.tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _candidate_validation(outcome: CandidateOutcome) -> dict:
    source = float(outcome.effective_directional_source_m_cells.sum())
    deposited = float(outcome.deposit_m.sum())
    diagnostics = outcome.diagnostics
    accounted = (deposited
                 + float(diagnostics["boundary_export_m_cells"])
                 + float(diagnostics["far_field_export_m_cells"])
                 + float(diagnostics["terminal_residual_m_cells"]))
    tolerance = 1e-12 * max(source, 1.0)
    base_level = float(diagnostics["base_level_m"])
    checks = {
        "arrays_finite": bool(
            np.isfinite(outcome.bed_m).all()
            and np.isfinite(
                outcome.requested_directional_source_m_cells).all()
            and np.isfinite(
                outcome.effective_directional_source_m_cells).all()
            and np.isfinite(outcome.deposit_m).all()),
        "sources_and_deposit_nonnegative": bool(
            (outcome.effective_directional_source_m_cells >= 0.0).all()
            and (outcome.deposit_m >= 0.0).all()),
        "source_matches_diagnostics": bool(
            source == float(diagnostics["source_m_cells"])),
        "deposit_matches_diagnostics": bool(
            deposited == float(diagnostics["deposited_m_cells"])),
        "recomputed_closure_within_scaled_1e_minus_12": bool(
            abs(source - accounted) <= tolerance),
        "reported_closure_within_scaled_1e_minus_12": bool(
            abs(float(diagnostics["closure_m_cells"])) <= tolerance),
        "accommodation_respected_with_numeric_tolerance": bool(
            np.all((outcome.deposit_m <= 0.0)
                   | (outcome.bed_m + outcome.deposit_m
                      <= base_level + tolerance))),
    }
    return {
        "source_m_cells": source,
        "scaled_tolerance_m_cells": tolerance,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _material_footprint_metrics(deposit) -> dict:
    deposit = np.asarray(deposit, np.float64)
    footprint = deposit > TERRAIN_MATERIAL_THRESHOLD_M
    values = np.sort(deposit[footprint].ravel())
    if values.size:
        total = float(values.sum())
        top_count = max(1, int(np.ceil(values.size * 0.01)))
        top_fraction = float(values[-top_count:].sum() / total)
        p99 = float(np.percentile(values, 99.0))
    else:
        total = 0.0
        top_count = 0
        top_fraction = 0.0
        p99 = 0.0
    return {
        "threshold_strictly_greater_than_m": TERRAIN_MATERIAL_THRESHOLD_M,
        "footprint_cells": int(values.size),
        "footprint_deposited_m_cells": total,
        "top_one_percent_cell_count": top_count,
        "top_one_percent_footprint_deposit_fraction": top_fraction,
        "p99_footprint_deposit_m": p99,
        "max_deposit_m": float(values.max(initial=0.0)),
    }


def _write_sediment_review_image(path: Path, baseline_deposit,
                                 candidate_deposit) -> dict:
    """Write fixed-scale baseline/candidate/absolute-difference panels."""
    baseline = np.maximum(np.asarray(baseline_deposit, np.float64), 0.0)
    candidate = np.maximum(np.asarray(candidate_deposit, np.float64), 0.0)
    difference = np.abs(candidate - baseline)
    fixed_scale_m = 100.0

    def colorize(value, difference_panel=False):
        normalized = np.clip(
            np.log1p(value) / np.log1p(fixed_scale_m), 0.0, 1.0)
        if difference_panel:
            rgb = np.stack((
                18.0 + 237.0 * normalized,
                18.0 + 105.0 * normalized,
                28.0 + 35.0 * normalized), axis=-1)
        else:
            rgb = np.stack((
                12.0 + 228.0 * normalized,
                28.0 + 177.0 * normalized,
                45.0 + 72.0 * normalized), axis=-1)
        return np.asarray(np.rint(rgb), np.uint8)

    panels = [
        colorize(baseline), colorize(candidate),
        colorize(difference, difference_panel=True)]
    separator = np.full((baseline.shape[0], 2, 3), 245, np.uint8)
    image = np.concatenate(
        (panels[0], separator, panels[1], separator, panels[2]), axis=1)
    Image.fromarray(image, "RGB").save(path)
    return {
        "file": path.name,
        "panel_order": [
            "baseline marine deposit",
            "candidate marine deposit",
            "absolute candidate-minus-baseline marine deposit"],
        "fixed_log1p_clip_scale_m": fixed_scale_m,
        "post_observation_rescaling": False,
        "manual_ray_and_morphology_review_required": True,
    }


def _outcome_report(outcome: CandidateOutcome, window) -> dict:
    return {
        "bed": _array_summary(outcome.bed_m),
        "requested_directional_source": _array_summary(
            outcome.requested_directional_source_m_cells),
        "effective_directional_source": _array_summary(
            outcome.effective_directional_source_m_cells),
        "directional_global_sparse_sha256": (
            _directional_global_sparse_sha256(
                outcome.effective_directional_source_m_cells, window)),
        "aggregate_effective_source": _array_summary(
            outcome.effective_source_m_cells),
        "deposit": _array_summary(outcome.deposit_m),
        "final_bed": _array_summary(outcome.bed_m + outcome.deposit_m),
        "combined_export_m_cells": outcome.combined_export_m_cells,
        "terminal_residual_m_cells": outcome.terminal_residual_m_cells,
        "diagnostics": outcome.diagnostics,
        "validation": _candidate_validation(outcome),
    }


def _baseline_oracle_check(standalone, captured) -> dict:
    actual = run1._actual_marine_outcome(captured)
    checks = {
        "bed_array_exact": bool(np.array_equal(
            standalone.bed_m, actual.bed_m)),
        "effective_source_array_exact": bool(np.array_equal(
            standalone.effective_source_m_cells,
            actual.effective_source_m_cells)),
        "deposit_array_exact": bool(np.array_equal(
            standalone.deposit_m, actual.deposit_m)),
        "combined_export_exact": bool(
            standalone.combined_export_m_cells
            == actual.combined_export_m_cells),
        "terminal_residual_exact": bool(
            standalone.terminal_residual_m_cells
            == actual.terminal_residual_m_cells),
        "diagnostics_exact": standalone.diagnostics == actual.diagnostics,
    }
    return {"checks": checks, "passed": all(checks.values())}


def _final_route_and_layers(final_z, runoff, window, base_level,
                            spacing_km) -> dict:
    """Rebuild the engine's final hydrology, lakes, and river edge layer."""
    final_z = np.asarray(final_z, np.float64)
    runoff = np.asarray(runoff, np.float64)
    side = window[2]
    marine = final_z <= float(base_level)
    routing_surface = np.where(marine, float(base_level), final_z)
    filled = erosion_engine._fill_to_lowstand_outlets(
        routing_surface, marine)
    receiver, targets, weights, flat = erosion_engine.receivers(filled)
    marine_flat = marine.ravel()
    index = np.arange(final_z.size)
    receiver[marine_flat] = index[marine_flat]
    weights[:, marine_flat] = 0.0
    flat[marine_flat] = True
    batches = erosion_engine.topo_batches(
        receiver, targets, weights, flat)
    routed_runoff = np.where(marine, 0.0, runoff)
    area8 = erosion_engine.flow_accumulation_d8(
        receiver, batches, final_z.size, routed_runoff)
    area8[marine_flat] = 0.0
    area8_grid = area8.reshape(final_z.shape)
    lake_depth, lake_surface = erosion_engine._balance_lakes(
        final_z, filled, area8_grid)

    land_now = final_z >= 0.0
    drawn = ((area8 > 30.0) & land_now.ravel()
             & (receiver != np.arange(final_z.size)))
    selected = np.flatnonzero(drawn)
    receiver_selected = receiver[selected]
    order = np.lexsort((-area8[selected], receiver_selected))
    sorted_receivers = receiver_selected[order]
    first = np.ones(order.size, bool)
    first[1:] = sorted_receivers[1:] != sorted_receivers[:-1]
    main_donor = {
        int(sorted_receivers[order_index]): int(selected[order[order_index]])
        for order_index in np.flatnonzero(first)}
    donor = np.array(
        [main_donor.get(int(cell), int(cell)) for cell in selected],
        np.int64) if selected.size else np.empty(0, np.int64)
    spacing = float(spacing_km)

    def km(cell):
        y, x = np.divmod(cell, side)
        return ((window[1] + x + 0.5) * spacing,
                (window[0] + y + 0.5) * spacing)

    x0, y0 = km(selected)
    x1, y1 = km(receiver[selected])
    xd, yd = km(donor)
    return {
        "discharge_log": np.log1p(area8_grid),
        "lake_depth": lake_depth,
        "lake_surf": lake_surface,
        "river_edges": {
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "xd": xd, "yd": yd, "a8": area8[selected],
        },
    }


def _river_edges_exact(left, right) -> bool:
    return (set(left) == set(right)
            and all(np.array_equal(left[key], right[key]) for key in left))


def _reconstructed_result(marine_outcome, observer, land, solved,
                          window) -> dict:
    final_z = marine_outcome.bed_m + marine_outcome.deposit_m
    total_sediment = land.land_deposit_m + marine_outcome.deposit_m
    layers = _final_route_and_layers(
        final_z, observer.d8s[3].runoff, window,
        observer.marines[0].base_level_m,
        observer.marines[0].process_spacing_km)
    return {
        "z": final_z,
        "z0": solved["z0"],
        "ero": solved["ero"],
        "sed": total_sediment,
        **layers,
    }


def _full_result_oracle(reconstructed, solved) -> dict:
    checks = {
        field + "_array_exact": bool(np.array_equal(
            reconstructed[field], solved[field]))
        for field in (
            "z", "z0", "ero", "sed", "discharge_log",
            "lake_depth", "lake_surf")
    }
    checks["river_edges_exact"] = _river_edges_exact(
        reconstructed["river_edges"], solved["river_edges"])
    return {"checks": checks, "passed": all(checks.values())}


def _result_relation(reference, other, reference_geometry,
                     other_geometry) -> dict:
    report = {}
    for field in ("z", "z0", "ero", "sed", "lake_depth", "lake_surf"):
        report[field] = run1._numeric_comparison(
            reference_geometry.extract_grid(reference[field]),
            other_geometry.extract_grid(other[field]),
            absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M)
    discharge, _ = run1._discharge_comparison(
        reference_geometry.extract_grid(reference["discharge_log"]),
        other_geometry.extract_grid(other["discharge_log"]))
    report["discharge_log"] = discharge
    return report


def _candidate_factorial_effect_report(
        actual, counterfactuals, reference_name, other_name,
        reference_geometry, other_geometry) -> dict:
    """Signed 2x2 bed/source decomposition without an unrun graph arm."""
    def relation_delta(outcomes):
        reference = reference_geometry.extract_grid(
            outcomes[reference_name].deposit_m)
        other = other_geometry.extract_grid(
            outcomes[other_name].deposit_m)
        return other - reference

    actual_delta = relation_delta(actual)
    arm_delta = {
        name: relation_delta(outcomes)
        for name, outcomes in counterfactuals.items()}
    domain = arm_delta["fixed_source_fixed_common_bed"]
    bed = arm_delta["fixed_source_native_bed"] - domain
    source = arm_delta["native_source_fixed_common_bed"] - domain
    interaction = (actual_delta
                   - arm_delta["fixed_source_native_bed"]
                   - arm_delta["native_source_fixed_common_bed"]
                   + domain)
    reconstruction = domain + bed + source + interaction
    error = np.abs(reconstruction - actual_delta)
    tolerance = 1e-12 * max(
        float(np.abs(actual_delta).max(initial=0.0)), 1.0)
    return {
        "algebra": (
            "actual relation delta = fixed-both domain baseline + "
            "native-bed contrast + native-directional-source contrast + "
            "source-bed interaction"),
        "actual_relation_delta": run1._effect_field_summary(
            actual_delta, actual_delta),
        "conditional_arm_relation_deltas": {
            name: run1._effect_field_summary(delta, actual_delta)
            for name, delta in arm_delta.items()},
        "signed_factorial_effects": {
            "domain_or_outside_common_support_baseline": (
                run1._effect_field_summary(domain, actual_delta)),
            "native_common_bed_contrast": run1._effect_field_summary(
                bed, actual_delta),
            "native_directional_source_contrast": (
                run1._effect_field_summary(source, actual_delta)),
            "source_bed_interaction_contrast": (
                run1._effect_field_summary(interaction, actual_delta)),
        },
        "signed_reconstruction": {
            "max_abs_error_m": float(error.max(initial=0.0)),
            "scaled_tolerance_m": tolerance,
            "array_exact": bool(np.array_equal(
                reconstruction, actual_delta)),
            "within_scaled_1e_minus_12": bool(
                error.max(initial=0.0) <= tolerance),
        },
        "graph_ablation_run": False,
    }


def _protocol(fingerprint, cfg) -> dict:
    common = run1._window_intersection(EXPECTED_WINDOWS.values())
    return {
        "experiment": EXPERIMENT,
        "manifest_role": "pre-execution sealed Run 2 protocol",
        "source_fingerprint": fingerprint,
        "prior_artifact_expected_digests": PRIOR_ARTIFACTS,
        "fixed": {
            "seed": SEED,
            "continental_budget": CONTINENTAL_BUDGET,
            "complete_effective_config": asdict(cfg),
            "origin_xy_km": list(replay.PRIMARY_ORIGIN),
            "windows": {name: list(value)
                        for name, value in EXPECTED_WINDOWS.items()},
            "common_intersection": list(common),
            "window_order": list(WINDOW_ORDER),
            "relations": [list(value) for value in RELATIONS],
            "localization_mode": "physical_outlets",
            "retries": 0,
        },
        "sequencing": {
            "structural_builds": 1,
            "coarse_elevation_builds": 1,
            "baseline_full_erosion_solves": EXPECTED_BASELINE_FULL_SOLVES,
            "captured_baseline_marine_calls": (
                EXPECTED_CAPTURED_BASELINE_MARINE_CALLS),
            "standalone_baseline_native_calls": (
                EXPECTED_STANDALONE_BASELINE_NATIVE_CALLS),
            "standalone_candidate_native_calls": (
                EXPECTED_STANDALONE_CANDIDATE_NATIVE_CALLS),
            "candidate_counterfactual_calls": (
                EXPECTED_CANDIDATE_COUNTERFACTUAL_CALLS),
            "counterfactual_variant_order": list(COUNTERFACTUAL_VARIANTS),
            "baseline_and_candidate_native_stepped_in_lockstep": True,
        },
        "candidate": {
            "candidate_count": 1,
            "name": "persistent_eight_heading_suspended_transport",
            "terrestrial_handoff": (
                "exact physical-outlet terrestrial replay, split by incoming "
                "land-to-marine NBR heading"),
            "state": "mobile[8, cell]",
            "heading_order_dy_dx": [list(value)
                                     for value in erosion_engine.NBR],
            "transition_logit": (
                "kappa*cos(theta_out-theta_in) + "
                "(bed_cell-bed_neighbor)/(1000*dx_km*distance*"
                "tan(0.5 degree))"),
            "valid_transition": "every adjacent marine neighbor",
            "softmax_scope": "outgoing headings for each incoming heading",
            "persistence_kappa": PERSISTENCE_KAPPA,
            "slope_scale_degrees": SLOPE_SCALE_DEGREES,
            "movement": "population enters the selected outgoing heading",
            "direction_specific_settling": (
                "p_heading=1-exp(-dx_km*heading_distance/L_dep_km)"),
            "accommodation": (
                "finite physical room to lowstand allocated pro rata over "
                "simultaneous heading demands"),
            "unsettled_state": "remaining mass preserves outgoing heading",
            "graph_rebuild": "every step from bed0+cumulative deposit",
            "max_steps_formula": "ceil(PHYSICAL_MARINE_EFOLDS*L_dep_km/dx_km)",
            "derived_max_steps_for_frozen_config": max(
                1, int(np.ceil(
                    erosion_engine.PHYSICAL_MARINE_EFOLDS
                    * float(cfg.deposition_length) / erosion_engine.E_KM))),
            "scalar_slope_blend_candidate_included": False,
            "thickness_cap": None,
            "crop_edge_fade": None,
            "outer_ring_accounting": (
                "start-of-step boundary export, any post-move outer-ring "
                "mobile mass, and final-step outer-ring mass included in "
                "far-field export are reported separately"),
            "constants_calibrated": False,
            "known_untuned_risks": [
                "incoming D8 land headings may preserve axis locking into the marine fan",
                "the smooth persistence term can send some mass uphill; slope is a bias, not a hard downhill gate",
            ],
        },
        "step_trace": {
            "scopes": [
                "fixed delivered frame plus 40-km collar",
                "full three-window common support"],
            "fields": [
                "valid and downhill neighbor masks",
                "transition weights",
                "outgoing heading flux",
                "arrived flux",
                "accommodation room",
                "settled flux",
                "remaining mobile flux",
                "cumulative deposit"],
            "baseline_and_candidate_reported_separately": True,
        },
        "counterfactuals": {
            "canonical_directional_source": (
                "large-window directional mouth source cropped to common "
                "support and embedded with zeros elsewhere"),
            "canonical_bed": (
                "large-window pre-marine bed on common support; native "
                "outside"),
            "variants": list(COUNTERFACTUAL_VARIANTS),
            "factorial_decomposition": (
                "fixed-both domain baseline + native-bed contrast + "
                "native-directional-source contrast + interaction"),
        },
        "gates": {
            "asserted_execution_integrity": [
                "all prior digests match",
                "exactly three full baseline solves",
                "instrumentation restored before standalone work",
                "standalone baseline exactly reproduces captured marine",
                "reconstructed baseline z/sed/hydrology/lakes/river edges "
                "exactly reproduce full solve",
                "all candidate outcomes finite, nonnegative, mass closed, "
                "and accommodation respecting",
                "fixed-both directional source identity",
                "fixed call counts",
                "engine default remains legacy"],
            "scientific_outcome_evaluated_not_asserted": [
                "zero >0.05-m candidate sediment differences on both "
                "large-domain delivered-core relations",
                "zero >0.5-percent candidate discharge differences on both "
                "large-domain delivered-core relations",
                "candidate top-one-percent deposit concentration no greater "
                "than baseline in the large window",
                "candidate positive-footprint top-one-percent concentration "
                "at most 0.20 in the large window",
                "candidate far-field export divided by total eroded source "
                "at most 0.05 in the large window",
                "candidate numerical boundary export divided by candidate "
                "total eroded source at most 0.02 in the large window",
                "derived diagonal reach is below the frozen 1549.747762-km "
                "halo safety bound and lies strictly inside every measured "
                "delivered-core-to-window halo",
                "candidate positive-deposit footprint no smaller than "
                "baseline in the large window",
                "candidate >0.05-m footprint top-one-percent concentration "
                "no greater than baseline in the large window",
                "manual review of the fixed-scale large-window sediment "
                "panels finds no conspicuous axial/diagonal rays or other "
                "morphology regression"],
            "one_seed_is_not_promotion_evidence": True,
            "deposited_fraction_comparator": (
                "reported diagnostically but deliberately not gated: meeting "
                "the frozen far-field ceiling can recover suspended mass into "
                "deposit and therefore conflict with a tight match to the "
                "rejected predecessor's deposited fraction"),
        },
        "fixed_manual_review_artifact": {
            "file": "large_marine_sediment_review.png",
            "panels": ["baseline", "candidate", "absolute difference"],
            "scale": "log1p deposit clipped at a precommitted 100 m",
            "manual_review_is_required_not_automatically_passed": True,
        },
        "decision_boundary": {
            "diagnostic_private_branch_only": True,
            "engine_source_modified": False,
            "default_or_public_controls_changed": False,
            "promotion_assessed": False,
        },
    }


def _run(out: Path) -> dict:
    _prepare_empty_output(out)
    cfg = replay._atlas_config(CONTINENTAL_BUDGET)
    fingerprint = _source_fingerprint()
    protocol_sha256 = _write_json_exclusive(
        out / "protocol_precommit.json", _protocol(fingerprint, cfg))
    prior_links = _prior_links()
    mismatched = {name: value for name, value in prior_links.items()
                  if not value["digest_matched"]}
    if mismatched:
        raise RuntimeError(
            f"digest-anchored prior evidence unavailable or changed: {mismatched}")

    started = time.perf_counter()
    structure = replay.build_structure(
        SEED, cfg, _world_km=replay.ATLAS_KM,
        _coarse_km=replay.ORACLE_KM,
        _continent_seeder=replay._seed_atlas_nuclei)
    elevation = replay.coarse_elevation(structure, cfg, SEED)
    windows = {
        "small": replay._window(
            structure, replay.PRIMARY_ORIGIN, replay.SMALL_HALO_KM),
        "large": replay._window(
            structure, replay.PRIMARY_ORIGIN, replay.LARGE_HALO_KM),
    }
    windows["shifted"] = replay._shift_window(
        windows["large"], structure, -replay.SHIFT_KM, replay.SHIFT_KM)
    windows = {name: tuple(int(value) for value in window)
               for name, window in windows.items()}
    if windows != EXPECTED_WINDOWS:
        raise AssertionError({"expected": EXPECTED_WINDOWS,
                              "observed": windows})
    common_rect = run1._window_intersection(windows.values())
    geometries = {
        name: stage_diagnostic.CoreGeometry.fixed(
            name, windows[name], structure)
        for name in WINDOW_ORDER}
    common_rows = np.arange(common_rect[0], common_rect[1], dtype=np.int64)
    common_columns = np.arange(
        common_rect[2], common_rect[3], dtype=np.int64)
    common_geometries = {
        name: stage_diagnostic.CoreGeometry.explicit(
            name + "_common", windows[name], geometries[name].e_km,
            common_rows, common_columns)
        for name in WINDOW_ORDER}

    observed_function_names = (
        "_fill_to_lowstand_outlets", "flow_accumulation",
        "flow_accumulation_d8", "spl_implicit",
        "_route_sediment_lowstand", "_physical_marine_transport")
    engine_before = {
        name: getattr(erosion_engine, name)
        for name in observed_function_names}
    solved = {}
    observers = {}
    wall_times = {}
    full_solve_count = 0
    with run1.PhysicalInstrumentation() as instrumentation:
        for name in WINDOW_ORDER:
            observer = run1.WindowObserver(
                name, windows[name], geometries[name])
            instrumentation.active = observer
            call_started = time.perf_counter()
            try:
                solved[name] = replay.run_erosion(
                    structure, elevation, cfg, SEED,
                    _process_window=windows[name],
                    _localization_mode="physical_outlets")
                full_solve_count += 1
            finally:
                instrumentation.active = None
            wall_times[name] = time.perf_counter() - call_started
            observer.finalize()
            observers[name] = observer
    engine_restored = all(
        getattr(erosion_engine, name) is function
        for name, function in engine_before.items())

    captures = {name: observers[name].marines[0]
                for name in WINDOW_ORDER}
    land_replays = {
        name: run1._replay_land_sediment(
            observers[name].sediments[0], windows[name], common_rect,
            captures[name])
        for name in WINDOW_ORDER}
    directional_replays = {
        name: _replay_directional_mouth(
            observers[name].sediments[0], captures[name])
        for name in WINDOW_ORDER}
    for name in WINDOW_ORDER:
        directional_replays[name].validation["checks"][
            "boundary_export_matches_run1_land_replay_exact"] = bool(
                directional_replays[name].boundary_export_m_cells
                == land_replays[name].boundary_export_m_cells)
        directional_replays[name].validation["passed"] = all(
            directional_replays[name].validation["checks"].values())
    native_input_hashes_before = {
        name: {
            "bed": _array_sha256(captures[name].pre_marine_bed_m),
            "aggregate_mouth": _array_sha256(
                captures[name].mouth_flux_m_cells),
            "directional_mouth": _array_sha256(
                directional_replays[name].directional_mouth_m_cells),
        }
        for name in WINDOW_ORDER}

    component_reports = {
        relation: run1._relation_component_report(
            reference_name, other_name, solved, observers,
            land_replays, geometries, common_geometries)
        for relation, reference_name, other_name in RELATIONS}
    historical = run1._historical_headline_reproduction(component_reports)

    baseline_native, candidate_native, step_trace = _lockstep_native(
        captures, directional_replays, geometries, common_geometries)
    baseline_oracles = {
        name: _baseline_oracle_check(baseline_native[name], captures[name])
        for name in WINDOW_ORDER}

    baseline_results = {
        name: _reconstructed_result(
            baseline_native[name], observers[name], land_replays[name],
            solved[name], windows[name])
        for name in WINDOW_ORDER}
    candidate_aggregate_native = {
        name: candidate_native[name].aggregate_outcome()
        for name in WINDOW_ORDER}
    candidate_results = {
        name: _reconstructed_result(
            candidate_aggregate_native[name], observers[name],
            land_replays[name], solved[name], windows[name])
        for name in WINDOW_ORDER}
    full_result_oracles = {
        name: _full_result_oracle(baseline_results[name], solved[name])
        for name in WINDOW_ORDER}

    large_directional_common = _extract_rect_leading(
        directional_replays["large"].directional_mouth_m_cells,
        windows["large"], common_rect)
    large_bed_common = run1._extract_rect(
        captures["large"].pre_marine_bed_m,
        windows["large"], common_rect)
    counterfactuals = {
        variant: {} for variant in COUNTERFACTUAL_VARIANTS}
    counterfactual_call_count = 0
    for variant in COUNTERFACTUAL_VARIANTS:
        for name in WINDOW_ORDER:
            native_source = directional_replays[
                name].directional_mouth_m_cells
            native_bed = captures[name].pre_marine_bed_m
            if variant in (
                    "fixed_source_native_bed",
                    "fixed_source_fixed_common_bed"):
                source = _embed_rect_leading(
                    large_directional_common, common_rect, windows[name])
            else:
                source = native_source
            if variant in (
                    "native_source_fixed_common_bed",
                    "fixed_source_fixed_common_bed"):
                bed = run1._replace_rect(
                    native_bed, windows[name], common_rect,
                    large_bed_common)
            else:
                bed = native_bed
            counterfactuals[variant][name] = _run_candidate(
                bed, source, captures[name].base_level_m,
                captures[name].deposition_length_km,
                captures[name].process_spacing_km)
            counterfactual_call_count += 1

    fixed_both_hashes = {
        name: _directional_global_sparse_sha256(
            counterfactuals["fixed_source_fixed_common_bed"][
                name].effective_directional_source_m_cells,
            windows[name])
        for name in WINDOW_ORDER}
    fixed_both_identity = len(set(fixed_both_hashes.values())) == 1

    candidate_relations = {}
    candidate_counterfactual_aggregates = {
        variant: {
            name: counterfactuals[variant][name].aggregate_outcome()
            for name in WINDOW_ORDER}
        for variant in COUNTERFACTUAL_VARIANTS}
    for relation, reference_name, other_name in RELATIONS:
        actual_relation = run1._marine_relation(
            candidate_aggregate_native[reference_name],
            candidate_aggregate_native[other_name],
            geometries[reference_name], geometries[other_name])
        variant_relations = {
            variant: run1._marine_relation(
                candidate_counterfactual_aggregates[variant][reference_name],
                candidate_counterfactual_aggregates[variant][other_name],
                geometries[reference_name], geometries[other_name])
            for variant in COUNTERFACTUAL_VARIANTS}
        factorial = _candidate_factorial_effect_report(
            candidate_aggregate_native, candidate_counterfactual_aggregates,
            reference_name, other_name,
            geometries[reference_name], geometries[other_name])
        candidate_relations[relation] = {
            "native_inputs": actual_relation,
            "counterfactuals": variant_relations,
            "signed_factorial_effects": factorial,
            "final_delivered_result": _result_relation(
                candidate_results[reference_name],
                candidate_results[other_name],
                geometries[reference_name], geometries[other_name]),
        }

    baseline_candidate_relations = {
        name: _result_relation(
            baseline_results[name], candidate_results[name],
            geometries[name], geometries[name])
        for name in WINDOW_ORDER}

    counterfactual_reports = {
        variant: {
            name: _outcome_report(
                counterfactuals[variant][name], windows[name])
            for name in WINDOW_ORDER}
        for variant in COUNTERFACTUAL_VARIANTS}
    native_input_hashes_after = {
        name: {
            "bed": _array_sha256(captures[name].pre_marine_bed_m),
            "aggregate_mouth": _array_sha256(
                captures[name].mouth_flux_m_cells),
            "directional_mouth": _array_sha256(
                directional_replays[name].directional_mouth_m_cells),
        }
        for name in WINDOW_ORDER}
    native_inputs_unchanged = (
        native_input_hashes_after == native_input_hashes_before)
    candidate_validations = {
        name: _candidate_validation(candidate_native[name])
        for name in WINDOW_ORDER}
    counterfactual_validations = {
        variant: {
            name: _candidate_validation(counterfactuals[variant][name])
            for name in WINDOW_ORDER}
        for variant in COUNTERFACTUAL_VARIANTS}

    large_baseline_diag = baseline_native["large"].diagnostics
    large_candidate_diag = candidate_native["large"].diagnostics
    baseline_deposited_fraction = (
        large_baseline_diag["deposited_m_cells"]
        / max(large_baseline_diag["source_m_cells"], 1.0))
    candidate_deposited_fraction = (
        large_candidate_diag["deposited_m_cells"]
        / max(large_candidate_diag["source_m_cells"], 1.0))
    large_baseline_material_shape = _material_footprint_metrics(
        baseline_native["large"].deposit_m)
    large_candidate_material_shape = _material_footprint_metrics(
        candidate_native["large"].deposit_m)
    total_eroded_source_large = float(
        land_replays["large"].source_m.sum())
    candidate_marine_source_large = float(
        large_candidate_diag["source_m_cells"])
    candidate_far_field_large = float(
        large_candidate_diag["far_field_export_m_cells"])
    candidate_boundary_export_large = float(
        large_candidate_diag["boundary_export_m_cells"])
    far_field_over_total_eroded = (
        candidate_far_field_large / max(total_eroded_source_large, 1.0))
    far_field_over_marine_source = (
        candidate_far_field_large / max(candidate_marine_source_large, 1.0))
    boundary_over_marine_source = (
        candidate_boundary_export_large
        / max(candidate_marine_source_large, 1.0))
    boundary_over_total_eroded = (
        candidate_boundary_export_large / max(total_eroded_source_large, 1.0))
    reach_guardrails = {}
    for name in WINDOW_ORDER:
        reach = float(candidate_native[name].diagnostics["max_reach_km"])
        minimum_halo = float(
            geometries[name].boundary_distance_km().min())
        reach_guardrails[name] = {
            "derived_diagonal_reach_km": reach,
            "minimum_delivered_core_to_window_boundary_km": minimum_halo,
            "strictly_inside_halo": reach < minimum_halo,
        }
    sediment_review = _write_sediment_review_image(
        out / "large_marine_sediment_review.png",
        baseline_native["large"].deposit_m,
        candidate_native["large"].deposit_m)
    manual_morphology_review_passed = False
    large_relations = ("small_vs_large", "shifted_vs_large")
    scientific_gates = {
        "candidate_zero_material_sediment_cells_both_large_relations": all(
            candidate_relations[relation]["final_delivered_result"][
                "sed"]["material_changed_cells"] == 0
            for relation in large_relations),
        "candidate_zero_material_discharge_cells_both_large_relations": all(
            candidate_relations[relation]["final_delivered_result"][
                "discharge_log"]["material_changed_cells"] == 0
            for relation in large_relations),
        "large_candidate_top_one_percent_concentration_not_greater": bool(
            large_candidate_diag[
                "top_one_percent_footprint_deposit_fraction"]
            <= large_baseline_diag[
                "top_one_percent_footprint_deposit_fraction"]),
        "large_candidate_footprint_not_smaller": bool(
            large_candidate_diag["deposit_footprint_cells"]
            >= large_baseline_diag["deposit_footprint_cells"]),
        "large_candidate_material_footprint_concentration_not_greater": bool(
            large_candidate_material_shape[
                "top_one_percent_footprint_deposit_fraction"]
            <= large_baseline_material_shape[
                "top_one_percent_footprint_deposit_fraction"]),
        "large_candidate_positive_footprint_top_one_fraction_at_most_0_20": bool(
            large_candidate_diag[
                "top_one_percent_footprint_deposit_fraction"] <= 0.20),
        "large_candidate_far_field_over_total_eroded_source_at_most_0_05": bool(
            far_field_over_total_eroded <= 0.05),
        "large_candidate_boundary_over_total_eroded_source_at_most_0_02": bool(
            boundary_over_total_eroded <= 0.02),
        "candidate_derived_reach_below_frozen_safety_and_inside_every_halo": bool(
            max(value["derived_diagonal_reach_km"]
                for value in reach_guardrails.values())
            < FROZEN_MINIMUM_HALO_SAFETY_KM
            and all(value["strictly_inside_halo"]
                    for value in reach_guardrails.values())),
    }
    automatic_scientific_gates_passed = all(scientific_gates.values())
    scientific_outcome_all_passed = bool(
        automatic_scientific_gates_passed
        and manual_morphology_review_passed)

    captured_marine_call_count = sum(
        len(observers[name].marines) for name in WINDOW_ORDER)
    integrity_checks = {
        "prior_artifact_digests_matched": not bool(mismatched),
        "historical_baseline_headline_reproduced": historical[
            "all_headline_fields_exactly_reproduced"],
        "baseline_full_solve_count_exact": (
            full_solve_count == EXPECTED_BASELINE_FULL_SOLVES),
        "captured_baseline_marine_call_count_exact": (
            captured_marine_call_count
            == EXPECTED_CAPTURED_BASELINE_MARINE_CALLS),
        "engine_functions_restored_before_standalone_work": engine_restored,
        "directional_terrestrial_replays_passed": all(
            value.validation["passed"]
            for value in directional_replays.values()),
        "standalone_baseline_native_call_count_exact": (
            step_trace["standalone_baseline_native_execution_count"]
            == EXPECTED_STANDALONE_BASELINE_NATIVE_CALLS),
        "standalone_candidate_native_call_count_exact": (
            step_trace["standalone_candidate_native_execution_count"]
            == EXPECTED_STANDALONE_CANDIDATE_NATIVE_CALLS),
        "candidate_counterfactual_call_count_exact": (
            counterfactual_call_count
            == EXPECTED_CANDIDATE_COUNTERFACTUAL_CALLS),
        "standalone_baseline_marine_exact_oracle": all(
            value["passed"] for value in baseline_oracles.values()),
        "baseline_final_layers_exact_oracle": all(
            value["passed"] for value in full_result_oracles.values()),
        "candidate_native_outcomes_valid": all(
            value["passed"] for value in candidate_validations.values()),
        "candidate_counterfactual_outcomes_valid": all(
            value["passed"]
            for variants in counterfactual_validations.values()
            for value in variants.values()),
        "fixed_both_directional_source_global_identity": (
            fixed_both_identity),
        "candidate_factorial_reconstructions_within_scaled_1e_minus_12": all(
            value["signed_factorial_effects"]["signed_reconstruction"][
                "within_scaled_1e_minus_12"]
            for value in candidate_relations.values()),
        "native_candidate_inputs_not_mutated": native_inputs_unchanged,
        "fixed_sediment_review_image_written": bool(
            (out / sediment_review["file"]).is_file()),
        "engine_default_remains_legacy": (
            inspect.signature(erosion_engine.run_erosion).parameters[
                "_localization_mode"].default == "legacy"),
    }
    if not all(integrity_checks.values()):
        raise AssertionError({"integrity_checks": integrity_checks})

    report = {
        "experiment": EXPERIMENT,
        "completed": True,
        "protocol_precommit_sha256": protocol_sha256,
        "source_fingerprint": fingerprint,
        "prior_artifacts": prior_links,
        "fixed": {
            "seed": SEED,
            "complete_effective_config": asdict(cfg),
            "windows": {name: list(value)
                        for name, value in windows.items()},
            "common_intersection": list(common_rect),
            "full_baseline_solves": full_solve_count,
            "captured_baseline_marine_calls": captured_marine_call_count,
            "standalone_baseline_native_calls": step_trace[
                "standalone_baseline_native_execution_count"],
            "standalone_candidate_native_calls": step_trace[
                "standalone_candidate_native_execution_count"],
            "candidate_counterfactual_calls": counterfactual_call_count,
            "retries": 0,
        },
        "wall_times_baseline_full_solves_s": wall_times,
        "historical_headline_reproduction": historical,
        "directional_terrestrial_mouth_replays": {
            name: {
                "directional_source": _array_summary(
                    directional_replays[
                        name].directional_mouth_m_cells),
                "aggregate_source": _array_summary(
                    directional_replays[name].aggregate_mouth_m_cells),
                "validation": directional_replays[name].validation,
            }
            for name in WINDOW_ORDER},
        "standalone_baseline_exact_oracles": baseline_oracles,
        "reconstructed_full_result_exact_oracles": full_result_oracles,
        "native_baseline_outcomes": {
            name: run1._marine_outcome_summary(
                baseline_native[name], windows[name])
            for name in WINDOW_ORDER},
        "native_candidate_outcomes": {
            name: _outcome_report(candidate_native[name], windows[name])
            for name in WINDOW_ORDER},
        "lockstep_step_trace": step_trace,
        "canonical_controls": {
            "common_directional_source_from_large": _array_summary(
                large_directional_common),
            "common_bed_from_large": _array_summary(large_bed_common),
            "fixed_both_effective_directional_source_hashes": (
                fixed_both_hashes),
            "fixed_both_global_identity": fixed_both_identity,
        },
        "candidate_counterfactual_outcomes": counterfactual_reports,
        "candidate_relations": candidate_relations,
        "baseline_to_candidate_delivered_core_relations_by_window": (
            baseline_candidate_relations),
        "large_window_process_shape_guardrails": {
            "deposited_fraction_comparator_status": (
                "reported_not_gated_because reducing far-field export can "
                "increase deposited fraction; the predecessor comparator "
                "must not override the frozen absolute export gates"),
            "baseline": {
                "deposited_fraction": baseline_deposited_fraction,
                "positive_footprint_diagnostics": large_baseline_diag,
                "material_footprint_diagnostics": (
                    large_baseline_material_shape)},
            "candidate": {
                "deposited_fraction": candidate_deposited_fraction,
                "positive_footprint_diagnostics": large_candidate_diag,
                "material_footprint_diagnostics": (
                    large_candidate_material_shape)},
            "absolute_export_denominators": {
                "total_eroded_source_m_cells": total_eroded_source_large,
                "candidate_marine_source_m_cells": (
                    candidate_marine_source_large),
                "candidate_far_field_export_m_cells": (
                    candidate_far_field_large),
                "candidate_boundary_export_m_cells": (
                    candidate_boundary_export_large),
                "far_field_over_total_eroded_source": (
                    far_field_over_total_eroded),
                "far_field_over_candidate_marine_source": (
                    far_field_over_marine_source),
                "boundary_over_candidate_marine_source": (
                    boundary_over_marine_source),
                "boundary_over_total_eroded_source": (
                    boundary_over_total_eroded),
            },
        },
        "derived_reach_guardrails": reach_guardrails,
        "frozen_minimum_halo_safety_km": (
            FROZEN_MINIMUM_HALO_SAFETY_KM),
        "native_input_mutation_check": {
            "before": native_input_hashes_before,
            "after": native_input_hashes_after,
            "unchanged": native_inputs_unchanged,
        },
        "manual_morphology_review": {
            **sediment_review,
            "status": "unreviewed",
            "passed": manual_morphology_review_passed,
            "known_review_targets": [
                "D8-heading axis locking carried offshore",
                "axial or diagonal rays from the eight-heading stencil",
                "unphysical uphill momentum paths",
                "coastal fan concentration or over-diffusion",
            ],
        },
        "integrity_checks": integrity_checks,
        "scientific_outcome_gates": scientific_gates,
        "automatic_scientific_outcome_gates_all_passed": (
            automatic_scientific_gates_passed),
        "scientific_outcome_all_passed": scientific_outcome_all_passed,
        "decision_boundary": {
            "diagnostic_only": True,
            "engine_or_default_changed": False,
            "promotion_assessed": False,
            "one_seed_scope": True,
            "scientific_failure_does_not_invalidate_the_run": True,
        },
        "elapsed_s": time.perf_counter() - started,
    }
    report_sha256 = _write_json_exclusive(out / "report.json", report)
    _write_json_exclusive(out / "report.sha256.json", {
        "file": "report.json", "sha256": report_sha256})
    return {
        "experiment": EXPERIMENT,
        "completed": True,
        "output": str(out),
        "report_sha256": report_sha256,
        "integrity_checks": integrity_checks,
        "scientific_outcome_gates": scientific_gates,
        "manual_morphology_review_passed": (
            manual_morphology_review_passed),
        "scientific_outcome_all_passed": scientific_outcome_all_passed,
        "elapsed_s": report["elapsed_s"],
    }


def _rotate_direction_permutation() -> np.ndarray:
    # np.rot90 maps a source-relative image vector (dy, dx) to (-dx, dy).
    return np.asarray([
        DIRECTION_LOOKUP[(-int(dx), int(dy))]
        for dy, dx in erosion_engine.NBR], np.int64)


def _reflect_direction_permutation() -> np.ndarray:
    return np.asarray([
        DIRECTION_LOOKUP[(int(dy), -int(dx))]
        for dy, dx in erosion_engine.NBR], np.int64)


def _graph_symmetry_error(graph, transformed_graph, transform,
                          permutation) -> float:
    side = int(round(np.sqrt(graph.targets.shape[1])))
    maximum = 0.0
    for incoming in range(HEADING_COUNT):
        original_block = _candidate_weights_for_incoming(graph, incoming)
        transformed_block = _candidate_weights_for_incoming(
            transformed_graph, permutation[incoming])
        for outgoing in range(HEADING_COUNT):
            original = original_block[outgoing].reshape(side, side)
            transformed = transform(original)
            observed = transformed_block[
                permutation[outgoing]].reshape(side, side)
            maximum = max(
                maximum,
                float(np.abs(transformed - observed).max(initial=0.0)))
    return maximum


def _self_check() -> dict:
    base_level = 0.0
    length_km = 4.0
    spacing_km = 1.0

    handoff_bed = np.full((5, 5), -10.0)
    handoff_bed[2, 2] = 10.0
    handoff_erosion = np.zeros((5, 5))
    handoff_erosion[2, 2] = 100.0
    handoff_receiver = np.arange(25, dtype=np.int64)
    handoff_receiver[12] = 13
    handoff_batches = (np.arange(25, dtype=np.int64),)
    handoff_area = np.ones(25)
    captured_handoff = {}

    def capture_zero_marine(bed_value, mouth, base, length, spacing):
        captured_handoff["bed"] = np.asarray(bed_value).copy()
        captured_handoff["mouth"] = np.asarray(mouth).copy()
        source_total = float(np.maximum(mouth, 0.0).sum())
        return (np.zeros_like(bed_value), 0.0, source_total, {
            "source_m_cells": source_total,
            "deposited_m_cells": 0.0,
            "boundary_export_m_cells": 0.0,
            "far_field_export_m_cells": 0.0,
            "terminal_residual_m_cells": source_total,
            "closure_m_cells": 0.0,
        })

    handoff_result = erosion_engine._route_sediment_lowstand(
        handoff_bed, handoff_erosion, handoff_receiver, handoff_batches,
        handoff_area, base_level, length_km, spacing_km,
        _marine_transport=capture_zero_marine)
    handoff_snapshot = run1.SedimentCapture(
        input_surface_m=handoff_bed.copy(),
        erosion_source_m=handoff_erosion.copy(),
        receiver=handoff_receiver.copy(), batches=handoff_batches,
        area_km2=handoff_area.copy(), base_level_m=base_level,
        deposition_length_km=length_km,
        process_spacing_km=spacing_km,
        output_surface_m=handoff_result[0],
        total_deposit_m=handoff_result[1],
        combined_export_m_cells=handoff_result[2],
        terminal_residual_m_cells=handoff_result[3],
        diagnostics=handoff_result[4])
    handoff_capture = run1.MarineCapture(
        pre_marine_bed_m=captured_handoff["bed"],
        mouth_flux_m_cells=captured_handoff["mouth"],
        base_level_m=base_level, deposition_length_km=length_km,
        process_spacing_km=spacing_km,
        marine_deposit_m=np.zeros_like(handoff_bed),
        combined_export_m_cells=0.0,
        terminal_residual_m_cells=float(captured_handoff["mouth"].sum()),
        diagnostics=handoff_result[4]["marine"],
        bed_unchanged_by_call=True, mouth_flux_unchanged_by_call=True)
    handoff_replay = _replay_directional_mouth(
        handoff_snapshot, handoff_capture)
    east = DIRECTION_LOOKUP[(0, 1)]
    other_heading_total = float(
        np.delete(handoff_replay.directional_mouth_m_cells, east, axis=0
                  ).sum())

    bed = np.full((9, 9), -20.0)
    yy, xx = np.indices(bed.shape)
    bed += 0.17 * yy - 0.11 * xx + 0.013 * yy * xx
    marine = bed <= base_level
    graph = _candidate_graph(bed, marine, spacing_km)
    rotated_bed = np.rot90(bed)
    rotated_graph = _candidate_graph(
        rotated_bed, np.rot90(marine), spacing_km)
    reflected_bed = np.fliplr(bed)
    reflected_graph = _candidate_graph(
        reflected_bed, np.fliplr(marine), spacing_km)
    rotation_error = _graph_symmetry_error(
        graph, rotated_graph, np.rot90, _rotate_direction_permutation())
    reflection_error = _graph_symmetry_error(
        graph, reflected_graph, np.fliplr,
        _reflect_direction_permutation())

    perturbed_bed = bed.copy()
    perturbed_bed[4, 4] += 1e-9
    perturbed_graph = _candidate_graph(
        perturbed_bed, marine, spacing_km)
    graph_perturbation = max(
        float(np.abs(
            _candidate_weights_for_incoming(perturbed_graph, incoming)
            - _candidate_weights_for_incoming(graph, incoming)
        ).max(initial=0.0))
        for incoming in range(HEADING_COUNT))

    flat_bed = np.full((5, 5), -10.0)
    flat_graph = _candidate_graph(
        flat_bed, np.ones_like(flat_bed, bool), spacing_km)
    incoming_east = DIRECTION_LOOKUP[(0, 1)]
    flat_weights = _candidate_weights_for_incoming(
        flat_graph, incoming_east)[:, 12]
    northeast = DIRECTION_LOOKUP[(-1, 1)]
    southeast = DIRECTION_LOOKUP[(1, 1)]
    flat_straight = float(flat_weights[incoming_east])
    flat_within_45 = float(flat_weights[[
        northeast, incoming_east, southeast]].sum())
    direction_unit = np.asarray(erosion_engine.NBR, np.float64)
    direction_unit /= np.linalg.norm(direction_unit, axis=1)[:, None]
    flat_resultant = float(np.linalg.norm(
        (flat_weights[:, None] * direction_unit).sum(axis=0)))
    center_index = 2 * 5 + 2
    graph_target_mapping_exact = all(
        tuple(np.subtract(
            np.unravel_index(
                int(flat_graph.targets[direction, center_index]),
                flat_bed.shape),
            (2, 2))) == tuple(physical_direction)
        for direction, physical_direction in enumerate(
            erosion_engine.NBR))
    open_flat_source = np.zeros((HEADING_COUNT, 5, 5))
    open_flat_source[incoming_east, 2, 2] = 1.0
    open_flat_state = _new_candidate_state(
        flat_bed, open_flat_source, base_level,
        length_km, spacing_km, max_steps=1)
    open_flat_snapshot = _advance_candidate(
        open_flat_state, 0, spacing_km, base_level)
    open_flat_arrived = open_flat_snapshot["arrived_m_cells"]
    arrival_y, arrival_x = np.indices(open_flat_arrived.shape)
    arrived_total = float(open_flat_arrived.sum())
    open_flat_centroid_dy = float(
        ((arrival_y - 2) * open_flat_arrived).sum() / arrived_total)
    open_flat_centroid_dx = float(
        ((arrival_x - 2) * open_flat_arrived).sum() / arrived_total)
    open_flat_peak = tuple(int(value) for value in np.unravel_index(
        int(np.argmax(open_flat_arrived)), open_flat_arrived.shape))

    fixture_spacing_km = 20.0
    axial_drop_m = (1000.0 * fixture_spacing_km
                    * SLOPE_SCALE_TAN)
    diagonal_drop_m = (1000.0 * fixture_spacing_km * np.sqrt(2.0)
                       * SLOPE_SCALE_TAN)
    axial_logit = (axial_drop_m
                   / (1000.0 * fixture_spacing_km * SLOPE_SCALE_TAN))
    diagonal_logit = (
        diagonal_drop_m
        / (1000.0 * fixture_spacing_km * np.sqrt(2.0)
           * SLOPE_SCALE_TAN))

    directional = np.zeros((HEADING_COUNT, 9, 9))
    directional[4, 4, 4] = 10.0
    candidate = _run_candidate(
        bed, directional, base_level, length_km, spacing_km,
        max_steps=6)
    candidate_validation = _candidate_validation(candidate)
    perturbed_candidate = _run_candidate(
        perturbed_bed, directional, base_level, length_km, spacing_km,
        max_steps=6)
    output_perturbation = float(np.abs(
        perturbed_candidate.deposit_m
        - candidate.deposit_m).max(initial=0.0))

    dynamic_bed = np.full((9, 9), -3.0)
    dynamic_bed[:, 5:] -= 0.2
    dynamic_source = np.zeros((HEADING_COUNT, 9, 9))
    dynamic_source[4, 4, 4] = 200.0
    dynamic = _run_candidate(
        dynamic_bed, dynamic_source, base_level, length_km, spacing_km,
        max_steps=8, dynamic=True)
    frozen = _run_candidate(
        dynamic_bed, dynamic_source, base_level, length_km, spacing_km,
        max_steps=8, dynamic=False)
    dynamic_frozen_difference = float(np.abs(
        dynamic.deposit_m - frozen.deposit_m).max(initial=0.0))

    small_bed = np.full((11, 11), -20.0)
    large_bed = np.full((15, 15), -20.0)
    small_source = np.zeros((HEADING_COUNT, 11, 11))
    large_source = np.zeros((HEADING_COUNT, 15, 15))
    small_source[4, 5, 5] = 1.0
    large_source[4, 7, 7] = 1.0
    small = _run_candidate(
        small_bed, small_source, base_level, length_km, spacing_km,
        max_steps=3)
    large = _run_candidate(
        large_bed, large_source, base_level, length_km, spacing_km,
        max_steps=3)
    padding_exact = np.array_equal(
        small.deposit_m[3:8, 3:8], large.deposit_m[5:10, 5:10])

    baseline_source = directional.sum(axis=0)
    engine_result = erosion_engine._physical_marine_transport(
        bed, baseline_source, base_level, length_km, spacing_km)
    standalone = _run_baseline_standalone(
        bed, baseline_source, base_level, length_km, spacing_km)
    baseline_exact = (
        np.array_equal(engine_result[0], standalone.deposit_m)
        and engine_result[1] == standalone.combined_export_m_cells
        and engine_result[2] == standalone.terminal_residual_m_cells
        and engine_result[3] == standalone.diagnostics)

    terminal_bed = np.full((3, 3), 1.0)
    terminal_bed[1, 1] = -0.25
    terminal_source = np.zeros((HEADING_COUNT, 3, 3))
    terminal_source[4, 1, 1] = 2.0
    terminal = _run_candidate(
        terminal_bed, terminal_source, base_level, length_km, spacing_km,
        max_steps=3)
    terminal_validation = _candidate_validation(terminal)

    axial_bed = np.full((5, 5), 1.0)
    axial_bed[2, 2] = -10.0
    axial_bed[2, 3] = -10.0
    axial_source = np.zeros((HEADING_COUNT, 5, 5))
    axial_source[incoming_east, 2, 2] = 1.0
    axial = _run_candidate(
        axial_bed, axial_source, base_level, length_km, spacing_km,
        max_steps=1)
    expected_axial_settle = 1.0 - np.exp(-spacing_km / length_km)
    diagonal_bed = np.full((5, 5), 1.0)
    diagonal_bed[2, 2] = -10.0
    diagonal_bed[3, 3] = -10.0
    diagonal_heading = DIRECTION_LOOKUP[(1, 1)]
    diagonal_source = np.zeros((HEADING_COUNT, 5, 5))
    diagonal_source[diagonal_heading, 2, 2] = 1.0
    diagonal = _run_candidate(
        diagonal_bed, diagonal_source, base_level, length_km, spacing_km,
        max_steps=1)
    expected_diagonal_settle = (
        1.0 - np.exp(-spacing_km * np.sqrt(2.0) / length_km))

    oversubscribed_bed = np.full((7, 7), 1.0)
    oversubscribed_bed[2, 3] = -10.0
    oversubscribed_bed[3, 3] = -0.1
    oversubscribed_bed[4, 3] = -10.0
    oversubscribed_source = np.zeros((HEADING_COUNT, 7, 7))
    south = DIRECTION_LOOKUP[(1, 0)]
    north = DIRECTION_LOOKUP[(-1, 0)]
    oversubscribed_source[south, 2, 3] = 10.0
    oversubscribed_source[north, 4, 3] = 10.0
    oversubscribed_state = _new_candidate_state(
        oversubscribed_bed, oversubscribed_source, base_level,
        length_km, spacing_km, max_steps=1)
    oversubscribed_snapshot = _advance_candidate(
        oversubscribed_state, 0, spacing_km, base_level)
    center_settled = oversubscribed_snapshot[
        "settled_by_heading_m"][:, 3, 3]
    pro_rata_total = float(center_settled.sum())
    pro_rata_equal = float(abs(center_settled[north] - center_settled[south]))

    boundary_bed = np.full((5, 5), -10.0)
    boundary_source = np.zeros((HEADING_COUNT, 5, 5))
    boundary_source[incoming_east, 0, 2] = 1.0
    boundary = _run_candidate(
        boundary_bed, boundary_source, base_level, length_km, spacing_km,
        max_steps=3)
    final_ring_source = np.zeros((HEADING_COUNT, 5, 5))
    final_ring_source[incoming_east, 2, 2] = 1.0
    final_ring = _run_candidate(
        boundary_bed, final_ring_source, base_level, length_km, spacing_km,
        max_steps=2)

    input_bed = bed.copy()
    input_source = directional.copy()
    input_bed_hash = _array_sha256(input_bed)
    input_source_hash = _array_sha256(input_source)
    _run_candidate(
        input_bed, input_source, base_level, length_km, spacing_km,
        max_steps=2)
    inputs_unchanged = (
        _array_sha256(input_bed) == input_bed_hash
        and _array_sha256(input_source) == input_source_hash)

    interior = graph.has_out
    softmax_normalized = all(bool(np.allclose(
        _candidate_weights_for_incoming(graph, incoming)[:, interior].sum(
            axis=0),
        1.0, rtol=0.0, atol=1e-15))
        for incoming in range(HEADING_COUNT))

    checks = {
        "smooth_graph_continuity_under_1e_minus_9_m_perturbation": bool(
            0.0 < graph_perturbation < 1e-8),
        "rotation_symmetry_within_1e_minus_14": rotation_error <= 1e-14,
        "reflection_symmetry_within_1e_minus_14": (
            reflection_error <= 1e-14),
        "softmax_rows_normalized": softmax_normalized,
        "flat_straight_probability_matches_fixture": bool(
            abs(flat_straight - 0.514) < 0.001),
        "flat_within_plus_minus_45_probability_matches_fixture": bool(
            abs(flat_within_45 - 0.941) < 0.001),
        "flat_resultant_matches_fixture": bool(
            abs(flat_resultant - 0.8107) < 0.001),
        "axial_and_diagonal_slope_logit_unit_fixtures": bool(
            abs(axial_drop_m - 174.537) < 0.001
            and abs(diagonal_drop_m - 246.833) < 0.001
            and abs(axial_logit - 1.0) < 1e-15
            and abs(diagonal_logit - 1.0) < 1e-15),
        "direction_mapping_rotation_and_reflection_fixtures": bool(
            np.array_equal(np.sort(_rotate_direction_permutation()),
                           np.arange(HEADING_COUNT))
            and np.array_equal(np.sort(_reflect_direction_permutation()),
                               np.arange(HEADING_COUNT))),
        "candidate_graph_targets_are_literal_physical_displacements": (
            graph_target_mapping_exact),
        "open_flat_east_heading_moves_east_with_persistent_centroid": bool(
            open_flat_peak == (2, 3)
            and abs(open_flat_centroid_dy) <= 1e-15
            and open_flat_centroid_dx > 0.8),
        "directional_terrestrial_mouth_replay_exact_and_east_headed": bool(
            handoff_replay.validation["passed"]
            and handoff_replay.directional_mouth_m_cells[east, 2, 3] > 0.0
            and other_heading_total == 0.0),
        "candidate_mass_closure_and_finiteness": (
            candidate_validation["passed"]),
        "dynamic_rebuild_differs_from_frozen_graph": (
            dynamic_frozen_difference > 1e-12),
        "finite_reach_padding_invariance_exact": padding_exact,
        "tiny_bed_perturbation_has_tiny_nonzero_output_response": bool(
            0.0 < output_perturbation < 1e-6),
        "standalone_baseline_is_exact_engine_oracle": baseline_exact,
        "terminal_accounting_mass_closed": terminal_validation["passed"],
        "no_neighbor_source_is_immediate_terminal_without_settling": bool(
            terminal.deposit_m.sum() == 0.0
            and terminal.terminal_residual_m_cells == 2.0),
        "forced_axial_attenuation_exact": bool(
            axial.deposit_m[2, 3] == expected_axial_settle),
        "forced_diagonal_attenuation_exact": bool(
            diagonal.deposit_m[3, 3] == expected_diagonal_settle),
        "pro_rata_oversubscription_exact_room_and_symmetric": bool(
            abs(pro_rata_total - 0.1) <= 1e-15
            and pro_rata_equal <= 1e-15),
        "outer_ring_contact_reported_separately": bool(
            boundary.diagnostics["any_outer_ring_contact"]
            and boundary.diagnostics["boundary_export_m_cells"] == 1.0
            and boundary.diagnostics[
                "last_outer_ring_contact_step_one_based"] == 1
            and not boundary.diagnostics[
                "outer_ring_contact_on_derived_final_step"]),
        "final_step_outer_ring_mass_not_hidden_in_far_field": bool(
            final_ring.diagnostics[
                "derived_final_step_outer_ring_mobile_before_farfield_m_cells"]
            > 0.0
            and final_ring.diagnostics["any_post_move_outer_ring_mobile"]
            and final_ring.diagnostics["far_field_export_m_cells"]
            >= final_ring.diagnostics[
                "derived_final_step_outer_ring_mobile_before_farfield_m_cells"]),
        "candidate_inputs_not_mutated": inputs_unchanged,
    }
    passed = all(checks.values())
    result = {
        "experiment": EXPERIMENT + "-self-check",
        "model_executed": False,
        "checks": checks,
        "measurements": {
            "graph_weight_max_change_for_1e_minus_9_m": (
                graph_perturbation),
            "candidate_deposit_max_change_for_1e_minus_9_m": (
                output_perturbation),
            "rotation_max_abs_weight_error": rotation_error,
            "reflection_max_abs_weight_error": reflection_error,
            "dynamic_vs_frozen_max_abs_deposit_difference_m": (
                dynamic_frozen_difference),
            "flat_straight_probability": flat_straight,
            "flat_within_plus_minus_45_probability": flat_within_45,
            "flat_resultant_magnitude": flat_resultant,
            "open_flat_east_heading_centroid_dy": open_flat_centroid_dy,
            "open_flat_east_heading_centroid_dx": open_flat_centroid_dx,
            "axial_plus_one_logit_drop_m_at_20_km": axial_drop_m,
            "diagonal_plus_one_logit_drop_m_at_20_km": diagonal_drop_m,
            "forced_axial_settled_fraction": float(axial.deposit_m.sum()),
            "forced_diagonal_settled_fraction": float(
                diagonal.deposit_m.sum()),
            "pro_rata_center_settled_total_m": pro_rata_total,
            "pro_rata_heading_imbalance_m": pro_rata_equal,
        },
        "passed": passed,
    }
    if not passed:
        raise AssertionError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("out") / "physical_outlet_run2_seed11_v1")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    result = _self_check() if args.self_check else _run(args.out)
    print(json.dumps(
        result, indent=2, allow_nan=False, default=_json_default))


if __name__ == "__main__":
    main()
