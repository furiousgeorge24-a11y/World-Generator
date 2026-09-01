"""Receiver-chain provenance for the frozen seed-11 legacy halo replay.

This is a separate, observation-only follow-up to
``process_halo_diagnostic``.  It builds the frozen structure once and runs
only the legacy small/large/shifted process windows.  Pass-through wrappers
capture the pre-sediment D8 graph and sediment inputs/outputs without
changing engine arguments or results.

The analysis is deliberately restricted to cells whose marine deposition
differs by more than 0.05 m.  For those cells it records downstream terminal
and overlap-exit provenance, upstream donor-set differences, and a replay of
the legacy sediment flux that attributes each deposit to sources inside or
outside the two windows' common natural-terrain overlap.

Run from ``pipeline_b`` with::

    python -B -m spikes.process_halo_provenance --out <fresh-directory>

The non-model mechanics check is::

    python -B -m spikes.process_halo_provenance --self-check
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from engine import erosion as erosion_engine
from spikes import atlas_replay as replay
from spikes import process_halo_diagnostic as stage_diagnostic


EXPERIMENT = "seed11-legacy-receiver-provenance-v1"
SEED = 11
CONTINENTAL_BUDGET = 0.65
WINDOW_ORDER = ("small", "large", "shifted")
RELATIONS = (
    ("small_vs_large", "large", "small"),
    ("shifted_vs_large", "large", "shifted"),
)
EXPECTED_WINDOWS = {
    "small": (79, 604, 365),
    "large": (39, 564, 445),
    "shifted": (19, 584, 445),
}
DEPOSIT_MATERIAL_THRESHOLD_M = 0.05
PRIOR_REPORT_RELATIVE = Path(
    "out/process_halo_seed11_stage_v1/report.json")
PRIOR_REPORT_SHA256 = (
    "d6dd696c14e7cc51a990d3a2f639b2dbd21f93c74426bf79ecb9bcdde5d46ae5")
EXPECTED_MATERIAL_TARGETS = {
    "small_vs_large": {"count": 10, "max_abs_m": 150.0},
    "shifted_vs_large": {"count": 123,
                          "max_abs_m": 79.4120593797473},
}
ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "engine/elevation.py",
    "engine/erosion.py",
    "engine/noise.py",
    "engine/rng.py",
    "engine/surface.py",
    "engine/tectonics.py",
    "spikes/atlas_replay.py",
    "spikes/atlas_survey.py",
    "spikes/process_halo_diagnostic.py",
    "spikes/process_halo_provenance.py",
)
MAX_EDGE_EXAMPLES = 12


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


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json_exclusive(path: Path, payload: dict) -> str:
    encoded = (json.dumps(
        payload, indent=2, allow_nan=False, default=_json_default)
        + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _prepare_empty_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"output path is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"output directory must be empty: {path}")
    else:
        path.mkdir(parents=True)


def _array_sha256(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def _coordinate_hash(coordinates) -> str:
    array = np.asarray(sorted(coordinates), np.int64).reshape(-1, 2)
    return _array_sha256(array)


@dataclass
class SedimentSnapshot:
    window_name: str
    window: tuple[int, int, int]
    e_km: float
    filled: np.ndarray
    surface: np.ndarray
    erosion: np.ndarray
    receiver: np.ndarray
    batches: tuple[np.ndarray, ...]
    area_km2: np.ndarray
    base_level_m: float
    deposition_length_km: float
    edge_length_km: np.ndarray | None
    output_surface: np.ndarray
    deposit: np.ndarray
    boundary_export_m_cells: float
    terminal_residual_m_cells: float

    @property
    def side(self) -> int:
        return self.window[2]

    def contains(self, coordinate: tuple[int, int]) -> bool:
        row, column = coordinate
        row0, column0, side = self.window
        return (row0 <= row < row0 + side
                and column0 <= column < column0 + side)

    def local_index(self, coordinate: tuple[int, int]) -> int:
        if not self.contains(coordinate):
            raise ValueError(
                f"coordinate {coordinate} outside {self.window_name}")
        row, column = coordinate
        row0, column0, side = self.window
        return (row - row0) * side + column - column0

    def global_coordinate(self, index: int) -> tuple[int, int]:
        local_row, local_column = divmod(int(index), self.side)
        return (self.window[0] + local_row,
                self.window[1] + local_column)

    def receiver_coordinate(self, coordinate: tuple[int, int]) \
            -> tuple[int, int]:
        index = self.local_index(coordinate)
        return self.global_coordinate(int(self.receiver[index]))

    def is_outer_ring(self, index: int) -> bool:
        row, column = divmod(int(index), self.side)
        return (row == 0 or row == self.side - 1
                or column == 0 or column == self.side - 1)


class WindowObserver:
    def __init__(self, window_name: str, window: tuple[int, int, int],
                 e_km: float):
        self.window_name = window_name
        self.window = window
        self.e_km = float(e_km)
        self.fill_count = 0
        self.route_count = 0
        self.pre_sediment_filled: np.ndarray | None = None
        self.snapshot: SedimentSnapshot | None = None

    def record_fill(self, value) -> None:
        index = self.fill_count
        self.fill_count += 1
        if index == erosion_engine.N_STEPS:
            self.pre_sediment_filled = np.asarray(value).copy()

    def record_sediment(self, z, ero, rcv, batches, area, base_level,
                        length_km, edge_length, result) -> None:
        self.route_count += 1
        if self.snapshot is not None:
            raise AssertionError("legacy sediment route called more than once")
        if self.pre_sediment_filled is None:
            raise AssertionError("pre-sediment fill was not captured")
        self.snapshot = SedimentSnapshot(
            window_name=self.window_name,
            window=self.window,
            e_km=self.e_km,
            filled=self.pre_sediment_filled.copy(),
            surface=np.asarray(z).copy(),
            erosion=np.asarray(ero).copy(),
            receiver=np.asarray(rcv, np.int64).copy(),
            batches=tuple(np.asarray(batch, np.int64).copy()
                          for batch in batches),
            area_km2=np.asarray(area).copy(),
            base_level_m=float(base_level),
            deposition_length_km=float(length_km),
            edge_length_km=(None if edge_length is None else
                            np.asarray(edge_length).copy()),
            output_surface=np.asarray(result[0]).copy(),
            deposit=np.asarray(result[1]).copy(),
            boundary_export_m_cells=float(result[2]),
            terminal_residual_m_cells=float(result[3]),
        )

    def finalize(self) -> SedimentSnapshot:
        expected_fills = erosion_engine.N_STEPS + 2
        if self.fill_count != expected_fills or self.route_count != 1:
            raise AssertionError({
                "window": self.window_name,
                "expected_fill_calls": expected_fills,
                "actual_fill_calls": self.fill_count,
                "expected_sediment_calls": 1,
                "actual_sediment_calls": self.route_count,
            })
        assert self.snapshot is not None
        return self.snapshot


class LegacyInstrumentation(AbstractContextManager):
    """Pass-through wrappers restored when the context exits."""

    def __init__(self):
        self.active: WindowObserver | None = None
        self.original_fill: Callable | None = None
        self.original_sediment: Callable | None = None

    def __enter__(self):
        self.original_fill = erosion_engine.fill_depressions
        self.original_sediment = erosion_engine.route_sediment

        def fill(h, *args, **kwargs):
            result = self.original_fill(h, *args, **kwargs)
            if self.active is not None:
                self.active.record_fill(result)
            return result

        def sediment(z, ero, rcv, batches, area, base_level,
                     length_km, dx_km, edge_len_km=None):
            result = self.original_sediment(
                z, ero, rcv, batches, area, base_level,
                length_km, dx_km, edge_len_km)
            if self.active is not None:
                if not np.isclose(dx_km, self.active.e_km,
                                  rtol=0.0, atol=1e-12):
                    raise AssertionError("unexpected process spacing")
                self.active.record_sediment(
                    z, ero, rcv, batches, area, base_level,
                    length_km, edge_len_km, result)
            return result

        erosion_engine.fill_depressions = fill
        erosion_engine.route_sediment = sediment
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        assert self.original_fill is not None
        assert self.original_sediment is not None
        erosion_engine.fill_depressions = self.original_fill
        erosion_engine.route_sediment = self.original_sediment
        self.active = None
        return False


@dataclass
class FluxReplay:
    source: np.ndarray
    before_deposit: np.ndarray
    remaining: np.ndarray
    deposit: np.ndarray
    survival_fraction: np.ndarray
    boundary_export_m_cells: float
    terminal_residual_m_cells: float
    validation: dict


def _replay_legacy_sediment(snapshot: SedimentSnapshot) -> FluxReplay:
    """Reproduce legacy routing while retaining per-cell flux states."""
    z = snapshot.surface
    zf = z.ravel()
    receiver = snapshot.receiver
    source = np.maximum(snapshot.erosion, 0.0).ravel()
    flux = source.copy()
    before = np.full(source.shape, np.nan, np.float64)
    remaining = np.zeros(source.shape, np.float64)
    deposit = np.zeros(source.shape, np.float64)
    marine = zf < snapshot.base_level_m
    border = np.zeros(z.shape, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    border = border.ravel()
    edge_length = snapshot.edge_length_km
    if edge_length is None:
        settle = 1.0 - np.exp(
            -snapshot.e_km / snapshot.deposition_length_km)
    else:
        edge_length = np.asarray(edge_length, np.float64)
        settle = 1.0 - np.exp(
            -edge_length / snapshot.deposition_length_km)
    capacity = (erosion_engine.KC_LAND
                * np.sqrt(np.maximum(snapshot.area_km2, 1.0)))
    export = 0.0
    residual = 0.0

    for batch in snapshot.batches:
        before[batch] = flux[batch]
        target = receiver[batch]
        movable = target != batch
        length = snapshot.e_km if edge_length is None else edge_length[batch]
        slope = np.maximum(zf[batch] - zf[target], 0.0) / (length * 1000.0)
        land = ~marine[batch]
        land_deposit = np.clip(
            flux[batch] - capacity[batch] * slope * 1000.0,
            0.0, erosion_engine.DEP_CAP) * land
        room = np.maximum(snapshot.base_level_m - zf[batch], 0.0)
        local_settle = settle if edge_length is None else settle[batch]
        marine_deposit = np.minimum(
            np.minimum(flux[batch] * local_settle,
                       erosion_engine.MAR_CAP), room) * marine[batch]
        local_deposit = np.minimum(
            land_deposit + marine_deposit, flux[batch])
        deposit[batch] += local_deposit
        rem = flux[batch] - local_deposit
        remaining[batch] = rem
        terminal = ~movable
        export += float(rem[terminal & border[batch]].sum())
        residual += float(rem[terminal & ~border[batch]].sum())
        np.add.at(flux, target[movable], rem[movable])

    if np.isnan(before).any():
        raise AssertionError("sediment batches did not process every cell")
    expected_deposit = snapshot.deposit.ravel()
    expected_surface = snapshot.output_surface
    replay_surface = snapshot.surface + deposit.reshape(snapshot.surface.shape)
    validation = {
        "deposit_array_exact": bool(np.array_equal(
            deposit, expected_deposit)),
        "surface_array_exact": bool(np.array_equal(
            replay_surface, expected_surface)),
        "boundary_export_exact": bool(
            export == snapshot.boundary_export_m_cells),
        "terminal_residual_exact": bool(
            residual == snapshot.terminal_residual_m_cells),
        "deposit_max_abs_difference_m": float(
            np.max(np.abs(deposit - expected_deposit), initial=0.0)),
    }
    if not all(value for key, value in validation.items()
               if key.endswith("_exact")):
        raise AssertionError(
            f"sediment replay failed for {snapshot.window_name}: {validation}")
    survival = np.divide(
        remaining, before, out=np.zeros_like(remaining), where=before > 0.0)
    return FluxReplay(
        source=source,
        before_deposit=before,
        remaining=remaining,
        deposit=deposit,
        survival_fraction=survival,
        boundary_export_m_cells=export,
        terminal_residual_m_cells=residual,
        validation=validation,
    )


def _overlap(a: SedimentSnapshot, b: SedimentSnapshot) \
        -> tuple[int, int, int, int]:
    ar, ac, an = a.window
    br, bc, bn = b.window
    return (max(ar, br), min(ar + an, br + bn),
            max(ac, bc), min(ac + an, bc + bn))


def _inside_overlap(coordinate: tuple[int, int], overlap) -> bool:
    row, column = coordinate
    row_min, row_max, column_min, column_max = overlap
    return (row_min <= row < row_max
            and column_min <= column < column_max)


def _path_summary(snapshot: SedimentSnapshot,
                  start: tuple[int, int], overlap) -> dict:
    current = start
    visited = set()
    digest = hashlib.sha256()
    hops = 0
    length_km = 0.0
    first_overlap_exit = None
    while True:
        if current in visited:
            return {
                "status": "cycle", "hops": hops,
                "path_length_km": length_km,
                "path_sha256": digest.hexdigest(),
                "cycle_coordinate": list(current),
                "first_common_overlap_exit": first_overlap_exit,
            }
        visited.add(current)
        index = snapshot.local_index(current)
        target_index = int(snapshot.receiver[index])
        target = snapshot.global_coordinate(target_index)
        digest.update(np.asarray(
            [current[0], current[1], target[0], target[1]],
            np.int64).tobytes())
        if target_index == index:
            return {
                "status": "terminal",
                "terminal_coordinate": list(current),
                "terminal_type": (
                    "numerical_outer_ring_self_receiver"
                    if snapshot.is_outer_ring(index)
                    else "interior_self_receiver"),
                "reaches_numerical_outer_ring": bool(
                    snapshot.is_outer_ring(index)),
                "hops": hops,
                "path_length_km": length_km,
                "path_sha256": digest.hexdigest(),
                "first_common_overlap_exit": first_overlap_exit,
            }
        if (first_overlap_exit is None
                and _inside_overlap(current, overlap)
                and not _inside_overlap(target, overlap)):
            first_overlap_exit = {
                "hop": hops,
                "edge_from_global_row_column": list(current),
                "edge_to_global_row_column": list(target),
            }
        dy = target[0] - current[0]
        dx = target[1] - current[1]
        length_km += float(np.hypot(dy, dx) * snapshot.e_km)
        hops += 1
        if hops > snapshot.receiver.size:
            raise AssertionError("receiver trace exceeded graph size")
        current = target


def _first_path_divergence(a: SedimentSnapshot,
                           b: SedimentSnapshot,
                           start: tuple[int, int], overlap) -> dict:
    current = start
    visited = set()
    for hop in range(max(a.receiver.size, b.receiver.size) + 1):
        if current in visited:
            return {"status": "shared_cycle", "hop": hop,
                    "coordinate": list(current)}
        visited.add(current)
        if not a.contains(current) or not b.contains(current):
            return {"status": "comparison_left_common_domain",
                    "hop": hop, "coordinate": list(current)}
        receiver_a = a.receiver_coordinate(current)
        receiver_b = b.receiver_coordinate(current)
        if receiver_a != receiver_b:
            return {
                "status": "receiver_edge_diverged",
                "hop": hop,
                "source_global_row_column": list(current),
                f"{a.window_name}_receiver_global_row_column": (
                    list(receiver_a)),
                f"{b.window_name}_receiver_global_row_column": (
                    list(receiver_b)),
                "source_inside_common_overlap": bool(
                    _inside_overlap(current, overlap)),
            }
        if receiver_a == current:
            return {"status": "shared_terminal", "hop": hop,
                    "terminal_global_row_column": list(current)}
        if not _inside_overlap(receiver_a, overlap):
            return {
                "status": "identical_until_common_overlap_exit",
                "hop": hop,
                "edge_from_global_row_column": list(current),
                "edge_to_global_row_column": list(receiver_a),
            }
        current = receiver_a
    raise AssertionError("path comparison exceeded graph size")


def _children(snapshot: SedimentSnapshot) -> tuple[np.ndarray, np.ndarray]:
    child = np.arange(snapshot.receiver.size, dtype=np.int64)
    movable = snapshot.receiver != child
    child = child[movable]
    parent = snapshot.receiver[movable]
    order = np.argsort(parent, kind="stable")
    return parent[order], child[order]


def _upstream_attribution(snapshot: SedimentSnapshot,
                          flux: FluxReplay,
                          target_coordinate: tuple[int, int],
                          overlap,
                          child_index) -> dict:
    sorted_parent, sorted_child = child_index
    target = snapshot.local_index(target_coordinate)
    deposit_fraction = (
        0.0 if flux.before_deposit[target] <= 0.0 else
        flux.deposit[target] / flux.before_deposit[target])
    stack = [(target, 1.0, 0)]
    seen = set()
    graph_coordinates = []
    positive_coordinates = []
    hop_by_coordinate = {}
    raw_source_inside = 0.0
    raw_source_outside = 0.0
    inbound_inside = 0.0
    inbound_outside = 0.0
    while stack:
        node, factor, hops = stack.pop()
        if node in seen:
            raise AssertionError("cycle encountered in upstream traversal")
        seen.add(node)
        coordinate = snapshot.global_coordinate(node)
        graph_coordinates.append(coordinate)
        hop_by_coordinate[coordinate] = hops
        inside = _inside_overlap(coordinate, overlap)
        source = float(flux.source[node])
        inbound = source * factor
        if source > 0.0:
            positive_coordinates.append(coordinate)
            if inside:
                raw_source_inside += source
                inbound_inside += inbound
            else:
                raw_source_outside += source
                inbound_outside += inbound
        first = int(np.searchsorted(sorted_parent, node, side="left"))
        last = int(np.searchsorted(sorted_parent, node, side="right"))
        for child in sorted_child[first:last]:
            child = int(child)
            stack.append((
                child,
                factor * float(flux.survival_fraction[child]),
                hops + 1,
            ))
    reconstructed_inbound = inbound_inside + inbound_outside
    reconstructed_deposit = reconstructed_inbound * deposit_fraction
    deposit = float(flux.deposit[target])
    return {
        "target_local_index": target,
        "graph_donor_count": len(graph_coordinates),
        "graph_donor_coordinates_sha256": _coordinate_hash(
            graph_coordinates),
        "positive_source_donor_count": len(positive_coordinates),
        "positive_source_coordinates_sha256": _coordinate_hash(
            positive_coordinates),
        "raw_source_m_cells": {
            "inside_common_overlap": raw_source_inside,
            "outside_common_overlap": raw_source_outside,
            "total": raw_source_inside + raw_source_outside,
        },
        "inbound_flux_m_cells": {
            "inside_common_overlap": inbound_inside,
            "outside_common_overlap": inbound_outside,
            "reconstructed_total": reconstructed_inbound,
            "engine_total": float(flux.before_deposit[target]),
            "reconstruction_abs_error": abs(
                reconstructed_inbound - flux.before_deposit[target]),
        },
        "deposit_attribution_m": {
            "inside_common_overlap": inbound_inside * deposit_fraction,
            "outside_common_overlap": inbound_outside * deposit_fraction,
            "reconstructed_total": reconstructed_deposit,
            "engine_total": deposit,
            "reconstruction_abs_error": abs(
                reconstructed_deposit - deposit),
        },
        "deposit_fraction_of_inbound": deposit_fraction,
        "_graph_coordinates": set(graph_coordinates),
        "_positive_coordinates": set(positive_coordinates),
        "_hop_by_coordinate": hop_by_coordinate,
    }


def _public_attribution(attribution: dict) -> dict:
    return {key: value for key, value in attribution.items()
            if not key.startswith("_")}


def _target_physics(snapshot: SedimentSnapshot,
                    flux: FluxReplay,
                    coordinate: tuple[int, int]) -> dict:
    index = snapshot.local_index(coordinate)
    target = int(snapshot.receiver[index])
    surface = float(snapshot.surface.ravel()[index])
    before = float(flux.before_deposit[index])
    marine = surface < snapshot.base_level_m
    if snapshot.edge_length_km is None:
        length = snapshot.e_km
    else:
        length = float(snapshot.edge_length_km[index])
    settle = 1.0 - np.exp(
        -length / snapshot.deposition_length_km)
    room = max(snapshot.base_level_m - surface, 0.0)
    candidates = {
        "available_flux_m_cells": before,
        "settling_limited_m": before * settle if marine else None,
        "marine_cap_m": erosion_engine.MAR_CAP if marine else None,
        "accommodation_room_m": room if marine else None,
    }
    numeric = {key: value for key, value in candidates.items()
               if value is not None}
    minimum = min(numeric.values()) if numeric else 0.0
    active = [key for key, value in numeric.items()
              if np.isclose(value, minimum, rtol=1e-12, atol=1e-12)]
    return {
        "surface_m": surface,
        "filled_surface_m": float(snapshot.filled.ravel()[index]),
        "initial_source_m_cells": float(flux.source[index]),
        "inbound_before_deposit_m_cells": before,
        "deposit_m": float(flux.deposit[index]),
        "remaining_after_deposit_m_cells": float(flux.remaining[index]),
        "receiver_global_row_column": list(
            snapshot.global_coordinate(target)),
        "receiver_is_self": bool(target == index),
        "marine_at_lowstand": bool(marine),
        "link_length_km": length,
        "settling_fraction": float(settle),
        "limiting_candidates": candidates,
        "active_minimum_constraints": active,
    }


def _receiver_edge_differences(a: SedimentSnapshot,
                               b: SedimentSnapshot,
                               attr_a: dict, attr_b: dict,
                               overlap) -> dict:
    union = attr_a["_graph_coordinates"] | attr_b["_graph_coordinates"]
    changed = []
    for coordinate in union:
        if (not _inside_overlap(coordinate, overlap)
                or not a.contains(coordinate)
                or not b.contains(coordinate)):
            continue
        receiver_a = a.receiver_coordinate(coordinate)
        receiver_b = b.receiver_coordinate(coordinate)
        if receiver_a == receiver_b:
            continue
        hop_a = attr_a["_hop_by_coordinate"].get(coordinate)
        hop_b = attr_b["_hop_by_coordinate"].get(coordinate)
        rank = min(value for value in (hop_a, hop_b)
                   if value is not None)
        changed.append((rank, coordinate, receiver_a, receiver_b,
                        hop_a, hop_b))
    changed.sort(key=lambda item: (item[0], item[1]))
    examples = []
    for rank, coordinate, receiver_a, receiver_b, hop_a, hop_b \
            in changed[:MAX_EDGE_EXAMPLES]:
        examples.append({
            "global_row_column": list(coordinate),
            f"{a.window_name}_receiver_global_row_column": list(receiver_a),
            f"{b.window_name}_receiver_global_row_column": list(receiver_b),
            f"{a.window_name}_upstream_hops_to_target": hop_a,
            f"{b.window_name}_upstream_hops_to_target": hop_b,
            "minimum_upstream_hops_to_target": rank,
            f"positive_source_in_{a.window_name}": (
                coordinate in attr_a["_positive_coordinates"]),
            f"positive_source_in_{b.window_name}": (
                coordinate in attr_b["_positive_coordinates"]),
        })
    return {
        "divergent_receiver_edge_count_in_common_overlap_donor_union": (
            len(changed)),
        "nearest_to_target_examples": examples,
    }


def _analyze_target(a: SedimentSnapshot, b: SedimentSnapshot,
                    flux_a: FluxReplay, flux_b: FluxReplay,
                    children_a, children_b,
                    coordinate: tuple[int, int], overlap) -> dict:
    attr_a = _upstream_attribution(
        a, flux_a, coordinate, overlap, children_a)
    attr_b = _upstream_attribution(
        b, flux_b, coordinate, overlap, children_b)
    graph_intersection = (attr_a["_graph_coordinates"]
                          & attr_b["_graph_coordinates"])
    graph_symdiff = (attr_a["_graph_coordinates"]
                     ^ attr_b["_graph_coordinates"])
    positive_intersection = (attr_a["_positive_coordinates"]
                             & attr_b["_positive_coordinates"])
    positive_symdiff = (attr_a["_positive_coordinates"]
                        ^ attr_b["_positive_coordinates"])
    deposit_a = float(
        flux_a.deposit[a.local_index(coordinate)])
    deposit_b = float(
        flux_b.deposit[b.local_index(coordinate)])
    inside_a = attr_a["deposit_attribution_m"][
        "inside_common_overlap"]
    inside_b = attr_b["deposit_attribution_m"][
        "inside_common_overlap"]
    outside_a = attr_a["deposit_attribution_m"][
        "outside_common_overlap"]
    outside_b = attr_b["deposit_attribution_m"][
        "outside_common_overlap"]
    return {
        "global_row_column": list(coordinate),
        "center_yx_km": [
            (coordinate[0] + 0.5) * a.e_km,
            (coordinate[1] + 0.5) * a.e_km,
        ],
        "deposit_difference": {
            f"{a.window_name}_m": deposit_a,
            f"{b.window_name}_m": deposit_b,
            f"{b.window_name}_minus_{a.window_name}_m": deposit_b - deposit_a,
            "absolute_m": abs(deposit_b - deposit_a),
        },
        "downstream": {
            a.window_name: _path_summary(a, coordinate, overlap),
            b.window_name: _path_summary(b, coordinate, overlap),
            "first_path_divergence": _first_path_divergence(
                a, b, coordinate, overlap),
        },
        "local_sediment_physics": {
            a.window_name: _target_physics(a, flux_a, coordinate),
            b.window_name: _target_physics(b, flux_b, coordinate),
        },
        "upstream": {
            a.window_name: _public_attribution(attr_a),
            b.window_name: _public_attribution(attr_b),
            "graph_donor_coordinate_intersection_count": len(
                graph_intersection),
            "graph_donor_coordinate_symmetric_difference_count": len(
                graph_symdiff),
            "positive_source_coordinate_intersection_count": len(
                positive_intersection),
            "positive_source_coordinate_symmetric_difference_count": len(
                positive_symdiff),
            "receiver_edge_differences": _receiver_edge_differences(
                a, b, attr_a, attr_b, overlap),
        },
        "attribution_deltas_m": {
            "inside_common_overlap_signed": inside_b - inside_a,
            "inside_common_overlap_absolute": abs(inside_b - inside_a),
            "outside_common_overlap_signed": outside_b - outside_a,
            "outside_common_overlap_absolute": abs(outside_b - outside_a),
        },
    }


def _core_deposit(snapshot: SedimentSnapshot,
                  geometry: stage_diagnostic.CoreGeometry) -> np.ndarray:
    return geometry.extract_grid(snapshot.deposit)


def _relation_report(name: str, a: SedimentSnapshot,
                     b: SedimentSnapshot,
                     flux_a: FluxReplay, flux_b: FluxReplay,
                     structure) -> dict:
    geometry_a = stage_diagnostic.CoreGeometry.fixed(
        a.window_name, a.window, structure)
    geometry_b = stage_diagnostic.CoreGeometry.fixed(
        b.window_name, b.window, structure)
    if not (np.array_equal(geometry_a.global_rows, geometry_b.global_rows)
            and np.array_equal(geometry_a.global_columns,
                               geometry_b.global_columns)):
        raise AssertionError("comparison cores differ")
    deposit_a = _core_deposit(a, geometry_a)
    deposit_b = _core_deposit(b, geometry_b)
    difference = np.abs(deposit_b - deposit_a)
    material = difference > DEPOSIT_MATERIAL_THRESHOLD_M
    coordinates = [
        (int(geometry_a.global_rows[row]),
         int(geometry_a.global_columns[column]))
        for row, column in np.argwhere(material)
    ]
    expected = EXPECTED_MATERIAL_TARGETS[name]
    observed_max = float(difference.max(initial=0.0))
    prior_reproduced = (
        len(coordinates) == expected["count"]
        and np.isclose(observed_max, expected["max_abs_m"],
                       rtol=1e-12, atol=1e-12)
    )
    if not prior_reproduced:
        raise AssertionError({
            "relation": name,
            "expected": expected,
            "observed_count": len(coordinates),
            "observed_max_abs_m": observed_max,
        })
    overlap = _overlap(a, b)
    children_a = _children(a)
    children_b = _children(b)
    targets = [
        _analyze_target(
            a, b, flux_a, flux_b, children_a, children_b,
            coordinate, overlap)
        for coordinate in coordinates
    ]

    status_counts = {}
    terminal_counts = {a.window_name: {}, b.window_name: {}}
    direct_rim_both = 0
    inside_delta_sum = 0.0
    outside_delta_sum = 0.0
    outside_attribution_targets = 0
    inside_delta_targets = 0
    divergent_upstream_edge_targets = 0
    for target in targets:
        status = target["downstream"]["first_path_divergence"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        reaches = []
        for window_name in (a.window_name, b.window_name):
            path = target["downstream"][window_name]
            terminal = path.get("terminal_type", path["status"])
            counts = terminal_counts[window_name]
            counts[terminal] = counts.get(terminal, 0) + 1
            reaches.append(path.get("reaches_numerical_outer_ring", False))
        direct_rim_both += int(all(reaches))
        delta = target["attribution_deltas_m"]
        inside_delta_sum += delta["inside_common_overlap_absolute"]
        outside_delta_sum += delta["outside_common_overlap_absolute"]
        outside_values = [
            target["upstream"][window_name]["deposit_attribution_m"][
                "outside_common_overlap"]
            for window_name in (a.window_name, b.window_name)
        ]
        outside_attribution_targets += int(
            max(outside_values) > DEPOSIT_MATERIAL_THRESHOLD_M)
        inside_delta_targets += int(
            delta["inside_common_overlap_absolute"]
            > DEPOSIT_MATERIAL_THRESHOLD_M)
        divergent_upstream_edge_targets += int(
            target["upstream"]["receiver_edge_differences"][
                "divergent_receiver_edge_count_in_common_overlap_donor_union"]
            > 0)
    return {
        "relation": name,
        "reference_window": a.window_name,
        "other_window": b.window_name,
        "common_overlap_global_row_column_half_open": {
            "row": [overlap[0], overlap[1]],
            "column": [overlap[2], overlap[3]],
        },
        "material_target_policy": {
            "field": "legacy sediment deposit",
            "absolute_difference_greater_than_m": (
                DEPOSIT_MATERIAL_THRESHOLD_M),
            "comparison_core": "delivered frame plus fixed 40-km collar",
        },
        "prior_stage_report_reproduced": prior_reproduced,
        "material_target_count": len(targets),
        "maximum_absolute_deposit_difference_m": observed_max,
        "aggregate": {
            "first_downstream_path_divergence_status_counts": status_counts,
            "downstream_terminal_type_counts": terminal_counts,
            "targets_reaching_numerical_outer_ring_in_both_windows": (
                direct_rim_both),
            "targets_with_divergent_upstream_receiver_edges_in_common_overlap": (
                divergent_upstream_edge_targets),
            "targets_with_more_than_0_05m_deposit_attributed_to_sources_outside_common_overlap": (
                outside_attribution_targets),
            "targets_with_more_than_0_05m_inside_overlap_attribution_delta": (
                inside_delta_targets),
            "sum_absolute_inside_common_overlap_attribution_delta_m": (
                inside_delta_sum),
            "sum_absolute_outside_common_overlap_attribution_delta_m": (
                outside_delta_sum),
            "sum_absolute_observed_deposit_difference_m": float(
                sum(item["deposit_difference"]["absolute_m"]
                    for item in targets)),
        },
        "targets": targets,
    }


def _prior_report_link() -> dict:
    path = ROOT.parent / PRIOR_REPORT_RELATIVE
    exists = path.is_file()
    actual = _sha256_file(path) if exists else None
    return {
        "relative_path_from_workspace": PRIOR_REPORT_RELATIVE.as_posix(),
        "artifact_exists": exists,
        "expected_sha256": PRIOR_REPORT_SHA256,
        "actual_sha256": actual,
        "digest_matched": actual == PRIOR_REPORT_SHA256,
    }


def _protocol(fingerprint: dict) -> dict:
    return {
        "experiment": EXPERIMENT,
        "manifest_role": "pre-execution fixed legacy provenance protocol",
        "source_fingerprint": fingerprint,
        "fixed": {
            "seed": SEED,
            "continental_budget": CONTINENTAL_BUDGET,
            "origin_xy_km": list(replay.PRIMARY_ORIGIN),
            "windows": {name: list(value)
                        for name, value in EXPECTED_WINDOWS.items()},
            "mode": "legacy",
            "structural_builds": 1,
            "coarse_elevation_builds": 1,
            "erosion_calls": 3,
            "window_order": list(WINDOW_ORDER),
            "retries": 0,
        },
        "target_selection": {
            "source": "prior digest-anchored stage report",
            "prior_report_sha256": PRIOR_REPORT_SHA256,
            "field": "sediment.output.deposit_m",
            "absolute_difference_greater_than_m": (
                DEPOSIT_MATERIAL_THRESHOLD_M),
            "expected": EXPECTED_MATERIAL_TARGETS,
            "reselection": False,
        },
        "captures": [
            "full pre-sediment filled surface",
            "full pre-sediment D8 receiver graph and batches",
            "full erosion source, drainage area, and surface",
            "legacy sediment input/output and flux replay",
            "changed-target downstream terminals and overlap exits",
            "changed-target upstream donors and inside/outside-overlap mass",
        ],
        "decision_boundary": {
            "numerical_outer_ring_is_process_window_not_delivered_frame": True,
            "natural_added_terrain_is_not_automatically_invalid": True,
            "outside_overlap_source_attribution_is_reported_separately": True,
            "no_contour_or_frame-alignment_gate": True,
            "diagnostic_only": True,
            "engine_behavior_modified": False,
        },
    }


def _run(out: Path) -> dict:
    _prepare_empty_output(out)
    fingerprint = _source_fingerprint()
    protocol_sha256 = _write_json_exclusive(
        out / "protocol_precommit.json", _protocol(fingerprint))
    prior_link = _prior_report_link()
    if not prior_link["digest_matched"]:
        raise RuntimeError(f"prior stage report unavailable or changed: {prior_link}")

    cfg = replay._atlas_config(CONTINENTAL_BUDGET)
    started = time.perf_counter()
    structure = replay.build_structure(
        SEED, cfg,
        _world_km=replay.ATLAS_KM,
        _coarse_km=replay.ORACLE_KM,
        _continent_seeder=replay._seed_atlas_nuclei,
    )
    elevation = replay.coarse_elevation(structure, cfg, SEED)
    windows = {
        "small": replay._window(
            structure, replay.PRIMARY_ORIGIN, replay.SMALL_HALO_KM),
        "large": replay._window(
            structure, replay.PRIMARY_ORIGIN, replay.LARGE_HALO_KM),
    }
    windows["shifted"] = replay._shift_window(
        windows["large"], structure,
        -replay.SHIFT_KM, replay.SHIFT_KM)
    windows = {name: tuple(int(value) for value in window)
               for name, window in windows.items()}
    if windows != EXPECTED_WINDOWS:
        raise AssertionError({"expected_windows": EXPECTED_WINDOWS,
                              "observed_windows": windows})
    n_world = int(round(structure.world_km / erosion_engine.E_KM))
    e_km = structure.world_km / n_world

    snapshots = {}
    wall_times = {}
    with LegacyInstrumentation() as instrumentation:
        for name in WINDOW_ORDER:
            observer = WindowObserver(name, windows[name], e_km)
            instrumentation.active = observer
            call_started = time.perf_counter()
            try:
                replay.run_erosion(
                    structure, elevation, cfg, SEED,
                    _process_window=windows[name],
                    _localization_mode="legacy",
                )
            finally:
                instrumentation.active = None
            wall_times[name] = time.perf_counter() - call_started
            snapshots[name] = observer.finalize()

    flux = {name: _replay_legacy_sediment(snapshot)
            for name, snapshot in snapshots.items()}
    relations = {
        relation_name: _relation_report(
            relation_name,
            snapshots[reference_name], snapshots[other_name],
            flux[reference_name], flux[other_name], structure)
        for relation_name, reference_name, other_name in RELATIONS
    }
    report = {
        "experiment": EXPERIMENT,
        "completed": True,
        "protocol_precommit_sha256": protocol_sha256,
        "source_fingerprint": fingerprint,
        "prior_stage_report": prior_link,
        "fixed": {
            "seed": SEED,
            "origin_xy_km": list(replay.PRIMARY_ORIGIN),
            "windows": {name: list(value)
                        for name, value in windows.items()},
            "mode": "legacy",
            "structural_builds": 1,
            "coarse_elevation_builds": 1,
            "erosion_calls": 3,
            "retries": 0,
        },
        "instrumentation": {
            "engine_behavior_modified": False,
            "wrappers_returned_original_values_unchanged": True,
            "wall_times_instrumented_s": wall_times,
            "sediment_flux_replay_validation": {
                name: value.validation for name, value in flux.items()
            },
        },
        "relations": relations,
        "interpretation_limits": [
            "Outside-overlap source mass is natural surrounding-terrain influence, not by itself an invalid border process.",
            "A numerical-rim terminal proves the legacy graph is finite-window conditioned; deposit attribution still distinguishes common and window-only sources.",
            "This one seed/origin/window trio does not establish general frequency.",
        ],
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
        "elapsed_s": report["elapsed_s"],
        "material_target_counts": {
            name: value["material_target_count"]
            for name, value in relations.items()
        },
    }


def _self_check() -> dict:
    z = np.array([[-100.0, -101.0, -102.0]], np.float64)
    ero = np.array([[100.0, 0.0, 0.0]], np.float64)
    receiver = np.array([1, 2, 2], np.int64)
    batches = (np.array([0]), np.array([1]), np.array([2]))
    area = np.ones(3, np.float64)
    result = erosion_engine.route_sediment(
        z, ero, receiver, batches, area,
        -80.0, 180.0, 20.0)
    snapshot = SedimentSnapshot(
        window_name="synthetic", window=(0, 0, 3), e_km=20.0,
        filled=z.copy(), surface=z.copy(), erosion=ero.copy(),
        receiver=receiver.copy(), batches=batches,
        area_km2=area.copy(), base_level_m=-80.0,
        deposition_length_km=180.0, edge_length_km=None,
        output_surface=result[0], deposit=result[1],
        boundary_export_m_cells=result[2],
        terminal_residual_m_cells=result[3],
    )
    flux = _replay_legacy_sediment(snapshot)
    path = _path_summary(snapshot, (0, 0), (0, 1, 0, 3))
    attribution = _upstream_attribution(
        snapshot, flux, (0, 1), (0, 1, 0, 3), _children(snapshot))
    checks = {
        "sediment_replay_exact": all(
            value for key, value in flux.validation.items()
            if key.endswith("_exact")),
        "terminal_is_numerical_ring": (
            path["terminal_type"]
            == "numerical_outer_ring_self_receiver"),
        "path_hops": path["hops"] == 2,
        "upstream_flux_reconstruction": (
            attribution["inbound_flux_m_cells"][
                "reconstruction_abs_error"] < 1e-12
            and attribution["deposit_attribution_m"][
                "reconstruction_abs_error"] < 1e-12),
        "coordinate_hash_stable": (
            _coordinate_hash([(1, 2), (0, 1)])
            == _coordinate_hash([(0, 1), (1, 2)])),
    }
    passed = all(checks.values())
    result = {
        "experiment": EXPERIMENT + "-self-check",
        "model_executed": False,
        "checks": checks,
        "passed": passed,
    }
    if not passed:
        raise AssertionError(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("out") / "process_halo_provenance_seed11_v1")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    result = _self_check() if args.self_check else _run(args.out)
    print(json.dumps(
        result, indent=2, allow_nan=False, default=_json_default))


if __name__ == "__main__":
    main()
