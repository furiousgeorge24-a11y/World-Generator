"""First-divergent-stage audit for the frozen physical-outlet replay.

This is a private, observation-only Run 1 diagnostic.  It repeats the
digest-anchored seed-11 small/large/shifted replay once, captures the four
lowstand routing stages and the terrestrial/marine sediment seam, then runs
precommitted marine-only cross-feeds.  It changes neither the engine nor the
public/default localization mode.

Run from ``pipeline_b`` with::

    python -B -m spikes.physical_outlet_causal_discriminator \
        --out ../out/physical_outlet_causal_seed11_v1

The non-model mechanics check is::

    python -B -m spikes.physical_outlet_causal_discriminator --self-check
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from engine import erosion as erosion_engine
from spikes import atlas_replay as replay
from spikes import physical_outlet_replay as physical_replay
from spikes import process_halo_diagnostic as stage_diagnostic


EXPERIMENT = "seed11-physical-outlet-causal-discriminator-v1"
SEED = 11
CONTINENTAL_BUDGET = 0.65
WINDOW_ORDER = ("small", "large", "shifted")
EXPECTED_WINDOWS = {
    "small": (79, 604, 365),
    "large": (39, 564, 445),
    "shifted": (19, 584, 445),
}
RELATIONS = (
    ("small_vs_large", "large", "small"),
    ("shifted_vs_large", "large", "shifted"),
    ("small_vs_shifted", "small", "shifted"),
)
ROUTING_STAGES = ("incision_0", "incision_1", "pre_sediment",
                  "post_sediment")
COUNTERFACTUAL_VARIANTS = (
    "fixed_source_native_bed",
    "native_source_fixed_common_bed",
    "fixed_source_fixed_common_bed",
)
TERRAIN_MATERIAL_THRESHOLD_M = 0.05
HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD = 0.005
NUMERIC_DIAGNOSTIC_THRESHOLD = 1e-9
MOUTH_NUMERIC_THRESHOLD_M_CELLS = 1e-9
MOUTH_MATERIAL_REPORTING_THRESHOLD_M_CELLS = 0.05
MAX_COORDINATE_EXAMPLES = 24
MAX_MOUTH_EXAMPLES = 24
EXPECTED_DIRECT_MARINE_CALLS = 9
EXPECTED_FROZEN_GRAPH_ABLATION_CALLS = 3
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PRIOR_ARTIFACTS = {
    "physical_outlet_replay": {
        "relative_path": "out/physical_outlet_seed11_v1/report.json",
        "sha256": (
            "a6fed11730e63c686f56a5f860e0a83647ad40523fa082da703f67e7005d92d2"),
    },
    "stage_diagnostic": {
        "relative_path": "out/process_halo_seed11_stage_v1/report.json",
        "sha256": (
            "d6dd696c14e7cc51a990d3a2f639b2dbd21f93c74426bf79ecb9bcdde5d46ae5"),
    },
    "legacy_provenance": {
        "relative_path": "out/process_halo_provenance_seed11_v1/report.json",
        "sha256": (
            "1b4c61fb93224db6a773b0fe26db47d6dcbb0d9605161f4c629c34cf75c01101"),
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
    "spikes/atlas_survey.py",
    "spikes/process_halo_diagnostic.py",
    "spikes/process_halo_provenance.py",
    "spikes/physical_outlet_replay.py",
    "spikes/physical_outlet_causal_discriminator.py",
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


def _array_summary(value) -> dict:
    array = np.asarray(value)
    result = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _array_sha256(array),
    }
    if array.dtype == np.dtype(bool):
        result.update({
            "true_count": int(np.count_nonzero(array)),
            "false_count": int(array.size - np.count_nonzero(array)),
        })
    elif np.issubdtype(array.dtype, np.number):
        finite = np.isfinite(array)
        result["finite_count"] = int(np.count_nonzero(finite))
        result["nonfinite_count"] = int(array.size - np.count_nonzero(finite))
        if finite.any():
            numeric = np.asarray(array[finite], np.float64)
            result.update({
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "sum": float(numeric.sum()),
                "nonzero_count": int(np.count_nonzero(numeric)),
            })
    return result


def _prior_links() -> dict:
    links = {}
    for name, fixed in PRIOR_ARTIFACTS.items():
        path = WORKSPACE / fixed["relative_path"]
        actual = _sha256_file(path) if path.is_file() else None
        links[name] = {
            "relative_path_from_workspace": fixed["relative_path"],
            "artifact_exists": path.is_file(),
            "expected_sha256": fixed["sha256"],
            "actual_sha256": actual,
            "digest_matched": actual == fixed["sha256"],
        }
    return links


def _window_intersection(windows) -> tuple[int, int, int, int]:
    row0 = max(window[0] for window in windows)
    column0 = max(window[1] for window in windows)
    row1 = min(window[0] + window[2] for window in windows)
    column1 = min(window[1] + window[2] for window in windows)
    if row0 >= row1 or column0 >= column1:
        raise ValueError("process windows have no common intersection")
    return row0, row1, column0, column1


def _rect_shape(rect) -> tuple[int, int]:
    return rect[1] - rect[0], rect[3] - rect[2]


def _extract_rect(value, window, rect) -> np.ndarray:
    array = np.asarray(value)
    row0, column0, side = window
    rr0, rr1, cc0, cc1 = rect
    if (rr0 < row0 or rr1 > row0 + side
            or cc0 < column0 or cc1 > column0 + side):
        raise ValueError(f"rectangle {rect} lies outside window {window}")
    return np.asarray(array[
        rr0 - row0:rr1 - row0,
        cc0 - column0:cc1 - column0,
    ]).copy()


def _embed_rect(value, rect, window, *, fill=0.0, dtype=None) -> np.ndarray:
    source = np.asarray(value)
    if source.shape != _rect_shape(rect):
        raise ValueError((source.shape, _rect_shape(rect)))
    row0, column0, side = window
    rr0, rr1, cc0, cc1 = rect
    if (rr0 < row0 or rr1 > row0 + side
            or cc0 < column0 or cc1 > column0 + side):
        raise ValueError(f"rectangle {rect} lies outside window {window}")
    result = np.full((side, side), fill,
                     dtype=source.dtype if dtype is None else dtype)
    result[
        rr0 - row0:rr1 - row0,
        cc0 - column0:cc1 - column0,
    ] = source
    return result


def _replace_rect(native, window, rect, replacement) -> np.ndarray:
    result = np.asarray(native).copy()
    row0, column0, _ = window
    rr0, rr1, cc0, cc1 = rect
    result[
        rr0 - row0:rr1 - row0,
        cc0 - column0:cc1 - column0,
    ] = replacement
    return result


def _global_sparse_summary(value, window) -> dict:
    array = np.asarray(value, np.float64)
    local_rows, local_columns = np.nonzero(array != 0.0)
    values = array[local_rows, local_columns]
    coordinates = np.column_stack((
        local_rows.astype(np.int64) + window[0],
        local_columns.astype(np.int64) + window[1],
    ))
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(coordinates).view(np.uint8).tobytes())
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(values).view(np.uint8).tobytes())
    if coordinates.size:
        bbox = {
            "global_row_range_inclusive": [
                int(coordinates[:, 0].min()), int(coordinates[:, 0].max())],
            "global_column_range_inclusive": [
                int(coordinates[:, 1].min()), int(coordinates[:, 1].max())],
        }
    else:
        bbox = None
    return {
        "global_sparse_coordinate_value_sha256": digest.hexdigest(),
        "nonzero_count": int(values.size),
        "sum_m_cells": float(values.sum()),
        "bounding_box": bbox,
    }


def _global_value(value, window, coordinate):
    row, column = coordinate
    row0, column0, side = window
    if not (row0 <= row < row0 + side
            and column0 <= column < column0 + side):
        raise ValueError(f"coordinate {coordinate} outside {window}")
    return np.asarray(value)[row - row0, column - column0]


def _core_coordinates(geometry, mask) -> list[tuple[int, int]]:
    return [
        (int(geometry.global_rows[row]),
         int(geometry.global_columns[column]))
        for row, column in np.argwhere(mask)
    ]


def _direction_core_global(targets, geometry) -> np.ndarray:
    targets = np.asarray(targets, np.int64)
    side = geometry.window[2]
    local_rows, local_columns = np.meshgrid(
        geometry.local_rows, geometry.local_columns, indexing="ij")
    source = (local_rows * side + local_columns).reshape(-1)
    selected = targets[:, source].T.reshape(
        geometry.core_shape + (8,))
    output = np.full(selected.shape + (2,), -1, np.int64)
    valid = selected >= 0
    target_rows = np.zeros_like(selected)
    target_columns = np.zeros_like(selected)
    target_rows[valid], target_columns[valid] = np.divmod(
        selected[valid], side)
    output[..., 0][valid] = target_rows[valid] + geometry.window[0]
    output[..., 1][valid] = target_columns[valid] + geometry.window[1]
    return output


def _direction_core_weights(weights, geometry) -> np.ndarray:
    weights = np.asarray(weights, np.float64)
    side = geometry.window[2]
    local_rows, local_columns = np.meshgrid(
        geometry.local_rows, geometry.local_columns, indexing="ij")
    source = (local_rows * side + local_columns).reshape(-1)
    return weights[:, source].T.reshape(geometry.core_shape + (8,)).copy()


@dataclass
class FillCapture:
    routing_surface_m: np.ndarray
    outlet_mask: np.ndarray
    filled_surface_m: np.ndarray


@dataclass
class FlowCapture:
    receiver: np.ndarray
    batches: tuple[np.ndarray, ...]
    raw_area: np.ndarray
    raw_area8: np.ndarray
    effective_area: np.ndarray
    effective_area8: np.ndarray
    runoff: np.ndarray
    target_core_global_row_column: np.ndarray
    weight_core: np.ndarray


@dataclass
class D8Capture:
    receiver: np.ndarray
    batches: tuple[np.ndarray, ...]
    raw_area8: np.ndarray
    effective_area8: np.ndarray
    runoff: np.ndarray


@dataclass
class SolveCapture:
    input_surface_m: np.ndarray
    base_mask_strict_less: np.ndarray
    output_surface_before_creep_m: np.ndarray
    cut_m: np.ndarray


@dataclass
class MarineCapture:
    pre_marine_bed_m: np.ndarray
    mouth_flux_m_cells: np.ndarray
    base_level_m: float
    deposition_length_km: float
    process_spacing_km: float
    marine_deposit_m: np.ndarray
    combined_export_m_cells: float
    terminal_residual_m_cells: float
    diagnostics: dict
    bed_unchanged_by_call: bool
    mouth_flux_unchanged_by_call: bool


@dataclass
class SedimentCapture:
    input_surface_m: np.ndarray
    erosion_source_m: np.ndarray
    receiver: np.ndarray
    batches: tuple[np.ndarray, ...]
    area_km2: np.ndarray
    base_level_m: float
    deposition_length_km: float
    process_spacing_km: float
    output_surface_m: np.ndarray
    total_deposit_m: np.ndarray
    combined_export_m_cells: float
    terminal_residual_m_cells: float
    diagnostics: dict


class WindowObserver:
    def __init__(self, name: str, window: tuple[int, int, int], geometry):
        self.name = name
        self.window = window
        self.geometry = geometry
        self.fills: list[FillCapture] = []
        self.flows: list[FlowCapture] = []
        self.d8s: list[D8Capture] = []
        self.solves: list[SolveCapture] = []
        self.marines: list[MarineCapture] = []
        self.sediments: list[SedimentCapture] = []

    def record_fill(self, h, outlet_mask, result) -> None:
        self.fills.append(FillCapture(
            np.asarray(h).copy(), np.asarray(outlet_mask, bool).copy(),
            np.asarray(result).copy()))

    def record_flow(self, rcv, batches, targets, weights, runoff,
                    result) -> None:
        weight_core = _direction_core_weights(weights, self.geometry)
        target_core = _direction_core_global(targets, self.geometry)
        target_core[weight_core <= 0.0] = -1
        raw_area = np.asarray(result[0]).copy()
        raw_area8 = np.asarray(result[1]).copy()
        effective_area = raw_area.copy()
        effective_area8 = raw_area8.copy()
        outlet = self.fills[-1].outlet_mask.ravel()
        effective_area[outlet] = 0.0
        effective_area8[outlet] = 0.0
        self.flows.append(FlowCapture(
            receiver=np.asarray(rcv, np.int64).copy(),
            batches=tuple(np.asarray(batch, np.int64).copy()
                          for batch in batches),
            raw_area=raw_area,
            raw_area8=raw_area8,
            effective_area=effective_area,
            effective_area8=effective_area8,
            runoff=np.asarray(runoff).copy(),
            target_core_global_row_column=target_core,
            weight_core=weight_core,
        ))

    def record_d8(self, rcv, batches, runoff, result) -> None:
        raw_area8 = np.asarray(result).copy()
        effective_area8 = raw_area8.copy()
        effective_area8[self.fills[-1].outlet_mask.ravel()] = 0.0
        self.d8s.append(D8Capture(
            receiver=np.asarray(rcv, np.int64).copy(),
            batches=tuple(np.asarray(batch, np.int64).copy()
                          for batch in batches),
            raw_area8=raw_area8,
            effective_area8=effective_area8,
            runoff=np.asarray(runoff).copy(),
        ))

    def record_solve(self, z, base, result) -> None:
        self.solves.append(SolveCapture(
            input_surface_m=np.asarray(z).copy(),
            base_mask_strict_less=np.asarray(base, bool).copy(),
            output_surface_before_creep_m=np.asarray(result[0]).copy(),
            cut_m=np.asarray(result[1]).copy(),
        ))

    def record_marine(self, z_before, mouth_before, base_level,
                      length_km, dx_km, result, bed_unchanged,
                      mouth_unchanged) -> None:
        self.marines.append(MarineCapture(
            pre_marine_bed_m=z_before,
            mouth_flux_m_cells=mouth_before,
            base_level_m=float(base_level),
            deposition_length_km=float(length_km),
            process_spacing_km=float(dx_km),
            marine_deposit_m=np.asarray(result[0]).copy(),
            combined_export_m_cells=float(result[1]),
            terminal_residual_m_cells=float(result[2]),
            diagnostics=json.loads(json.dumps(
                result[3], default=_json_default)),
            bed_unchanged_by_call=bed_unchanged,
            mouth_flux_unchanged_by_call=mouth_unchanged,
        ))

    def record_sediment(self, z, ero, rcv, batches, area, base_level,
                        length_km, dx_km, result) -> None:
        self.sediments.append(SedimentCapture(
            input_surface_m=np.asarray(z).copy(),
            erosion_source_m=np.asarray(ero).copy(),
            receiver=np.asarray(rcv, np.int64).copy(),
            batches=tuple(np.asarray(batch, np.int64).copy()
                          for batch in batches),
            area_km2=np.asarray(area).copy(),
            base_level_m=float(base_level),
            deposition_length_km=float(length_km),
            process_spacing_km=float(dx_km),
            output_surface_m=np.asarray(result[0]).copy(),
            total_deposit_m=np.asarray(result[1]).copy(),
            combined_export_m_cells=float(result[2]),
            terminal_residual_m_cells=float(result[3]),
            diagnostics=json.loads(json.dumps(
                result[4], default=_json_default)),
        ))

    def finalize(self) -> None:
        expected = {
            "fill": erosion_engine.N_STEPS + 2,
            "flow": erosion_engine.N_STEPS + 1,
            "d8": erosion_engine.N_STEPS + 2,
            "solve": erosion_engine.N_STEPS,
            "marine": 1,
            "sediment": 1,
        }
        observed = {
            "fill": len(self.fills), "flow": len(self.flows),
            "d8": len(self.d8s), "solve": len(self.solves),
            "marine": len(self.marines), "sediment": len(self.sediments),
        }
        if observed != expected:
            raise AssertionError({"window": self.name,
                                  "expected": expected,
                                  "observed": observed})


class PhysicalInstrumentation(AbstractContextManager):
    """Narrow pass-through wrappers, restored even after a failed solve."""

    def __init__(self):
        self.active: WindowObserver | None = None
        self.originals: dict[str, Callable] = {}

    def __enter__(self):
        names = (
            "_fill_to_lowstand_outlets", "flow_accumulation",
            "flow_accumulation_d8", "spl_implicit",
            "_route_sediment_lowstand", "_physical_marine_transport",
        )
        self.originals = {name: getattr(erosion_engine, name)
                          for name in names}

        def fill(h, outlet_mask, *args, **kwargs):
            result = self.originals["_fill_to_lowstand_outlets"](
                h, outlet_mask, *args, **kwargs)
            if self.active is not None:
                self.active.record_fill(h, outlet_mask, result)
            return result

        def flow(rcv, batches, n, targets, weights, runoff=None):
            result = self.originals["flow_accumulation"](
                rcv, batches, n, targets, weights, runoff)
            if self.active is not None:
                self.active.record_flow(
                    rcv, batches, targets, weights, runoff, result)
            return result

        def d8(rcv, batches, n, runoff=None):
            result = self.originals["flow_accumulation_d8"](
                rcv, batches, n, runoff)
            if self.active is not None:
                self.active.record_d8(rcv, batches, runoff, result)
            return result

        def solve(z, U, Kf, rcv, batches, area, dt_myr, dx_km, base,
                  edge_len_km=None):
            result = self.originals["spl_implicit"](
                z, U, Kf, rcv, batches, area, dt_myr, dx_km, base,
                edge_len_km)
            if self.active is not None:
                self.active.record_solve(z, base, result)
            return result

        def marine(z, mouth_flux, base_level, length_km, dx_km):
            z_before = np.asarray(z).copy()
            mouth_before = np.asarray(mouth_flux).copy()
            result = self.originals["_physical_marine_transport"](
                z, mouth_flux, base_level, length_km, dx_km)
            if self.active is not None:
                self.active.record_marine(
                    z_before, mouth_before, base_level, length_km, dx_km,
                    result, np.array_equal(z, z_before),
                    np.array_equal(mouth_flux, mouth_before))
            return result

        def sediment(z, ero, rcv, batches, area, base_level,
                     length_km, dx_km, *, _marine_transport=None):
            result = self.originals["_route_sediment_lowstand"](
                z, ero, rcv, batches, area, base_level, length_km, dx_km,
                _marine_transport=_marine_transport)
            if self.active is not None:
                self.active.record_sediment(
                    z, ero, rcv, batches, area, base_level,
                    length_km, dx_km, result)
            return result

        erosion_engine._fill_to_lowstand_outlets = fill
        erosion_engine.flow_accumulation = flow
        erosion_engine.flow_accumulation_d8 = d8
        erosion_engine.spl_implicit = solve
        erosion_engine._route_sediment_lowstand = sediment
        erosion_engine._physical_marine_transport = marine
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for name, original in self.originals.items():
            setattr(erosion_engine, name, original)
        self.active = None
        return False


@dataclass
class LandReplay:
    source_m: np.ndarray
    land_deposit_m: np.ndarray
    mouth_flux_m_cells: np.ndarray
    mouth_from_common_source_m_cells: np.ndarray
    mouth_from_window_only_source_m_cells: np.ndarray
    land_deposit_from_common_source_m: np.ndarray
    land_deposit_from_window_only_source_m: np.ndarray
    boundary_export_m_cells: float
    terminal_residual_m_cells: float
    validation: dict


def _common_source_mask(window, rect) -> np.ndarray:
    row0, column0, side = window
    rows = row0 + np.arange(side)[:, None]
    columns = column0 + np.arange(side)[None, :]
    return ((rows >= rect[0]) & (rows < rect[1])
            & (columns >= rect[2]) & (columns < rect[3]))


def _replay_land_sediment(snapshot: SedimentCapture, window,
                          common_rect, marine: MarineCapture) -> LandReplay:
    """Reproduce the terrestrial handoff and split provenance by support."""
    z = snapshot.input_surface_m
    zf = z.ravel()
    receiver = snapshot.receiver
    is_marine = zf <= snapshot.base_level_m
    source = np.maximum(snapshot.erosion_source_m, 0.0).ravel()
    common = _common_source_mask(window, common_rect).ravel()
    flux = np.where(is_marine, 0.0, source)
    flux_common = np.where(is_marine | ~common, 0.0, source)
    flux_window_only = np.where(is_marine | common, 0.0, source)
    deposit = np.zeros(source.shape, np.float64)
    deposit_common = np.zeros(source.shape, np.float64)
    deposit_window_only = np.zeros(source.shape, np.float64)
    mouth = np.zeros(source.shape, np.float64)
    mouth_common = np.zeros(source.shape, np.float64)
    mouth_window_only = np.zeros(source.shape, np.float64)
    border = np.zeros(z.shape, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    border = border.ravel()
    capacity = (erosion_engine.KC_LAND
                * np.sqrt(np.maximum(snapshot.area_km2, 1.0)))
    boundary_export = 0.0
    terminal_residual = float(source[is_marine].sum())

    for batch in snapshot.batches:
        land_batch = batch[~is_marine[batch]]
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
        survival = np.divide(
            remaining, flux[land_batch],
            out=np.zeros_like(remaining), where=flux[land_batch] > 0.0)
        common_remaining = flux_common[land_batch] * survival
        window_only_remaining = flux_window_only[land_batch] * survival
        deposit_common[land_batch] += (
            flux_common[land_batch] - common_remaining)
        deposit_window_only[land_batch] += (
            flux_window_only[land_batch] - window_only_remaining)

        to_marine = movable & is_marine[target]
        if to_marine.any():
            np.add.at(mouth, target[to_marine], remaining[to_marine])
            np.add.at(mouth_common, target[to_marine],
                      common_remaining[to_marine])
            np.add.at(mouth_window_only, target[to_marine],
                      window_only_remaining[to_marine])

        to_land = movable & ~is_marine[target]
        if to_land.any():
            np.add.at(flux, target[to_land], remaining[to_land])
            np.add.at(flux_common, target[to_land],
                      common_remaining[to_land])
            np.add.at(flux_window_only, target[to_land],
                      window_only_remaining[to_land])

        terminal = ~movable
        if terminal.any():
            terminal_cells = land_batch[terminal]
            terminal_flux = remaining[terminal]
            outer = border[terminal_cells]
            boundary_export += float(terminal_flux[outer].sum())
            terminal_residual += float(terminal_flux[~outer].sum())

    land_deposit = deposit.reshape(z.shape)
    mouth_grid = mouth.reshape(z.shape)
    route_deposit_reconstructed = land_deposit + marine.marine_deposit_m
    common_reconstruction_error = np.max(np.abs(
        mouth - mouth_common - mouth_window_only), initial=0.0)
    deposit_reconstruction_error = np.max(np.abs(
        deposit - deposit_common - deposit_window_only), initial=0.0)
    validation = {
        "mouth_flux_array_exact": bool(np.array_equal(
            mouth_grid, marine.mouth_flux_m_cells)),
        "pre_marine_bed_array_exact": bool(np.array_equal(
            z + land_deposit, marine.pre_marine_bed_m)),
        "land_deposit_total_exact": bool(
            float(deposit.sum())
            == float(snapshot.diagnostics["land_deposited_m_cells"])),
        "total_deposit_array_exact": bool(np.array_equal(
            route_deposit_reconstructed, snapshot.total_deposit_m)),
        "output_surface_array_exact": bool(np.array_equal(
            z + snapshot.total_deposit_m, snapshot.output_surface_m)),
        "provenance_mouth_reconstruction_max_abs_m_cells": float(
            common_reconstruction_error),
        "provenance_mouth_reconstruction_within_numeric_threshold": bool(
            common_reconstruction_error <= NUMERIC_DIAGNOSTIC_THRESHOLD),
        "provenance_land_deposit_reconstruction_max_abs_m": float(
            deposit_reconstruction_error),
        "provenance_land_deposit_reconstruction_within_numeric_threshold": (
            bool(deposit_reconstruction_error
                 <= NUMERIC_DIAGNOSTIC_THRESHOLD)),
    }
    if not all(value for key, value in validation.items()
               if key.endswith("_exact")):
        raise AssertionError({"land_replay_validation": validation})
    if not validation[
            "provenance_mouth_reconstruction_within_numeric_threshold"]:
        raise AssertionError({"land_provenance_validation": validation})
    if not validation[
            "provenance_land_deposit_reconstruction_within_numeric_threshold"]:
        raise AssertionError({"land_provenance_validation": validation})
    return LandReplay(
        source_m=source.reshape(z.shape),
        land_deposit_m=land_deposit,
        mouth_flux_m_cells=mouth_grid,
        mouth_from_common_source_m_cells=mouth_common.reshape(z.shape),
        mouth_from_window_only_source_m_cells=(
            mouth_window_only.reshape(z.shape)),
        land_deposit_from_common_source_m=deposit_common.reshape(z.shape),
        land_deposit_from_window_only_source_m=(
            deposit_window_only.reshape(z.shape)),
        boundary_export_m_cells=boundary_export,
        terminal_residual_m_cells=terminal_residual,
        validation=validation,
    )


def _fill_replay(capture: FillCapture, max_rounds=8) -> dict:
    """Repeat the fill solely to expose whether its fixed round cap fired."""
    h = capture.routing_surface_m
    outlets = capture.outlet_mask
    rows, columns = h.shape
    filled = np.full_like(h, np.inf)
    filled[0, :] = h[0, :]
    filled[-1, :] = h[-1, :]
    filled[:, 0] = h[:, 0]
    filled[:, -1] = h[:, -1]
    filled[outlets] = h[outlets]

    def relax_from(previous, current_height, current_filled):
        candidate = np.minimum(
            previous,
            np.minimum(np.r_[np.inf, previous[:-1]],
                       np.r_[previous[1:], np.inf])) + erosion_engine.EPS
        return np.maximum(current_height,
                          np.minimum(current_filled, candidate))

    converged = False
    rounds = 0
    for index in range(max_rounds):
        old = filled.copy()
        for row in range(1, rows - 1):
            filled[row] = relax_from(
                filled[row - 1], h[row], filled[row])
            filled[row, outlets[row]] = h[row, outlets[row]]
        for row in range(rows - 2, 0, -1):
            filled[row] = relax_from(
                filled[row + 1], h[row], filled[row])
            filled[row, outlets[row]] = h[row, outlets[row]]
        for column in range(1, columns - 1):
            filled[:, column] = relax_from(
                filled[:, column - 1], h[:, column], filled[:, column])
            filled[outlets[:, column], column] = h[
                outlets[:, column], column]
        for column in range(columns - 2, 0, -1):
            filled[:, column] = relax_from(
                filled[:, column + 1], h[:, column], filled[:, column])
            filled[outlets[:, column], column] = h[
                outlets[:, column], column]
        rounds = index + 1
        if np.array_equal(filled, old):
            converged = True
            break
    return {
        "maximum_rounds": int(max_rounds),
        "rounds_executed": rounds,
        "converged_before_or_at_limit": converged,
        "fixed_round_limit_reached": bool(rounds == max_rounds),
        "output_array_exact": bool(np.array_equal(
            filled, capture.filled_surface_m)),
    }


def _element_equal(left, right) -> np.ndarray:
    left = np.asarray(left)
    right = np.asarray(right)
    equal = left == right
    if (np.issubdtype(left.dtype, np.floating)
            and np.issubdtype(right.dtype, np.floating)):
        equal = equal | (np.isnan(left) & np.isnan(right))
    return equal


def _cell_mask(component_mask) -> np.ndarray:
    mask = np.asarray(component_mask, bool)
    if mask.ndim > 2:
        mask = np.any(mask, axis=tuple(range(2, mask.ndim)))
    return mask


def _numeric_comparison(left, right, *, absolute_threshold=None,
                        relative_threshold=None) -> dict:
    left = np.asarray(left)
    right = np.asarray(right)
    exact = _cell_mask(~_element_equal(left, right))
    difference_components = np.abs(
        np.asarray(right, np.float64) - np.asarray(left, np.float64))
    difference = (np.max(difference_components,
                         axis=tuple(range(2, difference_components.ndim)))
                  if difference_components.ndim > 2
                  else difference_components)
    result = {
        "array_exact": not bool(exact.any()),
        "exact_changed_cells": int(np.count_nonzero(exact)),
        "greater_than_1e_minus_9_cells": int(np.count_nonzero(
            difference > NUMERIC_DIAGNOSTIC_THRESHOLD)),
        "max_abs": float(difference.max(initial=0.0)),
        "p99_abs": float(np.percentile(difference, 99.0)),
    }
    if absolute_threshold is not None:
        material = difference > absolute_threshold
        result.update({
            "material_metric": "absolute_difference",
            "material_threshold": float(absolute_threshold),
            "material_changed_cells": int(np.count_nonzero(material)),
            "materially_equal": not bool(material.any()),
        })
    if relative_threshold is not None:
        left_numeric = np.asarray(left, np.float64)
        right_numeric = np.asarray(right, np.float64)
        relative_components = np.abs(right_numeric - left_numeric) / np.maximum(
            np.maximum(np.abs(left_numeric), np.abs(right_numeric)),
            np.finfo(np.float64).tiny)
        relative = (np.max(relative_components,
                           axis=tuple(range(2, relative_components.ndim)))
                    if relative_components.ndim > 2
                    else relative_components)
        material = relative > relative_threshold
        result.update({
            "material_metric": "relative_difference",
            "material_threshold": float(relative_threshold),
            "material_changed_cells": int(np.count_nonzero(material)),
            "materially_equal": not bool(material.any()),
            "max_relative": float(relative.max(initial=0.0)),
        })
    return result


def _mask_comparison(left, right) -> dict:
    changed = _cell_mask(np.asarray(left) != np.asarray(right))
    return {
        "array_exact": not bool(changed.any()),
        "changed_cells": int(np.count_nonzero(changed)),
        "topological_metric": "exact_inequality",
        "topological_changed_cells": int(np.count_nonzero(changed)),
    }


def _discharge_comparison(left_log, right_log) -> tuple[dict, np.ndarray]:
    left = np.expm1(np.asarray(left_log, np.float64))
    right = np.expm1(np.asarray(right_log, np.float64))
    relative = np.abs(right - left) / np.maximum(
        np.maximum(np.abs(left), np.abs(right)),
        np.finfo(np.float64).tiny)
    material = relative > HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD
    absolute = np.abs(right - left)
    drawn_relevant = material & ((left > 30.0) | (right > 30.0))
    delta_log = np.abs(np.asarray(right_log) - np.asarray(left_log))
    return ({
        "array_exact": bool(np.array_equal(left_log, right_log)),
        "exact_changed_cells": int(np.count_nonzero(left_log != right_log)),
        "material_metric": "linearized_relative_difference",
        "material_threshold": HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD,
        "material_changed_cells": int(np.count_nonzero(material)),
        "materially_equal": not bool(material.any()),
        "max_log_space_abs": float(delta_log.max(initial=0.0)),
        "max_linearized_relative": float(relative.max(initial=0.0)),
        "max_linearized_absolute": float(absolute.max(initial=0.0)),
        "material_cells_with_either_side_above_draw_threshold_30": int(
            np.count_nonzero(drawn_relevant)),
        "visibility_note": (
            "The frozen relative threshold has no absolute floor; cells below "
            "the engine's A8 > 30 drawn-channel threshold are not thereby "
            "established as visible or geomorphically important."),
    }, material)


def _receiver_core(capture, geometry) -> np.ndarray:
    return geometry.receiver_global_row_column(capture.receiver)


def _raw_stage_surface(observer: WindowObserver, stage: str) -> np.ndarray:
    if stage == "incision_0":
        return observer.solves[0].input_surface_m
    if stage == "incision_1":
        return observer.solves[1].input_surface_m
    if stage == "pre_sediment":
        return observer.sediments[0].input_surface_m
    if stage == "post_sediment":
        return observer.sediments[0].output_surface_m
    raise KeyError(stage)


def _stage_comparison(reference: WindowObserver, other: WindowObserver,
                      stage: str, reference_geometry=None,
                      other_geometry=None, *, include_direction_fields=True) \
        -> dict:
    index = ROUTING_STAGES.index(stage)
    ref_geometry = (reference.geometry if reference_geometry is None
                    else reference_geometry)
    other_geometry = (other.geometry if other_geometry is None
                      else other_geometry)
    ref_fill = reference.fills[index]
    other_fill = other.fills[index]
    raw_ref = ref_geometry.extract_grid(_raw_stage_surface(reference, stage))
    raw_other = other_geometry.extract_grid(_raw_stage_surface(other, stage))
    outlet_ref = ref_geometry.extract_grid(ref_fill.outlet_mask)
    outlet_other = other_geometry.extract_grid(other_fill.outlet_mask)
    routing_ref = ref_geometry.extract_grid(ref_fill.routing_surface_m)
    routing_other = other_geometry.extract_grid(other_fill.routing_surface_m)
    filled_ref = ref_geometry.extract_grid(ref_fill.filled_surface_m)
    filled_other = other_geometry.extract_grid(other_fill.filled_surface_m)
    d8_ref = reference.d8s[index]
    d8_other = other.d8s[index]
    receiver_ref = _receiver_core(d8_ref, ref_geometry)
    receiver_other = _receiver_core(d8_other, other_geometry)
    raw_accumulation_ref = ref_geometry.extract_grid(d8_ref.raw_area8)
    raw_accumulation_other = other_geometry.extract_grid(d8_other.raw_area8)
    accumulation_ref = ref_geometry.extract_grid(d8_ref.effective_area8)
    accumulation_other = other_geometry.extract_grid(
        d8_other.effective_area8)
    report = {
        "raw_surface_m": _numeric_comparison(
            raw_ref, raw_other,
            absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
        "lowstand_outlet_mask_surface_le_base": _mask_comparison(
            outlet_ref, outlet_other),
        "clamped_routing_surface_m": _numeric_comparison(
            routing_ref, routing_other,
            absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
        "filled_surface_m": _numeric_comparison(
            filled_ref, filled_other,
            absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
        "d8_receiver_global_row_column": _mask_comparison(
            receiver_ref, receiver_other),
        "runoff_input": _numeric_comparison(
            ref_geometry.extract_grid(d8_ref.runoff),
            other_geometry.extract_grid(d8_other.runoff),
            relative_threshold=HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD),
        "raw_d8_accumulation_before_intended_marine_zeroing": (
            _numeric_comparison(
                raw_accumulation_ref, raw_accumulation_other,
                relative_threshold=HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD)),
        "effective_d8_accumulation": _numeric_comparison(
            accumulation_ref, accumulation_other,
            relative_threshold=HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD),
    }
    if stage != "post_sediment":
        ref_flow = reference.flows[index]
        other_flow = other.flows[index]
        report.update({
            "raw_mfd_accumulation_before_intended_marine_zeroing": (
                _numeric_comparison(
                    ref_geometry.extract_grid(ref_flow.raw_area),
                    other_geometry.extract_grid(other_flow.raw_area),
                    relative_threshold=(
                        HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD))),
            "effective_mfd_accumulation": _numeric_comparison(
                ref_geometry.extract_grid(ref_flow.effective_area),
                other_geometry.extract_grid(other_flow.effective_area),
                relative_threshold=HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD),
        })
        if include_direction_fields:
            report.update({
                "mfd_target_global_row_column": _mask_comparison(
                    ref_flow.target_core_global_row_column,
                    other_flow.target_core_global_row_column),
                "mfd_weight": _numeric_comparison(
                    ref_flow.weight_core, other_flow.weight_core,
                    absolute_threshold=NUMERIC_DIAGNOSTIC_THRESHOLD),
            })
        if stage in ("incision_0", "incision_1"):
            solve = reference.solves[index]
            other_solve = other.solves[index]
            strict_ref = ref_geometry.extract_grid(
                solve.base_mask_strict_less)
            strict_other = other_geometry.extract_grid(
                other_solve.base_mask_strict_less)
            output_ref = ref_geometry.extract_grid(
                solve.output_surface_before_creep_m)
            output_other = other_geometry.extract_grid(
                other_solve.output_surface_before_creep_m)
            if stage == "incision_0":
                next_ref = ref_geometry.extract_grid(
                    reference.solves[1].input_surface_m)
                next_other = other_geometry.extract_grid(
                    other.solves[1].input_surface_m)
            else:
                next_ref = ref_geometry.extract_grid(
                    reference.sediments[0].input_surface_m)
                next_other = other_geometry.extract_grid(
                    other.sediments[0].input_surface_m)
            report.update({
                "solve_base_mask_surface_strictly_less_than_lowstand": (
                    _mask_comparison(strict_ref, strict_other)),
                "stream_power_cut_m": _numeric_comparison(
                    ref_geometry.extract_grid(solve.cut_m),
                    other_geometry.extract_grid(other_solve.cut_m),
                    absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
                "solve_output_before_creep_m": _numeric_comparison(
                    output_ref, output_other,
                    absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
                "creep_delta_m": _numeric_comparison(
                    next_ref - output_ref, next_other - output_other,
                    absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
            })
    return report


@dataclass
class MarineOutcome:
    bed_m: np.ndarray
    requested_source_m_cells: np.ndarray
    effective_source_m_cells: np.ndarray
    deposit_m: np.ndarray
    combined_export_m_cells: float
    terminal_residual_m_cells: float
    diagnostics: dict


def _run_direct_marine(bed, source, base_level, length_km,
                       spacing_km) -> MarineOutcome:
    bed = np.asarray(bed, np.float64).copy()
    source = np.asarray(source, np.float64).copy()
    result = erosion_engine._physical_marine_transport(
        bed, source, base_level, length_km, spacing_km)
    effective = np.where(
        bed <= base_level, np.maximum(source, 0.0), 0.0)
    return MarineOutcome(
        bed_m=bed,
        requested_source_m_cells=source,
        effective_source_m_cells=effective,
        deposit_m=np.asarray(result[0]).copy(),
        combined_export_m_cells=float(result[1]),
        terminal_residual_m_cells=float(result[2]),
        diagnostics=json.loads(json.dumps(
            result[3], default=_json_default)),
    )


def _run_frozen_graph_marine(bed, source, base_level, length_km,
                             spacing_km) -> MarineOutcome:
    """Private ablation: keep the initial marine graph for every step."""
    bed = np.asarray(bed, np.float64).copy()
    source = np.asarray(source, np.float64).copy()
    marine = bed <= base_level
    effective = np.where(marine, np.maximum(source, 0.0), 0.0)
    max_steps = max(1, int(np.ceil(
        erosion_engine.PHYSICAL_MARINE_EFOLDS
        * float(length_km) / spacing_km)))
    settle = 1.0 - np.exp(-spacing_km / float(length_km))
    targets, weights, has_out = erosion_engine._marine_transport_graph(
        bed, marine)
    deposit = np.zeros_like(bed)
    mobile = effective.copy()
    boundary_export = 0.0
    far_field_export = 0.0
    terminal_residual = 0.0
    accommodation_limited_events = 0
    border = np.zeros(bed.shape, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True

    for step in range(max_steps):
        if not np.any(mobile > 0.0):
            break
        at_boundary = border & (mobile > 0.0)
        boundary_export += float(mobile[at_boundary].sum())
        mobile[at_boundary] = 0.0
        if not np.any(mobile > 0.0):
            break
        next_mobile = np.zeros(bed.size, np.float64)
        mobile_flat = mobile.ravel()
        for direction in range(8):
            weight = weights[direction]
            moving = weight > 0.0
            if moving.any():
                np.add.at(next_mobile, targets[direction, moving],
                          mobile_flat[moving] * weight[moving])
        terminal = ~has_out
        next_mobile[terminal] += mobile_flat[terminal]
        arrived = next_mobile.reshape(bed.shape)
        demand = arrived * settle
        room = np.maximum(base_level - (bed + deposit), 0.0)
        accommodation_limited_events += int(np.count_nonzero(demand > room))
        settled = np.minimum(demand, room)
        deposit += settled
        mobile = arrived - settled
        if step + 1 == max_steps:
            terminal_mask = (~has_out.reshape(bed.shape)) & (mobile > 0.0)
            terminal_residual += float(mobile[terminal_mask].sum())
            mobile[terminal_mask] = 0.0
            far_field_export += float(mobile.sum())
            break

    source_total = float(effective.sum())
    deposited_total = float(deposit.sum())
    diagnostics = {
        "source_m_cells": source_total,
        "deposited_m_cells": deposited_total,
        "boundary_export_m_cells": boundary_export,
        "far_field_export_m_cells": far_field_export,
        "terminal_residual_m_cells": terminal_residual,
        "closure_m_cells": source_total - (
            deposited_total + boundary_export + far_field_export
            + terminal_residual),
        "max_steps": max_steps,
        "accommodation_limited_cell_events": accommodation_limited_events,
        "dynamic_aggradational_routing": False,
        "ablation": "initial_marine_graph_frozen_for_all_steps",
    }
    return MarineOutcome(
        bed_m=bed,
        requested_source_m_cells=source,
        effective_source_m_cells=effective,
        deposit_m=deposit,
        combined_export_m_cells=boundary_export + far_field_export,
        terminal_residual_m_cells=terminal_residual,
        diagnostics=diagnostics,
    )


def _marine_outcome_summary(outcome: MarineOutcome, window) -> dict:
    return {
        "bed": _array_summary(outcome.bed_m),
        "requested_source": {
            **_array_summary(outcome.requested_source_m_cells),
            **_global_sparse_summary(
                outcome.requested_source_m_cells, window),
        },
        "effective_source_after_bed_le_lowstand_mask": {
            **_array_summary(outcome.effective_source_m_cells),
            **_global_sparse_summary(
                outcome.effective_source_m_cells, window),
        },
        "deposit": _array_summary(outcome.deposit_m),
        "final_bed": _array_summary(outcome.bed_m + outcome.deposit_m),
        "combined_export_m_cells": outcome.combined_export_m_cells,
        "terminal_residual_m_cells": outcome.terminal_residual_m_cells,
        "diagnostics": outcome.diagnostics,
    }


def _marine_outcome_validation(outcome: MarineOutcome) -> dict:
    diagnostics = outcome.diagnostics
    source = float(outcome.effective_source_m_cells.sum())
    deposit = float(outcome.deposit_m.sum())
    closure = float(diagnostics["closure_m_cells"])
    boundary = float(diagnostics["boundary_export_m_cells"])
    far_field = float(diagnostics["far_field_export_m_cells"])
    residual = float(diagnostics["terminal_residual_m_cells"])
    tolerance = 1e-12 * max(source, 1.0)
    checks = {
        "arrays_finite": bool(
            np.isfinite(outcome.bed_m).all()
            and np.isfinite(outcome.requested_source_m_cells).all()
            and np.isfinite(outcome.effective_source_m_cells).all()
            and np.isfinite(outcome.deposit_m).all()),
        "effective_source_matches_diagnostics": bool(
            source == float(diagnostics["source_m_cells"])),
        "deposit_matches_diagnostics": bool(
            deposit == float(diagnostics["deposited_m_cells"])),
        "combined_export_matches_components": bool(
            outcome.combined_export_m_cells == boundary + far_field),
        "terminal_residual_matches_diagnostics": bool(
            outcome.terminal_residual_m_cells == residual),
        "mass_closure_within_scaled_1e_minus_12": bool(
            abs(closure) <= tolerance),
    }
    return {
        "source_m_cells": source,
        "scaled_closure_tolerance_m_cells": tolerance,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _marine_relation(reference: MarineOutcome, other: MarineOutcome,
                     reference_geometry, other_geometry) -> dict:
    bed_ref = reference_geometry.extract_grid(reference.bed_m)
    bed_other = other_geometry.extract_grid(other.bed_m)
    source_ref = reference_geometry.extract_grid(
        reference.effective_source_m_cells)
    source_other = other_geometry.extract_grid(
        other.effective_source_m_cells)
    dep_ref = reference_geometry.extract_grid(reference.deposit_m)
    dep_other = other_geometry.extract_grid(other.deposit_m)
    final_ref = bed_ref + dep_ref
    final_other = bed_other + dep_other
    return {
        "bed_m": _numeric_comparison(
            bed_ref, bed_other,
            absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
        "effective_source_m_cells": _numeric_comparison(
            source_ref, source_other,
            absolute_threshold=MOUTH_MATERIAL_REPORTING_THRESHOLD_M_CELLS),
        "marine_deposit_m": _numeric_comparison(
            dep_ref, dep_other,
            absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
        "final_marine_bed_m": _numeric_comparison(
            final_ref, final_other,
            absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
    }


def _effect_field_summary(effect, actual_delta) -> dict:
    effect = np.asarray(effect, np.float64)
    actual_delta = np.asarray(actual_delta, np.float64)
    effect_material = np.abs(effect) > TERRAIN_MATERIAL_THRESHOLD_M
    actual_material = np.abs(actual_delta) > TERRAIN_MATERIAL_THRESHOLD_M
    intersection = effect_material & actual_material
    union = effect_material | actual_material
    same_sign = intersection & (np.sign(effect) == np.sign(actual_delta))
    return {
        **_numeric_comparison(
            np.zeros_like(effect), effect,
            absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
        "actual_material_cells": int(np.count_nonzero(actual_material)),
        "material_overlap_with_actual_cells": int(np.count_nonzero(
            intersection)),
        "material_mask_jaccard_with_actual": (
            1.0 if not union.any() else float(
                np.count_nonzero(intersection) / np.count_nonzero(union))),
        "same_sign_material_overlap_cells": int(np.count_nonzero(same_sign)),
        "max_abs_effect_on_actual_material_cells": (
            None if not actual_material.any() else float(
                np.abs(effect[actual_material]).max(initial=0.0))),
        "l1_effect_on_actual_material_cells": float(
            np.abs(effect[actual_material]).sum()),
    }


def _factorial_effect_report(actual, counterfactuals, frozen,
                             reference_name, other_name,
                             reference_geometry, other_geometry) -> dict:
    def relation_delta(outcomes):
        reference = reference_geometry.extract_grid(
            outcomes[reference_name].deposit_m)
        other = other_geometry.extract_grid(outcomes[other_name].deposit_m)
        return other - reference

    actual_delta = relation_delta(actual)
    arm_delta = {
        name: relation_delta(outcomes)
        for name, outcomes in counterfactuals.items()
    }
    frozen_delta = relation_delta(frozen)
    domain = arm_delta["fixed_source_fixed_common_bed"]
    bed = arm_delta["fixed_source_native_bed"] - domain
    source = arm_delta["native_source_fixed_common_bed"] - domain
    interaction = (actual_delta
                   - arm_delta["fixed_source_native_bed"]
                   - arm_delta["native_source_fixed_common_bed"]
                   + domain)
    reconstruction = domain + bed + source + interaction
    error = np.abs(reconstruction - actual_delta)
    reconstruction_tolerance = 1e-12 * max(
        float(np.abs(actual_delta).max(initial=0.0)), 1.0)
    actual_material = np.abs(actual_delta) > TERRAIN_MATERIAL_THRESHOLD_M
    arm_reports = {}
    for name, delta in arm_delta.items():
        material = np.abs(delta) > TERRAIN_MATERIAL_THRESHOLD_M
        overlap = material & actual_material
        arm_reports[name] = {
            **_effect_field_summary(delta, actual_delta),
            "interpretation": (
                "conditional relation divergence under this arm; persistence "
                "does not alone identify a unique cause"),
            "material_overlap_coordinates_are_required_for_pattern_claim": (
                int(np.count_nonzero(overlap))),
        }
    effects = {
        "domain_or_outside_common_support_baseline": _effect_field_summary(
            domain, actual_delta),
        "native_common_bed_contrast": _effect_field_summary(
            bed, actual_delta),
        "native_source_contrast": _effect_field_summary(
            source, actual_delta),
        "source_bed_interaction_contrast": _effect_field_summary(
            interaction, actual_delta),
    }
    frozen_report = _effect_field_summary(frozen_delta, actual_delta)
    return {
        "algebra": (
            "actual relation delta = fixed-both domain baseline + native-bed "
            "contrast + native-source contrast + source-bed interaction"),
        "actual_relation_delta": _effect_field_summary(
            actual_delta, actual_delta),
        "conditional_arm_relation_deltas": arm_reports,
        "signed_factorial_effects": effects,
        "signed_reconstruction": {
            "max_abs_error_m": float(error.max(initial=0.0)),
            "scaled_tolerance_m": reconstruction_tolerance,
            "array_exact": bool(np.array_equal(reconstruction, actual_delta)),
            "within_scaled_1e_minus_12": bool(
                error.max(initial=0.0) <= reconstruction_tolerance),
        },
        "frozen_initial_graph_native_input_relation_delta": frozen_report,
        "frozen_graph_material_pattern_overlap_with_actual": int(
            frozen_report["material_overlap_with_actual_cells"]),
    }


def _actual_marine_outcome(capture: MarineCapture) -> MarineOutcome:
    effective = np.where(
        capture.pre_marine_bed_m <= capture.base_level_m,
        np.maximum(capture.mouth_flux_m_cells, 0.0), 0.0)
    return MarineOutcome(
        bed_m=capture.pre_marine_bed_m,
        requested_source_m_cells=capture.mouth_flux_m_cells,
        effective_source_m_cells=effective,
        deposit_m=capture.marine_deposit_m,
        combined_export_m_cells=capture.combined_export_m_cells,
        terminal_residual_m_cells=capture.terminal_residual_m_cells,
        diagnostics=capture.diagnostics,
    )


def _material_mask(left, right, threshold=TERRAIN_MATERIAL_THRESHOLD_M):
    return np.abs(np.asarray(right, np.float64)
                  - np.asarray(left, np.float64)) > threshold


def _relation_component_report(reference_name, other_name, solved,
                               observers, land_replays, geometries,
                               common_geometries) -> dict:
    reference_geometry = geometries[reference_name]
    other_geometry = geometries[other_name]
    reference_marine = observers[reference_name].marines[0]
    other_marine = observers[other_name].marines[0]
    reference_land = land_replays[reference_name]
    other_land = land_replays[other_name]
    reference_result = solved[reference_name]
    other_result = solved[other_name]
    common_reference_geometry = common_geometries[reference_name]
    common_other_geometry = common_geometries[other_name]

    def seam(ref_geometry, oth_geometry):
        return {
            "pre_sediment_surface_m": _numeric_comparison(
                ref_geometry.extract_grid(
                    observers[reference_name].sediments[0].input_surface_m),
                oth_geometry.extract_grid(
                    observers[other_name].sediments[0].input_surface_m),
                absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
            "cumulative_erosion_source_m": _numeric_comparison(
                ref_geometry.extract_grid(
                    observers[reference_name].sediments[0].erosion_source_m),
                oth_geometry.extract_grid(
                    observers[other_name].sediments[0].erosion_source_m),
                absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
            "land_deposit_m": _numeric_comparison(
                ref_geometry.extract_grid(reference_land.land_deposit_m),
                oth_geometry.extract_grid(other_land.land_deposit_m),
                absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
            "pre_marine_bed_m": _numeric_comparison(
                ref_geometry.extract_grid(
                    reference_marine.pre_marine_bed_m),
                oth_geometry.extract_grid(other_marine.pre_marine_bed_m),
                absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
            "mouth_flux_m_cells": _numeric_comparison(
                ref_geometry.extract_grid(
                    reference_marine.mouth_flux_m_cells),
                oth_geometry.extract_grid(other_marine.mouth_flux_m_cells),
                absolute_threshold=(
                    MOUTH_MATERIAL_REPORTING_THRESHOLD_M_CELLS)),
            "marine_deposit_m": _numeric_comparison(
                ref_geometry.extract_grid(
                    reference_marine.marine_deposit_m),
                oth_geometry.extract_grid(other_marine.marine_deposit_m),
                absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M),
        }

    fixed_seam = seam(reference_geometry, other_geometry)
    result = {
        "full_delivered_result": physical_replay._core_comparison(
            reference_result, other_result,
            reference_geometry, other_geometry),
        **fixed_seam,
        "sediment_seam_fixed_delivery_core": fixed_seam,
        "sediment_seam_three_window_common_support": seam(
            common_reference_geometry, common_other_geometry),
        "routing_stages_fixed_delivery_core": {
            stage: _stage_comparison(
                observers[reference_name], observers[other_name], stage)
            for stage in ROUTING_STAGES
        },
        "routing_stages_three_window_common_support": {
            stage: _stage_comparison(
                observers[reference_name], observers[other_name], stage,
                common_reference_geometry, common_other_geometry,
                include_direction_fields=False)
            for stage in ROUTING_STAGES
        },
    }
    discharge_report, _ = _discharge_comparison(
        reference_geometry.extract_grid(reference_result["discharge_log"]),
        other_geometry.extract_grid(other_result["discharge_log"]))
    result["final_discharge"] = discharge_report
    return result


def _common_mouth_report(reference_name, other_name, observers,
                         land_replays, windows, common_rect) -> dict:
    def fields(name):
        replayed = land_replays[name]
        window = windows[name]
        return {
            "total": _extract_rect(
                replayed.mouth_flux_m_cells, window, common_rect),
            "from_common_source": _extract_rect(
                replayed.mouth_from_common_source_m_cells,
                window, common_rect),
            "from_window_only_source": _extract_rect(
                replayed.mouth_from_window_only_source_m_cells,
                window, common_rect),
        }

    left = fields(reference_name)
    right = fields(other_name)
    return {
        name: _numeric_comparison(
            left[name], right[name],
            absolute_threshold=MOUTH_MATERIAL_REPORTING_THRESHOLD_M_CELLS)
        for name in left
    }


def _material_coordinate_sets(solved, geometries) -> tuple[dict, dict]:
    sediment_sets = {}
    discharge_sets = {}
    for relation, reference_name, other_name in RELATIONS:
        reference_geometry = geometries[reference_name]
        other_geometry = geometries[other_name]
        sed_ref = reference_geometry.extract_grid(
            solved[reference_name]["sed"])
        sed_other = other_geometry.extract_grid(solved[other_name]["sed"])
        sediment_sets[relation] = set(_core_coordinates(
            reference_geometry, _material_mask(sed_ref, sed_other)))
        discharge_ref = reference_geometry.extract_grid(
            solved[reference_name]["discharge_log"])
        discharge_other = other_geometry.extract_grid(
            solved[other_name]["discharge_log"])
        _, discharge_mask = _discharge_comparison(
            discharge_ref, discharge_other)
        discharge_sets[relation] = set(_core_coordinates(
            reference_geometry, discharge_mask))
    return sediment_sets, discharge_sets


def _set_relation_report(sets: dict) -> dict:
    result = {
        name: {
            "count": len(values),
            "global_row_column": [list(value) for value in sorted(values)],
        }
        for name, values in sets.items()
    }
    names = list(sets)
    intersections = {}
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            left = sets[left_name]
            right = sets[right_name]
            union = left | right
            intersections[f"{left_name}__{right_name}"] = {
                "intersection_count": len(left & right),
                "symmetric_difference_count": len(left ^ right),
                "jaccard": (1.0 if not union
                            else float(len(left & right) / len(union))),
            }
    return {"relations": result, "cross_relation_identity": intersections}


def _terminal_trace(observer: WindowObserver, stage: str,
                    coordinate) -> dict:
    index = ROUTING_STAGES.index(stage)
    d8 = observer.d8s[index]
    outlet = observer.fills[index].outlet_mask.ravel()
    side = observer.window[2]
    row0, column0, _ = observer.window
    local_row = coordinate[0] - row0
    local_column = coordinate[1] - column0
    current = int(local_row * side + local_column)
    visited = set()
    hops = 0
    while True:
        if current in visited:
            return {"terminal_type": "cycle", "hops": hops}
        visited.add(current)
        target = int(d8.receiver[current])
        if target == current:
            row, column = divmod(current, side)
            if outlet[current]:
                terminal_type = "physical_lowstand_outlet"
            elif row in (0, side - 1) or column in (0, side - 1):
                terminal_type = "numerical_outer_ring_self_receiver"
            else:
                terminal_type = "interior_self_receiver"
            return {
                "terminal_type": terminal_type,
                "terminal_global_row_column": [
                    row0 + row, column0 + column],
                "hops": hops,
                "reaches_numerical_outer_ring": (
                    terminal_type
                    == "numerical_outer_ring_self_receiver"),
            }
        current = target
        hops += 1
        if hops > d8.receiver.size:
            raise AssertionError("receiver path exceeded graph size")


def _cell_ledger(solved, observers, land_replays, windows,
                 sediment_sets, discharge_sets) -> list[dict]:
    coordinates = sorted(set().union(
        *sediment_sets.values(), *discharge_sets.values()))
    records = []
    for coordinate in coordinates:
        values = {}
        for name in WINDOW_ORDER:
            sediment = observers[name].sediments[0]
            marine = observers[name].marines[0]
            land = land_replays[name]
            window = windows[name]
            final_discharge = float(np.expm1(_global_value(
                solved[name]["discharge_log"], window, coordinate)))
            pre_surface = float(_global_value(
                sediment.input_surface_m, window, coordinate))
            final_surface = float(_global_value(
                solved[name]["z"], window, coordinate))
            stage_lowstand = {}
            for index, stage in enumerate(ROUTING_STAGES):
                surface_array = _raw_stage_surface(observers[name], stage)
                surface_value = float(_global_value(
                    surface_array, window, coordinate))
                stage_record = {
                    "surface_m": surface_value,
                    "surface_minus_lowstand_m": (
                        surface_value - sediment.base_level_m),
                    "route_outlet_surface_le_lowstand": bool(_global_value(
                        observers[name].fills[index].outlet_mask,
                        window, coordinate)),
                }
                if stage in ("incision_0", "incision_1"):
                    stage_record[
                        "solve_base_surface_strictly_less_than_lowstand"] = (
                            bool(_global_value(
                                observers[name].solves[index].base_mask_strict_less,
                                window, coordinate)))
                stage_lowstand[stage] = stage_record
            values[name] = {
                "z0_m": float(_global_value(
                    solved[name]["z0"], window, coordinate)),
                "pre_sediment_surface_m": pre_surface,
                "pre_sediment_lowstand_margin_m": (
                    pre_surface - sediment.base_level_m),
                "pre_sediment_lowstand_outlet": bool(
                    pre_surface <= sediment.base_level_m),
                "land_deposit_m": float(_global_value(
                    land.land_deposit_m, window, coordinate)),
                "pre_marine_bed_m": float(_global_value(
                    marine.pre_marine_bed_m, window, coordinate)),
                "mouth_flux_m_cells": float(_global_value(
                    marine.mouth_flux_m_cells, window, coordinate)),
                "marine_deposit_m": float(_global_value(
                    marine.marine_deposit_m, window, coordinate)),
                "total_deposit_m": float(_global_value(
                    solved[name]["sed"], window, coordinate)),
                "final_surface_m": final_surface,
                "post_sediment_lowstand_margin_m": (
                    final_surface - sediment.base_level_m),
                "post_sediment_lowstand_outlet": bool(
                    final_surface <= sediment.base_level_m),
                "all_stage_lowstand_ledger": stage_lowstand,
                "final_discharge_linearized": final_discharge,
                "final_route_terminal": _terminal_trace(
                    observers[name], "post_sediment", coordinate),
            }
        pairwise = {}
        for relation, reference_name, other_name in RELATIONS:
            pairwise[relation] = {
                "other_minus_reference": {
                    field: (values[other_name][field]
                            - values[reference_name][field])
                    for field in (
                        "pre_sediment_surface_m", "land_deposit_m",
                        "pre_marine_bed_m", "mouth_flux_m_cells",
                        "marine_deposit_m", "total_deposit_m",
                        "final_surface_m", "final_discharge_linearized")
                }
            }
        records.append({
            "global_row_column": list(coordinate),
            "material_membership": {
                "sediment": [name for name, selected in sediment_sets.items()
                             if coordinate in selected],
                "discharge": [name for name, selected in discharge_sets.items()
                              if coordinate in selected],
            },
            "windows": values,
            "pairwise": pairwise,
        })
    return records


def _mouth_examples(reference_name, other_name, land_replays, windows,
                    common_rect) -> list[dict]:
    fields = (
        ("total", "mouth_flux_m_cells"),
        ("from_common_source", "mouth_from_common_source_m_cells"),
        ("from_window_only_source",
         "mouth_from_window_only_source_m_cells"),
    )
    extracted = {}
    for name in (reference_name, other_name):
        extracted[name] = {
            public: _extract_rect(getattr(land_replays[name], attribute),
                                  windows[name], common_rect)
            for public, attribute in fields
        }
    difference = np.abs(
        extracted[other_name]["total"] - extracted[reference_name]["total"])
    selected = np.argwhere(difference > MOUTH_NUMERIC_THRESHOLD_M_CELLS)
    if selected.size == 0:
        return []
    ranked = sorted(
        ((float(difference[row, column]), int(row), int(column))
         for row, column in selected),
        key=lambda value: (-value[0], value[1], value[2]))
    records = []
    for delta, row, column in ranked[:MAX_MOUTH_EXAMPLES]:
        coordinate = (common_rect[0] + row, common_rect[2] + column)
        values = {
            name: {
                field: float(extracted[name][field][row, column])
                for field, _ in fields
            }
            for name in (reference_name, other_name)
        }
        records.append({
            "global_row_column": list(coordinate),
            "absolute_total_difference_m_cells": delta,
            "material_at_reporting_threshold": bool(
                delta > MOUTH_MATERIAL_REPORTING_THRESHOLD_M_CELLS),
            "windows": values,
        })
    return records


def _first_divergence(component_report) -> dict:
    exact_order = []
    topological_order = []
    material_order = []

    def append_fields(stage, report, fields, scope):
        for field in fields:
            if field not in report:
                continue
            value = report[field]
            changed = value.get("exact_changed_cells",
                                value.get("changed_cells", 0))
            if changed:
                exact_order.append({"stage": stage, "scope": scope,
                                    "field": field,
                                    "changed_cells": int(changed)})
            topological = value.get("topological_changed_cells")
            if topological:
                topological_order.append({
                    "stage": stage, "scope": scope, "field": field,
                    "changed_cells": int(topological),
                    "metric": value.get("topological_metric"),
                })
            material = value.get("material_changed_cells")
            if material:
                material_order.append({
                    "stage": stage, "scope": scope, "field": field,
                    "changed_cells": int(material),
                    "metric": value.get("material_metric"),
                    "threshold": value.get("material_threshold"),
                })

    routing = component_report[
        "routing_stages_three_window_common_support"]
    route_fields = (
        "raw_surface_m", "lowstand_outlet_mask_surface_le_base",
        "clamped_routing_surface_m", "filled_surface_m",
        "d8_receiver_global_row_column", "runoff_input",
        "effective_mfd_accumulation",
        "effective_d8_accumulation",
        "solve_base_mask_surface_strictly_less_than_lowstand",
        "stream_power_cut_m", "solve_output_before_creep_m",
        "creep_delta_m",
    )
    # Chronology matters: the post-sediment route is a consumer of the
    # terrestrial/marine seam and is intentionally evaluated afterward.
    for stage in ("incision_0", "incision_1", "pre_sediment"):
        append_fields(stage, routing[stage], route_fields,
                      "three_window_common_support")
    seam = component_report[
        "sediment_seam_three_window_common_support"]
    append_fields("sediment_land_handoff", seam, (
        "pre_sediment_surface_m", "cumulative_erosion_source_m",
        "land_deposit_m",
        "pre_marine_bed_m", "mouth_flux_m_cells"),
        "three_window_common_support")
    append_fields("physical_marine_transport", seam, (
        "marine_deposit_m",), "three_window_common_support")
    append_fields("post_sediment", routing["post_sediment"], route_fields,
                  "three_window_common_support")
    append_fields("delivered_output", {
        "final_discharge": component_report["final_discharge"]},
        ("final_discharge",), "fixed_delivery_core")
    return {
        "first_exact_divergence": exact_order[0] if exact_order else None,
        "first_topological_divergence": (
            topological_order[0] if topological_order else None),
        "first_material_divergence": (
            material_order[0] if material_order else None),
        "exact_divergence_order": exact_order,
        "topological_divergence_order": topological_order,
        "material_divergence_order": material_order,
        "warning": (
            "Exact numeric, exact topological, and threshold-material orders "
            "answer different questions; none is inferred from morphology. "
            "The common-support order is the earliest difference anywhere in "
            "that support, not proof that the cell lies on the causal path to "
            "the delivered residual."),
    }


def _threshold_xor_report(reference_mask, other_mask, reference_surface,
                          other_surface, reference_base, other_base,
                          reference_geometry, other_geometry) -> dict:
    reference_mask = reference_geometry.extract_grid(reference_mask)
    other_mask = other_geometry.extract_grid(other_mask)
    reference_margin = (reference_geometry.extract_grid(reference_surface)
                        - reference_base)
    other_margin = (other_geometry.extract_grid(other_surface)
                    - other_base)
    changed = reference_mask != other_mask
    near = (changed
            & (np.abs(reference_margin) <= NUMERIC_DIAGNOSTIC_THRESHOLD)
            & (np.abs(other_margin) <= NUMERIC_DIAGNOSTIC_THRESHOLD))
    coordinates = _core_coordinates(reference_geometry, changed)
    coordinate_array = np.asarray(coordinates, np.int64).reshape(-1, 2)
    examples = []
    for row, column in np.argwhere(changed)[:MAX_COORDINATE_EXAMPLES]:
        examples.append({
            "global_row_column": [
                int(reference_geometry.global_rows[row]),
                int(reference_geometry.global_columns[column]),
            ],
            "reference_margin_m": float(reference_margin[row, column]),
            "other_margin_m": float(other_margin[row, column]),
            "reference_mask": bool(reference_mask[row, column]),
            "other_mask": bool(other_mask[row, column]),
            "both_margins_within_1e_minus_9_m": bool(near[row, column]),
        })
    return {
        "xor_cells": int(np.count_nonzero(changed)),
        "near_threshold_xor_cells_both_margins_within_1e_minus_9_m": int(
            np.count_nonzero(near)),
        "xor_global_coordinate_sha256": _array_sha256(coordinate_array),
        "all_coordinates_included": len(coordinates) <= MAX_COORDINATE_EXAMPLES,
        "coordinate_and_margin_examples": examples,
        "maximum_absolute_margin_at_xor_m": (
            None if not changed.any() else float(max(
                np.abs(reference_margin[changed]).max(),
                np.abs(other_margin[changed]).max()))),
    }


def _lowstand_threshold_relation(reference_name, other_name, observers,
                                 reference_geometry, other_geometry) -> dict:
    reference = observers[reference_name]
    other = observers[other_name]
    stages = {}
    for index, stage in enumerate(ROUTING_STAGES):
        raw_reference = _raw_stage_surface(reference, stage)
        raw_other = _raw_stage_surface(other, stage)
        base_reference = reference.sediments[0].base_level_m
        base_other = other.sediments[0].base_level_m
        record = {
            "route_outlet_surface_le_lowstand": _threshold_xor_report(
                reference.fills[index].outlet_mask,
                other.fills[index].outlet_mask,
                raw_reference, raw_other, base_reference, base_other,
                reference_geometry, other_geometry),
        }
        if stage in ("incision_0", "incision_1"):
            record["solve_base_surface_strictly_less_than_lowstand"] = (
                _threshold_xor_report(
                    reference.solves[index].base_mask_strict_less,
                    other.solves[index].base_mask_strict_less,
                    raw_reference, raw_other, base_reference, base_other,
                    reference_geometry, other_geometry))
        stages[stage] = record
    return {
        "scope": {
            "reference": reference_geometry.report(),
            "other": other_geometry.report(),
        },
        "numeric_near_threshold_definition": (
            "both signed surface-minus-lowstand margins have absolute value "
            "<= 1e-9 m at an exact mask XOR"),
        "stages": stages,
    }


def _post_sediment_flip_evidence(reference_name, other_name, solved,
                                 observers, geometries) -> dict:
    reference_geometry = geometries[reference_name]
    other_geometry = geometries[other_name]
    discharge_report, discharge_material = _discharge_comparison(
        reference_geometry.extract_grid(
            solved[reference_name]["discharge_log"]),
        other_geometry.extract_grid(solved[other_name]["discharge_log"]))
    pre_ref = reference_geometry.extract_grid(
        observers[reference_name].fills[2].outlet_mask)
    pre_other = other_geometry.extract_grid(
        observers[other_name].fills[2].outlet_mask)
    post_ref = reference_geometry.extract_grid(
        observers[reference_name].fills[3].outlet_mask)
    post_other = other_geometry.extract_grid(
        observers[other_name].fills[3].outlet_mask)
    pre_xor = pre_ref != pre_other
    post_xor = post_ref != post_other
    discharge_coordinates = set(_core_coordinates(
        reference_geometry, discharge_material))
    pre_coordinates = set(_core_coordinates(reference_geometry, pre_xor))
    post_coordinates = set(_core_coordinates(reference_geometry, post_xor))
    colocated = bool(
        discharge_coordinates
        and discharge_coordinates <= post_coordinates
        and discharge_coordinates.isdisjoint(pre_coordinates))
    margins = []
    for coordinate in sorted(discharge_coordinates | post_coordinates):
        record = {"global_row_column": list(coordinate), "windows": {}}
        for name in (reference_name, other_name):
            surface = observers[name].sediments[0].output_surface_m
            base = observers[name].sediments[0].base_level_m
            value = float(_global_value(
                surface, observers[name].window, coordinate))
            record["windows"][name] = {
                "surface_m": value,
                "surface_minus_lowstand_m": value - base,
                "outlet_surface_le_lowstand": bool(value <= base),
            }
        margins.append(record)
    return {
        "discharge": discharge_report,
        "discharge_material_coordinates": [
            list(value) for value in sorted(discharge_coordinates)],
        "pre_sediment_lowstand_xor_coordinates": [
            list(value) for value in sorted(pre_coordinates)],
        "post_sediment_lowstand_xor_coordinates": [
            list(value) for value in sorted(post_coordinates)],
        "discharge_material_cells_subset_of_post_only_mask_flips": colocated,
        "classification": (
            "post_sediment_mask_flip_colocation_consistent_with_discharge_difference"
            if colocated else "no_complete_post_only_mask_flip_colocation"),
        "causal_status": (
            "strong_mechanistic_consistency_not_a_counterfactual_proof; final "
            "surface includes both terrestrial and marine deposition"),
        "threshold_margin_ledger": margins,
    }


def _counterfactual_classification(actual_relation, variant_relations,
                                   frozen_relation, factorial_effects,
                                   fixed_both_effective_source_equal,
                                   mouth_report, component_report) -> dict:
    actual_material = actual_relation[
        "marine_deposit_m"]["material_changed_cells"] > 0
    variant_material = {
        name: report["marine_deposit_m"]["material_changed_cells"] > 0
        for name, report in variant_relations.items()
    }
    frozen_material = frozen_relation[
        "marine_deposit_m"]["material_changed_cells"] > 0
    frozen_overlap = factorial_effects[
        "frozen_graph_material_pattern_overlap_with_actual"]
    if actual_material and not frozen_material:
        graph_feedback = (
            "dynamic_graph_rebuild_necessary_for_any_material_relation_"
            "divergence_in_this_case")
    elif actual_material and frozen_material and frozen_overlap:
        graph_feedback = (
            "frozen_graph_retains_material_cells_overlapping_actual_pattern")
    elif actual_material and frozen_material:
        graph_feedback = (
            "frozen_graph_has_material_divergence_but_not_at_actual_material_cells")
    else:
        graph_feedback = "not_applicable_without_material_actual_divergence"

    common_origin_material = mouth_report[
        "from_common_source"]["material_changed_cells"] > 0
    exterior_origin_material = mouth_report[
        "from_window_only_source"]["material_changed_cells"] > 0
    common_routing = component_report[
        "routing_stages_three_window_common_support"]
    pre_mask_changed = common_routing[
        "pre_sediment"]["lowstand_outlet_mask_surface_le_base"][
            "changed_cells"] > 0
    pre_receiver_changed = common_routing[
        "pre_sediment"]["d8_receiver_global_row_column"][
            "changed_cells"] > 0
    effects = factorial_effects["signed_factorial_effects"]
    material_effects = [
        name for name, value in effects.items()
        if value["material_changed_cells"] > 0
    ]
    overlapping_effects = [
        name for name, value in effects.items()
        if value["material_overlap_with_actual_cells"] > 0
    ]
    evidence = []
    if exterior_origin_material:
        evidence.append(
            "exterior-origin tagged share differs materially at common-support mouths under observed co-flow")
    if common_origin_material:
        evidence.append(
            "common-origin tagged share differs materially at common-support mouths under observed co-flow")
    if pre_mask_changed:
        evidence.append("pre-sediment lowstand classification differs")
    if pre_receiver_changed:
        evidence.append("pre-sediment common-support D8 receiver differs")
    if not evidence:
        evidence.append("no listed upstream material discriminator fired")
    return {
        "status": (
            "quantitative_conditional_causal_discrimination"
            if actual_material else "no_material_actual_marine_divergence"),
        "conditional_arm_material_persistence": variant_material,
        "material_signed_factorial_effects": material_effects,
        "material_effects_overlapping_actual_residual_cells": (
            overlapping_effects),
        "fixed_both_effective_source_globally_equal": (
            fixed_both_effective_source_equal),
        "direct_domain_or_outside_common_support_effect_supported": bool(
            fixed_both_effective_source_equal
            and effects[
                "domain_or_outside_common_support_baseline"][
                    "material_overlap_with_actual_cells"] > 0),
        "combined_subthreshold_effects_may_cross_output_threshold": bool(
            actual_material and not material_effects),
        "dynamic_graph_feedback_ablation": graph_feedback,
        "upstream_origin_and_routing_evidence": evidence,
        "exterior_origin_tagged_share_differs_under_observed_coflow": (
            exterior_origin_material),
        "common_origin_tagged_share_differs_under_observed_coflow": (
            common_origin_material),
        "pre_sediment_lowstand_mask_evidence": pre_mask_changed,
        "pre_sediment_receiver_evidence_in_common_support": (
            pre_receiver_changed),
        "interpretation_limits": [
            "Conditional arm persistence is not a unique-cause label; signed contrasts and actual-cell overlap carry the stronger interpretation.",
            "Exterior-origin tagging locates source mass but is not a removal counterfactual and does not prove its route was free of numerical-rim influence.",
            "The bed intervention fixes only the three-window common rectangle; exterior bed and domain remain native.",
            "A differing common-origin mouth share localizes the issue upstream but does not alone prove numerical-rim causation.",
            "The frozen-graph arm is a private causal ablation, not a proposed production process.",
            "Material equality uses the frozen 0.05 m output threshold; exact residuals are reported separately.",
        ],
    }


def _effective_config(cfg) -> dict:
    return asdict(cfg)


def _protocol(fingerprint: dict, cfg) -> dict:
    common = _window_intersection(EXPECTED_WINDOWS.values())
    return {
        "experiment": EXPERIMENT,
        "manifest_role": "pre-execution fixed causal-discrimination protocol",
        "source_fingerprint": fingerprint,
        "prior_artifact_expected_digests": PRIOR_ARTIFACTS,
        "fixed": {
            "seed": SEED,
            "continental_budget": CONTINENTAL_BUDGET,
            "complete_effective_config": _effective_config(cfg),
            "origin_xy_km": list(replay.PRIMARY_ORIGIN),
            "windows": {name: list(value)
                        for name, value in EXPECTED_WINDOWS.items()},
            "three_window_common_intersection_row_exclusive_row_column_exclusive_column": (
                list(common)),
            "localization_mode": "physical_outlets",
            "window_order": list(WINDOW_ORDER),
            "relations": [list(value) for value in RELATIONS],
            "retries": 0,
        },
        "sequencing": {
            "structural_builds": 1,
            "coarse_elevation_builds": 1,
            "erosion_calls": 3,
            "direct_original_marine_counterfactual_calls": (
                EXPECTED_DIRECT_MARINE_CALLS),
            "private_frozen_graph_ablation_calls": (
                EXPECTED_FROZEN_GRAPH_ABLATION_CALLS),
            "counterfactual_variant_order": list(COUNTERFACTUAL_VARIANTS),
        },
        "canonical_inputs": {
            "source": (
                "large-window mouth flux cropped to the three-window common "
                "intersection, globally embedded, zero elsewhere"),
            "bed": (
                "large-window pre-marine bed over the common intersection; "
                "native bed retained outside it"),
            "variants": {
                "fixed_source_native_bed": (
                    "canonical source; each window's native bed"),
                "native_source_fixed_common_bed": (
                    "each full native source; canonical common bed"),
                "fixed_source_fixed_common_bed": (
                    "canonical source and canonical common bed"),
            },
            "source_identity_is_global_sparse_coordinate_value_identity": True,
            "effective_source_after_bed_mask_is_reported": True,
            "fixed_both_effective_source_identity_required_for_domain_label": (
                True),
        },
        "comparison_policy": {
            "terrain_and_deposit_material_absolute_threshold_m": (
                TERRAIN_MATERIAL_THRESHOLD_M),
            "hydrology_linearized_relative_threshold": (
                HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD),
            "fill_numeric_diagnostic_threshold_m": (
                NUMERIC_DIAGNOSTIC_THRESHOLD),
            "mouth_numeric_diagnostic_threshold_m_cells": (
                MOUTH_NUMERIC_THRESHOLD_M_CELLS),
            "mouth_material_reporting_threshold_m_cells": (
                MOUTH_MATERIAL_REPORTING_THRESHOLD_M_CELLS),
            "mask_receiver_terminal_and_graph_rule": (
                "exact topological inequality, reported separately from "
                "threshold-material numeric divergence"),
            "first_exact_topological_and_material_divergence_separate": True,
            "routing_comparison_scopes": [
                "fixed delivered frame plus 40-km collar",
                "full three-window common support"],
            "thresholds_are_not_retuned_after_observation": True,
        },
        "classification_policy": {
            "factorial_decomposition": (
                "signed actual relation delta is decomposed algebraically into "
                "fixed-both domain baseline, native-bed contrast, native-source "
                "contrast, and source-bed interaction"),
            "actual_pattern_requirement": (
                "each material effect reports overlap and sign agreement with "
                "the actual material residual cells"),
            "exterior_origin_tag": (
                "a proportional observed-coflow mass partition, not a source-"
                "removal counterfactual or proof of boundary-free routing"),
            "direct_marine_domain_sensitivity": (
                "fixed source plus fixed common bed has equal effective source "
                "yet retains a signed material effect overlapping the actual "
                "residual"),
            "dynamic_feedback_required": (
                "actual native-input divergence is material but frozen-initial-graph "
                "native-input divergence is not"),
            "post_sediment_discharge_colocation": (
                "material discharge coordinates are post-sediment lowstand XORs "
                "and absent pre-sediment; this is mechanistic consistency, not "
                "counterfactual proof"),
            "unresolved_is_allowed": True,
        },
        "instrumentation": {
            "pass_through_wrappers_restored_before_counterfactuals": True,
            "captured_routing_stages": list(ROUTING_STAGES),
            "raw_and_derived_effective_accumulations_reported_separately": True,
            "strict_incision_and_nonstrict_route_lowstand_masks_captured": True,
            "land_handoff_replayed_and_exactly_validated": True,
            "mixed_flux_origin_attribution": (
                "common and window-only source components share each cell's "
                "observed survival fraction; their sum is validated against "
                "the exact total mouth flux"),
            "engine_source_modified": False,
            "historical_spikes_modified": False,
            "timings_are_not_performance_authoritative": True,
        },
        "decision_boundary": {
            "diagnostic_private_branch_only": True,
            "correction_implemented": False,
            "default_or_public_controls_changed": False,
            "promotion_assessed": False,
            "one_seed_does_not_establish_population_frequency": True,
        },
    }


def _historical_headline_reproduction(component_reports) -> dict:
    path = WORKSPACE / PRIOR_ARTIFACTS[
        "physical_outlet_replay"]["relative_path"]
    prior = json.loads(path.read_text(encoding="utf-8"))
    relations = {}
    all_matched = True
    for relation in ("small_vs_large", "shifted_vs_large"):
        expected_fields = prior["relations"][relation]["core"]["fields"]
        observed_fields = component_reports[relation][
            "full_delivered_result"]["fields"]
        fields = {}
        for field in ("z", "z0", "ero", "sed", "discharge_log",
                      "lake_depth", "lake_surf"):
            keys = [
                "exact_changed_cells", "material_changed_cells", "max_abs"]
            if field == "discharge_log":
                keys.append("max_linearized_relative")
            expected = {key: expected_fields[field][key] for key in keys}
            observed = {key: observed_fields[field][key] for key in keys}
            matched = observed == expected
            fields[field] = {
                "expected": expected, "observed": observed,
                "matched": matched,
            }
            all_matched &= matched
        relations[relation] = fields
    return {
        "prior_report_sha256": PRIOR_ARTIFACTS[
            "physical_outlet_replay"]["sha256"],
        "relations": relations,
        "all_headline_fields_exactly_reproduced": all_matched,
    }


def _window_capture_report(name, observer, land, window) -> dict:
    marine = observer.marines[0]
    actual = _actual_marine_outcome(marine)
    return {
        "process_window_row_column_side": list(window),
        "process_spacing_km": marine.process_spacing_km,
        "base_level_m": marine.base_level_m,
        "deposition_length_km": marine.deposition_length_km,
        "instrumentation_call_counts": {
            "fill": len(observer.fills), "flow": len(observer.flows),
            "d8": len(observer.d8s), "solve": len(observer.solves),
            "sediment": len(observer.sediments),
            "physical_marine": len(observer.marines),
        },
        "input_mutation_checks": {
            "marine_bed_unchanged": marine.bed_unchanged_by_call,
            "mouth_flux_unchanged": marine.mouth_flux_unchanged_by_call,
        },
        "fill_fixed_round_replays": {
            stage: _fill_replay(observer.fills[index])
            for index, stage in enumerate(ROUTING_STAGES)
        },
        "fill_replay_limit": (
            "This reproduces whether the shipped eight-round loop observed "
            "exact equality; it does not continue to convergence and is not "
            "itself a causal ablation of the round cap."),
        "land_handoff_replay_validation": land.validation,
        "origin_attribution_method": (
            "At mixed terrestrial cells, common-support and window-only "
            "components receive the same observed post-deposition survival "
            "fraction. This is an exact mass partition of total flux up to "
            "the reported numerical reconstruction error, not particle "
            "tracking."),
        "terrestrial": {
            "erosion_source": _array_summary(land.source_m),
            "land_deposit": _array_summary(land.land_deposit_m),
            "mouth_flux": {
                **_array_summary(land.mouth_flux_m_cells),
                **_global_sparse_summary(land.mouth_flux_m_cells, window),
            },
            "mouth_from_common_source": {
                **_array_summary(
                    land.mouth_from_common_source_m_cells),
                **_global_sparse_summary(
                    land.mouth_from_common_source_m_cells, window),
            },
            "mouth_from_window_only_source": {
                **_array_summary(
                    land.mouth_from_window_only_source_m_cells),
                **_global_sparse_summary(
                    land.mouth_from_window_only_source_m_cells, window),
            },
            "boundary_export_m_cells": land.boundary_export_m_cells,
            "terminal_residual_m_cells": land.terminal_residual_m_cells,
        },
        "marine_actual": _marine_outcome_summary(actual, window),
        "marine_actual_validation": _marine_outcome_validation(actual),
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
        SEED, cfg,
        _world_km=replay.ATLAS_KM,
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
        windows["large"], structure,
        -replay.SHIFT_KM, replay.SHIFT_KM)
    windows = {name: tuple(int(value) for value in window)
               for name, window in windows.items()}
    if windows != EXPECTED_WINDOWS:
        raise AssertionError({"expected_windows": EXPECTED_WINDOWS,
                              "observed_windows": windows})
    common_rect = _window_intersection(windows.values())
    expected_common = (
        EXPECTED_WINDOWS["small"][0],
        EXPECTED_WINDOWS["small"][0] + EXPECTED_WINDOWS["small"][2],
        EXPECTED_WINDOWS["small"][1],
        EXPECTED_WINDOWS["small"][1] + EXPECTED_WINDOWS["small"][2],
    )
    if common_rect != expected_common:
        raise AssertionError({"expected_common": expected_common,
                              "observed_common": common_rect})
    geometries = {
        name: stage_diagnostic.CoreGeometry.fixed(
            name, windows[name], structure)
        for name in WINDOW_ORDER
    }
    common_rows = np.arange(common_rect[0], common_rect[1], dtype=np.int64)
    common_columns = np.arange(
        common_rect[2], common_rect[3], dtype=np.int64)
    common_geometries = {
        name: stage_diagnostic.CoreGeometry.explicit(
            name + "_three_window_common", windows[name],
            geometries[name].e_km, common_rows, common_columns)
        for name in WINDOW_ORDER
    }

    solved = {}
    observers = {}
    wall_times = {}
    observed_function_names = (
        "_fill_to_lowstand_outlets", "flow_accumulation",
        "flow_accumulation_d8", "spl_implicit",
        "_route_sediment_lowstand", "_physical_marine_transport")
    engine_functions_before = {
        name: getattr(erosion_engine, name) for name in observed_function_names}
    with PhysicalInstrumentation() as instrumentation:
        for name in WINDOW_ORDER:
            observer = WindowObserver(name, windows[name], geometries[name])
            instrumentation.active = observer
            call_started = time.perf_counter()
            try:
                solved[name] = replay.run_erosion(
                    structure, elevation, cfg, SEED,
                    _process_window=windows[name],
                    _localization_mode="physical_outlets")
            finally:
                instrumentation.active = None
            wall_times[name] = time.perf_counter() - call_started
            observer.finalize()
            observers[name] = observer
    engine_functions_restored = all(
        getattr(erosion_engine, name) is function
        for name, function in engine_functions_before.items())

    land_replays = {
        name: _replay_land_sediment(
            observers[name].sediments[0], windows[name], common_rect,
            observers[name].marines[0])
        for name in WINDOW_ORDER
    }
    component_reports = {
        relation: _relation_component_report(
            reference_name, other_name, solved, observers,
            land_replays, geometries, common_geometries)
        for relation, reference_name, other_name in RELATIONS
    }
    historical_reproduction = _historical_headline_reproduction(
        component_reports)
    if not historical_reproduction[
            "all_headline_fields_exactly_reproduced"]:
        raise AssertionError({
            "historical_headline_reproduction": historical_reproduction})

    common_mouth = {
        relation: _common_mouth_report(
            reference_name, other_name, observers, land_replays,
            windows, common_rect)
        for relation, reference_name, other_name in RELATIONS
    }
    mouth_examples = {
        relation: _mouth_examples(
            reference_name, other_name, land_replays, windows, common_rect)
        for relation, reference_name, other_name in RELATIONS
    }

    large_marine = observers["large"].marines[0]
    canonical_source_common = _extract_rect(
        large_marine.mouth_flux_m_cells, windows["large"], common_rect)
    canonical_bed_common = _extract_rect(
        large_marine.pre_marine_bed_m, windows["large"], common_rect)
    counterfactuals: dict[str, dict[str, MarineOutcome]] = {
        variant: {} for variant in COUNTERFACTUAL_VARIANTS}
    direct_call_count = 0
    for variant in COUNTERFACTUAL_VARIANTS:
        for name in WINDOW_ORDER:
            native = observers[name].marines[0]
            if variant in (
                    "fixed_source_native_bed",
                    "fixed_source_fixed_common_bed"):
                source = _embed_rect(
                    canonical_source_common, common_rect, windows[name])
            else:
                source = native.mouth_flux_m_cells
            if variant in (
                    "native_source_fixed_common_bed",
                    "fixed_source_fixed_common_bed"):
                bed = _replace_rect(
                    native.pre_marine_bed_m, windows[name], common_rect,
                    canonical_bed_common)
            else:
                bed = native.pre_marine_bed_m
            counterfactuals[variant][name] = _run_direct_marine(
                bed, source, native.base_level_m,
                native.deposition_length_km, native.process_spacing_km)
            direct_call_count += 1
    if direct_call_count != EXPECTED_DIRECT_MARINE_CALLS:
        raise AssertionError((direct_call_count,
                              EXPECTED_DIRECT_MARINE_CALLS))

    frozen_graph = {}
    for name in WINDOW_ORDER:
        native = observers[name].marines[0]
        frozen_graph[name] = _run_frozen_graph_marine(
            native.pre_marine_bed_m, native.mouth_flux_m_cells,
            native.base_level_m, native.deposition_length_km,
            native.process_spacing_km)
    if len(frozen_graph) != EXPECTED_FROZEN_GRAPH_ABLATION_CALLS:
        raise AssertionError((len(frozen_graph),
                              EXPECTED_FROZEN_GRAPH_ABLATION_CALLS))

    actual_outcomes = {
        name: _actual_marine_outcome(observers[name].marines[0])
        for name in WINDOW_ORDER
    }
    fixed_both_effective_source_summaries = {
        name: _global_sparse_summary(
            counterfactuals[
                "fixed_source_fixed_common_bed"][name].effective_source_m_cells,
            windows[name])
        for name in WINDOW_ORDER
    }
    fixed_both_effective_source_identity = (
        len({value["global_sparse_coordinate_value_sha256"]
             for value in fixed_both_effective_source_summaries.values()}) == 1
        and len({value["sum_m_cells"]
                 for value in fixed_both_effective_source_summaries.values()})
        == 1)
    marine_relations = {}
    for relation, reference_name, other_name in RELATIONS:
        actual_relation = _marine_relation(
            actual_outcomes[reference_name], actual_outcomes[other_name],
            geometries[reference_name], geometries[other_name])
        variant_relations = {
            variant: _marine_relation(
                counterfactuals[variant][reference_name],
                counterfactuals[variant][other_name],
                geometries[reference_name], geometries[other_name])
            for variant in COUNTERFACTUAL_VARIANTS
        }
        frozen_relation = _marine_relation(
            frozen_graph[reference_name], frozen_graph[other_name],
            geometries[reference_name], geometries[other_name])
        factorial_effects = _factorial_effect_report(
            actual_outcomes, counterfactuals, frozen_graph,
            reference_name, other_name,
            geometries[reference_name], geometries[other_name])
        marine_relations[relation] = {
            "actual_native_inputs": actual_relation,
            "counterfactuals": variant_relations,
            "frozen_initial_graph_native_inputs": frozen_relation,
            "factorial_signed_effects_and_pattern_overlap": factorial_effects,
            "classification": _counterfactual_classification(
                actual_relation, variant_relations, frozen_relation,
                factorial_effects, fixed_both_effective_source_identity,
                common_mouth[relation], component_reports[relation]),
        }

    sediment_sets, discharge_sets = _material_coordinate_sets(
        solved, geometries)
    cell_ledger = _cell_ledger(
        solved, observers, land_replays, windows,
        sediment_sets, discharge_sets)
    first_divergence = {
        relation: _first_divergence(component_reports[relation])
        for relation, _, _ in RELATIONS
    }
    lowstand_threshold_relations = {
        relation: {
            "fixed_delivery_core": _lowstand_threshold_relation(
                reference_name, other_name, observers,
                geometries[reference_name], geometries[other_name]),
            "three_window_common_support": _lowstand_threshold_relation(
                reference_name, other_name, observers,
                common_geometries[reference_name],
                common_geometries[other_name]),
        }
        for relation, reference_name, other_name in RELATIONS
    }
    discharge_flip_evidence = {
        relation: _post_sediment_flip_evidence(
            reference_name, other_name, solved, observers, geometries)
        for relation, reference_name, other_name in RELATIONS
    }

    canonical_source_summaries = {
        name: _global_sparse_summary(
            _embed_rect(canonical_source_common, common_rect, windows[name]),
            windows[name])
        for name in WINDOW_ORDER
    }
    canonical_hashes = {
        value["global_sparse_coordinate_value_sha256"]
        for value in canonical_source_summaries.values()
    }
    canonical_totals = {
        value["sum_m_cells"] for value in canonical_source_summaries.values()
    }
    canonical_identity = (
        len(canonical_hashes) == 1 and len(canonical_totals) == 1)
    large_source_total = float(large_marine.mouth_flux_m_cells.sum())
    canonical_source_total = float(canonical_source_common.sum())

    window_reports = {
        name: _window_capture_report(
            name, observers[name], land_replays[name], windows[name])
        for name in WINDOW_ORDER
    }
    actual_validations = {
        name: _marine_outcome_validation(actual_outcomes[name])
        for name in WINDOW_ORDER
    }
    counterfactual_validations = {
        variant: {
            name: _marine_outcome_validation(counterfactuals[variant][name])
            for name in WINDOW_ORDER
        }
        for variant in COUNTERFACTUAL_VARIANTS
    }
    frozen_validations = {
        name: _marine_outcome_validation(frozen_graph[name])
        for name in WINDOW_ORDER
    }
    counterfactual_reports = {
        variant: {
            name: {
                **_marine_outcome_summary(
                    counterfactuals[variant][name], windows[name]),
                "validation": counterfactual_validations[variant][name],
            }
            for name in WINDOW_ORDER
        }
        for variant in COUNTERFACTUAL_VARIANTS
    }
    frozen_reports = {
        name: {
            **_marine_outcome_summary(frozen_graph[name], windows[name]),
            "validation": frozen_validations[name],
        }
        for name in WINDOW_ORDER
    }

    accumulation_capture_validation = {}
    for name in WINDOW_ORDER:
        observer = observers[name]
        spacing = observer.marines[0].process_spacing_km
        sediment_area = observer.sediments[0].area_km2
        final_linearized = np.expm1(solved[name]["discharge_log"]).ravel()
        accumulation_capture_validation[name] = {
            "nested_d8_raw_matches_flow_raw": all(
                np.array_equal(observer.d8s[index].raw_area8,
                               observer.flows[index].raw_area8)
                for index in range(erosion_engine.N_STEPS + 1)),
            "nested_d8_effective_matches_flow_effective": all(
                np.array_equal(observer.d8s[index].effective_area8,
                               observer.flows[index].effective_area8)
                for index in range(erosion_engine.N_STEPS + 1)),
            "pre_sediment_effective_mfd_matches_consumed_area_km2": bool(
                np.array_equal(
                    observer.flows[2].effective_area * spacing * spacing,
                    sediment_area)),
            "post_sediment_effective_d8_matches_delivered_discharge": bool(
                np.allclose(
                    observer.d8s[3].effective_area8, final_linearized,
                    rtol=1e-14, atol=1e-14)),
        }

    checks = {
        "prior_artifact_digests_matched": not bool(mismatched),
        "historical_physical_headline_reproduced": historical_reproduction[
            "all_headline_fields_exactly_reproduced"],
        "instrumented_marine_inputs_not_mutated": all(
            observer.marines[0].bed_unchanged_by_call
            and observer.marines[0].mouth_flux_unchanged_by_call
            for observer in observers.values()),
        "land_handoff_replays_exact": all(
            all(value for key, value in land.validation.items()
                if key.endswith("_exact"))
            for land in land_replays.values()),
        "land_origin_partitions_within_numeric_threshold": all(
            land.validation[
                "provenance_mouth_reconstruction_within_numeric_threshold"]
            and land.validation[
                "provenance_land_deposit_reconstruction_within_numeric_threshold"]
            for land in land_replays.values()),
        "raw_and_effective_accumulation_capture_validated": all(
            all(value.values())
            for value in accumulation_capture_validation.values()),
        "fill_replays_exact": all(
            _fill_replay(observer.fills[index])["output_array_exact"]
            for observer in observers.values()
            for index in range(len(ROUTING_STAGES))),
        "canonical_global_source_identity": canonical_identity,
        "fixed_both_effective_global_source_identity": (
            fixed_both_effective_source_identity),
        "actual_marine_outcomes_finite_and_closed": all(
            value["passed"] for value in actual_validations.values()),
        "counterfactual_marine_outcomes_finite_and_closed": all(
            value["passed"]
            for variant in counterfactual_validations.values()
            for value in variant.values()),
        "frozen_graph_outcomes_finite_and_closed": all(
            value["passed"] for value in frozen_validations.values()),
        "factorial_signed_reconstructions_within_scaled_1e_minus_12": all(
            relation[
                "factorial_signed_effects_and_pattern_overlap"][
                    "signed_reconstruction"]["within_scaled_1e_minus_12"]
            for relation in marine_relations.values()),
        "direct_original_marine_call_count_fixed": (
            direct_call_count == EXPECTED_DIRECT_MARINE_CALLS),
        "frozen_graph_ablation_call_count_fixed": (
            len(frozen_graph) == EXPECTED_FROZEN_GRAPH_ABLATION_CALLS),
        "instrumented_engine_functions_restored": engine_functions_restored,
        "engine_default_unchanged_by_run": (
            inspect.signature(erosion_engine.run_erosion).parameters[
                "_localization_mode"].default == "legacy"),
    }
    if not all(checks.values()):
        raise AssertionError({"checks": checks})

    report = {
        "experiment": EXPERIMENT,
        "completed": True,
        "protocol_precommit_sha256": protocol_sha256,
        "source_fingerprint": fingerprint,
        "prior_artifacts": prior_links,
        "fixed": {
            "seed": SEED,
            "origin_xy_km": list(replay.PRIMARY_ORIGIN),
            "complete_effective_config": _effective_config(cfg),
            "windows": {name: list(value)
                        for name, value in windows.items()},
            "common_intersection_row_exclusive_row_column_exclusive_column": (
                list(common_rect)),
            "localization_mode": "physical_outlets",
            "structural_builds": 1,
            "coarse_elevation_builds": 1,
            "erosion_calls": 3,
            "direct_original_marine_counterfactual_calls": direct_call_count,
            "private_frozen_graph_ablation_calls": len(frozen_graph),
            "retries": 0,
        },
        "wall_times_instrumented_s": wall_times,
        "historical_headline_reproduction": historical_reproduction,
        "window_captures": window_reports,
        "accumulation_capture_validation": accumulation_capture_validation,
        "component_relations": component_reports,
        "first_divergent_stage": first_divergence,
        "all_stage_lowstand_threshold_relations": (
            lowstand_threshold_relations),
        "common_support_mouth_provenance": {
            relation: {
                "comparisons": common_mouth[relation],
                "largest_numeric_difference_examples": (
                    mouth_examples[relation]),
            }
            for relation, _, _ in RELATIONS
        },
        "canonical_control": {
            "common_source_from_large": _array_summary(
                canonical_source_common),
            "common_bed_from_large": _array_summary(
                canonical_bed_common),
            "canonical_source_by_embedding": canonical_source_summaries,
            "global_sparse_identity_across_embeddings": canonical_identity,
            "fixed_both_effective_source_by_embedding": (
                fixed_both_effective_source_summaries),
            "fixed_both_effective_global_sparse_identity": (
                fixed_both_effective_source_identity),
            "large_native_source_total_m_cells": large_source_total,
            "canonical_common_source_total_m_cells": canonical_source_total,
            "large_source_discarded_outside_common_m_cells": (
                large_source_total - canonical_source_total),
        },
        "counterfactual_outcomes": counterfactual_reports,
        "frozen_graph_ablation_outcomes": frozen_reports,
        "marine_relations_and_classification": marine_relations,
        "material_cell_identity": {
            "sediment": _set_relation_report(sediment_sets),
            "discharge": _set_relation_report(discharge_sets),
        },
        "affected_cell_ledger": cell_ledger,
        "post_sediment_discharge_flip_evidence": discharge_flip_evidence,
        "checks": checks,
        "decision_boundary": {
            "diagnostic_only": True,
            "correction_implemented": False,
            "engine_or_default_changed": False,
            "promotion_assessed": False,
            "one_seed_scope": True,
            "marine_only_controls_do_not_reproduce_final_discharge_tail": True,
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
        "checks": checks,
        "relation_classifications": {
            relation: value["classification"]
            for relation, value in marine_relations.items()
        },
        "discharge_classifications": {
            relation: value["classification"]
            for relation, value in discharge_flip_evidence.items()
        },
        "elapsed_s": report["elapsed_s"],
    }


def _self_check() -> dict:
    windows = {
        "small": (2, 3, 3),
        "large": (1, 2, 5),
        "shifted": (0, 3, 5),
    }
    common = _window_intersection(windows.values())
    source = np.arange(9, dtype=np.float64).reshape(3, 3)
    embedded = {
        name: _embed_rect(source, common, window)
        for name, window in windows.items()
    }
    extraction_ok = all(np.array_equal(
        _extract_rect(value, windows[name], common), source)
        for name, value in embedded.items())
    sparse_hashes = {
        _global_sparse_summary(embedded[name], windows[name])[
            "global_sparse_coordinate_value_sha256"]
        for name in windows
    }

    z = np.full((3, 3), -100.0)
    z[1, 1] = 10.0
    erosion = np.zeros((3, 3))
    erosion[1, 1] = 100.0
    receiver = np.arange(9, dtype=np.int64)
    receiver[4] = 5
    batches = (np.arange(9, dtype=np.int64),)
    area = np.ones(9)
    marine_input = {}

    def zero_marine(bed, mouth, base_level, length_km, spacing_km):
        marine_input["bed"] = np.asarray(bed).copy()
        marine_input["mouth"] = np.asarray(mouth).copy()
        residual = float(np.maximum(mouth, 0.0).sum())
        return (np.zeros_like(bed), 0.0, residual, {
            "source_m_cells": residual,
            "deposited_m_cells": 0.0,
            "boundary_export_m_cells": 0.0,
            "far_field_export_m_cells": 0.0,
            "terminal_residual_m_cells": residual,
            "closure_m_cells": 0.0,
        })

    routed = erosion_engine._route_sediment_lowstand(
        z, erosion, receiver, batches, area,
        -80.0, 180.0, 20.0, _marine_transport=zero_marine)
    sediment = SedimentCapture(
        input_surface_m=z.copy(), erosion_source_m=erosion.copy(),
        receiver=receiver.copy(), batches=batches, area_km2=area.copy(),
        base_level_m=-80.0, deposition_length_km=180.0,
        process_spacing_km=20.0, output_surface_m=routed[0],
        total_deposit_m=routed[1], combined_export_m_cells=routed[2],
        terminal_residual_m_cells=routed[3], diagnostics=routed[4])
    marine = MarineCapture(
        pre_marine_bed_m=marine_input["bed"],
        mouth_flux_m_cells=marine_input["mouth"],
        base_level_m=-80.0, deposition_length_km=180.0,
        process_spacing_km=20.0, marine_deposit_m=np.zeros_like(z),
        combined_export_m_cells=0.0,
        terminal_residual_m_cells=float(marine_input["mouth"].sum()),
        diagnostics=routed[4]["marine"],
        bed_unchanged_by_call=True, mouth_flux_unchanged_by_call=True)
    land = _replay_land_sediment(
        sediment, (0, 0, 3), (0, 3, 0, 3), marine)

    frozen_bed = np.full((5, 5), -100.0)
    frozen_source = np.zeros((5, 5))
    frozen_source[2, 2] = 10.0
    frozen = _run_frozen_graph_marine(
        frozen_bed, frozen_source, -80.0, 180.0, 20.0)
    closure = abs(float(frozen.diagnostics["closure_m_cells"]))
    numeric = _numeric_comparison(
        np.zeros((2, 2)), np.array([[0.0, 0.1], [0.0, 0.0]]),
        absolute_threshold=TERRAIN_MATERIAL_THRESHOLD_M)

    effect_geometry = stage_diagnostic.CoreGeometry.explicit(
        "effect", (0, 0, 2), 1.0, np.arange(2), np.arange(2))

    def effect_outcome(deposit):
        deposit = np.asarray(deposit, np.float64)
        zeros = np.zeros_like(deposit)
        return MarineOutcome(
            bed_m=zeros.copy(), requested_source_m_cells=zeros.copy(),
            effective_source_m_cells=zeros.copy(), deposit_m=deposit.copy(),
            combined_export_m_cells=0.0, terminal_residual_m_cells=0.0,
            diagnostics={})

    zero2 = np.zeros((2, 2))
    actual_other = zero2.copy()
    actual_other[0, 0] = 0.06
    subeffect = zero2.copy()
    subeffect[0, 0] = 0.03
    actual_effect = {
        "reference": effect_outcome(zero2),
        "other": effect_outcome(actual_other),
    }
    counter_effect = {
        "fixed_source_native_bed": {
            "reference": effect_outcome(zero2),
            "other": effect_outcome(subeffect),
        },
        "native_source_fixed_common_bed": {
            "reference": effect_outcome(zero2),
            "other": effect_outcome(subeffect),
        },
        "fixed_source_fixed_common_bed": {
            "reference": effect_outcome(zero2),
            "other": effect_outcome(zero2),
        },
    }
    frozen_effect = {
        "reference": effect_outcome(zero2),
        "other": effect_outcome(zero2),
    }
    effect_report = _factorial_effect_report(
        actual_effect, counter_effect, frozen_effect,
        "reference", "other", effect_geometry, effect_geometry)

    capture_geometry = stage_diagnostic.CoreGeometry.explicit(
        "capture", (0, 0, 3), 1.0, np.arange(3), np.arange(3))
    capture_observer = WindowObserver("capture", (0, 0, 3), capture_geometry)
    capture_outlet = np.zeros((3, 3), bool)
    capture_outlet[1, 1] = True
    capture_observer.fills.append(FillCapture(
        np.zeros((3, 3)), capture_outlet, np.zeros((3, 3))))
    capture_receiver = np.arange(9, dtype=np.int64)
    capture_batches = (np.arange(9, dtype=np.int64),)
    capture_targets = np.tile(capture_receiver, (8, 1))
    capture_weights = np.zeros((8, 9))
    capture_runoff = np.ones((3, 3))
    capture_observer.record_d8(
        capture_receiver, capture_batches, capture_runoff, np.ones(9))
    capture_observer.record_flow(
        capture_receiver, capture_batches, capture_targets, capture_weights,
        capture_runoff, (np.ones(9), np.ones(9)))

    no_room_bed = np.full((5, 5), -80.0)
    no_room_source = np.zeros((5, 5))
    no_room_source[2, 2] = 10.0
    dynamic_no_room = _run_direct_marine(
        no_room_bed, no_room_source, -80.0, 180.0, 20.0)
    frozen_no_room = _run_frozen_graph_marine(
        no_room_bed, no_room_source, -80.0, 180.0, 20.0)

    threshold_reference = np.zeros((2, 2))
    threshold_other = np.zeros((2, 2))
    threshold_reference[0, 0] = 5e-10
    threshold_other[0, 0] = -5e-10
    threshold_report = _threshold_xor_report(
        threshold_reference <= 0.0, threshold_other <= 0.0,
        threshold_reference, threshold_other, 0.0, 0.0,
        effect_geometry, effect_geometry)
    checks = {
        "common_intersection": common == (2, 5, 3, 6),
        "global_embed_extract": extraction_ok,
        "global_sparse_hash_independent_of_embedding_shape": (
            len(sparse_hashes) == 1),
        "land_replay_exact": all(
            value for key, value in land.validation.items()
            if key.endswith("_exact")),
        "land_replay_mouth_attribution": land.validation[
            "provenance_mouth_reconstruction_within_numeric_threshold"],
        "frozen_graph_mass_closure": closure <= 1e-12,
        "material_threshold_strictly_detected": (
            numeric["material_changed_cells"] == 1),
        "factorial_reconstruction": effect_report[
            "signed_reconstruction"]["within_scaled_1e_minus_12"],
        "additive_subthreshold_effects_not_mislabeled_interaction": (
            effect_report["signed_factorial_effects"][
                "source_bed_interaction_contrast"][
                    "material_changed_cells"] == 0
            and effect_report["actual_relation_delta"][
                "material_changed_cells"] == 1),
        "effective_accumulation_zeroing": (
            capture_observer.d8s[0].raw_area8[4] == 1.0
            and capture_observer.d8s[0].effective_area8[4] == 0.0
            and capture_observer.flows[0].raw_area[4] == 1.0
            and capture_observer.flows[0].effective_area[4] == 0.0),
        "frozen_equals_dynamic_when_no_aggradation_possible": (
            np.array_equal(dynamic_no_room.deposit_m,
                           frozen_no_room.deposit_m)
            and dynamic_no_room.combined_export_m_cells
            == frozen_no_room.combined_export_m_cells
            and dynamic_no_room.terminal_residual_m_cells
            == frozen_no_room.terminal_residual_m_cells),
        "near_threshold_mask_xor_detected": (
            threshold_report[
                "near_threshold_xor_cells_both_margins_within_1e_minus_9_m"]
            == 1),
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
        default=Path("out") / "physical_outlet_causal_seed11_v1")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    result = _self_check() if args.self_check else _run(args.out)
    print(json.dumps(
        result, indent=2, allow_nan=False, default=_json_default))


if __name__ == "__main__":
    main()
