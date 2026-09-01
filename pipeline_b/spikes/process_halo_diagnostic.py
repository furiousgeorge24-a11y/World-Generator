"""Stage-resolved process-halo diagnostic for the frozen seed-11 replay.

This is a private, one-shot evidence script.  It does not add a control,
change an engine branch, move the frozen crop, or alter either historical
atlas spike.  One 40-km structural atlas is shared by two ordered trios:

1. the historical ``legacy`` small/large/shifted process windows; then
2. the existing ``lowstand_outlets`` localization branch as a control.

Each window is executed exactly once.  Temporary pass-through wrappers
observe the existing erosion functions and are restored in ``finally``.
Raw accumulation outputs are copied over the fixed core at hook return;
references retained until finalization are explicitly the effective state
seen by downstream consumers (including lowstand marine zeroing).  The
wrappers do not modify arguments or return values.  Their timings are
diagnostic because even thin Python interposition adds a small amount of
overhead.

Run from ``pipeline_b`` with::

    python -m spikes.process_halo_diagnostic --out out/process_halo_seed11_v1

The non-model mechanics self-check is::

    python -m spikes.process_halo_diagnostic --self-check
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from engine import elevation as elevation_engine
from engine import erosion as erosion_engine
from spikes import atlas_replay as replay


EXPERIMENT = "seed11-process-halo-stage-diagnostic-v1"
SEED = 11
CONTINENTAL_BUDGET = 0.65
WINDOW_ORDER = ("small", "large", "shifted")
MODE_ORDER = ("legacy", "lowstand_outlets")
BOUNDARY_BIN_EDGES_KM = (
    0.0, 400.0, 800.0, 1200.0, 1600.0,
    2000.0, 2400.0, 2800.0, 3200.0, float("inf"),
)
ABSOLUTE_DIFFERENCE_THRESHOLDS = (1e-9, 0.05, 0.5)
TERRAIN_MATERIAL_THRESHOLD_M = 0.05
FILL_DISTANCE_MATERIAL_THRESHOLD = 1e-9
HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD = 0.005
MAX_FIRST_DIFFERENCES = 12

HISTORICAL_REPORT_RELATIVE = Path(
    "out/atlas_replay_seed11_065_v2/report.json")
HISTORICAL_REPORT_SHA256 = (
    "9fcb7741f42b5399ead3931c93164ff9dd50f87f2c57c4c448b31bf18d82c12d")

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
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint() -> dict:
    files = {
        relative: _sha256_file(ROOT / relative)
        for relative in SOURCE_FILES
    }
    combined = hashlib.sha256()
    for relative, digest in sorted(files.items()):
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {"combined_sha256": combined.hexdigest(), "files": files}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _json_clone(value):
    return json.loads(json.dumps(
        value, allow_nan=False, default=_json_default))


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
            raise ValueError(f"diagnostic output directory must be empty: {path}")
    else:
        path.mkdir(parents=True)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.view(np.uint8).tobytes())
    return digest.hexdigest()


def _array_summary(array: np.ndarray) -> dict:
    array = np.asarray(array)
    summary = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _array_sha256(array),
    }
    if array.dtype == np.dtype(bool):
        summary.update({
            "true_count": int(np.count_nonzero(array)),
            "false_count": int(array.size - np.count_nonzero(array)),
        })
        return summary
    if np.issubdtype(array.dtype, np.number):
        numeric = np.asarray(array, np.float64)
        finite = np.isfinite(numeric)
        summary["finite_count"] = int(np.count_nonzero(finite))
        summary["nonfinite_count"] = int(array.size - np.count_nonzero(finite))
        if finite.any():
            values = numeric[finite]
            summary.update({
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            })
    return summary


@dataclass
class CoreGeometry:
    """Absolute process-grid geometry for one local solve window."""

    window_name: str
    window: tuple[int, int, int]
    e_km: float
    global_rows: np.ndarray
    global_columns: np.ndarray

    @classmethod
    def fixed(cls, window_name: str, window: tuple[int, int, int],
              structure) -> "CoreGeometry":
        n_world = int(round(structure.world_km / erosion_engine.E_KM))
        e_km = structure.world_km / n_world
        q = (np.arange(n_world) + 0.5) * e_km
        x0, y0 = replay.PRIMARY_ORIGIN
        rows = np.flatnonzero(
            (q >= y0 - replay.CORE_COLLAR_KM)
            & (q < y0 + replay.FRAME_KM + replay.CORE_COLLAR_KM))
        columns = np.flatnonzero(
            (q >= x0 - replay.CORE_COLLAR_KM)
            & (q < x0 + replay.FRAME_KM + replay.CORE_COLLAR_KM))
        geometry = cls(
            window_name=window_name,
            window=tuple(int(value) for value in window),
            e_km=float(e_km),
            global_rows=rows.astype(np.int64),
            global_columns=columns.astype(np.int64),
        )
        geometry._validate()
        return geometry

    @classmethod
    def explicit(cls, window_name: str, window: tuple[int, int, int],
                 e_km: float, rows, columns) -> "CoreGeometry":
        geometry = cls(
            window_name=window_name,
            window=window,
            e_km=float(e_km),
            global_rows=np.asarray(rows, np.int64),
            global_columns=np.asarray(columns, np.int64),
        )
        geometry._validate()
        return geometry

    def _validate(self) -> None:
        row0, column0, side = self.window
        if side < 2:
            raise ValueError("process window must have at least two cells")
        if self.global_rows.size == 0 or self.global_columns.size == 0:
            raise ValueError("comparison core must be non-empty")
        if (self.global_rows.min() < row0
                or self.global_rows.max() >= row0 + side
                or self.global_columns.min() < column0
                or self.global_columns.max() >= column0 + side):
            raise ValueError(
                f"comparison core lies outside {self.window_name} window")

    @property
    def local_rows(self) -> np.ndarray:
        return self.global_rows - self.window[0]

    @property
    def local_columns(self) -> np.ndarray:
        return self.global_columns - self.window[1]

    @property
    def core_shape(self) -> tuple[int, int]:
        return (int(self.global_rows.size), int(self.global_columns.size))

    def extract_grid(self, value) -> np.ndarray:
        array = np.asarray(value)
        side = self.window[2]
        if array.ndim == 1:
            if array.size != side * side:
                raise ValueError(
                    f"cannot reshape {array.size} cells for {self.window_name}")
            array = array.reshape(side, side)
        if array.ndim < 2 or array.shape[:2] != (side, side):
            raise ValueError(
                f"stage array {array.shape} does not match "
                f"{self.window_name} window {(side, side)}")
        index = np.ix_(self.local_rows, self.local_columns)
        return np.asarray(array[index]).copy()

    def extract_direction_grid(self, value) -> np.ndarray:
        """Freeze direction-major ``(8, n)`` data as core ``(y, x, 8)``."""
        array = np.asarray(value)
        side = self.window[2]
        if array.shape == (8, side * side):
            array = array.reshape(8, side, side).transpose(1, 2, 0)
        elif array.shape == (8, side, side):
            array = array.transpose(1, 2, 0)
        if array.ndim != 3 or array.shape != (side, side, 8):
            raise ValueError(
                f"direction array {array.shape} does not match "
                f"{self.window_name} window {(side, side, 8)}")
        index = np.ix_(self.local_rows, self.local_columns)
        return np.asarray(array[index]).copy()

    def receiver_global_row_column(self, receiver) -> np.ndarray:
        receiver = np.asarray(receiver, np.int64).reshape(-1)
        side = self.window[2]
        if receiver.size != side * side:
            raise ValueError("receiver graph size does not match process window")
        local_y, local_x = np.meshgrid(
            self.local_rows, self.local_columns, indexing="ij")
        source = (local_y * side + local_x).reshape(-1)
        target = receiver[source]
        target_y, target_x = np.divmod(target, side)
        row0, column0, _ = self.window
        result = np.stack((
            target_y.reshape(self.core_shape) + row0,
            target_x.reshape(self.core_shape) + column0,
        ), axis=-1)
        return result.astype(np.int64, copy=False)

    def boundary_side_distances_km(self) -> np.ndarray:
        row0, column0, side = self.window
        rows, columns = np.meshgrid(
            self.global_rows, self.global_columns, indexing="ij")
        distance_cells = np.stack((
            rows + 0.5 - row0,
            row0 + side - (rows + 0.5),
            columns + 0.5 - column0,
            column0 + side - (columns + 0.5),
        ), axis=-1)
        return distance_cells * self.e_km

    def boundary_distance_km(self) -> np.ndarray:
        return self.boundary_side_distances_km().min(axis=-1)

    def delivered_frame_mask(self) -> np.ndarray:
        x0, y0 = replay.PRIMARY_ORIGIN
        rows, columns = np.meshgrid(
            self.global_rows, self.global_columns, indexing="ij")
        y_km = (rows + 0.5) * self.e_km
        x_km = (columns + 0.5) * self.e_km
        return ((y_km >= y0) & (y_km < y0 + replay.FRAME_KM)
                & (x_km >= x0) & (x_km < x0 + replay.FRAME_KM))

    def global_coordinate_records(self, changed: np.ndarray,
                                  limit=MAX_FIRST_DIFFERENCES) -> list[dict]:
        records = []
        for local_y, local_x in np.argwhere(changed)[:limit]:
            global_y = int(self.global_rows[local_y])
            global_x = int(self.global_columns[local_x])
            records.append({
                "global_row_column": [global_y, global_x],
                "center_yx_km": [
                    (global_y + 0.5) * self.e_km,
                    (global_x + 0.5) * self.e_km,
                ],
            })
        return records

    def report(self) -> dict:
        distance = self.boundary_distance_km()
        return {
            "window_name": self.window_name,
            "process_window_row_column_side": list(self.window),
            "process_spacing_km": self.e_km,
            "comparison_core_shape": list(self.core_shape),
            "comparison_core_global_row_range_inclusive": [
                int(self.global_rows[0]), int(self.global_rows[-1])],
            "comparison_core_global_column_range_inclusive": [
                int(self.global_columns[0]), int(self.global_columns[-1])],
            "comparison_core_distance_to_window_boundary_km": {
                "min": float(distance.min()),
                "max": float(distance.max()),
            },
        }


@dataclass
class _RawArray:
    value: Any
    kind: str = "grid"


@dataclass
class FinalizedCapture:
    mode: str
    window_name: str
    geometry: CoreGeometry
    arrays: dict[str, dict[str, np.ndarray]]
    report: dict


def _routing_stage(index: int) -> str:
    if index < erosion_engine.N_STEPS:
        return f"routing.solve_step_{index}"
    if index == erosion_engine.N_STEPS:
        return "routing.pre_sediment"
    if index == erosion_engine.N_STEPS + 1:
        return "routing.post_sediment"
    return f"routing.unexpected_{index}"


def _stage_order() -> list[str]:
    result = ["initial_surface", "runoff_distance"]
    for step in range(erosion_engine.N_STEPS):
        result.extend((
            f"routing.solve_step_{step}",
            f"solve.step_{step}.input",
            f"solve.step_{step}.post_stream_power",
            f"solve.step_{step}.post_creep",
        ))
    result.extend((
        "routing.pre_sediment",
        "sediment.input",
        "sediment.output",
        "routing.post_sediment",
        "lakes.input",
        "lakes.output",
        "delivered",
    ))
    return result


class StageObserver:
    """Collect effective references and eager raw snapshots for one run."""

    def __init__(self, mode: str, geometry: CoreGeometry):
        self.mode = mode
        self.geometry = geometry
        self.raw: dict[str, dict[str, _RawArray | Any]] = {}
        self.call_counts: dict[str, int] = {}
        self.call_timings: list[dict] = []
        self.flow_accumulation_depth = 0

    def count(self, name: str) -> int:
        index = self.call_counts.get(name, 0)
        self.call_counts[name] = index + 1
        return index

    def timing(self, hook: str, stage: str, elapsed_s: float) -> None:
        self.call_timings.append({
            "hook": hook,
            "stage": stage,
            "original_call_wall_s": float(elapsed_s),
        })

    def _freeze(self, value, kind: str) -> np.ndarray:
        if kind == "receiver":
            return self.geometry.receiver_global_row_column(value).copy()
        if kind == "directions":
            return self.geometry.extract_direction_grid(value)
        if kind == "grid":
            return self.geometry.extract_grid(value)
        raise ValueError(f"unknown capture kind: {kind}")

    def array(self, stage: str, field: str, value,
              *, kind: str = "grid", snapshot_core: bool = False) -> None:
        fields = self.raw.setdefault(stage, {})
        if field in fields:
            raise AssertionError(f"duplicate captured field: {stage}.{field}")
        if snapshot_core:
            fields[field] = _RawArray(
                value=self._freeze(value, kind), kind="frozen_core")
        else:
            fields[field] = _RawArray(value=value, kind=kind)

    def scalar(self, stage: str, field: str, value) -> None:
        fields = self.raw.setdefault(stage, {})
        if field in fields:
            raise AssertionError(f"duplicate captured field: {stage}.{field}")
        fields[field] = _json_clone(value)

    def _expected_counts(self) -> dict[str, int]:
        route_count = erosion_engine.N_STEPS + 2
        mfd_count = erosion_engine.N_STEPS + 1
        return {
            "chamfer_km": 1,
            "fill_total": route_count,
            "fill_depressions": route_count if self.mode == "legacy" else 0,
            "fill_to_lowstand_outlets": (
                route_count if self.mode == "lowstand_outlets" else 0),
            "flow_accumulation": mfd_count,
            "flow_accumulation_d8_total": mfd_count + 1,
            "flow_accumulation_d8_nested": mfd_count,
            "flow_accumulation_d8_standalone": 1,
            "spl_implicit": erosion_engine.N_STEPS,
            "soil_creep": erosion_engine.N_STEPS,
            "route_sediment_total": 1,
            "route_sediment": 1 if self.mode == "legacy" else 0,
            "route_sediment_lowstand": (
                1 if self.mode == "lowstand_outlets" else 0),
            "balance_lakes": 1,
        }

    def _hook_integrity(self) -> dict:
        expected = self._expected_counts()
        actual = {name: int(self.call_counts.get(name, 0))
                  for name in expected}
        mismatches = {
            name: {"expected": expected[name], "actual": actual[name]}
            for name in expected if actual[name] != expected[name]
        }
        return {
            "passed": not mismatches,
            "expected": expected,
            "actual": actual,
            "mismatches": mismatches,
        }

    def _localization_limitations(self, result: dict) -> dict | None:
        if self.mode != "lowstand_outlets":
            return None
        diagnostics = result.get("_localization_diagnostics")
        sediment_input = self.raw["sediment.input"]["surface_m"]
        sediment_output = self.raw["sediment.output"]["deposit_m"]
        assert isinstance(sediment_input, _RawArray)
        assert isinstance(sediment_output, _RawArray)
        surface = np.asarray(sediment_input.value)
        deposit = np.asarray(sediment_output.value)
        base_level = -float(replay._atlas_config(
            CONTINENTAL_BUDGET).lowstand_drop)
        marine = surface <= base_level
        footprint = marine & (deposit > 0.0)
        saturated = footprint & (
            deposit >= erosion_engine.MAR_CAP - 1e-9)
        marine_deposit = float(deposit[marine].sum())
        saturated_deposit = float(deposit[saturated].sum())
        source = (0.0 if diagnostics is None else float(
            diagnostics["source_m_cells"]))
        far_field = (0.0 if diagnostics is None else float(
            diagnostics["marine"]["far_field_export_m_cells"]))
        return {
            "role": "same-structure control; not promotion evidence",
            "known_unresolved_defects": [
                "marine_fan_deposit_concentration",
                "far_field_sediment_export",
            ],
            "marine_deposit_footprint_cells": int(np.count_nonzero(footprint)),
            "saturated_marine_deposit_cells": int(np.count_nonzero(saturated)),
            "saturated_cell_fraction_of_marine_deposit_footprint": (
                0.0 if not footprint.any() else
                float(np.count_nonzero(saturated) / np.count_nonzero(footprint))),
            "saturated_deposit_fraction": (
                0.0 if marine_deposit <= 0.0 else
                saturated_deposit / marine_deposit),
            "far_field_export_fraction_of_total_source": (
                0.0 if source <= 0.0 else far_field / source),
            "localization_diagnostics": _json_clone(diagnostics),
        }

    def finalize(self, result: dict) -> FinalizedCapture:
        self.array("initial_surface", "surface_m", result["z0"])
        for field in ("z", "z0", "ero", "sed", "discharge_log",
                      "lake_depth", "lake_surf"):
            self.array("delivered", field, result[field])

        hook_integrity = self._hook_integrity()
        if not hook_integrity["passed"]:
            raise AssertionError(
                f"instrumentation call counts changed for {self.mode}/"
                f"{self.geometry.window_name}: "
                f"{hook_integrity['mismatches']}")

        limitations = self._localization_limitations(result)
        arrays: dict[str, dict[str, np.ndarray]] = {}
        stages_report = {}
        for stage, fields in self.raw.items():
            array_fields = {}
            scalar_fields = {}
            for field, captured in fields.items():
                if isinstance(captured, _RawArray):
                    if captured.kind == "frozen_core":
                        core = np.asarray(captured.value).copy()
                    else:
                        core = self._freeze(captured.value, captured.kind)
                    array_fields[field] = core
                else:
                    scalar_fields[field] = captured
            arrays[stage] = array_fields
            stages_report[stage] = {
                "arrays": {
                    field: _array_summary(value)
                    for field, value in array_fields.items()
                },
                "scalars": scalar_fields,
            }

        budget = _sediment_budget(result)
        report = {
            "geometry": self.geometry.report(),
            "flow_capture_semantics": {
                "raw_at_hook_return": (
                    "fixed-core copy made before caller normalization"),
                "effective_for_route_consumer": (
                    "reference frozen after route_graph normalization; "
                    "lowstand marine zeroing is intentionally included"),
                "receiver_coordinates": "global row/column after route edits",
                "mfd_targets": (
                    "not stored because they are fixed-neighbor geometry; "
                    "directional weights and global D8 receiver are stored"),
            },
            "hook_integrity": hook_integrity,
            "hook_original_call_timings": self.call_timings,
            "engine_aggregate_timings_instrumented_s": _json_clone(
                result["timings"]),
            "stage_evidence": stages_report,
            "sediment_budget": budget,
        }
        if limitations is not None:
            report["localization_control_limitations"] = limitations

        # Release the full-window intermediate references before the next
        # process run; only fixed-core copies remain in ``arrays``.
        self.raw.clear()
        return FinalizedCapture(
            mode=self.mode,
            window_name=self.geometry.window_name,
            geometry=self.geometry,
            arrays=arrays,
            report=report,
        )


class EngineInstrumentation(AbstractContextManager):
    """Temporarily interpose observation-only wrappers on erosion globals."""

    def __init__(self):
        self.active: StageObserver | None = None
        self.originals: list[tuple[object, str, Callable]] = []

    def _observer(self) -> StageObserver | None:
        return self.active

    def _patch(self, module, name: str, replacement: Callable) -> None:
        self.originals.append((module, name, getattr(module, name)))
        setattr(module, name, replacement)

    @staticmethod
    def _timed(original: Callable, *args, **kwargs):
        started = time.perf_counter()
        result = original(*args, **kwargs)
        return result, time.perf_counter() - started

    def __enter__(self):
        original_chamfer = elevation_engine._chamfer_km
        original_fill = erosion_engine.fill_depressions
        original_lowstand_fill = erosion_engine._fill_to_lowstand_outlets
        original_flow = erosion_engine.flow_accumulation
        original_flow_d8 = erosion_engine.flow_accumulation_d8
        original_spl = erosion_engine.spl_implicit
        original_creep = erosion_engine.soil_creep
        original_sediment = erosion_engine.route_sediment
        original_lowstand_sediment = erosion_engine._route_sediment_lowstand
        original_lakes = erosion_engine._balance_lakes

        def chamfer(source, ck):
            observer = self._observer()
            if observer is None:
                return original_chamfer(source, ck)
            observer.count("chamfer_km")
            result, elapsed = self._timed(original_chamfer, source, ck)
            observer.timing("_chamfer_km", "runoff_distance", elapsed)
            observer.array("runoff_distance", "initial_sea_mask", source)
            observer.array(
                "runoff_distance", "distance_to_initial_sea_km", result)
            return result

        def fill_depressions(surface, *args, **kwargs):
            observer = self._observer()
            if observer is None:
                return original_fill(surface, *args, **kwargs)
            index = observer.count("fill_total")
            observer.count("fill_depressions")
            stage = _routing_stage(index)
            result, elapsed = self._timed(
                original_fill, surface, *args, **kwargs)
            observer.timing("fill_depressions", stage, elapsed)
            observer.array(stage, "filled_surface_m", result)
            return result

        def fill_to_lowstand(surface, outlet_mask, *args, **kwargs):
            observer = self._observer()
            if observer is None:
                return original_lowstand_fill(
                    surface, outlet_mask, *args, **kwargs)
            index = observer.count("fill_total")
            observer.count("fill_to_lowstand_outlets")
            stage = _routing_stage(index)
            result, elapsed = self._timed(
                original_lowstand_fill, surface, outlet_mask,
                *args, **kwargs)
            observer.timing("_fill_to_lowstand_outlets", stage, elapsed)
            observer.array(stage, "filled_surface_m", result)
            observer.array(stage, "lowstand_outlet_mask", outlet_mask)
            return result

        def flow_accumulation(rcv, batches, n, targets, weights,
                              runoff=None):
            observer = self._observer()
            if observer is None:
                return original_flow(
                    rcv, batches, n, targets, weights, runoff)
            index = observer.count("flow_accumulation")
            stage = _routing_stage(index)
            observer.flow_accumulation_depth += 1
            try:
                result, elapsed = self._timed(
                    original_flow, rcv, batches, n, targets, weights,
                    runoff)
            finally:
                observer.flow_accumulation_depth -= 1
            observer.timing("flow_accumulation", stage, elapsed)
            observer.array(
                stage, "receiver_global_row_column_effective", rcv,
                kind="receiver", snapshot_core=True)
            observer.array(
                stage, "mfd_weights_directional_effective", weights,
                kind="directions", snapshot_core=True)
            if index == 0:
                if runoff is None:
                    observer.scalar(
                        stage, "first_route_runoff_effective_input",
                        "implicit_uniform_one")
                else:
                    observer.array(
                        stage, "first_route_runoff_effective_input", runoff,
                        snapshot_core=True)
            observer.array(
                stage, "mfd_accumulation_raw_at_hook_return", result[0],
                snapshot_core=True)
            observer.array(
                stage, "d8_accumulation_raw_at_hook_return", result[1],
                snapshot_core=True)
            observer.array(
                stage, "mfd_accumulation_effective_for_route_consumer",
                result[0])
            observer.array(
                stage, "d8_accumulation_effective_for_route_consumer",
                result[1])
            return result

        def flow_accumulation_d8(rcv, batches, n, runoff=None):
            observer = self._observer()
            if observer is None:
                return original_flow_d8(rcv, batches, n, runoff)
            observer.count("flow_accumulation_d8_total")
            nested = observer.flow_accumulation_depth > 0
            count_name = (
                "flow_accumulation_d8_nested" if nested
                else "flow_accumulation_d8_standalone")
            index = observer.count(count_name)
            result, elapsed = self._timed(
                original_flow_d8, rcv, batches, n, runoff)
            stage = (_routing_stage(index) if nested
                     else "routing.post_sediment")
            observer.timing("flow_accumulation_d8", stage, elapsed)
            if not nested:
                observer.array(
                    stage, "receiver_global_row_column_effective", rcv,
                    kind="receiver", snapshot_core=True)
                if runoff is None:
                    observer.scalar(
                        stage, "final_route_runoff_effective_input",
                        "implicit_uniform_one")
                else:
                    observer.array(
                        stage, "final_route_runoff_effective_input", runoff,
                        snapshot_core=True)
                observer.array(
                    stage, "d8_accumulation_raw_at_hook_return", result,
                    snapshot_core=True)
                observer.array(
                    stage, "d8_accumulation_effective_for_route_consumer",
                    result)
            return result

        def spl_implicit(z, uplift, erodibility, rcv, batches, area_km2,
                         dt_myr, dx_km, base, edge_len_km=None):
            observer = self._observer()
            if observer is None:
                return original_spl(
                    z, uplift, erodibility, rcv, batches, area_km2,
                    dt_myr, dx_km, base, edge_len_km)
            index = observer.count("spl_implicit")
            input_stage = f"solve.step_{index}.input"
            output_stage = f"solve.step_{index}.post_stream_power"
            if index == 0:
                observer.array(
                    "initial_surface", "uplift_m_per_myr", uplift,
                    snapshot_core=True)
                observer.array(
                    "initial_surface", "erodibility", erodibility,
                    snapshot_core=True)
            observer.array(input_stage, "surface_m", z)
            observer.array(
                input_stage, "area_km2_effective_for_stream_power",
                area_km2)
            observer.array(input_stage, "base_mask", base)
            result, elapsed = self._timed(
                original_spl, z, uplift, erodibility, rcv, batches,
                area_km2, dt_myr, dx_km, base, edge_len_km)
            observer.timing("spl_implicit", output_stage, elapsed)
            observer.array(output_stage, "surface_m", result[0])
            observer.array(output_stage, "cut_m", result[1])
            return result

        def soil_creep(z, diffusivity_km2_myr, dt_myr, dx_km, base_lvl):
            observer = self._observer()
            if observer is None:
                return original_creep(
                    z, diffusivity_km2_myr, dt_myr, dx_km, base_lvl)
            index = observer.count("soil_creep")
            stage = f"solve.step_{index}.post_creep"
            result, elapsed = self._timed(
                original_creep, z, diffusivity_km2_myr, dt_myr,
                dx_km, base_lvl)
            observer.timing("soil_creep", stage, elapsed)
            observer.array(stage, "surface_m", result)
            return result

        def route_sediment(z, ero, rcv, batches, area_km2, base_lvl,
                           length_km, dx_km, edge_len_km=None):
            observer = self._observer()
            if observer is None:
                return original_sediment(
                    z, ero, rcv, batches, area_km2, base_lvl,
                    length_km, dx_km, edge_len_km)
            observer.count("route_sediment_total")
            observer.count("route_sediment")
            observer.array("sediment.input", "surface_m", z)
            observer.array("sediment.input", "cumulative_cut_m", ero)
            observer.array(
                "sediment.input", "area_km2_effective_for_sediment",
                area_km2)
            result, elapsed = self._timed(
                original_sediment, z, ero, rcv, batches, area_km2,
                base_lvl, length_km, dx_km, edge_len_km)
            observer.timing("route_sediment", "sediment.output", elapsed)
            observer.array("sediment.output", "surface_m", result[0])
            observer.array("sediment.output", "deposit_m", result[1])
            observer.scalar(
                "sediment.output", "boundary_export_m_cells", result[2])
            observer.scalar(
                "sediment.output", "terminal_residual_m_cells", result[3])
            return result

        def route_sediment_lowstand(z, ero, rcv, batches, area_km2,
                                    base_lvl, length_km, dx_km):
            observer = self._observer()
            if observer is None:
                return original_lowstand_sediment(
                    z, ero, rcv, batches, area_km2, base_lvl,
                    length_km, dx_km)
            observer.count("route_sediment_total")
            observer.count("route_sediment_lowstand")
            observer.array("sediment.input", "surface_m", z)
            observer.array("sediment.input", "cumulative_cut_m", ero)
            observer.array(
                "sediment.input", "area_km2_effective_for_sediment",
                area_km2)
            result, elapsed = self._timed(
                original_lowstand_sediment, z, ero, rcv, batches,
                area_km2, base_lvl, length_km, dx_km)
            observer.timing(
                "_route_sediment_lowstand", "sediment.output", elapsed)
            observer.array("sediment.output", "surface_m", result[0])
            observer.array("sediment.output", "deposit_m", result[1])
            observer.scalar(
                "sediment.output", "boundary_export_m_cells", result[2])
            observer.scalar(
                "sediment.output", "terminal_residual_m_cells", result[3])
            observer.scalar(
                "sediment.output", "localization_diagnostics", result[4])
            return result

        def balance_lakes(z, filled, area8):
            observer = self._observer()
            if observer is None:
                return original_lakes(z, filled, area8)
            observer.count("balance_lakes")
            observer.array("lakes.input", "surface_m", z)
            observer.array("lakes.input", "filled_surface_m", filled)
            observer.array(
                "lakes.input", "d8_accumulation_effective_for_lakes",
                area8)
            result, elapsed = self._timed(
                original_lakes, z, filled, area8)
            observer.timing("_balance_lakes", "lakes.output", elapsed)
            observer.array("lakes.output", "lake_depth_m", result[0])
            observer.array("lakes.output", "lake_surface_m", result[1])
            return result

        self._patch(elevation_engine, "_chamfer_km", chamfer)
        self._patch(erosion_engine, "fill_depressions", fill_depressions)
        self._patch(
            erosion_engine, "_fill_to_lowstand_outlets", fill_to_lowstand)
        self._patch(
            erosion_engine, "flow_accumulation", flow_accumulation)
        self._patch(
            erosion_engine, "flow_accumulation_d8", flow_accumulation_d8)
        self._patch(erosion_engine, "spl_implicit", spl_implicit)
        self._patch(erosion_engine, "soil_creep", soil_creep)
        self._patch(erosion_engine, "route_sediment", route_sediment)
        self._patch(
            erosion_engine, "_route_sediment_lowstand",
            route_sediment_lowstand)
        self._patch(erosion_engine, "_balance_lakes", balance_lakes)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.active = None
        for module, name, original in reversed(self.originals):
            setattr(module, name, original)
        self.originals.clear()
        return False


def _sediment_budget(result: dict) -> dict:
    cell_area_m2 = (float(result["e_km"]) * 1000.0) ** 2
    source_m3 = float(np.maximum(result["ero"], 0.0).sum() * cell_area_m2)
    deposited_m3 = float(result["sed"].sum() * cell_area_m2)
    exported_m3 = float(result["sediment_export_m3"])
    terminal_m3 = float(result["sediment_terminal_residual_m3"])
    closure_m3 = source_m3 - deposited_m3 - exported_m3 - terminal_m3
    return {
        "source_m3": source_m3,
        "deposited_m3": deposited_m3,
        "exported_m3": exported_m3,
        "terminal_residual_m3": terminal_m3,
        "closure_m3": closure_m3,
        "relative_abs_closure": abs(closure_m3) / max(source_m3, 1.0),
    }


def _element_equal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    equal = left == right
    if (np.issubdtype(left.dtype, np.floating)
            and np.issubdtype(right.dtype, np.floating)):
        equal = equal | (np.isnan(left) & np.isnan(right))
    return equal


def _cell_changed(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    equal = _element_equal(left, right)
    if equal.ndim > 2:
        axes = tuple(range(2, equal.ndim))
        equal = np.all(equal, axis=axes)
    return ~equal


def _cell_any(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, bool)
    if mask.ndim > 2:
        mask = np.any(mask, axis=tuple(range(2, mask.ndim)))
    return mask


def _boundary_histogram(values: np.ndarray) -> list[dict]:
    records = []
    for lower, upper in zip(
            BOUNDARY_BIN_EDGES_KM[:-1], BOUNDARY_BIN_EDGES_KM[1:]):
        selected = (values >= lower) & (values < upper)
        records.append({
            "min_inclusive_km": lower,
            "max_exclusive_km": None if np.isinf(upper) else upper,
            "changed_cells": int(np.count_nonzero(selected)),
        })
    return records


def _distance_statistics(values: np.ndarray) -> dict:
    values = np.asarray(values, np.float64)
    if values.size == 0:
        return {name: None for name in ("min", "median", "p95", "p99", "max")}
    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(values.max()),
    }


def _ring_counts(values: np.ndarray, spacing_km: float) -> list[dict]:
    values = np.asarray(values, np.float64)
    if values.size == 0:
        return []
    # A boundary-cell centre is 0.5 spacing from the edge, hence ring 0.
    rings = np.floor(values / spacing_km).astype(np.int64)
    unique, counts = np.unique(rings, return_counts=True)
    return [
        {"boundary_ring_index": int(ring), "changed_cells": int(count)}
        for ring, count in zip(unique, counts)
    ]


def _geometry_distance_evidence(changed: np.ndarray,
                                geometry: CoreGeometry) -> dict:
    side_names = ("row_min", "row_max", "column_min", "column_max")
    side_distances = geometry.boundary_side_distances_km()
    distances = side_distances.min(axis=-1)[changed]
    nearest = np.argmin(side_distances, axis=-1)[changed]
    return {
        "name": geometry.window_name,
        "distance_definition": (
            "perpendicular geometric cell-centre distance; not flow-path "
            "distance"),
        "process_spacing_km": geometry.e_km,
        "distance_statistics_km": _distance_statistics(distances),
        "maximum_inward_penetration_km": (
            None if distances.size == 0 else float(distances.max())),
        "nearest_side_counts": {
            side: int(np.count_nonzero(nearest == index))
            for index, side in enumerate(side_names)
        },
        "nearest_side_tie_break_order": list(side_names),
        "process_cell_boundary_ring_counts": _ring_counts(
            distances, geometry.e_km),
        "supplementary_400km_bins": _boundary_histogram(distances),
    }


def _minimum_distance_evidence(changed: np.ndarray,
                               reference: CoreGeometry,
                               other: CoreGeometry) -> dict:
    reference_distance = reference.boundary_distance_km()
    other_distance = other.boundary_distance_km()
    minimum = np.minimum(reference_distance, other_distance)[changed]
    choose_reference = (
        reference_distance[changed] <= other_distance[changed])
    return {
        "distance_definition": (
            "minimum perpendicular geometric cell-centre distance to either "
            "tested process-window boundary; not flow-path distance"),
        "distance_statistics_km": _distance_statistics(minimum),
        "maximum_inward_penetration_km": (
            None if minimum.size == 0 else float(minimum.max())),
        "nearest_window_boundary_counts": {
            reference.window_name: int(np.count_nonzero(choose_reference)),
            other.window_name: int(np.count_nonzero(~choose_reference)),
        },
        "window_tie_break": reference.window_name,
        "process_cell_boundary_ring_counts": _ring_counts(
            minimum, reference.e_km),
        "supplementary_400km_bins": _boundary_histogram(minimum),
    }


def _changed_distance_evidence(changed: np.ndarray,
                               reference: CoreGeometry,
                               other: CoreGeometry) -> dict:
    delivered = reference.delivered_frame_mask()
    return {
        "changed_core_cells": int(np.count_nonzero(changed)),
        "changed_cells_by_output_region": {
            "delivered_frame": int(np.count_nonzero(changed & delivered)),
            "fixed_40km_collar": int(np.count_nonzero(changed & ~delivered)),
        },
        "reference_window": _geometry_distance_evidence(
            changed, reference),
        "other_window": _geometry_distance_evidence(changed, other),
        "minimum_distance_to_either_window_boundary": (
            _minimum_distance_evidence(changed, reference, other)),
    }


def _relative_difference(left: np.ndarray, right: np.ndarray,
                         *, logarithmic_discharge: bool) -> np.ndarray:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    result = np.full(left.shape, np.nan, np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if logarithmic_discharge:
        # ``discharge_log`` stores log1p(A8); materiality is measured after
        # restoring the consumer quantity, not as a raw log-space delta.
        linear_left = np.expm1(np.clip(left[finite], -700.0, 700.0))
        linear_right = np.expm1(np.clip(right[finite], -700.0, 700.0))
        difference = np.abs(linear_right - linear_left)
        scale = np.maximum(np.abs(linear_left), np.abs(linear_right))
        result[finite] = difference / np.maximum(
            scale, np.finfo(np.float64).tiny)
    else:
        difference = np.abs(right[finite] - left[finite])
        scale = np.maximum(np.abs(left[finite]), np.abs(right[finite]))
        result[finite] = difference / np.maximum(
            scale, np.finfo(np.float64).tiny)
    return result


def _material_policy(stage: str, field: str, dtype: np.dtype) -> dict:
    label = f"{stage}.{field}".lower()
    if dtype == np.dtype(bool) or "mask" in field or "receiver" in field:
        return {"metric": "exact_inequality", "threshold": 0.0}
    if "distance_to_initial_sea" in label or "filled_surface" in label:
        return {
            "metric": "absolute_difference",
            "threshold": FILL_DISTANCE_MATERIAL_THRESHOLD,
        }
    if any(term in label for term in (
            "accumulation", "area_km2", "discharge", "runoff")):
        return {
            "metric": (
                "multiplicative_difference_from_log" if
                "discharge_log" in label else "relative_difference"),
            "threshold": HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD,
        }
    if "weight" in label or "uplift" in label:
        return {
            "metric": "absolute_difference",
            "threshold": FILL_DISTANCE_MATERIAL_THRESHOLD,
        }
    if "erodibility" in label:
        return {
            "metric": "relative_difference",
            "threshold": HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD,
        }
    return {
        "metric": "absolute_difference",
        "threshold": TERRAIN_MATERIAL_THRESHOLD_M,
    }


def _material_components(left: np.ndarray, right: np.ndarray,
                         stage: str, field: str) -> tuple[np.ndarray, dict, np.ndarray]:
    exact_components = ~_element_equal(left, right)
    policy = _material_policy(stage, field, left.dtype)
    if policy["metric"] == "exact_inequality":
        relative = np.full(left.shape, np.nan, np.float64)
        return exact_components, policy, relative

    numeric_left = np.asarray(left, np.float64)
    numeric_right = np.asarray(right, np.float64)
    finite = np.isfinite(numeric_left) & np.isfinite(numeric_right)
    nonfinite_change = exact_components & ~finite
    if policy["metric"] == "absolute_difference":
        metric = np.abs(numeric_left - numeric_right)
        relative = _relative_difference(
            numeric_left, numeric_right, logarithmic_discharge=False)
    else:
        relative = _relative_difference(
            numeric_left, numeric_right,
            logarithmic_discharge=(
                policy["metric"] == "multiplicative_difference_from_log"))
        metric = relative
    material = (finite & (metric > policy["threshold"])) | nonfinite_change
    return material, policy, relative


def _field_comparison(left: np.ndarray, right: np.ndarray,
                      reference_geometry: CoreGeometry,
                      other_geometry: CoreGeometry,
                      reference_land: np.ndarray,
                      stage: str = "unspecified",
                      field: str = "unspecified") -> dict:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "array_exactly_equal": False,
            "array_materially_equal": False,
            "shape_or_dtype_mismatch": {
                "reference_shape": list(left.shape),
                "other_shape": list(right.shape),
                "reference_dtype": str(left.dtype),
                "other_dtype": str(right.dtype),
            },
        }
    exact_changed = _cell_changed(left, right)
    material_components, policy, relative = _material_components(
        left, right, stage, field)
    material_changed = _cell_any(material_components)
    exact_count = int(np.count_nonzero(exact_changed))
    material_count = int(np.count_nonzero(material_changed))
    report = {
        "array_exactly_equal": exact_count == 0,
        "array_materially_equal": material_count == 0,
        "material_policy": policy,
        "exact_changed_core_cells": exact_count,
        "material_changed_core_cells": material_count,
        "exact_changed_core_fraction": exact_count / exact_changed.size,
        "material_changed_core_fraction": (
            material_count / material_changed.size),
        "exact_changed_by_initial_surface_region": {
            "land": int(np.count_nonzero(exact_changed & reference_land)),
            "ocean": int(np.count_nonzero(exact_changed & ~reference_land)),
        },
        "material_changed_by_initial_surface_region": {
            "land": int(np.count_nonzero(material_changed & reference_land)),
            "ocean": int(np.count_nonzero(
                material_changed & ~reference_land)),
        },
        "first_exact_changed_coordinates": (
            reference_geometry.global_coordinate_records(exact_changed)),
        "first_material_changed_coordinates": (
            reference_geometry.global_coordinate_records(material_changed)),
        "exact_changed_cell_boundary_evidence": _changed_distance_evidence(
            exact_changed, reference_geometry, other_geometry),
        "material_changed_cell_boundary_evidence": (
            _changed_distance_evidence(
                material_changed, reference_geometry, other_geometry)),
    }
    if (np.issubdtype(left.dtype, np.number)
            and left.dtype != np.dtype(bool)):
        difference = np.abs(np.asarray(left, np.float64)
                            - np.asarray(right, np.float64))
        finite = np.isfinite(difference)
        if finite.any():
            values = difference[finite]
            threshold_counts = {}
            for threshold in ABSOLUTE_DIFFERENCE_THRESHOLDS:
                threshold_mask = _cell_any(finite & (difference > threshold))
                threshold_counts[f"greater_than_{threshold:g}"] = int(
                    np.count_nonzero(threshold_mask))
            finite_relative = relative[np.isfinite(relative)]
            report["numeric_difference"] = {
                "max": float(values.max(initial=0.0)),
                "p99": float(np.percentile(values, 99.0)),
                "mean": float(values.mean()),
                "core_cell_counts_by_absolute_difference": threshold_counts,
                "relative_max": (
                    None if finite_relative.size == 0 else
                    float(finite_relative.max(initial=0.0))),
                "relative_definition": (
                    "relative difference after expm1(log1p discharge)" if
                    "discharge_log" in f"{stage}.{field}".lower() else
                    "abs(delta) / max(abs(reference), abs(other))"),
            }
    return report


def _field_class(stage: str, field: str) -> str:
    hydrology_terms = (
        "runoff", "routing", "receiver", "accumulation", "area",
        "distance_to_initial_sea", "initial_sea_mask", "discharge",
        "filled_surface", "weight", "lake",
    )
    label = f"{stage}.{field}"
    return "hydrology" if any(term in label for term in hydrology_terms) \
        else "terrain"


def _causal_field_order(stage: str, fields: set[str]) -> list[str]:
    if not stage.startswith("routing."):
        return sorted(fields)
    priority_terms = (
        "lowstand_outlet_mask",
        "filled_surface_m",
        "receiver_global_row_column_effective",
        "mfd_weights_directional_effective",
        "runoff_effective_input",
        "accumulation_raw_at_hook_return",
        "accumulation_effective_for_route_consumer",
    )

    def key(field: str) -> tuple[int, str]:
        for index, term in enumerate(priority_terms):
            if term in field:
                return index, field
        return len(priority_terms), field

    return sorted(fields, key=key)


def _compare_captures(reference: FinalizedCapture,
                      other: FinalizedCapture) -> dict:
    if not (np.array_equal(reference.geometry.global_rows,
                           other.geometry.global_rows)
            and np.array_equal(reference.geometry.global_columns,
                               other.geometry.global_columns)):
        raise AssertionError("stage captures do not cover the same core")
    reference_land = (
        reference.arrays["initial_surface"]["surface_m"] >= 0.0)
    stages = {}
    ordered = _stage_order()
    all_names = list(dict.fromkeys(
        ordered + sorted(set(reference.arrays) | set(other.arrays))))
    earliest_exact = None
    earliest_exact_hydrology = None
    earliest_exact_terrain = None
    earliest_material = None
    earliest_material_hydrology = None
    earliest_material_terrain = None
    all_exact = True
    all_material = True
    for stage in all_names:
        left_fields = reference.arrays.get(stage, {})
        right_fields = other.arrays.get(stage, {})
        field_names = _causal_field_order(
            stage, set(left_fields) | set(right_fields))
        fields = {}
        exact_divergent_fields = []
        material_divergent_fields = []
        for field in field_names:
            if field not in left_fields or field not in right_fields:
                comparison = {
                    "array_exactly_equal": False,
                    "array_materially_equal": False,
                    "missing_from": (
                        "reference" if field not in left_fields else "other"),
                }
            else:
                comparison = _field_comparison(
                    left_fields[field], right_fields[field],
                    reference.geometry, other.geometry, reference_land,
                    stage, field)
            fields[field] = comparison
            record = {"stage": stage, "field": field}
            field_class = _field_class(stage, field)
            if not comparison["array_exactly_equal"]:
                all_exact = False
                exact_divergent_fields.append(field)
                if earliest_exact is None:
                    earliest_exact = record
                if (field_class == "hydrology"
                        and earliest_exact_hydrology is None):
                    earliest_exact_hydrology = record
                if (field_class == "terrain"
                        and earliest_exact_terrain is None):
                    earliest_exact_terrain = record
            if not comparison["array_materially_equal"]:
                all_material = False
                material_divergent_fields.append(field)
                if earliest_material is None:
                    earliest_material = record
                if (field_class == "hydrology"
                        and earliest_material_hydrology is None):
                    earliest_material_hydrology = record
                if (field_class == "terrain"
                        and earliest_material_terrain is None):
                    earliest_material_terrain = record
        stages[stage] = {
            "all_fields_exactly_equal": not exact_divergent_fields,
            "all_fields_materially_equal": not material_divergent_fields,
            "exact_divergent_fields_in_causal_order": exact_divergent_fields,
            "material_divergent_fields_in_causal_order": (
                material_divergent_fields),
            "fields": fields,
        }
    return {
        "reference_window": reference.window_name,
        "other_window": other.window_name,
        "scope": "fixed delivered frame plus 40-km collar",
        "all_stage_fields_exactly_equal": all_exact,
        "all_stage_fields_materially_equal": all_material,
        "earliest_exact_divergence": earliest_exact,
        "earliest_exact_hydrology_divergence": earliest_exact_hydrology,
        "earliest_exact_terrain_divergence": earliest_exact_terrain,
        "earliest_material_divergence": earliest_material,
        "earliest_material_hydrology_divergence": (
            earliest_material_hydrology),
        "earliest_material_terrain_divergence": earliest_material_terrain,
        "stages": stages,
    }


def _mode_report(mode: str, structure, elevation, cfg,
                 windows: dict[str, tuple[int, int, int]],
                 instrumentation: EngineInstrumentation) -> dict:
    solved = {}
    captured = {}
    wall_times = {}
    for window_name in WINDOW_ORDER:
        geometry = CoreGeometry.fixed(
            window_name, windows[window_name], structure)
        observer = StageObserver(mode, geometry)
        instrumentation.active = observer
        started = time.perf_counter()
        try:
            solved[window_name] = replay.run_erosion(
                structure, elevation, cfg, SEED,
                _process_window=windows[window_name],
                _localization_mode=mode,
            )
        finally:
            instrumentation.active = None
        wall_times[window_name] = time.perf_counter() - started
        captured[window_name] = observer.finalize(solved[window_name])

    stage_comparisons = {
        f"{name}_vs_large": _compare_captures(
            captured["large"], captured[name])
        for name in ("small", "shifted")
    }
    final_domain_comparisons = {
        f"{name}_vs_large": replay._compare_domains(
            solved["large"], solved[name], replay.PRIMARY_ORIGIN,
            cfg.river_density)
        for name in ("small", "shifted")
    }
    render_comparisons = {
        f"{name}_vs_large": {
            str(size): replay._compare_rendered(
                structure, elevation, solved["large"], solved[name],
                cfg, SEED, replay.PRIMARY_ORIGIN, size)
            for size in (512, 1024)
        }
        for name in ("small", "shifted")
    }
    stage_exact = all(
        comparison["all_stage_fields_exactly_equal"]
        for comparison in stage_comparisons.values())
    stage_material = all(
        comparison["all_stage_fields_materially_equal"]
        for comparison in stage_comparisons.values())
    final_threshold = all(
        comparison["passed"]
        for comparison in final_domain_comparisons.values())
    render_exact = all(
        item["passed_exact"]
        for relation in render_comparisons.values()
        for item in relation.values())

    report = {
        "mode": mode,
        "role": (
            "historical fixed replay diagnosis" if mode == "legacy" else
            "same-structure localization control; not promotion evidence"),
        "erosion_calls": {
            "small": 1, "large": 1, "shifted": 1, "total": 3},
        "wall_times_instrumented_s": wall_times,
        "windows": {
            name: captured[name].report for name in WINDOW_ORDER},
        "stage_comparisons": stage_comparisons,
        "final_domain_comparisons": final_domain_comparisons,
        "render_comparisons": render_comparisons,
        "tested_core_invariance_scope": {
            "seed": SEED,
            "origin_xy_km": list(replay.PRIMARY_ORIGIN),
            "windows": {name: list(windows[name]) for name in WINDOW_ORDER},
            "relations": ["small_vs_large", "shifted_vs_large"],
            "spatial_scope": "delivered frame plus fixed 40-km collar",
            "field_scope": "only arrays captured by this diagnostic",
            "mode": mode,
        },
        "tested_core_captured_fields_exactly_invariant": stage_exact,
        "tested_core_captured_fields_materially_invariant": stage_material,
        "historical_final_domain_threshold_passed": final_threshold,
        "render_exact": render_exact,
        "promotion_assessed": False,
    }
    if mode == "lowstand_outlets":
        report["promotion_not_assessed_reasons"] = [
            "marine fan deposit concentration remains unresolved",
            "far-field sediment export remains unresolved",
            "one seed/origin/window trio cannot establish general invariance",
        ]
    return report


def _headline_equal(expected, observed) -> bool:
    if isinstance(expected, bool) or isinstance(observed, bool):
        return type(expected) is type(observed) and expected == observed
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return bool(np.isclose(
            float(expected), float(observed), rtol=1e-12, atol=1e-12))
    return expected == observed


def _historical_headline_reproduction(structure, candidate,
                                      windows: dict[str, tuple[int, int, int]],
                                      legacy_report: dict) -> dict:
    artifact = ROOT.parent / HISTORICAL_REPORT_RELATIVE
    link = {
        "relative_path_from_workspace": HISTORICAL_REPORT_RELATIVE.as_posix(),
        "expected_sha256": HISTORICAL_REPORT_SHA256,
        "artifact_exists": artifact.is_file(),
    }
    if not artifact.is_file():
        return {
            "historical_report_link": link,
            "frozen_headline_reproduction_checked": False,
            "frozen_headline_reproduction_passed": None,
            "reason": "historical artifact unavailable",
        }

    actual_sha256 = _sha256_file(artifact)
    link.update({
        "actual_sha256": actual_sha256,
        "digest_matched": actual_sha256 == HISTORICAL_REPORT_SHA256,
    })
    if actual_sha256 != HISTORICAL_REPORT_SHA256:
        return {
            "historical_report_link": link,
            "frozen_headline_reproduction_checked": False,
            "frozen_headline_reproduction_passed": False,
            "reason": "historical artifact digest mismatch",
        }

    reference = json.loads(artifact.read_text(encoding="utf-8"))
    checks = {}

    def add(name: str, expected, observed) -> None:
        checks[name] = {
            "expected": expected,
            "observed": observed,
            "passed": _headline_equal(expected, observed),
        }

    add("seed", reference["seed"], SEED)
    add("continental_budget", reference["continental_budget"],
        CONTINENTAL_BUDGET)
    add("origin_xy_km", reference["origin_xy_km"],
        list(replay.PRIMARY_ORIGIN))
    add("structure_n", reference["structure_n"], int(structure.n))
    add("structure_spacing_km", reference["structure_spacing_km"],
        float(structure.world_km / structure.n))
    for name in WINDOW_ORDER:
        add(f"process_windows.{name}", reference["process_windows"][name],
            list(windows[name]))

    current_candidate = asdict(candidate)
    for field, expected in reference["structural"]["candidate"].items():
        add(f"candidate.{field}", expected, current_candidate.get(field))

    domain_fields = (
        "ocean_mask_xor",
        "lake_mask_xor",
        "drawn_discharge_land_cell_count",
        "drawn_discharge_max_relative",
        "river_edge_count_reference",
        "river_edge_count_other",
        "river_topology_symmetric_difference",
        "river_common_edge_a8_max_relative",
        "river_render_class_difference",
        "passed",
    )
    relation_sources = {
        "small_vs_large": "nested_small_vs_large",
        "shifted_vs_large": "shifted_vs_large",
    }
    for current_name, historical_name in relation_sources.items():
        expected_relation = reference[historical_name]
        observed_relation = legacy_report[
            "final_domain_comparisons"][current_name]
        for field in domain_fields:
            add(f"final_domain.{current_name}.{field}",
                expected_relation[field], observed_relation.get(field))
        for size in ("512", "1024"):
            add(f"render.{current_name}.{size}.passed_exact",
                reference["render_convergence"][current_name][size][
                    "passed_exact"],
                legacy_report["render_comparisons"][current_name][size].get(
                    "passed_exact"))

    passed = all(item["passed"] for item in checks.values())
    return {
        "historical_report_link": link,
        "frozen_headline_reproduction_checked": True,
        "frozen_headline_reproduction_passed": passed,
        "headline_definition": (
            "frozen structure/candidate/window identity plus legacy final-domain "
            "and render headline metrics"),
        "historical_obsolete_contour_gate_is_not_reapplied": True,
        "checks": checks,
    }


def _protocol(fingerprint: dict) -> dict:
    return {
        "experiment": EXPERIMENT,
        "manifest_role": "pre-execution fixed diagnostic protocol",
        "source_fingerprint": fingerprint,
        "fixed": {
            "seed": SEED,
            "continental_budget": CONTINENTAL_BUDGET,
            "origin_xy_km": list(replay.PRIMARY_ORIGIN),
            "atlas_km": replay.ATLAS_KM,
            "structural_spacing_km_requested": replay.ORACLE_KM,
            "small_halo_km": replay.SMALL_HALO_KM,
            "large_halo_km": replay.LARGE_HALO_KM,
            "shift_km": replay.SHIFT_KM,
            "comparison_core_collar_km": replay.CORE_COLLAR_KM,
            "process_spacing_km_nominal": erosion_engine.E_KM,
            "solver_steps": erosion_engine.N_STEPS,
        },
        "sequencing": {
            "structural_builds": 1,
            "coarse_elevation_builds": 1,
            "mode_order": list(MODE_ORDER),
            "window_order_per_mode": list(WINDOW_ORDER),
            "erosion_calls_per_window_per_mode": 1,
            "total_erosion_calls": 6,
            "retries": 0,
        },
        "instrumentation": {
            "engine_files_modified": False,
            "historical_spikes_modified": False,
            "wrappers_return_original_values_unchanged": True,
            "raw_flow_core_copied_at_hook_return": True,
            "effective_flow_alias_frozen_after_route_normalization": True,
            "lowstand_effective_flow_state_includes_intended_marine_zeroing": (
                True),
            "other_full_window_arrays_retained_only_until_each_run_is_frozen": (
                True),
            "timings_are_diagnostic_not_performance_authoritative": True,
            "observed_functions": [
                "elevation._chamfer_km",
                "erosion.fill_depressions",
                "erosion._fill_to_lowstand_outlets",
                "erosion.flow_accumulation",
                "erosion.flow_accumulation_d8",
                "erosion.spl_implicit",
                "erosion.soil_creep",
                "erosion.route_sediment",
                "erosion._route_sediment_lowstand",
                "erosion._balance_lakes",
            ],
        },
        "comparison_policy": {
            "exact_and_material_divergence_reported_separately": True,
            "numeric_absolute_difference_core_cell_counts": [
                ">1e-9", ">0.05", ">0.5"],
            "terrain_material_absolute_threshold_m": (
                TERRAIN_MATERIAL_THRESHOLD_M),
            "fill_and_sea_distance_material_absolute_threshold": (
                FILL_DISTANCE_MATERIAL_THRESHOLD),
            "hydrology_accumulation_area_discharge_material_relative_threshold": (
                HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD),
            "receiver_and_mask_material_rule": "exact inequality",
            "routing_field_order": [
                "outlet mask", "filled surface", "global receiver",
                "MFD weights", "runoff", "raw accumulations",
                "effective consumer accumulations",
            ],
        },
        "historical_reference": {
            "relative_path_from_workspace": (
                HISTORICAL_REPORT_RELATIVE.as_posix()),
            "expected_sha256": HISTORICAL_REPORT_SHA256,
            "headline_reproduction_checked_if_artifact_exists": True,
        },
        "decision_policy": {
            "diagnostic_only": True,
            "contours_used": False,
            "crop_reselection": False,
            "promotion_assessed": False,
            "lowstand_control_known_open_defects": [
                "marine fan deposit concentration",
                "far-field sediment export",
            ],
            "full_receiver_path_provenance": {
                "status": "explicitly deferred",
                "follow_up_trigger": (
                    "captured stage evidence remains causally ambiguous"),
                "geometric_boundary_distance_is_not_flow_path_distance": True,
            },
        },
    }


def _run(out: Path) -> dict:
    _prepare_empty_output(out)
    fingerprint = _source_fingerprint()
    protocol = _protocol(fingerprint)
    protocol_sha256 = _write_json_exclusive(
        out / "protocol_precommit.json", protocol)

    cfg = replay._atlas_config(CONTINENTAL_BUDGET)
    started = time.perf_counter()
    structure = replay.build_structure(
        SEED,
        cfg,
        _world_km=replay.ATLAS_KM,
        _coarse_km=replay.ORACLE_KM,
        _continent_seeder=replay._seed_atlas_nuclei,
    )
    elevation = replay.coarse_elevation(structure, cfg, SEED)
    candidates = replay._evaluate_candidates(
        structure, elevation, [elevation], SEED)
    candidate = {
        item.origin: item for item in candidates["safe"]
    }.get(replay.PRIMARY_ORIGIN)
    if candidate is None:
        raise RuntimeError(
            "frozen seed-11 origin is not in the historical conservative "
            "head-time water-safe pool")

    windows = {
        "small": replay._window(
            structure, replay.PRIMARY_ORIGIN, replay.SMALL_HALO_KM),
        "large": replay._window(
            structure, replay.PRIMARY_ORIGIN, replay.LARGE_HALO_KM),
    }
    windows["shifted"] = replay._shift_window(
        windows["large"], structure, -replay.SHIFT_KM, replay.SHIFT_KM)

    modes = {}
    with EngineInstrumentation() as instrumentation:
        # The approved sequencing is material: diagnose the historical path
        # first, then run the existing localization branch as a control on
        # the same structure/elevation objects.
        modes["legacy"] = _mode_report(
            "legacy", structure, elevation, cfg, windows, instrumentation)
        modes["lowstand_outlets"] = _mode_report(
            "lowstand_outlets", structure, elevation, cfg, windows,
            instrumentation)

    historical_reproduction = _historical_headline_reproduction(
        structure, candidate, windows, modes["legacy"])

    report = {
        "experiment": EXPERIMENT,
        "protocol_precommit_sha256": protocol_sha256,
        "source_fingerprint": fingerprint,
        "fixed_candidate": {
            "seed": SEED,
            "origin_xy_km": list(replay.PRIMARY_ORIGIN),
            "candidate": asdict(candidate),
            "candidate_pool": (
                "historical conservative head-time water-safe pool; not "
                "final border acceptance; contour diagnostics non-gating"),
        },
        "structure": {
            "n": int(structure.n),
            "world_km": float(structure.world_km),
            "spacing_km": float(structure.world_km / structure.n),
            "build_count": 1,
            "coarse_elevation_count": 1,
        },
        "process_windows": {name: list(window)
                            for name, window in windows.items()},
        "historical_replay_reproduction": historical_reproduction,
        "modes": modes,
        "conclusion": {
            "legacy_tested_core_captured_fields_exactly_invariant": modes[
                "legacy"][
                    "tested_core_captured_fields_exactly_invariant"],
            "legacy_tested_core_captured_fields_materially_invariant": modes[
                "legacy"][
                    "tested_core_captured_fields_materially_invariant"],
            "lowstand_control_tested_core_captured_fields_exactly_invariant": (
                modes["lowstand_outlets"][
                    "tested_core_captured_fields_exactly_invariant"]),
            "lowstand_control_tested_core_captured_fields_materially_invariant": (
                modes["lowstand_outlets"][
                    "tested_core_captured_fields_materially_invariant"]),
            "invariance_scope": {
                "seed": SEED,
                "origin_xy_km": list(replay.PRIMARY_ORIGIN),
                "windows": list(WINDOW_ORDER),
                "core": "delivered frame plus fixed 40-km collar",
                "fields": "captured fields only",
            },
            "full_receiver_path_provenance": (
                "deferred unless captured stage evidence is ambiguous"),
            "promotion_assessed": False,
            "why_lowstand_is_not_promotion_evidence": [
                "marine fan deposit concentration remains unresolved",
                "far-field sediment export remains unresolved",
            ],
        },
        "elapsed_s": time.perf_counter() - started,
        "completed": True,
    }
    report_sha256 = _write_json_exclusive(out / "report.json", report)
    _write_json_exclusive(out / "report.sha256.json", {
        "file": "report.json", "sha256": report_sha256})
    return report


def _self_check() -> dict:
    rows = np.array([1, 2, 3], np.int64)
    columns = np.array([1, 2, 3], np.int64)
    large_geometry = CoreGeometry.explicit(
        "large", (0, 0, 5), 20.0, rows, columns)
    shifted_geometry = CoreGeometry.explicit(
        "shifted", (0, 0, 5), 20.0, rows, columns)

    receiver = np.arange(25, dtype=np.int64)
    receiver[2 * 5 + 2] = 2 * 5 + 3
    canonical = large_geometry.receiver_global_row_column(receiver)
    receiver_ok = bool(np.array_equal(canonical[1, 1], [2, 3]))

    base = np.ones((3, 3), np.float64)
    fill_exact_only = base.copy()
    fill_exact_only[0, 0] += 5e-10
    hydrology = base.copy()
    hydrology[0, 1] = 2.0
    terrain = base.copy()
    terrain[2, 1] = 3.0
    land = np.ones((3, 3), bool)
    reference = FinalizedCapture(
        mode="self-check",
        window_name="large",
        geometry=large_geometry,
        arrays={
            "initial_surface": {"surface_m": base},
            "routing.solve_step_0": {
                "filled_surface_m": base,
                "mfd_accumulation_raw_at_hook_return": base,
            },
            "solve.step_0.post_stream_power": {"surface_m": base},
        },
        report={},
    )
    other = FinalizedCapture(
        mode="self-check",
        window_name="shifted",
        geometry=shifted_geometry,
        arrays={
            "initial_surface": {"surface_m": base.copy()},
            "routing.solve_step_0": {
                "filled_surface_m": fill_exact_only,
                "mfd_accumulation_raw_at_hook_return": hydrology,
            },
            "solve.step_0.post_stream_power": {"surface_m": terrain},
        },
        report={},
    )
    comparison = _compare_captures(reference, other)
    exact_ok = comparison["earliest_exact_divergence"] == {
        "stage": "routing.solve_step_0", "field": "filled_surface_m"}
    material_ok = comparison["earliest_material_divergence"] == {
        "stage": "routing.solve_step_0",
        "field": "mfd_accumulation_raw_at_hook_return"}
    hydrology_ok = (
        comparison["earliest_material_hydrology_divergence"] == {
            "stage": "routing.solve_step_0",
            "field": "mfd_accumulation_raw_at_hook_return"})
    terrain_ok = comparison["earliest_material_terrain_divergence"] == {
        "stage": "solve.step_0.post_stream_power", "field": "surface_m"}
    routing_stage = comparison["stages"]["routing.solve_step_0"]
    causal_order_ok = routing_stage[
        "exact_divergent_fields_in_causal_order"] == [
            "filled_surface_m", "mfd_accumulation_raw_at_hook_return"]
    fill_report = routing_stage["fields"]["filled_surface_m"]
    exact_material_split_ok = bool(
        not fill_report["array_exactly_equal"]
        and fill_report["array_materially_equal"])

    changed = _cell_changed(base, hydrology)
    field = _field_comparison(
        base, hydrology, large_geometry, shifted_geometry, land,
        "routing.solve_step_0", "mfd_accumulation_raw_at_hook_return")
    evidence = field["material_changed_cell_boundary_evidence"]
    bin_count = sum(
        item["changed_cells"]
        for item in evidence["other_window"][
            "process_cell_boundary_ring_counts"])
    numeric = field["numeric_difference"]
    bins_ok = bool(
        np.count_nonzero(changed) == 1
        and bin_count == 1
        and evidence["changed_cells_by_output_region"][
            "fixed_40km_collar"] == 1
        and evidence["other_window"]["distance_statistics_km"][
            "median"] == 30.0)
    numeric_thresholds_ok = bool(
        numeric["core_cell_counts_by_absolute_difference"][
            "greater_than_1e-09"] == 1
        and numeric["core_cell_counts_by_absolute_difference"][
            "greater_than_0.05"] == 1
        and numeric["core_cell_counts_by_absolute_difference"][
            "greater_than_0.5"] == 1
        and numeric["relative_max"] == 0.5)

    full = np.arange(25, dtype=np.float64).reshape(5, 5)
    snapshot_observer = StageObserver("self-check", large_geometry)
    snapshot_observer.array(
        "snapshot", "raw", full, snapshot_core=True)
    snapshot_observer.array("snapshot", "effective", full)
    full.fill(-1.0)
    raw_capture = snapshot_observer.raw["snapshot"]["raw"]
    effective_capture = snapshot_observer.raw["snapshot"]["effective"]
    assert isinstance(raw_capture, _RawArray)
    assert isinstance(effective_capture, _RawArray)
    snapshot_ok = bool(
        raw_capture.kind == "frozen_core"
        and np.asarray(raw_capture.value)[1, 1] == 12.0
        and np.asarray(effective_capture.value)[2, 2] == -1.0)

    directions = np.arange(8 * 25, dtype=np.float64).reshape(8, 25)
    direction_core = large_geometry.extract_direction_grid(directions)
    directions_ok = direction_core.shape == (3, 3, 8)
    historical_artifact = ROOT.parent / HISTORICAL_REPORT_RELATIVE
    historical_digest_ok = bool(
        not historical_artifact.is_file()
        or _sha256_file(historical_artifact) == HISTORICAL_REPORT_SHA256)
    digest_ok = _array_sha256(base) == _array_sha256(base.copy())
    passed = all((
        receiver_ok, exact_ok, material_ok, hydrology_ok, terrain_ok,
        causal_order_ok, exact_material_split_ok, bins_ok,
        numeric_thresholds_ok, snapshot_ok, directions_ok,
        historical_digest_ok, digest_ok,
    ))
    report = {
        "experiment": f"{EXPERIMENT}-self-check",
        "model_executed": False,
        "checks": {
            "receiver_global_coordinates": receiver_ok,
            "earliest_exact_divergence": exact_ok,
            "earliest_material_divergence": material_ok,
            "earliest_material_hydrology_divergence": hydrology_ok,
            "earliest_material_terrain_divergence": terrain_ok,
            "causal_routing_field_order": causal_order_ok,
            "exact_vs_material_threshold_split": exact_material_split_ok,
            "changed_boundary_distance_and_ring_evidence": bins_ok,
            "numeric_threshold_counts_and_relative_max": numeric_thresholds_ok,
            "raw_snapshot_vs_effective_alias": snapshot_ok,
            "directional_mfd_weight_core_extraction": directions_ok,
            "historical_digest_if_present": historical_digest_ok,
            "stable_array_digest": digest_ok,
        },
        "passed": passed,
    }
    if not passed:
        raise AssertionError(f"process halo diagnostic self-check failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("out") / "process_halo_seed11_v1")
    parser.add_argument(
        "--self-check", action="store_true",
        help="run only synthetic comparison/instrumentation mechanics")
    args = parser.parse_args()
    report = _self_check() if args.self_check else _run(args.out)
    print(json.dumps(
        report, indent=2, allow_nan=False, default=_json_default))


if __name__ == "__main__":
    main()
