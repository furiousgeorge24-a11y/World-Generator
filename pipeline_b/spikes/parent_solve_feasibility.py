"""Private one-parent feasibility run for naturally water-bordered frames.

This spike asks whether one 12,288-km parent, solved once by the current
full-domain legacy surface process, contains three useful 4,096-km delivered
frames near 20%, 35%, and 50% land.  The crop selector runs only after the
parent terrain and process layers exist.  It reads them; it never edits,
tapers, scores contour alignment, or feeds a frame mask back into formation.

This is availability and morphology evidence for crops of one finite parent.
It is not a certificate of finite-parent independence: legacy depression fill
and sediment routing still depend on the parent numerical domain, and exact
overlap between crops from the same parent is true by construction.

Run the sealed experiment from ``pipeline_b`` with::

    python -B -m spikes.parent_solve_feasibility \
        --out ../out/parent_solve_feasibility_seed137_v1

The no-model mechanics check is::

    python -B -m spikes.parent_solve_feasibility --self-check

The harness is private.  It changes no engine source, default, registry
control, historical report, or public API.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from engine import erosion as erosion_engine
from engine.elevation import coarse_elevation
from engine.render_map import MAP_VIEWS, render_map_view
from engine.rng import fnv1a64
from engine.surface import sample_map
from engine.tectonics import Config, FRAME_KM, build_structure
from spikes import atlas_survey
from spikes.causal_border_acceptance import evaluate_causal_border


EXPERIMENT = "parent-solve-feasibility-seed137-v1"
SEED = 137
PARENT_KM = 12288.0
STRUCTURE_NOMINAL_KM = 40.0
PROCESS_NOMINAL_KM = erosion_engine.E_KM
PRIVATE_CONTINENTAL_BUDGET = 0.65
PUBLIC_CONTINENTAL_BUDGET_MAX = 0.45
NUMERICAL_GUARD_KM = 2560.0
# A separate, mechanically derived structural reach screen.  This is not the
# candidate guard and is not claimed to bound depression fill or the legacy
# marine settling tail.
STRUCTURAL_KINEMATIC_RIM_KM = float(round(
    2.2 * Config().plate_speed * Config().eras, 9))
STRUCTURAL_GUARD_REMAINDER_KM = (
    NUMERICAL_GUARD_KM - STRUCTURAL_KINEMATIC_RIM_KM)
CANDIDATE_STRIDE_KM = 256.0
SCAN_SIZE = 256
AUTHORITY_SIZE = 1024
SHORTLIST_PER_TARGET = 8
TARGET_TOLERANCE = 0.05
TARGETS = (
    ("low", 0.20),
    ("medium", 0.35),
    ("high", 0.50),
)
MIN_ORIGIN_CHEBYSHEV_SEPARATION_KM = 0.5 * FRAME_KM
SEDIMENT_CLOSURE_RELATIVE_TOLERANCE = 1e-12
EXPECTED_STRUCTURE_BUILDS = 1
EXPECTED_ELEVATION_BUILDS = 1
EXPECTED_EROSION_SOLVES = 1
EXPECTED_LEGACY_ROUTE_SEDIMENT_CALLS = 1
EXPECTED_LEGACY_ROUTE_GRAPH_STAGES = 4
EXPECTED_SELECTOR_CALLS = 1
EXPECTED_FRAME_COUNT = len(TARGETS)
EXPECTED_STRUCTURE_N = int(round(PARENT_KM / STRUCTURE_NOMINAL_KM))
EXPECTED_PROCESS_N = int(round(PARENT_KM / PROCESS_NOMINAL_KM))
EXPECTED_STRUCTURE_ACTUAL_KM = PARENT_KM / EXPECTED_STRUCTURE_N
EXPECTED_PROCESS_ACTUAL_KM = PARENT_KM / EXPECTED_PROCESS_N

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PRIOR_ARTIFACTS = {
    "recovered_seed11_causal_border_fixture": {
        "relative_path": (
            "pipeline_b/tests/fixtures/recovered_seed11_causal_border.json"),
        "sha256": (
            "61ad01cc900d77e001b63c69e216d6312143b67c3dfb6a6e4ef4e9ce51d1a748"),
        "role": "historical comparison only; never a selector input",
    },
    "recovered_seed11_source_report": {
        "relative_path": "out/atlas_replay_seed11_065_v2/report.json",
        "sha256": (
            "9fcb7741f42b5399ead3931c93164ff9dd50f87f2c57c4c448b31bf18d82c12d"),
        "role": "historical comparison only; never a selector input",
    },
}
SOURCE_FILES = (
    "engine/elevation.py",
    "engine/erosion.py",
    "engine/noise.py",
    "engine/registry.py",
    "engine/render_map.py",
    "engine/rng.py",
    "engine/surface.py",
    "engine/tectonics.py",
    "spikes/atlas_survey.py",
    "spikes/causal_border_acceptance.py",
    "spikes/parent_solve_feasibility.py",
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
    result = {}
    for name, expected in PRIOR_ARTIFACTS.items():
        path = WORKSPACE / expected["relative_path"]
        observed = _sha256_file(path) if path.is_file() else None
        result[name] = {
            **expected,
            "exists": path.is_file(),
            "observed_sha256": observed,
            "digest_matched": observed == expected["sha256"],
        }
    return result


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
            values = np.asarray(array[finite], np.float64)
            result.update({
                "min": float(values.min()),
                "max": float(values.max()),
                "sum": float(values.sum()),
            })
    return result


def _private_parent_config() -> tuple[Config, dict]:
    default = Config()
    reference_world = FRAME_KM * (1.0 + 2.0 * default.world_margin)
    area_ratio = (PARENT_KM / reference_world) ** 2
    plates = max(4, int(round(default.plates * area_ratio)))
    nuclei = max(2, int(round(default.nuclei * area_ratio)))
    cfg = replace(
        default,
        plates=plates,
        nuclei=nuclei,
        continental_budget=PRIVATE_CONTINENTAL_BUDGET,
    )
    assembly_parent_count = max(3, int(round(nuclei / 3.0)))
    target_inventory_fraction_of_parent = (
        assembly_parent_count * PRIVATE_CONTINENTAL_BUDGET
        * FRAME_KM ** 2 / PARENT_KM ** 2)
    return cfg, {
        "reference_default_world_km": reference_world,
        "parent_to_reference_area_ratio": area_ratio,
        "default_plates": default.plates,
        "default_nuclei": default.nuclei,
        "scaled_plates": plates,
        "scaled_nuclei": nuclei,
        "atlas_assembly_parent_count": assembly_parent_count,
        "target_continental_inventory_fraction_of_parent": (
            target_inventory_fraction_of_parent),
        "inventory_interpretation": (
            "The atlas seeder maps 7 nuclei to 3 assembly provinces and "
            "allocates 0.65 delivered-frame areas per province: a target "
            "inventory equal to 21.67% of the 3x parent's area before "
            "transport, overlap, collision survival, and flooding."),
        "private_continental_budget": PRIVATE_CONTINENTAL_BUDGET,
        "public_registry_continental_budget_max": (
            PUBLIC_CONTINENTAL_BUDGET_MAX),
        "private_budget_exceeds_current_public_max": bool(
            PRIVATE_CONTINENTAL_BUDGET
            > PUBLIC_CONTINENTAL_BUDGET_MAX),
        "interpretation": (
            "A pass is framing/availability feasibility at a private 0.65 "
            "budget, not proof that the current public slider supplies the "
            "same land-fraction range."),
    }


def _candidate_origins() -> list[tuple[float, float]]:
    available = PARENT_KM - 2.0 * NUMERICAL_GUARD_KM - FRAME_KM
    steps = int(np.floor(available / CANDIDATE_STRIDE_KM))
    residual = available - steps * CANDIDATE_STRIDE_KM
    first = NUMERICAL_GUARD_KM + 0.5 * residual
    axis = first + CANDIDATE_STRIDE_KM * np.arange(steps + 1)
    return [(float(x), float(y)) for y in axis for x in axis]


def _candidate_tie_key(origin) -> int:
    x0, y0 = origin
    return fnv1a64(
        f"parent-candidate-v1:{SEED}:{x0:.0f}:{y0:.0f}")


def _assignment_tie_key(assignment: dict) -> int:
    ordered = []
    for label, _ in TARGETS:
        candidate = assignment[label]
        ordered.append(
            f"{label}:{candidate.x0_km:.0f}:{candidate.y0_km:.0f}")
    return fnv1a64(
        f"parent-assignment-v1:{SEED}:" + ";".join(ordered))


def _outer_ring_mask(shape) -> np.ndarray:
    rows, columns = (int(value) for value in shape)
    ring = np.zeros((rows, columns), bool)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    return ring


@dataclass(frozen=True)
class CandidateScore:
    x0_km: float
    y0_km: float
    size_px: int
    land_fraction: float
    water_fraction: float
    topographic_nonnegative_fraction: float
    border_passed: bool
    ring_cell_count: int
    ring_water_cells: int
    ring_ocean_cells: int
    ring_lake_cells: int
    ring_non_water_cells: int
    nearest_land_to_border_km: float | None
    tie_key: int

    @property
    def origin(self) -> tuple[float, float]:
        return self.x0_km, self.y0_km


def _score_sample(sampled, origin) -> CandidateScore:
    water = np.asarray(sampled["water"], bool)
    ocean = np.asarray(sampled["ocean"], bool)
    lake = np.asarray(sampled["lake"], bool)
    h = np.asarray(sampled["h"], np.float64)
    # Contract section 3 defines the delivered categorical water mask as
    # authority.  Therefore "nearest land" means nearest non-water cell,
    # even in the rare case that categorical sampling and h>=0 differ.
    land_rows, land_columns = np.nonzero(~water)
    if land_rows.size:
        pixel_distance = np.minimum.reduce((
            land_rows,
            land_columns,
            h.shape[0] - 1 - land_rows,
            h.shape[1] - 1 - land_columns,
        ))
        nearest_land_km = float(
            pixel_distance.min() * (FRAME_KM / h.shape[0]))
    else:
        nearest_land_km = None
    border = evaluate_causal_border(water)
    ring = _outer_ring_mask(water.shape)
    return CandidateScore(
        x0_km=float(origin[0]),
        y0_km=float(origin[1]),
        size_px=int(water.shape[0]),
        land_fraction=float(np.mean(~water)),
        water_fraction=float(np.mean(water)),
        topographic_nonnegative_fraction=float(np.mean(h >= 0.0)),
        border_passed=bool(border["passed"]),
        ring_cell_count=int(np.count_nonzero(ring)),
        ring_water_cells=int(np.count_nonzero(water & ring)),
        ring_ocean_cells=int(np.count_nonzero(ocean & ring)),
        ring_lake_cells=int(np.count_nonzero(lake & ring)),
        ring_non_water_cells=int(np.count_nonzero(~water & ring)),
        nearest_land_to_border_km=nearest_land_km,
        tie_key=_candidate_tie_key(origin),
    )


def _bounded_shortlist(scan_scores: list[CandidateScore]) -> dict:
    by_target = {}
    union = {}
    water_safe = [score for score in scan_scores if score.border_passed]
    for label, target in TARGETS:
        ranked = sorted(
            water_safe,
            key=lambda score: (
                abs(score.land_fraction - target), score.tie_key))
        chosen = ranked[:SHORTLIST_PER_TARGET]
        by_target[label] = chosen
        for score in chosen:
            union[score.origin] = score
    union_scores = sorted(union.values(), key=lambda score: score.tie_key)
    return {"by_target": by_target, "union": union_scores}


def _separated(left: CandidateScore, right: CandidateScore) -> bool:
    return max(
        abs(left.x0_km - right.x0_km),
        abs(left.y0_km - right.y0_km),
    ) >= MIN_ORIGIN_CHEBYSHEV_SEPARATION_KM


def _joint_assignment(authority_scores: list[CandidateScore]) -> dict:
    pools = {}
    for label, target in TARGETS:
        pools[label] = [
            score for score in authority_scores
            if (score.border_passed
                and abs(score.land_fraction - target)
                <= TARGET_TOLERANCE
                and (label != "high"
                     or (score.land_fraction < 0.50
                         and score.water_fraction > 0.50)))
        ]
    viable = []
    labels = [label for label, _ in TARGETS]
    targets = {label: target for label, target in TARGETS}
    for values in itertools.product(*(pools[label] for label in labels)):
        if len({value.origin for value in values}) != len(values):
            continue
        if not all(_separated(left, right)
                   for left, right in itertools.combinations(values, 2)):
            continue
        assignment = dict(zip(labels, values))
        errors = {
            label: abs(assignment[label].land_fraction - targets[label])
            for label in labels}
        key = (
            max(value / TARGET_TOLERANCE for value in errors.values()),
            errors["high"],
            errors["medium"],
            errors["low"],
            sum(errors.values()),
            _assignment_tie_key(assignment),
        )
        viable.append((key, assignment, errors))
    if not viable:
        return {
            "found": False,
            "assignment": {},
            "errors": {},
            "objective": None,
            "viable_assignment_count": 0,
            "pool_counts": {
                label: len(pool) for label, pool in pools.items()},
        }
    viable.sort(key=lambda item: item[0])
    key, assignment, errors = viable[0]
    return {
        "found": True,
        "assignment": assignment,
        "errors": errors,
        "objective": {
            "max_normalized_target_error": float(key[0]),
            "high_abs_target_error": float(key[1]),
            "medium_abs_target_error": float(key[2]),
            "low_abs_target_error": float(key[3]),
            "sum_abs_target_error": float(key[4]),
            "deterministic_assignment_hash": int(key[5]),
        },
        "viable_assignment_count": len(viable),
        "pool_counts": {
            label: len(pool) for label, pool in pools.items()},
    }


def _best_available_by_target(
        authority_scores: list[CandidateScore]) -> dict:
    """Return independent deterministic diagnostics, never an acceptance.

    A target may have no entry when the frozen authority shortlist contains no
    exact-water-ring candidate.  Origins may repeat across targets: unlike the
    joint assignment these views answer only "what was the closest inspected
    morphology?" and cannot satisfy the availability gate.
    """
    water_safe = [score for score in authority_scores if score.border_passed]
    result = {}
    for label, target in TARGETS:
        ranked = sorted(
            water_safe,
            key=lambda score: (
                abs(score.land_fraction - target), score.tie_key))
        result[label] = ranked[0] if ranked else None
    return result


def _scan_and_select(structure, elevation, erosion_result, cfg) -> dict:
    scan_scores = []
    for origin in _candidate_origins():
        x0, y0 = origin
        sampled = sample_map(
            structure, elevation, erosion_result, cfg, SEED, SCAN_SIZE,
            _frame_window_km=(y0, x0, FRAME_KM))
        scan_scores.append(_score_sample(sampled, origin))
    shortlist = _bounded_shortlist(scan_scores)
    authority_scores = []
    authority_samples = {}
    for scan_score in shortlist["union"]:
        origin = scan_score.origin
        x0, y0 = origin
        sampled = sample_map(
            structure, elevation, erosion_result, cfg, SEED,
            AUTHORITY_SIZE,
            _frame_window_km=(y0, x0, FRAME_KM))
        authority_samples[origin] = sampled
        authority_scores.append(_score_sample(sampled, origin))
    assignment = _joint_assignment(authority_scores)
    accepted_samples = {
        label: authority_samples[score.origin]
        for label, score in assignment["assignment"].items()}
    diagnostic_best = _best_available_by_target(authority_scores)
    if assignment["found"]:
        render_role = "accepted_joint_assignment"
        render_scores = dict(assignment["assignment"])
        render_samples = dict(accepted_samples)
    else:
        render_role = "diagnostic_best_available_not_accepted"
        render_scores = {
            label: score for label, score in diagnostic_best.items()
            if score is not None}
        render_samples = {
            label: authority_samples[score.origin]
            for label, score in render_scores.items()}
    return {
        "scan_scores": scan_scores,
        "shortlist": shortlist,
        "authority_scores": authority_scores,
        "assignment": assignment,
        "accepted_samples": accepted_samples,
        "diagnostic_best_available": diagnostic_best,
        "render_role": render_role,
        "render_scores": render_scores,
        "render_samples": render_samples,
    }


@dataclass
class RouteStageCapture:
    raw_z_m: np.ndarray
    filled_z_m: np.ndarray
    receiver: np.ndarray | None = None
    batches: tuple[np.ndarray, ...] | None = None
    mfd_targets: np.ndarray | None = None
    mfd_weights: np.ndarray | None = None
    flat: np.ndarray | None = None
    full_flow_rim_ancestor: np.ndarray | None = None
    positive_mfd_edge_count: int = 0
    d8_fallback_edge_count: int = 0


@dataclass
class RouteSedimentCapture:
    z_m: np.ndarray
    erosion_source_m: np.ndarray
    receiver: np.ndarray
    batches: tuple[np.ndarray, ...]
    area_km2: np.ndarray
    base_level_m: float
    deposition_length_km: float
    process_spacing_km: float
    edge_length_km: np.ndarray | None
    output_z_m: np.ndarray
    deposit_m: np.ndarray
    boundary_export_m_cells: float
    terminal_residual_m_cells: float
    inputs_unchanged: bool


class LegacySedimentInstrumentation(AbstractContextManager):
    """Pass-through capture of four legacy graphs and one sediment call."""

    def __init__(self):
        self.original_route_sediment: Callable | None = None
        self.original_fill_depressions: Callable | None = None
        self.original_receivers: Callable | None = None
        self.original_topo_batches: Callable | None = None
        self.call_count = 0
        self.capture: RouteSedimentCapture | None = None
        self.route_stages: list[RouteStageCapture] = []

    def __enter__(self):
        self.original_route_sediment = erosion_engine.route_sediment
        self.original_fill_depressions = erosion_engine.fill_depressions
        self.original_receivers = erosion_engine.receivers
        self.original_topo_batches = erosion_engine.topo_batches

        def fill_depressions(surface, *args, **kwargs):
            raw = np.asarray(surface).copy()
            result = self.original_fill_depressions(
                surface, *args, **kwargs)
            if not np.array_equal(surface, raw):
                raise AssertionError("fill_depressions mutated its input")
            self.route_stages.append(RouteStageCapture(
                raw_z_m=raw, filled_z_m=np.asarray(result).copy()))
            return result

        def receivers(filled):
            if (not self.route_stages
                    or self.route_stages[-1].receiver is not None):
                raise AssertionError(
                    "legacy receivers call did not follow one captured fill")
            result = self.original_receivers(filled)
            stage = self.route_stages[-1]
            if not np.array_equal(filled, stage.filled_z_m):
                raise AssertionError(
                    "legacy receivers input differs from captured fill")
            stage.receiver = np.asarray(result[0]).copy()
            # Retain these only until topo_batches supplies the matching
            # source-to-outlet order.  The completed capture stores the
            # propagated full-flow ancestry mask, not four redundant copies
            # of the 8-neighbour target lattice and weight arrays.
            stage.mfd_targets = np.asarray(result[1])
            stage.mfd_weights = np.asarray(result[2])
            stage.flat = np.asarray(result[3])
            return result

        def topo_batches(rcv, targets, weights, flat):
            if (not self.route_stages
                    or self.route_stages[-1].receiver is None
                    or self.route_stages[-1].batches is not None):
                raise AssertionError(
                    "legacy topo_batches call did not follow receivers")
            result = self.original_topo_batches(
                rcv, targets, weights, flat)
            stage = self.route_stages[-1]
            if not np.array_equal(rcv, stage.receiver):
                raise AssertionError(
                    "legacy topo receiver differs from captured receiver")
            stage.batches = tuple(
                np.asarray(batch).copy() for batch in result)
            spacing_km = PARENT_KM / stage.raw_z_m.shape[0]
            rim = _rim_mask(stage.raw_z_m.shape, spacing_km)
            stage.full_flow_rim_ancestor = (
                _propagate_full_flow_rim_ancestor(
                    stage.receiver, stage.mfd_targets,
                    stage.mfd_weights, stage.flat, stage.batches, rim))
            stage.positive_mfd_edge_count = int(np.count_nonzero(
                stage.mfd_weights > 0.0))
            index = np.arange(stage.receiver.size)
            stage.d8_fallback_edge_count = int(np.count_nonzero(
                stage.flat & (stage.receiver != index)))
            stage.mfd_targets = None
            stage.mfd_weights = None
            stage.flat = None
            return result

        def route(z, ero, rcv, batches, area, base_level, length_km,
                  spacing_km, edge_len_km=None):
            if self.call_count != 0:
                raise AssertionError(
                    "sealed parent run permits exactly one route_sediment call")
            self.call_count += 1
            z_before = np.asarray(z).copy()
            erosion_before = np.asarray(ero).copy()
            receiver_before = np.asarray(rcv).copy()
            batches_before = tuple(np.asarray(batch).copy()
                                   for batch in batches)
            area_before = np.asarray(area).copy()
            edge_before = (None if edge_len_km is None
                           else np.asarray(edge_len_km).copy())
            result = self.original_route_sediment(
                z, ero, rcv, batches, area, base_level, length_km,
                spacing_km, edge_len_km)
            unchanged = (
                np.array_equal(z, z_before)
                and np.array_equal(ero, erosion_before)
                and np.array_equal(rcv, receiver_before)
                and all(np.array_equal(left, right)
                        for left, right in zip(batches, batches_before))
                and np.array_equal(area, area_before)
                and ((edge_len_km is None and edge_before is None)
                     or np.array_equal(edge_len_km, edge_before))
            )
            self.capture = RouteSedimentCapture(
                z_m=z_before,
                erosion_source_m=erosion_before,
                receiver=receiver_before,
                batches=batches_before,
                area_km2=area_before,
                base_level_m=float(base_level),
                deposition_length_km=float(length_km),
                process_spacing_km=float(spacing_km),
                edge_length_km=edge_before,
                output_z_m=np.asarray(result[0]).copy(),
                deposit_m=np.asarray(result[1]).copy(),
                boundary_export_m_cells=float(result[2]),
                terminal_residual_m_cells=float(result[3]),
                inputs_unchanged=unchanged,
            )
            return result

        erosion_engine.fill_depressions = fill_depressions
        erosion_engine.receivers = receivers
        erosion_engine.topo_batches = topo_batches
        erosion_engine.route_sediment = route
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_route_sediment is not None:
            erosion_engine.route_sediment = self.original_route_sediment
        if self.original_fill_depressions is not None:
            erosion_engine.fill_depressions = self.original_fill_depressions
        if self.original_receivers is not None:
            erosion_engine.receivers = self.original_receivers
        if self.original_topo_batches is not None:
            erosion_engine.topo_batches = self.original_topo_batches
        return False


def _sediment_budget(capture: RouteSedimentCapture,
                     erosion_result) -> dict:
    source = float(np.maximum(capture.erosion_source_m, 0.0).sum())
    deposited = float(capture.deposit_m.sum())
    export = float(capture.boundary_export_m_cells)
    terminal = float(capture.terminal_residual_m_cells)
    closure = source - (deposited + export + terminal)
    tolerance = SEDIMENT_CLOSURE_RELATIVE_TOLERANCE * max(source, 1.0)
    cell_area_m2 = (capture.process_spacing_km * 1000.0) ** 2
    checks = {
        "arrays_finite": bool(
            np.isfinite(capture.z_m).all()
            and np.isfinite(capture.output_z_m).all()
            and np.isfinite(capture.erosion_source_m).all()
            and np.isfinite(capture.deposit_m).all()
            and np.isfinite(capture.area_km2).all()
            and np.isfinite(source)
            and np.isfinite(deposited)
            and np.isfinite(export)
            and np.isfinite(terminal)
            and np.isfinite(closure)),
        "source_and_deposit_nonnegative": bool(
            (capture.erosion_source_m >= 0.0).all()
            and (capture.deposit_m >= 0.0).all()),
        "export_and_terminal_nonnegative": bool(
            export >= 0.0 and terminal >= 0.0),
        "closure_within_scaled_1e_minus_12": abs(closure) <= tolerance,
        "explicit_terminal_residual_exactly_zero_m_cells": terminal == 0.0,
        "delivered_export_volume_matches_capture": bool(
            erosion_result["sediment_export_m3"] == export * cell_area_m2),
        "delivered_terminal_volume_matches_capture": bool(
            erosion_result["sediment_terminal_residual_m3"]
            == terminal * cell_area_m2),
        "captured_inputs_unchanged": capture.inputs_unchanged,
        "captured_output_z_reconstructs_exact": bool(np.array_equal(
            capture.output_z_m, capture.z_m + capture.deposit_m)),
        "delivered_final_z_matches_capture_exact": bool(np.array_equal(
            erosion_result["z"], capture.output_z_m)),
        "delivered_deposit_matches_capture_exact": bool(np.array_equal(
            erosion_result["sed"], capture.deposit_m)),
        "delivered_erosion_source_matches_capture_exact": bool(
            np.array_equal(
                erosion_result["ero"], capture.erosion_source_m)),
    }
    return {
        "source_m_cells": source,
        "deposited_m_cells": deposited,
        "boundary_export_m_cells": export,
        "terminal_residual_m_cells": terminal,
        "closure_m_cells": closure,
        "scaled_closure_tolerance_m_cells": tolerance,
        "checks": checks,
        "all_validity_checks_passed": all(checks.values()),
        "closure_passed": checks["closure_within_scaled_1e_minus_12"],
        "terminal_zero_passed": checks[
            "explicit_terminal_residual_exactly_zero_m_cells"],
    }


def _sediment_stage_oracle(capture: RouteSedimentCapture,
                           route_stages: list[RouteStageCapture]) -> dict:
    """Bind the sediment wrapper to the third legacy route_graph call."""
    supported = len(route_stages) >= 3
    if not supported:
        return {
            "supported": False,
            "stage_index": 2,
            "checks": {},
            "passed": False,
        }
    stage = route_stages[2]
    complete = stage.receiver is not None and stage.batches is not None
    checks = {
        "raw_surface_exact": bool(
            np.array_equal(capture.z_m, stage.raw_z_m)),
        "receiver_exact": bool(
            complete and np.array_equal(
                capture.receiver, stage.receiver)),
        "batch_count_exact": bool(
            complete and len(capture.batches) == len(stage.batches)),
        "batches_exact": bool(
            complete and len(capture.batches) == len(stage.batches)
            and all(np.array_equal(left, right)
                    for left, right in zip(
                        capture.batches, stage.batches))),
    }
    return {
        "supported": True,
        "stage_index": 2,
        "stage_label": ROUTE_STAGE_LABELS[2],
        "checks": checks,
        "passed": all(checks.values()),
    }


PATH_UNRESOLVED = 0
PATH_LOWSTAND_BEFORE_RIM = 1
PATH_RIM_BEFORE_LOWSTAND = 2
PATH_INTERIOR_TERMINAL = 3


def _rim_mask(shape, spacing_km,
              rim_km=STRUCTURAL_KINEMATIC_RIM_KM):
    rows, columns = shape
    y = (np.arange(rows) + 0.5) * float(spacing_km)
    x = (np.arange(columns) + 0.5) * float(spacing_km)
    world_y = rows * float(spacing_km)
    world_x = columns * float(spacing_km)
    return (
        (y[:, None] < rim_km)
        | (y[:, None] >= world_y - rim_km)
        | (x[None, :] < rim_km)
        | (x[None, :] >= world_x - rim_km)
    )


def _propagate_rim_ancestor(receiver, batches, rim) -> np.ndarray:
    """Propagate a rim flag down only the concentrated D8 graph.

    Kept as a narrow synthetic oracle.  Production causal screening uses
    ``_propagate_full_flow_rim_ancestor`` because erosion discharge follows
    every positive MFD edge, with D8 only as the flat-cell fallback.
    """
    receiver = np.asarray(receiver, np.int64)
    contaminated = np.asarray(rim, bool).ravel().copy()
    for batch in batches:
        batch = np.asarray(batch, np.int64)
        target = receiver[batch]
        moving = target != batch
        sources = batch[moving & contaminated[batch]]
        if sources.size:
            np.logical_or.at(contaminated, receiver[sources], True)
    return contaminated.reshape(rim.shape)


def _propagate_full_flow_rim_ancestor(
        receiver, targets, weights, flat, batches, rim) -> np.ndarray:
    """Propagate rim ancestry over the production MFD + D8-fallback graph."""
    receiver = np.asarray(receiver, np.int64)
    targets = np.asarray(targets, np.int64)
    weights = np.asarray(weights, np.float64)
    flat = np.asarray(flat, bool)
    contaminated = np.asarray(rim, bool).ravel().copy()
    index = np.arange(receiver.size)
    for batch in batches:
        batch = np.asarray(batch, np.int64)
        contaminated_sources = contaminated[batch]
        if not contaminated_sources.any():
            continue
        for direction in range(weights.shape[0]):
            moving = contaminated_sources & (weights[direction, batch] > 0.0)
            if moving.any():
                contaminated[targets[direction, batch[moving]]] = True
        fallback = (contaminated_sources & flat[batch]
                    & (receiver[batch] != index[batch]))
        if fallback.any():
            contaminated[receiver[batch[fallback]]] = True
    return contaminated.reshape(rim.shape)


def _classify_paths_before_rim(z, base_level, receiver, batches,
                               rim) -> np.ndarray:
    """Classify the concentrated D8 terrain/sediment path to base level.

    Incision's implicit downstream elevation and legacy sediment transport
    both consume this receiver.  MFD is separately covered by the full-flow
    rim-to-crop ancestry screen; an MFD branch leaving a crop cannot feed its
    own upstream accumulation back from the numerical rim.
    """
    z = np.asarray(z, np.float64)
    receiver = np.asarray(receiver, np.int64)
    status = np.full(z.size, PATH_UNRESOLVED, np.uint8)
    rim_flat = np.asarray(rim, bool).ravel()
    lowstand = np.asarray(z < float(base_level), bool).ravel()
    # The conservative screen requires lowstand before entering the derived
    # structural-risk band. A lowstand cell already inside it is rim-first.
    status[lowstand & ~rim_flat] = PATH_LOWSTAND_BEFORE_RIM
    status[rim_flat] = PATH_RIM_BEFORE_LOWSTAND
    index = np.arange(z.size)
    self_receiver = receiver == index
    status[self_receiver & ~rim_flat & ~lowstand] = PATH_INTERIOR_TERMINAL
    for batch in reversed(batches):
        batch = np.asarray(batch, np.int64)
        unknown = status[batch] == PATH_UNRESOLVED
        cells = batch[unknown]
        if cells.size:
            status[cells] = status[receiver[cells]]
    return status.reshape(z.shape)


def _frame_process_mask(shape, spacing_km, origin,
                        support_km: float = 0.0) -> np.ndarray:
    rows, columns = shape
    x0, y0 = origin
    y = (np.arange(rows) + 0.5) * float(spacing_km)
    x = (np.arange(columns) + 0.5) * float(spacing_km)
    return (
        (y[:, None] >= y0 - support_km)
        & (y[:, None] < y0 + FRAME_KM + support_km)
        & (x[None, :] >= x0 - support_km)
        & (x[None, :] < x0 + FRAME_KM + support_km)
    )


ROUTE_STAGE_LABELS = (
    "erosion_step_1_pre_solve",
    "erosion_step_2_pre_solve",
    "pre_sediment",
    "final_post_deposition_delivery",
)


def _one_stage_graph_causal_audit(stage: RouteStageCapture,
                                  base_level_m: float,
                                  process_spacing_km: float,
                                  assignment: dict,
                                  stage_index: int) -> dict:
    complete = bool(
        stage.receiver is not None
        and stage.batches is not None
        and stage.full_flow_rim_ancestor is not None)
    label = (ROUTE_STAGE_LABELS[stage_index]
             if stage_index < len(ROUTE_STAGE_LABELS)
             else f"unexpected_stage_{stage_index + 1}")
    if not complete:
        return {
            "stage_index": stage_index,
            "stage_label": label,
            "capture_complete": False,
            "frames": {},
            "all_frames_zero_structural_band_ancestry": False,
            "all_frame_concentrated_d8_land_paths_reach_strict_lowstand_before_band": False,
        }
    rim = _rim_mask(stage.raw_z_m.shape, process_spacing_km)
    contaminated = stage.full_flow_rim_ancestor
    path_status = _classify_paths_before_rim(
        stage.raw_z_m, base_level_m, stage.receiver, stage.batches, rim)
    frames = {}
    for frame_label, score in assignment.items():
        frame_mask = _frame_process_mask(
            stage.raw_z_m.shape, process_spacing_km, score.origin)
        support_mask = _frame_process_mask(
            stage.raw_z_m.shape, process_spacing_km, score.origin,
            # Four process cells conservatively cover lateral provenance
            # through 2 creep substeps x 2 erosion steps; two more cover
            # Catmull-Rom sampling support at delivered resolution.
            support_km=6.0 * process_spacing_km)
        land = frame_mask & (stage.raw_z_m >= 0.0)
        status_values, status_counts = np.unique(
            path_status[land], return_counts=True)
        counts = {
            str(int(value)): int(count)
            for value, count in zip(status_values, status_counts)}
        frames[frame_label] = {
            "origin_xy_km": list(score.origin),
            "process_mask_cells": int(np.count_nonzero(frame_mask)),
            "process_plus_cubic_support_mask_cells": int(
                np.count_nonzero(support_mask)),
            "conservative_lateral_plus_cubic_support_km": float(
                6.0 * process_spacing_km),
            "raw_present_land_cells": int(np.count_nonzero(land)),
            "structural_band_ancestor_cells_all_mask": int(
                np.count_nonzero(contaminated & support_mask)),
            "structural_band_ancestor_cells_on_land": int(
                np.count_nonzero(contaminated & land)),
            "land_path_status_counts": {
                "lowstand_before_band": counts.get(
                    str(PATH_LOWSTAND_BEFORE_RIM), 0),
                "band_before_lowstand": counts.get(
                    str(PATH_RIM_BEFORE_LOWSTAND), 0),
                "interior_terminal": counts.get(
                    str(PATH_INTERIOR_TERMINAL), 0),
                "unresolved": counts.get(str(PATH_UNRESOLVED), 0),
            },
            "zero_structural_band_ancestry": not bool(
                np.any(contaminated & support_mask)),
            "all_crop_concentrated_d8_land_paths_reach_strict_lowstand_before_band": bool(
                np.count_nonzero(land) > 0
                and np.all(path_status[land]
                           == PATH_LOWSTAND_BEFORE_RIM)),
        }
    frame_complete = len(frames) == EXPECTED_FRAME_COUNT
    return {
        "stage_index": stage_index,
        "stage_label": label,
        "capture_complete": True,
        "raw_z_sha256": _array_sha256(stage.raw_z_m),
        "filled_z_sha256": _array_sha256(stage.filled_z_m),
        "receiver_sha256": _array_sha256(stage.receiver),
        "positive_mfd_edge_count": stage.positive_mfd_edge_count,
        "d8_fallback_edge_count": stage.d8_fallback_edge_count,
        "rim_cells": int(np.count_nonzero(rim)),
        "structural_band_ancestor_reachable_cells": int(
            np.count_nonzero(contaminated)),
        "frames": frames,
        "all_frames_zero_structural_band_ancestry": bool(
            frame_complete and all(
                value["zero_structural_band_ancestry"]
                for value in frames.values())),
        "all_frame_concentrated_d8_land_paths_reach_strict_lowstand_before_band": bool(
            frame_complete and all(value[
                "all_crop_concentrated_d8_land_paths_reach_strict_lowstand_before_band"]
                for value in frames.values())),
    }


def _graph_causal_audit(route_stages: list[RouteStageCapture],
                        base_level_m: float,
                        process_spacing_km: float,
                        assignment: dict) -> dict:
    stages = [
        _one_stage_graph_causal_audit(
            stage, base_level_m, process_spacing_km, assignment, index)
        for index, stage in enumerate(route_stages)]
    exactly_four = len(stages) == EXPECTED_LEGACY_ROUTE_GRAPH_STAGES
    all_complete = bool(
        exactly_four and all(stage["capture_complete"] for stage in stages))
    return {
        "candidate_guard_km": NUMERICAL_GUARD_KM,
        "derived_structural_kinematic_band_km": (
            STRUCTURAL_KINEMATIC_RIM_KM),
        "guard_beyond_structural_band_km": (
            STRUCTURAL_GUARD_REMAINDER_KM),
        "derivation": "2.2 * 45 * 20 km = 1,980 km",
        "strict_lowstand_threshold_m": float(base_level_m),
        "strict_lowstand_operator": "raw_z < base_level",
        "expected_stage_count": EXPECTED_LEGACY_ROUTE_GRAPH_STAGES,
        "captured_stage_count": len(stages),
        "all_expected_stages_captured": all_complete,
        "path_status_codes": {
            "unresolved": PATH_UNRESOLVED,
            "lowstand_before_band": PATH_LOWSTAND_BEFORE_RIM,
            "band_before_lowstand": PATH_RIM_BEFORE_LOWSTAND,
            "interior_terminal": PATH_INTERIOR_TERMINAL,
        },
        "stages": stages,
        "all_stages_all_frames_zero_structural_band_ancestry": bool(
            all_complete and all(
                stage["all_frames_zero_structural_band_ancestry"]
                for stage in stages)),
        "all_stages_all_frame_concentrated_d8_land_paths_reach_strict_lowstand_before_band": bool(
            all_complete and all(stage[
                "all_frame_concentrated_d8_land_paths_reach_strict_lowstand_before_band"]
                for stage in stages)),
        "scope_limit": (
            "Conservative exclusion screen, not a causation test or proof. "
            "The 1,980-km band is derived only from structural kinematics. "
            "Legacy depression fill has no finite reach bound and legacy "
            "marine sediment has an unbounded exponential tail, so one "
            "parent cannot prove section 3b; band ancestry can also be a "
            "wholly natural routed contributor."),
    }


def _river_overlap_witness(erosion_result, capture, assignment) -> dict:
    edges = erosion_result["river_edges"]
    frames = {}
    for label, score in assignment.items():
        x0, y0 = score.origin
        keep = (
            (edges["x0"] >= x0) & (edges["x0"] < x0 + FRAME_KM)
            & (edges["y0"] >= y0) & (edges["y0"] < y0 + FRAME_KM)
        )
        indices = np.flatnonzero(keep)
        witness = None
        if indices.size:
            index = int(indices[0])
            row = int(np.floor(edges["y0"][index]
                               / capture.process_spacing_km))
            column = int(np.floor(edges["x0"][index]
                                  / capture.process_spacing_km))
            mask = _frame_process_mask(
                capture.z_m.shape, capture.process_spacing_km, score.origin)
            witness = {
                "edge_index": index,
                "source_xy_km": [
                    float(edges["x0"][index]),
                    float(edges["y0"][index])],
                "receiver_xy_km": [
                    float(edges["x1"][index]),
                    float(edges["y1"][index])],
                "a8": float(edges["a8"][index]),
                "process_row_column": [row, column],
                "source_cell_inside_frame_mask": bool(mask[row, column]),
            }
        frames[label] = {
            "river_edge_sources_inside_frame": int(indices.size),
            "witness": witness,
            "overlap_witness_passed": bool(
                witness is not None
                and witness["source_cell_inside_frame_mask"]),
        }
    return {
        "frames": frames,
        "all_frames_have_mask_river_overlap_witness": bool(
            len(frames) == EXPECTED_FRAME_COUNT
            and all(value["overlap_witness_passed"]
                    for value in frames.values())),
    }


def _authority_overlap_audit(chosen_samples, assignment) -> dict:
    labels = [label for label, _ in TARGETS if label in chosen_samples]
    pairs = {}
    all_exact = len(labels) == EXPECTED_FRAME_COUNT
    for left_label, right_label in itertools.combinations(labels, 2):
        left_score = assignment[left_label]
        right_score = assignment[right_label]
        x_start = max(left_score.x0_km, right_score.x0_km)
        x_end = min(left_score.x0_km + FRAME_KM,
                    right_score.x0_km + FRAME_KM)
        y_start = max(left_score.y0_km, right_score.y0_km)
        y_end = min(left_score.y0_km + FRAME_KM,
                    right_score.y0_km + FRAME_KM)
        pair_name = f"{left_label}_vs_{right_label}"
        if x_end <= x_start or y_end <= y_start:
            all_exact = False
            pairs[pair_name] = {
                "overlap_area_km2": 0.0,
                "overlap_fraction_of_frame": 0.0,
                "exact_shared_parent_fields": None,
            }
            continue
        pixels_per_km = AUTHORITY_SIZE / FRAME_KM

        def slices(score):
            row0 = int(round((y_start - score.y0_km) * pixels_per_km))
            row1 = int(round((y_end - score.y0_km) * pixels_per_km))
            column0 = int(round((x_start - score.x0_km) * pixels_per_km))
            column1 = int(round((x_end - score.x0_km) * pixels_per_km))
            return np.s_[row0:row1, column0:column1]

        left_slice = slices(left_score)
        right_slice = slices(right_score)
        fields = {}
        for field in ("h", "hc", "water", "ocean", "lake",
                      "lake_level", "riv_log", "sed"):
            fields[field] = bool(np.array_equal(
                chosen_samples[left_label][field][left_slice],
                chosen_samples[right_label][field][right_slice]))
        exact = all(fields.values())
        all_exact &= exact
        pairs[pair_name] = {
            "overlap_area_km2": float(
                (x_end - x_start) * (y_end - y_start)),
            "overlap_fraction_of_frame": float(
                ((x_end - x_start) * (y_end - y_start))
                / (FRAME_KM ** 2)),
            "field_array_exact": fields,
            "exact_shared_parent_fields": exact,
        }
    overlap_fractions = [
        value["overlap_fraction_of_frame"] for value in pairs.values()]
    return {
        "pairs": pairs,
        "all_overlapping_authority_fields_exact": all_exact,
        "all_three_pairs_reported": len(pairs) == 3,
        "minimum_pair_overlap_fraction": (
            min(overlap_fractions) if overlap_fractions else None),
        "all_pairs_overlap_at_least_0_0625": bool(
            len(overlap_fractions) == 3
            and min(overlap_fractions) >= 0.0625),
        "interpretation": (
            "Exact overlap is expected by construction because these frames "
            "sample one solved parent on an aligned lattice; it is not a "
            "finite-parent or process-domain independence test."),
    }


def _save_parent_overview(erosion_result, display_scores, display_role,
                          out: Path) -> dict:
    z = np.asarray(erosion_result["z"], np.float64)
    image = Image.fromarray(atlas_survey._terrain_rgb(z), "RGB")
    size = 768
    image = image.resize((size, size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    scale = size / PARENT_KM
    guard0 = int(round(NUMERICAL_GUARD_KM * scale))
    guard1 = int(round((PARENT_KM - NUMERICAL_GUARD_KM) * scale))
    draw.rectangle(
        (guard0, guard0, guard1, guard1),
        outline=(230, 210, 80), width=2)
    risk0 = int(round(STRUCTURAL_KINEMATIC_RIM_KM * scale))
    risk1 = int(round(
        (PARENT_KM - STRUCTURAL_KINEMATIC_RIM_KM) * scale))
    draw.rectangle(
        (risk0, risk0, risk1, risk1),
        outline=(255, 145, 60), width=2)
    colors = {
        "low": (80, 220, 255),
        "medium": (100, 255, 120),
        "high": (255, 100, 210),
    }
    for label, score in display_scores.items():
        x0 = int(round(score.x0_km * scale))
        y0 = int(round(score.y0_km * scale))
        x1 = int(round((score.x0_km + FRAME_KM) * scale))
        y1 = int(round((score.y0_km + FRAME_KM) * scale))
        draw.rectangle((x0, y0, x1, y1), outline=colors[label], width=4)
        text_label = (label if display_role == "accepted_joint_assignment"
                      else f"diagnostic {label}")
        draw.text((x0 + 5, y0 + 5), text_label, fill=colors[label])
    path = out / "seed137_parent_overview.png"
    if path.exists():
        raise ValueError(f"overview already exists: {path}")
    image.save(path)
    return {
        "file": path.name,
        "sha256": _sha256_file(path),
        "size_px": size,
        "yellow_rectangle": "interior outside the 2560-km candidate guard",
        "orange_rectangle": (
            "interior outside the derived 1980-km structural-risk band"),
        "display_role": display_role,
        "frame_colors": colors,
    }


def _render_chosen_frames(chosen_samples, cfg, out: Path,
                          display_role: str) -> dict:
    result = {}
    diagnostic = display_role != "accepted_joint_assignment"
    for label, _ in TARGETS:
        if label not in chosen_samples:
            continue
        sampled = chosen_samples[label]
        images = []
        for view in MAP_VIEWS:
            middle = f"diagnostic_{label}" if diagnostic else label
            path = out / f"seed137_{middle}_{view}_1024.png"
            if path.exists():
                raise ValueError(f"frame image already exists: {path}")
            render_map_view(
                sampled, view, cfg.river_density).save(path)
            images.append({
                "view": view,
                "file": path.name,
                "sha256": _sha256_file(path),
                "role": display_role,
            })
        result[label] = images
    return result


def _parent_outer_ring_report(erosion_result, base_level) -> dict:
    z = np.asarray(erosion_result["z"], np.float64)
    ring = _outer_ring_mask(z.shape)
    return {
        "ring_cells": int(np.count_nonzero(ring)),
        "present_ocean_z_below_zero_cells": int(np.count_nonzero(
            (z < 0.0) & ring)),
        "raw_lowstand_z_below_base_cells": int(np.count_nonzero(
            (z < float(base_level)) & ring)),
        "outer_process_ring_all_present_ocean": bool(np.all(z[ring] < 0.0)),
        "outer_process_ring_all_below_raw_lowstand": bool(np.all(
            z[ring] < float(base_level))),
    }


def _protocol(fingerprint, cfg, density) -> dict:
    origins = _candidate_origins()
    return {
        "experiment": EXPERIMENT,
        "manifest_role": "pre-execution sealed parent feasibility protocol",
        "source_fingerprint": fingerprint,
        "prior_artifact_expected_digests": PRIOR_ARTIFACTS,
        "fixed": {
            "seed": SEED,
            "parent_km": PARENT_KM,
            "delivered_frame_km": FRAME_KM,
            "structure_nominal_km": STRUCTURE_NOMINAL_KM,
            "process_nominal_km": PROCESS_NOMINAL_KM,
            "private_complete_config": asdict(cfg),
            "density_scaling": density,
            "structure_seeder": "stationary atlas _seed_atlas_nuclei",
            "process_call": (
                "run_erosion(structure, elevation, cfg, seed) with no "
                "routing/localization/window override"),
            "process_localization_mode_default": "legacy",
            "retries": 0,
        },
        "candidate_lattice": {
            "guard_km": NUMERICAL_GUARD_KM,
            "stride_km": CANDIDATE_STRIDE_KM,
            "origin_count": len(origins),
            "first_origin_xy_km": list(origins[0]),
            "last_origin_xy_km": list(origins[-1]),
            "inclusive_axis_km": [NUMERICAL_GUARD_KM, 5632.0],
        },
        "selection": {
            "called_only_after_full_parent_solve": True,
            "scan_size_px": SCAN_SIZE,
            "scan_border_policy": "exact outer ring of final water mask",
            "shortlist_per_target": SHORTLIST_PER_TARGET,
            "shortlist_union_max": (
                SHORTLIST_PER_TARGET * len(TARGETS)),
            "authority_size_px": AUTHORITY_SIZE,
            "authority_border_policy": (
                "evaluate_causal_border(sampled['water']) exact outer ring"),
            "land_fraction": (
                "mean(~sampled['water']), matching engine.report and the "
                "visible delivered land category"),
            "topographic_nonnegative_fraction": (
                "mean(sampled['h'] >= 0.0), reported as a diagnostic only"),
            "targets": {label: target for label, target in TARGETS},
            "absolute_target_tolerance": TARGET_TOLERANCE,
            "distinct_assignment": True,
            "pairwise_origin_chebyshev_separation_km": (
                MIN_ORIGIN_CHEBYSHEV_SEPARATION_KM),
            "joint_objective_order": [
                "minimum maximum error normalized by frozen tolerance",
                "minimum high absolute error",
                "minimum medium absolute error",
                "minimum low absolute error",
                "minimum sum absolute target error",
                "minimum deterministic assignment hash"],
            "high_target_additional_eligibility": (
                "final land_fraction < 0.50 and final categorical "
                "water_fraction > 0.50 so water remains dominant"),
            "greedy_high_first_selection": False,
            "contour_or_frame_alignment_score": None,
            "candidate_or_control_retry": None,
            "failure_behavior": (
                "never abort or widen K/tolerance/retry; serialize every "
                "coarse and authority score and render the independently "
                "closest exact-water authority candidate per target, when "
                "one exists, explicitly as diagnostic-not-accepted"),
        },
        "instrumentation": {
            "pass_through_legacy_route_sediment_calls": 1,
            "capture_budget_and_full_MFD_plus_D8_fallback_graph": True,
            "restore_before_selector": True,
            "source_or_engine_modified": False,
        },
        "causal_audits": {
            "candidate_guard_km": NUMERICAL_GUARD_KM,
            "derived_structural_kinematic_band_km": (
                STRUCTURAL_KINEMATIC_RIM_KM),
            "guard_beyond_structural_band_km": (
                STRUCTURAL_GUARD_REMAINDER_KM),
            "derivation": "2.2 * 45 * 20 km = 1,980 km",
            "captured_legacy_route_stages": (
                EXPECTED_LEGACY_ROUTE_GRAPH_STAGES),
            "crop_influence_support_process_cells": 6,
            "crop_influence_support_components": (
                "4 cells for 2 creep substeps x 2 erosion steps plus "
                "2 cells of Catmull-Rom sampling support"),
            "band_ancestor_flag_propagated_down_each_full_flow_graph": (
                "positive MFD edges plus D8 fallback on flat cells"),
            "crop_land_path_requirement": (
                "the concentrated D8 receiver used by incision downstream "
                "elevation and sediment transport reaches raw z < lowstand "
                "before the structural-risk band at every captured stage"),
            "conservative_exclusion_screen_not_causation_test": True,
            "failure_interpretation": (
                "A band-ancestry failure means this conservative screen "
                "cannot exclude parent-rim influence; it does not prove "
                "that a naturally routed contributor was boundary-caused."),
            "legacy_fill_finite_reach_claimed": False,
            "finite_parent_independence_claimed": False,
            "same_parent_overlap_independence_claimed": False,
        },
        "acceptance": {
            "exact_terminal_residual_zero": True,
            "sediment_closure_relative_tolerance": (
                SEDIMENT_CLOSURE_RELATIVE_TOLERANCE),
            "outer_process_ring_ocean_reported": True,
            "three_authoritative_water_bordered_targets": True,
            "mask_river_overlap_witness_each_frame": (
                "diagnostic only; not a causal or availability gate"),
            "parent_overview_and_five_existing_views_each_frame": True,
            "manual_morphology_review_required": True,
            "manual_morphology_default": "unreviewed_false",
        },
        "outcome_logic": {
            "availability_pass": (
                "three distinct final-1024 exact-water-ring frames, all "
                "within target tolerance, high land fraction below 0.50, "
                "high categorical water fraction above 0.50, "
                "and exact aligned same-parent overlaps"),
            "causal_screen_pass": (
                "order/frame-blind/guard support plus all four graph screens, "
                "sediment closure, and exact terminal zero"),
            "architectural_feasibility_pass": (
                "availability_pass and causal_screen_pass"),
            "section3b_status": "unresolved_single_domain",
            "promotion": False,
        },
        "sequencing": {
            "structure_builds": EXPECTED_STRUCTURE_BUILDS,
            "coarse_elevation_builds": EXPECTED_ELEVATION_BUILDS,
            "full_parent_erosion_solves": EXPECTED_EROSION_SOLVES,
            "legacy_route_sediment_calls": (
                EXPECTED_LEGACY_ROUTE_SEDIMENT_CALLS),
            "captured_legacy_route_graph_stages": (
                EXPECTED_LEGACY_ROUTE_GRAPH_STAGES),
            "selector_calls": EXPECTED_SELECTOR_CALLS,
        },
        "decision_boundary": {
            "private_diagnostic_only": True,
            "public_budget_range_proven": False,
            "finite_parent_or_process_independence_proven": False,
            "promotion_assessed": False,
        },
    }


def _serialize_score(score: CandidateScore) -> dict:
    return asdict(score)


def _run(out: Path) -> dict:
    _prepare_empty_output(out)
    cfg, density = _private_parent_config()
    fingerprint = _source_fingerprint()
    protocol_sha256 = _write_json_exclusive(
        out / "protocol_precommit.json", _protocol(fingerprint, cfg, density))
    prior_links = _prior_links()
    mismatched = {
        name: value for name, value in prior_links.items()
        if not value["digest_matched"]}
    if mismatched:
        raise RuntimeError(f"historical prior evidence changed: {mismatched}")

    events = ["protocol_precommit_written", "prior_digests_verified"]
    counters = {
        "structure_builds": 0,
        "coarse_elevation_builds": 0,
        "full_parent_erosion_solves": 0,
        "selector_calls": 0,
    }
    started = time.perf_counter()
    structure = build_structure(
        SEED, cfg,
        _world_km=PARENT_KM,
        _coarse_km=STRUCTURE_NOMINAL_KM,
        _continent_seeder=atlas_survey._seed_atlas_nuclei)
    counters["structure_builds"] += 1
    events.append("structure_complete")
    elevation = coarse_elevation(structure, cfg, SEED)
    counters["coarse_elevation_builds"] += 1
    events.append("coarse_elevation_complete")

    engine_functions_before = {
        "route_sediment": erosion_engine.route_sediment,
        "fill_depressions": erosion_engine.fill_depressions,
        "receivers": erosion_engine.receivers,
        "topo_batches": erosion_engine.topo_batches,
    }
    process_args = (structure, elevation, cfg, SEED)
    process_kwargs = {}
    process_call_observation = {
        "positional_arg_count": len(process_args),
        "structure_identity_exact": process_args[0] is structure,
        "elevation_identity_exact": process_args[1] is elevation,
        "config_identity_exact": process_args[2] is cfg,
        "seed_exact": process_args[3] == SEED,
        "keyword_arguments": dict(process_kwargs),
        "crop_or_process_window_keyword_present": bool(
            {"_process_window", "_frame_window_km"}
            & set(process_kwargs)),
    }
    with LegacySedimentInstrumentation() as instrumentation:
        erosion_result = erosion_engine.run_erosion(
            *process_args, **process_kwargs)
        counters["full_parent_erosion_solves"] += 1
    process_call_observation.update({
        "result_process_window_present": "process_window" in erosion_result,
        "result_process_origin_present": (
            "process_origin_km" in erosion_result),
        "observed_frame_blind_full_parent_call": bool(
            len(process_args) == 4
            and process_args[0] is structure
            and process_args[1] is elevation
            and process_args[2] is cfg
            and process_args[3] == SEED
            and not process_kwargs
            and "process_window" not in erosion_result
            and "process_origin_km" not in erosion_result),
    })
    engine_functions_restored = {
        name: getattr(erosion_engine, name) is function
        for name, function in engine_functions_before.items()}
    events.append("full_parent_legacy_process_complete")
    if instrumentation.capture is None:
        raise AssertionError("legacy route_sediment was not captured")
    capture = instrumentation.capture

    counters["selector_calls"] += 1
    events.append("selector_started_after_full_solve")
    selection = _scan_and_select(
        structure, elevation, erosion_result, cfg)
    events.append("selector_authority_and_joint_assignment_complete")
    assignment = selection["assignment"]["assignment"]
    accepted_samples = selection["accepted_samples"]
    render_scores = selection["render_scores"]
    render_samples = selection["render_samples"]
    render_role = selection["render_role"]

    budget = _sediment_budget(capture, erosion_result)
    sediment_stage_oracle = _sediment_stage_oracle(
        capture, instrumentation.route_stages)
    graph_audit = _graph_causal_audit(
        instrumentation.route_stages, capture.base_level_m,
        capture.process_spacing_km, assignment)
    diagnostic_graph_audit = None
    if not selection["assignment"]["found"]:
        diagnostic_graph_audit = _graph_causal_audit(
            instrumentation.route_stages, capture.base_level_m,
            capture.process_spacing_km, render_scores)
    river_witness = _river_overlap_witness(
        erosion_result, capture, assignment)
    overlap_audit = _authority_overlap_audit(
        accepted_samples, assignment)
    parent_ring = _parent_outer_ring_report(
        erosion_result, capture.base_level_m)
    overview = _save_parent_overview(
        erosion_result, render_scores, render_role, out)
    rendered_frames = _render_chosen_frames(
        render_samples, cfg, out, render_role)
    events.append("rendering_complete")

    authority_by_origin = {
        score.origin: score for score in selection["authority_scores"]}
    selected_report = {}
    for label, target in TARGETS:
        score = assignment.get(label)
        if score is None:
            continue
        sampled = accepted_samples[label]
        border = evaluate_causal_border(sampled["water"])
        selected_report[label] = {
            "target_land_fraction": target,
            "absolute_tolerance": TARGET_TOLERANCE,
            "score": _serialize_score(score),
            "absolute_target_error": abs(score.land_fraction - target),
            "causal_border": border,
            "ring_composition": {
                "water_cells": score.ring_water_cells,
                "ocean_cells": score.ring_ocean_cells,
                "lake_cells": score.ring_lake_cells,
                "non_water_cells": score.ring_non_water_cells,
            },
            "images": rendered_frames.get(label, []),
        }

    diagnostic_best_report = {}
    for label, target in TARGETS:
        score = selection["diagnostic_best_available"][label]
        if score is None:
            diagnostic_best_report[label] = {
                "available": False,
                "accepted": False,
                "reason": (
                    "no exact-final-water-ring candidate existed in the "
                    "frozen final authority shortlist"),
                "images": [],
            }
            continue
        diagnostic_best_report[label] = {
            "available": True,
            "accepted": False,
            "independent_per_target_rank_may_repeat_origin": True,
            "target_land_fraction": target,
            "absolute_target_error": abs(score.land_fraction - target),
            "within_frozen_tolerance": bool(
                abs(score.land_fraction - target) <= TARGET_TOLERANCE),
            "high_water_dominance_eligible": bool(
                label != "high"
                or (score.land_fraction < 0.50
                    and score.water_fraction > 0.50)),
            "score": _serialize_score(score),
            "rendered_because_joint_assignment_failed": bool(
                render_role == "diagnostic_best_available_not_accepted"),
            "images": (rendered_frames.get(label, [])
                       if render_role
                       == "diagnostic_best_available_not_accepted" else []),
        }

    expected_event_order = (
        events.index("full_parent_legacy_process_complete")
        < events.index("selector_started_after_full_solve")
        < events.index("selector_authority_and_joint_assignment_complete")
        < events.index("rendering_complete"))
    engine_default_legacy = (
        inspect.signature(erosion_engine.run_erosion).parameters[
            "_localization_mode"].default == "legacy"
        and inspect.signature(erosion_engine.run_erosion).parameters[
            "_routing_mode"].default == "legacy")
    scan_scores = selection["scan_scores"]
    shortlist = selection["shortlist"]
    authority_scores = selection["authority_scores"]
    selected_complete = len(assignment) == EXPECTED_FRAME_COUNT
    accepted_images_complete = (
        selected_complete
        and render_role == "accepted_joint_assignment"
        and all(len(rendered_frames.get(label, [])) == len(MAP_VIEWS)
                for label, _ in TARGETS))
    expected_render_image_count = len(render_scores) * len(MAP_VIEWS)
    observed_render_image_count = sum(
        len(images) for images in rendered_frames.values())
    render_artifacts_complete = bool(
        set(rendered_frames) == set(render_scores)
        and observed_render_image_count == expected_render_image_count
        and all(len(images) == len(MAP_VIEWS)
                for images in rendered_frames.values()))

    integrity_checks = {
        "prior_artifact_digests_matched": not bool(mismatched),
        "private_density_exactly_17_plates_7_nuclei": bool(
            cfg.plates == 17 and cfg.nuclei == 7),
        "private_continental_budget_exactly_0_65": bool(
            cfg.continental_budget == PRIVATE_CONTINENTAL_BUDGET),
        "structure_grid_exact_307": structure.n == EXPECTED_STRUCTURE_N,
        "process_grid_exact_614": tuple(erosion_result["z"].shape)
        == (EXPECTED_PROCESS_N, EXPECTED_PROCESS_N),
        "structure_actual_spacing_exact": bool(np.isclose(
            structure.world_km / structure.n,
            EXPECTED_STRUCTURE_ACTUAL_KM, rtol=0.0, atol=1e-12)),
        "process_actual_spacing_exact": bool(np.isclose(
            float(erosion_result["e_km"]), EXPECTED_PROCESS_ACTUAL_KM,
            rtol=0.0, atol=1e-12)),
        "structure_build_count_exact": (
            counters["structure_builds"] == EXPECTED_STRUCTURE_BUILDS),
        "coarse_elevation_build_count_exact": (
            counters["coarse_elevation_builds"]
            == EXPECTED_ELEVATION_BUILDS),
        "full_parent_erosion_solve_count_exact": (
            counters["full_parent_erosion_solves"]
            == EXPECTED_EROSION_SOLVES),
        "legacy_route_sediment_call_count_exact": (
            instrumentation.call_count
            == EXPECTED_LEGACY_ROUTE_SEDIMENT_CALLS),
        "selector_call_count_exact": (
            counters["selector_calls"] == EXPECTED_SELECTOR_CALLS),
        "selector_strictly_after_solve": expected_event_order,
        "observed_frame_blind_full_parent_process_call": (
            process_call_observation[
                "observed_frame_blind_full_parent_call"]),
        "candidate_lattice_exact_169": len(scan_scores) == 169,
        "shortlist_union_bounded_at_24": (
            len(shortlist["union"])
            <= SHORTLIST_PER_TARGET * len(TARGETS)),
        "authority_only_rescores_shortlist_union": (
            len(authority_scores) == len(shortlist["union"])
            and set(authority_by_origin)
            == {score.origin for score in shortlist["union"]}),
        "all_instrumented_engine_functions_restored": all(
            engine_functions_restored.values()),
        "engine_defaults_remain_legacy": engine_default_legacy,
        "captured_sediment_inputs_unchanged": capture.inputs_unchanged,
        "sediment_call_matches_third_route_stage_exactly": (
            sediment_stage_oracle["passed"]),
        "rendered_artifacts_complete_for_available_display_candidates": (
            render_artifacts_complete),
        "overview_written": bool((out / overview["file"]).is_file()),
    }
    if not all(integrity_checks.values()):
        raise AssertionError({"integrity_checks": integrity_checks})

    availability_gates = {
        "joint_authoritative_assignment_found": bool(
            selection["assignment"]["found"] and selected_complete),
        "three_distinct_origins": bool(
            selected_complete
            and len({score.origin for score in assignment.values()})
            == EXPECTED_FRAME_COUNT),
        "all_pairs_meet_frozen_separation": bool(
            selected_complete
            and all(_separated(left, right)
                    for left, right in itertools.combinations(
                        assignment.values(), 2))),
        "all_authority_land_fractions_within_tolerance": bool(
            selected_complete
            and all(abs(assignment[label].land_fraction - target)
                    <= TARGET_TOLERANCE
                    for label, target in TARGETS)),
        "high_authority_land_fraction_strictly_below_0_50": bool(
            selected_complete and assignment["high"].land_fraction < 0.50),
        "high_authority_categorical_water_fraction_strictly_above_0_50": bool(
            selected_complete and assignment["high"].water_fraction > 0.50),
        "all_authority_outer_rings_exact_water": bool(
            selected_complete
            and all(assignment[label].border_passed
                    for label, _ in TARGETS)),
        "same_parent_authority_overlap_exact": overlap_audit[
            "all_overlapping_authority_fields_exact"],
        "all_three_same_parent_pairs_overlap_at_least_0_0625": (
            overlap_audit["all_pairs_overlap_at_least_0_0625"]),
    }
    availability_pass = all(availability_gates.values())
    causal_screen_gates = {
        "selector_strictly_after_one_full_parent_solve": bool(
            expected_event_order
            and counters["full_parent_erosion_solves"] == 1),
        "full_parent_process_call_frame_blind_no_window_override": (
            process_call_observation[
                "observed_frame_blind_full_parent_call"]),
        "candidate_guard_exceeds_structural_band_by_580_km": bool(
            STRUCTURAL_GUARD_REMAINDER_KM == 580.0),
        "outer_process_ring_all_present_ocean": parent_ring[
            "outer_process_ring_all_present_ocean"],
        "all_four_legacy_route_stages_captured": graph_audit[
            "all_expected_stages_captured"],
        "all_stages_zero_1980km_band_ancestry": graph_audit[
            "all_stages_all_frames_zero_structural_band_ancestry"],
        "all_stage_concentrated_d8_land_paths_hit_raw_z_below_minus80_before_band": (
            graph_audit[
                "all_stages_all_frame_concentrated_d8_land_paths_reach_strict_lowstand_before_band"]),
        "sediment_mass_closure_within_scaled_1e_minus_12": budget[
            "closure_passed"],
        "explicit_terminal_residual_zero": budget["terminal_zero_passed"],
        "all_sediment_budget_validity_checks_pass": budget[
            "all_validity_checks_passed"],
    }
    causal_screen_pass = all(causal_screen_gates.values())
    architectural_feasibility_pass = bool(
        availability_pass and causal_screen_pass)
    manual_morphology_review_passed = False
    overall_passed = bool(
        architectural_feasibility_pass and manual_morphology_review_passed)

    report = {
        "experiment": EXPERIMENT,
        "completed": True,
        "protocol_precommit_sha256": protocol_sha256,
        "source_fingerprint": fingerprint,
        "prior_artifacts": prior_links,
        "fixed": {
            "seed": SEED,
            "parent_km": PARENT_KM,
            "structure_nominal_km": STRUCTURE_NOMINAL_KM,
            "structure_actual_km": structure.world_km / structure.n,
            "process_nominal_km": PROCESS_NOMINAL_KM,
            "process_actual_km": float(erosion_result["e_km"]),
            "complete_private_config": asdict(cfg),
            "density_scaling": density,
            "retries": 0,
        },
        "sequencing_events": events,
        "execution_counters": {
            **counters,
            "legacy_route_sediment_calls": instrumentation.call_count,
            "captured_legacy_route_graph_stages": len(
                instrumentation.route_stages),
        },
        "observed_parent_process_call": process_call_observation,
        "timings": {
            "structure": structure.timings,
            "erosion": erosion_result["timings"],
            "elapsed_total_s": time.perf_counter() - started,
        },
        "parent_fields": {
            "coarse_elevation_h": _array_summary(elevation["h"]),
            "final_process_z": _array_summary(erosion_result["z"]),
            "final_process_sediment": _array_summary(erosion_result["sed"]),
            "final_discharge_log": _array_summary(
                erosion_result["discharge_log"]),
        },
        "sediment_budget": budget,
        "parent_outer_process_ring": parent_ring,
        "selection": {
            "scan_size_px": SCAN_SIZE,
            "scan_scores": [_serialize_score(score)
                            for score in scan_scores],
            "water_safe_scan_count": int(sum(
                score.border_passed for score in scan_scores)),
            "shortlist_by_target": {
                label: [_serialize_score(score) for score in scores]
                for label, scores in shortlist["by_target"].items()},
            "shortlist_union": [_serialize_score(score)
                                for score in shortlist["union"]],
            "authority_size_px": AUTHORITY_SIZE,
            "authority_scores": [_serialize_score(score)
                                 for score in authority_scores],
            "joint_assignment": {
                "found": selection["assignment"]["found"],
                "assignment": {
                    label: _serialize_score(score)
                    for label, score in assignment.items()},
                "errors": selection["assignment"]["errors"],
                "objective": selection["assignment"]["objective"],
                "viable_assignment_count": selection[
                    "assignment"]["viable_assignment_count"],
                "pool_counts": selection["assignment"]["pool_counts"],
            },
            "selected_authority_frames": selected_report,
            "diagnostic_best_available_by_target": (
                diagnostic_best_report),
            "render_role": render_role,
            "failure_semantics": (
                "A frozen K=8/target shortlist failure is one-sided: it "
                "does not prove the parent lacks a qualifying triple. No "
                "K/tolerance/seed expansion or retry is performed."),
            "contour_or_frame_alignment_score_used": False,
            "retries": 0,
        },
        "sediment_call_to_route_stage_oracle": sediment_stage_oracle,
        "captured_legacy_graph_causal_audit": graph_audit,
        "diagnostic_render_frame_graph_audit_if_no_assignment": (
            diagnostic_graph_audit),
        "mask_river_overlap_witness": river_witness,
        "same_parent_overlap_audit": overlap_audit,
        "rendered_artifacts": {
            "parent_overview": overview,
            "frame_views": rendered_frames,
            "display_role": render_role,
            "expected_frame_image_count": expected_render_image_count,
            "observed_frame_image_count": observed_render_image_count,
        },
        "integrity_checks": integrity_checks,
        "availability_gates": availability_gates,
        "availability_pass": availability_pass,
        "causal_screen_gates": causal_screen_gates,
        "causal_screen_pass": causal_screen_pass,
        "architectural_feasibility_pass": architectural_feasibility_pass,
        "section3b_status": "unresolved_single_domain",
        "manual_morphology_review": {
            "status": "unreviewed",
            "passed": manual_morphology_review_passed,
            "required_artifacts": {
                "parent_overview": overview["file"],
                "five_existing_views_per_frame": list(MAP_VIEWS),
                "frame_artifact_role": render_role,
            },
            "review_targets": [
                "natural-looking coast and landmass distribution",
                "absence of obvious parent-rim artifacts",
                "mountain and bathymetry morphology",
                "river/lake coherence",
                "sediment concentration and edge behavior",
            ],
        },
        "overall_feasibility_passed": overall_passed,
        "interpretation_limits": [
            "This is one seed and one finite parent with no retry.",
            "The private 0.65 continental budget exceeds the current public 0.45 maximum.",
            "A full parent solve removes localized process-window dependence but does not prove independence from the parent numerical rim.",
            "Legacy depression fill is seeded by the numerical rim and legacy marine sediment has an unbounded exponential tail.",
            "The 1,980-km graph audit is a conservative exclusion screen, not a causation test: a pass is limited evidence and a failure is inconclusive; the separate candidate guard is 2,560 km.",
            "Exact overlapping crop fields are guaranteed by same-parent sampling and are not an independence result.",
            "No contour shape or frame-alignment metric participates in eligibility or ranking.",
            "A shortlist pass proves one qualifying same-parent triple exists under this frozen screen; failure does not prove none exists outside K=8.",
        ],
        "decision_boundary": {
            "private_diagnostic_only": True,
            "engine_or_default_changed": False,
            "public_slider_range_proven": False,
            "finite_parent_independence_proven": False,
            "section3b_status": "unresolved_single_domain",
            "promotion_assessed": False,
            "promotion": False,
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
        "availability_pass": availability_pass,
        "causal_screen_pass": causal_screen_pass,
        "architectural_feasibility_pass": architectural_feasibility_pass,
        "section3b_status": "unresolved_single_domain",
        "manual_morphology_review_passed": manual_morphology_review_passed,
        "overall_feasibility_passed": overall_passed,
        "elapsed_s": report["elapsed_s"],
    }


def _synthetic_score(origin, land_fraction, *, passed=True, size=1024):
    ring_count = 4 * size - 4
    return CandidateScore(
        x0_km=float(origin[0]), y0_km=float(origin[1]), size_px=size,
        land_fraction=float(land_fraction), border_passed=bool(passed),
        water_fraction=float(1.0 - land_fraction),
        topographic_nonnegative_fraction=float(land_fraction),
        ring_cell_count=ring_count,
        ring_water_cells=ring_count if passed else ring_count - 1,
        ring_ocean_cells=ring_count if passed else ring_count - 1,
        ring_lake_cells=0,
        ring_non_water_cells=0 if passed else 1,
        nearest_land_to_border_km=(FRAME_KM / size if passed else 0.0),
        tie_key=_candidate_tie_key(origin),
    )


def _self_check() -> dict:
    cfg, density = _private_parent_config()
    origins = _candidate_origins()

    water = np.ones((6, 8), bool)
    water[1:-1, 1:-1] = False
    ocean = water.copy()
    lake = np.zeros_like(water)
    h = np.full(water.shape, -10.0, np.float32)
    h[1:-1, 1:-1] = 100.0
    synthetic_sample = {
        "water": water, "ocean": ocean, "lake": lake, "h": h}
    score = _score_sample(synthetic_sample, origins[0])
    water_bad = water.copy()
    water_bad[0, 3] = False
    sample_bad = {**synthetic_sample, "water": water_bad}
    bad_score = _score_sample(sample_bad, origins[1])

    scan_scores = [
        _synthetic_score((2560.0, 2560.0), 0.20, size=256),
        _synthetic_score((4608.0, 2560.0), 0.35, size=256),
        _synthetic_score((2560.0, 4608.0), 0.50, size=256),
        _synthetic_score((5632.0, 5632.0), 0.36, size=256),
        _synthetic_score((2816.0, 2816.0), 0.19, passed=False, size=256),
    ]
    shortlist = _bounded_shortlist(scan_scores)
    authority = [
        _synthetic_score((2560.0, 2560.0), 0.201),
        _synthetic_score((4608.0, 2560.0), 0.348),
        _synthetic_score((2560.0, 4608.0), 0.497),
        _synthetic_score((5632.0, 5632.0), 0.36),
    ]
    assignment = _joint_assignment(authority)
    chosen = assignment["assignment"]
    overlap_fields = (
        "h", "hc", "water", "ocean", "lake", "lake_level",
        "riv_log", "sed")
    overlap_samples = {
        label: {
            field: np.full(
                (AUTHORITY_SIZE, AUTHORITY_SIZE),
                field in ("water", "ocean"),
                dtype=(bool if field in ("water", "ocean", "lake")
                       else np.float32))
            for field in overlap_fields}
        for label in chosen}
    synthetic_overlap = _authority_overlap_audit(
        overlap_samples, chosen)

    shape = (7, 7)
    n = shape[0] * shape[1]
    receiver = np.arange(n, dtype=np.int64)
    center = 3 * shape[1] + 3
    lowstand_cell = 3 * shape[1] + 4
    rim_cell = 3 * shape[1] + 6
    receiver[center] = lowstand_cell
    receiver[lowstand_cell] = rim_cell
    batches = (
        np.asarray([center], np.int64),
        np.asarray([lowstand_cell], np.int64),
        np.asarray([rim_cell], np.int64),
        np.asarray([value for value in range(n)
                    if value not in (center, lowstand_cell, rim_cell)],
                   np.int64),
    )
    z = np.full(shape, 10.0)
    z.flat[lowstand_cell] = -100.0
    rim = _rim_mask(shape, 1.0, rim_km=1.0)
    status = _classify_paths_before_rim(
        z, -80.0, receiver, batches, rim)

    contaminated_receiver = np.arange(n, dtype=np.int64)
    rim_source = 3 * shape[1]
    contaminated_receiver[rim_source] = center
    contamination_batches = (
        np.asarray([rim_source], np.int64),
        np.asarray([center], np.int64),
        np.asarray([value for value in range(n)
                    if value not in (rim_source, center)], np.int64),
    )
    contamination = _propagate_rim_ancestor(
        contaminated_receiver, contamination_batches, rim)

    # The full-flow oracle must catch ancestry carried by a positive MFD
    # branch even when the concentrated D8 receiver does not use that edge.
    mfd_receiver = np.arange(n, dtype=np.int64)
    mfd_targets = np.tile(np.arange(n, dtype=np.int64), (8, 1))
    mfd_weights = np.zeros((8, n), np.float64)
    mfd_targets[0, rim_source] = center
    mfd_weights[0, rim_source] = 1.0
    mfd_flat = mfd_weights.sum(axis=0) <= 0.0
    mfd_contamination = _propagate_full_flow_rim_ancestor(
        mfd_receiver, mfd_targets, mfd_weights, mfd_flat,
        contamination_batches, rim)
    mfd_d8_only = _propagate_rim_ancestor(
        mfd_receiver, contamination_batches, rim)

    instrument_function_before = erosion_engine.route_sediment
    test_z = np.full((3, 3), -100.0)
    test_z[1, 1] = 10.0
    test_erosion = np.zeros((3, 3))
    test_erosion[1, 1] = 1.0
    test_receiver = np.arange(9, dtype=np.int64)
    test_receiver[4] = 5
    # Source and receiver cannot share a Kahn batch; the receiving cell must
    # be processed later for the synthetic mass budget to be meaningful.
    test_batches = (
        np.asarray([4], np.int64),
        np.asarray([0, 1, 2, 3, 5, 6, 7, 8], np.int64),
    )
    test_area = np.ones(9)
    with LegacySedimentInstrumentation() as instrumentation:
        routed = erosion_engine.route_sediment(
            test_z, test_erosion, test_receiver, test_batches,
            test_area, -80.0, 180.0, 20.0)
    instrument_restored = (
        erosion_engine.route_sediment is instrument_function_before)
    fake_result = {
        "z": routed[0],
        "sed": routed[1],
        "ero": test_erosion,
        "sediment_export_m3": routed[2] * (20_000.0 ** 2),
        "sediment_terminal_residual_m3": routed[3] * (20_000.0 ** 2),
    }
    budget = _sediment_budget(instrumentation.capture, fake_result)

    protocol = _protocol(
        {"combined_sha256": "synthetic", "files": {}}, cfg, density)
    protocol_serializable = True
    try:
        json.dumps(protocol, allow_nan=False, default=_json_default)
    except (TypeError, ValueError):
        protocol_serializable = False

    checks = {
        "private_config_density_exact": bool(
            cfg.plates == 17 and cfg.nuclei == 7
            and abs(density["parent_to_reference_area_ratio"]
                    - 2.4930747922437675) < 1e-15),
        "private_budget_exact_and_above_public_max": bool(
            cfg.continental_budget == 0.65
            and cfg.continental_budget > PUBLIC_CONTINENTAL_BUDGET_MAX),
        "candidate_lattice_exact": bool(
            len(origins) == 169
            and origins[0] == (2560.0, 2560.0)
            and origins[-1] == (5632.0, 5632.0)),
        "kinematic_band_and_guard_remainder_exact": bool(
            STRUCTURAL_KINEMATIC_RIM_KM == 1980.0
            and STRUCTURAL_GUARD_REMAINDER_KM == 580.0),
        "causal_water_ring_accepts_interior_land": bool(
            score.border_passed and score.ring_non_water_cells == 0
            and score.land_fraction == 0.5),
        "causal_water_ring_rejects_one_border_land_cell": bool(
            not bad_score.border_passed
            and bad_score.ring_non_water_cells == 1),
        "shortlist_excludes_failed_water_border": bool(
            all(value.border_passed for value in shortlist["union"])
            and len(shortlist["union"]) <= 24),
        "joint_assignment_found_distinct_separated_targets": bool(
            assignment["found"] and len(chosen) == 3
            and len({value.origin for value in chosen.values()}) == 3
            and all(_separated(left, right)
                    for left, right in itertools.combinations(
                        chosen.values(), 2))),
        "joint_assignment_all_errors_within_tolerance": bool(
            assignment["found"]
            and all(error <= TARGET_TOLERANCE
                    for error in assignment["errors"].values())),
        "same_parent_overlap_all_required_fields_exact": bool(
            synthetic_overlap["all_overlapping_authority_fields_exact"]
            and synthetic_overlap["all_three_pairs_reported"]),
        "raw_lowstand_before_rim_path_classified": bool(
            status.flat[center] == PATH_LOWSTAND_BEFORE_RIM),
        "rim_ancestor_propagation_detects_inward_path": bool(
            contamination.flat[center]),
        "full_flow_ancestry_detects_mfd_only_branch": bool(
            mfd_contamination.flat[center]
            and not mfd_d8_only.flat[center]),
        "route_instrumentation_exactly_one_call_and_restored": bool(
            instrumentation.call_count == 1 and instrument_restored),
        "route_instrumentation_inputs_unchanged": bool(
            instrumentation.capture is not None
            and instrumentation.capture.inputs_unchanged),
        "synthetic_sediment_budget_closed": budget["closure_passed"],
        "protocol_json_serializable": protocol_serializable,
        "no_contour_or_frame_alignment_score_in_protocol": bool(
            protocol["selection"][
                "contour_or_frame_alignment_score"] is None),
        "engine_default_localization_remains_legacy": (
            inspect.signature(erosion_engine.run_erosion).parameters[
                "_localization_mode"].default == "legacy"),
    }
    passed = all(checks.values())
    result = {
        "experiment": EXPERIMENT + "-self-check",
        "full_generation_executed": False,
        "seed_137_spent": False,
        "production_sediment_kernel_executed_on_synthetic_3x3_fixture": True,
        "checks": checks,
        "measurements": {
            "parent_area_ratio": density[
                "parent_to_reference_area_ratio"],
            "candidate_origin_count": len(origins),
            "shortlist_union_count": len(shortlist["union"]),
            "joint_viable_assignment_count": assignment[
                "viable_assignment_count"],
            "synthetic_budget_closure_m_cells": budget[
                "closure_m_cells"],
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
        default=Path("out") / "parent_solve_feasibility_seed137_v1")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    result = _self_check() if args.self_check else _run(args.out)
    print(json.dumps(
        result, indent=2, allow_nan=False, default=_json_default))


if __name__ == "__main__":
    main()
