"""One-shot field-accretion parent/crop-last feasibility experiment.

This private harness answers one availability question: after an unchanged
full M3 solve of one naturally formed parent, do three separated 4096-km
windows exist whose exact final categorical-water masks satisfy the frozen
20/35/50-percent composition bands and an all-water outer ring?

Formation never receives a crop or target.  Selection begins only after the
single parent process solve.  A pass is not promotion evidence: the finite
legacy parent still leaves section 3b numerical-boundary independence
unresolved.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import hashlib
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
from engine.tectonics import FRAME_KM, build_structure
from spikes import field_accretion_oracle as formation
from spikes import field_accretion_parent_mosaic as mosaic_engine


EXPERIMENT = "field-accretion-parent-feasibility-seed138-v1"
SEED = 138
PARENT_KM = 6.0 * FRAME_KM
STRUCTURE_NOMINAL_KM = 40.0
PRIVATE_CONTINENTAL_BUDGET = 0.65
NUMERICAL_GUARD_KM = 2560.0
CANDIDATE_STRIDE_KM = 256.0
AUTHORITY_SIZE = 1024
AUTHORITY_KM_PER_PX = FRAME_KM / AUTHORITY_SIZE
MOSAIC_CHUNK_ROWS = 128
SHORTLIST_PER_TARGET = 8
TARGET_TOLERANCE = 0.05
TARGETS = (("low", 0.20), ("medium", 0.35), ("high", 0.50))
MIN_ORIGIN_CHEBYSHEV_SEPARATION_KM = 0.5 * FRAME_KM
SEDIMENT_CLOSURE_RELATIVE_TOLERANCE = 1e-12

SOURCE_FILES = (
    "engine/elevation.py",
    "engine/erosion.py",
    "engine/noise.py",
    "engine/render_map.py",
    "engine/rng.py",
    "engine/surface.py",
    "engine/tectonics.py",
    "spikes/atlas_survey.py",
    "spikes/field_accretion_oracle.py",
    "spikes/field_accretion_parent_mosaic.py",
    "spikes/field_accretion_parent_feasibility.py",
    "spikes/parent_solve_feasibility.py",
    "spikes/visible_contour_gate.py",
)

PRIOR_EVIDENCE = {
    "parent_solve_harness": (
        "spikes/parent_solve_feasibility.py",
        "f7911f253e8ba27dfec49ddcfc84667c6b2611f6f64fc557e7c957e08dd8a2d0",
    ),
    "parent_solve_report": (
        "../out/parent_solve_feasibility_seed137_v1/report.json",
        "0c6de3bbb439995b1ca5cb9d9400808315a5e9f6178f6e8c38afc1661aad76ca",
    ),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_fingerprint() -> dict:
    root = _root()
    digest = hashlib.sha256()
    files = {}
    for relative in SOURCE_FILES:
        payload = (root / relative).read_bytes()
        value = _sha256_bytes(payload)
        files[relative] = value
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return {"combined_sha256": digest.hexdigest(), "files": files}


def _verify_prior_evidence() -> dict:
    root = _root()
    records = {}
    for label, (relative, expected) in PRIOR_EVIDENCE.items():
        actual = _sha256_file((root / relative).resolve())
        records[label] = {
            "file": relative,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": actual == expected,
        }
    if not all(item["matched"] for item in records.values()):
        raise RuntimeError(f"historical evidence changed: {records}")
    return records


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_json_exclusive(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True,
                          default=_json_default) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
    return _sha256_bytes(encoded)


def _prepare_empty_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(path)
        if any(path.iterdir()):
            raise FileExistsError(f"output directory is not empty: {path}")
    else:
        path.mkdir(parents=True)


def _candidate_axis() -> np.ndarray:
    available = PARENT_KM - 2.0 * NUMERICAL_GUARD_KM - FRAME_KM
    steps = int(round(available / CANDIDATE_STRIDE_KM))
    if steps * CANDIDATE_STRIDE_KM != available:
        raise AssertionError("candidate lattice is not exact")
    return NUMERICAL_GUARD_KM + CANDIDATE_STRIDE_KM * np.arange(steps + 1)


def _candidate_origins() -> list[tuple[float, float]]:
    axis = _candidate_axis()
    return [(float(x), float(y)) for y in axis for x in axis]


def _protocol() -> dict:
    cfg = formation._atlas_config(PRIVATE_CONTINENTAL_BUDGET)
    axis = _candidate_axis()
    union_span = float(axis[-1] + FRAME_KM - axis[0])
    return {
        "experiment": EXPERIMENT,
        "manifest_role": "exclusive_pre_generation_protocol_precommit",
        "source_fingerprint": _source_fingerprint(),
        "prior_evidence": _verify_prior_evidence(),
        "seed_policy": {
            "seed": SEED,
            "selection": "mechanical next integer after sealed seed 137",
            "retry": None,
            "seed_cli": None,
        },
        "formation": {
            "mechanism": "unchanged field-accretion private hooks",
            "parent_km": PARENT_KM,
            "structure_nominal_km": STRUCTURE_NOMINAL_KM,
            "continental_budget": PRIVATE_CONTINENTAL_BUDGET,
            "plates": cfg.plates,
            "nuclei_metadata_only": cfg.nuclei,
            "crop_or_target_input": None,
        },
        "sequencing": {
            "structure_builds": 1,
            "coarse_elevation_builds": 1,
            "full_parent_process_solves": 1,
            "selector_starts_only_after_process": True,
        },
        "selection": {
            "candidate_guard_km": NUMERICAL_GUARD_KM,
            "stride_km": CANDIDATE_STRIDE_KM,
            "axis_first_km": float(axis[0]),
            "axis_last_km": float(axis[-1]),
            "axis_count": int(axis.size),
            "candidate_count": int(axis.size ** 2),
            "authority_size_px": AUTHORITY_SIZE,
            "authority_km_per_px": AUTHORITY_KM_PER_PX,
            "guarded_union_span_km": union_span,
            "guarded_union_size_px": int(round(
                union_span / AUTHORITY_KM_PER_PX)),
            "mosaic_chunk_rows": MOSAIC_CHUNK_ROWS,
            "bit_exact_full_map_verification_origins": [
                [float(axis[0]), float(axis[0])],
                [float(axis[axis.size // 2]), float(axis[axis.size // 2])],
                [float(axis[-1]), float(axis[-1])],
            ],
            "targets": dict(TARGETS),
            "absolute_target_tolerance": TARGET_TOLERANCE,
            "high_additional_rule": "land_fraction < 0.50",
            "shortlist_per_target": SHORTLIST_PER_TARGET,
            "shortlist_role": "reporting only; assignment uses every in-range exact-authority candidate",
            "joint_objective_order": [
                "minimum maximum normalized target error",
                "minimum high absolute error",
                "minimum medium absolute error",
                "minimum low absolute error",
                "minimum summed absolute error",
                "minimum high/medium/low candidate tie-key tuple",
            ],
            "minimum_pairwise_origin_chebyshev_km": (
                MIN_ORIGIN_CHEBYSHEV_SEPARATION_KM),
            "border_rule": "every exact final outer-ring cell is water",
            "contour_clearance_tag_or_cohort_gate": None,
        },
        "interpretation": {
            "availability_only": True,
            "section3b_status": "unresolved_single_finite_parent",
            "promotion_assessed": False,
            "public_slider_range_proven": False,
        },
    }


@dataclass(frozen=True)
class CandidateScore:
    x0_km: float
    y0_km: float
    land_fraction: float
    water_fraction: float
    topographic_nonnegative_fraction: float
    border_passed: bool
    ring_cell_count: int
    ring_water_cells: int
    ring_ocean_cells: int
    ring_lake_cells: int
    ring_non_water_cells: int
    tie_key: int

    @property
    def origin(self) -> tuple[float, float]:
        return self.x0_km, self.y0_km


def _candidate_tie_key(origin) -> int:
    return fnv1a64(
        f"field-parent-candidate-v1:{SEED}:{origin[0]:.0f}:{origin[1]:.0f}")


@dataclass
class SedimentCapture:
    z_m: np.ndarray
    erosion_source_m: np.ndarray
    area_km2: np.ndarray
    process_spacing_km: float
    output_z_m: np.ndarray
    deposit_m: np.ndarray
    boundary_export_m_cells: float
    terminal_residual_m_cells: float
    inputs_unchanged: bool


class SedimentInstrumentation(AbstractContextManager):
    """Capture the one unchanged legacy sediment call with minimal memory."""

    def __init__(self):
        self.original: Callable | None = None
        self.call_count = 0
        self.capture: SedimentCapture | None = None

    def __enter__(self):
        self.original = erosion_engine.route_sediment

        def route(z, ero, rcv, batches, area, base_level, length_km,
                  spacing_km, edge_len_km=None):
            if self.call_count:
                raise AssertionError("more than one sediment call")
            self.call_count += 1
            z0 = np.asarray(z).copy()
            ero0 = np.asarray(ero).copy()
            rcv0 = np.asarray(rcv).copy()
            area0 = np.asarray(area).copy()
            batch0 = tuple(np.asarray(item).copy() for item in batches)
            edge0 = (None if edge_len_km is None
                     else np.asarray(edge_len_km).copy())
            result = self.original(
                z, ero, rcv, batches, area, base_level, length_km,
                spacing_km, edge_len_km)
            unchanged = bool(
                np.array_equal(z, z0)
                and np.array_equal(ero, ero0)
                and np.array_equal(rcv, rcv0)
                and np.array_equal(area, area0)
                and all(np.array_equal(a, b)
                        for a, b in zip(batches, batch0))
                and ((edge_len_km is None and edge0 is None)
                     or np.array_equal(edge_len_km, edge0)))
            self.capture = SedimentCapture(
                z_m=z0,
                erosion_source_m=ero0,
                area_km2=area0,
                process_spacing_km=float(spacing_km),
                output_z_m=np.asarray(result[0]).copy(),
                deposit_m=np.asarray(result[1]).copy(),
                boundary_export_m_cells=float(result[2]),
                terminal_residual_m_cells=float(result[3]),
                inputs_unchanged=unchanged,
            )
            return result

        erosion_engine.route_sediment = route
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original is not None:
            erosion_engine.route_sediment = self.original
        return False


def _sediment_budget(capture: SedimentCapture, erosion_result: dict) -> dict:
    source = float(np.maximum(capture.erosion_source_m, 0.0).sum())
    deposited = float(capture.deposit_m.sum())
    export = capture.boundary_export_m_cells
    terminal = capture.terminal_residual_m_cells
    closure = source - deposited - export - terminal
    tolerance = SEDIMENT_CLOSURE_RELATIVE_TOLERANCE * max(source, 1.0)
    cell_area_m2 = (capture.process_spacing_km * 1000.0) ** 2
    checks = {
        "arrays_finite": bool(
            np.isfinite(capture.z_m).all()
            and np.isfinite(capture.erosion_source_m).all()
            and np.isfinite(capture.deposit_m).all()
            and np.isfinite(capture.output_z_m).all()),
        "nonnegative_source_and_deposit": bool(
            (capture.erosion_source_m >= 0.0).all()
            and (capture.deposit_m >= 0.0).all()),
        "closure_within_1e_minus_12_relative": abs(closure) <= tolerance,
        "terminal_residual_exactly_zero": terminal == 0.0,
        "inputs_unchanged": capture.inputs_unchanged,
        "output_reconstructs": bool(np.array_equal(
            capture.output_z_m, capture.z_m + capture.deposit_m)),
        "delivered_z_exact": bool(np.array_equal(
            erosion_result["z"], capture.output_z_m)),
        "delivered_sediment_exact": bool(np.array_equal(
            erosion_result["sed"], capture.deposit_m)),
        "delivered_source_exact": bool(np.array_equal(
            erosion_result["ero"], capture.erosion_source_m)),
        "delivered_export_exact": bool(
            erosion_result["sediment_export_m3"] == export * cell_area_m2),
        "delivered_terminal_exact": bool(
            erosion_result["sediment_terminal_residual_m3"]
            == terminal * cell_area_m2),
    }
    return {
        "source_m_cells": source,
        "deposited_m_cells": deposited,
        "boundary_export_m_cells": export,
        "terminal_residual_m_cells": terminal,
        "closure_m_cells": closure,
        "tolerance_m_cells": tolerance,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _integral(mask: np.ndarray) -> np.ndarray:
    result = np.asarray(mask, np.int32).cumsum(axis=0, dtype=np.int32)
    result.cumsum(axis=1, dtype=np.int32, out=result)
    return result


def _window_sum(integral: np.ndarray, y0: int, x0: int,
                side: int) -> int:
    y1, x1 = y0 + side - 1, x0 + side - 1
    value = int(integral[y1, x1])
    if y0:
        value -= int(integral[y0 - 1, x1])
    if x0:
        value -= int(integral[y1, x0 - 1])
    if y0 and x0:
        value += int(integral[y0 - 1, x0 - 1])
    return value


def _unique_ring(array: np.ndarray) -> np.ndarray:
    return np.concatenate((
        array[0, :], array[-1, :],
        array[1:-1, 0], array[1:-1, -1],
    ))


def _score_mosaic(mosaic: dict) -> list[CandidateScore]:
    water = np.asarray(mosaic["water"], bool)
    ocean = np.asarray(mosaic["ocean"], bool)
    lake = np.asarray(mosaic["lake"], bool)
    topo = np.asarray(mosaic["topographic"], bool)
    land_integral = _integral(~water)
    topo_integral = _integral(topo)
    side = AUTHORITY_SIZE
    total = side * side
    first = float(mosaic["origin_yx_km"][1])
    scores = []
    for origin in _candidate_origins():
        x0_km, y0_km = origin
        x0 = int(round((x0_km - first) / AUTHORITY_KM_PER_PX))
        y0 = int(round((y0_km - first) / AUTHORITY_KM_PER_PX))
        land_count = _window_sum(land_integral, y0, x0, side)
        topo_count = _window_sum(topo_integral, y0, x0, side)
        ys = slice(y0, y0 + side)
        xs = slice(x0, x0 + side)
        ring_water = _unique_ring(water[ys, xs])
        ring_ocean = _unique_ring(ocean[ys, xs])
        ring_lake = _unique_ring(lake[ys, xs])
        ring_cells = int(ring_water.size)
        water_count = int(np.count_nonzero(ring_water))
        scores.append(CandidateScore(
            x0_km=x0_km,
            y0_km=y0_km,
            land_fraction=land_count / total,
            water_fraction=1.0 - land_count / total,
            topographic_nonnegative_fraction=topo_count / total,
            border_passed=water_count == ring_cells,
            ring_cell_count=ring_cells,
            ring_water_cells=water_count,
            ring_ocean_cells=int(np.count_nonzero(ring_ocean)),
            ring_lake_cells=int(np.count_nonzero(ring_lake)),
            ring_non_water_cells=ring_cells - water_count,
            tie_key=_candidate_tie_key(origin),
        ))
    return scores


def _separated(left: CandidateScore, right: CandidateScore) -> bool:
    return max(abs(left.x0_km - right.x0_km),
               abs(left.y0_km - right.y0_km)) >= (
                   MIN_ORIGIN_CHEBYSHEV_SEPARATION_KM)


def _select(scores: list[CandidateScore]) -> dict:
    water_safe = [item for item in scores if item.border_passed]
    shortlists = {}
    pools = {}
    for label, target in TARGETS:
        rankable = [
            item for item in water_safe
            if label != "high" or item.land_fraction < 0.50
        ]
        ranked = sorted(
            rankable,
            key=lambda item: (abs(item.land_fraction - target),
                              item.tie_key))
        shortlists[label] = ranked[:SHORTLIST_PER_TARGET]
        pools[label] = [
            item for item in ranked
            if (abs(item.land_fraction - target) <= TARGET_TOLERANCE
                and (label != "high" or item.land_fraction < 0.50))
        ]
    targets = dict(TARGETS)
    lows = pools["low"]
    mediums = pools["medium"]
    highs = pools["high"]

    # Exact, non-truncated assignment search.  Python integers are compact
    # compatibility bitsets over the low pool; this keeps the worst-case
    # 3,721-origin search bounded while still counting every viable triple.
    def low_compatibility_bits(candidate):
        bits = 0
        for index, low in enumerate(lows):
            if _separated(candidate, low):
                bits |= 1 << index
        return bits

    high_low_bits = [low_compatibility_bits(item) for item in highs]
    medium_low_bits = [low_compatibility_bits(item) for item in mediums]
    best = None
    viable_count = 0
    for high_index, high in enumerate(highs):
        for medium_index, medium in enumerate(mediums):
            if not _separated(high, medium):
                continue
            common = (high_low_bits[high_index]
                      & medium_low_bits[medium_index])
            if not common:
                continue
            viable_count += common.bit_count()
            first_bit = common & -common
            low_index = first_bit.bit_length() - 1
            low = lows[low_index]
            assignment = {"low": low, "medium": medium, "high": high}
            errors = {label: abs(assignment[label].land_fraction
                                 - targets[label]) for label in targets}
            key = (
                max(value / TARGET_TOLERANCE for value in errors.values()),
                errors["high"], errors["medium"], errors["low"],
                sum(errors.values()), high.tie_key, medium.tie_key,
                low.tie_key,
            )
            if best is None or key < best[0]:
                best = (key, assignment, errors)
    if best is None:
        key, assignment, errors = None, {}, {}
    else:
        key, assignment, errors = best
    diagnostics = {}
    for label, target in TARGETS:
        ranked = sorted(scores, key=lambda item: (
            0 if item.border_passed else 1,
            item.ring_non_water_cells,
            abs(item.land_fraction - target),
            item.tie_key,
        ))
        diagnostics[label] = ranked[0]
    return {
        "water_safe_count": len(water_safe),
        "shortlists": shortlists,
        "pool_counts": {label: len(value)
                        for label, value in pools.items()},
        "viable_assignment_count": viable_count,
        "found": best is not None,
        "assignment": assignment,
        "errors": errors,
        "objective": None if key is None else {
            "max_normalized_error": float(key[0]),
            "high_error": float(key[1]),
            "medium_error": float(key[2]),
            "low_error": float(key[3]),
            "sum_error": float(key[4]),
            "candidate_tie_keys_high_medium_low": [
                int(key[5]), int(key[6]), int(key[7])],
        },
        "diagnostics": diagnostics,
    }


def _score_dict(score: CandidateScore) -> dict:
    return asdict(score)


def _selection_dict(selection: dict) -> dict:
    return {
        "water_safe_count": selection["water_safe_count"],
        "pool_counts": selection["pool_counts"],
        "viable_assignment_count": selection["viable_assignment_count"],
        "found": selection["found"],
        "assignment": {label: _score_dict(item)
                       for label, item in selection["assignment"].items()},
        "errors": selection["errors"],
        "objective": selection["objective"],
        "shortlists": {
            label: [_score_dict(item) for item in values]
            for label, values in selection["shortlists"].items()
        },
        "diagnostics": {label: _score_dict(item)
                        for label, item in selection["diagnostics"].items()},
    }


def _render(structure, elevation, erosion_result, cfg, scores: dict,
            role: str, out: Path, mosaic: dict) -> tuple[dict, dict]:
    records = {}
    exact_checks = {}
    first = float(mosaic["origin_yx_km"][1])
    for label, score in scores.items():
        sample = sample_map(
            structure, elevation, erosion_result, cfg, SEED,
            AUTHORITY_SIZE,
            _frame_window_km=(score.y0_km, score.x0_km, FRAME_KM),
        )
        x0 = int(round((score.x0_km - first) / AUTHORITY_KM_PER_PX))
        y0 = int(round((score.y0_km - first) / AUTHORITY_KM_PER_PX))
        exact_checks[label] = {
            key: bool(np.array_equal(
                np.asarray(sample[key] if key != "topographic"
                           else sample["h"] >= 0.0),
                mosaic[key][y0:y0 + AUTHORITY_SIZE,
                            x0:x0 + AUTHORITY_SIZE],
            ))
            for key in ("water", "ocean", "lake", "topographic")
        }
        images = []
        middle = label if role == "accepted" else f"diagnostic_{label}"
        for view in MAP_VIEWS:
            path = out / f"seed138_{middle}_{view}_1024.png"
            render_map_view(sample, view, cfg.river_density).save(path)
            images.append({"view": view, "file": path.name,
                           "sha256": _sha256_file(path)})
        records[label] = images
    return records, exact_checks


def _parent_overview(erosion_result: dict, scores: dict, role: str,
                     out: Path) -> dict:
    image = Image.fromarray(
        formation._terrain_rgb(np.asarray(erosion_result["z"])), "RGB"
    ).resize((1024, 1024), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    scale = 1024.0 / PARENT_KM
    g0 = int(round(NUMERICAL_GUARD_KM * scale))
    g1 = int(round((PARENT_KM - NUMERICAL_GUARD_KM) * scale))
    draw.rectangle((g0, g0, g1, g1), outline=(230, 210, 80), width=2)
    colors = {"low": (80, 220, 255), "medium": (100, 255, 120),
              "high": (255, 100, 210)}
    for label, score in scores.items():
        box = tuple(int(round(item * scale)) for item in (
            score.x0_km, score.y0_km,
            score.x0_km + FRAME_KM, score.y0_km + FRAME_KM))
        draw.rectangle(box, outline=colors[label], width=4)
        draw.text((box[0] + 4, box[1] + 4), label, fill=colors[label])
    path = out / "seed138_parent_overview.png"
    image.save(path)
    return {"file": path.name, "sha256": _sha256_file(path),
            "display_role": role}


def _formation_summary(layout: dict, structure, cfg) -> dict:
    tags = np.asarray(structure._material_tag_samples)
    valid = tags[tags >= 0]
    return {
        "canonical_km": float(layout["canonical_km"]),
        "carrier_count": len(layout["carriers"]),
        "raw_domain_count": int(layout["raw_domain_count"]),
        "nucleated_domain_count": int(layout["nucleated_domain_count"]),
        "configured_plates": int(cfg.plates),
        "alive_plates": int(structure.alive_plates),
        "structure_n": int(structure.n),
        "actual_structure_km": float(structure.world_km / structure.n),
        "continental_structure_fraction": float(structure.cont_frac.mean()),
        "material_tag_shape": list(tags.shape),
        "material_tag_sha256": _sha256_bytes(
            np.ascontiguousarray(tags).tobytes()),
        "continental_tag_samples": int(valid.size),
        "represented_domain_count": int(np.unique(valid).size),
        "partition_diagnostics": formation.FORMATION_DIAGNOSTICS.get(
            "partition"),
    }


def _require_execute_output(out: Path, expected_sha256: str) -> tuple[dict, str]:
    if not out.is_dir():
        raise FileNotFoundError(out)
    entries = {item.name for item in out.iterdir()}
    if entries != {"protocol_precommit.json"}:
        raise FileExistsError(
            "execute requires only protocol_precommit.json")
    encoded = (out / "protocol_precommit.json").read_bytes()
    actual = _sha256_bytes(encoded)
    if actual != expected_sha256:
        raise ValueError("precommit SHA-256 does not match")
    payload = json.loads(encoded.decode("utf-8"))
    current = _protocol()
    if payload != current:
        raise ValueError("source/configuration no longer matches precommit")
    return payload, actual


def _phase_precommit(out: Path) -> dict:
    _prepare_empty_output(out)
    payload = _protocol()
    value = _write_json_exclusive(out / "protocol_precommit.json", payload)
    result = {
        "experiment": EXPERIMENT,
        "phase": "precommit",
        "protocol_precommit_sha256": value,
        "next_phase": "execute",
    }
    print(json.dumps(result, indent=2))
    return result


def _phase_execute(out: Path, expected_sha256: str) -> dict:
    protocol, protocol_sha256 = _require_execute_output(
        out, expected_sha256)
    started = time.perf_counter()
    cfg = formation._atlas_config(PRIVATE_CONTINENTAL_BUDGET)
    events = ["protocol_verified"]
    counters = {
        "structure_builds": 0,
        "coarse_elevation_builds": 0,
        "full_parent_process_solves": 0,
        "selector_calls": 0,
    }

    # Formation is wholly world-addressed and receives no crop information.
    layout = formation._formation_layout(SEED, PARENT_KM, int(cfg.plates))
    sample_continent = formation._continent_sampler(
        SEED, layout, PRIVATE_CONTINENTAL_BUDGET)
    sample_material_tag = formation._material_tag_sampler(
        layout, PRIVATE_CONTINENTAL_BUDGET)
    sites = formation._plate_sites(SEED, PARENT_KM, int(cfg.plates))
    structure = build_structure(
        SEED, cfg,
        _world_km=PARENT_KM,
        _coarse_km=STRUCTURE_NOMINAL_KM,
        _continent_seeder=formation._continent_seeder(sample_continent),
        _partitioner=formation._partition_field_accretion,
        _initial_age_sampler=formation._initial_ocean_age,
        _plate_pivots=sites,
        _continent_sampler=sample_continent,
        _material_tag_sampler=sample_material_tag,
    )
    counters["structure_builds"] += 1
    events.append("structure_complete")
    elevation = coarse_elevation(structure, cfg, SEED)
    counters["coarse_elevation_builds"] += 1
    events.append("coarse_elevation_complete")

    original_route = erosion_engine.route_sediment
    with SedimentInstrumentation() as instrumentation:
        erosion_result = erosion_engine.run_erosion(
            structure, elevation, cfg, SEED)
        counters["full_parent_process_solves"] += 1
    route_restored = erosion_engine.route_sediment is original_route
    if instrumentation.capture is None:
        raise AssertionError("sediment call was not captured")
    events.append("full_parent_process_complete")

    # Selection begins here, after natural geography is complete.
    counters["selector_calls"] += 1
    events.append("selector_started")
    axis = _candidate_axis()
    first = float(axis[0])
    union_span = float(axis[-1] + FRAME_KM - first)
    union_px = int(round(union_span / AUTHORITY_KM_PER_PX))
    mosaic = mosaic_engine.sample_final_boolean_window(
        structure, erosion_result, cfg, SEED,
        y0_km=first, x0_km=first,
        height_px=union_px, width_px=union_px,
        row_chunk=MOSAIC_CHUNK_ROWS,
    )
    verification = []
    for x0, y0 in protocol["selection"][
            "bit_exact_full_map_verification_origins"]:
        verification.append(mosaic_engine.verify_4096_subwindow(
            mosaic, structure, elevation, erosion_result, cfg, SEED,
            y0_km=y0, x0_km=x0))
    if not all(item["passed"] for item in verification):
        raise AssertionError(f"mosaic verification failed: {verification}")
    scores = _score_mosaic(mosaic)
    selection = _select(scores)
    events.append("exact_authority_selection_complete")

    display = (selection["assignment"] if selection["found"]
               else selection["diagnostics"])
    role = "accepted" if selection["found"] else "diagnostic_not_accepted"
    overview = _parent_overview(erosion_result, display, role, out)
    renders, selected_exact_checks = _render(
        structure, elevation, erosion_result, cfg, display, role, out,
        mosaic)
    sediment = _sediment_budget(instrumentation.capture, erosion_result)
    formation_summary = _formation_summary(layout, structure, cfg)

    assignment_gates = {
        "joint_assignment_found": selection["found"],
        "all_selected_exact_water": bool(
            selection["found"]
            and all(item.border_passed
                    for item in selection["assignment"].values())),
        "all_selected_within_target_ranges": bool(
            selection["found"]
            and all(abs(selection["assignment"][label].land_fraction - target)
                    <= TARGET_TOLERANCE
                    for label, target in TARGETS)),
        "high_strictly_below_half": bool(
            selection["found"]
            and selection["assignment"]["high"].land_fraction < 0.50),
        "three_distinct_origins": bool(
            selection["found"]
            and len({item.origin for item
                     in selection["assignment"].values()}) == 3),
        "pairwise_separation": bool(
            selection["found"]
            and all(_separated(a, b) for a, b in itertools.combinations(
                selection["assignment"].values(), 2))),
    }
    integrity_checks = {
        "source_fingerprint_still_matches": (
            _source_fingerprint() == protocol["source_fingerprint"]),
        "prior_evidence_still_matches": (
            _verify_prior_evidence() == protocol["prior_evidence"]),
        "one_structure_build": counters["structure_builds"] == 1,
        "one_elevation_build": counters["coarse_elevation_builds"] == 1,
        "one_full_parent_process_solve": (
            counters["full_parent_process_solves"] == 1),
        "one_selector_call": counters["selector_calls"] == 1,
        "selector_after_process": events.index("selector_started")
            > events.index("full_parent_process_complete"),
        "one_sediment_call": instrumentation.call_count == 1,
        "route_function_restored": route_restored,
        "candidate_count_3721": len(scores) == 3721,
        "mosaic_water_is_ocean_or_lake": bool(np.array_equal(
            mosaic["water"], mosaic["ocean"] | mosaic["lake"])),
        "three_fixed_full_map_verifications_passed": bool(
            len(verification) == 3
            and all(item["passed"] for item in verification)),
        "display_samples_match_mosaic": bool(
            all(all(checks.values())
                for checks in selected_exact_checks.values())),
        "sediment_budget_passed": sediment["passed"],
    }
    integrity_pass = all(integrity_checks.values())
    availability_pass = all(assignment_gates.values())
    report = {
        "experiment": EXPERIMENT,
        "completed": True,
        "protocol_precommit_sha256": protocol_sha256,
        "events": events,
        "execution_counters": counters,
        "formation": formation_summary,
        "parent_process": {
            "shape": list(np.asarray(erosion_result["z"]).shape),
            "e_km": float(erosion_result["e_km"]),
            "full_parent_call": "run_erosion(structure,elevation,cfg,seed)",
            "crop_or_process_window": None,
        },
        "mosaic": {
            "shape": list(mosaic["shape"]),
            "origin_yx_km": list(mosaic["origin_yx_km"]),
            "km_per_px": mosaic["km_per_px"],
            "verification": verification,
        },
        "selection": _selection_dict(selection),
        "scan_scores": [_score_dict(item) for item in scores],
        "assignment_gates": assignment_gates,
        "availability_pass": availability_pass,
        "sediment_budget": sediment,
        "integrity_checks": integrity_checks,
        "integrity_pass": integrity_pass,
        "render_role": role,
        "rendered_artifacts": renders,
        "parent_overview": overview,
        "manual_morphology_review": {
            "status": "unreviewed",
            "required_if_availability_passes": True,
            "targets": [
                "few substantial coherent land domains",
                "natural coastline and mountain morphology",
                "no repeated blob/stamp character",
                "river/lake/sediment coherence",
            ],
        },
        "automatic_feasibility_pass": bool(
            availability_pass and integrity_pass),
        "section3b_status": "unresolved_single_finite_parent",
        "promotion_assessed": False,
        "promotion": False,
        "interpretation_limits": [
            "One fresh seed, no retry.",
            "The private 0.65 budget exceeds the public 0.45 maximum.",
            "A finite full parent removes crop-local process boundaries but does not prove independence from the parent numerical rim.",
            "Legacy depression fill is rim seeded and legacy marine settlement has no hard reach.",
            "The fixed 256-km origin lattice is a one-sided availability screen.",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    report_sha = _write_json_exclusive(out / "report.json", report)
    _write_json_exclusive(out / "report.sha256.json", {
        "file": "report.json", "sha256": report_sha,
        "protocol_precommit_sha256": protocol_sha256,
    })
    summary = {
        "experiment": EXPERIMENT,
        "completed": True,
        "availability_pass": availability_pass,
        "integrity_pass": integrity_pass,
        "water_safe_count": selection["water_safe_count"],
        "assignment": report["selection"]["assignment"],
        "report_sha256": report_sha,
        "elapsed_s": report["elapsed_s"],
    }
    print(json.dumps(summary, indent=2))
    return report


def _synthetic_score(x, y, land, passed=True) -> CandidateScore:
    ring = 4 * AUTHORITY_SIZE - 4
    return CandidateScore(
        float(x), float(y), float(land), float(1.0 - land), float(land),
        bool(passed), ring, ring if passed else ring - 1,
        ring if passed else ring - 1, 0, 0 if passed else 1,
        _candidate_tie_key((x, y)),
    )


def _self_check() -> dict:
    helper = mosaic_engine.self_check()
    axis = _candidate_axis()
    lattice = {
        "axis_count": int(axis.size),
        "candidate_count": len(_candidate_origins()),
        "union_px": int(round(
            (axis[-1] + FRAME_KM - axis[0]) / AUTHORITY_KM_PER_PX)),
    }
    if lattice != {"axis_count": 61, "candidate_count": 3721,
                   "union_px": 4864}:
        raise AssertionError(lattice)
    mask = np.arange(100, dtype=np.int32).reshape(10, 10) % 3 == 0
    integral = _integral(mask)
    if _window_sum(integral, 2, 3, 4) != int(mask[2:6, 3:7].sum()):
        raise AssertionError("integral window mismatch")
    scores = [
        _synthetic_score(0, 0, 0.20),
        _synthetic_score(4096, 0, 0.35),
        _synthetic_score(8192, 0, 0.47),
    ]
    selection = _select(scores)
    if not selection["found"]:
        raise AssertionError(selection)
    # The exact-authority assignment must not be limited to the eight-entry
    # reporting shortlist: only the ninth low candidate is spatially valid.
    untruncated = [
        _synthetic_score(0, 0, 0.47),
        _synthetic_score(4096, 0, 0.35),
    ]
    untruncated.extend(
        _synthetic_score(100 * index, 0, 0.20)
        for index in range(1, 9))
    untruncated.append(_synthetic_score(8192, 0, 0.21))
    untruncated_result = _select(untruncated)
    if (not untruncated_result["found"]
            or untruncated_result["assignment"]["low"].x0_km != 8192):
        raise AssertionError(untruncated_result)
    return {"passed": True, "mosaic_helper": helper,
            "lattice": lattice, "synthetic_assignment": True,
            "untruncated_assignment": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--phase", choices=("precommit", "execute"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-precommit-sha256")
    args = parser.parse_args()
    if args.self_check:
        if (args.phase is not None or args.out is not None
                or args.expected_precommit_sha256 is not None):
            parser.error("--self-check is exclusive")
        print(json.dumps(_self_check(), indent=2, default=_json_default))
        return
    if args.phase is None or args.out is None:
        parser.error("--phase and --out are required")
    if args.phase == "precommit":
        if args.expected_precommit_sha256 is not None:
            parser.error("expected SHA is execute-only")
        _phase_precommit(args.out)
        return
    expected = args.expected_precommit_sha256
    if (expected is None or len(expected) != 64
            or any(char not in "0123456789abcdef" for char in expected)):
        parser.error("execute requires a lowercase 64-hex expected SHA")
    _phase_execute(args.out, expected)


if __name__ == "__main__":
    main()
