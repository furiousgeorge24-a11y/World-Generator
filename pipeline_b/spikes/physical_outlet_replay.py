"""Fixed seed-11 validation of the private physical-outlet successor.

One structural atlas and one coarse elevation are shared by exactly three
``physical_outlets`` erosion calls: small, large, and shifted.  The script
checks core invariance, sediment budgets, fan concentration, causal reach,
and final rendered views without changing the shipped default branch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from engine import erosion as erosion_engine
from engine.render_map import render_map_view
from engine.surface import sample_map
from spikes import atlas_replay as replay
from spikes import process_halo_diagnostic as stage_diagnostic


EXPERIMENT = "seed11-physical-outlet-replay-v1"
SEED = 11
CONTINENTAL_BUDGET = 0.65
WINDOW_ORDER = ("small", "large", "shifted")
EXPECTED_WINDOWS = {
    "small": (79, 604, 365),
    "large": (39, 564, 445),
    "shifted": (19, 584, 445),
}
PRIOR_CONTROL_REPORT_RELATIVE = Path(
    "out/process_halo_seed11_stage_v1/report.json")
PRIOR_CONTROL_REPORT_SHA256 = (
    "d6dd696c14e7cc51a990d3a2f639b2dbd21f93c74426bf79ecb9bcdde5d46ae5")
MAX_FAR_FIELD_FRACTION_TOTAL_SOURCE = 0.05
MAX_BOUNDARY_EXPORT_FRACTION_TOTAL_SOURCE = 0.02
MAX_TOP_ONE_PERCENT_DEPOSIT_FRACTION = 0.20
TERRAIN_MATERIAL_THRESHOLD_M = 0.05
HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD = 0.005
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
    "spikes/physical_outlet_replay.py",
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


def _prior_control_link() -> dict:
    path = ROOT.parent / PRIOR_CONTROL_REPORT_RELATIVE
    exists = path.is_file()
    actual = _sha256_file(path) if exists else None
    return {
        "relative_path_from_workspace": (
            PRIOR_CONTROL_REPORT_RELATIVE.as_posix()),
        "artifact_exists": exists,
        "expected_sha256": PRIOR_CONTROL_REPORT_SHA256,
        "actual_sha256": actual,
        "digest_matched": actual == PRIOR_CONTROL_REPORT_SHA256,
    }


def _protocol(fingerprint: dict) -> dict:
    return {
        "experiment": EXPERIMENT,
        "manifest_role": "pre-execution fixed validation protocol",
        "source_fingerprint": fingerprint,
        "fixed": {
            "seed": SEED,
            "continental_budget": CONTINENTAL_BUDGET,
            "origin_xy_km": list(replay.PRIMARY_ORIGIN),
            "windows": {name: list(value)
                        for name, value in EXPECTED_WINDOWS.items()},
            "localization_mode": "physical_outlets",
            "structural_builds": 1,
            "coarse_elevation_builds": 1,
            "erosion_calls": 3,
            "window_order": list(WINDOW_ORDER),
            "retries": 0,
        },
        "acceptance": {
            "nested_and_shifted_final_domain_threshold_pass": True,
            "terrain_material_absolute_threshold_m": (
                TERRAIN_MATERIAL_THRESHOLD_M),
            "hydrology_material_relative_threshold": (
                HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD),
            "maximum_far_field_fraction_of_total_source": (
                MAX_FAR_FIELD_FRACTION_TOTAL_SOURCE),
            "maximum_boundary_export_fraction_of_total_source": (
                MAX_BOUNDARY_EXPORT_FRACTION_TOTAL_SOURCE),
            "maximum_top_one_percent_footprint_deposit_fraction": (
                MAX_TOP_ONE_PERCENT_DEPOSIT_FRACTION),
            "relative_mass_closure": 1e-12,
            "marine_reach_must_be_inside_small_core_halo": True,
            "marine_thickness_cap_applied": False,
            "exact_rendering_is_reported_but_not_a_morphology_proxy": True,
        },
        "decision_policy": {
            "diagnostic_private_branch_only": True,
            "default_branch_unchanged": True,
            "no_seed_crop_or_parameter_reselection": True,
            "passing_does_not_authorize_public_promotion": True,
            "manual_morphology_review_required": True,
        },
    }


def _core_arrays(result, geometry) -> dict[str, np.ndarray]:
    return {
        name: geometry.extract_grid(result[name])
        for name in ("z", "z0", "ero", "sed", "discharge_log",
                     "lake_depth", "lake_surf")
    }


def _relative_difference(left, right, logarithmic=False) -> np.ndarray:
    left = np.asarray(left, np.float64)
    right = np.asarray(right, np.float64)
    if logarithmic:
        left = np.expm1(left)
        right = np.expm1(right)
    return np.abs(right - left) / np.maximum(
        np.maximum(np.abs(left), np.abs(right)),
        np.finfo(np.float64).tiny)


def _core_comparison(reference, other, reference_geometry,
                     other_geometry) -> dict:
    left = _core_arrays(reference, reference_geometry)
    right = _core_arrays(other, other_geometry)
    fields = {}
    terrain_material = True
    hydrology_material = True
    for name in left:
        delta = np.abs(right[name] - left[name])
        if name == "discharge_log":
            relative = _relative_difference(
                left[name], right[name], logarithmic=True)
            material = relative > HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD
            hydrology_material &= not material.any()
            policy = {
                "metric": "linearized_relative_difference",
                "threshold": HYDROLOGY_MATERIAL_RELATIVE_THRESHOLD,
            }
        else:
            material = delta > TERRAIN_MATERIAL_THRESHOLD_M
            terrain_material &= not material.any()
            policy = {
                "metric": "absolute_difference_m",
                "threshold": TERRAIN_MATERIAL_THRESHOLD_M,
            }
        fields[name] = {
            "array_exact": bool(np.array_equal(left[name], right[name])),
            "exact_changed_cells": int(np.count_nonzero(
                left[name] != right[name])),
            "material_changed_cells": int(np.count_nonzero(material)),
            "max_abs": float(delta.max(initial=0.0)),
            "p99_abs": float(np.percentile(delta, 99.0)),
            "material_policy": policy,
        }
        if name == "discharge_log":
            fields[name]["max_linearized_relative"] = float(
                relative.max(initial=0.0))
    return {
        "all_terrain_fields_materially_equal": terrain_material,
        "discharge_materially_equal": hydrology_material,
        "fields": fields,
    }


def _morphology(result: dict) -> dict:
    diagnostics = result["_localization_diagnostics"]
    marine = diagnostics["marine"]
    source = float(diagnostics["source_m_cells"])
    marine_source = float(marine["source_m_cells"])
    return {
        "source_m_cells": source,
        "land_deposited_m_cells": float(
            diagnostics["land_deposited_m_cells"]),
        "mouth_flux_m_cells": float(diagnostics["mouth_flux_m_cells"]),
        "total_relative_closure": abs(float(
            diagnostics["closure_m_cells"])) / max(source, 1.0),
        "marine": {
            "source_m_cells": marine_source,
            "deposited_m_cells": float(marine["deposited_m_cells"]),
            "deposited_fraction_of_marine_source": float(
                marine["deposited_m_cells"] / max(marine_source, 1.0)),
            "boundary_export_m_cells": float(
                marine["boundary_export_m_cells"]),
            "boundary_export_fraction_of_total_source": float(
                marine["boundary_export_m_cells"] / max(source, 1.0)),
            "far_field_export_m_cells": float(
                marine["far_field_export_m_cells"]),
            "far_field_export_fraction_of_total_source": float(
                marine["far_field_export_m_cells"] / max(source, 1.0)),
            "far_field_export_fraction_of_marine_source": float(
                marine["far_field_export_m_cells"]
                / max(marine_source, 1.0)),
            "terminal_residual_m_cells": float(
                marine["terminal_residual_m_cells"]),
            "max_reach_km": float(marine["max_reach_km"]),
            "max_deposit_m": float(marine["max_deposit_m"]),
            "p99_positive_deposit_m": float(
                marine["p99_positive_deposit_m"]),
            "deposit_footprint_cells": int(
                marine["deposit_footprint_cells"]),
            "top_one_percent_footprint_deposit_fraction": float(
                marine["top_one_percent_footprint_deposit_fraction"]),
            "aggraded_to_lowstand_cells": int(
                marine["aggraded_to_lowstand_cells"]),
            "marine_thickness_cap_applied": bool(
                marine["marine_thickness_cap_applied"]),
            "dynamic_aggradational_routing": bool(
                marine["dynamic_aggradational_routing"]),
        },
    }


def _minimum_core_halo_km(structure, windows) -> float:
    geometry = stage_diagnostic.CoreGeometry.fixed(
        "small", windows["small"], structure)
    return float(geometry.boundary_distance_km().min())


def _render_large(structure, elevation, result, cfg, out: Path) -> dict:
    x0, y0 = replay.PRIMARY_ORIGIN
    sampled = sample_map(
        structure, elevation, result, cfg, SEED, 1024,
        _frame_window_km=(y0, x0, replay.FRAME_KM))
    for view in ("hypsometric", "isobaths", "slope", "drainage",
                 "sediment"):
        render_map_view(sampled, view, cfg.river_density).save(
            out / f"seed11_physical_{view}_1024.png")
    ring = np.zeros((1024, 1024), bool)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    land = sampled["h"] >= 0.0
    land_rows, land_columns = np.nonzero(land)
    if land_rows.size:
        distance_pixels = np.minimum.reduce((
            land_rows,
            1023 - land_rows,
            land_columns,
            1023 - land_columns,
        ))
        nearest_land_km = float(
            distance_pixels.min() * sampled["km_per_px"])
    else:
        nearest_land_km = None
    return {
        "size": 1024,
        "outermost_ring_water": bool(np.all(sampled["water"][ring])),
        "nearest_land_to_border_km": nearest_land_km,
        "land_fraction": float(np.mean(land)),
        "images": [
            f"seed11_physical_{view}_1024.png"
            for view in ("hypsometric", "isobaths", "slope",
                         "drainage", "sediment")
        ],
    }


def _run(out: Path) -> dict:
    _prepare_empty_output(out)
    fingerprint = _source_fingerprint()
    protocol_sha256 = _write_json_exclusive(
        out / "protocol_precommit.json", _protocol(fingerprint))
    prior = _prior_control_link()
    if not prior["digest_matched"]:
        raise RuntimeError(f"prior control evidence changed: {prior}")

    cfg = replay._atlas_config(CONTINENTAL_BUDGET)
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
        raise AssertionError({"expected": EXPECTED_WINDOWS,
                              "observed": windows})

    solved = {}
    wall_times = {}
    for name in WINDOW_ORDER:
        call_started = time.perf_counter()
        solved[name] = replay.run_erosion(
            structure, elevation, cfg, SEED,
            _process_window=windows[name],
            _localization_mode="physical_outlets")
        wall_times[name] = time.perf_counter() - call_started

    geometries = {
        name: stage_diagnostic.CoreGeometry.fixed(
            name, windows[name], structure)
        for name in WINDOW_ORDER
    }
    relations = {}
    for name in ("small", "shifted"):
        relation = f"{name}_vs_large"
        relations[relation] = {
            "core": _core_comparison(
                solved["large"], solved[name],
                geometries["large"], geometries[name]),
            "historical_final_domain": replay._compare_domains(
                solved["large"], solved[name], replay.PRIMARY_ORIGIN,
                cfg.river_density),
            "render": {
                str(size): replay._compare_rendered(
                    structure, elevation,
                    solved["large"], solved[name], cfg, SEED,
                    replay.PRIMARY_ORIGIN, size)
                for size in (512, 1024)
            },
        }
    morphology = {name: _morphology(solved[name])
                  for name in WINDOW_ORDER}
    small_halo = _minimum_core_halo_km(structure, windows)
    render = _render_large(
        structure, elevation, solved["large"], cfg, out)

    checks = {
        "historical_final_domain_thresholds_pass": all(
            relation["historical_final_domain"]["passed"]
            for relation in relations.values()),
        "core_terrain_materially_invariant": all(
            relation["core"]["all_terrain_fields_materially_equal"]
            for relation in relations.values()),
        "core_discharge_materially_invariant": all(
            relation["core"]["discharge_materially_equal"]
            for relation in relations.values()),
        "mass_closure": all(
            value["total_relative_closure"] <= 1e-12
            for value in morphology.values()),
        "far_field_export_within_limit": all(
            value["marine"][
                "far_field_export_fraction_of_total_source"]
            <= MAX_FAR_FIELD_FRACTION_TOTAL_SOURCE
            for value in morphology.values()),
        "boundary_export_within_limit": all(
            value["marine"][
                "boundary_export_fraction_of_total_source"]
            <= MAX_BOUNDARY_EXPORT_FRACTION_TOTAL_SOURCE
            for value in morphology.values()),
        "top_one_percent_concentration_within_limit": all(
            value["marine"][
                "top_one_percent_footprint_deposit_fraction"]
            <= MAX_TOP_ONE_PERCENT_DEPOSIT_FRACTION
            for value in morphology.values()),
        "causal_reach_inside_small_core_halo": all(
            value["marine"]["max_reach_km"] < small_halo
            for value in morphology.values()),
        "no_marine_thickness_cap": all(
            not value["marine"]["marine_thickness_cap_applied"]
            for value in morphology.values()),
        "outermost_delivered_ring_water": render[
            "outermost_ring_water"],
    }
    report = {
        "experiment": EXPERIMENT,
        "completed": True,
        "protocol_precommit_sha256": protocol_sha256,
        "source_fingerprint": fingerprint,
        "prior_control_report": prior,
        "fixed": {
            "seed": SEED,
            "origin_xy_km": list(replay.PRIMARY_ORIGIN),
            "windows": {name: list(value)
                        for name, value in windows.items()},
            "localization_mode": "physical_outlets",
            "erosion_calls": 3,
            "retries": 0,
        },
        "wall_times_s": wall_times,
        "small_core_minimum_process_halo_km": small_halo,
        "relations": relations,
        "morphology": morphology,
        "rendered_large_window": render,
        "checks": checks,
        "passed_fixed_validation": all(checks.values()),
        "promotion_assessed": False,
        "promotion_not_assessed_reasons": [
            "one seed and origin do not establish population behavior",
            "saved morphology images require review",
            "the public default and version stamp remain unchanged",
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
        "passed_fixed_validation": report["passed_fixed_validation"],
        "failed_checks": [name for name, passed in checks.items()
                          if not passed],
        "elapsed_s": report["elapsed_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("out") / "physical_outlet_seed11_v1")
    args = parser.parse_args()
    result = _run(args.out)
    print(json.dumps(
        result, indent=2, allow_nan=False, default=_json_default))


if __name__ == "__main__":
    main()
