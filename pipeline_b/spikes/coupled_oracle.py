"""Frozen 120-km shortlist versus an 80-km coupled-atlas oracle.

This is an atlas-only architecture probe.  It rebuilds the deterministic
120-km survey once, writes its contour-eligible shortlist before the finer
world exists, then tests those exact origins against one 80-km oracle.
Nothing here is connected to public controls or production generation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from engine.elevation import coarse_elevation
from engine.tectonics import FRAME_KM, _partition, build_structure
from spikes.atlas_survey import (
    ATLAS_KM,
    ATLAS_PARTITION_DIAGNOSTICS,
    ATLAS_SEEDER_DIAGNOSTICS,
    BASIN_RECALL_KM,
    EVALUATION_KM,
    ORACLE_KM,
    SHORTLIST_ORACLE_CLEARANCE_M,
    SURVEY_KM,
    _atlas_config,
    _atlas_parent_provinces,
    _common_fields,
    _diagnostic_panel,
    _evaluate_candidates,
    _partition_atlas_coupled_anisotropic,
    _rank_comparison,
    _seed_atlas_nuclei_coupled_anisotropic,
)


def _plate_stats(label: np.ndarray, count: int) -> dict:
    area = np.bincount(label.ravel(), minlength=count).astype(np.float64)
    positive = area[area > 0.0]
    return {
        "cell_count": int(label.size),
        "plate_count": int(count),
        "empty_plates": int(np.count_nonzero(area == 0.0)),
        "coefficient_of_variation": float(positive.std() / positive.mean()),
        "max_to_min": float(positive.max() / positive.min()),
        "min_cells": int(positive.min()),
        "median_cells": float(np.median(positive)),
        "max_cells": int(positive.max()),
    }


def _crop_land(q: np.ndarray, h: np.ndarray, origin) -> np.ndarray:
    x0, y0 = origin
    xs = np.flatnonzero((q >= x0) & (q < x0 + FRAME_KM))
    ys = np.flatnonzero((q >= y0) & (q < y0 + FRAME_KM))
    return h[np.ix_(ys, xs)] > 0.0


def _shared_land_iou(survey, survey_elevation, oracle,
                     oracle_elevation, origins) -> dict:
    survey_q, survey_h, _ = _common_fields(survey, survey_elevation)
    oracle_q, oracle_h, _ = _common_fields(oracle, oracle_elevation)
    if not np.array_equal(survey_q, oracle_q):
        raise AssertionError("survey and oracle do not share evaluation grid")
    result = {}
    for origin in origins:
        left = _crop_land(survey_q, survey_h, origin)
        right = _crop_land(oracle_q, oracle_h, origin)
        union = int(np.count_nonzero(left | right))
        intersection = int(np.count_nonzero(left & right))
        result[f"{origin[0]:.0f},{origin[1]:.0f}"] = {
            "intersection_cells": intersection,
            "union_cells": union,
            "iou": 1.0 if union == 0 else intersection / union,
            "mask_xor_cells": int(np.count_nonzero(left ^ right)),
        }
    return result


def _candidate_record(candidate, result) -> dict:
    record = asdict(candidate)
    record["visible_contour_gate"] = result[
        "contour_gate_by_origin"][candidate.origin]
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--continental-budget", type=float, default=0.65)
    parser.add_argument(
        "--out", type=Path,
        default=Path("out") / "coupled_anisotropic_seed11_oracle80",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = _atlas_config(args.continental_budget)
    started = time.perf_counter()

    built = {}
    for label, spacing in (("survey", SURVEY_KM), ("oracle", ORACLE_KM)):
        stage_started = time.perf_counter()
        structure = build_structure(
            args.seed,
            cfg,
            _world_km=ATLAS_KM,
            _coarse_km=spacing,
            _continent_seeder=_seed_atlas_nuclei_coupled_anisotropic,
            _partitioner=_partition_atlas_coupled_anisotropic,
        )
        elevation = coarse_elevation(structure, cfg, args.seed)
        candidates = _evaluate_candidates(
            structure, elevation, [elevation], args.seed)
        built[label] = {
            "structure": structure,
            "elevation": elevation,
            "candidates": candidates,
            "elapsed_s": time.perf_counter() - stage_started,
            "seeder_diagnostic": json.loads(json.dumps(
                ATLAS_SEEDER_DIAGNOSTICS["coupled-anisotropic"])),
            "partition_diagnostic": json.loads(json.dumps(
                ATLAS_PARTITION_DIAGNOSTICS["coupled-anisotropic"])),
        }

        if label == "survey":
            frozen = [
                _candidate_record(candidate, candidates)
                for candidate in candidates["contour_shortlist"]
            ]
            (args.out / "frozen_120km_shortlist.json").write_text(
                json.dumps({
                    "seed": args.seed,
                    "continental_budget": args.continental_budget,
                    "spacing_km": SURVEY_KM,
                    "written_before_oracle": True,
                    "shortlist": frozen,
                }, indent=2),
                encoding="utf-8",
            )

    survey_result = built["survey"]["candidates"]
    oracle_result = built["oracle"]["candidates"]
    comparison = _rank_comparison(
        survey_result, oracle_result, contour_eligible=True)
    origins = [candidate.origin
               for candidate in survey_result["contour_shortlist"]]
    oracle_safe_by_origin = {
        candidate.origin: candidate
        for candidate in oracle_result["safe"]
    }
    oracle_eligible_by_origin = {
        candidate.origin: candidate
        for candidate in oracle_result["contour_eligible"]
    }

    survivor_records = []
    for origin in origins:
        safe_candidate = oracle_safe_by_origin.get(origin)
        eligible_candidate = oracle_eligible_by_origin.get(origin)
        survivor_records.append({
            "origin_xy_km": list(origin),
            "water_safe": safe_candidate is not None,
            "contour_gate_evaluated": safe_candidate is not None,
            "contour_gate_passed": eligible_candidate is not None,
            "survived_combined_gate": eligible_candidate is not None,
            "oracle_water_safe_candidate": (
                None if safe_candidate is None else
                _candidate_record(safe_candidate, oracle_result)
            ),
            "oracle_eligible_candidate": (
                None if eligible_candidate is None else
                _candidate_record(eligible_candidate, oracle_result)
            ),
        })

    survey_structure = built["survey"]["structure"]
    legacy_label = _partition(
        args.seed, survey_structure.n,
        survey_structure.world_km / survey_structure.n,
        cfg.plates,
    )
    coupled_label = survey_structure.initial_label
    _, parents, _ = _atlas_parent_provinces(
        args.seed, ATLAS_KM, int(cfg.nuclei))
    coupled_area = np.bincount(
        coupled_label.ravel(), minlength=cfg.plates).astype(np.float64)
    parent_count = int(parents.shape[0])
    plate_comparison = {
        "legacy": _plate_stats(legacy_label, cfg.plates),
        "coupled": _plate_stats(coupled_label, cfg.plates),
        "coupled_parent_plate_mean_cells": float(
            coupled_area[:parent_count].mean()),
        "coupled_ocean_plate_mean_cells": float(
            coupled_area[parent_count:].mean()),
        "coupled_parent_to_ocean_mean_ratio": float(
            coupled_area[:parent_count].mean()
            / coupled_area[parent_count:].mean()),
    }

    _diagnostic_panel(
        args.seed,
        survey_structure, built["survey"]["elevation"],
        built["oracle"]["structure"], built["oracle"]["elevation"],
        comparison,
        args.out / "coupled_seed11_120_vs_80.png",
    )

    report = {
        "experiment": "coupled-anisotropic-frozen-oracle-v1",
        "seed": args.seed,
        "continental_budget": args.continental_budget,
        "constants": {
            "survey_km": SURVEY_KM,
            "oracle_km": ORACLE_KM,
            "evaluation_km": EVALUATION_KM,
            "required_oracle_clearance_m":
                SHORTLIST_ORACLE_CLEARANCE_M,
            "basin_recall_km": BASIN_RECALL_KM,
        },
        "elapsed_s": time.perf_counter() - started,
        "stage_elapsed_s": {
            key: value["elapsed_s"] for key, value in built.items()
        },
        "survey": {
            "safe_count": len(survey_result["safe"]),
            "eligible_count": len(survey_result["contour_eligible"]),
            "frozen_shortlist": [
                _candidate_record(candidate, survey_result)
                for candidate in survey_result["contour_shortlist"]
            ],
            "seeder_diagnostic": built["survey"]["seeder_diagnostic"],
            "partition_diagnostic":
                built["survey"]["partition_diagnostic"],
        },
        "oracle": {
            "safe_count": len(oracle_result["safe"]),
            "eligible_count": len(oracle_result["contour_eligible"]),
            "best": (None if not oracle_result["contour_eligible"] else
                     _candidate_record(
                         oracle_result["contour_eligible"][0],
                         oracle_result)),
            "seeder_diagnostic": built["oracle"]["seeder_diagnostic"],
            "partition_diagnostic":
                built["oracle"]["partition_diagnostic"],
        },
        "comparison": comparison,
        "frozen_shortlist_at_oracle": survivor_records,
        "shared_64km_land_iou": _shared_land_iou(
            survey_structure, built["survey"]["elevation"],
            built["oracle"]["structure"], built["oracle"]["elevation"],
            origins,
        ),
        "plate_partition_comparison_120km": plate_comparison,
    }
    report["acceptance"] = {
        "passed": bool(comparison["passed"]),
        "survey_shortlist_count": len(origins),
        "required_shortlist_count": 3,
        "oracle_water_safe_count": len(oracle_result["safe"]),
        "oracle_contour_eligible_count": len(
            oracle_result["contour_eligible"]),
        "frozen_survivors_combined_gate": sum(
            record["survived_combined_gate"]
            for record in survivor_records),
        "stop_reason": (
            "no_80km_contour_eligible_candidate"
            if not oracle_result["contour_eligible"] else
            "shortlist_acceptance_failed"
        ),
    }
    report["passed"] = report["acceptance"]["passed"]
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
