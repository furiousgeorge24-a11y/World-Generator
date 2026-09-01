"""Two-phase, single-head oracle for transported water-framed domains.

Phase ``precommit`` freezes seed 77's formation-native shortlist and the
complete protocol without running tectonics or elevation.  Phase ``execute``
accepts only that untouched seal, builds one tagged 120 km structural head,
translates the sealed origins by exact material-tag centroid displacement,
and seals selection evidence before any elevation call.

The exact-sample 256 km continental-tag collar is diagnostic only.  Water
clearance remains an elevation-stage question and is never smuggled into the
transport selector.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np

from spikes import atlas_survey as survey
from spikes import field_accretion_oracle as base


# --------------------------------------------------------------- protocol

EXPERIMENT = "field-accretion-transported-water-oracle-v1"
SEED = 77
CONTINENTAL_BUDGET = 0.65
STRUCTURAL_SPACING_KM = 120.0
ORIGIN_SNAP_KM = 64.0
COMMON_HEAD_KM = 64.0
DENSE_HEAD_KM = 16.0
COLLAR_KM = 256.0
CLEARANCE_M = 160.0

MIN_FOOTPRINT_FRACTION = 0.20
MAX_FOOTPRINT_FRACTION = 0.50
MIN_OWNER_FRACTION = 0.85
MIN_CAPTURE_FRACTION = 0.80
MIN_MEMBER_FRAME_FRACTION = 0.04
MIN_EXPECTED_MEMBERS = 2
MAX_EXPECTED_MEMBERS = 4

FINAL_SUBCELL_OFFSETS = (
    (-0.25, -0.25), (-0.25, 0.25),
    (0.25, -0.25), (0.25, 0.25),
)

SOURCE_FINGERPRINT_FILES = tuple(dict.fromkeys((
    *base.SOURCE_FINGERPRINT_FILES,
    "spikes/transported_water_oracle.py",
)))


def _require_protocol_constants() -> dict:
    """Fail closed if the delegated evaluator's grids or bounds drift."""
    checks = {
        "common_head_matches_atlas_evaluation_grid": bool(
            COMMON_HEAD_KM == survey.EVALUATION_KM),
        "dense_head_matches_base_dense_grid": bool(
            DENSE_HEAD_KM == base.DENSE_HEAD_KM),
        "collar_matches_base_edge_band": bool(
            COLLAR_KM == base.EDGE_BAND_KM),
        "clearance_matches_base_requirement": bool(
            CLEARANCE_M == base.REQUIRED_CLEARANCE_M),
        "subcell_offsets_match_base_sampler": bool(
            tuple(FINAL_SUBCELL_OFFSETS)
            == tuple(base.FINAL_SUBCELL_OFFSETS)),
        "atlas_extent_matches_base": bool(base.ATLAS_KM == survey.ATLAS_KM),
        "frame_extent_matches_base": bool(base.FRAME_KM == survey.FRAME_KM),
        "atlas_guard_matches_base": bool(
            base.ATLAS_GUARD_KM == survey.ATLAS_GUARD_KM),
        "atlas_guard_covers_diagnostic_collar": bool(
            base.ATLAS_GUARD_KM >= COLLAR_KM),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(
            "sealed protocol constants no longer match evaluator behavior: "
            + ", ".join(failed))
    return checks


def _source_fingerprint() -> dict:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    per_file = {}
    for relative in SOURCE_FINGERPRINT_FILES:
        payload = (root / relative).read_bytes()
        per_file[relative] = hashlib.sha256(payload).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return {"combined_sha256": digest.hexdigest(), "files": per_file}


def _write_json_exclusive(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _json_clone(value):
    return json.loads(json.dumps(value))


def _sealed_formation_record(record: dict) -> dict:
    """Normalize every selector output field into immutable JSON evidence."""
    sealed = _json_clone(record)
    sealed["formation_origin_xy_km"] = sealed.pop("origin_xy_km")
    return sealed


def _formation_selection() -> tuple[object, dict, list[dict]]:
    """Run only the fixed formation selector; never build structure."""
    cfg = base._atlas_config(CONTINENTAL_BUDGET)
    layout = base._formation_layout(SEED, base.ATLAS_KM, int(cfg.plates))
    frozen, _ = base._formation_crop_records(layout)
    records = [_sealed_formation_record(item) for item in frozen]
    _require_formation_integrity(records)
    return cfg, layout, records


def _require_formation_integrity(records: list[dict]):
    if len(records) != 3:
        raise ValueError("fixed formation selector did not return 3 records")
    group_ids = [item["domain_id"] for item in records]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("formation group IDs are not unique")
    member_sets = []
    carrier_sets = []
    for record in records:
        members = tuple(record["member_domain_ids"])
        carriers = tuple(record["carrier_plate_ids"])
        if (not record.get("passed", False)
                or not members or len(set(members)) != len(members)):
            raise ValueError("formation record is not a unique passing group")
        if not (MIN_EXPECTED_MEMBERS <= len(members)
                <= MAX_EXPECTED_MEMBERS):
            raise ValueError("formation group has the wrong member count")
        if not carriers or len(set(carriers)) != len(carriers):
            raise ValueError("formation group has invalid carriers")
        member_sets.append(set(members))
        carrier_sets.append(set(carriers))
    for left, right in combinations(range(len(records)), 2):
        if member_sets[left] & member_sets[right]:
            raise ValueError("formation member sets overlap")
        if carrier_sets[left] & carrier_sets[right]:
            raise ValueError("formation carrier sets overlap")


def _protocol_payload(records: list[dict], fingerprint: dict) -> dict:
    return {
        "experiment": EXPERIMENT,
        "manifest_role": "exclusive_pre_structure_protocol_precommit",
        "phase": "precommit",
        "sequencing": {
            "written_before_any_build_structure_call": True,
            "written_before_any_elevation_call": True,
            "execute_requires_this_to_be_the_only_output_artifact": True,
            "structural_build_count": 1,
            "elevation_call_count_maximum": 1,
            "stages_km": [STRUCTURAL_SPACING_KM],
            "later_stages": [],
        },
        "seed_policy": {
            "seed": SEED,
            "seed_order": [11, 63, 77],
            "seed_63_status": "known_formation_ineligible_no_structure_run",
            "seed_77_status": (
                "next_predeclared_field_accretion_centroid_trial_without_"
                "prior_result"
            ),
            "rationale": (
                "Seed 63 failed the already-fixed formation selector and is "
                "not eligible for structural spending; seed 77 is the next "
                "predeclared eligible seed with no prior field-accretion "
                "transported-centroid structural or elevation result.  This "
                "does not claim that seed 77 is globally unused."
            ),
        },
        "delegated_evaluator_constant_checks": _require_protocol_constants(),
        "fixed_inputs": {
            "continental_budget": CONTINENTAL_BUDGET,
            "world_km": base.ATLAS_KM,
            "requested_structural_spacing_km": STRUCTURAL_SPACING_KM,
            "expected_structural_n": int(round(
                base.ATLAS_KM / STRUCTURAL_SPACING_KM)),
            "expected_actual_structural_spacing_km": (
                base.ATLAS_KM
                / int(round(base.ATLAS_KM / STRUCTURAL_SPACING_KM))
            ),
            "common_head_km": COMMON_HEAD_KM,
            "dense_head_km": DENSE_HEAD_KM,
            "origin_snap_km": ORIGIN_SNAP_KM,
            "diagnostic_two_sided_tag_collar_km": COLLAR_KM,
            "final_subcell_offsets_yx": [
                list(item) for item in FINAL_SUBCELL_OFFSETS],
            "formation_frozen_records": records,
        },
        "origin_rule": {
            "canonical_centroid": (
                "equal-area mean of canonical cell centres in the sealed "
                "member set with accretion_time <= the fixed budget limit"
            ),
            "final_centroid": (
                "equal-area mean of exact final winning tag coordinates "
                "y=(i+0.5+oy)*actual_ck, x=(j+0.5+ox)*actual_ck"
            ),
            "displacement": "final group centroid minus canonical centroid",
            "unsnapped_origin": (
                "sealed formation origin plus centroid displacement"
            ),
            "snap": (
                "independent Decimal ROUND_HALF_UP of coordinate / 64 km, "
                "multiplied by 64 km exactly once"
            ),
            "candidate_count_per_group": 1,
        },
        "pre_elevation_hard_gates": {
            "exact_sealed_group_member_carrier_ids": True,
            "every_expected_member_final_tag_count": "> 0",
            "dense_layout_labels_and_tag_range": True,
            "guarded_origin_range_km": [
                base.ATLAS_GUARD_KM,
                base.ATLAS_KM - base.ATLAS_GUARD_KM - base.FRAME_KM,
            ],
            "pairwise_frame_overlap_count": 0,
            "tag_footprint_fraction": [
                MIN_FOOTPRINT_FRACTION, MAX_FOOTPRINT_FRACTION],
            "tag_footprint_upper_bound_exclusive": True,
            "minimum_expected_owner_fraction": MIN_OWNER_FRACTION,
            "minimum_capture_each_expected_member": MIN_CAPTURE_FRACTION,
            "expected_member_count_inclusive": [
                MIN_EXPECTED_MEMBERS, MAX_EXPECTED_MEMBERS],
            "minimum_frame_fraction_each_expected_member":
                MIN_MEMBER_FRAME_FRACTION,
            "foreign_ids_at_or_above_significant_fraction": 0,
        },
        "diagnostic_only_not_in_selection_passed": {
            "continental_tag_count_in_exact_two_sided_256km_collar": True,
            "collar_bounds_completeness_reported_explicitly": True,
        },
        "post_elevation_hard_gates": {
            "all_frozen_transport_identities_present": True,
            "every_expected_member_observed_on_emerged_land": True,
            "minimum_emerged_land_owner_fraction": MIN_OWNER_FRACTION,
            "minimum_transport_capture_fraction": MIN_CAPTURE_FRACTION,
            "emerged_land_fraction": [
                MIN_FOOTPRINT_FRACTION, MAX_FOOTPRINT_FRACTION],
            "emerged_land_fraction_upper_bound_exclusive": True,
            "significant_component_count_inclusive": [
                MIN_EXPECTED_MEMBERS, MAX_EXPECTED_MEMBERS],
            "common_head_km": COMMON_HEAD_KM,
            "dense_head_km": DENSE_HEAD_KM,
            "minimum_common_clearance_m": CLEARANCE_M,
            "minimum_dense_clearance_m": CLEARANCE_M,
            "common_exterior_ocean_coverage": 1.0,
            "dense_exterior_ocean_coverage": 1.0,
            "common_visible_border_contour_gate": True,
            "dense_visible_border_contour_gate": True,
        },
        "post_elevation_water_authority": {
            "field": "conservative_elevation_derived_late_envelope",
            "definition": (
                "current elevation plus the fixed positive bounded later "
                "relief/deposition allowance plus the fixed positive "
                "erosion-window uplift allowance"
            ),
            "positive_relief_and_deposition_bound_m":
                base.POSITIVE_RELIEF_BOUND_M,
            "positive_uplift_window_myr": base.EROSION_TIME_MAX_MYR,
            "required_exterior_connected_coverage": 1.0,
            "required_clearance_m": CLEARANCE_M,
            "continental_material_tags_affect_authority": False,
        },
        "selection_input_allowlist": [
            "sealed formation records",
            "budget-limited canonical domain_label/accretion_time/q fields",
            "one 120-km Structure world_km and n",
            "that Structure's exact final _material_tag_samples",
            "fixed constants and subcell offsets in this manifest",
        ],
        "selection_forbidden_inputs": [
            "elevation", "uplift", "sea_level", "coastline",
            "water_mask", "boundary_distance", "contours", "palette",
            "rendered_images", "prior_seed_structural_results",
        ],
        "prohibited_adaptation": [
            "crop_search", "origin_clamp", "retry", "seed_substitution",
            "identity_substitution", "group_substitution",
            "resolution_substitution", "threshold_relaxation",
        ],
        "source_fingerprint": fingerprint,
    }


# --------------------------------------------------------- tag geometry

def _snap_half_up_64(value: float) -> float:
    quantum = Decimal(str(ORIGIN_SNAP_KM))
    units = (Decimal(str(float(value))) / quantum).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP)
    return float(units * quantum)


def _material_tags(structure) -> np.ndarray:
    if not hasattr(structure, "_material_tag_samples"):
        raise ValueError("structural head has no material tag evidence")
    tags = np.asarray(structure._material_tag_samples)
    expected = (len(FINAL_SUBCELL_OFFSETS), structure.n, structure.n)
    if tags.shape != expected:
        raise ValueError(f"material tag shape {tags.shape} != {expected}")
    if not np.issubdtype(tags.dtype, np.integer):
        raise ValueError("material tags are not integers")
    return tags


def _domain_maps(layout: dict) -> tuple[dict, dict]:
    by_id = {item["domain_id"]: item for item in layout["domains"]}
    by_label = {int(item["label"]): item for item in layout["domains"]}
    return by_id, by_label


def _canonical_group_stats(layout: dict, record: dict) -> dict:
    by_id, _ = _domain_maps(layout)
    member_ids = tuple(record["member_domain_ids"])
    labels = np.asarray(
        [by_id[item]["label"] for item in member_ids], np.int64)
    limit = base._accretion_time_limit(CONTINENTAL_BUDGET)
    eligible = layout["accretion_time"] <= limit
    support = np.isin(layout["domain_label"], labels) & eligible
    ys, xs = np.nonzero(support)
    centroid = None
    if ys.size:
        q = layout["q"]
        centroid = [float(np.mean(q[ys])), float(np.mean(q[xs]))]
    member_counts = {
        domain_id: int(np.count_nonzero(
            (layout["domain_label"] == by_id[domain_id]["label"])
            & eligible))
        for domain_id in member_ids
    }
    canonical_cell_area = float(layout["canonical_km"] ** 2)
    return {
        "labels": labels,
        "centroid_yx_km": centroid,
        "support_cells": int(ys.size),
        "support_area_km2": float(ys.size * canonical_cell_area),
        "member_support_cells": member_counts,
        "member_support_area_km2": {
            item: float(count * canonical_cell_area)
            for item, count in member_counts.items()
        },
        "accretion_time_limit_km": float(limit),
    }


def _final_group_stats(tags: np.ndarray, structure,
                       labels: np.ndarray, member_ids: tuple[str, ...]) -> dict:
    ck = structure.world_km / structure.n
    sample_area = ck ** 2 / len(FINAL_SUBCELL_OFFSETS)
    count = 0
    sum_y = 0.0
    sum_x = 0.0
    for sample_index, (oy, ox) in enumerate(FINAL_SUBCELL_OFFSETS):
        ys, xs = np.nonzero(np.isin(tags[sample_index], labels))
        count += int(ys.size)
        sum_y += float(np.sum((ys + 0.5 + oy) * ck))
        sum_x += float(np.sum((xs + 0.5 + ox) * ck))
    centroid = None if count == 0 else [sum_y / count, sum_x / count]
    member_counts = {
        domain_id: int(np.count_nonzero(tags == int(label)))
        for domain_id, label in zip(member_ids, labels)
    }
    return {
        "centroid_yx_km": centroid,
        "support_tag_samples": count,
        "support_area_km2": float(count * sample_area),
        "member_global_tag_samples": member_counts,
        "member_global_tag_area_km2": {
            item: float(value * sample_area)
            for item, value in member_counts.items()
        },
        "sample_area_km2": float(sample_area),
    }


def _crop_metrics(tags: np.ndarray, structure, layout: dict,
                  origin_xy_km: list[float], labels: np.ndarray,
                  member_ids: tuple[str, ...]) -> dict:
    ck = structure.world_km / structure.n
    sample_area = ck ** 2 / len(FINAL_SUBCELL_OFFSETS)
    q = (np.arange(structure.n) + 0.5) * ck
    x0, y0 = origin_xy_km
    domain_count = len(layout["domains"])
    crop_counts = np.zeros(domain_count, np.int64)
    crop_total = 0
    crop_continental = 0
    collar_total = 0
    collar_continental = 0
    collar_bounds_complete = bool(
        x0 - COLLAR_KM >= 0.0
        and y0 - COLLAR_KM >= 0.0
        and x0 + base.FRAME_KM + COLLAR_KM <= structure.world_km
        and y0 + base.FRAME_KM + COLLAR_KM <= structure.world_km)

    for sample_index, (oy, ox) in enumerate(FINAL_SUBCELL_OFFSETS):
        sample_y = q + oy * ck
        sample_x = q + ox * ck
        iy = np.flatnonzero(
            (sample_y >= y0) & (sample_y < y0 + base.FRAME_KM))
        ix = np.flatnonzero(
            (sample_x >= x0) & (sample_x < x0 + base.FRAME_KM))
        values = tags[sample_index][np.ix_(iy, ix)]
        crop_total += int(values.size)
        valid = values[values >= 0]
        crop_continental += int(valid.size)
        crop_counts += np.bincount(
            valid, minlength=domain_count)[:domain_count].astype(np.int64)

        cy = np.flatnonzero(
            (sample_y >= y0 - COLLAR_KM)
            & (sample_y < y0 + base.FRAME_KM + COLLAR_KM))
        cx = np.flatnonzero(
            (sample_x >= x0 - COLLAR_KM)
            & (sample_x < x0 + base.FRAME_KM + COLLAR_KM))
        yr = sample_y[cy] - y0
        xr = sample_x[cx] - x0
        collar = (
            (xr[None, :] < COLLAR_KM)
            | (xr[None, :] >= base.FRAME_KM - COLLAR_KM)
            | (yr[:, None] < COLLAR_KM)
            | (yr[:, None] >= base.FRAME_KM - COLLAR_KM)
        )
        collar_values = tags[sample_index][np.ix_(cy, cx)][collar]
        collar_total += int(collar_values.size)
        collar_continental += int(np.count_nonzero(collar_values >= 0))

    global_counts = np.bincount(
        tags[tags >= 0], minlength=domain_count)[:domain_count].astype(np.int64)
    expected_crop = {
        domain_id: int(crop_counts[int(label)])
        for domain_id, label in zip(member_ids, labels)
    }
    expected_global = {
        domain_id: int(global_counts[int(label)])
        for domain_id, label in zip(member_ids, labels)
    }
    capture = {
        item: float(expected_crop[item] / max(expected_global[item], 1))
        for item in member_ids
    }
    frame_fractions = {
        item: float(expected_crop[item] / max(crop_total, 1))
        for item in member_ids
    }
    significant_labels = np.flatnonzero(
        crop_counts >= MIN_MEMBER_FRAME_FRACTION * max(crop_total, 1))
    _, by_label = _domain_maps(layout)
    significant_ids = [
        by_label[int(label)]["domain_id"] for label in significant_labels]
    expected_set = set(member_ids)
    foreign_significant = [
        item for item in significant_ids if item not in expected_set]
    footprint = float(crop_continental / max(crop_total, 1))
    owner = float(sum(expected_crop.values()) / max(crop_continental, 1))
    return {
        "sample_area_km2": float(sample_area),
        "crop_total_exact_tag_samples": crop_total,
        "crop_total_quadrature_area_km2": float(crop_total * sample_area),
        "crop_continental_tag_samples": crop_continental,
        "crop_continental_tag_area_km2":
            float(crop_continental * sample_area),
        "tag_footprint_fraction": footprint,
        "expected_owner_fraction": owner,
        "expected_crop_tag_samples": expected_crop,
        "expected_global_tag_samples": expected_global,
        "capture_by_domain_id": capture,
        "minimum_capture_fraction": min(capture.values(), default=0.0),
        "member_frame_fractions": frame_fractions,
        "significant_domain_ids": significant_ids,
        "foreign_significant_domain_ids": foreign_significant,
        "diagnostic_two_sided_256km_tag_collar": {
            "affects_passed": False,
            "bounds_complete": collar_bounds_complete,
            "total_exact_tag_samples": collar_total,
            "continental_tag_samples": collar_continental,
            "continental_tag_area_km2": float(
                collar_continental * sample_area),
        },
    }


def _frames_nonoverlap(left: dict, right: dict) -> bool:
    lx, ly = left["transported_origin_xy_km"]
    rx, ry = right["transported_origin_xy_km"]
    return bool(
        abs(lx - rx) >= base.FRAME_KM
        or abs(ly - ry) >= base.FRAME_KM)


def _transport_selection(structure, layout: dict, records: list[dict],
                         precommit_sha256: str) -> tuple[dict, list[dict]]:
    expected_n = int(round(base.ATLAS_KM / STRUCTURAL_SPACING_KM))
    structure_contract = {
        "world_extent_matches_sealed_atlas": bool(
            structure.world_km == base.ATLAS_KM),
        "grid_count_matches_fixed_spacing_request": bool(
            structure.n == expected_n),
    }
    if not all(structure_contract.values()):
        failed = [name for name, passed in structure_contract.items()
                  if not passed]
        raise RuntimeError(
            "structural head violates sealed protocol: "
            + ", ".join(failed))
    raw_tags = _material_tags(structure)
    raw_tag_sha256 = hashlib.sha256(
        np.ascontiguousarray(raw_tags).view(np.uint8)).hexdigest()
    tags = raw_tags.astype(np.int64, copy=False)
    normalized_tag_sha256 = hashlib.sha256(
        np.ascontiguousarray(tags).view(np.uint8)).hexdigest()
    valid_tags = tags[tags >= 0]
    domain_labels = sorted(int(item["label"]) for item in layout["domains"])
    layout_labels_dense = domain_labels == list(range(len(domain_labels)))
    tag_labels_in_range = bool(
        np.all(tags >= -1)
        and (valid_tags.size == 0
             or int(valid_tags.max()) < len(domain_labels)))
    dense_tag_labels = bool(layout_labels_dense and tag_labels_in_range)
    by_id, _ = _domain_maps(layout)
    selected = []
    evaluator_records = []

    for sealed in records:
        member_ids = tuple(sealed["member_domain_ids"])
        canonical = _canonical_group_stats(layout, sealed)
        final = _final_group_stats(
            tags, structure, canonical["labels"], member_ids)
        canonical_centroid = canonical["centroid_yx_km"]
        final_centroid = final["centroid_yx_km"]
        displacement_xy = None
        raw_origin = None
        transported_origin = None
        snap_residual = None
        if canonical_centroid is not None and final_centroid is not None:
            dy = final_centroid[0] - canonical_centroid[0]
            dx = final_centroid[1] - canonical_centroid[1]
            displacement_xy = [float(dx), float(dy)]
            old_x, old_y = sealed["formation_origin_xy_km"]
            raw_origin = [float(old_x + dx), float(old_y + dy)]
            transported_origin = [
                _snap_half_up_64(raw_origin[0]),
                _snap_half_up_64(raw_origin[1]),
            ]
            snap_residual = [
                float(transported_origin[0] - raw_origin[0]),
                float(transported_origin[1] - raw_origin[1]),
            ]

        metrics = None
        if transported_origin is not None and dense_tag_labels:
            metrics = _crop_metrics(
                tags, structure, layout, transported_origin,
                canonical["labels"], member_ids)
        nonzero = bool(
            all(value > 0 for value in
                canonical["member_support_cells"].values())
            and all(value > 0 for value in
                    final["member_global_tag_samples"].values()))
        guarded = bool(
            transported_origin is not None
            and base.ATLAS_GUARD_KM <= transported_origin[0]
            <= base.ATLAS_KM - base.ATLAS_GUARD_KM - base.FRAME_KM
            and base.ATLAS_GUARD_KM <= transported_origin[1]
            <= base.ATLAS_KM - base.ATLAS_GUARD_KM - base.FRAME_KM)
        if (guarded and metrics is not None
                and not metrics[
                    "diagnostic_two_sided_256km_tag_collar"
                ]["bounds_complete"]):
            raise RuntimeError(
                "guarded origin did not produce a complete tag collar")
        members_exist = bool(all(item in by_id for item in member_ids))
        actual_carrier_ids = (
            sorted({int(by_id[item]["carrier_plate_id"])
                    for item in member_ids})
            if members_exist else [])
        sealed_carrier_ids = sorted(
            int(item) for item in sealed["carrier_plate_ids"])
        exact_ids = bool(
            sealed["domain_id"]
            == f"{base.fnv1a64('field-accretion-group-v1:' + ':'.join(member_ids)):016x}"
            and members_exist
            and actual_carrier_ids == sealed_carrier_ids)
        gates = {
            "exact_sealed_group_member_carrier_ids": exact_ids,
            "nonzero_expected_members": nonzero,
            "dense_tag_labels": dense_tag_labels,
            "guarded_origin": guarded,
            "pairwise_nonoverlap": False,
            "tag_footprint_0_20_to_0_50_exclusive": False,
            "expected_owner_at_least_0_85": False,
            "every_expected_capture_at_least_0_80": False,
            "expected_member_count_2_to_4": bool(
                MIN_EXPECTED_MEMBERS <= len(member_ids)
                <= MAX_EXPECTED_MEMBERS),
            "every_expected_id_at_least_0_04_frame": False,
            "no_significant_foreign_id": False,
            "significant_ids_exactly_expected": False,
        }
        if metrics is not None:
            gates.update({
                "tag_footprint_0_20_to_0_50_exclusive": bool(
                    MIN_FOOTPRINT_FRACTION
                    <= metrics["tag_footprint_fraction"]
                    < MAX_FOOTPRINT_FRACTION),
                "expected_owner_at_least_0_85": bool(
                    metrics["expected_owner_fraction"]
                    >= MIN_OWNER_FRACTION),
                "every_expected_capture_at_least_0_80": bool(
                    all(value >= MIN_CAPTURE_FRACTION for value in
                        metrics["capture_by_domain_id"].values())),
                "every_expected_id_at_least_0_04_frame": bool(
                    all(value >= MIN_MEMBER_FRAME_FRACTION for value in
                        metrics["member_frame_fractions"].values())),
                "no_significant_foreign_id": bool(
                    not metrics["foreign_significant_domain_ids"]),
                "significant_ids_exactly_expected": bool(
                    set(metrics["significant_domain_ids"])
                    == set(member_ids)),
            })
        selected.append({
            "domain_id": sealed["domain_id"],
            "member_domain_ids": list(member_ids),
            "carrier_plate_ids": list(sealed["carrier_plate_ids"]),
            "actual_carrier_plate_ids": actual_carrier_ids,
            "formation_origin_xy_km": list(
                sealed["formation_origin_xy_km"]),
            "canonical_centroid_xy_km": (
                None if canonical_centroid is None
                else [canonical_centroid[1], canonical_centroid[0]]),
            "final_centroid_xy_km": (
                None if final_centroid is None
                else [final_centroid[1], final_centroid[0]]),
            "centroid_displacement_xy_km": displacement_xy,
            "raw_transported_origin_xy_km": raw_origin,
            "transported_origin_xy_km": transported_origin,
            "snap_residual_xy_km": snap_residual,
            "canonical_support": {
                key: value for key, value in canonical.items()
                if key != "labels"
            },
            "final_support": final,
            "crop_metrics": metrics,
            "hard_gates": gates,
        })
        if transported_origin is not None and metrics is not None:
            evaluator_records.append({
                "domain_id": sealed["domain_id"],
                "member_domain_ids": list(member_ids),
                "carrier_plate_ids": list(sealed["carrier_plate_ids"]),
                "canonical_pivots_yx_km": [
                    by_id[item]["pivot_yx_km"] for item in member_ids],
                "formation_origin_xy_km": list(
                    sealed["formation_origin_xy_km"]),
                "origin_xy_km": transported_origin,
                "owner_fraction": metrics["expected_owner_fraction"],
                "capture_fraction": metrics["minimum_capture_fraction"],
            })

    pairwise = []
    pairwise_passed = len(selected) == 3
    for left, right in combinations(selected, 2):
        if (left["transported_origin_xy_km"] is None
                or right["transported_origin_xy_km"] is None):
            nonoverlap = False
        else:
            nonoverlap = _frames_nonoverlap(left, right)
        pairwise_passed = pairwise_passed and nonoverlap
        pairwise.append({
            "left_domain_id": left["domain_id"],
            "right_domain_id": right["domain_id"],
            "nonoverlap": nonoverlap,
        })
    for item in selected:
        item["hard_gates"]["pairwise_nonoverlap"] = bool(pairwise_passed)
        item["passed"] = bool(all(item["hard_gates"].values()))

    passed = bool(
        len(selected) == 3
        and len(evaluator_records) == 3
        and all(item["passed"] for item in selected))
    selection = {
        "experiment": EXPERIMENT,
        "manifest_role": "exclusive_pre_elevation_transport_selection",
        "phase": "execute",
        "hash_chain": {
            "parent_protocol_precommit_sha256": precommit_sha256,
        },
        "sequencing": {
            "written_after_exactly_one_120km_structure": True,
            "written_before_any_elevation": True,
            "images_written_before_this_manifest": 0,
        },
        "seed": SEED,
        "continental_budget": CONTINENTAL_BUDGET,
        "requested_spacing_km": STRUCTURAL_SPACING_KM,
        "actual_spacing_km": float(structure.world_km / structure.n),
        "world_km": float(structure.world_km),
        "n": int(structure.n),
        "structure_contract": structure_contract,
        "tag_evidence": {
            "raw_tensor_sha256": raw_tag_sha256,
            "raw_shape": list(raw_tags.shape),
            "raw_dtype": str(raw_tags.dtype),
            "normalized_int64_sha256": normalized_tag_sha256,
            "normalized_dtype": str(tags.dtype),
            "final_subcell_offsets_yx": [
                list(item) for item in FINAL_SUBCELL_OFFSETS],
            "continental_tag_samples": int(valid_tags.size),
            "dense_layout_labels": layout_labels_dense,
            "tag_labels_in_range": tag_labels_in_range,
        },
        "pairwise_frames": pairwise,
        "groups": selected,
        "diagnostic_only_fields": [
            "groups[].crop_metrics."
            "diagnostic_two_sided_256km_tag_collar"
        ],
        "passed": passed,
    }
    return selection, evaluator_records


# ---------------------------------------------------------- phase control

def _prepare_empty_output(path: Path):
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(path)
        if any(path.iterdir()):
            raise FileExistsError(
                f"precommit output directory must be empty: {path}")
    else:
        path.mkdir(parents=True)


def _require_execute_output(path: Path, expected_sha256: str
                            ) -> tuple[Path, dict, str]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    entries = {item.name for item in path.iterdir()}
    if entries != {"protocol_precommit.json"}:
        raise FileExistsError(
            "execute requires only protocol_precommit.json and no report")
    precommit_path = path / "protocol_precommit.json"
    try:
        encoded = precommit_path.read_bytes()
        actual_sha256 = hashlib.sha256(encoded).hexdigest()
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid protocol precommit") from exc
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "protocol precommit does not match externally supplied SHA-256")
    return precommit_path, payload, actual_sha256


def _phase_precommit(out: Path):
    _require_protocol_constants()
    _prepare_empty_output(out)
    fingerprint = _source_fingerprint()
    _, _, records = _formation_selection()
    if _source_fingerprint() != fingerprint:
        raise RuntimeError("source changed during formation-only precommit")
    payload = _protocol_payload(records, fingerprint)
    sha256 = _write_json_exclusive(out / "protocol_precommit.json", payload)
    print(json.dumps({
        "experiment": EXPERIMENT,
        "phase": "precommit",
        "protocol_precommit_sha256": sha256,
        "formation_group_ids": [item["domain_id"] for item in records],
        "next_phase": "execute",
    }, indent=2))


def _verify_precommit(payload: dict) -> tuple[object, dict, list[dict]]:
    _require_protocol_constants()
    fingerprint = _source_fingerprint()
    if payload.get("source_fingerprint") != fingerprint:
        raise ValueError("source fingerprint no longer matches precommit")
    cfg, layout, records = _formation_selection()
    if _source_fingerprint() != fingerprint:
        raise RuntimeError("source changed while verifying precommit")
    expected = _protocol_payload(records, fingerprint)
    if payload != expected:
        raise ValueError(
            "recomputed formation records or protocol differ from precommit")
    return cfg, layout, records


def _collar_cell_counts(origin_xy_km: list[float], spacing_km: float,
                        coverage: float) -> dict:
    count = int(round(base.ATLAS_KM / spacing_km))
    q = (np.arange(count) + 0.5) * (base.ATLAS_KM / count)
    x0, y0 = origin_xy_km
    _, _, collar = base._collar_indices(q, x0, y0)
    total = int(np.count_nonzero(collar))
    covered = int(round(float(coverage) * total))
    if not (0 <= covered <= total):
        raise ValueError("recorded exterior coverage is outside [0, 1]")
    return {
        "covered": covered,
        "total": total,
        "spacing_km": float(base.ATLAS_KM / count),
    }


def _report_post_elevation(results: list[dict]) -> bool:
    all_passed = True
    for result in results:
        expected = set(result["member_domain_ids"])
        observed_land = set(result["observed_land_domain_ids"])
        emerged_members = expected.issubset(observed_land)
        base_passed = bool(result["passed"])
        identities_present = bool(result.pop("transport_identity_exact"))
        common_cell_counts = _collar_cell_counts(
            result["origin_xy_km"], COMMON_HEAD_KM,
            result["common_exterior_ocean_coverage"])
        dense_cell_counts = _collar_cell_counts(
            result["origin_xy_km"], DENSE_HEAD_KM,
            result["dense_exterior_ocean_coverage"])
        gates = {
            "base_evaluator_passed": base_passed,
            "all_frozen_transport_identities_present": identities_present,
            "every_expected_member_observed_on_emerged_land": bool(
                emerged_members),
            "emerged_land_owner_at_least_0_85": bool(
                result["transport_owner_fraction"] >= MIN_OWNER_FRACTION),
            "transport_capture_at_least_0_80": bool(
                result["transport_capture_fraction"] >= MIN_CAPTURE_FRACTION),
            "emerged_land_fraction_0_20_to_0_50_exclusive": bool(
                MIN_FOOTPRINT_FRACTION <= result["land_fraction"]
                < MAX_FOOTPRINT_FRACTION),
            "significant_components_2_to_4": bool(
                MIN_EXPECTED_MEMBERS
                <= result["components"]["significant_count"]
                <= MAX_EXPECTED_MEMBERS),
            "common_clearance_at_least_160m": bool(
                result["common_clearance_m"] >= CLEARANCE_M),
            "dense_clearance_at_least_160m": bool(
                result["dense_clearance_m"] >= CLEARANCE_M),
            "common_exterior_ocean_coverage_1": bool(
                result["common_exterior_ocean_coverage"] == 1.0),
            "dense_exterior_ocean_coverage_1": bool(
                result["dense_exterior_ocean_coverage"] == 1.0),
            "common_visible_contour_gate": bool(
                result["common_contour_gate"]["passed"]),
            "dense_visible_contour_gate": bool(
                result["dense_contour_gate"]["passed"]),
        }
        result["base_evaluator_passed"] = base_passed
        result["all_frozen_transport_identities_present"] = identities_present
        result["all_expected_members_observed_on_emerged_land"] = bool(
            emerged_members)
        result["common_exterior_ocean_cell_counts"] = common_cell_counts
        result["dense_exterior_ocean_cell_counts"] = dense_cell_counts
        result["water_authority"] = (
            "conservative_elevation_derived_late_envelope")
        result["transported_water_hard_gates"] = gates
        result["passed"] = bool(all(gates.values()))
        result["pre_elevation_tag_owner_fraction"] = result.pop(
            "formation_owner_fraction")
        result["pre_elevation_tag_capture_fraction"] = result.pop(
            "formation_capture_fraction")
        all_passed = all_passed and result["passed"]
    return bool(len(results) == 3 and all_passed)


def _write_report(out: Path, report: dict,
                  precommit_sha256: str, selection_sha256: str):
    report["hash_chain"] = {
        "protocol_precommit_sha256": precommit_sha256,
        "transport_selection_sha256": selection_sha256,
    }
    report_sha256 = _write_json_exclusive(out / "report.json", report)
    _write_json_exclusive(out / "report.sha256.json", {
        "experiment": EXPERIMENT,
        "file": "report.json",
        "sha256": report_sha256,
        "parent_transport_selection_sha256": selection_sha256,
    })


def _phase_execute(out: Path, expected_precommit_sha256: str):
    _, precommit, precommit_sha256 = _require_execute_output(
        out, expected_precommit_sha256)
    cfg, layout, records = _verify_precommit(precommit)
    sample_continent = base._continent_sampler(
        SEED, layout, CONTINENTAL_BUDGET)
    sample_material_tag = base._material_tag_sampler(
        layout, CONTINENTAL_BUDGET)
    sites = base._plate_sites(SEED, base.ATLAS_KM, int(cfg.plates))
    started = time.perf_counter()

    # The only structural build in this phase and experiment.
    structure = base.build_structure(
        SEED,
        cfg,
        _world_km=base.ATLAS_KM,
        _coarse_km=STRUCTURAL_SPACING_KM,
        _continent_seeder=base._continent_seeder(sample_continent),
        _partitioner=base._partition_field_accretion,
        _initial_age_sampler=base._initial_ocean_age,
        _plate_pivots=sites,
        _continent_sampler=sample_continent,
        _material_tag_sampler=sample_material_tag,
    )

    selection, transported = _transport_selection(
        structure, layout, records, precommit_sha256)
    selection_sha256 = _write_json_exclusive(
        out / "transport_selection.json", selection)
    _write_json_exclusive(out / "transport_selection.sha256.json", {
        "experiment": EXPERIMENT,
        "file": "transport_selection.json",
        "sha256": selection_sha256,
        "parent_protocol_precommit_sha256": precommit_sha256,
        "written_before_any_elevation": True,
    })

    if not selection["passed"]:
        report = {
            "experiment": EXPERIMENT,
            "phase": "execute",
            "seed": SEED,
            "continental_budget": CONTINENTAL_BUDGET,
            "passed": False,
            "stop_stage": "pre_elevation_transport_selection",
            "structural_build_count": 1,
            "elevation_call_count": 0,
            "images_written": [],
            "stages_run": ["120km_structure_only"],
            "elapsed_s": time.perf_counter() - started,
        }
        _write_report(
            out, report, precommit_sha256, selection_sha256)
        print(json.dumps(report, indent=2))
        return

    # Selection and its hash seal both exist before this first elevation call.
    elevation = base.coarse_elevation(structure, cfg, SEED)
    results, _ = base._evaluate_frozen_stage(
        structure, elevation, transported, layout)
    stage_passed = _report_post_elevation(results)
    base._head_panel(
        structure, elevation, transported, "120",
        out / "head_120km.png")
    report = {
        "experiment": EXPERIMENT,
        "phase": "execute",
        "seed": SEED,
        "continental_budget": CONTINENTAL_BUDGET,
        "passed": stage_passed,
        "stop_stage": None if stage_passed else "120km_head",
        "structural_build_count": 1,
        "elevation_call_count": 1,
        "stages_run": ["120km"],
        "later_stages_run": [],
        "requested_spacing_km": STRUCTURAL_SPACING_KM,
        "actual_spacing_km": float(structure.world_km / structure.n),
        "n": int(structure.n),
        "images_written": ["head_120km.png"],
        "elapsed_s": time.perf_counter() - started,
        "results": results,
    }
    _write_report(out, report, precommit_sha256, selection_sha256)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True, choices=("precommit", "execute"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-precommit-sha256")
    args = parser.parse_args()
    if args.phase == "precommit":
        if args.expected_precommit_sha256 is not None:
            parser.error(
                "--expected-precommit-sha256 is only valid for execute")
        _phase_precommit(args.out)
    else:
        expected = args.expected_precommit_sha256
        if (expected is None or len(expected) != 64
                or any(character not in "0123456789abcdef"
                       for character in expected)):
            parser.error(
                "execute requires a lowercase 64-hex "
                "--expected-precommit-sha256")
        _phase_execute(args.out, expected)


if __name__ == "__main__":
    main()
