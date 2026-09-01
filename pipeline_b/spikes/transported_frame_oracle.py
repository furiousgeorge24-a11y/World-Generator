"""Single-head oracle for elevation-independent transported crop frames.

This private spike freezes the three seed-11 field-accretion identities in
source, builds exactly one tagged 120 km structural head, and translates each
formation-native crop by the measured transport of its own material identity.
The translated origins are snapped once on a fixed 64 km lattice.  Exact
sub-cell tag evidence must pass every formation/transport gate before the
first elevation call; failure stops the experiment immediately.

There is no crop search, clamping, retry, seed substitution, identity
substitution, or later-resolution ladder in this oracle.
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

from spikes import field_accretion_oracle as base


# --------------------------------------------------------------- protocol

SEED = 11
CONTINENTAL_BUDGET = 0.65
STRUCTURAL_SPACING_KM = 120.0
ORIGIN_SNAP_KM = 64.0
COLLAR_KM = 256.0
MIN_FOOTPRINT_FRACTION = 0.20
MAX_FOOTPRINT_FRACTION = 0.50
MIN_OWNER_FRACTION = 0.85
MIN_CAPTURE_FRACTION = 0.80
MIN_MEMBER_FRAME_FRACTION = 0.04
MIN_EXPECTED_MEMBERS = 2
MAX_EXPECTED_MEMBERS = 4

# This order is part of the evidence contract.  It is the exact order used
# by engine.tectonics for the four final material reads.
FINAL_SUBCELL_OFFSETS = (
    (-0.25, -0.25), (-0.25, 0.25),
    (0.25, -0.25), (0.25, 0.25),
)

# Frozen from the seed-11 formation-native oracle.  Member order is lexical
# and therefore stable.  Origins remain birth-frame evidence; they are never
# overwritten by their transported replacements.
FROZEN_GROUPS = (
    {
        "domain_id": "6e523bbde838aed9",
        "member_domain_ids": (
            "54742f5884ad5964",
            "89340d1f9b7b6216",
            "9406995be1a1060e",
            "a98d473e1bfb1c3e",
        ),
        "carrier_plate_ids": (11,),
        "formation_origin_xy_km": (17728.0, 4800.0),
    },
    {
        "domain_id": "d66854bfac91ded7",
        "member_domain_ids": (
            "09c73261d566e589",
            "0e8e51eac0f50c05",
            "56e1f9ec97bfd8dd",
            "6735e15ab88b1d08",
        ),
        "carrier_plate_ids": (14,),
        "formation_origin_xy_km": (9152.0, 12928.0),
    },
    {
        "domain_id": "41c6915061be585e",
        "member_domain_ids": (
            "3020175597839fe3",
            "563450b146839374",
            "736eb55aa97a8aec",
        ),
        "carrier_plate_ids": (2,),
        "formation_origin_xy_km": (15168.0, 17728.0),
    },
)

SOURCE_FINGERPRINT_FILES = tuple(dict.fromkeys((
    *base.SOURCE_FINGERPRINT_FILES,
    "spikes/transported_frame_oracle.py",
)))


def _source_fingerprint() -> dict:
    """Hash the complete allowlisted source closure for this oracle."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    per_file = {}
    for relative in SOURCE_FINGERPRINT_FILES:
        payload = (root / relative).read_bytes()
        file_digest = hashlib.sha256(payload).hexdigest()
        per_file[relative] = file_digest
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return {"combined_sha256": digest.hexdigest(), "files": per_file}


def _frozen_manifest_records() -> list[dict]:
    return [
        {
            "domain_id": item["domain_id"],
            "member_domain_ids": list(item["member_domain_ids"]),
            "carrier_plate_ids": list(item["carrier_plate_ids"]),
            "formation_origin_xy_km": list(
                item["formation_origin_xy_km"]),
        }
        for item in FROZEN_GROUPS
    ]


def _protocol_manifest() -> dict:
    return {
        "experiment": "field-accretion-transported-frame-oracle-v1",
        "manifest_role": "exclusive_pre_structure_protocol_precommit",
        "sequencing": {
            "written_before_any_structural_build": True,
            "transport_selection_written_before_any_elevation": True,
            "structural_build_count": 1,
            "structural_stages_km": [STRUCTURAL_SPACING_KM],
            "elevation_stages_km": [STRUCTURAL_SPACING_KM],
            "surface_process_stages": [],
        },
        "fixed_inputs": {
            "seed": SEED,
            "continental_budget": CONTINENTAL_BUDGET,
            "world_km": base.ATLAS_KM,
            "requested_structural_spacing_km": STRUCTURAL_SPACING_KM,
            "origin_snap_km": ORIGIN_SNAP_KM,
            "two_sided_collar_km": COLLAR_KM,
            "final_subcell_offsets_yx": [
                list(item) for item in FINAL_SUBCELL_OFFSETS],
            "frozen_groups": _frozen_manifest_records(),
        },
        "origin_rule": {
            "claim_scope": (
                "centroid-following translation only; this is not claimed "
                "to be a full rotating or deforming Lagrangian frame"
            ),
            "canonical_centroid": (
                "equal-area mean of canonical cell centres in each fixed "
                "member set with accretion_time <= the budget-fixed limit"
            ),
            "final_centroid": (
                "equal-area mean of exact winning continental tag sample "
                "coordinates (i + 0.5 + offset) * structural_ck"
            ),
            "displacement_xy_km": "final centroid minus canonical centroid",
            "unsnapped_origin_xy_km": (
                "formation_origin_xy_km plus displacement_xy_km"
            ),
            "snap": (
                "independent Decimal ROUND_HALF_UP of coordinate / 64 km, "
                "multiplied by 64 km, exactly once"
            ),
        },
        "pre_elevation_gates": {
            "exact_frozen_group_and_member_ids": True,
            "every_expected_member_global_tag_count": "> 0",
            "guarded_origin_range_km": [
                base.ATLAS_GUARD_KM,
                base.ATLAS_KM - base.ATLAS_GUARD_KM - base.FRAME_KM,
            ],
            "pairwise_frame_overlap_count": 0,
            "tag_footprint_fraction": [
                MIN_FOOTPRINT_FRACTION, MAX_FOOTPRINT_FRACTION],
            "tag_footprint_upper_bound_exclusive": True,
            "minimum_expected_owner_fraction": MIN_OWNER_FRACTION,
            "minimum_capture_fraction_each_expected_member":
                MIN_CAPTURE_FRACTION,
            "expected_member_count_inclusive": [
                MIN_EXPECTED_MEMBERS,
                MAX_EXPECTED_MEMBERS,
            ],
            "minimum_frame_fraction_each_expected_member":
                MIN_MEMBER_FRAME_FRACTION,
            "foreign_ids_at_or_above_significant_fraction": 0,
            "continental_tag_samples_in_complete_exact_two_sided_collar": 0,
        },
        "post_elevation_gates": {
            "all_frozen_transport_identities_present": True,
            "every_frozen_member_observed_on_emergent_land": True,
            "emergent_identity_set_equality_required": False,
            "minimum_land_owner_fraction": MIN_OWNER_FRACTION,
            "minimum_transport_capture_fraction": MIN_CAPTURE_FRACTION,
            "land_fraction": [
                MIN_FOOTPRINT_FRACTION, MAX_FOOTPRINT_FRACTION],
            "land_fraction_upper_bound_exclusive": True,
            "significant_component_count_inclusive": [
                MIN_EXPECTED_MEMBERS,
                MAX_EXPECTED_MEMBERS,
            ],
            "minimum_common_clearance_m": base.REQUIRED_CLEARANCE_M,
            "minimum_dense_clearance_m": base.REQUIRED_CLEARANCE_M,
            "common_exterior_ocean_coverage": 1.0,
            "dense_exterior_ocean_coverage": 1.0,
            "verified_two_sided_water_collar_km": COLLAR_KM,
            "common_visible_border_contour_gate": True,
            "dense_visible_border_contour_gate": True,
        },
        "input_allowlist": {
            "pre_structure": [
                "constants and frozen identities in this script",
                "fingerprinted source files listed in source_fingerprint",
            ],
            "transport_selection": [
                "budget-limited canonical formation support",
                "one 120-km Structure world_km and n metadata",
                "that Structure's four final _material_tag_samples",
            ],
            "post_selection": [
                "the same 120-km Structure",
                "coarse_elevation output from that Structure",
                "the exclusively recorded transported origins",
            ],
            "cli": ["output_directory_only"],
        },
        "prohibited_adaptation": {
            "crop_search": False,
            "origin_clamp": False,
            "retry": False,
            "seed_substitution": False,
            "identity_substitution": False,
            "group_substitution": False,
            "resolution_substitution": False,
            "elevation_in_transport_selection": False,
            "contour_input_in_transport_selection": False,
        },
        "source_fingerprint": _source_fingerprint(),
    }


def _write_json_exclusive(path: Path, payload: dict):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2))


def _prepare_empty_output(path: Path):
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(path)
        if any(path.iterdir()):
            raise FileExistsError(
                f"oracle output directory must be empty: {path}")
    else:
        path.mkdir(parents=True)


# --------------------------------------------------------- tag geometry

def _snap_half_up_64(value: float) -> float:
    quantum = Decimal(str(ORIGIN_SNAP_KM))
    units = (Decimal(str(float(value))) / quantum).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP)
    return float(units * quantum)


def _pin_audit(layout: dict) -> dict:
    """Reproduce the formation shortlist only to verify the fixed records."""
    domain_by_id = {
        item["domain_id"]: item for item in layout["domains"]}
    reproduced_frozen, _ = base._formation_crop_records(layout)
    reproduced_by_id = {
        item["domain_id"]: item for item in reproduced_frozen}
    records = []
    for frozen in FROZEN_GROUPS:
        member_ids = tuple(frozen["member_domain_ids"])
        members_present = all(item in domain_by_id for item in member_ids)
        group_key = "field-accretion-group-v1:" + ":".join(member_ids)
        computed_group_id = f"{base.fnv1a64(group_key):016x}"
        actual_carriers = sorted({
            int(domain_by_id[item]["carrier_plate_id"])
            for item in member_ids if item in domain_by_id
        })
        reproduced = reproduced_by_id.get(frozen["domain_id"])
        reproduced_members = (
            None if reproduced is None
            else tuple(reproduced["member_domain_ids"]))
        reproduced_origin = (
            None if reproduced is None
            else tuple(reproduced["origin_xy_km"]))
        exact = bool(
            members_present
            and tuple(sorted(member_ids)) == member_ids
            and computed_group_id == frozen["domain_id"]
            and actual_carriers == list(frozen["carrier_plate_ids"])
            and reproduced_members == member_ids
            and reproduced_origin == frozen["formation_origin_xy_km"]
        )
        records.append({
            "domain_id": frozen["domain_id"],
            "member_domain_ids": list(member_ids),
            "members_present_in_layout": members_present,
            "computed_group_id": computed_group_id,
            "actual_carrier_plate_ids": actual_carriers,
            "reproduced_member_domain_ids": (
                None if reproduced_members is None
                else list(reproduced_members)),
            "reproduced_formation_origin_xy_km": (
                None if reproduced_origin is None
                else list(reproduced_origin)),
            "passed": exact,
        })
    return {
        "records": records,
        "passed": len(records) == 3 and all(item["passed"] for item in records),
    }


def _material_tags(structure) -> np.ndarray:
    if not hasattr(structure, "_material_tag_samples"):
        raise ValueError("structural head has no material identity evidence")
    tags = np.asarray(structure._material_tag_samples)
    expected = (len(FINAL_SUBCELL_OFFSETS), structure.n, structure.n)
    if tags.shape != expected:
        raise ValueError(
            f"material identity evidence shape {tags.shape} != {expected}")
    if not np.issubdtype(tags.dtype, np.integer):
        raise ValueError("material identity evidence is not integer-valued")
    if tags.size and int(tags.min()) < -1:
        raise ValueError("material identity evidence contains a tag below -1")
    return tags.astype(np.int64, copy=False)


def _canonical_group_stats(layout: dict, frozen: dict) -> dict:
    domain_by_id = {
        item["domain_id"]: item for item in layout["domains"]}
    labels = np.asarray([
        domain_by_id[item]["label"]
        for item in frozen["member_domain_ids"]
    ], np.int64)
    limit = base._accretion_time_limit(CONTINENTAL_BUDGET)
    support = (
        np.isin(layout["domain_label"], labels)
        & (layout["accretion_time"] <= limit)
    )
    ys, xs = np.nonzero(support)
    centroid_yx = None
    if ys.size:
        q = layout["q"]
        centroid_yx = [float(np.mean(q[ys])), float(np.mean(q[xs]))]
    member_counts = {
        domain_id: int(np.count_nonzero(
            (layout["domain_label"] == domain_by_id[domain_id]["label"])
            & (layout["accretion_time"] <= limit)))
        for domain_id in frozen["member_domain_ids"]
    }
    return {
        "labels": labels,
        "centroid_yx_km": centroid_yx,
        "support_cells": int(ys.size),
        "member_support_cells": member_counts,
        "accretion_time_limit_km": float(limit),
    }


def _final_group_stats(tags: np.ndarray, structure,
                       labels: np.ndarray, member_ids: tuple[str, ...]) -> dict:
    ck = structure.world_km / structure.n
    total = 0
    sum_y = 0.0
    sum_x = 0.0
    for sample_index, (oy, ox) in enumerate(FINAL_SUBCELL_OFFSETS):
        ys, xs = np.nonzero(np.isin(tags[sample_index], labels))
        total += int(ys.size)
        sum_y += float(np.sum((ys + 0.5 + oy) * ck))
        sum_x += float(np.sum((xs + 0.5 + ox) * ck))
    centroid_yx = None
    if total:
        centroid_yx = [sum_y / total, sum_x / total]
    member_counts = {
        domain_id: int(np.count_nonzero(tags == int(label)))
        for domain_id, label in zip(member_ids, labels)
    }
    return {
        "centroid_yx_km": centroid_yx,
        "support_tag_samples": total,
        "member_global_tag_samples": member_counts,
    }


def _crop_tag_metrics(tags: np.ndarray, structure, layout: dict,
                      origin_xy_km: list[float], labels: np.ndarray,
                      member_ids: tuple[str, ...]) -> dict:
    ck = structure.world_km / structure.n
    base_q = (np.arange(structure.n) + 0.5) * ck
    x0, y0 = origin_xy_km
    domain_count = len(layout["domains"])
    crop_counts = np.zeros(domain_count, np.int64)
    crop_total = 0
    crop_continental = 0
    collar_total = 0
    collar_continental = 0

    for sample_index, (oy, ox) in enumerate(FINAL_SUBCELL_OFFSETS):
        sample_y = base_q + oy * ck
        sample_x = base_q + ox * ck
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
    expected_crop_counts = {
        domain_id: int(crop_counts[int(label)])
        for domain_id, label in zip(member_ids, labels)
    }
    expected_global_counts = {
        domain_id: int(global_counts[int(label)])
        for domain_id, label in zip(member_ids, labels)
    }
    capture_by_id = {
        domain_id: float(
            expected_crop_counts[domain_id]
            / max(expected_global_counts[domain_id], 1))
        for domain_id in member_ids
    }
    member_frame_fractions = {
        domain_id: float(expected_crop_counts[domain_id] / max(crop_total, 1))
        for domain_id in member_ids
    }
    significant = np.flatnonzero(
        crop_counts >= MIN_MEMBER_FRAME_FRACTION * max(crop_total, 1))
    significant_ids = [
        layout["domains"][int(label)]["domain_id"]
        for label in significant]
    expected_set = set(member_ids)
    foreign_significant_ids = [
        item for item in significant_ids if item not in expected_set]
    owner_fraction = float(
        sum(expected_crop_counts.values()) / max(crop_continental, 1))
    footprint = float(crop_continental / max(crop_total, 1))

    return {
        "crop_total_exact_tag_samples": crop_total,
        "crop_continental_tag_samples": crop_continental,
        "tag_footprint_fraction": footprint,
        "expected_owner_fraction": owner_fraction,
        "expected_crop_tag_samples": expected_crop_counts,
        "expected_global_tag_samples": expected_global_counts,
        "capture_by_domain_id": capture_by_id,
        "minimum_capture_fraction": min(capture_by_id.values(), default=0.0),
        "member_frame_fractions": member_frame_fractions,
        "significant_domain_ids": significant_ids,
        "foreign_significant_domain_ids": foreign_significant_ids,
        "collar_total_exact_tag_samples": collar_total,
        "collar_continental_tag_samples": collar_continental,
    }


def _frames_nonoverlap(left: dict, right: dict) -> bool:
    lx, ly = left["transported_origin_xy_km"]
    rx, ry = right["transported_origin_xy_km"]
    return bool(
        abs(lx - rx) >= base.FRAME_KM
        or abs(ly - ry) >= base.FRAME_KM)


def _transport_selection(structure, layout: dict, pin_audit: dict,
                         protocol_sha256: str,
                         source_fingerprint: dict
                         ) -> tuple[dict, list[dict]]:
    tags = _material_tags(structure)
    domain_count = len(layout["domains"])
    dense_domain_labels = bool(
        [int(item["label"]) for item in layout["domains"]]
        == list(range(domain_count)))
    valid_tags = tags[tags >= 0]
    tags_in_range = bool(
        valid_tags.size == 0 or int(valid_tags.max()) < domain_count)
    tag_layout_safe = bool(tags_in_range and dense_domain_labels)
    tag_hash = hashlib.sha256(
        np.ascontiguousarray(tags).view(np.uint8)).hexdigest()
    records = []
    evaluator_records = []

    domain_by_id = {
        item["domain_id"]: item for item in layout["domains"]}
    for frozen in FROZEN_GROUPS:
        canonical = _canonical_group_stats(layout, frozen)
        final = _final_group_stats(
            tags, structure, canonical["labels"],
            frozen["member_domain_ids"])
        canonical_centroid = canonical["centroid_yx_km"]
        final_centroid = final["centroid_yx_km"]
        origin = None
        unsnapped = None
        displacement_xy = None
        snap_residual = None
        if canonical_centroid is not None and final_centroid is not None:
            dy = final_centroid[0] - canonical_centroid[0]
            dx = final_centroid[1] - canonical_centroid[1]
            displacement_xy = [float(dx), float(dy)]
            old_x, old_y = frozen["formation_origin_xy_km"]
            unsnapped = [float(old_x + dx), float(old_y + dy)]
            origin = [
                _snap_half_up_64(unsnapped[0]),
                _snap_half_up_64(unsnapped[1]),
            ]
            snap_residual = [
                float(origin[0] - unsnapped[0]),
                float(origin[1] - unsnapped[1]),
            ]

        metrics = None
        if origin is not None and tag_layout_safe:
            metrics = _crop_tag_metrics(
                tags, structure, layout, origin, canonical["labels"],
                frozen["member_domain_ids"])
        member_count_ok = (
            MIN_EXPECTED_MEMBERS
            <= len(frozen["member_domain_ids"])
            <= MAX_EXPECTED_MEMBERS)
        nonzero_members = bool(
            all(value > 0 for value in
                canonical["member_support_cells"].values())
            and all(value > 0 for value in
                    final["member_global_tag_samples"].values()))
        guarded = bool(
            origin is not None
            and base.ATLAS_GUARD_KM <= origin[0]
            <= base.ATLAS_KM - base.ATLAS_GUARD_KM - base.FRAME_KM
            and base.ATLAS_GUARD_KM <= origin[1]
            <= base.ATLAS_KM - base.ATLAS_GUARD_KM - base.FRAME_KM)
        gate = {
            "exact_frozen_ids": bool(pin_audit["passed"]),
            "tag_labels_in_layout_range": tags_in_range,
            "domain_labels_dense_and_list_indexed": dense_domain_labels,
            "nonzero_expected_members": nonzero_members,
            "guarded_origin": guarded,
            "expected_member_count_2_to_4": member_count_ok,
            "tag_footprint_0_20_to_0_50_exclusive": False,
            "expected_owner_at_least_0_85": False,
            "every_expected_capture_at_least_0_80": False,
            "every_expected_id_at_least_0_04_frame": False,
            "no_foreign_significant_id": False,
            "significant_ids_exactly_equal_frozen_members": False,
            "zero_continental_tags_in_complete_two_sided_collar": False,
            "pairwise_nonoverlap": False,
        }
        if metrics is not None:
            gate.update({
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
                    all(value >= MIN_MEMBER_FRAME_FRACTION
                        for value in
                        metrics["member_frame_fractions"].values())),
                "no_foreign_significant_id": bool(
                    not metrics["foreign_significant_domain_ids"]),
                "significant_ids_exactly_equal_frozen_members": bool(
                    set(metrics["significant_domain_ids"])
                    == set(frozen["member_domain_ids"])),
                "zero_continental_tags_in_complete_two_sided_collar": bool(
                    metrics["collar_continental_tag_samples"] == 0),
            })
        records.append({
            "domain_id": frozen["domain_id"],
            "member_domain_ids": list(frozen["member_domain_ids"]),
            "carrier_plate_ids": list(frozen["carrier_plate_ids"]),
            "formation_origin_xy_km": list(
                frozen["formation_origin_xy_km"]),
            "canonical_centroid_xy_km": (
                None if canonical_centroid is None
                else [canonical_centroid[1], canonical_centroid[0]]),
            "final_centroid_xy_km": (
                None if final_centroid is None
                else [final_centroid[1], final_centroid[0]]),
            "centroid_displacement_xy_km": displacement_xy,
            "unsnapped_transported_origin_xy_km": unsnapped,
            "transported_origin_xy_km": origin,
            "snap_residual_xy_km": snap_residual,
            "canonical_support_cells": canonical["support_cells"],
            "canonical_member_support_cells":
                canonical["member_support_cells"],
            "final_support_tag_samples": final["support_tag_samples"],
            "final_member_global_tag_samples":
                final["member_global_tag_samples"],
            "crop_metrics": metrics,
            "gates": gate,
        })

        if origin is not None and metrics is not None:
            evaluator_records.append({
                "domain_id": frozen["domain_id"],
                "member_domain_ids": list(frozen["member_domain_ids"]),
                "carrier_plate_ids": list(frozen["carrier_plate_ids"]),
                "canonical_pivots_yx_km": [
                    domain_by_id[item]["pivot_yx_km"]
                    for item in frozen["member_domain_ids"]],
                "formation_origin_xy_km": list(
                    frozen["formation_origin_xy_km"]),
                "origin_xy_km": origin,
                "owner_fraction": metrics["expected_owner_fraction"],
                "capture_fraction": metrics["minimum_capture_fraction"],
            })

    pair_records = []
    pairwise_ok = len(records) == len(FROZEN_GROUPS)
    for left, right in combinations(records, 2):
        if (left["transported_origin_xy_km"] is None
                or right["transported_origin_xy_km"] is None):
            nonoverlap = False
        else:
            nonoverlap = _frames_nonoverlap(left, right)
        pairwise_ok = pairwise_ok and nonoverlap
        pair_records.append({
            "left_domain_id": left["domain_id"],
            "right_domain_id": right["domain_id"],
            "nonoverlap": nonoverlap,
        })
    for record in records:
        record["gates"]["pairwise_nonoverlap"] = bool(pairwise_ok)
        record["passed"] = bool(all(record["gates"].values()))

    passed = bool(
        len(records) == 3
        and len(evaluator_records) == 3
        and pairwise_ok
        and all(record["passed"] for record in records))
    selection = {
        "experiment": "field-accretion-transported-frame-oracle-v1",
        "manifest_role": "exclusive_pre_elevation_transport_selection",
        "protocol_precommit_file": "protocol_precommit.json",
        "protocol_precommit_sha256": protocol_sha256,
        "source_fingerprint": source_fingerprint,
        "sequencing": {
            "written_after_exactly_one_120km_structural_build": True,
            "written_before_any_elevation": True,
        },
        "seed": SEED,
        "continental_budget": CONTINENTAL_BUDGET,
        "requested_spacing_km": STRUCTURAL_SPACING_KM,
        "actual_spacing_km": float(structure.world_km / structure.n),
        "n": int(structure.n),
        "tag_evidence": {
            "sha256": tag_hash,
            "shape": list(tags.shape),
            "dtype": str(tags.dtype),
            "final_subcell_offsets_yx": [
                list(item) for item in FINAL_SUBCELL_OFFSETS],
            "continental_tag_samples": int(valid_tags.size),
            "tag_labels_in_layout_range": tags_in_range,
            "domain_labels_dense_and_list_indexed": dense_domain_labels,
        },
        "pin_audit": pin_audit,
        "pairwise_frames": pair_records,
        "groups": records,
        "formation_and_transport_only_feasibility": {
            "uses_elevation": False,
            "uses_contours": False,
            "uses_crop_search": False,
            "passed": passed,
        },
        "passed": passed,
    }
    return selection, evaluator_records


# ------------------------------------------------------------------- run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path("out") / "transported_frame_seed11",
    )
    args = parser.parse_args()
    _prepare_empty_output(args.out)

    # This is intentionally the first artifact and precedes formation-layout
    # construction as well as the sole structural build.
    protocol = _protocol_manifest()
    protocol_path = args.out / "protocol_precommit.json"
    _write_json_exclusive(protocol_path, protocol)
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()

    started = time.perf_counter()
    cfg = base._atlas_config(CONTINENTAL_BUDGET)
    layout = base._formation_layout(SEED, base.ATLAS_KM, int(cfg.plates))
    pin_audit = _pin_audit(layout)
    sample_continent = base._continent_sampler(
        SEED, layout, CONTINENTAL_BUDGET)
    sample_material_tag = base._material_tag_sampler(
        layout, CONTINENTAL_BUDGET)
    sites = base._plate_sites(SEED, base.ATLAS_KM, int(cfg.plates))

    # Sole structural build in this experiment.  Do not move this call into a
    # loop or repeat it for presentation evidence.
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
        structure, layout, pin_audit, protocol_sha256,
        protocol["source_fingerprint"])
    selection_path = args.out / "transport_selection.json"
    _write_json_exclusive(selection_path, selection)
    selection_sha256 = hashlib.sha256(
        selection_path.read_bytes()).hexdigest()

    if not selection["passed"]:
        report = {
            "experiment": selection["experiment"],
            "seed": SEED,
            "continental_budget": CONTINENTAL_BUDGET,
            "passed": False,
            "stop_stage": "pre_elevation_transport_selection",
            "elevation_called": False,
            "structural_build_count": 1,
            "stages_run": ["120km_structure_only"],
            "elapsed_s": time.perf_counter() - started,
            "protocol_precommit_sha256": protocol_sha256,
            "transport_selection_file": "transport_selection.json",
            "transport_selection_sha256": selection_sha256,
        }
        _write_json_exclusive(args.out / "report.json", report)
        print(json.dumps(report, indent=2))
        return

    # The exclusive selection record now exists; elevation may consume the
    # same in-memory Structure but cannot alter the frozen origins.
    elevation = base.coarse_elevation(structure, cfg, SEED)
    results, _ = base._evaluate_frozen_stage(
        structure, elevation, transported, layout)
    for result in results:
        result["all_frozen_transport_identities_present"] = result.pop(
            "transport_identity_exact")
        expected = set(result["member_domain_ids"])
        observed_land = set(result["observed_land_domain_ids"])
        emergent_identity_exact = bool(expected.issubset(observed_land))
        result["every_frozen_member_observed_on_emergent_land"] = (
            emergent_identity_exact)
        result["unexpected_emergent_identity_ids"] = sorted(
            observed_land - expected)
        result["passed"] = bool(result["passed"]
                                and emergent_identity_exact)
        result["pre_elevation_tag_owner_fraction"] = result.pop(
            "formation_owner_fraction")
        result["pre_elevation_tag_capture_fraction"] = result.pop(
            "formation_capture_fraction")
    base._head_panel(
        structure, elevation, transported, "120",
        args.out / "head_120km.png")

    stage_passed = bool(all(item["passed"] for item in results))
    report = {
        "experiment": selection["experiment"],
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
        "elapsed_s": time.perf_counter() - started,
        "protocol_precommit_sha256": protocol_sha256,
        "transport_selection_file": "transport_selection.json",
        "transport_selection_sha256": selection_sha256,
        "results": results,
    }
    _write_json_exclusive(args.out / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
