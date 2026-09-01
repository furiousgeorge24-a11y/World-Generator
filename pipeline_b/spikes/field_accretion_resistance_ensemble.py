"""Sealed formation-only test of continuous accretion resistance.

The rejected conserved-inventory predecessor confined growth to the positive
side of a hard assembly threshold.  This successor keeps its exact global
crust inventory, but makes every canonical cell traversable through one
positive continuous resistance law.  Nuclei come from the positive phase of
the already-existing broadest assembly octave; that broad field chooses where
continental assembly can begin, never where it must stop.

Formation has no crop, frame, border, elevation, sea-level, or target-window
input.  This private spike is not imported by the public registry or adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from engine import noise
from engine.rng import fnv1a64, stage_rng, stage_salt
from engine.tectonics import FRAME_KM, build_structure
from spikes import field_accretion_inventory_ensemble as base
from spikes import field_accretion_oracle as legacy


EXPERIMENT = "field-accretion-resistance-ensemble-seed159-166-v1"
SEEDS = tuple(range(159, 167))
EXPOSED_DEVELOPMENT_SEEDS = tuple(range(151, 159))
RUN_ROLE = "fresh_validation"
SEED_SELECTION_DESCRIPTION = (
    "mechanical next eight integers after exposed development seeds 151-158")
PARENT_KM = 6.0 * FRAME_KM
CANONICAL_KM = legacy.CANONICAL_KM
TARGET_INITIAL_CONTINENTAL_FRACTION = 0.28
PREFIX_INITIAL_CONTINENTAL_FRACTION = 0.14
STRUCTURE_NOMINAL_KM = 80.0
FINE_SENTINEL_SEED = SEEDS[0]
FINE_STRUCTURE_NOMINAL_KM = 40.0
NESTED_PADDING_KM = FRAME_KM

# This is the unique simple positive exponential continuation whose speed and
# first derivative match the legacy law at its former hard boundary:
# v(0.12)=0.72 and v'(0.12)=1.8, hence d(log v)/da=1.8/0.72=2.5.
RESISTANCE_REFERENCE_ASSEMBLY = legacy.CARRIER_THRESHOLD
RESISTANCE_REFERENCE_SPEED = 0.72
RESISTANCE_LOG_SENSITIVITY = 1.8 / RESISTANCE_REFERENCE_SPEED
BROAD_PROVINCE_PHASE = 0.0
ARRIVAL_CONTOUR_INTERVAL = FRAME_KM / 16.0

MIN_READY_SEEDS = len(SEEDS)
MIN_SENTINEL_RESOLUTION_IOU = base.MIN_SENTINEL_RESOLUTION_IOU
MAX_SENTINEL_FRACTION_DELTA = base.MAX_SENTINEL_FRACTION_DELTA
MIN_NESTED_INNER_IOU = base.MIN_NESTED_INNER_IOU

SOURCE_FILES = tuple(dict.fromkeys(
    (*base.SOURCE_FILES,
     "spikes/field_accretion_resistance_ensemble.py")))

PRIOR_EVIDENCE = {
    "hard_support_harness": (
        "spikes/field_accretion_inventory_ensemble.py",
        "773a58d29f2f13d1356b782309ef197cc57561209f9567d1c10a5f95433686a1",
    ),
    "hard_support_report": (
        "../out/field_accretion_inventory_ensemble_seed151_158_v1/report.json",
        "e15329c6bb9efa9624718e2bd74cbb6ed1bef4f7d6112136f7192ed4258f714c",
    ),
    "hard_support_manual_review": (
        "../out/field_accretion_inventory_ensemble_seed151_158_v1/manual_review.md",
        "525a613df8eff623ee673460d506efa7b57fd4c60c50edae59917abd43bd58e9",
    ),
}


def _configure_exposed_development() -> None:
    """Select the one fixed, already-exposed development suite."""
    global EXPERIMENT, SEEDS, FINE_SENTINEL_SEED, MIN_READY_SEEDS
    global RUN_ROLE, SEED_SELECTION_DESCRIPTION
    EXPERIMENT = "field-accretion-resistance-development-seed151-158-v1"
    SEEDS = EXPOSED_DEVELOPMENT_SEEDS
    FINE_SENTINEL_SEED = SEEDS[0]
    MIN_READY_SEEDS = len(SEEDS)
    RUN_ROLE = "exposed_development"
    SEED_SELECTION_DESCRIPTION = (
        "fixed previously exposed hard-support validation block; development "
        "evidence only and never promotion evidence")


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
        files[relative] = _sha256_bytes(payload)
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


def _absolute_fields(
        seed: int, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full assembly, broad first octave, and craton at absolute positions."""
    X, Y = np.meshgrid(q, q)
    salt = stage_salt(seed, "atlas-field-accretion-assembly-v1")
    assembly = noise.fbm(
        X, Y, legacy.ASSEMBLY_WAVELENGTH_KM,
        legacy.ASSEMBLY_OCTAVES, salt)
    broad = noise.fbm(
        X, Y, legacy.ASSEMBLY_WAVELENGTH_KM, 1, salt,
        norm_octaves=legacy.ASSEMBLY_OCTAVES)
    craton = noise.fbm(
        X, Y, legacy.CRATON_WAVELENGTH_KM,
        legacy.CRATON_OCTAVES,
        stage_salt(seed, "atlas-field-accretion-craton-v1"))
    return assembly, broad, craton


def _resistance(assembly: np.ndarray) -> np.ndarray:
    value = (
        np.exp(-RESISTANCE_LOG_SENSITIVITY * (
            np.asarray(assembly, np.float64)
            - RESISTANCE_REFERENCE_ASSEMBLY))
        / RESISTANCE_REFERENCE_SPEED)
    if not np.all(np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError("continuous resistance is not positive and finite")
    return value


def _discover_nuclei(seed: int, q: np.ndarray, broad: np.ndarray,
                     craton: np.ndarray) -> dict:
    """One strongest eligible craton per positive broad assembly province."""
    province_raw, province_components = legacy._label_components(
        broad > BROAD_PROVINCE_PHASE)
    eligible = (
        (province_raw >= 0)
        & (craton > legacy.NUCLEUS_CRATON_THRESHOLD))
    ties = base._coordinate_ties(seed, q)
    records = []
    for raw_label, (province_ys, province_xs) in enumerate(
            province_components):
        inside = eligible[province_ys, province_xs]
        if not inside.any():
            continue
        ys = province_ys[inside]
        xs = province_xs[inside]
        py, px = base._strongest_cell(craton, ys, xs, ties)
        records.append({
            "domain_id": legacy._field_id(
                seed, "resistance-domain", q[py], q[px]),
            "canonical_yx": [py, px],
            "pivot_yx_km": [float(q[py]), float(q[px])],
            "province_raw_label": int(raw_label),
            "eligible_craton_cells": int(inside.sum()),
            "peak_craton": float(craton[py, px]),
        })
    records.sort(key=lambda item: item["domain_id"])
    if not records:
        raise ValueError("broad assembly produced no eligible craton nucleus")
    return {
        "province_raw": province_raw,
        "province_components": province_components,
        "eligible_nuclei": eligible,
        "records": records,
    }


def _continuous_growth(
        assembly: np.ndarray,
        nuclei: list[dict],
        canonical_km: float,
        prefix_cells: int,
        target_cells: int,
        tie_grid: np.ndarray) -> dict:
    """One full-support chronology with exact 14% and 28% snapshots."""
    if not 0 < prefix_cells < target_cells <= assembly.size:
        raise ValueError((prefix_cells, target_cells, assembly.size))
    rho = _resistance(assembly)
    shape = assembly.shape
    arrival = np.full(shape, np.inf, np.float64)
    provisional_owner = np.full(shape, -1, np.int32)
    selected_owner = np.full(shape, -1, np.int32)
    prefix_owner = np.full(shape, -1, np.int32)
    settled = np.zeros(shape, bool)
    queue = []
    for domain_index, record in enumerate(nuclei):
        y, x = record["canonical_yx"]
        arrival[y, x] = 0.0
        provisional_owner[y, x] = domain_index
        heapq.heappush(queue, (
            0.0, int(tie_grid[y, x]), domain_index, y, x))

    neighbors = tuple(
        (dy, dx, float(np.hypot(dy, dx)))
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy or dx)
    settled_count = 0
    prefix_cutoff = None
    target_cutoff = None
    while queue and settled_count < target_cells:
        elapsed, tie, domain_index, y, x = heapq.heappop(queue)
        if settled[y, x]:
            continue
        if (elapsed != arrival[y, x]
                or domain_index != provisional_owner[y, x]):
            continue
        settled[y, x] = True
        selected_owner[y, x] = domain_index
        settled_count += 1
        if settled_count <= prefix_cells:
            prefix_owner[y, x] = domain_index
        if settled_count == prefix_cells:
            prefix_cutoff = {
                "arrival": float(elapsed), "tie": int(tie)}
        if settled_count == target_cells:
            target_cutoff = {
                "arrival": float(elapsed), "tie": int(tie)}
            break
        for dy, dx, diagonal in neighbors:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < shape[0] and 0 <= xx < shape[1]):
                continue
            if settled[yy, xx]:
                continue
            edge_cost = (
                canonical_km * diagonal
                * 0.5 * (rho[y, x] + rho[yy, xx]))
            if not np.isfinite(edge_cost) or edge_cost <= 0.0:
                raise AssertionError("edge resistance is not positive finite")
            candidate = elapsed + edge_cost
            old = arrival[yy, xx]
            old_owner = provisional_owner[yy, xx]
            candidate_key = (
                candidate, int(tie_grid[yy, xx]), domain_index)
            old_key = (old, int(tie_grid[yy, xx]), int(old_owner))
            if candidate_key < old_key:
                arrival[yy, xx] = candidate
                provisional_owner[yy, xx] = domain_index
                heapq.heappush(queue, (
                    float(candidate), int(tie_grid[yy, xx]),
                    domain_index, yy, xx))
    if settled_count != target_cells:
        raise AssertionError((settled_count, target_cells))
    if prefix_cutoff is None or target_cutoff is None:
        raise AssertionError("growth snapshots were not captured")
    return {
        "selected_owner": selected_owner,
        "prefix_owner": prefix_owner,
        "selected": selected_owner >= 0,
        "prefix_selected": prefix_owner >= 0,
        "arrival": arrival,
        "resistance": rho,
        "prefix_cells": int(prefix_cells),
        "target_cells": int(target_cells),
        "prefix_cutoff": prefix_cutoff,
        "target_cutoff": target_cutoff,
        "selected_cutoff_cohort_cells": int(np.count_nonzero(
            settled & (arrival == target_cutoff["arrival"]))),
        "discovered_unselected_cutoff_cohort_cells": int(np.count_nonzero(
            ~settled & np.isfinite(arrival)
            & (arrival == target_cutoff["arrival"]))),
    }


def _domain_boundary_stats(mask: np.ndarray, pivot_yx: tuple[int, int],
                           canonical_km: float) -> dict:
    mask = np.asarray(mask, bool)
    interior = np.zeros(mask.shape, bool)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1] & mask[2:, 1:-1]
        & mask[1:-1, :-2] & mask[1:-1, 2:])
    boundary = mask & ~interior
    ys, xs = np.nonzero(boundary)
    py, px = pivot_yx
    radii = canonical_km * np.hypot(ys - py, xs - px)
    mean = float(radii.mean()) if radii.size else 0.0
    return {
        "boundary_cells": int(radii.size),
        "radius_mean_km": mean,
        "radius_std_km": float(radii.std()) if radii.size else 0.0,
        "normalized_radius_cv": (
            float(radii.std() / mean) if mean > 0.0 else 0.0),
        "radius_p10_km": float(np.percentile(radii, 10.0)),
        "radius_p90_km": float(np.percentile(radii, 90.0)),
        "touches_world_rim": bool(
            mask[0].any() or mask[-1].any()
            or mask[:, 0].any() or mask[:, -1].any()),
    }


def _strict_predecessor_paths(selected_owner: np.ndarray,
                              arrival: np.ndarray,
                              nuclei: list[dict]) -> bool:
    for domain_index, record in enumerate(nuclei):
        pivot = tuple(record["canonical_yx"])
        mask = selected_owner == domain_index
        if not mask[pivot]:
            return False
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys, xs):
            if (y, x) == pivot:
                if arrival[y, x] != 0.0:
                    return False
                continue
            earlier = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = int(y + dy), int(x + dx)
                    if (0 <= yy < mask.shape[0]
                            and 0 <= xx < mask.shape[1]
                            and selected_owner[yy, xx] == domain_index
                            and arrival[yy, xx] < arrival[y, x]):
                        earlier = True
                        break
                if earlier:
                    break
            if not earlier:
                return False
    return True


def _inventory_layout(seed: int, world_km: float,
                      plate_count: int, *, origin_km: float = 0.0) -> dict:
    count = int(round(world_km / CANONICAL_KM))
    canonical_km = world_km / count
    if abs(canonical_km - CANONICAL_KM) > 1e-12:
        raise ValueError("world extent must preserve canonical lattice")
    q = origin_km + (np.arange(count) + 0.5) * canonical_km
    assembly, broad, craton = _absolute_fields(seed, q)
    discovery = _discover_nuclei(seed, q, broad, craton)
    nuclei = discovery["records"]
    if len(nuclei) > plate_count:
        raise ValueError("broad provinces exceed configured plate count")
    target_cells = int(round(
        TARGET_INITIAL_CONTINENTAL_FRACTION * assembly.size))
    prefix_cells = int(round(
        PREFIX_INITIAL_CONTINENTAL_FRACTION * assembly.size))
    ties = base._coordinate_ties(seed, q)
    growth = _continuous_growth(
        assembly, nuclei, canonical_km,
        prefix_cells, target_cells, ties)

    carrier_records = []
    carrier_plate_by_raw = {}
    for plate_id, record in enumerate(nuclei):
        raw_label = record["province_raw_label"]
        ys, xs = discovery["province_components"][raw_label]
        carrier_records.append({
            "raw_label": raw_label,
            "carrier_id": legacy._field_id(
                seed, "resistance-province",
                record["pivot_yx_km"][0], record["pivot_yx_km"][1]),
            "pivot_yx_km": record["pivot_yx_km"],
            "canonical_cells": int(ys.size),
            "touches_world_rim": bool(
                np.any(ys == 0) or np.any(xs == 0)
                or np.any(ys == count - 1)
                or np.any(xs == count - 1)),
            "plate_id": plate_id,
        })
        carrier_plate_by_raw[raw_label] = plate_id

    carrier_owner = np.full(assembly.shape, -1, np.int32)
    for raw_label, plate_id in carrier_plate_by_raw.items():
        carrier_owner[
            discovery["province_raw"] == raw_label] = plate_id

    domain_plate_by_label = np.arange(len(nuclei), dtype=np.int32)
    domain_records = []
    for label, record in enumerate(nuclei):
        mask = growth["selected_owner"] == label
        ys, xs = np.nonzero(mask)
        pivot = tuple(record["canonical_yx"])
        domain_records.append({
            "label": label,
            "domain_id": record["domain_id"],
            "carrier_plate_id": label,
            "carrier_raw_label": record["province_raw_label"],
            "pivot_yx_km": record["pivot_yx_km"],
            "canonical_yx": list(pivot),
            "canonical_cells": int(ys.size),
            "area_km2": float(ys.size * canonical_km ** 2),
            "nucleus_cells": 1,
            "eligible_craton_cells": record["eligible_craton_cells"],
            "peak_craton": record["peak_craton"],
            "connected_to_seed": base._connected_to_seed(mask, pivot),
            "centroid_yx_km": [
                float(np.mean(q[ys])), float(np.mean(q[xs]))],
            "bbox_xyxy_km": [
                float(q[xs.min()] - 0.5 * canonical_km),
                float(q[ys.min()] - 0.5 * canonical_km),
                float(q[xs.max()] + 0.5 * canonical_km),
                float(q[ys.max()] + 0.5 * canonical_km)],
            **_domain_boundary_stats(mask, pivot, canonical_km),
        })

    selected = growth["selected"]
    prefix = growth["prefix_selected"]
    selected_assembly = assembly[selected]
    unselected_assembly = assembly[~selected]
    prefix_checks = {
        "exact_prefix_cells": int(prefix.sum()) == prefix_cells,
        "exact_target_cells": int(selected.sum()) == target_cells,
        "strict_mask_prefix": bool(
            np.all(~prefix | selected) and np.any(selected & ~prefix)),
        "owner_identical_on_prefix": bool(np.array_equal(
            growth["prefix_owner"][prefix],
            growth["selected_owner"][prefix])),
        "arrival_finite_on_prefix": bool(
            np.all(np.isfinite(growth["arrival"][prefix]))),
    }
    prefix_checks["passed"] = all(prefix_checks.values())
    nucleus_mask = np.zeros(assembly.shape, bool)
    for record in nuclei:
        nucleus_mask[tuple(record["canonical_yx"])] = True
    domain_id_by_label = np.asarray(
        [int(record["domain_id"], 16) for record in nuclei],
        dtype=np.uint64)
    selected_domain_id = np.zeros(assembly.shape, np.uint64)
    selected_domain_id[selected] = domain_id_by_label[
        growth["selected_owner"][selected]]
    prefix_domain_id = np.zeros(assembly.shape, np.uint64)
    prefix_domain_id[prefix] = domain_id_by_label[
        growth["prefix_owner"][prefix]]
    layout = {
        "canonical_km": canonical_km,
        "q": q,
        "assembly": assembly,
        "broad_assembly": broad,
        "craton": craton,
        "resistance": growth["resistance"],
        "province_raw": discovery["province_raw"],
        "eligible_nuclei": discovery["eligible_nuclei"],
        "nucleus_mask": nucleus_mask,
        "carrier_owner": carrier_owner,
        "selected": selected,
        "prefix_selected": prefix,
        "domain_label": growth["selected_owner"],
        "prefix_domain_label": growth["prefix_owner"],
        "domain_id_grid": selected_domain_id,
        "prefix_domain_id_grid": prefix_domain_id,
        "domain_plate_by_label": domain_plate_by_label,
        "carriers": carrier_records,
        "domains": domain_records,
        "nuclei": nuclei,
        "requested_cells": target_cells,
        "selected_cells": int(selected.sum()),
        "prefix_cells": int(prefix.sum()),
        "capacity_cells": int(assembly.size),
        "capacity_passed": True,
        "arrival": growth["arrival"],
        "prefix_cutoff": growth["prefix_cutoff"],
        "target_cutoff": growth["target_cutoff"],
        "selected_cutoff_cohort_cells":
            growth["selected_cutoff_cohort_cells"],
        "discovered_unselected_cutoff_cohort_cells":
            growth["discovered_unselected_cutoff_cohort_cells"],
        "target_fraction": TARGET_INITIAL_CONTINENTAL_FRACTION,
        "selected_fraction": float(selected.mean()),
        "prefix_checks": prefix_checks,
        "strict_predecessor_paths": _strict_predecessor_paths(
            growth["selected_owner"], growth["arrival"], nuclei),
        "resistance_diagnostics": {
            "selected_assembly_mean": float(selected_assembly.mean()),
            "unselected_assembly_mean": float(unselected_assembly.mean()),
            "selected_assembly_median": float(np.median(selected_assembly)),
            "unselected_assembly_median": float(
                np.median(unselected_assembly)),
            "selected_below_minus_0p10_fraction": float(
                np.mean(selected_assembly < -0.10)),
            "selected_inside_positive_broad_phase_fraction": float(
                np.mean(broad[selected] > BROAD_PROVINCE_PHASE)),
            "selected_inside_legacy_carrier_fraction": float(
                np.mean(assembly[selected] > legacy.CARRIER_THRESHOLD)),
            "resistance_min": float(growth["resistance"].min()),
            "resistance_median": float(np.median(growth["resistance"])),
            "resistance_max": float(growth["resistance"].max()),
        },
    }
    return layout


def _plate_sites(seed: int, layout: dict, world_km: float,
                 plate_count: int) -> np.ndarray:
    """Quota-independent sites: nucleus pivots plus maximin background sites."""
    parent_sites = np.asarray(
        [item["pivot_yx_km"] for item in layout["carriers"]],
        np.float64)
    remaining = plate_count - parent_sites.shape[0]
    rng = stage_rng(seed, "atlas-continuous-resistance-ocean-sites-v1")
    proposal_count = max(8192, 192 * plate_count)
    candidates = rng.uniform(
        0.02 * world_km, 0.98 * world_km,
        (proposal_count, 2))
    chosen = [point.copy() for point in parent_sites]
    if chosen:
        min_distance2 = np.min(
            ((candidates[:, None, :] - parent_sites[None, :, :]) ** 2)
            .sum(axis=2), axis=1)
    else:
        min_distance2 = np.full(candidates.shape[0], np.inf)
    available = np.ones(candidates.shape[0], bool)
    for _ in range(remaining):
        scores = np.where(available, min_distance2, -1.0)
        selected = int(np.argmax(scores))
        site = candidates[selected]
        chosen.append(site.copy())
        available[selected] = False
        distance2 = ((candidates - site) ** 2).sum(axis=1)
        min_distance2 = np.minimum(min_distance2, distance2)
    return np.asarray(chosen, np.float64)


def _make_partitioner(layout: dict, sites: np.ndarray,
                      expected_seed: int, expected_world_km: float):
    def partition(seed, n, ck, cfg):
        world = n * ck
        if seed != expected_seed or abs(world - expected_world_km) > 1e-9:
            raise ValueError("partition request differs from frozen layout")
        q = (np.arange(n) + 0.5) * ck
        X, Y = np.meshgrid(q, q)
        deformation_km = 0.045 * world
        wave = world / 3.5
        salt = stage_salt(seed, "atlas-field-accretion-partition-v1")
        dy = deformation_km * noise.fbm(X, Y, wave, 4, salt)
        dx = deformation_km * noise.fbm(X, Y, wave, 4, salt + 1)
        site_dy = deformation_km * noise.fbm(
            sites[:, 1], sites[:, 0], wave, 4, salt)
        site_dx = deformation_km * noise.fbm(
            sites[:, 1], sites[:, 0], wave, 4, salt + 1)
        Yw, Xw = Y + dy, X + dx
        best = np.full((n, n), np.inf)
        label = np.zeros((n, n), np.int32)
        for plate, (site_y, site_x) in enumerate(sites):
            cost = np.hypot(
                Yw - (site_y + site_dy[plate]),
                Xw - (site_x + site_dx[plate]))
            take = cost < best
            best[take] = cost[take]
            label[take] = plate
        domain = legacy._nearest_canonical(
            layout["domain_label"], Y, X,
            layout["canonical_km"], fill=-1)
        selected = domain >= 0
        owner = np.full(domain.shape, -1, np.int32)
        owner[selected] = layout["domain_plate_by_label"][domain[selected]]
        label[selected] = owner[selected]
        return label

    return partition


def _build(seed: int, layout: dict, nominal_km: float):
    cfg = legacy._atlas_config(TARGET_INITIAL_CONTINENTAL_FRACTION)
    sites = _plate_sites(seed, layout, PARENT_KM, int(cfg.plates))
    continent, material_tag = base._make_samplers(layout)
    structure = build_structure(
        seed, cfg,
        _world_km=PARENT_KM,
        _coarse_km=nominal_km,
        _continent_seeder=legacy._continent_seeder(continent),
        _partitioner=_make_partitioner(layout, sites, seed, PARENT_KM),
        _initial_age_sampler=legacy._initial_ocean_age,
        _plate_pivots=sites,
        _continent_sampler=continent,
        _material_tag_sampler=material_tag,
    )
    return structure, cfg, sites


def _seed_result(seed: int, layout: dict, structure,
                 cfg, sites: np.ndarray):
    authority = base._sample_structure_authority(structure, layout["q"])
    canonical_scan = base._scan_windows(layout["selected"])
    transported_scan = base._scan_windows(authority["proxy"])
    qualification = base._qualify_scan_morphology(
        transported_scan, authority["binary"],
        authority["dominant_tag"])
    transported_scan["selection"] = qualification.pop("selection")
    transported_scan["morphology_qualification"] = qualification
    selection = transported_scan["selection"]
    window_reviews = base._assigned_window_reviews(
        authority["binary"], authority["dominant_tag"], selection)

    partition = _make_partitioner(
        layout, sites, seed, PARENT_KM)(
            seed, layout["q"].size, layout["canonical_km"], cfg)
    selected = layout["selected"]
    expected_plate = np.full(layout["domain_label"].shape, -1, np.int32)
    expected_plate[selected] = layout["domain_plate_by_label"][
        layout["domain_label"][selected]]
    source_plate_consistent = bool(np.array_equal(
        partition[selected], expected_plate[selected]))

    domains = base._domain_summary(layout)
    represented = base._represented_domains(structure)
    nucleus_selected = bool(np.all(layout["selected"][
        layout["nucleus_mask"]]))
    resistance_preference = (
        layout["resistance_diagnostics"]["selected_assembly_mean"]
        > layout["resistance_diagnostics"]["unselected_assembly_mean"]
        and layout["resistance_diagnostics"]["selected_assembly_median"]
        > layout["resistance_diagnostics"]["unselected_assembly_median"])
    inventory_exact = (
        layout["selected_cells"] == layout["requested_cells"]
        and layout["selected_cells"]
        == int(round(TARGET_INITIAL_CONTINENTAL_FRACTION
                     * layout["selected"].size)))
    formation_invariants = all((
        inventory_exact,
        layout["prefix_checks"]["passed"],
        nucleus_selected,
        domains["all_connected_to_seed"],
        layout["strict_predecessor_paths"],
        domains["starved_domain_count"] == 0,
        len(layout["domains"]) == len(layout["carriers"]),
        source_plate_consistent,
        resistance_preference,
    ))
    ready_gates = {
        "inventory_exact": inventory_exact,
        "formation_invariants": formation_invariants,
        "all_domains_represented": (
            represented["represented_domain_count"]
            == domains["active_domain_count"]),
        "transported_proxy_assignment": selection["found"],
        "assigned_window_components": window_reviews["passed"],
    }
    substantial = [
        item for item in layout["domains"]
        if item["canonical_cells"]
        >= domains["substantial_threshold_cells"]]
    radial_cvs = np.asarray(
        [item["normalized_radius_cv"] for item in substantial],
        np.float64)
    result = {
        "seed": seed,
        "status": "complete",
        "inventory": {
            "requested_cells": layout["requested_cells"],
            "selected_cells": layout["selected_cells"],
            "prefix_cells": layout["prefix_cells"],
            "world_cells": int(layout["selected"].size),
            "target_world_fraction": TARGET_INITIAL_CONTINENTAL_FRACTION,
            "selected_world_fraction": layout["selected_fraction"],
            "quantization_error_cells": float(
                layout["selected_cells"]
                - TARGET_INITIAL_CONTINENTAL_FRACTION
                * layout["selected"].size),
            "prefix_cutoff": layout["prefix_cutoff"],
            "target_cutoff": layout["target_cutoff"],
            "selected_cutoff_cohort_cells":
                layout["selected_cutoff_cohort_cells"],
            "discovered_unselected_cutoff_cohort_cells":
                layout["discovered_unselected_cutoff_cohort_cells"],
        },
        "nucleation": {
            "broad_positive_component_count": int(
                np.max(layout["province_raw"]) + 1),
            "active_nucleus_count": len(layout["nuclei"]),
            "nuclei": [{
                "domain_id": item["domain_id"],
                "pivot_yx_km": item["pivot_yx_km"],
                "province_raw_label": item["province_raw_label"],
                "eligible_craton_cells": item["eligible_craton_cells"],
                "peak_craton": item["peak_craton"],
            } for item in layout["nuclei"]],
        },
        "resistance": layout["resistance_diagnostics"],
        "domains": domains,
        "domain_records": layout["domains"],
        "domain_shape_diagnostics": {
            "substantial_domain_count": len(substantial),
            "substantial_normalized_radius_cv_min": (
                float(radial_cvs.min()) if radial_cvs.size else None),
            "substantial_normalized_radius_cv_median": (
                float(np.median(radial_cvs)) if radial_cvs.size else None),
            "substantial_normalized_radius_cv_max": (
                float(radial_cvs.max()) if radial_cvs.size else None),
        },
        "prefix_checks": layout["prefix_checks"],
        "source_plate_consistent": source_plate_consistent,
        "transport": {
            "structure_n": int(structure.n),
            "actual_structure_km": float(
                structure.world_km / structure.n),
            "alive_plates": int(structure.alive_plates),
            **represented,
        },
        "canonical_scan": base._scan_report(canonical_scan),
        "transported_proxy_scan": base._scan_report(transported_scan),
        "assigned_window_reviews": window_reviews,
        "transported_global_proxy_fraction": float(
            authority["proxy"].mean()),
        "transported_global_binary_fraction": float(
            authority["binary"].mean()),
        "ready_gates": ready_gates,
        "ready": all(ready_gates.values()),
    }
    return result, authority, canonical_scan, transported_scan


def _render_resistance_diagnostic(seed: int, layout: dict,
                                  out: Path) -> dict:
    shape = layout["selected"].shape[0]
    left_rgb = np.zeros((shape, shape, 3), np.uint8)
    left_rgb[:] = np.asarray((10, 27, 47), np.uint8)
    for raw_label in np.unique(
            layout["province_raw"][layout["province_raw"] >= 0]):
        left_rgb[layout["province_raw"] == raw_label] = base._palette(
            int(raw_label))
    log_rho = np.log(layout["resistance"])
    lo, hi = np.percentile(log_rho, (2.0, 98.0))
    shade = np.clip((log_rho - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    left_rgb = np.clip(
        0.70 * left_rgb + 55.0 * shade[..., None], 0, 255).astype(np.uint8)

    right_rgb = np.zeros((shape, shape, 3), np.uint8)
    right_rgb[:] = np.asarray((10, 27, 47), np.uint8)
    for domain in layout["domains"]:
        label = domain["label"]
        mask = layout["domain_label"] == label
        bands = np.floor(
            layout["arrival"][mask] / ARRIVAL_CONTOUR_INTERVAL).astype(int)
        factors = 0.52 + 0.16 * (bands % 4)
        color = base._palette(label).astype(np.float64)
        right_rgb[mask] = np.clip(
            factors[:, None] * color[None, :], 0, 255).astype(np.uint8)

    left = Image.fromarray(left_rgb, "RGB").resize(
        (768, 768), Image.Resampling.NEAREST)
    right = Image.fromarray(right_rgb, "RGB").resize(
        (768, 768), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (1536, 816), (7, 16, 28))
    canvas.paste(left, (0, 48))
    canvas.paste(right, (768, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12),
              f"seed {seed} - broad nucleation provinces / resistance",
              fill=(238, 238, 225))
    draw.text((780, 12),
              (f"selected domains - arrival bands every "
               f"{ARRIVAL_CONTOUR_INTERVAL:.0f} resistance-km"),
              fill=(238, 238, 225))
    scale = 768.0 / shape
    for record in layout["nuclei"]:
        y, x = record["canonical_yx"]
        for xoff in (0, 768):
            cx = xoff + int(round((x + 0.5) * scale))
            cy = 48 + int(round((y + 0.5) * scale))
            draw.rectangle(
                (cx - 3, cy - 3, cx + 3, cy + 3),
                fill=(255, 250, 205))
    path = out / f"seed{seed}_resistance_diagnostic.png"
    canvas.save(path)
    return {"file": path.name, "sha256": _sha256_file(path)}


def _guarded_nucleus_signature(layout: dict, guard_cells: int) -> list[tuple]:
    low = float(layout["q"][guard_cells])
    high = float(layout["q"][-guard_cells - 1])
    return sorted(
        (item["domain_id"], tuple(item["pivot_yx_km"]))
        for item in layout["nuclei"]
        if (low <= item["pivot_yx_km"][0] <= high
            and low <= item["pivot_yx_km"][1] <= high))


def _nested_probe(seed: int, plate_count: int, small: dict) -> dict:
    padded_world = PARENT_KM + 2.0 * NESTED_PADDING_KM
    padded = _inventory_layout(
        seed, padded_world, plate_count,
        origin_km=-NESTED_PADDING_KM)
    offset = int(round(NESTED_PADDING_KM / CANONICAL_KM))
    size = small["selected"].shape[0]
    central_slice = np.s_[offset:offset + size, offset:offset + size]
    central = padded["selected"][central_slice]
    central_prefix = padded["prefix_selected"][central_slice]
    central_domain_id = padded["domain_id_grid"][central_slice]
    central_prefix_domain_id = padded["prefix_domain_id_grid"][central_slice]
    guard = int(round(FRAME_KM / CANONICAL_KM))
    small_inner = small["selected"][guard:-guard, guard:-guard]
    padded_inner = central[guard:-guard, guard:-guard]
    small_prefix_inner = small["prefix_selected"][guard:-guard, guard:-guard]
    padded_prefix_inner = central_prefix[guard:-guard, guard:-guard]
    small_domain_id_inner = small["domain_id_grid"][
        guard:-guard, guard:-guard]
    padded_domain_id_inner = central_domain_id[
        guard:-guard, guard:-guard]
    small_prefix_domain_id_inner = small["prefix_domain_id_grid"][
        guard:-guard, guard:-guard]
    padded_prefix_domain_id_inner = central_prefix_domain_id[
        guard:-guard, guard:-guard]
    target_overlap = small_inner & padded_inner
    prefix_overlap = small_prefix_inner & padded_prefix_inner
    target_owner_identical = bool(
        target_overlap.any()
        and np.array_equal(
            small_domain_id_inner[target_overlap],
            padded_domain_id_inner[target_overlap]))
    prefix_owner_identical = bool(
        prefix_overlap.any()
        and np.array_equal(
            small_prefix_domain_id_inner[prefix_overlap],
            padded_prefix_domain_id_inner[prefix_overlap]))
    addressing = {
        "assembly_identical": bool(np.array_equal(
            small["assembly"], padded["assembly"][central_slice])),
        "broad_assembly_identical": bool(np.array_equal(
            small["broad_assembly"],
            padded["broad_assembly"][central_slice])),
        "craton_identical": bool(np.array_equal(
            small["craton"], padded["craton"][central_slice])),
        "coordinate_ties_identical": bool(np.array_equal(
            base._coordinate_ties(seed, small["q"]),
            base._coordinate_ties(seed, padded["q"])[central_slice])),
        "broad_positive_phase_identical": bool(np.array_equal(
            small["broad_assembly"] > BROAD_PROVINCE_PHASE,
            padded["broad_assembly"][central_slice]
            > BROAD_PROVINCE_PHASE)),
        "eligible_nucleus_field_identical": bool(np.array_equal(
            small["eligible_nuclei"],
            padded["eligible_nuclei"][central_slice])),
        "guarded_nucleus_identity_identical": (
            _guarded_nucleus_signature(small, guard)
            == _guarded_nucleus_signature(padded, guard + offset)),
        "target_owner_identity_on_inner_overlap": target_owner_identical,
        "prefix_owner_identity_on_inner_overlap": prefix_owner_identical,
    }
    addressing_passed = all(addressing.values())
    target_inner_iou = base._mask_iou(small_inner, padded_inner)
    prefix_inner_iou = base._mask_iou(
        small_prefix_inner, padded_prefix_inner)
    return {
        "padded_world_km": padded_world,
        "padding_km_each_side": NESTED_PADDING_KM,
        "padded_inventory_exact": (
            padded["selected_cells"] == padded["requested_cells"]),
        "absolute_addressing_checks": addressing,
        "absolute_addressing_passed": addressing_passed,
        "full_parent_iou": base._mask_iou(small["selected"], central),
        "full_parent_prefix_iou": base._mask_iou(
            small["prefix_selected"], central_prefix),
        "inner_guard_km": FRAME_KM,
        "inner_iou": target_inner_iou,
        "inner_prefix_iou": prefix_inner_iou,
        "inner_target_owner_identity_on_overlap": target_owner_identical,
        "inner_prefix_owner_identity_on_overlap": prefix_owner_identical,
        "passed": (
            padded["selected_cells"] == padded["requested_cells"]
            and addressing_passed
            and target_inner_iou >= MIN_NESTED_INNER_IOU
            and prefix_inner_iou >= MIN_NESTED_INNER_IOU),
        "central_mask": central,
        "central_prefix_mask": central_prefix,
        "central_domain_id": central_domain_id,
        "central_prefix_domain_id": central_prefix_domain_id,
        "padded_full_mask": padded["selected"],
        "padded_full_prefix_mask": padded["prefix_selected"],
        "padded_full_domain_id": padded["domain_id_grid"],
        "padded_full_prefix_domain_id": padded["prefix_domain_id_grid"],
    }


def _protocol() -> dict:
    cfg = legacy._atlas_config(TARGET_INITIAL_CONTINENTAL_FRACTION)
    return {
        "experiment": EXPERIMENT,
        "manifest_role": (
            f"exclusive_pre_generation_protocol_precommit_{RUN_ROLE}"),
        "source_fingerprint": _source_fingerprint(),
        "prior_evidence": _verify_prior_evidence(),
        "seed_policy": {
            "seeds": list(SEEDS),
            "selection": SEED_SELECTION_DESCRIPTION,
            "evidence_role": RUN_ROLE,
            "exposed_development_seeds_not_validation":
                list(EXPOSED_DEVELOPMENT_SEEDS),
            "retry": None,
            "arbitrary_seed_cli": None,
            "fixed_suite_cli": (
                "--exposed-development"
                if RUN_ROLE == "exposed_development" else None),
            "fresh_validation_is_default_without_suite_flag": (
                RUN_ROLE == "fresh_validation"),
            "all_seeds_count": True,
        },
        "formation": {
            "world_km": PARENT_KM,
            "canonical_km": CANONICAL_KM,
            "target_initial_continental_fraction":
                TARGET_INITIAL_CONTINENTAL_FRACTION,
            "diagnostic_prefix_fraction":
                PREFIX_INITIAL_CONTINENTAL_FRACTION,
            "inventory_semantics": (
                "exact world-area crust inventory; 14% and 28% are snapshots "
                "from one global chronology"),
            "broad_nucleation_field": (
                "first 5000-km octave of existing assembly stack, normalized "
                "as the unchanged three-octave stack"),
            "broad_nucleation_phase": BROAD_PROVINCE_PHASE,
            "broad_field_scope": (
                "nucleus discovery and diagnostics only; it is neither "
                "growth support nor a plate-boundary override"),
            "nucleus_rule": (
                "one strongest craton>0.20 point per positive broad "
                "component; components without an eligible point get no "
                "fallback"),
            "resistance_law": (
                "rho(a)=exp(-2.5*(a-0.12))/0.72; symmetric edge cost is "
                "distance times endpoint-mean rho"),
            "resistance_reference_assembly":
                RESISTANCE_REFERENCE_ASSEMBLY,
            "resistance_reference_speed": RESISTANCE_REFERENCE_SPEED,
            "resistance_log_sensitivity": RESISTANCE_LOG_SENSITIVITY,
            "support": "all canonical cells; no hard carrier wall",
            "plate_binding": (
                "each nucleus has a fixed plate; every selected domain cell "
                "is overlaid onto that source plate; remaining plate sites "
                "are quota-independent maximin proposals"),
            "per_domain_quota": None,
            "crop_border_elevation_sea_level_or_target_input": None,
            "configured_plates": int(cfg.plates),
        },
        "execution": {
            "primary_inventory_chronologies": len(SEEDS),
            "diagnostic_prefix_reruns": 0,
            "nested_inventory_chronologies": 1,
            "coarse_structure_builds": len(SEEDS),
            "coarse_structure_nominal_km": STRUCTURE_NOMINAL_KM,
            "fine_sentinel_builds": 1,
            "fine_sentinel_seed": FINE_SENTINEL_SEED,
            "fine_structure_nominal_km": FINE_STRUCTURE_NOMINAL_KM,
            "elevation_builds": 0,
            "surface_process_solves": 0,
            "selectors_start_only_after_each_structure": True,
        },
        "post_structure_composition_screen": {
            "guard_km": base.NUMERICAL_GUARD_KM,
            "stride_km": base.CANDIDATE_STRIDE_KM,
            "candidate_count_per_mask": 61 ** 2,
            "target_bands": {
                label: {"minimum": band[0], "maximum": band[1],
                        "maximum_inclusive": band[2]}
                for label, band in base.TARGET_BANDS.items()},
            "proxy_semantics": (
                "mean transported continental occupancy; not final land"),
            "assignment_candidate_eligibility": (
                "target band plus frozen component-count/coverage gate "
                "before exact separated triplet search"),
            "selection_authority": (
                "morphology-qualified transported selection; composition-"
                "only selection is diagnostic"),
            "minimum_origin_chebyshev_separation_km":
                base.MIN_ORIGIN_CHEBYSHEV_SEPARATION_KM,
            "water_or_contour_gate": None,
        },
        "automatic_readiness_gates": {
            "exact_inventory_and_formation_invariants": "8 of 8 seeds",
            "ready_seed_count": f"{MIN_READY_SEEDS} of {len(SEEDS)}",
            "ready_seed_requires": [
                "exact 14% prefix and exact 28% inventory from one chronology",
                "positive finite symmetric resistance and field preference",
                "strict earlier path from every selected cell to its nucleus",
                "every selected cell initially belongs to its source plate",
                "every domain receives >1 selected cell and survives transport",
                "one separated morphology-qualified low/medium/high assignment",
                (f"each assigned window has {base.MIN_SIGNIFICANT_COMPONENTS}-"
                 f"{base.MAX_SIGNIFICANT_COMPONENTS} significant components"),
                (f"significant components cover >= "
                 f"{base.MIN_SIGNIFICANT_COMPONENT_COVERAGE:.2f} of binary "
                 "proxy land"),
            ],
            "fine_sentinel_iou": f">= {MIN_SENTINEL_RESOLUTION_IOU}",
            "fine_sentinel_fraction_delta":
                f"<= {MAX_SENTINEL_FRACTION_DELTA}",
            "fine_sentinel_morphology_qualified_assignment": True,
            "nested_target_and_prefix_inner_iou":
                f">= {MIN_NESTED_INNER_IOU}",
            "nested_owner_authority": (
                "stable domain IDs must be identical on all overlapping "
                "guarded-inner target and prefix cells"),
        },
        "manual_readiness_gate": {
            "required_after_automatic_readiness": True,
            "batch_veto": (
                "repeated round/equal-area bodies, common-radius or isochrone "
                "domain shells (excluding the deliberately drawn diagnostic "
                "arrival bands), broad-zero-contour stamping, carrier lace/"
                "tendrils, grid/tie directionality, or one ubiquitous amoeba "
                "blocks a full solve"),
        },
        "persisted_evidence": [
            "full assembly, broad assembly, craton, resistance, province, "
            "nucleus, 14%, 28%, stable owner ID, and selected-arrival arrays "
            "for every seed",
            "all transported masks and every complete scan/status row",
            "fixed-interval arrival-band diagnostics and formation panels",
            "complete padded target and prefix authority",
        ],
        "interpretation": [
            "Formation-only readiness cannot validate final water borders.",
            "Exact initial inventory cannot promise final visible land.",
            "Regularity is a causal tripwire and requires manual review.",
            "The nested probe measures but cannot alone close finite-rim causality.",
            (
                "Passing exposed development only merits sealing fresh "
                "validation; it is never promotion or full-solve evidence."
                if RUN_ROLE == "exposed_development" else
                "Passing fresh validation plus manual review permits one "
                "later full solve; it changes no production behavior."
            ),
        ],
    }


def _require_execute_output(out: Path,
                            expected_sha256: str) -> tuple[dict, str]:
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
    base._prepare_empty_output(out)
    payload = _protocol()
    value = base._write_json_exclusive(
        out / "protocol_precommit.json", payload)
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
    cfg = legacy._atlas_config(TARGET_INITIAL_CONTINENTAL_FRACTION)
    counters = {
        "inventory_chronologies": 0,
        "primary_prefix_snapshots": 0,
        "coarse_structure_builds": 0,
        "fine_sentinel_builds": 0,
        "elevation_builds": 0,
        "surface_process_solves": 0,
        "post_structure_scans": 0,
    }
    results = []
    layouts = {}
    masks = {}
    scans = {}
    panel_artifacts = []
    assignment_artifacts = []
    resistance_artifacts = []
    for seed in SEEDS:
        layout = _inventory_layout(seed, PARENT_KM, int(cfg.plates))
        counters["inventory_chronologies"] += 1
        counters["primary_prefix_snapshots"] += 1
        layouts[seed] = layout
        structure, build_cfg, sites = _build(
            seed, layout, STRUCTURE_NOMINAL_KM)
        counters["coarse_structure_builds"] += 1
        result, authority, canonical_scan, transported_scan = _seed_result(
            seed, layout, structure, build_cfg, sites)
        counters["post_structure_scans"] += 2
        results.append(result)
        scans[seed] = (canonical_scan, transported_scan)
        masks[seed] = {
            "canonical_assembly": layout["assembly"],
            "canonical_broad_assembly": layout["broad_assembly"],
            "canonical_craton": layout["craton"],
            "canonical_resistance": layout["resistance"],
            "canonical_province_raw": layout["province_raw"],
            "canonical_carrier_owner": layout["carrier_owner"],
            "canonical_eligible_nuclei": layout["eligible_nuclei"],
            "canonical_nucleus_mask": layout["nucleus_mask"],
            "canonical_prefix_selected": layout["prefix_selected"],
            "canonical_selected": layout["selected"],
            "canonical_prefix_domain_label": layout[
                "prefix_domain_label"],
            "canonical_domain_label": layout["domain_label"],
            "canonical_prefix_domain_id": layout["prefix_domain_id_grid"],
            "canonical_domain_id": layout["domain_id_grid"],
            "canonical_selected_arrival": np.where(
                layout["selected"], layout["arrival"], np.inf),
            "transported_80km_proxy": authority["proxy"].astype(np.float32),
            "transported_80km_binary": authority["binary"],
            "transported_80km_dominant_tag": authority["dominant_tag"],
        }
        panel_artifacts.append(base._render_seed(
            seed, layout, authority, result, out))
        assignment_artifacts.append(base._render_assignment_windows(
            seed, authority, result, out))
        resistance_artifacts.append(_render_resistance_diagnostic(
            seed, layout, out))

    sentinel_layout = layouts[FINE_SENTINEL_SEED]
    fine_structure, _, _ = _build(
        FINE_SENTINEL_SEED, sentinel_layout,
        FINE_STRUCTURE_NOMINAL_KM)
    counters["fine_sentinel_builds"] += 1
    fine_authority = base._sample_structure_authority(
        fine_structure, sentinel_layout["q"])
    fine_scan = base._scan_windows(fine_authority["proxy"])
    fine_qualification = base._qualify_scan_morphology(
        fine_scan, fine_authority["binary"],
        fine_authority["dominant_tag"])
    fine_scan["selection"] = fine_qualification.pop("selection")
    fine_scan["morphology_qualification"] = fine_qualification
    counters["post_structure_scans"] += 1
    coarse_binary = masks[FINE_SENTINEL_SEED]["transported_80km_binary"]
    coarse_proxy = masks[FINE_SENTINEL_SEED]["transported_80km_proxy"]
    resolution_iou = base._mask_iou(
        coarse_binary, fine_authority["binary"])
    fraction_delta = abs(
        float(coarse_proxy.mean())
        - float(fine_authority["proxy"].mean()))
    fine_sentinel = {
        "seed": FINE_SENTINEL_SEED,
        "coarse_nominal_km": STRUCTURE_NOMINAL_KM,
        "fine_nominal_km": FINE_STRUCTURE_NOMINAL_KM,
        "fine_actual_km": float(
            fine_structure.world_km / fine_structure.n),
        "binary_mask_iou": resolution_iou,
        "global_proxy_fraction_delta": fraction_delta,
        "fine_global_proxy_fraction": float(
            fine_authority["proxy"].mean()),
        "fine_scan": base._scan_report(fine_scan),
        "passed": (
            resolution_iou >= MIN_SENTINEL_RESOLUTION_IOU
            and fraction_delta <= MAX_SENTINEL_FRACTION_DELTA
            and fine_scan["selection"]["found"]),
    }
    masks[FINE_SENTINEL_SEED].update({
        "transported_40km_proxy":
            fine_authority["proxy"].astype(np.float32),
        "transported_40km_binary": fine_authority["binary"],
        "transported_40km_dominant_tag": fine_authority["dominant_tag"],
    })

    nested = _nested_probe(
        FINE_SENTINEL_SEED, int(cfg.plates), sentinel_layout)
    counters["inventory_chronologies"] += 1
    masks[FINE_SENTINEL_SEED]["nested_central_selected"] = nested.pop(
        "central_mask")
    masks[FINE_SENTINEL_SEED]["nested_central_prefix_selected"] = nested.pop(
        "central_prefix_mask")
    masks[FINE_SENTINEL_SEED]["nested_central_domain_id"] = nested.pop(
        "central_domain_id")
    masks[FINE_SENTINEL_SEED][
        "nested_central_prefix_domain_id"] = nested.pop(
            "central_prefix_domain_id")
    masks[FINE_SENTINEL_SEED]["nested_padded_full_selected"] = nested.pop(
        "padded_full_mask")
    masks[FINE_SENTINEL_SEED][
        "nested_padded_full_prefix_selected"] = nested.pop(
            "padded_full_prefix_mask")
    masks[FINE_SENTINEL_SEED]["nested_padded_full_domain_id"] = nested.pop(
        "padded_full_domain_id")
    masks[FINE_SENTINEL_SEED][
        "nested_padded_full_prefix_domain_id"] = nested.pop(
            "padded_full_prefix_domain_id")

    mask_artifacts = []
    scan_artifacts = []
    for seed in SEEDS:
        mask_artifacts.append(base._save_masks(
            seed, out, **masks[seed]))
        scan_artifacts.append(base._save_scan_table(
            seed, out, scans[seed][0], scans[seed][1],
            fine_scan if seed == FINE_SENTINEL_SEED else None))
    panel_montage = base._render_montage(
        panel_artifacts, out, "formation_panels_montage.png")
    assignment_montage = base._render_montage(
        assignment_artifacts, out, "assignment_windows_montage.png")
    resistance_montage = base._render_montage(
        resistance_artifacts, out, "resistance_diagnostics_montage.png")

    exact_count = sum(
        item["ready_gates"]["inventory_exact"] for item in results)
    invariant_count = sum(
        item["ready_gates"]["formation_invariants"] for item in results)
    assignment_count = sum(
        item["ready_gates"]["transported_proxy_assignment"]
        for item in results)
    passed_window_count = sum(
        sum(window["passed"] for window in
            item["assigned_window_reviews"]["windows"].values())
        for item in results)
    ready_count = sum(item["ready"] for item in results)
    aggregate_gates = {
        "exact_inventory_all_seeds": exact_count == len(SEEDS),
        "formation_invariants_all_seeds": invariant_count == len(SEEDS),
        "all_seeds_have_proxy_assignment": assignment_count == len(SEEDS),
        "all_assigned_windows_pass_components": passed_window_count == 24,
        "all_seeds_ready": ready_count == MIN_READY_SEEDS,
        "fine_sentinel": fine_sentinel["passed"],
        "nested_target_and_prefix_inner_stability": nested["passed"],
        "execution_counts": counters == {
            "inventory_chronologies": len(SEEDS) + 1,
            "primary_prefix_snapshots": len(SEEDS),
            "coarse_structure_builds": len(SEEDS),
            "fine_sentinel_builds": 1,
            "elevation_builds": 0,
            "surface_process_solves": 0,
            "post_structure_scans": 2 * len(SEEDS) + 1,
        },
    }
    automatic_pass = all(aggregate_gates.values())
    report = {
        "experiment": EXPERIMENT,
        "evidence_role": RUN_ROLE,
        "completed": True,
        "protocol_precommit_sha256": protocol_sha256,
        "source_fingerprint": protocol["source_fingerprint"],
        "prior_evidence": protocol["prior_evidence"],
        "elapsed_s": float(time.perf_counter() - started),
        "execution_counters": counters,
        "seed_results": results,
        "fine_sentinel": fine_sentinel,
        "nested_formation_probe": nested,
        "aggregate": {
            "seed_count": len(SEEDS),
            "exact_inventory_seed_count": exact_count,
            "formation_invariant_seed_count": invariant_count,
            "proxy_assignment_seed_count": assignment_count,
            "passed_assigned_window_count": passed_window_count,
            "required_assigned_window_count": 24,
            "ready_seed_count": ready_count,
            "required_ready_seed_count": MIN_READY_SEEDS,
            "gates": aggregate_gates,
        },
        "automatic_formation_readiness_pass": automatic_pass,
        "manual_morphology_review": {
            "required_before_full_solve": True,
            "status": "unreviewed",
            "criteria": [
                "few unequal non-stamped domains rather than metric balls",
                ("no domain boundaries revealing common-radius or isochrone "
                 "shells; fixed-interval diagnostic bands are expected"),
                "no broad-zero-contour stamping",
                "no carrier lace or tendrils",
                "no grid/tie directionality",
                "no ubiquitous single amoeba across target windows",
            ],
        },
        "artifacts": {
            "formation_panels": panel_artifacts,
            "assigned_proxy_windows": assignment_artifacts,
            "resistance_diagnostics": resistance_artifacts,
            "formation_montage": panel_montage,
            "assignment_windows_montage": assignment_montage,
            "resistance_diagnostics_montage": resistance_montage,
            "authority_masks": mask_artifacts,
            "complete_scan_tables": scan_artifacts,
        },
        "interpretation_limits": protocol["interpretation"],
        "recommend_full_parent_solve": False,
        "promotion": False,
        "section3b_status": "unresolved_finite_parent_continuous_resistance",
    }
    report_sha256 = base._write_json_exclusive(out / "report.json", report)
    base._write_json_exclusive(out / "report.sha256.json", {
        "file": "report.json",
        "protocol_precommit_sha256": protocol_sha256,
        "sha256": report_sha256,
    })
    print(json.dumps({
        "completed": True,
        "automatic_formation_readiness_pass": automatic_pass,
        "ready_seed_count": ready_count,
        "proxy_assignment_seed_count": assignment_count,
        "passed_assigned_window_count": passed_window_count,
        "exact_inventory_seed_count": exact_count,
        "fine_sentinel_pass": fine_sentinel["passed"],
        "nested_probe_pass": nested["passed"],
        "report_sha256": report_sha256,
    }, indent=2))
    return report


def _self_check() -> dict:
    assembly = np.full((32, 32), -0.20, np.float64)
    assembly[3:14, 3:14] = 0.30
    assembly[18:29, 18:29] = 0.24
    assembly[13:19, 13:19] = -0.45
    ties_q = (np.arange(32) + 0.5) * CANONICAL_KM
    ties = base._coordinate_ties(17, ties_q)
    nuclei = [
        {"domain_id": "left", "canonical_yx": [7, 7]},
        {"domain_id": "right", "canonical_yx": [24, 24]},
    ]
    growth = _continuous_growth(
        assembly, nuclei, CANONICAL_KM,
        prefix_cells=160, target_cells=360, tie_grid=ties)
    if int(growth["prefix_selected"].sum()) != 160:
        raise AssertionError("prefix inventory is not exact")
    if int(growth["selected"].sum()) != 360:
        raise AssertionError("target inventory is not exact")
    if not np.all(~growth["prefix_selected"] | growth["selected"]):
        raise AssertionError("prefix is not nested")
    if not np.array_equal(
            growth["prefix_owner"][growth["prefix_selected"]],
            growth["selected_owner"][growth["prefix_selected"]]):
        raise AssertionError("prefix owner changed")
    if not _strict_predecessor_paths(
            growth["selected_owner"], growth["arrival"], nuclei):
        raise AssertionError("strict predecessor path failed")
    if not np.any(
            growth["selected"]
            & (assembly <= legacy.CARRIER_THRESHOLD)):
        raise AssertionError("continuous support retained a hidden hard wall")

    reversed_growth = _continuous_growth(
        assembly, list(reversed(nuclei)), CANONICAL_KM,
        prefix_cells=160, target_cells=360, tie_grid=ties)
    if not np.array_equal(
            growth["selected"], reversed_growth["selected"]):
        raise AssertionError("nucleus enumeration changed selected authority")

    value = RESISTANCE_REFERENCE_ASSEMBLY
    speed = RESISTANCE_REFERENCE_SPEED * np.exp(
        RESISTANCE_LOG_SENSITIVITY
        * (value - RESISTANCE_REFERENCE_ASSEMBLY))
    slope = speed * RESISTANCE_LOG_SENSITIVITY
    if abs(speed - 0.72) > 1e-12 or abs(slope - 1.8) > 1e-12:
        raise AssertionError("resistance law no longer matches legacy tangent")
    rho = _resistance(np.asarray([-0.5, 0.0, 0.5]))
    if not (np.all(np.isfinite(rho)) and np.all(rho > 0.0)):
        raise AssertionError("resistance is not positive finite")

    q = (np.arange(96) + 0.5) * CANONICAL_KM
    assembly_field, broad, craton = _absolute_fields(17, q)
    discovery = _discover_nuclei(17, q, broad, craton)
    if not discovery["records"]:
        raise AssertionError("broad field found no nuclei")
    padded_q = (np.arange(112) - 8 + 0.5) * CANONICAL_KM
    padded_fields = _absolute_fields(17, padded_q)
    if not all(np.array_equal(
            field, padded[8:104, 8:104])
            for field, padded in zip(
                (assembly_field, broad, craton), padded_fields)):
        raise AssertionError("absolute fields changed under nested extent")
    if not np.array_equal(
            base._coordinate_ties(17, q),
            base._coordinate_ties(17, padded_q)[8:104, 8:104]):
        raise AssertionError("absolute ties changed under nested extent")

    dependency = base._self_check()
    if not dependency["passed"]:
        raise AssertionError("base diagnostic dependency failed")
    protocol = _protocol()
    if protocol["seed_policy"]["seeds"] != list(SEEDS):
        raise AssertionError("validation seed block changed")
    return {
        "passed": True,
        "exact_prefix_cells": 160,
        "exact_target_cells": 360,
        "strict_prefix": True,
        "strict_predecessor_paths": True,
        "full_continuous_support": True,
        "enumeration_invariant_selected_mask": True,
        "legacy_tangent_match": True,
        "positive_finite_resistance": True,
        "broad_nucleus_discovery": len(discovery["records"]),
        "absolute_fields_and_ties": True,
        "base_dependency_self_check": True,
        "frozen_validation_seeds": list(SEEDS),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--exposed-development", action="store_true")
    parser.add_argument("--phase", choices=("precommit", "execute"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-precommit-sha256")
    args = parser.parse_args()
    if args.self_check:
        if (args.phase is not None or args.out is not None
                or args.expected_precommit_sha256 is not None
                or args.exposed_development):
            parser.error("--self-check is exclusive")
        print(json.dumps(_self_check(), indent=2))
        return
    if args.exposed_development:
        _configure_exposed_development()
    if args.phase is None or args.out is None:
        parser.error("--phase and --out are required")
    if args.phase == "precommit":
        if args.expected_precommit_sha256 is not None:
            parser.error("expected SHA is execute-only")
        _phase_precommit(args.out)
    else:
        if args.expected_precommit_sha256 is None:
            parser.error("execute requires --expected-precommit-sha256")
        _phase_execute(args.out, args.expected_precommit_sha256)


if __name__ == "__main__":
    main()
