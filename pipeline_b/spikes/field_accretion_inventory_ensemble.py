"""Sealed formation-only ensemble for conserved continental inventory.

This private experiment replaces the field-accretion oracle's time-cutoff
proxy with one exact, world-relative crust-production quota.  The existing
assembly and craton fields still determine where growth can occur.  Each
connected active carrier receives one strongest-craton nucleus, and all
fronts share one chronological event queue until the global inventory is
exhausted.

Formation receives no crop, border, elevation, sea-level, or target-window
information.  Composition windows are inspected only after each complete
structural build.  This file is a spike and is never imported by the public
registry or adapter.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import heapq
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from engine import noise
from engine.rng import fnv1a64, stage_rng, stage_salt
from engine.surface import _bicubic
from engine.tectonics import FRAME_KM, build_structure
from spikes import field_accretion_oracle as legacy_formation


EXPERIMENT = "field-accretion-inventory-ensemble-seed151-158-v1"
SEEDS = tuple(range(151, 159))
EXPOSED_DEVELOPMENT_SEEDS = tuple(range(139, 151))
PARENT_KM = 6.0 * FRAME_KM
CANONICAL_KM = legacy_formation.CANONICAL_KM
TARGET_INITIAL_CONTINENTAL_FRACTION = 0.28
PREFIX_INITIAL_CONTINENTAL_FRACTION = 0.14
STRUCTURE_NOMINAL_KM = 80.0
FINE_SENTINEL_SEED = SEEDS[0]
FINE_STRUCTURE_NOMINAL_KM = 40.0
NESTED_PADDING_KM = FRAME_KM
NUMERICAL_GUARD_KM = 2560.0
CANDIDATE_STRIDE_KM = 256.0
TARGET_BANDS = {
    "low": (0.15, 0.25, True),
    "medium": (0.30, 0.40, True),
    "high": (0.45, 0.50, False),
}

# Readiness gates are formation-only necessities, never promotion criteria.
MIN_READY_SEEDS = len(SEEDS)
MIN_ORIGIN_CHEBYSHEV_SEPARATION_KM = 0.5 * FRAME_KM
SIGNIFICANT_COMPONENT_FRAME_FRACTION = 0.04
MIN_SIGNIFICANT_COMPONENTS = 2
MAX_SIGNIFICANT_COMPONENTS = 4
MIN_SIGNIFICANT_COMPONENT_COVERAGE = 0.85
DIAGNOSTIC_SUBSTANTIAL_DOMAIN_WORLD_FRACTION = 0.01
MIN_SENTINEL_RESOLUTION_IOU = 0.85
MAX_SENTINEL_FRACTION_DELTA = 0.03
MIN_NESTED_INNER_IOU = 0.80

SOURCE_FILES = (
    "engine/__init__.py",
    "engine/elevation.py",  # imported transitively by atlas_survey
    "engine/noise.py",
    "engine/rng.py",
    "engine/surface.py",
    "engine/tectonics.py",
    "spikes/atlas_survey.py",
    "spikes/field_accretion_oracle.py",
    "spikes/field_accretion_inventory_ensemble.py",
    "spikes/visible_contour_gate.py",  # atlas_survey import closure
)

PRIOR_EVIDENCE = {
    "seed138_harness": (
        "spikes/field_accretion_parent_feasibility.py",
        "b943ba07371017e795211548675e250c22c6da18947e42ca16e29dcc92270814",
    ),
    "seed138_report": (
        "../out/field_accretion_parent_seed138_v1/report.json",
        "fa6ea4d3a8eefb631fa8356177b05e2789b5edc95a837fd667915579b066dacc",
    ),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _absolute_fields(seed: int, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Existing stationary formation fields at absolute world coordinates."""
    X, Y = np.meshgrid(q, q)
    assembly = noise.fbm(
        X, Y,
        legacy_formation.ASSEMBLY_WAVELENGTH_KM,
        legacy_formation.ASSEMBLY_OCTAVES,
        stage_salt(seed, "atlas-field-accretion-assembly-v1"),
    )
    craton = noise.fbm(
        X, Y,
        legacy_formation.CRATON_WAVELENGTH_KM,
        legacy_formation.CRATON_OCTAVES,
        stage_salt(seed, "atlas-field-accretion-craton-v1"),
    )
    return assembly, craton


def _coordinate_ties(seed: int, q: np.ndarray) -> np.ndarray:
    """Absolute-coordinate SplitMix64 ties, stable under nested extents."""
    cell = np.rint(q / CANONICAL_KM - 0.5).astype(np.int64)
    uy = cell[:, None].astype(np.uint64)
    ux = cell[None, :].astype(np.uint64)
    with np.errstate(over="ignore"):
        value = (
            uy * np.uint64(0x9E3779B185EBCA87)
            ^ ux * np.uint64(0xC2B2AE3D27D4EB4F)
            ^ np.uint64(fnv1a64(
                f"field-accretion-inventory-tie-v1:{seed}"))
        )
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return value


def _strongest_cell(craton: np.ndarray, ys: np.ndarray, xs: np.ndarray,
                    ties: np.ndarray) -> tuple[int, int]:
    values = craton[ys, xs]
    best = values.max()
    candidates = np.flatnonzero(values == best)
    if candidates.size == 1:
        chosen = int(candidates[0])
    else:
        local_ties = ties[ys[candidates], xs[candidates]]
        chosen = int(candidates[int(np.argmin(local_ties))])
    return int(ys[chosen]), int(xs[chosen])


def _global_inventory_growth(
        assembly: np.ndarray,
        carrier_raw: np.ndarray,
        seeds: list[dict],
        canonical_km: float,
        target_cells: int,
        tie_grid: np.ndarray,
        *,
        strict_capacity: bool = False) -> dict:
    """Take the first K events from one assembly-weighted growth chronology."""
    active_labels = {item["carrier_raw_label"] for item in seeds}
    reachable = np.isin(carrier_raw, list(active_labels))
    capacity_cells = int(reachable.sum())
    if target_cells > capacity_cells and strict_capacity:
        raise ValueError(
            f"inventory {target_cells} exceeds carrier capacity {capacity_cells}")
    requested = int(target_cells)
    take_cells = min(requested, capacity_cells)

    shape = assembly.shape
    arrival = np.full(shape, np.inf, np.float64)
    provisional_owner = np.full(shape, -1, np.int32)
    settled = np.zeros(shape, bool)
    selected_owner = np.full(shape, -1, np.int32)
    queue = []
    for domain_index, seed_record in enumerate(seeds):
        y, x = seed_record["canonical_yx"]
        arrival[y, x] = 0.0
        provisional_owner[y, x] = domain_index
        heapq.heappush(queue, (
            0.0, int(tie_grid[y, x]), domain_index, y, x))

    neighbors = (
        (-1, -1, np.sqrt(2.0)), (-1, 0, 1.0),
        (-1, 1, np.sqrt(2.0)), (0, -1, 1.0),
        (0, 1, 1.0), (1, -1, np.sqrt(2.0)),
        (1, 0, 1.0), (1, 1, np.sqrt(2.0)),
    )
    settled_count = 0
    last_arrival = 0.0
    last_tie = 0
    while queue and settled_count < take_cells:
        elapsed, tie, domain_index, y, x = heapq.heappop(queue)
        if settled[y, x]:
            continue
        if (elapsed != arrival[y, x]
                or domain_index != provisional_owner[y, x]):
            continue
        settled[y, x] = True
        selected_owner[y, x] = domain_index
        settled_count += 1
        last_arrival = float(elapsed)
        last_tie = int(tie)
        carrier = seeds[domain_index]["carrier_raw_label"]
        for dy, dx, diagonal in neighbors:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < shape[0] and 0 <= xx < shape[1]):
                continue
            if settled[yy, xx] or carrier_raw[yy, xx] != carrier:
                continue
            potential = 0.5 * (assembly[y, x] + assembly[yy, xx])
            speed = np.clip(
                0.72 + 1.8 * (
                    potential - legacy_formation.CARRIER_THRESHOLD),
                0.45, 1.35,
            )
            candidate = elapsed + canonical_km * diagonal / speed
            old = arrival[yy, xx]
            old_owner = provisional_owner[yy, xx]
            candidate_key = (candidate, int(tie_grid[yy, xx]), domain_index)
            old_key = (old, int(tie_grid[yy, xx]), int(old_owner))
            if candidate_key < old_key:
                arrival[yy, xx] = candidate
                provisional_owner[yy, xx] = domain_index
                heapq.heappush(queue, (
                    float(candidate), int(tie_grid[yy, xx]),
                    domain_index, yy, xx))

    if settled_count != take_cells:
        raise AssertionError((settled_count, take_cells, capacity_cells))
    selected_cutoff_cohort = int(np.count_nonzero(
        settled & (arrival == last_arrival)))
    discovered_unselected_cutoff_cohort = int(np.count_nonzero(
        ~settled & np.isfinite(arrival) & (arrival == last_arrival)))
    return {
        "selected_owner": selected_owner,
        "selected": selected_owner >= 0,
        "arrival": arrival,
        "reachable": reachable,
        "requested_cells": requested,
        "capacity_cells": capacity_cells,
        "selected_cells": settled_count,
        "capacity_passed": requested <= capacity_cells,
        "cutoff_arrival": last_arrival,
        "cutoff_tie": last_tie,
        "selected_cutoff_cohort_cells": selected_cutoff_cohort,
        "discovered_unselected_cutoff_cohort_cells":
            discovered_unselected_cutoff_cohort,
    }


def _connected_to_seed(mask: np.ndarray, seed_yx: tuple[int, int]) -> bool:
    expected = int(mask.sum())
    if expected == 0 or not mask[seed_yx]:
        return False
    seen = np.zeros(mask.shape, bool)
    queue = deque([seed_yx])
    seen[seed_yx] = True
    count = 0
    while queue:
        y, x = queue.popleft()
        count += 1
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if (0 <= yy < mask.shape[0]
                        and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    queue.append((yy, xx))
    return count == expected


def _strict_predecessor_paths(
        selected_owner: np.ndarray,
        arrival: np.ndarray,
        seed_records: list[dict]) -> bool:
    """Every non-seed cell has an earlier selected neighbor of one owner."""
    seeds = {index: tuple(item["canonical_yx"])
             for index, item in enumerate(seed_records)}
    for y, x in np.argwhere(selected_owner >= 0):
        owner = int(selected_owner[y, x])
        if (int(y), int(x)) == seeds[owner]:
            if arrival[y, x] != 0.0:
                return False
            continue
        earlier = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = int(y + dy), int(x + dx)
                if (0 <= yy < selected_owner.shape[0]
                        and 0 <= xx < selected_owner.shape[1]
                        and selected_owner[yy, xx] == owner
                        and arrival[yy, xx] < arrival[y, x]):
                    earlier = True
                    break
            if earlier:
                break
        if not earlier:
            return False
    return True


def _layout_from_fields(
        seed: int,
        q: np.ndarray,
        assembly: np.ndarray,
        craton: np.ndarray,
        plate_count: int,
        target_fraction: float,
        *,
        reverse_components_for_test: bool = False,
        strict_capacity: bool = False) -> dict:
    if assembly.shape != craton.shape or assembly.shape != (q.size, q.size):
        raise ValueError("formation fields have inconsistent shapes")
    canonical_km = float(q[1] - q[0])
    tie_grid = _coordinate_ties(seed, q)
    carrier_raw, carrier_components = legacy_formation._label_components(
        assembly > legacy_formation.CARRIER_THRESHOLD)
    raw_nuclei = (
        (carrier_raw >= 0)
        & (craton > legacy_formation.NUCLEUS_CRATON_THRESHOLD)
    )
    active_labels = sorted(set(int(item) for item in carrier_raw[raw_nuclei]))
    if reverse_components_for_test:
        active_labels = list(reversed(active_labels))

    seed_records = []
    for raw_label in active_labels:
        ys, xs = np.nonzero(raw_nuclei & (carrier_raw == raw_label))
        py, px = _strongest_cell(craton, ys, xs, tie_grid)
        seed_records.append({
            "domain_id": legacy_formation._field_id(
                seed, "inventory-domain", q[py], q[px]),
            "canonical_yx": [py, px],
            "pivot_yx_km": [float(q[py]), float(q[px])],
            "carrier_raw_label": raw_label,
            "nucleus_cells": int(ys.size),
        })
    seed_records.sort(key=lambda item: item["domain_id"])
    if not seed_records:
        raise ValueError("formation produced no active carriers")
    if len(seed_records) > plate_count:
        raise ValueError("active carriers exceed configured plates")

    target_cells = int(round(float(target_fraction) * assembly.size))
    growth = _global_inventory_growth(
        assembly, carrier_raw, seed_records, canonical_km, target_cells,
        tie_grid, strict_capacity=strict_capacity)
    selected_owner = growth["selected_owner"]

    carrier_records = []
    for raw_label in sorted(active_labels):
        ys, xs = carrier_components[raw_label]
        py, px = _strongest_cell(assembly, ys, xs, tie_grid)
        carrier_records.append({
            "raw_label": raw_label,
            "carrier_id": legacy_formation._field_id(
                seed, "inventory-carrier", q[py], q[px]),
            "pivot_yx_km": [float(q[py]), float(q[px])],
            "canonical_cells": int(ys.size),
            "touches_world_rim": bool(
                np.any(ys == 0) or np.any(xs == 0)
                or np.any(ys == q.size - 1)
                or np.any(xs == q.size - 1)),
        })
    carrier_records.sort(key=lambda item: item["carrier_id"])
    carrier_plate_by_raw = {}
    for plate_id, record in enumerate(carrier_records):
        record["plate_id"] = plate_id
        carrier_plate_by_raw[record["raw_label"]] = plate_id

    carrier_owner = np.full(carrier_raw.shape, -1, np.int32)
    for raw_label, plate_id in carrier_plate_by_raw.items():
        carrier_owner[carrier_raw == raw_label] = plate_id

    domain_plate_by_label = np.full(len(seed_records), -1, np.int32)
    domain_records = []
    for label, seed_record in enumerate(seed_records):
        raw_label = seed_record["carrier_raw_label"]
        plate_id = int(carrier_plate_by_raw[raw_label])
        domain_plate_by_label[label] = plate_id
        ys, xs = np.nonzero(selected_owner == label)
        connected = _connected_to_seed(
            selected_owner == label,
            tuple(seed_record["canonical_yx"]),
        )
        domain_records.append({
            "label": label,
            "domain_id": seed_record["domain_id"],
            "carrier_plate_id": plate_id,
            "carrier_raw_label": raw_label,
            "pivot_yx_km": seed_record["pivot_yx_km"],
            "canonical_cells": int(ys.size),
            "area_km2": float(ys.size * canonical_km ** 2),
            "nucleus_cells": seed_record["nucleus_cells"],
            "connected_to_seed": connected,
            "touches_world_rim": next(
                item["touches_world_rim"] for item in carrier_records
                if item["raw_label"] == raw_label),
            "centroid_yx_km": (
                [float(np.mean(q[ys])), float(np.mean(q[xs]))]
                if ys.size else None),
            "bbox_xyxy_km": (
                [float(q[xs.min()] - 0.5 * canonical_km),
                 float(q[ys.min()] - 0.5 * canonical_km),
                 float(q[xs.max()] + 0.5 * canonical_km),
                 float(q[ys.max()] + 0.5 * canonical_km)]
                if ys.size else None),
        })

    return {
        "canonical_km": canonical_km,
        "q": q,
        "assembly": assembly,
        "craton": craton,
        "raw_nuclei": raw_nuclei,
        "carrier_raw": carrier_raw,
        "carrier_owner": carrier_owner,
        "selected": growth["selected"],
        "domain_label": selected_owner,
        "domain_plate_by_label": domain_plate_by_label,
        "carriers": carrier_records,
        "domains": domain_records,
        "requested_cells": growth["requested_cells"],
        "capacity_cells": growth["capacity_cells"],
        "selected_cells": growth["selected_cells"],
        "capacity_passed": growth["capacity_passed"],
        "cutoff_arrival": growth["cutoff_arrival"],
        "cutoff_tie": growth["cutoff_tie"],
        "selected_cutoff_cohort_cells":
            growth["selected_cutoff_cohort_cells"],
        "discovered_unselected_cutoff_cohort_cells":
            growth["discovered_unselected_cutoff_cohort_cells"],
        "arrival": growth["arrival"],
        "target_fraction": float(target_fraction),
        "selected_fraction": float(growth["selected_cells"] / assembly.size),
        "strict_predecessor_paths": _strict_predecessor_paths(
            selected_owner, growth["arrival"], seed_records),
    }


def _inventory_layout(seed: int, world_km: float, plate_count: int,
                      *, origin_km: float = 0.0,
                      target_fraction: float =
                      TARGET_INITIAL_CONTINENTAL_FRACTION,
                      strict_capacity: bool = False) -> dict:
    count = int(round(world_km / CANONICAL_KM))
    canonical_km = world_km / count
    if abs(canonical_km - CANONICAL_KM) > 1e-12:
        raise ValueError("world extent must preserve the canonical lattice")
    q = origin_km + (np.arange(count) + 0.5) * canonical_km
    assembly, craton = _absolute_fields(seed, q)
    return _layout_from_fields(
        seed, q, assembly, craton, plate_count,
        target_fraction,
        strict_capacity=strict_capacity,
    )


def _plate_sites(seed: int, layout: dict, world_km: float,
                 plate_count: int) -> np.ndarray:
    parent_sites = np.asarray(
        [item["pivot_yx_km"] for item in layout["carriers"]],
        np.float64,
    )
    remaining = plate_count - parent_sites.shape[0]
    rng = stage_rng(seed, "atlas-field-accretion-ocean-sites-v1")
    proposal_count = max(8192, 192 * plate_count)
    proposals = rng.uniform(
        0.02 * world_km, 0.98 * world_km, (proposal_count, 2))
    owner = legacy_formation._nearest_canonical(
        layout["carrier_owner"], proposals[:, 0], proposals[:, 1],
        layout["canonical_km"], fill=-1)
    candidates = proposals[owner < 0]
    if candidates.shape[0] < remaining:
        raise ValueError("not enough natural-ocean plate-site proposals")

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


def _make_partitioner(layout: dict, sites: np.ndarray, expected_seed: int,
                      expected_world_km: float):
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
                Xw - (site_x + site_dx[plate]),
            )
            take = cost < best
            best[take] = cost[take]
            label[take] = plate
        owner = legacy_formation._nearest_canonical(
            layout["carrier_owner"], Y, X,
            layout["canonical_km"], fill=-1)
        carrier = owner >= 0
        label[carrier] = owner[carrier]
        return label
    return partition


def _make_samplers(layout: dict):
    def continent(plate_id, material_y_km, material_x_km):
        label = legacy_formation._nearest_canonical(
            layout["domain_label"], material_y_km, material_x_km,
            layout["canonical_km"], fill=-1)
        valid = label >= 0
        owner = np.full(label.shape, -1, np.int32)
        owner[valid] = layout["domain_plate_by_label"][label[valid]]
        return valid & (owner == plate_id)

    def material_tag(plate_id, material_y_km, material_x_km):
        label = legacy_formation._nearest_canonical(
            layout["domain_label"], material_y_km, material_x_km,
            layout["canonical_km"], fill=-1)
        valid = label >= 0
        owner = np.full(label.shape, -1, np.int32)
        owner[valid] = layout["domain_plate_by_label"][label[valid]]
        return np.where(valid & (owner == plate_id), label, -1).astype(
            np.int32, copy=False)

    return continent, material_tag


def _prefix_checks(seed: int, main: dict, prefix: dict,
                   plate_count: int) -> dict:
    main_sites = _plate_sites(seed, main, PARENT_KM, plate_count)
    prefix_sites = _plate_sites(seed, prefix, PARENT_KM, plate_count)
    def chosen_nuclei(layout: dict) -> list[tuple]:
        return sorted(
            (
                str(item["domain_id"]),
                tuple(float(value) for value in item["pivot_yx_km"]),
                int(item["carrier_raw_label"]),
                int(item["nucleus_cells"]),
            )
            for item in layout["domains"]
        )

    strict_subset = (
        np.all(~prefix["selected"] | main["selected"])
        and np.any(main["selected"] & ~prefix["selected"]))
    checks = {
        "prefix_capacity_passed": prefix["capacity_passed"],
        "strict_mask_prefix": bool(strict_subset),
        "assembly_identical": bool(np.array_equal(
            main["assembly"], prefix["assembly"])),
        "craton_identical": bool(np.array_equal(
            main["craton"], prefix["craton"])),
        "nuclei_identical": bool(np.array_equal(
            main["raw_nuclei"], prefix["raw_nuclei"])),
        "chosen_nucleus_ids_pivots_carriers_identical": (
            chosen_nuclei(main) == chosen_nuclei(prefix)),
        "carrier_owner_identical": bool(np.array_equal(
            main["carrier_owner"], prefix["carrier_owner"])),
        "carrier_records_identical": main["carriers"] == prefix["carriers"],
        "plate_sites_identical": bool(np.array_equal(
            main_sites, prefix_sites)),
    }
    checks["passed"] = all(checks.values())
    return checks


def _build(seed: int, layout: dict, nominal_km: float):
    cfg = legacy_formation._atlas_config(
        TARGET_INITIAL_CONTINENTAL_FRACTION)
    sites = _plate_sites(seed, layout, PARENT_KM, int(cfg.plates))
    continent, material_tag = _make_samplers(layout)
    structure = build_structure(
        seed, cfg,
        _world_km=PARENT_KM,
        _coarse_km=nominal_km,
        _continent_seeder=legacy_formation._continent_seeder(continent),
        _partitioner=_make_partitioner(layout, sites, seed, PARENT_KM),
        _initial_age_sampler=legacy_formation._initial_ocean_age,
        _plate_pivots=sites,
        _continent_sampler=continent,
        _material_tag_sampler=material_tag,
    )
    return structure, cfg, sites


def _sample_structure_authority(structure, q: np.ndarray) -> dict:
    X, Y = np.meshgrid(q, q)
    proxy = np.clip(_bicubic(
        np.asarray(structure.cont_frac, np.float64),
        Y, X, structure.world_km / structure.n), 0.0, 1.0)
    raw_tags = np.asarray(structure._material_tag_samples)
    sampled_tags = np.stack([
        legacy_formation._nearest_canonical(
            raw_tags[index], Y, X, structure.world_km / structure.n,
            fill=-1)
        for index in range(raw_tags.shape[0])
    ])
    dominant = np.full(proxy.shape, -1, np.int32)
    best_count = np.zeros(proxy.shape, np.int8)
    for label in np.unique(sampled_tags[sampled_tags >= 0]):
        count = np.count_nonzero(sampled_tags == label, axis=0)
        take = ((count > best_count)
                | ((count == best_count) & (count > 0)
                   & ((dominant < 0) | (label < dominant))))
        dominant[take] = int(label)
        best_count[take] = count[take]
    return {
        "proxy": proxy,
        "binary": proxy >= 0.5,
        "dominant_tag": dominant,
    }


def _candidate_axis_cells(shape: int) -> np.ndarray:
    frame_cells = int(round(FRAME_KM / CANONICAL_KM))
    guard_cells = int(round(NUMERICAL_GUARD_KM / CANONICAL_KM))
    stride_cells = int(round(CANDIDATE_STRIDE_KM / CANONICAL_KM))
    last = shape - guard_cells - frame_cells
    axis = np.arange(guard_cells, last + 1, stride_cells, dtype=np.int32)
    if (axis.size != 61 or axis[0] != guard_cells
            or axis[-1] != last):
        raise AssertionError((shape, axis.size, axis[0], axis[-1], last))
    return axis


def _in_band(value: float, band: tuple[float, float, bool]) -> bool:
    low, high, upper_inclusive = band
    return value >= low and (value <= high if upper_inclusive else value < high)


def _separated(left: dict, right: dict) -> bool:
    return max(
        abs(left["x0_km"] - right["x0_km"]),
        abs(left["y0_km"] - right["y0_km"]),
    ) >= MIN_ORIGIN_CHEBYSHEV_SEPARATION_KM


def _select_assignment(records: list[dict]) -> dict:
    targets = {"low": 0.20, "medium": 0.35, "high": 0.50}
    if not records:
        return {
            "pool_counts": {label: 0 for label in TARGET_BANDS},
            "viable_assignment_count": 0,
            "found": False,
            "assignment": {},
            "errors": {},
            "objective": None,
            "diagnostics": {},
        }
    pools = {}
    for label, band in TARGET_BANDS.items():
        pools[label] = sorted(
            [item for item in records
             if _in_band(item["continental_fraction"], band)],
            key=lambda item: (
                abs(item["continental_fraction"] - targets[label]),
                item["tie_key"],
            ),
        )
    lows = pools["low"]
    mediums = pools["medium"]
    highs = pools["high"]

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
            common = high_low_bits[high_index] & medium_low_bits[medium_index]
            if not common:
                continue
            viable_count += common.bit_count()
            bit = common & -common
            low_index = bit.bit_length() - 1
            low = lows[low_index]
            assignment = {"low": low, "medium": medium, "high": high}
            errors = {
                label: abs(
                    assignment[label]["continental_fraction"]
                    - targets[label])
                for label in targets
            }
            key = (
                errors["high"], errors["medium"], errors["low"],
                sum(errors.values()),
                high["tie_key"], medium["tie_key"], low["tie_key"],
            )
            if best is None or key < best[0]:
                best = (key, assignment, errors)
    if best is None:
        key, assignment, errors = None, {}, {}
    else:
        key, assignment, errors = best
    diagnostics = {
        label: min(records, key=lambda item: (
            abs(item["continental_fraction"] - target),
            item["tie_key"],
        ))
        for label, target in targets.items()
    }
    return {
        "pool_counts": {label: len(items)
                        for label, items in pools.items()},
        "viable_assignment_count": viable_count,
        "found": best is not None,
        "assignment": assignment,
        "errors": errors,
        "objective": None if key is None else {
            "high_error": float(key[0]),
            "medium_error": float(key[1]),
            "low_error": float(key[2]),
            "sum_error": float(key[3]),
            "tie_keys_high_medium_low": [
                int(key[4]), int(key[5]), int(key[6])],
        },
        "diagnostics": diagnostics,
    }


def _scan_windows(field: np.ndarray) -> dict:
    field = np.asarray(field, np.float64)
    axis = _candidate_axis_cells(field.shape[0])
    frame_cells = int(round(FRAME_KM / CANONICAL_KM))
    integral = np.pad(
        np.cumsum(np.cumsum(field, axis=0), axis=1),
        ((1, 0), (1, 0)), constant_values=0)
    records = []
    for y0 in axis:
        y1 = int(y0 + frame_cells)
        for x0 in axis:
            x1 = int(x0 + frame_cells)
            total = float(
                integral[y1, x1] - integral[y0, x1]
                - integral[y1, x0] + integral[y0, x0])
            fraction = total / float(frame_cells ** 2)
            record = {
                "x0_km": float(x0 * CANONICAL_KM),
                "y0_km": float(y0 * CANONICAL_KM),
                "continental_sum": total,
                "continental_fraction": fraction,
                "tie_key": int(fnv1a64(
                    f"inventory-window:{x0 * CANONICAL_KM:.0f}:"
                    f"{y0 * CANONICAL_KM:.0f}")),
            }
            records.append(record)
    selection = _select_assignment(records)
    fractions = np.asarray(
        [item["continental_fraction"] for item in records])
    return {
        "candidate_count": len(records),
        "minimum_continental_fraction": float(fractions.min()),
        "median_continental_fraction": float(np.median(fractions)),
        "maximum_continental_fraction": float(fractions.max()),
        "composition_only_selection": selection,
        "records": records,
    }


def _scan_report(scan: dict) -> dict:
    return {key: value for key, value in scan.items() if key != "records"}


def _component_records(mask: np.ndarray) -> list[dict]:
    labels, components = legacy_formation._label_components(
        np.asarray(mask, bool), diagonal=True)
    records = []
    for label, (ys, xs) in enumerate(components):
        records.append({
            "label": label,
            "cells": int(ys.size),
            "fraction_of_frame": float(ys.size / mask.size),
        })
    records.sort(key=lambda item: (-item["cells"], item["label"]))
    return records


def _single_window_review(binary: np.ndarray, dominant_tag: np.ndarray,
                          candidate: dict) -> dict:
    frame_cells = int(round(FRAME_KM / CANONICAL_KM))
    threshold_cells = int(np.ceil(
        SIGNIFICANT_COMPONENT_FRAME_FRACTION * frame_cells ** 2))
    x0 = int(round(candidate["x0_km"] / CANONICAL_KM))
    y0 = int(round(candidate["y0_km"] / CANONICAL_KM))
    sub = binary[y0:y0 + frame_cells, x0:x0 + frame_cells]
    components = _component_records(sub)
    significant = [item for item in components
                   if item["cells"] >= threshold_cells]
    land_cells = int(sub.sum())
    coverage = sum(item["cells"] for item in significant) / max(land_cells, 1)
    tag_sub = dominant_tag[y0:y0 + frame_cells, x0:x0 + frame_cells]
    tags, counts = np.unique(
        tag_sub[sub & (tag_sub >= 0)], return_counts=True)
    identity_records = sorted(
        [{"tag": int(tag), "cells": int(count),
          "fraction_of_frame": float(count / sub.size)}
         for tag, count in zip(tags, counts)],
        key=lambda item: (-item["cells"], item["tag"]),
    )
    significant_identities = [
        item for item in identity_records if item["cells"] >= threshold_cells]
    passed = (
        MIN_SIGNIFICANT_COMPONENTS <= len(significant)
        <= MAX_SIGNIFICANT_COMPONENTS
        and coverage >= MIN_SIGNIFICANT_COMPONENT_COVERAGE)
    return {
        "candidate": candidate,
        "land_cells_binary_proxy": land_cells,
        "component_count": len(components),
        "components": components,
        "significant_threshold_cells": threshold_cells,
        "significant_component_count": len(significant),
        "significant_component_coverage": float(coverage),
        "identity_count": len(identity_records),
        "identities": identity_records,
        "significant_identity_count": len(significant_identities),
        "passed": passed,
    }


def _qualify_scan_morphology(scan: dict, binary: np.ndarray,
                             dominant_tag: np.ndarray) -> dict:
    qualified = []
    evaluated = 0
    passed = 0
    for candidate in scan["records"]:
        in_any_band = any(
            _in_band(candidate["continental_fraction"], band)
            for band in TARGET_BANDS.values())
        if not in_any_band:
            candidate["component_gate_status"] = -1
            continue
        review = _single_window_review(binary, dominant_tag, candidate)
        evaluated += 1
        candidate["component_gate_status"] = int(review["passed"])
        if review["passed"]:
            qualified.append(candidate)
            passed += 1
    selection = _select_assignment(qualified)
    return {
        "evaluated_candidate_count": evaluated,
        "passed_candidate_count": passed,
        "selection": selection,
    }


def _assigned_window_reviews(binary: np.ndarray, dominant_tag: np.ndarray,
                             selection: dict) -> dict:
    if not selection["found"]:
        return {"passed": False, "reason": "no_assignment", "windows": {}}
    windows = {}
    for label, candidate in selection["assignment"].items():
        windows[label] = _single_window_review(
            binary, dominant_tag, candidate)
    return {
        "passed": all(item["passed"] for item in windows.values()),
        "reason": None,
        "windows": windows,
    }


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.asarray(left, bool) | np.asarray(right, bool)
    if not union.any():
        return 1.0
    return float((np.asarray(left, bool) & np.asarray(right, bool)).sum()
                 / union.sum())


def _domain_summary(layout: dict) -> dict:
    cells = np.asarray(
        [item["canonical_cells"] for item in layout["domains"]],
        np.int64)
    total = int(cells.sum())
    shares = cells / max(total, 1)
    substantial_cells = int(round(
        DIAGNOSTIC_SUBSTANTIAL_DOMAIN_WORLD_FRACTION
        * layout["selected"].size))
    return {
        "active_domain_count": len(layout["domains"]),
        "selected_cells_by_domain": cells.tolist(),
        "minimum_domain_cells": int(cells.min()),
        "median_domain_cells": float(np.median(cells)),
        "maximum_domain_cells": int(cells.max()),
        "substantial_threshold_cells": substantial_cells,
        "substantial_domain_count": int((cells >= substantial_cells).sum()),
        "largest_inventory_share": float(shares.max()),
        "effective_domain_count": float(1.0 / np.sum(shares ** 2)),
        "all_connected_to_seed": all(
            item["connected_to_seed"] for item in layout["domains"]),
        "starved_domain_count": int((cells <= 1).sum()),
        "rim_touching_domain_count": sum(
            item["touches_world_rim"] for item in layout["domains"]),
    }


def _represented_domains(structure) -> dict:
    tags = np.asarray(structure._material_tag_samples)
    valid = tags[tags >= 0]
    unique = np.unique(valid)
    return {
        "material_tag_shape": list(tags.shape),
        "material_tag_sha256": _sha256_bytes(
            np.ascontiguousarray(tags).tobytes()),
        "tagged_samples": int(valid.size),
        "represented_domain_count": int(unique.size),
        "represented_domain_labels": unique.tolist(),
    }


def _palette(label: int) -> np.ndarray:
    colors = np.asarray([
        (171, 196, 118), (211, 173, 101), (133, 181, 153),
        (184, 143, 128), (132, 169, 204), (195, 181, 121),
        (154, 146, 190), (196, 151, 102), (121, 187, 183),
    ], np.uint8)
    return colors[label % len(colors)]


def _formation_rgb(layout: dict) -> np.ndarray:
    rgb = np.zeros((*layout["selected"].shape, 3), np.uint8)
    rgb[:] = np.asarray((12, 31, 55), np.uint8)
    active = layout["carrier_owner"] >= 0
    rgb[active] = np.asarray((31, 70, 78), np.uint8)
    for item in layout["domains"]:
        rgb[layout["domain_label"] == item["label"]] = _palette(item["label"])
        y, x = (int(value) for value in next(
            record["canonical_yx"]
            for record in _seed_records_from_layout(layout)
            if record["domain_id"] == item["domain_id"]))
        rgb[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = (250, 240, 180)
    return rgb


def _seed_records_from_layout(layout: dict) -> list[dict]:
    records = []
    for item in layout["domains"]:
        pivot_y, pivot_x = item["pivot_yx_km"]
        y = int(np.floor((pivot_y - layout["q"][0]
                          + 0.5 * layout["canonical_km"])
                         / layout["canonical_km"]))
        x = int(np.floor((pivot_x - layout["q"][0]
                          + 0.5 * layout["canonical_km"])
                         / layout["canonical_km"]))
        records.append({"domain_id": item["domain_id"],
                        "canonical_yx": [y, x]})
    return records


def _transport_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    rgb[:] = np.asarray((16, 42, 73), np.uint8)
    rgb[mask] = np.asarray((165, 190, 119), np.uint8)
    return rgb


def _render_seed(seed: int, layout: dict, authority: dict,
                 result: dict, out: Path) -> dict:
    left = Image.fromarray(_formation_rgb(layout), "RGB").resize(
        (768, 768), Image.Resampling.NEAREST)
    right = Image.fromarray(_transport_rgb(authority["binary"]), "RGB").resize(
        (768, 768), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (1536, 816), (8, 18, 30))
    canvas.paste(left, (0, 48))
    canvas.paste(right, (768, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12),
              f"seed {seed} — initial inventory/domain labels",
              fill=(238, 238, 225))
    draw.text((780, 12),
              (f"transported 80-km structural proxy — mean "
               f"{100.0 * result['transported_global_proxy_fraction']:.2f}%"),
              fill=(238, 238, 225))
    selection = result["transported_proxy_scan"]["selection"]
    colors = {"low": (80, 220, 255), "medium": (100, 255, 120),
              "high": (255, 100, 210)}
    if selection["found"]:
        scale = 768.0 / authority["binary"].shape[0]
        for label, candidate in selection["assignment"].items():
            x0 = int(round(candidate["x0_km"] / CANONICAL_KM * scale))
            y0 = int(round(candidate["y0_km"] / CANONICAL_KM * scale))
            size = int(round(FRAME_KM / CANONICAL_KM * scale))
            box = (768 + x0, 48 + y0, 768 + x0 + size, 48 + y0 + size)
            draw.rectangle(box, outline=colors[label], width=4)
            draw.text((box[0] + 4, box[1] + 4), label,
                      fill=colors[label])
    else:
        draw.text((780, 32), "no separated target assignment",
                  fill=(255, 120, 120))
    path = out / f"seed{seed}_formation_panel.png"
    canvas.save(path)
    return {"file": path.name, "sha256": _sha256_file(path)}


def _component_rgb(mask: np.ndarray) -> np.ndarray:
    labels, components = legacy_formation._label_components(
        np.asarray(mask, bool), diagonal=True)
    rgb = np.zeros((*mask.shape, 3), np.uint8)
    rgb[:] = np.asarray((16, 42, 73), np.uint8)
    for label, _ in enumerate(components):
        rgb[labels == label] = _palette(label)
    return rgb


def _render_assignment_windows(seed: int, authority: dict,
                               result: dict, out: Path) -> dict:
    canvas = Image.new("RGB", (1152, 432), (7, 16, 28))
    draw = ImageDraw.Draw(canvas)
    selection = result["transported_proxy_scan"]["selection"]
    if not selection["found"]:
        draw.text((20, 20), f"seed {seed}: no separated assignment",
                  fill=(255, 130, 130))
    else:
        frame_cells = int(round(FRAME_KM / CANONICAL_KM))
        for index, label in enumerate(("low", "medium", "high")):
            candidate = selection["assignment"][label]
            x0 = int(round(candidate["x0_km"] / CANONICAL_KM))
            y0 = int(round(candidate["y0_km"] / CANONICAL_KM))
            sub = authority["binary"][
                y0:y0 + frame_cells, x0:x0 + frame_cells]
            image = Image.fromarray(_component_rgb(sub), "RGB").resize(
                (384, 384), Image.Resampling.NEAREST)
            canvas.paste(image, (384 * index, 48))
            review = result["assigned_window_reviews"]["windows"][label]
            draw.text(
                (384 * index + 8, 10),
                (f"{label} {100.0 * candidate['continental_fraction']:.2f}% "
                 f"sig={review['significant_component_count']} "
                 f"cover={100.0 * review['significant_component_coverage']:.1f}%"),
                fill=(238, 238, 225),
            )
    path = out / f"seed{seed}_assigned_proxy_windows.png"
    canvas.save(path)
    return {"file": path.name, "sha256": _sha256_file(path)}


def _render_capacity_failure(seed: int, layout: dict, out: Path) -> dict:
    image = Image.fromarray(_formation_rgb(layout), "RGB").resize(
        (768, 768), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (768, 816), (8, 18, 30))
    canvas.paste(image, (0, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 12),
        (f"seed {seed} — capacity failure "
         f"{layout['capacity_cells']}/{layout['requested_cells']} cells"),
        fill=(255, 125, 125),
    )
    path = out / f"seed{seed}_capacity_failure.png"
    canvas.save(path)
    return {"file": path.name, "sha256": _sha256_file(path)}


def _render_montage(artifacts: list[dict], out: Path,
                    filename: str) -> dict:
    panels = [Image.open(out / item["file"]).convert("RGB")
              for item in artifacts]
    thumb_size = (768, 408)
    canvas = Image.new("RGB", (1536, 1632), (5, 12, 21))
    for index, panel in enumerate(panels):
        panel.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        x = (index % 2) * thumb_size[0]
        y = (index // 2) * thumb_size[1]
        canvas.paste(panel, (x, y))
    path = out / filename
    canvas.save(path)
    for panel in panels:
        panel.close()
    return {"file": path.name, "sha256": _sha256_file(path)}


def _save_masks(seed: int, out: Path, **arrays) -> dict:
    path = out / f"seed{seed}_authority_masks.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, **{
            key: np.asarray(value) for key, value in arrays.items()})
    return {"file": path.name, "sha256": _sha256_file(path),
            "arrays": sorted(arrays)}


def _save_scan_table(seed: int, out: Path, canonical_scan: dict,
                     transported_scan: dict,
                     fine_scan: dict | None = None) -> dict:
    canonical = canonical_scan["records"]
    transported = transported_scan["records"]
    fine = None if fine_scan is None else fine_scan["records"]
    if len(canonical) != len(transported):
        raise AssertionError("scan tables have different lengths")
    if fine is not None and len(canonical) != len(fine):
        raise AssertionError("fine scan table has different length")
    path = out / f"seed{seed}_complete_scan_table.npz"
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            x0_km=np.asarray([item["x0_km"] for item in canonical]),
            y0_km=np.asarray([item["y0_km"] for item in canonical]),
            tie_key=np.asarray(
                [item["tie_key"] for item in canonical], dtype=np.uint64),
            canonical_fraction=np.asarray(
                [item["continental_fraction"] for item in canonical]),
            transported_proxy_fraction=np.asarray(
                [item["continental_fraction"] for item in transported]),
            transported_component_gate_status=np.asarray(
                [item.get("component_gate_status", -1)
                 for item in transported], dtype=np.int8),
            fine_40km_proxy_fraction=np.asarray(
                ([np.nan] * len(canonical) if fine is None else
                 [item["continental_fraction"] for item in fine])),
            fine_40km_component_gate_status=np.asarray(
                ([-1] * len(canonical) if fine is None else
                 [item.get("component_gate_status", -1)
                  for item in fine]), dtype=np.int8),
        )
    return {"file": path.name, "sha256": _sha256_file(path),
            "row_count": len(canonical)}


def _protocol() -> dict:
    cfg = legacy_formation._atlas_config(
        TARGET_INITIAL_CONTINENTAL_FRACTION)
    return {
        "experiment": EXPERIMENT,
        "manifest_role": "exclusive_pre_generation_protocol_precommit",
        "source_fingerprint": _source_fingerprint(),
        "prior_evidence": _verify_prior_evidence(),
        "seed_policy": {
            "seeds": list(SEEDS),
            "selection": (
                "mechanical next eight integers after development-exposed "
                "seeds 139-150"),
            "exposed_development_seeds_not_validation":
                list(EXPOSED_DEVELOPMENT_SEEDS),
            "retry": None,
            "seed_cli": None,
            "all_seeds_count": True,
        },
        "formation": {
            "world_km": PARENT_KM,
            "canonical_km": CANONICAL_KM,
            "target_initial_continental_fraction":
                TARGET_INITIAL_CONTINENTAL_FRACTION,
            "diagnostic_prefix_fraction":
                PREFIX_INITIAL_CONTINENTAL_FRACTION,
            "quota_semantics": (
                "exact world-area canonical-cell crust-production inventory; "
                "capacity exhaustion is persisted as a diagnostic support "
                "mask, stops that seed before structure/selection, and is "
                "never accepted or renormalized"),
            "carrier_threshold": legacy_formation.CARRIER_THRESHOLD,
            "nucleus_threshold":
                legacy_formation.NUCLEUS_CRATON_THRESHOLD,
            "nuclei": "one strongest-craton cell per active carrier",
            "growth": (
                "unchanged assembly-speed Dijkstra, globally ranked by "
                "arrival then coordinate tie until quota exhaustion"),
            "per_carrier_quota": None,
            "prefix_requirement": (
                "0.14 mask is a strict nested prefix with identical fields, "
                "eligible nuclei, selected nucleus IDs/pivots/carriers, "
                "carrier ownership, and plate sites"),
            "crop_border_elevation_sea_level_or_target_input": None,
            "configured_plates": int(cfg.plates),
        },
        "execution": {
            "primary_inventory_layouts": len(SEEDS),
            "diagnostic_prefix_layouts": len(SEEDS),
            "coarse_structure_builds": (
                "one per capacity-passing seed; capacity failure stops only "
                "that seed before structural work"),
            "coarse_structure_nominal_km": STRUCTURE_NOMINAL_KM,
            "fine_sentinel_builds": (
                "one iff seed 151 passes capacity; otherwise zero"),
            "fine_sentinel_seed": FINE_SENTINEL_SEED,
            "fine_structure_nominal_km": FINE_STRUCTURE_NOMINAL_KM,
            "nested_formation_probe_seed": FINE_SENTINEL_SEED,
            "nested_padding_km_each_side": NESTED_PADDING_KM,
            "elevation_builds": 0,
            "surface_process_solves": 0,
            "selectors_start_only_after_each_structure": True,
        },
        "post_structure_composition_screen": {
            "guard_km": NUMERICAL_GUARD_KM,
            "stride_km": CANDIDATE_STRIDE_KM,
            "candidate_count_per_mask": 61 ** 2,
            "target_bands": {
                label: {"minimum": band[0], "maximum": band[1],
                        "maximum_inclusive": band[2]}
                for label, band in TARGET_BANDS.items()},
            "water_or_contour_gate": None,
            "proxy_semantics": (
                "mean transported continental occupancy; not final land"),
            "distinct_low_medium_high_assignment": True,
            "assignment_candidate_eligibility": (
                "candidate must be in its target band and pass the frozen "
                "component-count/coverage gate before the exact separated "
                "triplet search"),
            "selection_authority": (
                "transported selection is morphology-qualified; the "
                "composition-only selection is retained as an explicitly "
                "named diagnostic and never gates readiness"),
            "minimum_origin_chebyshev_separation_km":
                MIN_ORIGIN_CHEBYSHEV_SEPARATION_KM,
        },
        "automatic_readiness_gates": {
            "exact_inventory_and_formation_invariants": "8 of 8 seeds",
            "ready_seed_count": f"{MIN_READY_SEEDS} of {len(SEEDS)}",
            "ready_seed_requires": [
                "every active domain receives >1 selected cell",
                "every selected domain has a strictly earlier 8-connected path to its nucleus",
                "every active domain remains represented after transport",
                "one separated low/medium/high transported-proxy assignment",
                (f"each assigned window has {MIN_SIGNIFICANT_COMPONENTS}-"
                 f"{MAX_SIGNIFICANT_COMPONENTS} connected components at >= "
                 f"{SIGNIFICANT_COMPONENT_FRAME_FRACTION:.2f} frame area"),
                (f"significant components cover >= "
                 f"{MIN_SIGNIFICANT_COMPONENT_COVERAGE:.2f} of binary proxy land"),
            ],
            "fine_sentinel_iou": f">= {MIN_SENTINEL_RESOLUTION_IOU}",
            "fine_sentinel_fraction_delta":
                f"<= {MAX_SENTINEL_FRACTION_DELTA}",
            "fine_sentinel_separated_assignment": True,
            "nested_inner_binary_iou": f">= {MIN_NESTED_INNER_IOU}",
            "manual_batch_veto": (
                "systematic round bodies, carrier lace, isochrone rings, or "
                "grid/tie directionality blocks a full solve"),
        },
        "interpretation": [
            "Formation-only readiness cannot validate final water borders.",
            "Exact initial inventory cannot promise final visible land.",
            "The nested probe quantifies but cannot close finite-rim causality.",
            "Manual morphology review is required before a full solve.",
            "Complete primary/fine authority masks and executed scan tables, "
            "plus the complete padded nested binary authority, are persisted.",
            "No result from this run changes production behavior.",
        ],
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


def _seed_result(seed: int, layout: dict, structure):
    authority = _sample_structure_authority(structure, layout["q"])
    canonical_scan = _scan_windows(layout["selected"])
    transported_scan = _scan_windows(authority["proxy"])
    morphology_qualification = _qualify_scan_morphology(
        transported_scan, authority["binary"],
        authority["dominant_tag"])
    transported_scan["selection"] = morphology_qualification.pop(
        "selection")
    transported_scan["morphology_qualification"] = morphology_qualification
    qualified_selection = transported_scan["selection"]
    window_reviews = _assigned_window_reviews(
        authority["binary"], authority["dominant_tag"],
        qualified_selection)
    domains = _domain_summary(layout)
    represented = _represented_domains(structure)
    inventory_exact = (
        layout["capacity_passed"]
        and layout["selected_cells"] == layout["requested_cells"])
    formation_invariants = all((
        inventory_exact,
        bool(np.all(~layout["selected"]
                    | (layout["carrier_owner"] >= 0))),
        domains["all_connected_to_seed"],
        layout["strict_predecessor_paths"],
        domains["starved_domain_count"] == 0,
        len(layout["domains"]) == len(layout["carriers"]),
        layout["prefix_checks"]["passed"],
    ))
    ready_gates = {
        "inventory_exact": inventory_exact,
        "formation_invariants": formation_invariants,
        "all_domains_represented": (
            represented["represented_domain_count"]
            == domains["active_domain_count"]),
        "transported_proxy_assignment":
            qualified_selection["found"],
        "assigned_window_components": window_reviews["passed"],
    }
    result = {
        "seed": seed,
        "status": "complete",
        "inventory": {
            "requested_cells": layout["requested_cells"],
            "capacity_cells": layout["capacity_cells"],
            "selected_cells": layout["selected_cells"],
            "target_world_fraction": layout["target_fraction"],
            "capacity_world_fraction": float(
                layout["capacity_cells"] / layout["selected"].size),
            "selected_world_fraction": layout["selected_fraction"],
            "capacity_passed": layout["capacity_passed"],
            "cutoff_arrival": layout["cutoff_arrival"],
            "cutoff_tie": layout["cutoff_tie"],
            "selected_cutoff_cohort_cells":
                layout["selected_cutoff_cohort_cells"],
            "discovered_unselected_cutoff_cohort_cells":
                layout["discovered_unselected_cutoff_cohort_cells"],
            "quantization_error_cells": float(
                layout["selected_cells"]
                - TARGET_INITIAL_CONTINENTAL_FRACTION
                * layout["selected"].size),
        },
        "domains": domains,
        "prefix_checks": layout["prefix_checks"],
        "transport": {
            "structure_n": int(structure.n),
            "actual_structure_km": float(
                structure.world_km / structure.n),
            "alive_plates": int(structure.alive_plates),
            **represented,
        },
        "canonical_scan": _scan_report(canonical_scan),
        "transported_proxy_scan": _scan_report(transported_scan),
        "assigned_window_reviews": window_reviews,
        "transported_global_proxy_fraction": float(
            authority["proxy"].mean()),
        "transported_global_binary_fraction": float(
            authority["binary"].mean()),
        "ready_gates": ready_gates,
        "ready": all(ready_gates.values()),
    }
    return result, authority, canonical_scan, transported_scan


def _nested_probe(seed: int, plate_count: int, small_layout: dict) -> dict:
    padded_world = PARENT_KM + 2.0 * NESTED_PADDING_KM
    padded = _inventory_layout(
        seed, padded_world, plate_count,
        origin_km=-NESTED_PADDING_KM,
        strict_capacity=False,
    )
    offset = int(round(NESTED_PADDING_KM / CANONICAL_KM))
    central = padded["selected"][
        offset:offset + small_layout["selected"].shape[0],
        offset:offset + small_layout["selected"].shape[1],
    ]
    central_slice = np.s_[
        offset:offset + small_layout["selected"].shape[0],
        offset:offset + small_layout["selected"].shape[1],
    ]
    padded_ties = _coordinate_ties(seed, padded["q"])[central_slice]
    small_ties = _coordinate_ties(seed, small_layout["q"])
    addressing_checks = {
        "assembly_identical": bool(np.array_equal(
            small_layout["assembly"], padded["assembly"][central_slice])),
        "craton_identical": bool(np.array_equal(
            small_layout["craton"], padded["craton"][central_slice])),
        "coordinate_ties_identical": bool(np.array_equal(
            small_ties, padded_ties)),
        "carrier_eligibility_identical": bool(np.array_equal(
            small_layout["carrier_raw"] >= 0,
            padded["carrier_raw"][central_slice] >= 0)),
        "eligible_nuclei_identical": bool(np.array_equal(
            small_layout["raw_nuclei"],
            padded["raw_nuclei"][central_slice])),
    }
    addressing_passed = all(addressing_checks.values())
    inner = int(round(FRAME_KM / CANONICAL_KM))
    small_inner = small_layout["selected"][inner:-inner, inner:-inner]
    central_inner = central[inner:-inner, inner:-inner]
    return {
        "padded_world_km": padded_world,
        "padding_km_each_side": NESTED_PADDING_KM,
        "padded_capacity_passed": padded["capacity_passed"],
        "absolute_addressing_checks": addressing_checks,
        "absolute_addressing_passed": addressing_passed,
        "full_parent_iou": _mask_iou(small_layout["selected"], central),
        "inner_guard_km": FRAME_KM,
        "inner_iou": _mask_iou(small_inner, central_inner),
        "passed": (
            padded["capacity_passed"]
            and addressing_passed
            and _mask_iou(small_inner, central_inner)
            >= MIN_NESTED_INNER_IOU),
        "central_mask": central,
        "padded_full_mask": padded["selected"],
    }


def _capacity_failure_result(seed: int, layout: dict):
    domains = _domain_summary(layout)
    ready_gates = {
        "inventory_exact": False,
        "formation_invariants": False,
        "all_domains_represented": False,
        "transported_proxy_assignment": False,
        "assigned_window_components": False,
    }
    result = {
        "seed": seed,
        "status": "capacity_failure_no_structural_build",
        "inventory": {
            "requested_cells": layout["requested_cells"],
            "capacity_cells": layout["capacity_cells"],
            "selected_diagnostic_cells": layout["selected_cells"],
            "target_world_fraction": layout["target_fraction"],
            "capacity_world_fraction": float(
                layout["capacity_cells"] / layout["selected"].size),
            "capacity_passed": False,
        },
        "domains": domains,
        "prefix_checks": layout["prefix_checks"],
        "transport": None,
        "canonical_capacity_exhaustion_scan": None,
        "transported_proxy_scan": None,
        "assigned_window_reviews": {
            "passed": False,
            "reason": "capacity_failure",
            "windows": {},
        },
        "ready_gates": ready_gates,
        "ready": False,
    }
    return result


def _phase_execute(out: Path, expected_sha256: str) -> dict:
    protocol, protocol_sha256 = _require_execute_output(out, expected_sha256)
    started = time.perf_counter()
    cfg = legacy_formation._atlas_config(
        TARGET_INITIAL_CONTINENTAL_FRACTION)
    counters = {
        "inventory_layouts": 0,
        "prefix_layouts": 0,
        "coarse_structure_builds": 0,
        "fine_sentinel_builds": 0,
        "elevation_builds": 0,
        "surface_process_solves": 0,
        "post_structure_scans": 0,
    }
    results = []
    masks = {}
    layouts = {}
    scans = {}
    panel_artifacts = []
    assignment_artifacts = []
    for seed in SEEDS:
        layout = _inventory_layout(
            seed, PARENT_KM, int(cfg.plates), strict_capacity=False)
        counters["inventory_layouts"] += 1
        prefix = _layout_from_fields(
            seed, layout["q"], layout["assembly"], layout["craton"],
            int(cfg.plates), PREFIX_INITIAL_CONTINENTAL_FRACTION,
            strict_capacity=False)
        counters["prefix_layouts"] += 1
        layout["prefix_checks"] = _prefix_checks(
            seed, layout, prefix, int(cfg.plates))
        layouts[seed] = layout
        masks[seed] = {
            "canonical_selected": layout["selected"],
            "canonical_prefix_selected": prefix["selected"],
            "canonical_domain_label": layout["domain_label"],
            "canonical_carrier_owner": layout["carrier_owner"],
            "canonical_selected_arrival": np.where(
                layout["selected"], layout["arrival"], np.inf),
        }
        if not layout["capacity_passed"]:
            result = _capacity_failure_result(seed, layout)
            scans[seed] = None
            panel_artifacts.append(_render_capacity_failure(seed, layout, out))
            results.append(result)
            continue

        structure, _, _ = _build(seed, layout, STRUCTURE_NOMINAL_KM)
        counters["coarse_structure_builds"] += 1
        result, authority, canonical_scan, transported_scan = _seed_result(
            seed, layout, structure)
        counters["post_structure_scans"] += 2
        scans[seed] = (canonical_scan, transported_scan)
        masks[seed].update({
            "transported_80km_proxy": authority["proxy"].astype(np.float32),
            "transported_80km_binary": authority["binary"],
            "transported_80km_dominant_tag": authority["dominant_tag"],
        })
        panel_artifacts.append(_render_seed(
            seed, layout, authority, result, out))
        assignment_artifacts.append(_render_assignment_windows(
            seed, authority, result, out))
        results.append(result)

    sentinel_layout = layouts[FINE_SENTINEL_SEED]
    if sentinel_layout["capacity_passed"]:
        fine_structure, _, _ = _build(
            FINE_SENTINEL_SEED, sentinel_layout,
            FINE_STRUCTURE_NOMINAL_KM)
        counters["fine_sentinel_builds"] += 1
        fine_authority = _sample_structure_authority(
            fine_structure, sentinel_layout["q"])
        fine_scan = _scan_windows(fine_authority["proxy"])
        fine_morphology_qualification = _qualify_scan_morphology(
            fine_scan, fine_authority["binary"],
            fine_authority["dominant_tag"])
        fine_scan["selection"] = fine_morphology_qualification.pop(
            "selection")
        fine_scan["morphology_qualification"] = (
            fine_morphology_qualification)
        counters["post_structure_scans"] += 1
        coarse_binary = masks[FINE_SENTINEL_SEED][
            "transported_80km_binary"]
        coarse_proxy = masks[FINE_SENTINEL_SEED][
            "transported_80km_proxy"]
        resolution_iou = _mask_iou(
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
            "fine_scan": _scan_report(fine_scan),
            "passed": (
                resolution_iou >= MIN_SENTINEL_RESOLUTION_IOU
                and fraction_delta <= MAX_SENTINEL_FRACTION_DELTA
                and fine_scan["selection"]["found"]),
        }
        masks[FINE_SENTINEL_SEED].update({
            "transported_40km_proxy":
                fine_authority["proxy"].astype(np.float32),
            "transported_40km_binary": fine_authority["binary"],
            "transported_40km_dominant_tag":
                fine_authority["dominant_tag"],
        })
    else:
        fine_sentinel = {
            "seed": FINE_SENTINEL_SEED,
            "passed": False,
            "reason": "sentinel_capacity_failure",
        }

    nested = _nested_probe(
        FINE_SENTINEL_SEED, int(cfg.plates), sentinel_layout)
    counters["inventory_layouts"] += 1
    masks[FINE_SENTINEL_SEED]["nested_central_selected"] = nested.pop(
        "central_mask")
    masks[FINE_SENTINEL_SEED]["nested_padded_full_selected"] = nested.pop(
        "padded_full_mask")

    mask_artifacts = []
    scan_artifacts = []
    for seed in SEEDS:
        mask_artifacts.append(_save_masks(seed, out, **masks[seed]))
        if scans[seed] is not None:
            scan_artifacts.append(_save_scan_table(
                seed, out, scans[seed][0], scans[seed][1],
                fine_scan if seed == FINE_SENTINEL_SEED else None))
    panel_montage = _render_montage(
        panel_artifacts, out, "formation_panels_montage.png")
    if assignment_artifacts:
        assignment_montage = _render_montage(
            assignment_artifacts, out, "assignment_windows_montage.png")
    else:
        assignment_montage = None

    exact_count = sum(
        item["ready_gates"]["inventory_exact"] for item in results)
    invariant_count = sum(
        item["ready_gates"]["formation_invariants"] for item in results)
    ready_count = sum(item["ready"] for item in results)
    assignment_count = sum(
        item["ready_gates"]["transported_proxy_assignment"]
        for item in results)
    passed_window_count = sum(
        sum(window["passed"] for window in
            item["assigned_window_reviews"]["windows"].values())
        for item in results)
    capacity_pass_count = sum(
        item["ready_gates"]["inventory_exact"] for item in results)
    expected_fine = int(sentinel_layout["capacity_passed"])
    aggregate_gates = {
        "exact_inventory_all_seeds": exact_count == len(SEEDS),
        "formation_invariants_all_seeds": invariant_count == len(SEEDS),
        "all_seeds_have_proxy_assignment": assignment_count == len(SEEDS),
        "all_assigned_windows_pass_components": passed_window_count == 24,
        "all_seeds_ready": ready_count == MIN_READY_SEEDS,
        "fine_sentinel": fine_sentinel["passed"],
        "nested_inner_stability": nested["passed"],
        "execution_counts": counters == {
            "inventory_layouts": len(SEEDS) + 1,
            "prefix_layouts": len(SEEDS),
            "coarse_structure_builds": capacity_pass_count,
            "fine_sentinel_builds": expected_fine,
            "elevation_builds": 0,
            "surface_process_solves": 0,
            "post_structure_scans": (
                2 * capacity_pass_count
                + expected_fine),
        },
    }
    automatic_pass = all(aggregate_gates.values())
    report = {
        "experiment": EXPERIMENT,
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
                "fewer substantial coherent domains than seed 138",
                "varied non-stamped domain scale and outline",
                "no new grid or common-radius frontier signature",
                "no carrier-threshold lace or tendrils",
            ],
        },
        "artifacts": {
            "formation_panels": panel_artifacts,
            "assigned_proxy_windows": assignment_artifacts,
            "formation_montage": panel_montage,
            "assignment_windows_montage": assignment_montage,
            "authority_masks": mask_artifacts,
            "complete_scan_tables": scan_artifacts,
        },
        "interpretation_limits": protocol["interpretation"],
        "recommend_full_parent_solve": False,
        "promotion": False,
        "section3b_status": "unresolved_finite_parent_global_quota",
    }
    report_sha256 = _write_json_exclusive(out / "report.json", report)
    _write_json_exclusive(out / "report.sha256.json", {
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
    q = (np.arange(12) + 0.5) * 64.0
    assembly = np.full((12, 12), -1.0)
    assembly[1:7, 1:6] = 0.35
    assembly[5:11, 7:11] = 0.28
    craton = np.full((12, 12), -1.0)
    craton[2:4, 2:4] = 0.5
    craton[7:9, 8:10] = 0.55
    low = _layout_from_fields(
        7, q, assembly, craton, 4, 0.15, strict_capacity=True)
    high = _layout_from_fields(
        7, q, assembly, craton, 4, 0.25, strict_capacity=True)
    reversed_layout = _layout_from_fields(
        7, q, assembly, craton, 4, 0.25,
        reverse_components_for_test=True, strict_capacity=True)
    if len(high["domains"]) != 2:
        raise AssertionError("one nucleus per carrier failed")
    if high["selected_cells"] != round(0.25 * assembly.size):
        raise AssertionError("exact inventory failed")
    if not np.all(~low["selected"] | high["selected"]):
        raise AssertionError("inventory masks are not nested")
    if not np.array_equal(high["domain_label"],
                          reversed_layout["domain_label"]):
        raise AssertionError("component enumeration affects authority")
    if not all(item["connected_to_seed"] for item in high["domains"]):
        raise AssertionError("selected domain disconnected from seed")
    if not high["strict_predecessor_paths"]:
        raise AssertionError("strict predecessor path check failed")
    if np.any(high["selected"] & (high["carrier_owner"] < 0)):
        raise AssertionError("selected material escaped carrier support")
    try:
        _layout_from_fields(
            7, q, assembly, craton, 4, 0.90, strict_capacity=True)
    except ValueError:
        capacity_overflow_raised = True
    else:
        raise AssertionError("capacity overflow did not raise")

    continent, tags = _make_samplers(high)
    X, Y = np.meshgrid(q, q)
    union = np.zeros(high["selected"].shape, bool)
    tag_union = np.full(high["selected"].shape, -1, np.int32)
    for plate in range(4):
        present = continent(plate, Y, X)
        sampled_tags = tags(plate, Y, X)
        if np.any((sampled_tags >= 0) != present):
            raise AssertionError("sampler/tag disagreement")
        union |= present
        tag_union[present] = sampled_tags[present]
    if not np.array_equal(union, high["selected"]):
        raise AssertionError("sampler differs from selected mask")
    if not np.array_equal(tag_union, high["domain_label"]):
        raise AssertionError("tag sampler differs from authority labels")

    padded_q = (np.arange(16) - 2 + 0.5) * 64.0
    if not np.array_equal(
            _coordinate_ties(7, q),
            _coordinate_ties(7, padded_q)[2:14, 2:14]):
        raise AssertionError("coordinate ties changed under nested extent")

    synthetic_records = [
        {"x0_km": 2560.0, "y0_km": 2560.0,
         "continental_fraction": 0.20, "tie_key": 1},
        {"x0_km": 6656.0, "y0_km": 2560.0,
         "continental_fraction": 0.35, "tie_key": 2},
        {"x0_km": 2560.0, "y0_km": 6656.0,
         "continental_fraction": 0.49, "tie_key": 3},
    ]
    synthetic_selection = _select_assignment(synthetic_records)
    if not synthetic_selection["found"]:
        raise AssertionError("separated assignment search failed")
    morphology = np.zeros((384, 384), bool)
    morphology_tags = np.full((384, 384), -1, np.int32)
    for index, candidate in enumerate(synthetic_records):
        x0 = int(candidate["x0_km"] / 64.0)
        y0 = int(candidate["y0_km"] / 64.0)
        morphology[y0 + 4:y0 + 20, x0 + 4:x0 + 20] = True
        morphology[y0 + 36:y0 + 56, x0 + 36:x0 + 56] = True
        morphology_tags[y0 + 4:y0 + 20, x0 + 4:x0 + 20] = 2 * index
        morphology_tags[y0 + 36:y0 + 56, x0 + 36:x0 + 56] = 2 * index + 1
    reviews = _assigned_window_reviews(
        morphology, morphology_tags, synthetic_selection)
    if not reviews["passed"]:
        raise AssertionError("assigned-window component review failed")
    qualified = _qualify_scan_morphology(
        {"records": synthetic_records}, morphology, morphology_tags)
    if not qualified["selection"]["found"]:
        raise AssertionError("morphology-qualified assignment search failed")

    if round(TARGET_INITIAL_CONTINENTAL_FRACTION * 384 ** 2) != 41288:
        raise AssertionError("frozen world quota changed")

    return {
        "passed": True,
        "one_seed_per_carrier": True,
        "exact_inventory": high["selected_cells"],
        "nested_inventories": True,
        "enumeration_invariant": True,
        "connected_domains": True,
        "strict_predecessor_paths": True,
        "capacity_overflow_raised": capacity_overflow_raised,
        "sampler_tag_authority": True,
        "absolute_coordinate_ties": True,
        "separated_assignment": True,
        "assigned_window_components": True,
        "morphology_qualified_assignment": True,
        "frozen_quota_cells": 41288,
    }


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
        print(json.dumps(_self_check(), indent=2))
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
