"""Fail-fast structural oracle for field-nucleated continental domains.

This is a private formation experiment.  It never reads a delivered-frame
coordinate while forming crust, never retries a seed, and never edits an
elevation field near a crop.  A fixed world-km assembly field supplies the
material through which continental accretion can persist; an independent
shorter field supplies ancient-crust nuclei.  A fixed finite accretion
interval lets fronts advance through favorable assembly material; the
resulting connected support becomes continental crust.

Three formation-native identities and crop origins are frozen before the
first structural/elevation build.  The same identities and origins must pass
120, 80, and 40 km in order.  Failure stops the ladder immediately.  No
surface-process solve is part of this spike.
"""

from __future__ import annotations

import argparse
from collections import deque
from functools import lru_cache
import hashlib
import heapq
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from engine import noise
from engine.elevation import coarse_elevation
from engine.rng import fnv1a64, stage_rng, stage_salt
from engine.surface import _bicubic
from engine.tectonics import FRAME_KM, build_structure
from spikes.atlas_survey import (
    ATLAS_GUARD_KM,
    ATLAS_KM,
    CORE_INSET_KM,
    EDGE_BAND_KM,
    EROSION_TIME_MAX_MYR,
    FINE_RELIEF_MAX_M,
    MARINE_DEPOSITION_MAX_M,
    MID_RELIEF_MAX_M,
    ORACLE_KM,
    SHORTLIST_ORACLE_CLEARANCE_M,
    SURVEY_KM,
    _atlas_config,
    _common_fields,
    _smoothstep,
    _terrain_rgb,
)
from spikes.visible_contour_gate import evaluate_visible_border_contours


# ------------------------------------------------------------------ model

CANONICAL_KM = 64.0
DENSE_HEAD_KM = 16.0
ASSEMBLY_WAVELENGTH_KM = 5000.0
ASSEMBLY_OCTAVES = 3
CRATON_WAVELENGTH_KM = 1800.0
CRATON_OCTAVES = 3

# The plate carrier is a lower phase of the same assembly field.  It leaves
# a natural oceanic apron outside the higher continental-growth phase.
CARRIER_THRESHOLD = 0.12
DESIGN_BUDGET = 0.65
PUBLIC_REFERENCE_BUDGET = 0.30
DESIGN_ACCRETION_TIME_KM = 900.0
PUBLIC_REFERENCE_ACCRETION_TIME_KM = 600.0
NUCLEUS_CRATON_THRESHOLD = 0.20

SHORTLIST_SIZE = 3
MIN_LAND_FRACTION = 0.20
MAX_LAND_FRACTION = 0.50
MIN_SIGNIFICANT_COMPONENT_FRACTION = 0.04
MIN_SIGNIFICANT_COMPONENTS = 2
MAX_SIGNIFICANT_COMPONENTS = 4
MIN_OWNER_FRACTION = 0.85
MIN_CAPTURE_FRACTION = 0.80
REQUIRED_CLEARANCE_M = SHORTLIST_ORACLE_CLEARANCE_M
REQUIRED_LAND_IOU = 0.90
MAX_RESOLUTION_DELTA = 0.05
FINAL_SUBCELL_OFFSETS = (
    (-0.25, -0.25), (-0.25, 0.25),
    (0.25, -0.25), (0.25, 0.25),
)

POSITIVE_RELIEF_BOUND_M = (
    MID_RELIEF_MAX_M + FINE_RELIEF_MAX_M + MARINE_DEPOSITION_MAX_M
)

FORMATION_DIAGNOSTICS: dict[str, object] = {}

SOURCE_FINGERPRINT_FILES = (
    "engine/elevation.py",
    "engine/noise.py",
    "engine/rng.py",
    "engine/surface.py",
    "engine/tectonics.py",
    "spikes/atlas_survey.py",
    "spikes/field_accretion_oracle.py",
    "spikes/visible_contour_gate.py",
)


def _source_fingerprint() -> dict:
    """Hash every source file that can affect this private oracle."""
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


def _accretion_time_limit(continental_budget: float) -> float:
    """Fixed finite-growth interval; never fitted to a seed or crop."""
    budget = np.clip(
        float(continental_budget), PUBLIC_REFERENCE_BUDGET, DESIGN_BUDGET)
    fraction = ((budget - PUBLIC_REFERENCE_BUDGET)
                / (DESIGN_BUDGET - PUBLIC_REFERENCE_BUDGET))
    return float(
        PUBLIC_REFERENCE_ACCRETION_TIME_KM
        + fraction * (DESIGN_ACCRETION_TIME_KM
                      - PUBLIC_REFERENCE_ACCRETION_TIME_KM))


def _canonical_fields(seed: int, q: np.ndarray) -> tuple[np.ndarray, ...]:
    X, Y = np.meshgrid(q, q)
    assembly = noise.fbm(
        X, Y, ASSEMBLY_WAVELENGTH_KM, ASSEMBLY_OCTAVES,
        stage_salt(seed, "atlas-field-accretion-assembly-v1"),
    )
    craton = noise.fbm(
        X, Y, CRATON_WAVELENGTH_KM, CRATON_OCTAVES,
        stage_salt(seed, "atlas-field-accretion-craton-v1"),
    )
    return X, Y, assembly, craton


def _label_components(mask: np.ndarray, *, diagonal: bool = True):
    """Deterministic labels plus cell coordinates for every component."""
    mask = np.asarray(mask, bool)
    labels = np.full(mask.shape, -1, np.int32)
    components = []
    if diagonal:
        neighbors = tuple(
            (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if dy or dx
        )
    else:
        neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    height, width = mask.shape
    for sy, sx in np.argwhere(mask):
        if labels[sy, sx] >= 0:
            continue
        index = len(components)
        labels[sy, sx] = index
        queue = deque([(int(sy), int(sx))])
        ys, xs = [], []
        while queue:
            y, x = queue.popleft()
            ys.append(y)
            xs.append(x)
            for dy, dx in neighbors:
                yy, xx = y + dy, x + dx
                if (0 <= yy < height and 0 <= xx < width
                        and mask[yy, xx] and labels[yy, xx] < 0):
                    labels[yy, xx] = index
                    queue.append((yy, xx))
        components.append((np.asarray(ys, np.int32),
                           np.asarray(xs, np.int32)))
    return labels, components


def _field_id(seed: int, kind: str, y_km: float, x_km: float) -> str:
    value = fnv1a64(
        f"field-accretion-v1:{seed}:{kind}:{y_km:.0f}:{x_km:.0f}")
    return f"{value:016x}"


def _finite_accretion(assembly: np.ndarray, carrier_label: np.ndarray,
                      seeds: list[dict], canonical_km: float):
    """Multi-source finite-time growth through favorable lithosphere.

    Front speed rises smoothly with assembly potential.  Each front stays
    inside the broad carrier component containing its nucleus.  Fronts may
    meet naturally; no separator or post-growth erosion is inserted.
    """
    shape = assembly.shape
    arrival = np.full(shape, np.inf, np.float64)
    owner = np.full(shape, -1, np.int32)
    queue = []
    for domain_index, seed in enumerate(seeds):
        y, x = seed["canonical_yx"]
        arrival[y, x] = 0.0
        owner[y, x] = domain_index
        heapq.heappush(queue, (0.0, domain_index, y, x))

    neighbors = (
        (-1, -1, np.sqrt(2.0)), (-1, 0, 1.0),
        (-1, 1, np.sqrt(2.0)), (0, -1, 1.0),
        (0, 1, 1.0), (1, -1, np.sqrt(2.0)),
        (1, 0, 1.0), (1, 1, np.sqrt(2.0)),
    )
    max_time = DESIGN_ACCRETION_TIME_KM
    while queue:
        elapsed, domain_index, y, x = heapq.heappop(queue)
        if elapsed != arrival[y, x] or owner[y, x] != domain_index:
            continue
        carrier = seeds[domain_index]["carrier_raw_label"]
        for dy, dx, diagonal in neighbors:
            yy, xx = y + dy, x + dx
            if not (0 <= yy < shape[0] and 0 <= xx < shape[1]):
                continue
            if carrier_label[yy, xx] != carrier:
                continue
            potential = 0.5 * (assembly[y, x] + assembly[yy, xx])
            speed = np.clip(
                0.72 + 1.8 * (potential - CARRIER_THRESHOLD),
                0.45, 1.35)
            candidate = elapsed + canonical_km * diagonal / speed
            if candidate > max_time:
                continue
            if (candidate < arrival[yy, xx]
                    or (candidate == arrival[yy, xx]
                        and domain_index < owner[yy, xx])):
                arrival[yy, xx] = candidate
                owner[yy, xx] = domain_index
                heapq.heappush(
                    queue, (candidate, domain_index, yy, xx))
    return owner, arrival


@lru_cache(maxsize=16)
def _formation_layout(seed: int, world_km: float, plate_count: int) -> dict:
    count = int(round(world_km / CANONICAL_KM))
    ck = world_km / count
    q = (np.arange(count) + 0.5) * ck
    _, _, assembly, craton = _canonical_fields(seed, q)

    carrier_raw, carrier_components = _label_components(
        assembly > CARRIER_THRESHOLD)
    raw_nuclei = ((carrier_raw >= 0)
                  & (craton > NUCLEUS_CRATON_THRESHOLD))
    _, nucleus_components = _label_components(raw_nuclei)
    seeds = []
    for ys, xs in nucleus_components:
        local = int(np.argmax(craton[ys, xs]))
        py, px = int(ys[local]), int(xs[local])
        seeds.append({
            "domain_id": _field_id(seed, "domain", q[py], q[px]),
            "canonical_yx": [py, px],
            "pivot_yx_km": [float(q[py]), float(q[px])],
            "carrier_raw_label": int(carrier_raw[py, px]),
            "nucleus_cells": int(ys.size),
        })
    seeds.sort(key=lambda item: item["domain_id"])
    domain_label, accretion_time = _finite_accretion(
        assembly, carrier_raw, seeds, ck)
    grown = domain_label >= 0

    active_carrier_labels = sorted({
        item["carrier_raw_label"] for item in seeds})
    if len(active_carrier_labels) > plate_count:
        raise ValueError(
            "field accretion produced more carriers than plates")

    carrier_records = []
    carrier_plate_by_raw = {}
    for raw_label in active_carrier_labels:
        ys, xs = carrier_components[raw_label]
        local = int(np.argmax(assembly[ys, xs]))
        py, px = int(ys[local]), int(xs[local])
        record = {
            "raw_label": int(raw_label),
            "carrier_id": _field_id(seed, "carrier", q[py], q[px]),
            "pivot_yx_km": [float(q[py]), float(q[px])],
            "canonical_cells": int(ys.size),
            "touches_world_rim": bool(
                np.any(ys == 0) or np.any(xs == 0)
                or np.any(ys == count - 1)
                or np.any(xs == count - 1)),
        }
        carrier_records.append(record)
    carrier_records.sort(key=lambda item: item["carrier_id"])
    for plate, record in enumerate(carrier_records):
        record["plate_id"] = plate
        carrier_plate_by_raw[record["raw_label"]] = plate

    carrier_owner = np.full(carrier_raw.shape, -1, np.int32)
    for raw_label, plate in carrier_plate_by_raw.items():
        carrier_owner[carrier_raw == raw_label] = plate

    domain_records = []
    domain_plate_by_label = np.full(len(seeds), -1, np.int32)
    for label, seed_record in enumerate(seeds):
        ys, xs = np.nonzero(domain_label == label)
        py, px = seed_record["canonical_yx"]
        carrier_raw_label = seed_record["carrier_raw_label"]
        plate_id = int(carrier_plate_by_raw[carrier_raw_label])
        domain_plate_by_label[label] = plate_id
        domain_records.append({
            "label": label,
            "domain_id": seed_record["domain_id"],
            "carrier_plate_id": plate_id,
            "pivot_yx_km": seed_record["pivot_yx_km"],
            "centroid_yx_km": [
                float(np.mean(q[ys])), float(np.mean(q[xs]))],
            "canonical_cells": int(ys.size),
            "area_km2": float(ys.size * ck ** 2),
            "nucleus_cells": seed_record["nucleus_cells"],
            "bbox_xyxy_km": [
                float(q[xs.min()] - 0.5 * ck),
                float(q[ys.min()] - 0.5 * ck),
                float(q[xs.max()] + 0.5 * ck),
                float(q[ys.max()] + 0.5 * ck),
            ],
        })

    return {
        "canonical_km": ck,
        "q": q,
        "assembly": assembly,
        "craton": craton,
        "nuclei": raw_nuclei,
        "grown": grown,
        "domain_label": domain_label,
        "accretion_time": accretion_time,
        "domain_plate_by_label": domain_plate_by_label,
        "carrier_owner": carrier_owner,
        "carriers": carrier_records,
        "domains": domain_records,
        "raw_domain_count": len(nucleus_components),
        "nucleated_domain_count": len(seeds),
    }


def _nearest_canonical(array: np.ndarray, y_km, x_km,
                       canonical_km: float, fill=-1):
    y = np.asarray(y_km, np.float64)
    x = np.asarray(x_km, np.float64)
    shape = np.broadcast(y, x).shape
    yy = np.broadcast_to(y, shape)
    xx = np.broadcast_to(x, shape)
    iy = np.floor(yy / canonical_km).astype(np.int64)
    ix = np.floor(xx / canonical_km).astype(np.int64)
    inside = ((iy >= 0) & (iy < array.shape[0])
              & (ix >= 0) & (ix < array.shape[1]))
    result = np.full(shape, fill, dtype=array.dtype)
    result[inside] = array[iy[inside], ix[inside]]
    return result


@lru_cache(maxsize=16)
def _plate_sites(seed: int, world_km: float, plate_count: int) -> np.ndarray:
    layout = _formation_layout(seed, world_km, plate_count)
    parent_sites = np.asarray(
        [item["pivot_yx_km"] for item in layout["carriers"]],
        np.float64,
    )
    remaining = plate_count - parent_sites.shape[0]
    rng = stage_rng(seed, "atlas-field-accretion-ocean-sites-v1")
    proposal_count = max(8192, 192 * plate_count)
    proposals = rng.uniform(
        0.02 * world_km, 0.98 * world_km, (proposal_count, 2))
    owner = _nearest_canonical(
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


def _partition_field_accretion(seed, n, ck, cfg):
    world = n * ck
    layout = _formation_layout(seed, world, int(cfg.plates))
    sites = _plate_sites(seed, world, int(cfg.plates))
    q = (np.arange(n) + 0.5) * ck
    X, Y = np.meshgrid(q, q)

    # One shared, continuous mantle-coordinate deformation gives irregular
    # plate boundaries without per-plate multiplicative bias.
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

    owner = _nearest_canonical(
        layout["carrier_owner"], Y, X, layout["canonical_km"], fill=-1)
    carrier = owner >= 0
    label[carrier] = owner[carrier]
    FORMATION_DIAGNOSTICS["partition"] = {
        "carrier_plate_count": len(layout["carriers"]),
        "ocean_plate_count": int(cfg.plates - len(layout["carriers"])),
        "sites_yx_km": sites.tolist(),
        "forced_carrier_fraction": float(carrier.mean()),
    }
    return label


def _continent_sampler(seed: int, layout: dict, budget: float):
    time_limit = _accretion_time_limit(budget)

    def sample(plate_id, material_y_km, material_x_km):
        label = _nearest_canonical(
            layout["domain_label"], material_y_km, material_x_km,
            layout["canonical_km"], fill=-1)
        valid = label >= 0
        owner = np.full(label.shape, -1, np.int32)
        owner[valid] = layout["domain_plate_by_label"][label[valid]]
        accretion_time = _nearest_canonical(
            layout["accretion_time"], material_y_km, material_x_km,
            layout["canonical_km"], fill=np.inf)
        return valid & (owner == plate_id) & (accretion_time <= time_limit)

    return sample


def _material_tag_sampler(layout: dict, budget: float):
    """Return the canonical domain label carried by continental material."""
    time_limit = _accretion_time_limit(budget)

    def sample(plate_id, material_y_km, material_x_km):
        label = _nearest_canonical(
            layout["domain_label"], material_y_km, material_x_km,
            layout["canonical_km"], fill=-1)
        valid = label >= 0
        owner = np.full(label.shape, -1, np.int32)
        owner[valid] = layout["domain_plate_by_label"][label[valid]]
        accretion_time = _nearest_canonical(
            layout["accretion_time"], material_y_km, material_x_km,
            layout["canonical_km"], fill=np.inf)
        belongs = valid & (owner == plate_id) & (accretion_time <= time_limit)
        return np.where(belongs, label, -1).astype(np.int32, copy=False)

    return sample


def _continent_seeder(sample):
    def seed(seed, n, ck, cfg, label, plates):
        q = (np.arange(n) + 0.5) * ck
        X, Y = np.meshgrid(q, q)
        for plate in range(cfg.plates):
            selected = sample(plate, Y, X) & (label == plate)
            if selected.any():
                plates[plate].cont |= selected
    return seed


def _initial_ocean_age(seed, n, ck, cfg):
    """Coordinate-addressed initial age provinces, in birth eras."""
    q = (np.arange(n) + 0.5) * ck
    X, Y = np.meshgrid(q, q)
    age_field = noise.fbm(
        X, Y, 2400.0, 4,
        stage_salt(seed, "atlas-field-accretion-initial-ocean-age-v1"),
    )
    age_eras = np.rint(np.clip(3.5 + 10.0 * age_field, 0.0, 7.0))
    return -age_eras.astype(np.int16)


# ---------------------------------------------------------- frozen crops

def _formation_crop_records(layout: dict) -> tuple[list[dict], list[dict]]:
    q = layout["q"]
    ck = layout["canonical_km"]
    labels = layout["domain_label"]
    frame_cells = (FRAME_KM / ck) ** 2
    lower = ATLAS_GUARD_KM
    upper = ATLAS_KM - ATLAS_GUARD_KM - FRAME_KM
    all_records = []

    seen = set()
    domains = layout["domains"]
    # A delivered composition is derived from the material-weighted midpoint
    # of a native domain pair.  The resulting crop may naturally contain a
    # third or fourth significant identity; every identity actually captured
    # is frozen. Enumeration happens before elevation or contour information.
    for proposed_group in combinations(domains, 2):
        weights = np.asarray(
            [item["canonical_cells"] for item in proposed_group],
            np.float64)
        centroids = np.asarray(
            [item["centroid_yx_km"] for item in proposed_group],
            np.float64)
        cy, cx = np.average(centroids, axis=0, weights=weights)
        x0 = float(np.round((cx - 0.5 * FRAME_KM) / ck) * ck)
        y0 = float(np.round((cy - 0.5 * FRAME_KM) / ck) * ck)
        reasons = []
        if not (lower <= x0 <= upper and lower <= y0 <= upper):
            reasons.append("guarded_atlas_rim")
        xs = np.flatnonzero((q >= x0) & (q < x0 + FRAME_KM))
        ys = np.flatnonzero((q >= y0) & (q < y0 + FRAME_KM))
        if xs.size == 0 or ys.size == 0:
            reasons.append("empty_crop")
            sub = np.empty((0, 0), np.int32)
        else:
            sub = labels[np.ix_(ys, xs)]

        land_fraction = float(np.mean(sub >= 0)) if sub.size else 0.0
        if np.any(sub >= 0):
            counts = np.bincount(
                sub[sub >= 0], minlength=len(layout["domains"]))
        else:
            counts = np.zeros(len(layout["domains"]), int)
        significant = np.flatnonzero(
            counts >= MIN_SIGNIFICANT_COMPONENT_FRACTION * frame_cells)
        significant_ids = [
            domains[int(index)]["domain_id"] for index in significant]
        proposed_ids = {
            item["domain_id"] for item in proposed_group}
        if not proposed_ids.issubset(set(significant_ids)):
            reasons.append("proposed_pair_not_captured")
        capture_by_id = {}
        for index in significant:
            domain = domains[int(index)]
            capture_by_id[domain["domain_id"]] = float(
                counts[index] / max(domain["canonical_cells"], 1))
        min_capture = min(capture_by_id.values(), default=0.0)
        if min_capture < MIN_CAPTURE_FRACTION:
            reasons.append("domain_capture")
        if not (MIN_LAND_FRACTION <= land_fraction
                < MAX_LAND_FRACTION):
            reasons.append("formation_land_fraction")
        if not (MIN_SIGNIFICANT_COMPONENTS <= len(significant)
                <= MAX_SIGNIFICANT_COMPONENTS):
            reasons.append("formation_component_count")

        member_ids = sorted(significant_ids)
        owned_cells = int(counts[significant].sum())
        land_cells = int(np.count_nonzero(sub >= 0))
        owner_fraction = float(owned_cells / max(land_cells, 1))
        if owner_fraction < MIN_OWNER_FRACTION:
            reasons.append("formation_owner_fraction")
        group_key = "field-accretion-group-v1:" + ":".join(member_ids)
        formation_id = f"{fnv1a64(group_key):016x}"
        key = (formation_id, x0, y0)
        if key in seen:
            continue
        seen.add(key)
        member_set = set(member_ids)
        member_records = [
            domain for domain in domains
            if domain["domain_id"] in member_set
        ]
        all_records.append({
            "domain_id": formation_id,
            "member_domain_ids": member_ids,
            "carrier_plate_ids": sorted({
                item["carrier_plate_id"] for item in member_records}),
            "canonical_pivots_yx_km": [
                item["pivot_yx_km"] for item in member_records],
            "origin_xy_km": [x0, y0],
            "owner_fraction": owner_fraction,
            "capture_fraction": float(min_capture),
            "capture_by_domain_id": capture_by_id,
            "formation_land_fraction": land_fraction,
            "significant_domain_count": int(len(significant)),
            "significant_domain_ids": significant_ids,
            "passed": not reasons,
            "reasons": reasons,
        })

    ranked = sorted(
        (record for record in all_records if record["passed"]),
        key=lambda item: (-item["formation_land_fraction"],
                          item["domain_id"]),
    )
    frozen = []
    frozen_group_ids = set()
    frozen_member_ids = set()
    for record in ranked:
        member_ids = set(record["member_domain_ids"])
        if (record["domain_id"] in frozen_group_ids
                or member_ids & frozen_member_ids):
            continue
        x0, y0 = record["origin_xy_km"]
        if all(
            max(abs(x0 - prior["origin_xy_km"][0]),
                abs(y0 - prior["origin_xy_km"][1])) >= FRAME_KM
            for prior in frozen
        ):
            frozen.append(record)
            frozen_group_ids.add(record["domain_id"])
            frozen_member_ids.update(member_ids)
            if len(frozen) == SHORTLIST_SIZE:
                break
    return frozen, all_records


def _layout_panel(layout: dict, frozen: list[dict], out_path: Path):
    assembly = layout["assembly"]
    lo, hi = np.percentile(assembly, (2.0, 98.0))
    shade = np.clip((assembly - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    rgb = np.empty((*assembly.shape, 3), np.uint8)
    rgb[..., 0] = 20 + 30 * shade
    rgb[..., 1] = 48 + 55 * shade
    rgb[..., 2] = 84 + 80 * shade
    carrier = layout["carrier_owner"] >= 0
    grown = layout["grown"]
    rgb[carrier] = np.array((45, 92, 112), np.uint8)
    rgb[grown] = np.array((142, 161, 103), np.uint8)
    rgb[layout["nuclei"] & grown] = np.array((229, 201, 91), np.uint8)
    image = Image.fromarray(rgb, "RGB").resize(
        (768, 768), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    scale = 768.0 / ATLAS_KM
    for index, record in enumerate(frozen):
        x0, y0 = record["origin_xy_km"]
        box = tuple(int(round(value * scale)) for value in (
            x0, y0, x0 + FRAME_KM, y0 + FRAME_KM))
        draw.rectangle(box, outline=(255, 255, 255), width=3)
        draw.text((box[0] + 4, box[1] + 4), str(index + 1),
                  fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


# -------------------------------------------------------------- head gate

def _exterior_component(water: np.ndarray) -> np.ndarray:
    water = np.asarray(water, bool)
    connected = np.zeros_like(water)
    queue = deque()
    for y in (0, water.shape[0] - 1):
        for x in range(water.shape[1]):
            if water[y, x] and not connected[y, x]:
                connected[y, x] = True
                queue.append((y, x))
    for x in (0, water.shape[1] - 1):
        for y in range(water.shape[0]):
            if water[y, x] and not connected[y, x]:
                connected[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if (0 <= yy < water.shape[0] and 0 <= xx < water.shape[1]
                    and water[yy, xx] and not connected[yy, xx]):
                connected[yy, xx] = True
                queue.append((yy, xx))
    return connected


def _component_summary(land: np.ndarray, cell_km: float) -> dict:
    _, components = _label_components(land)
    areas = sorted((ys.size * cell_km ** 2 for ys, _ in components),
                   reverse=True)
    significant_min = (
        MIN_SIGNIFICANT_COMPONENT_FRACTION * FRAME_KM ** 2)
    return {
        "count": len(areas),
        "significant_count": sum(area >= significant_min for area in areas),
        "largest_area_fractions": [
            float(area / FRAME_KM ** 2) for area in areas[:8]],
    }


def _collar_indices(q: np.ndarray, x0: float, y0: float):
    xs = np.flatnonzero(
        (q >= x0 - EDGE_BAND_KM)
        & (q < x0 + FRAME_KM + EDGE_BAND_KM))
    ys = np.flatnonzero(
        (q >= y0 - EDGE_BAND_KM)
        & (q < y0 + FRAME_KM + EDGE_BAND_KM))
    xr = q[xs] - x0
    yr = q[ys] - y0
    collar = (
        (xr[None, :] < EDGE_BAND_KM)
        | (xr[None, :] >= FRAME_KM - EDGE_BAND_KM)
        | (yr[:, None] < EDGE_BAND_KM)
        | (yr[:, None] >= FRAME_KM - EDGE_BAND_KM)
    )
    return ys, xs, collar


def _dense_world_fields(elevation: dict, structure):
    """Sample the complete atlas before exterior-ocean flood filling.

    Connectivity cannot be established from an isolated crop collar: a
    narrow positive bridge elsewhere could disconnect that collar from the
    world exterior.  The dense field is therefore global and authoritative.
    """
    ck = structure.world_km / structure.n
    count = int(round(structure.world_km / DENSE_HEAD_KM))
    q = (np.arange(count) + 0.5) * (structure.world_km / count)
    h = _bicubic(
        np.asarray(elevation["h"], np.float64),
        q[:, None], q[None, :], ck)
    uplift = _bicubic(
        np.asarray(elevation["uplift"], np.float64),
        q[:, None], q[None, :], ck)
    envelope = (
        h + POSITIVE_RELIEF_BOUND_M
        + EROSION_TIME_MAX_MYR * np.maximum(uplift, 0.0)
    )
    exterior = _exterior_component(envelope < 0.0)
    return q, h, envelope, exterior


def _sample_material_tags(structure, q: np.ndarray) -> np.ndarray:
    """Nearest structural-cell tag quadrature on a shared world grid."""
    if not hasattr(structure, "_material_tag_samples"):
        raise ValueError("structural head has no material identity evidence")
    samples = np.asarray(structure._material_tag_samples)
    if samples.shape != (len(FINAL_SUBCELL_OFFSETS),
                         structure.n, structure.n):
        raise ValueError("structural material identity evidence is malformed")
    ck = structure.world_km / structure.n
    index = np.floor(np.asarray(q, np.float64) / ck).astype(np.int64)
    if np.any(index < 0) or np.any(index >= structure.n):
        raise ValueError("material identity query falls outside the atlas")
    return np.take(np.take(samples, index, axis=1), index, axis=2)


def _transport_identity_metrics(structure, common_tags: np.ndarray,
                                crop_y: np.ndarray, crop_x: np.ndarray,
                                land: np.ndarray, frozen_record: dict,
                                layout: dict) -> dict:
    """Measure identity ownership and capture after tectonic transport."""
    domain_by_id = {
        item["domain_id"]: item for item in layout["domains"]}
    expected_ids = frozen_record["member_domain_ids"]
    expected_labels = np.asarray(
        [domain_by_id[item]["label"] for item in expected_ids],
        np.int64)

    crop_tags = np.take(
        np.take(common_tags, crop_y, axis=1), crop_x, axis=2)
    land_samples = np.broadcast_to(land, crop_tags.shape)
    owned_land_samples = land_samples & np.isin(
        crop_tags, expected_labels)
    owner_fraction = float(
        np.count_nonzero(owned_land_samples)
        / max(np.count_nonzero(land_samples), 1))

    raw = np.asarray(structure._material_tag_samples)
    domain_count = len(layout["domains"])
    global_valid = raw[raw >= 0]
    global_counts = np.bincount(
        global_valid, minlength=domain_count).astype(np.int64)
    crop_counts = np.zeros(domain_count, np.int64)
    ck = structure.world_km / structure.n
    base = (np.arange(structure.n) + 0.5) * ck
    x0, y0 = frozen_record["origin_xy_km"]
    for sample_index, (oy, ox) in enumerate(FINAL_SUBCELL_OFFSETS):
        sample_y = base + oy * ck
        sample_x = base + ox * ck
        iy = np.flatnonzero((sample_y >= y0)
                            & (sample_y < y0 + FRAME_KM))
        ix = np.flatnonzero((sample_x >= x0)
                            & (sample_x < x0 + FRAME_KM))
        values = raw[sample_index][np.ix_(iy, ix)]
        valid = values[values >= 0]
        crop_counts += np.bincount(
            valid, minlength=domain_count).astype(np.int64)

    capture_by_id = {}
    for domain_id, label in zip(expected_ids, expected_labels):
        capture_by_id[domain_id] = float(
            crop_counts[label] / max(global_counts[label], 1))
    observed_labels = np.unique(crop_tags[land_samples & (crop_tags >= 0)])
    observed_land_ids = [
        layout["domains"][int(label)]["domain_id"]
        for label in observed_labels]
    observed_expected_ids = [
        domain_id for domain_id, label in zip(expected_ids, expected_labels)
        if crop_counts[label] > 0]
    return {
        "owner_fraction": owner_fraction,
        "capture_fraction": min(capture_by_id.values(), default=0.0),
        "capture_by_domain_id": capture_by_id,
        "observed_frozen_member_domain_ids": observed_expected_ids,
        "observed_land_domain_ids": observed_land_ids,
    }


def _evaluate_frozen_stage(structure, elevation, frozen: list[dict],
                           layout: dict):
    q, h_world, uplift_world = _common_fields(structure, elevation)
    common_tags = _sample_material_tags(structure, q)
    late_envelope = (
        h_world + POSITIVE_RELIEF_BOUND_M
        + EROSION_TIME_MAX_MYR * np.maximum(uplift_world, 0.0)
    )
    exterior = _exterior_component(late_envelope < 0.0)
    dense_q, dense_h_world, dense_envelope_world, dense_exterior = (
        _dense_world_fields(elevation, structure))
    results = []
    masks = {}

    for frozen_record in frozen:
        x0, y0 = frozen_record["origin_xy_km"]
        xs = np.flatnonzero((q >= x0) & (q < x0 + FRAME_KM))
        ys = np.flatnonzero((q >= y0) & (q < y0 + FRAME_KM))
        sub_h = h_world[np.ix_(ys, xs)]
        land = sub_h > 0.0
        masks[frozen_record["domain_id"]] = land
        land_fraction = float(land.mean())

        core_x = np.flatnonzero(
            (q[xs] >= x0 + CORE_INSET_KM)
            & (q[xs] < x0 + FRAME_KM - CORE_INSET_KM))
        core_y = np.flatnonzero(
            (q[ys] >= y0 + CORE_INSET_KM)
            & (q[ys] < y0 + FRAME_KM - CORE_INSET_KM))
        capacity = float(_smoothstep(
            sub_h[np.ix_(core_y, core_x)], -100.0, 300.0).mean())

        cy, cx, collar = _collar_indices(q, x0, y0)
        collar_envelope = late_envelope[np.ix_(cy, cx)]
        collar_exterior = exterior[np.ix_(cy, cx)]
        clearance = -float(np.max(collar_envelope[collar]))
        open_coverage = float(np.mean(collar_exterior[collar]))

        common_gate = evaluate_visible_border_contours(
            sub_h, float(q[1] - q[0]))
        dense_x = np.flatnonzero(
            (dense_q >= x0) & (dense_q < x0 + FRAME_KM))
        dense_y = np.flatnonzero(
            (dense_q >= y0) & (dense_q < y0 + FRAME_KM))
        dense_h = dense_h_world[np.ix_(dense_y, dense_x)]
        dcy, dcx, dense_collar = _collar_indices(dense_q, x0, y0)
        dense_envelope = dense_envelope_world[np.ix_(dcy, dcx)]
        dense_exterior_collar = dense_exterior[np.ix_(dcy, dcx)]
        dense_clearance = -float(np.max(dense_envelope[dense_collar]))
        dense_open_coverage = float(
            np.mean(dense_exterior_collar[dense_collar]))
        dense_gate = evaluate_visible_border_contours(
            dense_h, DENSE_HEAD_KM)
        components = _component_summary(land, float(q[1] - q[0]))
        identity = _transport_identity_metrics(
            structure, common_tags, ys, xs, land, frozen_record, layout)
        identity_exact = (
            identity["observed_frozen_member_domain_ids"]
            == frozen_record["member_domain_ids"])

        passed = bool(
            identity_exact
            and identity["owner_fraction"] >= MIN_OWNER_FRACTION
            and identity["capture_fraction"] >= MIN_CAPTURE_FRACTION
            and MIN_LAND_FRACTION <= land_fraction < MAX_LAND_FRACTION
            and MIN_SIGNIFICANT_COMPONENTS
            <= components["significant_count"]
            <= MAX_SIGNIFICANT_COMPONENTS
            and clearance >= REQUIRED_CLEARANCE_M
            and dense_clearance >= REQUIRED_CLEARANCE_M
            and open_coverage == 1.0
            and dense_open_coverage == 1.0
            and common_gate["passed"]
            and dense_gate["passed"]
        )
        results.append({
            "domain_id": frozen_record["domain_id"],
            "member_domain_ids": frozen_record["member_domain_ids"],
            "carrier_plate_ids": frozen_record["carrier_plate_ids"],
            "origin_xy_km": frozen_record["origin_xy_km"],
            "formation_owner_fraction": frozen_record["owner_fraction"],
            "formation_capture_fraction": frozen_record["capture_fraction"],
            "transport_identity_exact": identity_exact,
            "transport_owner_fraction": identity["owner_fraction"],
            "transport_capture_fraction": identity["capture_fraction"],
            "transport_capture_by_domain_id":
                identity["capture_by_domain_id"],
            "observed_frozen_member_domain_ids":
                identity["observed_frozen_member_domain_ids"],
            "observed_land_domain_ids": identity["observed_land_domain_ids"],
            "land_fraction": land_fraction,
            "land_capacity_score": capacity,
            "components": components,
            "common_clearance_m": clearance,
            "dense_clearance_m": dense_clearance,
            "common_exterior_ocean_coverage": open_coverage,
            "dense_exterior_ocean_coverage": dense_open_coverage,
            "verified_two_sided_water_collar_km": (
                EDGE_BAND_KM
                if (open_coverage == 1.0
                    and dense_open_coverage == 1.0)
                else 0.0),
            "common_contour_gate": {
                "passed": common_gate["passed"],
                "max_parallel_span_km":
                    common_gate["max_parallel_span_km"],
                "violations": common_gate["violations"],
            },
            "dense_contour_gate": {
                "passed": dense_gate["passed"],
                "max_parallel_span_km": dense_gate["max_parallel_span_km"],
                "violations": dense_gate["violations"],
            },
            "passed": passed,
        })
    return results, masks


def _resolution_comparison(previous_results, previous_masks,
                           results, masks):
    prior_by_id = {item["domain_id"]: item for item in previous_results}
    current_by_id = {item["domain_id"]: item for item in results}
    records = []
    for domain_id in prior_by_id:
        left = previous_masks[domain_id]
        right = masks[domain_id]
        union = int(np.count_nonzero(left | right))
        intersection = int(np.count_nonzero(left & right))
        land_iou = 1.0 if union == 0 else intersection / union
        land_delta = (current_by_id[domain_id]["land_fraction"]
                      - prior_by_id[domain_id]["land_fraction"])
        capacity_delta = (
            current_by_id[domain_id]["land_capacity_score"]
            - prior_by_id[domain_id]["land_capacity_score"])
        passed = bool(
            land_iou >= REQUIRED_LAND_IOU
            and abs(land_delta) <= MAX_RESOLUTION_DELTA
            and abs(capacity_delta) <= MAX_RESOLUTION_DELTA)
        records.append({
            "domain_id": domain_id,
            "land_iou": land_iou,
            "land_mask_xor_cells": int(np.count_nonzero(left ^ right)),
            "land_fraction_delta": land_delta,
            "capacity_delta": capacity_delta,
            "passed": passed,
        })
    return records


def _head_panel(structure, elevation, frozen, stage, out_path):
    image = Image.fromarray(_terrain_rgb(elevation["h"]), "RGB")
    image = image.resize((768, 768), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    scale = 768.0 / structure.world_km
    for index, record in enumerate(frozen):
        x0, y0 = record["origin_xy_km"]
        box = tuple(int(round(value * scale)) for value in (
            x0, y0, x0 + FRAME_KM, y0 + FRAME_KM))
        draw.rectangle(box, outline=(255, 255, 255), width=3)
        draw.text((box[0] + 4, box[1] + 4), str(index + 1),
                  fill=(255, 255, 255))
    draw.rectangle((0, 0, 220, 24), fill=(245, 245, 238))
    draw.text((5, 5), f"{stage}-km structural head", fill=(20, 20, 20))
    image.save(out_path)


# ------------------------------------------------------------------- run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--continental-budget", type=float, default=0.65)
    parser.add_argument(
        "--out", type=Path,
        default=Path("out") / "field_accretion_seed11",
    )
    args = parser.parse_args()
    if args.continental_budget != DESIGN_BUDGET:
        parser.error(
            "this frozen oracle is valid only at continental budget 0.65")
    if args.out.exists():
        if any(args.out.iterdir()):
            raise FileExistsError(
                f"oracle output directory must be empty: {args.out}")
    else:
        args.out.mkdir(parents=True)

    cfg = _atlas_config(args.continental_budget)
    layout = _formation_layout(args.seed, ATLAS_KM, int(cfg.plates))
    frozen, formation_records = _formation_crop_records(layout)

    precommit = {
        "experiment": "field-accretion-domain-oracle-v1",
        "written_before_any_structural_head": True,
        "source_fingerprint": _source_fingerprint(),
        "seed": args.seed,
        "seed_order": [11, 63, 77],
        "continental_budget": args.continental_budget,
        "constants": {
            "canonical_km": layout["canonical_km"],
            "assembly_wavelength_km": ASSEMBLY_WAVELENGTH_KM,
            "assembly_octaves": ASSEMBLY_OCTAVES,
            "craton_wavelength_km": CRATON_WAVELENGTH_KM,
            "craton_octaves": CRATON_OCTAVES,
            "carrier_threshold": CARRIER_THRESHOLD,
            "accretion_time_limit_km": _accretion_time_limit(
                args.continental_budget),
            "nucleus_craton_threshold": NUCLEUS_CRATON_THRESHOLD,
            "required_transport_identity_exact": True,
            "required_transport_owner_fraction": MIN_OWNER_FRACTION,
            "required_transport_capture_fraction": MIN_CAPTURE_FRACTION,
            "required_land_fraction_range": [
                MIN_LAND_FRACTION, MAX_LAND_FRACTION],
            "required_significant_component_range": [
                MIN_SIGNIFICANT_COMPONENTS, MAX_SIGNIFICANT_COMPONENTS],
            "required_clearance_m": REQUIRED_CLEARANCE_M,
            "required_common_exterior_ocean_coverage": 1.0,
            "required_dense_exterior_ocean_coverage": 1.0,
            "required_two_sided_water_collar_km": EDGE_BAND_KM,
            "required_land_iou": REQUIRED_LAND_IOU,
            "dense_head_km": DENSE_HEAD_KM,
        },
        "layout": {
            "carrier_count": len(layout["carriers"]),
            "raw_domain_count": layout["raw_domain_count"],
            "nucleated_domain_count": layout["nucleated_domain_count"],
            "carriers": layout["carriers"],
            "domains": layout["domains"],
        },
        "formation_candidate_count": len(formation_records),
        "formation_pass_count": sum(
            item["passed"] for item in formation_records),
        "frozen": frozen,
    }
    with (args.out / "frozen_domains.json").open(
            "x", encoding="utf-8") as manifest:
        manifest.write(json.dumps(precommit, indent=2))
    _layout_panel(layout, frozen, args.out / "formation_layout.png")

    report = dict(precommit)
    report["stages"] = {}
    report["resolution_comparisons"] = {}
    if len(frozen) != SHORTLIST_SIZE:
        report.update({
            "passed": False,
            "stop_stage": "formation_precommit",
            "reason": "fewer_than_three_nonoverlapping_formation_crops",
            "formation_candidates": formation_records,
        })
        (args.out / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    sample_continent = _continent_sampler(
        args.seed, layout, args.continental_budget)
    sample_material_tag = _material_tag_sampler(
        layout, args.continental_budget)
    sites = _plate_sites(args.seed, ATLAS_KM, int(cfg.plates))
    previous_results = previous_masks = None
    started = time.perf_counter()

    for stage_name, spacing in (
            ("120", SURVEY_KM), ("80", ORACLE_KM), ("40", 40.0)):
        stage_started = time.perf_counter()
        structure = build_structure(
            args.seed,
            cfg,
            _world_km=ATLAS_KM,
            _coarse_km=spacing,
            _continent_seeder=_continent_seeder(sample_continent),
            _partitioner=_partition_field_accretion,
            _initial_age_sampler=_initial_ocean_age,
            _plate_pivots=sites,
            _continent_sampler=sample_continent,
            _material_tag_sampler=sample_material_tag,
        )
        elevation = coarse_elevation(structure, cfg, args.seed)
        results, masks = _evaluate_frozen_stage(
            structure, elevation, frozen, layout)
        _head_panel(
            structure, elevation, frozen, stage_name,
            args.out / f"head_{stage_name}km.png")

        stage_passed = all(item["passed"] for item in results)
        stage_record = {
            "spacing_km": structure.world_km / structure.n,
            "n": structure.n,
            "elapsed_s": time.perf_counter() - stage_started,
            "results": results,
            "passed": stage_passed,
        }
        if previous_results is not None:
            comparison = _resolution_comparison(
                previous_results, previous_masks, results, masks)
            report["resolution_comparisons"][
                f"{previous_stage_name}_to_{stage_name}"] = comparison
            stage_passed = stage_passed and all(
                item["passed"] for item in comparison)
            stage_record["passed"] = stage_passed
        report["stages"][stage_name] = stage_record

        if not stage_passed:
            report.update({
                "passed": False,
                "stop_stage": f"{stage_name}km_head",
                "elapsed_s": time.perf_counter() - started,
            })
            break
        previous_results, previous_masks = results, masks
        previous_stage_name = stage_name
    else:
        report.update({
            "passed": True,
            "stop_stage": None,
            "elapsed_s": time.perf_counter() - started,
        })

    report["partition_diagnostic"] = FORMATION_DIAGNOSTICS.get(
        "partition")
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
