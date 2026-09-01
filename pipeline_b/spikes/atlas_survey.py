"""Large-world structural survey rank-stability experiment.

This is deliberately a spike, not a production crop selector.  It asks
one narrow question before the more expensive replay/local-process work:

    Can a 120-km structural atlas identify water-bordered 4096-km
    windows that remain good when the same atlas is solved at 80 km?

The experiment uses coordinate-addressed continental nuclei and a fixed
candidate lattice.  Candidate qualification includes the full late-control
positive-elevation envelope (fine relief, marine deposition, and 40 Myr of
rock uplift), so changing an erosion control would not move the crop.

It does *not* prove sparse replay equivalence or causal isolation from the
atlas rim.  Those are later experiments, and this module must not be wired
into the registry, adapter, or public engine API.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from engine import noise
from engine.elevation import coarse_elevation
from engine.rng import fnv1a64, stage_rng, stage_salt
from engine.surface import _bicubic
from engine.tectonics import Config, FRAME_KM, build_structure
from spikes.visible_contour_gate import evaluate_visible_border_contours


ATLAS_KM = 6.0 * FRAME_KM
SURVEY_KM = 120.0
ORACLE_KM = 80.0
CANDIDATE_STRIDE_KM = 128.0
EVALUATION_KM = 64.0
BASIN_RECALL_KM = 256.0

# This numerical guard exceeds the default maximum accumulated drift
# (2.2 * 45 km/era * 20 eras = 1980 km).  It is not a substitute for the
# trajectory-hull certificate required by a production replay system.
ATLAS_GUARD_KM = 2560.0

# A head-time certificate for all currently supported late controls.
EDGE_BAND_KM = 256.0
CORE_INSET_KM = 512.0
CONTOUR_CORRIDOR_KM = 640.0
MID_RELIEF_MAX_M = 170.0
FINE_RELIEF_MAX_M = 48.0
MARINE_DEPOSITION_MAX_M = 150.0
EROSION_TIME_MAX_MYR = 40.0
SELECTOR_WATER_CLEARANCE_M = 60.0
SHORTLIST_ORACLE_CLEARANCE_M = 160.0
MIN_LAND_FRACTION = 0.02
PARALLEL_SPAN_LIMIT_KM = 0.20 * FRAME_KM

# Survey shortlist acceptance.  Exact rank identity is reported too, but
# retaining at least 95% of the finer atlas's best land coverage is the
# useful architectural question.
SHORTLIST_SIZE = 3
MAX_ORACLE_REGRET = 0.02

# Private coupled-partition design certificate.  Plate sites must not move
# when the public continental-budget control changes, so their craton
# exclusion radius is sized once for the largest budget this atlas spike
# currently supports rather than from the configured value.
COUPLED_PARTITION_BUDGET_ENVELOPE = 0.65

# Fixed continuous-shape normalization.  The squared radial energy of the
# two angular harmonics plus the zero-mean fBm outline is about 0.6%; radius
# therefore scales by sqrt(1 / 1.006).  Crucially this is independent of the
# structural sampling lattice.
ANISOTROPIC_EXPECTED_AREA_SCALE = 0.997


@dataclass(frozen=True)
class Candidate:
    x0_km: float
    y0_km: float
    land_capacity_score: float
    land_fraction: float
    edge_envelope_max_m: float
    water_clearance_m: float
    max_parallel_span_km: float

    @property
    def origin(self) -> tuple[float, float]:
        return self.x0_km, self.y0_km


def _atlas_config(continental_budget: float) -> Config:
    """Preserve current plate/nucleus density on the enlarged atlas."""
    reference_world = FRAME_KM * (1.0 + 2.0 * Config().world_margin)
    area_ratio = (ATLAS_KM / reference_world) ** 2
    return Config(
        plates=max(4, int(round(Config().plates * area_ratio))),
        nuclei=max(2, int(round(Config().nuclei * area_ratio))),
        continental_budget=float(continental_budget),
    )


def _atlas_parent_provinces(seed: int, world: float, count: int):
    """One authority for stationary atlas province parents and weights."""
    rng = stage_rng(seed, "atlas-nuclei-v1")
    parent_count = max(3, int(round(count / 3.0)))
    parents = []
    for _ in range(parent_count):
        proposals = rng.uniform(0.0, world, (16, 2))
        if not parents:
            chosen = proposals[0]
        else:
            prior = np.asarray(parents)
            delta = np.abs(proposals[:, None, :] - prior[None, :, :])
            delta = np.minimum(delta, world - delta)
            distance2 = (delta ** 2).sum(axis=2)
            chosen = proposals[np.argmax(distance2.min(axis=1))]
        parents.append(chosen)
    parents = np.asarray(parents)
    weights = rng.uniform(0.72, 1.28, parent_count)
    weights /= weights.mean()
    return rng, parents, weights


def _seed_atlas_nuclei(seed, n, ck, cfg, label, plates):
    """Seed resolution-independent, world-addressed cratonic nuclei.

    Every centre, lobe offset, radius, and outline-noise salt is drawn
    independently of the raster.  Sampling at 120 or 80 km therefore
    observes the same continuous initial geometry.  Continental material
    belongs to the plate under each sampled point, as in the production
    seeder; no delivered-frame coordinates are available here.
    """
    rng = stage_rng(seed, "atlas-nuclei-v1")
    salt = stage_salt(seed, "atlas-nuclei-v1")
    world = n * ck
    xs = (np.arange(n) + 0.5) * ck
    X, Y = np.meshgrid(xs, xs)

    count = int(cfg.nuclei)

    # Cratons are not a homogeneous mist over the planet.  A small
    # number of mantle-scale assembly provinces collect several nuclei,
    # leaving broad ocean basins between them.  Wrapped proposal distance
    # makes the point process stationary instead of attracting provinces
    # to the numerical atlas rim.  The delivered crop is never wrapped.
    parent_count = max(3, int(round(count / 3.0)))
    parents = []
    for _ in range(parent_count):
        proposals = rng.uniform(0.0, world, (16, 2))
        if not parents:
            chosen = proposals[0]
        else:
            prior = np.asarray(parents)
            delta = np.abs(proposals[:, None, :] - prior[None, :, :])
            delta = np.minimum(delta, world - delta)
            distance2 = (delta ** 2).sum(axis=2)
            chosen = proposals[np.argmax(distance2.min(axis=1))]
        parents.append(chosen)
    parents = np.asarray(parents)

    weights = rng.uniform(0.72, 1.28, parent_count)
    weights /= weights.mean()

    # The public budget is a fraction of one delivered frame, not a
    # fraction of an arbitrarily enlarged computational atlas.  Allocate
    # one such material budget to each independent assembly province.
    # This retains the control's meaning while broad ocean basins occupy
    # the space between provinces.
    target_area_per_assembly = (
        float(cfg.continental_budget) * FRAME_KM ** 2
    )
    # Approximate union area of the main body and two overlapping lobes.
    # This global factor is not seed- or candidate-fitted.
    lobe_area_factor = 1.28
    base_radius = np.sqrt(
        target_area_per_assembly / (np.pi * lobe_area_factor)
    )

    for k, ((cy, cx), weight) in enumerate(zip(parents, weights)):
        radius = base_radius * np.sqrt(weight)
        lobe_specs = [(cy, cx, 1.0)]
        # Fixed-size draws: changing the raster spacing cannot change the
        # random stream or any later nucleus.
        for j, fraction in enumerate((0.52, 0.39), start=1):
            angle = rng.uniform(0.0, 2.0 * np.pi)
            offset = rng.uniform(0.35, 0.72) * radius
            lobe_specs.append((
                cy + offset * np.sin(angle),
                cx + offset * np.cos(angle),
                fraction,
            ))

        blob = np.zeros((n, n), bool)
        for j, (ly, lx, fraction) in enumerate(lobe_specs):
            lobe_radius = fraction * radius
            wobble = np.clip(
                noise.fbm(
                    X, Y, max(lobe_radius, 480.0), 5,
                    salt + 97 * k + 17 * j,
                ),
                -0.9, 0.9,
            )
            reach = lobe_radius * (1.0 + 0.38 * wobble)
            blob |= np.hypot(Y - ly, X - lx) < reach

        # A cratonic assembly is one coherent material domain at birth.
        # Clipping it to its host plate avoids the previous unphysical
        # initialization where one continent was pre-split across several
        # independently moving plates.  Later modeled tectonics can still
        # collide, subduct, and fragment it.
        iy = int(np.clip(np.floor(cy / ck), 0, n - 1))
        ix = int(np.clip(np.floor(cx / ck), 0, n - 1))
        host = int(label[iy, ix])
        selected = blob & (label == host)
        if selected.any():
            plates[host].cont |= selected


def _seed_atlas_nuclei_shared_scalene(seed, n, ck, cfg, label, plates):
    """Three-body province on one authoritative host plate.

    This experimental alternative preserves the legacy stationary parent
    process and each province's expected continental area.  It changes only
    the within-province geometry: three separately noised proto-cratons are
    arranged on a randomly rotated scalene triangle, then assigned together
    to the plate beneath the province parent.  Their shared kinematics rule
    out the previously observed multi-megametre independent-plate trains.

    No delivered-frame coordinate, crop candidate, contour measurement,
    retry, or target mask participates in formation.
    """
    rng = stage_rng(seed, "atlas-nuclei-v1")
    salt = stage_salt(seed, "atlas-nuclei-shared-scalene-v1")
    world = n * ck
    xs = (np.arange(n) + 0.5) * ck
    X, Y = np.meshgrid(xs, xs)

    count = int(cfg.nuclei)
    parent_count = max(3, int(round(count / 3.0)))
    parents = []
    for _ in range(parent_count):
        proposals = rng.uniform(0.0, world, (16, 2))
        if not parents:
            chosen = proposals[0]
        else:
            prior = np.asarray(parents)
            delta = np.abs(proposals[:, None, :] - prior[None, :, :])
            delta = np.minimum(delta, world - delta)
            distance2 = (delta ** 2).sum(axis=2)
            chosen = proposals[np.argmax(distance2.min(axis=1))]
        parents.append(chosen)
    parents = np.asarray(parents)

    weights = rng.uniform(0.72, 1.28, parent_count)
    weights /= weights.mean()
    target_area = float(cfg.continental_budget) * FRAME_KM ** 2

    # Fixed scalene skeleton in local (y, x) coordinates.  Random rotation
    # makes its orientation world-stationary rather than frame-aligned.  The
    # weighted recentering below keeps the material barycentre on the parent
    # even after the area shares are permuted.
    skeleton = np.array([
        (-0.62, -0.56),
        (-0.42, 0.72),
        (0.82, -0.08),
    ], np.float64)
    base_shares = np.array([0.40, 0.34, 0.26], np.float64)

    for province, ((cy, cx), weight) in enumerate(zip(parents, weights)):
        shares = base_shares[rng.permutation(3)]
        local = skeleton - np.sum(
            shares[:, None] * skeleton, axis=0, keepdims=True)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array(((cosine, -sine), (sine, cosine)))

        # The legacy main radius uses a 1.28 union-area allowance for its
        # overlapping lobes.  Here the exact analytic split is three disks;
        # summing pi*r_i^2 therefore preserves the same target area before
        # outline noise and the same one-host clipping applied by legacy.
        equivalent_radius = np.sqrt(target_area * weight / np.pi)
        legacy_main_radius = np.sqrt(target_area * weight / (np.pi * 1.28))
        centres = np.array((cy, cx))[None, :] + (
            local @ rotation.T) * (0.80 * legacy_main_radius)
        radii = equivalent_radius * np.sqrt(shares)

        province_blob = np.zeros((n, n), bool)
        for body, ((body_y, body_x), radius) in enumerate(
                zip(centres, radii)):
            wobble = np.clip(
                noise.fbm(
                    X, Y, max(float(radius), 480.0), 5,
                    salt + 101 * province + 29 * body,
                ),
                -0.9, 0.9,
            )
            reach = radius * (1.0 + 0.38 * wobble)
            province_blob |= np.hypot(Y - body_y, X - body_x) < reach

        iy = int(np.clip(np.floor(cy / ck), 0, n - 1))
        ix = int(np.clip(np.floor(cx / ck), 0, n - 1))
        host = int(label[iy, ix])
        selected = province_blob & (label == host)
        if selected.any():
            plates[host].cont |= selected


ATLAS_SEEDER_DIAGNOSTICS: dict[str, dict] = {}


def _seed_atlas_nuclei_anisotropic_impl(seed, n, ck, cfg, label, plates,
                                        diagnostic_key):
    """One-host, area-normalized anisotropic craton per province.

    Each legacy stationary province receives one randomly oriented ellipse
    with a moderate seed-drawn aspect ratio.  Independent world-coordinate
    outline noise and low-order angular lobing keep it from being a styled
    geometric oval.  One fixed, resolution-independent expected-area factor
    preserves the province's legacy target approximately before its natural
    clipping to the authoritative host plate.  Formation never reads a frame
    edge, crop, or contour result.
    """
    rng, parents, weights = _atlas_parent_provinces(
        seed, n * ck, int(cfg.nuclei))
    salt = stage_salt(seed, "atlas-nuclei-single-anisotropic-v1")
    world = n * ck
    xs = (np.arange(n) + 0.5) * ck
    X, Y = np.meshgrid(xs, xs)

    target_area_per_province = (
        float(cfg.continental_budget) * FRAME_KM ** 2
    )
    cell_area = ck ** 2
    province_records = []

    for province, ((cy, cx), weight) in enumerate(zip(parents, weights)):
        target_area = target_area_per_province * float(weight)
        equivalent_radius = np.sqrt(target_area / np.pi)
        aspect = rng.uniform(1.25, 1.65)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        phase2 = rng.uniform(0.0, 2.0 * np.pi)
        phase3 = rng.uniform(0.0, 2.0 * np.pi)
        cosine, sine = np.cos(angle), np.sin(angle)
        dx, dy = X - cx, Y - cy
        major_coordinate = cosine * dx + sine * dy
        minor_coordinate = -sine * dx + cosine * dy
        outline_noise = np.clip(
            noise.fbm(
                X, Y, max(float(equivalent_radius), 480.0), 5,
                salt + 101 * province,
            ),
            -0.9, 0.9,
        )

        def paint(scale):
            major = equivalent_radius * np.sqrt(aspect) * scale
            minor = equivalent_radius / np.sqrt(aspect) * scale
            u = major_coordinate / major
            v = minor_coordinate / minor
            theta = np.arctan2(v, u)
            radial_outline = (
                1.0
                + 0.18 * outline_noise
                + 0.085 * np.cos(3.0 * theta + phase3)
                + 0.045 * np.cos(2.0 * theta + phase2)
            )
            radial_outline = np.clip(radial_outline, 0.68, 1.32)
            return np.hypot(u, v) < radial_outline

        area_scale = ANISOTROPIC_EXPECTED_AREA_SCALE
        blob = paint(area_scale)

        iy = int(np.clip(np.floor(cy / ck), 0, n - 1))
        ix = int(np.clip(np.floor(cx / ck), 0, n - 1))
        host = int(label[iy, ix])
        selected = blob & (label == host)
        new_selected = selected & ~plates[host].cont
        plates[host].cont |= selected

        raw_area = float(blob.sum()) * cell_area
        selected_area = float(selected.sum()) * cell_area
        new_selected_area = float(new_selected.sum()) * cell_area
        province_records.append({
            "province": int(province),
            "host_plate": host,
            "target_area_km2": target_area,
            "raw_area_km2": raw_area,
            "selected_area_km2": selected_area,
            "new_selected_area_km2": new_selected_area,
            "raw_area_error_fraction":
                (raw_area - target_area) / target_area,
            "host_selected_loss_fraction":
                (target_area - selected_area) / target_area,
            "unique_selected_loss_fraction":
                (target_area - new_selected_area) / target_area,
            "aspect_ratio": float(aspect),
            "orientation_radians": float(angle),
            "fixed_expected_area_scale": float(area_scale),
        })

    target_total = sum(item["target_area_km2"]
                       for item in province_records)
    raw_total = sum(item["raw_area_km2"] for item in province_records)
    selected_total = sum(item["selected_area_km2"]
                         for item in province_records)
    unique_total = sum(item["new_selected_area_km2"]
                       for item in province_records)
    ATLAS_SEEDER_DIAGNOSTICS[diagnostic_key] = {
        "province_count": len(province_records),
        "target_area_km2": target_total,
        "raw_area_km2": raw_total,
        "host_selected_area_km2": selected_total,
        "unique_selected_area_km2": unique_total,
        "raw_area_error_fraction":
            (raw_total - target_total) / target_total,
        "host_selected_loss_fraction":
            (target_total - selected_total) / target_total,
        "unique_selected_loss_fraction":
            (target_total - unique_total) / target_total,
        "provinces": province_records,
    }


def _seed_atlas_nuclei_single_anisotropic(seed, n, ck, cfg, label,
                                          plates):
    _seed_atlas_nuclei_anisotropic_impl(
        seed, n, ck, cfg, label, plates, "single-anisotropic")


def _seed_atlas_nuclei_coupled_anisotropic(seed, n, ck, cfg, label,
                                           plates):
    _seed_atlas_nuclei_anisotropic_impl(
        seed, n, ck, cfg, label, plates, "coupled-anisotropic")


ATLAS_PARTITION_DIAGNOSTICS: dict[str, dict] = {}


def _partition_atlas_coupled_anisotropic(seed, n, ck, cfg):
    """Warped Voronoi partition with one site at every craton parent.

    The remaining sites come from one fixed world-wide proposal set.  Sites
    inside a radius derived solely from each province's nominal cratonic area
    are ineligible; deterministic farthest-point selection then spreads the
    oceanic sites through the remaining world.  There are no rejection
    retries and no delivered-frame or candidate information.
    """
    world = n * ck
    _, parents, weights = _atlas_parent_provinces(
        seed, world, int(cfg.nuclei))
    parent_count = parents.shape[0]
    if parent_count > cfg.plates:
        raise ValueError("more craton parents than experimental plates")

    remaining = int(cfg.plates - parent_count)
    rng = stage_rng(seed, "atlas-coupled-ocean-sites-v1")
    proposal_count = max(4096, 128 * int(cfg.plates))
    proposals = rng.uniform(0.02 * world, 0.98 * world,
                            (proposal_count, 2))
    target_area = COUPLED_PARTITION_BUDGET_ENVELOPE * FRAME_KM ** 2
    equivalent_radii = np.sqrt(target_area * weights / np.pi)
    exclusion_radii = 1.90 * equivalent_radii
    distances = np.linalg.norm(
        proposals[:, None, :] - parents[None, :, :], axis=2)
    valid = np.all(distances >= exclusion_radii[None, :], axis=1)
    ocean_candidates = proposals[valid]
    if ocean_candidates.shape[0] < remaining:
        raise ValueError("fixed coupled-partition proposal set is exhausted")

    selected_sites = [point.copy() for point in parents]
    min_distance2 = np.min(
        ((ocean_candidates[:, None, :] - parents[None, :, :]) ** 2)
        .sum(axis=2),
        axis=1,
    )
    available = np.ones(ocean_candidates.shape[0], bool)
    for _ in range(remaining):
        scores = np.where(available, min_distance2, -1.0)
        chosen = int(np.argmax(scores))
        site = ocean_candidates[chosen]
        selected_sites.append(site.copy())
        available[chosen] = False
        distance2 = ((ocean_candidates - site) ** 2).sum(axis=1)
        min_distance2 = np.minimum(min_distance2, distance2)
    sites = np.asarray(selected_sites, np.float64)

    salt = stage_salt(seed, "atlas-coupled-partition-v1")
    xs = (np.arange(n) + 0.5) * ck
    X, Y = np.meshgrid(xs, xs)
    best = np.full((n, n), np.inf)
    label = np.zeros((n, n), np.int32)
    for plate, (site_y, site_x) in enumerate(sites):
        distance = np.hypot(Y - site_y, X - site_x)
        warp = 1.0 + 0.5 * np.clip(
            noise.fbm(X, Y, world / 3.5, 5, salt + plate),
            -0.9, 0.9,
        )
        cost = distance * warp
        take = cost < best
        best[take] = cost[take]
        label[take] = plate

    ATLAS_PARTITION_DIAGNOSTICS["coupled-anisotropic"] = {
        "parent_site_count": int(parent_count),
        "ocean_site_count": remaining,
        "proposal_count": proposal_count,
        "eligible_ocean_proposals": int(ocean_candidates.shape[0]),
        "continental_budget_design_envelope":
            COUPLED_PARTITION_BUDGET_ENVELOPE,
        "exclusion_radius_km": {
            "min": float(exclusion_radii.min()),
            "median": float(np.median(exclusion_radii)),
            "max": float(exclusion_radii.max()),
        },
        "sites_yx_km": sites.tolist(),
    }
    return label


ATLAS_SEEDERS = {
    "legacy": _seed_atlas_nuclei,
    "shared-scalene": _seed_atlas_nuclei_shared_scalene,
    "single-anisotropic": _seed_atlas_nuclei_single_anisotropic,
    "coupled-anisotropic": _seed_atlas_nuclei_coupled_anisotropic,
}

ATLAS_PARTITIONERS = {
    "legacy": None,
    "shared-scalene": None,
    "single-anisotropic": None,
    "coupled-anisotropic": _partition_atlas_coupled_anisotropic,
}


def _candidate_origins() -> list[tuple[float, float]]:
    available = ATLAS_KM - 2.0 * ATLAS_GUARD_KM - FRAME_KM
    steps = int(np.floor(available / CANDIDATE_STRIDE_KM))
    residual = available - steps * CANDIDATE_STRIDE_KM
    first = ATLAS_GUARD_KM + 0.5 * residual
    axis = first + CANDIDATE_STRIDE_KM * np.arange(steps + 1)
    return [(float(x), float(y)) for y in axis for x in axis]


def _coast_mask(land: np.ndarray) -> np.ndarray:
    padded = np.pad(land, 1, mode="constant", constant_values=False)
    interior = padded[1:-1, 1:-1]
    enclosed = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return interior & ~enclosed


def _component_max_span(mask: np.ndarray, tangential_axis: int,
                        ck: float) -> float:
    """Maximum tangential span of an 8-connected raster component."""
    todo = mask.copy()
    best = 0.0
    height, width = todo.shape
    while todo.any():
        sy, sx = np.argwhere(todo)[0]
        todo[sy, sx] = False
        stack = [(int(sy), int(sx))]
        lo = hi = int(sx if tangential_axis == 1 else sy)
        while stack:
            y, x = stack.pop()
            coord = x if tangential_axis == 1 else y
            lo = min(lo, coord)
            hi = max(hi, coord)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < height and 0 <= xx < width
                            and todo[yy, xx]):
                        todo[yy, xx] = False
                        stack.append((yy, xx))
        best = max(best, (hi - lo + 1) * ck)
    return best


def _parallel_coast_span(land: np.ndarray, ck: float) -> float:
    """Coarse proxy for a long coastline paralleling a crop edge."""
    coast = _coast_mask(land)
    rows, cols = land.shape
    band = max(1, int(np.ceil(CONTOUR_CORRIDOR_KM / ck)))
    masks = (
        (coast & (np.arange(rows)[:, None] < band), 1),
        (coast & (np.arange(rows)[:, None] >= rows - band), 1),
        (coast & (np.arange(cols)[None, :] < band), 0),
        (coast & (np.arange(cols)[None, :] >= cols - band), 0),
    )
    return max(_component_max_span(mask, axis, ck)
               for mask, axis in masks)


def _land_component_diagnostic(h: np.ndarray, q: np.ndarray) -> list[dict]:
    """Largest emergent components on the shared physical lattice."""
    todo = h > 0.0
    height, width = todo.shape
    components = []
    while todo.any():
        sy, sx = np.argwhere(todo)[0]
        todo[sy, sx] = False
        stack = [(int(sy), int(sx))]
        count = 0
        min_y = max_y = int(sy)
        min_x = max_x = int(sx)
        while stack:
            y, x = stack.pop()
            count += 1
            min_y, max_y = min(min_y, y), max(max_y, y)
            min_x, max_x = min(min_x, x), max(max_x, x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < height and 0 <= xx < width
                            and todo[yy, xx]):
                        todo[yy, xx] = False
                        stack.append((yy, xx))
        components.append({
            "area_fraction_of_frame":
                count * EVALUATION_KM ** 2 / FRAME_KM ** 2,
            "width_km": (max_x - min_x + 1) * EVALUATION_KM,
            "height_km": (max_y - min_y + 1) * EVALUATION_KM,
            "bbox_km": [
                float(q[min_x] - 0.5 * EVALUATION_KM),
                float(q[min_y] - 0.5 * EVALUATION_KM),
                float(q[max_x] + 0.5 * EVALUATION_KM),
                float(q[max_y] + 0.5 * EVALUATION_KM),
            ],
        })
    components.sort(key=lambda item: -item["area_fraction_of_frame"])
    return components[:12]


def _land_component_count(h: np.ndarray) -> int:
    """Eight-connected emergent component count for crop diagnostics."""
    todo = np.asarray(h) > 0.0
    height, width = todo.shape
    count = 0
    while todo.any():
        sy, sx = np.argwhere(todo)[0]
        todo[sy, sx] = False
        stack = [(int(sy), int(sx))]
        count += 1
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    yy, xx = y + dy, x + dx
                    if (0 <= yy < height and 0 <= xx < width
                            and todo[yy, xx]):
                        todo[yy, xx] = False
                        stack.append((yy, xx))
    return count


def _smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((value - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _common_fields(structure, elevation) -> tuple[np.ndarray, ...]:
    """Sample every atlas resolution on one physical quadrature grid."""
    ck = structure.world_km / structure.n
    count = int(round(structure.world_km / EVALUATION_KM))
    q = (np.arange(count) + 0.5) * (structure.world_km / count)
    y = q[:, None]
    x = q[None, :]
    h = _bicubic(np.asarray(elevation["h"], np.float64), y, x, ck)
    uplift = _bicubic(
        np.asarray(elevation["uplift"], np.float64), y, x, ck)
    return q, h, np.maximum(uplift, 0.0)


def _selector_fields(structure, reference_elevation,
                     envelope_elevations) -> tuple[np.ndarray, ...]:
    q, h_reference, _ = _common_fields(structure, reference_elevation)
    envelope = np.full_like(h_reference, -np.inf)
    native_envelope = np.full_like(
        np.asarray(reference_elevation["h"], np.float64), -np.inf)
    positive_bound = (
        MID_RELIEF_MAX_M + FINE_RELIEF_MAX_M
        + MARINE_DEPOSITION_MAX_M
    )
    for elevation in envelope_elevations:
        _, h, uplift = _common_fields(structure, elevation)
        envelope = np.maximum(
            envelope,
            h + positive_bound + EROSION_TIME_MAX_MYR * uplift,
        )
        native_h = np.asarray(elevation["h"], np.float64)
        native_uplift = np.maximum(
            np.asarray(elevation["uplift"], np.float64), 0.0)
        native_envelope = np.maximum(
            native_envelope,
            native_h + positive_bound
            + EROSION_TIME_MAX_MYR * native_uplift,
        )
    return q, h_reference, envelope, native_envelope


def _tie_key(seed: int, candidate: Candidate) -> int:
    return fnv1a64(
        f"atlas-candidate-v1:{seed}:{candidate.x0_km:.0f}:"
        f"{candidate.y0_km:.0f}"
    )


def _shortlist(candidates: list[Candidate], seed: int) -> list[Candidate]:
    """Three high-capacity, non-overlapping survey regions."""
    ranked = sorted(candidates, key=lambda c: (
        -c.land_capacity_score,
        _tie_key(seed, c),
    ))
    chosen: list[Candidate] = []
    for candidate in ranked:
        if all(max(abs(candidate.x0_km - prior.x0_km),
                   abs(candidate.y0_km - prior.y0_km))
               >= 0.5 * FRAME_KM for prior in chosen):
            chosen.append(candidate)
            if len(chosen) == SHORTLIST_SIZE:
                break
    return chosen


def _evaluate_candidates(structure, reference_elevation,
                         envelope_elevations, seed: int) -> dict:
    ck = structure.world_km / structure.n
    q, h, late_envelope, native_envelope = _selector_fields(
        structure, reference_elevation, envelope_elevations)
    native_q = (np.arange(structure.n) + 0.5) * ck

    safe: list[Candidate] = []
    contour_eligible: list[Candidate] = []
    contour_gate_by_origin = {}
    rejected_water = 0
    rejected_parallel = 0
    rejected_visible_contour = 0
    rejected_empty = 0
    contour_violation_kinds: dict[str, int] = {}
    common_edge_maxima = []
    source_edge_maxima = []
    hard_edge_maxima = []
    edge_origins = []
    common_water_pass = 0
    common_water_and_land_pass = 0
    native_water_pass = 0
    for x0, y0 in _candidate_origins():
        xs = np.flatnonzero((q >= x0) & (q < x0 + FRAME_KM))
        ys = np.flatnonzero((q >= y0) & (q < y0 + FRAME_KM))
        if xs.size < 2 or ys.size < 2:
            continue
        sub_h = h[np.ix_(ys, xs)]
        # Two-sided physical moat: potential land is excluded both just
        # inside and just outside the delivered boundary.  This cuts a
        # window through existing open ocean; it never edits or tapers a
        # field toward the rectangular crop.
        collar_x = np.flatnonzero(
            (q >= x0 - EDGE_BAND_KM)
            & (q < x0 + FRAME_KM + EDGE_BAND_KM))
        collar_y = np.flatnonzero(
            (q >= y0 - EDGE_BAND_KM)
            & (q < y0 + FRAME_KM + EDGE_BAND_KM))
        x_rel = q[collar_x] - x0
        y_rel = q[collar_y] - y0
        collar = (
            (x_rel[None, :] < EDGE_BAND_KM)
            | (x_rel[None, :] >= FRAME_KM - EDGE_BAND_KM)
            | (y_rel[:, None] < EDGE_BAND_KM)
            | (y_rel[:, None] >= FRAME_KM - EDGE_BAND_KM)
        )
        collar_env = late_envelope[np.ix_(collar_y, collar_x)]
        common_edge_max = float(np.max(collar_env[collar]))
        if common_edge_max <= -SELECTOR_WATER_CLEARANCE_M:
            common_water_pass += 1
            if float((sub_h > 0.0).mean()) >= MIN_LAND_FRACTION:
                common_water_and_land_pass += 1

        # Bicubic output is clamped to its source-cell extrema.  Include
        # every native source cell whose interpolation support can touch
        # the physical moat, so the 64-km quadrature cannot miss a peak.
        support = EDGE_BAND_KM + 2.0 * ck
        native_x = np.flatnonzero(
            (native_q >= x0 - support)
            & (native_q < x0 + FRAME_KM + support))
        native_y = np.flatnonzero(
            (native_q >= y0 - support)
            & (native_q < y0 + FRAME_KM + support))
        nx_rel = native_q[native_x] - x0
        ny_rel = native_q[native_y] - y0
        native_collar = (
            (nx_rel[None, :] < support)
            | (nx_rel[None, :] >= FRAME_KM - support)
            | (ny_rel[:, None] < support)
            | (ny_rel[:, None] >= FRAME_KM - support)
        )
        source_env = native_envelope[np.ix_(native_y, native_x)]
        source_edge_max = float(np.max(source_env[native_collar]))
        if source_edge_max <= -SELECTOR_WATER_CLEARANCE_M:
            native_water_pass += 1
        # The common physical lattice is the actual survey gate.  The
        # broader native-source maximum is diagnostic only: _bicubic is
        # clamped to its four bracketing values, so treating every raw
        # cubic-support cell as attainable would enlarge the moat and
        # reject valid candidates.  Exact 40-km replay is the final proof.
        edge_max = common_edge_max
        common_edge_maxima.append(common_edge_max)
        source_edge_maxima.append(source_edge_max)
        hard_edge_maxima.append(edge_max)
        edge_origins.append((x0, y0))
        if edge_max > -SELECTOR_WATER_CLEARANCE_M:
            rejected_water += 1
            continue

        land = sub_h > 0.0
        land_fraction = float(land.mean())
        if land_fraction < MIN_LAND_FRACTION:
            rejected_empty += 1
            continue

        # Retain the original binary-land coastline proxy as a legacy
        # diagnostic.  It is neither an eligibility decision nor a rank
        # score; the component-aware visible-level gate below is the hard
        # eligibility test requested by this experiment.
        parallel_span = _parallel_coast_span(land, EVALUATION_KM)
        if parallel_span >= PARALLEL_SPAN_LIMIT_KM:
            rejected_parallel += 1

        core_x = np.flatnonzero(
            (q[xs] >= x0 + CORE_INSET_KM)
            & (q[xs] < x0 + FRAME_KM - CORE_INSET_KM))
        core_y = np.flatnonzero(
            (q[ys] >= y0 + CORE_INSET_KM)
            & (q[ys] < y0 + FRAME_KM - CORE_INSET_KM))
        core_h = sub_h[np.ix_(core_y, core_x)]
        capacity = float(_smoothstep(core_h, -100.0, 300.0).mean())

        candidate = Candidate(
            x0_km=x0,
            y0_km=y0,
            land_capacity_score=capacity,
            land_fraction=land_fraction,
            edge_envelope_max_m=edge_max,
            water_clearance_m=-edge_max,
            max_parallel_span_km=parallel_span,
        )
        safe.append(candidate)

        # Eligibility only: examine the naturally sampled common-grid crop
        # without changing it.  A passing candidate later competes solely on
        # the existing land-capacity key; no contour score enters ranking.
        contour_gate = evaluate_visible_border_contours(
            sub_h, EVALUATION_KM)
        contour_gate_by_origin[candidate.origin] = {
            "passed": contour_gate["passed"],
            "max_parallel_span_km":
                contour_gate["max_parallel_span_km"],
            "longest_runs": contour_gate["longest_runs"][:3],
            "violations": contour_gate["violations"],
        }
        if contour_gate["passed"]:
            contour_eligible.append(candidate)
        else:
            rejected_visible_contour += 1
            for violation in contour_gate["violations"]:
                kind = str(violation["kind"])
                contour_violation_kinds[kind] = (
                    contour_violation_kinds.get(kind, 0) + 1)

    safe.sort(key=lambda c: (-c.land_capacity_score, _tie_key(seed, c)))
    contour_eligible.sort(
        key=lambda c: (-c.land_capacity_score, _tie_key(seed, c)))
    hard_maxima = np.asarray(hard_edge_maxima, np.float64)
    best_edge_index = int(np.argmin(hard_maxima)) if hard_maxima.size else None
    return {
        "safe": safe,
        "shortlist": _shortlist(safe, seed),
        "contour_eligible": contour_eligible,
        "contour_shortlist": _shortlist(contour_eligible, seed),
        "contour_gate_by_origin": contour_gate_by_origin,
        "contour_gate_evaluated": len(safe),
        "contour_gate_passed": len(contour_eligible),
        "rejected_visible_contour": rejected_visible_contour,
        "contour_violation_kinds": contour_violation_kinds,
        "rejected_water": rejected_water,
        "rejected_parallel": rejected_parallel,
        "rejected_empty": rejected_empty,
        "total": len(_candidate_origins()),
        "largest_land_components": _land_component_diagnostic(h, q),
        "edge_envelope_diagnostic": {
            "min_m": None if not hard_maxima.size else float(
                hard_maxima.min()),
            "p10_m": None if not hard_maxima.size else float(
                np.percentile(hard_maxima, 10.0)),
            "median_m": None if not hard_maxima.size else float(
                np.median(hard_maxima)),
            "max_m": None if not hard_maxima.size else float(
                hard_maxima.max()),
            "best_origin_km": None if best_edge_index is None else
                list(edge_origins[best_edge_index]),
            "best_common_sample_m": None if best_edge_index is None else
                float(common_edge_maxima[best_edge_index]),
            "best_native_support_m": None if best_edge_index is None else
                float(source_edge_maxima[best_edge_index]),
            "common_water_pass": common_water_pass,
            "common_water_and_land_pass": common_water_and_land_pass,
            "native_water_pass": native_water_pass,
        },
    }


def _candidate_lookup(candidates: list[Candidate]) -> dict:
    return {candidate.origin: candidate for candidate in candidates}


def _rank_comparison(survey_result: dict, oracle_result: dict,
                     *, contour_eligible: bool = False) -> dict:
    if contour_eligible:
        survey_top = survey_result["contour_shortlist"]
        oracle_safe = oracle_result["contour_eligible"]
    else:
        survey_top = survey_result["shortlist"]
        oracle_safe = oracle_result["safe"]
    oracle_best = oracle_safe[0] if oracle_safe else None
    oracle_by_origin = _candidate_lookup(oracle_safe)
    surviving = [oracle_by_origin[c.origin] for c in survey_top
                 if c.origin in oracle_by_origin]

    if oracle_best is None:
        # There is no finer-resolution target against which rank regret or
        # basin recall can be defined.  Keep these values explicitly N/A;
        # a numeric/False sentinel would look like a measured result.
        regret = None
        exact_best = None
        basin_recalled = None
    else:
        exact_best = oracle_best.origin in {c.origin for c in survey_top}
        basin_recalled = any(
            max(abs(candidate.x0_km - oracle_best.x0_km),
                abs(candidate.y0_km - oracle_best.y0_km))
            <= BASIN_RECALL_KM
            for candidate in survey_top
        )
        regret = (None if not surviving else
                  oracle_best.land_capacity_score
                  - max(c.land_capacity_score for c in surviving))

    oracle_clear = [candidate.water_clearance_m for candidate in surviving]

    passed = (
        len(survey_top) == SHORTLIST_SIZE
        and oracle_best is not None
        and len(surviving) == SHORTLIST_SIZE
        and all(value >= SHORTLIST_ORACLE_CLEARANCE_M
                for value in oracle_clear)
        and basin_recalled is True
        and regret is not None
        and regret <= MAX_ORACLE_REGRET
    )
    return {
        "contour_eligibility_filter": contour_eligible,
        "passed": passed,
        "survey_top": [asdict(c) for c in survey_top],
        "oracle_best": None if oracle_best is None else asdict(oracle_best),
        "oracle_membership_pool": (
            "contour_eligible" if contour_eligible else "water_safe"),
        "survey_top_surviving_at_oracle": len(surviving),
        "survey_top_oracle_clearance_m": oracle_clear,
        "oracle_regret_from_shortlist": regret,
        "exact_oracle_best_in_shortlist": exact_best,
        "oracle_best_basin_recalled": basin_recalled,
    }


def _terrain_rgb(h: np.ndarray) -> np.ndarray:
    cuts = np.array([
        -6000.0, -3000.0, -1200.0, -300.0, 0.0,
        350.0, 1000.0, 2200.0, 4200.0, 9000.0,
    ])
    colors = np.array([
        [13, 36, 77], [20, 61, 112], [39, 103, 151],
        [91, 168, 195], [166, 190, 137], [126, 158, 95],
        [156, 126, 82], [116, 78, 61], [79, 54, 57],
        [235, 238, 232],
    ], dtype=np.uint8)
    return colors[np.searchsorted(cuts, h, side="right").clip(0, len(colors)-1)]


def _diagnostic_panel(seed: int, survey_structure, survey_elevation,
                      oracle_structure, oracle_elevation, comparison: dict,
                      out_path: Path) -> None:
    size = 640

    def panel(structure, elevation):
        image = Image.fromarray(_terrain_rgb(elevation["h"]), "RGB")
        image = image.resize((size, size), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)

        def rect(candidate, color, width):
            scale = size / structure.world_km
            x0 = int(round(candidate["x0_km"] * scale))
            y0 = int(round(candidate["y0_km"] * scale))
            x1 = int(round((candidate["x0_km"] + FRAME_KM) * scale))
            y1 = int(round((candidate["y0_km"] + FRAME_KM) * scale))
            draw.rectangle((x0, y0, x1, y1), outline=color, width=width)

        for candidate in comparison["survey_top"]:
            rect(candidate, (95, 255, 110), 3)
        if comparison["oracle_best"] is not None:
            rect(comparison["oracle_best"], (255, 83, 205), 3)
        return image

    left = panel(survey_structure, survey_elevation)
    right = panel(oracle_structure, oracle_elevation)
    canvas = Image.new("RGB", (2 * size, size + 38), (244, 244, 240))
    canvas.paste(left, (0, 38))
    canvas.paste(right, (size, 38))
    draw = ImageDraw.Draw(canvas)
    status = "PASS" if comparison["passed"] else "FAIL"
    draw.text((8, 10), f"seed {seed} | 120-km survey", fill=(20, 20, 20))
    shortlist_count = len(comparison["survey_top"])
    draw.text(
        (size + 8, 10),
        f"80-km oracle | {status} | green=survey shortlist "
        f"({shortlist_count}), magenta=oracle best",
        fill=(20, 20, 20),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def run_seed(seed: int, continental_budget: float, out_dir: Path,
             *, contour_filter: bool = False,
             seeder_variant: str = "legacy") -> dict:
    try:
        continent_seeder = ATLAS_SEEDERS[seeder_variant]
        partitioner = ATLAS_PARTITIONERS[seeder_variant]
    except KeyError as exc:
        raise ValueError(f"unknown atlas seeder {seeder_variant!r}") from exc
    cfg = _atlas_config(continental_budget)
    built = {}
    for label, spacing in (("survey", SURVEY_KM), ("oracle", ORACLE_KM)):
        started = time.perf_counter()
        structure = build_structure(
            seed,
            cfg,
            _world_km=ATLAS_KM,
            _coarse_km=spacing,
            _continent_seeder=continent_seeder,
            _partitioner=partitioner,
        )
        elevation = coarse_elevation(structure, cfg, seed)
        # Early surface controls belong to the rebuilt head and may
        # legitimately produce a new crop.  The selector must remain
        # fixed across late controls, so this one configured surface is
        # certified against their full positive-elevation envelope.
        envelope_elevations = [elevation]
        candidates = _evaluate_candidates(
            structure, elevation, envelope_elevations, seed)
        built[label] = {
            "structure": structure,
            "elevation": elevation,
            "candidates": candidates,
            "seeder_diagnostic": deepcopy(
                ATLAS_SEEDER_DIAGNOSTICS.get(seeder_variant)),
            "partition_diagnostic": deepcopy(
                ATLAS_PARTITION_DIAGNOSTICS.get(seeder_variant)),
            "elapsed_s": time.perf_counter() - started,
        }

    legacy_comparison = _rank_comparison(
        built["survey"]["candidates"], built["oracle"]["candidates"])
    contour_comparison = _rank_comparison(
        built["survey"]["candidates"], built["oracle"]["candidates"],
        contour_eligible=True)
    comparison = (contour_comparison if contour_filter
                  else legacy_comparison)
    _diagnostic_panel(
        seed,
        built["survey"]["structure"], built["survey"]["elevation"],
        built["oracle"]["structure"], built["oracle"]["elevation"],
        comparison,
        out_dir / f"atlas_survey_seed{seed}.png",
    )

    def candidate_summary(which):
        result = built[which]["candidates"]
        eligible = result["contour_eligible"]
        structure = built[which]["structure"]
        cell_area = (structure.world_km / structure.n) ** 2
        plate_areas = np.bincount(
            structure.initial_label.ravel(),
            minlength=structure.alive_plates,
        ).astype(np.float64) * cell_area
        parent_count = (built[which]["partition_diagnostic"] or {}).get(
            "parent_site_count", 0)
        plate_size_diagnostic = {
            "min_km2": float(plate_areas.min()),
            "p10_km2": float(np.percentile(plate_areas, 10.0)),
            "median_km2": float(np.median(plate_areas)),
            "mean_km2": float(plate_areas.mean()),
            "p90_km2": float(np.percentile(plate_areas, 90.0)),
            "max_km2": float(plate_areas.max()),
            "coefficient_of_variation": float(
                plate_areas.std() / plate_areas.mean()),
            "max_to_min": float(
                plate_areas.max() / max(plate_areas.min(), cell_area)),
            "parent_plate_mean_km2": None if not parent_count else float(
                plate_areas[:parent_count].mean()),
            "ocean_plate_mean_km2": None if not parent_count else float(
                plate_areas[parent_count:].mean()),
        }

        def eligible_record(candidate):
            record = asdict(candidate)
            record["visible_contour_gate"] = (
                result["contour_gate_by_origin"][candidate.origin])
            return record

        return {
            "spacing_km": SURVEY_KM if which == "survey" else ORACLE_KM,
            "n": built[which]["structure"].n,
            "elapsed_s": built[which]["elapsed_s"],
            "seeder_diagnostic": built[which]["seeder_diagnostic"],
            "partition_diagnostic":
                built[which]["partition_diagnostic"],
            "plate_size_diagnostic": plate_size_diagnostic,
            "safe_candidates": len(result["safe"]),
            "visible_contour_gate_evaluated":
                result["contour_gate_evaluated"],
            "visible_contour_gate_passed":
                result["contour_gate_passed"],
            "visible_contour_gate_rejected":
                result["rejected_visible_contour"],
            "visible_contour_violation_kinds":
                result["contour_violation_kinds"],
            "rejected_water": result["rejected_water"],
            "parallel_tripwire_candidates": result["rejected_parallel"],
            "rejected_empty": result["rejected_empty"],
            "total_candidates": result["total"],
            "edge_envelope_diagnostic":
                result["edge_envelope_diagnostic"],
            "largest_land_components": result["largest_land_components"],
            "best": None if not result["safe"] else asdict(result["safe"][0]),
            "eligible_best": None if not eligible else
                eligible_record(eligible[0]),
            "eligible_top": [eligible_record(candidate)
                             for candidate in eligible[:SHORTLIST_SIZE]],
            "eligible_shortlist": [eligible_record(candidate)
                                   for candidate
                                   in result["contour_shortlist"]],
        }

    return {
        "seed": seed,
        "config": {
            "plates": cfg.plates,
            "nuclei": cfg.nuclei,
            "continental_budget": cfg.continental_budget,
            "contour_filter_applied": contour_filter,
            "seeder_variant": seeder_variant,
        },
        "survey": candidate_summary("survey"),
        "oracle": candidate_summary("oracle"),
        "comparison": comparison,
        "legacy_comparison": legacy_comparison,
        "contour_eligible_comparison": contour_comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 63, 77])
    parser.add_argument("--continental-budget", type=float, default=0.30)
    parser.add_argument(
        "--contour-filter", action="store_true",
        help=("hard-filter water-safe common-grid crops by the visible "
              "contour gate before the existing land-capacity ranking"),
    )
    parser.add_argument(
        "--seeder", choices=tuple(ATLAS_SEEDERS), default="legacy",
        help="experimental atlas-only continental seeder",
    )
    parser.add_argument("--out", type=Path,
                        default=Path("out") / "atlas_survey")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = [run_seed(seed, args.continental_budget, args.out,
                        contour_filter=args.contour_filter,
                        seeder_variant=args.seeder)
               for seed in args.seeds]
    best_key = "eligible_best" if args.contour_filter else "best"
    best_land = [
        result["oracle"][best_key]["land_fraction"]
        if result["oracle"][best_key] is not None else 0.0
        for result in results
    ]
    rank_stability_passed = all(
        result["comparison"]["passed"] for result in results)
    if args.continental_budget >= 0.60:
        land_utility_passed = (
            all(value >= 0.33 for value in best_land)
            and sum(value >= 0.35 for value in best_land)
            >= int(np.ceil(2.0 * len(best_land) / 3.0))
        )
    else:
        land_utility_passed = None
    report = {
        "experiment": "atlas-survey-rank-stability-v1",
        "contour_filter_applied": args.contour_filter,
        "seeder_variant": args.seeder,
        "passed": (rank_stability_passed
                   and land_utility_passed is not False),
        "rank_stability_passed": rank_stability_passed,
        "land_utility_passed": land_utility_passed,
        "elapsed_s": time.perf_counter() - started,
        "constants": {
            "atlas_km": ATLAS_KM,
            "crop_km": FRAME_KM,
            "survey_km": SURVEY_KM,
            "oracle_km": ORACLE_KM,
            "candidate_stride_km": CANDIDATE_STRIDE_KM,
            "basin_recall_km": BASIN_RECALL_KM,
            "evaluation_km": EVALUATION_KM,
            "atlas_guard_km": ATLAS_GUARD_KM,
            "edge_band_km": EDGE_BAND_KM,
            "core_inset_km": CORE_INSET_KM,
            "mid_relief_max_m": MID_RELIEF_MAX_M,
            "fine_relief_max_m": FINE_RELIEF_MAX_M,
            "marine_deposition_max_m": MARINE_DEPOSITION_MAX_M,
            "erosion_time_max_myr": EROSION_TIME_MAX_MYR,
            "selector_water_clearance_m": SELECTOR_WATER_CLEARANCE_M,
            "shortlist_oracle_clearance_m":
                SHORTLIST_ORACLE_CLEARANCE_M,
            "parallel_span_limit_km": PARALLEL_SPAN_LIMIT_KM,
            "coupled_partition_budget_design_envelope":
                COUPLED_PARTITION_BUDGET_ENVELOPE,
            "anisotropic_expected_area_scale":
                ANISOTROPIC_EXPECTED_AREA_SCALE,
            "visible_contour_gate": {
                "role": "hard eligibility only; never a ranking score",
                "surface": "unmodified 64-km common-grid crop",
            },
            "shortlist_size": SHORTLIST_SIZE,
            "max_oracle_regret": MAX_ORACLE_REGRET,
            "surface_selector_scope": {
                "early_controls": "configured values",
                "late_controls": "full positive-elevation envelope",
            },
        },
        "results": results,
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
