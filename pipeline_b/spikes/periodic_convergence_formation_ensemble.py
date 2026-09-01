"""Sealed private feasibility for convergence-driven continental formation.

The complete 24,576-km formation domain is a flat torus.  Canonical material
columns are traced through a continuous kinematic history.  Continental
material matures only where those histories accumulate sustained convergence
between moving plates, modulated by a material-following lithospheric survival
field.  Plate-boundary curvature comes from analytic periodic shears.

There are no land nuclei, distance fronts, Cartesian neighbor propagation,
finite atlas rims, or crop-relative inputs.  One fixed maturation threshold
defines physical support; a single first-crossing chronology supplies exact
nested 14% and 28% snapshots only when that support is large enough.  Crops
are scanned after the 28% authority freezes.

This exposed-seed experiment is formation-only.  It performs no structural
transport, elevation, bathymetry, hydrology, sea-level, surface-process, or
final-water-border work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from engine.rng import fnv1a64, stage_rng, stage_salt
from engine.tectonics import FRAME_KM
from spikes import field_accretion_inventory_ensemble as base
from spikes import field_accretion_periodic_ensemble as prior


EXPERIMENT = "periodic-convergence-formation-development-seed151-158-v1"
SEEDS = tuple(range(151, 159))
RUN_ROLE = "exposed_development"
SEED_SELECTION_DESCRIPTION = (
    "fixed previously exposed block; development evidence only")

PARENT_KM = 24576.0
CANONICAL_KM = prior.CANONICAL_KM
TARGET_INITIAL_CONTINENTAL_FRACTION = 0.28
PREFIX_INITIAL_CONTINENTAL_FRACTION = 0.14
FORMATION_RECUT_CELLS_YX = (97, 149)

# Kinematic history.  Plate points define ownership and motion, never land.
PLATE_COUNT = 16
PLATE_SITE_CANDIDATES = 2048
PLATE_SPEED_KM_PER_MYR_MIN = 12.0
PLATE_SPEED_KM_PER_MYR_MAX = 38.0
PLATE_TURN_PERIOD_MYR_MIN = 180.0
PLATE_TURN_PERIOD_MYR_MAX = 420.0
SOFT_DISTANCE_SIGMA_KM = 1650.0
HISTORY_MYR = 192.0
HISTORY_STEPS = 32
HISTORY_DT_MYR = HISTORY_MYR / HISTORY_STEPS
# A fixed 45% accumulated shortening-equivalent is the physical support
# threshold.  It is set from the dimensional convergence scale before any
# exposed seed is run and is never moved to fill the requested quota.
MATURATION_THRESHOLD = 0.45
SURVIVAL_BETA = 0.65
CHARACTERISTIC_PLATE_SPACING_KM = PARENT_KM / math.sqrt(PLATE_COUNT)
SURVIVAL_WAVELENGTH_RATIOS = (1.15, 0.53, 0.24)
SURVIVAL_WAVELENGTHS_KM = tuple(
    CHARACTERISTIC_PLATE_SPACING_KM * ratio
    for ratio in SURVIVAL_WAVELENGTH_RATIOS)
SURVIVAL_WEIGHTS = (1.0, 0.52, 0.20)

# Fixed analytic diffeomorphism that curves the otherwise Voronoi-like
# kinematic boundaries.  It changes boundary geometry, not crop composition.
WARP_SHEAR_COUNT = 6
WARP_MODE_MIN_RADIUS = 1.0
WARP_MODE_MAX_RADIUS = 5.5
WARP_AMPLITUDE_KM_MIN = 300.0
WARP_AMPLITUDE_KM_MAX = 760.0

# Conspicuous construction-geometry tripwires.  Passing them never proves
# naturalness; fixed-view manual review remains mandatory.
RULER_RUN_KM = prior.RULER_RUN_KM
D4_ALPHA = 0.01
BLOB_SUBSTANTIAL_CELLS = int(math.ceil(
    0.01 * (PARENT_KM / CANONICAL_KM) ** 2))
BLOB_COMPACTNESS_MIN = 0.72
BLOB_SOLIDITY_MIN = 0.90
BLOB_ROUNDED_COUNT_FRACTION_MIN = 0.60
BLOB_ROUNDED_AREA_FRACTION_MIN = 0.60
BLOB_AREA_GINI_MAX = 0.20
BLOB_UNION_MIN_COMPONENTS = 3
BLOB_IDENTITY_MIN_COMPONENTS = 6
BLOB_COHORT_VETO_SEEDS = 6

SOURCE_FILES = tuple(dict.fromkeys((
    *prior.SOURCE_FILES,
    "spikes/periodic_convergence_formation_ensemble.py",
)))

PRIOR_EVIDENCE = {
    "rejected_periodic_harness": (
        "spikes/field_accretion_periodic_ensemble.py",
        "4b702a87b9f828fb66e69b2981ce27a8eb99fe99266a46617eddef4b4a8b93fa",
    ),
    "rejected_periodic_report": (
        "../out/field_accretion_periodic_development_seed151_158_v1/report.json",
        "4d8469b53a1c571e7e74dacd740d9eb9cddcb9732b99f2c1bf3354104379f57f",
    ),
    "rejected_periodic_manual_review": (
        "../out/field_accretion_periodic_development_seed151_158_v1/manual_review.md",
        "acee173e63a18394048444c62918ee1186ad9a7a0c4a6a60f84ea33bdc65efb9",
    ),
}


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json_bytes(payload) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=base._json_default).encode("utf-8")


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
    result = {}
    for label, (relative, expected) in PRIOR_EVIDENCE.items():
        actual = _sha256_file((root / relative).resolve())
        result[label] = {
            "file": relative,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": actual == expected,
        }
    if not all(item["matched"] for item in result.values()):
        raise RuntimeError(f"historical evidence changed: {result}")
    return result


def _minimum_image(delta):
    return ((np.asarray(delta, np.float64) + 0.5 * PARENT_KM)
            % PARENT_KM - 0.5 * PARENT_KM)


def _smooth_chord_distance_terms(delta) -> tuple[np.ndarray, np.ndarray]:
    """Squared circle-chord distance and its derivative on one torus axis."""
    value = np.asarray(delta, np.float64)
    radius = PARENT_KM / (2.0 * np.pi)
    angle = value / radius
    distance2 = 2.0 * radius ** 2 * (1.0 - np.cos(angle))
    derivative = 2.0 * radius * np.sin(angle)
    return distance2, derivative


def _maximin_sites(seed: int) -> np.ndarray:
    """Continuous torus sites with no crop, land, or raster-grid input."""
    rng = stage_rng(seed, "periodic-convergence-plate-sites-v1")
    sites = [rng.uniform(0.0, PARENT_KM, size=2)]
    for _ in range(1, PLATE_COUNT):
        candidates = rng.uniform(
            0.0, PARENT_KM, size=(PLATE_SITE_CANDIDATES, 2))
        best_distance2 = np.full(PLATE_SITE_CANDIDATES, np.inf)
        for site in sites:
            delta = _minimum_image(candidates - site[None, :])
            best_distance2 = np.minimum(
                best_distance2, np.sum(delta ** 2, axis=1))
        key = np.lexsort((
            candidates[:, 1], candidates[:, 0], -best_distance2))
        sites.append(candidates[int(key[0])])
    return np.asarray(sites, np.float64)


def _primitive_modes() -> tuple[tuple[int, int], ...]:
    reach = int(math.ceil(WARP_MODE_MAX_RADIUS))
    modes = []
    for ky in range(-reach, reach + 1):
        for kx in range(-reach, reach + 1):
            if not (ky > 0 or (ky == 0 and kx > 0)):
                continue
            radius = math.hypot(kx, ky)
            if not WARP_MODE_MIN_RADIUS <= radius <= WARP_MODE_MAX_RADIUS:
                continue
            if math.gcd(abs(kx), abs(ky)) != 1:
                continue
            modes.append((kx, ky))
    if len(modes) < WARP_SHEAR_COUNT:
        raise AssertionError("insufficient primitive periodic modes")
    return tuple(modes)


def _warp_table(seed: int) -> list[dict]:
    candidates = _primitive_modes()
    rng = stage_rng(seed, "periodic-convergence-boundary-warp-v1")
    chosen = rng.choice(
        len(candidates), size=WARP_SHEAR_COUNT, replace=False)
    table = []
    for order, raw_index in enumerate(chosen):
        kx, ky = candidates[int(raw_index)]
        radius = float(math.hypot(kx, ky))
        amplitude = float(rng.uniform(
            WARP_AMPLITUDE_KM_MIN, WARP_AMPLITUDE_KM_MAX))
        amplitude *= (2.5 / radius) ** 0.18
        table.append({
            "order": order,
            "kx": int(kx),
            "ky": int(ky),
            "radius": radius,
            "phase_radians": float(rng.uniform(0.0, 2.0 * np.pi)),
            "amplitude_km": float(amplitude),
            "analytic_jacobian_determinant": 1.0,
        })
    return table


def _plate_model(seed: int) -> dict:
    rng = stage_rng(seed, "periodic-convergence-plate-motion-v1")
    sites = _maximin_sites(seed)
    speed = rng.uniform(
        PLATE_SPEED_KM_PER_MYR_MIN,
        PLATE_SPEED_KM_PER_MYR_MAX,
        size=PLATE_COUNT)
    heading = rng.uniform(0.0, 2.0 * np.pi, size=PLATE_COUNT)
    period = rng.uniform(
        PLATE_TURN_PERIOD_MYR_MIN,
        PLATE_TURN_PERIOD_MYR_MAX,
        size=PLATE_COUNT)
    turn_sign = rng.choice(np.asarray((-1.0, 1.0)), size=PLATE_COUNT)
    omega = turn_sign * 2.0 * np.pi / period
    size_bias = rng.normal(0.0, 0.28, size=PLATE_COUNT)
    size_bias -= size_bias.mean()
    warp = _warp_table(seed)
    serial = {
        "sites0_xy_km": sites.tolist(),
        "speed_km_per_myr": speed.tolist(),
        "heading0_radians": heading.tolist(),
        "turn_period_myr": period.tolist(),
        "turn_sign": turn_sign.astype(int).tolist(),
        "soft_size_bias": size_bias.tolist(),
        "warp": warp,
    }
    return {
        "sites0": sites,
        "speed": speed,
        "heading": heading,
        "omega": omega,
        "size_bias": size_bias,
        "warp": warp,
        "serial": serial,
        "sha256": _sha256_bytes(_canonical_json_bytes(serial)),
    }


def _site_state(model: dict, t_myr: float) -> tuple[np.ndarray, np.ndarray]:
    theta0 = model["heading"]
    omega = model["omega"]
    theta = theta0 + omega * float(t_myr)
    speed = model["speed"]
    velocity = np.column_stack((
        speed * np.cos(theta), speed * np.sin(theta)))
    # Integral of a constant-speed turning velocity.
    x_offset = speed / omega * (np.sin(theta) - np.sin(theta0))
    y_offset = -speed / omega * (np.cos(theta) - np.cos(theta0))
    sites = np.mod(
        model["sites0"] + np.column_stack((x_offset, y_offset)),
        PARENT_KM)
    return sites, velocity


def _warp_with_jacobian(x_km: np.ndarray, y_km: np.ndarray,
                        table: list[dict]) -> tuple[np.ndarray, ...]:
    """Compose analytic, area-preserving periodic shears and Jacobians."""
    x = np.asarray(x_km, np.float64).copy()
    y = np.asarray(y_km, np.float64).copy()
    j00 = np.ones_like(x)
    j01 = np.zeros_like(x)
    j10 = np.zeros_like(x)
    j11 = np.ones_like(x)
    factor = 2.0 * np.pi / PARENT_KM
    for record in table:
        kx = float(record["kx"])
        ky = float(record["ky"])
        radius = float(record["radius"])
        amplitude = float(record["amplitude_km"])
        phase = (
            factor * (kx * x + ky * y)
            + float(record["phase_radians"]))
        sine = np.sin(phase)
        cosine_scale = amplitude * factor * np.cos(phase)
        tx = -ky / radius
        ty = kx / radius
        l00 = 1.0 + cosine_scale * tx * kx
        l01 = cosine_scale * tx * ky
        l10 = cosine_scale * ty * kx
        l11 = 1.0 + cosine_scale * ty * ky
        next_j00 = l00 * j00 + l01 * j10
        next_j01 = l00 * j01 + l01 * j11
        next_j10 = l10 * j00 + l11 * j10
        next_j11 = l10 * j01 + l11 * j11
        x = np.mod(x + amplitude * tx * sine, PARENT_KM)
        y = np.mod(y + amplitude * ty * sine, PARENT_KM)
        j00, j01, j10, j11 = (
            next_j00, next_j01, next_j10, next_j11)
    return x, y, j00, j01, j10, j11


def _kinematic_state(model: dict, x_km: np.ndarray,
                     y_km: np.ndarray, t_myr: float,
                     *, include_pair: bool) -> dict:
    """Continuous plate velocity and analytic convergence at points."""
    zx, zy, j00, j01, j10, j11 = _warp_with_jacobian(
        x_km, y_km, model["warp"])
    sites, velocities = _site_state(model, t_myr)
    dx2, derivative_x = _smooth_chord_distance_terms(
        zx[None, ...] - sites[:, 0, None, None])
    dy2, derivative_y = _smooth_chord_distance_terms(
        zy[None, ...] - sites[:, 1, None, None])
    sigma2 = SOFT_DISTANCE_SIGMA_KM ** 2
    logits = (
        -0.5 * (dx2 + dy2) / sigma2
        + model["size_bias"][:, None, None])
    logits -= logits.max(axis=0, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=0, keepdims=True)

    gz_x = -0.5 * derivative_x / sigma2
    gz_y = -0.5 * derivative_y / sigma2
    gradient_x = j00[None, ...] * gz_x + j10[None, ...] * gz_y
    gradient_y = j01[None, ...] * gz_x + j11[None, ...] * gz_y
    mean_gradient_x = np.sum(weights * gradient_x, axis=0)
    mean_gradient_y = np.sum(weights * gradient_y, axis=0)
    dw_dx = weights * (gradient_x - mean_gradient_x[None, ...])
    dw_dy = weights * (gradient_y - mean_gradient_y[None, ...])
    vx = velocities[:, 0, None, None]
    vy = velocities[:, 1, None, None]
    velocity_x = np.sum(weights * vx, axis=0)
    velocity_y = np.sum(weights * vy, axis=0)
    divergence = np.sum(vx * dw_dx + vy * dw_dy, axis=0)
    convergence = np.maximum(-divergence, 0.0)
    result = {
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "convergence_per_myr": convergence,
        "dominant_plate": np.argmax(weights, axis=0).astype(np.int16),
        "membership_entropy": -np.sum(
            weights * np.log(np.maximum(weights, 1e-300)), axis=0),
    }
    if include_pair:
        top2 = np.argpartition(weights, -2, axis=0)[-2:]
        result["plate_pair_low"] = np.minimum(
            top2[0], top2[1]).astype(np.int16)
        result["plate_pair_high"] = np.maximum(
            top2[0], top2[1]).astype(np.int16)
    return result


def _survival_field(seed: int, x_material_km: np.ndarray,
                    y_material_km: np.ndarray) -> np.ndarray:
    value = np.zeros(np.broadcast(
        x_material_km, y_material_km).shape, np.float64)
    norm2 = 0.0
    for index, (wavelength, weight) in enumerate(zip(
            SURVIVAL_WAVELENGTHS_KM, SURVIVAL_WEIGHTS)):
        value += float(weight) * prior._periodic_spectral_octave(
            x_material_km, y_material_km, PARENT_KM,
            float(wavelength), stage_salt(
                seed, f"periodic-convergence-survival-v1:{index}"))
        norm2 += float(weight) ** 2
    return value / math.sqrt(norm2)


def _trace_history(seed: int, qy: np.ndarray, qx: np.ndarray,
                   model: dict) -> dict:
    """Backtrace final material columns; no raster field is interpolated."""
    x, y = np.meshgrid(qx, qy)
    raw_dose = np.empty((HISTORY_STEPS, *x.shape), np.float32)
    maximum_convergence = np.full(x.shape, -np.inf, np.float64)
    dominant_epoch = np.full(x.shape, -1, np.int16)
    pair_low = np.full(x.shape, -1, np.int16)
    pair_high = np.full(x.shape, -1, np.int16)
    # A true explicit midpoint/RK2 characteristic step.  Both velocity
    # evaluations are continuous point queries, never gathers from a raster.
    for reverse_index in range(HISTORY_STEPS - 1, -1, -1):
        t_end = (reverse_index + 1.0) * HISTORY_DT_MYR
        t_mid = (reverse_index + 0.5) * HISTORY_DT_MYR
        state_end = _kinematic_state(
            model, x, y, t_end, include_pair=False)
        x_mid = np.mod(
            x - 0.5 * HISTORY_DT_MYR * state_end["velocity_x"],
            PARENT_KM)
        y_mid = np.mod(
            y - 0.5 * HISTORY_DT_MYR * state_end["velocity_y"],
            PARENT_KM)
        state_mid = _kinematic_state(
            model, x_mid, y_mid, t_mid, include_pair=True)
        convergence = state_mid["convergence_per_myr"]
        raw_dose[reverse_index] = (
            HISTORY_DT_MYR * convergence).astype(np.float32)
        stronger = convergence > maximum_convergence
        maximum_convergence[stronger] = convergence[stronger]
        dominant_epoch[stronger] = reverse_index
        pair_low[stronger] = state_mid["plate_pair_low"][stronger]
        pair_high[stronger] = state_mid["plate_pair_high"][stronger]
        x = np.mod(
            x - HISTORY_DT_MYR * state_mid["velocity_x"], PARENT_KM)
        y = np.mod(
            y - HISTORY_DT_MYR * state_mid["velocity_y"], PARENT_KM)

    survival = _survival_field(seed, x, y)
    survival_factor = np.exp(SURVIVAL_BETA * survival)
    cumulative = np.zeros(x.shape, np.float64)
    crossing_time = np.full(x.shape, np.inf, np.float64)
    for index in range(HISTORY_STEPS):
        increment = raw_dose[index].astype(np.float64) * survival_factor
        before = cumulative.copy()
        cumulative += increment
        crossed = (
            ~np.isfinite(crossing_time)
            & (cumulative >= MATURATION_THRESHOLD)
            & (increment > 0.0))
        crossing_time[crossed] = (
            index * HISTORY_DT_MYR
            + HISTORY_DT_MYR
            * (MATURATION_THRESHOLD - before[crossed])
            / increment[crossed])
    return {
        "x_material_km": x,
        "y_material_km": y,
        "raw_convergence_dose": raw_dose,
        "survival": survival,
        "survival_factor": survival_factor,
        "maturity": cumulative,
        "crossing_time_myr": crossing_time,
        "crossed": np.isfinite(crossing_time),
        "maximum_convergence_per_myr": maximum_convergence,
        "dominant_epoch": dominant_epoch,
        "dominant_plate_pair_low": pair_low,
        "dominant_plate_pair_high": pair_high,
    }


def _physical_keys(qy: np.ndarray, qx: np.ndarray) -> np.ndarray:
    n = qy.size
    py = np.rint(qy / CANONICAL_KM - 0.5).astype(np.int64) % n
    px = np.rint(qx / CANONICAL_KM - 0.5).astype(np.int64) % n
    return py[:, None] * n + px[None, :]


def _activation_rank(seed: int, history: dict,
                     qy: np.ndarray, qx: np.ndarray) -> np.ndarray:
    crossed = history["crossed"].ravel()
    indices = np.flatnonzero(crossed)
    rank = np.full(crossed.size, np.iinfo(np.int32).max, np.int32)
    if indices.size:
        crossing = history["crossing_time_myr"].ravel()[indices]
        maturity = history["maturity"].ravel()[indices]
        ties = prior._coordinate_ties(seed, qy, qx).ravel()[indices]
        physical = _physical_keys(qy, qx).ravel()[indices]
        order = np.lexsort((physical, ties, -maturity, crossing))
        rank[indices[order]] = np.arange(indices.size, dtype=np.int32)
    return rank.reshape(history["crossed"].shape)


def _stable_components(selected: np.ndarray, prefix: np.ndarray,
                       maturity: np.ndarray, qy: np.ndarray,
                       qx: np.ndarray) -> dict:
    raw_labels, components = prior._periodic_components(selected)
    physical = _physical_keys(qy, qx)
    records = []
    for raw_label, (ys, xs) in enumerate(components):
        values = maturity[ys, xs]
        peak = float(values.max())
        choices = np.flatnonzero(values == peak)
        keys = physical[ys[choices], xs[choices]]
        chosen = int(choices[int(np.argmin(keys))])
        y = int(ys[chosen])
        x = int(xs[chosen])
        representative_key = int(physical[y, x])
        records.append({
            "raw_label": int(raw_label),
            "domain_id": representative_key + 1,
            "representative_physical_key": representative_key,
            "pivot_yx_km": [float(qy[y]), float(qx[x])],
            "storage_yx": [y, x],
            "canonical_cells": int(ys.size),
            "prefix_cells": int(np.count_nonzero(prefix[ys, xs])),
            "peak_maturity": peak,
        })
    records.sort(key=lambda item: item["domain_id"])
    dense_by_raw = np.full(len(records), -1, np.int32)
    for dense, record in enumerate(records):
        record["label"] = dense
        dense_by_raw[record["raw_label"]] = dense
    domain_label = np.full(selected.shape, -1, np.int32)
    if records:
        domain_label[selected] = dense_by_raw[raw_labels[selected]]
    prefix_domain_label = np.where(prefix, domain_label, -1).astype(np.int32)
    ids = np.asarray([item["domain_id"] for item in records], np.uint64)
    domain_id_grid = np.zeros(selected.shape, np.uint64)
    prefix_domain_id_grid = np.zeros(selected.shape, np.uint64)
    if records:
        domain_id_grid[selected] = ids[domain_label[selected]]
        prefix_domain_id_grid[prefix] = ids[prefix_domain_label[prefix]]
    return {
        "domain_label": domain_label,
        "prefix_domain_label": prefix_domain_label,
        "domain_id_grid": domain_id_grid,
        "prefix_domain_id_grid": prefix_domain_id_grid,
        "representatives": records,
        "domain_plate_by_label": np.arange(len(records), dtype=np.int32),
    }


def _layout(seed: int, *, cut_cells_yx=(0, 0)) -> dict:
    n = int(round(PARENT_KM / CANONICAL_KM))
    if abs(PARENT_KM / n - CANONICAL_KM) > 1e-12:
        raise AssertionError("canonical torus does not divide exactly")
    cut_y = int(cut_cells_yx[0]) % n
    cut_x = int(cut_cells_yx[1]) % n
    qy = ((np.arange(n) + cut_y) % n + 0.5) * CANONICAL_KM
    qx = ((np.arange(n) + cut_x) % n + 0.5) * CANONICAL_KM
    model = _plate_model(seed)
    history = _trace_history(seed, qy, qx, model)
    activation_rank = _activation_rank(seed, history, qy, qx)
    prefix_cells = int(round(
        PREFIX_INITIAL_CONTINENTAL_FRACTION * activation_rank.size))
    target_cells = int(round(
        TARGET_INITIAL_CONTINENTAL_FRACTION * activation_rank.size))
    prefix = activation_rank < prefix_cells
    selected = activation_rank < target_cells
    crossed_cells = int(history["crossed"].sum())
    capacity_passed = crossed_cells >= target_cells
    components = _stable_components(
        selected, prefix, history["maturity"], qy, qx)
    checks = {
        "physical_threshold_capacity": capacity_passed,
        "exact_prefix_cells": int(prefix.sum()) == prefix_cells,
        "exact_target_cells": int(selected.sum()) == target_cells,
        "strict_mask_prefix": bool(
            np.all(~prefix | selected) and np.any(selected & ~prefix)),
        "all_selected_crossed_threshold": bool(np.all(
            history["crossed"][selected]
            & (history["maturity"][selected] >= MATURATION_THRESHOLD))),
        "prefix_identity_inherited_from_target": bool(np.array_equal(
            components["prefix_domain_label"],
            np.where(prefix, components["domain_label"], -1))),
    }
    checks["passed"] = all(checks.values())
    return {
        "periodic": True,
        "world_km": PARENT_KM,
        "canonical_km": CANONICAL_KM,
        "cut_cells_yx": (cut_y, cut_x),
        "qy": qy,
        "qx": qx,
        "plate_model": model,
        "plate_model_sha256": model["sha256"],
        "history": history,
        "activation_rank": activation_rank,
        "prefix_selected": prefix,
        "selected": selected,
        "requested_prefix_cells": prefix_cells,
        "requested_target_cells": target_cells,
        "crossed_cells": crossed_cells,
        "capacity_passed": capacity_passed,
        "formation_checks": checks,
        **components,
    }


def _roll_to_canonical(array: np.ndarray,
                       cut_cells_yx: tuple[int, int]) -> np.ndarray:
    return np.roll(
        array, shift=(int(cut_cells_yx[0]), int(cut_cells_yx[1])),
        axis=(-2, -1))


def _layout_recut_probe(seed: int, canonical: dict) -> tuple[dict, dict]:
    recut = _layout(seed, cut_cells_yx=FORMATION_RECUT_CELLS_YX)
    layout_arrays = (
        "activation_rank", "prefix_selected", "selected", "domain_label",
        "prefix_domain_label", "domain_id_grid", "prefix_domain_id_grid",
    )
    history_arrays = (
        "x_material_km", "y_material_km", "raw_convergence_dose",
        "survival", "survival_factor", "maturity", "crossing_time_myr",
        "crossed", "maximum_convergence_per_myr", "dominant_epoch",
        "dominant_plate_pair_low", "dominant_plate_pair_high",
    )
    checks = {
        name: bool(np.array_equal(
            canonical[name],
            _roll_to_canonical(recut[name], FORMATION_RECUT_CELLS_YX)))
        for name in layout_arrays
    }
    checks.update({
        f"history.{name}": bool(np.array_equal(
            canonical["history"][name],
            _roll_to_canonical(
                recut["history"][name], FORMATION_RECUT_CELLS_YX)))
        for name in history_arrays
    })
    checks["plate_model"] = (
        canonical["plate_model_sha256"] == recut["plate_model_sha256"])
    signature = lambda layout: [
        (item["domain_id"], tuple(item["pivot_yx_km"]),
         item["canonical_cells"], item["prefix_cells"])
        for item in layout["representatives"]]
    checks["stable_component_records"] = (
        signature(canonical) == signature(recut))
    checks["domain_plate_order"] = bool(np.array_equal(
        canonical["domain_plate_by_label"],
        recut["domain_plate_by_label"]))
    return {
        "cut_cells_yx": list(FORMATION_RECUT_CELLS_YX),
        "checks": checks,
        "passed": all(checks.values()),
    }, recut


def _crofton4(mask: np.ndarray) -> float:
    value = np.asarray(mask, bool)
    tx = np.count_nonzero(value != np.roll(value, 1, axis=1))
    ty = np.count_nonzero(value != np.roll(value, 1, axis=0))
    td1 = np.count_nonzero(
        value != np.roll(value, (1, 1), (0, 1)))
    td2 = np.count_nonzero(
        value != np.roll(value, (1, -1), (0, 1)))
    return float((np.pi / 8.0) * (
        tx + ty + (td1 + td2) / np.sqrt(2.0)))


def _polygon_area(hull: np.ndarray) -> float:
    if hull.shape[0] < 3:
        return 0.0
    x = hull[:, 0]
    y = hull[:, 1]
    return float(0.5 * abs(
        np.dot(x, np.roll(y, -1))
        - np.dot(y, np.roll(x, -1))))


def _component_blob_metrics(component: np.ndarray,
                            pivot_yx: tuple[int, int]) -> dict:
    area = int(component.sum())
    perimeter = _crofton4(component)
    compactness = (
        None if perimeter <= 0.0
        else float(4.0 * np.pi * area / perimeter ** 2))
    unwrapped = prior._unwrap_periodic_component(component, pivot_yx)
    solidity = None
    if not unwrapped["component_winds_torus"]:
        ys, xs = np.nonzero(component)
        ux = unwrapped["relative_x"][ys, xs].astype(np.float64)
        uy = unwrapped["relative_y"][ys, xs].astype(np.float64)
        centers = np.column_stack((ux, uy))
        offsets = np.asarray((
            (-0.5, -0.5), (-0.5, 0.5),
            (0.5, -0.5), (0.5, 0.5)), np.float64)
        corners = np.unique(
            (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 2),
            axis=0)
        hull = prior._convex_hull(corners)
        hull_area = _polygon_area(hull)
        if hull_area + 1e-10 < area:
            raise AssertionError("convex hull smaller than raster area")
        if hull_area > 0.0:
            solidity = float(area / hull_area)
    rounded = bool(
        compactness is not None and solidity is not None
        and compactness >= BLOB_COMPACTNESS_MIN
        and solidity >= BLOB_SOLIDITY_MIN)
    return {
        "cells": area,
        "crofton4_perimeter_cells": perimeter,
        "compactness": compactness,
        "solidity": solidity,
        "rounded": rounded,
        "component_winds_torus": unwrapped["component_winds_torus"],
        "winding_vectors_yx": unwrapped["winding_vectors_yx"],
    }


def _normalized_gini(areas: list[int]) -> float | None:
    values = np.sort(np.asarray(areas, np.float64))
    if values.size == 0:
        return None
    if np.any(values <= 0.0):
        raise ValueError("Gini requires positive areas")
    if values.size == 1:
        return 0.0
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    raw = (
        2.0 * np.dot(ranks, values) / (values.size * values.sum())
        - (values.size + 1.0) / values.size)
    return float(np.clip(
        values.size / (values.size - 1.0) * raw, 0.0, 1.0))


def _blob_channel(mask: np.ndarray, *, channel: str,
                  substantial_cells: int = BLOB_SUBSTANTIAL_CELLS) -> dict:
    labels, found = prior._periodic_components(mask)
    records = []
    for component_label, (ys, xs) in enumerate(found):
        if ys.size < substantial_cells:
            continue
        component = labels == component_label
        records.append({
            "component_label": int(component_label),
            **_component_blob_metrics(
                component, (int(ys[0]), int(xs[0]))),
        })
    count = len(records)
    total_area = sum(item["cells"] for item in records)
    rounded_count = sum(item["rounded"] for item in records)
    rounded_area = sum(
        item["cells"] for item in records if item["rounded"])
    count_fraction = rounded_count / max(count, 1)
    area_fraction = rounded_area / max(total_area, 1)
    gini = _normalized_gini([item["cells"] for item in records])
    minimum = (
        BLOB_UNION_MIN_COMPONENTS
        if channel == "visible_union" else BLOB_IDENTITY_MIN_COMPONENTS)
    flagged = bool(
        count >= minimum
        and count_fraction >= BLOB_ROUNDED_COUNT_FRACTION_MIN
        and area_fraction >= BLOB_ROUNDED_AREA_FRACTION_MIN
        and gini is not None and gini <= BLOB_AREA_GINI_MAX)
    return {
        "channel": channel,
        "substantial_component_cells": substantial_cells,
        "substantial_component_count": count,
        "rounded_component_count": rounded_count,
        "rounded_count_fraction": float(count_fraction),
        "rounded_area_fraction": float(area_fraction),
        "normalized_area_gini": gini,
        "winding_component_count": sum(
            item["component_winds_torus"] for item in records),
        "similar_rounded_blob_field": flagged,
        "components": records,
    }


def _blob_diagnostics(mask: np.ndarray, identities: np.ndarray) -> dict:
    union = _blob_channel(mask, channel="visible_union")
    identity_records = []
    for tag in sorted(int(value) for value in np.unique(
            identities[mask]) if value >= 0):
        channel = _blob_channel(
            mask & (identities == tag), channel="identity")
        for record in channel["components"]:
            identity_records.append({"material_tag": tag, **record})
    count = len(identity_records)
    total_area = sum(item["cells"] for item in identity_records)
    rounded_count = sum(item["rounded"] for item in identity_records)
    rounded_area = sum(
        item["cells"] for item in identity_records if item["rounded"])
    gini = _normalized_gini([item["cells"] for item in identity_records])
    identity = {
        "channel": "identity",
        "substantial_component_cells": BLOB_SUBSTANTIAL_CELLS,
        "substantial_component_count": count,
        "rounded_component_count": rounded_count,
        "rounded_count_fraction": float(rounded_count / max(count, 1)),
        "rounded_area_fraction": float(rounded_area / max(total_area, 1)),
        "normalized_area_gini": gini,
        "winding_component_count": sum(
            item["component_winds_torus"] for item in identity_records),
        "similar_rounded_blob_field": bool(
            count >= BLOB_IDENTITY_MIN_COMPONENTS
            and rounded_count / max(count, 1)
                >= BLOB_ROUNDED_COUNT_FRACTION_MIN
            and rounded_area / max(total_area, 1)
                >= BLOB_ROUNDED_AREA_FRACTION_MIN
            and gini is not None and gini <= BLOB_AREA_GINI_MAX),
        "components": identity_records,
    }
    return {
        "visible_union": union,
        "identity": identity,
        "flagged": bool(
            union["similar_rounded_blob_field"]
            or identity["similar_rounded_blob_field"]),
    }


def _snapshot_geometry(mask: np.ndarray, identities: np.ndarray) -> dict:
    geometry = prior._transported_geometry({
        "binary": np.asarray(mask, bool),
        "dominant_tag": np.asarray(identities, np.int32),
    })
    geometry["blob_diagnostics"] = _blob_diagnostics(mask, identities)
    return geometry


def _rulers_for_seed(seed: int, snapshot: str,
                     geometry: dict) -> list[dict]:
    result = []
    for channel, records in (
            ("visible_union", geometry["union_components"]),
            ("identity", geometry["identity_components"])):
        for record in records:
            angle = record["maximum_ruler_run_angle_degrees"]
            if (angle is None
                    or record["maximum_ruler_run_km"] < RULER_RUN_KM):
                continue
            result.append({
                "seed": seed,
                "snapshot": snapshot,
                "channel": channel,
                "component_label": record["component_label"],
                "material_tag": record.get("material_tag"),
                "length_km": record["maximum_ruler_run_km"],
                "angle_degrees": angle,
            })
    return result


def _seed_result(seed: int, layout: dict) -> tuple[dict, dict]:
    prefix_geometry = _snapshot_geometry(
        layout["prefix_selected"], layout["prefix_domain_label"])
    target_geometry = _snapshot_geometry(
        layout["selected"], layout["domain_label"])
    scan = prior._periodic_scan_windows(
        layout["selected"].astype(np.float64))
    qualification = prior._periodic_qualify_scan(
        scan, layout["selected"], layout["domain_label"])
    scan["selection"] = qualification.pop("selection")
    scan["morphology_qualification"] = qualification
    reviews = prior._periodic_assigned_reviews(
        layout["selected"], layout["domain_label"], scan["selection"])
    all_geometry = (prefix_geometry, target_geometry)
    no_rectangle = all(
        item["tripwires"]["no_severe_oriented_rectangle"]
        for item in all_geometry)
    no_repetition = all(
        item["tripwires"]["no_exact_frame_width_repetition"]
        for item in all_geometry)
    ready_gates = {
        "physical_threshold_capacity": layout["capacity_passed"],
        "formation_invariants": layout["formation_checks"]["passed"],
        "no_severe_oriented_rectangle": no_rectangle,
        "no_exact_frame_width_repetition": no_repetition,
        "separated_morphology_qualified_assignment":
            scan["selection"]["found"],
        "assigned_window_components": reviews["passed"],
    }
    raw_total = layout["history"]["raw_convergence_dose"].astype(
        np.float64).sum(axis=0)
    selected = layout["selected"]
    result = {
        "seed": seed,
        "status": (
            "complete" if layout["capacity_passed"]
            else "physical_support_capacity_failure"),
        "physical_support": {
            "crossed_cells": layout["crossed_cells"],
            "crossed_fraction": float(
                layout["crossed_cells"] / layout["selected"].size),
            "required_target_cells": layout["requested_target_cells"],
            "capacity_passed": layout["capacity_passed"],
            "maturity_min": float(layout["history"]["maturity"].min()),
            "maturity_median": float(np.median(
                layout["history"]["maturity"])),
            "maturity_max": float(layout["history"]["maturity"].max()),
            "selected_raw_convergence_mean": float(
                raw_total[selected].mean()) if selected.any() else None,
            "unselected_raw_convergence_mean": float(
                raw_total[~selected].mean()) if (~selected).any() else None,
            "selected_survival_mean": float(
                layout["history"]["survival"][selected].mean())
                if selected.any() else None,
            "unselected_survival_mean": float(
                layout["history"]["survival"][~selected].mean())
                if (~selected).any() else None,
            "selected_maturity_min": float(
                layout["history"]["maturity"][selected].min())
                if selected.any() else None,
        },
        "formation_checks": layout["formation_checks"],
        "plate_model_sha256": layout["plate_model_sha256"],
        "component_count": len(layout["representatives"]),
        "component_records": layout["representatives"],
        "prefix_geometry": prefix_geometry,
        "target_geometry": target_geometry,
        "scan": {
            key: value for key, value in scan.items()
            if key != "records"
        },
        "assigned_window_reviews": reviews,
        "ready_gates": ready_gates,
        "ready_before_cohort_and_manual": all(ready_gates.values()),
    }
    return result, scan


def _geometry_cohort(results: list[dict]) -> dict:
    rulers = []
    for result in results:
        rulers.extend(_rulers_for_seed(
            result["seed"], "prefix14", result["prefix_geometry"]))
        rulers.extend(_rulers_for_seed(
            result["seed"], "target28", result["target_geometry"]))
    d4 = prior._seed_blocked_d4_randomization(rulers)
    blob_seed_flags = []
    for result in results:
        flagged = bool(
            result["prefix_geometry"]["blob_diagnostics"]["flagged"]
            or result["target_geometry"]["blob_diagnostics"]["flagged"])
        blob_seed_flags.append({"seed": result["seed"], "flagged": flagged})
    blob_count = sum(item["flagged"] for item in blob_seed_flags)
    winding_count = sum(
        int(record["component_winds_torus"])
        for result in results
        for geometry in (
            result["prefix_geometry"], result["target_geometry"])
        for records in (
            geometry["union_components"], geometry["identity_components"])
        for record in records)
    return {
        "long_rulers": rulers,
        "long_ruler_count": len(rulers),
        "d4_seed_blocked_randomization": d4,
        "d4_gate_passed": d4["randomization_upper_tail_p"] >= D4_ALPHA,
        "blob_seed_flags": blob_seed_flags,
        "blob_flagged_seed_count": blob_count,
        "blob_cohort_veto_threshold": BLOB_COHORT_VETO_SEEDS,
        "blob_gate_passed": blob_count < BLOB_COHORT_VETO_SEEDS,
        "winding_component_count": winding_count,
        "euclidean_shape_coverage_complete": winding_count == 0,
    }


def _artifact(path: Path) -> dict:
    return {"file": path.name, "sha256": _sha256_file(path)}


def _scale_field(field: np.ndarray, low=None, high=None) -> np.ndarray:
    value = np.asarray(field, np.float64)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return np.zeros(value.shape, np.float64)
    if low is None:
        low = float(np.percentile(finite, 2.0))
    if high is None:
        high = float(np.percentile(finite, 98.0))
    if high <= low:
        high = low + 1.0
    result = np.zeros(value.shape, np.float64)
    result[np.isfinite(value)] = np.clip(
        (value[np.isfinite(value)] - low) / (high - low), 0.0, 1.0)
    return result


def _heat_rgb(field: np.ndarray) -> np.ndarray:
    value = _scale_field(field)
    stops = np.asarray((
        (5, 16, 38), (18, 58, 92), (39, 126, 132),
        (194, 174, 88), (245, 228, 178)), np.float64)
    position = value * (len(stops) - 1)
    index = np.minimum(position.astype(np.int32), len(stops) - 2)
    fraction = position - index
    rgb = (
        stops[index] * (1.0 - fraction[..., None])
        + stops[index + 1] * fraction[..., None])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _render_grid(panels: list[tuple[str, np.ndarray]], path: Path,
                 columns: int = 4, cell_size: int = 300) -> dict:
    rows = int(math.ceil(len(panels) / columns))
    title_h = 30
    canvas = Image.new(
        "RGB", (columns * cell_size, rows * (cell_size + title_h)),
        (5, 12, 21))
    draw = ImageDraw.Draw(canvas)
    for index, (title, rgb) in enumerate(panels):
        x0 = (index % columns) * cell_size
        y0 = (index // columns) * (cell_size + title_h)
        image = Image.fromarray(np.asarray(rgb, np.uint8), "RGB")
        image = image.resize(
            (cell_size, cell_size), Image.Resampling.NEAREST)
        canvas.paste(image, (x0, y0 + title_h))
        draw.text((x0 + 8, y0 + 8), title, fill=(238, 238, 226))
    canvas.save(path)
    return _artifact(path)


def _render_cause_panel(seed: int, layout: dict, out: Path) -> dict:
    history = layout["history"]
    raw_total = history["raw_convergence_dose"].astype(
        np.float64).sum(axis=0)
    pair = (
        history["dominant_plate_pair_low"].astype(np.int32) * PLATE_COUNT
        + history["dominant_plate_pair_high"].astype(np.int32))
    pair_rgb = prior._label_rgb(pair)
    pair_rgb[history["maximum_convergence_per_myr"] <= 0.0] = (8, 25, 47)
    prefix_target = np.full(layout["selected"].shape, -1, np.int32)
    prefix_target[layout["selected"]] = 0
    prefix_target[layout["prefix_selected"]] = 1
    chronology_rgb = prior._label_rgb(prefix_target)
    chronology_rgb[prior._mask_outline(layout["selected"])] = (245, 245, 235)
    identities = prior._label_rgb(layout["domain_label"])
    identities[prior._mask_outline(layout["selected"])] = (245, 245, 235)
    crossing = history["crossing_time_myr"].copy()
    crossing[~history["crossed"]] = HISTORY_MYR
    panels = [
        ("maximum-convergence plate pair", pair_rgb),
        ("maximum convergence / Myr", _heat_rgb(
            history["maximum_convergence_per_myr"])),
        ("accumulated raw convergence", _heat_rgb(raw_total)),
        ("material survival field", _heat_rgb(history["survival"])),
        ("maturation dose", _heat_rgb(history["maturity"])),
        ("first threshold crossing (dark=early)", _heat_rgb(crossing)),
        ("14% core inside 28% material", chronology_rgb),
        ("post-formation component identities", identities),
    ]
    return _render_grid(
        panels, out / f"seed{seed}_formation_cause_panel.png")


def _render_unmarked_periodic(seed: int, layout: dict,
                              out: Path) -> dict:
    target = np.zeros((*layout["selected"].shape, 3), np.uint8)
    target[:] = (9, 29, 58)
    target[layout["selected"]] = (181, 158, 105)
    prefix = np.zeros_like(target)
    prefix[:] = (9, 29, 58)
    prefix[layout["selected"]] = (161, 143, 101)
    prefix[layout["prefix_selected"]] = (104, 73, 58)
    target3 = np.tile(target, (3, 3, 1))
    prefix3 = np.tile(prefix, (3, 3, 1))
    canvas = Image.new("RGB", (1152, 576), (5, 12, 21))
    left = Image.fromarray(prefix3, "RGB").resize(
        (576, 576), Image.Resampling.NEAREST)
    right = Image.fromarray(target3, "RGB").resize(
        (576, 576), Image.Resampling.NEAREST)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (576, 0))
    path = out / f"seed{seed}_unmarked_3x3_prefix_target.png"
    canvas.save(path)
    return _artifact(path)


def _render_geometry_panel(seed: int, layout: dict,
                           result: dict, out: Path) -> dict:
    panels = []
    for name, labels, geometry in (
            ("prefix14", layout["prefix_domain_label"],
             result["prefix_geometry"]),
            ("target28", layout["domain_label"],
             result["target_geometry"])):
        rgb = prior._label_rgb(labels)
        segments = []
        for record in (
                geometry["union_components"]
                + geometry["identity_components"]):
            segments.extend(prior._rectangle_segments(
                record, (255, 210, 65), width=1))
            endpoints = record.get(
                "maximum_ruler_run_endpoint_yx_unwrapped_cells")
            if endpoints is not None:
                segments.append((endpoints, (255, 70, 205), 2))
        rgb = prior._draw_periodic_segments(rgb, segments)
        panels.append((
            (f"{name}: yellow OBB / magenta ruler; "
             f"wind={sum(item['component_winds_torus'] for item in geometry['union_components'])}"),
            rgb))
    path = out / f"seed{seed}_geometry_overlay.png"
    return _render_grid(panels, path, columns=2, cell_size=512)


def _periodic_crop(array: np.ndarray, candidate: dict) -> np.ndarray:
    frame_cells = int(round(FRAME_KM / CANONICAL_KM))
    y0 = int(round(candidate["y0_km"] / CANONICAL_KM))
    x0 = int(round(candidate["x0_km"] / CANONICAL_KM))
    tiled = np.tile(array, (2, 2) + ((1,) if array.ndim == 3 else ()))
    return tiled[y0:y0 + frame_cells, x0:x0 + frame_cells]


def _render_assignment_panel(seed: int, layout: dict,
                             result: dict, out: Path) -> dict:
    selection = result["scan"]["selection"]
    path = out / f"seed{seed}_assigned_crops.png"
    if not selection["found"]:
        canvas = Image.new("RGB", (900, 330), (5, 12, 21))
        ImageDraw.Draw(canvas).text(
            (24, 24), "no separated morphology-qualified assignment",
            fill=(255, 120, 120))
        canvas.save(path)
        return _artifact(path)
    source = prior._label_rgb(layout["domain_label"])
    panels = []
    for label in ("low", "medium", "high"):
        candidate = selection["assignment"][label]
        crop = _periodic_crop(source, candidate)
        review = result["assigned_window_reviews"]["windows"][label]
        panels.append((
            (f"{label} land={candidate['continental_fraction']:.3f} "
             f"sig={review['significant_component_count']} "
             f"cover={review['significant_component_coverage']:.3f}"),
            crop))
    return _render_grid(panels, path, columns=3, cell_size=300)


def _save_npz(seed: int, out: Path, layout: dict,
              recut: dict) -> dict:
    path = out / f"seed{seed}_formation_authority.npz"
    arrays = {
        "prefix_selected": layout["prefix_selected"],
        "selected": layout["selected"],
        "activation_rank": layout["activation_rank"],
        "domain_label": layout["domain_label"],
        "prefix_domain_label": layout["prefix_domain_label"],
        "domain_id_grid": layout["domain_id_grid"],
        "prefix_domain_id_grid": layout["prefix_domain_id_grid"],
        "x_material_km": layout["history"]["x_material_km"],
        "y_material_km": layout["history"]["y_material_km"],
        "raw_convergence_dose": layout["history"]["raw_convergence_dose"],
        "survival": layout["history"]["survival"],
        "maturity": layout["history"]["maturity"],
        "crossing_time_myr": layout["history"]["crossing_time_myr"],
        "maximum_convergence_per_myr":
            layout["history"]["maximum_convergence_per_myr"],
        "dominant_epoch": layout["history"]["dominant_epoch"],
        "dominant_plate_pair_low":
            layout["history"]["dominant_plate_pair_low"],
        "dominant_plate_pair_high":
            layout["history"]["dominant_plate_pair_high"],
        "recut_prefix_selected": recut["prefix_selected"],
        "recut_selected": recut["selected"],
        "recut_activation_rank": recut["activation_rank"],
        "recut_domain_label": recut["domain_label"],
        "recut_prefix_domain_label": recut["prefix_domain_label"],
        "recut_domain_id_grid": recut["domain_id_grid"],
        "recut_prefix_domain_id_grid": recut["prefix_domain_id_grid"],
        "recut_x_material_km": recut["history"]["x_material_km"],
        "recut_y_material_km": recut["history"]["y_material_km"],
        "recut_raw_convergence_dose":
            recut["history"]["raw_convergence_dose"],
        "recut_survival": recut["history"]["survival"],
        "recut_maturity": recut["history"]["maturity"],
        "recut_crossing_time_myr":
            recut["history"]["crossing_time_myr"],
        "recut_maximum_convergence_per_myr":
            recut["history"]["maximum_convergence_per_myr"],
        "recut_dominant_epoch": recut["history"]["dominant_epoch"],
        "recut_dominant_plate_pair_low":
            recut["history"]["dominant_plate_pair_low"],
        "recut_dominant_plate_pair_high":
            recut["history"]["dominant_plate_pair_high"],
    }
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return {
        "file": path.name,
        "sha256": _sha256_file(path),
        "arrays": sorted(arrays),
    }


def _save_plate_model(seed: int, out: Path, layout: dict) -> dict:
    path = out / f"seed{seed}_plate_model.json"
    sha = base._write_json_exclusive(
        path, layout["plate_model"]["serial"])
    return {"file": path.name, "sha256": sha}


def _save_scan(seed: int, out: Path, scan: dict) -> dict:
    path = out / f"seed{seed}_complete_periodic_scan.csv"
    assigned = {}
    if scan["selection"]["found"]:
        for label, candidate in scan["selection"]["assignment"].items():
            assigned[(candidate["y0_km"], candidate["x0_km"])] = label
    fields = (
        "x0_km", "y0_km", "continental_fraction", "continental_sum",
        "tie_key", "wraps_x", "wraps_y", "component_gate_status",
        "assigned_band")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in scan["records"]:
            row = {key: record.get(key) for key in fields}
            row["assigned_band"] = assigned.get(
                (record["y0_km"], record["x0_km"]), "")
            writer.writerow(row)
    return {
        "file": path.name,
        "sha256": _sha256_file(path),
        "rows": len(scan["records"]),
    }


def _protocol() -> dict:
    prefix_cells = int(round(
        PREFIX_INITIAL_CONTINENTAL_FRACTION
        * (PARENT_KM / CANONICAL_KM) ** 2))
    target_cells = int(round(
        TARGET_INITIAL_CONTINENTAL_FRACTION
        * (PARENT_KM / CANONICAL_KM) ** 2))
    return {
        "experiment": EXPERIMENT,
        "manifest_role": "exclusive_formation_precommit_exposed_development",
        "source_fingerprint": _source_fingerprint(),
        "prior_evidence": _verify_prior_evidence(),
        "seed_policy": {
            "seeds": list(SEEDS),
            "selection": SEED_SELECTION_DESCRIPTION,
            "evidence_role": RUN_ROLE,
            "fresh_validation": False,
            "retry": None,
            "replacement_seed": None,
            "arbitrary_seed_cli": None,
            "interpretation": (
                "may reject or freeze a candidate; cannot establish "
                "generalization or improvement"),
        },
        "scope": {
            "claim": "periodic convergence-driven formation only",
            "structural_transport_builds": 0,
            "elevation_builds": 0,
            "bathymetry_builds": 0,
            "hydrology_builds": 0,
            "surface_process_solves": 0,
            "final_water_border_claim": False,
            "topography_or_bathymetry_claim": False,
        },
        "topology_and_border_causality": {
            "world_km": PARENT_KM,
            "canonical_km": CANONICAL_KM,
            "topology": "complete flat square torus",
            "formation_reads_crop": False,
            "formation_reads_frame_distance_or_direction": False,
            "formation_reads_finite_world_rim": False,
            "formation_uses_cartesian_neighbor_propagation": False,
            "formation_uses_point_land_nuclei": False,
            "formation_uses_distance_front": False,
            "crop_scan_starts_after_target_authority_freezes": True,
            "diagnostic_tiling_enters_generation": False,
            "naturally_frame_parallel_features_are_not_rejected": True,
            "forbidden": [
                "edge copy", "edge blend", "edge fade", "edge mask",
                "forced-water ring", "mirrored geography",
                "smaller generated patch tiling", "crop-relative modifier",
            ],
        },
        "formation_law": {
            "plate_count": PLATE_COUNT,
            "plate_sites": (
                "continuous minimum-image maximin points; plate ownership "
                "and motion only; never land seeds"),
            "speed_km_per_myr": [
                PLATE_SPEED_KM_PER_MYR_MIN,
                PLATE_SPEED_KM_PER_MYR_MAX],
            "turn_period_myr": [
                PLATE_TURN_PERIOD_MYR_MIN,
                PLATE_TURN_PERIOD_MYR_MAX],
            "soft_distance_sigma_km": SOFT_DISTANCE_SIGMA_KM,
            "soft_membership_distance": (
                "globally smooth squared chord distance on each circle of "
                "the product torus; no minimum-image cut-locus derivative"),
            "boundary_warp": {
                "law": (
                    "composition of analytic divergence-free periodic "
                    "integer-mode shears"),
                "count": WARP_SHEAR_COUNT,
                "mode_radius": [
                    WARP_MODE_MIN_RADIUS, WARP_MODE_MAX_RADIUS],
                "base_amplitude_draw_km": [
                    WARP_AMPLITUDE_KM_MIN, WARP_AMPLITUDE_KM_MAX],
                "realized_amplitude_formula": (
                    "base_amplitude_km * (2.5 / mode_radius)^0.18"),
                "continuous_jacobian_determinant": 1.0,
                "purpose": (
                    "curve kinematic boundaries without a raster stencil"),
            },
            "history_myr": HISTORY_MYR,
            "steps": HISTORY_STEPS,
            "dt_myr": HISTORY_DT_MYR,
            "characteristics": (
                "final canonical material columns backtraced pointwise by "
                "explicit midpoint/RK2 through continuous blended plate "
                "velocity; two point evaluations per interval; no raster "
                "gather"),
            "convergence": (
                "positive part of minus globally analytic divergence of "
                "smooth chord-kernel plate velocity membership"),
            "survival": {
                "material_following": True,
                "characteristic_plate_spacing_km":
                    CHARACTERISTIC_PLATE_SPACING_KM,
                "wavelength_ratios_to_plate_spacing":
                    list(SURVIVAL_WAVELENGTH_RATIOS),
                "wavelengths_km": list(SURVIVAL_WAVELENGTHS_KM),
                "weights": list(SURVIVAL_WEIGHTS),
                "exponential_beta": SURVIVAL_BETA,
                "delivered_frame_scale_is_not_an_input": True,
            },
            "maturation_threshold": MATURATION_THRESHOLD,
            "threshold_units": "accumulated dimensionless shortening dose",
            "threshold_fixed_before_exposed_execution": True,
            "capacity_rule": (
                "at least the target count must cross the fixed threshold; "
                "otherwise fail with no threshold move, dilation, fill, or "
                "fallback population"),
            "chronology_order": (
                "first threshold-crossing time, then descending final dose, "
                "then absolute-coordinate tie"),
            "prefix_cells": prefix_cells,
            "target_cells": target_cells,
            "strict_nested_snapshots": True,
            "post_formation_components": (
                "periodic target-mask components receive stable IDs from "
                "their maximum-dose physical representative; they never "
                "feed shape production"),
        },
        "recut_authority": {
            "cut_cells_yx": list(FORMATION_RECUT_CELLS_YX),
            "rerun_every_seed": True,
            "inverse_roll_exact_fields": [
                "material coordinates", "per-step convergence dose",
                "survival", "maturity", "crossing time", "activation rank",
                "prefix/target masks", "stable component labels and IDs",
                "dominant epoch and plate pair"],
            "plate_model_table_exact": True,
            "rolling_completed_output_is_not_the_test": True,
        },
        "crop_availability": {
            "authority": "frozen 28% formation mask; not final land",
            "delivered_frame_km": FRAME_KM,
            "stride_km": base.CANDIDATE_STRIDE_KM,
            "periodic_origin_count": 96 * 96,
            "wrapping_windows_allowed": True,
            "target_bands": {
                label: {
                    "minimum": band[0],
                    "maximum": band[1],
                    "maximum_inclusive": band[2],
                }
                for label, band in base.TARGET_BANDS.items()
            },
            "assignment_separation_km":
                base.MIN_ORIGIN_CHEBYSHEV_SEPARATION_KM,
            "morphology_before_assignment": True,
            "window_gate": {
                "significant_component_frame_fraction":
                    base.SIGNIFICANT_COMPONENT_FRAME_FRACTION,
                "component_count": [
                    base.MIN_SIGNIFICANT_COMPONENTS,
                    base.MAX_SIGNIFICANT_COMPONENTS],
                "minimum_land_coverage":
                    base.MIN_SIGNIFICANT_COMPONENT_COVERAGE,
            },
            "water_border_or_contour_input": None,
        },
        "geometry_gates": {
            "snapshots": ["14% prefix", "28% target"],
            "channels": ["visible union", "post-formation identity"],
            "severe_oriented_rectangle": {
                "minimum_component_cells": prior.SUBSTANTIAL_COMPONENT_CELLS,
                "minimum_box_fill": prior.RECTANGLE_FILL_MIN,
                "minimum_boundary_side_coverage":
                    prior.RECTANGLE_SIDE_COVERAGE_MIN,
                "automatic_veto": True,
            },
            "exact_nonzero_4096km_repetition_veto": True,
            "d4_long_rulers": {
                "minimum_km": RULER_RUN_KM,
                "near_axis_degrees": prior.D4_TOLERANCE_DEGREES,
                "seed_blocked_randomization_trials":
                    prior.GEOMETRY_BLOCK_RANDOMIZATION_TRIALS,
                "alpha": D4_ALPHA,
                "automatic_veto": True,
            },
            "similar_rounded_blob_field": {
                "substantial_world_fraction": 0.01,
                "substantial_cells": BLOB_SUBSTANTIAL_CELLS,
                "compactness_min": BLOB_COMPACTNESS_MIN,
                "solidity_min": BLOB_SOLIDITY_MIN,
                "rounded_count_fraction_min":
                    BLOB_ROUNDED_COUNT_FRACTION_MIN,
                "rounded_area_fraction_min":
                    BLOB_ROUNDED_AREA_FRACTION_MIN,
                "normalized_area_gini_max": BLOB_AREA_GINI_MAX,
                "union_min_components": BLOB_UNION_MIN_COMPONENTS,
                "identity_min_components": BLOB_IDENTITY_MIN_COMPONENTS,
                "cohort_veto_seed_count": BLOB_COHORT_VETO_SEEDS,
                "heuristic_not_calibrated_statistical_test": True,
            },
            "winding_components": {
                "bfs_unwrapped": True,
                "euclidean_obb_ruler_solidity_skipped": True,
                "automatic_veto": False,
                "forces_automatic_shape_coverage_incomplete": True,
            },
            "causal_rule": (
                "frame alignment is diagnostic only; exact recut dependence, "
                "systematic lattice lock, rectangular construction, exact "
                "frame-width stamping, or repeated rounded fields can veto"),
        },
        "execution": {
            "primary_histories": len(SEEDS),
            "recut_histories": len(SEEDS),
            "primary_prefix_target_snapshot_pairs": len(SEEDS),
            "recut_prefix_target_snapshot_pairs": len(SEEDS),
            "complete_periodic_scans": len(SEEDS),
            "structural_transport_builds": 0,
            "elevation_builds": 0,
            "surface_process_solves": 0,
        },
        "automatic_readiness": {
            "every_seed": [
                "physical threshold supplies exact target without fallback",
                "exact nested 14%/28% chronology",
                "exact nonzero recut",
                "no severe oriented rectangle",
                "no exact nonzero frame-width repetition",
                "separated morphology-qualified low/medium/high assignment",
                "all assigned crops pass component gate",
            ],
            "cohort": [
                "D4 seed-blocked p >= 0.01",
                "similar-rounded-blob flags in fewer than six seeds",
            ],
            "permitted_automatic_claim": (
                "Conspicuous formation-artifact tripwires passed."),
            "natural_shapes_claim": False,
        },
        "manual_readiness": {
            "required": True,
            "initial_status": "unreviewed",
            "views": [
                "unmarked 3x3 prefix/target continuity",
                "formation cause panels",
                "union/identity OBB and ruler overlays",
                "all assigned low/medium/high crops",
            ],
            "veto": [
                "square or rectangular surfaces", "unearned straight rulers",
                "parallel/even-width convergence lace", "one connected web",
                "repeated compact rounded bodies", "storage-seam discontinuity",
                "frame-scale stamping with a causal trace",
            ],
        },
        "interpretation": [
            "This is formation support, not final coastline or land.",
            "No claim is made about elevation, bathymetry, or water borders.",
            "Exposed passage could only justify freezing unchanged bytes for a separately authorized fresh cohort.",
            "Production/default/public controls remain unchanged.",
        ],
    }


def _require_execute_output(out: Path,
                            expected_sha256: str) -> tuple[dict, str]:
    if not out.is_dir():
        raise FileNotFoundError(out)
    if {item.name for item in out.iterdir()} != {"protocol_precommit.json"}:
        raise FileExistsError(
            "execute requires only protocol_precommit.json")
    encoded = (out / "protocol_precommit.json").read_bytes()
    actual = _sha256_bytes(encoded)
    if actual != expected_sha256:
        raise ValueError("precommit SHA-256 does not match")
    protocol = json.loads(encoded.decode("utf-8"))
    if protocol != _protocol():
        raise ValueError("source/configuration no longer matches precommit")
    return protocol, actual


def _phase_precommit(out: Path) -> dict:
    base._prepare_empty_output(out)
    value = base._write_json_exclusive(
        out / "protocol_precommit.json", _protocol())
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
    counters = {
        "primary_histories": 0,
        "recut_histories": 0,
        "primary_prefix_target_snapshot_pairs": 0,
        "recut_prefix_target_snapshot_pairs": 0,
        "complete_periodic_scans": 0,
        "structural_transport_builds": 0,
        "elevation_builds": 0,
        "surface_process_solves": 0,
    }
    results = []
    npz_artifacts = []
    model_artifacts = []
    scan_artifacts = []
    cause_panels = []
    periodic_panels = []
    geometry_panels = []
    assignment_panels = []
    for seed in SEEDS:
        layout = _layout(seed)
        counters["primary_histories"] += 1
        counters["primary_prefix_target_snapshot_pairs"] += 1
        recut_probe, recut = _layout_recut_probe(seed, layout)
        counters["recut_histories"] += 1
        counters["recut_prefix_target_snapshot_pairs"] += 1
        result, scan = _seed_result(seed, layout)
        result["recut_probe"] = recut_probe
        result["ready_gates"]["exact_recut"] = recut_probe["passed"]
        result["ready_before_cohort_and_manual"] = all(
            result["ready_gates"].values())
        counters["complete_periodic_scans"] += 1
        npz_artifacts.append(_save_npz(seed, out, layout, recut))
        model_artifacts.append(_save_plate_model(seed, out, layout))
        scan_artifacts.append(_save_scan(seed, out, scan))
        cause_panels.append(_render_cause_panel(seed, layout, out))
        periodic_panels.append(_render_unmarked_periodic(
            seed, layout, out))
        geometry_panels.append(_render_geometry_panel(
            seed, layout, result, out))
        assignment_panels.append(_render_assignment_panel(
            seed, layout, result, out))
        results.append(result)
        print(json.dumps({
            "seed_complete": seed,
            "physical_support_fraction":
                result["physical_support"]["crossed_fraction"],
            "component_count": result["component_count"],
            "assignment": result["scan"]["selection"]["found"],
            "recut": recut_probe["passed"],
        }), flush=True)

    geometry_cohort = _geometry_cohort(results)
    expected_counters = protocol["execution"]
    per_seed_ready_count = sum(
        result["ready_before_cohort_and_manual"] for result in results)
    capacity_count = sum(
        result["ready_gates"]["physical_threshold_capacity"]
        for result in results)
    assignment_count = sum(
        result["ready_gates"][
            "separated_morphology_qualified_assignment"]
        for result in results)
    recut_count = sum(
        result["ready_gates"]["exact_recut"] for result in results)
    aggregate_gates = {
        "all_physical_threshold_capacity": capacity_count == len(SEEDS),
        "all_exact_recut": recut_count == len(SEEDS),
        "all_seed_gates_before_cohort": per_seed_ready_count == len(SEEDS),
        "all_separated_assignments": assignment_count == len(SEEDS),
        "d4_gate": geometry_cohort["d4_gate_passed"],
        "rounded_blob_gate": geometry_cohort["blob_gate_passed"],
        "execution_counts": counters == expected_counters,
    }
    automatic_pass = all(aggregate_gates.values())
    cause_montage = base._render_montage(
        cause_panels, out, "formation_cause_montage.png")
    periodic_montage = base._render_montage(
        periodic_panels, out, "unmarked_periodic_montage.png")
    geometry_montage = base._render_montage(
        geometry_panels, out, "geometry_overlay_montage.png")
    assignment_montage = base._render_montage(
        assignment_panels, out, "assigned_crops_montage.png")
    report = {
        "experiment": EXPERIMENT,
        "run_role": RUN_ROLE,
        "protocol_precommit_sha256": protocol_sha256,
        "source_fingerprint": protocol["source_fingerprint"],
        "elapsed_seconds": float(time.perf_counter() - started),
        "counters": counters,
        "seeds": results,
        "geometry_cohort": geometry_cohort,
        "aggregate": {
            "seed_count": len(SEEDS),
            "capacity_seed_count": capacity_count,
            "exact_recut_seed_count": recut_count,
            "assignment_seed_count": assignment_count,
            "ready_before_manual_seed_count": per_seed_ready_count,
            "gates": aggregate_gates,
            "automatic_formation_artifact_tripwires_passed":
                automatic_pass,
            "permitted_automatic_claim": (
                "Conspicuous formation-artifact tripwires passed."
                if automatic_pass else None),
            "shape_naturalness_status": "unresolved_pending_manual_review",
            "euclidean_shape_coverage_complete":
                geometry_cohort["euclidean_shape_coverage_complete"],
        },
        "artifacts": {
            "formation_authority_npz": npz_artifacts,
            "plate_models": model_artifacts,
            "complete_scans": scan_artifacts,
            "cause_panels": cause_panels,
            "unmarked_periodic_panels": periodic_panels,
            "geometry_panels": geometry_panels,
            "assignment_panels": assignment_panels,
            "cause_montage": cause_montage,
            "unmarked_periodic_montage": periodic_montage,
            "geometry_montage": geometry_montage,
            "assignment_montage": assignment_montage,
        },
        "manual_review": {
            "required": True,
            "status": "unreviewed",
        },
        "recommend_structural_transport": False,
        "production_changed": False,
        "fresh_seeds_touched": False,
    }
    report_sha = base._write_json_exclusive(out / "report.json", report)
    summary = {
        "completed": True,
        "automatic_formation_artifact_tripwires_passed": automatic_pass,
        "capacity_seed_count": capacity_count,
        "assignment_seed_count": assignment_count,
        "exact_recut_seed_count": recut_count,
        "d4_p": geometry_cohort[
            "d4_seed_blocked_randomization"][
                "randomization_upper_tail_p"],
        "blob_flagged_seed_count":
            geometry_cohort["blob_flagged_seed_count"],
        "report_sha256": report_sha,
        "manual_review": "required",
        "recommend_structural_transport": False,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _fixture_xy(n: int, center_yx: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((n, n))
    dy = (yy - center_yx[0] + n / 2.0) % n - n / 2.0
    dx = (xx - center_yx[1] + n / 2.0) % n - n / 2.0
    return dx, dy


def _fixture_disk(n: int, center_yx: tuple[int, int],
                  radius: float) -> np.ndarray:
    x, y = _fixture_xy(n, center_yx)
    return x ** 2 + y ** 2 <= radius ** 2


def _fixture_ellipse(n: int, center_yx: tuple[int, int],
                     rx: float, ry: float, degrees: float) -> np.ndarray:
    x, y = _fixture_xy(n, center_yx)
    angle = np.radians(degrees)
    u = np.cos(angle) * x + np.sin(angle) * y
    v = -np.sin(angle) * x + np.cos(angle) * y
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def _fixture_lobed(n: int, center_yx: tuple[int, int],
                   radius: float) -> np.ndarray:
    x, y = _fixture_xy(n, center_yx)
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    boundary = radius * (
        1.0 + 0.28 * np.sin(3.0 * theta + 0.4)
        + 0.18 * np.sin(7.0 * theta - 0.2))
    return rho <= boundary


def _metric_for_fixture(mask: np.ndarray) -> dict:
    ys, xs = np.nonzero(mask)
    return _component_blob_metrics(
        mask, (int(ys[0]), int(xs[0])))


def _self_check() -> dict:
    epsilon = 1e-3
    chord_left = _smooth_chord_distance_terms(
        0.5 * PARENT_KM - epsilon)
    chord_mid = _smooth_chord_distance_terms(0.5 * PARENT_KM)
    chord_right = _smooth_chord_distance_terms(
        0.5 * PARENT_KM + epsilon)
    if (abs(float(chord_mid[1])) > 1e-9
            or abs(float(chord_left[0] - chord_right[0])) > 1e-8
            or abs(float(chord_left[1] + chord_right[1])) > 1e-8):
        raise AssertionError("smooth chord kernel has an antipodal cut")
    # Analytic warp is periodic and area-preserving in continuous space.
    model = _plate_model(900001)
    q = (np.arange(9) + 0.5) * 731.0
    x, y = np.meshgrid(q, q + 123.0)
    warped = _warp_with_jacobian(x, y, model["warp"])
    shifted = _warp_with_jacobian(
        x + PARENT_KM, y - PARENT_KM, model["warp"])
    for left, right in zip(warped[:2], shifted[:2]):
        if not np.allclose(left, right, rtol=0.0, atol=1e-9):
            raise AssertionError("analytic warp is not periodic")
    determinant = warped[2] * warped[5] - warped[3] * warped[4]
    if not np.allclose(determinant, 1.0, rtol=0.0, atol=1e-10):
        raise AssertionError("analytic warp is not area preserving")
    state = _kinematic_state(
        model, x, y, 77.0, include_pair=True)
    for name in (
            "velocity_x", "velocity_y", "convergence_per_myr",
            "membership_entropy"):
        if not np.all(np.isfinite(state[name])):
            raise AssertionError(f"non-finite kinematic state: {name}")
    if np.any(state["convergence_per_myr"] < 0.0):
        raise AssertionError("convergence positive-part failed")
    shifted_state = _kinematic_state(
        model, x + PARENT_KM, y - PARENT_KM, 77.0,
        include_pair=True)
    for name in state:
        left = state[name]
        right = shifted_state[name]
        if np.issubdtype(np.asarray(left).dtype, np.floating):
            equal = np.allclose(left, right, rtol=0.0, atol=1e-12)
        else:
            equal = np.array_equal(left, right)
        if not equal:
            raise AssertionError(f"kinematic state is not periodic: {name}")

    # Exact physical-coordinate ranking is storage-order invariant.
    n = 24
    q0 = (np.arange(n) + 0.5) * CANONICAL_KM
    xx, yy = np.meshgrid(q0, q0)
    crossing = np.mod(
        0.013 * xx + 0.019 * yy
        + 0.2 * np.sin(2.0 * np.pi * xx / (n * CANONICAL_KM)),
        HISTORY_MYR)
    maturity = 1.0 + 0.1 * np.cos(
        2.0 * np.pi * yy / (n * CANONICAL_KM))
    synthetic = {
        "crossed": np.ones((n, n), bool),
        "crossing_time_myr": crossing,
        "maturity": maturity,
    }
    rank = _activation_rank(900002, synthetic, q0, q0)
    cut = (7, 11)
    qr_y = ((np.arange(n) + cut[0]) % n + 0.5) * CANONICAL_KM
    qr_x = ((np.arange(n) + cut[1]) % n + 0.5) * CANONICAL_KM
    xr, yr = np.meshgrid(qr_x, qr_y)
    synthetic_recut = {
        "crossed": np.ones((n, n), bool),
        "crossing_time_myr": np.mod(
            0.013 * xr + 0.019 * yr
            + 0.2 * np.sin(
                2.0 * np.pi * xr / (n * CANONICAL_KM)),
            HISTORY_MYR),
        "maturity": 1.0 + 0.1 * np.cos(
            2.0 * np.pi * yr / (n * CANONICAL_KM)),
    }
    recut_rank = _activation_rank(
        900002, synthetic_recut, qr_y, qr_x)
    if not np.array_equal(rank, _roll_to_canonical(recut_rank, cut)):
        raise AssertionError("physical activation rank changed under recut")

    # Frozen rounded-blob fixtures.
    fixture_n = 384
    disk = _fixture_disk(fixture_n, (192, 192), 32.0)
    ellipse = _fixture_ellipse(
        fixture_n, (192, 192), 40.0, 24.0, 27.0)
    lobed = _fixture_lobed(fixture_n, (192, 192), 34.0)
    x0, y0 = _fixture_xy(fixture_n, (192, 192))
    radius2 = x0 ** 2 + y0 ** 2
    annulus = (radius2 <= 40.0 ** 2) & (radius2 >= 20.0 ** 2)
    disk_metric = _metric_for_fixture(disk)
    ellipse_metric = _metric_for_fixture(ellipse)
    lobed_metric = _metric_for_fixture(lobed)
    annulus_metric = _metric_for_fixture(annulus)
    if not (disk_metric["compactness"] > 0.90
            and disk_metric["solidity"] > 0.95
            and disk_metric["rounded"]):
        raise AssertionError(f"disk fixture did not classify: {disk_metric}")
    if not (ellipse_metric["compactness"] > 0.80
            and ellipse_metric["solidity"] > 0.95
            and ellipse_metric["rounded"]):
        raise AssertionError(
            f"ellipse fixture did not classify: {ellipse_metric}")
    if lobed_metric["rounded"] or annulus_metric["rounded"]:
        raise AssertionError("irregular fixture classified as rounded")

    centers = (
        (64, 64), (64, 192), (64, 320),
        (256, 64), (256, 192), (256, 320))
    equal_disks = np.zeros((fixture_n, fixture_n), bool)
    equal_tags = np.full((fixture_n, fixture_n), -1, np.int32)
    irregular = np.zeros((fixture_n, fixture_n), bool)
    for label, center in enumerate(centers):
        body = _fixture_disk(fixture_n, center, 24.0)
        equal_disks |= body
        equal_tags[body] = label
        ix, iy = _fixture_xy(fixture_n, center)
        r2 = ix ** 2 + iy ** 2
        outer = 27.0 + 2.0 * (label % 3)
        inner = 11.0 + 2.0 * (label % 2)
        irregular |= (r2 <= outer ** 2) & (r2 >= inner ** 2)
    equal_blob = _blob_diagnostics(equal_disks, equal_tags)
    irregular_blob = _blob_channel(
        irregular, channel="visible_union")
    if not (equal_blob["visible_union"]["similar_rounded_blob_field"]
            and equal_blob["identity"]["similar_rounded_blob_field"]):
        raise AssertionError("equal-disk blob fixture did not trip")
    if irregular_blob["similar_rounded_blob_field"]:
        raise AssertionError("irregular field tripped rounded-blob gate")

    band = np.zeros((fixture_n, fixture_n), bool)
    band[170:214, :] = True
    band_metric = _metric_for_fixture(band)
    if (not band_metric["component_winds_torus"]
            or band_metric["solidity"] is not None):
        raise AssertionError("winding blob fixture mishandled")
    rolled = np.roll(disk, FORMATION_RECUT_CELLS_YX, (0, 1))
    rolled_metric = _metric_for_fixture(rolled)
    for name in ("cells", "crofton4_perimeter_cells", "compactness",
                 "solidity", "rounded"):
        left = disk_metric[name]
        right = rolled_metric[name]
        if isinstance(left, float):
            equal = abs(left - right) <= 1e-12
        else:
            equal = left == right
        if not equal:
            raise AssertionError(f"blob metric changed under roll: {name}")
    rotated_metric = _metric_for_fixture(np.rot90(disk))
    if abs(disk_metric["compactness"]
           - rotated_metric["compactness"]) > 1e-12:
        raise AssertionError("Crofton compactness changed under D4 rotation")

    protocol = _protocol()
    if protocol["seed_policy"]["seeds"] != list(SEEDS):
        raise AssertionError("fixed exposed suite changed")
    if protocol["scope"]["structural_transport_builds"] != 0:
        raise AssertionError("formation-only scope changed")
    return {
        "passed": True,
        "analytic_periodic_warp": True,
        "analytic_area_preserving_warp": True,
        "smooth_antipodal_chord_kernel": True,
        "continuous_periodic_kinematics": True,
        "physical_coordinate_rank_recut": True,
        "rounded_blob_fixtures": True,
        "winding_fixture": True,
        "fixed_suite": list(SEEDS),
        "source_fingerprint": protocol["source_fingerprint"],
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
    else:
        if args.expected_precommit_sha256 is None:
            parser.error("execute requires --expected-precommit-sha256")
        _phase_execute(args.out, args.expected_precommit_sha256)


if __name__ == "__main__":
    main()
