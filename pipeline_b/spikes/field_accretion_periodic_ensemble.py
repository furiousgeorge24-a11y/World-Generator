"""Sealed private feasibility for boundaryless field accretion.

The complete 24,576-km formation and structural atlas is a flat torus.
Periodic gradient fields, components, accretion, plate partitioning,
translation-only transport, material reads, event neighborhoods, and
authority sampling all wrap intrinsically.  No smaller patch is tiled and no
edge is copied, blended, mirrored, faded, or forced to water.

The delivered 4,096-km rectangle does not exist until post-transport scans.
This experiment therefore addresses formation and structural-transport rims
only; it makes no claim yet about elevation, bathymetry, hydrology, surface
processes, or final water borders.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import heapq
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from engine import noise
from engine.rng import fnv1a64, stage_rng, stage_salt
from engine.surface import _cr_w
from engine.tectonics import DT_MYR, FRAME_KM
from spikes import field_accretion_inventory_ensemble as base
from spikes import field_accretion_oracle as legacy
from spikes.periodic_tectonics import (
    build_periodic_structure,
    self_check as periodic_tectonics_self_check,
)


EXPERIMENT = "field-accretion-periodic-ensemble-seed159-166-v1"
SEEDS = tuple(range(159, 167))
EXPOSED_DEVELOPMENT_SEEDS = tuple(range(151, 159))
RUN_ROLE = "fresh_validation"
SEED_SELECTION_DESCRIPTION = (
    "mechanical next eight integers after exposed development seeds 151-158")

PARENT_KM = 24576.0
CANONICAL_KM = legacy.CANONICAL_KM
TARGET_INITIAL_CONTINENTAL_FRACTION = 0.28
PREFIX_INITIAL_CONTINENTAL_FRACTION = 0.14
STRUCTURE_NOMINAL_KM = 80.0
FINE_STRUCTURE_NOMINAL_KM = 40.0
FINE_SENTINEL_SEED = SEEDS[0]
FORMATION_RECUT_CELLS_YX = (97, 149)
STRUCTURE_RECUT_CELLS_YX = (97, 149)

RESISTANCE_REFERENCE_ASSEMBLY = legacy.CARRIER_THRESHOLD
RESISTANCE_REFERENCE_SPEED = 0.72
RESISTANCE_LOG_SENSITIVITY = 1.8 / RESISTANCE_REFERENCE_SPEED
BROAD_PROVINCE_PHASE = 0.0

# Geometry tripwires reject only conspicuous construction geometry.  Natural
# isolated straightness remains a manual/causal question under DESIGN.md.
RULER_RUN_KM = 1024.0
RULER_RUN_CELLS = int(np.ceil(RULER_RUN_KM / CANONICAL_KM))
RULER_COMPONENT_MIN_CELLS = max(
    2, int(np.ceil(RULER_RUN_CELLS / np.sqrt(2.0))))
RECTANGLE_FILL_MIN = 0.90
RECTANGLE_SIDE_COVERAGE_MIN = 0.85
RECTANGLE_SIDE_TOLERANCE_CELLS = 1.0
D4_TOLERANCE_DEGREES = 5.0
GEOMETRY_BLOCK_RANDOMIZATION_TRIALS = 100000
GEOMETRY_BLOCK_RANDOMIZATION_SALT = (
    "periodic-geometry-d4-seed-block-v1")
SUBSTANTIAL_COMPONENT_CELLS = int(np.ceil(
    base.SIGNIFICANT_COMPONENT_FRAME_FRACTION
    * (FRAME_KM / CANONICAL_KM) ** 2))

MIN_READY_SEEDS = len(SEEDS)
MIN_SENTINEL_RESOLUTION_IOU = base.MIN_SENTINEL_RESOLUTION_IOU
MAX_SENTINEL_FRACTION_DELTA = base.MAX_SENTINEL_FRACTION_DELTA

SOURCE_FILES = tuple(dict.fromkeys((
    *base.SOURCE_FILES,
    "spikes/periodic_tectonics.py",
    "spikes/field_accretion_periodic_ensemble.py",
)))

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
    global EXPERIMENT, SEEDS, RUN_ROLE, SEED_SELECTION_DESCRIPTION
    global FINE_SENTINEL_SEED, MIN_READY_SEEDS
    EXPERIMENT = "field-accretion-periodic-development-seed151-158-v1"
    SEEDS = EXPOSED_DEVELOPMENT_SEEDS
    RUN_ROLE = "exposed_development"
    SEED_SELECTION_DESCRIPTION = (
        "fixed previously exposed block; development evidence only")
    FINE_SENTINEL_SEED = SEEDS[0]
    MIN_READY_SEEDS = len(SEEDS)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


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


def _periodic_spectral_octave(x_km, y_km, world_km: float,
                              nominal_wavelength_km: float,
                              salt: int) -> np.ndarray:
    """Radially selected narrow-band random field on a square torus.

    Every Fourier wave vector is an integer torus mode.  A one-fundamental-
    bin raised-cosine shell around the requested radial frequency avoids a
    square gradient-noise lattice while retaining a reported physical scale.
    The finite reciprocal lattice has residual D4 sampling symmetry.  The
    fixed cohort tests systematic D4 lock of long land-boundary rulers;
    remaining field and structural texture stays a manual-review obligation.
    """
    target = world_km / nominal_wavelength_km
    reach = int(np.ceil(target + 1.0))
    modes = []
    for ky in range(-reach, reach + 1):
        for kx in range(-reach, reach + 1):
            if not (ky > 0 or (ky == 0 and kx > 0)):
                continue
            radius = float(np.hypot(kx, ky))
            offset = abs(radius - target)
            if offset > 1.0:
                continue
            weight = 0.5 * (1.0 + np.cos(np.pi * offset))
            if weight > 1e-12:
                modes.append((kx, ky, weight))
    if len(modes) < 6:
        raise ValueError((nominal_wavelength_km, target, len(modes)))
    x = np.mod(np.asarray(x_km, np.float64), world_km)
    y = np.mod(np.asarray(y_km, np.float64), world_km)
    result = np.zeros(np.broadcast(x, y).shape, np.float64)
    weight2 = 0.0
    factor = 2.0 * np.pi / world_km
    for kx, ky, weight in modes:
        hashed = noise._lattice_hash(
            np.asarray(kx, np.int64), np.asarray(ky, np.int64), salt)
        phase = float(hashed) * noise._INV64 * noise._TWO_PI
        result += weight * np.cos(
            factor * (kx * x + ky * y) + phase)
        weight2 += weight ** 2
    # Gradient Perlin's octave-scale standard deviation is about 0.3; this
    # fixed normalization preserves that stochastic-condition magnitude.
    return 0.30 * result / np.sqrt(0.5 * weight2)


def _periodic_fbm(x_km, y_km, world_km: float,
                  base_wavelength_km: float, octaves: int, salt: int,
                  *, gain: float = 0.55, norm_octaves=None) -> np.ndarray:
    total = np.zeros(np.broadcast(x_km, y_km).shape, np.float64)
    amplitude = 1.0
    normalization = 0.0
    for octave in range(octaves):
        nominal = base_wavelength_km / (2.0 ** octave)
        osalt = int(noise._mix(np.uint64(
            (salt + 0x9E37 * (octave + 1)) & ((1 << 64) - 1))))
        total += amplitude * _periodic_spectral_octave(
            x_km, y_km, world_km, nominal, osalt)
        amplitude *= gain
    count = octaves if norm_octaves is None else int(norm_octaves)
    amplitude = 1.0
    for _ in range(count):
        normalization += amplitude
        amplitude *= gain
    return total / normalization


def _periodic_fields(seed: int, qy: np.ndarray,
                     qx: np.ndarray) -> tuple[np.ndarray, ...]:
    X, Y = np.meshgrid(qx, qy)
    assembly_salt = stage_salt(
        seed, "periodic-field-accretion-assembly-v1")
    assembly = _periodic_fbm(
        X, Y, PARENT_KM, legacy.ASSEMBLY_WAVELENGTH_KM,
        legacy.ASSEMBLY_OCTAVES, assembly_salt)
    broad = _periodic_fbm(
        X, Y, PARENT_KM, legacy.ASSEMBLY_WAVELENGTH_KM,
        1, assembly_salt, norm_octaves=legacy.ASSEMBLY_OCTAVES)
    craton = _periodic_fbm(
        X, Y, PARENT_KM, legacy.CRATON_WAVELENGTH_KM,
        legacy.CRATON_OCTAVES,
        stage_salt(seed, "periodic-field-accretion-craton-v1"))
    return assembly, broad, craton


def _coordinate_ties(seed: int, qy: np.ndarray,
                     qx: np.ndarray) -> np.ndarray:
    cy = np.rint(qy / CANONICAL_KM - 0.5).astype(np.int64)
    cx = np.rint(qx / CANONICAL_KM - 0.5).astype(np.int64)
    uy = cy[:, None].astype(np.uint64)
    ux = cx[None, :].astype(np.uint64)
    with np.errstate(over="ignore"):
        value = (
            uy * np.uint64(0x9E3779B185EBCA87)
            ^ ux * np.uint64(0xC2B2AE3D27D4EB4F)
            ^ np.uint64(fnv1a64(
                f"periodic-field-accretion-tie-v1:{seed}")))
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return value


def _periodic_components(mask: np.ndarray) -> tuple[np.ndarray, list]:
    mask = np.asarray(mask, bool)
    n0, n1 = mask.shape
    labels = np.full(mask.shape, -1, np.int32)
    components = []
    neighbors = tuple(
        (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
        if dy or dx)
    for y0, x0 in zip(*np.nonzero(mask)):
        if labels[y0, x0] >= 0:
            continue
        label = len(components)
        labels[y0, x0] = label
        queue = deque([(int(y0), int(x0))])
        ys = []
        xs = []
        while queue:
            y, x = queue.popleft()
            ys.append(y)
            xs.append(x)
            for dy, dx in neighbors:
                yy = (y + dy) % n0
                xx = (x + dx) % n1
                if mask[yy, xx] and labels[yy, xx] < 0:
                    labels[yy, xx] = label
                    queue.append((yy, xx))
        components.append((
            np.asarray(ys, np.int32), np.asarray(xs, np.int32)))
    return labels, components


def _strongest_cell(field: np.ndarray, ys: np.ndarray, xs: np.ndarray,
                    ties: np.ndarray) -> tuple[int, int]:
    values = field[ys, xs]
    candidates = np.flatnonzero(values == values.max())
    if candidates.size == 1:
        chosen = int(candidates[0])
    else:
        chosen = int(candidates[int(np.argmin(
            ties[ys[candidates], xs[candidates]]))])
    return int(ys[chosen]), int(xs[chosen])


def _field_id(seed: int, kind: str, y_km: float, x_km: float) -> str:
    value = fnv1a64(
        f"periodic-field-accretion-v1:{seed}:{kind}:"
        f"{y_km:.0f}:{x_km:.0f}")
    return f"{value:016x}"


def _discover_nuclei(seed: int, qy: np.ndarray, qx: np.ndarray,
                     broad: np.ndarray, craton: np.ndarray) -> dict:
    province, components = _periodic_components(
        broad > BROAD_PROVINCE_PHASE)
    eligible = (
        (province >= 0) & (craton > legacy.NUCLEUS_CRATON_THRESHOLD))
    ties = _coordinate_ties(seed, qy, qx)
    records = []
    for raw_label, (component_y, component_x) in enumerate(components):
        inside = eligible[component_y, component_x]
        if not inside.any():
            continue
        ys = component_y[inside]
        xs = component_x[inside]
        y, x = _strongest_cell(craton, ys, xs, ties)
        physical_y = float(qy[y])
        physical_x = float(qx[x])
        records.append({
            "domain_id": _field_id(
                seed, "periodic-domain", physical_y, physical_x),
            "storage_yx": [y, x],
            "pivot_yx_km": [physical_y, physical_x],
            "province_raw_label": int(raw_label),
            "eligible_craton_cells": int(inside.sum()),
            "peak_craton": float(craton[y, x]),
        })
    records.sort(key=lambda item: item["domain_id"])
    if not records:
        raise ValueError("periodic broad field produced no eligible nucleus")
    return {
        "province_raw": province,
        "province_components": components,
        "eligible_nuclei": eligible,
        "records": records,
    }


def _resistance(assembly: np.ndarray) -> np.ndarray:
    value = (
        np.exp(-RESISTANCE_LOG_SENSITIVITY * (
            np.asarray(assembly, np.float64)
            - RESISTANCE_REFERENCE_ASSEMBLY))
        / RESISTANCE_REFERENCE_SPEED)
    if not np.all(np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError("periodic resistance is not positive finite")
    return value


def _periodic_growth(assembly: np.ndarray, nuclei: list[dict],
                     ties: np.ndarray, qy: np.ndarray, qx: np.ndarray,
                     prefix_cells: int, target_cells: int) -> dict:
    """Periodic Cartesian first-order fast marching of scalar resistance."""
    n0, n1 = assembly.shape
    if n0 != n1 or not 0 < prefix_cells < target_cells <= assembly.size:
        raise ValueError("invalid periodic inventory")
    resistance = _resistance(assembly)
    arrival = np.full(assembly.shape, np.inf, np.float64)
    trial_owner = np.full(assembly.shape, -1, np.int32)
    selected_owner = np.full(assembly.shape, -1, np.int32)
    prefix_owner = np.full(assembly.shape, -1, np.int32)
    settled = np.zeros(assembly.shape, bool)
    physical_y = np.rint(qy / CANONICAL_KM - 0.5).astype(np.int64)
    physical_x = np.rint(qx / CANONICAL_KM - 0.5).astype(np.int64)
    physical_key = (
        physical_y[:, None] * n1 + physical_x[None, :])
    queue = []
    for owner, record in enumerate(nuclei):
        y, x = record["storage_yx"]
        arrival[y, x] = 0.0
        trial_owner[y, x] = owner
        heapq.heappush(queue, (
            0.0, int(ties[y, x]), owner,
            int(physical_key[y, x]), y, x))
    neighbors = ((0, 1), (0, -1), (1, 0), (-1, 0))

    def trial_candidate(y: int, x: int) -> tuple[float, int] | None:
        owner_candidates = set()
        for dy, dx in neighbors:
            yy = (y + dy) % n0
            xx = (x + dx) % n1
            if settled[yy, xx]:
                owner_candidates.add(int(selected_owner[yy, xx]))
        best = None
        local_cost = CANONICAL_KM * resistance[y, x]
        for owner in owner_candidates:
            horizontal = [
                arrival[y, (x + dx) % n1]
                for dx in (-1, 1)
                if (settled[y, (x + dx) % n1]
                    and selected_owner[y, (x + dx) % n1] == owner)
            ]
            vertical = [
                arrival[(y + dy) % n0, x]
                for dy in (-1, 1)
                if (settled[(y + dy) % n0, x]
                    and selected_owner[(y + dy) % n0, x] == owner)
            ]
            if not horizontal and not vertical:
                continue
            if horizontal and vertical:
                a = float(min(horizontal))
                b = float(min(vertical))
                difference = abs(a - b)
                if difference >= local_cost:
                    value = min(a, b) + local_cost
                else:
                    radicand = max(
                        2.0 * local_cost ** 2 - difference ** 2, 0.0)
                    value = 0.5 * (a + b + np.sqrt(radicand))
            else:
                value = float(min(horizontal or vertical) + local_cost)
            key = (value, owner)
            if best is None or key < best:
                best = key
        return best
    settled_count = 0
    prefix_cutoff = None
    target_cutoff = None
    while queue and settled_count < target_cells:
        elapsed, tie, owner, _, y, x = heapq.heappop(queue)
        if settled[y, x]:
            continue
        if (elapsed != arrival[y, x]
                or owner != trial_owner[y, x]):
            continue
        settled[y, x] = True
        selected_owner[y, x] = owner
        settled_count += 1
        if settled_count <= prefix_cells:
            prefix_owner[y, x] = owner
        if settled_count == prefix_cells:
            prefix_cutoff = {"arrival": float(elapsed), "tie": int(tie)}
        if settled_count == target_cells:
            target_cutoff = {"arrival": float(elapsed), "tie": int(tie)}
            break
        for dy, dx in neighbors:
            yy = (y + dy) % n0
            xx = (x + dx) % n1
            if settled[yy, xx]:
                continue
            proposal = trial_candidate(yy, xx)
            if proposal is None:
                continue
            candidate, candidate_owner = proposal
            candidate_key = (
                candidate, int(ties[yy, xx]), candidate_owner)
            old_key = (
                arrival[yy, xx], int(ties[yy, xx]),
                int(trial_owner[yy, xx]))
            if candidate_key < old_key:
                arrival[yy, xx] = candidate
                trial_owner[yy, xx] = candidate_owner
                heapq.heappush(queue, (
                    float(candidate), int(ties[yy, xx]), candidate_owner,
                    int(physical_key[yy, xx]), yy, xx))
    if settled_count != target_cells:
        raise AssertionError("periodic queue did not reach exact inventory")
    return {
        "selected_owner": selected_owner,
        "prefix_owner": prefix_owner,
        "selected": selected_owner >= 0,
        "prefix_selected": prefix_owner >= 0,
        "arrival": arrival,
        "resistance": resistance,
        "prefix_cutoff": prefix_cutoff,
        "target_cutoff": target_cutoff,
    }


def _cyclic_run(vector: np.ndarray) -> int:
    vector = np.asarray(vector, bool).ravel()
    if not vector.any():
        return 0
    if vector.all():
        return int(vector.size)
    doubled = np.concatenate((vector, vector))
    best = current = 0
    for value in doubled:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return min(best, int(vector.size))


def _maximum_cardinal_boundary_run(mask: np.ndarray) -> int:
    mask = np.asarray(mask, bool)
    vertical_edges = mask != np.roll(mask, 1, axis=1)
    horizontal_edges = mask != np.roll(mask, 1, axis=0)
    vertical = max(
        (_cyclic_run(vertical_edges[:, x])
         for x in range(mask.shape[1])), default=0)
    horizontal = max(
        (_cyclic_run(horizontal_edges[y, :])
         for y in range(mask.shape[0])), default=0)
    return int(max(vertical, horizontal))


def _unwrap_periodic_component(mask: np.ndarray,
                               pivot_yx: tuple[int, int]) -> dict:
    """BFS-unroll one 8-connected torus component and detect winding."""
    mask = np.asarray(mask, bool)
    n0, n1 = mask.shape
    py, px = map(int, pivot_yx)
    if not mask[py, px]:
        raise ValueError("component pivot is outside mask")
    sentinel = np.iinfo(np.int32).min
    relative_y = np.full(mask.shape, sentinel, np.int32)
    relative_x = np.full(mask.shape, sentinel, np.int32)
    relative_y[py, px] = 0
    relative_x[py, px] = 0
    queue = deque([(py, px)])
    winding_vectors = set()
    neighbors = tuple(
        (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
        if dy or dx)
    visited = 0
    while queue:
        y, x = queue.popleft()
        visited += 1
        for dy, dx in neighbors:
            yy = (y + dy) % n0
            xx = (x + dx) % n1
            if not mask[yy, xx]:
                continue
            proposal_y = int(relative_y[y, x]) + dy
            proposal_x = int(relative_x[y, x]) + dx
            if relative_y[yy, xx] == sentinel:
                relative_y[yy, xx] = proposal_y
                relative_x[yy, xx] = proposal_x
                queue.append((yy, xx))
                continue
            difference_y = proposal_y - int(relative_y[yy, xx])
            difference_x = proposal_x - int(relative_x[yy, xx])
            if difference_y or difference_x:
                if difference_y % n0 or difference_x % n1:
                    raise AssertionError(
                        "periodic unwrap produced non-topological cycle")
                winding_vectors.add((
                    int(difference_y // n0),
                    int(difference_x // n1),
                ))
    if visited != int(mask.sum()):
        raise AssertionError("component unwrap did not visit entire mask")
    return {
        "relative_y": relative_y,
        "relative_x": relative_x,
        "component_winds_torus": bool(winding_vectors),
        "winding_vectors_yx": [
            list(vector) for vector in sorted(winding_vectors)],
    }


def _maximum_ruler_run(mask: np.ndarray,
                       pivot_yx: tuple[int, int],
                       unwrapped: dict | None = None) -> dict:
    """Longest boundary chord staying within one cell of any straight line."""
    mask = np.asarray(mask, bool)
    n = mask.shape[0]
    if unwrapped is None:
        unwrapped = _unwrap_periodic_component(mask, pivot_yx)
    if unwrapped["component_winds_torus"]:
        return {
            "cells": 0.0,
            "angle_degrees": None,
            "endpoint_yx_unwrapped_cells": None,
            "normal_bin_cells": None,
            "boundary_point_density": 0.0,
            "component_winds_torus": True,
            "winding_vectors_yx": unwrapped["winding_vectors_yx"],
        }
    interior = mask.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        interior &= np.roll(mask, (dy, dx), (0, 1))
    ys, xs = np.nonzero(mask & ~interior)
    if ys.size < 2:
        return {
            "cells": 0.0,
            "angle_degrees": None,
            "endpoint_yx_unwrapped_cells": None,
            "normal_bin_cells": None,
            "boundary_point_density": 0.0,
            "component_winds_torus": False,
            "winding_vectors_yx": [],
        }
    py, px = pivot_yx
    y = unwrapped["relative_y"][ys, xs].astype(np.float64)
    x = unwrapped["relative_x"][ys, xs].astype(np.float64)
    best_length = 0.0
    best_angle = None
    best_endpoints = None
    best_normal_bin = None
    best_density = 0.0
    for degrees in range(180):
        angle = np.radians(float(degrees))
        cosine = np.cos(angle)
        sine = np.sin(angle)
        tangent = x * cosine + y * sine
        normal = -x * sine + y * cosine
        groups = {}
        normal_bin = np.rint(normal).astype(np.int32)
        for index, centre in enumerate(normal_bin):
            for key in (int(centre - 1), int(centre), int(centre + 1)):
                if abs(normal[index] - key) <= 1.0:
                    groups.setdefault(key, []).append((
                        float(tangent[index]),
                        float(y[index]),
                        float(x[index]),
                    ))
        for normal_key, values in groups.items():
            if len(values) < 2:
                continue
            ordered = sorted(values, key=lambda item: (
                item[0], item[1], item[2]))
            start = 0
            for stop in range(1, len(ordered) + 1):
                at_end = stop == len(ordered)
                gap = (np.inf if at_end else
                       ordered[stop][0] - ordered[stop - 1][0])
                if gap <= 2.0:
                    continue
                segment = ordered[start:stop]
                span = float(segment[-1][0] - segment[0][0] + 1.0)
                density = float(len(segment) / max(span, 1.0))
                if density >= 0.70 and span > best_length:
                    best_length = span
                    best_angle = float(degrees)
                    best_normal_bin = int(normal_key)
                    best_density = density
                    best_endpoints = [
                        [
                            float(py + segment[0][1]),
                            float(px + segment[0][2]),
                        ],
                        [
                            float(py + segment[-1][1]),
                            float(px + segment[-1][2]),
                        ],
                    ]
                start = stop
    return {
        "cells": best_length,
        "angle_degrees": best_angle,
        "endpoint_yx_unwrapped_cells": best_endpoints,
        "normal_bin_cells": best_normal_bin,
        "boundary_point_density": float(best_density),
        "component_winds_torus": False,
        "winding_vectors_yx": [],
    }


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted(set(map(tuple, np.asarray(points, np.float64))))
    if len(unique) <= 1:
        return np.asarray(unique, np.float64)

    def cross(origin, left, right):
        return ((left[0] - origin[0]) * (right[1] - origin[1])
                - (left[1] - origin[1]) * (right[0] - origin[0]))

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(
                lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(
                upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], np.float64)


def _oriented_rectangle_stats(mask: np.ndarray,
                              pivot_yx: tuple[int, int],
                              unwrapped: dict | None = None) -> dict:
    """Conspicuous filled-rectangle tripwire on one toroidal component."""
    mask = np.asarray(mask, bool)
    py, px = pivot_yx
    if unwrapped is None:
        unwrapped = _unwrap_periodic_component(mask, pivot_yx)
    if unwrapped["component_winds_torus"]:
        return {
            "minimum_oriented_rectangle_fill": None,
            "boundary_near_rectangle_sides_fraction": None,
            "rectangle_angle_degrees": None,
            "rectangle_corners_yx_unwrapped_cells": None,
            "component_winds_torus": True,
            "winding_vectors_yx": unwrapped["winding_vectors_yx"],
            "severe_rectangle": False,
        }
    interior = mask.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        interior &= np.roll(mask, (dy, dx), (0, 1))
    boundary_y, boundary_x = np.nonzero(mask & ~interior)
    if boundary_y.size < 3:
        return {
            "minimum_oriented_rectangle_fill": 0.0,
            "boundary_near_rectangle_sides_fraction": 0.0,
            "rectangle_angle_degrees": None,
            "rectangle_corners_yx_unwrapped_cells": None,
            "component_winds_torus": False,
            "winding_vectors_yx": [],
            "severe_rectangle": False,
        }
    dy = unwrapped["relative_y"][boundary_y, boundary_x]
    dx = unwrapped["relative_x"][boundary_y, boundary_x]
    points = np.column_stack((dx, dy)).astype(np.float64)
    hull = _convex_hull(points)
    if hull.shape[0] < 2:
        angles = np.asarray([0.0])
    else:
        edges = np.roll(hull, -1, axis=0) - hull
        angles = np.arctan2(edges[:, 1], edges[:, 0])
    best = None
    for angle in angles:
        cosine = np.cos(angle)
        sine = np.sin(angle)
        tangent = points[:, 0] * cosine + points[:, 1] * sine
        normal = -points[:, 0] * sine + points[:, 1] * cosine
        t0, t1 = float(tangent.min()), float(tangent.max())
        n0, n1 = float(normal.min()), float(normal.max())
        box_cells = max(t1 - t0 + 1.0, 1.0) * max(n1 - n0 + 1.0, 1.0)
        fill = float(mask.sum() / box_cells)
        distance = np.minimum.reduce((
            np.abs(tangent - t0), np.abs(tangent - t1),
            np.abs(normal - n0), np.abs(normal - n1)))
        side_fraction = float(np.mean(
            distance <= RECTANGLE_SIDE_TOLERANCE_CELLS))
        key = (box_cells, -side_fraction, float(angle))
        if best is None or key < best[0]:
            best = (
                key, fill, side_fraction, angle,
                (t0, t1, n0, n1))
    _, fill, side_fraction, angle, bounds = best
    cosine = np.cos(angle)
    sine = np.sin(angle)
    t0, t1, n0, n1 = bounds
    corners = []
    for tangent, normal in (
            (t0, n0), (t1, n0), (t1, n1), (t0, n1)):
        x = tangent * cosine - normal * sine
        y = tangent * sine + normal * cosine
        corners.append([float(py + y), float(px + x)])
    severe = (
        mask.sum() >= SUBSTANTIAL_COMPONENT_CELLS
        and fill >= RECTANGLE_FILL_MIN
        and side_fraction >= RECTANGLE_SIDE_COVERAGE_MIN)
    return {
        "minimum_oriented_rectangle_fill": float(fill),
        "boundary_near_rectangle_sides_fraction": side_fraction,
        "rectangle_angle_degrees": float(np.degrees(angle) % 90.0),
        "rectangle_corners_yx_unwrapped_cells": corners,
        "component_winds_torus": False,
        "winding_vectors_yx": [],
        "severe_rectangle": bool(severe),
    }


def _connected_to_seed_periodic(mask: np.ndarray,
                                pivot: tuple[int, int]) -> bool:
    labels, _ = _periodic_components(mask)
    label = labels[pivot]
    return bool(label >= 0 and np.all(~mask | (labels == label)))


def _strict_predecessor_paths(owner: np.ndarray, arrival: np.ndarray,
                              nuclei: list[dict]) -> bool:
    n0, n1 = owner.shape
    for domain, record in enumerate(nuclei):
        pivot = tuple(record["storage_yx"])
        mask = owner == domain
        if not mask[pivot] or arrival[pivot] != 0.0:
            return False
        for y, x in zip(*np.nonzero(mask)):
            if (int(y), int(x)) == pivot:
                continue
            if not any(
                owner[(y + dy) % n0, (x + dx) % n1] == domain
                and arrival[(y + dy) % n0, (x + dx) % n1]
                < arrival[y, x]
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if dy or dx
            ):
                return False
    return True


def _layout(seed: int, *, cut_cells_yx=(0, 0)) -> dict:
    count = int(round(PARENT_KM / CANONICAL_KM))
    if abs(PARENT_KM / count - CANONICAL_KM) > 1e-12:
        raise AssertionError("canonical torus does not divide exactly")
    cut_y = int(cut_cells_yx[0]) % count
    cut_x = int(cut_cells_yx[1]) % count
    qy = ((np.arange(count) + cut_y) % count + 0.5) * CANONICAL_KM
    qx = ((np.arange(count) + cut_x) % count + 0.5) * CANONICAL_KM
    assembly, broad, craton = _periodic_fields(seed, qy, qx)
    discovery = _discover_nuclei(seed, qy, qx, broad, craton)
    cfg = legacy._atlas_config(TARGET_INITIAL_CONTINENTAL_FRACTION)
    if len(discovery["records"]) > cfg.plates:
        raise ValueError("periodic nuclei exceed configured plate count")
    prefix_cells = int(round(
        PREFIX_INITIAL_CONTINENTAL_FRACTION * assembly.size))
    target_cells = int(round(
        TARGET_INITIAL_CONTINENTAL_FRACTION * assembly.size))
    ties = _coordinate_ties(seed, qy, qx)
    growth = _periodic_growth(
        assembly, discovery["records"], ties, qy, qx,
        prefix_cells, target_cells)

    nuclei = discovery["records"]
    domain_id_by_label = np.asarray(
        [int(item["domain_id"], 16) for item in nuclei], np.uint64)
    selected = growth["selected"]
    prefix = growth["prefix_selected"]
    domain_id_grid = np.zeros(assembly.shape, np.uint64)
    domain_id_grid[selected] = domain_id_by_label[
        growth["selected_owner"][selected]]
    prefix_domain_id_grid = np.zeros(assembly.shape, np.uint64)
    prefix_domain_id_grid[prefix] = domain_id_by_label[
        growth["prefix_owner"][prefix]]

    carrier_owner = np.full(assembly.shape, -1, np.int32)
    carriers = []
    for plate_id, record in enumerate(nuclei):
        raw = record["province_raw_label"]
        component_y, component_x = discovery["province_components"][raw]
        carrier_owner[component_y, component_x] = plate_id
        carriers.append({
            "raw_label": raw,
            "carrier_id": _field_id(
                seed, "periodic-province", *record["pivot_yx_km"]),
            "pivot_yx_km": record["pivot_yx_km"],
            "canonical_cells": int(component_y.size),
            "crosses_storage_seam": bool(
                (np.any(component_y == 0) and np.any(component_y == count - 1))
                or (np.any(component_x == 0)
                    and np.any(component_x == count - 1))),
            "plate_id": plate_id,
        })

    domains = []
    for label, record in enumerate(nuclei):
        mask = growth["selected_owner"] == label
        pivot = tuple(record["storage_yx"])
        unwrapped = _unwrap_periodic_component(mask, pivot)
        rectangle = _oriented_rectangle_stats(mask, pivot, unwrapped)
        ruler = _maximum_ruler_run(mask, pivot, unwrapped)
        domains.append({
            "label": label,
            "domain_id": record["domain_id"],
            "carrier_plate_id": label,
            "carrier_raw_label": record["province_raw_label"],
            "pivot_yx_km": record["pivot_yx_km"],
            "storage_yx": list(pivot),
            "canonical_cells": int(mask.sum()),
            "area_km2": float(mask.sum() * CANONICAL_KM ** 2),
            "nucleus_cells": 1,
            "eligible_craton_cells": record["eligible_craton_cells"],
            "peak_craton": record["peak_craton"],
            "connected_to_seed": _connected_to_seed_periodic(mask, pivot),
            "touches_world_rim": False,
            "maximum_cardinal_boundary_run_cells":
                _maximum_cardinal_boundary_run(mask),
            "maximum_ruler_run_cells": float(ruler["cells"]),
            "maximum_ruler_run_km": float(
                ruler["cells"] * CANONICAL_KM),
            "maximum_ruler_run_angle_degrees": ruler["angle_degrees"],
            "maximum_ruler_run_endpoint_yx_unwrapped_cells":
                ruler["endpoint_yx_unwrapped_cells"],
            "maximum_ruler_run_normal_bin_cells":
                ruler["normal_bin_cells"],
            "maximum_ruler_run_boundary_point_density":
                ruler["boundary_point_density"],
            "component_winds_torus":
                unwrapped["component_winds_torus"],
            "winding_vectors_yx": unwrapped["winding_vectors_yx"],
            **rectangle,
        })

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
    }
    prefix_checks["passed"] = all(prefix_checks.values())
    nucleus_mask = np.zeros(assembly.shape, bool)
    for record in nuclei:
        nucleus_mask[tuple(record["storage_yx"])] = True
    return {
        "periodic": True,
        "world_km": PARENT_KM,
        "canonical_km": CANONICAL_KM,
        "cut_cells_yx": (cut_y, cut_x),
        "qy": qy,
        "qx": qx,
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
        "domain_id_grid": domain_id_grid,
        "prefix_domain_id_grid": prefix_domain_id_grid,
        "domain_plate_by_label": np.arange(len(nuclei), dtype=np.int32),
        "carriers": carriers,
        "domains": domains,
        "nuclei": nuclei,
        "requested_cells": target_cells,
        "selected_cells": int(selected.sum()),
        "prefix_cells": int(prefix.sum()),
        "arrival": growth["arrival"],
        "prefix_cutoff": growth["prefix_cutoff"],
        "target_cutoff": growth["target_cutoff"],
        "selected_fraction": float(selected.mean()),
        "prefix_checks": prefix_checks,
        "strict_predecessor_paths": _strict_predecessor_paths(
            growth["selected_owner"], growth["arrival"], nuclei),
        "resistance_diagnostics": {
            "selected_assembly_mean": float(selected_assembly.mean()),
            "unselected_assembly_mean": float(unselected_assembly.mean()),
            "selected_assembly_median": float(
                np.median(selected_assembly)),
            "unselected_assembly_median": float(
                np.median(unselected_assembly)),
            "resistance_min": float(growth["resistance"].min()),
            "resistance_median": float(
                np.median(growth["resistance"])),
            "resistance_max": float(growth["resistance"].max()),
        },
    }


def _roll_to_canonical(array: np.ndarray,
                       cut_cells_yx: tuple[int, int]) -> np.ndarray:
    return np.roll(
        array, shift=(int(cut_cells_yx[0]), int(cut_cells_yx[1])),
        axis=(-2, -1))


def _layout_recut_probe(seed: int, canonical: dict) -> tuple[dict, dict]:
    recut = _layout(seed, cut_cells_yx=FORMATION_RECUT_CELLS_YX)
    arrays = (
        "assembly", "broad_assembly", "craton", "resistance",
        "eligible_nuclei", "nucleus_mask", "selected",
        "prefix_selected", "domain_id_grid", "prefix_domain_id_grid",
    )
    checks = {
        name: bool(np.array_equal(
            canonical[name],
            _roll_to_canonical(recut[name], FORMATION_RECUT_CELLS_YX)))
        for name in arrays
    }
    selected_arrival = np.where(
        canonical["selected"], canonical["arrival"], np.inf)
    recut_arrival = np.where(
        recut["selected"], recut["arrival"], np.inf)
    checks["selected_arrival"] = bool(np.array_equal(
        selected_arrival,
        _roll_to_canonical(recut_arrival, FORMATION_RECUT_CELLS_YX)))
    signature = lambda layout: sorted(
        (item["domain_id"], tuple(item["pivot_yx_km"]))
        for item in layout["nuclei"])
    area = lambda layout: sorted(
        (item["domain_id"], item["canonical_cells"])
        for item in layout["domains"])
    checks["nucleus_identity"] = signature(canonical) == signature(recut)
    checks["domain_areas"] = area(canonical) == area(recut)
    return {
        "cut_cells_yx": list(FORMATION_RECUT_CELLS_YX),
        "checks": checks,
        "passed": all(checks.values()),
    }, recut


def _minimum_image(delta, world_km: float = PARENT_KM):
    return (np.asarray(delta, np.float64) + 0.5 * world_km) % world_km \
        - 0.5 * world_km


def _plate_sites(seed: int, layout: dict, plate_count: int) -> np.ndarray:
    parent_sites = np.asarray(
        [item["pivot_yx_km"] for item in layout["nuclei"]], np.float64)
    remaining = plate_count - parent_sites.shape[0]
    if remaining < 0:
        raise ValueError("periodic nuclei exceed plate count")
    rng = stage_rng(seed, "periodic-field-accretion-sites-v1")
    candidates = rng.uniform(
        0.0, PARENT_KM, (max(8192, 192 * plate_count), 2))
    chosen = [item.copy() for item in parent_sites]
    if chosen:
        delta = _minimum_image(
            candidates[:, None, :] - parent_sites[None, :, :])
        minimum_distance2 = np.min(np.sum(delta ** 2, axis=2), axis=1)
    else:
        minimum_distance2 = np.full(candidates.shape[0], np.inf)
    available = np.ones(candidates.shape[0], bool)
    for _ in range(remaining):
        scores = np.where(available, minimum_distance2, -1.0)
        selected = int(np.argmax(scores))
        site = candidates[selected]
        chosen.append(site.copy())
        available[selected] = False
        distance2 = np.sum(
            _minimum_image(candidates - site) ** 2, axis=1)
        minimum_distance2 = np.minimum(minimum_distance2, distance2)
    return np.asarray(chosen, np.float64)


def _periodic_nearest(array: np.ndarray, y_km, x_km,
                      world_km: float = PARENT_KM) -> np.ndarray:
    array = np.asarray(array)
    y = np.asarray(y_km, np.float64)
    x = np.asarray(x_km, np.float64)
    shape = np.broadcast(y, x).shape
    yy = np.broadcast_to(y, shape)
    xx = np.broadcast_to(x, shape)
    ck_y = world_km / array.shape[0]
    ck_x = world_km / array.shape[1]
    iy = np.floor(np.mod(yy, world_km) / ck_y).astype(np.int64)
    ix = np.floor(np.mod(xx, world_km) / ck_x).astype(np.int64)
    return array[iy % array.shape[0], ix % array.shape[1]]


def _make_partitioner(layout: dict, sites: np.ndarray, expected_seed: int):
    def partition(seed, Y, X, ck, cfg):
        if seed != expected_seed:
            raise ValueError("periodic partition seed changed")
        deformation_km = 0.045 * PARENT_KM
        wavelength = PARENT_KM / 3.5
        salt = stage_salt(seed, "periodic-field-accretion-partition-v1")
        dy = deformation_km * _periodic_fbm(
            X, Y, PARENT_KM, wavelength, 4, salt)
        dx = deformation_km * _periodic_fbm(
            X, Y, PARENT_KM, wavelength, 4, salt + 1)
        site_dy = deformation_km * _periodic_fbm(
            sites[:, 1], sites[:, 0], PARENT_KM,
            wavelength, 4, salt)
        site_dx = deformation_km * _periodic_fbm(
            sites[:, 1], sites[:, 0], PARENT_KM,
            wavelength, 4, salt + 1)
        warped_y = np.mod(Y + dy, PARENT_KM)
        warped_x = np.mod(X + dx, PARENT_KM)
        warped_site_y = np.mod(sites[:, 0] + site_dy, PARENT_KM)
        warped_site_x = np.mod(sites[:, 1] + site_dx, PARENT_KM)
        best = np.full(Y.shape, np.inf)
        label = np.zeros(Y.shape, np.int32)
        for plate in range(sites.shape[0]):
            cost = np.hypot(
                _minimum_image(warped_y - warped_site_y[plate]),
                _minimum_image(warped_x - warped_site_x[plate]))
            take = cost < best
            best[take] = cost[take]
            label[take] = plate
        domain = _periodic_nearest(
            layout["domain_label"], Y, X)
        selected = domain >= 0
        owner = np.full(domain.shape, -1, np.int32)
        owner[selected] = layout["domain_plate_by_label"][domain[selected]]
        label[selected] = owner[selected]
        return label
    return partition


def _make_samplers(layout: dict):
    def continent(plate_id, material_y_km, material_x_km):
        label = _periodic_nearest(
            layout["domain_label"], material_y_km, material_x_km)
        valid = label >= 0
        owner = np.full(label.shape, -1, np.int32)
        owner[valid] = layout["domain_plate_by_label"][label[valid]]
        return valid & (owner == plate_id)

    def material_tag(plate_id, material_y_km, material_x_km):
        label = _periodic_nearest(
            layout["domain_label"], material_y_km, material_x_km)
        valid = label >= 0
        owner = np.full(label.shape, -1, np.int32)
        owner[valid] = layout["domain_plate_by_label"][label[valid]]
        return np.where(
            valid & (owner == plate_id), label, -1).astype(
                np.int32, copy=False)
    return continent, material_tag


def _initial_ocean_age(seed, Y, X, ck, cfg):
    field = _periodic_fbm(
        X, Y, PARENT_KM, 2400.0, 4,
        stage_salt(seed, "periodic-field-accretion-initial-age-v1"))
    eras = np.rint(np.clip(3.5 + 10.0 * field, 0.0, 7.0))
    return -eras.astype(np.int16)


def _build(seed: int, layout: dict, nominal_km: float,
           *, cut_cells_yx=(0, 0)):
    cfg = legacy._atlas_config(TARGET_INITIAL_CONTINENTAL_FRACTION)
    sites = _plate_sites(seed, layout, int(cfg.plates))
    continent, material_tag = _make_samplers(layout)
    structure = build_periodic_structure(
        seed, cfg,
        world_km=PARENT_KM,
        coarse_km=nominal_km,
        partitioner=_make_partitioner(layout, sites, seed),
        initial_age_sampler=_initial_ocean_age,
        continent_sampler=continent,
        material_tag_sampler=material_tag,
        cut_cells_yx=cut_cells_yx,
    )
    return structure, cfg, sites


def _periodic_bicubic(field: np.ndarray, y_km, x_km,
                      world_km: float) -> np.ndarray:
    field = np.asarray(field, np.float64)
    n0, n1 = field.shape
    ck_y = world_km / n0
    ck_x = world_km / n1
    gy = np.mod(np.asarray(y_km, np.float64), world_km) / ck_y - 0.5
    gx = np.mod(np.asarray(x_km, np.float64), world_km) / ck_x - 0.5
    iy = np.floor(gy).astype(np.int64)
    ix = np.floor(gx).astype(np.int64)
    fy = gy - iy
    fx = gx - ix
    wy = _cr_w(fy)
    wx = _cr_w(fx)
    out = np.zeros(np.broadcast(gy, gx).shape, np.float64)
    for a in range(4):
        ia = (iy + (a - 1)) % n0
        row = np.zeros_like(out)
        for b in range(4):
            ib = (ix + (b - 1)) % n1
            row += wx[b] * field[ia, ib]
        out += wy[a] * row
    iy0 = iy % n0
    ix0 = ix % n1
    c00 = field[iy0, ix0]
    c10 = field[iy0, (ix0 + 1) % n1]
    c01 = field[(iy0 + 1) % n0, ix0]
    c11 = field[(iy0 + 1) % n0, (ix0 + 1) % n1]
    low = np.minimum.reduce((c00, c10, c01, c11))
    high = np.maximum.reduce((c00, c10, c01, c11))
    return np.clip(out, low, high)


def _sample_structure_authority(structure, layout: dict) -> dict:
    X, Y = np.meshgrid(layout["qx"], layout["qy"])
    proxy = np.clip(_periodic_bicubic(
        structure.cont_frac, Y, X, structure.world_km), 0.0, 1.0)
    tags = np.asarray(structure._material_tag_samples)
    sampled_tags = np.stack([
        _periodic_nearest(
            tags[index], Y, X, structure.world_km)
        for index in range(tags.shape[0])
    ])
    dominant = np.full(proxy.shape, -1, np.int32)
    best_count = np.zeros(proxy.shape, np.int8)
    for label in np.unique(sampled_tags[sampled_tags >= 0]):
        count = np.count_nonzero(sampled_tags == label, axis=0)
        take = (
            (count > best_count)
            | ((count == best_count) & (count > 0)
               & ((dominant < 0) | (label < dominant))))
        dominant[take] = int(label)
        best_count[take] = count[take]
    return {
        "proxy": proxy,
        "binary": proxy >= 0.5,
        "dominant_tag": dominant,
    }


def _structure_recut_probe(seed: int, layout: dict,
                           canonical_structure) -> tuple[dict, object]:
    recut, _, _ = _build(
        seed, layout, STRUCTURE_NOMINAL_KM,
        cut_cells_yx=STRUCTURE_RECUT_CELLS_YX)
    arrays = (
        "label", "cont", "cont_frac", "age_myr", "belt",
        "belt_age_era", "conv_recent", "div_recent", "coast",
        "active_margin", "passive_margin", "initial_label",
    )
    checks = {
        name: bool(np.array_equal(
            getattr(canonical_structure, name),
            _roll_to_canonical(
                getattr(recut, name), STRUCTURE_RECUT_CELLS_YX)))
        for name in arrays
    }
    checks["material_tags"] = bool(np.array_equal(
        canonical_structure._material_tag_samples,
        _roll_to_canonical(
            recut._material_tag_samples, STRUCTURE_RECUT_CELLS_YX)))
    checks["plate_displacements"] = bool(np.array_equal(
        canonical_structure._plate_displacements_yx_km,
        recut._plate_displacements_yx_km))
    return {
        "cut_cells_yx": list(STRUCTURE_RECUT_CELLS_YX),
        "checks": checks,
        "passed": all(checks.values()),
    }, recut


def _rank_fraction(value: float, values: np.ndarray) -> float:
    values = np.asarray(values, np.float64)
    return float((np.count_nonzero(values < value)
                  + 0.5 * np.count_nonzero(values == value)) / values.size)


def _geometry_diagnostics(mask: np.ndarray,
                          identity: np.ndarray | None = None) -> dict:
    mask = np.asarray(mask, bool)
    vertical_edges = mask != np.roll(mask, 1, axis=1)
    horizontal_edges = mask != np.roll(mask, 1, axis=0)
    vertical_profile = vertical_edges.sum(axis=0).astype(np.float64)
    horizontal_profile = horizontal_edges.sum(axis=1).astype(np.float64)
    if identity is None:
        identity_vertical_profile = np.zeros(mask.shape[1], np.float64)
        identity_horizontal_profile = np.zeros(mask.shape[0], np.float64)
    else:
        identity = np.asarray(identity)
        if identity.shape != mask.shape:
            raise ValueError("geometry identity shape does not match mask")
        identity_vertical = (
            mask & np.roll(mask, 1, axis=1)
            & (identity != np.roll(identity, 1, axis=1)))
        identity_horizontal = (
            mask & np.roll(mask, 1, axis=0)
            & (identity != np.roll(identity, 1, axis=0)))
        identity_vertical_profile = identity_vertical.sum(
            axis=0).astype(np.float64)
        identity_horizontal_profile = identity_horizontal.sum(
            axis=1).astype(np.float64)
    frame_cycles = int(round(PARENT_KM / FRAME_KM))

    def spectrum(profile):
        power = np.abs(np.fft.rfft(profile - profile.mean())) ** 2
        controls = np.asarray([
            power[index] for index in (
                frame_cycles - 2, frame_cycles - 1,
                frame_cycles + 1, frame_cycles + 2)
            if 0 < index < power.size
        ])
        return {
            "frame_cycles": frame_cycles,
            "frame_scale_power": float(power[frame_cycles]),
            "neighbor_scale_median_power": float(np.median(controls)),
            "frame_to_neighbor_power_ratio": float(
                power[frame_cycles] / max(np.median(controls), 1e-12)),
        }

    frame_cells = int(round(FRAME_KM / CANONICAL_KM))
    shifted_ious = {}
    for multiplier in range(1, frame_cycles):
        shift = multiplier * frame_cells
        shifted_ious[f"x_{multiplier}"] = base._mask_iou(
            mask, np.roll(mask, shift, axis=1))
        shifted_ious[f"y_{multiplier}"] = base._mask_iou(
            mask, np.roll(mask, shift, axis=0))
    return {
        "vertical_transition_profile": vertical_profile.astype(int).tolist(),
        "horizontal_transition_profile": horizontal_profile.astype(int).tolist(),
        "identity_vertical_transition_profile":
            identity_vertical_profile.astype(int).tolist(),
        "identity_horizontal_transition_profile":
            identity_horizontal_profile.astype(int).tolist(),
        "canonical_vertical_seam_rank": _rank_fraction(
            vertical_profile[0], vertical_profile),
        "canonical_horizontal_seam_rank": _rank_fraction(
            horizontal_profile[0], horizontal_profile),
        "maximum_cardinal_boundary_run_cells":
            _maximum_cardinal_boundary_run(mask),
        "maximum_cardinal_boundary_run_km": float(
            _maximum_cardinal_boundary_run(mask) * CANONICAL_KM),
        "vertical_frame_lock_spectrum": spectrum(vertical_profile),
        "horizontal_frame_lock_spectrum": spectrum(horizontal_profile),
        "identity_vertical_frame_lock_spectrum":
            spectrum(identity_vertical_profile),
        "identity_horizontal_frame_lock_spectrum":
            spectrum(identity_horizontal_profile),
        "frame_width_shift_ious": shifted_ious,
        "exact_nonzero_frame_width_repetition": any(
            value == 1.0 for value in shifted_ious.values()),
    }


def _periodic_shape_components(mask: np.ndarray) -> dict:
    """Oriented shape evidence for every nontrivial toroidal component."""
    labels, found = _periodic_components(mask)
    records = []
    tiny = 0
    for component_label, (ys, xs) in enumerate(found):
        cells = int(ys.size)
        if cells < RULER_COMPONENT_MIN_CELLS:
            tiny += 1
            continue
        component = labels == component_label
        pivot = (int(ys[0]), int(xs[0]))
        unwrapped = _unwrap_periodic_component(component, pivot)
        ruler = _maximum_ruler_run(component, pivot, unwrapped)
        rectangle = _oriented_rectangle_stats(
            component, pivot, unwrapped)
        cardinal = _maximum_cardinal_boundary_run(component)
        records.append({
            "component_label": int(component_label),
            "pivot_yx_cells": list(pivot),
            "canonical_cells": cells,
            "maximum_cardinal_boundary_run_cells": cardinal,
            "maximum_cardinal_boundary_run_km": float(
                cardinal * CANONICAL_KM),
            "maximum_ruler_run_cells": float(ruler["cells"]),
            "maximum_ruler_run_km": float(
                ruler["cells"] * CANONICAL_KM),
            "maximum_ruler_run_angle_degrees": ruler["angle_degrees"],
            "maximum_ruler_run_endpoint_yx_unwrapped_cells":
                ruler["endpoint_yx_unwrapped_cells"],
            "maximum_ruler_run_normal_bin_cells":
                ruler["normal_bin_cells"],
            "maximum_ruler_run_boundary_point_density":
                ruler["boundary_point_density"],
            "component_winds_torus":
                unwrapped["component_winds_torus"],
            "winding_vectors_yx": unwrapped["winding_vectors_yx"],
            **rectangle,
        })
    return {
        "component_count": len(found),
        "analyzed_component_count": len(records),
        "tiny_component_count": tiny,
        "minimum_analyzed_component_cells": RULER_COMPONENT_MIN_CELLS,
        "components": records,
    }


def _transported_geometry(authority: dict) -> dict:
    """Measure final transported land unions and material identities."""
    binary = np.asarray(authority["binary"], bool)
    dominant = np.asarray(authority["dominant_tag"], np.int32)
    union = _geometry_diagnostics(binary, dominant)
    union_shapes = _periodic_shape_components(binary)
    identity_components = []
    total_identity_components = 0
    tiny_identity_components = 0
    for material_tag in sorted(
            int(value) for value in np.unique(dominant[binary])
            if value >= 0):
        shapes = _periodic_shape_components(
            binary & (dominant == material_tag))
        total_identity_components += shapes["component_count"]
        tiny_identity_components += shapes["tiny_component_count"]
        for record in shapes["components"]:
            identity_components.append({
                "material_tag": material_tag,
                **record,
            })
    analyzed = (
        union_shapes["components"] + identity_components)
    tripwires = {
        "no_severe_oriented_rectangle": not any(
            item["severe_rectangle"] for item in analyzed),
        "oriented_shape_coverage_complete_diagnostic": not any(
            item["component_winds_torus"] for item in analyzed),
        "no_exact_frame_width_repetition": not union[
            "exact_nonzero_frame_width_repetition"],
        "maximum_cardinal_boundary_run_is_diagnostic_only": float(
            union["maximum_cardinal_boundary_run_km"]),
    }
    tripwires["passed"] = (
        tripwires["no_severe_oriented_rectangle"]
        and tripwires["no_exact_frame_width_repetition"])
    return {
        "union_geometry_diagnostics": union,
        "union_component_count": union_shapes["component_count"],
        "analyzed_union_component_count":
            union_shapes["analyzed_component_count"],
        "tiny_union_component_count": union_shapes["tiny_component_count"],
        "union_components": union_shapes["components"],
        "identity_component_count": total_identity_components,
        "analyzed_identity_component_count": len(identity_components),
        "tiny_identity_component_count": tiny_identity_components,
        "minimum_analyzed_component_cells": RULER_COMPONENT_MIN_CELLS,
        "identity_components": identity_components,
        "tripwires": tripwires,
    }


def _periodic_separated(left: dict, right: dict) -> bool:
    dy = abs(float(_minimum_image(
        left["y0_km"] - right["y0_km"])))
    dx = abs(float(_minimum_image(
        left["x0_km"] - right["x0_km"])))
    return max(dy, dx) >= base.MIN_ORIGIN_CHEBYSHEV_SEPARATION_KM


def _periodic_select_assignment(records: list[dict]) -> dict:
    targets = {"low": 0.20, "medium": 0.35, "high": 0.50}
    pools = {
        label: sorted(
            [item for item in records
             if base._in_band(item["continental_fraction"], band)],
            key=lambda item: (
                abs(item["continental_fraction"] - targets[label]),
                item["tie_key"]))
        for label, band in base.TARGET_BANDS.items()
    }
    lows = pools["low"]
    mediums = pools["medium"]
    highs = pools["high"]

    def low_compatibility(candidate):
        bits = 0
        for index, low in enumerate(lows):
            if _periodic_separated(candidate, low):
                bits |= 1 << index
        return bits

    high_low_bits = [low_compatibility(item) for item in highs]
    medium_low_bits = [low_compatibility(item) for item in mediums]
    best = None
    viable_count = 0
    for high_index, high in enumerate(highs):
        for medium_index, medium in enumerate(mediums):
            if not _periodic_separated(high, medium):
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
                label: abs(item["continental_fraction"] - targets[label])
                for label, item in assignment.items()
            }
            key = (
                errors["high"], errors["medium"], errors["low"],
                sum(errors.values()), high["tie_key"],
                medium["tie_key"], low["tie_key"])
            if best is None or key < best[0]:
                best = (key, assignment, errors)
    if best is None:
        key, assignment, errors = None, {}, {}
    else:
        key, assignment, errors = best
    diagnostics = {}
    if records:
        diagnostics = {
            label: min(records, key=lambda item: (
                abs(item["continental_fraction"] - target),
                item["tie_key"]))
            for label, target in targets.items()
        }
    return {
        "pool_counts": {
            label: len(items) for label, items in pools.items()},
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


def _periodic_scan_windows(field: np.ndarray) -> dict:
    field = np.asarray(field, np.float64)
    n = field.shape[0]
    if field.shape != (n, n):
        raise ValueError("periodic scan requires a square field")
    frame_cells = int(round(FRAME_KM / CANONICAL_KM))
    stride_cells = int(round(base.CANDIDATE_STRIDE_KM / CANONICAL_KM))
    axis = np.arange(0, n, stride_cells, dtype=np.int32)
    tiled = np.tile(field, (2, 2))
    integral = np.pad(
        np.cumsum(np.cumsum(tiled, axis=0), axis=1),
        ((1, 0), (1, 0)), constant_values=0)
    records = []
    for y0 in axis:
        y1 = int(y0 + frame_cells)
        for x0 in axis:
            x1 = int(x0 + frame_cells)
            total = float(
                integral[y1, x1] - integral[y0, x1]
                - integral[y1, x0] + integral[y0, x0])
            records.append({
                "x0_km": float(x0 * CANONICAL_KM),
                "y0_km": float(y0 * CANONICAL_KM),
                "continental_sum": total,
                "continental_fraction": total / frame_cells ** 2,
                "tie_key": int(fnv1a64(
                    f"periodic-window:{x0 * CANONICAL_KM:.0f}:"
                    f"{y0 * CANONICAL_KM:.0f}")),
                "wraps_x": bool(x0 + frame_cells > n),
                "wraps_y": bool(y0 + frame_cells > n),
            })
    fractions = np.asarray([
        item["continental_fraction"] for item in records])
    return {
        "candidate_count": len(records),
        "minimum_continental_fraction": float(fractions.min()),
        "median_continental_fraction": float(np.median(fractions)),
        "maximum_continental_fraction": float(fractions.max()),
        "composition_only_selection": _periodic_select_assignment(records),
        "records": records,
    }


def _periodic_qualify_scan(scan: dict, binary: np.ndarray,
                           dominant_tag: np.ndarray) -> dict:
    binary_tiled = np.tile(np.asarray(binary, bool), (2, 2))
    tag_tiled = np.tile(np.asarray(dominant_tag), (2, 2))
    qualified = []
    evaluated = passed = 0
    for candidate in scan["records"]:
        in_band = any(
            base._in_band(candidate["continental_fraction"], band)
            for band in base.TARGET_BANDS.values())
        if not in_band:
            candidate["component_gate_status"] = -1
            continue
        review = base._single_window_review(
            binary_tiled, tag_tiled, candidate)
        evaluated += 1
        candidate["component_gate_status"] = int(review["passed"])
        if review["passed"]:
            qualified.append(candidate)
            passed += 1
    return {
        "evaluated_candidate_count": evaluated,
        "passed_candidate_count": passed,
        "selection": _periodic_select_assignment(qualified),
    }


def _periodic_assigned_reviews(binary: np.ndarray, dominant_tag: np.ndarray,
                               selection: dict) -> dict:
    if not selection["found"]:
        return {"passed": False, "reason": "no_assignment", "windows": {}}
    binary_tiled = np.tile(np.asarray(binary, bool), (2, 2))
    tag_tiled = np.tile(np.asarray(dominant_tag), (2, 2))
    windows = {
        label: base._single_window_review(
            binary_tiled, tag_tiled, candidate)
        for label, candidate in selection["assignment"].items()
    }
    return {
        "passed": all(item["passed"] for item in windows.values()),
        "windows": windows,
    }


def _seed_result(seed: int, layout: dict, structure, cfg,
                 sites: np.ndarray) -> tuple[dict, dict, dict, dict]:
    authority = _sample_structure_authority(structure, layout)
    transported_geometry = _transported_geometry(authority)
    canonical_scan = _periodic_scan_windows(layout["selected"])
    transported_scan = _periodic_scan_windows(authority["proxy"])
    qualification = _periodic_qualify_scan(
        transported_scan, authority["binary"], authority["dominant_tag"])
    transported_scan["selection"] = qualification.pop("selection")
    transported_scan["morphology_qualification"] = qualification
    selection = transported_scan["selection"]
    window_reviews = _periodic_assigned_reviews(
        authority["binary"], authority["dominant_tag"], selection)

    Y, X = np.meshgrid(layout["qy"], layout["qx"], indexing="ij")
    partition = _make_partitioner(layout, sites, seed)(
        seed, Y, X, CANONICAL_KM, cfg)
    authority["source_partition"] = partition
    selected = layout["selected"]
    expected_plate = np.full(layout["domain_label"].shape, -1, np.int32)
    expected_plate[selected] = layout["domain_plate_by_label"][
        layout["domain_label"][selected]]
    source_plate_consistent = bool(np.array_equal(
        partition[selected], expected_plate[selected]))
    domains = base._domain_summary(layout)
    represented = base._represented_domains(structure)
    resistance_law_exact = bool(np.array_equal(
        layout["resistance"], _resistance(layout["assembly"])))
    geometry = _geometry_diagnostics(
        layout["selected"], layout["domain_id_grid"])
    initial_union_geometry = _periodic_shape_components(
        layout["selected"])
    initial_shape_records = (
        layout["domains"] + initial_union_geometry["components"])
    geometry_tripwires = {
        "no_severe_oriented_rectangle": not any(
            item["severe_rectangle"] for item in initial_shape_records),
        "oriented_shape_coverage_complete_diagnostic": not any(
            item["component_winds_torus"]
            for item in initial_shape_records),
        "maximum_cardinal_boundary_run_km_diagnostic":
            geometry["maximum_cardinal_boundary_run_km"],
        "no_exact_frame_width_repetition": not geometry[
            "exact_nonzero_frame_width_repetition"],
    }
    geometry_tripwires["passed"] = (
        geometry_tripwires["no_severe_oriented_rectangle"]
        and geometry_tripwires["no_exact_frame_width_repetition"])
    formation_invariants = all((
        layout["selected_cells"] == layout["requested_cells"],
        layout["prefix_checks"]["passed"],
        layout["strict_predecessor_paths"],
        domains["all_connected_to_seed"],
        domains["starved_domain_count"] == 0,
        source_plate_consistent,
        resistance_law_exact,
        geometry_tripwires["passed"],
    ))
    ready_gates = {
        "inventory_exact": (
            layout["selected_cells"] == layout["requested_cells"]),
        "formation_invariants": formation_invariants,
        "all_domains_represented": (
            represented["represented_domain_count"]
            == domains["active_domain_count"]),
        "transported_geometry":
            transported_geometry["tripwires"]["passed"],
        "transported_proxy_assignment": selection["found"],
        "assigned_window_components": window_reviews["passed"],
    }
    result = {
        "seed": seed,
        "status": "complete",
        "inventory": {
            "requested_cells": layout["requested_cells"],
            "selected_cells": layout["selected_cells"],
            "prefix_cells": layout["prefix_cells"],
            "world_cells": int(layout["selected"].size),
            "selected_world_fraction": layout["selected_fraction"],
            "prefix_cutoff": layout["prefix_cutoff"],
            "target_cutoff": layout["target_cutoff"],
        },
        "nucleation": {
            "positive_broad_component_count": int(
                np.max(layout["province_raw"]) + 1),
            "active_nucleus_count": len(layout["nuclei"]),
            "nuclei": [{
                "domain_id": item["domain_id"],
                "pivot_yx_km": item["pivot_yx_km"],
                "eligible_craton_cells": item["eligible_craton_cells"],
                "peak_craton": item["peak_craton"],
            } for item in layout["nuclei"]],
        },
        "domains": domains,
        "domain_records": layout["domains"],
        "resistance": layout["resistance_diagnostics"],
        "resistance_causal_check": {
            "pointwise_law_exact": resistance_law_exact,
            "selected_mean_exceeds_unselected_mean_diagnostic": (
                layout["resistance_diagnostics"]["selected_assembly_mean"]
                > layout["resistance_diagnostics"][
                    "unselected_assembly_mean"]),
            "selected_median_exceeds_unselected_median_diagnostic": (
                layout["resistance_diagnostics"][
                    "selected_assembly_median"]
                > layout["resistance_diagnostics"][
                    "unselected_assembly_median"]),
            "outcome_signs_are_not_per_seed_gates": True,
        },
        "prefix_checks": layout["prefix_checks"],
        "source_plate_consistent": source_plate_consistent,
        "geometry_diagnostics": geometry,
        "initial_union_geometry": initial_union_geometry,
        "geometry_tripwires": geometry_tripwires,
        "transported_geometry": transported_geometry,
        "transport": {
            "periodic": bool(structure._periodic),
            "translation_only": bool(structure._translation_only),
            "structure_n": int(structure.n),
            "actual_structure_km": float(
                structure.world_km / structure.n),
            "alive_plates": int(structure.alive_plates),
            "plate_sites_yx_km": np.asarray(sites).tolist(),
            "plate_displacements_yx_km": np.asarray(
                structure._plate_displacements_yx_km).tolist(),
            **represented,
        },
        "canonical_scan": base._scan_report(canonical_scan),
        "transported_proxy_scan": base._scan_report(transported_scan),
        "assigned_window_reviews": window_reviews,
        "transported_global_proxy_fraction": float(
            authority["proxy"].mean()),
        "transported_global_binary_fraction": float(
            authority["binary"].mean()),
        "transported_tag_coverage_diagnostics": {
            "binary_land_without_material_tag_cells": int(np.count_nonzero(
                authority["binary"] & (authority["dominant_tag"] < 0))),
            "material_tag_outside_binary_land_cells": int(np.count_nonzero(
                ~authority["binary"] & (authority["dominant_tag"] >= 0))),
            "binary_land_without_material_tag_fraction_of_world": float(
                np.mean(authority["binary"]
                        & (authority["dominant_tag"] < 0))),
            "material_tag_outside_binary_land_fraction_of_world": float(
                np.mean(~authority["binary"]
                        & (authority["dominant_tag"] >= 0))),
        },
        "ready_gates": ready_gates,
        "ready": all(ready_gates.values()),
    }
    return result, authority, canonical_scan, transported_scan


def _distance_to_d4_axis_degrees(angle_degrees) -> np.ndarray:
    folded = np.mod(np.asarray(angle_degrees, np.float64), 45.0)
    return np.minimum(folded, 45.0 - folded)


def _seed_blocked_d4_randomization(rulers: list[dict]) -> dict:
    """Test D4 ruler lock without pretending within-seed independence.

    Each null draw applies one whole-degree rotation to every initial and
    transported ruler from the same seed.  Relative geometry and all
    within-seed dependence are therefore preserved.
    """
    for item in rulers:
        distance = float(_distance_to_d4_axis_degrees(
            item["angle_degrees"]))
        item["distance_to_d4_axis_degrees"] = distance
        item["within_d4_tolerance"] = (
            distance <= D4_TOLERANCE_DEGREES)
    observed = sum(item["within_d4_tolerance"] for item in rulers)
    if not rulers:
        return {
            "observed_near_d4_count": 0,
            "randomization_upper_tail_p": 1.0,
            "trials": GEOMETRY_BLOCK_RANDOMIZATION_TRIALS,
            "salt": GEOMETRY_BLOCK_RANDOMIZATION_SALT,
            "block": "seed",
        }
    grouped = {}
    for item in rulers:
        grouped.setdefault(int(item["seed"]), []).append(
            float(item["angle_degrees"]))
    rng = np.random.default_rng(
        fnv1a64(GEOMETRY_BLOCK_RANDOMIZATION_SALT))
    at_least_observed = 0
    chunk_size = 5000
    remaining = GEOMETRY_BLOCK_RANDOMIZATION_TRIALS
    while remaining:
        count = min(chunk_size, remaining)
        simulated = np.zeros(count, np.int32)
        for seed in sorted(grouped):
            rotations = rng.integers(0, 180, size=count, dtype=np.int16)
            angles = np.asarray(grouped[seed], np.float64)
            rotated = np.mod(
                rotations[:, None].astype(np.float64)
                + angles[None, :], 180.0)
            simulated += np.count_nonzero(
                _distance_to_d4_axis_degrees(rotated)
                <= D4_TOLERANCE_DEGREES,
                axis=1).astype(np.int32)
        at_least_observed += int(np.count_nonzero(
            simulated >= observed))
        remaining -= count
    p_value = (
        1.0 + at_least_observed
    ) / (GEOMETRY_BLOCK_RANDOMIZATION_TRIALS + 1.0)
    return {
        "observed_near_d4_count": int(observed),
        "randomization_upper_tail_p": float(p_value),
        "trials": GEOMETRY_BLOCK_RANDOMIZATION_TRIALS,
        "salt": GEOMETRY_BLOCK_RANDOMIZATION_SALT,
        "block": "seed",
        "rotation_support_integer_degrees": [0, 179],
    }


def _geometry_cohort(results: list[dict]) -> dict:
    rulers = []
    for result in results:
        for domain in result["domain_records"]:
            angle = domain["maximum_ruler_run_angle_degrees"]
            if (angle is not None
                    and domain["maximum_ruler_run_km"] >= RULER_RUN_KM):
                rulers.append({
                    "seed": result["seed"],
                    "stage": "initial_domain",
                    "domain_id": domain["domain_id"],
                    "length_km": domain["maximum_ruler_run_km"],
                    "angle_degrees": float(angle),
                })
        for component in result[
                "initial_union_geometry"]["components"]:
            angle = component["maximum_ruler_run_angle_degrees"]
            if (angle is not None
                    and component["maximum_ruler_run_km"] >= RULER_RUN_KM):
                rulers.append({
                    "seed": result["seed"],
                    "stage": "initial_union",
                    "component_label": component["component_label"],
                    "length_km": component["maximum_ruler_run_km"],
                    "angle_degrees": float(angle),
                })
        for component in result[
                "transported_geometry"]["union_components"]:
            angle = component["maximum_ruler_run_angle_degrees"]
            if (angle is not None
                    and component["maximum_ruler_run_km"] >= RULER_RUN_KM):
                rulers.append({
                    "seed": result["seed"],
                    "stage": "transported_union",
                    "component_label": component["component_label"],
                    "length_km": component["maximum_ruler_run_km"],
                    "angle_degrees": float(angle),
                })
        for component in result[
                "transported_geometry"]["identity_components"]:
            angle = component["maximum_ruler_run_angle_degrees"]
            if (angle is not None
                    and component["maximum_ruler_run_km"] >= RULER_RUN_KM):
                rulers.append({
                    "seed": result["seed"],
                    "stage": "transported_identity",
                    "material_tag": component["material_tag"],
                    "component_label": component["component_label"],
                    "length_km": component["maximum_ruler_run_km"],
                    "angle_degrees": float(angle),
                })
    d4 = _seed_blocked_d4_randomization(rulers)
    initial_identity_rectangle_count = sum(
        item["severe_rectangle"]
        for result in results for item in result["domain_records"])
    initial_union_rectangle_count = sum(
        item["severe_rectangle"]
        for result in results
        for item in result["initial_union_geometry"]["components"])
    transported_union_rectangle_count = sum(
        item["severe_rectangle"]
        for result in results
        for item in result[
            "transported_geometry"]["union_components"])
    transported_identity_rectangle_count = sum(
        item["severe_rectangle"]
        for result in results
        for item in result[
            "transported_geometry"]["identity_components"])
    initial_exact_repetition_count = sum(
        result["geometry_diagnostics"][
            "exact_nonzero_frame_width_repetition"]
        for result in results)
    transported_exact_repetition_count = sum(
        result["transported_geometry"]["union_geometry_diagnostics"][
            "exact_nonzero_frame_width_repetition"]
        for result in results)
    rectangle_count = (
        initial_identity_rectangle_count
        + initial_union_rectangle_count
        + transported_union_rectangle_count
        + transported_identity_rectangle_count)
    winding_component_count = sum(
        item["component_winds_torus"]
        for result in results
        for item in (
            result["domain_records"]
            + result["initial_union_geometry"]["components"]
            + result["transported_geometry"]["union_components"]
            + result["transported_geometry"]["identity_components"]))
    exact_repetition_count = (
        initial_exact_repetition_count
        + transported_exact_repetition_count)
    p_value = d4["randomization_upper_tail_p"]
    return {
        "ruler_run_baseline_km": RULER_RUN_KM,
        "ruler_runs": rulers,
        "ruler_run_count": len(rulers),
        "d4_tolerance_degrees": D4_TOLERANCE_DEGREES,
        "d4_seed_blocked_randomization": d4,
        "d4_systematic_lock_rejected_at_family_alpha_0p01":
            p_value < 0.01,
        "initial_identity_severe_rectangle_count":
            initial_identity_rectangle_count,
        "initial_union_severe_rectangle_count":
            initial_union_rectangle_count,
        "transported_union_severe_rectangle_count":
            transported_union_rectangle_count,
        "transported_identity_severe_rectangle_count":
            transported_identity_rectangle_count,
        "severe_rectangle_count": rectangle_count,
        "shape_gate_incomplete_winding_component_count":
            winding_component_count,
        "initial_exact_frame_width_repetition_seed_count":
            initial_exact_repetition_count,
        "transported_exact_frame_width_repetition_seed_count":
            transported_exact_repetition_count,
        "exact_frame_width_repetition_seed_count": exact_repetition_count,
        "passed": (
            p_value >= 0.01
            and rectangle_count == 0
            and exact_repetition_count == 0),
    }


def _frame_profile_records(result: dict) -> list[dict]:
    records = []
    diagnostics = (
        ("initial", result["geometry_diagnostics"]),
        ("transported", result["transported_geometry"][
            "union_geometry_diagnostics"]),
    )
    for stage, geometry in diagnostics:
        for channel, prefix in (
                ("occupancy", ""), ("identity", "identity_")):
            for axis in ("vertical", "horizontal"):
                profile = np.asarray(geometry[
                    f"{prefix}{axis}_transition_profile"], np.float64)
                total = float(profile.sum())
                if total <= 0.0:
                    continue
                records.append({
                    "stage": stage,
                    "channel": channel,
                    "axis": axis,
                    "profile": profile / total,
                })
    return records


def _frame_phase_score(records: list[dict], dy: int, dx: int,
                       frame_frequency: int) -> float:
    best = 0.0
    n = records[0]["profile"].size if records else 1
    for record in records:
        coefficient = record["frame_coefficient"]
        offset = dx if record["axis"] == "vertical" else dy
        for delta in (-1, 0, 1):
            phase = np.exp(
                -2j * np.pi * frame_frequency
                * (offset + delta) / n)
            best = max(best, abs(float(np.real(coefficient * phase))))
    return float(best)


def _frame_lock_cohort(results: list[dict]) -> dict:
    """Seed-blocked frame-phase and exact-frame-scale resonance tests."""
    frame_cells = int(round(FRAME_KM / CANONICAL_KM))
    world_cells = int(round(PARENT_KM / CANONICAL_KM))
    frame_frequency = world_cells // frame_cells
    if world_cells % frame_cells:
        raise AssertionError("frame scale is not an exact torus frequency")
    phase_period = frame_cells
    records_by_seed = {
        int(result["seed"]): _frame_profile_records(result)
        for result in results
    }
    for records in records_by_seed.values():
        for record in records:
            spectrum = np.fft.rfft(record["profile"])
            record["frame_coefficient"] = spectrum[frame_frequency]
            record["power"] = np.abs(spectrum) ** 2
    ordered_seeds = sorted(records_by_seed)

    phase_tables = np.zeros(
        (len(ordered_seeds), phase_period, phase_period), np.float64)
    phase_seed_records = []
    for seed_index, seed in enumerate(ordered_seeds):
        records = records_by_seed[seed]
        for dy in range(phase_period):
            for dx in range(phase_period):
                phase_tables[seed_index, dy, dx] = _frame_phase_score(
                    records, dy, dx, frame_frequency)
        observed_seed = float(phase_tables[seed_index, 0, 0])
        percentile95 = float(np.quantile(
            phase_tables[seed_index], 0.95, method="higher"))
        null_minimum = float(phase_tables[seed_index].min())
        null_maximum = float(phase_tables[seed_index].max())
        positive_nonflat = (
            observed_seed > 0.0
            and null_maximum > null_minimum + 1e-15)
        phase_seed_records.append({
            "seed": seed,
            "observed_score": observed_seed,
            "translation_null_95th_percentile": percentile95,
            "translation_null_minimum": null_minimum,
            "translation_null_maximum": null_maximum,
            "positive_nonflat_signal": positive_nonflat,
            "at_or_above_95th_percentile":
                positive_nonflat and observed_seed >= percentile95,
            "nonempty_profile_count": len(records),
        })
    observed_phase = float(phase_tables[:, 0, 0].sum())
    phase_trials = 65536
    phase_salt = "periodic-frame-phase-null-v1"
    phase_rng = np.random.Generator(np.random.PCG64(fnv1a64(phase_salt)))
    phase_offsets = phase_rng.integers(
        0, phase_period,
        size=(phase_trials, len(ordered_seeds), 2),
        dtype=np.int16)
    phase_null = np.zeros(phase_trials, np.float64)
    for seed_index in range(len(ordered_seeds)):
        phase_null += phase_tables[
            seed_index,
            phase_offsets[:, seed_index, 0],
            phase_offsets[:, seed_index, 1],
        ]
    phase_p_raw = float((
        1 + np.count_nonzero(phase_null >= observed_phase)
    ) / (phase_trials + 1))
    phase_effect_count = sum(
        item["at_or_above_95th_percentile"]
        for item in phase_seed_records)
    phase_effect_eligible = phase_effect_count >= 6
    phase_evidence = phase_effect_eligible and phase_p_raw < 0.01
    phase_table_sha256 = _sha256_bytes(
        np.ascontiguousarray(phase_tables).tobytes())
    phase_offset_sha256 = _sha256_bytes(
        np.ascontiguousarray(phase_offsets).tobytes())
    packed_offsets = (
        phase_offsets[:, :, 0].astype(np.int32) * phase_period
        + phase_offsets[:, :, 1].astype(np.int32))
    minimum_unique_joint_pairs = min(
        np.unique(
            (packed_offsets[:, left].astype(np.int64) << 12)
            | packed_offsets[:, right].astype(np.int64)).size
        for left in range(len(ordered_seeds))
        for right in range(left + 1, len(ordered_seeds)))
    offset_correlation = np.corrcoef(packed_offsets, rowvar=False)
    maximum_absolute_inter_seed_correlation = float(np.max(np.abs(
        offset_correlation - np.eye(len(ordered_seeds)))))

    candidate_frequencies = np.arange(4, 9, dtype=np.int32)
    resonance = np.zeros(
        (len(ordered_seeds), candidate_frequencies.size), np.float64)
    scale_seed_records = []
    for seed_index, seed in enumerate(ordered_seeds):
        for frequency_index, frequency in enumerate(candidate_frequencies):
            best = 0.0
            for record in records_by_seed[seed]:
                power = record["power"]
                controls = power[[
                    frequency - 2, frequency - 1,
                    frequency + 1, frequency + 2,
                ]]
                ratio = float(
                    power[frequency]
                    / (np.median(controls) + 1e-12))
                best = max(best, ratio)
            resonance[seed_index, frequency_index] = best
        frame_ratio = float(resonance[seed_index, 2])
        scale_seed_records.append({
            "seed": seed,
            "maximum_ratio_by_candidate_frequency": {
                str(int(frequency)): float(value)
                for frequency, value in zip(
                    candidate_frequencies, resonance[seed_index])
            },
            "frame_frequency_ratio_at_least_4": frame_ratio >= 4.0,
        })
    log_resonance = np.log(np.maximum(1.0, resonance))
    observed_scale = float(log_resonance[:, 2].sum())
    exact_assignments = int(candidate_frequencies.size ** len(ordered_seeds))
    assignment_index = np.arange(exact_assignments, dtype=np.int64)
    null_scale = np.zeros(exact_assignments, np.float64)
    place = 1
    for seed_index in range(len(ordered_seeds)):
        choice = (assignment_index // place) % candidate_frequencies.size
        null_scale += log_resonance[seed_index, choice]
        place *= candidate_frequencies.size
    scale_reference_tail_fraction = float(np.count_nonzero(
        null_scale >= observed_scale - 1e-12) / exact_assignments)
    scale_effect_count = sum(
        item["frame_frequency_ratio_at_least_4"]
        for item in scale_seed_records)
    scale_effect_eligible = scale_effect_count >= 6
    scale_corroboration = (
        scale_effect_eligible and scale_reference_tail_fraction < 0.01)
    systematic_frame_stamping = phase_evidence and scale_corroboration
    return {
        "frame_cells": frame_cells,
        "frame_frequency_cycles_per_world": frame_frequency,
        "profiles_per_seed_maximum": 8,
        "pairing": (
            "initial/transported occupancy/identity and both axes remain "
            "inside one seed block"),
        "phase_endpoint": {
            "observed_sum": observed_phase,
            "trials": phase_trials,
            "translation_residues": [0, phase_period - 1],
            "one_cell_convention_allowance": [-1, 0, 1],
            "salt": phase_salt,
            "null_table_sha256": phase_table_sha256,
            "translation_offset_generator": (
                "NumPy PCG64 independently draws dy/dx per trial and "
                "seed block"),
            "translation_offset_table_sha256": phase_offset_sha256,
            "minimum_unique_joint_seed_translation_pairs": int(
                minimum_unique_joint_pairs),
            "maximum_absolute_inter_seed_packed_offset_correlation":
                maximum_absolute_inter_seed_correlation,
            "raw_randomization_upper_tail_p": phase_p_raw,
            "effect_seed_count": phase_effect_count,
            "effect_requires_at_least_6_of_8": phase_effect_eligible,
            "randomization_alpha": 0.01,
            "canonical_phase_evidence": phase_evidence,
            "seeds": phase_seed_records,
        },
        "scale_endpoint": {
            "candidate_frequencies": candidate_frequencies.tolist(),
            "observed_sum_log_max1_ratio": observed_scale,
            "exact_assignment_count": exact_assignments,
            "candidate_frequency_reference_tail_fraction":
                scale_reference_tail_fraction,
            "candidate_frequencies_are_not_exchangeable_null_draws": True,
            "effect_seed_count": scale_effect_count,
            "effect_requires_at_least_6_of_8_with_ratio_at_least_4":
                scale_effect_eligible,
            "reference_tail_threshold": 0.01,
            "scale_corroboration": scale_corroboration,
            "seeds": scale_seed_records,
        },
        "causal_veto_requires_phase_evidence_and_scale_corroboration": True,
        "systematic_frame_stamping_detected": systematic_frame_stamping,
        "passed": not systematic_frame_stamping,
    }


def _interpolate_rgb(field: np.ndarray, low: float, high: float,
                     colors: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    value = np.clip(
        (np.asarray(field, np.float64) - low) / (high - low), 0.0, 1.0)
    positions = value * (len(colors) - 1)
    left = np.floor(positions).astype(np.int32)
    right = np.minimum(left + 1, len(colors) - 1)
    fraction = positions - left
    palette = np.asarray(colors, np.float64)
    rgb = (
        palette[left] * (1.0 - fraction[..., None])
        + palette[right] * fraction[..., None])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _diverging_rgb(field: np.ndarray, limit: float) -> np.ndarray:
    return _interpolate_rgb(field, -limit, limit, (
        (16, 45, 105), (70, 145, 185), (232, 231, 211),
        (205, 118, 70), (116, 37, 45),
    ))


def _sequential_rgb(field: np.ndarray, low: float,
                    high: float) -> np.ndarray:
    return _interpolate_rgb(field, low, high, (
        (7, 18, 42), (25, 82, 118), (55, 155, 145),
        (179, 194, 123), (242, 225, 166),
    ))


def _label_rgb(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    rgb = np.zeros((*labels.shape, 3), np.uint8)
    rgb[:] = np.asarray((8, 25, 47), np.uint8)
    for label in np.unique(labels[labels >= 0]):
        rgb[labels == label] = base._palette(int(label))
    return rgb


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, bool)
    interior = mask.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        interior &= np.roll(mask, (dy, dx), (0, 1))
    return mask & ~interior


def _label_boundaries(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return (
        (labels != np.roll(labels, 1, axis=0))
        | (labels != np.roll(labels, 1, axis=1)))


def _resize_rgb(rgb: np.ndarray, size: int) -> np.ndarray:
    if rgb.shape[:2] == (size, size):
        return np.asarray(rgb, np.uint8).copy()
    image = Image.fromarray(np.asarray(rgb, np.uint8), "RGB")
    image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image, np.uint8).copy()


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape == (size, size):
        return np.asarray(mask, bool).copy()
    image = Image.fromarray(
        np.asarray(mask, np.uint8) * 255, "L")
    image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def _draw_periodic_segments(
        rgb: np.ndarray,
        segments: list[tuple[list, tuple[int, int, int], int]]) -> np.ndarray:
    if not segments:
        return np.asarray(rgb, np.uint8).copy()
    n = rgb.shape[0]
    tiled = np.tile(np.asarray(rgb, np.uint8), (3, 3, 1))
    image = Image.fromarray(tiled, "RGB")
    draw = ImageDraw.Draw(image)
    for endpoints, color, width in segments:
        if endpoints is None:
            continue
        y0, x0 = map(float, endpoints[0])
        y1, x1 = map(float, endpoints[1])
        dy = float(y1 - y0)
        dx = float(x1 - x0)
        base_y = y0 % n
        base_x = x0 % n
        for tile_y in range(-1, 4):
            for tile_x in range(-1, 4):
                start_x = base_x + tile_x * n
                start_y = base_y + tile_y * n
                draw.line((start_x, start_y,
                           start_x + dx, start_y + dy),
                          fill=color, width=width)
    return np.asarray(image.crop((n, n, 2 * n, 2 * n)), np.uint8)


def _rectangle_segments(record: dict, color, width=2) -> list:
    corners = record.get("rectangle_corners_yx_unwrapped_cells")
    if corners is None:
        return []
    return [
        ([corners[index], corners[(index + 1) % 4]], color, width)
        for index in range(4)
    ]


def _mark_points(rgb: np.ndarray, points_yx: list,
                 color=(255, 245, 170), radius=2) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    n = rgb.shape[0]
    for y, x in points_yx:
        y = float(y) % n
        x = float(x) % n
        draw.line((x - radius, y, x + radius, y),
                  fill=color, width=1)
        draw.line((x, y - radius, x, y + radius),
                  fill=color, width=1)
    return np.asarray(image, np.uint8)


def _render_audit_panel(seed: int, layout: dict, structure,
                        authority: dict, cfg, sites: np.ndarray,
                        result: dict, out: Path) -> dict:
    """Fixed-scale evidence view for formation and transported structure."""
    n = layout["selected"].shape[0]
    tiles = []

    assembly = _diverging_rgb(layout["broad_assembly"], 0.65)
    active_provinces = {
        int(item["province_raw_label"]) for item in layout["nuclei"]}
    for raw in np.unique(layout["province_raw"][
            layout["province_raw"] >= 0]):
        outline = _mask_outline(layout["province_raw"] == raw)
        assembly[outline] = (
            (255, 232, 120) if int(raw) in active_provinces
            else (255, 80, 205))
    tiles.append((
        "broad assembly +/-0.65; eligible yellow, skipped magenta",
        assembly))

    craton = _diverging_rgb(layout["craton"], 0.65)
    threshold = layout["craton"] > legacy.NUCLEUS_CRATON_THRESHOLD
    craton[_mask_outline(threshold)] = (240, 240, 230)
    craton = _mark_points(craton, [
        item["storage_yx"] for item in layout["nuclei"]])
    tiles.append(("craton +/-0.65; >0.20 outline; nuclei", craton))

    log_resistance = np.log(np.maximum(layout["resistance"], 1e-12))
    resistance = _sequential_rgb(log_resistance, -2.0, 2.5)
    selected = layout["selected"]
    normalized_arrival = np.zeros(selected.shape, np.float64)
    normalized_arrival[selected] = np.clip(
        layout["arrival"][selected]
        / max(float(layout["target_cutoff"]["arrival"]), 1e-12),
        0.0, 1.0)
    arrival_band = np.floor(4.0 * normalized_arrival).astype(np.int8)
    isoline = selected & _label_boundaries(arrival_band)
    resistance[isoline] = (145, 145, 140)
    resistance[_mask_outline(layout["prefix_selected"])] = (45, 235, 245)
    resistance[_mask_outline(selected)] = (250, 250, 240)
    initial_union_labels, _ = _periodic_components(selected)
    for item in layout["domains"]:
        if item["component_winds_torus"]:
            resistance[_mask_outline(
                layout["domain_label"] == item["label"])] = (
                    185, 70, 255)
    for item in result["initial_union_geometry"]["components"]:
        if item["component_winds_torus"]:
            resistance[_mask_outline(
                initial_union_labels == item["component_label"])] = (
                    185, 70, 255)
    initial_segments = [
        (item["maximum_ruler_run_endpoint_yx_unwrapped_cells"],
         (50, 225, 245), 2)
        for item in layout["domains"]
        if item["maximum_ruler_run_km"] >= RULER_RUN_KM
    ]
    initial_segments.extend([
        (item["maximum_ruler_run_endpoint_yx_unwrapped_cells"],
         (255, 190, 40), 2)
        for item in result["initial_union_geometry"]["components"]
        if item["maximum_ruler_run_km"] >= RULER_RUN_KM
    ])
    for item in layout["domains"]:
        if item["severe_rectangle"]:
            initial_segments.extend(_rectangle_segments(
                item, (255, 70, 220), 2))
    for item in result["initial_union_geometry"]["components"]:
        if item["severe_rectangle"]:
            initial_segments.extend(_rectangle_segments(
                item, (255, 45, 45), 2))
    resistance = _draw_periodic_segments(resistance, initial_segments)
    tiles.append((
        "gray=arrival Q overlay; rulers union Y/identity C; OBB",
        resistance))

    partition = _label_rgb(authority["source_partition"])
    partition[_label_boundaries(authority["source_partition"])] = (
        5, 8, 12)
    displacements = np.asarray(
        structure._plate_displacements_yx_km, np.float64)
    plate_segments = []
    site_points = []
    for plate_id, site in enumerate(np.asarray(sites, np.float64)):
        start = [
            site[0] / CANONICAL_KM - 0.5,
            site[1] / CANONICAL_KM - 0.5,
        ]
        delta = (
            (displacements[plate_id] + PARENT_KM / 2.0) % PARENT_KM
            - PARENT_KM / 2.0) / CANONICAL_KM
        end = [start[0] + delta[0], start[1] + delta[1]]
        plate_segments.append((
            [start, end], tuple(base._palette(plate_id).tolist()), 1))
        site_points.append(start)
    partition = _draw_periodic_segments(partition, plate_segments)
    partition = _mark_points(
        partition, site_points, color=(250, 250, 240), radius=1)
    tiles.append(("source partition; sites + final displacement", partition))

    transported = _sequential_rgb(authority["proxy"], 0.0, 1.0)
    transported[~authority["binary"]] = np.asarray(
        (8, 25, 47), np.uint8)
    for tag in np.unique(authority["dominant_tag"]):
        if tag < 0:
            continue
        mask = (
            authority["binary"]
            & (authority["dominant_tag"] == tag))
        color = base._palette(int(tag)).astype(np.float64)
        shade = 0.35 + 0.65 * authority["proxy"][mask]
        transported[mask] = np.clip(
            shade[:, None] * color[None, :], 0, 255).astype(np.uint8)
    untagged_land = (
        authority["binary"] & (authority["dominant_tag"] < 0))
    transported[untagged_land] = (225, 225, 220)
    transported[_mask_outline(authority["binary"])] = (245, 245, 235)
    transported_union_labels, _ = _periodic_components(
        authority["binary"])
    for item in result["transported_geometry"]["union_components"]:
        if item["component_winds_torus"]:
            transported[_mask_outline(
                transported_union_labels == item["component_label"])] = (
                    185, 70, 255)
    for item in result["transported_geometry"]["identity_components"]:
        if not item["component_winds_torus"]:
            continue
        tag_mask = (
            authority["binary"]
            & (authority["dominant_tag"] == item["material_tag"]))
        labels, _ = _periodic_components(tag_mask)
        transported[_mask_outline(
            labels == item["component_label"])] = (185, 70, 255)
    transported_segments = [
        (item["maximum_ruler_run_endpoint_yx_unwrapped_cells"],
         (255, 190, 40), 2)
        for item in result[
            "transported_geometry"]["union_components"]
        if item["maximum_ruler_run_km"] >= RULER_RUN_KM
    ]
    transported_segments.extend([
        (item["maximum_ruler_run_endpoint_yx_unwrapped_cells"],
         (50, 225, 245), 2)
        for item in result[
            "transported_geometry"]["identity_components"]
        if item["maximum_ruler_run_km"] >= RULER_RUN_KM
    ])
    for item in result["transported_geometry"]["union_components"]:
        if item["severe_rectangle"]:
            transported_segments.extend(_rectangle_segments(
                item, (255, 45, 45), 2))
    for item in result["transported_geometry"]["identity_components"]:
        if item["severe_rectangle"]:
            transported_segments.extend(_rectangle_segments(
                item, (255, 70, 220), 2))
    transported = _draw_periodic_segments(
        transported, transported_segments)
    tiles.append((
        "cont_frac; union yellow/identity cyan; winding violet; OBB",
        transported))

    final_labels = _resize_rgb(_label_rgb(structure.label), n)
    final_boundaries = _resize_mask(
        _label_boundaries(structure.label), n)
    final_labels[final_boundaries] = (5, 8, 12)
    material_boundaries = _label_boundaries(authority["dominant_tag"])
    final_labels[material_boundaries & authority["binary"]] = (
        250, 250, 235)
    tiles.append(("final plate labels; material boundaries white", final_labels))

    age = _sequential_rgb(
        structure.age_myr, 0.0, float(structure.eras * DT_MYR))
    age[structure.conv_recent] = (255, 55, 55)
    age[structure.div_recent] = (45, 225, 245)
    tiles.append((
        f"ocean age [0,{structure.eras * DT_MYR:.0f}] Myr; conv/div",
        _resize_rgb(age, n)))

    belt = _sequential_rgb(np.log1p(structure.belt), 0.0, np.log1p(8.0))
    belt[structure.coast] = (245, 245, 235)
    belt[structure.active_margin] = (255, 60, 220)
    belt[structure.passive_margin] = (255, 220, 55)
    tiles.append(("log1p belt [0,log9]; coast/active/passive", _resize_rgb(
        belt, n)))

    header = 24
    canvas = Image.new("RGB", (4 * n, 2 * (n + header)), (5, 13, 24))
    draw = ImageDraw.Draw(canvas)
    for index, (title, rgb) in enumerate(tiles):
        column = index % 4
        row = index // 4
        x0 = column * n
        y0 = row * (n + header)
        draw.text((x0 + 5, y0 + 5),
                  f"{title}; seam=center",
                  fill=(238, 238, 226))
        seam_centered = np.roll(
            rgb, (n // 2, n // 2), axis=(0, 1))
        canvas.paste(
            Image.fromarray(seam_centered, "RGB"),
            (x0, y0 + header))
    path = out / f"seed{seed}_formation_structure_audit.png"
    canvas.save(path)
    return {"file": path.name, "sha256": _sha256_file(path)}


def _draw_plot(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
               series: list[tuple[np.ndarray, tuple[int, int, int]]],
               low: float, high: float, vertical_marker=None) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=(85, 95, 105), width=1)
    if vertical_marker is not None:
        marker_x = x0 + vertical_marker[0] / vertical_marker[1] * (x1 - x0)
        draw.line((marker_x, y0, marker_x, y1),
                  fill=(255, 180, 50), width=1)
    for values, color in series:
        values = np.asarray(values, np.float64)
        if values.size < 2:
            continue
        clipped = np.clip(values, low, high)
        points = [
            (x0 + index / (values.size - 1) * (x1 - x0),
             y1 - (value - low) / max(high - low, 1e-12) * (y1 - y0))
            for index, value in enumerate(clipped)
        ]
        draw.line(points, fill=color, width=1)


def _render_geometry_spectrum_panel(seed: int, result: dict,
                                    out: Path) -> dict:
    width, height = 960, 520
    canvas = Image.new("RGB", (width, height), (6, 16, 28))
    draw = ImageDraw.Draw(canvas)
    colors = {
        ("occupancy", "vertical"): (255, 155, 70),
        ("occupancy", "horizontal"): (55, 215, 245),
        ("identity", "vertical"): (250, 85, 205),
        ("identity", "horizontal"): (105, 235, 125),
    }
    diagnostics = (
        ("initial", result["geometry_diagnostics"]),
        ("transported", result["transported_geometry"][
            "union_geometry_diagnostics"]),
    )
    for column, (stage, geometry) in enumerate(diagnostics):
        records = []
        for channel, prefix in (("occupancy", ""), ("identity", "identity_")):
            for axis in ("vertical", "horizontal"):
                profile = np.asarray(geometry[
                    f"{prefix}{axis}_transition_profile"], np.float64)
                records.append((channel, axis, profile))
        x0 = 20 + column * 480
        draw.text((x0, 10),
                  f"seed {seed} {stage}: boundary phase profiles",
                  fill=(238, 238, 226))
        maximum = max(1.0, max(float(item[2].max()) for item in records))
        _draw_plot(draw, (x0, 32, x0 + 440, 235), [
            (profile, colors[(channel, axis)])
            for channel, axis, profile in records
        ], 0.0, maximum)
        spectra = []
        for channel, axis, profile in records:
            if profile.sum() <= 0:
                power = np.full(16, -12.0)
            else:
                normalized = profile / profile.sum()
                raw = np.abs(np.fft.rfft(normalized)) ** 2
                power = np.log10(np.maximum(raw[1:17], 1e-12))
            spectra.append((power, colors[(channel, axis)]))
        draw.text((x0, 247),
                  "log10 normalized power, k=1..16; k=6 marker",
                  fill=(238, 238, 226))
        _draw_plot(draw, (x0, 270, x0 + 440, 492), spectra,
                   -12.0, 0.0, vertical_marker=(5, 15))
    draw.text((20, 500),
              "orange/cyan occupancy V/H; magenta/green identity V/H",
              fill=(190, 200, 205))
    path = out / f"seed{seed}_geometry_phase_spectrum.png"
    canvas.save(path)
    return {"file": path.name, "sha256": _sha256_file(path)}


def _render_periodic_panel(seed: int, layout: dict, authority: dict,
                           out: Path) -> dict:
    n = layout["selected"].shape[0]
    initial = np.zeros((n, n, 3), np.uint8)
    initial[:] = np.asarray((8, 25, 47), np.uint8)
    for domain in layout["domains"]:
        mask = layout["domain_label"] == domain["label"]
        initial[mask] = base._palette(domain["label"])
    transported = np.zeros((n, n, 3), np.uint8)
    transported[:] = np.asarray((8, 25, 47), np.uint8)
    for tag in np.unique(authority["dominant_tag"]):
        if tag < 0:
            continue
        mask = (
            authority["binary"]
            & (authority["dominant_tag"] == tag))
        color = base._palette(int(tag)).astype(np.float64)
        shade = 0.45 + 0.55 * authority["proxy"][mask]
        transported[mask] = np.clip(
            shade[:, None] * color[None, :], 0, 255).astype(np.uint8)
    transported[
        authority["binary"] & (authority["dominant_tag"] < 0)
    ] = (225, 225, 220)
    initial_tile = np.tile(initial, (3, 3, 1))
    transported_tile = np.tile(transported, (3, 3, 1))
    canvas = Image.new("RGB", (6 * n, 3 * n + 44), (5, 13, 24))
    canvas.paste(Image.fromarray(initial_tile, "RGB"), (0, 44))
    canvas.paste(Image.fromarray(transported_tile, "RGB"), (3 * n, 44))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10),
              f"seed {seed} initial torus - unmarked 3x3 continuity",
              fill=(238, 238, 226))
    draw.text((3 * n + 8, 10),
              "transported authority - unmarked 3x3 continuity",
              fill=(238, 238, 226))
    # Header-only ticks identify tile joins without painting over the data.
    for side in (0, 3 * n):
        for index in (1, 2):
            coordinate = index * n
            draw.line((side + coordinate, 38,
                       side + coordinate, 43),
                      fill=(235, 235, 220), width=1)
    path = out / f"seed{seed}_periodic_3x3.png"
    canvas.save(path)
    return {"file": path.name, "sha256": _sha256_file(path)}


def _render_assignment_windows(seed: int, authority: dict,
                               result: dict, out: Path) -> dict:
    selection = result["transported_proxy_scan"]["selection"]
    if not selection["found"]:
        canvas = Image.new("RGB", (768, 160), (6, 16, 28))
        ImageDraw.Draw(canvas).text(
            (12, 12), f"seed {seed}: no periodic assignment",
            fill=(238, 238, 226))
    else:
        frame_cells = int(round(FRAME_KM / CANONICAL_KM))
        tag_tiled = np.tile(authority["dominant_tag"], (2, 2))
        binary_tiled = np.tile(authority["binary"], (2, 2))
        panel_size = 384
        canvas = Image.new("RGB", (3 * panel_size, panel_size + 48),
                           (6, 16, 28))
        draw = ImageDraw.Draw(canvas)
        for index, label in enumerate(("low", "medium", "high")):
            candidate = selection["assignment"][label]
            y0 = int(round(candidate["y0_km"] / CANONICAL_KM))
            x0 = int(round(candidate["x0_km"] / CANONICAL_KM))
            binary = binary_tiled[
                y0:y0 + frame_cells, x0:x0 + frame_cells]
            tags = tag_tiled[
                y0:y0 + frame_cells, x0:x0 + frame_cells]
            rgb = np.zeros((frame_cells, frame_cells, 3), np.uint8)
            rgb[:] = np.asarray((11, 39, 72), np.uint8)
            for tag in np.unique(tags[binary & (tags >= 0)]):
                rgb[binary & (tags == tag)] = base._palette(int(tag))
            image = Image.fromarray(rgb, "RGB").resize(
                (panel_size, panel_size), Image.Resampling.NEAREST)
            canvas.paste(image, (index * panel_size, 48))
            review = result["assigned_window_reviews"]["windows"][label]
            draw.text(
                (index * panel_size + 6, 10),
                (f"{label} {candidate['continental_fraction']:.2%} "
                 f"sig={review['significant_component_count']} "
                 f"wrap=({int(candidate['wraps_y'])},"
                 f"{int(candidate['wraps_x'])})"),
                fill=(238, 238, 226))
    path = out / f"seed{seed}_periodic_assignment.png"
    canvas.save(path)
    return {"file": path.name, "sha256": _sha256_file(path)}


def _protocol() -> dict:
    cfg = legacy._atlas_config(TARGET_INITIAL_CONTINENTAL_FRACTION)
    return {
        "experiment": EXPERIMENT,
        "manifest_role": f"exclusive_periodic_precommit_{RUN_ROLE}",
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
        },
        "scope": {
            "claim": "periodic formation and structural transport only",
            "elevation_builds": 0,
            "surface_process_solves": 0,
            "final_water_border_claim": False,
            "bathymetry_topography_hydrology_claim": False,
        },
        "topology": {
            "world_km": PARENT_KM,
            "canonical_km": CANONICAL_KM,
            "topology": "flat square torus",
            "intrinsic_periodicity": True,
            "forbidden_constructions": [
                "edge copying", "edge blending", "edge fading", "mirroring",
                "tiling a smaller generated patch", "forced-water rim",
                "crop-relative modifier",
            ],
            "diagnostic_3x3_render_is_post_authority_only": True,
        },
        "formation": {
            "target_initial_continental_fraction":
                TARGET_INITIAL_CONTINENTAL_FRACTION,
            "diagnostic_prefix_fraction":
                PREFIX_INITIAL_CONTINENTAL_FRACTION,
            "inventory": (
                "exact 14% and 28% snapshots from one global chronology"),
            "field": (
                "intrinsic Fourier torus modes in one-bin radial raised-"
                "cosine shells; random phases; octave standard deviation "
                "0.30; no square gradient lattice; finite square reciprocal "
                "lattice D4 symmetry remains; systematic lock of long land-"
                "boundary rulers is cohort-tested and remaining texture is "
                "manual, not called exactly isotropic"),
            "assembly_nominal_wavelengths_km": [5000.0, 2500.0, 1250.0],
            "craton_nominal_wavelengths_km": [1800.0, 900.0, 450.0],
            "broad_nucleus_rule": (
                "one strongest craton>0.20 point per toroidally connected "
                "positive first-octave assembly province that contains an "
                "eligible craton cell; skipped provinces are rendered"),
            "resistance_law": (
                "rho(a)=exp(-2.5*(a-0.12))/0.72 on every cell"),
            "resistance_acceptance": (
                "pointwise law equality and strict predecessor chronology "
                "are automatic; selected-vs-unselected mean and median are "
                "reported outcome diagnostics, not causal per-seed gates"),
            "travel_time_solver": (
                "periodic multi-source Cartesian first-order fast marching; "
                "owner-specific upwind update; systematic D4 lock of long "
                "land-boundary rulers is cohort-tested and remaining D4 "
                "texture is manual"),
            "support": "every canonical cell",
            "crop_border_elevation_sea_level_or_target_input": None,
        },
        "structure": {
            "configured_plates": int(cfg.plates),
            "partition": (
                "full-domain proposals, minimum-image maximin sites, "
                "periodic spectral deformation, minimum-image Voronoi, "
                "selected source-domain overlay"),
            "transport": (
                "private translation-only torus with wrapped material reads, "
                "collision, divergence, event neighborhoods, age smoothing, "
                "coast detection, and authority sampling"),
            "arbitrary_plate_rotation": False,
            "reason_rotation_absent": (
                "general Euclidean rotation is not a continuous map of a "
                "square flat torus"),
            "coarse_nominal_km": STRUCTURE_NOMINAL_KM,
            "fine_sentinel_nominal_km": FINE_STRUCTURE_NOMINAL_KM,
        },
        "recut_authority": {
            "formation_cut_cells_yx": list(FORMATION_RECUT_CELLS_YX),
            "formation_rerun_each_seed": True,
            "structure_cut_cells_yx": list(STRUCTURE_RECUT_CELLS_YX),
            "structure_rerun_seed": FINE_SENTINEL_SEED,
            "required": (
                "inverse-rolled fields, nuclei, prefix/target masks, stable "
                "owners, selected arrivals, partition, transport, coast, "
                "events, ages, and material tags are exact"),
            "persisted_authority": (
                "paired canonical/raw-recut formation eligible-nucleus and "
                "nucleus masks plus every compared structural field, "
                "material-tag tensor, and displacement table are stored in "
                "the sentinel NPZ; all other compared formation arrays are "
                "paired in every seed NPZ"),
            "rolling_completed_output_is_not_the_test": True,
        },
        "post_transport_scan": {
            "starts_after_authority_is_frozen": True,
            "delivered_frame_km": FRAME_KM,
            "periodic_origins": 96 * 96,
            "stride_km": base.CANDIDATE_STRIDE_KM,
            "wrapping_windows_allowed": True,
            "toroidal_assignment_separation":
                base.MIN_ORIGIN_CHEBYSHEV_SEPARATION_KM,
            "target_bands": {
                label: {
                    "minimum": band[0], "maximum": band[1],
                    "maximum_inclusive": band[2],
                }
                for label, band in base.TARGET_BANDS.items()
            },
            "water_or_contour_input": None,
        },
        "geometry_gates": {
            "component_channels": (
                "initial visible-union plus owner identities and transported "
                "visible-union plus material identities"),
            "minimum_shape_analysis_component_cells":
                RULER_COMPONENT_MIN_CELLS,
            "severe_rectangle": {
                "minimum_component_cells": SUBSTANTIAL_COMPONENT_CELLS,
                "minimum_oriented_box_fill": RECTANGLE_FILL_MIN,
                "minimum_boundary_near_sides_fraction":
                    RECTANGLE_SIDE_COVERAGE_MIN,
                "side_tolerance_cells": RECTANGLE_SIDE_TOLERANCE_CELLS,
                "automatic_veto": True,
            },
            "ruler_diagnostics": {
                "minimum_length_km": RULER_RUN_KM,
                "orthogonal_tolerance_cells": 1.0,
                "single_natural_run_is_not_an_automatic_veto": True,
                "systematic_d4_lock_seed_blocked_randomization_trials":
                    GEOMETRY_BLOCK_RANDOMIZATION_TRIALS,
                "systematic_d4_lock_randomization_salt":
                    GEOMETRY_BLOCK_RANDOMIZATION_SALT,
                "d4_family_alpha": 0.01,
                "d4_near_axis_degrees": D4_TOLERANCE_DEGREES,
                "seed_block_preserves": (
                    "all initial/transported ruler angles within a seed"),
            },
            "toroidal_winding_components": {
                "bfs_unwrapped_with_cycle_detection": True,
                "euclidean_ruler_and_obb_skipped_when_winding": True,
                "automatic_veto": False,
                "manual_review_required": True,
                "render_color": "violet",
                "reason": (
                    "winding is topologically natural and is not itself "
                    "evidence of crop-border causality"),
            },
            "frame_phase_and_scale_family": {
                "profiles": (
                    "initial and transported occupancy and internal-identity "
                    "vertical/horizontal boundary profiles"),
                "frame_frequency_cycles_per_world": 6,
                "canonical_phase_trials": 65536,
                "canonical_phase_salt": "periodic-frame-phase-null-v1",
                "canonical_phase_offset_generator": (
                    "NumPy PCG64 independently draws dy/dx for every "
                    "trial and seed block; offset table SHA-256 persisted"),
                "translation_residues": [0, 63],
                "phase_effect": (
                    "at least six of eight positive nonflat seed scores at "
                    "or above their exact 64x64 translation-null 95th "
                    "percentile"),
                "scale_candidate_frequencies": [4, 5, 6, 7, 8],
                "scale_exact_assignments": 390625,
                "scale_effect": (
                    "at least six of eight seeds with frame-frequency "
                    "power/control-median ratio >=4"),
                "phase_randomization_alpha": 0.01,
                "scale_candidate_reference_tail_threshold": 0.01,
                "scale_candidate_frequencies_are_not_exchangeable_nulls":
                    True,
                "automatic_veto_requires_phase_evidence_and_scale_"
                "corroboration": True,
                "reason": (
                    "frame-scale coincidence alone is valid when generated "
                    "without crop-border knowledge"),
                "seed_block_preserves": (
                    "initial/transported, occupancy/identity, and both axes"),
            },
            "frame_width_exact_nonzero_repetition_veto": True,
            "canonical_seam_profiles_and_frame_scale_power_persisted": True,
            "causal_rule": (
                "a naturally straight or frame-parallel feature is valid; "
                "recut dependence, systematic lattice lock, rectangular "
                "construction, or exact frame-width stamping is not"),
        },
        "execution": {
            "primary_formation_chronologies": len(SEEDS),
            "recut_formation_chronologies": len(SEEDS),
            "primary_prefix_snapshots": len(SEEDS),
            "recut_prefix_snapshots": len(SEEDS),
            "primary_coarse_structure_builds": len(SEEDS),
            "structure_recut_builds": 1,
            "fine_sentinel_builds": 1,
            "post_structure_scans": 2 * len(SEEDS) + 1,
            "elevation_builds": 0,
            "surface_process_solves": 0,
        },
        "automatic_readiness": {
            "every_seed": [
                "exact inventory and strict prefix",
                "exact formation recut",
                "periodic connectivity, exact resistance law, and strict "
                "earlier paths",
                "no initial or transported severe rectangle or exact "
                "frame-width repetition",
                "source-plate consistency and all domains represented",
                "separated morphology-qualified low/medium/high assignment",
            ],
            "sentinel": [
                "exact full structural recut",
                f"40/80-km IoU >= {MIN_SENTINEL_RESOLUTION_IOU}",
                f"40/80-km fraction delta <= {MAX_SENTINEL_FRACTION_DELTA}",
                "fine morphology-qualified assignment",
            ],
            "cohort": (
                "no seed-blocked D4 long-ruler lock and no joint canonical-"
                "phase evidence plus exact-frame-scale corroboration; phase "
                "randomization alpha is 0.01 and scale alone never vetoes"),
        },
        "manual_readiness": {
            "required": True,
            "audit_views": (
                "unmarked 3x3 continuity, fixed-scale eight-tile formation/"
                "structure panels half-world-rolled to center both seams "
                "with union/identity ruler, OBB, and winding overlays; "
                "gray arrival quartiles are audit overlays rather than "
                "generated land or structural surfaces; "
                "boundary-phase/spectrum panels; and assigned crop panels "
                "for every seed"),
            "veto": (
                "square/rectangular bodies, repeated straight or parallel "
                "surfaces without tectonic cause, seam discontinuities, "
                "frame-width stamping, similar-sized blob fields, or "
                "translation-only transport visibly impoverishing structure; "
                "winding components are manual because Euclidean OBB/ruler "
                "tests are topologically invalid for them"),
        },
        "interpretation": [
            "Passing does not validate elevation, final land, or water borders.",
            "Diagnostic tiling never enters generation.",
            (
                "Passing exposed development only permits a separately sealed "
                "fresh validation; it is not promotion evidence."
                if RUN_ROLE == "exposed_development" else
                "Passing fresh validation plus manual review permits a later "
                "periodic elevation/process feasibility; production remains "
                "unchanged."
            ),
        ],
    }


def _require_execute_output(out: Path,
                            expected_sha256: str) -> tuple[dict, str]:
    if not out.is_dir():
        raise FileNotFoundError(out)
    if {item.name for item in out.iterdir()} != {"protocol_precommit.json"}:
        raise FileExistsError("execute requires only protocol_precommit.json")
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
        "primary_formation_chronologies": 0,
        "recut_formation_chronologies": 0,
        "primary_prefix_snapshots": 0,
        "recut_prefix_snapshots": 0,
        "primary_coarse_structure_builds": 0,
        "structure_recut_builds": 0,
        "fine_sentinel_builds": 0,
        "post_structure_scans": 0,
        "elevation_builds": 0,
        "surface_process_solves": 0,
    }
    results = []
    layouts = {}
    structures = {}
    masks = {}
    scans = {}
    panels = []
    audit_panels = []
    geometry_panels = []
    assignments = []

    for seed in SEEDS:
        layout = _layout(seed)
        counters["primary_formation_chronologies"] += 1
        counters["primary_prefix_snapshots"] += 1
        recut_probe, recut_layout = _layout_recut_probe(seed, layout)
        counters["recut_formation_chronologies"] += 1
        counters["recut_prefix_snapshots"] += 1
        structure, cfg, sites = _build(
            seed, layout, STRUCTURE_NOMINAL_KM)
        counters["primary_coarse_structure_builds"] += 1
        result, authority, canonical_scan, transported_scan = _seed_result(
            seed, layout, structure, cfg, sites)
        counters["post_structure_scans"] += 2
        result["formation_recut_probe"] = recut_probe
        results.append(result)
        layouts[seed] = layout
        if seed == FINE_SENTINEL_SEED:
            structures[seed] = structure
        scans[seed] = (canonical_scan, transported_scan)
        masks[seed] = {
            "canonical_assembly": layout["assembly"],
            "canonical_broad_assembly": layout["broad_assembly"],
            "canonical_craton": layout["craton"],
            "canonical_resistance": layout["resistance"],
            "canonical_province_raw": layout["province_raw"],
            "canonical_eligible_nuclei": layout["eligible_nuclei"],
            "canonical_nucleus_mask": layout["nucleus_mask"],
            "canonical_prefix_selected": layout["prefix_selected"],
            "canonical_selected": layout["selected"],
            "canonical_prefix_domain_id": layout["prefix_domain_id_grid"],
            "canonical_domain_id": layout["domain_id_grid"],
            "canonical_selected_arrival": np.where(
                layout["selected"], layout["arrival"], np.inf),
            "recut_assembly_raw": recut_layout["assembly"],
            "recut_broad_assembly_raw": recut_layout["broad_assembly"],
            "recut_craton_raw": recut_layout["craton"],
            "recut_resistance_raw": recut_layout["resistance"],
            "recut_eligible_nuclei_raw":
                recut_layout["eligible_nuclei"],
            "recut_nucleus_mask_raw": recut_layout["nucleus_mask"],
            "recut_prefix_selected_raw": recut_layout["prefix_selected"],
            "recut_selected_raw": recut_layout["selected"],
            "recut_prefix_domain_id_raw":
                recut_layout["prefix_domain_id_grid"],
            "recut_domain_id_raw": recut_layout["domain_id_grid"],
            "recut_selected_arrival_raw": np.where(
                recut_layout["selected"], recut_layout["arrival"], np.inf),
            "transported_80km_proxy": authority["proxy"].astype(np.float32),
            "transported_80km_binary": authority["binary"],
            "transported_80km_dominant_tag": authority["dominant_tag"],
            "source_plate_partition": authority["source_partition"],
        }
        panels.append(_render_periodic_panel(seed, layout, authority, out))
        audit_panels.append(_render_audit_panel(
            seed, layout, structure, authority, cfg, sites, result, out))
        geometry_panels.append(_render_geometry_spectrum_panel(
            seed, result, out))
        assignments.append(_render_assignment_windows(
            seed, authority, result, out))

    sentinel_layout = layouts[FINE_SENTINEL_SEED]
    sentinel_structure = structures[FINE_SENTINEL_SEED]
    structure_recut, recut_structure = _structure_recut_probe(
        FINE_SENTINEL_SEED, sentinel_layout, sentinel_structure)
    counters["structure_recut_builds"] += 1
    structural_fields = (
        "label", "cont", "cont_frac", "age_myr", "belt",
        "belt_age_era", "conv_recent", "div_recent", "coast",
        "active_margin", "passive_margin", "initial_label",
    )
    structural_evidence = {}
    for name in structural_fields:
        structural_evidence[f"structure_canonical_{name}"] = getattr(
            sentinel_structure, name)
        structural_evidence[f"structure_recut_{name}_raw"] = getattr(
            recut_structure, name)
    structural_evidence.update({
        "structure_canonical_material_tags":
            sentinel_structure._material_tag_samples,
        "structure_recut_material_tags_raw":
            recut_structure._material_tag_samples,
        "structure_canonical_plate_displacements_yx_km":
            sentinel_structure._plate_displacements_yx_km,
        "structure_recut_plate_displacements_yx_km_raw":
            recut_structure._plate_displacements_yx_km,
    })
    masks[FINE_SENTINEL_SEED].update(structural_evidence)

    fine_structure, _, _ = _build(
        FINE_SENTINEL_SEED, sentinel_layout,
        FINE_STRUCTURE_NOMINAL_KM)
    counters["fine_sentinel_builds"] += 1
    fine_authority = _sample_structure_authority(
        fine_structure, sentinel_layout)
    fine_scan = _periodic_scan_windows(fine_authority["proxy"])
    fine_qualification = _periodic_qualify_scan(
        fine_scan, fine_authority["binary"],
        fine_authority["dominant_tag"])
    fine_scan["selection"] = fine_qualification.pop("selection")
    fine_scan["morphology_qualification"] = fine_qualification
    counters["post_structure_scans"] += 1
    coarse_proxy = masks[FINE_SENTINEL_SEED]["transported_80km_proxy"]
    coarse_binary = masks[FINE_SENTINEL_SEED]["transported_80km_binary"]
    resolution_iou = base._mask_iou(
        coarse_binary, fine_authority["binary"])
    fraction_delta = abs(
        float(coarse_proxy.mean()) - float(fine_authority["proxy"].mean()))
    fine_sentinel = {
        "seed": FINE_SENTINEL_SEED,
        "coarse_nominal_km": STRUCTURE_NOMINAL_KM,
        "fine_nominal_km": FINE_STRUCTURE_NOMINAL_KM,
        "fine_actual_km": float(
            fine_structure.world_km / fine_structure.n),
        "binary_mask_iou": resolution_iou,
        "global_proxy_fraction_delta": fraction_delta,
        "fine_global_proxy_fraction": float(fine_authority["proxy"].mean()),
        "fine_scan": base._scan_report(fine_scan),
        "passed": (
            resolution_iou >= MIN_SENTINEL_RESOLUTION_IOU
            and fraction_delta <= MAX_SENTINEL_FRACTION_DELTA
            and fine_scan["selection"]["found"]),
    }
    masks[FINE_SENTINEL_SEED].update({
        "transported_40km_proxy": fine_authority["proxy"].astype(np.float32),
        "transported_40km_binary": fine_authority["binary"],
        "transported_40km_dominant_tag": fine_authority["dominant_tag"],
    })

    mask_artifacts = []
    scan_artifacts = []
    for seed in SEEDS:
        mask_artifacts.append(base._save_masks(
            seed, out, **masks[seed]))
        scan_artifacts.append(base._save_scan_table(
            seed, out, scans[seed][0], scans[seed][1],
            fine_scan if seed == FINE_SENTINEL_SEED else None))
    panel_montage = base._render_montage(
        panels, out, "periodic_formation_montage.png")
    audit_montage = base._render_montage(
        audit_panels, out, "formation_structure_audit_montage.png")
    geometry_montage = base._render_montage(
        geometry_panels, out, "geometry_phase_spectrum_montage.png")
    assignment_montage = base._render_montage(
        assignments, out, "periodic_assignment_montage.png")

    exact_count = sum(
        result["ready_gates"]["inventory_exact"] for result in results)
    invariant_count = sum(
        result["ready_gates"]["formation_invariants"] for result in results)
    recut_count = sum(
        result["formation_recut_probe"]["passed"] for result in results)
    assignment_count = sum(
        result["ready_gates"]["transported_proxy_assignment"]
        for result in results)
    passed_window_count = sum(
        sum(window["passed"] for window in
            result["assigned_window_reviews"]["windows"].values())
        for result in results)
    ready_count = sum(result["ready"] for result in results)
    geometry_cohort = _geometry_cohort(results)
    frame_lock_cohort = _frame_lock_cohort(results)
    expected_counters = {
        "primary_formation_chronologies": len(SEEDS),
        "recut_formation_chronologies": len(SEEDS),
        "primary_prefix_snapshots": len(SEEDS),
        "recut_prefix_snapshots": len(SEEDS),
        "primary_coarse_structure_builds": len(SEEDS),
        "structure_recut_builds": 1,
        "fine_sentinel_builds": 1,
        "post_structure_scans": 2 * len(SEEDS) + 1,
        "elevation_builds": 0,
        "surface_process_solves": 0,
    }
    aggregate_gates = {
        "exact_inventory_all_seeds": exact_count == len(SEEDS),
        "formation_invariants_all_seeds": invariant_count == len(SEEDS),
        "exact_formation_recut_all_seeds": recut_count == len(SEEDS),
        "exact_structure_recut_sentinel": structure_recut["passed"],
        "all_seeds_have_proxy_assignment": assignment_count == len(SEEDS),
        "all_assigned_windows_pass_components": passed_window_count == 24,
        "all_seeds_ready": ready_count == MIN_READY_SEEDS,
        "geometry_cohort": geometry_cohort["passed"],
        "frame_lock_cohort": frame_lock_cohort["passed"],
        "fine_sentinel": fine_sentinel["passed"],
        "execution_counts": counters == expected_counters,
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
        "structure_recut_probe": structure_recut,
        "fine_sentinel": fine_sentinel,
        "geometry_cohort": geometry_cohort,
        "frame_lock_cohort": frame_lock_cohort,
        "aggregate": {
            "seed_count": len(SEEDS),
            "exact_inventory_seed_count": exact_count,
            "formation_invariant_seed_count": invariant_count,
            "exact_formation_recut_seed_count": recut_count,
            "proxy_assignment_seed_count": assignment_count,
            "passed_assigned_window_count": passed_window_count,
            "required_assigned_window_count": 24,
            "ready_seed_count": ready_count,
            "required_ready_seed_count": MIN_READY_SEEDS,
            "gates": aggregate_gates,
        },
        "automatic_periodic_readiness_pass": automatic_pass,
        "manual_morphology_review": {
            "status": "unreviewed",
            "required": True,
            "criteria": protocol["manual_readiness"]["veto"],
        },
        "artifacts": {
            "periodic_panels": panels,
            "formation_structure_audit_panels": audit_panels,
            "geometry_phase_spectrum_panels": geometry_panels,
            "periodic_assignments": assignments,
            "periodic_panel_montage": panel_montage,
            "formation_structure_audit_montage": audit_montage,
            "geometry_phase_spectrum_montage": geometry_montage,
            "periodic_assignment_montage": assignment_montage,
            "authority_masks": mask_artifacts,
            "complete_scan_tables": scan_artifacts,
        },
        "interpretation_limits": protocol["interpretation"],
        "recommend_fresh_validation": False,
        "recommend_full_parent_solve": False,
        "promotion": False,
        "section3b_status": "periodic_structure_feasibility_unresolved",
    }
    report_sha256 = base._write_json_exclusive(out / "report.json", report)
    base._write_json_exclusive(out / "report.sha256.json", {
        "file": "report.json",
        "protocol_precommit_sha256": protocol_sha256,
        "sha256": report_sha256,
    })
    print(json.dumps({
        "completed": True,
        "automatic_periodic_readiness_pass": automatic_pass,
        "ready_seed_count": ready_count,
        "formation_recut_seed_count": recut_count,
        "structure_recut_pass": structure_recut["passed"],
        "geometry_cohort_pass": geometry_cohort["passed"],
        "frame_lock_cohort_pass": frame_lock_cohort["passed"],
        "fine_sentinel_pass": fine_sentinel["passed"],
        "report_sha256": report_sha256,
    }, indent=2))
    return report


def _self_check() -> dict:
    periodic_transport = periodic_tectonics_self_check()
    if not periodic_transport["passed"]:
        raise AssertionError("periodic transport helper failed")
    x = np.asarray([123.5, 4500.25, 24000.75])
    y = np.asarray([900.0, 12000.5, 88.25])
    field = _periodic_spectral_octave(
        x, y, PARENT_KM, 5000.0, 1234567)
    shifted = _periodic_spectral_octave(
        x + PARENT_KM, y - PARENT_KM,
        PARENT_KM, 5000.0, 1234567)
    if not np.array_equal(field, shifted):
        raise AssertionError("spectral field is not exactly periodic")

    corner_mask = np.zeros((12, 12), bool)
    corner_mask[0, 0] = True
    corner_mask[-1, -1] = True
    _, corner_components = _periodic_components(corner_mask)
    if len(corner_components) != 1:
        raise AssertionError("periodic components do not cross corner")

    n = 20
    q = (np.arange(n) + 0.5) * CANONICAL_KM
    X, Y = np.meshgrid(q, q)
    assembly = (
        0.12 * np.sin(2.0 * np.pi * X / (n * CANONICAL_KM))
        + 0.08 * np.cos(4.0 * np.pi * Y / (n * CANONICAL_KM)))
    ties = _coordinate_ties(41, q, q)
    physical_pivots = ((3, 4), (13, 15))
    nuclei = [{
        "storage_yx": [y0, x0],
        "domain_id": f"{index + 1:016x}",
        "pivot_yx_km": [float(q[y0]), float(q[x0])],
    } for index, (y0, x0) in enumerate(physical_pivots)]
    growth = _periodic_growth(
        assembly, nuclei, ties, q, q, 160, 360)
    if not (growth["prefix_selected"].sum() == 160
            and growth["selected"].sum() == 360
            and np.all(~growth["prefix_selected"] | growth["selected"])):
        raise AssertionError("synthetic periodic inventory failed")

    cut = (7, 11)
    qy = ((np.arange(n) + cut[0]) % n + 0.5) * CANONICAL_KM
    qx = ((np.arange(n) + cut[1]) % n + 0.5) * CANONICAL_KM
    Xr, Yr = np.meshgrid(qx, qy)
    assembly_recut = (
        0.12 * np.sin(2.0 * np.pi * Xr / (n * CANONICAL_KM))
        + 0.08 * np.cos(4.0 * np.pi * Yr / (n * CANONICAL_KM)))
    ties_recut = _coordinate_ties(41, qy, qx)
    nuclei_recut = [{
        "storage_yx": [
            (y0 - cut[0]) % n, (x0 - cut[1]) % n],
        "domain_id": f"{index + 1:016x}",
        "pivot_yx_km": [float(q[y0]), float(q[x0])],
    } for index, (y0, x0) in enumerate(physical_pivots)]
    growth_recut = _periodic_growth(
        assembly_recut, nuclei_recut, ties_recut,
        qy, qx, 160, 360)
    for name in ("selected", "prefix_selected",
                 "selected_owner", "prefix_owner"):
        if not np.array_equal(
                growth[name],
                _roll_to_canonical(growth_recut[name], cut)):
            raise AssertionError(f"synthetic recut changed {name}")
    canonical_arrival = np.where(
        growth["selected"], growth["arrival"], np.inf)
    recut_arrival = np.where(
        growth_recut["selected"], growth_recut["arrival"], np.inf)
    if not np.array_equal(
            canonical_arrival, _roll_to_canonical(recut_arrival, cut)):
        raise AssertionError("synthetic recut changed arrival authority")

    rectangle = np.zeros((32, 32), bool)
    rectangle[4:20, 5:21] = True
    rectangle_stats = _oriented_rectangle_stats(rectangle, (10, 12))
    if not rectangle_stats["severe_rectangle"]:
        raise AssertionError("rectangle fixture did not trip")
    if _maximum_ruler_run(rectangle, (10, 12))["cells"] < 16.0:
        raise AssertionError("rectangle fixture ruler was not detected")
    irregular = np.zeros((32, 32), bool)
    yy, xx = np.indices(irregular.shape)
    dy = _minimum_image(yy - 2, 32.0)
    dx = _minimum_image(xx - 3, 32.0)
    irregular[(dx / 8.0) ** 2 + (dy / 5.0) ** 2
              < 1.0 + 0.12 * np.sin(0.7 * xx)] = True
    if _oriented_rectangle_stats(irregular, (2, 3))["severe_rectangle"]:
        raise AssertionError("wrapped irregular fixture looks rectangular")
    transported_fixture = _transported_geometry({
        "binary": irregular,
        "dominant_tag": np.where(irregular, 0, -1),
    })
    if not transported_fixture["tripwires"][
            "no_severe_oriented_rectangle"]:
        raise AssertionError("wrapped transported fixture looks rectangular")
    checker_tags = np.where(
        rectangle, (yy + xx) % 2, -1).astype(np.int32)
    multitag_rectangle = _transported_geometry({
        "binary": rectangle,
        "dominant_tag": checker_tags,
    })
    if (not any(item["severe_rectangle"] for item in
                multitag_rectangle["union_components"])
            or multitag_rectangle["tripwires"][
                "no_severe_oriented_rectangle"]):
        raise AssertionError("multi-tag rectangle evaded union shape gate")
    winding_band = np.zeros((32, 32), bool)
    winding_band[13:18, :] = True
    winding_shapes = _periodic_shape_components(winding_band)
    if not any(item["component_winds_torus"]
               for item in winding_shapes["components"]):
        raise AssertionError("toroidal winding was not detected")

    rulers = [{
        "seed": 900 + index // 2,
        "angle_degrees": float((17 * index) % 180),
    } for index in range(16)]
    d4_fixture = _seed_blocked_d4_randomization(rulers)
    if not (0.0 < d4_fixture["randomization_upper_tail_p"] <= 1.0):
        raise AssertionError("seed-blocked D4 randomization failed")

    size = int(round(PARENT_KM / CANONICAL_KM))
    gy, gx = np.indices((size, size))
    ddy = (gy - 71 + size // 2) % size - size // 2
    ddx = (gx - 113 + size // 2) % size - size // 2
    phase_fixture = (
        (ddx / 91.0) ** 2 + (ddy / 57.0) ** 2
        < 1.0 + 0.10 * np.sin(0.09 * gx + 0.04 * gy))
    phase_results = []
    for index in range(8):
        initial_mask = np.roll(
            phase_fixture, (index * 19, index * 31), (0, 1))
        transported_mask = np.roll(
            phase_fixture, (index * 23 + 7, index * 11 + 5), (0, 1))
        initial_identity = np.where(initial_mask, 1, 0)
        transported_identity = np.where(transported_mask, 2, -1)
        phase_results.append({
            "seed": 950 + index,
            "geometry_diagnostics": _geometry_diagnostics(
                initial_mask, initial_identity),
            "transported_geometry": {
                "union_geometry_diagnostics": _geometry_diagnostics(
                    transported_mask, transported_identity),
            },
        })
    frame_fixture = _frame_lock_cohort(phase_results)
    if (frame_fixture["phase_endpoint"]["trials"] != 65536
            or frame_fixture["scale_endpoint"][
                "exact_assignment_count"] != 390625):
        raise AssertionError("frame-lock cohort protocol changed")
    if (frame_fixture["phase_endpoint"][
            "minimum_unique_joint_seed_translation_pairs"] < 65000
            or frame_fixture["phase_endpoint"][
                "maximum_absolute_inter_seed_packed_offset_correlation"]
            >= 0.02):
        raise AssertionError("phase translations are not independent blocks")

    protocol = _protocol()
    if protocol["seed_policy"]["seeds"] != list(SEEDS):
        raise AssertionError("fixed suite changed")
    return {
        "passed": True,
        "intrinsic_spectral_periodicity": True,
        "periodic_corner_connectivity": True,
        "exact_prefix_target": True,
        "exact_synthetic_recut": True,
        "rectangle_fixture_rejected": True,
        "wrapped_irregular_fixture_accepted": True,
        "transported_geometry_fixture_accepted": True,
        "multitag_union_rectangle_rejected": True,
        "toroidal_winding_detected": True,
        "seed_blocked_d4_randomization": True,
        "seed_blocked_frame_phase_and_scale": True,
        "translation_only_transport_disclosed": True,
        "periodic_transport_helpers": periodic_transport,
        "fixed_suite": list(SEEDS),
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
        if (args.exposed_development or args.phase is not None
                or args.out is not None
                or args.expected_precommit_sha256 is not None):
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
