"""Chunked final-mask sampling for the private field-accretion parent spike.

This helper samples already-solved absolute-world geography.  It deliberately
contains no candidate ranking or terrain mutation.  The mask equations mirror
``engine.surface.sample_map`` lines 170--198 at the fixed 4-km authority
spacing used by a 4096-km/1024-pixel delivered frame.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from engine import surface
from engine.rng import stage_salt


KM_PER_PX = 4.0
FRAME_KM = 4096.0
FRAME_PX = 1024
MASK_KEYS = ("water", "ocean", "lake", "topographic")


def _validate_window(s, er, y0_km, x0_km, height_px, width_px):
    height_px = int(height_px)
    width_px = int(width_px)
    if height_px <= 0 or width_px <= 0:
        raise ValueError("window dimensions must be positive")
    y0_km = float(y0_km)
    x0_km = float(x0_km)
    height_km = height_px * KM_PER_PX
    width_km = width_px * KM_PER_PX
    if (x0_km < 0.0 or y0_km < 0.0
            or x0_km + width_km > s.world_km
            or y0_km + height_km > s.world_km):
        raise ValueError("absolute-world window lies outside the world")

    process_y0, process_x0 = er.get("process_origin_km", (0.0, 0.0))
    e_km = float(er["e_km"])
    process_ny, process_nx = er["z"].shape
    support = 2.0 * e_km
    if (x0_km < process_x0 + support
            or y0_km < process_y0 + support
            or x0_km + width_km
            > process_x0 + process_nx * e_km - support
            or y0_km + height_km
            > process_y0 + process_ny * e_km - support):
        raise ValueError(
            "process domain lacks cubic support for absolute-world window")
    return y0_km, x0_km, height_px, width_px


def sample_final_boolean_window(
        s, er, cfg, seed, *, y0_km, x0_km, height_px, width_px,
        row_chunk=128):
    """Sample final categorical masks over an arbitrary 4-km rectangle.

    ``topographic`` is the externally visible ``float32(h) >= 0`` diagnostic,
    matching how callers classify the ``h`` returned by ``sample_map``.
    Ocean and lake decisions use the pre-cast float64 surface exactly as the
    authoritative implementation does.
    """
    y0_km, x0_km, height_px, width_px = _validate_window(
        s, er, y0_km, x0_km, height_px, width_px)
    row_chunk = int(row_chunk)
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")

    result = {
        key: np.empty((height_px, width_px), dtype=bool)
        for key in MASK_KEYS
    }
    x_km = (
        x0_km + (np.arange(width_px, dtype=np.float64) + 0.5) * KM_PER_PX
    )[None, :]
    process_y0, process_x0 = er.get("process_origin_km", (0.0, 0.0))
    x_sample = x_km - process_x0
    e_km = er["e_km"]
    lake_cells = er["lake_depth"] > 0.0

    lam = surface.BASE_LAM_KM / (2.0 ** surface.MID_OCTAVES)
    kept = 0
    for _ in range(surface.FULL_OCTAVES - surface.MID_OCTAVES):
        if lam < KM_PER_PX:
            break
        kept += 1
        lam /= 2.0
    kept = max(kept, 1)
    detail_salt = stage_salt(seed, "surface-detail")

    for first in range(0, height_px, row_chunk):
        last = min(first + row_chunk, height_px)
        y_km = (
            y0_km
            + (np.arange(first, last, dtype=np.float64) + 0.5)
            * KM_PER_PX
        )[:, None]
        y_sample = y_km - process_y0

        # Keep this block synchronized with engine.surface.sample_map 170--198.
        hc = surface._bicubic(er["z"], y_sample, x_sample, e_km)
        land_amp = 80.0 + 0.10 * np.maximum(hc, 0.0)
        ocean_amp = 16.0 + 24.0 * surface._smooth01(
            (hc + 2500.0) / 2250.0)
        amp = np.where(hc >= 0.0, land_amp, ocean_amp)
        det = surface.noise.fbm(
            x_km, y_km, surface.BASE_LAM_KM, kept, detail_salt,
            gain=surface.DETAIL_GAIN, first_octave=surface.MID_OCTAVES)
        h = hc + cfg.detail_amplitude * surface.FINE_SCALE * amp * det

        ocean = (h < 0.0) & (hc < 0.0)
        ld = surface._bilinear(
            er["lake_depth"], y_sample, x_sample, e_km)
        lsurf = surface._masked_bilinear(
            er["lake_surf"], lake_cells, y_sample, x_sample, e_km)
        lake = (ld > 1.5) & (h < lsurf) & ~ocean

        result["ocean"][first:last] = ocean
        result["lake"][first:last] = lake
        result["water"][first:last] = ocean | lake
        result["topographic"][first:last] = (
            h.astype(np.float32) >= 0.0)

    result["origin_yx_km"] = (y0_km, x0_km)
    result["km_per_px"] = KM_PER_PX
    result["shape"] = (height_px, width_px)
    return result


def _aligned_offset(origin_km, mosaic_origin_km):
    raw = (float(origin_km) - float(mosaic_origin_km)) / KM_PER_PX
    index = int(round(raw))
    if abs(raw - index) > 1e-9:
        raise ValueError("subwindow is not aligned to the 4-km mosaic")
    return index


def verify_4096_subwindow(
        mosaic, s, ce, er, cfg, seed, *, y0_km, x0_km):
    """Compare a mosaic subwindow with a complete authoritative sample_map."""
    if float(mosaic.get("km_per_px", np.nan)) != KM_PER_PX:
        raise ValueError("mosaic does not use the fixed 4-km spacing")
    mosaic_y0, mosaic_x0 = mosaic["origin_yx_km"]
    row0 = _aligned_offset(y0_km, mosaic_y0)
    col0 = _aligned_offset(x0_km, mosaic_x0)
    row1, col1 = row0 + FRAME_PX, col0 + FRAME_PX
    height, width = mosaic["shape"]
    if row0 < 0 or col0 < 0 or row1 > height or col1 > width:
        raise ValueError("4096-km subwindow lies outside the mosaic")

    reference = surface.sample_map(
        s, ce, er, cfg, seed, FRAME_PX,
        _frame_window_km=(float(y0_km), float(x0_km), FRAME_KM))
    expected = {
        "water": np.asarray(reference["water"], bool),
        "ocean": np.asarray(reference["ocean"], bool),
        "lake": np.asarray(reference["lake"], bool),
        "topographic": np.asarray(reference["h"]) >= 0.0,
    }
    mismatches = {}
    for key in MASK_KEYS:
        observed = mosaic[key][row0:row1, col0:col1]
        mismatches[key] = int(np.count_nonzero(observed != expected[key]))
    return {
        "passed": all(value == 0 for value in mismatches.values()),
        "mismatch_cells": mismatches,
        "origin_yx_km": (float(y0_km), float(x0_km)),
    }


def self_check():
    """Synthetic, generation-free chunk/full equivalence check."""
    world_km = 8192.0
    e_km = 16.0
    n = int(world_km / e_km)
    q = (np.arange(n, dtype=np.float64) + 0.5) * e_km
    X, Y = np.meshgrid(q, q)
    z = (620.0 * np.sin(X / 710.0)
         + 510.0 * np.cos(Y / 830.0) - 120.0)
    basin = ((X - 4100.0) ** 2 + (Y - 3900.0) ** 2) < 620.0 ** 2
    lake_depth = np.where(basin, 18.0, 0.0)
    lake_surf = np.where(basin, 1800.0, 0.0)
    zeros = np.zeros_like(z)
    er = {
        "e_km": e_km,
        "z": z,
        "lake_depth": lake_depth,
        "lake_surf": lake_surf,
        "discharge_log": zeros,
        "sed": zeros,
        "river_edges": {},
        "process_origin_km": (0.0, 0.0),
    }
    s = SimpleNamespace(world_km=world_km, n=128, frame_slice=(32, 96))
    cfg = SimpleNamespace(detail_amplitude=0.85)
    mosaic = sample_final_boolean_window(
        s, er, cfg, 17, y0_km=1984.0, x0_km=1920.0,
        height_px=1104, width_px=1088, row_chunk=37)
    check = verify_4096_subwindow(
        mosaic, s, None, er, cfg, 17, y0_km=2048.0, x0_km=2048.0)
    if not check["passed"]:
        raise AssertionError(check)
    if not np.array_equal(mosaic["water"], mosaic["ocean"] | mosaic["lake"]):
        raise AssertionError("water union mismatch")
    return check


if __name__ == "__main__":
    print(self_check())
