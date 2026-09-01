"""M3 detail stage: the delivered-frame fields at output resolution.

- The ERODED process-grid surface (engine/erosion.py, fixed km lattice)
  is sampled with clamped Catmull-Rom at each pixel's world-km position
  — the same continuous surface at every resolution (§2). Valleys,
  floodplains, shelf wedges, and drowned channels arrive already carved.
- Only the FINE band of the texture stack (octaves below the process
  grid) is added here; the mid band rode through the erosion solve.
  Sub-pixel octaves are trimmed with full-stack normalization (§2).
- Water: ocean where the detailed surface AND the process surface sit
  below present sea level (sub-grid bumps breach shelves as island
  fields; sub-grid pits cannot flood — §9 no speckle by construction);
  lakes come from the erosion stage's drainage-fed depressions.
- River discharge is sampled to output resolution for the render layer
  (drawn water is the footprint of computed discharge, §11).

Nothing here reads frame coordinates; the frame enters only as the km
window being sampled (§3b).
"""

import numpy as np

from . import noise
from .rng import stage_salt

BASE_LAM_KM = 360.0     # texture spectrum top wavelength (full stack)
FULL_OCTAVES = 9        # 360 km -> ~1.4 km
MID_OCTAVES = 4         # octaves 0..3 ride through the erosion solve
DETAIL_GAIN = 0.5
FINE_SCALE = 0.30       # fine-band amplitude vs the zone modulator


def _smooth01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _bilinear(F, y_km, x_km, ck):
    """Sample coarse world field F (cell centres at (i+0.5)*ck) at km
    coordinates, bilinear, edge-clamped."""
    n0, n1 = F.shape
    gy = y_km / ck - 0.5
    gx = x_km / ck - 0.5
    iy = np.clip(np.floor(gy).astype(np.int64), 0, n0 - 2)
    ix = np.clip(np.floor(gx).astype(np.int64), 0, n1 - 2)
    fy = np.clip(gy - iy, 0.0, 1.0)
    fx = np.clip(gx - ix, 0.0, 1.0)
    f00 = F[iy, ix]
    f10 = F[iy, ix + 1]
    f01 = F[iy + 1, ix]
    f11 = F[iy + 1, ix + 1]
    top = f00 + fx * (f10 - f00)
    bot = f01 + fx * (f11 - f01)
    return top + fy * (bot - top)


def _masked_bilinear(F, valid, y_km, x_km, ck):
    """Bilinear sample without diluting valid values with zero backing.

    Lake levels are sparse fields: zero means "no lake", not a zero-m
    water datum. Normalizing by interpolated validity preserves a flat
    basin level through its soft sampling footprint.
    """
    valid_f = valid.astype(np.float64)
    weight = _bilinear(valid_f, y_km, x_km, ck)
    numer = _bilinear(np.where(valid, F, 0.0), y_km, x_km, ck)
    out = np.zeros(np.broadcast(y_km, x_km).shape, np.float64)
    np.divide(numer, weight, out=out, where=weight > 1e-12)
    return out


def _cr_w(t):
    """Catmull-Rom weights for offsets (-1, 0, 1, 2) at fraction t."""
    t2 = t * t
    t3 = t2 * t
    return (-0.5 * t + t2 - 0.5 * t3,
            1.0 - 2.5 * t2 + 1.5 * t3,
            0.5 * t + 2.0 * t2 - 1.5 * t3,
            -0.5 * t2 + 0.5 * t3)


def _bicubic(F, y_km, x_km, ck):
    """Clamped C1 Catmull-Rom prolongation of a coarse world field.

    Bilinear prolongation is only C0: its level sets kink at every cell
    edge (isobath staircases, §11a — M2 instrument find). Plain
    Catmull-Rom overshoots at steep steps, inventing phantom bump rings
    (M2 eval find). The cubic sample is therefore clamped to its cell's
    corner range: smooth where the staircase lived, no extrema the
    coarse solution does not contain."""
    n0, n1 = F.shape
    gy = y_km / ck - 0.5
    gx = x_km / ck - 0.5
    iy = np.floor(gy).astype(np.int64)
    ix = np.floor(gx).astype(np.int64)
    fy = gy - iy
    fx = gx - ix
    wy = _cr_w(fy)
    wx = _cr_w(fx)
    out = np.zeros(np.broadcast(gy, gx).shape, np.float64)
    for a in range(4):
        ia = np.clip(iy + (a - 1), 0, n0 - 1)
        row = np.zeros_like(out)
        for b in range(4):
            ib = np.clip(ix + (b - 1), 0, n1 - 1)
            row += wx[b] * F[ia, ib]
        out += wy[a] * row
    iy0 = np.clip(iy, 0, n0 - 2)
    ix0 = np.clip(ix, 0, n1 - 2)
    c00 = F[iy0, ix0]
    c10 = F[iy0, ix0 + 1]
    c01 = F[iy0 + 1, ix0]
    c11 = F[iy0 + 1, ix0 + 1]
    lo = np.minimum(np.minimum(c00, c10), np.minimum(c01, c11))
    hi = np.maximum(np.maximum(c00, c10), np.maximum(c01, c11))
    return np.clip(out, lo, hi)


def sample_map(s, ce, er, cfg, seed, size, *, _frame_window_km=None):
    """Frame window at output resolution.

    ``_frame_window_km`` is a private atlas/local-process experiment seam
    in ``(y0_km, x0_km, span_km)`` order.  Public callers retain the
    original centered-frame path exactly.
    """
    ck = s.world_km / s.n
    if _frame_window_km is None:
        f0, f1 = s.frame_slice
        frame_km = (f1 - f0) * ck
        km_px = frame_km / size
        q = (np.arange(size) + 0.5) * km_px
        x_km = (f0 * ck + q)[None, :]
        y_km = (f0 * ck + q)[:, None]
        x_sample = x_km
        y_sample = y_km
        frame_x0 = frame_y0 = f0 * ck
    else:
        if len(_frame_window_km) != 3:
            raise ValueError(
                "_frame_window_km must be (y0_km, x0_km, span_km)")
        frame_y0, frame_x0, frame_km = (
            float(value) for value in _frame_window_km)
        if (frame_km <= 0.0 or frame_x0 < 0.0 or frame_y0 < 0.0
                or frame_x0 + frame_km > s.world_km
                or frame_y0 + frame_km > s.world_km):
            raise ValueError("_frame_window_km lies outside the world")
        km_px = frame_km / size
        q = (np.arange(size) + 0.5) * km_px
        x_km = (frame_x0 + q)[None, :]
        y_km = (frame_y0 + q)[:, None]

        process_y0, process_x0 = er.get("process_origin_km", (0.0, 0.0))
        e_km = float(er["e_km"])
        process_ny, process_nx = er["z"].shape
        # Catmull-Rom reads one cell beyond each bracketing interval.
        # Refuse to let edge clamping turn a local-domain edge into a
        # synthetic terrain contour.
        support = 2.0 * e_km
        if (frame_x0 < process_x0 + support
                or frame_y0 < process_y0 + support
                or frame_x0 + frame_km
                > process_x0 + process_nx * e_km - support
                or frame_y0 + frame_km
                > process_y0 + process_ny * e_km - support):
            raise ValueError(
                "localized process domain lacks cubic support for frame")
        x_sample = x_km - process_x0
        y_sample = y_km - process_y0

    e_km = er["e_km"]
    hc = _bicubic(er["z"], y_sample, x_sample, e_km)

    # fine-band texture (octaves MID_OCTAVES.. of the same stack)
    land_amp = 80.0 + 0.10 * np.maximum(hc, 0.0)
    ocean_amp = 16.0 + 24.0 * _smooth01((hc + 2500.0) / 2250.0)
    amp = np.where(hc >= 0.0, land_amp, ocean_amp)
    lam = BASE_LAM_KM / (2.0 ** MID_OCTAVES)
    kept = 0
    for _ in range(FULL_OCTAVES - MID_OCTAVES):
        if lam < km_px:
            break
        kept += 1
        lam /= 2.0
    kept = max(kept, 1)
    det = noise.fbm(x_km, y_km, BASE_LAM_KM, kept,
                    stage_salt(seed, "surface-detail"),
                    gain=DETAIL_GAIN, first_octave=MID_OCTAVES)
    h = hc + cfg.detail_amplitude * FINE_SCALE * amp * det

    ocean = (h < 0.0) & (hc < 0.0)
    ld = _bilinear(er["lake_depth"], y_sample, x_sample, e_km)
    lake_cells = er["lake_depth"] > 0.0
    lsurf = _masked_bilinear(er["lake_surf"], lake_cells,
                             y_sample, x_sample, e_km)
    # Lake shorelines are cut by OUTPUT terrain against the basin's flat
    # water surface. Bilinear depth keeps a soft process-cell footprint;
    # masked level sampling prevents dilution by zero-valued dry cells.
    lake = (ld > 1.5) & (h < lsurf) & ~ocean
    # discharge is a LINEAR feature one process-cell wide — bilinear
    # sampling smears the peak below any threshold; nearest keeps the
    # channel line intact
    ney, nex = er["discharge_log"].shape
    iy = np.clip((y_sample / e_km).astype(np.int64), 0, ney - 1)
    ix = np.clip((x_sample / e_km).astype(np.int64), 0, nex - 1)
    riv_log = er["discharge_log"][iy, ix]
    sed = _bilinear(er["sed"], y_sample, x_sample, e_km)

    result = {
        "h": h.astype(np.float32),
        "hc": hc.astype(np.float32),
        "water": ocean | lake,
        "ocean": ocean,
        "lake": lake,
        "lake_level": lsurf.astype(np.float32),
        "riv_log": riv_log.astype(np.float32),
        "sed": sed.astype(np.float32),
        "river_edges": er["river_edges"],
        "frame_origin_km": frame_x0,
        "km_per_px": km_px,
        "size": size,
    }
    if _frame_window_km is not None:
        result["_frame_window_km"] = (frame_y0, frame_x0, frame_km)
    return result
