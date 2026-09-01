"""Plan A pilot: the APPROVED M1 formation on a flat periodic parent.

The prior border program (B00-B17) rejected every attempt, but its
surviving premises are: boundaryless (periodic) parent domain, crop-last
window selection, and natural formation. Its systemic blind spot: the
approved M1 formation was never itself run on the surviving parent
architecture -- every periodic attempt substituted a new formation
(field accretion, convergence maturation) that then failed morphology.
This spike closes that gap.

Port rules (every deviation from `engine.tectonics.build_structure` is
listed here; everything else is copied verbatim):

1. DOMAIN: flat torus, WORLD_KM = 2.5 x FRAME_KM (10240 km), 256^2 at
   40 km. Plate count 12 preserves approved plate area (approved world
   7782.4^2 km^2 / 7 plates ~= parent 10240^2 / 12).
2. ROTATION IS KEPT. "Rotation is not well-defined on a torus" is true
   for global maps but irrelevant for bounded plates: each plate's
   material lives in an unwrapped PLANAR patch (universal cover); the
   approved composed `_Affine` applies unchanged; a world query point
   tests its periodic images (u + a*W, v + b*W) against the patch.
   First image wins deterministically; same-plate multi-image hits
   (plate wrapping into itself) are counted as `self_overlap_cells`
   and predicted zero at this parent size.
3. MATERIAL PATCHES: at era 0 each plate's owned torus cells are
   unwrapped into the n-cell window centred on the plate's circular
   centroid, stored in an (n+2*PAD)^2 array. PAD=96 cells covers the
   kinematic budget (drift ~50 cells + rotation ~10) plus growth, so
   fresh ridge crust always lands in bounds (`dropped_fresh_cells`
   counted, predicted zero).
4. PARTITION: warped Voronoi as approved, with minimum-image toroidal
   distance and the warp fbm made periodic by hashing lattice indices
   modulo the octave period (same gradient-noise mechanism; the
   standing noise-parameterization exception). Warp wavelength keeps
   the approved PHYSICAL scale: W / round(W / (approved_world/3.5)).
   Generators are uniform over the full torus -- the approved
   [0.05, 0.95]*W inset is a finite-rim device, meaningless here.
5. NUCLEI ARE FRAME-FREE: uniform over host-plate interiors. The
   approved frame box + inset is exactly the frame correlation being
   removed. Craton lobes/wobble are painted in a local chart centred
   on the nucleus (minimum-image deltas; reach <= 1.5 r << W/2), so
   outlines are seam-free WITHOUT periodic noise and ride the plate.
6. BUDGET: per-nucleus budget preserved; total =
   continental_budget * frame_cells * BUDGET_FRAMES over NUCLEI=6
   nuclei on the 6 largest plates (approved: 0.30 * frame_cells over
   3 nuclei on the 3 largest).
7. ROTATION CENTRES: cover-space centroid of the plate's existing
   material (approved used the claimed-label centroid; on the torus
   the material centroid is the well-defined equivalent).
8. All neighbourhood ops (_shift/_dilate/_fill_owner, seafloor-age
   smoothing, coasts/margins) wrap on both axes.

Everything downstream of the Structure (M2/M3) is NOT run here. The
window census is structural-level: a 103-cell (4120 km >= 4096 km)
window whose 1-cell outer ring is crust-free water is a conservative
SUPERSET test of the frame ring.

Failure handling per the author's instruction: if the implementation
fails its mechanics acceptance (determinism, seam blindness, zero
self-overlap, zero dropped crust), this code is NOT promoted and the
attempt is registered as failed. Supply shortfalls are measured
findings, not implementation failure.

Usage:
  python spikes/plan_a_periodic_m1.py --debug 3 7   # non-cohort seeds
  python spikes/plan_a_periodic_m1.py               # cohort 151-158
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import noise
from engine.rng import fnv1a64, stage_rng, stage_salt
from engine.tectonics import (CONT_BORN, COARSE_KM, DT_MYR, EVENT_MEMORY,
                              FRAME_KM, Config, Structure, _Affine)
from eval.geometry_instruments import (boundary_mask, component_roundness,
                                       label_components,
                                       rulers_for_mask, seed_blocked_d4_test,
                                       unwrap_periodic_component)

PARENT_FACTOR = 2.5
WORLD_KM = FRAME_KM * PARENT_FACTOR          # 10240 km
N = int(round(WORLD_KM / COARSE_KM))         # 256 cells
CK = WORLD_KM / N
PLATES = 12
NUCLEI = 6
BUDGET_FRAMES = 2.0
PAD = 96
M = N + 2 * PAD
APPROVED_WORLD_KM = FRAME_KM * 1.9           # default world_margin 0.45
WINDOW_CELLS = 103                            # ceil(4096 / 40)
COHORT = tuple(range(151, 159))
MIN_RULER_KM = 512.0
MIN_ROUND_CELLS = 64


# ------------------------------------------------------ periodic noise

def _periodic_perlin(x_km, y_km, wavelength_km, period_cells, salt):
    """Engine gradient noise with lattice indices wrapped modulo the
    period, making the field exactly periodic on the torus. Same
    hashing, fade, and normalization as `engine.noise.perlin`."""
    x = np.asarray(x_km, np.float64) / wavelength_km
    y = np.asarray(y_km, np.float64) / wavelength_km
    ix0 = np.floor(x).astype(np.int64)
    iy0 = np.floor(y).astype(np.int64)
    fx = x - ix0
    fy = y - iy0
    ux = noise._fade(fx)
    uy = noise._fade(fy)
    p = int(period_cells)

    def corner(dx, dy):
        h = noise._lattice_hash((ix0 + dx) % p, (iy0 + dy) % p, salt)
        ang = h.astype(np.float64) * noise._INV64 * noise._TWO_PI
        return np.cos(ang) * (fx - dx) + np.sin(ang) * (fy - dy)

    n00 = corner(0, 0)
    n10 = corner(1, 0)
    n01 = corner(0, 1)
    n11 = corner(1, 1)
    nx0 = n00 + ux * (n10 - n00)
    nx1 = n01 + ux * (n11 - n01)
    return np.sqrt(2.0) * (nx0 + uy * (nx1 - nx0))


def periodic_fbm(x_km, y_km, world_km, k_base, octaves, salt, gain=0.55):
    """fbm of periodic octaves; base wavelength world/k_base, lacunarity
    fixed at 2 so every octave period stays an integer cell count. Per-
    octave salts/offsets follow `engine.noise.fbm` exactly."""
    total = np.zeros(np.broadcast(x_km, y_km).shape, np.float64)
    amp = 1.0
    k = int(k_base)
    norm = 0.0
    for o in range(octaves):
        lam = world_km / k
        osalt = int(noise._mix(
            np.uint64((salt + 0x9e37 * (o + 1)) & ((1 << 64) - 1))))
        off_x = (osalt & 0xffffffff) / 0xffffffff * lam
        off_y = ((osalt >> 32) & 0xffffffff) / 0xffffffff * lam
        total += amp * _periodic_perlin(x_km + off_x, y_km + off_y,
                                        lam, k, osalt)
        norm += amp
        amp *= gain
        k *= 2
    return total / norm


# ---------------------------------------------------- periodic helpers

def wrap_delta(d, world):
    return (d + 0.5 * world) % world - 0.5 * world


def _proll(a, dy, dx):
    return np.roll(a, (dy, dx), (0, 1))


def _pdilate(mask, r):
    out = np.asarray(mask, bool).copy()
    for _ in range(r):
        g = out.copy()
        out |= _proll(g, 0, 1)
        out |= _proll(g, 0, -1)
        out |= _proll(g, 1, 0)
        out |= _proll(g, -1, 0)
    return out


def _pfill_owner(label):
    lab = np.asarray(label, np.int32).copy()
    empty = lab < 0
    for _ in range(lab.shape[0]):
        if not empty.any():
            break
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = _proll(lab, dy, dx)
            take = empty & (nb >= 0)
            lab[take] = nb[take]
            empty = lab < 0
    return lab


def circ_center_cell(idx, n):
    ang = np.asarray(idx, np.float64) * (2.0 * np.pi / n)
    a = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))
    return int(np.round(a * n / (2.0 * np.pi))) % n


# ------------------------------------------------------------ builders

class _TorusPlate:
    def __init__(self):
        self.exists = np.zeros((M, M), bool)
        self.cont = np.zeros((M, M), bool)
        self.born = np.zeros((M, M), np.int16)
        self.belt = np.zeros((M, M), np.float32)
        self.belt_age = np.full((M, M), -1, np.int16)
        self.T = _Affine()
        self.oy = 0
        self.ox = 0
        self.row_map = None   # torus row -> material row (era-0 chart)
        self.col_map = None


def partition_periodic(seed, cfg):
    """Approved warped Voronoi with toroidal distance and periodic warp
    noise. Same RNG stream/draw order; generator range is the full
    torus (deviation 4 in the module docstring)."""
    rng = stage_rng(seed, "tect-partition")
    salt = stage_salt(seed, "tect-partition")
    xs = (np.arange(N) + 0.5) * CK
    X, Y = np.meshgrid(xs, xs)
    pts = rng.uniform(0.0, WORLD_KM, (cfg.plates, 2))
    k_warp = max(1, int(round(WORLD_KM / (APPROVED_WORLD_KM / 3.5))))
    best = np.full((N, N), np.inf)
    label = np.zeros((N, N), np.int32)
    for p in range(cfg.plates):
        d = np.hypot(wrap_delta(Y - pts[p, 0], WORLD_KM),
                     wrap_delta(X - pts[p, 1], WORLD_KM))
        warp = 1.0 + 0.5 * np.clip(
            periodic_fbm(X, Y, WORLD_KM, k_warp, 5, salt + p), -0.9, 0.9)
        cost = d * warp
        take = cost < best
        best[take] = cost[take]
        label[take] = p
    return label


def seed_nuclei_periodic(seed, cfg, label, plates, budget_cells):
    """Approved `_seed_nuclei` with the frame box removed (deviation 5)
    and craton painting in a nucleus-centred local chart so outlines
    are seam-free. RNG draw order matches the approved seeder."""
    rng = stage_rng(seed, "tect-nuclei")
    salt = stage_salt(seed, "tect-nuclei")
    xs = (np.arange(N) + 0.5) * CK
    X, Y = np.meshgrid(xs, xs)

    sizes = np.bincount(label.ravel(), minlength=cfg.plates)
    hosts = np.argsort(-sizes)[:cfg.nuclei]
    weights = rng.uniform(0.7, 1.3, cfg.nuclei)
    weights = weights / weights.sum()

    for k, host in enumerate(hosts):
        interior = (label == host) & ~_pdilate(label != host, 3)
        cand = np.nonzero(interior)
        if not cand[0].size:
            cand = np.nonzero(label == host)
        i = int(rng.integers(0, cand[0].size))
        cy, cx = (cand[0][i] + 0.5) * CK, (cand[1][i] + 0.5) * CK
        target_cells = weights[k] * budget_cells
        r_km = float(np.sqrt(target_cells / np.pi) * CK)
        n_lobes = int(rng.integers(2, 4))
        lobe_draws = [(rng.uniform(0, 2 * np.pi),
                       rng.uniform(0.35, 0.75),
                       rng.uniform(0.40, 0.65))
                      for _ in range(n_lobes - 1)]
        lobes = [(cy, cx, 0.92 if cfg.multi_lobe else 1.0)]
        if cfg.multi_lobe:
            for a, df, fr in lobe_draws:
                d = df * r_km
                lobes.append((cy + d * np.sin(a), cx + d * np.cos(a), fr))

        # local chart: smooth minimum-image coordinates about the
        # nucleus; the engine's planar fbm applies directly.
        Ych = cy + wrap_delta(Y - cy, WORLD_KM)
        Xch = cx + wrap_delta(X - cx, WORLD_KM)

        def paint(scale):
            b = np.zeros((N, N), bool)
            for j, (ly, lx, fr) in enumerate(lobes):
                lr = fr * r_km * scale
                wob = np.clip(noise.fbm(Xch, Ych, max(lr, 3 * CK), 5,
                                        salt + 17 * k + j), -0.9, 0.9)
                b |= np.hypot(Ych - ly, Xch - lx) < lr * (1.0 + 0.38 * wob)
            return b

        blob = paint(1.0)
        measured = max(int(blob.sum()), 1)
        scale = float(np.sqrt(target_cells / measured))
        if abs(scale - 1.0) > 0.05:
            blob = paint(min(scale, 1.5))
        for p in range(cfg.plates):
            sel = blob & (label == p)
            if sel.any():
                sy, sx = np.nonzero(sel)
                plates[p].cont[plates[p].row_map[sy],
                               plates[p].col_map[sx]] = True


def build_parent(seed, cfg=None):
    """Approved `build_structure` era loop on the periodic parent."""
    cfg = cfg or Config(plates=PLATES, nuclei=NUCLEI)
    t_all = time.perf_counter()
    diag = {"self_overlap_cells": 0, "dropped_fresh_cells": 0,
            "overlap_by_plate": [0] * (cfg or Config()).plates,
            "overlap_first_detail": None}
    xs = (np.arange(N) + 0.5) * CK
    X, Y = np.meshgrid(xs, xs)

    label0 = partition_periodic(seed, cfg)
    plates = [_TorusPlate() for _ in range(cfg.plates)]
    rng_init = stage_rng(seed, "tect-initial-age")
    ocean_born = -rng_init.integers(0, 8, (N, N)).astype(np.int16)
    for p in range(cfg.plates):
        pl = plates[p]
        oy_cells, ox_cells = np.nonzero(label0 == p)
        cy = circ_center_cell(oy_cells, N)
        cx = circ_center_cell(ox_cells, N)
        pl.oy = cy - N // 2 - PAD
        pl.ox = cx - N // 2 - PAD
        pl.row_map = ((np.arange(N) - cy + N // 2) % N) + PAD
        pl.col_map = ((np.arange(N) - cx + N // 2) % N) + PAD
        pl.exists[pl.row_map[oy_cells], pl.col_map[ox_cells]] = True
        pl.born[pl.row_map[oy_cells], pl.col_map[ox_cells]] = \
            ocean_born[oy_cells, ox_cells]

    frame_cells = (FRAME_KM / CK) ** 2
    seed_nuclei_periodic(seed, cfg, label0, plates,
                         cfg.continental_budget * frame_cells * BUDGET_FRAMES)
    for p in range(cfg.plates):
        plates[p].born[plates[p].cont] = CONT_BORN

    rng_k = stage_rng(seed, "tect-kinematics")
    ang = rng_k.uniform(0.0, 2 * np.pi, cfg.plates)
    speed = rng_k.uniform(0.6, 1.4, cfg.plates) * cfg.plate_speed
    omega = rng_k.normal(0.0, 0.004, cfg.plates) * (cfg.plate_speed / 45.0)

    conv_hist = [np.zeros((N, N), bool) for _ in range(EVENT_MEMORY)]
    div_hist = [np.zeros((N, N), bool) for _ in range(EVENT_MEMORY)]

    def image_offsets(pl):
        """Periodic images (a, b) whose translate of the world tile can
        intersect the cover-space bbox of the material array."""
        corners = []
        for i in (0.0, float(M)):
            for j in (0.0, float(M)):
                q = pl.T.a @ np.array([(pl.oy + i) * CK, (pl.ox + j) * CK])
                corners.append(q + pl.T.b)
        corners = np.array(corners)
        ylo, xlo = corners.min(0)
        yhi, xhi = corners.max(0)
        ays = range(int(np.floor(ylo / WORLD_KM)) - 1,
                    int(np.floor(yhi / WORLD_KM)) + 2)
        axs = range(int(np.floor(xlo / WORLD_KM)) - 1,
                    int(np.floor(xhi / WORLD_KM)) + 2)
        return [(a, b) for a in ays for b in axs]

    def rasterize(Yq=None, Xq=None):
        Yq = Y if Yq is None else Yq
        Xq = X if Xq is None else Xq
        claims = []
        for p in range(cfg.plates):
            pl = plates[p]
            mu0, mv0 = pl.T.inverse_map(Yq, Xq)
            ai = pl.T.a.T
            mask = np.zeros(Yq.shape, bool)
            IY = np.zeros(Yq.shape, np.int64)
            IX = np.zeros(Yq.shape, np.int64)
            for a, b in image_offsets(pl):
                dy = ai[0, 0] * (a * WORLD_KM) + ai[0, 1] * (b * WORLD_KM)
                dx = ai[1, 0] * (a * WORLD_KM) + ai[1, 1] * (b * WORLD_KM)
                iy = np.floor((mu0 + dy) / CK).astype(np.int64) - pl.oy
                ix = np.floor((mv0 + dx) / CK).astype(np.int64) - pl.ox
                inside = (iy >= 0) & (iy < M) & (ix >= 0) & (ix < M)
                if not inside.any():
                    continue
                iyc = iy.clip(0, M - 1)
                ixc = ix.clip(0, M - 1)
                hit = inside & pl.exists[iyc, ixc]
                over = hit & mask
                n_over = int(np.count_nonzero(over))
                diag["self_overlap_cells"] += n_over
                diag["overlap_by_plate"][p] += n_over
                if n_over and diag["overlap_first_detail"] is None:
                    k = np.argwhere(over)[0]
                    diag["overlap_first_detail"] = {
                        "plate": p, "image": (a, b),
                        "prev_material_yx": (int(IY[tuple(k)]),
                                             int(IX[tuple(k)])),
                        "this_material_yx": (int(iy[tuple(k)]),
                                             int(ix[tuple(k)]))}
                new = hit & ~mask
                mask |= new
                IY[new] = iy[new]
                IX[new] = ix[new]
            claims.append((mask, IY, IX))
        return claims

    def resolve(claims, era, mutate):
        label_ = np.full((N, N), -1, np.int32)
        win_cont = np.zeros((N, N), bool)
        wmiy = np.zeros((N, N), np.int64)
        wmix = np.zeros((N, N), np.int64)
        conv = np.zeros((N, N), bool)
        for p in range(cfg.plates):
            mask, iy, ix = claims[p]
            if not mask.any():
                continue
            cont_p = np.zeros((N, N), bool)
            cont_p[mask] = plates[p].cont[iy[mask], ix[mask]]
            occupied = label_ >= 0
            collide = mask & occupied
            conv |= collide
            win = mask & (~occupied | (cont_p & ~win_cont))
            lose = mask & ~win
            disp = collide & win

            if mutate and lose.any():
                lose_oc = lose & ~cont_p
                plates[p].exists[iy[lose_oc], ix[lose_oc]] = False
                for q in np.unique(label_[lose]):
                    sel = lose & (label_ == q)
                    amt = np.where(cont_p[sel], 2.0, 1.0).astype(np.float32)
                    np.add.at(plates[q].belt, (wmiy[sel], wmix[sel]), amt)
                    plates[q].belt_age[wmiy[sel], wmix[sel]] = era
            if mutate and disp.any():
                for q in np.unique(label_[disp]):
                    sel = disp & (label_ == q)
                    plates[q].exists[wmiy[sel], wmix[sel]] = False
                np.add.at(plates[p].belt, (iy[disp], ix[disp]),
                          np.ones(int(disp.sum()), np.float32))
                plates[p].belt_age[iy[disp], ix[disp]] = era

            label_[win] = p
            win_cont[win] = cont_p[win]
            wmiy[win] = iy[win]
            wmix[win] = ix[win]
        return label_, conv

    label = label0.copy()
    for era in range(cfg.eras):
        ang = ang + rng_k.normal(0.0, cfg.wander, cfg.plates)
        speed = np.clip(speed + rng_k.normal(0.0, 0.03 * cfg.plate_speed,
                                             cfg.plates),
                        0.2 * cfg.plate_speed, 2.2 * cfg.plate_speed)
        cents = np.zeros((cfg.plates, 2))
        for p in range(cfg.plates):
            pl = plates[p]
            eiy, eix = np.nonzero(pl.exists)
            if eiy.size:
                mat = np.array([(pl.oy + eiy.mean() + 0.5) * CK,
                                (pl.ox + eix.mean() + 0.5) * CK])
                cents[p] = pl.T.a @ mat + pl.T.b
        for p in range(cfg.plates):
            vel = np.array([speed[p] * np.sin(ang[p]),
                            speed[p] * np.cos(ang[p])])
            plates[p].T.pre_step(omega[p], cents[p], vel)

        claims = rasterize()
        label, conv = resolve(claims, era, mutate=True)

        gap = label < 0
        if gap.any():
            label = _pfill_owner(label)
            for p in np.unique(label[gap]):
                pl = plates[p]
                sel = gap & (label == p)
                gy, gx = np.nonzero(sel)
                mu0, mv0 = pl.T.inverse_map((gy + 0.5) * CK,
                                            (gx + 0.5) * CK)
                ai = pl.T.a.T
                # Fresh crust accretes ADJACENT to the plate: among the
                # in-bounds periodic images pick the one nearest the
                # plate's existing material centroid. (The material
                # array is wider than one world period, so a naive
                # first-in-bounds choice can strand fresh crust a full
                # period away and snowball the patch span.)
                eiy, eix = np.nonzero(pl.exists)
                cy_mat = eiy.mean() if eiy.size else 0.5 * M
                cx_mat = eix.mean() if eix.size else 0.5 * M
                got = np.zeros(gy.size, bool)
                best_d2 = np.full(gy.size, np.inf)
                MIY = np.zeros(gy.size, np.int64)
                MIX = np.zeros(gy.size, np.int64)
                for a, b in image_offsets(pl):
                    dy = (ai[0, 0] * (a * WORLD_KM)
                          + ai[0, 1] * (b * WORLD_KM))
                    dx = (ai[1, 0] * (a * WORLD_KM)
                          + ai[1, 1] * (b * WORLD_KM))
                    iy = np.floor((mu0 + dy) / CK).astype(np.int64) - pl.oy
                    ix = np.floor((mv0 + dx) / CK).astype(np.int64) - pl.ox
                    inb = (iy >= 0) & (iy < M) & (ix >= 0) & (ix < M)
                    d2 = ((iy - cy_mat) ** 2 + (ix - cx_mat) ** 2)
                    take = inb & (d2 < best_d2)
                    MIY[take] = iy[take]
                    MIX[take] = ix[take]
                    best_d2[take] = d2[take]
                    got |= inb
                diag["dropped_fresh_cells"] += int(np.count_nonzero(~got))
                iy, ix = MIY[got], MIX[got]
                fresh = ~pl.exists[iy, ix]
                pl.exists[iy[fresh], ix[fresh]] = True
                pl.cont[iy[fresh], ix[fresh]] = False
                pl.born[iy[fresh], ix[fresh]] = era
        conv_hist[era % EVENT_MEMORY] = conv
        div_hist[era % EVENT_MEMORY] = gap

    # final snapshot: 2x2 supersampled read-only resolve (approved)
    claims = rasterize()
    label, _ = resolve(claims, cfg.eras, mutate=False)
    if (label < 0).any():
        label = _pfill_owner(label)

    cont_frac = np.zeros((N, N))
    born_f = np.zeros((N, N))
    belt = np.zeros((N, N), np.float32)
    belt_age_w = np.zeros((N, N))
    for oy, ox in ((-0.25, -0.25), (-0.25, 0.25),
                   (0.25, -0.25), (0.25, 0.25)):
        cl = rasterize(np.mod(Y + oy * CK, WORLD_KM),
                       np.mod(X + ox * CK, WORLD_KM))
        lab_s, _ = resolve(cl, cfg.eras, mutate=False)
        gap_s = lab_s < 0
        cont_s = np.zeros((N, N), bool)
        born_s = np.full((N, N), float(cfg.eras))
        belt_s = np.zeros((N, N), np.float32)
        bage_s = np.full((N, N), -1.0)
        for p in range(cfg.plates):
            mask, iy, ix = cl[p]
            own = (lab_s == p) & mask
            cont_s[own] = plates[p].cont[iy[own], ix[own]]
            born_s[own] = plates[p].born[iy[own], ix[own]]
            belt_s[own] = plates[p].belt[iy[own], ix[own]]
            bage_s[own] = plates[p].belt_age[iy[own], ix[own]]
        born_s[gap_s] = float(cfg.eras)
        cont_frac += 0.25 * cont_s
        born_f += 0.25 * born_s
        belt += 0.25 * belt_s
        belt_age_w += 0.25 * belt_s * np.maximum(bage_s, 0.0)
    cont = cont_frac >= 0.5
    with np.errstate(invalid="ignore"):
        belt_age = np.where(belt > 0, belt_age_w / np.maximum(belt, 1e-9),
                            -1.0).astype(np.float64)

    oc = ~cont
    ocw = oc.astype(np.float64)
    bw = born_f * ocw
    num = bw.copy()
    w = ocw.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)):
        num = num + _proll(bw, dy, dx)
        w = w + _proll(ocw, dy, dx)
    born_sm = np.where(oc & (w > 0), num / np.maximum(w, 1e-9), born_f)

    age = (cfg.eras - born_sm) * DT_MYR
    conv_recent = np.zeros((N, N), bool)
    div_recent = np.zeros((N, N), bool)
    for e in conv_hist:
        conv_recent |= e
    for e in div_hist:
        div_recent |= e

    ocean = ~cont
    coast = cont & (_proll(ocean, 0, 1) | _proll(ocean, 0, -1)
                    | _proll(ocean, 1, 0) | _proll(ocean, -1, 0))
    near_conv = _pdilate(conv_recent, 3)
    active = coast & near_conv
    passive = coast & ~near_conv
    alive = int(len(np.unique(label[label >= 0])))

    s = Structure(
        n=N, world_km=WORLD_KM, frame_slice=(0, N),
        label=label, cont=cont, cont_frac=cont_frac, age_myr=age,
        belt=belt, belt_age_era=belt_age, conv_recent=conv_recent,
        div_recent=div_recent, coast=coast, active_margin=active,
        passive_margin=passive, initial_label=label0,
        alive_plates=alive, eras=cfg.eras,
        timings={"structure_s": time.perf_counter() - t_all})
    spans = []
    for p in range(cfg.plates):
        eiy, eix = np.nonzero(plates[p].exists)
        if eiy.size:
            spans.append((int(eiy.max() - eiy.min() + 1),
                          int(eix.max() - eix.min() + 1)))
        else:
            spans.append((0, 0))
    diag["plate_spans"] = spans

    s._periodic = True
    s._rotation = True
    return s, diag


# ---------------------------------------------------------- instruments

def window_fields(cont):
    """Per-origin land fraction and 1-cell-ring water flag for every
    WINDOW_CELLS square window on the torus."""
    wsz = WINDOW_CELLS
    t = np.pad(cont.astype(np.int64), ((0, wsz), (0, wsz)), mode="wrap")
    S = np.zeros((N + wsz + 1, N + wsz + 1), np.int64)
    S[1:, 1:] = t.cumsum(0).cumsum(1)

    def sq(y0, x0, k):
        return (S[y0 + k: y0 + k + N, x0 + k: x0 + k + N]
                - S[y0 + k: y0 + k + N, x0: x0 + N]
                - S[y0: y0 + N, x0 + k: x0 + k + N]
                + S[y0: y0 + N, x0: x0 + N])

    full = sq(0, 0, wsz)
    inner = sq(1, 1, wsz - 2)
    ring_water = (full - inner) == 0
    frac = full / float(wsz * wsz)
    return frac, ring_water


def window_census(cont):
    """All-origin census of a WINDOW_CELLS square with a 1-cell ring."""
    wsz = WINDOW_CELLS
    frac, ring_water = window_fields(cont)
    bands = {"low_15_25": (frac >= 0.15) & (frac <= 0.25),
             "med_30_40": (frac >= 0.30) & (frac <= 0.40),
             "high_45_50": (frac >= 0.45) & (frac < 0.50)}
    out = {
        "origins": int(N * N),
        "ring_water_origins": int(ring_water.sum()),
        "ring_water_share": float(ring_water.mean()),
        "max_land_frac_any": float(frac.max()),
        "max_land_frac_ring_water": float(
            frac[ring_water].max()) if ring_water.any() else 0.0,
    }
    for name, sel in bands.items():
        out[f"{name}_origins"] = int(sel.sum())
        out[f"{name}_ring_water_origins"] = int((sel & ring_water).sum())
    return out


def seam_stats(cont):
    b = boundary_mask(cont, periodic=True)
    rows = b.sum(1).astype(np.float64)
    cols = b.sum(0).astype(np.float64)

    def z(v, arr):
        sd = arr.std()
        return float((v - arr.mean()) / sd) if sd > 0 else 0.0

    return {"row0_z": z(rows[0], rows), "rowN_z": z(rows[-1], rows),
            "col0_z": z(cols[0], cols), "colN_z": z(cols[-1], cols)}


def roundness_summary(cont):
    comps = []
    for ys, xs in label_components(cont, periodic=True):
        if ys.size < MIN_ROUND_CELLS:
            continue
        comp = np.zeros_like(cont)
        comp[ys, xs] = True
        un = unwrap_periodic_component(comp, (int(ys[0]), int(xs[0])))
        if un["component_winds_torus"]:
            comps.append({"cells": int(ys.size), "winds": True,
                          "compactness": None, "solidity": None,
                          "rounded": False})
            continue
        ry = un["relative_y"][ys, xs]
        rx = un["relative_x"][ys, xs]
        ry = ry - ry.min()
        rx = rx - rx.min()
        rec = component_roundness(
            ry, rx, (int(ry.max()) + 1, int(rx.max()) + 1))
        rec["winds"] = False
        comps.append(rec)
    return comps


def isotropic_baseline(land_frac, seeds):
    """Matched-null cohort: periodic isotropic fBm thresholded to the
    run's mean parent land fraction, same lattice, same ruler protocol.
    Per the 2026-08-31 calibration, the analytic rotation null fires on
    isotropic rasters; the run is judged AGAINST this baseline."""
    xs = (np.arange(N) + 0.5) * CK
    X, Y = np.meshgrid(xs, xs)
    rulers = []
    for s in seeds:
        salt = fnv1a64(f"plan-a-iso-baseline:{s}")
        f = periodic_fbm(X, Y, WORLD_KM, 3, 5, salt)
        mask = f >= np.quantile(f, 1.0 - land_frac)
        rs = rulers_for_mask(mask, CK, MIN_RULER_KM, periodic=True)
        for r in rs:
            r["seed"] = 100000 + s
        rulers.extend(rs)
    test = seed_blocked_d4_test(rulers)
    test["cohort"] = "isotropic_periodic_fbm"
    return test


# ------------------------------------------------------------ rendering

OCEAN_RGB = (24, 48, 78)
LAND_RGB = (206, 196, 166)
BELT_RGB = (150, 108, 78)
COAST_RGB = (240, 236, 220)


def render_seed(s, path):
    img = np.zeros((N, N, 3), np.uint8)
    img[:] = OCEAN_RGB
    # ocean age shading (older = darker)
    a = np.clip(s.age_myr / (s.eras * DT_MYR + 1500.0), 0.0, 1.0)
    for c in range(3):
        img[..., c] = np.where(
            ~s.cont,
            (OCEAN_RGB[c] * (1.0 - 0.35 * a)).astype(np.uint8),
            img[..., c])
    for c in range(3):
        img[..., c] = np.where(s.cont, LAND_RGB[c], img[..., c])
    b = np.clip(s.belt / max(float(s.belt.max()), 1e-9), 0.0, 1.0)
    strong = s.cont & (b > 0.15)
    for c in range(3):
        img[..., c] = np.where(
            strong,
            (LAND_RGB[c] * (1.0 - b) + BELT_RGB[c] * b).astype(np.uint8),
            img[..., c])
    for c in range(3):
        img[..., c] = np.where(s.coast, COAST_RGB[c], img[..., c])
    im = Image.fromarray(img, "RGB").resize((N * 3, N * 3), Image.NEAREST)
    im.save(path)
    return im


def render_labels(s, path):
    lab = s.label
    img = np.zeros((N, N, 3), np.uint8)
    for p in np.unique(lab[lab >= 0]):
        h = (p * 0.618033988749895) % 1.0
        rgb = _hsv(h, 0.45, 0.82 if not (p % 2) else 0.62)
        img[lab == p] = rgb
    img[s.cont] = (img[s.cont] * 0.55 + np.array(LAND_RGB) * 0.45
                   ).astype(np.uint8)
    Image.fromarray(img, "RGB").resize(
        (N * 3, N * 3), Image.NEAREST).save(path)


def _hsv(h, sat, val):
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = val * (1 - sat)
    q = val * (1 - f * sat)
    t = val * (1 - (1 - f) * sat)
    rgb = [(val, t, p), (q, val, p), (p, val, t),
           (p, q, val), (t, p, val), (val, p, q)][i]
    return tuple(int(255 * c) for c in rgb)


# ------------------------------------------------------------- pipeline

def run_seed(seed):
    t0 = time.perf_counter()
    s, diag = build_parent(seed)
    build_s = time.perf_counter() - t0
    cont = s.cont
    rulers = rulers_for_mask(cont, CK, MIN_RULER_KM, periodic=True)
    for r in rulers:
        r["seed"] = seed
    comps = roundness_summary(cont)
    rec = {
        "seed": seed,
        "build_s": round(build_s, 3),
        "land_frac": float(cont.mean()),
        "alive_plates": s.alive_plates,
        "components_ge_%d" % MIN_ROUND_CELLS: len(comps),
        "winding_components": sum(1 for c in comps if c["winds"]),
        "rounded_components": sum(1 for c in comps if c["rounded"]),
        "ruler_count": len(rulers),
        "self_overlap_cells": diag["self_overlap_cells"],
        "dropped_fresh_cells": diag["dropped_fresh_cells"],
        "seam": seam_stats(cont),
        "census": window_census(cont),
        "roundness": comps,
    }
    return s, rec, rulers


def main(argv):
    debug = "--debug" in argv
    if debug:
        seeds = [int(a) for a in argv[argv.index("--debug") + 1:]] or [3]
        outdir = ROOT / "out" / "plan_a_periodic_m1_debug"
    else:
        seeds = list(COHORT)
        outdir = ROOT / "out" / "plan_a_periodic_m1_run1"
    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    all_rulers = []
    montage = []
    for seed in seeds:
        s, rec, rulers = run_seed(seed)
        # determinism: rebuild the first seed and compare fields
        if seed == seeds[0]:
            s2, _ = build_parent(seed)
            rec["deterministic"] = bool(
                np.array_equal(s.cont, s2.cont)
                and np.array_equal(s.label, s2.label)
                and np.array_equal(s.age_myr, s2.age_myr)
                and np.array_equal(s.belt, s2.belt))
        records.append(rec)
        all_rulers.extend(rulers)
        im = render_seed(s, outdir / f"parent_cont_seed{seed}.png")
        montage.append((seed, im))
        render_labels(s, outdir / f"parent_label_seed{seed}.png")
        print(f"seed {seed}: land {rec['land_frac']:.3f} "
              f"build {rec['build_s']:.1f}s "
              f"rulers {rec['ruler_count']} "
              f"overlap {rec['self_overlap_cells']} "
              f"dropped {rec['dropped_fresh_cells']} "
              f"ring-water {rec['census']['ring_water_share']:.2f}")

    run_test = seed_blocked_d4_test(all_rulers) if all_rulers else None
    mean_land = float(np.mean([r["land_frac"] for r in records]))
    base_test = isotropic_baseline(mean_land, seeds)

    if len(montage) > 1:
        cols = 4
        rows = (len(montage) + cols - 1) // cols
        w, h = montage[0][1].size
        sheet = Image.new("RGB", (cols * w, rows * h), (10, 10, 10))
        for i, (_, im) in enumerate(montage):
            sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        sheet.save(outdir / "montage_cont.png")

    report = {
        "spike": "plan_a_periodic_m1",
        "date": "2026-08-31",
        "world_km": WORLD_KM, "n": N, "cell_km": CK,
        "plates": PLATES, "nuclei": NUCLEI,
        "budget_frames": BUDGET_FRAMES, "pad_cells": PAD,
        "window_cells": WINDOW_CELLS,
        "seeds": seeds,
        "records": records,
        "pooled_d4_run": run_test,
        "pooled_d4_isotropic_baseline": base_test,
        "mean_land_frac": mean_land,
    }
    (outdir / "report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")

    lines = ["# Plan A pilot report", ""]
    if run_test:
        lines.append(
            f"Pooled run rulers: {run_test['ruler_count']}, near-D4 "
            f"fraction {run_test['near_fraction']:.3f}, analytic-null "
            f"p={run_test['randomization_upper_tail_p']:.4f} "
            f"(evidence-grade only vs matched baseline)")
    lines.append(
        f"Matched isotropic baseline ({base_test['ruler_count']} rulers): "
        f"near-D4 fraction {base_test['near_fraction']:.3f}, "
        f"p={base_test['randomization_upper_tail_p']:.4f}")
    lines.append("")
    lines.append("| seed | land | build s | rulers | near-D4 | rounded/comps "
                 "| ring-water | low | med | high | seam max |z| | ovl | drop |")
    lines.append("|" + "---|" * 13)
    for rec in records:
        near = sum(1 for r in all_rulers
                   if r["seed"] == rec["seed"]
                   and abs((r["angle_degrees"] % 45.0)
                           if (r["angle_degrees"] % 45.0) <= 22.5
                           else 45.0 - (r["angle_degrees"] % 45.0)) <= 5.0)
        c = rec["census"]
        zmax = max(abs(v) for v in rec["seam"].values())
        lines.append(
            f"| {rec['seed']} | {rec['land_frac']:.3f} "
            f"| {rec['build_s']:.1f} | {rec['ruler_count']} | {near} "
            f"| {rec['rounded_components']}/{rec['components_ge_64']} "
            f"| {c['ring_water_share']:.2f} "
            f"| {c['low_15_25_ring_water_origins']} "
            f"| {c['med_30_40_ring_water_origins']} "
            f"| {c['high_45_50_ring_water_origins']} "
            f"| {zmax:.2f} | {rec['self_overlap_cells']} "
            f"| {rec['dropped_fresh_cells']} |")
    (outdir / "report.md").write_text("\n".join(lines) + "\n",
                                      encoding="utf-8")
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main(sys.argv[1:])
