"""B20 pilot: arc-accretion continental growth on the periodic parent.

Author-authorized (2026-09-01) formation extension addressing the
high-band supply gap and the ref12 aesthetic (one-sided mountain-spine
islands / suture-welded large landmasses). Mechanism grounded in real
accretionary orogenesis: sustained subduction manufactures juvenile
continental crust in the volcanic-arc band BEHIND the trench on the
OVERRIDING plate; arc complexes ride their plates, collide, and weld.

Design rules honoured:
- B17 lesson: convergence dose is a crust SOURCE, never a final mask.
  Land identity is created cell-by-cell in plate material frames and
  then lives under the ordinary approved process (persistence,
  subduction, belts, welding).
- No geometric dilation-as-growth: the arc band is a physical standoff
  (80-280 km behind the front, where the melt column sits); growth in
  time comes from boundary wander, plate motion, and welding — process,
  not painting.
- Patchy emergence: dose rate is modulated by static bounded noise in
  each plate's MATERIAL frame (edifice spacing ~220 km), so maturation
  crosses threshold at different times along-strike -> island chains
  first, welded strips later (fights the B17 ribbon failure and gives
  natural low-band dispersion).
- Overrider-only: a plate accrues dose only near collision cells that
  plate itself WON, so the subducting side grows nothing.
- §4: the arc stream uses its own salt ("tect-arc"); kinematics,
  partition, nuclei draws are untouched. With arc_productivity=0 the
  builder reproduces the B18 parent BIT-EXACTLY (checked in --debug).

Everything stays in spikes/; engine untouched. Register entry: B20.

Usage:
  python spikes/arc_accretion_m1.py --debug    # seeds 3, 7 + parity
  python spikes/arc_accretion_m1.py            # cohort 151-158 + finals
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
from engine.rng import stage_rng, stage_salt
from engine.tectonics import (CONT_BORN, DT_MYR, EVENT_MEMORY, FRAME_KM,
                              Config, Structure)
from spikes.plan_a_periodic_m1 import (BUDGET_FRAMES, CK, COHORT, M, N, PAD,
                                       NUCLEI, PLATES, WORLD_KM,
                                       _pdilate, _pfill_owner, _proll,
                                       _TorusPlate, build_parent,
                                       partition_periodic, render_seed,
                                       seam_stats, seed_nuclei_periodic,
                                       roundness_summary, window_census,
                                       isotropic_baseline, MIN_RULER_KM)
from spikes.plan_a_crop_adapter import (FRAME_LOCAL_KM, extract_structure,
                                        run_window, select_windows)
from eval.geometry_instruments import rulers_for_mask, seed_blocked_d4_test

# --- arc mechanism constants (frozen after seed-3/7 debug, before the
# --- cohort run; see PREDICTIONS.md)
ARC_PLATES = 7        # plate diameter (~3900 km) ~ window size, so
                      # windows can sit inside plate interiors (Earth
                      # relation; 12 window-sized-or-smaller plates put
                      # boundary arcs in every window ring)
ARC_NUCLEI = 5        # two plates stay craton-free (arc/ocean domains)
ARC_BUDGET_FRAMES = 2.0 * 5.0 / 6.0   # per-nucleus budget preserved
ARC_NEAR = 2          # forearc gap, cells (nothing matures closer)
ARC_FAR = 12          # 480 km: full arc + backarc complex reach
                      # (Indonesia/Philippines-scale terrane belt);
                      # the patch gates carve archipelago structure
                      # inside it. A migrating-front variant was tried
                      # and REDUCED yield (narrow instantaneous
                      # exposure lost more than sweeping gained).
ARC_RATE = 1.0        # dose per era of adjacent won convergence
ARC_T = 4.0           # maturation threshold (median ~4 sustained eras)
ARC_PATCH_LAM = 220.0  # km, edifice-spacing patchiness wavelength
ARC_PATCH_CUT = 0.12   # fbm below this NEVER matures (discrete edifice
                       # clusters, not continuous snakes — B17 lesson)
ARC_PATCH_GAIN = 14.0  # rate scale above the cutoff: productive-
                       # segment cells mature in ~2 sustained eras, so
                       # concentration redistributes mass rather than
                       # deleting it
ARC_SEG_LAM = 1600.0   # km, plate-scale productivity segmentation:
                       # only ~40% of margin length hosts arcs at all
                       # (slab age / convergence vigor analogue), so
                       # quiet corridors stay open water
ARC_SEG_CUT = 0.10
ARC_SEG_KNEE = 8.0


def build_parent_arc(seed, cfg=None, arc_productivity=1.0,
                     budget_frames=None):
    """The B18 periodic builder with the arc-accretion block added at
    the end of each era. With arc_productivity=0 and the B18 config
    (plates=12, nuclei=6, budget_frames=2.0) this takes the exact B18
    path; the pilot default is the large-plate arc configuration."""
    if cfg is None:
        cfg = Config(plates=ARC_PLATES, nuclei=ARC_NUCLEI)
        if budget_frames is None:
            budget_frames = ARC_BUDGET_FRAMES
    elif budget_frames is None:
        budget_frames = BUDGET_FRAMES
    t_all = time.perf_counter()
    diag = {"self_overlap_cells": 0, "dropped_fresh_cells": 0,
            "arc_matured_cells": 0, "subd_by_era": [],
            "band_by_era": [], "matured_by_era": []}
    xs = (np.arange(N) + 0.5) * CK
    X, Y = np.meshgrid(xs, xs)

    label0 = partition_periodic(seed, cfg)
    plates = [_TorusPlate() for _ in range(cfg.plates)]
    rng_init = stage_rng(seed, "tect-initial-age")
    ocean_born = -rng_init.integers(0, 8, (N, N)).astype(np.int16)
    for p in range(cfg.plates):
        pl = plates[p]
        oy_cells, ox_cells = np.nonzero(label0 == p)
        from spikes.plan_a_periodic_m1 import circ_center_cell
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
                         cfg.continental_budget * frame_cells * budget_frames)
    for p in range(cfg.plates):
        plates[p].born[plates[p].cont] = CONT_BORN

    # arc state: per-plate dose accumulator + lazily evaluated static
    # patchiness field in MATERIAL km (frame-free, rides the plate)
    arc_salt = stage_salt(seed, "tect-arc")
    arc_dose = [np.zeros((M, M), np.float32) for _ in range(cfg.plates)]
    arc_patch = [None] * cfg.plates

    def patch_field(p):
        if arc_patch[p] is None:
            pl = plates[p]
            my = (pl.oy + np.arange(M) + 0.5)[:, None] * CK
            mx = (pl.ox + np.arange(M) + 0.5)[None, :] * CK
            f = noise.fbm(mx, my, ARC_PATCH_LAM, 4, arc_salt + 977 * p)
            seg = np.clip((noise.fbm(mx, my, ARC_SEG_LAM, 3,
                                     arc_salt + 5501 * p)
                           - ARC_SEG_CUT) * ARC_SEG_KNEE, 0.0, 1.0)
            arc_patch[p] = (seg * ARC_PATCH_GAIN
                            * np.clip(f - ARC_PATCH_CUT, 0.0, None)
                            ).astype(np.float32)
        return arc_patch[p]

    rng_k = stage_rng(seed, "tect-kinematics")
    ang = rng_k.uniform(0.0, 2 * np.pi, cfg.plates)
    speed = rng_k.uniform(0.6, 1.4, cfg.plates) * cfg.plate_speed
    omega = rng_k.normal(0.0, 0.004, cfg.plates) * (cfg.plate_speed / 45.0)

    conv_hist = [np.zeros((N, N), bool) for _ in range(EVENT_MEMORY)]
    div_hist = [np.zeros((N, N), bool) for _ in range(EVENT_MEMORY)]

    def image_offsets(pl):
        corners = []
        for i in (0.0, float(M)):
            for j in (0.0, float(M)):
                q = pl.T.a @ np.array([(pl.oy + i) * CK, (pl.ox + j) * CK])
                corners.append(q + pl.T.b)
        corners = np.array(corners)
        ylo, xlo = corners.min(0)
        yhi, xhi = corners.max(0)
        return [(a, b)
                for a in range(int(np.floor(ylo / WORLD_KM)) - 1,
                               int(np.floor(yhi / WORLD_KM)) + 2)
                for b in range(int(np.floor(xlo / WORLD_KM)) - 1,
                               int(np.floor(xhi / WORLD_KM)) + 2)]

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
                diag["self_overlap_cells"] += int(
                    np.count_nonzero(hit & mask))
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
        subd = np.zeros((N, N), bool)   # oceanic crust consumed here
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
                subd |= lose_oc
                plates[p].exists[iy[lose_oc], ix[lose_oc]] = False
                for q in np.unique(label_[lose]):
                    sel = lose & (label_ == q)
                    amt = np.where(cont_p[sel], 2.0, 1.0).astype(np.float32)
                    np.add.at(plates[q].belt, (wmiy[sel], wmix[sel]), amt)
                    plates[q].belt_age[wmiy[sel], wmix[sel]] = era
            if mutate and disp.any():
                subd |= disp   # displaced winners are oceanic by rule
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
        return label_, conv, subd, win_cont, wmiy, wmix

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
        label, conv, subd, win_cont, wmiy, wmix = resolve(claims, era,
                                                          mutate=True)

        # --- arc accretion (the B20 mechanism) -----------------------
        diag["subd_by_era"].append(int(subd.sum()))
        band_total = 0
        matured_total = diag["arc_matured_cells"]
        if arc_productivity > 0.0 and subd.any():
            near = _pdilate(subd, ARC_NEAR)
            for p in range(cfg.plates):
                won_conv = subd & (label == p)
                if not won_conv.any():
                    continue
                # standoff annulus on the overrider's own crust behind
                # collision cells THIS plate won; oceanic cells only
                band = (_pdilate(won_conv, ARC_FAR) & ~near
                        & (label == p) & ~win_cont)
                if not band.any():
                    continue
                band_total += int(band.sum())
                miy = wmiy[band]
                mix = wmix[band]
                dose = arc_dose[p]
                dose[miy, mix] += (ARC_RATE * arc_productivity
                                   * patch_field(p)[miy, mix])
                pl = plates[p]
                ripe = (dose[miy, mix] >= ARC_T) & ~pl.cont[miy, mix]
                if ripe.any():
                    ry, rx = miy[ripe], mix[ripe]
                    pl.cont[ry, rx] = True
                    pl.born[ry, rx] = era   # young continental crust
                    diag["arc_matured_cells"] += int(ripe.sum())
        diag["band_by_era"].append(band_total)
        diag["matured_by_era"].append(
            diag["arc_matured_cells"] - matured_total)
        # -------------------------------------------------------------

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

    claims = rasterize()
    label, _, _, _, _, _ = resolve(claims, cfg.eras, mutate=False)
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
        lab_s, _, _, _, _, _ = resolve(cl, cfg.eras, mutate=False)
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
    s._periodic = True
    s._rotation = True
    s._arc_productivity = arc_productivity
    return s, diag


def parity_check(seed):
    """arc_productivity=0 under the B18 config must reproduce the B18
    parent bit-exactly."""
    a, _ = build_parent_arc(seed, cfg=Config(plates=12, nuclei=6),
                            arc_productivity=0.0, budget_frames=2.0)
    b, _ = build_parent(seed)
    same = (np.array_equal(a.cont, b.cont)
            and np.array_equal(a.label, b.label)
            and np.array_equal(a.age_myr, b.age_myr)
            and np.array_equal(a.belt, b.belt)
            and np.array_equal(a.cont_frac, b.cont_frac))
    return same


def main(argv):
    debug = "--debug" in argv
    seeds = [3, 7] if debug else list(COHORT)
    outdir = ROOT / "out" / ("arc_accretion_m1_debug" if debug
                             else "arc_accretion_m1_run1")
    outdir.mkdir(parents=True, exist_ok=True)

    if debug:
        ok = parity_check(3)
        print(f"parity (arc=0 vs B18, seed 3): {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    records = []
    all_rulers = []
    finals = []
    for seed in seeds:
        t0 = time.perf_counter()
        s, diag = build_parent_arc(seed)
        build_s = time.perf_counter() - t0
        if seed == seeds[0]:
            s2, _ = build_parent_arc(seed)
            deterministic = bool(np.array_equal(s.cont, s2.cont)
                                 and np.array_equal(s.belt, s2.belt))
        else:
            deterministic = None
        cont = s.cont
        rulers = rulers_for_mask(cont, CK, MIN_RULER_KM, periodic=True)
        for r in rulers:
            r["seed"] = seed
        all_rulers.extend(rulers)
        comps = roundness_summary(cont)
        rec = {
            "seed": seed, "build_s": round(build_s, 2),
            "deterministic": deterministic,
            "land_frac": float(cont.mean()),
            "arc_matured_cells": diag["arc_matured_cells"],
            "self_overlap_cells": diag["self_overlap_cells"],
            "dropped_fresh_cells": diag["dropped_fresh_cells"],
            "ruler_count": len(rulers),
            "components": len(comps),
            "rounded_components": sum(1 for c in comps if c["rounded"]),
            "seam": seam_stats(cont),
            "census": window_census(cont),
        }
        records.append(rec)
        render_seed(s, outdir / f"parent_cont_seed{seed}.png")
        c = rec["census"]
        print(f"seed {seed}: land {rec['land_frac']:.3f} "
              f"(+arc {diag['arc_matured_cells']}) build {build_s:.1f}s "
              f"ring-water {c['ring_water_share']:.2f} "
              f"low/med/high rw {c['low_15_25_ring_water_origins']}"
              f"/{c['med_30_40_ring_water_origins']}"
              f"/{c['high_45_50_ring_water_origins']} "
              f"maxrw {c['max_land_frac_ring_water']:.3f}")

        # delivered finals: best ring-water window per populated band
        for pick in select_windows(cont):
            frec, img = run_window(s, seed, pick, outdir)
            finals.append(frec)
            print(f"   -> {pick['band']}: census "
                  f"{pick['census_land_frac']:.3f} delivered "
                  f"{frec['delivered_land_frac']:.3f} ring "
                  f"{'WATER' if frec['ring_is_all_water'] else 'LAND ' + str(frec['ring_land_px'])}")

    run_test = seed_blocked_d4_test(all_rulers) if all_rulers else None
    mean_land = float(np.mean([r["land_frac"] for r in records]))
    base_test = isotropic_baseline(mean_land, seeds)

    fin_imgs = [outdir / f["image"] for f in finals]
    if len(fin_imgs) > 1:
        tw = 512
        cols = 4
        rows = (len(fin_imgs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, rows * tw), (12, 12, 12))
        for i, path in enumerate(fin_imgs):
            sheet.paste(Image.open(path).resize((tw, tw), Image.LANCZOS),
                        ((i % cols) * tw, (i // cols) * tw))
        sheet.save(outdir / "montage_final.png")

    report = {
        "spike": "arc_accretion_m1", "date": "2026-09-01",
        "arc_constants": {"near": ARC_NEAR, "far": ARC_FAR,
                          "rate": ARC_RATE, "threshold": ARC_T,
                          "patch_lam_km": ARC_PATCH_LAM,
                          "patch_cut": ARC_PATCH_CUT,
                          "patch_gain": ARC_PATCH_GAIN},
        "seeds": seeds, "records": records, "finals": finals,
        "pooled_d4_run": run_test,
        "pooled_d4_isotropic_baseline": base_test,
        "mean_land_frac": mean_land,
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=1),
                                        encoding="utf-8")
    print(f"\npooled D4: run {run_test['near_fraction']:.3f} "
          f"({run_test['ruler_count']} rulers, "
          f"p={run_test['randomization_upper_tail_p']:.4f}) vs isotropic "
          f"{base_test['near_fraction']:.3f} "
          f"({base_test['ruler_count']} rulers)")
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main(sys.argv[1:])
