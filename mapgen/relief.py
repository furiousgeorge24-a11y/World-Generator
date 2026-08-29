"""Stage: relief — two-regime elevation assembly (design.md; R1 replumb,
R3 fabric).

R3 additions: sharp crest spines and edged plateaus (kernel language),
era belts (worn ancient orogens), interior provinces (basins / shields /
raised interiors), tectonic grain (relief noise elongated along nearby
boundary tangents), and a coastal-complexity field (coast raggedness that
varies along the map — bold stretches, intricate stretches).
"""

import numpy as np

from .noise import fbm, value_noise
from .rng import salts_for
from .world import World

_MAX_OCT = 9
_POS_CAP = 4300.0   # feedback normalization reference only (K2: no tanh)
_CEIL = 4800.0      # isostatic ceiling: soft knee for stacked orogens


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _oct(scale_km: float, cell: float, cap: int = _MAX_OCT) -> int:
    fit = int(np.floor(np.log2(max(scale_km / (2.0 * cell), 2.0))))
    return max(1, min(cap, fit))


def stage_relief(world: World) -> None:
    c = world.controls
    pot = world["crust_potential"].astype(np.float64)
    thr = float(world.meta["crust"]["threshold"])
    p98 = max(float(world.meta["crust"]["p98"]), thr * 1.0001)
    xkm, ykm = world.coords_km()
    cell = world.cell_km

    pn0 = pot / max(thr, 1e-9)                  # provisional (pre-feedback)
    uf = world["uplift_falloff"].astype(np.float64)
    ocean_fade0 = _smoothstep((0.55 - pn0) / 0.55)

    # --- positive tectonics (K2): profile-painted anatomy stacks linearly;
    # saturation is applied per-orogen in the painting, so the only global
    # guard is a soft isostatic-ceiling knee for legitimate crossings of
    # independent belts (era belts over couplets, etc.) — replacing the
    # old whole-stack tanh, which compressed everything nonlinearly.
    pos = (world["tect_cordillera"].astype(np.float64)
           + world["tect_cordillera_n"].astype(np.float64)
           + world["tect_crest"].astype(np.float64)
           + world["tect_plateau"].astype(np.float64)
           + world["tect_apron"].astype(np.float64)
           + world["tect_foreland"].astype(np.float64)
           + world["tect_shoulder"].astype(np.float64)
           + world["tect_era_belt"].astype(np.float64)
           + (world["tect_arc_oo"].astype(np.float64)
              + world["tect_arc_comp"].astype(np.float64)) * ocean_fade0)
    pos = np.where(pos > _CEIL, _CEIL + (pos - _CEIL) * 0.22, pos)
    pos *= uf

    # --- feedback (R1): uplift bulges the potential; coast, shelf and
    # slope are all cut from the fed-back field ---------------------------
    # (K2: coefficient retuned 0.75 -> 0.55 — the linear profile stack
    # plus apron rows feeds back more area than the old tanh-capped sum)
    pn = (pot + thr * 0.55 * (np.maximum(pos, 0.0) / _POS_CAP)) / max(thr, 1e-9)

    # --- zones, with margin-typed shelf breadth (R1) ---------------------
    act = world["margin_activity"].astype(np.float64)
    shelf_w = float(c["shelf_width"])
    passive_lo = 0.62 - 0.45 * shelf_w
    s_lo = passive_lo + (0.78 - passive_lo) * act
    land = pn >= 1.0
    shelf = (pn >= s_lo) & ~land
    slope = (pn >= 0.10) & ~land & ~shelf

    # --- ocean base from the age law (R1) --------------------------------
    age = world["crust_age"].astype(np.float64)
    crest_lift = 1800.0 * float(c["ridge_swell"])
    ocean_base = -4400.0 + crest_lift * (1.0 - age) ** 1.3

    e = ocean_base.copy()
    dome = np.clip((pot - thr) / (p98 - thr), 0.0, 1.6) / 1.6
    e[land] = 25.0 + 320.0 * dome[land]
    e[shelf] = -150.0 + 148.0 * ((pn[shelf] - s_lo[shelf])
                                 / np.maximum(1.0 - s_lo[shelf], 1e-9))
    t_sl = (s_lo[slope] - pn[slope]) / np.maximum(s_lo[slope] - 0.10, 1e-9)
    e[slope] = -160.0 + (ocean_base[slope] + 160.0) * _smoothstep(t_sl)

    # --- interior provinces (R3): basins, shields, raised interiors ------
    prov_ctl = float(c["province_relief"])
    sp = salts_for(world.seed, "relief:prov", 3)
    p = fbm(xkm, ykm, 1400.0, sp)
    interior = _smoothstep((pn - 1.0) / 0.35)
    prov = (290.0 * p
            + 250.0 * _smoothstep((p - 0.28) / 0.22)
            - 210.0 * _smoothstep((-p - 0.28) / 0.22))
    e += prov_ctl * prov * interior

    # --- add features ----------------------------------------------------
    ocean_fade = _smoothstep((0.55 - pn) / 0.55)
    e += pos
    e += world["tect_graben"].astype(np.float64)
    e += world["tect_graben_w"].astype(np.float64)
    e += world["tect_backarc"].astype(np.float64) * ocean_fade
    e += (world["tect_hotspot"].astype(np.float64)
          + world["tect_hotspot_comp"].astype(np.float64)) * uf
    neg = (world["tect_trench"].astype(np.float64)
           + world["tect_axial"].astype(np.float64))
    e += -4200.0 * np.tanh(-neg / 4200.0) * ocean_fade
    e += (world["tect_fracture"].astype(np.float64)
          + world["tect_rise"].astype(np.float64)) * ocean_fade

    # --- relief noise, tectonically modulated ----------------------------
    rough = float(c["relief_roughness"])
    s = salts_for(world.seed, "relief", 12)
    land_noise = fbm(xkm, ykm, 640.0, s[0:_oct(640.0, cell)], gain=0.5)
    mont = np.clip(pos / 2200.0, 0.0, 1.3)
    amp_land = ((160.0 + 1300.0 * rough) * (0.30 + mont)
                * (1.0 + 0.35 * prov_ctl * p * interior))
    ocean_noise = fbm(xkm, ykm, 380.0, s[9:12])
    landish = pn >= s_lo
    e += np.where(landish, land_noise * amp_land,
                  ocean_noise * (90.0 + 260.0 * rough))

    # --- plains grain (K3): sub-grid texture of unresolved processes ----
    # Amplitude is process-modulated (mountains already carry noise; the
    # carve and deposition then work it: erosional country keeps grain,
    # valley floors get flattened back out of it). Applies to the whole
    # continental platform — the shelf was exposed land at lowstand.
    pg = float(c["plains_grain"])
    if pg > 0.0:
        sgp = salts_for(world.seed, "relief:plains", 5)
        grp = np.zeros(world.shape, dtype=np.float64)
        amp_o, wl, norm, used = 1.0, 55.0, 0.0, 0
        while wl >= 2.2 * cell and used < 5:   # world-km octaves down to
            grp += amp_o * value_noise(xkm, ykm, wl, sgp[used])  # ~cell scale
            norm += amp_o
            amp_o *= 0.62
            wl /= 2.0
            used += 1
        if used:
            grp /= norm
            amp_pg = (15.0 + 85.0 * pg) * (1.0 - 0.55 * np.clip(mont, 0.0, 1.0))
            e += grp * amp_pg * landish

    # --- tectonic grain (R3): noise elongated along boundary tangents ----
    grain_ctl = float(c["tectonic_grain"])
    if grain_ctl > 0.0:
        oc = world["tect_orient_c"].astype(np.float64)
        osn = world["tect_orient_s"].astype(np.float64)
        coh = np.hypot(oc, osn)
        ref = max(float(np.percentile(coh, 98.0)), 1e-9)
        coh = np.clip(coh / ref, 0.0, 1.0)
        theta = 0.5 * np.arctan2(osn, oc)
        u = np.cos(theta) * xkm + np.sin(theta) * ykm
        vq = (-np.sin(theta) * xkm + np.cos(theta) * ykm) * 2.6
        sg = salts_for(world.seed, "relief:grain", 3)
        grain = fbm(u, vq, 90.0, sg[0:_oct(90.0, cell, 3)])
        e += grain * (300.0 * grain_ctl * rough) * coh * landish

    # --- abyssal-hill fabric (C3): corduroy aligned to the ridge that
    # made the crust — orientation from the age gradient -------------------
    fab_ctl = float(c["seafloor_fabric"])
    if fab_ctl > 0.0:
        gya, gxa = np.gradient(age)
        gmag = np.hypot(gxa, gya)
        gref = max(float(np.percentile(gmag, 95.0)), 1e-12)
        gm = np.clip(gmag / gref, 0.0, 1.0)
        th = np.arctan2(gya, gxa)                # hills run perp to gradient
        uo = np.cos(th) * xkm + np.sin(th) * ykm
        vo = (-np.sin(th) * xkm + np.cos(th) * ykm) * 3.5
        sf = salts_for(world.seed, "relief:fabric", 2)
        fabric = fbm(uo, vo, 110.0, sf[0:_oct(110.0, cell, 2)])
        e += fabric * 95.0 * fab_ctl * (0.35 + 0.65 * gm) * ocean_fade

    # --- coastal complexity (R3): raggedness that varies along the map ---
    coast_ctl = float(c["coast_complexity"])
    if coast_ctl > 0.0:
        scst = salts_for(world.seed, "relief:coast", 6)
        band = np.exp(-(((pn - 1.0) / 0.18) ** 2))
        env = np.clip(0.5 + 0.5 * fbm(xkm, ykm, 900.0, scst[0:2]),
                      0.0, 1.0) ** 1.5
        fine = fbm(xkm, ykm, 120.0, scst[2:2 + _oct(120.0, cell, 4)])
        e += coast_ctl * 240.0 * band * env * fine

    # Frame guarantee stays constructive: potential, uplift, provinces and
    # coast band are all zero on the ring by their gates; the ocean base
    # there sits deeper than any bounded noise (contract section 7).

    world["elevation"] = e.astype(np.float32)   # sea level applies in tail

    landm = e >= 0.0
    coast = landm & ~(np.roll(landm, 1, 0) & np.roll(landm, -1, 0)
                      & np.roll(landm, 1, 1) & np.roll(landm, -1, 1))
    if coast.any():
        world.findings.append(
            {"check": "active_coast_fraction", "level": "info",
             "value": round(float((act[coast] > 0.45).mean()), 3)})
