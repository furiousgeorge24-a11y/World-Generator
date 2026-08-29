"""Stage: crust — continental potential via the border stack (design.md).

Interior-anchored nuclei (method B: compact-support kernels sampled away
from the frame) x shape noise x warped-margin falloff (method A) x an
absolute outer-ring floor (the contract 7 backstop, by construction — the
falloff multiplies potential before any landform exists; nothing is redrawn).

Nucleus *cluster centers* are placed by rejection sampling against the
plate mosaic (A1, `crust_plate_affinity`): a subset of plates is flagged
continent-carrying (size-weighted draw) and candidate centers are accepted
with probability (1-a) + a * interiority-on-those-plates. Correlated, not
coincident — kernels still scatter and spill across boundaries; only the
cores anchor. Centers draw from a dedicated substream so dragging the
knob slides continents without reshuffling their shapes, and candidates
are scored analytically in world-km (tectonics.sample_plate_affinity), so
placement is resolution-independent by construction.
"""

import numpy as np

from .noise import fbm
from .rng import rng_for, salts_for
from .tectonics import leading_edge, sample_plate_affinity
from .world import World

_TRIES = 48          # candidates per batch
_BATCHES = 3         # per cluster; then fall back to densest candidate seen

_LEAD_VN_MIN = 0.05      # closing rate below this: not a real trench
_LEAD_REACH = 1.0        # kernel-cloud effective radius, in r_base units
_LEAD_SETBACK_KM = 80.0  # crust margin stops short of the boundary


def _lead_shift(world: World, bias: float, u_lead: np.ndarray,
                cx: float, cy: float, cpl: int, q: float,
                ew: float, eh: float, r_base: float):
    """A2 (`active_margin_bias`): displace a cluster center toward its
    plate's convergent leading edge — the steady state of unmodeled
    drift is a continent jammed against its trench. No-op unless the
    plate drew leading status AND the march hits a closing boundary.
    The shift clamps to the placement margin q (border-stack guarantee:
    never pulls crust into the frame band). Consumes no RNG itself, so
    dragging the knob flips plates without reshuffling shapes."""
    info = {"plate": int(cpl), "leading": False, "applied": False,
            "hit_km": None, "neighbor": None, "vn": None, "shift_km": 0.0,
            "center_km": [round(cx, 1), round(cy, 1)]}
    if bias <= 0.0 or u_lead.size == 0:
        return cx, cy, info
    if cpl < 0:                    # affinity off: plate is still knowable
        pid, _ = sample_plate_affinity(world, np.array([cx]), np.array([cy]))
        cpl = int(pid[0])
        info["plate"] = cpl
    if not (0 <= cpl < u_lead.size) or u_lead[cpl] >= bias:
        return cx, cy, info
    info["leading"] = True
    hit = leading_edge(world, cpl, cx, cy)
    if hit is None:
        return cx, cy, info
    hx, hy, dist, nb, vn = hit
    info["hit_km"] = [round(hx, 1), round(hy, 1)]
    info["neighbor"] = nb
    info["vn"] = round(vn, 4)
    if vn <= _LEAD_VN_MIN or dist <= 0.0:
        return cx, cy, info
    keep = _LEAD_REACH * r_base + _LEAD_SETBACK_KM
    shift = max(0.0, dist - keep)
    if shift <= 0.0:
        return cx, cy, info
    ux, uy = (hx - cx) / dist, (hy - cy) / dist
    nx = float(np.clip(cx + ux * shift, q, ew - q))
    ny = float(np.clip(cy + uy * shift, q, eh - q))
    info["applied"] = True
    info["shift_km"] = round(float(np.hypot(nx - cx, ny - cy)), 1)
    info["from_km"] = [round(cx, 1), round(cy, 1)]
    info["center_km"] = [round(nx, 1), round(ny, 1)]
    return nx, ny, info


def _place_center(rng_p, a: float, cont: list[int], world: World,
                  q: float, ew: float, eh: float):
    """One cluster center via rejection sampling. Returns
    (x, y, plate_id, fell_back). a=0 accepts the first candidate."""
    use_aff = a > 0.0 and len(cont) > 0
    best = (-1.0, q, q, -1)
    for _ in range(_BATCHES):
        xs = q + rng_p.random(_TRIES) * (ew - 2.0 * q)
        ys = q + rng_p.random(_TRIES) * (eh - 2.0 * q)
        us = rng_p.random(_TRIES)
        if use_aff:
            pid, wgt = sample_plate_affinity(world, xs, ys)
            dens = (1.0 - a) + a * np.where(np.isin(pid, cont), wgt, 0.0)
        else:
            pid = np.full(_TRIES, -1, dtype=np.int16)
            dens = np.ones(_TRIES)
        for k in range(_TRIES):
            if dens[k] > best[0]:
                best = (float(dens[k]), float(xs[k]), float(ys[k]),
                        int(pid[k]))
            if us[k] < dens[k]:
                return float(xs[k]), float(ys[k]), int(pid[k]), False
    return best[1], best[2], best[3], True


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def stage_crust(world: World) -> None:
    c = world.controls
    rng = rng_for(world.seed, "crust")
    xkm, ykm = world.coords_km()
    eh, ew = world.extent_km
    min_e = min(eh, ew)
    area = eh * ew

    margin = float(c["border_sea_width"]) * min_e         # aesthetic band
    q = min(margin + 0.05 * min_e, 0.35 * min_e)          # placement margin

    # --- continent-carrying plates (A1): size-weighted, own substream ---
    aff = float(c["crust_plate_affinity"])
    rng_p = rng_for(world.seed, "crust:placement")
    n_clusters = int(c["continent_count"])
    cont: list[int] = []
    sampling = world.meta.get("plates", {}).get("sampling")
    if aff > 0.0 and sampling is not None:
        rates = np.asarray(sampling["rates"], dtype=np.float64)
        n_cont = max(1, min(len(rates), int(round(0.8 * n_clusters))))
        pw = rates ** 2                       # intended plate size as prior
        pick = rng_p.choice(len(rates), size=n_cont, replace=False,
                            p=pw / pw.sum())
        cont = sorted(int(p) for p in pick)

    # --- leading-plate draws (A2): own substream, one uniform per plate,
    # always drawn — dragging `active_margin_bias` flips plates
    # monotonically without reshuffling anything else -------------------
    bias = float(c["active_margin_bias"])
    n_plates = int(world.meta.get("plates", {}).get("count", 0))
    u_lead = rng_for(world.seed, "crust:leading").random(n_plates)
    lead_meta: list[dict] = []

    # --- nuclei (method B): compact bump kernels, interior-anchored -----
    kernels: list[tuple[float, float, float]] = []        # (cx, cy, r)
    per = rng.integers(2, 6, n_clusters)
    r_base = np.sqrt(float(c["land_fraction"]) * area * 1.7 / (np.pi * per.sum()))
    cluster_plates: list[int] = []
    cluster_fallback: list[bool] = []
    for ci in range(n_clusters):
        ccx, ccy, cpl, fb = _place_center(rng_p, aff, cont, world, q, ew, eh)
        ccx, ccy, li = _lead_shift(world, bias, u_lead, ccx, ccy, cpl,
                                   q, ew, eh, r_base)
        lead_meta.append(li)
        cluster_plates.append(cpl)
        cluster_fallback.append(fb)
        for _ in range(int(per[ci])):
            r = r_base * (0.55 + 0.9 * rng.random())
            kx = np.clip(ccx + rng.normal(0.0, 1.15 * r_base), 0.6 * q, ew - 0.6 * q)
            ky = np.clip(ccy + rng.normal(0.0, 1.15 * r_base), 0.6 * q, eh - 0.6 * q)
            kernels.append((float(kx), float(ky), float(r)))

    # Kernels are evaluated at domain-warped coordinates: the bumps
    # themselves stop being circles (blob-continent killer). Displacement
    # is bounded, so support stays bounded and the border stack holds.
    s = salts_for(world.seed, "crust", 15)
    irr = float(c["continent_irregularity"])
    wamp = (0.15 + 0.5 * irr) * r_base
    xw = xkm + wamp * fbm(xkm, ykm, 0.8 * r_base, s[0:3])
    yw = ykm + wamp * fbm(xkm, ykm, 0.8 * r_base, s[3:6])

    pot = np.zeros(world.shape, dtype=np.float64)
    for kx, ky, r in kernels:
        d2 = ((xw - kx) ** 2 + (yw - ky) ** 2) / (r * r)
        np.add(pot, np.maximum(0.0, 1.0 - d2) ** 3, out=pot)

    # --- landmass shape noise (multiplicative: support stays bounded) ---
    pot *= (1.0
            + 0.9 * irr * fbm(xkm, ykm, 0.9 * r_base, s[6:10])
            + 0.4 * irr * fbm(xkm, ykm, 0.28 * r_base, s[10:12]))
    np.maximum(pot, 0.0, out=pot)

    # --- guide-mask slot (authored placement density; feature later) ----
    gm = world.meta.get("guide_mask")
    if gm is not None:
        pot *= np.asarray(gm, dtype=np.float64)

    # --- warped-margin falloff (method A) -------------------------------
    d_edge = np.minimum(np.minimum(xkm, ew - xkm), np.minimum(ykm, eh - ykm))
    wander = float(c["border_irregularity"]) * 0.8 * margin
    d_eff = d_edge + wander * fbm(xkm, ykm, 0.35 * min_e, s[12:15])
    falloff = _smoothstep(d_eff / max(margin, 1e-9))
    pot *= falloff
    world["border_falloff"] = falloff.astype(np.float32)
    # Uplift gets a far tighter suppression than crust (issue 8): tectonics
    # may live near the margin sea; only the frame itself is off-limits.
    # Both factors are constructive (they scale sources, redraw nothing).
    uf = (_smoothstep(d_eff / max(0.35 * margin, 1e-9))
          * np.clip((d_edge - world.cell_km) / (2.0 * world.cell_km), 0.0, 1.0))
    world["uplift_falloff"] = uf.astype(np.float32)

    # --- absolute floor: outermost ring is water by construction --------
    pot[d_edge < world.cell_km] = 0.0

    # --- threshold to hit the land-fraction target ----------------------
    target = float(c["land_fraction"])
    thr = float(np.quantile(pot, 1.0 - target))
    if thr <= 0.0:
        pos = pot[pot > 0.0]
        thr = float(np.quantile(pos, 0.25)) if pos.size else 1.0
        world.findings.append(
            {"check": "crust", "level": "warn",
             "msg": "land_fraction target unreachable with current nuclei; "
                    "delivered what the potential supports"})
    crust = pot >= thr

    world["crust_potential"] = pot.astype(np.float32)
    world["crust"] = crust.astype(np.uint8)
    world.meta["crust"] = {"threshold": thr,
                           "p98": float(np.quantile(pot, 0.98)),
                           "kernels": len(kernels),
                           "affinity": {"value": aff,
                                        "cont_plates": cont,
                                        "cluster_plates": cluster_plates,
                                        "cluster_fallback": cluster_fallback},
                           "leading": {"bias": bias,
                                       "clusters": lead_meta}}
