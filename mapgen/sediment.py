"""Stage: sediment (M2/W3, K1) — the sea working the drowned surface.

Marine-only (touches cells below today's sea level exclusively):
wave-base planation of the lowstand shoreline (K1 — the crisp fossil
shelf break; cut-only, drowned valleys survive); masked-blur "burial"
that mutes the shelf and builds the continental-rise apron (drowned
valleys are muted, never erased — design.md's shelf grammar); submarine
canyons continuing rivers from the lowstand coast down the slope;
deep-sea fans at their feet. Deposition is capped by depth headroom, so
sediment can never surface as new land (a constructive cap on the
addition, not a redraw of the result).
"""

import numpy as np

from .boundaries import _fft_gauss
from .hydrology import fill_depressions, flow_accumulation, flow_directions
from .world import World

_OFF8 = ((-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1))


def _band(e: np.ndarray, hi: float, lo: float, feather: float) -> np.ndarray:
    """1 inside [lo, hi], smooth feather outside."""
    up = np.clip((e - lo) / feather, 0.0, 1.0)
    dn = np.clip((hi - e) / feather, 0.0, 1.0)
    return up * up * (3 - 2 * up) * dn * dn * (3 - 2 * dn)


def _masked_blur(e: np.ndarray, mask: np.ndarray, sigma_c: float) -> np.ndarray:
    num = _fft_gauss(e * mask, sigma_c)
    den = _fft_gauss(mask.astype(np.float64), sigma_c)
    return num / np.maximum(den, 1e-9)


def stage_sediment(world: World) -> None:
    c = world.controls
    cell = world.cell_km
    e = world["elevation"].astype(np.float64)
    flood = float(c["flood_rise_m"])
    n_canyon = n_fan = 0

    # --- wave-base planation (K1): the fossil shelf break ---------------
    # Waves planed interfluve highs at the lowstand shoreline (long
    # stillstand), and the transgressing surf bevelled the whole flooded
    # band on its way up. Cut-only, strongest at the old coast; drowned
    # valleys sit below the local mean surface and survive. Never touches
    # cells above -2 m, so today's coastline is untouched by construction.
    plan = float(c["wave_planation"])
    if plan > 0.0 and flood > 0.0:
        mu = _fft_gauss(e, max(10.0 / cell, 1.0))
        # positive relief only, capped at a physical ravinement thickness —
        # near cliffs (shelf break, trench walls) e - mu is thousands of
        # metres and is *structure*, not an interfluve to plane off
        bump = np.minimum(np.maximum(e - mu, 0.0), 45.0)
        wband = _band(e, -flood + 25.0, -flood - 35.0, 30.0)
        tband = _band(e, -2.0, -flood, 40.0)
        shave = np.clip(plan * (0.85 * wband + 0.35 * tband), 0.0, 0.95)
        e = e - bump * shave * (e < -2.0)

    ocean = e < 0.0

    # --- burial: mute shelf, build the rise apron -----------------------
    soft = float(c["sediment_softening"])
    if soft > 0.0 and ocean.any():
        om = ocean.astype(np.float64)
        b_shelf = _masked_blur(e, om, max(26.0 / cell, 1.2))
        b_rise = _masked_blur(e, om, max(48.0 / cell, 1.5))
        w = (0.55 * soft * _band(e, -2.0, -280.0, 120.0)
             + 0.75 * soft * _band(e, -2200.0, -4300.0, 500.0))
        w = np.clip(w, 0.0, 0.9) * ocean
        blend = np.where(e > -1000.0, b_shelf, b_rise)
        e = e * (1.0 - w) + np.minimum(blend, -2.0) * w

    # --- major river mouths ---------------------------------------------
    fan_ctl = float(c["fan_size"])
    cny_ctl = float(c["canyon_depth"])
    if (fan_ctl > 0.0 or cny_ctl > 0.0) and ocean.any():
        acc = world.meta.get("rough_acc")
        if acc is None:
            F, _ = fill_depressions(e)
            d8, recv = flow_directions(F, cell)
            acc = flow_accumulation(F, recv, cell * cell)
        acc = np.asarray(acc, dtype=np.float64)
        h, w_ = world.shape
        land = e >= 0.0
        # K1: canyons continue rivers from the *lowstand* coastline — the
        # coast the rivers actually graded to — down the slope.
        ls = e >= -flood
        coastal = ls & ~(np.roll(ls, 1, 0) & np.roll(ls, -1, 0)
                         & np.roll(ls, 1, 1) & np.roll(ls, -1, 1))
        cand = np.argwhere(coastal & (acc > 12.0 * cell * cell))
        if len(cand):
            order = np.argsort(-acc[cand[:, 0], cand[:, 1]], kind="stable")
            cand = cand[order]
            n_max = int(np.clip(round(land.sum() * cell * cell / 2.5e5),
                                4, 20))
            mouths: list[tuple[int, int]] = []
            min_d = 90.0 / cell
            for r, ccol in cand:
                if len(mouths) >= n_max:
                    break
                if all((r - mr) ** 2 + (ccol - mc) ** 2 >= min_d ** 2
                       for mr, mc in mouths):
                    mouths.append((int(r), int(ccol)))

            cny_src = np.zeros(world.shape)
            fan_src = np.zeros(world.shape)
            sig_cny = max(9.0 / cell, 1.2)
            norm_cny = np.sqrt(np.pi) * sig_cny
            a_ref = max(float(acc[coastal].max()), 1.0)
            for mr, mc in mouths:
                a_rel = (float(acc[mr, mc]) / a_ref) ** 0.3
                # walk downslope into the ocean, carving as we go
                r, ccol = mr, mc
                path: list[tuple[int, int]] = []
                for _step in range(int(220.0 / cell) + 2):
                    best, br, bc = e[r, ccol], -1, -1
                    for dr, dc in _OFF8:
                        r2, c2 = r + dr, ccol + dc
                        if 0 <= r2 < h and 0 <= c2 < w_ and e[r2, c2] < best:
                            best, br, bc = e[r2, c2], r2, c2
                    if br < 0:
                        break
                    r, ccol = br, bc
                    if e[r, ccol] < -flood:
                        path.append((r, ccol))
                    if e[r, ccol] < -3400.0:
                        break
                if len(path) >= 3:
                    n_canyon += 1
                    for k, (pr, pc) in enumerate(path):
                        t = k / max(len(path) - 1, 1)
                        cny_src[pr, pc] -= (330.0 * cny_ctl * a_rel
                                            * (1.0 - 0.75 * t) / norm_cny)
                    fr, fc = path[-1]
                    fan_src[fr, fc] += 620.0 * fan_ctl * a_rel
                    n_fan += 1

            if n_canyon and cny_ctl > 0.0:
                cf = _fft_gauss(cny_src, sig_cny)
                e += cf * (e < 0.0)
            if n_fan and fan_ctl > 0.0:
                sig_f = max(52.0 / cell, 1.5)
                ff = _fft_gauss(fan_src, sig_f)
                headroom = np.maximum(-60.0 - e, 0.0)
                e += np.minimum(ff, 0.65 * headroom) * (e < -0.0)

    world["elevation"] = e.astype(np.float32)
    world.findings.append({"check": "sediment", "level": "info",
                           "canyons": n_canyon, "fans": n_fan})
