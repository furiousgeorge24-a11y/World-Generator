"""Stage: erosion (M2/W2) — the carve.

Implicit Braun-Willett stream-power incision (n=1, m=1/2) with a
channel-initiation threshold: only cells with at least a few cells of
upstream catchment incise (hillslopes get diffusion only), so valleys cut
while ridges and interfluves stand — differential erosion is what makes
dissection *visible*, not just total volume. Solved cell-by-
cell in descending fill order against already-updated receivers: uncondi-
tionally stable, so a few large steps dissect deeply. Dissection-only by
design — no syn-erosion uplift re-injection — so `erosion_strength` keeps
one meaning: how carved the terrain is. Ridgelines survive because
incision scales with catchment; rivers grade to sea level (receiver pull
is floored near 0 m so coastal cells never dive toward the abyss).
Hillslope diffusion softens valley walls between cuts. Routing re-runs on
alternate steps (terrain moves slowly between cuts; cost halves).

K1 (drowned datum): the carve grades to the glacial *lowstand* coast,
-flood_rise_m, not to today's sea level — the shelf band is dissected as
exposed land and then drowned by the post-glacial rise, which is where
rias, drowned valleys, and shelf-crossing river scars come from. Today's
sea level never re-decides the carve; it just floods it.

K3 (mass balance): incision tapers continuously below the channel-
initiation catchment (no smooth/carved texture cliff), and the carved
mass is routed downstream and settled where carrying capacity drops —
valley-floor flats, filled closed basins, coastal wedges under the
lowstand sea — instead of vanishing. Hillslope diffusion has a no-flux
coast: it can soften a coastal plain but never drag a coastline down.

Young volcanic cones (hotspot chain heads, split in boundaries by
`volcano_youth`) are added after the loop: fresh edifices on dissected
terrain — the age split is ordering, not simulation.
"""

import numpy as np

from .hydrology import fill_depressions, flow_accumulation, flow_directions
from .world import World

_DIST8 = np.array([1.0, 1.0, 1.0, 1.0,
                   1.41421356, 1.41421356, 1.41421356, 1.41421356])


def _implicit_incise(e: np.ndarray, order: np.ndarray, dir8: np.ndarray,
                     recv: np.ndarray, acc: np.ndarray, K: float,
                     cell: float, floor: float, taper_p: float) -> np.ndarray:
    dx = _DIST8[np.clip(dir8, 0, 7)] * cell
    a_c = max(25.0, 3.5 * cell * cell)   # channel initiation (cells-relative)
    C = K * np.sqrt(np.maximum(acc, 0.0)) / dx
    # K3: incision tapers continuously into the hillslope regime instead of
    # switching off — plains get marked by fine valleys, no texture cliff
    C *= np.minimum(acc / a_c, 1.0) ** taper_p
    C[C < 1e-6] = 0.0
    C[dir8 < 0] = 0.0
    C[e < floor] = 0.0                   # below lowstand wave base: W3 owns it

    el = e.ravel().tolist()
    rl = recv.ravel().tolist()
    cl = C.ravel().tolist()
    for i in order.tolist():
        ci = cl[i]
        if ci <= 0.0:
            continue
        j = rl[i]
        if j < 0:
            continue
        ej = el[j]
        if ej < floor:
            ej = floor
        ei = el[i]
        if ei <= ej:
            continue
        el[i] = (ei + ci * ej) / (1.0 + ci)
    return np.asarray(el, dtype=np.float64).reshape(e.shape)


def _diffuse(e: np.ndarray, d: float, base: float) -> np.ndarray:
    """Hillslope diffusion on exposed land with a no-flux coast (K3 fix:
    the Laplacian previously read submerged neighbors, dragging coastal
    land down — diffusion must never move a coastline)."""
    m = e >= base
    p = np.pad(e, 1, mode="edge")
    pm = np.pad(m, 1, mode="edge")
    lap = np.zeros_like(e)
    for sl in ((slice(0, -2), slice(1, -1)), (slice(2, None), slice(1, -1)),
               (slice(1, -1), slice(0, -2)), (slice(1, -1), slice(2, None))):
        lap += np.where(pm[sl], p[sl], e) - e
    return e + d * lap * m


def _deposit(e_pre: np.ndarray, e_post: np.ndarray, order: np.ndarray,
             dir8: np.ndarray, recv: np.ndarray, cell: float, frac: float,
             base: float):
    """K3 mass balance: route this step's carved mass downstream and
    settle it where carrying capacity drops (capacity ~ slope). Valley
    floors flatten, closed basins fill toward flat, and below the
    lowstand coast the load builds wedges capped just under lowstand sea
    (never new land). Load passing the shelf edge is exported — the
    conceptual supply of W3's burial/fans. Per-step fill is capped for
    stability; re-routing every 3rd step resolves any local reversals.

    Returns (e, deposited, basin_trapped, exported) — volumes in
    metre-thickness units, caller scales to km^3."""
    _S_REF = 2.6        # m/km: below this slope rivers start dropping load
    _CAP = 3.5          # m per step per cell: stability bound on fill
    el = e_post.ravel().tolist()
    ml = np.maximum(e_pre - e_post, 0.0).ravel().tolist()
    rl = recv.ravel().tolist()
    dxl = (_DIST8[np.clip(dir8, 0, 7)] * cell).ravel().tolist()
    flux = [0.0] * len(el)
    dep = basin = exported = 0.0
    edge = base - 600.0
    for i in order.tolist():
        carry = flux[i] + ml[i]
        if carry <= 1e-9:
            continue
        ei = el[i]
        j = rl[i]
        if ei < base:                       # subaqueous (lowstand sea)
            d = min(carry * 0.5, _CAP, (base - 0.5) - ei)
            if d > 0.0:
                el[i] = ei + d
                dep += d
                carry -= d
            if j < 0 or el[j] < edge:       # off the shelf edge: exported
                exported += carry
            else:
                flux[j] += carry
            continue
        if j < 0:                           # closed basin: trap the load
            d = min(carry, 2.0 * _CAP)
            el[i] = ei + d
            dep += d
            basin += carry - d
            continue
        s = (ei - el[j]) / dxl[i]
        fs = frac * max(0.0, 1.0 - s / _S_REF)
        d = min(carry * fs, _CAP)
        if d > 0.0:
            el[i] = ei + d
            dep += d
            carry -= d
        flux[j] += carry
    return (np.asarray(el, dtype=np.float64).reshape(e_post.shape),
            dep, basin, exported)


def stage_erosion(world: World) -> None:
    c = world.controls
    strength = float(c["erosion_strength"])
    steps = int(c["erosion_steps"])
    cell = world.cell_km
    e = world["elevation"].astype(np.float64)
    base = -float(c["flood_rise_m"])     # K1: lowstand base level — rivers
    floor = base - 10.0                  # grade to the ice-age coast

    if strength > 0.0 and steps > 0:
        K = 0.25 * strength
        d = 0.10 * float(c["hillslope_smoothing"])
        taper_p = 8.0 / (1.0 + 7.0 * float(c["lowland_dissection"]))
        frac = float(c["deposition"])
        e0 = e.copy()
        dep_t = basin_t = exp_t = 0.0
        F = dir8 = recv = acc = order = None
        for step in range(steps):
            if step % 3 == 0 or F is None:      # re-route every 3rd step
                F, _ = fill_depressions(e)
                dir8, recv = flow_directions(F, cell)
                acc = flow_accumulation(F, recv, cell * cell)
                order = np.argsort(F, axis=None, kind="stable")[::-1]
                # keep the Python loops to the working set: exposed land
                # plus the wedge zone under the lowstand coast (deposition
                # reaches it; the deep floor beyond can only export)
                order = order[F.ravel()[order] >= floor - 600.0]
            e_pre = e
            e = _implicit_incise(e, order, dir8, recv, acc, K, cell,
                                 floor, taper_p)
            if frac > 0.0:
                e, dp, bs, ex = _deposit(e_pre, e, order, dir8, recv,
                                         cell, frac, base)
                dep_t += dp
                basin_t += bs
                exp_t += ex
            if d > 0.0:
                e = _diffuse(e, d, base)
        world.meta["rough_acc"] = acc.astype(np.float32)
        eroded = np.maximum(e0 - e, 0.0)
        shelf = (e > base) & (e < 0.0)   # carved while exposed, drowned now
        vol = cell * cell / 1e3
        world.findings.append(
            {"check": "erosion", "level": "info", "steps": steps,
             "eroded_volume_km3": round(float(eroded.sum()) * vol, 1),
             "shelf_incision_km3": round(float(eroded[shelf].sum()) * vol, 1),
             "deposited_km3": round(dep_t * vol, 1),
             "basin_trapped_km3": round(basin_t * vol, 1),
             "exported_km3": round(exp_t * vol, 1),
             "max_incision_m": round(float(eroded.max()), 1)})

    if "tect_hotspot_y" in world.layers:
        uf = world["uplift_falloff"].astype(np.float64)
        e = e + (world["tect_hotspot_y"].astype(np.float64)
                 + world["tect_hotspot_y_comp"].astype(np.float64)) * uf

    world["elevation"] = e.astype(np.float32)
