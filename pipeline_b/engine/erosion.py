"""M3 surface-process stage: coupled fluvial erosion, sediment routing,
and lakes on a FIXED world-domain process grid.

Architecture (each choice traceable to a recorded lesson):
- The solver runs at E_KM per cell over the WHOLE world, independent of
  output resolution — §2 resolution independence by construction (the
  same argument as the tectonic lattice), and drainage can never be a
  function of frame coordinates because the frame does not exist here
  (§3b). Output stages sample this grid.
- Mechanics productionized from spike S2 (see MILESTONES): epsilon
  depression-fill by directional sweeps, D8 steepest receivers with
  slope-weighted MFD accumulation, vectorized-Kahn topological batches,
  implicit stream-power incision processed downstream-first
  (unconditionally stable — a few large steps).
- The S2 axis-locking hazard is countered by all three named M-phase
  fixes: MFD participation in discharge, process-modulated erodibility
  heterogeneity (km-space noise + harder belt rock), and hillslope
  diffusion (soil creep) — shipped as a control whose range includes 0.
- Incision runs against a LOWSTAND base level; the present sea floods
  back afterwards (lowstand-then-flood): drowned valleys, estuaries,
  shelf channels.
- Sediment: eroded rock routes downstream; on land it deposits where
  transport capacity drops (floodplains, valley fill); at sea it
  settles with an e-folding travel distance (delta/shelf wedges near
  mouths, thinning fans basinward, a quiet drape beyond) — land
  erosion and bathymetry share one budget (§6e floating islands are
  impossible rather than forbidden).
- Lakes: depressions of the final surface hold water only if fed by
  routed drainage (crude water balance) — lakes get inlets and outlet
  context by construction (§9, M2 eval "lakes without drainage").
"""

import time

import numpy as np

from . import noise
from .rng import stage_salt
from .surface import _bicubic, _smooth01

E_KM = 20.0            # process-grid cell (fixed, world domain)
EPS = 1e-5             # routing-plumbing fill gradient (S2 lesson)
M_EXP = 0.5            # stream-power area exponent
K_BASE = 0.0065        # erodibility scale (1/Myr per sqrt(km^2)/km)
N_STEPS = 2            # implicit steps (unconditionally stable)
CREEP_DIFFUSIVITY_KM2_MYR = 8.8
# At the 20-km process spacing and a 10-Myr solver step, 8.8 km^2/Myr
# is .22 total explicit diffusion: the shipped two .11 substeps. The
# strength is now derived from elapsed time instead of merely testing
# whether erosion_time is nonzero.
CREEP_MIN_SUBSTEPS = 2
CREEP_MAX_ALPHA = 0.20  # safely below the 2-D explicit stability limit
KC_LAND = 0.9          # transport-capacity coefficient on land
DEP_CAP = 30.0         # max land deposition per cell (m) — capped so
                       # young structural depressions are not entirely
                       # buried within the window (lakes survive)
MAR_CAP = 150.0        # max marine deposition per cell (m)
# Private local-process experiment: after this many axial settling-length
# equivalents, unresolved suspension leaves the modeled coastal fan as
# far-field export. Diagonal links make the conservative spatial reach
# sqrt(2) larger and diagnostics report that full bound. This is a finite
# recent-process reach, not a crop-relative fade; callers must provide at
# least the reported bound as halo.
LOCAL_MARINE_EFOLDS = 6.0
# Private successor experiment: a finite depositional episode with enough
# settling time to remove 99.59% of unobstructed suspension.  The resulting
# compact support is about 1,414 km diagonally at the default 180-km settling
# length and 20-km process spacing, inside the frozen 1,550-km minimum halo.
# Unlike ``LOCAL_MARINE_EFOLDS``, this path has no arbitrary thickness cap;
# lobe switching follows aggradation against physical lowstand accommodation.
PHYSICAL_MARINE_EFOLDS = 5.5
LAKE_MIN_DEPTH = 4.0   # m below balance level to count as a lake
LAKE_FEED_CELLS = 10.0 # drainage cells that must feed a lake
WATER_YIELD = 0.25     # lake area per unit of water inflow (runoff /
                       # evaporation balance): basins hold water to
                       # the level their inflow sustains, capped at
                       # the spill. Combined with continentality this
                       # yields floor lakes in dry interiors and
                       # overflowing chain lakes on wet through-rivers
L_MOIST_KM = 1100.0    # continentality: moisture e-folding distance
                       # inland from the ocean (crude climate-lite;
                       # the full climate stage is M4+ scope)

NBR = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
       (0, 1), (1, -1), (1, 0), (1, 1)]
NBR_D = np.array([np.hypot(dy, dx) for dy, dx in NBR])


def _shiftf(F, dy, dx, fill):
    G0, G1 = F.shape
    sh = np.full_like(F, fill)
    sh[max(0, dy):G0 + min(0, dy), max(0, dx):G1 + min(0, dx)] = \
        F[max(0, -dy):G0 + min(0, -dy), max(0, -dx):G1 + min(0, -dx)]
    return sh


def fill_depressions(h, max_rounds=8):
    """Epsilon depression-fill by leak-free directional Gauss-Seidel
    reconstruction.

    NOTE: the S2 spike's in-row min-plus shortcut was WRONG — it
    propagated min(row[k] + eps*(j-k)) without clamping at the heights
    of the cells BETWEEN k and j, so reconstruction tunnelled through
    ridges and silently under-filled walled basins (caught by the M3
    known-positive lake calibration test; the spike's own tests never
    exercised a rimmed basin on a tilted plain). Here every
    propagation step clamps at the receiving cell's own height, so
    nothing leaks; four sweep directions (each using its
    already-updated neighbor line and that line's diagonals) give
    8-connected propagation, converging in a few rounds on terrain."""
    G0, G1 = h.shape
    F = np.full_like(h, np.inf)
    F[0, :] = h[0, :]
    F[-1, :] = h[-1, :]
    F[:, 0] = h[:, 0]
    F[:, -1] = h[:, -1]

    def relax_from(prev, cur_h, cur_F):
        cand = np.minimum(prev,
                          np.minimum(np.r_[np.inf, prev[:-1]],
                                     np.r_[prev[1:], np.inf])) + EPS
        return np.maximum(cur_h, np.minimum(cur_F, cand))

    for _ in range(max_rounds):
        Fprev = F.copy()
        for i in range(1, G0 - 1):
            F[i] = relax_from(F[i - 1], h[i], F[i])
        for i in range(G0 - 2, 0, -1):
            F[i] = relax_from(F[i + 1], h[i], F[i])
        for j in range(1, G1 - 1):
            F[:, j] = relax_from(F[:, j - 1], h[:, j], F[:, j])
        for j in range(G1 - 2, 0, -1):
            F[:, j] = relax_from(F[:, j + 1], h[:, j], F[:, j])
        if np.array_equal(F, Fprev):
            break
    return F


def _fill_to_lowstand_outlets(h, outlet_mask, max_rounds=8):
    """Depression fill seeded by physical lowstand water everywhere.

    This private localization variant differs from ``fill_depressions``
    only in its seed set.  The public/default solver still drains toward
    the numerical world rim; here every submerged cell is a base-level
    outlet, so a remote ocean-domain boundary cannot redirect a river.
    """
    G0, G1 = h.shape
    outlets = np.asarray(outlet_mask, dtype=bool)
    if outlets.shape != h.shape:
        raise ValueError("lowstand outlet mask must match the surface")
    F = np.full_like(h, np.inf)
    F[0, :] = h[0, :]
    F[-1, :] = h[-1, :]
    F[:, 0] = h[:, 0]
    F[:, -1] = h[:, -1]
    F[outlets] = h[outlets]

    def relax_from(prev, cur_h, cur_F):
        cand = np.minimum(prev,
                          np.minimum(np.r_[np.inf, prev[:-1]],
                                     np.r_[prev[1:], np.inf])) + EPS
        return np.maximum(cur_h, np.minimum(cur_F, cand))

    for _ in range(max_rounds):
        previous = F.copy()
        for i in range(1, G0 - 1):
            F[i] = relax_from(F[i - 1], h[i], F[i])
            F[i, outlets[i]] = h[i, outlets[i]]
        for i in range(G0 - 2, 0, -1):
            F[i] = relax_from(F[i + 1], h[i], F[i])
            F[i, outlets[i]] = h[i, outlets[i]]
        for j in range(1, G1 - 1):
            F[:, j] = relax_from(F[:, j - 1], h[:, j], F[:, j])
            F[outlets[:, j], j] = h[outlets[:, j], j]
        for j in range(G1 - 2, 0, -1):
            F[:, j] = relax_from(F[:, j + 1], h[:, j], F[:, j])
            F[outlets[:, j], j] = h[outlets[:, j], j]
        if np.array_equal(F, previous):
            break
    return F


def receivers(F):
    """D8 steepest receiver + slope-weighted MFD weights (S2)."""
    G = F.shape[0]
    n = G * G
    best = np.zeros((G, G))
    rcv = np.arange(n).reshape(G, G)
    idx = np.arange(n).reshape(G, G)
    targets = np.empty((8, G, G), np.int64)
    weights = np.zeros((8, G, G))
    for k, ((dy, dx), dist) in enumerate(zip(NBR, NBR_D)):
        sh = _shiftf(F, dy, dx, np.inf)
        drop = (F - sh) / dist
        src = np.full_like(rcv, -1)
        src[max(0, dy):G + min(0, dy), max(0, dx):G + min(0, dx)] = \
            idx[max(0, -dy):G + min(0, -dy), max(0, -dx):G + min(0, -dx)]
        targets[k] = src
        weights[k] = np.clip(drop, 0.0, None) ** 1.1
        take = drop > best
        best[take] = drop[take]
        rcv[take] = src[take]
    border = np.zeros((G, G), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    rcv[border] = idx[border]
    weights[:, border] = 0.0
    wsum = weights.sum(axis=0)
    flat = wsum <= 0.0
    weights /= np.where(wsum > 0, wsum, 1.0)[None]
    return rcv.ravel(), targets.reshape(8, n), weights.reshape(8, n), \
        flat.ravel()


def topo_batches(rcv, targets, weights, flat):
    """Vectorized Kahn over the MFD edge set (S2). Batch order runs
    sources -> outlets; reversed order is downstream-first."""
    n = rcv.size
    indeg = np.zeros(n, np.int64)
    for k in range(8):
        e = weights[k] > 0
        np.add.at(indeg, targets[k, e], 1)
    fb = flat & (rcv != np.arange(n))
    np.add.at(indeg, rcv[fb], 1)

    frontier = np.nonzero(indeg == 0)[0]
    batches = []
    seen = 0
    while frontier.size:
        batches.append(frontier)
        seen += frontier.size
        outs = []
        for k in range(8):
            e = weights[k, frontier] > 0
            outs.append(targets[k, frontier[e]])
        ff = frontier[flat[frontier]]
        ff = ff[rcv[ff] != ff]
        outs.append(rcv[ff])
        tgt = np.concatenate(outs) if outs else np.empty(0, np.int64)
        np.subtract.at(indeg, tgt, 1)
        cand = np.unique(tgt)
        frontier = cand[indeg[cand] == 0]
    assert seen == n, f"topo order covered {seen}/{n}"
    return batches


def flow_accumulation_d8(rcv, batches, n, runoff=None):
    """Concentrated D8 accumulation used by channels and lake inflow."""
    A8 = np.ones(n) if runoff is None else runoff.ravel().copy()
    for b in batches:
        r = rcv[b]
        keep = r != b
        np.add.at(A8, r[keep], A8[b[keep]])
    return A8


def flow_accumulation(rcv, batches, n, targets, weights, runoff=None):
    """MFD accumulation (discharge) + concentrated D8 accumulation
    (channels) over the same batches (S2). Seeded with per-cell runoff
    (continentality: moisture decays inland), so accumulations are
    water discharge, not raw area — interior rivers run leaner and
    interior basins receive what an arid interior actually yields."""
    A = np.ones(n) if runoff is None else runoff.ravel().copy()
    for b in batches:
        w = weights[:, b]
        has = w.sum(axis=0) > 0
        bb = b[has]
        if bb.size:
            for k in range(8):
                wk = weights[k, bb]
                nz = wk > 0
                if nz.any():
                    np.add.at(A, targets[k, bb[nz]], A[bb[nz]] * wk[nz])
        fb = b[~has]
        keep = rcv[fb] != fb
        if keep.any():
            np.add.at(A, rcv[fb[keep]], A[fb[keep]])
    A8 = flow_accumulation_d8(rcv, batches, n, runoff)
    return A, A8


def spl_implicit(z, U, Kf, rcv, batches, A_km2, dt_myr, dx_km, base,
                 edge_len_km=None):
    """Implicit stream-power update, downstream-first. Base-level cells
    (below the lowstand sea) and self-receivers do not erode, but every
    cell still receives its configured rock uplift. Returns both the new
    surface and the actual fluvial cut; sediment production must come
    from that cut, never from a post-hoc elevation difference that would
    count soil-creep redistribution a second time."""
    hf = (z + dt_myr * U).ravel().copy()
    numerator = (Kf.ravel() * dt_myr
                 * np.sqrt(np.maximum(A_km2, 1.0)))
    if edge_len_km is None:
        f = numerator / dx_km
    else:
        length = np.asarray(edge_len_km, dtype=np.float64)
        f = numerator / np.where(length > 0.0, length, dx_km)
    basef = base.ravel()
    cut = np.zeros_like(hf)
    for b in reversed(batches):
        r = rcv[b]
        upd = (r != b) & ~basef[b]
        bb, rr = b[upd], r[upd]
        raised = hf[bb]
        new = (raised + f[bb] * hf[rr]) / (1.0 + f[bb])
        # stream power only cuts: routing runs on the FILLED surface,
        # so the raw implicit form would haul closed-basin floors up
        # to their spill level — unphysical (it erased every lake).
        # Erosion may lower, uplift may raise; nothing else.
        hf[bb] = np.minimum(new, raised)
        cut[bb] = raised - hf[bb]
    return hf.reshape(z.shape), cut.reshape(z.shape)


def soil_creep(z, diffusivity_km2_myr, dt_myr, dx_km, base_lvl):
    """Mass-conserving hillslope diffusion over lowstand-exposed land.

    Diffusivity has physical units (km^2/Myr); the explicit coefficient
    is D*dt/dx^2 and is divided into enough stable substeps. Flux crosses
    only edges whose two cells are exposed during that substep, so the
    abyss is not spuriously treated as soil and the exposure boundary is
    no-flux rather than a mass sink.
    """
    total_alpha = (max(float(diffusivity_km2_myr), 0.0)
                   * max(float(dt_myr), 0.0) / float(dx_km) ** 2)
    if total_alpha <= 0.0:
        return z
    n_sub = max(CREEP_MIN_SUBSTEPS,
                int(np.ceil(total_alpha / CREEP_MAX_ALPHA)))
    alpha = total_alpha / n_sub
    for _ in range(n_sub):
        active = z >= base_lvl
        dz = np.zeros_like(z)

        edge = active[:, :-1] & active[:, 1:]
        flux = alpha * (z[:, 1:] - z[:, :-1]) * edge
        dz[:, :-1] += flux
        dz[:, 1:] -= flux

        edge = active[:-1, :] & active[1:, :]
        flux = alpha * (z[1:, :] - z[:-1, :]) * edge
        dz[:-1, :] += flux
        dz[1:, :] -= flux
        z = z + dz
    return z


def route_sediment(z, ero, rcv, batches, A_km2, base_lvl, L_dep_km,
                   dx_km, edge_len_km=None):
    """Single-pass sediment routing, sources -> outlets. Land cells
    deposit what exceeds transport capacity (floodplains/valley fill);
    marine cells settle an e-folding fraction per cell of travel
    (wedges -> fans -> drape), capped so deltas top out at the sea
    surface. Remaining flux at an outer-ring self-receiver crosses the
    open world boundary; an interior self-receiver is reported as a
    terminal residual rather than silently relabelled as export.

    Returns (z_new, deposit, boundary_export_m_cells,
    terminal_residual_m_cells). The scalar fluxes are summed equivalent
    metres over equal-area process cells; run_erosion converts them to
    physical volume.
    """
    n = z.size
    zf = z.ravel()
    Qs = np.maximum(ero, 0.0).ravel().copy()
    dep = np.zeros(n)
    export = 0.0
    terminal_residual = 0.0
    marine = zf < base_lvl
    border = np.zeros(z.shape, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    border = border.ravel()
    if edge_len_km is None:
        link_length = None
        settle = 1.0 - np.exp(-dx_km / L_dep_km)
    else:
        link_length = np.asarray(edge_len_km, dtype=np.float64)
        settle = 1.0 - np.exp(-link_length / L_dep_km)
    cap_coef = KC_LAND * np.sqrt(np.maximum(A_km2, 1.0))
    for b in batches:
        r = rcv[b]
        movable = r != b
        length = dx_km if link_length is None else link_length[b]
        S = np.maximum(zf[b] - zf[r], 0.0) / (length * 1000.0)
        onland = ~marine[b]
        d_land = np.clip(Qs[b] - cap_coef[b] * S * 1000.0,
                         0.0, DEP_CAP) * onland
        room = np.maximum(base_lvl - zf[b], 0.0)
        settle_b = settle if link_length is None else settle[b]
        d_mar = np.minimum(np.minimum(Qs[b] * settle_b, MAR_CAP),
                           room) * marine[b]
        d = np.minimum(d_land + d_mar, Qs[b])
        dep[b] += d
        rem = Qs[b] - d
        terminal = ~movable
        outlet = terminal & border[b]
        export += float(rem[outlet].sum())
        internal = terminal & ~border[b]
        terminal_residual += float(rem[internal].sum())
        np.add.at(Qs, r[movable], rem[movable])
    z_new = z + dep.reshape(z.shape)
    return (z_new, dep.reshape(z.shape), export,
            terminal_residual)


def _marine_transport_graph(z, marine):
    """Local dispersive transport stencil for the marine pass.

    Most flux follows downslope bathymetry, while a smaller isotropic
    share represents lateral fan spreading and bottom-current dispersion.
    Flats and closed basins therefore broaden rather than concentrating
    an entire catchment into one terminal process cell.  The graph is
    intentionally stepped for finite transport time, so cycles are safe.
    """
    G0, G1 = z.shape
    n = z.size
    idx = np.arange(n).reshape(z.shape)
    targets = np.full((8, n), -1, np.int64)
    downhill = np.zeros((8, n), np.float64)
    neighbors = np.zeros((8, n), bool)
    for k, ((dy, dx), dist) in enumerate(zip(NBR, NBR_D)):
        neighbor_z = _shiftf(z, dy, dx, np.inf)
        neighbor_marine = _shiftf(marine, dy, dx, False)
        target = np.full(z.shape, -1, np.int64)
        target[max(0, dy):G0 + min(0, dy),
               max(0, dx):G1 + min(0, dx)] = \
            idx[max(0, -dy):G0 + min(0, -dy),
                max(0, -dx):G1 + min(0, -dx)]
        drop = (z - neighbor_z) / dist
        adjacent = marine & neighbor_marine
        valid = adjacent & (drop > 0.0)
        targets[k] = target.ravel()
        downhill[k] = (np.clip(drop, 0.0, None) ** 1.1
                       * valid).ravel()
        neighbors[k] = adjacent.ravel()
    downhill_total = downhill.sum(axis=0)
    neighbor_count = neighbors.sum(axis=0)
    downhill /= np.where(downhill_total > 0.0,
                         downhill_total, 1.0)[None]
    isotropic = neighbors / np.where(neighbor_count > 0,
                                     neighbor_count, 1.0)[None]
    has_downhill = downhill_total > 0.0
    weights = np.where(
        has_downhill[None],
        0.82 * downhill + 0.18 * isotropic,
        isotropic,
    )
    return targets, weights, neighbor_count > 0


def _bounded_marine_transport(z, mouth_flux, base_lvl, L_dep_km,
                              dx_km):
    """Finite-reach, mass-conserving marine settling from river mouths.

    Flux spreads over marine neighbors rather than riding a single D8
    thread. A fixed fraction settles per process step; local accommodation
    and the finite-window thickness cap limit deposition.
    After six axial settling-length equivalents, remaining suspended load
    is reported as far-field export. Thus influence has a physical,
    control-scaled finite reach and never depends on a crop-edge taper;
    diagnostics report the larger conservative diagonal reach.
    """
    marine = z <= base_lvl
    source = np.maximum(np.asarray(mouth_flux, np.float64), 0.0)
    source = np.where(marine, source, 0.0)
    targets, weights, has_out = _marine_transport_graph(z, marine)
    max_steps = max(
        1, int(np.ceil(LOCAL_MARINE_EFOLDS * float(L_dep_km) / dx_km)))
    settle = 1.0 - np.exp(-dx_km / float(L_dep_km))

    deposit = np.zeros_like(z, dtype=np.float64)
    mobile = source.copy()
    boundary_export = 0.0
    far_field_export = 0.0
    terminal_residual = 0.0
    border = np.zeros(z.shape, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True

    for step in range(max_steps):
        if not np.any(mobile > 0.0):
            break
        # The outer ring is an open marine boundary.  A sufficiently
        # padded local solve never exercises it for flux that can affect
        # the delivered core, but accounting remains explicit.
        at_boundary = border & (mobile > 0.0)
        boundary_export += float(mobile[at_boundary].sum())
        mobile[at_boundary] = 0.0
        if not np.any(mobile > 0.0):
            break

        # Spread first, then settle.  A river mouth therefore feeds a
        # two-dimensional fan before any one process cell receives the
        # full e-fold fraction of a large catchment's load.
        next_mobile = np.zeros(z.size, np.float64)
        mobile_flat = mobile.ravel()
        for k in range(8):
            weight = weights[k]
            moving = weight > 0.0
            if moving.any():
                np.add.at(next_mobile, targets[k, moving],
                          mobile_flat[moving] * weight[moving])
        terminal = ~has_out
        next_mobile[terminal] += mobile_flat[terminal]
        arrived = next_mobile.reshape(z.shape)

        # MAR_CAP is a finite recent-window thickness limit, applied to
        # cumulative deposit rather than independently on every visit.
        # Any settling demand that exceeds accommodation remains mobile
        # and can spread farther; finite transport reach, not this cap,
        # is what makes localization exact.
        room = np.minimum(
            np.maximum(base_lvl - (z + deposit), 0.0),
            np.maximum(MAR_CAP - deposit, 0.0),
        )
        settled = np.minimum(arrived * settle, room)
        deposit += settled
        mobile = arrived - settled

        if step + 1 == max_steps:
            terminal_mask = (~has_out.reshape(z.shape)) & (mobile > 0.0)
            terminal_residual += float(mobile[terminal_mask].sum())
            mobile[terminal_mask] = 0.0
            far_field_export += float(mobile.sum())
            break

    source_total = float(source.sum())
    deposited_total = float(deposit.sum())
    accounted = (deposited_total + boundary_export + far_field_export
                 + terminal_residual)
    diagnostics = {
        "source_m_cells": source_total,
        "deposited_m_cells": deposited_total,
        "boundary_export_m_cells": boundary_export,
        "far_field_export_m_cells": far_field_export,
        "terminal_residual_m_cells": terminal_residual,
        "closure_m_cells": source_total - accounted,
        "max_steps": max_steps,
        "axial_reach_km": max_steps * float(dx_km),
        "max_reach_km": (np.sqrt(2.0) * max_steps * float(dx_km)),
        "max_deposit_m": float(deposit.max(initial=0.0)),
    }
    return (deposit, boundary_export + far_field_export,
            terminal_residual, diagnostics)


def _physical_marine_transport(z, mouth_flux, base_lvl, L_dep_km,
                               dx_km):
    """Finite-runout marine settling with aggradational lobe switching.

    River-mouth load spreads over the submerged neighbor graph, settles by
    the author control's e-folding distance, and aggrades only into physical
    accommodation below lowstand.  The transport graph is rebuilt from the
    aggraded bed every step, so a filling channel or lobe redirects later
    load instead of carrying one capped thread indefinitely.  There is no
    per-cell marine thickness cap.

    The depositional episode lasts ``PHYSICAL_MARINE_EFOLDS`` settling
    lengths.  Remaining suspension is explicit far-field export, while the
    finite duration supplies a crop-independent causal reach bound.  An
    adequate process halo must exceed the reported diagonal reach.
    """
    marine = z <= base_lvl
    source = np.maximum(np.asarray(mouth_flux, np.float64), 0.0)
    source = np.where(marine, source, 0.0)
    max_steps = max(
        1, int(np.ceil(
            PHYSICAL_MARINE_EFOLDS * float(L_dep_km) / dx_km)))
    settle = 1.0 - np.exp(-dx_km / float(L_dep_km))

    deposit = np.zeros_like(z, dtype=np.float64)
    mobile = source.copy()
    boundary_export = 0.0
    far_field_export = 0.0
    terminal_residual = 0.0
    accommodation_limited_events = 0
    border = np.zeros(z.shape, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True

    for step in range(max_steps):
        if not np.any(mobile > 0.0):
            break
        at_boundary = border & (mobile > 0.0)
        boundary_export += float(mobile[at_boundary].sum())
        mobile[at_boundary] = 0.0
        if not np.any(mobile > 0.0):
            break

        # Aggradation changes the next lobe's slope and flat geometry.
        # Rebuilding this local stencil is the modeled avulsion mechanism,
        # not a smoothing or appearance correction.
        bed = z + deposit
        targets, weights, has_out = _marine_transport_graph(bed, marine)
        next_mobile = np.zeros(z.size, np.float64)
        mobile_flat = mobile.ravel()
        for k in range(8):
            weight = weights[k]
            moving = weight > 0.0
            if moving.any():
                np.add.at(next_mobile, targets[k, moving],
                          mobile_flat[moving] * weight[moving])
        terminal = ~has_out
        next_mobile[terminal] += mobile_flat[terminal]
        arrived = next_mobile.reshape(z.shape)

        demand = arrived * settle
        room = np.maximum(base_lvl - bed, 0.0)
        accommodation_limited_events += int(np.count_nonzero(demand > room))
        settled = np.minimum(demand, room)
        deposit += settled
        mobile = arrived - settled

        if step + 1 == max_steps:
            terminal_mask = (~has_out.reshape(z.shape)) & (mobile > 0.0)
            terminal_residual += float(mobile[terminal_mask].sum())
            mobile[terminal_mask] = 0.0
            far_field_export += float(mobile.sum())
            break

    source_total = float(source.sum())
    deposited_total = float(deposit.sum())
    footprint = deposit > 0.0
    values = np.sort(deposit[footprint].ravel())
    if values.size:
        top_count = max(1, int(np.ceil(values.size * 0.01)))
        top_one_percent_fraction = float(
            values[-top_count:].sum() / deposited_total)
        p99_deposit = float(np.percentile(values, 99.0))
    else:
        top_one_percent_fraction = 0.0
        p99_deposit = 0.0
    accounted = (deposited_total + boundary_export + far_field_export
                 + terminal_residual)
    diagnostics = {
        "source_m_cells": source_total,
        "deposited_m_cells": deposited_total,
        "boundary_export_m_cells": boundary_export,
        "far_field_export_m_cells": far_field_export,
        "terminal_residual_m_cells": terminal_residual,
        "closure_m_cells": source_total - accounted,
        "max_steps": max_steps,
        "axial_reach_km": max_steps * float(dx_km),
        "max_reach_km": (np.sqrt(2.0) * max_steps * float(dx_km)),
        "max_deposit_m": float(deposit.max(initial=0.0)),
        "p99_positive_deposit_m": p99_deposit,
        "deposit_footprint_cells": int(np.count_nonzero(footprint)),
        "top_one_percent_footprint_deposit_fraction": (
            top_one_percent_fraction),
        "aggraded_to_lowstand_cells": int(np.count_nonzero(
            footprint & ((z + deposit) >= base_lvl - 1e-9))),
        "accommodation_limited_cell_events": accommodation_limited_events,
        "marine_thickness_cap_applied": False,
        "dynamic_aggradational_routing": True,
    }
    return (deposit, boundary_export + far_field_export,
            terminal_residual, diagnostics)


def _route_sediment_lowstand(z, ero, rcv, batches, A_km2, base_lvl,
                             L_dep_km, dx_km, *,
                             _marine_transport=None):
    """Private two-domain sediment pass for localized processing.

    Terrestrial sediment follows the lowstand drainage graph until its
    first submerged receiver.  The collected mouth flux then enters the
    finite-reach marine fan solver above.  No abyssal receiver can feed
    back into the river graph.
    """
    n = z.size
    zf = z.ravel()
    marine = zf <= base_lvl
    source = np.maximum(np.asarray(ero, np.float64), 0.0).ravel()
    flux = np.where(marine, 0.0, source)
    deposit = np.zeros(n, np.float64)
    mouth_flux = np.zeros(n, np.float64)
    boundary_export = 0.0
    terminal_residual = float(source[marine].sum())
    border = np.zeros(z.shape, bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    border = border.ravel()
    capacity = KC_LAND * np.sqrt(np.maximum(A_km2, 1.0))

    for batch in batches:
        land_batch = batch[~marine[batch]]
        if land_batch.size == 0:
            continue
        receiver = rcv[land_batch]
        movable = receiver != land_batch
        slope = (np.maximum(zf[land_batch] - zf[receiver], 0.0)
                 / (dx_km * 1000.0))
        local_deposit = np.clip(
            flux[land_batch] - capacity[land_batch] * slope * 1000.0,
            0.0, DEP_CAP)
        local_deposit = np.minimum(local_deposit, flux[land_batch])
        deposit[land_batch] += local_deposit
        remaining = flux[land_batch] - local_deposit

        to_marine = movable & marine[receiver]
        if to_marine.any():
            np.add.at(mouth_flux, receiver[to_marine],
                      remaining[to_marine])

        to_land = movable & ~marine[receiver]
        if to_land.any():
            np.add.at(flux, receiver[to_land], remaining[to_land])

        terminal = ~movable
        if terminal.any():
            terminal_cells = land_batch[terminal]
            terminal_flux = remaining[terminal]
            outer = border[terminal_cells]
            boundary_export += float(terminal_flux[outer].sum())
            terminal_residual += float(terminal_flux[~outer].sum())

    land_deposit_total = float(deposit.sum())
    marine_transport = (_bounded_marine_transport
                        if _marine_transport is None
                        else _marine_transport)
    marine_deposit, marine_export, marine_residual, marine_diag = \
        marine_transport(
            z + deposit.reshape(z.shape),
            mouth_flux.reshape(z.shape), base_lvl, L_dep_km, dx_km)
    deposit += marine_deposit.ravel()
    boundary_export += marine_export
    terminal_residual += marine_residual

    source_total = float(source.sum())
    deposited_total = float(deposit.sum())
    closure = source_total - (
        deposited_total + boundary_export + terminal_residual)
    diagnostics = {
        "source_m_cells": source_total,
        "land_deposited_m_cells": land_deposit_total,
        "mouth_flux_m_cells": float(mouth_flux.sum()),
        "marine": marine_diag,
        "boundary_and_far_field_export_m_cells": boundary_export,
        "terminal_residual_m_cells": terminal_residual,
        "closure_m_cells": closure,
    }
    dep = deposit.reshape(z.shape)
    return z + dep, dep, boundary_export, terminal_residual, diagnostics


def _dilate1(m):
    g = m.copy()
    g[:, 1:] |= m[:, :-1]
    g[:, :-1] |= m[:, 1:]
    g[1:, :] |= m[:-1, :]
    g[:-1, :] |= m[1:, :]
    return g


def _balance_lakes(z, F, A8g):
    """Per-basin water balance. Each connected depression above present
    sea level holds water up to the level where lake area ~= WATER_YIELD
    x catchment area (max routed accumulation through the basin), capped
    at the spill. Basins that would overflow stay brim-full (exorheic,
    outlet river intact); big dry-belt basins keep floor lakes only."""
    depth_full = F - z
    cand = (depth_full > 0.5) & (F > 0.0)
    lake_depth = np.zeros_like(z)
    lake_surf = np.zeros_like(z)
    if not cand.any():
        return lake_depth, lake_surf
    G0, G1 = z.shape
    seen = np.zeros_like(cand)
    ys, xs = np.nonzero(cand)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        stack = [(int(y0), int(x0))]
        seen[y0, x0] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < G0 and 0 <= nx < G1 \
                            and cand[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        cy = np.array([c[0] for c in cells])
        cx = np.array([c[1] for c in cells])
        catch = float(A8g[cy, cx].max())
        if catch < LAKE_FEED_CELLS:
            continue
        zb = z[cy, cx]
        n_target = int(WATER_YIELD * catch)
        if n_target < 1:
            continue
        # The epsilon fill is intentionally infinitesimally sloped for
        # routing, but a water surface is one horizontal plane. Use the
        # component's lowest escape level as its scalar spill datum.
        spill = float(F[cy, cx].min())
        if n_target >= zb.size:
            level = spill                    # brim-full to spill
        else:
            level = min(float(np.sort(zb)[n_target]), spill)
        d = level - zb
        wet = d > LAKE_MIN_DEPTH
        lake_depth[cy[wet], cx[wet]] = d[wet]
        lake_surf[cy[wet], cx[wet]] = level
    return lake_depth, lake_surf


def run_erosion(s, ce, cfg, seed, *, _routing_mode="legacy",
                _process_window=None, _localization_mode="legacy"):
    """Structure + coarse elevation -> process-grid dict.

    ``_routing_mode`` is an internal ablation seam, deliberately absent
    from the registry, adapter, report, and public controls.  ``legacy``
    is the shipped path and remains the default.

    ``_process_window`` is an experimental seam for the large-world /
    local-process spike.  It is ``(row0, col0, side)`` in cells of the
    full fixed process lattice.  Absolute world-km sampling and RNG
    coordinates are preserved; only the solve domain is cropped.  No
    public caller supplies it.

    Private localization modes terminate terrestrial drainage at the
    physical lowstand ocean and hand river-mouth load to a separate
    finite-reach marine transport pass. ``lowstand_outlets`` preserves the
    first capped experiment; ``physical_outlets`` is its uncapped,
    aggradational-routing successor. Neither is available with routing
    ablations.
    """
    valid_routing_modes = {
        "legacy", "legacy_lengths", "d8_flat", "ltd_mfd", "ltd_dinf"
    }
    if _routing_mode not in valid_routing_modes:
        raise ValueError(f"unknown internal routing mode: {_routing_mode}")
    valid_localization_modes = {
        "legacy", "lowstand_outlets", "physical_outlets"
    }
    if _localization_mode not in valid_localization_modes:
        raise ValueError(
            f"unknown internal localization mode: {_localization_mode}")
    if (_localization_mode != "legacy" and _routing_mode != "legacy"):
        raise ValueError(
            "physical-outlet localization currently requires legacy routing")
    t_all = time.perf_counter()
    n_world = int(round(s.world_km / E_KM))
    e_km = s.world_km / n_world
    if _process_window is None:
        iy0 = ix0 = 0
        n_e = n_world
    else:
        if len(_process_window) != 3:
            raise ValueError("_process_window must be (row0, col0, side)")
        iy0, ix0, n_e = (int(v) for v in _process_window)
        if (n_e < 2 or iy0 < 0 or ix0 < 0
                or iy0 + n_e > n_world or ix0 + n_e > n_world):
            raise ValueError("_process_window lies outside process world")
    qx = (ix0 + np.arange(n_e) + 0.5) * e_km
    qy = (iy0 + np.arange(n_e) + 0.5) * e_km
    x_km = qx[None, :]
    y_km = qy[:, None]
    ck = s.world_km / s.n

    hc = _bicubic(ce["h"], y_km, x_km, ck)
    oro = _bicubic(ce["oro"], y_km, x_km, ck)
    up = _bicubic(ce["uplift"], y_km, x_km, ck)

    # mid-band relief texture (octaves 0..3 of the surface stack) rides
    # THROUGH the solve — the fine band is added at output resolution
    land_amp = 80.0 + 0.12 * np.maximum(hc, 0.0) + 0.05 * oro
    ocean_amp = 18.0 + 27.0 * _smooth01((hc + 2500.0) / 2250.0)
    amp = np.where(hc >= 0.0, land_amp, ocean_amp)
    det = noise.fbm(x_km, y_km, 360.0, 4,
                    stage_salt(seed, "surface-detail"),
                    gain=0.5, norm_octaves=9)
    z0 = hc + cfg.detail_amplitude * amp * det

    # process-modulated erodibility: lithology (belt rock is harder)
    # x km-space heterogeneity — a named S2 counter to D8 axis-locking
    khet = np.exp(0.8 * noise.fbm(x_km, y_km, 190.0, 3,
                                  stage_salt(seed, "erodibility"),
                                  gain=0.55))
    K = (K_BASE * cfg.erodibility * np.clip(khet, 0.45, 2.2)
         * np.where(oro > 400.0, 0.55, 1.0))

    base_lvl = -float(cfg.lowstand_drop)
    erosion_time = max(float(cfg.erosion_time), 0.0)
    dt = erosion_time / N_STEPS if N_STEPS else 0.0
    creep_diffusivity = (CREEP_DIFFUSIVITY_KM2_MYR
                         * float(cfg.soil_creep))

    timings = {name: 0.0 for name in
               ("fill", "route", "accum", "solve", "sediment",
                "lakes")}

    def route_graph(surface, runoff, need_mfd=True):
        """Route one immutable surface and account its phase timings."""
        if _localization_mode in ("lowstand_outlets", "physical_outlets"):
            t0 = time.perf_counter()
            marine = surface <= base_lvl
            routing_surface = np.where(marine, base_lvl, surface)
            filled = _fill_to_lowstand_outlets(
                routing_surface, marine)
            t1 = time.perf_counter()
            rcv0, targets, weights, flat = receivers(filled)
            marine_flat = marine.ravel()
            index = np.arange(surface.size)
            rcv0[marine_flat] = index[marine_flat]
            weights[:, marine_flat] = 0.0
            flat[marine_flat] = True
            batches0 = topo_batches(rcv0, targets, weights, flat)
            t2 = time.perf_counter()
            routed_runoff = np.where(marine, 0.0, runoff)
            if need_mfd:
                area, area8 = flow_accumulation(
                    rcv0, batches0, surface.size, targets, weights,
                    routed_runoff)
            else:
                area = None
                area8 = flow_accumulation_d8(
                    rcv0, batches0, surface.size, routed_runoff)
            # Marine sink cells receive river-mouth flux while the graph
            # is accumulated, but submerged "river discharge" is not a
            # terrestrial observable and must not leak into map layers.
            area8[marine_flat] = 0.0
            if area is not None:
                area[marine_flat] = 0.0
            t3 = time.perf_counter()
            timings["fill"] += t1 - t0
            timings["route"] += t2 - t1
            timings["accum"] += t3 - t2
            return (filled, rcv0, batches0, area, area8, None, None)

        if _routing_mode in ("d8_flat", "ltd_mfd", "ltd_dinf"):
            from .routing_experiment import (
                accumulate_channel, accumulate_weighted, freeman_graph,
                routing_graph as experimental_routing_graph)
            t0 = time.perf_counter()
            channel_mode = ("dinf_d8" if _routing_mode == "d8_flat"
                            else "dinf_ltd")
            graph = experimental_routing_graph(
                surface, e_km, mode=channel_mode)
            if need_mfd and _routing_mode in ("d8_flat", "ltd_mfd"):
                diffuse_targets, diffuse_weights, diffuse_batches = \
                    freeman_graph(graph.filled_level, graph)
            else:
                diffuse_targets = graph.targets
                diffuse_weights = graph.weights
                diffuse_batches = graph.batches
            t1 = time.perf_counter()
            if need_mfd:
                area = accumulate_weighted(
                    diffuse_targets, diffuse_weights, diffuse_batches,
                    runoff)
                batches0 = diffuse_batches
            else:
                area = None
                batches0 = graph.batches
            area8 = accumulate_channel(graph.rcv, batches0, runoff)
            t2 = time.perf_counter()
            # The experimental constructor performs exact filling and
            # route construction together.  Keep total accounting exact;
            # its standalone probe reports the finer breakdown.
            timings["route"] += t1 - t0
            timings["accum"] += t2 - t1
            return (graph.filled_level, graph.rcv, batches0, area,
                    area8, graph.edge_len, graph.transport_len)

        t0 = time.perf_counter()
        filled = fill_depressions(surface)
        t1 = time.perf_counter()
        rcv0, targets, weights, flat = receivers(filled)
        batches0 = topo_batches(rcv0, targets, weights, flat)
        t2 = time.perf_counter()
        if need_mfd:
            area, area8 = flow_accumulation(
                rcv0, batches0, surface.size, targets, weights, runoff)
        else:
            area = None
            area8 = flow_accumulation_d8(
                rcv0, batches0, surface.size, runoff)
        t3 = time.perf_counter()
        timings["fill"] += t1 - t0
        timings["route"] += t2 - t1
        timings["accum"] += t3 - t2
        edge_len = transport_len = None
        if _routing_mode == "legacy_lengths":
            from .routing_experiment import (receiver_edge_lengths,
                                              receiver_transport_lengths)
            edge_len = receiver_edge_lengths(rcv0, surface.shape, e_km)
            transport_len = receiver_transport_lengths(
                rcv0, surface.shape, e_km)
        return (filled, rcv0, batches0, area, area8, edge_len,
                transport_len)

    z = z0.copy()
    ero = np.zeros_like(z)
    dep = np.zeros_like(z)
    export_m_cells = 0.0
    terminal_residual_m_cells = 0.0
    localization_diagnostics = None
    runoff = None
    if erosion_time > 0.0:
        # Continentality remains fixed from the initial present-day
        # coastline; changing it would be climate work, not rerouting.
        from .elevation import _chamfer_km
        d_sea = _chamfer_km(z0 < 0.0, e_km)
        runoff = np.exp(-d_sea / L_MOIST_KM)

    steps = N_STEPS if erosion_time > 0.0 else 0
    for _ in range(steps):
        # This graph drives only this immutable-surface solve. The
        # changed terrain is routed again before its next consumer.
        _, rcv, batches, A, _, edge_len, _ = route_graph(z, runoff)
        A_km2 = A * e_km * e_km
        t0 = time.perf_counter()
        base = z < base_lvl
        z, cut = spl_implicit(z, up, K, rcv, batches, A_km2, dt,
                              e_km, base, edge_len)
        ero += cut
        z = soil_creep(z, creep_diffusivity, dt, e_km, base_lvl)
        timings["solve"] += time.perf_counter() - t0

    if erosion_time > 0.0:
        # The final solve and creep changed the route they used. Route
        # sediment on the post-process terrain, never the stale graph.
        _, rcv, batches, A, _, _, transport_len = route_graph(z, runoff)
        A_km2 = A * e_km * e_km

        t0 = time.perf_counter()
        if _localization_mode == "lowstand_outlets":
            (z, dep, export_m_cells, terminal_residual_m_cells,
             localization_diagnostics) = _route_sediment_lowstand(
                z, ero, rcv, batches, A_km2, base_lvl,
                float(cfg.deposition_length), e_km)
        elif _localization_mode == "physical_outlets":
            (z, dep, export_m_cells, terminal_residual_m_cells,
             localization_diagnostics) = _route_sediment_lowstand(
                z, ero, rcv, batches, A_km2, base_lvl,
                float(cfg.deposition_length), e_km,
                _marine_transport=_physical_marine_transport)
        else:
            z, dep, export_m_cells, terminal_residual_m_cells = \
                route_sediment(z, ero, rcv, batches, A_km2, base_lvl,
                               float(cfg.deposition_length), e_km,
                               transport_len)
        timings["sediment"] = time.perf_counter() - t0

        # Deposition changes receivers once more. Delivered discharge,
        # rivers, and lake inflow all use this final-terrain graph. MFD
        # accumulation is unnecessary for those consumers.
        F_final, rcv, batches, _, A8, _, _ = route_graph(
            z, runoff, need_mfd=False)
        A8g = A8.reshape(z.shape)

        t0 = time.perf_counter()
        lake_depth, lake_surf = _balance_lakes(z, F_final, A8g)
        timings["lakes"] = time.perf_counter() - t0
    else:
        # The control promise is literal: no erosion-time window means
        # no carving, creep, sediment, routed rivers, or process lakes.
        rcv = np.arange(z.size)
        A8 = np.zeros(z.size)
        A8g = A8.reshape(z.shape)
        lake_depth = np.zeros_like(z)
        lake_surf = np.zeros_like(z)

    cell_area_m2 = (e_km * 1000.0) ** 2
    sediment_export_m3 = export_m_cells * cell_area_m2
    sediment_terminal_residual_m3 = (terminal_residual_m_cells
                                      * cell_area_m2)
    timings["erosion_total"] = time.perf_counter() - t_all

    # river network for the render: per drawn cell, its centre, its
    # receiver's centre, and its MAIN DONOR's centre (largest drawn
    # discharge flowing in). The channel's sub-cell position is unknown
    # at process-grid scale, so the render interpolates a smooth curve
    # through donor-midpoint -> cell -> receiver-midpoint — the same
    # numerics-of-sampling class as elevation prolongation; straight
    # cell-to-cell segments drew right-angle "drafting board" rivers
    # (the dominant M3 judge tell)
    land_now = z >= 0.0
    drawn = (A8 > 30.0) & land_now.ravel() & (rcv != np.arange(z.size))
    sel = np.flatnonzero(drawn)
    r_sel = rcv[sel]
    order = np.lexsort((-A8[sel], r_sel))
    r_sorted = r_sel[order]
    first = np.ones(order.size, bool)
    first[1:] = r_sorted[1:] != r_sorted[:-1]
    main_donor = {int(r_sorted[oi]): int(sel[order[oi]])
                  for oi in np.flatnonzero(first)}
    donor = np.array([main_donor.get(int(c), int(c)) for c in sel],
                     np.int64) if sel.size else np.empty(0, np.int64)

    def _km(idx):
        yi, xi = np.divmod(idx, n_e)
        return ((ix0 + xi + 0.5) * e_km,
                (iy0 + yi + 0.5) * e_km)

    x0, y0 = _km(sel)
    x1, y1 = _km(rcv[sel])
    xd, yd = _km(donor)
    river_edges = {
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "xd": xd, "yd": yd,
        "a8": A8[sel],
    }

    result = {
        "n_e": n_e, "e_km": e_km,
        "z": z, "z0": z0,
        "discharge_log": np.log1p(A8g),
        "sed": dep, "ero": ero,
        "sediment_export_m3": float(sediment_export_m3),
        "sediment_terminal_residual_m3":
            float(sediment_terminal_residual_m3),
        "lake_depth": lake_depth, "lake_surf": lake_surf,
        "river_edges": river_edges,
        "timings": timings,
    }
    if _process_window is not None:
        result["process_origin_km"] = (iy0 * e_km, ix0 * e_km)
        result["process_window"] = (iy0, ix0, n_e)
    if localization_diagnostics is not None:
        result["_localization_diagnostics"] = localization_diagnostics
    return result
