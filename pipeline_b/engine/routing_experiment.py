"""Internal routing spike for the post-M3 D8-bias investigation.

This module is intentionally not a public control surface.  It supplies
the same-surface ablation with four process pieces that must be measured
separately: an exact depression fill, a nonphysical rank for routing over
the resulting flats, D-infinity diffuse targets, and a nondispersive
D8-LTD (lambda=1) channel spine.

The D8-LTD implementation follows Orlandini et al. (2003): cells are
processed source to outlet over the potential D-infinity DAG.  At a
confluence the cumulative transverse error is inherited from the donor
with the largest geometric drainage area, then the cardinal or diagonal
edge bracketing the theoretical direction is selected to minimize the
absolute cumulative error.
"""

from dataclasses import dataclass

import numpy as np


OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
           (0, 1), (1, -1), (1, 0), (1, 1))
OFFSET_DISTANCE = np.array([np.hypot(dy, dx)
                            for dy, dx in OFFSETS])

# (cardinal offset, diagonal offset, signed-deviation orientation).
# The signs are Table 1's sigma values in Orlandini et al. (2003).
FACETS = (
    ((-1, 0), (-1, -1), +1),
    ((-1, 0), (-1, 1), -1),
    ((0, 1), (-1, 1), +1),
    ((0, 1), (1, 1), -1),
    ((1, 0), (1, 1), +1),
    ((1, 0), (1, -1), -1),
    ((0, -1), (1, -1), +1),
    ((0, -1), (-1, -1), -1),
)
FACET_ANGLE = np.pi / 4.0


@dataclass
class RoutingGraph:
    """Complete internal representation used by the routing spike."""

    filled_level: np.ndarray
    flat_mask: np.ndarray
    flat_rank: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    flow_angle: np.ndarray
    rcv: np.ndarray
    edge_len: np.ndarray
    transport_len: np.ndarray
    batches: list
    cum_transverse_error: np.ndarray
    channel_area_cells: np.ndarray
    main_donor: np.ndarray


def _neighbor(a, dy, dx, fill):
    """Value at (row + dy, col + dx) for every source cell."""
    g0, g1 = a.shape
    out = np.full(a.shape, fill, dtype=a.dtype)
    y0, y1 = max(0, -dy), min(g0, g0 - dy)
    x0, x1 = max(0, -dx), min(g1, g1 - dx)
    if y0 < y1 and x0 < x1:
        out[y0:y1, x0:x1] = a[y0 + dy:y1 + dy,
                               x0 + dx:x1 + dx]
    return out


def _neighbor_targets(shape, dy, dx):
    g0, g1 = shape
    idx = np.arange(g0 * g1, dtype=np.int64).reshape(shape)
    return _neighbor(idx, dy, dx, -1)


def fill_level(h, max_rounds=None):
    """Return an exact, level depression fill.

    This is the leak-free directional reconstruction used by the
    production solver with the epsilon plumbing increment removed.  A
    separate integer rank, never this elevation field, establishes
    drainage across the flats created by the fill.
    """
    h = np.asarray(h, dtype=np.float64)
    g0, g1 = h.shape
    filled = np.full_like(h, np.inf)
    filled[0, :] = h[0, :]
    filled[-1, :] = h[-1, :]
    filled[:, 0] = h[:, 0]
    filled[:, -1] = h[:, -1]

    def relax(prev, cur_h, cur_f):
        candidate = prev.copy()
        np.minimum(candidate[1:], prev[:-1], out=candidate[1:])
        np.minimum(candidate[:-1], prev[1:], out=candidate[:-1])
        return np.maximum(cur_h, np.minimum(cur_f, candidate))

    limit = max(g0, g1) if max_rounds is None else int(max_rounds)
    for _ in range(max(limit, 1)):
        before = filled.copy()
        for i in range(1, g0 - 1):
            filled[i] = relax(filled[i - 1], h[i], filled[i])
        for i in range(g0 - 2, 0, -1):
            filled[i] = relax(filled[i + 1], h[i], filled[i])
        for j in range(1, g1 - 1):
            filled[:, j] = relax(filled[:, j - 1], h[:, j],
                                  filled[:, j])
        for j in range(g1 - 2, 0, -1):
            filled[:, j] = relax(filled[:, j + 1], h[:, j],
                                  filled[:, j])
        if np.array_equal(filled, before):
            return filled
    raise RuntimeError("exact depression fill did not converge")


def _flat_neighbors(filled):
    """Neighbor table and local comparisons for exact-level routing."""
    lower = np.zeros_like(filled, dtype=bool)
    equal = np.zeros_like(filled, dtype=bool)
    higher = np.zeros_like(filled, dtype=bool)
    neighbors = []
    for dy, dx in OFFSETS:
        nb = _neighbor(filled, dy, dx, np.inf)
        target = _neighbor_targets(filled.shape, dy, dx)
        neighbors.append(target.ravel())
        valid = target >= 0
        lower |= valid & (nb < filled)
        equal |= valid & (nb == filled)
        higher |= valid & (nb > filled)

    border = np.zeros_like(filled, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    return lower, equal, higher, border, np.stack(neighbors)


def resolve_flats(filled):
    """Build a Barnes-style convergent rank without altering elevation.

    The rank superposes a gradient away from higher terrain with a
    twice-weighted gradient toward lower terrain.  Only ordering matters;
    these integer values are never interpreted as metres or slope.
    """
    lower, equal, higher, border, neighbors = _flat_neighbors(filled)
    n = filled.size
    level = filled.ravel()
    flat_mask = np.zeros(n, dtype=bool)
    dist_low = np.full(n, -1, dtype=np.int64)
    dist_high = np.full(n, -1, dtype=np.int64)
    lower_f = lower.ravel()
    higher_f = higher.ravel()
    border_f = border.ravel()

    def expand(starts, seen, distances=None):
        """Synchronous BFS constrained to equal filled elevations."""
        frontier = np.unique(np.asarray(starts, dtype=np.int64))
        seen[frontier] = True
        if distances is not None:
            distances[frontier] = 0
        distance = 0
        while frontier.size:
            candidate = neighbors[:, frontier].ravel()
            source = np.tile(frontier, neighbors.shape[0])
            valid = candidate >= 0
            candidate = candidate[valid]
            source = source[valid]
            valid = ((level[candidate] == level[source])
                     & ~seen[candidate])
            if not valid.any():
                break
            frontier = np.unique(candidate[valid])
            distance += 1
            seen[frontier] = True
            if distances is not None:
                distances[frontier] = distance

    # Identify whole equal-level components containing at least one cell
    # that has no local descent.  Their outlet and high-edge cells are
    # included even though those boundary members may themselves have a
    # physical downhill edge.
    seeds = np.flatnonzero((equal & ~lower & ~border).ravel())
    if seeds.size:
        expand(seeds, flat_mask)
    low_starts = np.flatnonzero(flat_mask & (lower_f | border_f))
    high_starts = np.flatnonzero(flat_mask & higher_f)
    if flat_mask.any() and low_starts.size == 0:
        raise RuntimeError("filled flat has no outlet")

    if low_starts.size:
        seen_low = ~flat_mask.copy()
        expand(low_starts, seen_low, dist_low)
    if high_starts.size:
        seen_high = ~flat_mask.copy()
        expand(high_starts, seen_high, dist_high)
    if (dist_low[flat_mask] < 0).any():
        raise RuntimeError("flat outlet rank did not cover component")

    # Barnes' per-component constant max(distance-from-high) can be
    # omitted: it shifts every member equally and cannot affect routing.
    # The equivalent ordering is 2*toward-low - away-from-high.
    rank = np.zeros(n, dtype=np.int64)
    away = np.where(dist_high[flat_mask] >= 0,
                    dist_high[flat_mask], 0)
    rank[flat_mask] = 2 * dist_low[flat_mask] - away
    return flat_mask.reshape(filled.shape), rank.reshape(filled.shape)


def _dinf_local(field, allowed, dx_km):
    """D-infinity facets over a scalar field and allowed neighbor edges."""
    shape = field.shape
    n = field.size
    center = field
    target_by_offset = {
        off: _neighbor_targets(shape, *off) for off in OFFSETS
    }
    value_by_offset = {
        off: _neighbor(field, *off, np.inf) for off in OFFSETS
    }

    best_slope = np.full(shape, -np.inf)
    target0 = np.full(shape, -1, dtype=np.int64)
    target1 = np.full(shape, -1, dtype=np.int64)
    weight0 = np.zeros(shape)
    weight1 = np.zeros(shape)
    angle = np.full(shape, np.nan)
    sigma = np.zeros(shape, dtype=np.int8)
    delta0 = np.zeros(shape)
    delta1 = np.zeros(shape)

    # Boundary directions are also candidates.  Facet interiors below
    # supersede them only when their true planar slope is steeper.
    for k, (dy, dx) in enumerate(OFFSETS):
        nb = value_by_offset[(dy, dx)]
        ok = allowed[k]
        slope = np.where(ok, (center - nb) /
                         (OFFSET_DISTANCE[k] * dx_km), -np.inf)
        take = slope > best_slope
        if take.any():
            best_slope[take] = slope[take]
            tgt = target_by_offset[(dy, dx)]
            target0[take] = tgt[take]
            target1[take] = tgt[take]
            weight0[take] = 1.0
            weight1[take] = 0.0
            angle[take] = np.arctan2(dy, dx)
            sigma[take] = 0
            delta0[take] = 0.0
            delta1[take] = 0.0

    for card, diag, facet_sigma in FACETS:
        k_card = OFFSETS.index(card)
        k_diag = OFFSETS.index(diag)
        valid = allowed[k_card] & allowed[k_diag]
        e1 = value_by_offset[card]
        e2 = value_by_offset[diag]
        with np.errstate(invalid="ignore"):
            s1 = (center - e1) / dx_km
            s2 = (e1 - e2) / dx_km
        r = np.arctan2(s2, s1)
        inside = valid & (r > 0.0) & (r < FACET_ANGLE)
        slope = np.where(inside, np.hypot(s1, s2), -np.inf)
        take = slope > best_slope
        if not take.any():
            continue
        frac = r / FACET_ANGLE
        best_slope[take] = slope[take]
        card_target = target_by_offset[card]
        diag_target = target_by_offset[diag]
        target0[take] = card_target[take]
        target1[take] = diag_target[take]
        weight0[take] = 1.0 - frac[take]
        weight1[take] = frac[take]
        theta0 = np.arctan2(card[0], card[1])
        theta1 = np.arctan2(diag[0], diag[1])
        turn = np.arctan2(np.sin(theta1 - theta0),
                          np.cos(theta1 - theta0))
        angle[take] = theta0 + np.sign(turn) * r[take]
        sigma[take] = facet_sigma
        delta0[take] = dx_km * np.sin(r[take])
        delta1[take] = (np.sqrt(2.0) * dx_km
                        * np.sin(FACET_ANGLE - r[take]))

    dead = ~(best_slope > 0.0)
    target0[dead] = np.arange(n, dtype=np.int64).reshape(shape)[dead]
    target1[dead] = target0[dead]
    weight0[dead] = 0.0
    weight1[dead] = 0.0
    return {
        "targets": np.stack((target0.ravel(), target1.ravel())),
        "weights": np.stack((weight0.ravel(), weight1.ravel())),
        "flow_angle": angle.ravel(),
        "sigma": sigma.ravel(),
        "delta0": delta0.ravel(),
        "delta1": delta1.ravel(),
    }


def _flow_directions(filled, flat_mask, flat_rank, dx_km):
    """D-infinity directions on slopes, falling back to flat rank."""
    physical_allowed = []
    for dy, dx in OFFSETS:
        nb = _neighbor(filled, dy, dx, np.inf)
        physical_allowed.append(nb < filled)
    result = _dinf_local(filled, physical_allowed, dx_km)

    dead = result["weights"].sum(axis=0) <= 0.0
    use_flat = dead & flat_mask.ravel()
    if use_flat.any():
        flat_allowed = []
        for dy, dx in OFFSETS:
            level_nb = _neighbor(filled, dy, dx, np.inf)
            rank_nb = _neighbor(flat_rank, dy, dx,
                                np.iinfo(np.int64).max)
            flat_allowed.append((level_nb == filled)
                                & (rank_nb < flat_rank))
        ranked = _dinf_local(flat_rank.astype(np.float64),
                             flat_allowed, dx_km)
        for name in result:
            if result[name].ndim == 2:
                result[name][:, use_flat] = ranked[name][:, use_flat]
            else:
                result[name][use_flat] = ranked[name][use_flat]

    # The outer ring is an open-domain terminal, matching production.
    g0, g1 = filled.shape
    border = np.zeros_like(filled, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    bf = border.ravel()
    idx = np.arange(filled.size, dtype=np.int64)
    result["targets"][:, bf] = idx[bf]
    result["weights"][:, bf] = 0.0
    return result


def topo_batches_weighted(targets, weights, n=None):
    """Kahn batches for any fixed-width weighted downstream graph."""
    targets = np.asarray(targets, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if n is None:
        n = targets.shape[1]
    indegree = np.zeros(n, dtype=np.int64)
    for k in range(targets.shape[0]):
        active = (weights[k] > 0.0) & (targets[k] >= 0) \
            & (targets[k] != np.arange(n))
        np.add.at(indegree, targets[k, active], 1)

    frontier = np.flatnonzero(indegree == 0)
    batches = []
    seen = 0
    while frontier.size:
        batches.append(frontier)
        seen += frontier.size
        outgoing = []
        for k in range(targets.shape[0]):
            active = ((weights[k, frontier] > 0.0)
                      & (targets[k, frontier] != frontier))
            if active.any():
                outgoing.append(targets[k, frontier[active]])
        if not outgoing:
            frontier = np.empty(0, dtype=np.int64)
            continue
        dst = np.concatenate(outgoing)
        np.subtract.at(indegree, dst, 1)
        candidate = np.unique(dst)
        frontier = candidate[indegree[candidate] == 0]
    if seen != n:
        raise RuntimeError(f"routing graph is cyclic: covered {seen}/{n}")
    return batches


def _d8_ltd(local, batches, shape, dx_km):
    """Select the lambda=1 D8-LTD channel spine over a potential DAG."""
    n = shape[0] * shape[1]
    targets = local["targets"]
    weights = local["weights"]
    sigma = local["sigma"].astype(np.float64)
    delta0 = local["delta0"]
    delta1 = local["delta1"]
    idx = np.arange(n, dtype=np.int64)
    rcv = idx.copy()
    path_error = np.zeros(n)
    inherited_error = np.zeros(n)
    area = np.ones(n)
    best_donor_area = np.full(n, -np.inf)
    best_donor_error = np.full(n, np.inf)
    best_donor = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    main_donor = idx.copy()

    for batch in batches:
        err0 = inherited_error[batch] + sigma[batch] * delta0[batch]
        err1 = inherited_error[batch] - sigma[batch] * delta1[batch]
        valid0 = (weights[0, batch] > 0.0) \
            & (targets[0, batch] != batch)
        valid1 = (weights[1, batch] > 0.0) \
            & (targets[1, batch] != batch)
        choose0 = valid0 & (~valid1 | (np.abs(err0) <= np.abs(err1)))
        choose1 = valid1 & ~choose0
        chosen = np.where(choose0, targets[0, batch],
                          np.where(choose1, targets[1, batch], batch))
        chosen_error = np.where(choose0, err0,
                                np.where(choose1, err1, 0.0))
        rcv[batch] = chosen
        path_error[batch] = chosen_error

        moving = chosen != batch
        if not moving.any():
            continue
        src = batch[moving]
        dst = chosen[moving]
        np.add.at(area, dst, area[src])

        # At a confluence, inherit the deviation from the path carrying
        # the largest geometric drainage area.  A lower source index is
        # the deterministic tie break.
        order = np.lexsort((src, np.abs(path_error[src]),
                            -area[src], dst))
        dst_sorted = dst[order]
        first = np.ones(order.size, dtype=bool)
        first[1:] = dst_sorted[1:] != dst_sorted[:-1]
        winner = src[order[first]]
        receiver = dst_sorted[first]
        winner_error = np.abs(path_error[winner])
        better = ((area[winner] > best_donor_area[receiver])
                  | ((area[winner] == best_donor_area[receiver])
                     & ((winner_error < best_donor_error[receiver])
                        | ((winner_error == best_donor_error[receiver])
                           & (winner < best_donor[receiver])))))
        if better.any():
            rr = receiver[better]
            ww = winner[better]
            best_donor_area[rr] = area[ww]
            best_donor_error[rr] = np.abs(path_error[ww])
            best_donor[rr] = ww
            main_donor[rr] = ww
            inherited_error[rr] = path_error[ww]

    y0, x0 = np.divmod(idx, shape[1])
    y1, x1 = np.divmod(rcv, shape[1])
    edge_len = dx_km * np.hypot(y1 - y0, x1 - x0)
    return rcv, edge_len, path_error, area, main_donor


def receiver_edge_lengths(rcv, shape, dx_km):
    """Physical center-to-center length of each selected raster edge."""
    idx = np.arange(rcv.size, dtype=np.int64)
    y0, x0 = np.divmod(idx, shape[1])
    y1, x1 = np.divmod(rcv, shape[1])
    length = float(dx_km) * np.hypot(y1 - y0, x1 - x0)
    return length


def receiver_transport_lengths(rcv, shape, dx_km):
    """Link length with the legacy terminal sediment residence step."""
    length = receiver_edge_lengths(rcv, shape, dx_km)
    length[rcv == np.arange(rcv.size)] = float(dx_km)
    return length


def _d8_lad(local, batches, shape, dx_km):
    """Local D8/LAD spine on the exact-level plus flat-rank potential."""
    n = shape[0] * shape[1]
    idx = np.arange(n, dtype=np.int64)
    valid0 = (local["weights"][0] > 0.0) \
        & (local["targets"][0] != idx)
    valid1 = (local["weights"][1] > 0.0) \
        & (local["targets"][1] != idx)
    choose0 = valid0 & (~valid1
                        | (local["weights"][0]
                           >= local["weights"][1]))
    choose1 = valid1 & ~choose0
    rcv = np.where(choose0, local["targets"][0],
                   np.where(choose1, local["targets"][1], idx))
    area = accumulate_channel(rcv, batches)
    return (rcv, receiver_edge_lengths(rcv, shape, dx_km),
            np.zeros(n), area, idx.copy())


def routing_graph(surface, dx_km=1.0, mode="dinf_ltd"):
    """Construct the internal exact-fill/D-infinity/D8-LTD graph."""
    if mode not in ("dinf_ltd", "dinf_d8"):
        raise ValueError(f"unknown experimental routing mode: {mode}")
    surface = np.asarray(surface, dtype=np.float64)
    filled = fill_level(surface)
    flat_mask, flat_rank = resolve_flats(filled)
    local = _flow_directions(filled, flat_mask, flat_rank,
                             float(dx_km))
    batches = topo_batches_weighted(local["targets"], local["weights"],
                                     surface.size)
    if mode == "dinf_ltd":
        rcv, edge_len, error, area, donor = _d8_ltd(
            local, batches, surface.shape, float(dx_km))
    else:
        rcv, edge_len, error, area, donor = _d8_lad(
            local, batches, surface.shape, float(dx_km))
    return RoutingGraph(
        filled_level=filled,
        flat_mask=flat_mask,
        flat_rank=flat_rank.ravel(),
        targets=local["targets"],
        weights=local["weights"],
        flow_angle=local["flow_angle"],
        rcv=rcv,
        edge_len=edge_len,
        transport_len=receiver_transport_lengths(
            rcv, surface.shape, float(dx_km)),
        batches=batches,
        cum_transverse_error=error,
        channel_area_cells=area,
        main_donor=donor,
    )


def accumulate_weighted(targets, weights, batches, runoff=None):
    """Accumulate a conservative source field over a weighted DAG."""
    n = targets.shape[1]
    area = (np.ones(n) if runoff is None
            else np.asarray(runoff, dtype=np.float64).ravel().copy())
    for batch in batches:
        for k in range(targets.shape[0]):
            active = ((weights[k, batch] > 0.0)
                      & (targets[k, batch] != batch))
            if active.any():
                src = batch[active]
                np.add.at(area, targets[k, src],
                          area[src] * weights[k, src])
    return area


def accumulate_channel(rcv, batches, runoff=None):
    """Accumulate a conservative source field over one channel spine."""
    n = rcv.size
    area = (np.ones(n) if runoff is None
            else np.asarray(runoff, dtype=np.float64).ravel().copy())
    for batch in batches:
        dst = rcv[batch]
        active = dst != batch
        if active.any():
            np.add.at(area, dst[active], area[batch[active]])
    return area


def freeman_graph(filled, channel_graph, exponent=1.1):
    """Current all-lower-neighbor MFD on exact levels plus flat spine."""
    n = filled.size
    targets = np.empty((len(OFFSETS), n), dtype=np.int64)
    weights = np.zeros((len(OFFSETS), n))
    for k, ((dy, dx), distance) in enumerate(
            zip(OFFSETS, OFFSET_DISTANCE)):
        nb = _neighbor(filled, dy, dx, np.inf)
        targets[k] = _neighbor_targets(filled.shape, dy, dx).ravel()
        slope = np.clip((filled - nb) / distance, 0.0, None)
        weights[k] = slope.ravel() ** float(exponent)
    total = weights.sum(axis=0)
    wet = total > 0.0
    weights[:, wet] /= total[wet]

    # Exact filled flats have no physical gradient.  Preserve the
    # current concentrated fallback, now supplied by the flat rank.
    fallback = ~wet & (channel_graph.rcv != np.arange(n))
    if fallback.any():
        targets[0, fallback] = channel_graph.rcv[fallback]
        weights[0, fallback] = 1.0

    border = np.zeros_like(filled, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    weights[:, border.ravel()] = 0.0
    batches = topo_batches_weighted(targets, weights, n)
    return targets, weights, batches
