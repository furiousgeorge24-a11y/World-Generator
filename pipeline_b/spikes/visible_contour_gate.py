"""Geometry-aware border-contour tripwire for atlas experiments.

This module is deliberately confined to ``spikes``.  It is an evaluation
instrument, not a crop-shaping operation: callers pass the *final rendered
elevation surface* after erosion/deposition and the instrument reports long
visible palette contours which appear to follow a frame edge.

The implementation has no geometry/image dependency beyond NumPy.  It uses
linear marching squares, joins segments through their shared raster edges,
and only measures pieces whose tangent is genuinely edge-parallel.  Short
breaks in an otherwise continuous same-level contour may be bridged, but
different contour components and different palette levels can never join.

Run ``python -m spikes.visible_contour_gate --self-check`` from the
``pipeline_b`` directory for focused synthetic checks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BorderContourConfig:
    """Physical thresholds for the visual border-contour gate."""

    corridor_km: float = 500.0
    tangent_max_degrees: float = 25.0
    normal_band_km: float = 140.0
    normal_band_phase_km: float = 60.0
    bridge_gap_km: float = 60.0
    reject_span_km: float = 819.2
    corner_arm_km: float = 409.6
    corner_corridor_km: float = 400.0
    corner_reach_km: float = 840.0


_EDGES = ("top", "bottom", "left", "right")


def visible_map_levels_m() -> np.ndarray:
    """Return every elevation boundary visible in the map palette.

    The private palette constants are intentionally read here rather than
    duplicated.  This remains an experimental consumer; it does not expand
    the production rendering API.
    """
    from engine.render_map import _LB, _OB

    return np.unique(np.concatenate((
        np.asarray(_OB, np.float64),
        np.array([0.0], np.float64),
        np.asarray(_LB, np.float64),
    )))


class _UnionFind:
    def __init__(self, count: int):
        self.parent = np.arange(count, dtype=np.int64)
        self.rank = np.zeros(count, dtype=np.uint8)

    def find(self, item: int) -> int:
        parent = self.parent
        root = item
        while int(parent[root]) != root:
            root = int(parent[root])
        while item != root:
            following = int(parent[item])
            parent[item] = root
            item = following
        return root

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _edge_intersection(
    z: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    edge: int,
    level: float,
    km_per_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates and stable grid-edge IDs for marching-square edges."""
    height, width = z.shape
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)

    if edge == 0:  # top, left -> right
        y0 = y1 = (rows + 0.5) * km_per_px
        x0 = (cols + 0.5) * km_per_px
        x1 = (cols + 1.5) * km_per_px
        v0 = z[rows, cols]
        v1 = z[rows, cols + 1]
        key = rows * (width - 1) + cols
    elif edge == 1:  # right, top -> bottom
        x0 = x1 = (cols + 1.5) * km_per_px
        y0 = (rows + 0.5) * km_per_px
        y1 = (rows + 1.5) * km_per_px
        v0 = z[rows, cols + 1]
        v1 = z[rows + 1, cols + 1]
        key = height * (width - 1) + rows * width + cols + 1
    elif edge == 2:  # bottom, left -> right
        y0 = y1 = (rows + 1.5) * km_per_px
        x0 = (cols + 0.5) * km_per_px
        x1 = (cols + 1.5) * km_per_px
        v0 = z[rows + 1, cols]
        v1 = z[rows + 1, cols + 1]
        key = (rows + 1) * (width - 1) + cols
    elif edge == 3:  # left, top -> bottom
        x0 = x1 = (cols + 0.5) * km_per_px
        y0 = (rows + 0.5) * km_per_px
        y1 = (rows + 1.5) * km_per_px
        v0 = z[rows, cols]
        v1 = z[rows + 1, cols]
        key = height * (width - 1) + rows * width + cols
    else:
        raise ValueError(f"marching-square edge must be 0..3, got {edge}")

    denominator = np.asarray(v1 - v0, np.float64)
    # A selected marching edge normally has opposite endpoint signs.  Exact
    # threshold plateaus are degenerate; the midpoint fallback is stable and
    # prevents NaNs without inventing a long off-grid segment.
    fraction = np.divide(
        level - v0,
        denominator,
        out=np.full(denominator.shape, 0.5, np.float64),
        where=np.abs(denominator) > np.finfo(np.float64).eps,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    x = np.asarray(x0 + fraction * (x1 - x0), np.float64)
    y = np.asarray(y0 + fraction * (y1 - y0), np.float64)
    return np.column_stack((x, y)), np.asarray(key, np.int64)


def _marching_segments(
    z: np.ndarray,
    level: float,
    km_per_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract linear marching-square segments and endpoint edge IDs."""
    above_tl = z[:-1, :-1] >= level
    above_tr = z[:-1, 1:] >= level
    above_br = z[1:, 1:] >= level
    above_bl = z[1:, :-1] >= level
    cases = (
        above_tl.astype(np.uint8)
        + 2 * above_tr.astype(np.uint8)
        + 4 * above_br.astype(np.uint8)
        + 8 * above_bl.astype(np.uint8)
    )

    # Non-saddle cases.  Edges are top=0, right=1, bottom=2, left=3.
    pairs = {
        1: ((3, 0),),
        2: ((0, 1),),
        3: ((3, 1),),
        4: ((1, 2),),
        6: ((0, 2),),
        7: ((2, 3),),
        8: ((3, 2),),
        9: ((0, 2),),
        11: ((1, 2),),
        12: ((3, 1),),
        13: ((0, 1),),
        14: ((3, 0),),
    }
    specifications: list[tuple[np.ndarray, tuple[int, int]]] = []
    for case, edge_pairs in pairs.items():
        indices = np.flatnonzero(cases.ravel() == case)
        for pair in edge_pairs:
            specifications.append((indices, pair))

    # Resolve the two alternating saddle cases from the bilinear cell-centre
    # value.  This avoids the directional bias of a fixed triangle split.
    centre_high = (
        z[:-1, :-1] + z[:-1, 1:] + z[1:, 1:] + z[1:, :-1]
    ) * 0.25 >= level
    flat_case = cases.ravel()
    flat_centre = centre_high.ravel()
    saddle_specs = (
        (5, False, ((3, 0), (1, 2))),
        (5, True, ((0, 1), (2, 3))),
        (10, False, ((0, 1), (2, 3))),
        (10, True, ((3, 0), (1, 2))),
    )
    for case, high, edge_pairs in saddle_specs:
        indices = np.flatnonzero((flat_case == case) & (flat_centre == high))
        for pair in edge_pairs:
            specifications.append((indices, pair))

    segments: list[np.ndarray] = []
    endpoint_keys: list[np.ndarray] = []
    cell_width = z.shape[1] - 1
    for flat_indices, (edge0, edge1) in specifications:
        if flat_indices.size == 0:
            continue
        rows = flat_indices // cell_width
        cols = flat_indices % cell_width
        point0, key0 = _edge_intersection(
            z, rows, cols, edge0, level, km_per_px)
        point1, key1 = _edge_intersection(
            z, rows, cols, edge1, level, km_per_px)
        segment = np.stack((point0, point1), axis=1)
        length = np.linalg.norm(segment[:, 1] - segment[:, 0], axis=1)
        keep = length > max(1e-10, km_per_px * 1e-10)
        if keep.any():
            segments.append(segment[keep])
            endpoint_keys.append(np.column_stack((key0[keep], key1[keep])))

    if not segments:
        return (np.empty((0, 2, 2), np.float64),
                np.empty((0, 2), np.int64))
    return np.concatenate(segments), np.concatenate(endpoint_keys)


def _component_roots(endpoint_keys: np.ndarray) -> np.ndarray:
    """Label segments which meet at the same interpolated grid edge."""
    count = endpoint_keys.shape[0]
    if count == 0:
        return np.empty(0, np.int64)
    union = _UnionFind(count)
    keys = endpoint_keys.ravel()
    segment_ids = np.repeat(np.arange(count, dtype=np.int64), 2)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    sorted_segments = segment_ids[order]
    starts = np.r_[0, 1 + np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1])]
    ends = np.r_[starts[1:], sorted_keys.size]
    for start, end in zip(starts, ends):
        if end - start < 2:
            continue
        first = int(sorted_segments[start])
        for position in range(start + 1, end):
            union.union(first, int(sorted_segments[position]))
    return np.fromiter((union.find(i) for i in range(count)),
                       dtype=np.int64, count=count)


def _segment_paths(
    segments: np.ndarray,
    endpoint_keys: np.ndarray,
    component_roots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[float, bool]]]:
    """Order non-branching contour components and assign arc coordinates.

    Marching-square contours normally give every endpoint degree two (closed
    loop) or one/two (open line).  Exact-level degeneracies can branch; those
    are split into separate trails so a 60-km tolerance can never jump across
    an ambiguous branch.
    """
    count = segments.shape[0]
    path_id = np.full(count, -1, np.int64)
    endpoint_arc = np.full((count, 2), np.nan, np.float64)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for index, (key0, key1) in enumerate(endpoint_keys):
        adjacency.setdefault(int(key0), []).append((index, 0))
        adjacency.setdefault(int(key1), []).append((index, 1))

    by_component: dict[int, list[int]] = {}
    for index, root in enumerate(component_roots):
        by_component.setdefault(int(root), []).append(index)

    path_info: dict[int, tuple[float, bool]] = {}
    next_path = 0
    for indices in by_component.values():
        remaining = set(indices)
        while remaining:
            remaining_degree: dict[int, int] = {}
            for index in remaining:
                for key in endpoint_keys[index]:
                    key_int = int(key)
                    remaining_degree[key_int] = (
                        remaining_degree.get(key_int, 0) + 1)
            ends = [key for key, degree in remaining_degree.items()
                    if degree == 1]
            if ends:
                current_key = min(ends)
            else:
                first = min(remaining)
                current_key = int(endpoint_keys[first, 0])
            start_key = current_key
            distance = 0.0
            while True:
                options = [entry for entry in adjacency[current_key]
                           if entry[0] in remaining]
                if not options:
                    break
                index, side = min(options)
                other_side = 1 - side
                endpoint_arc[index, side] = distance
                distance += float(lengths[index])
                endpoint_arc[index, other_side] = distance
                path_id[index] = next_path
                remaining.remove(index)
                current_key = int(endpoint_keys[index, other_side])
            cyclic = current_key == start_key
            path_info[next_path] = (distance, cyclic)
            next_path += 1
    if np.any(path_id < 0) or not np.isfinite(endpoint_arc).all():
        raise AssertionError("failed to order all marching-square segments")
    return path_id, endpoint_arc, path_info


def _clip_normal_slab(
    segments: np.ndarray,
    normal_axis: int,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clip line segments to a horizontal/vertical physical corridor."""
    p0 = segments[:, 0]
    p1 = segments[:, 1]
    n0 = p0[:, normal_axis]
    dn = p1[:, normal_axis] - n0
    t0 = np.zeros(segments.shape[0], np.float64)
    t1 = np.ones(segments.shape[0], np.float64)
    changing = np.abs(dn) > np.finfo(np.float64).eps
    lower_t = np.zeros_like(t0)
    upper_t = np.ones_like(t1)
    lower_t[changing] = (low - n0[changing]) / dn[changing]
    upper_t[changing] = (high - n0[changing]) / dn[changing]
    lo = np.minimum(lower_t, upper_t)
    hi = np.maximum(lower_t, upper_t)
    t0[changing] = np.maximum(t0[changing], lo[changing])
    t1[changing] = np.minimum(t1[changing], hi[changing])
    constant_inside = (~changing) & (n0 >= low) & (n0 <= high)
    valid = (changing & (t1 >= t0)) | constant_inside
    direction = p1 - p0
    clipped = np.stack((p0 + t0[:, None] * direction,
                        p0 + t1[:, None] * direction), axis=1)
    return clipped, valid, t0, t1


def _normal_bands(config: BorderContourConfig) -> list[tuple[int, int, float, float]]:
    """Two staggered physical clearance partitions of the edge corridor."""
    if config.normal_band_km <= 0.0:
        return [(0, 0, 0.0, config.corridor_km)]
    phases = (0.0, config.normal_band_phase_km)
    result = []
    for phase_index, phase in enumerate(phases):
        first = int(np.floor((0.0 - phase) / config.normal_band_km))
        last = int(np.ceil(
            (config.corridor_km - phase) / config.normal_band_km))
        for band_index in range(first, last):
            low = max(0.0, phase + band_index * config.normal_band_km)
            high = min(config.corridor_km,
                       phase + (band_index + 1) * config.normal_band_km)
            if high > low:
                result.append((phase_index, band_index, low, high))
    return result


def _qualified_fragments(
    segments: np.ndarray,
    roots: np.ndarray,
    path_ids: np.ndarray,
    endpoint_arc: np.ndarray,
    path_info: dict[int, tuple[float, bool]],
    width_km: float,
    height_km: float,
    config: BorderContourConfig,
    *,
    banded: bool = True,
    corridor_km: float | None = None,
) -> list[dict]:
    if segments.size == 0:
        return []
    delta = segments[:, 1] - segments[:, 0]
    length = np.linalg.norm(delta, axis=1)
    sine_limit = np.sin(np.deg2rad(config.tangent_max_degrees))
    horizontal = np.abs(delta[:, 1]) <= sine_limit * length
    vertical = np.abs(delta[:, 0]) <= sine_limit * length
    corridor = config.corridor_km if corridor_km is None else corridor_km
    if banded:
        bands = _normal_bands(config)
    else:
        bands = [(0, 0, 0.0, corridor)]
    descriptions = (
        ("top", 1, 0, height_km, False, horizontal),
        ("bottom", 1, 0, height_km, True, horizontal),
        ("left", 0, 1, width_km, False, vertical),
        ("right", 0, 1, width_km, True, vertical),
    )
    fragments: list[dict] = []
    for edge, normal_axis, tangent_axis, extent, reverse, aligned in descriptions:
        for phase, band, clearance_low, clearance_high in bands:
            clearance_high = min(clearance_high, corridor)
            if clearance_high <= clearance_low:
                continue
            if reverse:
                low = max(0.0, extent - clearance_high)
                high = min(extent, extent - clearance_low)
            else:
                low = clearance_low
                high = min(extent, clearance_high)
            clipped, inside, clip_t0, clip_t1 = _clip_normal_slab(
                segments, normal_axis, low, high)
            keep = inside & aligned
            for index in np.flatnonzero(keep):
                tangents = clipped[index, :, tangent_axis]
                normal_coordinates = clipped[index, :, normal_axis]
                if reverse:
                    clearances = extent - normal_coordinates
                else:
                    clearances = normal_coordinates
                arc0, arc1 = endpoint_arc[index]
                clipped_arcs = arc0 + np.array(
                    [clip_t0[index], clip_t1[index]]) * (arc1 - arc0)
                path = int(path_ids[index])
                path_length, path_cyclic = path_info[path]
                fragments.append({
                    "edge": edge,
                    "component": int(roots[index]),
                    "path": path,
                    "path_length": float(path_length),
                    "path_cyclic": bool(path_cyclic),
                    "phase": int(phase),
                    "band": int(band),
                    "band_clearance_min_km": float(clearance_low),
                    "band_clearance_max_km": float(clearance_high),
                    "t0": float(np.min(tangents)),
                    "t1": float(np.max(tangents)),
                    "n0": float(np.min(clearances)),
                    "n1": float(np.max(clearances)),
                    "a0": float(np.min(clipped_arcs)),
                    "a1": float(np.max(clipped_arcs)),
                    "segments": 1,
                })
    return fragments


def _arc_gap(left: dict, right: dict) -> float:
    linear = max(0.0, left["a0"] - right["a1"],
                 right["a0"] - left["a1"])
    if not left["path_cyclic"]:
        return float(linear)
    around_seam = (
        left["path_length"] - max(left["a1"], right["a1"])
        + min(left["a0"], right["a0"])
    )
    return float(min(linear, max(0.0, around_seam)))


def _merge_group(fragments: list[dict], bridge_gap_km: float) -> list[dict]:
    """Bridge only short tangent interruptions along the same polyline."""
    count = len(fragments)
    if count == 0:
        return []
    union = _UnionFind(count)
    order = sorted(range(count), key=lambda index: fragments[index]["a0"])
    for position, left_index in enumerate(order):
        left = fragments[left_index]
        for right_index in order[position + 1:]:
            right = fragments[right_index]
            # On an open path, later fragments can only get farther away.
            if (not left["path_cyclic"]
                    and right["a0"] - left["a1"] > bridge_gap_km):
                break
            if _arc_gap(left, right) <= bridge_gap_km:
                union.union(left_index, right_index)

    clusters: dict[int, dict] = {}
    for index, fragment in enumerate(fragments):
        root = union.find(index)
        if root not in clusters:
            clusters[root] = dict(fragment)
            continue
        cluster = clusters[root]
        for low_key in ("t0", "n0", "a0"):
            cluster[low_key] = min(cluster[low_key], fragment[low_key])
        for high_key in ("t1", "n1", "a1"):
            cluster[high_key] = max(cluster[high_key], fragment[high_key])
        cluster["segments"] += fragment["segments"]
    return list(clusters.values())


def _merge_fragments(fragments: list[dict], level: float) -> list[dict]:
    groups: dict[tuple[int, str, int, int, int], list[dict]] = {}
    for fragment in fragments:
        key = (fragment["component"], fragment["edge"], fragment["path"],
               fragment["phase"], fragment["band"])
        groups.setdefault(key, []).append(fragment)
    runs: list[dict] = []
    bridge = float(fragments[0]["_bridge_gap_km"]) if fragments else 0.0
    for (component, edge, path, phase, band), group in groups.items():
        for cluster in _merge_group(group, bridge):
            cluster.pop("_bridge_gap_km", None)
            cluster.update({
                "level_m": float(level),
                "component": int(component),
                "edge": edge,
                "path": int(path),
                "phase": int(phase),
                "band": int(band),
                "span_km": float(cluster["t1"] - cluster["t0"]),
            })
            runs.append(cluster)
    return runs


def _corner_violations(
    runs: list[dict],
    width_km: float,
    height_km: float,
    config: BorderContourConfig,
) -> list[dict]:
    by_component: dict[tuple[int, int], dict[str, list[dict]]] = {}
    for run in runs:
        key = (run["component"], run["path"])
        by_component.setdefault(key, {}).setdefault(
            run["edge"], []).append(run)

    # (name, horizontal edge, its corner coordinate, vertical edge,
    #  its corner coordinate)
    corners = (
        ("top_left", "top", 0.0, "left", 0.0),
        ("top_right", "top", width_km, "right", 0.0),
        ("bottom_left", "bottom", 0.0, "left", height_km),
        ("bottom_right", "bottom", width_km, "right", height_km),
    )
    violations = []
    for (component, path), edge_runs in by_component.items():
        for name, h_edge, h_corner, v_edge, v_corner in corners:
            horizontal = [
                (run, _corner_arm_span(
                    run, h_corner, width_km, config.corner_reach_km))
                for run in edge_runs.get(h_edge, [])
            ]
            horizontal = [(run, span) for run, span in horizontal
                          if span >= config.corner_arm_km]
            vertical = [
                (run, _corner_arm_span(
                    run, v_corner, height_km, config.corner_reach_km))
                for run in edge_runs.get(v_edge, [])
            ]
            vertical = [(run, span) for run, span in vertical
                        if span >= config.corner_arm_km]
            if horizontal and vertical:
                pairs = [
                    (h_run, h_span, v_run, v_span)
                    for h_run, h_span in horizontal
                    for v_run, v_span in vertical
                    if _arc_gap(h_run, v_run)
                    <= 2.0 * config.corner_reach_km
                ]
                if not pairs:
                    continue
                h_run, h_span, v_run, v_span = max(
                    pairs, key=lambda pair: min(pair[1], pair[3]))
                violations.append({
                    "kind": "corner_joined_arms",
                    "corner": name,
                    "level_m": float(h_run["level_m"]),
                    "component": int(component),
                    "path": int(path),
                    "horizontal_edge": h_edge,
                    "vertical_edge": v_edge,
                    "horizontal_arm_km": float(h_span),
                    "vertical_arm_km": float(v_span),
                    "connecting_arc_gap_km": _arc_gap(h_run, v_run),
                })
    return violations


def _interval_distance(run: dict, coordinate: float) -> float:
    return float(max(0.0, run["t0"] - coordinate,
                     coordinate - run["t1"]))


def _corner_arm_span(
    run: dict,
    corner_coordinate: float,
    edge_extent: float,
    reach_km: float,
) -> float:
    if corner_coordinate <= 0.0:
        low, high = 0.0, min(edge_extent, reach_km)
    else:
        low, high = max(0.0, edge_extent - reach_km), edge_extent
    return float(max(0.0, min(run["t1"], high) - max(run["t0"], low)))


def evaluate_visible_border_contours(
    final_surface_m: np.ndarray,
    km_per_px: float,
    levels_m: Iterable[float] | None = None,
    config: BorderContourConfig = BorderContourConfig(),
) -> dict:
    """Evaluate edge-parallel visible contours on a final elevation map.

    ``final_surface_m`` must be the post-erosion/deposition surface actually
    used by the renderer (for ``sample_map`` output, pass ``m["h"]``).  The
    gate never edits that surface.
    """
    z = np.asarray(final_surface_m, np.float64)
    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError("final_surface_m must be a 2-D array at least 2x2")
    if not np.isfinite(z).all():
        raise ValueError("final_surface_m contains non-finite values")
    if not np.isfinite(km_per_px) or km_per_px <= 0.0:
        raise ValueError("km_per_px must be positive and finite")
    for field, value in asdict(config).items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be non-negative and finite")
    if config.tangent_max_degrees > 90.0:
        raise ValueError("tangent_max_degrees must be <= 90")

    if levels_m is None:
        levels = visible_map_levels_m()
    else:
        levels = np.unique(np.asarray(list(levels_m), np.float64))
    levels = levels[np.isfinite(levels)]
    width_km = z.shape[1] * float(km_per_px)
    height_km = z.shape[0] * float(km_per_px)

    all_runs: list[dict] = []
    violations: list[dict] = []
    active_levels = 0
    for level in levels:
        # A level outside the sampled range cannot form a visible contour.
        if level < float(z.min()) or level > float(z.max()):
            continue
        segments, keys = _marching_segments(z, float(level), km_per_px)
        if segments.shape[0] == 0:
            continue
        active_levels += 1
        roots = _component_roots(keys)
        path_ids, endpoint_arc, path_info = _segment_paths(
            segments, keys, roots)
        fragments = _qualified_fragments(
            segments, roots, path_ids, endpoint_arc, path_info,
            width_km, height_km, config)
        for fragment in fragments:
            fragment["_bridge_gap_km"] = config.bridge_gap_km
        runs = _merge_fragments(fragments, float(level))
        all_runs.extend(runs)
        longest_by_component_edge: dict[tuple[int, str], dict] = {}
        for run in runs:
            key = (run["component"], run["edge"])
            prior = longest_by_component_edge.get(key)
            if prior is None or run["span_km"] > prior["span_km"]:
                longest_by_component_edge[key] = run
        for run in longest_by_component_edge.values():
            if run["span_km"] >= config.reject_span_km:
                violations.append({
                    "kind": "long_edge_parallel_contour",
                    "level_m": float(level),
                    "component": int(run["component"]),
                    "edge": run["edge"],
                    "span_km": float(run["span_km"]),
                    "normal_min_km": float(run["n0"]),
                    "normal_max_km": float(run["n1"]),
                    "normal_band_phase": int(run["phase"]),
                    "normal_band": int(run["band"]),
                    "normal_band_min_km": float(
                        run["band_clearance_min_km"]),
                    "normal_band_max_km": float(
                        run["band_clearance_max_km"]),
                })
        corner_fragments = _qualified_fragments(
            segments, roots, path_ids, endpoint_arc, path_info,
            width_km, height_km, config, banded=False,
            corridor_km=config.corner_corridor_km)
        for fragment in corner_fragments:
            fragment["_bridge_gap_km"] = config.bridge_gap_km
        corner_runs = _merge_fragments(corner_fragments, float(level))
        violations.extend(_corner_violations(
            corner_runs, width_km, height_km, config))

    # Keep the report compact while retaining every reject reason.  The top
    # runs make pass/fail decisions auditable without serializing each tiny
    # marching segment.
    ordered_runs = sorted(all_runs, key=lambda run: run["span_km"],
                          reverse=True)
    max_span = ordered_runs[0]["span_km"] if ordered_runs else 0.0
    return {
        "passed": not violations,
        "config": asdict(config),
        "shape": [int(z.shape[0]), int(z.shape[1])],
        "km_per_px": float(km_per_px),
        "frame_width_km": width_km,
        "frame_height_km": height_km,
        "levels_tested_m": [float(level) for level in levels],
        "active_level_count": active_levels,
        "qualified_run_count": len(all_runs),
        "max_parallel_span_km": float(max_span),
        "violations": violations,
        "longest_runs": ordered_runs[:12],
    }


def _synthetic_surface(
    size: int,
    km_per_px: float,
    coastline_y,
) -> np.ndarray:
    x = (np.arange(size) + 0.5) * km_per_px
    y = (np.arange(size) + 0.5) * km_per_px
    line = np.asarray(coastline_y(x), np.float64)
    return y[:, None] - line[None, :]


def run_self_checks() -> dict[str, bool]:
    """Focused, deterministic checks of orientation/gap/corner semantics."""
    km = 8.0
    size = 256
    level = [0.0]

    parallel = _synthetic_surface(size, km, lambda x: 181.0 + 0.0 * x)
    parallel_result = evaluate_visible_border_contours(parallel, km, level)

    # A 30-degree crossing occupies enough of the 500-km corridor that a
    # position-only detector would exceed 819.2 km.  The tangent gate must
    # exclude it.
    diagonal = _synthetic_surface(
        size, km, lambda x: 8.0 + np.tan(np.deg2rad(30.0)) * x)
    diagonal_result = evaluate_visible_border_contours(diagonal, km, level)

    # Two long horizontal arms separated by a 40x20-km, >25-degree ramp.
    # Its endpoints are <60 km apart, so the same contour should bridge.
    def short_break(x):
        return 181.0 + np.clip((x - 721.0) / 38.0, 0.0, 1.0) * 20.0

    bridged = _synthetic_surface(192, km, short_break)
    bridged_result = evaluate_visible_border_contours(bridged, km, level)

    # A 120x120-km diagonal interruption separates two individually short
    # arms.  It must not be bridged into a false long-span rejection.
    def long_break(x):
        return 181.0 + np.clip((x - 720.0) / 120.0, 0.0, 1.0) * 120.0

    separated = _synthetic_surface(192, km, long_break)
    separated_result = evaluate_visible_border_contours(separated, km, level)

    # min(x-a, y-a)=0 is a joined L in the top-left corner context.  Each arm
    # is deliberately below the ordinary 819.2-km limit but above 409.6 km.
    x = (np.arange(128) + 0.5) * km
    y = (np.arange(128) + 0.5) * km
    corner = np.minimum(x[None, :] - 303.0, y[:, None] - 303.0)
    corner_result = evaluate_visible_border_contours(corner, km, level)

    # A single natural-looking contour gradually migrates from 40 to 400 km
    # clearance through nine short 45-degree turns.  An unbanded transitive
    # bridge would concatenate its horizontal pieces into one frame-wide
    # false positive; staggered normal-clearance bands keep the measurement
    # local to comparable clearance.
    def clearance_chain(x):
        result = np.full(x.shape, 40.0)
        for step in range(9):
            transition = 150.0 + step * 140.0
            result += 40.0 * np.clip(
                (x - transition) / 40.0, 0.0, 1.0)
        return result

    curving = _synthetic_surface(210, km, clearance_chain)
    curving_result = evaluate_visible_border_contours(curving, km, level)
    unbanded_control = evaluate_visible_border_contours(
        curving, km, level,
        BorderContourConfig(normal_band_km=1000.0,
                            normal_band_phase_km=0.0),
    )

    checks = {
        "long_parallel_rejected": not parallel_result["passed"],
        "thirty_degree_crossing_allowed": diagonal_result["passed"],
        "sub_60km_break_bridged": not bridged_result["passed"],
        "over_60km_break_not_bridged": separated_result["passed"],
        "joined_corner_arms_rejected": any(
            violation["kind"] == "corner_joined_arms"
            for violation in corner_result["violations"]),
        "corner_arms_below_long_span": not any(
            violation["kind"] == "long_edge_parallel_contour"
            for violation in corner_result["violations"]),
        "curving_clearance_chain_allowed": curving_result["passed"],
        "curving_unbanded_control_rejected": not unbanded_control["passed"],
    }
    if not all(checks.values()):
        details = {name: result for name, result in (
            ("parallel", parallel_result),
            ("diagonal", diagonal_result),
            ("bridged", bridged_result),
            ("separated", separated_result),
            ("corner", corner_result),
            ("curving", curving_result),
            ("unbanded_control", unbanded_control),
        )}
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"visible contour self-checks failed: {failed}; "
                             f"details={details}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if not args.self_check:
        parser.error("this helper currently exposes only --self-check")
    checks = run_self_checks()
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
