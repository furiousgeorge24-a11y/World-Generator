"""40-km atlas oracle plus localized surface-process convergence test.

This consumes the precommitted hardest candidate from atlas_survey.py.
It does not reselect at 40 km: if the stored crop fails, the experiment
fails.  Structural fields are solved globally; only the expensive 20-km
erosion/hydrology tail is localized and compared across nested/shifted
domains.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from engine.elevation import coarse_elevation
from engine.erosion import E_KM, run_erosion
from engine.render_map import render_map_view
from engine.surface import sample_map
from engine.tectonics import FRAME_KM, build_structure
from spikes.atlas_survey import (
    ATLAS_KM,
    PARALLEL_SPAN_LIMIT_KM,
    _atlas_config,
    _evaluate_candidates,
    _parallel_coast_span,
    _seed_atlas_nuclei,
)
from spikes.visible_contour_gate import evaluate_visible_border_contours


PRIMARY_ORIGIN = (13696.0, 3200.0)  # x, y; frozen 120-km candidate ID
REFERENCE_80 = {
    "land_fraction": 0.444091796875,
    "land_capacity_score": 0.7592230879965918,
    "edge_envelope_max_m": -784.7588538516768,
}
ORACLE_KM = 40.0
CORE_COLLAR_KM = 40.0
SMALL_HALO_KM = 1600.0
LARGE_HALO_KM = 2400.0
SHIFT_KM = 400.0


def _window(structure, origin, halo_km):
    x0, y0 = origin
    n_world = int(round(structure.world_km / E_KM))
    e_km = structure.world_km / n_world
    side = int(np.ceil((FRAME_KM + 2.0 * halo_km) / e_km))
    cx = (x0 + 0.5 * FRAME_KM) / e_km
    cy = (y0 + 0.5 * FRAME_KM) / e_km
    ix0 = int(np.floor(cx - 0.5 * side))
    iy0 = int(np.floor(cy - 0.5 * side))
    ix0 = min(max(ix0, 0), n_world - side)
    iy0 = min(max(iy0, 0), n_world - side)
    return iy0, ix0, side


def _shift_window(window, structure, dy_km, dx_km):
    iy0, ix0, side = window
    n_world = int(round(structure.world_km / E_KM))
    e_km = structure.world_km / n_world
    dy = int(round(dy_km / e_km))
    dx = int(round(dx_km / e_km))
    shifted_y = min(max(iy0 + dy, 0), n_world - side)
    shifted_x = min(max(ix0 + dx, 0), n_world - side)
    return shifted_y, shifted_x, side


def _core_global_indices(er, origin):
    x0, y0 = origin
    iy0, ix0, side = er["process_window"]
    e_km = float(er["e_km"])
    gy = np.arange(iy0, iy0 + side)
    gx = np.arange(ix0, ix0 + side)
    yc = (gy + 0.5) * e_km
    xc = (gx + 0.5) * e_km
    keep_y = gy[(yc >= y0 - CORE_COLLAR_KM)
                & (yc < y0 + FRAME_KM + CORE_COLLAR_KM)]
    keep_x = gx[(xc >= x0 - CORE_COLLAR_KM)
                & (xc < x0 + FRAME_KM + CORE_COLLAR_KM)]
    return keep_y, keep_x


def _extract(er, key, global_y, global_x):
    iy0, ix0, _ = er["process_window"]
    return er[key][np.ix_(global_y - iy0, global_x - ix0)]


def _continuous_stats(left, right):
    difference = np.abs(np.asarray(left, np.float64)
                        - np.asarray(right, np.float64))
    return {
        "array_equal": bool(np.array_equal(left, right, equal_nan=True)),
        "max_abs": float(difference.max(initial=0.0)),
        "p99_abs": float(np.percentile(difference, 99.0)),
        "mean_abs": float(difference.mean()),
    }


def _masked_continuous_stats(left, right, mask):
    """Difference statistics restricted to a diagnosed surface region."""
    left = np.asarray(left)
    right = np.asarray(right)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {
            "count": 0,
            "array_equal": True,
            "max_abs": 0.0,
            "p99_abs": 0.0,
            "mean_abs": 0.0,
        }
    difference = np.abs(left[mask].astype(np.float64)
                        - right[mask].astype(np.float64))
    return {
        "count": int(mask.sum()),
        "array_equal": bool(np.array_equal(
            left[mask], right[mask], equal_nan=True)),
        "max_abs": float(difference.max(initial=0.0)),
        "p99_abs": float(np.percentile(difference, 99.0)),
        "mean_abs": float(difference.mean()),
    }


def _river_topology(er, origin):
    x0, y0 = origin
    edges = er["river_edges"]
    keep = (
        (edges["x0"] >= x0 - CORE_COLLAR_KM)
        & (edges["x0"] < x0 + FRAME_KM + CORE_COLLAR_KM)
        & (edges["y0"] >= y0 - CORE_COLLAR_KM)
        & (edges["y0"] < y0 + FRAME_KM + CORE_COLLAR_KM)
    )
    return {
        (round(float(x0_), 6), round(float(y0_), 6),
         round(float(x1_), 6), round(float(y1_), 6))
        for x0_, y0_, x1_, y1_ in zip(
            edges["x0"][keep], edges["y0"][keep],
            edges["x1"][keep], edges["y1"][keep])
    }


def _river_values(er, origin):
    """Discharge keyed by the same absolute geometry used for topology."""
    x0, y0 = origin
    edges = er["river_edges"]
    keep = (
        (edges["x0"] >= x0 - CORE_COLLAR_KM)
        & (edges["x0"] < x0 + FRAME_KM + CORE_COLLAR_KM)
        & (edges["y0"] >= y0 - CORE_COLLAR_KM)
        & (edges["y0"] < y0 + FRAME_KM + CORE_COLLAR_KM)
    )
    result = {}
    for index in np.flatnonzero(keep):
        key = (
            round(float(edges["x0"][index]), 6),
            round(float(edges["y0"][index]), 6),
            round(float(edges["x1"][index]), 6),
            round(float(edges["y1"][index]), 6),
        )
        result[key] = float(edges["a8"][index])
    return result


def _compare_domains(reference, other, origin, river_density):
    gy, gx = _core_global_indices(reference, origin)
    gy2, gx2 = _core_global_indices(other, origin)
    if not np.array_equal(gy, gy2) or not np.array_equal(gx, gx2):
        raise AssertionError("localized domains do not cover same core")

    extracted = {}
    fields = {}
    for key in ("z0", "z", "ero", "sed", "discharge_log",
                "lake_depth", "lake_surf"):
        ref_field = _extract(reference, key, gy, gx)
        other_field = _extract(other, key, gy, gx)
        extracted[key] = (ref_field, other_field)
        fields[key] = _continuous_stats(
            ref_field,
            other_field,
        )

    ref_z, other_z = extracted["z"]
    ref_lake = extracted["lake_depth"][0] > 0.0
    other_lake = extracted["lake_depth"][1] > 0.0
    ref_q = np.expm1(extracted["discharge_log"][0])
    other_q = np.expm1(extracted["discharge_log"][1])
    land = ref_z >= 0.0
    ocean = ~land
    field_regions = {}
    for region_name, mask in (("land", land), ("ocean", ocean)):
        field_regions[region_name] = {
            key: _masked_continuous_stats(left, right, mask)
            for key, (left, right) in extracted.items()
        }

    # The process solver stores only A8 > 30 river edges. The render may
    # impose a still-higher density threshold, and uses a tenfold cutoff
    # for trunk width. Diagnose those actual drawn classes separately
    # from the full discharge raster.
    render_threshold = max(
        30.0, 10.0 ** (3.1 - 2.3 * float(river_density)))
    drawn = (ref_q > render_threshold) & land
    relative_q = np.abs(other_q - ref_q) / np.maximum(ref_q, 1.0)

    ref_topology = _river_topology(reference, origin)
    other_topology = _river_topology(other, origin)
    topology_delta = ref_topology.symmetric_difference(other_topology)
    ref_values = _river_values(reference, origin)
    other_values = _river_values(other, origin)
    common_edges = sorted(ref_topology & other_topology)
    if common_edges:
        ref_a8 = np.array([ref_values[key] for key in common_edges])
        other_a8 = np.array([other_values[key] for key in common_edges])
        edge_relative = np.abs(other_a8 - ref_a8) / np.maximum(ref_a8, 1.0)

        def render_class(values):
            return ((values > render_threshold).astype(np.int8)
                    + (values > 10.0 * render_threshold).astype(np.int8))

        edge_class_delta = int(np.count_nonzero(
            render_class(ref_a8) != render_class(other_a8)))
        edge_relative_max = float(edge_relative.max(initial=0.0))
        edge_relative_p99 = float(np.percentile(edge_relative, 99.0))
    else:
        edge_class_delta = 0
        edge_relative_max = 0.0
        edge_relative_p99 = 0.0
    result = {
        "fields": fields,
        "fields_by_region": field_regions,
        "ocean_mask_xor": int(np.count_nonzero((ref_z < 0.0)
                                                ^ (other_z < 0.0))),
        "lake_mask_xor": int(np.count_nonzero(ref_lake ^ other_lake)),
        "drawn_discharge_land_cell_count": int(drawn.sum()),
        "drawn_discharge_max_relative": float(
            relative_q[drawn].max(initial=0.0)),
        "river_edge_count_reference": len(ref_topology),
        "river_edge_count_other": len(other_topology),
        "river_topology_symmetric_difference": len(topology_delta),
        "river_common_edge_a8_max_relative": edge_relative_max,
        "river_common_edge_a8_p99_relative": edge_relative_p99,
        "river_render_class_difference": edge_class_delta,
    }
    result["passed"] = bool(
        fields["z0"]["array_equal"]
        and fields["z"]["max_abs"] <= 0.5
        and fields["z"]["p99_abs"] <= 0.05
        and fields["ero"]["max_abs"] <= 0.5
        and fields["ero"]["p99_abs"] <= 0.05
        and fields["sed"]["max_abs"] <= 0.5
        and fields["sed"]["p99_abs"] <= 0.05
        and result["ocean_mask_xor"] == 0
        and result["lake_mask_xor"] == 0
        and result["drawn_discharge_max_relative"] <= 0.005
        and result["river_topology_symmetric_difference"] == 0
    )
    return result


def _compare_rendered(structure, elevation, reference, other, cfg,
                      seed, origin, size):
    """Compare the actual sampled and rendered crop, not solver rasters."""
    x0, y0 = origin
    kwargs = {"_frame_window_km": (y0, x0, FRAME_KM)}
    left = sample_map(
        structure, elevation, reference, cfg, seed, size, **kwargs)
    right = sample_map(
        structure, elevation, other, cfg, seed, size, **kwargs)
    fields = {
        key: _continuous_stats(left[key], right[key])
        for key in ("h", "hc", "riv_log", "sed", "lake_level")
    }
    masks = {
        key: int(np.count_nonzero(left[key] ^ right[key]))
        for key in ("water", "ocean", "lake")
    }
    views = {}
    for view in ("hypsometric", "isobaths", "slope", "drainage",
                 "sediment"):
        left_rgb = np.asarray(render_map_view(
            left, view, cfg.river_density))
        right_rgb = np.asarray(render_map_view(
            right, view, cfg.river_density))
        changed = np.any(left_rgb != right_rgb, axis=2)
        views[view] = {
            "array_equal": bool(np.array_equal(left_rgb, right_rgb)),
            "changed_pixels": int(changed.sum()),
            "changed_fraction": float(changed.mean()),
        }
    return {
        "size": size,
        "fields": fields,
        "mask_xor": masks,
        "views": views,
        "passed_exact": bool(
            all(item["array_equal"] for item in fields.values())
            and all(value == 0 for value in masks.values())
            and all(item["array_equal"] for item in views.values())
        ),
    }


def _render_audit(structure, elevation, er, cfg, seed, origin,
                  out_dir):
    x0, y0 = origin
    sizes = {}
    for size in (128, 512, 1024):
        sampled = sample_map(
            structure, elevation, er, cfg, seed, size,
            _frame_window_km=(y0, x0, FRAME_KM),
        )
        km_px = float(sampled["km_per_px"])
        band = max(1, int(np.ceil(16.1 / km_px)))
        ring = np.zeros((size, size), bool)
        ring[:band, :] = ring[-band:, :] = True
        ring[:, :band] = ring[:, -band:] = True
        ocean_ok = bool(np.all(sampled["ocean"][ring]))
        contour_span = _parallel_coast_span(
            sampled["h"] >= 0.0, km_px)
        sizes[str(size)] = {
            "edge_band_pixels": band,
            "ocean_edge_band_passed": ocean_ok,
            "land_fraction": float(np.mean(sampled["h"] >= 0.0)),
            "coarse_parallel_coast_span_km": contour_span,
            "coarse_parallel_tripwire":
                bool(contour_span >= PARALLEL_SPAN_LIMIT_KM),
        }
        if size == 1024:
            # Final authority for the visual edge-artifact question.  The
            # older land-mask span remains above as a deliberately coarse
            # diagnostic, but this gate follows connected, same-palette-level
            # contours on the actual post-process rendered surface.
            sizes[str(size)]["visible_contour_gate"] = (
                evaluate_visible_border_contours(
                    sampled["h"], km_px))
        if size in (512, 1024):
            for view in ("hypsometric", "isobaths"):
                image = render_map_view(sampled, view, cfg.river_density)
                image.save(out_dir / f"seed{seed}_{view}_{size}.png")
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--continental-budget", type=float, default=0.65)
    parser.add_argument("--out", type=Path,
                        default=Path("out") / "atlas_replay")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = _atlas_config(args.continental_budget)
    started = time.perf_counter()
    structure = build_structure(
        args.seed,
        cfg,
        _world_km=ATLAS_KM,
        _coarse_km=ORACLE_KM,
        _continent_seeder=_seed_atlas_nuclei,
    )
    elevation = coarse_elevation(structure, cfg, args.seed)
    candidates = _evaluate_candidates(
        structure, elevation, [elevation], args.seed)
    candidate_by_origin = {
        candidate.origin: candidate for candidate in candidates["safe"]
    }
    candidate = candidate_by_origin.get(PRIMARY_ORIGIN)
    structural_elapsed = time.perf_counter() - started

    if candidate is None:
        structural = {
            "passed": False,
            "reason": "precommitted candidate is not 40-km safe",
        }
    else:
        structural = {
            "passed": bool(
                candidate.water_clearance_m >= 160.0
                and candidate.land_fraction >= 0.35
                and abs(candidate.land_fraction
                        - REFERENCE_80["land_fraction"]) <= 0.05
                and abs(candidate.land_capacity_score
                        - REFERENCE_80["land_capacity_score"]) <= 0.05
                and candidate.max_parallel_span_km
                < PARALLEL_SPAN_LIMIT_KM
            ),
            "candidate": asdict(candidate),
            "delta_from_80": {
                "land_fraction": candidate.land_fraction
                - REFERENCE_80["land_fraction"],
                "land_capacity_score": candidate.land_capacity_score
                - REFERENCE_80["land_capacity_score"],
                "edge_envelope_max_m": candidate.edge_envelope_max_m
                - REFERENCE_80["edge_envelope_max_m"],
            },
        }

    report = {
        "experiment": "atlas-40km-local-process-v1",
        "seed": args.seed,
        "continental_budget": args.continental_budget,
        "origin_xy_km": list(PRIMARY_ORIGIN),
        "structure_n": structure.n,
        "structure_spacing_km": structure.world_km / structure.n,
        "structural_elapsed_s": structural_elapsed,
        "structural": structural,
    }

    # Continue even when only the contour tripwire failed: the exact
    # rendered process surface is the evidence needed to audit that proxy.
    if candidate is not None:
        windows = {
            "small": _window(structure, PRIMARY_ORIGIN, SMALL_HALO_KM),
            "large": _window(structure, PRIMARY_ORIGIN, LARGE_HALO_KM),
        }
        windows["shifted"] = _shift_window(
            windows["large"], structure, -SHIFT_KM, SHIFT_KM)

        eroded = {}
        erosion_timings = {}
        for name, window in windows.items():
            t0 = time.perf_counter()
            eroded[name] = run_erosion(
                structure, elevation, cfg, args.seed,
                _process_window=window,
            )
            erosion_timings[name] = time.perf_counter() - t0

        nested = _compare_domains(
            eroded["large"], eroded["small"], PRIMARY_ORIGIN,
            cfg.river_density)
        shifted = _compare_domains(
            eroded["large"], eroded["shifted"], PRIMARY_ORIGIN,
            cfg.river_density)
        render_convergence = {
            name: {
                str(size): _compare_rendered(
                    structure, elevation, eroded["large"], other,
                    cfg, args.seed, PRIMARY_ORIGIN, size)
                for size in (512, 1024)
            }
            for name, other in (
                ("small_vs_large", eroded["small"]),
                ("shifted_vs_large", eroded["shifted"]),
            )
        }
        render = _render_audit(
            structure, elevation, eroded["large"], cfg, args.seed,
            PRIMARY_ORIGIN, args.out)
        report.update({
            "process_windows": {key: list(value)
                                for key, value in windows.items()},
            "erosion_elapsed_s": erosion_timings,
            "nested_small_vs_large": nested,
            "shifted_vs_large": shifted,
            "render_convergence": render_convergence,
            "render": render,
        })
        report["passed"] = bool(
            structural["passed"]
            and nested["passed"]
            and shifted["passed"]
            and all(
                comparison["passed_exact"]
                for relation in render_convergence.values()
                for comparison in relation.values()
            )
            and all(item["ocean_edge_band_passed"]
                    for item in render.values())
            # The 1024 connected visible-level gate is authoritative.
            # The binary-land span above is retained only as a coarse
            # diagnostic and must neither hide nor manufacture a pass.
            and render["1024"]["visible_contour_gate"]["passed"]
        )
    else:
        report["passed"] = False

    report["elapsed_s"] = time.perf_counter() - started
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
