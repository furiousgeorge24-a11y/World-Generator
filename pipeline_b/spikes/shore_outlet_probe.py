"""Focused checks for the private lowstand-outlet localization model.

This is a synthetic mechanics probe, not a production test or public
control.  It verifies three properties before the atlas replay spends its
40-km structural-oracle cost:

* terrestrial discharge stops at physical lowstand water;
* the separate marine pass closes its sediment budget; and
* a core farther from every process edge than the marine reach is exactly
  invariant under nested and shifted solve windows.
"""

from __future__ import annotations

import json

import numpy as np

from engine.erosion import (
    MAR_CAP,
    _bounded_marine_transport,
    _fill_to_lowstand_outlets,
    _route_sediment_lowstand,
    flow_accumulation,
    receivers,
    topo_batches,
)


DX_KM = 20.0
BASE_LEVEL_M = -80.0
DEPOSITION_LENGTH_KM = 180.0

# Rejected private ablation, retained as evidence rather than dead engine
# code. Rebuilding the marine stencil after every deposition step did
# improve a synthetic saturated mouth, but barely improved the real map's
# flux distribution while tripling sediment cost. Same-process medians,
# default-world seed 11, three runs per variant.
REJECTED_DYNAMIC_ABLATION = {
    "reason": "modest real benefit, saturation unchanged, 3.20x sediment cost",
    "static": {
        "far_field_export_fraction": 0.34487367607217356,
        "saturated_cells": 640,
        "saturated_deposit_fraction": 0.396597056187101,
        "deposit_cells": 39407,
        "median_wall_s": 1.2803175000008196,
        "median_sediment_s": 0.683926400000928,
        "max_deposit_m": 150.0,
    },
    "dynamic": {
        "far_field_export_fraction": 0.3285160843089024,
        "saturated_cells": 659,
        "saturated_deposit_fraction": 0.39842298547150523,
        "deposit_cells": 40177,
        "median_wall_s": 2.7879423999984283,
        "median_sediment_s": 2.1892852999735624,
        "max_deposit_m": 150.0,
    },
}


def _synthetic_world(n=341):
    q = (np.arange(n) - n // 2) * DX_KM
    y, x = np.meshgrid(q, q, indexing="ij")
    radius = np.hypot(x, y)
    # One naturally isolated, asymmetric island.  The regional fall puts
    # its lowstand shore near 620 km; coordinate waves break routing ties
    # without reference to any solve window.
    z = (910.0 - 1.60 * radius
         + 42.0 * np.sin(x / 170.0)
         + 31.0 * np.cos(y / 145.0)
         + 18.0 * np.sin((x + y) / 93.0))
    # Synthetic incision source is confined to exposed terrain and varies
    # smoothly in absolute coordinates.
    ero = np.where(
        z > BASE_LEVEL_M,
        np.clip(5.0 + 0.022 * np.maximum(z, 0.0)
                + 2.0 * (1.0 + np.sin((2.0 * x - y) / 210.0)),
                0.0, None),
        0.0,
    )
    runoff = np.where(z > BASE_LEVEL_M,
                      0.65 + 0.35 * np.cos(y / 700.0) ** 2, 0.0)
    return z, ero, runoff


def _solve(z, ero, runoff):
    marine = z <= BASE_LEVEL_M
    routing_surface = np.where(marine, BASE_LEVEL_M, z)
    filled = _fill_to_lowstand_outlets(routing_surface, marine)
    rcv, targets, weights, flat = receivers(filled)
    marine_flat = marine.ravel()
    index = np.arange(z.size)
    rcv[marine_flat] = index[marine_flat]
    weights[:, marine_flat] = 0.0
    flat[marine_flat] = True
    batches = topo_batches(rcv, targets, weights, flat)
    area, area8 = flow_accumulation(
        rcv, batches, z.size, targets, weights,
        np.where(marine, 0.0, runoff))
    area[marine_flat] = 0.0
    area8[marine_flat] = 0.0
    routed = _route_sediment_lowstand(
        z, ero, rcv, batches, area * DX_KM * DX_KM,
        BASE_LEVEL_M, DEPOSITION_LENGTH_KM, DX_KM)
    z_final, deposit, export, residual, diagnostics = routed
    return {
        "z": z_final,
        "sed": deposit,
        "discharge": area8.reshape(z.shape),
        "marine": marine,
        "export": export,
        "residual": residual,
        "diagnostics": diagnostics,
    }


def _extract_global(result, window, core):
    y0, x0, _ = window
    cy0, cx0, side = core
    ys = np.arange(cy0, cy0 + side) - y0
    xs = np.arange(cx0, cx0 + side) - x0
    return {
        key: result[key][np.ix_(ys, xs)]
        for key in ("z", "sed", "discharge", "marine")
    }


def _run_window(world, window):
    z, ero, runoff = world
    y0, x0, side = window
    sl = np.s_[y0:y0 + side, x0:x0 + side]
    return _solve(z[sl].copy(), ero[sl].copy(), runoff[sl].copy())


def _cap_carry_probe():
    """A saturated mouth must spread excess instead of piling it up."""
    n = 41
    _, x = np.indices((n, n))
    z = -450.0 - 18.0 * x
    source = np.zeros_like(z)
    source[n // 2, 4] = 100_000.0
    deposit, export, residual, diagnostics = _bounded_marine_transport(
        z, source, BASE_LEVEL_M, DEPOSITION_LENGTH_KM, DX_KM)
    downstream = float(deposit[:, 6:].sum())
    return {
        "max_deposit_m": float(deposit.max(initial=0.0)),
        "downstream_deposit_m_cells": downstream,
        "closure_m_cells": float(diagnostics["closure_m_cells"]),
        "export_m_cells": float(export),
        "residual_m_cells": float(residual),
        "passed": bool(
            deposit.max(initial=0.0) <= MAR_CAP
            and downstream > 0.0
            and abs(diagnostics["closure_m_cells"]) <= 1e-8
        ),
    }


def main():
    world = _synthetic_world()
    # The 61-cell core has 80 cells (1,600 km) of halo in the small
    # window.  This exceeds the conservative 54-diagonal-step reach
    # (about 1,527 km), not merely its 1,080-km axial reach.
    core = (140, 140, 61)
    windows = {
        "small": (60, 60, 221),
        "large": (20, 20, 301),
        "shifted": (10, 30, 301),
    }
    solved = {name: _run_window(world, window)
              for name, window in windows.items()}
    extracted = {name: _extract_global(solved[name], window, core)
                 for name, window in windows.items()}

    comparisons = {}
    for name in ("small", "shifted"):
        comparisons[f"{name}_vs_large"] = {
            field: {
                "array_equal": bool(np.array_equal(
                    extracted[name][field], extracted["large"][field])),
                "max_abs": (0.0 if field == "marine" else float(
                    np.max(np.abs(extracted[name][field]
                                  - extracted["large"][field])))),
            }
            for field in ("z", "sed", "discharge", "marine")
        }

    budgets = {}
    for name, result in solved.items():
        diag = result["diagnostics"]
        source = float(diag["source_m_cells"])
        closure = float(diag["closure_m_cells"])
        budgets[name] = {
            "source_m_cells": source,
            "closure_m_cells": closure,
            "relative_closure": abs(closure) / max(source, 1.0),
            "marine_reach_km": float(
                diag["marine"]["max_reach_km"]),
            "max_marine_deposit_m": float(
                diag["marine"]["max_deposit_m"]),
            "submerged_discharge_max": float(
                result["discharge"][result["marine"]].max(initial=0.0)),
        }

    cap_carry = _cap_carry_probe()

    passed = bool(
        all(item["array_equal"]
            for comparison in comparisons.values()
            for item in comparison.values())
        and all(item["relative_closure"] <= 1e-12
                for item in budgets.values())
        and all(item["submerged_discharge_max"] == 0.0
                for item in budgets.values())
        and all(item["marine_reach_km"] < 1600.0
                for item in budgets.values())
        and all(item["max_marine_deposit_m"] <= MAR_CAP
                for item in budgets.values())
        and cap_carry["passed"]
    )
    report = {
        "experiment": "private-lowstand-outlet-localization-v1",
        "core": list(core),
        "windows": {key: list(value) for key, value in windows.items()},
        "comparisons": comparisons,
        "budgets": budgets,
        "cap_carry": cap_carry,
        "rejected_dynamic_ablation": REJECTED_DYNAMIC_ABLATION,
        "passed": passed,
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
