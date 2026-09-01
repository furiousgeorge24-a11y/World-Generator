"""Focused mechanics checks for the private physical-outlet successor.

These checks do not exercise or alter the shipped legacy default.  They
verify compact causal support, nested/shifted invariance, sediment closure,
and the absence of a marine thickness cap before an atlas replay is spent.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.erosion import (
    PHYSICAL_MARINE_EFOLDS,
    _fill_to_lowstand_outlets,
    _physical_marine_transport,
    _route_sediment_lowstand,
    flow_accumulation,
    receivers,
    topo_batches,
)
from spikes.shore_outlet_probe import _synthetic_world


DX_KM = 20.0
BASE_LEVEL_M = -80.0
DEPOSITION_LENGTH_KM = 180.0


def _solve(z, ero, runoff):
    marine = z <= BASE_LEVEL_M
    routing_surface = np.where(marine, BASE_LEVEL_M, z)
    filled = _fill_to_lowstand_outlets(routing_surface, marine)
    receiver, targets, weights, flat = receivers(filled)
    marine_flat = marine.ravel()
    index = np.arange(z.size)
    receiver[marine_flat] = index[marine_flat]
    weights[:, marine_flat] = 0.0
    flat[marine_flat] = True
    batches = topo_batches(receiver, targets, weights, flat)
    area, area8 = flow_accumulation(
        receiver, batches, z.size, targets, weights,
        np.where(marine, 0.0, runoff))
    area[marine_flat] = 0.0
    area8[marine_flat] = 0.0
    routed = _route_sediment_lowstand(
        z, ero, receiver, batches, area * DX_KM * DX_KM,
        BASE_LEVEL_M, DEPOSITION_LENGTH_KM, DX_KM,
        _marine_transport=_physical_marine_transport)
    return {
        "z": routed[0],
        "sed": routed[1],
        "discharge": area8.reshape(z.shape),
        "marine": marine,
        "export": routed[2],
        "residual": routed[3],
        "diagnostics": routed[4],
    }


def _run_window(world, window):
    z, ero, runoff = world
    row0, column0, side = window
    selection = np.s_[row0:row0 + side, column0:column0 + side]
    return _solve(
        z[selection].copy(), ero[selection].copy(),
        runoff[selection].copy())


def _extract(result, window, core):
    row0, column0, _ = window
    core_row, core_column, side = core
    rows = np.arange(core_row, core_row + side) - row0
    columns = np.arange(core_column, core_column + side) - column0
    return {
        name: result[name][np.ix_(rows, columns)]
        for name in ("z", "sed", "discharge", "marine")
    }


def _high_load_fan():
    n = 81
    _, x = np.indices((n, n))
    z = -450.0 - 18.0 * x
    source = np.zeros_like(z)
    source[n // 2, 8] = 100_000.0
    deposit, export, residual, diagnostics = _physical_marine_transport(
        z, source, BASE_LEVEL_M, DEPOSITION_LENGTH_KM, DX_KM)
    positive = deposit > 0.0
    rows = np.flatnonzero(np.any(positive, axis=1))
    return {
        "source_m_cells": float(source.sum()),
        "deposited_m_cells": float(deposit.sum()),
        "export_m_cells": float(export),
        "residual_m_cells": float(residual),
        "closure_m_cells": float(diagnostics["closure_m_cells"]),
        "max_deposit_m": float(deposit.max(initial=0.0)),
        "deposit_footprint_cells": int(np.count_nonzero(positive)),
        "deposit_row_span_cells": (
            0 if rows.size == 0 else int(rows[-1] - rows[0] + 1)),
        "top_one_percent_footprint_deposit_fraction": float(
            diagnostics["top_one_percent_footprint_deposit_fraction"]),
        "far_field_export_fraction": float(
            diagnostics["far_field_export_m_cells"] / source.sum()),
        "marine_thickness_cap_applied": bool(
            diagnostics["marine_thickness_cap_applied"]),
        "dynamic_aggradational_routing": bool(
            diagnostics["dynamic_aggradational_routing"]),
    }


def main():
    world = _synthetic_world()
    core = (140, 140, 61)
    windows = {
        "small": (60, 60, 221),
        "large": (20, 20, 301),
        "shifted": (10, 30, 301),
    }
    solved = {name: _run_window(world, window)
              for name, window in windows.items()}
    extracted = {name: _extract(solved[name], windows[name], core)
                 for name in windows}
    comparisons = {}
    for name in ("small", "shifted"):
        comparisons[f"{name}_vs_large"] = {
            field: {
                "array_equal": bool(np.array_equal(
                    extracted[name][field], extracted["large"][field])),
                "max_abs": (0.0 if field == "marine" else float(
                    np.max(np.abs(
                        extracted[name][field]
                        - extracted["large"][field])))),
            }
            for field in ("z", "sed", "discharge", "marine")
        }
    budgets = {}
    for name, result in solved.items():
        diagnostics = result["diagnostics"]
        marine = diagnostics["marine"]
        source = float(diagnostics["source_m_cells"])
        budgets[name] = {
            "source_m_cells": source,
            "relative_closure": abs(float(
                diagnostics["closure_m_cells"])) / max(source, 1.0),
            "max_marine_reach_km": float(marine["max_reach_km"]),
            "far_field_export_fraction": float(
                marine["far_field_export_m_cells"] / max(source, 1.0)),
            "marine_thickness_cap_applied": bool(
                marine["marine_thickness_cap_applied"]),
            "submerged_discharge_max": float(
                result["discharge"][result["marine"]].max(initial=0.0)),
        }
    fan = _high_load_fan()
    checks = {
        "nested_and_shifted_core_exact": all(
            value["array_equal"]
            for comparison in comparisons.values()
            for value in comparison.values()),
        "mass_closure": all(
            value["relative_closure"] <= 1e-12
            for value in budgets.values()),
        "causal_reach_inside_1600km_halo": all(
            value["max_marine_reach_km"] < 1600.0
            for value in budgets.values()),
        "submerged_discharge_suppressed": all(
            value["submerged_discharge_max"] == 0.0
            for value in budgets.values()),
        "no_marine_thickness_cap": (
            not fan["marine_thickness_cap_applied"]
            and all(not value["marine_thickness_cap_applied"]
                    for value in budgets.values())),
        "aggradational_routing_active": fan[
            "dynamic_aggradational_routing"],
        "high_load_fan_spreads_laterally": (
            fan["deposit_row_span_cells"] >= 9),
        "high_load_budget_closes": (
            abs(fan["closure_m_cells"])
            <= 1e-10 * max(fan["source_m_cells"], 1.0)),
    }
    passed = all(checks.values())
    report = {
        "experiment": "private-physical-outlet-mechanics-v1",
        "physical_marine_efolds": PHYSICAL_MARINE_EFOLDS,
        "comparisons": comparisons,
        "budgets": budgets,
        "high_load_fan": fan,
        "checks": checks,
        "passed": passed,
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
