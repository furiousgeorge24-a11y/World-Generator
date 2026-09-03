"""One world of the exploration lab, in a process of its own.

This module exists so a `ProcessPoolExecutor` on the spawn context has a
module-level function to call. Nothing but NumPy arrays, floats, ints, and
lists crosses the process boundary: the worker runs the history and reads off
what the lab's views and report need, and the parent draws the sheets.

Spawn sends the parent's `sys.path` to the child, but this module also puts
its own directory on the path before importing the engine, so it is
importable in a child however it was started.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

from engine.geometry import WorldGeometry  # noqa: E402
from engine.history.kinematics import HistoryParams, run_history  # noqa: E402
from engine.history.plates import (  # noqa: E402
    boundary_mask,
    label_plates,
    plate_areas,
    regime,
    weak_mask,
)

#: The three early times the lab looks at, in Myr. `EARLY_SNAPSHOT_STEPS`
#: keeps steps 2, 4, 8, 12 and 16, which at the 4 Myr step are 8, 16, 32, 48
#: and 64 Myr; these three are picked from that list by nearest time.
EARLY_MYR = (16.0, 32.0, 64.0)

#: The time the report quotes a weak fraction at, so a run that is still
#: growing can be told from one that settled early.
REPORT_MYR = 100.0


def _nearest_early(history, t_myr: float) -> np.ndarray:
    """The weak mask of the kept early snapshot closest to `t_myr`."""
    nearest = min(history.early, key=lambda row: abs(row[1] - t_myr))
    return weak_mask(nearest[2])


def run_one_world(seed: int, pixels: int, scale_km: int,
                  params_record: dict) -> dict:
    """Run one history and return the fields and numbers the lab reads.

    `params_record` is `HistoryParams.to_record()`, a plain dict, because a
    dataclass defined in a package the child has to import is a worse thing
    to pickle than nine numbers.
    """
    started = time.perf_counter()
    params = HistoryParams(**params_record)
    geometry = WorldGeometry(int(seed), int(pixels), int(scale_km))
    history = run_history(geometry, params=params)

    epochs = history.epochs
    final = epochs[-1]
    weak_final = weak_mask(final.strength)
    labels = label_plates(final.strength)

    cells = geometry.history_n ** 2
    areas = plate_areas(labels)
    percent = [round(100.0 * float(area) / cells, 4)
               for area in areas if area >= 0.01 * cells]

    strong = ~weak_final
    strength_mean_strong = (float(final.strength[strong].mean())
                            if strong.any() else 0.0)

    weak_fraction = [float(value) for value in history.weak_fraction]
    peak_index = int(np.argmax(weak_fraction))
    report_step = min(max(int(round(REPORT_MYR / history.step_myr)), 1),
                      history.steps)

    return {
        "seed": int(seed),
        "history_n": int(geometry.history_n),
        "steps": int(history.steps),
        "step_myr": float(history.step_myr),
        "history_myr": float(history.history_myr),
        "yield_strain_per_myr": float(history.yield_strain_per_myr),
        "yield_power": float(history.yield_power),
        # Fields, in the order the views ask for them.
        "labels": labels,
        "boundary": boundary_mask(labels, weak_final),
        "weak_early": [_nearest_early(history, t_myr) for t_myr in EARLY_MYR],
        "weak_epochs": [weak_mask(epoch.strength) for epoch in epochs[:3]],
        "strength_final": final.strength,
        "strength_epochs": [epoch.strength for epoch in epochs[:3]],
        "regime": regime(final.divergence, final.strain_rate, weak_final),
        "velocity": final.velocity,
        "strain_rate": final.strain_rate,
        "power": final.power,
        "stress": final.stress,
        "mismatch": final.mismatch,
        "piece_labels": final.piece_labels,
        "piece_centroid": final.piece_centroid,
        "piece_motion": final.piece_velocity,
        "intact_strength": history.sigma_c_field,
        "drive": history.drive.field(history.history_myr),
        # Trajectory and numbers.
        "weak_fraction": weak_fraction,
        # The same numbers under the seam formulation, where a weak cell is a
        # seam cell, kept under their own name so the record says which rule
        # produced them.
        "seam_fraction": [float(value) for value in history.seam_fraction],
        "tip_total": int(sum(history.tip_count)),
        "nucleation_total": int(sum(history.nucleation_count)),
        "advance_total": int(sum(history.advance_count)),
        "sigma_c": float(history.sigma_c),
        # The block model's record. `piece_count` and `largest_piece_share`
        # are per step, the pieces each step's rigid solve moved, so the
        # report can find the step at which a loop first cut a piece of a
        # given size; `piece_count_final` and `largest_piece_share_final` are
        # read off the final strength beside the plate labels, so they match
        # what the `plates` sheet shows. All zero under the sheet and under
        # `seams = 1`, which have no rigid pieces.
        "piece_count": [int(value) for value in history.piece_count],
        "largest_piece_share": [float(value)
                                for value in history.largest_piece_share],
        "second_piece_cells": [int(value)
                               for value in history.second_piece_cells],
        "force_residual_max": float(max(history.force_residual_max)),
        "torque_residual_max": float(max(history.torque_residual_max)),
        "wrapping_pieces_final": int(history.wrapping_pieces[-1]),
        "wrapping_pieces_max": int(max(history.wrapping_pieces)),
        "marker_count": [int(value) for value in history.marker_count],
        "gaps_closed_total": int(sum(history.gaps_closed)),
        # What the curve of `WORK_ORDER_C04_6.md` §1 did over the run:
        # advances that met another chain and linked to it, edges the move
        # stretched past the bound and the step split, markers left at or
        # above `WEAK_THRESHOLD` at the end — remembered vertices on intact
        # cells — markers that crossed back below it, and the points the last
        # step's raster drew. `degenerate_tip_total` is the tips whose
        # averaged stress tensor was within one per cent of isotropic. Zero
        # under the sheet and under `seams = 1`.
        "meeting_total": int(sum(history.meetings)),
        "subdivision_total": int(sum(history.subdivisions)),
        "suture_markers_final": int(history.suture_markers[-1]),
        "reactivation_total": int(sum(history.reactivations)),
        "sample_count_final": int(history.sample_count[-1]),
        "degenerate_tip_total": int(sum(history.degenerate_tips)),
        # Which part of the slip carried the damage, per step: the share of
        # the step's markers above the yield slip rate, and the elastic and
        # rigid parts averaged over the seam cells. Zero under the sheet and
        # under `seams = 1`.
        "seam_yield_share": [float(value)
                             for value in history.seam_yield_share],
        "elastic_slip_mean": [float(value)
                              for value in history.elastic_slip_mean],
        "rigid_slip_mean": [float(value)
                            for value in history.rigid_slip_mean],
        "piece_count_final": int(len(areas)),
        "largest_piece_share_final": (float(areas[0]) / cells
                                      if areas.size else 0.0),
        "second_piece_cells_final": (int(areas[1]) if areas.size > 1 else 0),
        "plate_percent": percent,
        "weak_final": weak_fraction[-1],
        "weak_peak": max(weak_fraction),
        "weak_peak_myr": (peak_index + 1) * float(history.step_myr),
        "weak_at_report_myr": weak_fraction[report_step - 1],
        "report_myr": float(report_step * history.step_myr),
        "strength_mean_strong": strength_mean_strong,
        "solver_cycles": [int(value) for value in history.solver_cycles],
        "solver_residual": [float(value) for value in history.solver_residual],
        "exhausted_steps": int(sum(1 for value in history.solver_cycles
                                   if value >= params.max_cycles)),
        "seconds": round(time.perf_counter() - started, 3),
    }


__all__ = ["EARLY_MYR", "REPORT_MYR", "run_one_world"]
