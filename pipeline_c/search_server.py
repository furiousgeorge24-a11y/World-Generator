"""The regime search, on port 5004, with a live gallery.

A third tab on its own port. It does not touch the shared shell in `webui/`:
it is a small Flask app of its own, serving one page and a handful of JSON
endpoints, with the search running on a background thread that owns the
process pool.

Nothing here decides anything. The search screens cells on measured
properties of a plate regime and the gallery shows what came out; a cell that
passes at twelve seeds is a candidate for the author's eyes, and this server
grants it nothing.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import re
import sys
import threading

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from flask import Flask, Response, jsonify, request, send_from_directory  # noqa: E402

import search  # noqa: E402
from engine.history.constants import SUPPORTED_SIZES  # noqa: E402

PORT = 5004
POOL_WORKERS = 8
_WEB = _HERE / "web_search"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_CELLS_LIMIT_DEFAULT = 200
_CELLS_LIMIT_MAX = 1000

UINT32 = 2**32 - 1

#: Every knob the page shows, as `(group, name, kind, lo, hi, meaning)`.
#: `kind` is `float`, `int`, or `set_int` (a comma-separated list of whole
#: numbers). The meanings are the one-liners printed beside each field.
KNOBS: tuple[tuple[str, str, str, float, float, str], ...] = (
    ("screen", "weak_min", "float", 0.0, 1.0,
     "Some lithosphere must have failed: the smallest final weak fraction."),
    ("screen", "weak_max", "float", 0.0, 1.0,
     "Most of it must not have: the largest final weak fraction."),
    ("screen", "peak_ratio_max", "float", 1.0, 100.0,
     "Peak weak fraction over final: the weak set is not collapsing back or "
     "overshooting."),
    ("screen", "flat_window_myr", "float", 0.0, 1000.0,
     "Window, ending at the last step, over which the weak fraction must be "
     "settled."),
    ("screen", "flat_tolerance", "float", 0.0, 1.0,
     "How far the weak fraction may drift across that window."),
    ("screen", "plates_min", "int", 0, 4096,
     "Fewest plates above 1 % of the parent."),
    ("screen", "plates_max", "int", 0, 4096,
     "Most plates above 1 % of the parent."),
    ("screen", "network_share_min", "float", 0.0, 1.0,
     "Largest 8-connected component of the weak set, as a share of it: the "
     "weak set is a connected network, not speckle."),
    ("screen", "edge_fraction_min", "float", 0.0, 1.0,
     "Weak cells with a strong neighbour, as a share of the weak set. A line "
     "w cells wide gives about 2 / w, so 0.5 means width four or less."),
    ("screen", "residual_max", "float", 1e-12, 1.0,
     "Worst solver relative residual. Below this the cell is a solve; above "
     "it the cell is invalid and is never scored."),
    ("screen", "pass_fraction", "float", 0.0, 1.0,
     "Share of a cell's worlds that must pass every term for the cell to "
     "pass."),

    ("space", "stiffness_fraction_lo", "float", 0.02, 4.0,
     "Fraction of the world over which a plate holds together: low end."),
    ("space", "stiffness_fraction_hi", "float", 0.02, 4.0,
     "High end. Sampled log-uniform."),
    ("space", "yield_percentile_lo", "float", 0.5, 50.0,
     "Percent of the first strain field above yield: low end."),
    ("space", "yield_percentile_hi", "float", 0.5, 50.0,
     "High end. Sampled log-uniform."),
    ("space", "heal_time_myr_lo", "float", 5.0, 2000.0,
     "Time for a fault to seal once it stops moving: low end."),
    ("space", "heal_time_myr_hi", "float", 5.0, 2000.0,
     "High end. Sampled log-uniform."),
    ("space", "damage_time_myr_lo", "float", 0.5, 200.0,
     "Time for intact rock at twice yield to fail: low end."),
    ("space", "damage_time_myr_hi", "float", 0.5, 200.0,
     "High end. Sampled log-uniform."),
    ("space", "strength_exponent_set", "set_int", 1, 8,
     "How steeply stiffness falls with damage. Sampled uniformly from this "
     "set."),
    ("space", "strength_spread_lo", "float", 0.0, 0.3,
     "Initial heterogeneity of the lithosphere: low end."),
    ("space", "strength_spread_hi", "float", 0.0, 0.3,
     "High end. Sampled uniformly."),
    ("space", "drive_wavelength_km_lo", "float", 640.0, 40960.0,
     "Coarsest mantle wavelength in kilometres, the same at every resolution "
     "and scale: low end. It sets how many mantle cells the world holds and "
     "so how many plates can form."),
    ("space", "drive_wavelength_km_hi", "float", 640.0, 40960.0,
     "High end, in kilometres. Sampled log-uniform. 5,120 km is two cells "
     "across the default 1024-px world."),
    ("space", "drive_shear_lo", "float", 0.0, 2.0,
     "Rotational drive relative to pushing drive: low end."),
    ("space", "drive_shear_hi", "float", 0.0, 2.0,
     "High end. Sampled uniformly."),
    ("space", "crack_speed_km_per_myr_lo", "float", 0.0, 400.0,
     "How fast a crack tip runs, in kilometres per million years: low end. "
     "Read only under the seam formulation."),
    ("space", "crack_speed_km_per_myr_hi", "float", 0.0, 400.0,
     "High end. Sampled log-uniform. A rift propagates at tens of kilometres "
     "per million years."),
    ("space", "nucleations_per_step_set", "set_int", 0, 20,
     "New cracks per step, at the highest-stress intact cells away from any "
     "existing seam. Sampled uniformly from this set."),
    ("space", "toughness_fraction_lo", "float", 0.05, 1.0,
     "Fracture toughness as a fraction of intact strength: low end. Cracks "
     "propagate at this fraction of the stress it takes to nucleate one, for "
     "a crack one cell long; longer cracks propagate at less."),
    ("space", "toughness_fraction_hi", "float", 0.05, 1.0,
     "High end. Sampled log-uniform. 1.0 is where the tip rule sat before "
     "the toughness was a dial."),
    ("space", "pixels", "int", min(SUPPORTED_SIZES), max(SUPPORTED_SIZES),
     "Delivered resolution of every world. Fixed for the run."),
    ("space", "scale_km", "int", 5, 20,
     "Kilometres per delivered pixel. Fixed for the run."),
    ("space", "history_myr", "float", 50.0, 1000.0,
     "How long each history runs. Fixed for the run."),
    ("space", "max_cycles", "int", 5, 200,
     "Solver effort per step. Fixed for the run."),
    ("space", "work_damage", "int", 0, 1,
     "Damage law, fixed for the run and never sampled: 0 compares the strain "
     "rate with its yield, 1 compares the dissipated work, stiffness times "
     "the square of the strain rate, with the same percentile of its own "
     "field. Fixed, so the same search seed draws the same cells either way "
     "and the two runs pair."),
    ("space", "solve_divisor", "int", 1, 2,
     "Kinematic cells per solve cell, fixed for the run and never sampled: 2 "
     "solves the velocity on half the kinematic grid and lifts strain back in "
     "2 x 2 blocks, so a zone cannot be narrower than two cells, and 1 solves "
     "on the full grid at about six times the cost. Fixed, so the same search "
     "seed draws the same cells either way and the two runs pair."),
    ("space", "seams", "int", 0, 2,
     "Damage rule, fixed for the run and never sampled: 0 is the sheet, "
     "diffuse damage wherever strain exceeds yield; 1 is the seam "
     "formulation, damage only on a seam, at its tip, or at a nucleation "
     "site, so boundaries are one cell wide by construction and the width "
     "term of the screen is satisfied without searching for it; 2 is the "
     "block model, where pieces are rigid bodies, the stress the seam rules "
     "read is the integral of the drag a piece fails to match, and seams "
     "are carried on markers that cannot duplicate. Fixed, so the same "
     "search seed draws the same cells whichever it is and the runs pair."),
    ("space", "base_seed", "int", 0, UINT32,
     "First seed of a stage-1 or stage-2 cell; the rest follow it."),

    ("stages", "stage1_cells", "int", 1, 20000,
     "Latin-hypercube samples of the space in one stage-1 pass."),
    ("stages", "stage1_seeds", "int", 1, 32,
     "Worlds per stage-1 cell."),
    ("stages", "stage2_top", "int", 0, 500,
     "Best stage-1 cells by soft score carried into stage 2, on top of every "
     "passer."),
    ("stages", "stage2_perturbations", "int", 0, 100,
     "Gaussian perturbations of each stage-2 candidate."),
    ("stages", "stage2_seeds", "int", 1, 32,
     "Worlds per stage-2 cell."),
    ("stages", "stage3_top", "int", 0, 100,
     "Best stage-2 cells confirmed on the twelve development seeds when none "
     "passed."),

    ("run", "search_seed", "int", 0, UINT32,
     "The whole search is reproducible from this. A second stage-1 pass uses "
     "this plus one."),
    ("run", "window", "int", 1, 32,
     "Cells in flight at once, so results arrive continuously."),
)

#: Range pairs that must be ordered, and whether the low end must be positive
#: because the dial is sampled in the logarithm.
_RANGE_PAIRS = (
    ("stiffness_fraction", True),
    ("yield_percentile", True),
    ("heal_time_myr", True),
    ("damage_time_myr", True),
    ("strength_spread", False),
    ("drive_wavelength_km", True),
    ("drive_shear", False),
    ("crack_speed_km_per_myr", True),
    ("toughness_fraction", True),
)


class ConfigError(ValueError):
    """A knob value the search would not accept."""


def _as_number(name: str, value, kind: str):
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a number, not a boolean")
    if kind == "int":
        if isinstance(value, str):
            try:
                value = int(value.strip())
            except ValueError:
                raise ConfigError(f"{name} must be a whole number") from None
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise ConfigError(f"{name} must be a whole number")
        return value
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise ConfigError(f"{name} must be a number") from None
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    return float(value)


def _as_set(name: str, value, lo: float, hi: float) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part for part in re.split(r"[,\s]+", value.strip()) if part]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise ConfigError(f"{name} must be a list of whole numbers")
    values: list[int] = []
    for part in parts:
        number = _as_number(name, part, "int")
        if not lo <= number <= hi:
            raise ConfigError(f"{name} entries must be between "
                              f"{int(lo)} and {int(hi)}")
        if number not in values:
            values.append(number)
    if not values:
        raise ConfigError(f"{name} must name at least one value")
    return tuple(values)


def default_values() -> dict:
    screen, space, stages = search.Screen(), search.Space(), search.Stages()
    config = search.SearchConfig()
    sources = {"screen": screen, "space": space, "stages": stages,
               "run": config}
    out: dict[str, dict] = {"screen": {}, "space": {}, "stages": {}, "run": {}}
    for group, name, kind, _lo, _hi, _meaning in KNOBS:
        value = getattr(sources[group], name)
        out[group][name] = list(value) if isinstance(value, tuple) else value
    return out


def config_from(body: dict) -> search.SearchConfig:
    """One `SearchConfig` from a knob body, defaulted and validated.

    A value may be given inside its group (`{"screen": {"weak_min": …}}`) or
    at the top level; the group wins. Anything absent keeps its default.
    """
    if not isinstance(body, dict):
        raise ConfigError("the request body must be a JSON object")
    values = default_values()
    for group, name, kind, lo, hi, _meaning in KNOBS:
        group_body = body.get(group)
        if isinstance(group_body, dict) and name in group_body:
            raw = group_body[name]
        elif name in body:
            raw = body[name]
        else:
            continue
        if kind == "set_int":
            values[group][name] = list(_as_set(name, raw, lo, hi))
            continue
        number = _as_number(name, raw, kind)
        if not lo <= number <= hi:
            raise ConfigError(f"{name} must be between {lo} and {hi}")
        values[group][name] = number

    if values["space"]["pixels"] not in SUPPORTED_SIZES:
        raise ConfigError(f"pixels must be one of {list(SUPPORTED_SIZES)}")
    for name, positive in _RANGE_PAIRS:
        low = values["space"][f"{name}_lo"]
        high = values["space"][f"{name}_hi"]
        if low > high:
            raise ConfigError(f"{name}_lo must not exceed {name}_hi")
        if positive and low <= 0.0:
            raise ConfigError(f"{name}_lo must be above zero: it is sampled "
                              "log-uniform")
    if values["screen"]["weak_min"] > values["screen"]["weak_max"]:
        raise ConfigError("weak_min must not exceed weak_max")
    if values["screen"]["plates_min"] > values["screen"]["plates_max"]:
        raise ConfigError("plates_min must not exceed plates_max")

    space = search.Space(**{
        key: (tuple(value) if isinstance(value, list) else value)
        for key, value in values["space"].items()})
    try:
        return search.SearchConfig(
            screen=search.Screen(**values["screen"]),
            space=space,
            stages=search.Stages(**values["stages"]),
            search_seed=int(values["run"]["search_seed"]),
            window=int(values["run"]["window"]),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc


class SearchState:
    """The server's one run, its pool, and the run it is showing on disk."""

    def __init__(self, executor_factory=None, root: Path | None = None) -> None:
        self._executor_factory = executor_factory or self._process_pool
        self._executor = None
        self._lock = threading.Lock()
        self.root = search.runs_root() if root is None else Path(root)
        self.run: search.SearchRun | None = None
        self.thread: threading.Thread | None = None

    @staticmethod
    def _process_pool():
        # This pool is the server's own, not the lab's, so the BLAS cap the
        # lab sets in `explore_adapter._pool` never reached it: the run of
        # 2026-09-03 stayed at 3.4 cells a minute after that fix. Same cap,
        # same reason, set before the children are spawned.
        import explore_adapter
        explore_adapter.cap_blas_threads()
        return ProcessPoolExecutor(
            max_workers=POOL_WORKERS,
            mp_context=multiprocessing.get_context("spawn"))

    def executor(self):
        """The one pool, created on the first start and kept for the process."""
        if self._executor is None:
            self._executor = self._executor_factory()
        return self._executor

    @property
    def active(self) -> bool:
        return self.run is not None and not self.run.finished

    def start(self, config: search.SearchConfig) -> str:
        with self._lock:
            if self.active:
                raise RuntimeError("a run is already active")
            run_dir = self.root / search.run_id_for(config.search_seed)
            index = 1
            while run_dir.exists():
                run_dir = self.root / (
                    f"{search.run_id_for(config.search_seed)}-{index}")
                index += 1
            run = search.SearchRun(config, run_dir, self.executor())
            self.run = run
            self.thread = threading.Thread(
                target=run.run, name=f"search-{run.run_id}", daemon=True)
            self.thread.start()
            return run.run_id

    def stop(self) -> None:
        if self.run is None:
            raise RuntimeError("no run to stop")
        self.run.stop()

    def cells_of(self, run_id: str | None) -> tuple[str | None, list[dict]]:
        if self.run is not None and (run_id is None or run_id == self.run.run_id):
            return self.run.run_id, list(self.run.cells)
        if run_id is None:
            return None, []
        directory = self._run_dir(run_id)
        return run_id, search.load_cells(directory)

    def _run_dir(self, run_id: str) -> Path:
        if not _SAFE_NAME.match(run_id or ""):
            raise ConfigError("bad run id")
        return self.root / run_id


def create_app(state: SearchState | None = None) -> Flask:
    state = SearchState() if state is None else state
    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.search_state = state

    def no_cache(response):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response

    @app.get("/")
    def index():
        return no_cache(send_from_directory(str(_WEB), "index.html"))

    @app.get("/api/config")
    def api_config():
        return jsonify({
            "defaults": default_values(),
            "knobs": [
                {"group": group, "name": name, "ctype": kind,
                 "lo": lo, "hi": hi, "meaning": meaning}
                for group, name, kind, lo, hi, meaning in KNOBS
            ],
            "development_seeds": list(search.DEVELOPMENT_SEEDS),
            "cell_sheets": list(search.CELL_SHEETS),
            "stage3_sheets": list(search.ALL_SHEETS),
            "terms": [name for name, _lo, _hi
                      in search.term_bounds(search.Screen())],
            "note": (
                "Every term of the screen is a measured property of a plate "
                "regime. A cell that passes on the twelve development seeds "
                "is a finding: a candidate for the author to look at in the "
                "exploration lab, and not an approval."
            ),
        })

    @app.post("/api/start")
    def api_start():
        body = request.get_json(silent=True)
        if body is None:
            body = {}
        try:
            config = config_from(body)
        except ConfigError as exc:
            return jsonify({"error": str(exc), "code": "invalid_knob"}), 400
        try:
            run_id = state.start(config)
        except RuntimeError as exc:
            return jsonify({"error": str(exc), "code": "run_active"}), 409
        except OSError as exc:
            return jsonify({"error": f"could not start the pool: {exc}",
                            "code": "pool_unavailable"}), 503
        return jsonify({"run_id": run_id, "config": config.to_json()})

    @app.post("/api/stop")
    def api_stop():
        try:
            state.stop()
        except RuntimeError as exc:
            return jsonify({"error": str(exc), "code": "no_run"}), 409
        return jsonify({"stopping": True, "run_id": state.run.run_id})

    @app.get("/api/status")
    def api_status():
        if state.run is None:
            return jsonify({"run_id": None, "stage": "idle", "cells_done": 0,
                            "passers": 0, "invalid": 0, "cells_per_minute": 0.0,
                            "best_soft_score": None, "elapsed_s": 0.0,
                            "running": False, "stopping": False,
                            "finding": None, "findings": 0, "error": None,
                            "round": 0})
        return jsonify(state.run.status())

    @app.get("/api/cells")
    def api_cells():
        try:
            after = int(request.args.get("after", -1))
            limit = int(request.args.get("limit", _CELLS_LIMIT_DEFAULT))
        except (TypeError, ValueError):
            return jsonify({"error": "after and limit must be whole numbers",
                            "code": "invalid_query"}), 400
        limit = max(1, min(limit, _CELLS_LIMIT_MAX))
        run_id = request.args.get("run") or None
        try:
            resolved, cells = state.cells_of(run_id)
        except ConfigError as exc:
            return jsonify({"error": str(exc), "code": "invalid_run"}), 400
        keep_passed = request.args.get("keep_passed", "0") not in ("", "0", "false")
        newer = [cell for cell in cells if cell["index"] > after]
        if keep_passed:
            # Reopening a long run: every passed cell is kept whatever the
            # limit, and the newest of the rest fill what room is left, so a
            # finding is never pushed off the page by cells that failed.
            kept = [cell for cell in newer if cell["passed"]]
            room = max(0, limit - len(kept))
            rest = [cell for cell in newer if not cell["passed"]]
            selected = sorted(kept + (rest[-room:] if room else []),
                              key=lambda cell: cell["index"])
        else:
            selected = newer[:limit]
        return jsonify({"run_id": resolved, "total": len(cells),
                        "cells": selected})

    @app.get("/api/cell/<cell_id>/<sheet>.png")
    def api_cell_sheet(cell_id: str, sheet: str):
        run_id = request.args.get("run") or (
            state.run.run_id if state.run is not None else None)
        if run_id is None:
            return jsonify({"error": "no run to read", "code": "no_run"}), 404
        if not (_SAFE_NAME.match(cell_id) and _SAFE_NAME.match(sheet)):
            return jsonify({"error": "bad cell or sheet name",
                            "code": "not_found"}), 404
        try:
            path = state._run_dir(run_id) / "cells" / cell_id / f"{sheet}.png"
        except ConfigError:
            return jsonify({"error": "bad run id", "code": "not_found"}), 404
        if not path.is_file():
            return jsonify({"error": "no such sheet", "code": "not_found"}), 404
        response = Response(path.read_bytes(), mimetype="image/png")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/runs")
    def api_runs():
        return jsonify({"runs": search.list_runs(state.root),
                        "active": state.run.run_id if state.active else None})

    return app


#: The adapter whose views the search draws its sheets through. It is a
#: command-line argument rather than an import constant for one reason: the
#: shared launcher check in `prepare_webui.ps1` identifies a stale listener by
#: the exact `--backend`, `--root` and `--port` on its command line, and this
#: server is launched through that check like the other two. Naming a
#: different adapter is refused rather than silently ignored.
SHEET_BACKEND = "explore_adapter"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="pipeline_c regime search")
    parser.add_argument("--backend", default=SHEET_BACKEND,
                        help="adapter the sheets are drawn through")
    parser.add_argument("--root", default=None,
                        help="directory to prepend to sys.path before import")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("SEARCH_PORT", PORT)))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.backend != SHEET_BACKEND:
        raise SystemExit(f"--backend must be {SHEET_BACKEND}")
    if args.root:
        root = os.path.abspath(args.root)
        if root not in sys.path:
            sys.path.insert(0, root)
    app = create_app()
    # The reloader restarts this process on any edit, which would leak the
    # pool's children and orphan a running search. It stays off.
    app.run(host="127.0.0.1", port=args.port, debug=False,
            use_reloader=False, use_debugger=False)


if __name__ == "__main__":
    main()
