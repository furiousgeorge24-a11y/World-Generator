"""Universal generator preview: control panel + pan/zoom viewer for any
seed-driven image generator.

    py -3.14 serve.py --backend <module> [--root <dir>] [--port 5000]

Everything shown — app name, version, controls (sliders), views, the
report — comes from the backend adapter module named on the command
line. This server and its front end contain no generator-specific
content; the adapter contract is documented in README.md next to this
file.

The server hot-reloads: editing any imported module (the backend
included) restarts it in place, and index.html is served no-cache, so
the whole dev loop is save-then-refresh. WEBUI_RELOAD=0 pins a single
stable process (timing runs, profilers). debug stays off: the
interactive debugger is a code-execution surface, and generation
failures belong in the backend's report, not on a traceback page.
"""

import argparse
import base64
import importlib
import json
import os
import sys
import time

from flask import Flask, jsonify, request, send_from_directory


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True,
                    help="import path of the adapter module")
    ap.add_argument("--root", default=None,
                    help="directory to prepend to sys.path before import")
    ap.add_argument("--port", type=int, default=5000)
    return ap.parse_args()


ARGS = _parse_args()
if ARGS.root:
    sys.path.insert(0, os.path.abspath(ARGS.root))
B = importlib.import_module(ARGS.backend)

app = Flask(__name__)
# Dev preview: never let a browser hold a stale index.html. Without this
# Flask sends no max-age and browsers apply heuristic caching, so an edited
# front end can survive a refresh. 0 forces revalidation (cheap 304s).
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
_WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

_cache: dict[str, object] = {}       # sim-key -> finished world handle
_head_cache: dict[str, object] = {}  # head-key -> reusable head snapshot
_CACHE_CAP = 8
_HEAD_CAP = 3                        # heads are heavy


def _inv(name: str) -> str:
    fn = getattr(B, "invalidation_of", None)
    return fn(name) if fn else "full"


def _sim_key(seed: int, size: int, controls: dict) -> str:
    sim = {k: v for k, v in controls.items() if _inv(k) != "render"}
    return json.dumps([seed, size, sim], sort_keys=True)


def _head_key(seed: int, size: int, controls: dict) -> str:
    head = {k: v for k, v in controls.items()
            if _inv(k) not in ("render", "late")}
    return json.dumps([seed, size, head], sort_keys=True)


def _png_data_uri(png_bytes: bytes) -> str:
    return ("data:image/png;base64,"
            + base64.b64encode(png_bytes).decode("ascii"))


@app.get("/")
def index():
    return send_from_directory(_WEB, "index.html")


@app.get("/api/registry")
def api_registry():
    return jsonify(B.meta())


@app.post("/api/generate")
def api_generate():
    body = request.get_json(force=True)
    seed = int(body.get("seed", 1))
    size = int(body.get("size", 256))
    controls = body.get("controls", {}) or {}
    view = body.get("view") or ""

    key = _sim_key(seed, size, controls)
    t0 = time.perf_counter()
    world = _cache.get(key)
    cached = "sim" if world is not None else None
    if world is None:
        if hasattr(B, "generate_head") and hasattr(B, "run_tail"):
            # late-class path: re-run only the cheap tail on a cached head
            hkey = _head_key(seed, size, controls)
            head = _head_cache.get(hkey)
            if head is None:
                head = B.generate_head(seed, controls, size)
                _head_cache[hkey] = head
                while len(_head_cache) > _HEAD_CAP:
                    _head_cache.pop(next(iter(_head_cache)))
            else:
                cached = "head"
            world = B.run_tail(head, controls)
        else:
            world = B.generate(seed, controls, size)
        _cache[key] = world
        while len(_cache) > _CACHE_CAP:
            _cache.pop(next(iter(_cache)))
    elif hasattr(B, "set_render_controls"):
        # render-class controls may have changed; refresh them in place
        B.set_render_controls(world, controls)

    # Render every available view: the client keeps a whole generation, so
    # swapping views is a client-side image swap, never a round trip.
    views = list(B.views(world))
    images = {name: _png_data_uri(B.render_png(world, name))
              for name in views}
    if view not in images:
        view = views[0] if views else ""
    return jsonify({
        "images": images,
        "report": B.report(world),
        "views": views,
        "view": view,
        "cached": bool(cached),
        "cache_level": cached,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    })


def main() -> None:
    use_reloader = os.environ.get("WEBUI_RELOAD", "1") != "0"
    app.run(host="127.0.0.1", port=ARGS.port, debug=False,
            use_reloader=use_reloader, use_debugger=False)


if __name__ == "__main__":
    main()
