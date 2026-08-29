"""Preview webui: sliders generated from the control registry.

    py -3.14 -m mapgen.webui        ->  http://127.0.0.1:5000

Render-class controls hit a World cache (no re-simulation); everything
else re-generates on demand. The registry is the single source of truth —
this file never needs editing when a control is added.

The server hot-reloads: editing any mapgen module restarts it in place, and
index.html is served no-cache, so the whole dev loop is save-then-refresh.
"""

import base64
import io
import json
import os
import time

from flask import Flask, jsonify, request, send_from_directory

from . import VERSION, pipeline, registry, render, report

app = Flask(__name__)
# Dev preview: never let a browser hold a stale index.html. Without this Flask
# sends no max-age and browsers apply heuristic caching, so an edited front end
# can survive a refresh. 0 forces revalidation (still a cheap 304 when unchanged).
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
_WEB = os.path.join(os.path.dirname(__file__), "web")

_cache: dict[str, object] = {}       # sim-key -> finished World
_head_cache: dict[str, object] = {}  # head-key -> pre-tail World snapshot
_CACHE_CAP = 8
_HEAD_CAP = 3                        # heads are heavy (all layers)


def _sim_key(seed: int, size: int, controls: dict) -> str:
    sim = {k: v for k, v in controls.items()
           if registry.invalidation_of(k) != registry.RENDER}
    return json.dumps([seed, size, sim], sort_keys=True)


def _head_key(seed: int, size: int, controls: dict) -> str:
    head = {k: v for k, v in controls.items()
            if registry.invalidation_of(k)
            not in (registry.RENDER, registry.LATE)}
    return json.dumps([seed, size, head], sort_keys=True)


def _png_data_uri(img, world) -> str:
    buf = io.BytesIO()
    render.save_png(img, buf, world)
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


@app.get("/")
def index():
    return send_from_directory(_WEB, "index.html")


@app.get("/api/registry")
def api_registry():
    return jsonify({"version": VERSION, "controls": registry.as_dicts()})


@app.post("/api/generate")
def api_generate():
    body = request.get_json(force=True)
    seed = int(body.get("seed", 1))
    size = int(body.get("size", 256))
    controls = body.get("controls", {}) or {}
    view = body.get("view") or "hypsometric"

    key = _sim_key(seed, size, controls)
    t0 = time.perf_counter()
    world = _cache.get(key)
    cached = "sim" if world is not None else None
    if world is None:
        # late-class path: re-run only the cheap tail against a cached head
        hkey = _head_key(seed, size, controls)
        head = _head_cache.get(hkey)
        if head is None:
            head = pipeline.generate_head(seed, controls, size)
            _head_cache[hkey] = head
            while len(_head_cache) > _HEAD_CAP:
                _head_cache.pop(next(iter(_head_cache)))
        else:
            cached = "head"
        world = pipeline.clone_world(head)
        vals, _ = registry.resolve(controls)
        world.controls = vals
        pipeline.run_tail(world)
        _cache[key] = world
        while len(_cache) > _CACHE_CAP:
            _cache.pop(next(iter(_cache)))
    else:
        # render-class controls may have changed; refresh them on the World
        vals, _ = registry.resolve(controls)
        for name in vals:
            if registry.invalidation_of(name) == registry.RENDER:
                world.controls[name] = vals[name]

    # Render every available view: the client keeps a whole generation, so
    # swapping views is a client-side image swap, never a round trip.
    views = render.available_views(world)
    images = {name: _png_data_uri(render.render_view(world, name), world)
              for name in views}
    if view not in images:
        view = views[0] if views else "hypsometric"
    return jsonify({
        "images": images,
        "report": report.build(world),
        "views": views,
        "view": view,
        "cached": bool(cached),
        "cache_level": cached,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    })


def main() -> None:
    # The reloader stats every imported mapgen module once a second and restarts
    # the server in place when one changes: edit -> save -> refresh the browser.
    # The World cache dies with the old process, which is the point — a World
    # simulated by pre-edit code would be a lie. index.html is not watched; it is
    # read from disk per request, so HTML edits need no restart at all.
    # MAPGEN_RELOAD=0 pins a single stable process (timing runs, profilers).
    # debug stays off: the interactive debugger is a code-execution surface, and
    # generation failures belong in the report sidecar, not on a traceback page.
    use_reloader = os.environ.get("MAPGEN_RELOAD", "1") != "0"
    app.run(host="127.0.0.1", port=5000, debug=False,
            use_reloader=use_reloader, use_debugger=False)


if __name__ == "__main__":
    main()
