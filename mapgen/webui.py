"""Preview webui: sliders generated from the control registry.

    py -3.14 -m mapgen.webui        ->  http://127.0.0.1:5000

Render-class controls hit a World cache (no re-simulation); everything
else re-generates on demand. The registry is the single source of truth —
this file never needs editing when a control is added.
"""

import base64
import io
import json
import os
import time

from flask import Flask, jsonify, request, send_from_directory

from . import VERSION, pipeline, registry, render, report

app = Flask(__name__)
_WEB = os.path.join(os.path.dirname(__file__), "web")

_cache: dict[str, object] = {}  # sim-key -> World, insertion-ordered
_CACHE_CAP = 8


def _sim_key(seed: int, size: int, controls: dict) -> str:
    sim = {k: v for k, v in controls.items()
           if registry.invalidation_of(k) != registry.RENDER}
    return json.dumps([seed, size, sim], sort_keys=True)


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
    cached = world is not None
    if not cached:
        world = pipeline.generate(seed, controls, size)
        _cache[key] = world
        while len(_cache) > _CACHE_CAP:
            _cache.pop(next(iter(_cache)))
    else:
        # render-class controls may have changed; refresh them on the World
        vals, _ = registry.resolve(controls)
        for name in vals:
            if registry.invalidation_of(name) == registry.RENDER:
                world.controls[name] = vals[name]

    img = render.render_view(world, view)
    buf = io.BytesIO()
    render.save_png(img, buf, world)
    return jsonify({
        "image": "data:image/png;base64,"
                 + base64.b64encode(buf.getvalue()).decode("ascii"),
        "report": report.build(world),
        "views": render.available_views(world),
        "view": view,
        "cached": cached,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    })


def main() -> None:
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
