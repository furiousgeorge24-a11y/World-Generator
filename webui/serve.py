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
import hashlib
import importlib
import json
import os
import sys
import time

from flask import Flask, Response, jsonify, request, send_from_directory


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True,
                    help="import path of the adapter module")
    ap.add_argument("--root", default=None,
                    help="directory to prepend to sys.path before import")
    ap.add_argument("--port", type=int, default=5000)
    return ap.parse_args()


_WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
_CACHE_CAP = 8
_HEAD_CAP = 3                        # heads are heavy
_INSPECTION_CAP = 64


def _png_data_uri(png_bytes: bytes) -> str:
    return ("data:image/png;base64,"
            + base64.b64encode(png_bytes).decode("ascii"))


def create_app(backend) -> Flask:
    """Create a preview app around one adapter module.

    Inspection is an optional, isolated capability. It never passes through
    the generator endpoint or shares generator cache entries.
    """

    app = Flask(__name__)
    # Dev preview: never let a browser hold a stale index.html. Without this
    # Flask sends no max-age and browsers apply heuristic caching, so an edited
    # front end can survive a refresh. 0 forces revalidation (cheap 304s).
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    sim_cache: dict[str, object] = {}
    head_cache: dict[str, object] = {}
    inspection_cache: dict[str, object] = {}

    def inv(name: str) -> str:
        fn = getattr(backend, "invalidation_of", None)
        return fn(name) if fn else "full"

    def sim_key(seed: int, size: int, controls: dict) -> str:
        sim = {key: value for key, value in controls.items()
               if inv(key) != "render"}
        return json.dumps([seed, size, sim], sort_keys=True)

    def head_key(seed: int, size: int, controls: dict) -> str:
        head = {key: value for key, value in controls.items()
                if inv(key) not in ("render", "late")}
        return json.dumps([seed, size, head], sort_keys=True)

    def json_object_body():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return None, (jsonify({
                "error": "request body must be a JSON object",
                "code": "invalid_json_body",
            }), 400)
        return body, None

    def adapter_error(exc: Exception, default_code: str = "invalid_request"):
        code = getattr(exc, "code", default_code)
        status = getattr(exc, "status_code", 400)
        if not isinstance(code, str) or not code:
            code = default_code
        if not isinstance(status, int) or not 400 <= status <= 599:
            status = 400
        payload = {"error": str(exc), "code": code}
        details = getattr(exc, "details", None)
        if details is None:
            details = getattr(exc, "failure_record", None)
        if isinstance(details, dict):
            try:
                json.dumps(details, allow_nan=False)
            except (TypeError, ValueError):
                pass
            else:
                payload["details"] = details
        return jsonify(payload), status

    def inspection_unavailable(meta: dict):
        status = meta.get("inspection_status")
        if not isinstance(status, str) or not status.strip():
            status = "The selected backend has no inspection laboratory."
        return jsonify({
            "error": status,
            "code": "inspection_not_ready",
            "inspect_ready": False,
        }), 503

    @app.get("/")
    def index():
        return send_from_directory(_WEB, "index.html")

    @app.get("/api/registry")
    def api_registry():
        return jsonify(backend.meta())

    @app.post("/api/generate")
    def api_generate():
        meta = backend.meta()
        if meta.get("ready", True) is False:
            status = meta.get("status")
            if not isinstance(status, str) or not status.strip():
                status = "The selected generator is not ready."
            return jsonify({
                "error": status,
                "code": "backend_not_ready",
                "ready": False,
            }), 503

        body, error = json_object_body()
        if error:
            return error
        try:
            seed = int(body.get("seed", 1))
            size = int(body.get("size", meta.get("default_size", 256)))
            controls = body.get("controls", {}) or {}
            if not isinstance(controls, dict):
                raise TypeError("controls must be a JSON object")
            view = body.get("view") or ""

            key = sim_key(seed, size, controls)
            t0 = time.perf_counter()
            world = sim_cache.get(key)
            cached = "sim" if world is not None else None
            if world is None:
                if (hasattr(backend, "generate_head")
                        and hasattr(backend, "run_tail")):
                    hkey = head_key(seed, size, controls)
                    head = head_cache.get(hkey)
                    if head is None:
                        head = backend.generate_head(seed, controls, size)
                        head_cache[hkey] = head
                        while len(head_cache) > _HEAD_CAP:
                            head_cache.pop(next(iter(head_cache)))
                    else:
                        cached = "head"
                    world = backend.run_tail(head, controls)
                else:
                    world = backend.generate(seed, controls, size)
                sim_cache[key] = world
                while len(sim_cache) > _CACHE_CAP:
                    sim_cache.pop(next(iter(sim_cache)))
            elif hasattr(backend, "set_render_controls"):
                backend.set_render_controls(world, controls)

            # Legacy generation remains eager for backward compatibility.
            views = list(backend.views(world))
            images = {
                name: _png_data_uri(backend.render_png(world, name))
                for name in views
            }
            if view not in images:
                view = views[0] if views else ""
            return jsonify({
                "images": images,
                "report": backend.report(world),
                "views": views,
                "view": view,
                "cached": bool(cached),
                "cache_level": cached,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            })
        except (TypeError, ValueError) as exc:
            return adapter_error(exc)

    @app.post("/api/inspect")
    def api_inspect():
        meta = backend.meta()
        required = ("inspect", "inspection_views", "inspection_report")
        if (meta.get("inspect_ready") is not True
                or any(not hasattr(backend, name) for name in required)):
            return inspection_unavailable(meta)
        body, error = json_object_body()
        if error:
            return error
        case_id = body.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            return jsonify({
                "error": "case_id must be a non-empty string",
                "code": "invalid_case_id",
            }), 400
        render_controls = body.get("render_controls")
        if render_controls is not None and not isinstance(render_controls, dict):
            return jsonify({
                "error": "render_controls must be a JSON object",
                "code": "invalid_render_controls",
            }), 400
        selection = body.get("selection")
        if selection is not None and not isinstance(selection, dict):
            return jsonify({
                "error": "selection must be a JSON object",
                "code": "invalid_inspection_selection",
            }), 400
        try:
            t0 = time.perf_counter()
            inspect_kwargs = {}
            if render_controls is not None:
                inspect_kwargs["render_controls"] = render_controls
            if selection is not None:
                inspect_kwargs["selection"] = selection
            packet = backend.inspect(case_id, **inspect_kwargs)
            report = backend.inspection_report(packet)
            views = list(backend.inspection_views(packet))
            if not isinstance(report, dict):
                raise TypeError("inspection_report must return an object")
            packet_id = report.get("packet_id")
            if not isinstance(packet_id, str) or not packet_id:
                raise ValueError("inspection report has no packet_id")
            inspection_cache[packet_id] = packet
            while len(inspection_cache) > _INSPECTION_CAP:
                inspection_cache.pop(next(iter(inspection_cache)))
            return jsonify({
                "packet_id": packet_id,
                "case_id": case_id,
                "views": views,
                "report": report,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            })
        except (TypeError, ValueError, KeyError) as exc:
            return adapter_error(exc, "inspection_failed")

    @app.post("/api/inspect/render")
    def api_inspect_render():
        meta = backend.meta()
        if (meta.get("inspect_ready") is not True
                or not hasattr(backend, "render_inspection_png")):
            return inspection_unavailable(meta)
        body, error = json_object_body()
        if error:
            return error
        packet_id = body.get("packet_id")
        view_id = body.get("view_id")
        role = body.get("role")
        if not all(isinstance(value, str) and value for value in (
                packet_id, view_id, role)):
            return jsonify({
                "error": "packet_id, view_id, and role are required strings",
                "code": "invalid_render_request",
            }), 400
        packet = inspection_cache.get(packet_id)
        if packet is None:
            return jsonify({
                "error": "inspection packet is not cached; inspect the case again",
                "code": "inspection_packet_not_cached",
            }), 404
        try:
            png = backend.render_inspection_png(packet, view_id, role)
            if not isinstance(png, bytes) or not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("inspection renderer did not return PNG bytes")
            response = Response(png, mimetype="image/png")
            response.headers["Cache-Control"] = "no-store"
            digest = hashlib.sha256(png).hexdigest()
            response.headers["X-Content-SHA256"] = digest
            response.set_etag(digest, weak=False)
            return response
        except (TypeError, ValueError, KeyError) as exc:
            return adapter_error(exc, "inspection_render_failed")

    @app.post("/api/inspect/promote")
    def api_inspect_promote():
        meta = backend.meta()
        if (meta.get("inspect_ready") is not True
                or not hasattr(backend, "promote_baseline")):
            return inspection_unavailable(meta)
        body, error = json_object_body()
        if error:
            return error
        if body.get("confirm") is not True:
            return jsonify({
                "error": "baseline promotion requires confirm=true",
                "code": "promotion_not_confirmed",
            }), 400
        packet_id = body.get("packet_id")
        author = body.get("author")
        question = body.get("question")
        if not all(isinstance(value, str) and value.strip() for value in (
                packet_id, author, question)):
            return jsonify({
                "error": "packet_id, author, and question are required strings",
                "code": "invalid_promotion_request",
            }), 400
        packet = inspection_cache.get(packet_id)
        if packet is None:
            return jsonify({
                "error": "inspection packet is not cached; inspect the case again",
                "code": "inspection_packet_not_cached",
            }), 404
        try:
            receipt = backend.promote_baseline(
                packet,
                author=author.strip(),
                question=question.strip(),
                confirm=True,
                expected_previous=body.get("expected_previous"),
            )
            if not isinstance(receipt, dict):
                raise TypeError("promotion receipt must be an object")
            return jsonify(receipt)
        except (TypeError, ValueError, KeyError) as exc:
            return adapter_error(exc, "baseline_promotion_failed")

    return app


def main() -> None:
    args = _parse_args()
    if args.root:
        sys.path.insert(0, os.path.abspath(args.root))
    backend = importlib.import_module(args.backend)
    app = create_app(backend)
    use_reloader = os.environ.get("WEBUI_RELOAD", "1") != "0"
    app.run(host="127.0.0.1", port=args.port, debug=False,
            use_reloader=use_reloader, use_debugger=False)


if __name__ == "__main__":
    main()
