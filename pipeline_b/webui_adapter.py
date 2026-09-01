"""webui adapter for pipeline_b — M3 surface-process slice.

Implements the adapter contract in webui/README.md against the real
engine, including the caching tiers: structure + coarse elevation form
the cached HEAD (full-class controls), the erosion/surface stages are
the LATE tail (erosion-class controls rerun only that), and river
prominence is RENDER-class (re-renders a cached world in
milliseconds). Structure runs at the coarse km lattice and erosion at
the fixed process grid regardless of requested size (§2); only the
detail sampling touches output resolution.
"""

import io
import json
import time

from PIL import PngImagePlugin

from engine import VERSION
from engine import registry
from engine.elevation import coarse_elevation
from engine.erosion import run_erosion
from engine.registry import CONTROLS, make_config
from engine.render_map import MAP_VIEWS, render_map_view
from engine.render_structure import VIEWS as STRUCT_VIEWS, render_view
from engine.report import map_report
from engine.surface import sample_map
from engine.tectonics import build_structure

_INVALIDATION = {c["name"]: c["invalidates"] for c in CONTROLS}


def meta():
    return registry.meta()


def generate_head(seed, controls, size):
    t0 = time.perf_counter()
    effective = registry.effective_controls(controls)
    cfg = make_config(effective)
    s = build_structure(int(seed), cfg)
    ce = coarse_elevation(s, cfg, int(seed))
    return {
        "structure": s,
        "coarse": ce,
        "seed": int(seed),
        "size": int(size),
        "controls": effective,
        "head_s": time.perf_counter() - t0,
    }


def run_tail(head, controls):
    t0 = time.perf_counter()
    merged = dict(head.get("controls", {}))
    # A cached head already embodies every full-tier control. Applying a
    # changed full value only to its provenance would make the echo lie;
    # the cache owner must rebuild the head for those invalidations.
    for name, value in (controls or {}).items():
        if _INVALIDATION.get(name) in ("late", "render"):
            merged[name] = value
    effective = registry.effective_controls(merged)
    cfg = make_config(effective)
    s = head["structure"]
    ce = head["coarse"]
    seed = head["seed"]
    size = head["size"]
    er = run_erosion(s, ce, cfg, seed)
    m = sample_map(s, ce, er, cfg, seed, size)
    return {
        "structure": s,
        "coarse": ce,
        "erosion": er,
        "map": m,
        "seed": seed,
        "controls": effective,
        "size": size,
        "river_density": float(cfg.river_density),
        "elapsed_s": head["head_s"] + (time.perf_counter() - t0),
    }


def generate(seed, controls, size):
    return run_tail(generate_head(seed, controls, size), controls)


def set_render_controls(world, controls):
    merged = dict(world.get("controls", {}))
    for name, value in (controls or {}).items():
        if _INVALIDATION.get(name) == "render":
            merged[name] = value
    effective = registry.effective_controls(merged)
    cfg = make_config(effective)
    world["controls"] = effective
    world["river_density"] = float(cfg.river_density)


def views(world):
    return MAP_VIEWS + STRUCT_VIEWS


def render_png(world, view):
    if view in MAP_VIEWS:
        im = render_map_view(world["map"], view,
                             river_density=world["river_density"])
    else:
        im = render_view(world["structure"], view, world["size"])
    info = PngImagePlugin.PngInfo()
    info.add_text("pipeline_b:seed", str(world["seed"]))
    info.add_text("pipeline_b:controls", json.dumps(world["controls"],
                                                    sort_keys=True))
    info.add_text("pipeline_b:version", VERSION)
    info.add_text("pipeline_b:stage", "map")
    info.add_text("pipeline_b:view", view)
    buf = io.BytesIO()
    im.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def report(world):
    return map_report(world["structure"], world["coarse"],
                      world["erosion"], world["map"], world["seed"],
                      world["controls"], world["elapsed_s"])


def invalidation_of(name):
    return _INVALIDATION.get(name, "full")
