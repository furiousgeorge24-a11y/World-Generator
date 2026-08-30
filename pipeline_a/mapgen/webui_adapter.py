"""Adapter binding this pipeline to the shared preview webui.

Implements the contract in webui/README.md (repo root). The shared
server imports only this module; everything pipeline-specific stays
behind it. Launch from pipeline_a/:

    py -3.14 ..\\webui\\serve.py --backend mapgen.webui_adapter --root .

(or run.bat, which does exactly that and opens the browser).
"""

import io

from . import VERSION, pipeline, registry, render
from . import report as report_mod
from .world import World


def meta() -> dict:
    return {"name": "mapgen", "version": VERSION,
            "controls": registry.as_dicts()}


def invalidation_of(name: str) -> str:
    return registry.invalidation_of(name)     # already "full"/"late"/"render"


def generate(seed: int, controls: dict, size: int) -> World:
    return pipeline.generate(seed, controls, size)


def generate_head(seed: int, controls: dict, size: int) -> World:
    return pipeline.generate_head(seed, controls, size)


def run_tail(head: World, controls: dict) -> World:
    world = pipeline.clone_world(head)        # the head stays reusable
    vals, _ = registry.resolve(controls)
    world.controls = vals
    pipeline.run_tail(world)
    return world


def set_render_controls(world: World, controls: dict) -> None:
    vals, _ = registry.resolve(controls)
    for name in vals:
        if registry.invalidation_of(name) == registry.RENDER:
            world.controls[name] = vals[name]


def views(world: World) -> list[str]:
    return render.available_views(world)


def render_png(world: World, view: str) -> bytes:
    buf = io.BytesIO()
    render.save_png(render.render_view(world, view), buf, world)
    return buf.getvalue()


def report(world: World) -> dict:
    return report_mod.build(world)
