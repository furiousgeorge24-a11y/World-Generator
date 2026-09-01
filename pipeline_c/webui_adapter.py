"""Generate-and-view adapter for the shared WebUI.

Pick a seed, generate, look at the result. There is no review packet,
baseline, delta, snapshot, or approval workflow here: the engine is
deterministic, so any world can be rebuilt from its seed on demand.

What this generates is the *tectonic fabric* stage — one categorical
affiliation field over a flat-torus parent world. It is not a map. There is
no elevation, water, coastline, island, or land anywhere in it yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import time

import numpy as np
from PIL import Image

if __package__:
    from .engine import registry
    from .engine.foundation import (
        ExecutionSeed,
        PARENT_RECT,
        build_identity_bundle,
    )
    from .engine.tectonic_fabric_c02 import (
        build_tectonic_fabric_state,
        resistance_array,
    )
    from .eval.palette import CATEGORY_COLORS, categorical_rgb, scalar_rgb
else:  # The shared WebUI puts pipeline_c directly on sys.path.
    from engine import registry
    from engine.foundation import ExecutionSeed, PARENT_RECT, build_identity_bundle
    from engine.tectonic_fabric_c02 import (
        build_tectonic_fabric_state,
        resistance_array,
    )
    from eval.palette import CATEGORY_COLORS, categorical_rgb, scalar_rgb


CANONICAL_SIZE = 1024

VIEWS = ("affiliation", "arrival", "resistance")

_VIEW_PURPOSE = {
    "affiliation": "Which primary actor owns each cell of the parent world.",
    "arrival": (
        "When each cell was claimed during competitive growth. Banding here "
        "means the growth is axis-locked."
    ),
    "resistance": (
        "The low-frequency resistance field that growth had to push through."
    ),
}


@dataclass(slots=True)
class World:
    seed: int
    size: int
    affiliation: np.ndarray
    arrival: np.ndarray
    resistance: np.ndarray
    family_id: int
    shares: tuple[float, ...]
    elapsed_s: float
    world_id: str
    affiliation_sha256: str


def meta() -> dict:
    result = registry.meta()
    result.update(
        {
            "ready": True,
            "stage": "tectonic_fabric.v2",
            "status": (
                "Tectonic fabric only: categorical actor affiliation over the "
                "parent world. No elevation, water, coastline, or land."
            ),
            "views": list(VIEWS),
            "view_purposes": dict(_VIEW_PURPOSE),
        }
    )
    return result


def generate(seed: int, controls: dict | None = None, size: int | None = None) -> World:
    """Build one tectonic fabric for `seed`. Deterministic and self-contained."""

    seed = int(seed)
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must fit in a uint32")
    registry.normalize_controls(controls)
    out_size = registry.normalize_size(size)

    started = time.perf_counter()
    state = build_tectonic_fabric_state(
        ExecutionSeed(role="development", index=0, seed=seed),
        build_identity_bundle(seed),
        PARENT_RECT,
    )
    elapsed = time.perf_counter() - started

    shape = (CANONICAL_SIZE, CANONICAL_SIZE)
    affiliation = np.frombuffer(state.affiliation_bytes, dtype=np.uint8).reshape(shape)
    arrival = np.frombuffer(state.arrival_times_bytes, dtype="<i4").reshape(shape)
    counts = np.bincount(affiliation.reshape(-1), minlength=len(CATEGORY_COLORS))

    return World(
        seed=seed,
        size=out_size,
        affiliation=affiliation,
        arrival=arrival,
        resistance=resistance_array(state.context.world_id),
        family_id=int(state.controls.family_id),
        shares=tuple(round(100 * c / affiliation.size, 4) for c in counts),
        elapsed_s=round(elapsed, 3),
        world_id=state.context.world_id,
        affiliation_sha256=state.certificate.affiliation_sha256,
    )


def views(world: World) -> list[str]:
    return list(VIEWS)


def render_png(world: World, view: str) -> bytes:
    if view not in VIEWS:
        raise ValueError(f"unknown view: {view!r}")
    if view == "affiliation":
        rgb = categorical_rgb(world.affiliation)
    else:
        rgb = scalar_rgb(world.arrival if view == "arrival" else world.resistance)

    image = Image.fromarray(rgb, mode="RGB")
    # Canonical row zero is the parent's minimum-y row; display north up.
    image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if world.size != CANONICAL_SIZE:
        image = image.resize((world.size, world.size), Image.Resampling.NEAREST)

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def report(world: World) -> dict:
    ranked = sorted(world.shares, reverse=True)
    return {
        "seed": world.seed,
        "stage": "tectonic_fabric.v2",
        "layout_family": ("scatter", "belt", "dual_focus", "arc_void")[world.family_id],
        "actors": len(world.shares),
        "actor_area_percent": ranked,
        "largest_actor_percent": ranked[0],
        "hierarchy_ratio": round(ranked[0] / ranked[-1], 3),
        "canonical_lattice": f"{CANONICAL_SIZE}x{CANONICAL_SIZE} flat torus",
        "delivered_size": world.size,
        "generation_seconds": world.elapsed_s,
        "world_id": world.world_id,
        "affiliation_sha256": world.affiliation_sha256,
        "contains": "categorical actor affiliation only",
        "does_not_contain": "elevation, water, coastline, islands, land, or a map",
    }


__all__ = [
    "World",
    "generate",
    "meta",
    "render_png",
    "report",
    "views",
]
