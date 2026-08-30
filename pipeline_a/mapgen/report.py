"""Report sidecar (contract section 8): every run explains itself."""

import json

from .world import World


def build(world: World) -> dict:
    return {
        "version": world.version,
        "seed": world.seed,
        "size": {"h": world.shape[0], "w": world.shape[1]},
        "cell_size_km": world.cell_km,
        "extent_km": {"h": world.extent_km[0], "w": world.extent_km[1]},
        "controls": world.controls,
        "timings_s": world.timings,
        "findings": world.findings,
    }


def write(world: World, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build(world), f, indent=2, sort_keys=True)
