"""World identity and the integer geometry every field is defined on.

Scale and resolution together size the delivered window, and the simulated
parent world is sized from that window. A different resolution or a different
scale is therefore a different world, not a different view of one.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .history.constants import (
    CELL_PX,
    MIN_HISTORY_N,
    PARENT_WINDOW_RATIO,
    SCALE_MAX,
    SCALE_MIN,
    SUPPORTED_SIZES,
)

WORLD_ID_SCHEMA = "pipeline-c-world-id:v2"
SEED_MAX = 2**32 - 1


def _require_plain_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, not a bool")
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an integer, not {type(value).__name__}")
    return value


@dataclass(frozen=True, slots=True)
class WorldGeometry:
    seed: int
    pixels: int
    scale_km: int

    def __post_init__(self) -> None:
        _require_plain_int(self.seed, "seed")
        _require_plain_int(self.pixels, "pixels")
        _require_plain_int(self.scale_km, "scale_km")
        if not 0 <= self.seed <= SEED_MAX:
            raise ValueError("seed must fit in a uint32")
        if self.pixels not in SUPPORTED_SIZES:
            raise ValueError(f"pixels must be one of {SUPPORTED_SIZES}")
        if not SCALE_MIN <= self.scale_km <= SCALE_MAX:
            raise ValueError(
                f"scale_km must be between {SCALE_MIN} and {SCALE_MAX} km per pixel")

    @property
    def history_n(self) -> int:
        """Cells per axis of the periodic history grid.

        The parent is `PARENT_WINDOW_RATIO` windows wide and one cell is
        `CELL_PX` delivered pixels, so the grid is that ratio times the window
        in cells, floored at `MIN_HISTORY_N`.
        """
        return max(PARENT_WINDOW_RATIO * self.pixels // CELL_PX, MIN_HISTORY_N)

    @property
    def cell_km(self) -> int:
        """Width of one history cell in kilometres."""
        return CELL_PX * self.scale_km

    @property
    def parent_km(self) -> int:
        """Width of the periodic parent world in kilometres."""
        return self.history_n * self.cell_km

    @property
    def window_km(self) -> int:
        """Width of the delivered window in kilometres."""
        return self.pixels * self.scale_km

    @property
    def window_cells(self) -> int:
        """Width of the delivered window in history cells."""
        return self.pixels // CELL_PX

    @property
    def cell_m(self) -> int:
        return self.cell_km * 1000

    @property
    def parent_m(self) -> int:
        return self.parent_km * 1000

    @property
    def world_id(self) -> str:
        payload = {
            "pixels": self.pixels,
            "scale_km": self.scale_km,
            "schema": WORLD_ID_SCHEMA,
            "seed": self.seed,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def cell_centre_m(self, index: int) -> int:
        """Sampler address of the centre of cell `index`, exact in metres."""
        return self.cell_m * index + self.cell_m // 2

    def to_record(self) -> dict[str, object]:
        return {
            "cell_km": self.cell_km,
            "cell_m": self.cell_m,
            "history_n": self.history_n,
            "parent_km": self.parent_km,
            "parent_m": self.parent_m,
            "pixels": self.pixels,
            "scale_km": self.scale_km,
            "seed": self.seed,
            "window_cells": self.window_cells,
            "window_km": self.window_km,
            "world_id": self.world_id,
        }


__all__ = ["SEED_MAX", "WORLD_ID_SCHEMA", "WorldGeometry"]
