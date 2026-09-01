"""Frozen Run 1 controls and their WebUI metadata.

This module intentionally contains no formation logic.  It gives the shared
WebUI and future engine one canonical control vocabulary while the engine is
still unavailable.
"""

from __future__ import annotations

import math

from . import VERSION


CONTROLS = (
    {
        "name": "target_land_percent",
        "ctype": "float",
        "default": 35.0,
        "lo": 0.0,
        "hi": 70.0,
        "tier": "primary",
        "invalidates": "full",
        "promise": (
            "requested final dry-land percentage; each delivered map must be "
            "within 10 percentage points"
        ),
    },
    {
        "name": "landmass_fragmentation",
        "ctype": "float",
        "default": 0.5,
        "lo": 0.0,
        "hi": 1.0,
        "tier": "primary",
        "invalidates": "full",
        "promise": (
            "0 strongly favors one dominant macro-landmass while allowing "
            "minor natural islands; higher values bias more separated major "
            "landmasses without specifying a count"
        ),
    },
)


def meta() -> dict:
    """Return adapter metadata without implying that generation exists."""

    return {
        "name": "pipeline_c land-origin lab",
        "version": VERSION,
        "controls": [dict(control) for control in CONTROLS],
        "ready": False,
        "status": (
            "Run 1 bootstrap only: the land-origin engine has not been "
            "implemented, so generation is unavailable."
        ),
    }


def normalize_controls(controls: dict | None = None) -> dict[str, float]:
    """Return a complete, finite, in-range control dictionary.

    Values outside the contract are rejected rather than silently clamped;
    this keeps UI echoes and later evaluation provenance honest.
    """

    supplied = {} if controls is None else controls
    if not isinstance(supplied, dict):
        raise TypeError("controls must be a dictionary")
    if any(not isinstance(name, str) for name in supplied):
        raise TypeError("control names must be strings")

    known = {control["name"] for control in CONTROLS}
    unknown = sorted(set(supplied) - known)
    if unknown:
        raise ValueError(f"unknown control(s): {', '.join(unknown)}")

    normalized: dict[str, float] = {}
    for control in CONTROLS:
        name = control["name"]
        raw = supplied.get(name, control["default"])
        if isinstance(raw, bool):
            raise TypeError(f"{name} must be a real number, not bool")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real number") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < control["lo"] or value > control["hi"]:
            raise ValueError(
                f"{name} must be in [{control['lo']}, {control['hi']}]"
            )
        normalized[name] = value
    return normalized


def effective_controls(controls: dict | None = None) -> dict[str, float]:
    """Alias used by adapters and provenance writers."""

    return normalize_controls(controls)
