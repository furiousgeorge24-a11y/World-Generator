"""Frozen Pipeline C controls and their WebUI metadata.

This module intentionally contains no formation logic.  It gives the shared
WebUI and future engine one canonical control vocabulary while the engine is
still unavailable.
"""

from __future__ import annotations

import math

from . import VERSION


# No author control is advertised yet: the WebUI offers only what actually
# works. `target_land_percent` (C11) and `landmass_fragmentation` (C13) are
# promised in CONTRACT.md and are added here when their causal stages exist.
CONTROLS: tuple[dict, ...] = ()

DEFAULT_SIZE = 1024
SUPPORTED_SIZES = (512, 1024)


def meta() -> dict:
    """Return the control vocabulary and geometry the adapter advertises."""

    return {
        "name": "pipeline_c land-origin lab",
        "version": VERSION,
        "controls": [dict(control) for control in CONTROLS],
        "default_size": DEFAULT_SIZE,
        "supported_sizes": list(SUPPORTED_SIZES),
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


def normalize_size(size: object = None) -> int:
    """Return a frozen C4 square sampling size, defaulting to 1024."""

    if size is None:
        return DEFAULT_SIZE
    if isinstance(size, bool) or not isinstance(size, int):
        raise TypeError("size must be a positive integer")
    if size not in SUPPORTED_SIZES:
        raise ValueError(
            "size must be one of " + ", ".join(str(value) for value in SUPPORTED_SIZES)
        )
    return size
