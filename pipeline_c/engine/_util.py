"""Strict-record helpers shared by the engine.

Small, dependency-free validation used by the sampler and the geometry
record. It knows nothing about fields, grids, or history.
"""

from __future__ import annotations

import re


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class EngineRecordError(ValueError):
    """An engine record is incomplete, ambiguous, or out of range."""


def require_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise EngineRecordError(f"{label} must be a safe non-empty identifier")
    return value


def require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise EngineRecordError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_int(value: object, label: str, *, minimum: int | None = None,
                maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EngineRecordError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise EngineRecordError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise EngineRecordError(f"{label} must be at most {maximum}")
    return value


__all__ = [
    "EngineRecordError",
    "require_hash",
    "require_id",
    "require_int",
]
