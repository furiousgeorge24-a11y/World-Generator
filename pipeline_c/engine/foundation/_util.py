"""Small strict-record helpers for the production C4 foundation.

This module deliberately depends only on the engine-independent artifact
primitives.  It does not import the C3 laboratory or any presentation code.
"""

from __future__ import annotations

import math
import re
from types import MappingProxyType
from typing import Mapping

try:  # Package import: pipeline_c.engine.foundation
    from ...artifacts import canonical_json_bytes, sha256_bytes
except ImportError:  # Shared-WebUI import: engine.foundation
    from artifacts import canonical_json_bytes, sha256_bytes


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class FoundationRecordError(ValueError):
    """A C4 production record is incomplete, ambiguous, or out of range."""


def require_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise FoundationRecordError(f"{label} must be a safe non-empty identifier")
    return value


def require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FoundationRecordError(f"{label} must be non-empty text")
    return value


def require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise FoundationRecordError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_int(value: object, label: str, *, minimum: int | None = None,
                maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FoundationRecordError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise FoundationRecordError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise FoundationRecordError(f"{label} must be at most {maximum}")
    return value


def require_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FoundationRecordError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FoundationRecordError(f"{label} must be finite")
    return result


def freeze_json(value: object, label: str = "value") -> object:
    """Validate JSON data and recursively make containers immutable."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FoundationRecordError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise FoundationRecordError(f"{label} contains an invalid key")
            frozen[key] = freeze_json(item, f"{label}.{key}")
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise FoundationRecordError(
        f"{label} contains non-JSON value {type(value).__name__}"
    )


def thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def content_sha256(record: object) -> str:
    return sha256_bytes(canonical_json_bytes(record))


__all__ = [
    "FoundationRecordError",
    "content_sha256",
    "freeze_json",
    "require_finite",
    "require_hash",
    "require_id",
    "require_int",
    "require_text",
    "thaw_json",
]
