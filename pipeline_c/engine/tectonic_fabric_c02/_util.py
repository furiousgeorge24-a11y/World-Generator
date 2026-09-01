"""Strict immutable-record helpers for the isolated C02 engine."""

from __future__ import annotations

import math
import re
from types import MappingProxyType
from typing import Mapping

try:  # Installed package import.
    from ...artifacts import canonical_json_bytes, sha256_bytes
except ImportError:  # Shared WebUI import.
    from artifacts import canonical_json_bytes, sha256_bytes


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class FabricRecordError(ValueError):
    """A C02 record is incomplete, ambiguous, mutable, or out of range."""


class FabricFormationError(FabricRecordError):
    """One frozen, non-retryable C02 formation attempt failed."""

    def __init__(self, code: str, message: str, **details: object) -> None:
        self.code = require_id(code, "formation failure code")
        self.details = freeze_json(details, "formation failure details")
        super().__init__(message)

    def to_record(self) -> dict[str, object]:
        return {
            "code": self.code,
            "details": thaw_json(self.details),
            "message": str(self),
            "retryable": False,
        }


def require_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise FabricRecordError(f"{label} must be a safe non-empty identifier")
    return value


def require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise FabricRecordError(f"{label} must be a lowercase SHA-256 digest")
    return value


def require_int(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FabricRecordError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise FabricRecordError(f"{label} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise FabricRecordError(f"{label} must be at most {maximum}")
    return value


def require_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FabricRecordError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FabricRecordError(f"{label} must be finite")
    return result


def freeze_json(value: object, label: str = "value") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FabricRecordError(f"{label} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise FabricRecordError(f"{label} contains an invalid key")
            frozen[key] = freeze_json(item, f"{label}.{key}")
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise FabricRecordError(
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
    "FabricFormationError",
    "FabricRecordError",
    "content_sha256",
    "freeze_json",
    "require_finite",
    "require_hash",
    "require_id",
    "require_int",
    "thaw_json",
]
