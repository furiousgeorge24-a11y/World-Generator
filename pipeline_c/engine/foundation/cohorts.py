"""Frozen C4 seed cohorts and validation-access guard."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ._util import FoundationRecordError, content_sha256, require_int
from .constants import COHORT_SCHEMA_ID, ROADMAP_RUN


DEBUG_SEEDS = (2849919200, 3534786579, 3413470326, 1140583516)
DEVELOPMENT_SEEDS = (
    2075014389,
    2477733044,
    476149591,
    151640007,
    2697441485,
    1504571935,
    548870008,
    2157195430,
    4108373596,
    4287772760,
    287488203,
    1833546021,
)
VALIDATION_SEEDS = (
    2791121701,
    3130115100,
    1455726917,
    1973562563,
    1338344000,
    1420467597,
    4162438652,
    1437289707,
    4007147443,
    1674537276,
    3118573802,
    3603308068,
    3071127819,
    829628780,
    1240803007,
    2251729814,
    3107823350,
    3992594243,
    3806252049,
    3159806290,
    1664116445,
    3232083246,
    3563470347,
    3886709965,
    1749459072,
    2995727652,
    4009585040,
    1292709806,
    3305252046,
    2145001681,
    1578577612,
    3129225758,
)
COHORT_MANIFEST_SHA256 = "a97323bceead5f55a6870354256f4279dfd4ca6d939df2728876d7ef3da4382a"
_RUN = re.compile(r"^C([0-9]+)$")


class ValidationAccessError(PermissionError):
    """A validation execution was attempted before frozen Run C15."""

    code = "VALIDATION_COHORT_SEALED_UNTIL_C15"


def derive_seed(role: str, index: int) -> int:
    if role not in {"debug", "development", "validation"}:
        raise FoundationRecordError("cohort role must be debug, development, or validation")
    require_int(index, "index", minimum=0, maximum=999)
    material = f"pipeline-c/cohort-v1/{role}/{index:03d}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[0:4], "big", signed=False)


@dataclass(frozen=True, slots=True)
class CohortManifest:
    debug: tuple[int, ...] = DEBUG_SEEDS
    development: tuple[int, ...] = DEVELOPMENT_SEEDS
    validation: tuple[int, ...] = VALIDATION_SEEDS

    def __post_init__(self) -> None:
        expected_lengths = {"debug": 4, "development": 12, "validation": 32}
        all_values: list[int] = []
        for role, expected_length in expected_lengths.items():
            values = getattr(self, role)
            if not isinstance(values, tuple) or len(values) != expected_length:
                raise FoundationRecordError(
                    f"{role} cohort must contain exactly {expected_length} seeds"
                )
            for index, seed in enumerate(values):
                require_int(seed, f"{role}[{index}]", minimum=0, maximum=2**32 - 1)
                if seed != derive_seed(role, index):
                    raise FoundationRecordError(
                        f"{role}[{index}] does not match frozen derivation"
                    )
            all_values.extend(values)
        if len(set(all_values)) != 48:
            raise FoundationRecordError("all 48 cohort identities must be distinct")
        if content_sha256(self.to_record()) != COHORT_MANIFEST_SHA256:
            raise FoundationRecordError("cohort manifest hash differs from C4 precommit")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_id": COHORT_SCHEMA_ID,
            "schema_version": 1,
            "seeds": {
                "debug": list(self.debug),
                "development": list(self.development),
                "validation": list(self.validation),
            },
        }

    @property
    def sha256(self) -> str:
        return content_sha256(self.to_record())


COHORT_MANIFEST = CohortManifest()


@dataclass(frozen=True, slots=True)
class ExecutionSeed:
    role: str
    index: int
    seed: int

    def __post_init__(self) -> None:
        if self.role not in {"debug", "development", "validation"}:
            raise FoundationRecordError("invalid execution-seed role")
        require_int(self.index, "index", minimum=0)
        require_int(self.seed, "seed", minimum=0, maximum=2**32 - 1)

    def to_record(self) -> dict[str, object]:
        return {"index": self.index, "role": self.role, "seed": self.seed}


def _validation_is_open(roadmap_run: str) -> bool:
    if not isinstance(roadmap_run, str):
        return False
    match = _RUN.fullmatch(roadmap_run)
    return match is not None and int(match.group(1)) >= 15


def seed_for_execution(
    role: str,
    index: int,
    *,
    roadmap_run: str = ROADMAP_RUN,
) -> ExecutionSeed:
    """Authorize then return a seed; validation is rejected before lookup."""

    if role == "validation" and not _validation_is_open(roadmap_run):
        raise ValidationAccessError(
            f"validation execution is sealed before C15 (requested {roadmap_run!r})"
        )
    if role not in {"debug", "development", "validation"}:
        raise FoundationRecordError("cohort role must be debug, development, or validation")
    require_int(index, "index", minimum=0)
    values = {
        "debug": COHORT_MANIFEST.debug,
        "development": COHORT_MANIFEST.development,
        "validation": COHORT_MANIFEST.validation,
    }[role]
    if index >= len(values):
        raise FoundationRecordError(f"{role} cohort index is out of range")
    return ExecutionSeed(role=role, index=index, seed=values[index])


def development_execution_plan() -> tuple[ExecutionSeed, ...]:
    return tuple(
        seed_for_execution("development", index)
        for index in range(len(DEVELOPMENT_SEEDS))
    )


def debug_execution_plan() -> tuple[ExecutionSeed, ...]:
    return tuple(
        seed_for_execution("debug", index)
        for index in range(len(DEBUG_SEEDS))
    )


__all__ = [
    "COHORT_MANIFEST",
    "COHORT_MANIFEST_SHA256",
    "CohortManifest",
    "DEBUG_SEEDS",
    "DEVELOPMENT_SEEDS",
    "ExecutionSeed",
    "VALIDATION_SEEDS",
    "ValidationAccessError",
    "debug_execution_plan",
    "derive_seed",
    "development_execution_plan",
    "seed_for_execution",
]
