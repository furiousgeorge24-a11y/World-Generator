"""Local label-invariant partition diversity for the C02 cohort."""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass

import numpy as np

from ._util import (
    FabricRecordError,
    content_sha256,
    require_finite,
    require_hash,
    require_int,
)
from .constants import (
    DIVERSITY_SCHEMA_ID,
    MAX_ADJUSTED_RAND,
    MIN_PAIR_DISAGREEMENT,
    PRIMARY_ACTOR_COUNT,
)
from .observation import FabricObservation


def canonical_partition_bytes(observation: FabricObservation) -> bytes:
    if not isinstance(observation, FabricObservation):
        raise TypeError("observation must be FabricObservation")
    mapping: dict[int, int] = {}
    next_slot = 0
    result = bytearray(len(observation.actor_slots_bytes))
    for index, value in enumerate(observation.actor_slots_bytes):
        if value not in mapping:
            mapping[value] = next_slot
            next_slot += 1
        result[index] = mapping[value]
    return bytes(result)


def canonical_partition_fingerprint(observation: FabricObservation) -> str:
    return hashlib.sha256(canonical_partition_bytes(observation)).hexdigest()


def contingency_table(
    left: FabricObservation,
    right: FabricObservation,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(left, FabricObservation) or not isinstance(right, FabricObservation):
        raise TypeError("left and right must be FabricObservation")
    if left.grid != right.grid:
        raise FabricRecordError("diversity observations must share one registered grid")
    first = left.slots_array().ravel()
    second = right.slots_array().ravel()
    codes = first.astype(np.int16) * PRIMARY_ACTOR_COUNT + second.astype(np.int16)
    counts = np.bincount(codes, minlength=PRIMARY_ACTOR_COUNT**2).reshape(
        PRIMARY_ACTOR_COUNT, PRIMARY_ACTOR_COUNT
    )
    return tuple(tuple(int(value) for value in row) for row in counts)


def optimal_label_agreement_from_table(
    table: tuple[tuple[int, ...], ...],
) -> tuple[int, tuple[int, ...]]:
    if (
        not isinstance(table, tuple)
        or len(table) != PRIMARY_ACTOR_COUNT
        or any(not isinstance(row, tuple) or len(row) != PRIMARY_ACTOR_COUNT for row in table)
    ):
        raise FabricRecordError("contingency table must be exact 7x7 tuples")
    best_count = -1
    best_permutation: tuple[int, ...] | None = None
    for permutation in itertools.permutations(range(PRIMARY_ACTOR_COUNT)):
        count = sum(table[slot][permutation[slot]] for slot in range(PRIMARY_ACTOR_COUNT))
        if count > best_count:
            best_count = count
            best_permutation = permutation
    assert best_permutation is not None
    return best_count, best_permutation


def optimal_label_agreement(left: FabricObservation, right: FabricObservation) -> float:
    table = contingency_table(left, right)
    matched, _ = optimal_label_agreement_from_table(table)
    return matched / sum(sum(row) for row in table)


def _choose_two(value: int) -> int:
    return value * (value - 1) // 2


def adjusted_rand_similarity_from_table(
    table: tuple[tuple[int, ...], ...],
) -> float:
    total = sum(sum(row) for row in table)
    if total < 2:
        return 1.0
    sum_cells = sum(_choose_two(value) for row in table for value in row)
    row_pairs = sum(_choose_two(sum(row)) for row in table)
    column_pairs = sum(
        _choose_two(sum(table[row][column] for row in range(PRIMARY_ACTOR_COUNT)))
        for column in range(PRIMARY_ACTOR_COUNT)
    )
    total_pairs = _choose_two(total)
    expected = row_pairs * column_pairs / total_pairs
    maximum = (row_pairs + column_pairs) / 2
    denominator = maximum - expected
    return 1.0 if denominator == 0 else (sum_cells - expected) / denominator


def adjusted_rand_similarity(left: FabricObservation, right: FabricObservation) -> float:
    return adjusted_rand_similarity_from_table(contingency_table(left, right))


@dataclass(frozen=True, slots=True)
class PairwiseDiversity:
    left_index: int
    right_index: int
    contingency: tuple[tuple[int, ...], ...]
    optimal_permutation: tuple[int, ...]
    matched_cell_count: int
    total_cell_count: int
    agreement: float
    disagreement: float
    adjusted_rand_similarity: float

    def __post_init__(self) -> None:
        require_int(self.left_index, "left_index", minimum=0, maximum=11)
        require_int(self.right_index, "right_index", minimum=0, maximum=11)
        if self.left_index >= self.right_index:
            raise FabricRecordError("pair indices must be increasing")
        require_int(self.matched_cell_count, "matched_cell_count", minimum=0)
        require_int(self.total_cell_count, "total_cell_count", minimum=1)
        if self.matched_cell_count > self.total_cell_count:
            raise FabricRecordError("matched count exceeds total")
        if sorted(self.optimal_permutation) != list(range(PRIMARY_ACTOR_COUNT)):
            raise FabricRecordError("optimal permutation must contain slots 0..6")
        for name in ("agreement", "disagreement", "adjusted_rand_similarity"):
            require_finite(getattr(self, name), name)
        if not math.isclose(self.agreement + self.disagreement, 1.0, abs_tol=1e-12):
            raise FabricRecordError("agreement and disagreement must sum to one")

    def to_record(self) -> dict[str, object]:
        return {
            "adjusted_rand_similarity": round(self.adjusted_rand_similarity, 12),
            "agreement": round(self.agreement, 12),
            "contingency": [list(row) for row in self.contingency],
            "disagreement": round(self.disagreement, 12),
            "left_index": self.left_index,
            "matched_cell_count": self.matched_cell_count,
            "optimal_permutation": list(self.optimal_permutation),
            "right_index": self.right_index,
            "total_cell_count": self.total_cell_count,
        }


@dataclass(frozen=True, slots=True)
class FabricDiversityReport:
    partition_fingerprints: tuple[str, ...]
    pairs: tuple[PairwiseDiversity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.partition_fingerprints, tuple) or len(self.partition_fingerprints) != 12:
            raise FabricRecordError("diversity report requires twelve fingerprints")
        for value in self.partition_fingerprints:
            require_hash(value, "partition fingerprint")
        if not isinstance(self.pairs, tuple) or len(self.pairs) != 66:
            raise FabricRecordError("diversity report requires all 66 pairs")

    @property
    def all_fingerprints_unique(self) -> bool:
        return len(set(self.partition_fingerprints)) == 12

    @property
    def all_pairs_pass(self) -> bool:
        return all(
            record["disagreement"] >= MIN_PAIR_DISAGREEMENT
            and record["adjusted_rand_similarity"] < MAX_ADJUSTED_RAND
            for record in (item.to_record() for item in self.pairs)
        )

    def to_record(self) -> dict[str, object]:
        return {
            "all_fingerprints_unique": self.all_fingerprints_unique,
            "all_pairs_pass": self.all_pairs_pass,
            "pair_count": len(self.pairs),
            "pairs": [item.to_record() for item in self.pairs],
            "partition_fingerprints": list(self.partition_fingerprints),
            "schema_id": DIVERSITY_SCHEMA_ID,
            "schema_version": 1,
            "thresholds": {
                "adjusted_rand_strictly_below": MAX_ADJUSTED_RAND,
                "minimum_disagreement": MIN_PAIR_DISAGREEMENT,
            },
        }

    @property
    def canonical_sha256(self) -> str:
        return content_sha256(self.to_record())


def build_diversity_report(
    observations: tuple[FabricObservation, ...],
) -> FabricDiversityReport:
    if not isinstance(observations, tuple) or len(observations) != 12:
        raise FabricRecordError("diversity requires twelve parent observations")
    if any(item.observation_kind != "parent_census" for item in observations):
        raise FabricRecordError("diversity requires registered parent censuses")
    fingerprints = tuple(canonical_partition_fingerprint(item) for item in observations)
    pairs: list[PairwiseDiversity] = []
    for left_index in range(12):
        for right_index in range(left_index + 1, 12):
            table = contingency_table(observations[left_index], observations[right_index])
            matched, permutation = optimal_label_agreement_from_table(table)
            total = sum(sum(row) for row in table)
            agreement = matched / total
            pairs.append(
                PairwiseDiversity(
                    left_index=left_index,
                    right_index=right_index,
                    contingency=table,
                    optimal_permutation=permutation,
                    matched_cell_count=matched,
                    total_cell_count=total,
                    agreement=agreement,
                    disagreement=1.0 - agreement,
                    adjusted_rand_similarity=adjusted_rand_similarity_from_table(table),
                )
            )
    return FabricDiversityReport(fingerprints, tuple(pairs))


__all__ = [
    "FabricDiversityReport",
    "PairwiseDiversity",
    "adjusted_rand_similarity",
    "adjusted_rand_similarity_from_table",
    "build_diversity_report",
    "canonical_partition_bytes",
    "canonical_partition_fingerprint",
    "contingency_table",
    "optimal_label_agreement",
    "optimal_label_agreement_from_table",
]
