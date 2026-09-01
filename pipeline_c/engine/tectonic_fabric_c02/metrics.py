"""Independent census, stability, and morphology diagnostics for C02."""

from __future__ import annotations

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
    CENSUS_SCHEMA_ID,
    MORPHOLOGY_SCHEMA_ID,
    PRIMARY_ACTOR_COUNT,
)
from .layout import signed_periodic_delta
from .observation import CanonicalAffiliationObservation, observe_canonical_affiliation
from .records import TectonicFabricState


def _component_summary(labels: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Count exact toroidal 4-components with row runs and union-find.

    ``labels`` is an int array containing actor slots and optionally ``-1`` as
    background.  Horizontal and vertical wrap are joined explicitly.
    """

    height, width = labels.shape
    run_ids = np.empty((height, width), dtype=np.int32)
    run_labels: list[int] = []
    run_sizes: list[int] = []
    horizontal_wrap_pairs: list[tuple[int, int]] = []
    next_id = 0
    for row_index in range(height):
        row = labels[row_index]
        starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.flatnonzero(row[1:] != row[:-1]) + 1)
        )
        stops = np.concatenate((starts[1:], np.array([width], dtype=np.int64)))
        row_first = next_id
        for start, stop in zip(starts.tolist(), stops.tolist()):
            run_ids[row_index, start:stop] = next_id
            run_labels.append(int(row[start]))
            run_sizes.append(stop - start)
            next_id += 1
        if row[0] == row[-1] and next_id - row_first > 1:
            horizontal_wrap_pairs.append((row_first, next_id - 1))

    union_parent = np.arange(next_id, dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while int(union_parent[root]) != root:
            root = int(union_parent[root])
        while value != root:
            following = int(union_parent[value])
            union_parent[value] = root
            value = following
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            union_parent[right_root] = left_root
        else:
            union_parent[left_root] = right_root

    for left, right in horizontal_wrap_pairs:
        union(left, right)
    next_rows = np.roll(labels, -1, axis=0)
    next_runs = np.roll(run_ids, -1, axis=0)
    same = labels == next_rows
    if np.any(same):
        left = run_ids[same].astype(np.int64, copy=False)
        right = next_runs[same].astype(np.int64, copy=False)
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        codes = np.unique(low * next_id + high)
        for code in codes.tolist():
            union(int(code // next_id), int(code % next_id))

    component_cells: dict[tuple[int, int], int] = {}
    for run_id, (label, size) in enumerate(zip(run_labels, run_sizes)):
        if label < 0:
            continue
        key = (label, find(run_id))
        component_cells[key] = component_cells.get(key, 0) + size
    counts = [0] * PRIMARY_ACTOR_COUNT
    largest = [0] * PRIMARY_ACTOR_COUNT
    for (label, _), size in component_cells.items():
        counts[label] += 1
        largest[label] = max(largest[label], size)
    return tuple(counts), tuple(largest)


def _contact_edges(slots: np.ndarray) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    counts: dict[tuple[int, int], int] = {}
    for other in (np.roll(slots, -1, axis=0), np.roll(slots, -1, axis=1)):
        changed = slots != other
        first = slots[changed].astype(np.int16, copy=False)
        second = other[changed].astype(np.int16, copy=False)
        low = np.minimum(first, second)
        high = np.maximum(first, second)
        codes = low * PRIMARY_ACTOR_COUNT + high
        bins = np.bincount(codes, minlength=PRIMARY_ACTOR_COUNT**2)
        for code in np.flatnonzero(bins):
            pair = (int(code) // PRIMARY_ACTOR_COUNT, int(code) % PRIMARY_ACTOR_COUNT)
            counts[pair] = counts.get(pair, 0) + int(bins[code])
    pairs = tuple(sorted(counts))
    return pairs, tuple(counts[pair] for pair in pairs)


def _contact_graph_connected(pairs: tuple[tuple[int, int], ...]) -> bool:
    graph = [set() for _ in range(PRIMARY_ACTOR_COUNT)]
    for left, right in pairs:
        graph[left].add(right)
        graph[right].add(left)
    visited = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    return len(visited) == PRIMARY_ACTOR_COUNT


@dataclass(frozen=True, slots=True)
class FabricCensusMetrics:
    source_state_sha256: str
    source_observation_sha256: str
    total_cell_count: int
    actor_cell_counts: tuple[int, ...]
    toroidal_component_counts: tuple[int, ...]
    observed_contact_pairs: tuple[tuple[int, int], ...]
    contact_edge_counts: tuple[int, ...]
    contact_graph_connected: bool
    normalized_entropy: float

    def __post_init__(self) -> None:
        require_hash(self.source_state_sha256, "source_state_sha256")
        require_hash(self.source_observation_sha256, "source_observation_sha256")
        require_int(self.total_cell_count, "total_cell_count", minimum=1)
        for name in ("actor_cell_counts", "toroidal_component_counts"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != PRIMARY_ACTOR_COUNT:
                raise FabricRecordError(f"{name} must contain seven values")
            for value in values:
                require_int(value, f"{name} item", minimum=0)
        if sum(self.actor_cell_counts) != self.total_cell_count:
            raise FabricRecordError("actor counts do not cover the canonical parent")
        if len(self.contact_edge_counts) != len(self.observed_contact_pairs):
            raise FabricRecordError("contact edge counts do not match contact pairs")
        for pair, count in zip(self.observed_contact_pairs, self.contact_edge_counts):
            if len(pair) != 2 or not 0 <= pair[0] < pair[1] < PRIMARY_ACTOR_COUNT:
                raise FabricRecordError("invalid contact pair")
            require_int(count, "contact edge count", minimum=1)
        if not isinstance(self.contact_graph_connected, bool):
            raise FabricRecordError("contact_graph_connected must be boolean")
        entropy = require_finite(self.normalized_entropy, "normalized_entropy")
        if not 0.0 <= entropy <= 1.0:
            raise FabricRecordError("normalized entropy must lie in [0,1]")

    @property
    def area_shares(self) -> tuple[float, ...]:
        return tuple(value / self.total_cell_count for value in self.actor_cell_counts)

    @property
    def dominant_actor_slot(self) -> int:
        return max(range(PRIMARY_ACTOR_COUNT), key=self.actor_cell_counts.__getitem__)

    @property
    def hierarchy_ratio(self) -> float:
        smallest = min(self.actor_cell_counts)
        return math.inf if smallest == 0 else max(self.actor_cell_counts) / smallest

    def to_record(self) -> dict[str, object]:
        return {
            "actor_areas": [
                {
                    "actor_slot": slot,
                    "cell_count": self.actor_cell_counts[slot],
                    "fraction": {
                        "denominator": self.total_cell_count,
                        "numerator": self.actor_cell_counts[slot],
                    },
                    "toroidal_4_component_count": self.toroidal_component_counts[slot],
                }
                for slot in range(PRIMARY_ACTOR_COUNT)
            ],
            "contact_graph_connected": self.contact_graph_connected,
            "contacts": [
                {"cardinal_edge_count": count, "slots": list(pair)}
                for pair, count in zip(self.observed_contact_pairs, self.contact_edge_counts)
            ],
            "dominant_actor_slot": self.dominant_actor_slot,
            "hierarchy_ratio": round(self.hierarchy_ratio, 12),
            "normalized_entropy": round(self.normalized_entropy, 12),
            "schema_id": CENSUS_SCHEMA_ID,
            "schema_version": 1,
            "source_observation_sha256": self.source_observation_sha256,
            "source_state_sha256": self.source_state_sha256,
            "total_cell_count": self.total_cell_count,
        }

    @property
    def canonical_sha256(self) -> str:
        return content_sha256(self.to_record())


@dataclass(frozen=True, slots=True)
class ActorMorphologyMetrics:
    slot: int
    area_cell_count: int
    nearest_nucleus_distance_m: int
    mean_endpoint_agreement: float
    strict_all_eight_cell_count: int
    aspect_ratio: float
    compactness_penalty: float
    principal_axis_degrees: float
    angle_to_nearest_cardinal_degrees: float
    erosion_cell_count: int
    erosion_retention: float
    eroded_component_count: int
    eroded_largest_component_cell_count: int
    eroded_largest_component_fraction: float

    def __post_init__(self) -> None:
        require_int(self.slot, "slot", minimum=0, maximum=PRIMARY_ACTOR_COUNT - 1)
        require_int(self.area_cell_count, "area_cell_count", minimum=1)
        require_int(self.nearest_nucleus_distance_m, "nearest_nucleus_distance_m", minimum=1)
        require_int(self.strict_all_eight_cell_count, "strict_all_eight_cell_count", minimum=0, maximum=self.area_cell_count)
        require_int(self.erosion_cell_count, "erosion_cell_count", minimum=0, maximum=self.area_cell_count)
        require_int(self.eroded_component_count, "eroded_component_count", minimum=0)
        require_int(self.eroded_largest_component_cell_count, "eroded_largest_component_cell_count", minimum=0, maximum=self.erosion_cell_count)
        for name in (
            "mean_endpoint_agreement",
            "aspect_ratio",
            "compactness_penalty",
            "principal_axis_degrees",
            "angle_to_nearest_cardinal_degrees",
            "erosion_retention",
            "eroded_largest_component_fraction",
        ):
            require_finite(getattr(self, name), name)

    @property
    def strict_all_eight_share(self) -> float:
        return self.strict_all_eight_cell_count / self.area_cell_count

    def to_record(self) -> dict[str, object]:
        return {
            "actor_slot": self.slot,
            "angle_to_nearest_cardinal_degrees": round(self.angle_to_nearest_cardinal_degrees, 12),
            "area_cell_count": self.area_cell_count,
            "aspect_ratio": round(self.aspect_ratio, 12),
            "compactness_penalty": round(self.compactness_penalty, 12),
            "eroded_component_count": self.eroded_component_count,
            "eroded_largest_component_cell_count": self.eroded_largest_component_cell_count,
            "eroded_largest_component_fraction": round(self.eroded_largest_component_fraction, 12),
            "erosion_cell_count": self.erosion_cell_count,
            "erosion_retention": round(self.erosion_retention, 12),
            "mean_endpoint_agreement": round(self.mean_endpoint_agreement, 12),
            "nearest_nucleus_distance_m": self.nearest_nucleus_distance_m,
            "principal_axis_degrees": round(self.principal_axis_degrees, 12),
            "strict_all_eight_cell_count": self.strict_all_eight_cell_count,
            "strict_all_eight_share_of_actor": round(self.strict_all_eight_share, 12),
        }


@dataclass(frozen=True, slots=True)
class FabricMorphologyMetrics:
    source_state_sha256: str
    source_observation_sha256: str
    family_id: int
    nucleus_nearest_neighbor_cv: float
    total_mean_endpoint_agreement: float
    strict_all_eight_total_cell_count: int
    actor_metrics: tuple[ActorMorphologyMetrics, ...]
    adjacency_signature_sha256: str

    def __post_init__(self) -> None:
        require_hash(self.source_state_sha256, "source_state_sha256")
        require_hash(self.source_observation_sha256, "source_observation_sha256")
        require_hash(self.adjacency_signature_sha256, "adjacency_signature_sha256")
        require_int(self.family_id, "family_id", minimum=0, maximum=3)
        require_finite(self.nucleus_nearest_neighbor_cv, "nucleus_nearest_neighbor_cv")
        require_finite(self.total_mean_endpoint_agreement, "total_mean_endpoint_agreement")
        require_int(self.strict_all_eight_total_cell_count, "strict_all_eight_total_cell_count", minimum=0)
        if (
            not isinstance(self.actor_metrics, tuple)
            or tuple(item.slot for item in self.actor_metrics) != tuple(range(PRIMARY_ACTOR_COUNT))
        ):
            raise FabricRecordError("morphology requires seven ordered actor records")

    @property
    def strict_all_eight_total_share(self) -> float:
        return self.strict_all_eight_total_cell_count / sum(
            item.area_cell_count for item in self.actor_metrics
        )

    def to_record(self) -> dict[str, object]:
        return {
            "actor_metrics": [item.to_record() for item in self.actor_metrics],
            "adjacency_signature_sha256": self.adjacency_signature_sha256,
            "definitions": {
                "aspect": "sqrt(largest/smallest nucleus-unwrapped covariance eigenvalue)",
                "compactness": "cardinal_grid_perimeter_squared_over_4pi_area",
                "cv": "population_standard_deviation_over_mean",
                "erosion": "retain iff all_four_cardinal_neighbors_share_owner",
                "eroded_largest_fraction": "largest_toroidal_component_over_eroded_cells",
                "principal_axis_degrees": "counterclockwise_from_positive_x_modulo_180",
            },
            "family_id": self.family_id,
            "nucleus_nearest_neighbor_cv": round(self.nucleus_nearest_neighbor_cv, 12),
            "schema_id": MORPHOLOGY_SCHEMA_ID,
            "schema_version": 1,
            "source_observation_sha256": self.source_observation_sha256,
            "source_state_sha256": self.source_state_sha256,
            "strict_all_eight_total_cell_count": self.strict_all_eight_total_cell_count,
            "strict_all_eight_total_share": round(self.strict_all_eight_total_share, 12),
            "total_mean_endpoint_agreement": round(self.total_mean_endpoint_agreement, 12),
        }

    @property
    def canonical_sha256(self) -> str:
        return content_sha256(self.to_record())


def measure_census(
    state: TectonicFabricState,
    observation: CanonicalAffiliationObservation | None = None,
) -> FabricCensusMetrics:
    if not isinstance(state, TectonicFabricState):
        raise TypeError("state must be TectonicFabricState")
    if observation is None:
        observation = observe_canonical_affiliation(state)
    if not isinstance(observation, CanonicalAffiliationObservation):
        raise TypeError("observation must be CanonicalAffiliationObservation")
    if observation.source_state_sha256 != state.canonical_sha256:
        raise FabricRecordError("canonical observation belongs to another state")
    slots = observation.slots_array()
    counts = tuple(
        int(np.count_nonzero(slots == slot)) for slot in range(PRIMARY_ACTOR_COUNT)
    )
    component_counts, _ = _component_summary(slots.astype(np.int8, copy=False))
    pairs, edge_counts = _contact_edges(slots)
    shares = [value / slots.size for value in counts if value]
    entropy = -sum(value * math.log(value) for value in shares) / math.log(PRIMARY_ACTOR_COUNT)
    return FabricCensusMetrics(
        source_state_sha256=state.canonical_sha256,
        source_observation_sha256=observation.canonical_sha256,
        total_cell_count=slots.size,
        actor_cell_counts=counts,
        toroidal_component_counts=component_counts,
        observed_contact_pairs=pairs,
        contact_edge_counts=edge_counts,
        contact_graph_connected=_contact_graph_connected(pairs),
        normalized_entropy=entropy,
    )


def measure_morphology(
    state: TectonicFabricState,
    observation: CanonicalAffiliationObservation | None = None,
    census: FabricCensusMetrics | None = None,
) -> FabricMorphologyMetrics:
    if not isinstance(state, TectonicFabricState):
        raise TypeError("state must be TectonicFabricState")
    if observation is None:
        observation = observe_canonical_affiliation(state)
    if census is None:
        census = measure_census(state, observation)
    if observation.source_state_sha256 != state.canonical_sha256 or census.source_state_sha256 != state.canonical_sha256:
        raise FabricRecordError("metrics inputs belong to another state")
    slots = observation.slots_array()
    agreement = observation.endpoint_agreement_array()
    strict = observation.strict_all_eight_array().astype(np.bool_, copy=False)
    eroded = (
        (slots == np.roll(slots, 1, axis=0))
        & (slots == np.roll(slots, -1, axis=0))
        & (slots == np.roll(slots, 1, axis=1))
        & (slots == np.roll(slots, -1, axis=1))
    )
    eroded_labels = np.where(
        eroded, slots.astype(np.int16, copy=False), -1
    ).astype(np.int16, copy=False)
    eroded_components, eroded_largest = _component_summary(eroded_labels)
    actor_records: list[ActorMorphologyMetrics] = []
    for actor in state.actors:
        mask = slots == actor.slot
        row, column = np.nonzero(mask)
        x_center = column.astype(np.int64) * 40_000 + 20_000
        y_center = row.astype(np.int64) * 40_000 + 20_000
        dx = ((x_center - actor.nucleus_x_m + 20_480_000) % 40_960_000) - 20_480_000
        dy = ((y_center - actor.nucleus_y_m + 20_480_000) % 40_960_000) - 20_480_000
        covariance = np.cov(np.stack((dx, dy)), bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        smallest = max(float(eigenvalues[0]), 1e-12)
        largest = max(float(eigenvalues[-1]), 1e-12)
        aspect = math.sqrt(largest / smallest)
        vector = eigenvectors[:, -1]
        principal = math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 180.0
        folded = principal % 90.0
        angle_to_cardinal = min(folded, 90.0 - folded)
        perimeter = (
            int(np.count_nonzero(mask & (np.roll(slots, 1, axis=0) != actor.slot)))
            + int(np.count_nonzero(mask & (np.roll(slots, -1, axis=0) != actor.slot)))
            + int(np.count_nonzero(mask & (np.roll(slots, 1, axis=1) != actor.slot)))
            + int(np.count_nonzero(mask & (np.roll(slots, -1, axis=1) != actor.slot)))
        )
        area = int(mask.sum())
        erosion_count = int(np.count_nonzero(eroded & mask))
        actor_records.append(
            ActorMorphologyMetrics(
                slot=actor.slot,
                area_cell_count=area,
                nearest_nucleus_distance_m=actor.nearest_nucleus_distance_m,
                mean_endpoint_agreement=float(agreement[mask].sum() / (8 * area)),
                strict_all_eight_cell_count=int(np.count_nonzero(strict & mask)),
                aspect_ratio=aspect,
                compactness_penalty=(perimeter * perimeter) / (4 * math.pi * area),
                principal_axis_degrees=principal,
                angle_to_nearest_cardinal_degrees=angle_to_cardinal,
                erosion_cell_count=erosion_count,
                erosion_retention=erosion_count / area,
                eroded_component_count=eroded_components[actor.slot],
                eroded_largest_component_cell_count=eroded_largest[actor.slot],
                eroded_largest_component_fraction=(
                    eroded_largest[actor.slot] / erosion_count if erosion_count else 0.0
                ),
            )
        )
    nearest = np.asarray(
        [actor.nearest_nucleus_distance_m for actor in state.actors], dtype=np.float64
    )
    return FabricMorphologyMetrics(
        source_state_sha256=state.canonical_sha256,
        source_observation_sha256=observation.canonical_sha256,
        family_id=state.family_id,
        nucleus_nearest_neighbor_cv=float(nearest.std(ddof=0) / nearest.mean()),
        total_mean_endpoint_agreement=float(agreement.sum() / (8 * slots.size)),
        strict_all_eight_total_cell_count=int(strict.sum()),
        actor_metrics=tuple(actor_records),
        adjacency_signature_sha256=state.adjacency_signature_sha256,
    )


__all__ = [
    "ActorMorphologyMetrics",
    "FabricCensusMetrics",
    "FabricMorphologyMetrics",
    "measure_census",
    "measure_morphology",
]
