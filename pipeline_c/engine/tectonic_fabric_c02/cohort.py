"""Typed, position-preserving C02 development cohort execution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..foundation import (
    COHORT_MANIFEST_SHA256,
    ExecutionSeed,
    ValidationAccessError,
    build_development_cohort,
    development_execution_plan,
    seed_for_execution,
)
from ._util import (
    FabricFormationError,
    FabricRecordError,
    content_sha256,
    freeze_json,
    require_hash,
    require_id,
    require_int,
    thaw_json,
)
from .constants import (
    ATTEMPT_ID,
    COHORT_MEMBER_SCHEMA_ID,
    COHORT_SCHEMA_ID,
    ROADMAP_RUN,
)
from .construction import build_tectonic_fabric_state
from .diversity import FabricDiversityReport, build_diversity_report
from .metrics import FabricCensusMetrics, FabricMorphologyMetrics, measure_census, measure_morphology
from .observation import (
    CanonicalAffiliationObservation,
    FabricObservation,
    observe_analysis,
    observe_canonical_affiliation,
    observe_parent_census,
)
from .records import TectonicFabricState


@dataclass(frozen=True, slots=True)
class FormationFailureRecord:
    code: str
    message: str
    details: object

    def __post_init__(self) -> None:
        require_id(self.code, "failure code")
        if not isinstance(self.message, str) or not self.message:
            raise FabricRecordError("failure message must be non-empty")
        object.__setattr__(self, "details", freeze_json(self.details, "failure details"))

    @classmethod
    def from_exception(cls, error: FabricFormationError) -> "FormationFailureRecord":
        if not isinstance(error, FabricFormationError):
            raise TypeError("error must be FabricFormationError")
        return cls(error.code, str(error), error.details)

    def to_record(self) -> dict[str, object]:
        return {
            "code": self.code,
            "details": thaw_json(self.details),
            "message": self.message,
            "retryable": False,
        }


@dataclass(frozen=True, slots=True)
class CohortReceipt:
    index: int
    role: str
    seed: int
    attempt_count: int
    outcome: str
    state_sha256: str | None
    failure: FormationFailureRecord | None

    def __post_init__(self) -> None:
        require_int(self.index, "receipt index", minimum=0, maximum=11)
        if self.role != "development":
            raise FabricRecordError("C02 production receipts must be development")
        require_int(self.seed, "receipt seed", minimum=0, maximum=2**32 - 1)
        if self.attempt_count != 1:
            raise FabricRecordError("every C02 receipt must record one attempt")
        if self.outcome not in {"success", "failed"}:
            raise FabricRecordError("unsupported receipt outcome")
        if self.outcome == "success":
            require_hash(self.state_sha256, "state_sha256")
            if self.failure is not None:
                raise FabricRecordError("successful receipt cannot retain a failure")
        elif self.state_sha256 is not None or not isinstance(self.failure, FormationFailureRecord):
            raise FabricRecordError("failed receipt requires one typed failure and no state")

    def to_record(self) -> dict[str, object]:
        return {
            "attempt_count": self.attempt_count,
            "failure": None if self.failure is None else self.failure.to_record(),
            "index": self.index,
            "outcome": self.outcome,
            "role": self.role,
            "seed": self.seed,
            "state_sha256": self.state_sha256,
        }


@dataclass(frozen=True, slots=True)
class DevelopmentFabricMember:
    member_id: str
    execution_seed: ExecutionSeed
    outcome: str
    state: TectonicFabricState | None
    canonical_affiliation: CanonicalAffiliationObservation | None
    parent_census: FabricObservation | None
    analysis_512: FabricObservation | None
    analysis_1024: FabricObservation | None
    census_metrics: FabricCensusMetrics | None
    morphology_metrics: FabricMorphologyMetrics | None
    failure: FormationFailureRecord | None = None

    def __post_init__(self) -> None:
        require_id(self.member_id, "member_id")
        if not isinstance(self.execution_seed, ExecutionSeed):
            raise FabricRecordError("member execution_seed must be ExecutionSeed")
        expected_id = f"dev-{self.execution_seed.index:02d}"
        if self.member_id != expected_id or self.execution_seed.role != "development":
            raise FabricRecordError("member ID or role differs from cohort position")
        if self.outcome not in {"success", "failed"}:
            raise FabricRecordError("unsupported member outcome")
        products = (
            self.state,
            self.canonical_affiliation,
            self.parent_census,
            self.analysis_512,
            self.analysis_1024,
            self.census_metrics,
            self.morphology_metrics,
        )
        if self.outcome == "failed":
            if any(value is not None for value in products) or not isinstance(self.failure, FormationFailureRecord):
                raise FabricRecordError("failed member must retain only its typed failure")
            return
        expected_types = (
            TectonicFabricState,
            CanonicalAffiliationObservation,
            FabricObservation,
            FabricObservation,
            FabricObservation,
            FabricCensusMetrics,
            FabricMorphologyMetrics,
        )
        if any(not isinstance(value, kind) for value, kind in zip(products, expected_types)):
            raise FabricRecordError("successful member is missing a typed product")
        if self.failure is not None:
            raise FabricRecordError("successful member cannot retain a failure")
        assert self.state is not None
        assert self.canonical_affiliation is not None
        assert self.parent_census is not None
        assert self.analysis_512 is not None
        assert self.analysis_1024 is not None
        assert self.census_metrics is not None
        assert self.morphology_metrics is not None
        if self.state.execution_seed != self.execution_seed:
            raise FabricRecordError("member state uses another execution seed")
        state_hash = self.state.canonical_sha256
        sources = (
            self.canonical_affiliation.source_state_sha256,
            self.parent_census.source_state_sha256,
            self.analysis_512.source_state_sha256,
            self.analysis_1024.source_state_sha256,
            self.census_metrics.source_state_sha256,
            self.morphology_metrics.source_state_sha256,
        )
        if any(value != state_hash for value in sources):
            raise FabricRecordError("member products do not close one state")
        if self.parent_census.observation_kind != "parent_census":
            raise FabricRecordError("member parent census has the wrong kind")
        if self.analysis_512.width_px != 512 or self.analysis_1024.width_px != 1024:
            raise FabricRecordError("member analysis observations have wrong sizes")

    @property
    def receipt(self) -> CohortReceipt:
        return CohortReceipt(
            index=self.execution_seed.index,
            role=self.execution_seed.role,
            seed=self.execution_seed.seed,
            attempt_count=1,
            outcome=self.outcome,
            state_sha256=(None if self.state is None else self.state.canonical_sha256),
            failure=self.failure,
        )

    def to_record(self, *, include_arrays: bool = False) -> dict[str, object]:
        return {
            "analysis_1024": (
                None if self.analysis_1024 is None else self.analysis_1024.to_record(include_data=include_arrays)
            ),
            "analysis_512": (
                None if self.analysis_512 is None else self.analysis_512.to_record(include_data=include_arrays)
            ),
            "canonical_affiliation": (
                None
                if self.canonical_affiliation is None
                else self.canonical_affiliation.to_record(include_data=include_arrays)
            ),
            "census_metrics": None if self.census_metrics is None else self.census_metrics.to_record(),
            "execution_seed": self.execution_seed.to_record(),
            "failure": None if self.failure is None else self.failure.to_record(),
            "member_id": self.member_id,
            "morphology_metrics": (
                None if self.morphology_metrics is None else self.morphology_metrics.to_record()
            ),
            "outcome": self.outcome,
            "parent_census": (
                None if self.parent_census is None else self.parent_census.to_record(include_data=include_arrays)
            ),
            "schema_id": COHORT_MEMBER_SCHEMA_ID,
            "schema_version": 1,
            "state": None if self.state is None else self.state.to_record(include_arrays=include_arrays),
        }


@dataclass(frozen=True, slots=True)
class DevelopmentFabricCohort:
    members: tuple[DevelopmentFabricMember, ...]
    diversity: FabricDiversityReport | None

    def __post_init__(self) -> None:
        if not isinstance(self.members, tuple) or len(self.members) != 12:
            raise FabricRecordError("C02 cohort requires twelve positional members")
        plan = development_execution_plan()
        if tuple(member.execution_seed for member in self.members) != plan:
            raise FabricRecordError("C02 members differ from the frozen execution plan")
        if self.complete_success:
            if not isinstance(self.diversity, FabricDiversityReport):
                raise FabricRecordError("complete cohort requires diversity report")
        elif self.diversity is not None:
            raise FabricRecordError("failed cohort cannot claim complete diversity")

    @property
    def complete_success(self) -> bool:
        return all(member.outcome == "success" for member in self.members)

    @property
    def receipts(self) -> tuple[CohortReceipt, ...]:
        return tuple(member.receipt for member in self.members)

    @property
    def states(self) -> tuple[TectonicFabricState, ...]:
        return tuple(member.state for member in self.members if member.state is not None)

    @property
    def canonical_affiliations(self) -> tuple[CanonicalAffiliationObservation, ...]:
        return tuple(
            member.canonical_affiliation
            for member in self.members
            if member.canonical_affiliation is not None
        )

    @property
    def parent_censuses(self) -> tuple[FabricObservation, ...]:
        return tuple(member.parent_census for member in self.members if member.parent_census is not None)

    @property
    def analysis_512(self) -> tuple[FabricObservation, ...]:
        return tuple(member.analysis_512 for member in self.members if member.analysis_512 is not None)

    @property
    def analysis_1024(self) -> tuple[FabricObservation, ...]:
        return tuple(member.analysis_1024 for member in self.members if member.analysis_1024 is not None)

    @property
    def census_metrics(self) -> tuple[FabricCensusMetrics, ...]:
        return tuple(member.census_metrics for member in self.members if member.census_metrics is not None)

    @property
    def morphology_metrics(self) -> tuple[FabricMorphologyMetrics, ...]:
        return tuple(member.morphology_metrics for member in self.members if member.morphology_metrics is not None)

    @property
    def family_ids(self) -> tuple[int | None, ...]:
        return tuple(None if member.state is None else member.state.family_id for member in self.members)

    def family_conditioned_record(self) -> dict[str, object]:
        """Return complete, unselected morphology distributions by family."""

        result: dict[str, object] = {}
        for family_id in range(4):
            selected = [
                member
                for member in self.members
                if member.state is not None and member.state.family_id == family_id
            ]
            result[str(family_id)] = {
                "contact_pair_counts": [
                    len(member.census_metrics.observed_contact_pairs)
                    for member in selected
                    if member.census_metrics is not None
                ],
                "maximum_actor_aspect_ratios": [
                    max(actor.aspect_ratio for actor in member.morphology_metrics.actor_metrics)
                    for member in selected
                    if member.morphology_metrics is not None
                ],
                "member_indices": [member.execution_seed.index for member in selected],
                "nucleus_nearest_neighbor_cv": [
                    member.morphology_metrics.nucleus_nearest_neighbor_cv
                    for member in selected
                    if member.morphology_metrics is not None
                ],
                "strict_all_eight_total_shares": [
                    member.morphology_metrics.strict_all_eight_total_share
                    for member in selected
                    if member.morphology_metrics is not None
                ],
                "total_mean_endpoint_agreements": [
                    member.morphology_metrics.total_mean_endpoint_agreement
                    for member in selected
                    if member.morphology_metrics is not None
                ],
            }
        return result

    def member_record(self, index: int, *, include_arrays: bool = False) -> dict[str, object]:
        require_int(index, "member index", minimum=0, maximum=11)
        return self.members[index].to_record(include_arrays=include_arrays)

    def to_record(self, *, include_arrays: bool = False) -> dict[str, object]:
        return {
            "attempt_id": ATTEMPT_ID,
            "cohort_manifest_sha256": COHORT_MANIFEST_SHA256,
            "complete_success": self.complete_success,
            "derived_family_ids": list(self.family_ids),
            "diversity": None if self.diversity is None else self.diversity.to_record(),
            "family_conditioned_morphology": self.family_conditioned_record(),
            "member_count": len(self.members),
            "members": [member.to_record(include_arrays=include_arrays) for member in self.members],
            "receipts": [receipt.to_record() for receipt in self.receipts],
            "roadmap_run": ROADMAP_RUN,
            "schema_id": COHORT_SCHEMA_ID,
            "schema_version": 1,
            "validation": {
                "artifact_count": 0,
                "receipt_count": 0,
                "state": "sealed_unopened",
            },
        }

    @property
    def canonical_sha256(self) -> str:
        return content_sha256(self.to_record())


@lru_cache(maxsize=1)
def build_development_fabric_cohort() -> DevelopmentFabricCohort:
    """Execute each frozen development identity once, retaining typed failures."""

    foundations = build_development_cohort()
    members: list[DevelopmentFabricMember] = []
    for foundation in foundations:
        execution_seed = foundation.execution_seed
        try:
            state = build_tectonic_fabric_state(
                execution_seed,
                foundation.identities,
                foundation.spec.parent_rectangle,
            )
            canonical = observe_canonical_affiliation(state)
            census = measure_census(state, canonical)
            morphology = measure_morphology(state, canonical, census)
            members.append(
                DevelopmentFabricMember(
                    member_id=f"dev-{execution_seed.index:02d}",
                    execution_seed=execution_seed,
                    outcome="success",
                    state=state,
                    canonical_affiliation=canonical,
                    parent_census=observe_parent_census(state),
                    analysis_512=observe_analysis(state, 512),
                    analysis_1024=observe_analysis(state, 1024),
                    census_metrics=census,
                    morphology_metrics=morphology,
                )
            )
        except FabricFormationError as error:
            members.append(
                DevelopmentFabricMember(
                    member_id=f"dev-{execution_seed.index:02d}",
                    execution_seed=execution_seed,
                    outcome="failed",
                    state=None,
                    canonical_affiliation=None,
                    parent_census=None,
                    analysis_512=None,
                    analysis_1024=None,
                    census_metrics=None,
                    morphology_metrics=None,
                    failure=FormationFailureRecord.from_exception(error),
                )
            )
    member_tuple = tuple(members)
    complete = all(member.outcome == "success" for member in member_tuple)
    diversity = (
        build_diversity_report(
            tuple(member.parent_census for member in member_tuple if member.parent_census is not None)
        )
        if complete
        else None
    )
    # The derived family draw is a cohort gate, not a reason to discard the
    # twelve already-executed positional outcomes.  Audit G03 records/fails a
    # mismatch without retrying or replacing any member.
    return DevelopmentFabricCohort(member_tuple, diversity)


def validation_guard_is_closed() -> bool:
    try:
        seed_for_execution("validation", 0, roadmap_run=ROADMAP_RUN)
    except ValidationAccessError:
        return True
    return False


__all__ = [
    "CohortReceipt",
    "DevelopmentFabricCohort",
    "DevelopmentFabricMember",
    "FormationFailureRecord",
    "build_development_fabric_cohort",
    "validation_guard_is_closed",
]
