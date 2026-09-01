"""Complete-cohort C02 gates with one process-cached 1024-lattice execution."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
import unittest


PIPELINE_C = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
for path in (PIPELINE_C, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from engine.foundation import (  # noqa: E402
    COHORT_MANIFEST,
    ValidationAccessError,
    seed_for_execution,
)
from engine.tectonic_fabric_c02.constants import (  # noqa: E402
    FROZEN_DEVELOPMENT_FAMILY_IDS,
    MAX_ACTOR_AREA_PERCENT,
    MAX_ADJUSTED_RAND,
    MAX_COMPACTNESS_PENALTY,
    MAX_CONTACT_PAIR_COUNT,
    MAX_HIERARCHY,
    MIN_ACTOR_AREA_PERCENT,
    MIN_ACTOR_MEAN_ENDPOINT_AGREEMENT,
    MIN_ASPECT_RATIO,
    MIN_ASPECT_WORLD_COUNT,
    MIN_BELT_ASPECT_RATIO,
    MIN_CONTACT_PAIR_COUNT,
    MIN_EROSION_RETENTION,
    MIN_HIERARCHY_DENOMINATOR,
    MIN_HIERARCHY_NUMERATOR,
    MIN_LARGEST_ACTOR_PERCENT,
    MIN_NUCLEUS_NEIGHBOR_CV,
    MIN_PAIR_DISAGREEMENT,
    MIN_TOTAL_MEAN_ENDPOINT_AGREEMENT,
)
from engine.tectonic_fabric_c02.cohort import validation_guard_is_closed  # noqa: E402
from engine.tectonic_fabric_c02.audit import audit_bundle_record  # noqa: E402
from test_c5_c02_support import cohort_audits, development_cohort  # noqa: E402


class C5C02CompleteCohortGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cohort = development_cohort()

    def test_c01_all_twelve_members_and_receipts_are_positional_and_successful(self) -> None:
        cohort = self.cohort
        self.assertTrue(cohort.complete_success)
        self.assertEqual(len(cohort.members), 12)
        self.assertEqual(len(cohort.states), 12)
        self.assertEqual(
            tuple(member.execution_seed.seed for member in cohort.members),
            COHORT_MANIFEST.development,
        )
        self.assertEqual(
            tuple(member.member_id for member in cohort.members),
            tuple(f"dev-{index:02d}" for index in range(12)),
        )
        for index, receipt in enumerate(cohort.receipts):
            self.assertEqual(receipt.index, index)
            self.assertEqual(receipt.attempt_count, 1)
            self.assertEqual(receipt.outcome, "success")
            self.assertIsNotNone(receipt.state_sha256)
            self.assertIsNone(receipt.failure)

    def test_c02_family_draw_is_derived_exact_and_not_quality_selected(self) -> None:
        self.assertEqual(self.cohort.family_ids, FROZEN_DEVELOPMENT_FAMILY_IDS)
        counts = Counter(self.cohort.family_ids)
        self.assertEqual(set(counts), {0, 1, 2, 3})
        self.assertLessEqual(max(counts.values()), 6)
        self.assertEqual(self.cohort.to_record()["derived_family_ids"], list(FROZEN_DEVELOPMENT_FAMILY_IDS))

    def test_c03_every_world_passes_completeness_connectivity_and_hierarchy(self) -> None:
        for member in self.cohort.members:
            with self.subTest(member=member.member_id):
                state = member.state
                census = member.census_metrics
                assert state is not None and census is not None
                total = census.total_cell_count
                self.assertTrue(state.certificate.passes)
                self.assertEqual(census.toroidal_component_counts, (1,) * 7)
                self.assertTrue(census.contact_graph_connected)
                self.assertTrue(all(
                    MIN_ACTOR_AREA_PERCENT * total <= 100 * count
                    <= MAX_ACTOR_AREA_PERCENT * total
                    for count in census.actor_cell_counts
                ))
                self.assertGreaterEqual(
                    100 * max(census.actor_cell_counts),
                    MIN_LARGEST_ACTOR_PERCENT * total,
                )
                self.assertGreaterEqual(
                    max(census.actor_cell_counts) * MIN_HIERARCHY_DENOMINATOR,
                    min(census.actor_cell_counts) * MIN_HIERARCHY_NUMERATOR,
                )
                self.assertLessEqual(census.hierarchy_ratio, MAX_HIERARCHY)

    def test_c04_anti_cellularity_shape_envelope_and_stability_are_complete(self) -> None:
        elongated_worlds = 0
        for member in self.cohort.members:
            morphology = member.morphology_metrics
            census = member.census_metrics
            assert morphology is not None and census is not None
            maximum_aspect = max(
                item.aspect_ratio for item in morphology.actor_metrics
            )
            elongated_worlds += maximum_aspect >= MIN_ASPECT_RATIO
            with self.subTest(member=member.member_id, family=morphology.family_id):
                self.assertGreaterEqual(
                    morphology.nucleus_nearest_neighbor_cv,
                    MIN_NUCLEUS_NEIGHBOR_CV,
                )
                if morphology.family_id == 1:
                    self.assertGreaterEqual(maximum_aspect, MIN_BELT_ASPECT_RATIO)
                self.assertGreaterEqual(
                    len(census.observed_contact_pairs), MIN_CONTACT_PAIR_COUNT
                )
                self.assertLessEqual(
                    len(census.observed_contact_pairs), MAX_CONTACT_PAIR_COUNT
                )
                self.assertGreaterEqual(
                    morphology.total_mean_endpoint_agreement,
                    MIN_TOTAL_MEAN_ENDPOINT_AGREEMENT,
                )
                self.assertTrue(all(
                    item.compactness_penalty <= MAX_COMPACTNESS_PENALTY
                    and item.erosion_retention >= MIN_EROSION_RETENTION
                    and item.mean_endpoint_agreement
                    >= MIN_ACTOR_MEAN_ENDPOINT_AGREEMENT
                    for item in morphology.actor_metrics
                ))
        self.assertGreaterEqual(elongated_worlds, MIN_ASPECT_WORLD_COUNT)

    def test_c05_seeded_structural_and_label_invariant_variety_is_complete(self) -> None:
        cohort = self.cohort
        retained_hash_fields = (
            "canonical_sha256",
            "layout_sha256",
            "construction_sha256",
            "partition_sha256",
            "adjacency_signature_sha256",
        )
        for field in retained_hash_fields:
            with self.subTest(field=field):
                values = [getattr(state, field) for state in cohort.states]
                self.assertEqual(len(values), 12)
                self.assertTrue(all(len(value) == 64 for value in values))
        # Unique-count diagnostics for these labeled structural hashes are not
        # frozen gates.  Variety is gated only by the label-invariant products
        # and all 66 pair thresholds below.
        diversity = cohort.diversity
        self.assertIsNotNone(diversity)
        assert diversity is not None
        self.assertEqual(len(diversity.pairs), 66)
        self.assertTrue(diversity.all_fingerprints_unique)
        self.assertTrue(all(
            pair.disagreement >= MIN_PAIR_DISAGREEMENT
            and pair.adjusted_rand_similarity < MAX_ADJUSTED_RAND
            for pair in diversity.pairs
        ))

    def test_c06_all_observation_products_close_their_own_state(self) -> None:
        for member in self.cohort.members:
            state = member.state
            assert state is not None
            state_hash = state.canonical_sha256
            products = (
                member.canonical_affiliation,
                member.parent_census,
                member.analysis_512,
                member.analysis_1024,
                member.census_metrics,
                member.morphology_metrics,
            )
            self.assertTrue(all(
                product is not None
                and product.source_state_sha256 == state_hash
                for product in products
            ))

    def test_c07_validation_remains_sealed_with_no_receipt_or_artifact(self) -> None:
        self.assertTrue(validation_guard_is_closed())
        self.assertEqual(
            self.cohort.to_record()["validation"],
            {"artifact_count": 0, "receipt_count": 0, "state": "sealed_unopened"},
        )
        with self.assertRaises(ValidationAccessError):
            seed_for_execution("validation", 0, roadmap_run="C5")

    def test_c08_quick_audit_covers_all_ids_without_claiming_skipped_replay(self) -> None:
        audits = cohort_audits()
        self.assertEqual(
            set(audits),
            {
                "c5.c02.g01.scope-readiness.v1",
                "c5.c02.g02.c4-inheritance-isolation.v1",
                "c5.c02.g03.layout.v1",
                "c5.c02.g04.growth-completeness.v1",
                "c5.c02.g05.connectivity.v1",
                "c5.c02.g06.determinism.v1",
                "c5.c02.g07.low-count-hierarchy.v1",
                "c5.c02.g08.anti-cellularity.v1",
                "c5.c02.g09.shape-envelope.v1",
                "c5.c02.g10.stability-diagnostics.v1",
                "c5.c02.g11.seeded-variety.v1",
                "c5.c02.g12.physical-observations.v1",
                "c5.c02.g13.cohort-validation.v1",
            },
        )
        replay_id = "c5.c02.g06.determinism.v1"
        self.assertFalse(audits[replay_id].passes)
        replay_details = audits[replay_id].details
        self.assertIs(replay_details["conditions"]["replay_executed"], False)
        self.assertEqual(
            replay_details["members"],
            ({"reason": "verify_replay_false", "skipped": True},),
        )
        self.assertTrue(all(
            result.passes
            for audit_id, result in audits.items()
            if audit_id != replay_id
        ))
        bundle = audit_bundle_record(audits)
        self.assertIs(bundle["all_pass"], False)
        self.assertEqual(bundle["audit_count"], 13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
