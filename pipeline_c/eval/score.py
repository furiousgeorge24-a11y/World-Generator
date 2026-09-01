"""Mechanical 2AFC scoring; it never decides whether morphology is natural."""

from __future__ import annotations

import itertools
from collections import Counter

from .keys import validate_panel_key
from .verdicts import (
    PROMPT_2AFC,
    PROMPT_CRITIQUE,
    PROMPT_SWEEP,
    VerdictError,
    validate,
)


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def score_2afc(key: list[dict], verdicts: dict[str, dict[int, dict]]) -> dict:
    """Score validated verdicts against a hidden key.

    Critique and sweep responses intentionally have no automatic aggregate
    quality score: they become an evidence-verified work list for the author.
    """
    if not isinstance(verdicts, dict) or len(verdicts) < 2:
        raise VerdictError("2AFC scoring requires at least two judge submissions")
    if any(not isinstance(judge, str) or not judge.strip() for judge in verdicts):
        raise VerdictError("judge identifiers must be non-empty strings")
    if not isinstance(key, list):
        raise VerdictError("2AFC key must be an array")
    by_trial = {}
    for item in key:
        if not isinstance(item, dict):
            raise VerdictError("every 2AFC key row must be an object")
        if set(item) != {"trial", "kind", "reference_side"}:
            raise VerdictError("every 2AFC key row needs exactly trial/kind/reference_side")
        trial = item["trial"]
        if type(trial) is not int or trial < 1 or trial in by_trial:
            raise VerdictError(
                "2AFC key trial numbers must be unique positive integers")
        if item["kind"] not in ("reference_vs_candidate", "calibration"):
            raise VerdictError("invalid 2AFC key kind")
        if item["kind"] == "reference_vs_candidate":
            if item["reference_side"] not in ("A", "B"):
                raise VerdictError("candidate arms need an A/B reference_side")
        elif item["reference_side"] is not None:
            raise VerdictError("calibration arms need null reference_side")
        by_trial[trial] = item
    if not by_trial:
        raise VerdictError("2AFC key is empty")
    expected = set(by_trial)
    validated_verdicts = {}
    for judge, rows in verdicts.items():
        if not isinstance(rows, dict):
            raise VerdictError(f"judge {judge} verdicts must be keyed rows")
        if set(rows) != expected:
            raise VerdictError(f"judge {judge} does not cover the complete key")
        validated_verdicts[judge] = validate(
            PROMPT_2AFC, [rows[number] for number in sorted(rows)], expected)
    verdicts = validated_verdicts

    candidate_ids = [n for n, row in by_trial.items()
                     if row["kind"] == "reference_vs_candidate"]
    calibration_ids = [n for n, row in by_trial.items()
                       if row["kind"] == "calibration"]
    if not candidate_ids or not calibration_ids:
        raise VerdictError(
            "2AFC key needs candidate arms and reference/reference calibration arms")
    per_judge = {}
    for judge, rows in sorted(verdicts.items()):
        valid = [n for n in candidate_ids if not rows[n]["void"]]
        correct = sum(
            rows[n]["pick"] == by_trial[n]["reference_side"] for n in valid)
        calibration = [rows[n] for n in calibration_ids if not rows[n]["void"]]
        per_judge[judge] = {
            "correct_reference_picks": correct,
            "valid_candidate_arms": len(valid),
            "candidate_accuracy": correct / len(valid) if valid else None,
            "candidate_voids": len(candidate_ids) - len(valid),
            "candidate_mean_confidence": _mean(
                [rows[n]["confidence"] for n in valid]),
            "calibration_voids": len(calibration_ids) - len(calibration),
            "calibration_mean_confidence": _mean(
                [row["confidence"] for row in calibration]),
            "calibration_high_confidence_count": sum(
                row["confidence"] >= 4 for row in calibration),
        }

    agreements = comparisons = 0
    for first, second in itertools.combinations(sorted(verdicts), 2):
        for trial in candidate_ids:
            a, b = verdicts[first][trial], verdicts[second][trial]
            if a["void"] or b["void"]:
                continue
            comparisons += 1
            agreements += a["pick"] == b["pick"]
    return {
        "authority": (
            "mechanical discrimination evidence only; perceptual claims "
            "require verification and the author decides acceptance"),
        "judge_count": len(verdicts),
        "per_judge": per_judge,
        "pairwise_agreements": agreements,
        "pairwise_comparisons": comparisons,
        "pairwise_agreement": agreements / comparisons if comparisons else None,
    }


def _critique_signature(row: dict) -> dict[str, object]:
    severity = Counter(claim["severity"] for claim in row["done_poorly"])
    return {
        "severity_counts": {
            value: severity.get(value, 0) for value in ("A", "B", "C", "D")
        },
        "done_poorly_count": len(row["done_poorly"]),
        "done_well_count": len(row["done_well"]),
        "cannot_identify_count": len(row["cannot_identify"]),
    }


def _sweep_signature(row: dict) -> dict[str, str]:
    return {
        dimension: row[dimension]["assessment"]
        for dimension in (
            "target_continuity", "fragmentation_response",
            "land_amount_leakage", "naturalness",
        )
    }


def score_duplicate_reliability(
    panel_key: dict, verdicts: dict[str, dict[int, dict]]
) -> dict[str, object]:
    """Mechanically compare responses to declared byte-identical panels.

    Matching bucket counts or assessment labels are only a coarse reliability
    signal. Semantic consistency of location/evidence claims remains a manual,
    evidence-based verification task.
    """
    key = validate_panel_key(panel_key)
    if not isinstance(verdicts, dict) or len(verdicts) < 2:
        raise VerdictError(
            "duplicate reliability scoring requires at least two judge submissions")
    if any(not isinstance(judge, str) or not judge.strip() for judge in verdicts):
        raise VerdictError("judge identifiers must be non-empty strings")
    if not key["duplicate_groups"]:
        raise VerdictError(
            "duplicate reliability scoring requires a declared duplicate group")
    expected = set(key["by_panel"])
    validated_verdicts = {}
    for judge, rows in verdicts.items():
        if not isinstance(rows, dict):
            raise VerdictError(f"judge {judge} verdicts must be keyed rows")
        if set(rows) != expected:
            raise VerdictError(f"judge {judge} does not cover the complete panel key")
        validated_verdicts[judge] = validate(
            key["prompt_id"], [rows[number] for number in sorted(rows)], expected)
    verdicts = validated_verdicts
    signature = (
        _critique_signature if key["prompt_id"] == PROMPT_CRITIQUE
        else _sweep_signature
    )
    groups = {}
    for group, panels in key["duplicate_groups"].items():
        anchor = panels[0]
        per_judge = {}
        for judge, rows in sorted(verdicts.items()):
            anchor_signature = signature(rows[anchor])
            comparisons = []
            for panel in panels[1:]:
                other_signature = signature(rows[panel])
                comparisons.append({
                    "panels": [anchor, panel],
                    "mechanical_signature_match": (
                        anchor_signature == other_signature),
                    "anchor_signature": anchor_signature,
                    "other_signature": other_signature,
                    "semantic_consistency": "requires manual evidence verification",
                })
            per_judge[judge] = comparisons
        groups[group] = {
            "panels": panels,
            "stimulus_sha256": key["by_panel"][anchor]["stimulus_sha256"],
            "per_judge": per_judge,
        }
    return {
        "authority": (
            "mechanical duplicate reliability only; cited claims require "
            "manual semantic verification"),
        "judge_count": len(verdicts),
        "duplicate_groups": groups,
    }
