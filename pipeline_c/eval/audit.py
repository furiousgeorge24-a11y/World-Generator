"""Mechanical scoring for the single-image layer audit.

This module never says a layer is acceptable. It answers three narrower
questions that have checkable answers:

1. Did the judge work at all? Known-formulaic controls must be caught and
   known-process controls must not be condemned. Failing either voids the
   batch and discards its verdict on the candidates.
2. Can the judge name mechanisms it is shown? The true mechanism of an
   intermediate view is known, so this is scored rather than trusted.
3. Is the judge stable? Byte-identical panels under different numbers should
   draw the same call.

Everything it emits is evidence for a person to read. A batch that is not void
is not an approval; it only means nothing in it was caught.
"""

from __future__ import annotations

from collections import Counter

from .keys import CANDIDATE, CONTROL_FORMULAIC, CONTROL_PROCESS, validate_layer_panel_key
from .verdicts import PROMPT_LAYER_AUDIT, VerdictError, validate

AUTHORITY = (
    "calibration and consistency evidence only; a non-void batch is not an "
    "approval, and the author decides acceptance"
)


def _rows_in_order(verdicts: object, expected: set[int]) -> dict[int, dict]:
    if isinstance(verdicts, dict):
        verdicts = [verdicts[number] for number in sorted(verdicts)]
    return validate(PROMPT_LAYER_AUDIT, verdicts, expected)


def score_layer_audit(
    panel_key: object, verdicts: object, *, judge_id: str = "judge"
) -> dict[str, object]:
    """Score one judge's audit of one batch against its hidden key."""
    if not isinstance(judge_id, str) or not judge_id.strip():
        raise VerdictError("judge identifier must be a non-empty string")
    key = validate_layer_panel_key(panel_key)
    by_panel = key["by_panel"]
    rows = _rows_in_order(verdicts, set(by_panel))

    controls = []
    void_reasons = []
    for panel in sorted(key["expected_verdicts"]):
        entry, row = by_panel[panel], rows[panel]
        expected = key["expected_verdicts"][panel]
        got = row["verdict"]
        controls.append({
            "panel": panel,
            "source_id": entry["source_id"],
            "hidden_kind": entry["hidden_kind"],
            "crop_factor": entry["crop_factor"],
            "expected_verdict": expected,
            "verdict": got,
            "mechanism": row["mechanism"]["label"],
            "true_mechanism": entry["true_mechanism"],
            "correct": got == expected,
        })
        # Asymmetric on purpose. The instrument exists to catch formulas, so
        # missing a known one is disqualifying; being unsure about a known
        # process panel is merely unhelpful, but condemning one is not.
        if entry["hidden_kind"] == CONTROL_FORMULAIC and got != "formula":
            void_reasons.append(
                f"panel {panel} is a known formulaic control ({entry['source_id']}) "
                f"and the judge returned {got!r}")
        if entry["hidden_kind"] == CONTROL_PROCESS and got == "formula":
            void_reasons.append(
                f"panel {panel} is a known process control ({entry['source_id']}) "
                "and the judge called it formulaic")

    scored = [
        {
            "panel": panel,
            "source_id": by_panel[panel]["source_id"],
            "hidden_kind": by_panel[panel]["hidden_kind"],
            "true_mechanism": by_panel[panel]["true_mechanism"],
            "claimed_mechanism": rows[panel]["mechanism"]["label"],
            "confidence": rows[panel]["mechanism"]["confidence"],
            "correct": (rows[panel]["mechanism"]["label"]
                        == by_panel[panel]["true_mechanism"]),
        }
        for panel in sorted(by_panel)
        if by_panel[panel]["true_mechanism"] is not None
    ]
    correct = sum(item["correct"] for item in scored)
    abstained = sum(
        item["claimed_mechanism"] == "cannot_determine" for item in scored)

    duplicates = {}
    for group, panels in key["duplicate_groups"].items():
        calls = [rows[panel]["verdict"] for panel in panels]
        labels = [rows[panel]["mechanism"]["label"] for panel in panels]
        duplicates[group] = {
            "panels": panels,
            "verdicts": calls,
            "mechanisms": labels,
            "verdict_agrees": len(set(calls)) == 1,
            "mechanism_agrees": len(set(labels)) == 1,
        }

    candidates: dict[str, dict] = {}
    for panel in sorted(by_panel):
        entry = by_panel[panel]
        if entry["hidden_kind"] != CANDIDATE:
            continue
        row = rows[panel]
        record = candidates.setdefault(entry["source_id"], {
            "panels": [], "verdicts": [], "mechanisms": [], "closures": [],
            "regularity_kinds": [], "predictions_to_verify": [],
        })
        record["panels"].append(panel)
        record["verdicts"].append(row["verdict"])
        record["mechanisms"].append(row["mechanism"]["label"])
        record["closures"].append(row["generating_rule"]["closure"])
        record["regularity_kinds"].extend(
            claim["kind"] for claim in row["regularities"])
        forecast = row["off_frame_prediction"]
        if forecast["predictable"]:
            record["predictions_to_verify"].append({
                "panel": panel,
                "crop_factor": entry["crop_factor"],
                "prediction": forecast["prediction"],
                "period_px": forecast["period_px"],
                "orientation_deg": forecast["orientation_deg"],
            })

    for source_id, record in candidates.items():
        record["regularity_kinds"] = dict(
            sorted(Counter(record["regularity_kinds"]).items()))
        record["called_formula_on"] = [
            panel for panel, call in zip(record["panels"], record["verdicts"])
            if call == "formula"
        ]
        record["generating_rule_closed_on"] = [
            panel for panel, closure in zip(record["panels"], record["closures"])
            if closure == "closed"
        ]

    return {
        "authority": AUTHORITY,
        "judge_id": judge_id,
        "judge_count": 1,
        "limitations": [
            "one judge; agreement across independent judges is not measured",
            "off-frame predictions are claims, not measurements, until the "
            "adjacent region is rendered and compared",
        ],
        "batch_void": bool(void_reasons),
        "void_reasons": void_reasons,
        "controls": controls,
        "control_summary": {
            "formulaic_total": sum(
                item["hidden_kind"] == CONTROL_FORMULAIC for item in controls),
            "formulaic_caught": sum(
                item["hidden_kind"] == CONTROL_FORMULAIC and item["correct"]
                for item in controls),
            "process_total": sum(
                item["hidden_kind"] == CONTROL_PROCESS for item in controls),
            "process_cleared": sum(
                item["hidden_kind"] == CONTROL_PROCESS and item["correct"]
                for item in controls),
        },
        "mechanism": {
            "scored_panels": len(scored),
            "correct": correct,
            "accuracy": correct / len(scored) if scored else None,
            "abstentions": abstained,
            "per_panel": scored,
        },
        "duplicates": duplicates,
        "candidates": candidates,
    }


__all__ = ["AUTHORITY", "score_layer_audit"]
