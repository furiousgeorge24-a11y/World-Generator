"""Strict stdlib validators for the versioned perceptual verdict schemas."""

from __future__ import annotations

import json
from pathlib import Path

PROMPT_2AFC = "land_origin_2afc_v1"
PROMPT_CRITIQUE = "land_origin_critique_v1"
PROMPT_SWEEP = "land_controls_sweep_v1"
PROMPT_LAYER_AUDIT = "layer_audit_v1"
PROMPT_IDS = {PROMPT_2AFC, PROMPT_CRITIQUE, PROMPT_SWEEP, PROMPT_LAYER_AUDIT}
SEVERITIES = {"A", "B", "C", "D"}
ASSESSMENTS = {"supports", "concern", "cannot_assess"}

# The layer audit's fixed vocabularies. The mechanism list is closed on purpose:
# the true mechanism of an intermediate view is always known, so a forced choice
# from a fixed list can be scored instead of merely read.
MECHANISMS = {
    "periodic_waves",
    "distance_or_cost_field",
    "filtered_noise",
    "iterative_growth",
    "thresholded_field",
    "cannot_determine",
}
REGULARITY_KINDS = {
    "grid_locking",
    "periodicity",
    "constant_scale",
    "constant_curvature",
    "symmetry",
    "straight_runs",
    "degenerate_morphology",
    "seams",
}
VERDICT_CALLS = {"process", "formula", "undecided"}
CLOSURES = {"closed", "partial", "open"}


class VerdictError(ValueError):
    """A judge response is incomplete, ambiguous, or schema-incompatible."""


def _closed_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerdictError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _invalid_json_constant(value: str):
    raise VerdictError(f"JSON contains non-standard numeric constant {value}")


def read_json(path: str | Path) -> object:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_closed_json_object,
            parse_constant=_invalid_json_constant,
        )
    except VerdictError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerdictError(f"invalid JSON file {path}: {exc}") from exc


def _exact(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict):
        raise VerdictError(f"{label} must be an object")
    if set(value) != keys:
        raise VerdictError(
            f"{label} must contain exactly {sorted(keys)}; got {sorted(value)}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerdictError(f"{label} must be a non-empty string")
    return value


def _numbered_rows(data: object, expected: set[int], field: str) -> dict[int, dict]:
    if not isinstance(expected, set) or not expected:
        raise VerdictError(f"expected {field} set must be non-empty")
    if any(type(number) is not int or number < 1 for number in expected):
        raise VerdictError(f"expected {field} set contains invalid identifiers")
    if not isinstance(data, list):
        raise VerdictError("top level must be a JSON array")
    if not data:
        raise VerdictError("top-level verdict array must be non-empty")
    rows: dict[int, dict] = {}
    encountered: list[int] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise VerdictError(f"item {index} must be an object")
        number = item.get(field)
        if isinstance(number, bool) or not isinstance(number, int):
            raise VerdictError(f"item {index} has an invalid {field}")
        if number not in expected:
            raise VerdictError(f"item {index} has unexpected {field} {number}")
        if number in rows:
            raise VerdictError(f"duplicate {field} {number}")
        rows[number] = item
        encountered.append(number)
    missing = sorted(expected - set(rows))
    if missing:
        raise VerdictError(f"missing {field} values {missing}")
    if encountered != sorted(expected):
        raise VerdictError(f"{field} rows must be in ascending identifier order")
    return rows


def validate_2afc(data: object, expected: set[int]) -> dict[int, dict]:
    rows = _numbered_rows(data, expected, "trial")
    keys = {"trial", "void", "pick", "confidence", "evidence", "void_reason"}
    for trial, item in rows.items():
        _exact(item, keys, f"trial {trial}")
        if type(item["void"]) is not bool:
            raise VerdictError(f"trial {trial} void must be boolean")
        if item["void"]:
            if any(item[field] is not None for field in ("pick", "confidence", "evidence")):
                raise VerdictError(f"void trial {trial} needs null pick/confidence/evidence")
            _text(item["void_reason"], f"trial {trial} void_reason")
        else:
            if item["pick"] not in ("A", "B"):
                raise VerdictError(f"trial {trial} pick must be A or B")
            confidence = item["confidence"]
            if type(confidence) is not int or not 1 <= confidence <= 5:
                raise VerdictError(f"trial {trial} confidence must be 1..5")
            evidence = _exact(
                item["evidence"], {"A", "B", "comparison"},
                f"trial {trial} evidence")
            for field in ("A", "B", "comparison"):
                _text(evidence[field], f"trial {trial} evidence.{field}")
            if item["void_reason"] is not None:
                raise VerdictError(f"valid trial {trial} needs null void_reason")
    return rows


def _claim(value: object, label: str, *, severity: bool) -> None:
    keys = {"what", "where", "evidence", "severity"} if severity else {
        "what", "where", "evidence"
    }
    value = _exact(value, keys, label)
    for field in ("what", "where", "evidence"):
        _text(value[field], f"{label}.{field}")
    if severity and value["severity"] not in SEVERITIES:
        raise VerdictError(f"{label}.severity must be A, B, C, or D")


def validate_critique(data: object, expected: set[int]) -> dict[int, dict]:
    rows = _numbered_rows(data, expected, "panel")
    keys = {"panel", "done_poorly", "done_well", "cannot_identify"}
    caps = {"done_poorly": 5, "done_well": 5, "cannot_identify": 3}
    for panel, item in rows.items():
        _exact(item, keys, f"panel {panel}")
        for bucket, cap in caps.items():
            claims = item[bucket]
            if not isinstance(claims, list) or len(claims) > cap:
                raise VerdictError(f"panel {panel} {bucket} must have at most {cap} claims")
            for index, claim in enumerate(claims):
                _claim(
                    claim, f"panel {panel} {bucket}[{index}]",
                    severity=bucket == "done_poorly")
    return rows


def _assessment(value: object, label: str) -> None:
    value = _exact(value, {"assessment", "where", "evidence"}, label)
    if value["assessment"] not in ASSESSMENTS:
        raise VerdictError(f"{label}.assessment is invalid")
    _text(value["where"], f"{label}.where")
    _text(value["evidence"], f"{label}.evidence")


def validate_sweep(data: object, expected: set[int]) -> dict[int, dict]:
    rows = _numbered_rows(data, expected, "panel")
    dimensions = {
        "target_continuity", "fragmentation_response",
        "land_amount_leakage", "naturalness",
    }
    keys = {"panel"} | dimensions
    for panel, item in rows.items():
        _exact(item, keys, f"panel {panel}")
        for dimension in sorted(dimensions):
            _assessment(item[dimension], f"panel {panel} {dimension}")
    return rows


def _off_frame_prediction(value: object, label: str) -> None:
    forecast = _exact(
        value, {"predictable", "prediction", "period_px", "orientation_deg"}, label)
    if type(forecast["predictable"]) is not bool:
        raise VerdictError(f"{label}.predictable must be boolean")
    if not forecast["predictable"]:
        if any(forecast[field] is not None
               for field in ("prediction", "period_px", "orientation_deg")):
            raise VerdictError(
                f"{label} claims no prediction but supplies one")
        return
    _text(forecast["prediction"], f"{label}.prediction")
    period = forecast["period_px"]
    if period is not None and (type(period) is not int or period < 2):
        raise VerdictError(f"{label}.period_px must be null or an integer >= 2")
    angle = forecast["orientation_deg"]
    if angle is None:
        return
    if isinstance(angle, bool) or not isinstance(angle, (int, float)):
        raise VerdictError(f"{label}.orientation_deg must be null or a number")
    if not 0.0 <= float(angle) < 180.0:
        raise VerdictError(f"{label}.orientation_deg must lie in [0, 180)")


def validate_layer_audit(data: object, expected: set[int]) -> dict[int, dict]:
    """Validate single-image mechanism audit rows.

    Nothing here decides whether a panel is acceptable. It only guarantees the
    judge answered every forced question in a form that can be scored against a
    hidden key.
    """
    rows = _numbered_rows(data, expected, "panel")
    keys = {
        "panel", "verdict", "generating_rule", "off_frame_prediction",
        "mechanism", "regularities",
    }
    for panel, item in rows.items():
        label = f"panel {panel}"
        _exact(item, keys, label)
        if item["verdict"] not in VERDICT_CALLS:
            raise VerdictError(
                f"{label}.verdict must be one of {sorted(VERDICT_CALLS)}")

        rule = _exact(
            item["generating_rule"], {"description", "closure"},
            f"{label} generating_rule")
        _text(rule["description"], f"{label} generating_rule.description")
        if rule["closure"] not in CLOSURES:
            raise VerdictError(f"{label} generating_rule.closure is invalid")

        _off_frame_prediction(
            item["off_frame_prediction"], f"{label} off_frame_prediction")

        mechanism = _exact(
            item["mechanism"], {"label", "confidence", "evidence"},
            f"{label} mechanism")
        if mechanism["label"] not in MECHANISMS:
            raise VerdictError(f"{label} mechanism.label is invalid")
        confidence = mechanism["confidence"]
        if type(confidence) is not int or not 1 <= confidence <= 5:
            raise VerdictError(f"{label} mechanism.confidence must be 1..5")
        _text(mechanism["evidence"], f"{label} mechanism.evidence")

        claims = item["regularities"]
        if not isinstance(claims, list) or len(claims) > 5:
            raise VerdictError(f"{label} regularities must have at most 5 claims")
        for index, claim in enumerate(claims):
            entry = _exact(
                claim, {"what", "where", "evidence", "kind"},
                f"{label} regularities[{index}]")
            for field in ("what", "where", "evidence"):
                _text(entry[field], f"{label} regularities[{index}].{field}")
            if entry["kind"] not in REGULARITY_KINDS:
                raise VerdictError(f"{label} regularities[{index}].kind is invalid")
    return rows


def validate(prompt_id: str, data: object, expected: set[int]) -> dict[int, dict]:
    if prompt_id == PROMPT_2AFC:
        return validate_2afc(data, expected)
    if prompt_id == PROMPT_CRITIQUE:
        return validate_critique(data, expected)
    if prompt_id == PROMPT_SWEEP:
        return validate_sweep(data, expected)
    if prompt_id == PROMPT_LAYER_AUDIT:
        return validate_layer_audit(data, expected)
    raise VerdictError(f"unknown prompt_id: {prompt_id}")
