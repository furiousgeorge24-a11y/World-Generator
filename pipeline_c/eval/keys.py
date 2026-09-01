"""Strict hidden-key validation for perceptual panel cohorts."""

from __future__ import annotations

import re

from .verdicts import (
    MECHANISMS,
    PROMPT_CRITIQUE,
    PROMPT_LAYER_AUDIT,
    PROMPT_SWEEP,
    VerdictError,
)

PANEL_KEY_SCHEMA_ID = "urn:mapgen:pipeline-c:eval:panel-key:v1"
LAYER_PANEL_KEY_SCHEMA_ID = "urn:mapgen:pipeline-c:eval:layer-panel-key:v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")

CANDIDATE = "candidate"
CONTROL_FORMULAIC = "control_formulaic"
CONTROL_PROCESS = "control_process"

# What a working judge must return for each kind of hidden calibration panel.
# Candidates deliberately have no expected verdict: that is the open question.
CONTROL_EXPECTED_VERDICT = {
    CONTROL_FORMULAIC: "formula",
    CONTROL_PROCESS: "process",
}


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerdictError(f"{label} must be a non-empty string")
    return value


def _check_duplicate_groups(
    groups: dict[str, list[dict]], hashes: dict[str, list[dict]],
    agreeing_fields: tuple[str, ...],
) -> None:
    """Enforce that declared duplicates really are indistinguishable.

    A group must be byte-identical and identical in every respect the judge
    could otherwise use to tell its members apart.
    """
    for group, members in groups.items():
        if len(members) < 2:
            raise VerdictError(f"duplicate group {group!r} needs at least two panels")
        if len({member["stimulus_sha256"] for member in members}) != 1:
            raise VerdictError(f"duplicate group {group!r} is not byte-identical")
        for field in agreeing_fields:
            if len({member[field] for member in members}) != 1:
                raise VerdictError(f"duplicate group {group!r} changes {field}")

    for digest, members in hashes.items():
        if len(members) < 2:
            continue
        member_groups = {member["duplicate_group"] for member in members}
        if len(member_groups) != 1 or None in member_groups:
            raise VerdictError(
                f"repeated stimulus hash {digest} is not one declared duplicate group")


def validate_layer_panel_key(data: object) -> dict[str, object]:
    """Validate a layer-audit key, including its supervised answer side.

    Every calibration control must declare the mechanism that actually produced
    it. A candidate may declare `null` when the mechanism is genuinely unknown,
    in which case it is simply excluded from mechanism accuracy.
    """
    root_keys = {"schema_id", "schema_version", "prompt_id", "panels"}
    if not isinstance(data, dict) or set(data) != root_keys:
        raise VerdictError(f"layer panel key must contain exactly {sorted(root_keys)}")
    if (data["schema_id"] != LAYER_PANEL_KEY_SCHEMA_ID
            or data["schema_version"] != 1):
        raise VerdictError("layer panel key has an unsupported schema identity")
    if data["prompt_id"] != PROMPT_LAYER_AUDIT:
        raise VerdictError("layer panel key prompt_id must be the layer audit prompt")
    panels = data["panels"]
    if not isinstance(panels, list) or not panels:
        raise VerdictError("layer panel key panels must be a non-empty array")

    item_keys = {
        "panel", "hidden_kind", "source_id", "duplicate_group",
        "stimulus_sha256", "true_mechanism", "crop_factor", "window",
    }
    kinds = {CANDIDATE, CONTROL_FORMULAIC, CONTROL_PROCESS}
    by_panel: dict[int, dict] = {}
    groups: dict[str, list[dict]] = {}
    hashes: dict[str, list[dict]] = {}
    for index, item in enumerate(panels):
        if not isinstance(item, dict) or set(item) != item_keys:
            raise VerdictError(
                f"layer panel key item {index} must contain exactly {sorted(item_keys)}")
        panel = item["panel"]
        if type(panel) is not int or panel < 1 or panel in by_panel:
            raise VerdictError(f"layer panel key item {index} has an invalid panel number")
        kind = item["hidden_kind"]
        if kind not in kinds:
            raise VerdictError(f"panel {panel} has an invalid hidden_kind")
        _text(item["source_id"], f"panel {panel} source_id")
        mechanism = item["true_mechanism"]
        if mechanism is not None and mechanism not in MECHANISMS:
            raise VerdictError(f"panel {panel} true_mechanism is not a known mechanism")
        if kind != CANDIDATE and mechanism is None:
            raise VerdictError(
                f"panel {panel} is a calibration control and must declare its mechanism")
        if mechanism == "cannot_determine":
            raise VerdictError(
                f"panel {panel} true_mechanism may not be the judge's escape label")
        crop = item["crop_factor"]
        if type(crop) is not int or crop < 1:
            raise VerdictError(f"panel {panel} crop_factor must be a positive integer")
        window = item["window"]
        if not isinstance(window, dict) or set(window) != {"top", "left", "extent"}:
            raise VerdictError(f"panel {panel} window needs exactly top/left/extent")
        if any(type(window[field]) is not int for field in ("top", "left", "extent")):
            raise VerdictError(f"panel {panel} window values must be integers")
        if window["top"] < 0 or window["left"] < 0 or window["extent"] < 1:
            raise VerdictError(f"panel {panel} window is out of range")
        group = item["duplicate_group"]
        if group is not None:
            _text(group, f"panel {panel} duplicate_group")
            groups.setdefault(group, []).append(item)
        digest = item["stimulus_sha256"]
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise VerdictError(f"panel {panel} has an invalid stimulus_sha256")
        hashes.setdefault(digest, []).append(item)
        by_panel[panel] = item

    _check_duplicate_groups(
        groups, hashes,
        ("hidden_kind", "source_id", "true_mechanism", "crop_factor"))
    for group, members in groups.items():
        if len({(member["window"]["top"], member["window"]["left"],
                 member["window"]["extent"]) for member in members}) != 1:
            raise VerdictError(f"duplicate group {group!r} changes window")

    if not any(item["hidden_kind"] == CONTROL_FORMULAIC for item in by_panel.values()):
        raise VerdictError("a layer audit batch needs at least one formulaic control")
    if not any(item["hidden_kind"] == CONTROL_PROCESS for item in by_panel.values()):
        raise VerdictError("a layer audit batch needs at least one process control")

    return {
        "prompt_id": data["prompt_id"],
        "by_panel": by_panel,
        "duplicate_groups": {
            group: sorted(member["panel"] for member in members)
            for group, members in sorted(groups.items())
        },
        "expected_verdicts": {
            panel: CONTROL_EXPECTED_VERDICT[item["hidden_kind"]]
            for panel, item in by_panel.items()
            if item["hidden_kind"] != CANDIDATE
        },
    }


def validate_panel_key(data: object) -> dict[str, object]:
    """Validate a critique/sweep key and its byte-identical duplicate groups."""
    root_keys = {"schema_id", "schema_version", "prompt_id", "panels"}
    if not isinstance(data, dict) or set(data) != root_keys:
        raise VerdictError(f"panel key must contain exactly {sorted(root_keys)}")
    if data["schema_id"] != PANEL_KEY_SCHEMA_ID or data["schema_version"] != 1:
        raise VerdictError("panel key has an unsupported schema identity")
    if data["prompt_id"] not in (PROMPT_CRITIQUE, PROMPT_SWEEP):
        raise VerdictError("panel key prompt_id must be a panel prompt")
    panels = data["panels"]
    if not isinstance(panels, list) or not panels:
        raise VerdictError("panel key panels must be a non-empty array")

    item_keys = {
        "panel", "hidden_kind", "source_id", "duplicate_group",
        "stimulus_sha256",
    }
    by_panel: dict[int, dict] = {}
    groups: dict[str, list[dict]] = {}
    hashes: dict[str, list[dict]] = {}
    for index, item in enumerate(panels):
        if not isinstance(item, dict) or set(item) != item_keys:
            raise VerdictError(
                f"panel key item {index} must contain exactly {sorted(item_keys)}")
        panel = item["panel"]
        if type(panel) is not int or panel < 1 or panel in by_panel:
            raise VerdictError(f"panel key item {index} has an invalid panel number")
        if item["hidden_kind"] not in ("candidate", "reference"):
            raise VerdictError(f"panel {panel} has an invalid hidden_kind")
        _text(item["source_id"], f"panel {panel} source_id")
        group = item["duplicate_group"]
        if group is not None:
            _text(group, f"panel {panel} duplicate_group")
            groups.setdefault(group, []).append(item)
        digest = item["stimulus_sha256"]
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise VerdictError(f"panel {panel} has an invalid stimulus_sha256")
        hashes.setdefault(digest, []).append(item)
        by_panel[panel] = item

    _check_duplicate_groups(groups, hashes, ("hidden_kind", "source_id"))

    return {
        "prompt_id": data["prompt_id"],
        "by_panel": by_panel,
        "duplicate_groups": {
            group: [member["panel"] for member in members]
            for group, members in sorted(groups.items())
        },
    }
