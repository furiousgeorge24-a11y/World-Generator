"""Strict hidden-key validation for perceptual panel cohorts."""

from __future__ import annotations

import re

from .verdicts import PROMPT_CRITIQUE, PROMPT_SWEEP, VerdictError

PANEL_KEY_SCHEMA_ID = "urn:mapgen:pipeline-c:eval:panel-key:v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerdictError(f"{label} must be a non-empty string")
    return value


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

    for group, members in groups.items():
        if len(members) < 2:
            raise VerdictError(f"duplicate group {group!r} needs at least two panels")
        if len({member["stimulus_sha256"] for member in members}) != 1:
            raise VerdictError(f"duplicate group {group!r} is not byte-identical")
        if len({member["hidden_kind"] for member in members}) != 1:
            raise VerdictError(f"duplicate group {group!r} changes hidden kind")
        if len({member["source_id"] for member in members}) != 1:
            raise VerdictError(f"duplicate group {group!r} changes source identity")

    for digest, members in hashes.items():
        if len(members) < 2:
            continue
        member_groups = {member["duplicate_group"] for member in members}
        if len(member_groups) != 1 or None in member_groups:
            raise VerdictError(
                f"repeated stimulus hash {digest} is not one declared duplicate group")

    return {
        "prompt_id": data["prompt_id"],
        "by_panel": by_panel,
        "duplicate_groups": {
            group: [member["panel"] for member in members]
            for group, members in sorted(groups.items())
        },
    }
