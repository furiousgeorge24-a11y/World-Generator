"""Focused checks for the single-image layer audit.

The audit's whole claim is that a judge can be scored rather than trusted, so
these checks concentrate on the two places that claim could quietly fail: the
hidden key refusing to describe a batch it cannot calibrate, and the scorer
voiding a batch whose judge missed a planted formula.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.audit import score_layer_audit  # noqa: E402
from eval.controls import FORMULAIC, PROCESS, build_controls  # noqa: E402
from eval.keys import (  # noqa: E402
    LAYER_PANEL_KEY_SCHEMA_ID,
    validate_layer_panel_key,
)
from eval.stimulus import (  # noqa: E402
    HIDDEN_ROOT,
    KEY_NAME,
    PLAN_NAME,
    Source,
    build_batch,
    plan_chunks,
)
from eval.verdicts import (  # noqa: E402
    PROMPT_LAYER_AUDIT,
    VerdictError,
    validate,
)

PASS: list[str] = []


def check(name: str, condition: object) -> None:
    if not condition:
        raise AssertionError(name)
    PASS.append(name)
    print(f"PASS  {name}")


def rejects(name: str, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except VerdictError:
        check(name, True)
    else:
        raise AssertionError(name)


def row(panel: int, *, verdict: str = "process", mechanism: str = "filtered_noise",
        closure: str = "open", predictable: bool = False) -> dict:
    return {
        "panel": panel,
        "verdict": verdict,
        "generating_rule": {
            "description": "Correlated random values over a length scale.",
            "closure": closure,
        },
        "off_frame_prediction": {
            "predictable": predictable,
            "prediction": "The stripe set repeats." if predictable else None,
            "period_px": 120 if predictable else None,
            "orientation_deg": 45.0 if predictable else None,
        },
        "mechanism": {
            "label": mechanism,
            "confidence": 4,
            "evidence": "Blotches vary in size with no repeat anywhere.",
        },
        "regularities": [],
    }


def key_panel(panel: int, kind: str, source: str, mechanism: str | None, *,
              digest: str | None = None, group: str | None = None,
              crop: int = 1, top: int = 0) -> dict:
    return {
        "panel": panel,
        "hidden_kind": kind,
        "source_id": source,
        "duplicate_group": group,
        "stimulus_sha256": digest or f"{panel:064d}",
        "true_mechanism": mechanism,
        "crop_factor": crop,
        "window": {"top": top, "left": 0, "extent": 512 // crop},
    }


def make_key(panels: list[dict]) -> dict:
    return {
        "schema_id": LAYER_PANEL_KEY_SCHEMA_ID,
        "schema_version": 1,
        "prompt_id": PROMPT_LAYER_AUDIT,
        "panels": panels,
    }


BASE_PANELS = [
    key_panel(1, "candidate", "arrival", None),
    key_panel(2, "control_formulaic", "triangle_lattice", "periodic_waves"),
    key_panel(3, "control_process", "fractal_noise", "filtered_noise"),
]


def check_prompt_and_verdicts() -> None:
    fence = chr(96) * 3
    text = (ROOT / "eval" / "prompts" / f"{PROMPT_LAYER_AUDIT}.md").read_text(
        encoding="utf-8")
    blocks = re.findall(fence + r"json\s*(.*?)\s*" + fence, text, re.DOTALL)
    check("the audit prompt carries exactly one JSON example", len(blocks) == 1)
    example = json.loads(blocks[0])
    rows = validate(PROMPT_LAYER_AUDIT, example, {1})
    check("the prompt example is a literal valid verdict",
          rows[1]["verdict"] == "formula")
    check("the prompt example demonstrates a committed prediction",
          rows[1]["off_frame_prediction"]["predictable"]
          and rows[1]["off_frame_prediction"]["period_px"] == 120)

    check("a plain verdict validates", bool(validate(PROMPT_LAYER_AUDIT, [row(1)], {1})))

    bad = row(1)
    bad["unexpected"] = "no"
    rejects("verdict rejects extra fields", validate, PROMPT_LAYER_AUDIT, [bad], {1})

    bad = row(1, mechanism="vibes")
    rejects("verdict rejects an invented mechanism label",
            validate, PROMPT_LAYER_AUDIT, [bad], {1})

    bad = row(1)
    bad["verdict"] = "probably"
    rejects("verdict rejects an invented call",
            validate, PROMPT_LAYER_AUDIT, [bad], {1})

    bad = row(1, closure="mostly")
    rejects("verdict rejects an invented closure",
            validate, PROMPT_LAYER_AUDIT, [bad], {1})

    bad = row(1)
    bad["mechanism"]["confidence"] = 9
    rejects("verdict rejects confidence outside 1..5",
            validate, PROMPT_LAYER_AUDIT, [bad], {1})

    bad = row(1)
    bad["off_frame_prediction"]["prediction"] = "it continues"
    rejects("an unpredictable continuation may not carry a prediction",
            validate, PROMPT_LAYER_AUDIT, [bad], {1})

    bad = row(1, predictable=True)
    bad["off_frame_prediction"]["orientation_deg"] = 270.0
    rejects("orientation must lie in [0, 180)",
            validate, PROMPT_LAYER_AUDIT, [bad], {1})

    bad = row(1, predictable=True)
    bad["off_frame_prediction"]["period_px"] = 1
    rejects("a declared period must be larger than one pixel",
            validate, PROMPT_LAYER_AUDIT, [bad], {1})

    bad = row(1)
    bad["regularities"] = [
        {"what": "w", "where": "here", "evidence": "e", "kind": "periodicity"}
    ] * 6
    rejects("at most five regularities per panel",
            validate, PROMPT_LAYER_AUDIT, [bad], {1})

    bad = row(1)
    bad["regularities"] = [
        {"what": "w", "where": "here", "evidence": "e", "kind": "looks_odd"}
    ]
    rejects("regularity kinds come from the fixed vocabulary",
            validate, PROMPT_LAYER_AUDIT, [bad], {1})


def check_hidden_key() -> None:
    result = validate_layer_panel_key(make_key(copy.deepcopy(BASE_PANELS)))
    check("a calibrated key validates", set(result["by_panel"]) == {1, 2, 3})
    check("control panels carry an expected verdict",
          result["expected_verdicts"] == {2: "formula", 3: "process"})
    check("candidates carry no expected verdict",
          1 not in result["expected_verdicts"])

    panels = copy.deepcopy(BASE_PANELS)
    panels[1]["true_mechanism"] = None
    rejects("a control must declare the mechanism that produced it",
            validate_layer_panel_key, make_key(panels))

    panels = copy.deepcopy(BASE_PANELS)
    panels[1]["true_mechanism"] = "cannot_determine"
    rejects("the judge's escape label is not an answer key",
            validate_layer_panel_key, make_key(panels))

    rejects("a batch without a formulaic control cannot calibrate",
            validate_layer_panel_key,
            make_key([copy.deepcopy(BASE_PANELS[0]), copy.deepcopy(BASE_PANELS[2])]))
    rejects("a batch without a process control cannot calibrate",
            validate_layer_panel_key,
            make_key([copy.deepcopy(BASE_PANELS[0]), copy.deepcopy(BASE_PANELS[1])]))

    panels = copy.deepcopy(BASE_PANELS)
    panels[0]["hidden_kind"] = "control_maybe"
    rejects("hidden kinds come from the fixed vocabulary",
            validate_layer_panel_key, make_key(panels))

    shared = "ab" * 32
    panels = copy.deepcopy(BASE_PANELS) + [
        key_panel(4, "candidate", "arrival", None, digest=shared, group="g"),
        key_panel(5, "candidate", "arrival", None, digest=shared, group="g", top=7),
    ]
    rejects("a duplicate group may not shift its window",
            validate_layer_panel_key, make_key(panels))

    panels[4]["window"] = dict(panels[3]["window"])
    check("a genuine duplicate group validates",
          validate_layer_panel_key(make_key(panels))["duplicate_groups"] == {"g": [4, 5]})

    panels = copy.deepcopy(BASE_PANELS) + [
        key_panel(4, "candidate", "arrival", None, digest=shared),
        key_panel(5, "candidate", "arrival", None, digest=shared),
    ]
    rejects("a repeated stimulus must be a declared duplicate group",
            validate_layer_panel_key, make_key(panels))


def check_scoring() -> None:
    key = make_key(copy.deepcopy(BASE_PANELS))
    clean = [
        row(1, verdict="process", mechanism="iterative_growth"),
        row(2, verdict="formula", mechanism="periodic_waves"),
        row(3, verdict="process", mechanism="filtered_noise"),
    ]
    result = score_layer_audit(key, clean)
    check("a calibrated judge does not void the batch", not result["batch_void"])
    check("mechanism accuracy is scored only where truth is declared",
          result["mechanism"]["scored_panels"] == 2
          and result["mechanism"]["correct"] == 2)
    check("the candidate is reported without being approved",
          result["candidates"]["arrival"]["called_formula_on"] == [])
    check("scoring never claims authority to approve",
          "not an approval" in result["authority"])

    missed = copy.deepcopy(clean)
    missed[1] = row(2, verdict="process", mechanism="filtered_noise")
    result = score_layer_audit(key, missed)
    check("missing a planted formula voids the batch", result["batch_void"])
    check("the void reason names the control that was missed",
          "triangle_lattice" in result["void_reasons"][0])

    undecided = copy.deepcopy(clean)
    undecided[1] = row(2, verdict="undecided", mechanism="cannot_determine")
    check("being unsure about a planted formula also voids the batch",
          score_layer_audit(key, undecided)["batch_void"])

    overeager = copy.deepcopy(clean)
    overeager[2] = row(3, verdict="formula", mechanism="periodic_waves")
    result = score_layer_audit(key, overeager)
    check("condemning a known process control voids the batch too",
          result["batch_void"])
    check("an indiscriminate judge is called out by name",
          "fractal_noise" in result["void_reasons"][0])

    cautious = copy.deepcopy(clean)
    cautious[2] = row(3, verdict="undecided", mechanism="cannot_determine")
    check("being unsure about a process control is not disqualifying",
          not score_layer_audit(key, cautious)["batch_void"])

    flagging = copy.deepcopy(clean)
    flagging[0] = row(1, verdict="formula", mechanism="periodic_waves",
                      closure="closed", predictable=True)
    result = score_layer_audit(key, flagging)
    record = result["candidates"]["arrival"]
    check("a flagged candidate is reported with its panel",
          record["called_formula_on"] == [1])
    check("a closed generating rule is surfaced",
          record["generating_rule_closed_on"] == [1])
    check("a committed prediction is queued for verification",
          record["predictions_to_verify"][0]["period_px"] == 120)

    rejects("a verdict that does not cover the key is rejected",
            score_layer_audit, key, clean[:2])


def check_controls() -> None:
    controls = build_controls(11, 64)
    kinds = {control.kind for control in controls}
    check("controls supply both calibration sides", kinds == {FORMULAIC, PROCESS})
    check("every control declares a real mechanism",
          all(control.true_mechanism and control.true_mechanism != "cannot_determine"
              for control in controls))
    check("controls render as plain RGB rasters", all(
        control.rgb.shape == (64, 64, 3) and control.rgb.dtype == np.uint8
        for control in controls))
    repeat = build_controls(11, 64)
    check("controls are deterministic for a seed", all(
        np.array_equal(a.rgb, b.rgb) for a, b in zip(controls, repeat)))
    check("a different seed gives different controls", not all(
        np.array_equal(a.rgb, b.rgb)
        for a, b in zip(controls, build_controls(12, 64))))


def check_batch_build() -> None:
    rng = np.random.default_rng(3)
    sources = [
        Source("candidate_field", rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)),
        Source("planted_lattice",
               rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
               FORMULAIC, "periodic_waves"),
        Source("planted_noise",
               rng.integers(0, 255, (128, 128, 3), dtype=np.uint8),
               PROCESS, "filtered_noise"),
    ]
    with tempfile.TemporaryDirectory() as temporary:
        destination = Path(temporary) / "run" / "bundle"
        summary = build_batch(sources, destination, seed=5, panel_px=64,
                              crop_factors=(1, 2), duplicate_panels=1,
                              provenance={"note": "test"})
        check("every source contributes a panel at every crop factor",
              summary["panel_count"] == len(sources) * 2 + 1)

        packet = Path(summary["judge_packet"])
        panels = sorted(packet.glob("panel_*.png"))
        check("the packet holds only panels and the prompt",
              {path.name for path in packet.iterdir()}
              == {path.name for path in panels} | {"PROMPT.md"})
        check("the hidden key is not inside the judge packet",
              not (packet / KEY_NAME).exists())

        from PIL import Image
        images = [Image.open(path) for path in panels]
        check("every panel is the same size, so size reveals nothing",
              {image.size for image in images} == {(64, 64)})
        check("no panel carries readable metadata",
              all(not getattr(image, "text", {}) for image in images))

        key = json.loads(
            (destination / HIDDEN_ROOT / KEY_NAME).read_text(encoding="utf-8"))
        validated = validate_layer_panel_key(key)
        check("the emitted key validates against its own rules",
              len(validated["by_panel"]) == summary["panel_count"])
        groups = validated["duplicate_groups"]
        check("one duplicate group was declared", len(groups) == 1)
        members = next(iter(groups.values()))
        payloads = {(packet / f"panel_{number:02d}.png").read_bytes()
                    for number in members}
        check("declared duplicates are byte-identical", len(payloads) == 1)
        check("provenance is written to the hidden root",
              (destination / HIDDEN_ROOT / "provenance.json").exists())


def check_judging_plan() -> None:
    rng = np.random.default_rng(0)
    panels = [
        key_panel(1, "candidate", "arrival", None, digest="a" * 64, group="g"),
        key_panel(2, "candidate", "arrival", None, digest="a" * 64, group="g"),
        key_panel(3, "control_formulaic", "lattice", "periodic_waves"),
        key_panel(4, "control_formulaic", "radial", "distance_or_cost_field"),
        key_panel(5, "control_process", "noise", "filtered_noise"),
        key_panel(6, "control_process", "growth", "iterative_growth"),
        key_panel(7, "candidate", "resistance", None),
    ]
    plan = plan_chunks(panels, 2, rng)
    flat = sorted(number for chunk in plan for number in chunk)
    check("the plan covers every panel exactly once",
          flat == sorted(panel["panel"] for panel in panels))
    where = {number: index for index, chunk in enumerate(plan) for number in chunk}
    check("duplicate group members go to different judge calls",
          where[1] != where[2])
    for index, chunk in enumerate(plan):
        kinds = {panel["hidden_kind"] for panel in panels
                 if panel["panel"] in chunk}
        check(f"call {index} carries a full calibration pair",
              {"control_formulaic", "control_process"} <= kinds)

    try:
        plan_chunks(panels, 3, np.random.default_rng(0))
    except ValueError:
        check("a chunk count that starves calibration is refused", True)
    else:
        raise AssertionError("a chunk count that starves calibration is refused")

    crowded = [
        key_panel(1, "candidate", "arrival", None, digest="a" * 64, group="g"),
        key_panel(2, "candidate", "arrival", None, digest="a" * 64, group="g"),
        key_panel(3, "control_formulaic", "lattice", "periodic_waves"),
        key_panel(4, "control_process", "noise", "filtered_noise"),
    ]
    try:
        plan_chunks(crowded, 1, np.random.default_rng(0))
    except ValueError:
        check("a duplicate group larger than the chunk count is refused", True)
    else:
        raise AssertionError("a duplicate group larger than the chunk count is refused")


def main() -> None:
    print("== prompt and verdict validation ==")
    check_prompt_and_verdicts()
    print("\n== hidden key and calibration ==")
    check_hidden_key()
    print("\n== scoring and voiding ==")
    check_scoring()
    print("\n== calibration controls ==")
    check_controls()
    print("\n== batch assembly ==")
    check_batch_build()
    print("\n== judging plan ==")
    check_judging_plan()
    print(f"\n{len(PASS)}/{len(PASS)} checks passed")


if __name__ == "__main__":
    main()
