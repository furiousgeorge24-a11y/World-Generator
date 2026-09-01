"""Focused checks for the engine-independent Pipeline C eval scaffold."""

from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.bundle import (  # noqa: E402
    BundleError,
    create_staging_directory,
    load_and_verify_stage,
    make_stage_manifest,
    publish_staged_directory,
    safe_relative_path,
    sha256_file,
    validate_manifest_shape,
    write_stage_manifest,
)
from eval.keys import PANEL_KEY_SCHEMA_ID, validate_panel_key  # noqa: E402
from eval.metrics import (  # noqa: E402
    fragmentation_metrics,
    land_fraction,
    land_percent,
    monotonic_land_percentages,
    outer_ring_is_water,
    target_interval_percent,
    target_within_tolerance_percent,
)
from eval.score import score_2afc, score_duplicate_reliability  # noqa: E402
from eval.verdicts import (  # noqa: E402
    PROMPT_2AFC,
    PROMPT_CRITIQUE,
    PROMPT_SWEEP,
    VerdictError,
    read_json,
    validate,
)

PASS: list[str] = []


def check(name: str, condition: object) -> None:
    if not condition:
        raise AssertionError(name)
    PASS.append(name)
    print(f"PASS  {name}")


def rejects(name: str, error: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error:
        check(name, True)
    else:
        raise AssertionError(name)


def prompt_example(path: Path) -> object:
    matches = re.findall(
        r"```json\s*(.*?)\s*```", path.read_text(encoding="utf-8"), re.DOTALL)
    if len(matches) != 1:
        raise AssertionError(f"{path} must contain exactly one JSON example")
    return json.loads(matches[0])


def assert_strict_objects(node: object, location: str = "$") -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and node.get("additionalProperties") is not False:
            raise AssertionError(f"object schema is not strict at {location}")
        for key, value in node.items():
            assert_strict_objects(value, f"{location}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            assert_strict_objects(value, f"{location}/{index}")


def artifact_path_patterns(node: object) -> list[str]:
    patterns: list[str] = []
    if isinstance(node, dict):
        path_schema = node.get("properties", {}).get("path")
        if isinstance(path_schema, dict) and isinstance(path_schema.get("pattern"), str):
            patterns.append(path_schema["pattern"])
        for value in node.values():
            patterns.extend(artifact_path_patterns(value))
    elif isinstance(node, list):
        for value in node:
            patterns.extend(artifact_path_patterns(value))
    return patterns


def check_schemas_and_prompts() -> tuple[dict[int, dict], dict[int, dict], dict[int, dict]]:
    schema_dir = ROOT / "eval" / "schemas"
    schemas = {}
    for path in sorted(schema_dir.glob("*.schema.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        check(f"{path.name} uses Draft 2020-12",
              schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema")
        check(f"{path.name} has a stable ID",
              isinstance(schema.get("$id"), str) and schema["$id"].startswith("urn:mapgen:pipeline-c:"))
        assert_strict_objects(schema)
        schemas[path.name] = schema
    check("all seven scaffold schemas are present", set(schemas) == {
        "candidate_manifest_v1.schema.json",
        "eval_stage_manifest_v1.schema.json",
        "judge_submission_v1.schema.json",
        "land_controls_sweep_v1.schema.json",
        "land_origin_2afc_v1.schema.json",
        "land_origin_critique_v1.schema.json",
        "panel_key_v1.schema.json",
    })
    check("schema IDs are unique",
          len({schema["$id"] for schema in schemas.values()}) == len(schemas))
    path_patterns = [
        pattern for schema in schemas.values()
        for pattern in artifact_path_patterns(schema)
    ]
    check("artifact schemas reject non-canonical and absolute paths",
          len(path_patterns) >= 3 and all(
              re.fullmatch(pattern, "judge/file.json")
              and not any(re.fullmatch(pattern, unsafe) for unsafe in (
                  "C:/file.json", "/file.json", "./file.json",
                  "judge/../hidden/key.json", "judge//file.json",
                  "judge\\file.json",
              ))
              for pattern in path_patterns
          ))
    check("all verdict schemas forbid an empty top-level array", all(
        schemas[name].get("minItems") == 1 for name in (
            "land_controls_sweep_v1.schema.json",
            "land_origin_2afc_v1.schema.json",
            "land_origin_critique_v1.schema.json",
        )))

    prompts = ROOT / "eval" / "prompts"
    two = prompt_example(prompts / "land_origin_2afc_v1.md")
    critique = prompt_example(prompts / "land_origin_critique_v1.md")
    sweep = prompt_example(prompts / "land_controls_sweep_v1.md")
    two_rows = validate(PROMPT_2AFC, two, {1, 2})
    critique_rows = validate(PROMPT_CRITIQUE, critique, {1})
    sweep_rows = validate(PROMPT_SWEEP, sweep, {1})
    check("all prompt examples are literal valid JSON", True)
    check("2AFC example exercises valid and void forms",
          not two_rows[1]["void"] and two_rows[2]["void"])

    bad = copy.deepcopy(critique)
    bad[0]["unexpected"] = "not allowed"
    rejects("strict verdict validator rejects extra fields", VerdictError,
            validate, PROMPT_CRITIQUE, bad, {1})
    bad = copy.deepcopy(two)
    bad[0]["evidence"]["B"] = ""
    rejects("strict verdict validator rejects empty side evidence", VerdictError,
            validate, PROMPT_2AFC, bad, {1, 2})
    bad = copy.deepcopy(sweep)
    bad[0]["naturalness"]["assessment"] = "pass"
    rejects("sweep verdict rejects invented assessment labels", VerdictError,
            validate, PROMPT_SWEEP, bad, {1})
    rejects("verdict rows must remain in declared identifier order", VerdictError,
            validate, PROMPT_2AFC, list(reversed(two)), {1, 2})
    rejects("manual validator rejects an empty expected-ID set", VerdictError,
            validate, PROMPT_CRITIQUE, [], set())
    bad = copy.deepcopy(critique)
    bad[0]["done_well"][0]["evidence"] = "   "
    rejects("manual validator rejects whitespace-only evidence", VerdictError,
            validate, PROMPT_CRITIQUE, bad, {1})
    return two_rows, critique_rows, sweep_rows


def check_endpoint_and_mask_metrics() -> None:
    check("zero target accepts zero through ten percent",
          target_interval_percent(0) == (0.0, 10.0)
          and target_within_tolerance_percent(0, 0)
          and target_within_tolerance_percent(0, 10)
          and not target_within_tolerance_percent(0, 10.001))
    check("seventy target accepts sixty through eighty percent",
          target_interval_percent(70) == (60.0, 80.0)
          and target_within_tolerance_percent(70, 60)
          and target_within_tolerance_percent(70, 80)
          and not target_within_tolerance_percent(70, 80.001))
    rejects("requests above seventy are rejected", ValueError,
            target_interval_percent, 70.001)

    mask = [[False] * 9 for _ in range(9)]
    for y in range(3, 6):
        for x in range(3, 6):
            mask[y][x] = True
    mask[1][1] = True
    check("land fraction and percentage agree",
          land_fraction(mask) == 10 / 81 and land_percent(mask) == 1000 / 81)
    check("exact outer-ring water uses the final mask", outer_ring_is_water(mask))
    frame_land = copy.deepcopy(mask)
    frame_land[0][4] = True
    check("one land cell on the outer ring fails", not outer_ring_is_water(frame_land))

    metrics = fragmentation_metrics(mask)
    check("fragmentation metrics use eight-connected component areas",
          metrics["component_areas_cells"] == [9, 1]
          and metrics["largest_component_land_share"] == 0.9)
    empty = fragmentation_metrics([[False, False], [False, False]])
    check("fragmentation is not applicable at zero realized land",
          empty["applicable"] is False
          and empty["largest_component_land_share"] is None)

    monotonic = monotonic_land_percentages([
        (0, 8.0, 10_000), (20, 22.0, 10_000), (40, 39.0, 10_000)])
    check("increasing same-family target sweep passes", monotonic["passes"])
    backwards = monotonic_land_percentages([
        (20, 22.0, 10_000), (40, 18.0, 10_000)])
    check("backwards land response beyond one cell is reported",
          not backwards["passes"] and len(backwards["violations"]) == 1)


def check_immutable_stages() -> None:
    for unsafe in (".", "../key.json", "/absolute.json", "C:/key.json",
                   "hidden\\key.json", "./x"):
        rejects(f"unsafe path rejected: {unsafe}", BundleError,
                safe_relative_path, unsafe)

    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        destination = parent / "bundle"
        staging = create_staging_directory(destination)
        (staging / "judge").mkdir()
        (staging / "hidden").mkdir()
        (staging / "judge" / "instructions.md").write_text(
            "provider-neutral instructions\n", encoding="utf-8")
        (staging / "hidden" / "key.json").write_text(
            "[]\n", encoding="utf-8")
        manifest = make_stage_manifest(
            staging,
            stage="bundle",
            stage_id="eval-fixture",
            created_at_utc="2026-09-01T12:00:00+00:00",
            parent_manifest_sha256s=[],
            prompt_ids=[PROMPT_2AFC],
            schema_ids=["urn:mapgen:pipeline-c:eval:land-origin-2afc-verdict:v1"],
            judge_visible_roots=["judge"],
            hidden_roots=["hidden"],
        )
        write_stage_manifest(staging, manifest)
        check("closed stage manifest verifies", load_and_verify_stage(staging) == manifest)
        published = publish_staged_directory(staging, destination)
        check("verified stage publishes to a new destination", published == destination)
        check("published manifest has a stable hash", len(sha256_file(
            destination / "manifest.json")) == 64)

        bundle_hash = sha256_file(destination / "manifest.json")
        submission_manifest = copy.deepcopy(manifest)
        submission_manifest["stage"] = "submission"
        submission_manifest["stage_id"] = "submission-fixture"
        submission_manifest["parent_manifest_sha256s"] = [bundle_hash]
        check("submission provenance names exactly one bundle manifest",
              validate_manifest_shape(submission_manifest) == submission_manifest)

        result_manifest = copy.deepcopy(manifest)
        result_manifest["stage"] = "result"
        result_manifest["stage_id"] = "result-fixture"
        result_manifest["parent_manifest_sha256s"] = sorted([
            bundle_hash, "a" * 64, "b" * 64,
        ])
        check("result provenance closes bundle and multiple submissions",
              validate_manifest_shape(result_manifest) == result_manifest)

        invalid = copy.deepcopy(submission_manifest)
        invalid["parent_manifest_sha256s"] = []
        rejects("submission provenance cannot omit its bundle", BundleError,
                validate_manifest_shape, invalid)
        invalid["parent_manifest_sha256s"] = sorted(["a" * 64, "b" * 64])
        rejects("submission provenance cannot name multiple parents", BundleError,
                validate_manifest_shape, invalid)
        invalid = copy.deepcopy(result_manifest)
        invalid["parent_manifest_sha256s"] = []
        rejects("result provenance cannot omit all inputs", BundleError,
                validate_manifest_shape, invalid)
        invalid["parent_manifest_sha256s"] = ["a" * 64, "a" * 64]
        rejects("parent manifest hashes must be unique", BundleError,
                validate_manifest_shape, invalid)
        invalid["parent_manifest_sha256s"] = ["f" * 64, "0" * 64]
        rejects("parent manifest hashes must be canonically sorted", BundleError,
                validate_manifest_shape, invalid)

        rejects("published destination cannot be recreated", FileExistsError,
                create_staging_directory, destination)
        second = parent / ".second-build"
        second.mkdir()
        (second / "judge").mkdir()
        (second / "hidden").mkdir()
        (second / "judge" / "instructions.md").write_text("new\n", encoding="utf-8")
        (second / "hidden" / "key.json").write_text("[]\n", encoding="utf-8")
        second_manifest = make_stage_manifest(
            second,
            stage="bundle",
            stage_id="eval-fixture-2",
            created_at_utc="2026-09-01T12:01:00Z",
            parent_manifest_sha256s=[],
            prompt_ids=[PROMPT_2AFC],
            schema_ids=["urn:mapgen:pipeline-c:eval:land-origin-2afc-verdict:v1"],
            judge_visible_roots=["judge"],
            hidden_roots=["hidden"],
        )
        write_stage_manifest(second, second_manifest)
        rejects("publish refuses to overwrite an existing stage", FileExistsError,
                publish_staged_directory, second, destination)

        (destination / "judge" / "instructions.md").write_text(
            "tampered\n", encoding="utf-8")
        rejects("hash mutation invalidates a published stage", BundleError,
                load_and_verify_stage, destination)

        closure_destination = parent / "closure"
        closure_stage = create_staging_directory(closure_destination)
        (closure_stage / "judge").mkdir()
        (closure_stage / "hidden").mkdir()
        (closure_stage / "judge" / "instructions.md").write_text(
            "closed\n", encoding="utf-8")
        (closure_stage / "hidden" / "key.json").write_text("[]\n", encoding="utf-8")
        closure_manifest = make_stage_manifest(
            closure_stage,
            stage="bundle",
            stage_id="closure-fixture",
            created_at_utc="2026-09-01T12:02:00Z",
            parent_manifest_sha256s=[],
            prompt_ids=[PROMPT_2AFC],
            schema_ids=["urn:mapgen:pipeline-c:eval:land-origin-2afc-verdict:v1"],
            judge_visible_roots=["judge"],
            hidden_roots=["hidden"],
        )
        write_stage_manifest(closure_stage, closure_manifest)
        (closure_stage / "judge" / "unmanifested.txt").write_text(
            "not closed\n", encoding="utf-8")
        rejects("unmanifested additions invalidate stage closure", BundleError,
                load_and_verify_stage, closure_stage)

        strict_json = parent / "strict-verdict.json"
        strict_json.write_text('{"panel": 1, "panel": 2}\n', encoding="utf-8")
        rejects("strict JSON input rejects duplicate object keys", VerdictError,
                read_json, strict_json)


def check_scoring(two_rows: dict[int, dict], critique_rows: dict[int, dict]) -> None:
    key = [
        {"trial": 1, "kind": "reference_vs_candidate", "reference_side": "A"},
        {"trial": 2, "kind": "calibration", "reference_side": None},
    ]
    first = copy.deepcopy(two_rows)
    second_data = copy.deepcopy(list(two_rows.values()))
    second_data[0]["pick"] = "B"
    second = validate(PROMPT_2AFC, second_data, {1, 2})
    score = score_2afc(key, {"judge_alpha": first, "judge_beta": second})
    check("2AFC scorer requires and records judge disagreement",
          score["judge_count"] == 2
          and score["pairwise_comparisons"] == 1
          and score["pairwise_agreement"] == 0.0)
    rejects("one judge cannot produce an official 2AFC score", VerdictError,
            score_2afc, key, {"judge_alpha": first})

    digest = "a" * 64
    panel_key = {
        "schema_id": PANEL_KEY_SCHEMA_ID,
        "schema_version": 1,
        "prompt_id": PROMPT_CRITIQUE,
        "panels": [
            {"panel": 1, "hidden_kind": "candidate", "source_id": "sample-1",
             "duplicate_group": "repeat-1", "stimulus_sha256": digest},
            {"panel": 2, "hidden_kind": "candidate", "source_id": "sample-1",
             "duplicate_group": "repeat-1", "stimulus_sha256": digest},
            {"panel": 3, "hidden_kind": "reference", "source_id": "reference-1",
             "duplicate_group": None, "stimulus_sha256": "b" * 64},
        ],
    }
    key_record = validate_panel_key(panel_key)
    check("hidden panel key represents a byte-identical duplicate group",
          key_record["duplicate_groups"] == {"repeat-1": [1, 2]})

    def critique_cohort(*, change_duplicate: bool) -> dict[int, dict]:
        rows = []
        for panel in (1, 2, 3):
            row = copy.deepcopy(critique_rows[1])
            row["panel"] = panel
            if panel == 3:
                row["done_poorly"] = []
                row["done_well"] = []
                row["cannot_identify"] = []
            rows.append(row)
        if change_duplicate:
            rows[1]["done_well"] = []
        return validate(PROMPT_CRITIQUE, rows, {1, 2, 3})

    reliability = score_duplicate_reliability(panel_key, {
        "judge_alpha": critique_cohort(change_duplicate=False),
        "judge_beta": critique_cohort(change_duplicate=True),
    })
    comparisons = reliability["duplicate_groups"]["repeat-1"]["per_judge"]
    check("duplicate scorer records matching and divergent mechanical signatures",
          comparisons["judge_alpha"][0]["mechanical_signature_match"] is True
          and comparisons["judge_beta"][0]["mechanical_signature_match"] is False)
    bad_key = copy.deepcopy(panel_key)
    bad_key["panels"][1]["stimulus_sha256"] = "c" * 64
    rejects("duplicate key rejects a non-identical claimed repeat", VerdictError,
            validate_panel_key, bad_key)
    no_duplicate_key = copy.deepcopy(panel_key)
    no_duplicate_key["panels"] = no_duplicate_key["panels"][2:]
    rejects("duplicate scoring requires a declared reliability probe",
            VerdictError, score_duplicate_reliability, no_duplicate_key, {
                "judge_alpha": {3: critique_cohort(change_duplicate=False)[3]},
                "judge_beta": {3: critique_cohort(change_duplicate=False)[3]},
            })


def check_engine_independence() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "eval").glob("*.py")))
    check("evaluation Python has no engine import",
          not re.search(r"(?m)^\s*(?:from\s+engine|import\s+engine)\b", source))
    check("scaffold ships no fabricated raster stimuli",
          not list((ROOT / "eval").rglob("*.png")))


def main() -> None:
    print("== schemas and fresh prompts ==")
    two_rows, critique_rows, _ = check_schemas_and_prompts()
    print("\n== deterministic land-origin metrics ==")
    check_endpoint_and_mask_metrics()
    print("\n== immutable append-only stages ==")
    check_immutable_stages()
    print("\n== mechanical scoring and isolation ==")
    check_scoring(two_rows, critique_rows)
    check_engine_independence()
    print(f"\n{len(PASS)}/{len(PASS)} checks passed")


if __name__ == "__main__":
    main()
