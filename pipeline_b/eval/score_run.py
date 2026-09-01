"""Validate and score a persisted blind M3-family evaluation run.

The scorer handles only mechanical facts: trial accuracy/agreement,
confidence, critique severity counts, and duplicate count deltas. It
does not decide whether a cited visual claim is true; that remains an
evidence-verification task for the orchestrator.
"""

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

JUDGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*\.json$")
SEVERITIES = {"A", "B", "C", "D"}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file {path}: {exc}") from exc


def nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def verdict_files(directory):
    directory = Path(directory)
    files = sorted(
        path for path in directory.glob("*.json")
        if JUDGE_RE.fullmatch(path.name)
    )
    return files


def validate_trials(path, expected):
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path}: top level must be a JSON list")
    by_trial = {}
    for i, item in enumerate(data):
        label = f"{path}: item {i}"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        trial = item.get("trial")
        if type(trial) is not int or trial not in expected:
            raise ValueError(f"{label} has invalid trial number {trial!r}")
        if trial in by_trial:
            raise ValueError(f"{path}: duplicate trial {trial}")
        if type(item.get("void")) is not bool:
            raise ValueError(f"{path}: trial {trial} needs boolean void")
        if not nonempty_string(item.get("evidence")):
            raise ValueError(f"{path}: trial {trial} needs evidence")
        if item["void"]:
            if item.get("pick") is not None or item.get("confidence") is not None:
                raise ValueError(
                    f"{path}: void trial {trial} needs null pick/confidence")
        else:
            if item.get("pick") not in ("A", "B"):
                raise ValueError(f"{path}: trial {trial} has invalid pick")
            confidence = item.get("confidence")
            if type(confidence) is not int or not 1 <= confidence <= 5:
                raise ValueError(
                    f"{path}: trial {trial} confidence must be 1..5")
        by_trial[trial] = item
    missing = sorted(expected - set(by_trial))
    if missing:
        raise ValueError(f"{path}: missing trials {missing}")
    return by_trial


def validate_claim(path, panel, bucket, index, claim):
    label = f"{path}: panel {panel} {bucket}[{index}]"
    if not isinstance(claim, dict):
        raise ValueError(f"{label} must be an object")
    for field in ("what", "where", "evidence"):
        if not nonempty_string(claim.get(field)):
            raise ValueError(f"{label} needs non-empty {field}")
    if bucket == "done_poorly":
        if claim.get("severity") not in SEVERITIES:
            raise ValueError(f"{label} has invalid severity")
    elif "severity" in claim:
        raise ValueError(f"{label} must not carry severity")


def validate_panels(path, expected):
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path}: top level must be a JSON list")
    by_panel = {}
    caps = {"done_poorly": 5, "done_well": 5, "cannot_identify": 3}
    for i, item in enumerate(data):
        label = f"{path}: item {i}"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        panel = item.get("panel")
        if type(panel) is not int or panel not in expected:
            raise ValueError(f"{label} has invalid panel number {panel!r}")
        if panel in by_panel:
            raise ValueError(f"{path}: duplicate panel {panel}")
        for bucket, cap in caps.items():
            claims = item.get(bucket)
            if not isinstance(claims, list) or len(claims) > cap:
                raise ValueError(
                    f"{path}: panel {panel} {bucket} must be a list of <= {cap}")
            for index, claim in enumerate(claims):
                validate_claim(path, panel, bucket, index, claim)
        by_panel[panel] = item
    missing = sorted(expected - set(by_panel))
    if missing:
        raise ValueError(f"{path}: missing panels {missing}")
    return by_panel


def mean(values):
    return sum(values) / len(values) if values else None


def score_trials(key, verdicts):
    key_by_trial = {item["trial"]: item for item in key}
    candidate_ids = [
        item["trial"] for item in key if item["kind"] == "ref_vs_cand"
    ]
    calibration_ids = [
        item["trial"] for item in key if item["kind"] == "calibration"
    ]
    per_judge = {}
    for judge, rows in verdicts.items():
        candidate = [rows[n] for n in candidate_ids]
        valid_candidate = [row for row in candidate if not row["void"]]
        correct = sum(
            rows[n]["pick"] == key_by_trial[n]["ref_side"]
            for n in candidate_ids if not rows[n]["void"]
        )
        calibration = [rows[n] for n in calibration_ids]
        valid_calibration = [row for row in calibration if not row["void"]]
        per_judge[judge] = {
            "correct_reference_picks": correct,
            "valid_candidate_arms": len(valid_candidate),
            "total_candidate_arms": len(candidate_ids),
            "candidate_accuracy": (
                correct / len(valid_candidate) if valid_candidate else None),
            "candidate_voids": len(candidate) - len(valid_candidate),
            "candidate_mean_confidence": mean(
                [row["confidence"] for row in valid_candidate]),
            "valid_calibration_arms": len(valid_calibration),
            "total_calibration_arms": len(calibration_ids),
            "calibration_voids": len(calibration) - len(valid_calibration),
            "calibration_mean_confidence": mean(
                [row["confidence"] for row in valid_calibration]),
            "calibration_high_confidence_count": sum(
                row["confidence"] >= 4 for row in valid_calibration),
        }

    unanimous = 0
    fully_covered = 0
    trial_agreement = {}
    for trial in candidate_ids:
        picks = {
            judge: rows[trial]["pick"]
            for judge, rows in verdicts.items() if not rows[trial]["void"]
        }
        all_present = len(picks) == len(verdicts)
        if all_present:
            fully_covered += 1
            unanimous += len(set(picks.values())) == 1
        trial_agreement[str(trial)] = {
            "valid_picks": picks,
            "unanimous_all_judges": (
                all_present and len(set(picks.values())) == 1),
        }

    pairwise_hits = 0
    pairwise_total = 0
    for a, b in itertools.combinations(sorted(verdicts), 2):
        for trial in candidate_ids:
            ra, rb = verdicts[a][trial], verdicts[b][trial]
            if ra["void"] or rb["void"]:
                continue
            pairwise_total += 1
            pairwise_hits += ra["pick"] == rb["pick"]

    total_correct = sum(v["correct_reference_picks"] for v in per_judge.values())
    total_valid = sum(v["valid_candidate_arms"] for v in per_judge.values())
    return {
        "judge_count": len(verdicts),
        "official_minimum_judges_met": len(verdicts) >= 2,
        "per_judge": per_judge,
        "combined": {
            "correct_reference_picks": total_correct,
            "valid_candidate_judgments": total_valid,
            "candidate_accuracy": total_correct / total_valid if total_valid else None,
            "unanimous_candidate_arms": unanimous,
            "fully_covered_candidate_arms": fully_covered,
            "pairwise_agreement": (
                pairwise_hits / pairwise_total if pairwise_total else None),
            "pairwise_agreements": pairwise_hits,
            "pairwise_comparisons": pairwise_total,
        },
        "candidate_trial_agreement": trial_agreement,
    }


def score_panels(key, verdicts):
    key_by_panel = {item["panel"]: item for item in key}
    per_judge = {}
    for judge, rows in verdicts.items():
        counts = defaultdict(Counter)
        praise = Counter()
        unknowns = Counter()
        for panel, row in rows.items():
            kind = key_by_panel[panel]["kind"]
            for claim in row["done_poorly"]:
                counts[kind][claim["severity"]] += 1
            praise[kind] += len(row["done_well"])
            unknowns[kind] += len(row["cannot_identify"])

        duplicate = next(
            item for item in key if item["kind"] == "duplicate_of_candidate")
        original = next(
            item for item in key
            if item["kind"] == "candidate"
            and item["seed"] == duplicate["seed"])
        orig_counts = Counter(
            claim["severity"] for claim
            in rows[original["panel"]]["done_poorly"])
        dup_counts = Counter(
            claim["severity"] for claim
            in rows[duplicate["panel"]]["done_poorly"])
        per_judge[judge] = {
            "severity_counts_by_hidden_kind": {
                kind: {severity: counter.get(severity, 0)
                       for severity in sorted(SEVERITIES)}
                for kind, counter in sorted(counts.items())
            },
            "done_well_counts_by_hidden_kind": dict(sorted(praise.items())),
            "cannot_identify_counts_by_hidden_kind": dict(
                sorted(unknowns.items())),
            "duplicate_probe": {
                "original_panel": original["panel"],
                "duplicate_panel": duplicate["panel"],
                "severity_count_delta": {
                    severity: abs(orig_counts.get(severity, 0)
                                  - dup_counts.get(severity, 0))
                    for severity in sorted(SEVERITIES)
                },
                "semantic_consistency": "requires manual verification",
            },
        }

    aggregate = defaultdict(Counter)
    for rows in verdicts.values():
        for panel, row in rows.items():
            kind = key_by_panel[panel]["kind"]
            for claim in row["done_poorly"]:
                aggregate[kind][claim["severity"]] += 1
    return {
        "judge_count": len(verdicts),
        "official_minimum_judges_met": len(verdicts) >= 2,
        "per_judge": per_judge,
        "aggregate_severity_counts_by_hidden_kind": {
            kind: {severity: counter.get(severity, 0)
                   for severity in sorted(SEVERITIES)}
            for kind, counter in sorted(aggregate.items())
        },
        "evidence_truth": "requires orchestrator spot verification",
    }


def load_trial_cohort(directory, key):
    expected = {item["trial"] for item in key}
    files = verdict_files(directory)
    return {
        path.stem: validate_trials(path, expected) for path in files
    }, files


def load_panel_cohort(directory, key):
    expected = {item["panel"] for item in key}
    files = verdict_files(directory)
    return {
        path.stem: validate_panels(path, expected) for path in files
    }, files


def score(eval_dir):
    eval_dir = Path(eval_dir).resolve()
    output = eval_dir / "scores.json"
    if output.exists():
        raise FileExistsError(f"scores already exist and will not be overwritten: {output}")

    manifest = read_json(eval_dir / "manifest.json")
    for rel, expected_hash in manifest["generated_artifact_sha256"].items():
        path = eval_dir / rel
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"immutable bundle hash mismatch: {rel}")

    trials_key = read_json(eval_dir / "key" / "m3_trials_key.json")
    bridge_key = read_json(
        eval_dir / "key" / "bridge_m3_0_3_0_trials_key.json")
    panels_key = read_json(eval_dir / "key" / "m3_panels_key.json")

    trial_verdicts, trial_files = load_trial_cohort(
        eval_dir / "verdicts" / "trials", trials_key)
    bridge_verdicts, bridge_files = load_trial_cohort(
        eval_dir / "verdicts" / "bridge", bridge_key)
    panel_verdicts, panel_files = load_panel_cohort(
        eval_dir / "verdicts" / "panels", panels_key)
    for name, cohort in (
            ("Run 1 trials", trial_verdicts),
            ("bridge trials", bridge_verdicts),
            ("Run 1 panels", panel_verdicts)):
        if len(cohort) < 2:
            raise ValueError(f"{name} needs at least two independent judges")

    result = {
        "schema": 1,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_manifest_sha256": sha256_file(eval_dir / "manifest.json"),
        "raw_verdict_sha256": {
            path.relative_to(eval_dir).as_posix(): sha256_file(path)
            for path in sorted(trial_files + bridge_files + panel_files)
        },
        "run1_trials": score_trials(trials_key, trial_verdicts),
        "bridge_m3_0_3_0_trials": score_trials(
            bridge_key, bridge_verdicts),
        "run1_panels": score_panels(panels_key, panel_verdicts),
        "limitations": [
            "reference screenshot km/px is unknown; visually mismatched arms may be voided",
            "severity counts are mechanical; claim truth and duplicate semantic consistency require verification",
        ],
    }
    with output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"scores -> {output}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    args = parser.parse_args()
    score(args.eval_dir)


if __name__ == "__main__":
    main()
