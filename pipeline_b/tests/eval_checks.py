"""Evaluation-harness archive, calibration, and bundle checks."""

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import m3_panels, m3_trials
from eval.build_m3_run import sha256_file
from eval.m2_trials import REFS, SIZE, bland
from eval.score_run import validate_panels, validate_trials

PASS = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    PASS.append(name)
    print(f"PASS  {name}")


def crop(ref_id, x, y):
    image = Image.open(REFS / f"ref{ref_id}.png").convert("RGB")
    return image.crop((x, y, x + SIZE, y + SIZE))


def main():
    eval_dir = ROOT / "out" / "m3_run1" / "eval"
    try:
        manifest = json.loads(
            (eval_dir / "manifest.json").read_text(encoding="utf-8"))
    except PermissionError as exc:
        print("FAIL  eval bundle unreadable: out/m3_run1/eval is "
              "ACL-locked (created under a sandboxed account; the "
              "current user is denied even ACL reads).")
        print("      Fix from an elevated shell, then re-run:")
        print(r'        takeown /f "out\m3_run1\eval" /r /d y')
        print(r'        icacls "out\m3_run1\eval" /reset /t')
        print(f"      underlying error: {exc}")
        sys.exit(1)

    print("== archive safety ==")
    try:
        m3_trials._safe_out(ROOT / "out" / "m3" / "eval")
    except ValueError:
        rejected_trials = True
    else:
        rejected_trials = False
    try:
        m3_panels._safe_out(ROOT / "out" / "m3" / "eval")
    except ValueError:
        rejected_panels = True
    else:
        rejected_panels = False
    check("historical M3 trial target rejected", rejected_trials)
    check("historical M3 panel target rejected", rejected_panels)
    check("no unpublished build directories remain",
          not list((ROOT / "out" / "m3_run1").glob(".eval-build-*")))

    print("== bland-filter calibration ==")
    check("judgeable dark ref14 crop is retained",
          not bland(crop(14, 125, 117)))
    check("documented dim ref14 crop is rejected",
          bland(crop(14, 158, 210)))

    print("== immutable bundle ==")
    check("expected stimulus counts",
          len(list((eval_dir / "trials").glob("trial_*.png"))) == 12
          and len(list((eval_dir / "panels").glob("panel_*.png"))) == 6
          and len(list((eval_dir / "bridge" / "m3_0_3_0" / "trials")
                       .glob("trial_*.png"))) == 12)
    check("keys stay outside judge-visible directories",
          not list((eval_dir / "trials").glob("*key*"))
          and not list((eval_dir / "panels").glob("*key*"))
          and not list((eval_dir / "bridge" / "m3_0_3_0" / "trials")
                       .glob("*key*")))
    check("source-controlled prompts copied byte-identically",
          (eval_dir / "trials" / "instructions.md").read_bytes()
          == (ROOT / "eval" / "prompts" / "2afc_v2.md").read_bytes()
          and (eval_dir / "panels" / "rubric.md").read_bytes()
          == (ROOT / "eval" / "prompts" / "critique_v2.md").read_bytes())
    check("all manifested artifact hashes match",
          all(sha256_file(eval_dir / rel) == digest
              for rel, digest
              in manifest["generated_artifact_sha256"].items()))

    panel_key = json.loads(
        (eval_dir / "key" / "m3_panels_key.json").read_text())
    duplicate = next(
        item for item in panel_key if item["kind"] == "duplicate_of_candidate")
    original = next(
        item for item in panel_key
        if item["kind"] == "candidate" and item["seed"] == duplicate["seed"])
    check("duplicate reliability panels are byte-identical",
          (eval_dir / "panels" / f"panel_{original['panel']:02d}.png").read_bytes()
          == (eval_dir / "panels" / f"panel_{duplicate['panel']:02d}.png").read_bytes())

    print("== verdict schema ==")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        trial_path = tmp / "trials.json"
        trial_path.write_text(json.dumps([
            {"trial": 1, "void": False, "pick": "A", "confidence": 3,
             "evidence": "A north ridge; B south coast"},
            {"trial": 2, "void": True, "pick": None, "confidence": None,
             "evidence": "B is predominantly dark"},
        ]), encoding="utf-8")
        panel_path = tmp / "panels.json"
        panel_path.write_text(json.dumps([{
            "panel": 1,
            "done_poorly": [{"what": "line", "where": "north",
                              "evidence": "straight for 200 px",
                              "severity": "A"}],
            "done_well": [],
            "cannot_identify": [{"what": "patch", "where": "east",
                                  "evidence": "two plausible readings"}],
        }]), encoding="utf-8")
        check("valid trial schema accepted",
              set(validate_trials(trial_path, {1, 2})) == {1, 2})
        check("valid critique schema accepted",
              set(validate_panels(panel_path, {1})) == {1})
        bad = json.loads(panel_path.read_text())
        del bad[0]["cannot_identify"][0]["evidence"]
        panel_path.write_text(json.dumps(bad), encoding="utf-8")
        try:
            validate_panels(panel_path, {1})
        except ValueError:
            rejected = True
        else:
            rejected = False
        check("missing uncertainty evidence rejected", rejected)

    print(f"\n{len(PASS)}/{len(PASS)} checks passed")


if __name__ == "__main__":
    main()
