"""Build, score, and verify a blind layer audit of the current engine.

This is the work-approval gate described in `VIEWS.md`. It is not a per-run
filter: it exists to be run after a stage is implemented, so an obviously
constructed field is caught before anyone spends time looking at it.

    py -3.14 pipeline_c/run_layer_audit.py build --seed 4287772760
    py -3.14 pipeline_c/run_layer_audit.py score --run <id> --verdict v.json
    py -3.14 pipeline_c/run_layer_audit.py verify --run <id> --verdict v.json

`build` publishes an immutable batch whose judge packet is a directory of
numbered panels and nothing else. The judge sees only that directory. Whoever
runs the judge must not read the hidden key first; that is a protocol
obligation, not an enforced sandbox.

The audit asks what kind of rule produced a field, from native-resolution
windows. It is not a judgement of world-scale composition — that stays with
the author, who has the WebUI for it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import webui_adapter  # noqa: E402
from eval.audit import score_layer_audit  # noqa: E402
from eval.bundle import (  # noqa: E402
    create_staging_directory,
    make_stage_manifest,
    publish_staged_directory,
    sha256_file,
    write_stage_manifest,
)
from eval.controls import build_controls  # noqa: E402
from eval.keys import CANDIDATE  # noqa: E402
from eval.stimulus import (  # noqa: E402
    HIDDEN_ROOT,
    KEY_NAME,
    PROVENANCE_NAME,
    Source,
    build_batch,
)
from eval.verdicts import PROMPT_LAYER_AUDIT, read_json  # noqa: E402

DEFAULT_RUNS = ROOT / "audit_runs"
PANEL_PX = 512

# Epoch and tiled views are for the author, not for the blind audit: the audit
# asks what kind of rule made a field, and the same field at four times is one
# question, not four.
DEFAULT_VIEWS = (
    "plates", "boundaries", "regime", "strength", "strength_banded",
    "velocity", "strain_rate", "strain_rate_banded", "drive",
    "strength_initial",
)

# What the runner is willing to assert about a candidate view's mechanism.
# `None` means genuinely arguable, and the panel is then excluded from
# mechanism accuracy rather than scored against a guess. Only claim a value
# that can be defended from the source of the stage that produced it.
DECLARED_MECHANISM = {
    # Smoothstep-interpolated periodic value noise from the sampler, or a
    # first derivative of one. Nothing else here is defensible as a formula:
    # the strength, velocity, strain, plate, and boundary fields are the
    # residue of a hundred and fifty solve-damage-advect steps.
    "drive": "filtered_noise",
    "drive_phi": "filtered_noise",
    "drive_psi": "filtered_noise",
    "strength_initial": "filtered_noise",
}


def _rgb(world, view: str) -> np.ndarray:
    """Exactly the pixels the WebUI shows, so the audit reviews what is seen."""
    image = Image.open(BytesIO(webui_adapter.render_png(world, view)))
    return np.asarray(image.convert("RGB"))


def _panel_px(sources: list[Source]) -> int:
    """The largest panel every candidate can actually supply.

    Views render at native history resolution, which is half the delivered
    resolution, so a 512 px world has 256 px views and cannot be cropped to a
    512 px panel. Candidates and calibration controls must share one panel
    size or the judge could separate them by shape alone.
    """
    smallest = min(min(source.rgb.shape[:2]) for source in sources)
    return min(PANEL_PX, smallest)


def _candidate_sources(seed: int, views: list[str], pixels: int,
                       scale: int) -> tuple[list[Source], dict]:
    world = webui_adapter.generate(seed, {"scale_km": scale}, pixels)
    sources = [
        Source(view, _rgb(world, view), CANDIDATE, DECLARED_MECHANISM.get(view))
        for view in views
    ]
    return sources, {
        "world_seed": world.seed,
        "world_id": world.world_id,
        "pixels": world.pixels,
        "scale_km": world.scale_km,
        "stage": webui_adapter.meta()["stage"],
        "views": list(views),
    }


def command_build(args: argparse.Namespace) -> int:
    views = [view.strip() for view in args.views.split(",") if view.strip()]
    unknown = sorted(set(views) - set(webui_adapter.VIEWS))
    if unknown:
        raise SystemExit(f"unknown view(s): {unknown}")

    sources, world_provenance = _candidate_sources(
        args.seed, views, args.pixels, args.scale)
    panel_px = _panel_px(sources)
    controls = build_controls(args.control_seed, panel_px)
    sources.extend(
        Source(control.control_id, control.rgb, control.kind, control.true_mechanism)
        for control in controls
    )

    run_id = args.run_id or (
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-seed{args.seed}")
    summary = build_batch(
        sources,
        Path(args.runs) / run_id / "bundle",
        seed=args.batch_seed,
        panel_px=panel_px,
        crop_factors=tuple(int(value) for value in args.crops.split(",")),
        duplicate_panels=args.duplicates,
        chunks=args.chunks,
        provenance={
            "world": world_provenance,
            "control_seed": args.control_seed,
            "batch_seed": args.batch_seed,
            "panel_px": panel_px,
            "declared_mechanisms": {
                view: DECLARED_MECHANISM.get(view) for view in views
            },
        },
    )

    print(f"run id        {run_id}")
    print(f"panels        {summary['panel_count']} at {panel_px}px")
    print(f"judge packet  {summary['judge_packet']}")
    print(f"hidden key    {summary['hidden_key']}")
    print()
    if summary["chunks"]:
        print("Judge these panel sets in separate fresh-context calls, then")
        print("concatenate the returned arrays before scoring:")
        for index, chunk in enumerate(summary["chunks"], start=1):
            print(f"  call {index}: {chunk}")
        print()
    print("Give the judge the packet directory and nothing else. It holds")
    print("PROMPT.md and the numbered panels. Do not read the hidden key first.")
    return 0


def _bundle_dir(runs: Path, run_id: str) -> Path:
    bundle = runs / run_id / "bundle"
    if not bundle.is_dir():
        raise SystemExit(f"no bundle at {bundle}")
    return bundle


def _publish(destination: Path, root: str, name: str, payload: object, *,
             stage: str, stage_id: str, parents: list[str]) -> Path:
    staging = create_staging_directory(destination)
    (staging / root).mkdir()
    with (staging / root / name).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    write_stage_manifest(staging, make_stage_manifest(
        staging,
        stage=stage,
        stage_id=stage_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        parent_manifest_sha256s=sorted(parents),
        prompt_ids=[PROMPT_LAYER_AUDIT],
        schema_ids=["urn:mapgen:pipeline-c:eval:layer-audit-verdict:v1"],
        judge_visible_roots=[],
        hidden_roots=[root],
    ))
    return publish_staged_directory(staging, destination)


def command_score(args: argparse.Namespace) -> int:
    runs = Path(args.runs)
    bundle = _bundle_dir(runs, args.run)
    key = read_json(bundle / HIDDEN_ROOT / KEY_NAME)
    verdict = read_json(args.verdict)
    result = score_layer_audit(key, verdict, judge_id=args.judge)

    bundle_hash = sha256_file(bundle / "manifest.json")
    submission = _publish(
        runs / args.run / "submissions" / args.judge, "submission",
        "verdict.json", verdict,
        stage="submission", stage_id=args.judge, parents=[bundle_hash])
    _publish(
        runs / args.run / "results" / args.judge, "result",
        "score.json", result,
        stage="result", stage_id=args.judge,
        parents=[bundle_hash, sha256_file(submission / "manifest.json")])

    summary = result["control_summary"]
    print(f"judge         {result['judge_id']}")
    print(f"controls      {summary['formulaic_caught']}/{summary['formulaic_total']} "
          f"formulaic caught, {summary['process_cleared']}/{summary['process_total']} "
          "process cleared")
    mechanism = result["mechanism"]
    if mechanism["scored_panels"]:
        print(f"mechanism     {mechanism['correct']}/{mechanism['scored_panels']} "
              f"correct ({mechanism['abstentions']} abstained)")
    for group, record in sorted(result["duplicates"].items()):
        agree = "agrees" if record["verdict_agrees"] else "DISAGREES"
        print(f"duplicate     {group}: panels {record['panels']} {agree} "
              f"({', '.join(record['verdicts'])})")
    print()
    if result["batch_void"]:
        print("BATCH VOID — the judge failed calibration, so its verdict on the")
        print("candidates is discarded. Nothing is claimed about the layers.")
        for reason in result["void_reasons"]:
            print(f"  - {reason}")
        print()
    for source_id, record in sorted(result["candidates"].items()):
        flagged = record["called_formula_on"]
        state = f"FORMULA on panels {flagged}" if flagged else "not flagged"
        print(f"{source_id:14s} {state}; closures {record['closures']}; "
              f"mechanisms {record['mechanisms']}")
        if record["regularity_kinds"]:
            print(f"{'':14s} regularities {record['regularity_kinds']}")
    print()
    print("A non-void batch is not an approval. It means nothing was caught.")
    print(f"full result   {runs / args.run / 'results' / args.judge}")
    return 2 if result["batch_void"] else 0


def command_verify(args: argparse.Namespace) -> int:
    """Render what is actually past the right edge of each predicted panel."""
    runs = Path(args.runs)
    bundle = _bundle_dir(runs, args.run)
    key = read_json(bundle / HIDDEN_ROOT / KEY_NAME)
    provenance = read_json(bundle / HIDDEN_ROOT / PROVENANCE_NAME)
    verdict = read_json(args.verdict)
    rows = {row["panel"]: row for row in verdict}
    by_panel = {item["panel"]: item for item in key["panels"]}

    world = webui_adapter.generate(
        provenance["world"]["world_seed"],
        {"scale_km": provenance["world"]["scale_km"]},
        provenance["world"]["pixels"])
    fields = {view: _rgb(world, view) for view in provenance["world"]["views"]}
    for control in build_controls(provenance["control_seed"], provenance["panel_px"]):
        fields[control.control_id] = control.rgb

    out = runs / args.run / "verification"
    out.mkdir(parents=True, exist_ok=True)
    checked = 0
    for panel in sorted(by_panel):
        row = rows.get(panel)
        if row is None or not row["off_frame_prediction"]["predictable"]:
            continue
        entry = by_panel[panel]
        field = fields[entry["source_id"]]
        window, factor = entry["window"], entry["crop_factor"]
        extent = window["extent"]
        row_index = (np.arange(extent) + window["top"]) % field.shape[0]
        column_index = (np.arange(extent) + window["left"] + extent) % field.shape[1]
        image = Image.fromarray(
            np.ascontiguousarray(field[np.ix_(row_index, column_index)]), mode="RGB")
        if factor > 1:
            image = image.resize(
                (image.width * factor, image.height * factor),
                Image.Resampling.NEAREST)
        target = out / f"panel_{panel:02d}_actual_right.png"
        image.save(target, format="PNG", optimize=True)
        forecast = row["off_frame_prediction"]
        print(f"panel {panel:2d}  {entry['source_id']} ({entry['hidden_kind']})")
        print(f"  predicted   {forecast['prediction']}")
        print(f"  period/deg  {forecast['period_px']} / {forecast['orientation_deg']}")
        print(f"  actual      {target}")
        checked += 1
    print()
    print(f"{checked} prediction(s) rendered. Comparing them is a judgement:")
    print("show each pair to a fresh judge, or look yourself. A correct")
    print("prediction is evidence the field is reproducible by a rule.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default=str(DEFAULT_RUNS),
                        help="directory holding audit runs")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="publish a blind audit batch")
    build.add_argument("--seed", type=int, required=True, help="world seed")
    build.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    build.add_argument("--pixels", type=int, default=1024,
                       help="delivered resolution of the audited world")
    build.add_argument("--scale", type=int, default=5,
                       help="kilometres per delivered pixel")
    build.add_argument("--crops", default="1,2",
                       help="integer magnifications, e.g. 1,2")
    build.add_argument("--duplicates", type=int, default=2)
    build.add_argument("--chunks", type=int, default=2,
                       help="independent judge calls to split the batch across")
    build.add_argument("--control-seed", type=int, default=1741)
    build.add_argument("--batch-seed", type=int, default=90210)
    build.add_argument("--run-id", default=None)
    build.set_defaults(func=command_build)

    score = sub.add_parser("score", help="score a judge verdict against the key")
    score.add_argument("--run", required=True)
    score.add_argument("--verdict", required=True)
    score.add_argument("--judge", default="subagent")
    score.set_defaults(func=command_score)

    verify = sub.add_parser("verify", help="render the actual off-frame regions")
    verify.add_argument("--run", required=True)
    verify.add_argument("--verdict", required=True)
    verify.set_defaults(func=command_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
