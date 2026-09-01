"""Assemble a blind layer-audit batch: numbered panels plus a hidden key.

Deliberately dumb. It takes already-rendered RGB fields and turns them into
panels; it does not know what any field means, and it never decides anything.
The caller chooses what to submit.

Two rules shape every choice here:

* **No resampling that invents or destroys detail.** Panels are native-
  resolution windows, optionally magnified by an integer factor with
  nearest-neighbour replication. Nothing is ever reduced or interpolated,
  because a smoothing or decimating step can both hide a lattice and
  manufacture one.
* **Nothing distinguishes a control from a candidate except the image.**
  Identical size, identical palette, identical encoder, sequential filenames
  after a shuffle, and no metadata in the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .bundle import (
    create_staging_directory,
    make_stage_manifest,
    publish_staged_directory,
    write_stage_manifest,
)
from .keys import (
    CANDIDATE,
    LAYER_PANEL_KEY_SCHEMA_ID,
    validate_layer_panel_key,
)
from .verdicts import PROMPT_LAYER_AUDIT

VISIBLE_ROOT = "judge"
HIDDEN_ROOT = "hidden"
KEY_NAME = "panel_key.json"
PROVENANCE_NAME = "provenance.json"
PLAN_NAME = "judging_plan.json"
AUDIT_SCHEMA_IDS = (
    "urn:mapgen:pipeline-c:eval:layer-audit-verdict:v1",
    LAYER_PANEL_KEY_SCHEMA_ID,
)


@dataclass(slots=True, frozen=True)
class Source:
    """One rendered field offered to the audit.

    `true_mechanism` is the supervised answer. It may be `None` only for a
    candidate whose mechanism is genuinely unknown; a control must declare it.
    """

    source_id: str
    rgb: np.ndarray
    hidden_kind: str = CANDIDATE
    true_mechanism: str | None = None


def _window(rgb: np.ndarray, top: int, left: int, extent: int) -> np.ndarray:
    """A wrapped native-resolution window. Fields are tori, so this has no edge."""
    rows = (np.arange(extent) + top) % rgb.shape[0]
    cols = (np.arange(extent) + left) % rgb.shape[1]
    return rgb[np.ix_(rows, cols)]


def _choose_window(rgb: np.ndarray, rng: np.random.Generator, extent: int,
                   attempts: int = 8) -> tuple[int, int]:
    """Pick the most informative of several random offsets.

    A window holding one class and one boundary asks the judge nothing, and a
    forced answer about it is noise in the result. Distinct-colour count is a
    blunt content measure, applied identically to candidates and controls, so
    it cannot separate them; it only avoids spending a panel on a blank crop.
    """
    best, best_content = (0, 0), -1
    for _ in range(attempts):
        top, left = (int(v) for v in rng.integers(0, rgb.shape[0], size=2))
        sample = _window(rgb, top, left, extent).reshape(-1, 3)
        content = len(np.unique(sample[::7], axis=0))
        if content > best_content:
            best, best_content = (top, left), content
    return best


def _encode(rgb: np.ndarray, magnify: int) -> bytes:
    image = Image.fromarray(np.ascontiguousarray(rgb), mode="RGB")
    if magnify > 1:
        image = image.resize(
            (image.width * magnify, image.height * magnify),
            Image.Resampling.NEAREST,
        )
    from io import BytesIO

    buffer = BytesIO()
    # No pnginfo: a panel must carry nothing a judge could read instead of look.
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _panels_for(source: Source, rng: np.random.Generator, panel_px: int,
                crop_factors: tuple[int, ...]) -> list[dict]:
    panels = []
    for factor in crop_factors:
        extent = panel_px // factor
        if extent < 1:
            raise ValueError(f"crop factor {factor} exceeds the panel size")
        if extent > min(source.rgb.shape[:2]):
            raise ValueError(
                f"source {source.source_id!r} is smaller than a {extent}px window")
        top, left = _choose_window(source.rgb, rng, extent)
        panels.append({
            "source_id": source.source_id,
            "hidden_kind": source.hidden_kind,
            "true_mechanism": source.true_mechanism,
            "crop_factor": factor,
            "window": {"top": top, "left": left, "extent": extent},
            "png": _encode(_window(source.rgb, top, left, extent), factor),
        })
    return panels


def plan_chunks(key_panels: list[dict], chunks: int,
                rng: np.random.Generator) -> list[list[int]]:
    """Split a batch across independent judge calls.

    Two constraints, both load-bearing:

    * Members of a duplicate group go to *different* chunks. Inside one call a
      judge sees both copies at once, so agreement there measures internal
      consistency; across calls it measures whether the same image draws the
      same call from a fresh reader, which is the thing worth knowing.
    * Every chunk carries at least one formulaic and one process control. A
      chunk without both cannot be calibrated, and an uncalibrated chunk's
      verdict on candidates means nothing.
    """
    if chunks < 1:
        raise ValueError("a batch needs at least one chunk")
    buckets: list[list[int]] = [[] for _ in range(chunks)]
    assigned: set[int] = set()

    groups: dict[str, list[int]] = {}
    for panel in key_panels:
        if panel["duplicate_group"]:
            groups.setdefault(panel["duplicate_group"], []).append(panel["panel"])
    for group, members in sorted(groups.items()):
        if len(members) > chunks:
            raise ValueError(
                f"duplicate group {group!r} has more members than there are chunks")
        for bucket, number in zip(rng.permutation(chunks), sorted(members)):
            buckets[int(bucket)].append(number)
            assigned.add(number)

    def deal(candidates: list[int]) -> None:
        for number in candidates:
            if number in assigned:
                continue
            buckets.sort(key=len)
            buckets[0].append(number)
            assigned.add(number)

    by_kind = {kind: [panel["panel"] for panel in key_panels
                      if panel["hidden_kind"] == kind]
               for kind in ("control_formulaic", "control_process", CANDIDATE)}
    deal(by_kind["control_formulaic"])
    deal(by_kind["control_process"])
    deal(by_kind[CANDIDATE])

    for index, bucket in enumerate(buckets):
        kinds = {panel["hidden_kind"] for panel in key_panels
                 if panel["panel"] in bucket}
        if not {"control_formulaic", "control_process"} <= kinds:
            raise ValueError(
                f"chunk {index} lacks a full calibration pair; use fewer chunks")
    return [sorted(bucket) for bucket in buckets]


def build_batch(
    sources: list[Source],
    destination: str | Path,
    *,
    seed: int,
    panel_px: int = 512,
    crop_factors: tuple[int, ...] = (1, 2),
    duplicate_panels: int = 2,
    chunks: int = 1,
    prompt_path: str | Path | None = None,
    provenance: dict | None = None,
) -> dict[str, object]:
    """Build and publish one immutable audit batch.

    Returns a summary for the operator. The hidden key is written under a root
    disjoint from the judge packet; keeping the judge out of it is a protocol
    obligation on whoever runs the batch, not an enforced sandbox.
    """
    if not sources:
        raise ValueError("an audit batch needs at least one source")
    if panel_px < 16 or panel_px % max(crop_factors) != 0:
        raise ValueError("panel_px must be a sensible multiple of every crop factor")
    rng = np.random.default_rng(seed)

    built: list[dict] = []
    for source in sources:
        built.extend(_panels_for(source, rng, panel_px, crop_factors))

    # Byte-identical repeats under different numbers: the within-judge
    # agreement measure. A duplicate must be indistinguishable, so it is the
    # same encoded bytes, not a re-render.
    for index in rng.choice(
        len(built), size=min(duplicate_panels, len(built)), replace=False
    ):
        original = built[int(index)]
        group = f"dup_{original['source_id']}_{original['crop_factor']}x"
        original["duplicate_group"] = group
        built.append(dict(original))

    order = rng.permutation(len(built))
    ordered = [built[int(index)] for index in order]

    destination = Path(destination)
    staging = create_staging_directory(destination)
    judge_dir = staging / VISIBLE_ROOT
    hidden_dir = staging / HIDDEN_ROOT
    judge_dir.mkdir()
    hidden_dir.mkdir()

    width = max(2, len(str(len(ordered))))
    key_panels = []
    for number, panel in enumerate(ordered, start=1):
        name = f"panel_{number:0{width}d}.png"
        (judge_dir / name).write_bytes(panel["png"])
        key_panels.append({
            "panel": number,
            "hidden_kind": panel["hidden_kind"],
            "source_id": panel["source_id"],
            "duplicate_group": panel.get("duplicate_group"),
            "stimulus_sha256": hashlib.sha256(panel["png"]).hexdigest(),
            "true_mechanism": panel["true_mechanism"],
            "crop_factor": panel["crop_factor"],
            "window": panel["window"],
        })

    key = {
        "schema_id": LAYER_PANEL_KEY_SCHEMA_ID,
        "schema_version": 1,
        "prompt_id": PROMPT_LAYER_AUDIT,
        "panels": key_panels,
    }
    validate_layer_panel_key(key)
    with (hidden_dir / KEY_NAME).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(key, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    plan = plan_chunks(key_panels, chunks, rng) if chunks > 1 else None
    if plan is not None:
        # Answer-side: the plan reveals which panels are duplicates, so it
        # belongs with the key and never with the packet.
        with (hidden_dir / PLAN_NAME).open(
            "x", encoding="utf-8", newline=chr(10)
        ) as handle:
            json.dump({"chunks": plan}, handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write(chr(10))

    if provenance is not None:
        # Answer-side only: it names what produced each source, which is what
        # makes an off-frame prediction checkable after the fact.
        with (hidden_dir / PROVENANCE_NAME).open(
            "x", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")

    prompt_path = Path(
        prompt_path or Path(__file__).resolve().parent / "prompts"
        / f"{PROMPT_LAYER_AUDIT}.md")
    shutil.copyfile(prompt_path, judge_dir / "PROMPT.md")

    manifest = make_stage_manifest(
        staging,
        stage="bundle",
        stage_id=destination.name,
        created_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        parent_manifest_sha256s=[],
        prompt_ids=[PROMPT_LAYER_AUDIT],
        schema_ids=list(AUDIT_SCHEMA_IDS),
        judge_visible_roots=[VISIBLE_ROOT],
        hidden_roots=[HIDDEN_ROOT],
    )
    write_stage_manifest(staging, manifest)
    published = publish_staged_directory(staging, destination)

    return {
        "batch": str(published),
        "judge_packet": str(published / VISIBLE_ROOT),
        "hidden_key": str(published / HIDDEN_ROOT / KEY_NAME),
        "panel_count": len(ordered),
        "panel_px": panel_px,
        "sources": sorted({source.source_id for source in sources}),
        "duplicate_groups": sorted(
            {panel["duplicate_group"] for panel in key_panels
             if panel["duplicate_group"]}),
        "chunks": plan,
    }


__all__ = [
    "HIDDEN_ROOT",
    "KEY_NAME",
    "PLAN_NAME",
    "PROVENANCE_NAME",
    "Source",
    "VISIBLE_ROOT",
    "build_batch",
    "plan_chunks",
]
