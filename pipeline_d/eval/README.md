# Land-origin evaluation scaffold

This directory contains the evaluation boundary for `pipeline_c`. It is
deliberately independent of the future generator: evaluation consumes a
versioned candidate manifest and already-produced artifacts; it does not
import an engine, generate terrain, repair masks, choose seeds, or invent
stimuli.

## Authority

Evaluation has three distinct layers:

1. **Deterministic conformance** decides measurable promises: requested-land
   error, exact outer-ring water, determinism, same-family target monotonicity,
   resolution/domain checks, and declared provenance. A failure blocks a claim
   of conformance but does not erase diagnostic evidence.
2. **Perceptual instruments** provide morphology evidence. Blind 2AFC is a
   discrimination measure; critique and control-sweep panels produce an
   evidence-anchored work list. A visual regularity is a causal hypothesis,
   never proof that a construction hack exists.
3. **The author decides** whether the result is suitable. The builder may
   assemble packets and mechanically validate responses but may not grade its
   own output.

The prompts and schemas are provider-neutral. Every judging submission must
record provider, model, fresh-context status, independence limitations, raw
output, and validated output through the submission schema. A cohort drawn
from one provider or model family is reported as a limitation.

## Frozen controller semantics

- The author-facing `target_land_percent` control is in **0 through 70
  inclusive**.
- Acceptance is an absolute error of at most **ten percentage points** on
  every map rather than on a batch mean.
- A request of 0 accepts 0% through 10% realized land.
- A request of 70 accepts 60% through 80% realized land. The 70% value is
  the maximum request, not a hard ceiling on realized land.
- Across an increasing-target same-family sweep, realized land may not move
  backwards beyond one final-mask cell of measurement tolerance.
- `landmass_fragmentation` is a continuous 0..1 process weight, not an island
  or continent count, with a default of 0.5. At 0 it creates a strong likelihood of one dominant
  broad landmass while allowing small coastal, barrier, volcanic, and other
  secondary islands. Increasing it should reorganize approximately the same
  land budget into more separated major bodies through formation, not by
  cutting or editing the finished mask.

Component-area metrics are diagnostics for that promise. They do not impose a
specific number of islands. At zero realized land, fragmentation metrics are
explicitly not applicable. At low land targets, flooding can naturally divide
the exposed remnants even when fragmentation is zero.

## Fresh perceptual instruments

- `land_origin_2afc_v1`: blind reference-versus-candidate morphology
  discrimination with reference-versus-reference calibration arms.
- `land_origin_critique_v1`: single-panel, evidence-anchored diagnostic review.
- `land_controls_sweep_v1`: same-family land-target/fragmentation matrices for
  continuity, rerolling, artificial-cut appearance, and control leakage.

These IDs are new. They must never be silently edited after a bundle uses
them; a semantic change creates a new prompt ID and a harness-change record.
Prompt examples are literal valid JSON and are covered by the focused tests.

Exact candidate outer-ring water is checked from the authoritative final mask,
outside perceptual judging. Land approaching the frame, a straight coast, or a
coast that happens to parallel the frame is not evidence of a hack by itself.
Only a causal trace to the delivered frame or a numerical boundary can confirm
that violation.

## Append-only stage lifecycle

An evaluation run is a container of separately immutable stages:

```text
eval/runs/<evaluation-id>/
  bundle/       # prompts, schemas, stimuli, hidden keys, closed manifest
  submissions/  # one immutable child stage per independent judge
  results/      # immutable scored/adjudicated result stages
```

Each stage is built in a sibling temporary directory and published only after
its closed manifest verifies. Publication refuses an existing destination.
The manifest lists every regular file other than itself; added, missing, or
changed files invalidate the stage. Symlinks, absolute paths, backslashes,
path traversal, and files outside declared visible/hidden roots are rejected.

The bundle's judge packet belongs below a declared visible root such as
`judge/`. Keys, source identities, candidate provenance, and answer-side data
belong below a disjoint hidden root such as `hidden/`. Judges receive only the
visible packet. Protocol-enforced blindness must not be described as an OS
sandbox unless one actually exists.

`schemas/panel_key_v1.schema.json` records hidden candidate/reference identity
and explicit duplicate groups. Every member of a duplicate group must carry
the same source identity and stimulus SHA-256; the strict key validator also
rejects repeated hashes that were not declared as one group. Mechanical
bucket/severity or sweep-assessment consistency is scored, while semantic
consistency of the cited evidence remains a manual verification.

Every stage carries a canonical, unique `parent_manifest_sha256s` list. A
bundle's list is empty; a submission names exactly its one bundle; a result
names the bundle plus every submission manifest it consumes. The result schema
allows one or more inputs, while an official multi-judge result naturally has
the bundle and at least two submissions. A judge submission retains raw model
output, strict validated JSON, judge metadata, and their hashes. Scoring is
mechanical and append-only; critique severity counts are not an acceptance
score, and cited claims require spot verification.

## Candidate interface and future hooks

`schemas/candidate_manifest_v1.schema.json` defines the engine-neutral handoff:
same-family identity, fixed window identity, `target_land_percent`, realized
land percentage, `landmass_fragmentation`, final-mask dimensions/scale,
stable latent-randomness hash,
selection disclosure, and hashed mask/morphology/cause artifacts.

The following remain honest future hooks because Run 1 has no generator output:

- consume and validate a real candidate manifest;
- build neutral, scale-matched stimuli from real masks and a separately frozen
  reference manifest;
- calibrate bland/feature-content filters on known positive and negative land
  panels;
- predeclare development and fresh validation cohorts;
- publish judge submissions and results through the immutable-stage utilities;
- add calibrated resolution, shifted/nested-domain, structural-continuity, and
  narrow-bridge robustness instruments.

No fake gallery, reference mask, stimulus, key, verdict, or score is shipped by
this scaffold.

Run the current infrastructure checks from the repository root with:

```powershell
python pipeline_c/tests/eval_checks.py
```
