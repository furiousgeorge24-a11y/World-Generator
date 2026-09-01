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

## Controller semantics

The author-facing control ranges, defaults, per-map tolerance, endpoint
intervals, and monotonicity rule are normative in
[`../CONTRACT.md`](../CONTRACT.md) §§3–4 and are not restated here. Evaluation
consumes them; it does not define them.

What follows from them for this scaffold:

- Component-area metrics are diagnostics for the fragmentation promise. They
  never impose a specific number of islands.
- At zero realized land, fragmentation metrics are explicitly not applicable.
- At low land targets, flooding can naturally divide the exposed remnants even
  when fragmentation is zero.

## Fresh perceptual instruments

- `land_origin_2afc_v1`: blind reference-versus-candidate morphology
  discrimination with reference-versus-reference calibration arms.
- `land_origin_critique_v1`: single-panel, evidence-anchored diagnostic review.
- `land_controls_sweep_v1`: same-family land-target/fragmentation matrices for
  continuity, rerolling, artificial-cut appearance, and control leakage.
- `layer_audit_v1`: single-image mechanism audit for intermediate layers. Not
  a land instrument and not a quality rating — it asks what kind of rule
  produced a field, from one panel, with no reference to compare against.

These IDs are new. They must never be silently edited after a bundle uses
them; a semantic change creates a new prompt ID and a harness-change record.
Prompt examples are literal valid JSON and are covered by the focused tests.

## The layer audit

A direct "does this look natural?" prompt is unreliable in both directions, so
`layer_audit_v1` replaces the aesthetic question with four that have checkable
answers: state the rule that reproduces this image and whether that description
closes; predict what lies past the right edge; name the mechanism from a fixed
list; then call it `process`, `formula`, or `undecided`. Random and
noise-derived fields are `process` — no short rule writes down their values.

The audit is **supervised**, which is what separates it from the other
instruments. The true mechanism of an intermediate view is known, because the
code that produced it was just written, so the judge can be scored instead of
merely read.

| Module | Role |
|---|---|
| `palette.py` | The one ramp and class table every panel is drawn with, controls included |
| `controls.py` | Synthetic calibration panels with known mechanisms |
| `stimulus.py` | Native-resolution windows, duplicates, hidden key, judging plan |
| `audit.py` | Voiding, mechanism accuracy, duplicate agreement, candidate findings |

`controls.py` generates calibration panels; it does not ship them, and they are
never presented as engine output. A control is an instrument with a declared
answer, recorded in the key as `control_formulaic` or `control_process`, and
`audit.py` refuses to score a batch that has neither. Formulaic controls that
go uncaught void the batch; so does condemning a known process control, because
an indiscriminate judge cannot clear anything. Being *unsure* about a process
control is not disqualifying — the instrument exists to catch formulas.

Controls are rendered through the same palette, at the same size, by the same
encoder as candidates, and panels carry no metadata. Nothing but the image
distinguishes them.

The gate that drives all of this lives outside the scaffold, in
`../run_layer_audit.py`, because it has to touch the engine to render views.
Nothing in `eval/` imports an engine.

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

The following remain honest future hooks while no generator output exists:

- consume and validate a real candidate manifest;
- build neutral, scale-matched *land* stimuli from real masks and a separately
  frozen reference manifest;
- predeclare development and fresh validation cohorts;
- add calibrated resolution, shifted/nested-domain, structural-continuity, and
  narrow-bridge robustness instruments.

No fake gallery, reference mask, candidate, key, verdict, or score is shipped
by this scaffold. Synthetic calibration controls are generated on demand by
`controls.py`, never committed, and never scored as candidates; a repository
check enforces that no raster ships here at all.

Run the infrastructure checks from the repository root with:

```powershell
py -3.14 pipeline_c/tests/eval_checks.py
py -3.14 pipeline_c/tests/layer_audit_checks.py
```
