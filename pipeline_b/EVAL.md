# Evaluation protocol — pipeline_b

[`CONTRACT.md`](CONTRACT.md) is normative; this file is the operational
source of truth for applying it; [`DESIGN.md`](DESIGN.md) records the
architecture. On conflict, the contract wins. The protocol grew out of
Phase 0 spike S4 (see [`MILESTONES.md`](MILESTONES.md) for production
history and [`ATTEMPT_REGISTER.md`](ATTEMPT_REGISTER.md) for private
experiments) and supersedes procedural fragments in those records.

**Why this exists:** an earlier self-evaluation eagerly approved its own
output. The failure had three causes — self-grading,
gestalt questions, no evidence requirement — and every rule below
traces to one of them.

## The hierarchy

1. **Checks have explicit authority.** Hard invariants and confirmed
   causal violations veto conformance. Calibrated necessity gates can
   veto promotion by proving a required property absent (a map with no
   shelf-depth mass has no shelves). Diagnostic regularity metrics
   trigger causal audit but cannot reject natural terrain by themselves.
   No metric certifies beauty.
2. **Blind judges compare; they never see their own work.** All
   perceptual judgment is comparative (against canon or rubric), by
   fresh-context judges with no knowledge of what was built or which
   side is which.
3. **The author decides.** The harness's job is narrower than taste:
   nothing may reach the author carrying an unearned recommendation.

Every run still emits diagnostic artifacts and a report (§5). Failed
hard invariants mark the image non-conforming and block delivery or
promotion as a contract-satisfying map; ordinary findings do not erase
the run evidence.

## Layer 1 — instruments (M2+)

Views designed to make defects glaring rather than arguable. Every
persisted/material field and stage output needed to audit a contract
promise has a view (§14); these are the derived ones:

- **Isobath/contour render** — floating islands are invisible in color,
  glaring in contours (do they bow around the island or pass under?).
- **Slope map** — shoreline plummet reads as a bright coastal halo.
- **Shore-normal transect overlays** — the §6b shelf sequence as 1D
  curves; fifty overlaid transects expose uniform shelf width (§6c)
  instantly.
- **Same-seed difference images** — control orthogonality (§4) and the
  author's proven even-band catcher.
- **Autocorrelation / spectrum views** — stamps, tiling, even spacing
  (§11a) appear as spectral peaks.
- **Row/column statistics** — frame correlation (§3b). This is a causal
  audit trigger, not an appearance-only veto: investigate whether the
  delivered frame or a numerical-domain boundary influenced the field.
- **Cause-field overlays** — §11 cross-examination: shelf width against
  margin type, trenches against boundary fields, valleys against flow.
  A pretty picture whose supposed cause doesn't correlate is a painted
  symptom.

## Layer 2 — deterministic checks (from M1 on)

Run on every batch; results go into the run report. Their authority is
classified rather than assumed.

### Hard conformance checks

- Exact final outer-ring water + nearest-land-to-border distance (§3a).
- Sediment budget closure: actual fluvial source = in-world deposition
  + explicit process-domain export + surfaced terminal residual; the
  last term is expected to be zero (§6h).
- Determinism: same seed+controls → bit-identical output; control
  isolation: out-of-stage fields untouched when a downstream control
  moves (§4).
- Confirmed crop/frame-coordinate terrain modification or other
  frame-caused geography (§3b). Where a finite rim or localized solver
  exists, promotion also requires an applicable independence proof:
  periodic/boundary-neutral construction, a sufficient causal-reach
  bound, or adequate nested/shifted-domain invariance.

### Calibrated necessity gates

- Depth-histogram mass in the shelf band (§6b) — no mass, no shelves.
- Shoreline gradient distribution — plummet detection (§6a).
- Dendritic-texture-below-the-slope detector (§6g).
- Lake census: count/size distribution (speckle and mega-lake, §9).

These gates veto promotion only after calibration on a known positive
and a known negative establishes that the measured absence is real.

### Diagnostic tripwires

- Per-landmass shelf-width distribution — uniformity investigation
  (§6c).
- Spectral/autocorrelation peaks — tiling and even-spacing investigation
  (§11a).
- Frame-correlation and contour-alignment statistics (§3b).

Diagnostic tripwires place the result on hold for causal audit. They do
not veto frame-blind natural terrain by appearance alone.

## Layer 3 — trial type 1: blind 2AFC ("imposter") — the scalar metric

On scored candidate arms, judges see pairs of tiles with exactly one
from the canon (`../examples/`) and pick which is canon-grade.
Calibration arms may contain two canon tiles. Accuracy near 100% on the
scored arms = trivially distinguishable; **progress = accuracy falling
toward chance.**

Construction rules (each learned the hard way in S4/S4b):

1. **Feature-targeted crops.** Random crops land on featureless ocean
   and void the trial. Crops must contain judgeable formation; the
   bland-tile filter (variance + gradient + near-black tests) is
   mandatory, calibrated on known positive and negative tiles, and its
   voids are recorded.
2. **Scale matching.** Use candidate scale metadata and the best
   available approximate canon match. Canon km/px remains unknown, so
   disclose that uncertainty and void severe scale mismatches rather
   than treating zoom as formation evidence.
3. **Palette preserved.** Never grayscale — CONTRACT §12, valued
   feature 7: the ramp carries half the look. Render candidates through
   a comparable stepped ramp.
   Judges are told palettes may differ and hue alone is not evidence.
4. **Calibration arms.** Ref-vs-ref pairs mixed into every run. They
   have no right answer; they measure invented certainty and judge
   bias, and they catch a broken trial build before it is trusted
   (round 1's chance-level score was diagnosed this way).
5. **Blindness mechanics.** L/R randomized (seeded); answer key stored
   outside the trials directory; judges receive the trials directory
   only, fresh context, no provenance, ≥2 independent judges.
6. **Scoring.** Per-judge accuracy on ref-vs-cand arms; inter-judge
   agreement; confidence calibration (high confidence on calibration
   arms is a judge defect). Whole-run bundle — trials, key, verdicts,
   scores — persists to disk at run time.
7. **Frame condition scored separately.** Reference land touching the
   crop frame is a known source-program property and is excluded from
   blinded formation scoring. Candidate outer-ring conformance is
   decided by the exact deterministic §3a test, not by judge taste.

## Layer 3 — trial type 2: diagnostic critique panels — the work list

Author-specified format (2026-08-29). Single images reviewed blind
against the formation rubric — never "spot differences vs this
reference"; difference from any particular ref is not a defect
(refs are feature-level exemplars, not composition targets).

1. **Three buckets, capped:** up to 5 *done poorly*, up to 5 *done
   well*, up to 3 *cannot identify*. Caps force prioritization.
2. **Evidence anchoring on every claim** — what / where / evidence.
   **Praise bears the same burden as defects**: unearned praise is the
   original sign-off failure mode.
3. **The cannot-identify bucket is load-bearing.** Honest unknowns
   beat confabulated stories; a feature a viewer cannot name is itself
   a §11 signal, and canon ambiguities (see literacy notes) surface
   here instead of as false defects.
4. **Canon in the defendant's seat.** Some panels are secretly
   reference crops, presented identically. Judge harshness on those is
   the baseline; provenance framing stays neutral.
5. **Panel hygiene.** Canon picks deduplicated (distinct refs,
   disjoint regions). Occasional *deliberate* duplicate panels are a
   test-retest reliability probe (near-identical panels drew
   near-identical critiques in the s4c run — keep measuring that).
6. **Scoring by severity class, not count.** In s4c, canon and
   candidate panels drew similar defect *counts*; the separation was
   entirely in *class*. Tag every defect claim:
   - **Class A — suspected artifact/regularity (§11a).** Repeated identical
     marks, axis alignment, right angles, starbursts, rings/halos,
     even spacing, and processing artifacts (dotted columns, seams).
     Frame correlation is Class A only when the frame or a numerical
     boundary caused it; visual alignment alone is diagnostic. A
     visual Class-A claim is a causal hypothesis, not a veto by itself.
     A confirmed Class-A construction mechanism on delivered output =
     the rejected look, full stop.
   - **Class B — formation implausibility.** Features ignoring their
     causes or surroundings: floating islands, rivers terminating
     without an endorheic basin or other drainage context, lakes without
     drainage context, unbroken escarpments,
     uniform shelf width.
   - **Class C — character/quality.** Blandness, weak anatomy,
     insufficient variety. Direction for improvement, not rejection.
   - **Class D — render/palette literacy.** Not a terrain defect;
     routes to render decisions or the literacy notes.
7. **Spot verification.** The orchestrator verifies a sample of claims
   against images/instruments; per-judge false-claim rate is tracked.
   A claim that names a location that shows nothing is a judge defect.

## Judge literacy notes (canon facts — prevent false positives)

- All references are **hypsometric**: color is elevation only. Lowland
  dark-green mottling is terrain texture, not vegetation (§12 —
  mis-read once already).
- Grey-violet plateau fill = high-elevation coloring, not a material
  overlay.
- Single-pixel bright dots on canon plateaus are §7e floor lakes —
  a protected feature that reads as "noise speckle" at crop scale.
- Small volcanic islands can read as cross/star marks at pixel scale —
  the confirmed "stamp" false-positive mode (both judges, twice).
- Some references carry day/night darkening and globe-projection
  stretching; neither is a canon property (§12). Nor is
  land-touching-frame.
- At native screenshot zoom the references show pixel-scale
  square-step crenellation on every boundary and 1–2 px constant-width
  shoreline strokes — raster quantization of the source screenshots,
  not landform detail (flagged by two s4d judges as "repeated
  geometric marks"). Judge at feature scale, not pixel scale.
- Drowned-shelf hills with pale summit cores ("green ring + pale core
  islets") read as "repeated bullseye stamps" in 1024² crops (both M2
  panel judges, against ref2) — they are the §10 drowned-landscape
  look; require size/spacing evidence before accepting such a claim
  against shelf island fields.
- Canon lowland mottle reads as "one constant-grain noise field
  stamped over everything" at 1024² crop scale (both M2 panel judges,
  both refs) — same crop-scale artifact family as the crenellation
  note; texture-response claims must be judged at feature scale.
- The volcanic star-massif false positive recurred at trial scale (M2
  judge JA dinged canon for it twice, and both panel judges A-flagged
  ref2's large star massif) — it remains the most reliable way for a
  judge to mis-score canon.

## The author verdict library

Every author ruling is recorded (the negative-baseline catalog of
2026-08-29 is the seed). Production rulings live in
[`MILESTONES.md`](MILESTONES.md); private border/composition dispositions
live in [`ATTEMPT_REGISTER.md`](ATTEMPT_REGISTER.md); saved evaluation
bundles preserve the underlying reviews. Each future addition records
the source image or hash, ruling, date, and supersession state.
Protocols and judges are regression-tested against this durable record:
**any protocol or judge that would approve a previously rejected image
is broken by definition** and gets fixed before its output is trusted
again.

## Operational rules

- The builder never grades its own output; orchestrating a judging run
  and verifying judges' cited evidence is fine — issuing the verdict
  is not.
- Judges: fresh context, blind, ≥2, independent; disagreement across
  reruns = low-confidence signal, discounted.
- The prompts and evidence schema are provider/model neutral. Record
  provider and model metadata for every judge, and disclose a
  one-provider/model-family cohort as a limitation rather than implying
  cross-family independence.
- Every judging run persists its full bundle beside the trial set.
- Review is batch galleries across seeds/sizes (§14) — never a single
  hand-picked example. Contact sheets are first-class output.
- Evaluation harness code is versioned with the pipeline; harness
  changes that alter scores are called out at review like any
  output-changing change.
- **Probes are instruments and get calibrated before their numbers are
  trusted**: validate the detector on a known positive and a known
  negative first. (Learned 2026-08-29: a 4-neighbor isolation test
  counted diagonal river chains as "speckle," inflating a count ~15×,
  and a palette-tuned mask silently returned zero on canon images —
  both read as findings until checked.)
- **D4 long-ruler statistics are calibrated as of 2026-08-31**
  (`eval/geometry_instruments.py`, `tests/geometry_instrument_checks.py`):
  thresholded, perfectly isotropic fBm masks show a ~0.70 near-D4
  long-ruler fraction (p ≈ 0 under the analytic 11/45 rotation null),
  because curved digital boundaries locally quantize into axis/diagonal
  runs regardless of formation. **The analytic-null D4 gate is
  therefore not evidence-grade on its own** — it fires on any raster.
  Formation-caused grid lock claims require comparison against a
  matched isotropic baseline on the same grid and component scale, or
  independent evidence (rectangle tripwires, manual review). The
  B16/B17 D4 legs used the analytic null and their near-fractions
  (0.447, 0.385) sit *below* the isotropic baseline; those two
  rejections stand on their manual-morphology legs, not on D4.
  Approved M1 output measures 0.36–0.46 near-D4 on the same
  instrument. Known-negative and power-floor arms are also pinned in
  the calibration suite (a blocked null with k single-ruler blocks
  cannot beat p = (11/45)^k).

**Frozen prompt debt (2026-08-31).** The source `2afc_v2` and
`critique_v2` templates are byte-locked to the existing immutable Run 1
bundle. Their examples are schema-like pseudocode rather than literal
JSON, and their older Class-A wording does not fully encode the current
causal-only treatment of frame alignment and visual regularity. Do not
silently edit or reuse them for a new run: introduce a new prompt ID,
valid JSON examples, and the current causal rules, then treat the result
as harness-changed.

The archived M2/M3 panel rubrics at
`out/m2/eval/panels/rubric.md` and `out/m3/eval/panels/rubric.md` are
Windows-1252 evidence artifacts. Decode them accordingly; do not
normalize their bytes in place and invalidate archival identity.

## Current private-evaluation status (2026-08-31)

No private border/composition attempt has been promoted. Those
feasibility runs used frozen deterministic gates, causal evidence, and
manual morphology review; none claims a complete canon 2AFC production
evaluation. The authoritative sequence and dispositions are in
[`ATTEMPT_REGISTER.md`](ATTEMPT_REGISTER.md).

## Historical run snapshots — non-normative

The sections below preserve what was true at each dated run. Their
pending items and provisional interpretations do not override the
current protocol or later records.

### Most recent production snapshot (2026-08-30, Run 1 / 0.3.1-m3)

Run 1 evaluation is complete under `m3-eval-v2`: three fresh blind
judges each for the archived-M3 bridge, Run 1 2AFC, and diagnostic
panels. The bridge reproduced M3 at **24/24, 8/8 arms unanimous**.
Run 1 scored **17/24 (70.8%), only 1/8 arms unanimous, pairwise
agreement 10/24**: TA 8/8, TB 8/8, TC 1/8. Those aggregate/unanimity
figures are lower than the separate bridge cohort's, but fresh
judge/context sampling confounds the comparison; it is a weak
cohort-level signal, not causal evidence of aesthetic improvement. It
also does **not** pass the standing progress rule because the strongest
detecting judges remain 8/8. TA/TC and CB2 raw confidence values are reported
but not treated as reliable supporting evidence after high-confidence
ref-vs-ref calibration choices; selections remain unweighted in the
score.

All 25 Class-A/B claims on distinct candidate panels were verified:
23 supported, two partly supported, none wholly unsupported. The
remaining output tells are D8-axis river reaches, continuous-terrain
geometric/aligned submarine relief, a connected perimeter orogenic
belt, distance-dominated shelf halos, and low-relief interiors whose
appearance the stepped palette amplifies. Canon controls exposed a
large rubric defect: of ten canon A/B claims, six were unsupported,
two were severity overcalls, and only two (the same orthogonal island
coastline) were supported. Raw severity counts are therefore not a
valid comparison without adjudication. The byte-identical duplicate
probe passed exactly for all three judges.

Harness changes affecting comparability are explicit: v2 prompts add
void/evidence requirements and expanded feature-scale literacy; the
near-black crop filter has known-positive/negative calibration; builds
are atomic, no-overwrite, archive-safe, and fully hashed; scoring fails
closed. Exact scale matching remains unresolved because canon images
lack km/px metadata. Blindness was protocol-enforced rather than an OS
sandbox, all judges were one provider/model family, and no author
sign-off is implied. Full record:
[`out/m3_run1/run1_evaluation.md`](out/m3_run1/run1_evaluation.md);
machine-readable scores and verification live beside its immutable
bundle.

### Historical snapshot (2026-08-30, post-M3)

M3 run (3 2AFC judges, 2 critique judges, first duplicate-panel
reliability probe): **2AFC 24/24, unanimous, uniform confidence 4 — a
recorded REGRESSION on the yardstick** (series: s4b 15/16 → M2 9/16
judge-split → M3 24/24). Cause named in nearly every verdict: the new
river layer's right-angle geometry, a single dominant Class-A tell
that overrode M3's coupling gains (verified; render fixed same-day —
smooth channel curves, water-mask compositing, terrain-cut lake
shorelines — next run measures the residue). Duplicate-panel probe
PASSED cleanly (near-verbatim reproduction by both judges).
Calibration arms clean across all five judges. Canon panels drew 1–3
A-claims each, all inside documented literacy families
(stroke/crenellation/mottle) — the rubric notes do not fully suppress
them at 1024; revise before the next run. Full record:
out/m3/m3_judges.md (+ verbatim critique companion).

### Historical snapshot (2026-08-29, post-M2)

Exists: productionized builders `eval/m2_trials.py` (2AFC, bland
filter, calibration arms, external key) and `eval/m2_panels.py`
(severity-tagged critique, canon defendants); instruments live in the
engine (isobaths, slope); tripwires wired into the map report
(§3a/§6a/§6b/§7i/§8 + nearest-land-to-border); full M2 run bundle at
`out/m2/eval/` with verdicts + scoring + verification in
`out/m2/m2_judges.md` (+ verbatim critique companion file).

**M2 baseline (the standing yardstick):** s4b spike baseline was
15/16 detectable. M2 run: **JB 8/8, JA 1/8 (combined 9/16), 1/8
inter-judge agreement on candidate arms; calibration arms clean (low
confidence, split picks) for all judges.** Interpretation of record:
M2 output is no longer *unanimously* distinguishable from canon — one
judge was systematically convinced by the coupling features (ridge
lineations, arc–trench pairing, shelf gradation) — but a calibrated
judge still detects it 8/8 via a specific verified tell list (see
m2_judges.md §2). Track BOTH numbers next milestone; progress claims
require the detecting-judge accuracy to fall, not just the average.

Severity scoring: applied for the first time — candidate panels drew
novel verified A-class claims; canon panels drew A/B only in
documented literacy classes. Class separation intact.

Pending: km/px matching (scale metadata now exists in the report —
wire it into trial construction next run); bland-filter tightening
(one dim night-side ref crop passed); a reliability-probe duplicate
panel next panel run; regression-test judges against the M2 verdicts
now in the library.
