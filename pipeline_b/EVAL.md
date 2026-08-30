# Evaluation protocol — pipeline_b

The single source of truth for how output quality is judged. Grew out
of Phase 0 spike S4 (see MILESTONES.md for the run history); supersedes
the rule fragments scattered through those status blocks. CONTRACT.md
§11/§11a/§14 govern; DESIGN.md carries the one-paragraph summary.

**Why this exists:** a prior pipeline's output was eagerly approved by
the AI that built it. The failure had three causes — self-grading,
gestalt questions, no evidence requirement — and every rule below
traces to one of them.

## The hierarchy

1. **Deterministic tripwires veto; they never approve.** Cheap
   necessity tests prove specific ugliness (a map with no shelf-depth
   mass has no shelves); no metric certifies beauty.
2. **Blind judges compare; they never see their own work.** All
   perceptual judgment is comparative (against canon or rubric), by
   fresh-context judges with no knowledge of what was built or which
   side is which.
3. **The author decides.** The harness's job is narrower than taste:
   nothing may reach the author carrying an unearned recommendation.

Generation is never blocked by any of this (§5): tripwire violations
ship as report findings beside the map.

## Layer 1 — instruments (M2+)

Views designed to make defects glaring rather than arguable. Every
generated field ships a view (§14); these are the derived ones:

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
- **Row/column statistics** — frame correlation (§3b).
- **Cause-field overlays** — §11 cross-examination: shelf width against
  margin type, trenches against boundary fields, valleys against flow.
  A pretty picture whose supposed cause doesn't correlate is a painted
  symptom.

## Layer 2 — deterministic tripwires (from M1 on)

Run on every batch; results into the run report. Veto-only.

- Outer-ring land check + nearest-land-to-border distance (§3a).
- Depth-histogram mass in the shelf band (§6b) — no mass, no shelves.
- Shoreline gradient distribution — plummet detection (§6a).
- Per-landmass shelf-width distribution — uniformity detection (§6c).
- Dendritic-texture-below-the-slope detector (§6g — contract names
  this check).
- Lake census: count/size distribution (speckle and mega-lake, §9).
- Spectral/autocorrelation peaks — tiling, even spacing (§11a).
- Frame-correlation statistics (§3b).
- Determinism: same seed+controls → bit-identical output; control
  isolation: out-of-stage fields untouched when a downstream control
  moves (§4).

## Layer 3 — trial type 1: blind 2AFC ("imposter") — the scalar metric

Judges see pairs of tiles, exactly one from the canon (`examples/`),
and pick which is canon-grade. Accuracy near 100% = trivially
distinguishable; **progress = accuracy falling toward chance.**

Construction rules (each learned the hard way in S4/S4b):

1. **Feature-targeted crops.** Random crops land on featureless ocean
   and void the trial. Crops must contain judgeable formation; the
   bland-tile filter (variance + gradient + near-black tests) is
   mandatory and still needs tightening — a featureless pair slipped
   through and drew a confidence-5 verdict.
2. **Scale matching.** Compare like km/px with like once the pipeline
   exposes scale; unmatched zoom levels are a confound.
3. **Palette preserved.** Never grayscale — §12.7: the ramp carries
   half the look. Render candidates through a comparable stepped ramp.
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
   - **Class A — artifact/regularity (§11a).** Repeated identical
     marks, axis alignment, right angles, starbursts, rings/halos,
     even spacing, frame correlation, processing artifacts (dotted
     columns, seams). Confirmed class-A on delivered output = the
     rejected look, full stop.
   - **Class B — formation implausibility.** Features ignoring their
     causes or surroundings: floating islands, rivers that never reach
     the sea, lakes without drainage context, unbroken escarpments,
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
2026-08-29 is the seed; MILESTONES and memory hold the additions).
Protocols and judges are regression-tested against it: **any protocol
or judge that would approve a previously-rejected image is broken by
definition** and gets fixed before its output is trusted again.

## Operational rules

- The builder never grades its own output; orchestrating a judging run
  and verifying judges' cited evidence is fine — issuing the verdict
  is not.
- Judges: fresh context, blind, ≥2, independent; disagreement across
  reruns = low-confidence signal, discounted.
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

## Status (2026-08-30, post-M3)

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

## Status (2026-08-29, post-M2)

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
