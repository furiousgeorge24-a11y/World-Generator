# Evaluation protocol — land-origin laboratory

[`CONTRACT.md`](CONTRACT.md) is normative. This file defines how its promises
are tested; on conflict, the contract wins. [`DESIGN.md`](DESIGN.md) records
architecture obligations and hypotheses, not acceptance evidence.

Run 1 establishes evaluation interfaces and fresh Pipeline C instruments. It
contains no generator, candidate output, scored gallery, or promotion claim.

## 1. Authority hierarchy

Evaluation separates three kinds of evidence:

1. **Deterministic hard gates** decide objective conformance: target error,
   exact final water ring, determinism, fixed-window identity, forbidden
   operations, and the applicable numerical-domain proof.
2. **Calibrated behavioral gates and diagnostics** measure control response,
   structural continuity, component organization, and suspicious regularity.
   A metric may veto promotion only when its necessity and calibration have
   been predeclared.
3. **Perceptual morphology review** judges whether the land actually looks
   naturally formed. The builder may prepare evidence and verify citations;
   it may not issue the perceptual verdict on its own output.

The author decides promotion. Numeric success cannot certify naturalness, and
a visual impression cannot waive a hard invariant.

## 2. Precommit and seed roles

Every model attempt declares before execution:

- implementation/version hash and allowed debug scope;
- debug seeds, development seeds, and untouched validation seeds;
- map geometry and supported resolutions;
- the exact target/fragmentation matrix;
- hard gates, calibrated behavioral gates, diagnostics, and stop rules;
- expected evidence bundle and perceptual-review protocol.

Debug seeds may be inspected while implementing. Development seeds may guide
an architecture decision. Validation seeds stay untouched until M3 freezes
the implementation and evaluation protocol. A failed seed is evidence, not
permission to substitute another. Hidden retries and best-seed galleries are
prohibited.

## 3. Minimum control sweeps by milestone

Before an M1 or later attempt runs, its exact matrix is frozen.

**M1 target-development minimum:**

- `target_land_percent`: `0, 10, 20, 30, 40, 50, 60, 70`;
- `landmass_fragmentation`: fixed at its registry default, `0.5`;
- identical seeds, geometry, latent-world IDs, and window IDs at every target;
- at least two supported output resolutions for structural comparison once a
  model claims resolution independence.

M1 establishes only the land-target architecture. Holding fragmentation at
its default is an explicit scope restriction, not evidence that the
fragmentation promise works. Because the registry already advertises that
control, the generator adapter remains `ready: false` throughout target-only
M1 work.

**M2 fragmentation-development minimum:** the full cross-product of
`target_land_percent = {0, 10, 20, 30, 40, 50, 60, 70}` and
`landmass_fragmentation = {0, 0.5, 1}`, using the same seeds, geometry,
latent-world IDs, and fixed window at every matrix cell. M2 may predeclare
denser fragmentation samples without discarding this common matrix.

Focused implementation probes may use a subset, but they cannot establish
range-wide conformance. No M1 or M2 development cohort constitutes fresh M3
validation.

## 4. Deterministic hard gates

Run on every relevant map or same-seed sweep:

### 4.1 Final land target

- Compute land percentage from the authoritative final-resolution boolean
  mask using CONTRACT §2.
- The public control is expressed as percentage points (`0` through `70`).
  Evaluator functions or manifests may use normalized fractions (`0` through
  `0.70`) only in explicitly fraction-named fields, with the exact relation
  `target_land_fraction = target_land_percent / 100`.
- Require absolute error at most 10 percentage points on every map.
- Check endpoint intervals explicitly: 0-request gives 0–10%; 70-request
  gives 60–80%.
- A batch mean cannot rescue an individual failure.
- Across each increasing same-seed target sweep at fixed fragmentation,
  realized land percentage must not reverse by more than one final-mask
  cell's fractional contribution.

### 4.2 Exact causal water border

- Every final outer-ring cell is water.
- Report nearest land to the border.
- Static and runtime audits reject reads of crop-relative distance,
  direction, masks, or border state by forming processes.
- Promotion requires a declared boundary-independence route: periodic or
  boundary-neutral construction, sufficient causal reach, or calibrated
  nested/shifted-domain invariance.
- Frame alignment alone never fails this gate; causal dependence does.

### 4.3 Determinism, identity, and no reroll

- Repeat identical inputs and require bit-identical masks, material fields,
  reports apart from explicitly non-authoritative timing data, and images.
- Across the full target/fragmentation matrix require identical latent-world,
  parent, and delivered-window identifiers.
- Verify common keyed latent random fields or their stable hashes.
- Reject seed substitution, target-driven crop search, control-dependent crop
  relocation, and unreported fallback attempts.

### 4.4 Prohibited final-mask operations

Static causal inspection and registered ablations must establish that no edge
edit, final-mask target patch, or connected-component surgery is present.
Passing numeric output is non-conforming when its cause violates CONTRACT §6
or §7.

### 4.5 Adapter readiness and control wiring

- A `ready: true` adapter must have an implemented generation path for every
  advertised control and every in-range value.
- Each control must be traceable into the promised causal stage and echoed in
  provenance; values may not be ignored, silently clamped, or replaced by a
  constant.
- A same-seed comparison plus registered cause-field evidence verifies wiring.
  A stochastic control need not visibly change every individual map, but its
  calibrated cohort must show the promised response.
- If the engine or any advertised control is unimplemented, the adapter stays
  `ready: false`, generation fails closed, and no placeholder artifact is
  accepted as output.
- M1 target-development evidence may be produced by its predeclared harness
  while the shared UI remains unavailable. Fragmentation and readiness first
  become eligible for acceptance in M2.

## 5. Control-behavior instruments

Fresh Pipeline C instruments must operate on semantic arrays rather than
palette inference. Their results live in the run report and are rendered in
same-seed panels.

### 5.1 Target authorability

- requested-versus-realized land curve and absolute error;
- monotonic-reversal count and magnitude;
- pairwise mask overlap/change maps between adjacent targets;
- macro-landmaterial identity retention and major-body correspondence;
- fixed-window and latent-field hash equality.

The scalar target and identity checks are deterministic. Structural-overlap
thresholds become gating only after calibration on known continuous and
rerolled controls; until then they are diagnostics paired with morphology
review.

### 5.2 Fragmentation behavior

Do not optimize or gate on raw island count. The M0 instruments report
8-connected raw component areas, largest-component land share, inverse
Simpson effective components, and component-area entropy as non-gating
diagnostics. Before M2 promotion, predeclare and calibrate a physical-area
macro/micro threshold, then report at minimum:

- largest macro-landmass share of total land;
- area-ranked macro-component spectrum;
- an area-weighted dispersion or entropy measure;
- separation/connectivity of major bodies;
- the same quantities for underlying land-causing material when low exposure
  makes the final mask misleading;
- realized-land spread across the fragmentation sweep.

The macro/micro area threshold is stated in physical units and calibrated
before it gains authority so barrier, coastal, and volcanic islands do not
overturn the zero-fragmentation promise. `F=0` is judged as a strong cohort
tendency toward one dominant macro-landmass where applicable, not as a
per-map component count. Increasing F should produce a consistent cohort
trend toward greater major-body separation while every map continues to pass
the land target.

### 5.3 Geometry tripwires

Views and metrics should expose repeated rounded bodies, common-radius arcs,
parallel/even bands, long rulers, square corners, D4/grid preference, lace or
web morphology, and feature-size collapse. These are causal-audit triggers,
not proof by appearance alone.

Raster geometry instruments must be calibrated against matched isotropic
controls on the same grid and component scale. An analytic direction null by
itself is not evidence-grade for digital boundaries.

## 6. Perceptual morphology review

Perceptual review uses fresh-context, blind, independent judges and the new
Pipeline-C-specific prompt IDs `land_origin_2afc_v1`,
`land_origin_critique_v1`, and `land_controls_sweep_v1`, as applicable. A
frozen prompt is never silently edited or reused under the same ID after an
evidence bundle consumes it. Provider/model metadata and any lack of model
family diversity are recorded. The implementation boundary is documented in
[`eval/README.md`](eval/README.md).

Judges receive formation-focused panels, not a finished terrain product:

- final land/water silhouette at a neutral diagnostic palette;
- parent geography with the fixed delivered window marked;
- causal material/organization views where appropriate;
- same-seed adjacent-target and adjacent-fragmentation comparisons;
- control panels and, when useful, root reference-image crops at an
  approximately comparable scale.

Reference land touching its image frame is excluded from scoring. Naturally
straight or frame-parallel geography is not a defect without causal evidence.

Each claim names what, where, and visible evidence, and is sorted into:

- suspected construction/regularity artifact;
- formation implausibility or missing causal response;
- character/quality weakness;
- cannot identify from the supplied evidence.

Praise carries the same evidence burden as criticism. Full judge responses,
trial order, keys, metadata, and scores persist beside the run.

## 7. Required evidence bundle

Every attempt bundle contains:

- precommit and source/version hashes;
- complete control matrix and seed-role record;
- machine-readable per-map and cohort reports;
- final masks and neutral diagnostic renders;
- requested-versus-realized curves;
- target difference/overlap sheets;
- fragmentation organization plots and panels;
- parent/window and causal-field views;
- deterministic, resolution, and domain-independence results;
- perceptual-review materials and verdicts when that stage is reached;
- final disposition, retained premises, and links to the ledgers.

Failed hard gates stop promotion but do not delete downstream-safe evidence.
An attempt may deliberately stop before perceptual review when no conforming
image exists; the absence of that review must be stated.

## 8. Milestone application

- **M0:** validate documentation, schemas, instruments, packaging, adapter
  boundary, and quarantine. There is no generator to score.
- **M1:** evaluate target accuracy, monotonicity, fixed identity/window,
  border causality, resolution/domain premises, and preliminary morphology at
  fragmentation `0.5`; keep the generator adapter unavailable.
- **M2:** add full fragmentation and target/fragmentation-orthogonality
  evaluation across the common matrix; only then assess whether every
  advertised control is implemented well enough for adapter readiness.
- **M3:** freeze implementation and protocol, then use untouched validation
  seeds for complete deterministic, domain, resolution, and blind perceptual
  review. No validation-seed tuning is allowed.
