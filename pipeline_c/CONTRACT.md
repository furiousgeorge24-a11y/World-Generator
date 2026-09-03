# Contract — land-origin output requirements

This document is normative. It defines what a conforming Pipeline C result
must provide and the causal constraints under which that result may be
obtained. It deliberately does not prescribe a particular land-formation
algorithm.

This contract commits to tectonic organization as a high-level
responsibility. It ratifies no numerical tectonic algorithm, and nothing in it
is evidence that its promises are already feasible.

## 1. Scope

Pipeline C is a **land-origin module**. Its eventual deliverable is a bounded
land/water geography with enough causal state to show why land formed and why
it is exposed. At minimum, a run must provide:

- the authoritative final-resolution boolean land/water mask;
- the minimal elevation, sea-level, material, and formation fields used to
  derive that mask;
- the parent-domain and delivered-window identity;
- a machine-readable report and diagnostic views.

Detailed mountains, seafloor anatomy, erosion, rivers, lakes, sediment,
climate, biomes, settlements, names, lore, and finished cartography are out
of scope. Diagnostic rendering exists to judge land formation; it is not a
claim to a finished map.

Land origin includes the causal systems logically required before first land
exposure. Tectonic organization is the primary intended source of persistent
structure. Crustal or material history and vertical realization belong here
only to the extent needed to establish the initial exposed geography and its
provenance. The module ends at the authoritative initial land/water state;
downstream terrain finishing may not be used to rescue it.

The following formation character is a normative perceptual obligation:

- primary tectonic actors remain few enough to read at world scale and leave
  broad, comparatively quiet interiors; dense equal-scale cellular tilings
  are contrary to the intent;
- boundaries form long, coherent, curving systems with segmentation and local
  raggedness rather than straight polygon seams or uncorrelated noisy lines;
- islands arise in varied causal families, including boundary arcs, margin
  continuations, detached blocks, drowned shelves or platforms, fragments of
  larger masses, and occasional isolated sources;
- tectonic boundaries often influence coastlines, but the two are not the
  same object: boundaries may be inland, offshore, submerged, or inherited,
  and passive coasts may have no nearby active boundary;
- coastlines derive from continuous vertical state and water exposure. A
  boundary may influence that state but may not simply be traced into the
  final mask.

Until calibrated thresholds exist, these obligations are judged across
predeclared cohorts and stage views. They do not imply an exact plate,
boundary, landmass, or island count.

Input is a seed, map geometry, `target_land_percent`,
`landmass_fragmentation`, and versioned implementation settings.

## 2. Authoritative grid and land measurement

- The delivered raster is bounded and non-wrapping. An internal parent may
  be periodic or otherwise boundary-neutral, but wrapping is never a
  property of the delivered map.
- Horizontal geometry is stated in physical world units. Scale, in
  kilometres per delivered pixel, is an author input with a fixed default.
  Resolution and scale together size the delivered window, and the simulated
  parent world is sized from that window, so a different resolution or scale
  is a different world. Features keep their physical size and their on-screen
  size at every resolution; a smaller map is a smaller piece of a smaller
  world, never a coarser sampling of the same one.
- The authoritative land mask is evaluated at the final delivered
  resolution after the module's minimal elevation and water solve. Structural
  tags, continental material, a coarse census, or a pre-water proxy cannot
  substitute for it.
- A cell is land only when the final authoritative water classification says
  it is dry. Every water cell, including any inland water the eventual model
  may produce, counts as water.
- Realized land percentage is:

  `100 × final dry-land cell count / delivered cell count`

  The denominator includes the required water border.

The delivery target is a square `size × size` raster with default
`size = 1024` at a default scale of `5 km/px`. Every material attempt
predeclares its supported sizes, its scale range, the parent-to-window
ratio, and its sampling convention. Internal geometry carries width and
height independently so square delivery is not a hidden architectural
assumption. Rectangular delivery is not yet claimed. Scale is world
geometry, not a formation control: it is held fixed, like the seed, by
every same-family sweep of the author controls, and it is never swept.

## 3. Land-amount control

`target_land_percent` is a continuous author control with the inclusive
request range **0 through 70** and default **50**.

For every conforming delivered map:

`abs(realized_land_percent - target_land_percent) <= 10 percentage points`

This is an absolute percentage-point tolerance, not a relative percentage
and not a batch-average promise. Each individual delivered map must pass.

Endpoint semantics are frozen as follows:

- A request of `0` accepts a realized result from **0% through 10%** land.
- A request of `70` accepts a realized result from **60% through 80%** land.
- `70` is the maximum author request, not a hard ceiling on realized land.
  Values above 80% are non-conforming for every permitted request.

Across an increasing same-seed target sweep, realized land percentage must
not decrease beyond one final-mask cell of measurement tolerance. The
same-seed geography must remain recognizably related; the control may expose
or causally create more land, but it may not reroll the world. This does not
require every lower-target land cell to remain land at every higher target,
because a legitimate physical response may reorganize local boundaries. It
does require structural continuity and a globally non-decreasing response
apart from that explicit one-cell tolerance.

## 4. Fragmentation control

`landmass_fragmentation` is a continuous author control with the inclusive
range **0 through 1** and default **0.5**.

- At `0`, where enough land exists for a macro-landmass to be meaningful,
  the formation process has a strong likelihood of producing one dominant
  contiguous macro-landmass.
- Small boundary islands, barrier islands, coastal islands, volcanic
  islands, and comparable secondary bodies remain natural and valid at `0`.
- Increasing the control shifts the land organization toward more separated
  major bodies and, at the high end, more archipelagic organization.
- The control does **not** promise a continent count, island count, or exact
  connected-component count on any individual map.
- At very low realized land, flooding may divide exposed remnants even at
  `0`. Evaluation must distinguish the organization of the underlying major
  land material from incidental tiny exposed components.

Fragmentation is a cohort-level tendency as well as an author control. Its
behavior must be visible across predeclared same-seed sweeps, using
world-scale component-area and dominance diagnostics rather than a hard body
count.

Fragmentation is approximately land-budget-orthogonal: it reorganizes the
available land instead of serving as a second land-amount slider. Every
fragmentation setting must independently satisfy §3, and the residual land
change across a fragmentation sweep must be reported.

## 5. Common latent world and fixed delivered window

For a fixed seed, physical delivered geometry, and implementation version:

- all target and fragmentation variants use the same keyed latent
  randomness;
- exactly one delivered-window identity is selected for the entire control
  family, and its physical origin, extent, orientation, and parent identity
  are reused at every target and fragmentation value;
- a target or fragmentation change may not select a different window,
  replace the seed, or trigger hidden retries;
- unrelated latent structure may not reshuffle merely because a control
  moved.

A deterministic selector may observe a larger, already-formed parent world.
That selector is conforming only when all of the following are true:

- candidate enumeration, inspected fields, eligibility rules, ordering, and
  tie-breaking are predeclared and versioned;
- candidate geography forms without candidate-frame distance, direction, or
  border inputs;
- selection occurs once for the whole control family, never independently
  for a requested target or fragmentation setting;
- selection may test declared deliverability predicates, including exact
  border eligibility, but may not rank target accuracy, resemblance to a
  reference, perceptual attractiveness, or any undeclared convenience;
- every candidate, observation, rejection reason, and final decision is
  preserved in the report.

This fully reported observation of existing geography is not the hidden
best-of-many selection prohibited by §7. The exact selector and the fields it
may inspect remain an implementation precommit. If no eligible window exists,
the run reports `NO_VALID_WINDOW`; it does not retry a seed or alter geography.
The report exposes stable latent-world and delivered-window identifiers so
these promises can be checked directly.

## 6. The water border

Two requirements apply simultaneously.

**6a. Exact output invariant.** Every cell in the outermost ring of the final
delivered-resolution mask is water on every conforming map. Water depth is
irrelevant. The final mask is authoritative.

**6b. Causal border.** No formation, elevation, water, or corrective law may
consume distance or direction to the delivered frame. Prohibited mechanisms
include forced-water rings, edge masks, fades, tapers, clamps, border-aware
relief changes, and any equivalent post-hoc operation. A numerical boundary
that can affect delivered geography also violates this rule unless
independence is closed by a boundary-neutral construction, a sufficient
causal-reach proof, or adequate nested and shifted-domain invariance.

The fixed window may extract already-formed geography; extraction may not
alter it. A coastline or other feature that happens to run straight,
parallel the frame, turn near a corner, or crowd an edge remains valid when a
causal audit establishes that its formation was independent of the frame.
Visual alignment is a diagnostic tripwire, never proof of a violation by
itself.

Observational extraction of a rectangle is allowed. “Cropping land to fit”
is not: clipping, deleting, lowering, flooding, tapering, adding a moat, or
changing elevation, bathymetry, or water level as a function of frame distance
or direction all alter the parent geography and are prohibited. The water
depth of the exact one-pixel ring is irrelevant, and no wider clearance or
moat is required. A naturally formed shoreline may parallel, turn with,
closely approach, or perfectly contour the rectangle. The decisive audit is
whether overlapping parent state is unchanged when the frame is absent,
moved, or differently visualized.

## 7. Process-footprint and naturalness rule

> Every visible land body and separation must be the footprint of a natural
> process the generator actually models, never a finished shape placed or
> edited to satisfy a desired appearance or score.

A deterministic global solve for causal parameters is allowed. For example,
an implementation may solve a global material, buoyancy, or water budget to
meet the requested land amount, provided the solved parameter drives the
modeled process and is fully reported.

The following are prohibited:

- painting, dilating, eroding, clipping, or threshold-patching the finished
  land mask to hit the target;
- identifying finished components and cutting channels, deleting bridges,
  welding bodies, scattering islands, or otherwise performing component
  surgery to obtain a fragmentation setting;
- placing a feature because a crop needs land or water at a location;
- target-driven crop search, seed retry, best-of-many hidden selection, or a
  fallback geography;
- stamping, tiling, mirroring, or repeatedly placing recognizable landform
  shapes;
- a post-hoc noise, smoothing, or blending layer whose only justification is
  to conceal an unnatural construction signature.

Process-modulated stochastic initial conditions or parameter fields are
allowed when their causal role is explicit and auditable. Regularity is a
reason to investigate, not automatic proof that a natural process is absent.

## 8. Determinism and resolution independence

- Same seed, geometry, controls, settings, and version produce bit-identical
  output in the same environment and structurally equivalent output across
  supported environments.
- Wall-clock time, unordered iteration, and unordered reductions may not
  affect output.
- Randomness is keyed by stable stage and process identities. Moving one
  control cannot consume a different random stream and thereby reroll other
  structure.
- The same world rendered at different resolutions retains major landmass
  topology, placement, and coastline structure within a stated physical
  tolerance. Fine features may converge rather than match cell for cell.
- Any implementation or default change that alters output for the same inputs
  changes the version stamp.

## 9. Evidence, reporting, and provenance

Every attempted generation must preserve its diagnostic artifact and report,
including non-conforming runs. A hard failure blocks delivery or promotion as
a contract-satisfying result; it does not erase the evidence.

Outcomes use this top-level taxonomy:

- `generation_failure`: no eligible window, an unsupported geometry, an
  unbracketed or non-convergent target solve, or another declared condition
  prevents delivery; no map is emitted;
- `hard_gate_rejection`: determinism, identity, border causality, prohibited
  operation, provenance, domain, resolution, or other objective invariant
  fails;
- `behavioral_rejection`: a calibrated control-response, orthogonality, or
  structural-continuity obligation fails;
- `perceptual_rejection`: the author or predeclared independent review rejects
  morphology without claiming an objective causal violation;
- `infrastructure_error`: an exception, schema, renderer, artifact, or harness
  failure prevents a valid evaluation.

Every non-success records a stable code, class, stage, observed condition,
required condition, trace, and available artifacts. Honest failure is
permitted during development only as explicit, preserved non-delivery. It
never authorizes a placeholder or nonconforming map. The acceptable eventual
failure rate must be frozen before validation; it is not chosen after results
are seen.

Each report includes at least:

- seed, version, complete control and geometry echo;
- latent-world and fixed-window identifiers;
- requested and realized land percentage and absolute error;
- fragmentation setting and registered organization diagnostics;
- final outer-ring water result and nearest-land-to-border distance;
- every globally solved parameter, bracket, convergence result, and fallback;
- parent-domain construction and numerical-boundary proof route;
- timings, findings, and artifact hashes.

Every persisted or material field required to audit a promise has a
diagnostic view. Provenance accompanies every rendered artifact.

## 10. Controls and review

- Controls are data: name, type, range, default, tier, and promise.
- A promise holds across its entire range, including both endpoints.
- Controls are described in causal/process terms, not as instructions to draw
  a desired silhouette.
- Advertising a control while the bootstrap adapter is unavailable is allowed
  to freeze the interface. Marking the adapter ready is not: `ready: true`
  requires every advertised control to be implemented, connected to its
  promised causal process, and accepted across its range. An ignored,
  constant, silently clamped, or otherwise no-op advertised control keeps the
  adapter `ready: false`.
- While any advertised control or generation stage is unavailable, generation
  fails closed and emits no placeholder map that could be mistaken for a
  result.
- Review uses predeclared batches across seeds, target values,
  fragmentation values, and resolutions. A hand-picked success is not
  evidence.
- Deterministic conformance and causal proof are separate from perceptual
  morphology review. The builder may run checks but may not issue the
  perceptual verdict on its own output.
- The author makes promotion and taste decisions.
- Every meaningful engine-stage increment exposes the actual fields that
  exist through an inspection-ready WebUI mode and then stops for author
  review. Each registered stage supplies cause and state views and, once a
  consequence exists, an appropriate consequence or difference view.
- Comparison modes are current, the most recent explicitly author-accepted
  compatible baseline, and a typed semantic delta. A baseline never advances
  automatically to the most recent execution. Rejected snapshots remain
  immutable evidence.
- Stage approval is limited to the reviewed responsibility. It neither waives
  other contract requirements nor certifies the complete engine.

## 11. Quarantine

A narrowly authorized
read-only consultation identified only high-level categories of work that can
precede land origin; no implementation detail, algorithm, output, or model
claim was imported. That event is closed and recorded without preserving its
contents in [`AUTHOR_RULINGS.md`](AUTHOR_RULINGS.md).

Pipeline C may now use only its own implementation and evidence plus the
declared shared root WebUI shell. Root reference images may be consumed only
as external perceptual evidence under
[`AUTHOR_RULINGS.md`](AUTHOR_RULINGS.md). Future development may not
import from or consult `pipeline_a` or `pipeline_b` code, output, experiments,
or evolving documentation.
