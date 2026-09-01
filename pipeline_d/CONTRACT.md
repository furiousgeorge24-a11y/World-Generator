# Contract — land-origin output requirements

This document is normative. It defines what a conforming Pipeline C result
must provide and the causal constraints under which that result may be
obtained. It deliberately does not prescribe a particular land-formation
algorithm.

Run 1 only freezes this contract and its supporting laboratory. It contains
no generator, model, generated output, or evidence that these promises are
already feasible.

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

Input is a seed, map geometry, `target_land_percent`,
`landmass_fragmentation`, and versioned implementation settings.

## 2. Authoritative grid and land measurement

- The delivered raster is bounded and non-wrapping. An internal parent may
  be periodic or otherwise boundary-neutral, but wrapping is never a
  property of the delivered map.
- Horizontal geometry is stated in physical world units. Resolution changes
  sample the same world more finely rather than changing what the world is.
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

## 3. Land-amount control

`target_land_percent` is a continuous author control with the inclusive
request range **0 through 70** and default **35**.

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

For a fixed seed and map geometry:

- all target and fragmentation variants use the same keyed latent
  randomness;
- the delivered window origin, size, orientation, and parent identity are
  fixed before either control is varied;
- a target or fragmentation change may not select a different crop, search
  for a more convenient window, replace the seed, or trigger hidden retries;
- unrelated latent structure may not reshuffle merely because a control
  moved.

The report must expose stable identifiers for the latent world and delivered
window so these promises can be checked directly.

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

## 11. Quarantine

The one-time bootstrap inheritance is recorded in
[`BOOTSTRAP_MANIFEST.md`](BOOTSTRAP_MANIFEST.md). After M0 closes, Pipeline C
may use only its own implementation and evidence plus the declared shared
root WebUI shell. Root reference images may be consumed only as external
perceptual evidence after the selected assets and hashes are frozen into the
evaluation bundle. Future development may not import from or consult
`pipeline_b` code, output, private experiments, or evolving documentation.
