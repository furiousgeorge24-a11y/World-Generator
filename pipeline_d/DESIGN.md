# Design — land-origin laboratory

The promises live in [`CONTRACT.md`](CONTRACT.md); evaluation lives in
[`EVAL.md`](EVAL.md). This document records architecture obligations and
candidate seams. During M0 it deliberately ratifies **no land-formation
mechanism**.

Run 1 contains no generator, model, generated output, or empirical result.

## 1. Design objective

Build the smallest independently useful module that can eventually answer:

1. Can naturally formed geography deliver any requested land amount from
   0% through 70% within ±10 percentage points on each map?
2. Can an author continuously shift organization from a likely dominant
   macro-landmass toward more fragmented major bodies without requesting a
   body count?
3. Can both controls preserve one seed's latent geography, one fixed window,
   and an exact causally obtained water border?

No downstream terrain system is allowed to compensate for a weak answer.

## 2. Non-negotiable architecture boundaries

Any candidate architecture must maintain these stage boundaries:

1. **Input and keying.** Seed, physical geometry, target, fragmentation, and
   versioned settings enter through a typed control schema. Stable keyed
   randomness defines a common latent world for every same-seed sweep.
2. **Latent parent context.** Control-independent stochastic fields, parent
   coordinates, and the delivered-window identity are established without
   inspecting a target-specific finished result.
3. **Causal control solve.** If global parameters must be solved to satisfy
   the requested land amount, that deterministic solve operates on causal
   budgets or rates, never on local final-mask edits. Candidate evaluations
   reuse the same latent context and fixed window.
4. **Upstream formation and minimal physical realization.** Land-causing
   state forms without frame knowledge, with fragmentation acting here or in
   an earlier causal parameter stage. The module produces only the elevation
   and water state needed to classify authoritative dry land and expose its
   causes.
5. **Fixed extraction.** One seed-and-geometry window identity is fixed
   independently of target and fragmentation and reused across their entire
   sweep.
6. **Evidence.** Final mask, causal fields, control response, and boundary
   proof are reported and renderable.

These are obligations, not an endorsement of a particular implementation.

## 3. Candidate hypotheses, not M0 decisions

The following are reasonable hypotheses for M1 design work but are not
ratified merely by appearing here:

- a periodic or otherwise boundary-neutral parent geography;
- an auditable inventory of buoyant or continent-capable material;
- a global water inventory or sea-level parameter that controls exposure;
- a deterministic scalar or low-dimensional solve coupling material supply
  and water exposure to the requested target;
- upstream growth, rifting, welding, separation, or accretion behavior that
  responds to fragmentation;
- a minimal continuous elevation field used to turn causal material state
  into the final water classification.

M1 may choose, combine, or reject these only through a predeclared attempt.
Nothing here authorizes copying an existing formation implementation.

## 4. Target-control obligations

The target controller must solve the authoritative final land percentage,
not a continental tag or pre-water proxy. If it uses a root finder or search:

- the searched parameters and bounds are global and physically interpretable;
- the solve is deterministic and uses no seed substitution;
- every iteration uses the already-fixed latent world and window;
- convergence, iteration count, brackets, and any failure are reported;
- failure survives as evidence and does not trigger a fallback geography;
- the final mask is observed, never patched.

Material supply and water exposure are expected to have different roles.
Water level alone may expose implausible substrate at high targets; material
supply alone may leave large seed-dependent exposure error. That is a risk to
test, not a conclusion that both mechanisms are mandatory.

Across increasing targets, realized land must not reverse beyond the one-cell
measurement tolerance and macro structure must remain recognizable. Exact set
inclusion of masks is not an architecture requirement, but wholesale
reorganization is a control failure.

## 5. Fragmentation-control obligations

Fragmentation enters before the final mask exists. A candidate may vary
natural rates or propensities such as separation, rifting, welding,
connectivity, survival, or accretion, but it may not inspect connected
components and edit them afterward.

The controller must aim to reorganize approximately the same land budget.
The target solver remains responsible for bringing every fragmentation
variant into the requested land tolerance. Reports expose the residual
target drift rather than hiding it.

The control uses the same latent random fields at every setting. A slider
change alters the promised process response; it does not draw from a new
world. Small secondary islands are intentionally not eliminated at zero.

Because `0` is a likelihood statement, its outcome is evaluated across a
predeclared cohort and supported with continuous dominance/area-spectrum
diagnostics. No implementation should optimize an island-count score.

## 6. Frame and numerical-domain obligations

The delivered frame is an address into geography, never a forming force.
Its origin, size, orientation, and parent identity are stable across every
target/fragmentation variant for a seed and geometry.

An implementation must establish at least one applicable independence route:

- intrinsically periodic or boundary-neutral construction;
- a proven causal-reach bound that cannot reach the delivered window; or
- adequate nested and shifted-domain invariance.

A large halo alone is not proof. The exact outer water ring is checked only
after final-resolution classification, and no edge operation is available as
a recovery path.

## 7. Audit fields and module interface

The eventual engine interface should return plain, portable arrays and a
structured report. Names may evolve during M1, but the semantic groups are
fixed:

- final land and water masks;
- minimal elevation and final sea level;
- causal land-material/support and organization fields;
- stable latent-world, parent-domain, and fixed-window metadata;
- target-solver trace;
- fragmentation diagnostics;
- boundary-independence diagnostics and findings.

Every material field needed to defend a contract promise receives a
diagnostic view. Scratch arrays with no evidentiary value need not be exposed.

Core logic should remain algorithmically portable: explicit stages, physical
units, ordinary array operations, deterministic iteration, and no UI state in
the model.

## 8. WebUI and shared-resource boundary

Pipeline C may use the declared root WebUI shell through a C-specific adapter
and control schema. The UI may display reports, compare same-seed sweeps, and
request diagnostic views; it may not contain formation behavior or quietly
rewrite controls.

The Run 1 adapter advertises the frozen two-control interface but reports
`ready: false` and fails generation closed. M1 target development may use
offline harnesses and may expose diagnostic artifacts for inspection, but it
does not make the adapter ready while fragmentation is absent. The adapter
may report ready only after every advertised control has an implemented causal
path and its promised range has passed the applicable milestone evaluation.
A control may legitimately have little visible effect on an individual seed;
it may not be disconnected, ignored, or constant across the calibrated
cohort.

Selected root reference images may calibrate perceptual formation review only
after their paths and hashes are frozen into that evaluation bundle. They are
external evidence rather than runtime dependencies or composition targets,
and their land touching the image frame is not a property to reproduce.

No engine dependency crosses the quarantine boundary described in the
bootstrap manifest and contract §11.

## 9. Known feasibility risks entering M1

- A fixed window that remains causally water-bordered across the full target
  and fragmentation ranges may be difficult to supply at high land targets.
- A ±10-point final target may require coupling material inventory and water
  exposure without turning either into a final-mask correction.
- Fragmentation zero is meaningful only with enough macro-scale land; low
  targets can expose separated remnants even over coherent underlying
  material.
- A process can pass numeric composition while producing repeated blobs,
  lace, rulers, common-radius bodies, or other visibly constructed geometry.
- Raster boundary statistics can accuse natural curved geometry; instruments
  require calibrated controls and causal follow-up.
- Resolution or parent-domain changes can alter a seemingly stable
  composition solve.

These risks are reasons for the staged evaluation plan, not permission to
weaken the contract after seeing output.
