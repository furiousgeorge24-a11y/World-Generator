# Roadmap

What still has to be built, in order, and the design questions each stage has
to answer. Current state is in [`STATUS.md`](STATUS.md); the promises being
built toward are in [`CONTRACT.md`](CONTRACT.md).

## Build order

C03 (foundation and kinematic history) exists. The rest do not. Each run
owes a view for every layer, mask, and intermediate field it introduces, per
[`VIEWS.md`](VIEWS.md).

| Run | Responsibility | What it has to get right |
|---|---|---|
| C03 | Foundation and kinematic history — periodic domain, sampler, mantle drive, strength and damage, velocity solve, emergent plates and boundaries. No crust. | Boundaries that curve, segment, and change regime along their length, on a spread of seeds |
| C04 | The seam formulation of [`DESIGN.md`](DESIGN.md) §3.6 behind a switch, production byte-identical: seam damage by slip work, tip propagation, nucleation, healing and merging. The corner search's dials and the existing views. | Whether cracks close loops and cut pieces, or craze or dead-end. Plate count and network share on the twelve development seeds |
| C04.1 | A slipping seam stays weak: `work_damage` honoured under `seams`, so a seam damages by its slip rate and heals only when it stops | Whether seams persist, and whether the loops they close cut pieces |
| C04.2 | The block model of [`DESIGN.md`](DESIGN.md) §3.6's last paragraph behind `seams = 2`: pieces are rigid bodies coupled through seam tractions, the stress the seam rules read is the sheet solve of the drag a piece failed to match, and seams are carried on markers that cannot duplicate | Whether the velocity view shows bodies, and whether a loop that closes encloses a plate rather than a crumb |
| C6 | Crust on markers — creation, transport, thickening, subduction, arcs, rifting, hotspots, age | Coherent at world scale while curved and irregular locally, without becoming independent noisy squiggles |
| C7 | Structural history and crustal state — material identity, age, deformation, margin classes, inherited structures | Believable large masses, asymmetric margins, subordinate fragments, broad quiet interiors |
| C8 | Vertical structural response — uplift and subsidence causes, continuous structural elevation and depth | Relief that follows structure without tracing it |
| C9 | First exposure — water reference, sea-relative height, the first authoritative land mask | Coastlines that read as flooded structure, not as thresholded noise |
| C10 | Island and margin families — arcs, margin continuations, detached blocks, drowned platforms | Varied causal island families with visible structural continuity |
| C11 | Land-target controller — deterministic global solve for the requested land amount | Target met without final-mask edits |
| C12 | Window selection and exact delivery border — deterministic observer over already-formed parent geography | Exact one-pixel water ring obtained naturally |
| C13 | Fragmentation — upstream reorganization of roughly the same land budget | Credible separation, not cut channels or scattered stamps |
| C14.n | Focused failure-reduction runs, one predeclared failure class each | Failure rate falling without relaxed gates |
| C15 | Frozen validation on the untouched 32 seeds | All hard gates pass on fresh evidence |
| C16 | Port-readiness and generator-readiness decision | Interface, limits, and risks are explicit |

## Milestone exits

**M1 — land-target architecture** (C4–C12). Every delivered development map
inside its ±10-point band with an exact causal water ring; target sweeps that
do not reverse beyond one cell; no mask edits, seed retries, crop relocation,
or numerical-boundary dependence; morphology plausible enough to justify
adding the second control.

**M2 — fragmentation and orthogonality** (C13–C14.n). Every cell of the
target × fragmentation matrix still target-conforming; fragmentation acting
upstream on common latent randomness with a stable cohort-level response and
no body-count promise; ablation evidence justifying the mechanism. Adapter
readiness first becomes considerable here.

**M3 — validation and port-readiness** (C15–C16). Implementation, defaults,
controls, matrix, gates, and stop rules frozen before execution. No validation
result may cause tuning inside the same run.

## Open design questions

Each needs a real answer before the run named, not an assumed default.

| Question | Due |
|---|---|
| What kinematic/history model and boundary classifications are materially necessary? | C6 |
| How do structural causes map to broad elevation and the water reference, with no frame-relative term and no rule that traces every boundary into coast? | C8 |
| What window-candidate generation, ordering, budget, and rejection policy? Geography must already exist; exhaustion fails honestly. | C12 |
| How is one window certified for an entire control family without becoming a control itself? | C12 |
| At what upstream stage, and by what mechanism, does `landmass_fragmentation` act without component surgery? | C13 |
| What failure classes and rates are acceptable? Decide from development evidence, never after seeing validation. | C15 |
| What target × fragmentation × resolution matrix is feasible and sufficient? | C15 |
| What performance envelope and dependency ceiling does a port need? | C16 |
| Should rectangular delivered maps ever be supported? Internals stay rectangle-capable; no public promise. | C16+ |
| What exact evidence permits the adapter to become `ready: true`? | C16 |

## Known feasibility risks

- A fixed window that stays causally water-bordered across the full target and
  fragmentation range may be hard to supply at high land targets.
- A ±10-point final target may require coupling material inventory and water
  exposure without turning either into a final-mask correction.
- Fragmentation `0` is only meaningful with enough macro-scale land; low
  targets can expose separated remnants over coherent material.
- A process can pass every numeric composition gate while producing repeated
  blobs, lace, rulers, common-radius bodies, or banding. This has now happened
  twice — see [`STATUS.md`](STATUS.md).
- Raster geometry instruments can accuse naturally curved geometry. They need
  calibration against matched isotropic controls on the same grid and
  component scale before they gain any authority.
- Resolution or parent-domain changes can destabilize a composition solve.

## Working rules

What survives now that the review/approval apparatus is gone:

- **Change one mechanism at a time**, and say what you expect it to do before
  running it. Otherwise a cohort of twelve worlds tells you nothing about
  which change caused what.
- **Judge on a spread of seeds, not a good one.** Both rejected attempts
  looked defensible on a hand-picked example and repetitive across twelve.
- **Numeric gates do not settle morphology.** C01 and C02 each passed every
  diversity and anti-cellularity threshold declared for them and were still
  wrong on sight. Metrics are for catching regressions, not for granting
  approval.
- **Every layer, mask, and process gets a view**, and an agentic judge screens
  it before the author looks. An unviewed field is an unexamined assumption —
  see [`VIEWS.md`](VIEWS.md). This applies to every stage from C6 onward.
- **The engine stays deterministic**: same seed, same world, no wall-clock or
  traversal-order dependence. This is what makes regenerating cheaper than
  storing.
- **Honest non-delivery beats a placeholder.** If a stage cannot produce
  something real, it produces nothing and says so.
- **Validation seeds stay unused** until the implementation is frozen, so
  there is at least one honest test left at the end.
