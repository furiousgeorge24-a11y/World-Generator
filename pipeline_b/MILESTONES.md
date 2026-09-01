# Milestones — pipeline_b

Sequencing and exit criteria. Every phase ends with a batch gallery,
a value-ledger update, and a commit recommendation; the author judges
galleries and decides advancement.

Normative requirements are in [`CONTRACT.md`](CONTRACT.md); operational
judging rules are in [`EVAL.md`](EVAL.md). This file records production
milestone state and the detailed historical narrative. The canonical
cross-attempt index for private border/composition work is
[`ATTEMPT_REGISTER.md`](ATTEMPT_REGISTER.md), and feature cost/yield is
in [`VALUE_LEDGER.md`](VALUE_LEDGER.md).

## Current milestone status (2026-08-31)

| Milestone | State | Current qualification |
|---|---|---|
| Phase 0 / S1–S4 | Complete | Architecture and evaluation lessons carried forward. |
| M1 — structure | Complete | Advanced to M2; later border work rejected fixed central-frame closure and `center_pull`, not the shipped structural stage as a whole. |
| M2 — crude end-to-end | Complete | Advanced to M3; the current inherited suite is 23/24 because the 256² performance tier misses §15. |
| M3 — surface processes | Built and evaluated | `m3_checks` 53/53; 256² performance remains unresolved and author gallery sign-off is pending. |
| M4 — anatomy/provinces/texture | Not started | Held while the natural water-border and land-composition parent architecture remains unresolved. |
| M5+ — canon convergence | Not started | Depends on M4 and a promotable border/composition architecture. |

The current private conclusion is at [Private-attempt audit synthesis](#private-attempt-audit-synthesis-2026-08-31).

## Phase 0 — de-risking spikes (throwaway code, `spikes/`)

Spike code answers a question and is then discarded or rewritten
properly; it sets no architecture precedent and gets no controls,
provenance, or polish.

- **S1 — coarse tectonic history.** *Question:* does a simple
  time-stepped plate model at fixed coarse scale produce diverse,
  believable structure — few large domains, long connected belt
  systems, margins that vary along one continent — in negligible time?
  *Judged on:* structure views (crust type/age, belts, margin classes)
  across ~12 seeds; boundaries must not read as smooth splines
  **without any post-hoc roughening** (initial-condition noise and
  richer history are the only legal fixes).
  *Status: **pass with lessons** (2026-08-29). ~50 ms/world at 128²·20
  eras. v1 exposed two conceptual bugs, both fixed as process: belts
  must advect with the crust that carries them (they are crust
  properties, not world-fixed stains), and cratons must ride single
  plates. v2 shows large coherent domains, long advected belt systems
  (coastal and suture), margin classes varying along one continent,
  ridge-parallel ocean-age banding. Carry-forward for M1: iterated
  nearest-neighbor gathers accumulate striation artifacts — compose
  per-plate transforms and resample original fields once; handle
  boundary slivers sub-cell.*
- **S2 — erosion solver.** *Question:* can flow routing + implicit
  stream-power incision (+ sediment routing) hit §15 at 512/1024/2048,
  and do valley webs read at map scale?
  *Status: **pass** (2026-08-29). Full loop (epsilon-fill by directional
  sweeps, D8 receivers + MFD weights, vectorized-Kahn topo batches over
  the MFD edge set, MFD+D8 accumulation, downstream-first implicit
  stream-power) at 10 geologic steps: 0.6 s @256², 2.6 s @512²,
  9.8 s @1024², 39 s @2048² — inside §15 untuned, numpy-only. Implicit
  solve is unconditionally stable (few large steps suffice); dendritic
  webs read at map scale by ~24 steps. Named hazard for M-phase: D8 on
  smooth surfaces axis-locks channels (§11a regularity) — counter with
  process-modulated heterogeneity (erodibility/uplift texture), MFD,
  enough incision; sediment routing not yet timed (~2× estimate).*

  *Artifact diagnosis (2026-08-29, corrected after audit): the
  "vertical dotted column" at map center is ROOT-CAUSED. The
  depression-fill's epsilon floor equals EPS × drainage-path distance
  to the border outlets; on a flat plane it peaks at EPS·G/2 = 0.511
  at G=1024 — marginally above the 0.5 fill-depth threshold the spike
  render uses to color lakes — so low flat ground near the map's
  central columns flickers into phantom shallow-lake speckle.
  Verified: flat-plane fill puts cells over threshold exactly in
  cols/rows [500,523]; observed dots concentrate in cols [500,522] in
  all four candidate panels (6–17× enrichment, 24–67% of all true
  isolated water dots); zero over-threshold cells at G=512 (artifact
  is resolution-dependent — also a §2 violation; would worsen at
  2048). Seed-independent because it is geometry, not noise. Class A
  artifact. Fix is process-level for M-phase: lakes must come from
  routed water, never from fill-depth-vs-threshold (the epsilon is
  routing plumbing and must be subtracted/scaled by km-per-cell so it
  never approaches physical magnitudes). Retractions from the same
  audit: the upsampler-degeneracy columns (341/682) show NO dot
  enrichment — that hypothesis stays only for the separate
  land-colored diamond/×-stamp texture (untested); the earlier
  "3,300–5,000 map-wide speckle" figure was inflated ~15–20× by a
  4-neighbor isolation bug counting diagonal river chains (true count
  110–300/map); the "canon control showed zero dots" comparison was
  void (color mask cannot detect canon water).*

  *Rectangular lake (seed 31) — ROOT-CAUSED (2026-08-29). Step-axis
  bisect: the basin is BORN NATURAL (irregular lobed depression in the
  initial noise, inside a zero-clipped-uplift zone, floor below base
  level, median depth ~15 — a real lake, unrelated to the
  epsilon-column artifact) and is progressively SQUARED by the erosion
  loop: lobed at step 0, still lobed at step 4, fully squared by step
  16. Mechanism: implicit stream-power under D8 is a strong
  along-flow smoother with no cross-slope process, so surrounding
  hillslopes lose their initial fine structure and grade into
  near-planar ramps whose channels and aspect lock to the grid axes
  (the known D8 axis-locking hazard); a lake surface is a horizontal
  plane, so its shoreline — the level-set of those axis-graded planar
  flanks — renders as straight axis-aligned segments meeting at
  right-angle corners. Same root pathology as the comb rivers, new
  symptom. A candidate-crease/lattice hypothesis was tested and
  refuted (U ≡ 0 across 86% of the lake; edge |d²U| ≈ background).
  Note: single-pixel-line straightness is only 9–16% — the shore is
  stepwise-straight within ±2 px, which reads as a clean rectangle at
  map scale; strict modal metrics understate visual straightness.
  M-phase fixes, all process-level: MFD/D∞ participation in the solve,
  process-modulated erodibility/uplift heterogeneity, gradient-class
  noise — plus a NEW named candidate for M3: **hillslope diffusion
  (soil creep) alongside stream power**, the standard geomorphic
  process that keeps graded slopes curved; its absence is why flanks
  planarize. Zero-clipped uplift dead zones are a spike-input quirk
  (real uplift comes from tectonic history), but note the class:
  interior basins below base level are legitimate endorheic features
  that will eventually need water-balance treatment, not suppression.*

  *Corrections applied + rerun (author-authorized, 2026-08-29): EPS
  1e-3 → 1e-5 (routing plumbing scaled far below physical thresholds)
  and hillslope diffusion (soil creep, alpha 0.2 × 3 substeps/step;
  +0.8 s per 24 steps at 1024²). Field verification: zero isolated
  single-cell lakes on all seeds (column AND map-wide marginal speckle
  shared the epsilon cause). Blind rerun s4d (same seeds, fresh judges
  C/D, verdicts in out/spikes/s4d_judges.md): the dotted column is
  gone from all rulings; the seed-31 lake downgraded from
  right-angled rectangle to "rounded-rectangle" — corners cured, two
  near-flat parallel edges persist exactly as the planarization
  mechanism predicts (full fix = M3 heterogeneity/MFD participation).
  Drainage integration (orphaned streams, lakes without inlets, water
  not marking terrain) is now the top-salience defect class — M3 core
  scope. Diffusion side-effects recorded: stamps morph into
  crater/ring marks, a featureless lowland void where fine texture was
  erased, and — usefully — the noise lattice's row/column ordering
  became VISIBLE to judges, hardening the case for gradient-class
  noise. Canon panels (ref1/ref9, deduped) drew mostly
  raster-literacy flags (screenshot pixel crenellation, 1-px shore
  strokes at this zoom) — add to EVAL literacy notes; class separation
  between canon (C/D-class) and candidates (A/B-class) remains clean.*
- **S3 — border mechanism.** *Question:* with the world larger than the
  frame and land-causes confined/tapered toward the world rim, is the
  no-land-in-outer-ring guarantee structurally airtight across hundreds
  of seeds (tripwire stats, worst-case gallery)? Depth at the frame is
  unconstrained; frame-correlation of any structure is the fail.
  *Status: **findings in, closure deferred to M2** (2026-08-29). 300
  seeds × 6 configs on the conservative proxy (continental crust =
  land): initial-placement confinement alone cannot guarantee the ring
  — craton reach must be strictly bounded (multiplicative outline noise
  gave 3× nominal radius: fixed to additive/clipped), and 20-era drift
  budget exceeds any feasible margin. Key reframe: the hard border is
  decided at the elevation+sea-level stage — crust crossing the frame
  is legal (its margin floods into shelf; §3-clarified), only emergent
  cores must stay interior. Guarantee architecture = kinematic drift
  budget × world:frame ratio × flooded margins; tripwire re-runs on
  real land in M2. Named candidates for the author: (a) derived
  cartographic window (near-zero frame-hug even on the conservative
  proxy: 0.013 mean vs 0.12–0.22 fixed-frame; it is selection, not
  formation — flagged); (b) mantle-circulation clustering pull
  (supercontinent cycles; helps composition §8 and border tails;
  process-footed but conveniently frame-friendly — flagged).*

  *Evidence-audit note (2026-08-31): the 300×6 numerical table was printed
  to the console and was not serialized. The executable source and six
  worst-case sheets remain at `spikes/s3_border_stats.py` and
  `out/spikes/s3_worst_*.png`. Under the later causal rule, post-history
  crop selection is allowed; `F_window` is therefore an admissible premise,
  not a demonstrated solution. It did not test exact final water,
  low/medium/high availability, morphology, or process-domain independence.*
- **S4 — evaluation-harness seed.** *Question:* do the instruments
  and the blind imposter/2AFC protocol work end-to-end? Calibrate
  ref-vs-ref detectability; sanity-check that obviously-non-canon
  output is ~100% detectable.
  *Status: **mechanics pass; first calibration run invalidated naive
  trial construction — by design** (2026-08-29). Two blind
  fresh-context judges on 12 grayscale 2AFC trials: 9/16 correct on
  canon-vs-S2-render (chance ≈ 8/16), calibration pairs split 2/2
  each. NOT evidence the S2 spike approaches canon quality — evidence
  the trial build was unfair: random small crops of the references
  often land on featureless ocean/flat regions; crop scales were
  unmatched (zoomed globe texture vs whole-landscape map render); and
  grayscale+autocontrast destroys the ramp, which CONTRACT §12 valued
  feature 7 says carries
  half the look. Judges attended to exactly the right cues (coupling,
  dendritic connectivity, stamped-vs-formed edges), so the
  discrimination axis is sound. Production protocol requirements
  extracted: (1) feature-targeted cropping, (2) km/px scale matching,
  (3) palette-preserving normalization (render candidates through a
  comparable ramp; don't grayscale away the look), (4) bland-tile
  filtering, (5) always run calibration arms so a broken yardstick is
  caught before it is trusted. Rebuild trials to these rules for the
  M2 baseline score.*

  *S4b re-run at 1024×1024 (author-requested, 2026-08-29): whole S2
  candidate maps in a reference-like stepped ramp vs full-size crops of
  the six refs that support 1024² (1, 2, 6, 9, 10, 14); bland-tile
  filter; palettes preserved; judges told hue/subject differences are
  not evidence. Result: **15/16 correct discrimination** (Judge A 7/8,
  Judge B 8/8; first build was 9/16 ≈ chance) — the repaired protocol
  is decisive and becomes the standing yardstick: progress = accuracy
  falling toward chance on later pipeline output. Consistent tells
  against the candidate = the work list: radial starburst peaks with
  comb drainage (blob uplift + D8); repeated identical diamond/x
  speckle (bilinear lattice-noise artifact — a §11a stamp texture from
  numerics; use gradient-class noise); vertical dotted seam columns
  (fill/upsample axis bias); ring-haloed lakes (depth-band rendering =
  miniature bullseyes); slab/lozenge mesas (fill + ramp on smooth fbm);
  plus-sign river junctions. What judges praised in refs = the target
  list: coastal belts coupled to offshore arcs/trenches, foothill
  fringes, dendritic valleys reaching the sea, shelf-to-deep gradients
  tracking coasts, asymmetric belts. Calibration arms caught a judge
  false-positive mode: small canon volcanic islands crop-read as
  "stamps" out of context — rubric note for production judging.*

  *Trial type 2 adopted (author, 2026-08-29): diagnostic critique
  panels — single images reviewed blind against the formation rubric in
  three buckets (done poorly / done well / cannot identify), every
  claim evidence-anchored, praise bearing the same burden as defects,
  neutral provenance framing with canon-in-the-defendant-slot
  calibration panels. Mechanics run (s4c, 4 candidates + 2 canon,
  2 judges): **pass.** Judges critiqued canon panels freely (no
  sycophancy); canon-vs-candidate separation lives in defect *class*
  (§11a-regularity violations vs subtler formation quibbles) — add a
  severity taxonomy to make that quantitative. Candidate critiques
  reproduced the S4b work list independently and added new reproducible
  defects: a vertical dotted column at x≈50% on multiple maps
  (systematic center-seam artifact in the S2 stack), a rectangular
  right-angled lake, parallel N-S water slivers, an N/W map-edge seam.
  Unknown bucket used honestly (canon's grey-violet plateau fill and
  tan/green boundary flagged rather than confabulated — the same
  ambiguity class as the author's vegetation question; §7e plateau lake
  speckle reads as noise at crop scale — rubric literacy notes). Build
  flaw found: both canon panels drew overlapping crops of one ref —
  dedupe picks; keep occasional deliberate duplicates as a reliability
  probe (judges gave near-identical critiques to near-identical panels
  — good test-retest signal). Verdicts persisted: out/spikes/
  s4c_judges.md. 2AFC remains the scalar metric; critique panels feed
  the work list.*

## Milestones (after spikes ratify the architecture)

- **M1 — structure.** Tectonic stage productionized: world domain,
  plates/terranes, crust fields, margins; land/sea mask at composition
  level (2–4 large domains, water dominates, land-fraction knob;
  border emerges causally). Judged on structure views before any
  relief exists.
  **Status:** complete; advanced to M2 (2026-08-29).
  Delivered: `engine/` package — km-space gradient noise (calibration
  checks incl. positional degeneracy detector: pass), composed-
  transform plate history (static-world invariance proves resampling
  artifacts structurally impossible; continental crust persists at
  sutures — shortens/thickens, never vanishes; belts/ages ride their
  plate), margin classification, five structure views, controls
  registry (7 controls, process-term promises), tripwire report,
  provenance-embedded PNGs, webui adapter live (~0.7 s generation),
  m1_checks (all pass: determinism, perf, static invariance, control
  isolation eras/wander vs partition, activity sanity, seed variety),
  12-seed gallery at out/m1/m1_gallery.png. Observations queued for
  author review: (a) some seeds read as several similar-size cratons
  rather than aggregated continents — candidate responses at the time
  were fewer/larger nuclei or mantle-circulation clustering; the later
  `center_pull` formulation was rejected in B00/B02 because it restored
  fixed-center bias; (b) seeded continental budget maps to ~0.55–0.65× visible
  in-frame fraction (drift out of frame + suture stacking) — knob
  covers the range, default retune is the author's call; (c) belts can
  wrap a craton's whole perimeter when hit from many sides over
  history — watch at M2 that this does not become a §7f rim/bullseye
  read; (d) world-rim fresh-crust band verified to stay outside the
  frame at world_margin 0.45 (buffer must exceed kinematic budget —
  now stated in the control's promise).
- **M2 — crude end-to-end slice.** Subsidence + isostasy baseline,
  first orogenic relief, sea level, stepped-ramp render, report,
  provenance, webui adapter live. The full map exists, crude; record
  the baseline imposter score as the yardstick.
  **Status:** complete; advanced to M3 (2026-08-29). The original suite
  passed 24/24; the current inherited suite is 23/24 because the 256²
  performance tier misses §15.
  Delivered: `engine/elevation.py` (all terms named processes — GDH1
  plate-cooling subsidence from crust age; isostatic cratonic
  freeboard; stretched-margin subsidence profile whose width comes from
  the M1 margin class, passive broad / active narrow, segmented along
  strike by bounded km-space noise — §6c from process + the sanctioned
  noise exception; sediment-apron continental rise toward passive
  margins, standing in for M3 routed sediment; subduction trenches
  offshore active margins only — §6d earned plunges; orogeny saturating
  with accumulated shortening and decaying with belt age, oceanic belts
  standing as arc ridges/island chains; exact eustatic sea-level solve
  from a hydrosphere-inventory control, calibrated default 4930 m so
  the stand sits near datum 0), `engine/surface.py` (Catmull-Rom
  prolongation of the coarse surface + process-modulated km-space fBm
  detail — rough land rising with orogeny, muted shelf, quiet deep;
  sub-pixel octaves trimmed with full-stack normalization so shared
  octaves are identical at every resolution; water rule: wet iff
  detail-h AND crustal surface below sea level — islands can breach
  shelves, sub-grid pits cannot flood, no speckle by construction),
  `engine/render_map.py` (stepped hypsometric ramp anchored to the
  canon register, isobath + slope instrument views), map report with
  §3a/§6a/§6b/§7i/§8 tripwires (incl. the contract-named
  nearest-land-to-border every run), webui adapter live (11 controls,
  8 views, ~1.1 s at 512²), m2_checks 24/24 (determinism; sea level
  exact across sizes; structure corr 0.999 across 128↔512; §4
  isolation incl. hydrosphere = pure constant crust offset and
  orogeny = off-belt crust bit-identical with recorded eustatic
  shift; GDH1 wired exactly; perf 0.48 s/0.79 s/1.95 s at
  256²/512²/1024²; re-render 43 ms). Galleries at out/m2/ (12-seed,
  instruments, same-seed pairs incl. the queued lobes ablation).
  Mid-build corrections, all process- or numerics-level: hydrosphere
  default was 15% under the world's hypsometry (sea level solved
  −728 m and stranded the margin-flooding architecture — recalibrated);
  bilinear prolongation's C0 level-sets stamped cell-scale right-angle
  staircases on every isobath (Class A, caught by the isobath
  instrument; fixed by C1 Catmull-Rom sampling of the same surface —
  the gradient-noise class of fix); binary margin width drew
  uniform-width slope halos (§6c; fixed by along-strike segmentation
  noise); land palette was washed-out vs canon (Class D; ramp
  re-anchored against ref9).
  **§3a closure evidence (the S3 question, answered on real land):
  56/60 seeds place land in the outer ring, and it is crustal
  emergence (56) not noise islands (3) — continental interiors drift
  across the frame line and cannot flood. The
  confinement × drift-budget × flooded-margins architecture is NOT
  sufficient for the hard border. The flagged candidates stand for
  the author's decision at that time: (a) derived cartographic window
  (selection, not formation — near-zero frame-hug in S3), (b)
  mantle-circulation clustering pull. Later B00/B02 work retained the
  crop-last selection premise but rejected `center_pull` because it
  merely renamed fixed-center bias.**
  Also queued for the author: continental_budget default retune —
  observed land fraction 0.05–0.17 across the gallery vs the ~1/3
  center of §8 (budget 0.30 × ~0.6 in-frame × margin flooding); M1
  craton-scatter and belt-perimeter-wrap observations now visible on
  real maps (seed 7's coastal mountain ring around a low interior —
  its highstand pair floods it into a mountain-ringed epeiric sea).
  Known-crude by design, M3+ scope: no rivers/valleys/erosion texture,
  shelf ribbons narrow until sediment fill (shelf_band_fraction warns
  at ~0.01), enclosed coarse seas render wet without water-balance,
  interiors flat outside belts (M4 anatomy/provinces), frame_km knob
  deferred (window-semantics decision).
  **Baseline eval run (full record: out/m2/m2_judges.md + EVAL.md
  status): 2AFC = JB 8/8 vs JA 1/8 (s4b spike baseline was 15/16;
  output is no longer unanimously distinguishable, but the tell list
  is real); severity-tagged critique panels held class separation
  (canon drew A/B claims only in documented false-positive modes).
  Post-eval verified artifacts + dispositions: (1) Catmull-Rom
  overshoot rang phantom concentric bump rings at steep coarse steps
  — judge-flagged, probe-verified (143 phantom bumps seed 88), FIXED
  same-day by clamping the cubic sample to its cell corner range
  (post-fix 55 < the 325-bump no-overshoot bilinear baseline;
  staircase stays cured; galleries regenerated — eval images on disk
  remain the judged pre-fix renders); (2) dotted/dashed ridge
  lineations (three judges) — thin young-crust axes sliced by band
  quantization; M3 candidate: coherent ridge-axis crust record at
  divergence; (3) lattice-aligned coast scarps / rectangular bar
  islets / right-angle rift channels — verified (seed 40); the S1
  sub-cell-boundary carry-forward is now judge-confirmed on delivered
  maps; structure-side fix awaits author authorization; (4)
  "distance-field buffer" shelf read (all four judges) — the margin
  taper's nested even isobands; real cure is M3 sediment/erosion
  texturing the platform; (5) freestanding belt ring enclosing abyss
  (seed 19, both panel judges) — belts persist on consumed carriers;
  tectonic bookkeeping fix, M3 structure pass; (6) summit bands
  pinned to waterlines, one-sided and unbroken — belt cross-section
  anatomy, M4 (M1 obs (c) now fully realized); (7) featureless
  interiors / pancake islands / single-scale texture — the declared
  M3/M4 crudeness, now with judge wording. Judges' keep-list:
  arc–trench pairing with correct polarity, shared land/seafloor
  structural grain, setting-responsive shelf width, collision-zone
  relief concentration, branching lineation networks — the coupling
  the architecture exists to buy.**
- **M3 — surface-process coupling.** Flow routing, incision, sediment
  routing/deposition; rivers, valleys, floodplains, deltas; shelves
  and fans fed by land erosion; lakes with drainage context;
  lowstand-then-flood.
  **Status:** built and evaluated (2026-08-30); author gallery sign-off
  pending. `m3_checks` is 53/53; the 256² §15 performance miss remains.
  Delivered: `engine/erosion.py` — the S2 mechanics productionized on
  a FIXED 20-km world-domain process grid (output resolution samples
  below it — §2 by the same argument as the tectonic lattice; drainage
  cannot see the frame — §3b): leak-free epsilon depression-fill, D8
  receivers + slope-weighted MFD discharge, vectorized-Kahn batches,
  implicit stream-power incision downstream-first against a LOWSTAND
  base with flood-back (drowned valleys, shelf channels), hillslope
  soil creep (control, 0 = off), process-modulated erodibility
  (belt rock harder × km-noise heterogeneity), single-pass sediment
  routing (land deposition where capacity drops -> floodplains;
  marine e-folding settlement -> delta/shelf wedges, fans, drape;
  one budget, §6h), continentality runoff (moisture decays inland —
  crude climate-lite standing in for M4), and per-basin water-balance
  lakes (floor lakes in dry interiors, brim-full chain lakes on
  through-rivers, outlet rivers intact). Rivers ship as VECTOR edges
  drawn from computed discharge (rasters dotted the diagonals);
  river_density is render-class. Views + report grew (drainage,
  sediment; lake census, mass balance, river coverage). webui adapter
  now uses all three caching tiers (head = tectonics+crust, late =
  erosion tail ~0.8 s, render = river prominence in ms). Structure
  pass shipped the three authorized fixes, all probe-verified:
  sub-cell supersampled material snapshot (lattice-aligned scarps and
  bar islets gone — seed 40), continuous seafloor-age reconstruction
  (dotted ridge threads gone), fast remnant-arc subsidence (seed 19's
  freestanding belt ring now a faint shoal trace). Checks: m3_checks
  29/29, m2_checks 23/24, m1+noise green.
  **Solver bug found by the probe-calibration rule: the S2 spike's
  in-row min-plus depression fill LEAKED THROUGH RIDGES** (it never
  clamped at the heights along the propagation path) — caught by the
  M3 known-positive lake calibration test, latent through all of S2
  and the M2 build; replaced by leak-free directional Gauss-Seidel
  reconstruction verified against a reference implementation. The fix
  un-hid every walled basin, which forced the lake physics honest:
  stream power may only cut (the implicit form was hauling basin
  floors up to their spills), and basins hold water to a
  runoff/evaporation balance level, not to the brim.
  **§15: the 256² tier MISSES its 1 s budget — 1.1–1.4 s** (fixed
  process-grid cost dominates; 512²≈1.3–1.6 s ✓, 1024²≈1.9–2.3 s ✓,
  2048² well inside ✓, re-render ≈50 ms ✓, late-tier control tweaks
  ≈0.8 s). Remediation is the sanctioned perf path (the planned Rust
  port; numpy has no cheap headroom left in the Kahn batches).
  Builder pre-screen watch items for review: straight river segments
  on several seeds (D8 axis-locking at process-grid scale, partially
  countered but visible — candidates: stronger heterogeneity, D-inf
  routing); seed 7/19 lake arcs hugging the inside of perimeter belt
  rings (downstream echo of the M1 belt-wrap item); faint chamfer
  facets in the drainage instrument's ocean field (diagnostic-only).
  Authorization notes: sub-cell treatment + soil creep (as a
  zeroable control) + continentality runoff shipped under "go ahead
  with M3" as parts of the approved work list — each is isolated and
  cheap to revert if the author objects. At this historical M3
  checkpoint, border closure and `continental_budget` were the two open
  author decisions (decision aids at `out/m2/m2_decision_*.png`); later
  private work supersedes that framing.
  **Eval outcome.** Full record `out/m3/m3_judges.md`: 2AFC 24/24
  unanimous — a recorded regression vs M2's 9/16 split, caused almost
  entirely by ONE tell: the river layer's straight cell-segment
  geometry ("drafting-board rivers", every judge). Post-eval
  corrections, verified and applied same-day (judged images
  archived): rivers now draw as C1 curves through the network
  (prolongation-class interpolation of the coarse path), river
  overlays composite through the output water mask (floating-dash
  fix), lake shorelines are cut by output terrain against the lake
  surface (square-lake fix). Duplicate-panel reliability probe
  passed. Remaining judge-confirmed opens, by root: D8 axis
  preference in the SOLVE (straight valley reaches, parallel
  channels — M4 candidates: D-infinity participation, stronger
  heterogeneity); distance-halo shelf isobands (M4 platform
  texture); belt-wrap family (perimeter rims, the subsided seed-19
  ring still faintly ring-read, range-front moat-lake chains — the
  oldest open structural item, author candidates due at M4);
  stamped arc-bump chains on ridges; endorheic river terminations
  lacking a dry-floor marker; flat interiors (the M4 anatomy scope
  itself).
  **Run 1 foundation-correctness pass shipped as 0.3.1-m3
  (2026-08-30; blind evaluation complete, still awaiting author gallery
  review).** Soil creep is now a
  time-scaled, preservation-calibrated effective 8.8 km²/Myr
  diffusivity, conservatively restricted to
  lowstand-exposed neighbors; sediment is sourced only from actual
  stream-power incision, so redistributed creep and hypothetical uplift
  on excluded cells are not counted again. Sediment reroutes after the
  final solve/creep mutation, and delivered discharge, river edges, and
  lake inflow reroute again after deposition. Each basin uses one
  horizontal level, preserved through masked output sampling rather
  than diluted by zero-valued dry cells. World-boundary sediment export
  is explicit in m³; the report independently closes source = deposit +
  export + terminal residual and warns if an interior residual appears.
  `erosion_time=0` now literally disables the whole process stage.
  Report/PNG provenance now echoes the complete effective control set,
  including render-only changes. Evidence: `m3_checks` 53/53; M1 and
  noise suites green; M2 remains its documented 23/24 with only the
  256² performance miss (1.38 s; 512² 1.45 s, 1024² 1.97 s); cached
  late tail ~0.99 s. The refreshed 12-seed gallery, instruments, and
  same-seed pairs are preserved separately at `out/m3_run1/`. Blind
  `m3-eval-v2` results: the separate archived-M3 bridge cohort
  reproduced 24/24; Run 1's fresh cohort scored 17/24 with only 1/8
  candidate arms unanimous. That difference is a confounded cohort-
  level signal, not causal evidence; two Run 1 judges still detected
  8/8, so the documented detecting-judge progress gate did not pass.
  Verification confirms residual D8-axis rivers,
  geometric/aligned submarine relief, the seed-19 perimeter belt,
  distance-dominated shelf halos, and genuinely low-relief interiors
  amplified by the stepped palette. It also finds six unsupported and
  two overcalled Class-A claims on ten canon-control claims, so raw
  critique severity counts remain unusable without adjudication. Full
  record: `out/m3_run1/run1_evaluation.md`; no author sign-off is
  implied.

## Private border/composition experiment record (2026-08-31)

The narratives below preserve the historical reasoning in full. The
canonical attempt IDs, dispositions, qualifications, and evidence paths
are maintained in [`ATTEMPT_REGISTER.md`](ATTEMPT_REGISTER.md); if a
summary here and that register differ, the register is current.
### B02 — Post-M3 Run 1 border/composition formation spike rejected (2026-08-31)

  No engine behavior was retained. The authorized pilot
  removed nucleus placement from the fixed central frame and tested a
  seed-varying, world-space assembly prior, measured world-relative
  continental inventory, plate-interior hosts, and movable 4096-km
  crops. The old S3 `center_pull` candidate was explicitly disqualified:
  it accelerates plates toward the world midpoint, which is also the
  delivered-frame midpoint, so it only renames the center bias.
  The most promising affordable pilot used a 3× world, 8 nuclei in 6
  assembly provinces, world continental fraction 0.22, and fixed
  hydrosphere 4200 m. Before world-rim adjudication it appeared to give
  water-buffered candidates in 12/12 standard seeds, 10/12 above 20%
  land, and five near 35–50% (maximum 0.497). That was a false-positive
  headline: visual review showed several richest crops touching the
  simulated-world rim, with bathymetry flattening and running parallel
  to the map edge. A 2000-km numerical-rim exclusion reduced the same
  run to 8/12 available, 4/12 above 20%, and none at or above 35%
  (maximum was below 35% before rounding, reported as 0.350); even a
  permissive 1000-km exclusion left only 5/12
  above 20% and one above 35%. A 4× world still missed one of six probe
  seeds and cost about 16 s per generation, outside the viable default
  path. Inventory sweeps were strongly non-monotonic per seed because
  more continental material legitimately changes collision survival
  and displaces a fixed water inventory; lowering the fixed
  hydrosphere to 3000 m did not cure that. No compensating water curve,
  formation retry, center-biased province, or land-target mask was
  accepted. The private engine seam was removed and production hashes
  for seeds 3/11/63 were reverified bit-identical. Exploratory visual
  evidence remains at `out/formation_spatial_audit.png`,
  `out/formation_candidate_gallery.png`, and
  `out/formation_seed77_world3.png`. These are exploratory PNGs; the printed
  sweep table was not serialized into a protocol/report. Their visible
  morphology also retains repeated rounded/pancake bodies and broad uniform
  shelf halos, so only the seed-varying spatial-placement premise remains
  plausible. Border independence and natural morphology did not pass, and
  this finite planar world cannot supply reliable land-rich, rim-independent
  crops cheaply; the next
  architectural candidate must remove the simulation rim from the
  candidate domain—by periodicizing it or by separating a much larger,
  cheap structural world from a haloed local surface-process solve—before
  a selector is promoted.
### B03–B05 — Large-atlas/local-process follow-up rejected for promotion (2026-08-31)

  Private experiment seams and spikes only. A 6×
  (24,576-km) structural atlas and movable 4,096-km crop can guarantee
  water on the delivered rim without editing, tapering, or masking the
  terrain. That guarantee alone was insufficient: the frozen 40-km
  legacy replay still put a -5,250-m isobath parallel to the frame for
  1,081 km along the top and 852 km along the left. Under that historical
  precommit, a connected visible-level contour gate was therefore made a
  hard eligibility test; the later causal rule supersedes appearance-only
  use of that gate.
  A coupled plate/craton formation variant then produced one promising
  120-km shortlist basin with 39.28% land, 88.7 m water clearance, and a
  passing 496-km maximum visible parallel run. It was frozen before the
  finer build. At the 80-km oracle the same origin failed the water-safe
  screen; although its shared-grid land mask remained similar (IoU
  0.921), its edge envelope changed from -88.7 m to +1,320.2 m. More
  decisively, all 15 other water-safe oracle crops failed the then-sealed
  contour gate, leaving zero eligible crops. The predeclared shortlist
  stability gate therefore failed and no 40-km replay or multi-seed
  promotion run was spent on this variant.

  The local-process half also remains experimental. A lowstand-outlet,
  finite-reach marine-routing prototype made nested and shifted
  synthetic halos bit-identical and closed sediment mass to floating-
  point precision, while preserving the shipped default branch. On a
  real-world smoke case, however, 34.49% of mouth load remained far-field
  export and 1.62% of the marine footprint held 39.66% of its deposit at
  the thickness cap. Rebuilding marine routes dynamically reduced export
  by only 1.64 percentage points, slightly worsened concentration, and
  more than doubled full runtime, so that ablation was removed. Nothing
  from either experiment is wired into the registry, adapter, or public
  controls; `m3_checks` remains 53/53. Conclusion: do not tune the
  current ellipse/crop selector further. Any later attempt should first
  demonstrate multiple naturally bounded continental domains whose exact
  final water ring survives structural resolution and whose delivered
  terrain is process-domain independent; visible-contour alignment is now
  diagnostic only under the current causal rule below. Marine fan
  morphology remains a separate process-design problem. Evidence for this
  chain is preserved at `../out/atlas_replay_seed11_065_v2/`,
  `../out/coupled_anisotropic_seed11_oracle80/`,
  `../out/process_halo_seed11_stage_v1/`, and
  `../out/process_halo_provenance_seed11_v1/`. Their report SHA-256 values
  are respectively
  `9fcb7741f42b5399ead3931c93164ff9dd50f87f2c57c4c448b31bf18d82c12d`,
  `9fe0063546f789404582574ee68389e72885d78183182088235c2409ade2eb72`,
  `d6dd696c14e7cc51a990d3a2f639b2dbd21f93c74426bf79ecb9bcdde5d46ae5`,
  and
  `1b4c61fb93224db6a773b0fe26db47d6dcbb0d9605161f4c629c34cf75c01101`.
### B06 — Formation-first field-accretion follow-up rejected at the first structural gate (2026-08-31)

  Private spike only. A fixed 64-km
  canonical formation lattice used independent broad assembly and
  cratonization fields, with finite-time accretion through favorable
  carrier lithosphere. It therefore formed irregular domains without
  placed ellipses, crop masks, retries, or border tapers. Three
  nonoverlapping, carrier-disjoint groups containing three or four
  formation-native identities were frozen before terrain evidence.
  The oracle was hardened before execution so that final collision-
  winner material tags measure transported identity, ownership, and
  whole-atlas capture; a complete 16-km atlas flood fill, rather than an
  isolated collar, is authoritative for exterior-ocean connectivity.
  At the frozen 120-km screen all three identities remained exact, but
  zero of three candidates passed: transported owner/capture were
  0.734/0.943, 0.927/0.353, and 0.833/0.797 against 0.85/0.80 minima,
  while dense exterior-ocean collar coverage was only 0.725, 0.844,
  and 0.862 against the required 1.0. No candidate retained the verified
  two-sided 256-km water collar, so no 80-km, 40-km, or surface-process
  run was spent and no post-result formation tuning was attempted. The
  evidence is preserved at
  `../out/field_accretion_seed11/audit_120km_rejection.json`. Public
  behavior remains unchanged (default fingerprint matched; `m3_checks`
  53/53). The next credible architecture should derive the sampling
  frame from transported carrier/domain geometry before examining
  elevation or contours; altering terrain near a fixed crop would only
  recreate the prohibited border fix.
### B07 — One-shot transported-centroid frames rejected before elevation (2026-08-31)

  Private spike only. The same three seed-11 identity
  groups and birth-frame origins were source-pinned. One 120-km tagged
  structural build translated each origin by the exact winning-material
  centroid displacement and snapped it once to 64 km; there was no crop
  search, neighboring-origin probe, clamp, retry, or elevation/contour
  input. The shifted frames remained pairwise disjoint and recovered
  minimum per-member capture of 0.985, 0.849, and 1.000. Nevertheless,
  all three failed the predeclared conservative requirement of zero
  continental tag samples in the complete two-sided 256-km collar
  (676/2389, 372/2252, and 122/2389 sampled collar points). The richest
  frame also had 0.546 tagged footprint, only 0.805 frozen-group
  ownership, and one significant foreign identity. The other two passed
  every non-collar formation gate. The oracle therefore stopped after
  one structural build and made zero elevation or surface-process calls;
  no origins were adjusted after inspection. Chained manifests and the
  rejection report are preserved under
  `../out/transported_frame_seed11_centroid_v1/`. This establishes that
  centroid translation alone does not preserve a continent-free natural
  apron. It does *not* establish that the literal water border would
  fail: submerged continental shelf is natural, and the tag-collar gate
  is deliberately stronger than the author's depth-irrelevant water
  requirement. Any test that makes elevation authoritative instead must
  be a separately precommitted protocol, not a same-run relaxation.
### B08 — Seed-77 transported-water trial stopped before elevation (2026-08-31)

  Private spike only. The replacement protocol
  made continental material in the 256-km collar diagnostic only and
  reserved the literal border decision for the conservative elevation-
  derived late envelope: complete-atlas exterior connectivity and at
  least 160 m clearance at both 64-km and 16-km sampling, followed by the
  then-frozen visible-contour gates, now superseded as appearance-only
  criteria. Source, constants, seed order, formation records,
  one-build/at-most-one-elevation sequencing, and three formation groups
  were frozen in a formation-only precommit with SHA-256
  `1b301da9084c001e42a8415ba6de9a07be1b3729dc94f9e65a878cd644de4f5e`.
  One requested-120-km structural atlas (actual spacing 119.883 km) then
  transported each origin once by winning-material centroid displacement;
  there was no crop search, clamp, water input, or same-seed retry. Two
  frames passed every pre-elevation gate. The third had acceptable tagged
  footprint (0.331), frozen-group ownership (0.869), minimum member capture
  (0.988), and all four members above 4% of the frame, but also contained
  one unsealed natural identity (`c08ae926cb62ae80`) above the same 4%
  significance threshold. It therefore failed only the no-significant-
  foreign-identity/exact-significant-set gates. Because the sealed protocol
  required all three frames to pass together, execution stopped after its
  single structural build: zero elevation calls, zero contour evaluations,
  and zero images. Diagnostic continental collar counts were 348/2252,
  0/2329, and 209/2389; their variation supports treating material tags as
  a poor water proxy but says nothing about actual border water because no
  elevation was evaluated. The complete hash-chained evidence is preserved
  under `../out/transported_water_seed77_v1/` (selection SHA-256
  `0c9e95bef066834b658d24ae721610df046232c739c64be338876391495c715f`).
  This trial neither validates nor rejects the late-envelope water test;
  any continuation must use a fresh, separately precommitted seed and must
  decide prospectively whether candidates are independent or an all-three
  cohort.

## Deferred milestones

- **M4 — anatomy, provinces, texture.** **Status: not started; held
  pending border/composition architecture.** Range anatomy (nested
  bands, asymmetry, saddles, massifs), plateaus/terrane provinces,
  process-modulated sub-grid texture, coast character variety.
- **M5+ — canon convergence.** **Status: not started.** Iterate against
  the evaluation harness and author verdict library; controls tiering
  and promise audits; performance passes; version/provenance hygiene for
  release.

## Current causal border rule and evidence re-audit (2026-08-31)

The historical entries above retain the gates that were actually sealed
and run. They are not retroactively rewritten. From this point forward,
border closure uses the causal rule in [`DESIGN.md`](DESIGN.md): the exact outermost
ring of the final delivered-resolution result must be water, with no
minimum depth; a selector may choose among already-generated natural
geographies, and naturally frame-aligned features are allowed. Contour
alignment is diagnostic only. A result still fails if formation or surface
processes consume the crop border/distance or use its selected origin for a
crop-relative terrain change (rather than only absolute addressing/domain
scheduling), or if the delivered result depends on the finite atlas rim or
a localized solver boundary. Guards and halos are evidence, not proof of
independence.

Evidence is reclassified narrowly under that rule:

- S3's `F_window` and the later M2 derived-window demonstration are no
  longer disqualified merely because they select a crop. They remain
  unvalidated: neither was a sealed final-resolution selector, neither
  established the accepted land-composition bands or natural morphology,
  and neither proved independence from downstream numerical domains.
- The recovered seed-11 40-km replay is a **border/land pass**: its final
  delivered outer ring is water at every rendered tier and its 1024px land
  fraction is 43.508%, while the long frame-parallel isobath is not a
  validity failure. It remains a
  **process-domain independence fail**, so it is not a promotable result;
  the localized surface solution changed with its numerical process domain.
- The coupled-atlas result no longer loses all 15 historically "water-safe"
  80-km alternatives merely because their contours align with the frame.
  The one prospectively frozen origin genuinely lost its conservative
  256-km moat/envelope classification at 80 km; the saved evidence does not
  isolate the literal outer ring. The alternatives were neither serialized
  individually nor carried through a final surface-process proof. They are
  recovered candidates, not validated maps.
- The first seed-11 formation-first field-accretion oracle likewise made
  zero elevation calls. Its retained failures are the frozen ownership,
  member-capture, and conservative-collar cohort gates. It rejects those
  fixed formation frames, but neither validates nor rejects literal final
  outer-ring water.
- The seed-11 transported-centroid trial's zero-continental-tag collar is
  not evidence against a literal water border; two frames passed its other
  formation gates, while the richest frame retained separate composition
  failures. The seed-77 trial likewise left two frames structurally
  prequalified, but its third-frame/cohort composition gate stopped the run.
  Because both trials made zero elevation calls, neither validates nor
  rejects final border water. Their former or prospective contour gates are
  non-gating under the causal rule.

### B09 — Physical-outlet marine successor rejected (2026-08-31)

A separately precommitted seed-11 replay tested an uncapped,
mass-conserving lowstand-outlet successor with dynamically rebuilt marine
routes. The shipped/default branch was not changed. The three fixed small,
large, and shifted solves closed mass to better than 7.2e-16 relative,
kept explicit numerical-boundary export below 0.91% of total source, bounded
reported marine reach to 1,413.98 km inside the 1,549.75-km minimum core
halo, applied no thickness cap, and delivered an all-water outer ring.

The branch nevertheless failed its frozen acceptance criteria. Far-field
export remained 14.04-16.29% of total eroded source against a 5% limit, and
the top 1% of positive marine-deposit cells held 47.02-48.85% of marine
deposit against a 20% limit. The saved sediment view shows conspicuous
radial/branching routes and a bright coastal belt rather than credible fan
morphology. Both nested/shifted comparisons also changed the same count of
material cells: 71 final-elevation/sediment cells (maximum 3.733 m) and one
discharge cell. Thus the marine reach bound did not establish independence
of the coupled surface-process result; the exact residual land/lowstand
causal path remains unresolved.

No post-result threshold or parameter adjustment was made and the branch is
not promotable. Evidence is preserved under
`../out/physical_outlet_seed11_v1/`; report SHA-256 is
`a6fed11730e63c686f56a5f860e0a83647ad40523fa082da703f67e7005d92d2`.
The visual rejection above was a post-seal review of the saved morphology
images; the immutable report itself still records promotion as unassessed
and has no separate `manual_review.md`.

### B10 — Physical-outlet Run 1 causal discriminator completed (2026-08-31)

Diagnostic only. A separately sealed replay reproduced the identical 71 delivered-core sediment
cells and one discharge cell in both large-domain comparisons. All captured
upstream and land-handoff differences stayed below `1e-9`; the first
threshold-material difference in their common support occurred in physical
marine transport. Fixing both the effective marine source and common-support
bed eliminated the delivered residual exactly, while fixing either one alone
retained all 71 cells. Those cells lay at least 95 neighbor steps inside the
common support, beyond the 50-step marine transport duration, so native source
outside that support could not reach them in the source-only arm. A private
frozen-initial-graph ablation eliminated every delivered-core difference above
`0.05 m`, so dynamic marine weight rebuilding is necessary for this seed's
material divergence under this solver. The evidence is consistent with
alternate lobe selection but does not trace the first divergent marine step,
identify a unique initiating perturbation, or make frozen routing a candidate
production process.

The lone discharge cell, global `[230, 712]`, coincided with a post-deposition
`surface <= -80 m` classification split only `1.42e-14 m` from the threshold
and was below the drawn-channel threshold. That is strong mechanistic
consistency, not counterfactual proof. The run changed no engine default or
public control and made no promotion decision; one seed does not establish
frequency. Evidence is preserved under
`../out/physical_outlet_causal_seed11_v1/`; report SHA-256 is
`30fdcfa8edd1b602c1b1b7727a5eeb0e1c6c876e5091827c3578ee55c315df7c`.

### B11 — Physical-outlet Run 2 smooth heading-transport candidate rejected (2026-08-31)

Private diagnostic. One precommitted eight-heading suspended-load
formulation replaced the baseline's discrete downhill/fallback switch with a
continuous direction-and-slope softmax, retained per-link settling and dynamic
aggradational rebuilding, and added no cap, border fade, retry, noise, or
public control. The frozen harness passed 25/25 synthetic checks and all 18
sealed-run integrity checks, including exact reproduction of the captured
baseline marine result and reconstructed final layers.

The routing hypothesis passed. In the reproduced baseline, the first
delivered-core downhill/weight branch change appeared at step 3 and the first
material cumulative-deposit change at step 4. In the candidate, transition,
flux, and cumulative-deposit differences stayed below `1e-9` through all 50
steps. Final small/large and shifted/large comparisons had zero cells above
`0.05 m` in elevation or sediment and zero discharge cells above the frozen
relative threshold; their maxima were only `4.10e-12 m` and `7.96e-15`
relative. Fixed-source/native-bed and native-source/fixed-bed arms likewise
had no differences above `1e-9`, and the fixed-both arm was exact. Smooth
marine weights are therefore sufficient to stop the seed-11 microscopic
input differences from selecting materially different lobes.

The candidate itself failed acceptance. Only 5/9 serialized automatic gates
passed. Large-window far-field export was 14.684% of total eroded source
against 5%; the positive-footprint top-1% share was 47.831% against 20%; and
although the `>0.05 m` footprint expanded from 12,480 to 20,007 cells, its
top-1% share worsened from 22.438% to 29.352%. A post-run fixed-scale visual
review found that long comb-like rays were reduced, but replaced by a bright,
nearly continuous shoreline rim and broad uniform halo, so the morphology
veto failed. Marine terminal residual also increased from 3,361.811 to
3,405.509 m-cells. The previously frozen zero-terminal criterion was
mistakenly omitted from the serialized nine-gate dictionary; explicit
candidate ocean/lake-mask XOR and river-topology/render checks were also not
serialized. Those are report-coverage defects, not reasons to repeat or
reinterpret the run: four automatic gates and the visual veto already reject
the candidate.

No post-result constant or threshold was changed, and no engine default or
public control changed. The executed harness SHA-256 is
`8968c417a8466aa52a3abf720fa8f6f4002037be654babbd4c359fc6ac4cfb24`.
Evidence is preserved under `../out/physical_outlet_run2_seed11_v1/`; report
SHA-256 is
`2b2299c4007fad87c0e68957afaf747a921753692279a1e67c599d0d854c70df`.
The visual rejection above was completed after the sealed execution. The
immutable report therefore still records manual morphology as `unreviewed`;
the adjudication is preserved here and in `ATTEMPT_REGISTER.md`.

### B12 — One-parent crop-last feasibility run rejected on availability (2026-08-31)

Private run. Seed 137 was precommitted with no retry. One
12,288-km parent was formed at 307², elevated once, and processed once by the
unchanged default legacy surface solver at 614². Only after that solve did a
frame-blind selector inspect 169 guarded origins on a 256-km lattice. Its
frozen targets were 20%, 35%, and just-under-50% categorical land; eligibility
used the exact final water mask's outer ring, never contour alignment, a
clearance proxy, or a crop-relative terrain change. The private stress
configuration used 17 plates, 7 nuclei, and continental budget 0.65. That
budget exceeds the public maximum 0.45, and the atlas seeder turns the seven
nuclei into three assembly provinces (a target inventory equal to 21.67% of
the 3x parent's area), so this was architecture feasibility rather than a
public-slider claim.

Availability failed at the first frozen screen. Zero of 169 256px candidates
had an all-water outer ring; the best still had 56 non-water cells among the
1,020 unique perimeter cells (5.49%), and the median had 147. Land fractions
did span the low and medium targets, but the richest candidate was only
35.783%, below the high target's 45-50% acceptance interval. Consequently the
K=8-per-target authority shortlist was empty, no 1024px frame was selected,
and no crop morphology verdict was possible. The parent overview is useful
only as a layout diagnostic: the guarded interior contains a few large
domains, but their placement makes every frozen 4,096-km window cut land.

Run integrity passed completely: exactly one structure/elevation/process
sequence, one legacy sediment call, selection strictly afterward, all four
routing stages captured, engine functions restored, and all artifacts
written. Sediment closed exactly in captured metre-cells
(690,674.648 source = 593,298.537 deposit + 97,376.111 boundary export), with
zero terminal residual and all validity checks passing. The serialized
`causal_screen_pass=false` is downstream of the missing assignment: there
were no selected frames on which to evaluate ancestry or lowstand paths. It
is not evidence that a qualifying crop was boundary-caused. Section 3b
therefore remains explicitly `unresolved_single_domain`, as it would even
after a clean screen because legacy fill is numerical-rim seeded and marine
settlement has no hard reach.

This rejects the frozen 3x/three-province parent layout for seed 137; it does
not prove that no continuous origin, finer-only screen, other seed, or other
formation architecture can work. No shortlist, tolerance, seed, budget, or
terrain rule was changed after the result, and production behavior remains
untouched. Evidence is preserved under
`../out/parent_solve_feasibility_seed137_v1/`; harness SHA-256 is
`f7911f253e8ba27dfec49ddcfc84667c6b2611f6f64fc557e7c957e08dd8a2d0` and
report SHA-256 is
`0c6de3bbb439995b1ca5cb9d9400808315a5e9f6178f6e8c38afc1661aad76ca`.

### B13 — Field-accretion one-parent feasibility run rejected on availability (2026-08-31)

Private run. Fresh seed 138 was mechanically chosen
after seed 137 and precommitted with no retry. One 24,576-km parent was formed
once at 40.026-km structural spacing, elevated once, and processed once by the
unchanged full-parent legacy M3 solver at approximately 20-km spacing. Only
after that solve did the selector inspect all 3,721 origins on its frozen
61×61, 256-km lattice inside a 2,560-km guard. Authority came from one exact
4-km final-boolean mosaic, equivalent to 1024² delivered 4,096-km frames;
three predeclared full `sample_map` probes matched that mosaic bit-for-bit in
water, ocean, lake, and topographic masks. Eligibility was only an all-water
outer ring plus the frozen land intervals: low 15–25%, medium 30–40%, and high
45–<50%. No contour, collar, clearance, or frame-shape veto was used.

The natural-border premise succeeded but the composition premise did not.
There were 583 exact-water origins (15.668% of the lattice), compared with
zero of 169 in the preceding layout, and their richest frame contained
21.263790% land. However, 324 were all water, 528 were below 5% land, only six
entered the low interval, and none entered medium or high. The failure is
upstream supply rather than assignment separation: even with the water-ring
requirement ignored, the richest of all 3,721 frames was only 26.240730%, so
no candidate reached the 30% medium floor. The realized continental structure
fraction was 13.792255%, distributed among many isolated island-scale domains
from 20 carriers and 87 represented domains. Thus private budget 0.65 at the
field-accretion design cap does not mean 65% continental area and cannot be
treated as a direct land-fraction control.

The diagnostic frame verifies that crop-unaware formation can naturally put
water on every border cell without a fade or forced edit: its three coherent
islands do not track the square frame. It is not an accepted target sample.
All low/medium/high diagnostic labels intentionally fall back to that same
21.263790% origin and are byte-identical per view. Manual review also found
repeated compact, similarly scaled rounded domains, limited range anatomy,
the already-deferred broadly parallel offshore-slope character, and visible
rectilinear structure in slope/drainage/sediment views. Formal morphology
acceptance was not reached because a complete assignment did not exist, and
the available diagnostic morphology was unfavorable.

Run integrity passed. The single structure/elevation/process/sediment/selector
counts, sealed source and prior-evidence hashes, post-solve sequencing, route
restoration, candidate lattice, image hashes, and mosaic/full-map comparisons
all verified. Sediment closed to `-8.731149137020111e-11` metre-cells against
a `3.09e-6` tolerance, with zero terminal residual. `m3_checks` remains 53/53,
the causal-border suite remains 10/10, and both private harness self-checks
pass. Section 3b remains `unresolved_single_finite_parent`: a finite parent
removes crop-local boundaries but does not prove independence from legacy
rim-seeded fill or unbounded marine settlement.

Two non-outcome-reversing audit-coverage limits remain. The four complete
mosaic masks were not persisted, so an independent audit can verify every
serialized candidate identity and arithmetic relation but must replay the
sealed source to reconstruct all 3,721 classifications; the three frozen
full-map probes alone do not serialize the entire authority mosaic. Also,
the source closure omitted imported `engine/__init__.py`. That file currently
contains documentation and the version constant only, so the omission has no
numerical effect, but a successor harness should include it.

No engine default, public control, threshold, seed, or sealed harness changed
after the result. Evidence is preserved under
`../out/field_accretion_parent_seed138_v1/`; precommit SHA-256 is
`80b264c91a6daff5f6473670ff73ca3a0c61037f32db3c4c0f0d5e046e19c34f`,
harness SHA-256 is
`b943ba07371017e795211548675e250c22c6da18947e42ca16e29dcc92270814`,
exact-mosaic helper SHA-256 is
`2b3ace3111e0550784b8e503e360e8b2becd2ac27f28a2cc46c6bbb1e1bfc717`,
and report SHA-256 is
`fa6ea4d3a8eefb631fa8356177b05e2789b5edc95a837fd667915579b066dacc`.

### B14 — Conserved-inventory formation ensemble rejected on hard support (2026-08-31)

Private run. Fresh seeds 151-158 were mechanically
reserved after exposed development seeds 139-150 and executed once with no
retry or tunable CLI. Each 24,576-km parent used the existing absolute
assembly/craton fields, one strongest-craton nucleus per active carrier, and
one global chronological growth queue. A conserved 28% world inventory meant
exactly 41,288 selected 64-km cells; a separately built 14% mask had to be a
strict prefix with identical fields, chosen nuclei, carrier ownership, and
plate sites. A seed whose hard `assembly > 0.12` carrier support could not
supply the quota stopped before structural work. No elevation, surface,
border, crop, sea-level, or target-window information entered formation.

The hard support ceiling failed availability. Only seeds 152 and 153 had at
least 41,288 eligible cells, with capacities of 29.133% and 28.733%. The other
six stopped honestly at 23.797-25.722%; median capacity across all eight was
24.796%. Seed 152 selected the exact inventory but left one of its 16 domains
at only its nucleus cell, so its frozen formation-invariant gate failed. Seed
153 was the only fully ready seed. Aggregate readiness was therefore 2/8
exact inventories, 1/8 invariant passes, 2/8 qualified proxy assignments,
6/24 passing assigned windows, and 1/8 ready seeds. This is a clear rejection
of the binary-carrier implementation, not a reason to clamp, renormalize,
retry seeds, or lower the accepted land targets.

The successful cases retain useful evidence. After 80-km structural
transport, seeds 152 and 153 both supplied separated low/medium/high windows
inside the frozen 15-25%, 30-40%, and 45-<50% proxy bands. All six assignments
passed the 2-4 significant-component and 85% coverage gates. An independent
manual review passed the frozen round-body, carrier-lace, isochrone-ring, and
grid/tie vetoes: these formations are visibly fewer, larger, and more varied
than seed 138. Seed 152 had 6 substantial initial domains and seed 153 had 7,
versus seed 138's repeated compact island-scale result. This is formation-only
evidence and says nothing yet about final coastline, elevation, bathymetry,
drainage, sediment, or water-border quality.

The fixed fine sentinel was correctly skipped because seed 151 failed
capacity; this is not a resolution-stability failure. The padded seed-151
probe likewise missed its 28% quota (70,897 of 73,400 cells), but every
absolute assembly/craton/tie/eligibility/nucleus check passed, full-parent IoU
was 0.992270, and the guarded inner IoU was exactly 1.0. Thus the serialized
nested gate fails solely on capacity while its actual inner stability result
is strong. Section 3b remains `unresolved_finite_parent_global_quota`.

All 22 sealed artifact hashes, all 7,442 morphology statuses, every inventory
and prefix mask, all domain/path invariants, and the execution counters were
independently reproduced. The run performed 9 inventory layouts, 8 prefix
layouts, 2 coarse structure builds, 0 fine builds, 4 scans, and zero elevation
or surface solves. Production behavior and public controls remain unchanged;
no full parent solve is recommended from this result. Evidence is preserved
under `../out/field_accretion_inventory_ensemble_seed151_158_v1/`. Precommit
SHA-256 is
`1efd4243543951796baa954a7d179f80b4ac163f0a218cd247d419cbfc895913`,
harness SHA-256 is
`773a58d29f2f13d1356b782309ef197cc57561209f9567d1c10a5f95433686a1`,
report SHA-256 is
`e15329c6bb9efa9624718e2bd74cbb6ed1bef4f7d6112136f7192ed4258f714c`,
and the post-run manual review SHA-256 is
`525a613df8eff623ee673460d506efa7b57fd4c60c50edae59917abd43bd58e9`.

### Standing cross-domain conclusions

A later attempt must separately solve (1) physically credible suspended
marine transport and (2) terminal/coupled-domain causality, such as
watershed-closed processing or a boundary-neutral atlas solve. Physical-outlet Run 2 shows
that continuity can cure the seed-11 amplification, but not that this
heading-persistent softmax is a viable natural process. Enlarging a rectangular
halo, restoring a thickness cap, or tuning its two constants on seed 11 is not
a demonstrated fix.

Likewise, the earlier 256-km collar, 160-m clearance, and complete-atlas
exterior-connectivity tests remain useful robustness diagnostics, but are
not substitutes for the exact final-resolution outer-ring test. No existing
pre-elevation record may be promoted on that basis alone.

### B15 — Continuous-resistance field-accretion successor implemented but not run (2026-08-31)

Prospectively stopped. A complete private harness replaced
the rejected hard assembly support with one positive exponential resistance
law while retaining exact nested 14%/28% chronology, crop-blind formation,
post-structure crop scans, resolution and padded-extent probes, and the same
manual morphology obligations. Its default manifest named fresh seeds
159-166, and an exposed-development mode named 151-158.

No protocol precommit, seed execution, report, or output directory was ever
created. The design was stopped because it solved only the demonstrated
capacity wall while retaining a finite 6x planar rim and point-nucleus,
eight-neighbor arrival geometry already implicated in rounded/isochrone and
D4 morphology. Spending fresh seeds could not answer the more important rim
and construction-geometry questions. The following periodic ensemble
subsumed the continuous-support premise on exposed seeds while removing the
finite rim. This is an abandoned implementation, not a failed empirical run;
seeds 159-166 remain untouched. Source is
`spikes/field_accretion_resistance_ensemble.py`, SHA-256
`24f44ce3bcd5c80581dec0e808d3cbf607830bcff96281a45cb3c7ceaf6b97f2`.

### B16 — Boundaryless periodic field-accretion ensemble rejected (2026-08-31)

Private exposed feasibility run. Fixed development seeds 151-158 ran
once on a complete 24,576-km flat torus. Formation used intrinsic integer
Fourier-torus modes, exact 14%/28% snapshots from one global chronology, and
periodic Cartesian first-order multi-source accretion. Plate partitioning,
translation-only transport, collision/divergence neighborhoods, material
reads, age smoothing, coast detection, and authority sampling all wrapped.
Only after authority froze did the selector inspect all 9,216 periodic crop
origins. Nothing copied, blended, faded, mirrored, tiled a smaller generated
patch, or used crop-border distance. This run built no elevation, bathymetry,
hydrology, or final water border.

The boundaryless mechanics succeeded. All 8 formation re-cuts were exact; the
independently rebuilt full structural re-cut was exact; every inventory,
prefix, causal-resistance, connectivity, and domain-representation invariant
passed. The 40/80-km sentinel passed at 0.990820 binary IoU and 0.00003948
global proxy-fraction delta. Seam-neutral 3x3 views, half-world-rolled audit
tiles, and numeric gates found no discontinuous join, severe oriented
rectangle, winding blind spot, exact 4,096-km repetition, or joint canonical-
phase/frame-scale stamping. Thus a periodic parent remains credible as a way
to remove finite-rim causality; this result does not demonstrate final
all-water borders.

The morphology-producing architecture failed. Its 123 land-boundary rulers
of at least 1,024 km included 55 within 5 degrees of a square-grid axis or
diagonal. The sealed seed-blocked rotation test rejected the null at
`p=0.0043299567`; the excess occurred in initial domains/unions and transported
unions/identities. Manual review independently rejected repeated compact,
similarly rounded bodies and limited anatomical change under translation-only
transport. Only seeds 151, 152, 153, and 155 supplied separated qualified
low/medium/high crops. Seeds 156/158 had no qualified high pool, while
154/157 had all pools but no separated triple. Aggregate readiness was 4/8,
12/24 required assigned windows, and automatic readiness was false.

Fresh validation seeds 159-166 remain untouched. No tuning, retry, default,
public control, engine path, elevation path, or promotion followed the result.
This rejects the radially nucleated Cartesian-FMM plus translation-only
implementation, not intrinsic periodicity and not naturally frame-parallel
features as a class. Evidence is preserved under
`../out/field_accretion_periodic_development_seed151_158_v1/`. Precommit
SHA-256 is
`4ddb123446278ea444569aec05cbd0155fb081b27c5f095143a7c28c49efc204`,
ensemble source SHA-256 is
`4b702a87b9f828fb66e69b2981ce27a8eb99fe99266a46617eddef4b4a8b93fa`,
transport source SHA-256 is
`50da3f6a582524b9dd579cc8105bde3bec2c0668766df6dba393fbd44b8b5e9a`,
report SHA-256 is
`4d8469b53a1c571e7e74dacd740d9eb9cddcb9732b99f2c1bf3354104379f57f`,
and manual-review SHA-256 is
`acee173e63a18394048444c62918ee1186ad9a7a0c4a6a60f84ea33bdc65efb9`.

### B17 — Periodic convergence-driven formation ensemble rejected (2026-08-31)

Private exposed feasibility run. Development seeds 151-158 ran once
on a complete 24,576-km flat torus. Sixteen moving soft plate provinces were
defined by a globally smooth chord-distance kernel and analytically
area-preserving spectral shears. A true backward midpoint/RK2 trace carried
material columns through 192 Myr of history. Continental material matured
only from positive analytic convergence, modulated by material-following
survival. A fixed physical threshold of 0.45 supplied the final support;
there was no fallback, fill, dilation, border edit, crop-relative formation
input, or tuned per-seed threshold. Exact nested 14%/28% inventories came
from first-crossing chronology, and crops were scanned only after formation
froze. The run built no elevation, bathymetry, hydrology, or structural
transport.

The availability mechanics succeeded. Every seed exceeded the fixed support
floor, with support spanning 36.085-42.146%. All 8 exact full-history re-cuts
matched, all 8 seeds supplied separated morphology-qualified low/medium/high
crop assignments in the frozen 15-25%, 30-40%, and 45-<50% bands, and all
per-seed gates passed before cohort adjudication. The automatic round-body
flag was 0/8. Numeric and visual audits found no severe rectangle, exact
nonzero 4,096-km repetition, seam cut, square outer footprint, edge fade, or
feature deliberately made to contour a crop frame. Selected material also
received 3.24-4.07 times the raw convergence dose of unselected material in
every seed, while survival differences were much smaller. The morphology is
therefore genuinely caused by the modeled convergence rather than by a
decorative survival field or a hidden border correction.

That causal architecture nevertheless failed naturalness. The frozen D4
test found 90 of 234 long all-channel rulers near a square-grid axis or
diagonal (`p=0.0033799662`); restricting to post-formation union geometry did
not change the cohort result, while the 14% prefix alone passed. Eight
records wound around the torus, so their Euclidean component-shape coverage
was necessarily incomplete. More decisively, independent manual reviews of
all original panels, geometry overlays, causal views, and assigned crops
rejected every seed. Positive convergence matured narrow plate-boundary
ribbons into repeated lace, webs, antlers, hooks, crescents, starfish, and
cap-on-stalk forms rather than broad two-dimensional continental interiors.
The numerical crop success therefore does not establish natural map
availability. The D4 endpoint may be somewhat sensitive to local facets and
shards, but the visible convergence-lace failure is independent of that
statistic.

No structural transport, fresh validation seed, retry, threshold tuning,
default change, public control, or production-path change followed. Fresh
seeds 159-166 remain untouched. This rejects convergence-as-final-binary-
shape, not the smooth torus, exact chronology, fixed physical support, or
crop-last premises. If this line is revisited, convergence should remain a
source of juvenile crust while a natural material process carries, overlaps,
accretes, stacks, and welds terranes into broad rafts separated by seaways;
simple belt widening, dilation, smoothing, or global land inflation would
only disguise the demonstrated cause.

Evidence is preserved under
`../out/periodic_convergence_formation_development_seed151_158_v1/`.
Precommit SHA-256 is
`6248d082ad3484ee0fd719c4f8e987034d9506ff0d99d38a416cb0db6cb1d236`,
harness SHA-256 is
`8a21caaf1717e8820773029b52ecbf74f27602ce9b2e1b9647835b56d8760ba9`,
report SHA-256 is
`189ab3a80347ef79120f2bafce42d8edef97557040d9a60f3a85064af9228de4`,
and manual-review SHA-256 is
`984ab7607ceabd95b84880393be182c040c874ae478548be2785e022e733ff1d`.

## Private-attempt audit synthesis (2026-08-31)

The complete cross-attempt index is
[`ATTEMPT_REGISTER.md`](ATTEMPT_REGISTER.md). It includes the
early S3/M2 selectors, finite planar formation pilot, atlas and local-process
oracles, fixed and transported field-accretion frames, both physical-outlet
runs and their diagnostics, both one-parent solves, the inventory ensemble,
the implemented-but-unexecuted continuous-resistance successor, and both
periodic ensembles. It also records which historical contour/collar failures
were superseded by the clarified causal rule, which sealed reports retained
stale pre-review status, and which exploratory results lack serialized
numeric reports.

No attempt has simultaneously passed exact final-ring water,
process-domain independence, reliable low/medium/high land availability, and
natural morphology. Production/default/public behavior remains unchanged.
