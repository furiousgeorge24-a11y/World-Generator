# Milestones — pipeline_b

Sequencing and exit criteria. Every phase ends with a batch gallery,
a value-ledger update, and a commit recommendation; the author judges
galleries and decides advancement.

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
  grayscale+autocontrast destroys the ramp, which §12.7 says carries
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
  *Status: **built, checks green, in author review** (2026-08-29).
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
  rather than aggregated continents — candidate responses: fewer/
  larger nuclei default, or the deferred mantle-circulation clustering
  candidate; (b) seeded continental budget maps to ~0.55–0.65× visible
  in-frame fraction (drift out of frame + suture stacking) — knob
  covers the range, default retune is the author's call; (c) belts can
  wrap a craton's whole perimeter when hit from many sides over
  history — watch at M2 that this does not become a §7f rim/bullseye
  read; (d) world-rim fresh-crust band verified to stay outside the
  frame at world_margin 0.45 (buffer must exceed kinematic budget —
  now stated in the control's promise).*
- **M2 — crude end-to-end slice.** Subsidence + isostasy baseline,
  first orogenic relief, sea level, stepped-ramp render, report,
  provenance, webui adapter live. The full map exists, crude; record
  the baseline imposter score as the yardstick.
  *Status: **built, checks green, in author review** (2026-08-29).
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
  the author's decision: (a) derived cartographic window (selection,
  not formation — near-zero frame-hug in S3), (b) mantle-circulation
  clustering pull (process-footed, helps §8 composition too).**
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
  *Status: **built, checks green, in author review** (2026-08-30).
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
  one budget, §6e), continentality runoff (moisture decays inland —
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
  cheap to revert if the author objects. Border closure and
  continental_budget remain THE two open author decisions (unchanged
  by M3; decision aids at out/m2/m2_decision_*.png).
  **Eval outcome (full record out/m3/m3_judges.md): 2AFC 24/24
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
  itself).*
- **M4 — anatomy, provinces, texture.** Range anatomy (nested bands,
  asymmetry, saddles, massifs), plateaus/terrane provinces,
  process-modulated sub-grid texture, coast character variety.
- **M5+ — canon convergence.** Iterate against the evaluation harness
  and author verdict library; controls tiering and promise audits;
  perf passes; version/provenance hygiene for release.

The evaluation harness grows every milestone; judges are
regression-tested against accumulated author verdicts.
