# Design — pipeline_b

Architecture and working rules for the clean-room generator. The
promises live in [`CONTRACT.md`](CONTRACT.md); this file records how we
intend to keep them. Current milestone state is in
[`MILESTONES.md`](MILESTONES.md), and private experiments are indexed in
[`ATTEMPT_REGISTER.md`](ATTEMPT_REGISTER.md). Ratified by the author
2026-08-29; refined as we learn.

## Paradigm (ratified)

**Hybrid: simulated history at the structural level, steady-state
solvers below it.**

- **Tectonic history** runs time-stepped at a **fixed coarse scale**
  (coarse lattice/mesh in km-space). Output resolution never touches
  this stage, so structural resolution independence holds by
  construction, and cost is capped. History is what buys coupling: in
  subduction settings, a collision zone can share one cause with its
  trench offshore, belt onshore, and intervening narrow shelf. Products
  are crust type and age, belt geometry/age/polarity, margin classes,
  boundary systems, and hotspot tracks.
- **Steady-state derivation on resolution-independent physical grids**,
  sampled at the requested output resolution: baseline elevation from
  isostasy + thermal subsidence (seafloor depth tracks crust age —
  ridges, abyssal plains, and gradual margins from one law), orogenic
  relief with anatomy from belt structure, terranes as erodibility
  provinces (the causal route to plateaus, block patchworks, lone
  massifs).
- **Coupled surface processes**: flow routing, stream-power incision,
  and sediment routing/deposition. River sediment builds shelves, fans,
  and the quiet abyss. Land erosion and bathymetry share an auditable
  budget, so the surrounding bathymetry must respond to island and
  continental sediment sources rather than ignoring them.
- **Sea level applied last** using the current lowstand-then-flood
  baseline: erode against a lower stand, then flood to final level →
  drowned valleys, estuaries, island fields,
  shelf-as-drowned-landscape.
- **Render is separate and cheap**: stepped hypsometric ramp, dense
  stops near sea level; render-class controls never regenerate.

*Full time-stepping (surface evolution simulated through time as well)
was deliberately NOT rejected — the author may explore it later. Keep
stage boundaries clean enough that the structural stage could be
swapped for a deeper simulation without rewriting downstream.*

## The frame is a window

The authoritative geography is larger than the delivered map and may be
finite, periodic, or otherwise boundary-neutral. The delivered frame is
selected from and extracted out of that geography; it does not
participate in forming it. The delivered raster itself remains bounded
and non-wrapping.

### Causal border rule (clarified by the author 2026-08-31)

- The hard delivered-output condition is exact and narrow: after all
  terrain and surface-process stages, every cell in the outermost ring at
  the final delivered resolution must be water. Depth is irrelevant. No
  minimum clearance, deep-water target, or wider water collar is implied.
- Formation, tectonics, elevation, sea level, erosion, deposition, and
  relief generation must not consume the selected crop mask, crop border,
  or distance/direction to that border. Extraction may use the origin to
  address already-defined absolute world-coordinate fields, and a local
  solve may use it only to schedule a causally sufficient numerical domain;
  neither may feed a crop-relative value back into geography.
- Selecting a crop after natural geography exists is allowed, including
  selection informed by final water. Selection conditions which existing
  geography is delivered; it must not regenerate, taper, mask, or otherwise
  alter that geography after the choice.
- A naturally generated coastline, isobath, range, river, or other feature
  remains valid if it happens to parallel or turn near the frame. Contour-
  alignment and frame-correlation instruments remain useful diagnostics,
  but they are non-gating unless a causal trace shows dependence on the
  selected crop boundary or on a numerical boundary.
- Dependence on the finite atlas rim or on a localized solver boundary is
  still invalid. A crop guard or halo reduces that risk but does not prove
  independence; periodic or boundary-neutral construction, a sufficient
  causal reach bound, or nested/shifted-domain invariance must close it.
  Until then the affected result is unresolved, not rescued by visual
  plausibility.

## Working rules

- **The author drives design.** Findings get named candidates; only
  what is explicitly authorized gets implemented.
- **No pure-math fixes** (author, 2026-08-29): when output looks wrong,
  the correction goes into the process that produced it — never a
  post-hoc falloff, mask, blend, smoothing pass, or clamp justified by
  appearance. Noise is the one standing exception, as process-modulated
  parameterization or stochastic initial conditions. A mechanism with
  no natural footing may only be proposed in an explicitly flagged
  request ("this is an unnatural process"), with support, and stays out
  until approved.
- **Builder does not grade its own output.** The builder may orchestrate
  evaluation and verify cited evidence, but may not serve as the
  perceptual judge or issue the verdict. The ratified harness uses
  instrument views (isobaths, slope, transect overlays, spectra),
  deterministic hard checks and calibrated necessity gates, blind
  canon-discrimination (imposter/2AFC) by fresh-context judges — the
  scalar progress metric — and diagnostic critique panels (author
  format, 2026-08-29): three buckets per image — done poorly / done
  well / cannot identify — every claim evidence-anchored (praise bears
  the same burden as defects), rubric-anchored rather than
  ref-difference-anchored, neutral provenance framing with
  canon-in-the-defendant-slot calibration panels. The cannot-identify
  bucket is load-bearing: honest unknowns beat confabulated stories,
  and unrecognizable features are themselves a §11 signal. Judge
  verdicts persist to disk beside their trial sets. Author verdicts
  accumulate as a calibration library. Review is by batch gallery,
  never a hand-picked single.
- **Every material stage is auditable.** Every persisted/material field
  and stage output needed to audit a contract promise has a diagnostic
  view; transient scratch arrays are exempt. Cause fields stay
  renderable so features can be cross-examined against their causes.
- **Determinism discipline**: per-stage RNG keying; no wall clock; no
  unordered iteration feeding results. Controls are staged so a
  downstream slider cannot reshuffle upstream structure (§4).
- **Sample in world-space km**, never in cell space, wherever a field
  must survive resolution change (§2).
- **Portability**: once the pipeline is approved the author intends a
  port to a more performant language (probably Rust). Keep the core
  algorithmically portable — clean stage boundaries, plain array math,
  no Python-only cleverness in the hot path.
- **Dependencies**: numpy + Pillow (+ Flask via the shared webui).
  Anything heavier is argued case-by-case.
- **Value ledger**: every feature gets predicted yield at build time,
  ablation evidence at review, and author-decided trims in
  [`VALUE_LEDGER.md`](VALUE_LEDGER.md). Private experimental chronology
  belongs in [`ATTEMPT_REGISTER.md`](ATTEMPT_REGISTER.md).

## Aesthetic register

Positive canon: `../examples/ref1.png` through `../examples/ref14.png`
(Gleba screenshots) — emulate
formation qualities feature-by-feature; composition is contract-governed.
Ignore globe distortion, day/night lighting, land-on-frame.

Known rejection signatures (from author-reviewed failures; see CONTRACT
§11a plus author sessions of 2026-08-29). They trigger a hold and causal
audit; regularity alone is not proof:

1. Floating islands / bathymetry that ignores land; shoreline plunge
   where no fault lies offshore.
2. Spline tectonics — smooth, even-width, embossed boundary grooves;
   constant-width water canals where a boundary crosses land.
3. Independently composited layers — trenches undeflected by
   coastlines, ranges without foothills, lakes without drainage
   context.
4. Airbrushed worm ranges — uniform width, no segmentation or
   anatomy, abrupt terminations on open plain.
5. Blank single-tone land; all elevation interest inside range stamps.
6. One-frequency blobby coastlines; hard mask-edge land/sea steps.
7. Stamped uniform lakes; mega-lakes against ranges; lake speckle
   across interiors (excessive inland water generally).
8. Empty seas — no arcs, seamounts, banks; ranges never continue
   offshore.
9. Emboss/hillshade aesthetic instead of the stepped hypsometric ramp.
10. Similar-sized blob landmasses with short disconnected range
    strokes.
11. Frame-caused border contouring — land or seafloor altered by an edge
    fade, mask, forced-water operation, crop-relative modifier, or leaking
    numerical boundary. Similar geometry produced by frame-blind natural
    processes is diagnostic evidence, not a rejection by itself.
12. Systematically repeated or causally unsupported construction
    geometry — common-radius rings, ruler-straight/even-width bands, or
    inferred common spacing. Individual naturally caused arcs or
    straight reaches are audit triggers, not automatic failures.

## Perf targets and named risks

Contract §15: 256² < 1 s, 512² < 3 s, 1024² < 15 s, 2048² < 90 s;
re-render in milliseconds.

Current risks, tracked in [`MILESTONES.md`](MILESTONES.md),
[`VALUE_LEDGER.md`](VALUE_LEDGER.md), and
[`ATTEMPT_REGISTER.md`](ATTEMPT_REGISTER.md): the 256² fixed-process-grid
performance miss; resolution independence of coupled surface processes;
D8 river directionality; distance-dominated shelf halos and aligned bathymetry;
perimeter belt-wrap; and a natural parent geography that jointly
supplies the exact water border, accepted land-composition bands, and
broad two-dimensional continental morphology. Spikes S1–S4 were the
initial de-risking work; the architecture has since advanced through M3
and the private border/composition experiments recorded in those files.
