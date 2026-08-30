# Design — pipeline_b

Architecture and working rules for the clean-room generator. The
promises live in `CONTRACT.md`; this file records how we intend to keep
them. Ratified by the author 2026-08-29; refined as we learn.

## Paradigm (ratified)

**Hybrid: simulated history at the structural level, steady-state
solvers below it.**

- **Tectonic history** runs time-stepped at a **fixed coarse scale**
  (coarse lattice/mesh in km-space). Output resolution never touches
  this stage, so structural resolution independence holds by
  construction, and cost is capped. History is what buys coupling: a
  collision zone automatically has its trench offshore, its belt
  onshore, its narrow shelf between. Products: crust type, crust age,
  belt geometry/age/polarity, margin classes, boundary systems,
  hotspot tracks.
- **Steady-state derivation at output resolution**: baseline elevation
  from isostasy + thermal subsidence (seafloor depth tracks crust age —
  ridges, abyssal plains, and gradual margins from one law), orogenic
  relief with anatomy from belt structure, terranes as erodibility
  provinces (the causal route to plateaus, block patchworks, lone
  massifs).
- **Coupled surface processes**: flow routing, stream-power incision,
  and sediment routing/deposition. River sediment builds shelves, fans,
  and the quiet abyss — land erosion and bathymetry share one budget,
  so floating islands are impossible rather than forbidden.
- **Sea level applied last** (lowstand-then-flood candidate): erode
  against a lower stand, flood to final level → drowned valleys,
  estuaries, island fields, shelf-as-drowned-landscape.
- **Render is separate and cheap**: stepped hypsometric ramp, dense
  stops near sea level; render-class controls never regenerate.

*Full time-stepping (surface evolution simulated through time as well)
was deliberately NOT rejected — the author may explore it later. Keep
stage boundaries clean enough that the structural stage could be
swapped for a deeper simulation without rewriting downstream.*

## The frame is a window

The simulated world extends beyond the delivered map on all sides.
Ranges, trenches, and coasts cross the frame without paralleling it
because they cannot see it. **Nothing in the pipeline may compute a
function of frame coordinates** (the crop itself excepted).

Border rule, clarified by the author: the hard requirement is only
**no land in the outermost ring** — depth is irrelevant there, so a
flooded shelf may legally run under the frame. Frame-correlation of
*any* visible structure (including bathymetry) remains a rejected look.
The guarantee must come from where the causes of land can exist
(continental nuclei and volcanic sources confined to the world
interior, tapering at the *world* rim — which lies outside the frame).
Spike S3 proves or breaks this.

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
- **Builder never judges alone.** Evaluation harness per the ratified
  scheme: instrument views (isobaths, slope, transect overlays,
  spectra), deterministic tripwires that veto but never approve, blind
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
- **Every stage ships a view.** Work that cannot be seen cannot be
  judged; cause-fields stay renderable so features can be
  cross-examined against their causes.
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
  ablation evidence at review, author-decided trims.

## Aesthetic register

Positive canon: `examples/ref1–14.png` (Gleba screenshots) — emulate
formation qualities feature-by-feature; composition is contract-governed.
Ignore globe distortion, day/night lighting, land-on-frame.

Rejected on sight (from author-reviewed failures; see CONTRACT §11a
plus author sessions of 2026-08-29):

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
11. Border contouring — any structure (land *or* seafloor) tracing the
    frame; right-angle corner features.
12. Obviously geometric features anywhere — arcs, rings, straight
    lines, even spacing, even widths.

## Perf targets and named risks

Contract §15: 256² < 1 s, 512² < 3 s, 1024² < 15 s, 2048² < 90 s;
re-render in milliseconds.

Risks, tracked openly: erosion-solver cost at 2048²; resolution
independence of the erosion stage specifically (large valley structure
must hold across res; the contract permits fine-texture divergence);
the border guarantee's tail behavior across seeds. Spikes S1–S4 exist
to de-risk these before architecture commits.
