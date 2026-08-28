# Milestones

Status: **M1 in progress** — C0 (walking skeleton) done; C1 (partition) next.

Cross-cutting foundations, built into M1 because they're retrofit-hostile:
per-stage RNG keying; control registry as data (name/range/default/promise/
invalidation/tier); never-fail + report sidecar; PNG provenance; physical
units + world-space sampling (structural resolution independence); batch/
gallery CLI as the formal review vehicle; **every layer a milestone adds
ships a named view** (render.py VIEWS; webui selector + batch --views);
**webui shell** (Flask; sliders
generated from the registry, seed control, generate button, preview panel —
built first, against a stub pipeline, so the registry and generate/render
split are real from day one; it grows itself as stages add controls, with
invalidation classes going live incrementally: render-class instant from
v0, late-class caching as stages accumulate). Every milestone exits
through: gallery review (with ablation pairs for predicted-marginal
features) + current value-ledger rows + a commit recommendation.

## M1 — Skeleton
*Continent silhouettes with tectonic intent, on a guaranteed water frame.*

- Webui v0 shell + stub pipeline (first deliverable, then the stages below
  land behind it).
- Plates: seeding, growth (varied rates), Euler-pole motion, crust content,
  oceanic-frame bias.
- Continental potential: interior-anchored nuclei, warped placement density,
  absolute border floor backstop. Guide-mask *slot* in the seeding API
  (feature itself later).
- Boundary classification → uplift painting: couplet with arc–trench gap,
  collision ridge+plateau, rift graben+shoulders, ridge swell + age field,
  segmentation + fracture zones, along-strike modulation, curvature stack.
- Flourishes (knobbed, ledgered): jigsaw seas, failed rifts, back-arc
  basins, outer rise.
- Volcanism placement (arc sites, hotspot tracks) as uplift contribution.
- Two-regime elevation from day one: platform + oceanic floor
  (age-subsidence, ridges, crude trench/shelf bands); sea level fixed
  before any erosion exists; bimodal hypsometry.
- Base relief noise modulated by tectonic field.
- Hypsometric render (reference palette, nearest-neighbor), batch galleries.
- First image review doubles as aesthetic elicitation: style variants
  (hillshade on/off, quantized vs. smooth ramp) + scale ladder
  (`cell_size_km` sweep).
- Expectation: maps look *uncarved* — no erosion yet, no dendrites.

Gate: author "go." Related open call: eras extension (design.md log #1).

## M2 — Carve
*The reference look, complete.*

- Flow routing; depression handling (scipy-vs-numpy benchmark here, ask
  with numbers); stream-power incision; talus; lakes.
- Sediment appearance: shelf smoothing, rise, fans.
- Submarine detail: canyons opposite major rivers, drowned shelf valleys,
  abyssal fabric, seamount edifices.
- Volcano age split (dissected old / fresh young).
- New report finding: roughness above vs. below shelf break.
- Webui: late-class invalidation (cached-field recompute) should be fully
  live by end of M2 — erosion is the stage that makes it matter.

## M3 — Climate
- N/S temperature profile − lapse ± continentality; aridity profile +
  moisture march + rain shadow; wetness from hydrology.
- Review by field renders (temperature, moisture, wetness galleries).
- **Gated on design.md log #2 (center-arid default).**

## M4 — Export
- Layer set finalized; schema co-designed with the target editor (its API).
- `hexify`: area-weighted per-hex aggregates, pointy-top odd-r, border
  hexes water, apron covers border-hex footprints.
- Exit: target system consumes a map end-to-end and derives biomes.

## M5 — Glaciation *(optional, unauthorized)*
- Fjords, U-valleys, cirques as post-climate carving; one-shot circularity
  accepted. Built only on explicit authorization.
