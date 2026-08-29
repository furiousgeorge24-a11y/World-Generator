# Milestones

Status: **M2 code done** — W1–W3 complete; **A1 addendum landed**
(crust-plate affinity, author-authorized 2026-08-28; version 0.4.0). Both
review packs regenerated at 0.4.0 — formal M1 (out/m1_review/) and M2
(out/m2_review/) image reviews pending, plus the A1 sweep
(out/a1_affinity/). **Aesthetic canon landed** (examples/ref1–14, all
hypsometric; see design.md "Aesthetic canon") and the 2026-08-28 canon
review produced the process-footprint principle (CLAUDE.md + design.md),
a named feature-formation candidate list, and a zero-based review naming
three wrong-path foundations. Proposed next: the **K-series** foundation
rework (below) — awaiting author authorization (design.md open q. 7).
Smaller candidates and the M1/M2 formal reviews wait behind it.

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

## K-series — Foundation rework *(proposed, unauthorized)*
*Fix old work before moving on: the three wrong-path foundations from the
zero-based review (design.md), in dependency order. Each run: version
bump, galleries, ledger rows, commit recommendation.*

- **KR — Ramp rework** — **code done (0.4.1)**: `render_palette` control
  (classic / canon / canon-soft / canon-crisp; stops calibrated to
  measured hypsometry), sqrt-space quantize, lake color floor. Style
  sheets in out/kr_ramp/ (_palettes.png, _quantize.png); author default
  pick pending.
- **K1 — Drowned datum** — **code done (0.5.0)**: implemented as
  formation base level −`flood_rise_m` (default 120 m, author-picked)
  with the relief assembly unchanged — land_fraction stays exact by
  construction and `sea_level_m` stays the late trim. Carve grades to the
  lowstand coast; canyons continue real rivers from it; `wave_planation`
  (author-approved scope) benches the fossil shelf break, cut-only.
  19 smoke tests green (border at flood extremes, coastline invariance
  under planation, shelf incision > 0). Galleries: out/k1_flood/
  (_flood sweep, _planation, _coasts zooms). **Honest exit status:**
  platform seas gained interior structure and river-mouth notches, but
  map-scale ria intricacy is capped by smooth lowlands — the K1 exit
  criterion completes when K3's worked plains land (dependency confirmed,
  not a defect). Perf finding: 1024² ≈ 19.8 s vs 15 s budget, ~+0.8 s of
  it K1's; perf pass nominated post-K-series (see ledger).
- **K3 — Erosion mass balance** — **code done (0.6.0)**: tapered channel
  initiation (`lowland_dissection`) replacing the hard gate; downstream
  deposition (`deposition`) with conservation bookkeeping
  (deposited/basin/exported findings); no-flux-coast diffusion (audit
  defect closed, drift 0.022→0.005); `plains_grain` (fixed-km octaves to
  ~2.2·cell). Also fixed en route: K1's planation shaved kilometre-deep
  craters near cliffs — cut now capped at 45 m ravinement thickness.
  22 smoke tests green. Galleries: out/k3_carve/ (_defaults, _deposition,
  _grain, _dissection sweeps + _coasts zooms). **Exit status:** worked
  mottled lowlands, island-field platforms, and intricate coasts landed
  (K1's deferred coastal payoff visibly arrives with grain ≥ 0.5);
  deposition is mechanically correct but conservative at defaults —
  floodplain prominence is an author tuning call at review. Perf: 1024²
  ≈ 23.7 s vs 15 s budget (deposit loop ~doubles erosion); perf pass
  stays nominated post-K-series. Author default picks pending from the
  three sweeps. flood_rise_m default raised to 250 (author K1 pick).
- **K2 — Physical profiles** — **code done (0.7.0)**: convergent belts
  rebuilt as profiles — saturating crest height (isostatic ceiling),
  width breathing with convergence intensity, apron flanks, foreland
  flexure (folded into `outer_rise`), and past saturation the orogen
  spreads into a rim-enclosed plateau (`plateau_tendency`; ref14
  grammar — verified in out/k2_profiles/_plateau.png: doubled rims,
  elevated floor, trapped lakes). Retired: whole-stack tanh (→ linear +
  ceiling knee), edged-plateau smoothstep, collision-plateau blob.
  Calibration: land_fraction held (feedback 0.75→0.55), peaks bounded
  (max ≈ 5,900 m, snow confined to crest cores after two amplitude
  trims). 24 smoke tests green. Anatomy zooms show nested bands, high
  belt lakes, foreland lake chains. Edifice moat/bench: existing comp
  rings re-grounded in docs; per-edifice benches deferred (planation
  covers the lowstand case). Author picks pending: `plateau_tendency`
  default from the sweep. Perf: 1024² ≈ 24.2 s (+0.5 s).
- **Combined formal image review** closes the series — replaces the
  separately pending M1/M2 reviews (author-approved fold). **Pack built
  at 0.7.0, rebuilt at 0.7.1 after the tier-1 retune: out/k_review/** —
  README.md inside maps every sheet to the ledger verdicts, default
  picks, and open questions it settles. Awaiting the author's sitting;
  ledger observed-yield columns fill there.
- **Tier-1 default retune** — **done (0.7.1)**: the k_review-vs-examples
  canon comparison (2026-08-28) produced a ranked gap list; the author
  authorized tier 1 (default tweaks). Picks from out/tier1_defaults/
  evidence sweeps: `plate_count` 6, `render_quantize` 12,
  `lake_min_depth_m` 6, `deposition` 0.8, `plains_grain` 0.7. Smoke
  24/24; ledger row current. Still awaiting authorization: tier 2
  (belt raggedness, crest-zone mass, foreland along-strike modulation,
  lake palette quieting) and tier 3 (provincial interiors, margin
  asymmetry, river-overlay test), plus the nominated perf pass.

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
