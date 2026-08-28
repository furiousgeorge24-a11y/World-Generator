# Design record

Accumulated architecture and stage design. This is the "how we intend to do
it" companion to `contract.md` ("what must be true"). It churns; the
contract shouldn't.

## World layers (provisional)

| Layer | Type | Meaning |
|---|---|---|
| `elevation` | f32 | metres, sea level = 0 |
| `plate_id` | i16 | plate membership |
| `crust` | u8 | oceanic / continental |
| `crust_age` | f32 | oceanic crust age (drives subsidence) |
| `uplift` | f32 | tectonic forcing |
| `flow_dir` | i8 | drainage direction |
| `flow_acc` | f32 | upstream catchment |
| `lake_id` | i32 | filled depressions |
| `temperature` | f32 | after N/S profile + lapse |
| `moisture` | f32 | after aridity profile + rain shadow |
| `wetness` | f32 | waterlogging (swamp/bog ingredient) |
| `local_relief`, `slope` | f32 | landform discriminators |
| water masks | u8 | ocean / lake / river (+ depth) |
| flags | u8 | volcanic; glacial if M6 |

## The spine

1. **Plates** — seeds, growth (varied rates → size heterogeneity), Euler
   motion, crust content; oceanic-frame-plate bias.
2. **Boundaries** — per-point classification → uplift field.
3. **Volcanism** — arc sites, hotspot tracks, rift volcanoes.
4. **Base relief** — noise modulated by the tectonic field.
5. **Erosion** — flow routing, depression handling, stream-power incision,
   talus. Sea level is fixed *before* this stage (it is erosion's base level).
6. **Hydrology** — final flow accumulation, lakes, coastline, bathymetry
   detail.
7. **Climate** — stylized profiles + rain shadow + wetness.
8. *(optional)* **Glaciation** — cold-region carving; one-shot circularity
   accepted (carve after climate, no iteration).
9. **Export** — layer emission + hexify.
10. **Render** — hypsometric (reference look); field renders for climate.

## Two-regime elevation (the central architecture)

Earth's hypsometry is bimodal (continental platform vs. abyssal floor);
one fractal field thresholded at a percentile is unimodal and reads as
"land, but blue" underwater. Therefore:

- **Crust type, not elevation sign, splits the regimes.** Continental cells
  (including sub-sea-level shelf) get the land system: uplift + noise +
  erosion. Oceanic cells get the marine system: age-subsidence + tectonic
  features + sediment smoothing.
- **Coastlines and plate boundaries are independent.** Continental crust
  comes from interior-anchored nuclei; plates are a separate partition. A
  coast is active only where a convergent boundary runs near the crust edge
  (west-coast trench vs. east-coast shelf asymmetry).
- **Land fraction** is set mainly by continental platform area; sea level is
  a fine-tune that slides the coast across the low-gradient shelf. Because
  the shelf is flooded land carrying real drowned valleys, moderate
  sea-level moves produce rias — correct without re-running erosion. The
  slider's promised range rides the margin band only.

## Bathymetry appearance spec

Governing grammar: **land is rough because rivers cut it; the seafloor is
smooth because sediment buries it — its drama is tectonic: sparse, linear,
deliberate.** High-frequency detail dies below the shelf break.

Coast → abyss: shelf (flat, muted, wide on passive margins, absent on
active) → shelf break (legible edge) → slope (steepness = margin type;
canyons sparse, opposite major river mouths, ending in fans) → rise
(passive) → abyssal plain (flattest thing on the map; seamount chains;
faint abyssal-hill fabric aligned to the ridge that made the crust) →
ridge swell / trenches / fracture zones.

Dendritic underwater is legal in exactly two places: muted drowned valleys
on the shelf, and submarine canyons notching the slope. A report finding
guards the rest.

## Border stack (contract §7 mechanism)

Named methods, verdicts:

- **A — warped-margin falloff**: noise-warped distance-to-edge falloff on
  continental potential. Adopted (as warp on B's placement density).
  Overdone-artifact: vignette.
- **B — interior-anchored sources**: all land-building (nuclei, orogeny,
  hotspots, edifices) sampled from a placement density reaching zero near
  the frame; bounded kernels give a guarantee by construction. **Backbone.**
  Same machinery as the future guide-mask feature (an authored mask is a
  hand-drawn placement density). Overdone-artifact: blob continents —
  break with noise + tectonic structure.
- **C — boundary-conditioned diffusion**: ocean-valued Dirichlet solve.
  Viable alternate; not adopted (soapy margins, least direct control).
- **D — oceanic frame plates**: seeding bias. Adopted as assist; cannot
  guarantee alone.
- **E — window selection**: generate large, crop a water-ringed window.
  Set aside (needs a fallback, which means building a second system).
- **F — sea-level accommodation**: last-millimeter trim only.
- Plus: thin absolute potential floor in the outer ring (provably below any
  later positive contribution) as the contract's backstop — construction,
  not masking.

Controls: `border_sea_width` (primary), `border_irregularity` (advanced).

## Tectonics

- **Motion**: per-plate Euler pole + angular rate (+ optional translation),
  not uniform drift. Relative velocity varies along each boundary, so
  boundary character transitions along its length; obliquity partitions
  into normal (builds relief) and shear (offsets, fault grain) components.
- **Convergent interactions** (polarity fixed per segment: oceanic under
  continental; older/denser under younger):
  - ocean under continent: trench + coastal cordillera at an **arc–trench
    gap** inland (mountains behind a coastal strip, never on the trench lip).
  - ocean under ocean: trench + volcanic island arc, **convex toward the
    subducting ocean**.
  - continent + continent: no trench; collision belt = ridge **+ plateau
    term** (Tibet, not just Himalaya) + subtle foreland strip.
  - Amplitudes scale with convergence rate; along-strike low-frequency
    modulation so arcs aren't uniform sausages. Outer rise: faint oceanward
    flex band (flourish).
- **Divergent**:
  - Oceanic: the ridge is the basin's skeleton — crust age = distance from
    ridge along spreading; depth follows √age subsidence. The swell is wide
    and gentle; only the axial detail is sharp. Spreading rate selects
    axial form: slow → median rift valley, fast → smooth axial high.
    **Segmentation**: offset spreading segments linked by transforms,
    fracture-zone scars running into old floor. Basins with no ridge are
    uniformly old: flat, deep. Ridge-hotspot coincidence may surface an
    island (rare).
  - Continental rifts: graben **with uplifted shoulders** (no shoulders =
    scratch, not rift). Graben floors are genuine closed depressions, so
    hydrology fills them: rift lakes for free. Rifts branch (triple
    junctions); some arms fail. Maturity ladder per rift: valley → lake
    chain → narrow sea. Narrow-sea coasts share the same 1D noise along the
    rift line: **jigsaw-fit coastlines** without simulating drift. Failed
    rifts imprint mild linear lowlands; erosion routes major rivers down
    them later.
- **Curvature**: (1) convex-arc construction at subduction segments (the
  signature), (2) low-frequency domain warp of the partition, (3) Euler
  kinematics making intensity vary along curves. Growth-noise raggedness as
  the fine scale.
- **Grammar**: convergent features are narrow/sharp/asymmetric/paired;
  divergent features are broad/gentle/symmetric with only thin axial detail
  sharp. Painting a ridge like a sunken cordillera is the canonical failure.
- **History fork**: **A — kinematic snapshot** adopted for M1. **C —
  pseudo-history eras** (2–3 successive snapshots; older orogens imprinted
  first, aged/blunted — composes with erosion ordering) proposed as
  extension, **status: open, needs author call**. **B — time-stepped
  simulation: rejected.**
- Flourishes (default-build under the breadth stance, each behind a knob,
  each ledgered): jigsaw seas, failed rifts, back-arc basins, outer rise.
- Volcano **age split**: edifices placed pre-erosion come out dissected;
  post-erosion stay fresh cones. Ordering, not simulation.

## Climate (no seasons — closed by author)

- Temperature: author-shaped north–south profile (default: cold both ends)
  minus altitude lapse (snowy peaks fall out), plus optional continentality.
- Moisture: author-shaped aridity profile + ocean moisture marched along
  prevailing wind (control), orographic rain on ascent, rain shadow behind.
  Border-ring ocean makes every map edge a clean moisture source.
- Wetness: waterlogging from slope + flow accumulation + water proximity.
- **Open (gates M4 defaults): "center arid" = geometric (equator band) or
  continental (interior) as the default look.** Both exist as controls
  regardless.

## Export ingredients (proves sufficiency for the target's biome calls)

| Target's call | Ingredients |
|---|---|
| swamp | warm + waterlogged + near fresh water |
| bog | cold + waterlogged |
| savanna | hot + mid moisture |
| fields/prairie | temperate + low-mid moisture + low relief |
| hills | mid local relief |
| mountain | high relief/elevation |
| snowy mountain | high elevation + sub-freezing |

Layers: elevation, slope, local_relief, temperature, moisture, wetness,
water masks/depth, dist_to_coast, volcanic flag (glacial flag if M6).

## Open questions log

| # | Question | Status |
|---|---|---|
| 1 | Eras (history option C) authorized as extension? | open — author call |
| 2 | "Center arid": geometric vs. continental default | open — gates M4 |
| 3 | scipy for depression fill | deferred — benchmark first, ask with numbers (M2) |
| 4 | Export schema details | deferred — co-design at M5 |
| 5 | Contract v0.1 → v1.0 blessing | pending author review |
