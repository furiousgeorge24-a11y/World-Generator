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
- **Coastlines and plate boundaries are correlated, not coincident.**
  *(Amended by the A1 run, author-authorized 2026-08-28; the original
  decision was full independence, which read as two unrelated random
  fields.)* Continental crust still comes from interior-anchored nuclei,
  but nucleus *cluster centers* are rejection-sampled toward the interiors
  of a size-weighted subset of continent-carrying plates
  (`crust_plate_affinity`; 0 restores independence). Candidates are scored
  analytically in world-km against the same warped plate metric as the
  raster partition, so placement is resolution-independent. Kernels still
  scatter and spill across boundaries — coasts are never redrawn to track
  plates. A coast is active only where a convergent boundary runs near the
  crust edge (west-coast trench vs. east-coast shelf asymmetry); affinity
  makes that configuration common instead of lucky.
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

*Formation note (canon review 2026-08-28):* the canon confirms this
grammar and names its likely best mechanism — the shelf zone formed as
land at glacial lowstand, then flooded (`postglacial-flood` candidate
below); the shelf break is a fossil coastline, which is why it reads as a
crisp line.

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

**2026-08-28 finding (author, with crops):** land often contours the
frame closely, and three of four map corners grew right-angle-ish
coastlines. Mechanism identified: `d_edge` is min-distance-to-edge,
whose iso-contours are sharp-cornered rectangles; and when a kernel is
large or near the frame, `pot × falloff` crosses the land threshold
inside the falloff band — method A *authors* the coast (frame geometry
shows) instead of vetoing it. The single-octave wander (≤ 0.8·margin)
only wobbles the rectangle. The author's blessed counter-example is the
intended behavior: a small landmass whose own kernel edge authors the
coast. Named candidates — **border naturalization** run, unauthorized:
(1) rounded-rectangle SDF for `d_edge` (kills corner right angles at
the metric level); (2) frame-fit kernel budget — shrink/reject kernels
whose footprint would lean on the falloff band, so frame-adjacent
continents are naturally small and the floor returns to backstop duty
(method B strengthened); (3) multi-octave wander, amplitude still
bounded inside the guarantee ring. **Constraint baked into A2:** the
leading-edge shift must clamp against the placement margin — a
convergent boundary near the frame must not pull crust into the border
band. Minor annotation: stray single-pixel deep water off some
shorelines (canyon notch or quantize band edge?) — inspect during this
run.

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

## The process-footprint principle (standing agreement, 2026-08-28)

Features are the **steady-state footprints of named natural processes**,
parameterized by causal fields the pipeline already computes (boundary
kinematics, crust age, margin activity, flow accumulation). Nothing is
placed for its appearance: paint the cause and let the render reveal the
consequence. Canonical example — a volcanic island is magma flux (edifice)
+ load/age (flexural moat, subsidence) + wave base (planation bench); the
concentric rings in the hypsometric view are emergent, never drawn. A
feature built the wrong way fails predictably: it disagrees with its
surroundings and stops responding coherently to controls.

Boundaries of the principle, blessed with it:

- "Process" means *steady-state footprint*, not time-stepped simulation.
  History option B stays rejected (cost, chaos under control-dragging,
  promises that emergence makes unkeepable). Eras remain snapshots.
- Sub-grid texture noise is legitimate as **parameterization of unresolved
  processes**; its amplitude must be process-modulated (depositional
  lowlands smooth, erosional uplands rough, young volcanic surfaces rough,
  sedimented abyss smooth) — never uniform decal jitter. Precedent: even
  the canon's source generator ships a "small stochastic uplifts" control.
- Control promises are worded in process terms — occurrence rates and
  physical magnitudes (cm/yr, metres, °C) — not outcome shapes. The source
  generator's author-facing surface is an existence proof: all process
  frequencies plus climate, zero appearance dials, and the canon look
  emerges.

## Aesthetic canon (examples/)

`examples/ref1.png`–`ref14.png`: author-blessed excellent outputs from a
separate, external globe-generating program (**not** the quarantined
parent lineage; positive references are legal here). All are hypsometric
views — early mis-reads of lowland mottling as vegetation are corrected
below. Three source artifacts are explicitly *not* goals: globe
distortion, day/night shading (some images are dark), and land touching
the frame (our sea ring stands). ref14 was added by the author
specifically so plateau features are not trained away.

Confirmed canon qualities — formation-focused; highland/lowland *balance*
is deliberately out of canon (it stays in author knobs):

1. **Belt anatomy.** A belt is nested ragged bands (foothill apron → flank
   → maroon core → near-black crest), each band edge fractal at cell
   scale. Asymmetric: steep on the trench/pro side, wide apron on the
   retro side. Width pinches and swells along strike. Crests are clumped
   massif strings carrying small high lakes; flanks are visibly dissected
   while crests stay clean. Terminations taper into hill country or exit
   coasts as island-arc tails.
2. **Plateaus are first-class.** Two kinds: rim-enclosed orogenic plateaus
   (flat high floor between belt crests, speckle lakes, interior calm,
   dissection only at the rims) and vast tabular uplands with escarpment
   edges gnawed by canyon-head fringes.
3. **Interior features.** Lone inlier massifs in plains; stepped
   terrane-block patchwork in interiors; volcano edifices whose concentric
   shelf-ring structure is emergent.
4. **Worked lowlands.** Cell-scale mottling everywhere, faint incised
   valley webs converging tree-wise, floodplain lake chains, and a
   distinct dissected hill-country register between plains and belts.
5. **Coasts and shelves are drowned landscape.** The platform is flooded
   terrain — banks, drowned valleys, island fields with land's own fabric;
   the shelf break is a crisp fossil-coastline at one depth; rias notch
   the coast. Breadth: extreme on passive margins, razor-thin on active.
6. **Bathymetric drama is linear and sparse.** Trenches are narrow crisp
   lines at the arc–trench gap; behind arcs sits a distinct marginal-sea
   depth register; remnant arcs ghost as pale drowned bands; the abyss is
   calm and sedimented with one or two deep pools per basin. (Matches the
   existing bathymetry spec; the canon confirms it.)
7. **The ramp carries half the look.** Hypsometric stops dense near sea
   level on both sides, compressed mids, dark summits with sparse
   snow-grey caps. Banded feature anatomy = terrain crossing thresholds
   noisily × a ramp with visible stops.
8. *(open — author ruling pending)* Chunky cell-scale grain on coasts and
   texture: source-engine artifact or a wanted style?

## Feature-formation candidates (canon review, 2026-08-28)

Named at the canon review; process-grounded per the footprint principle.
**None implemented; none authorized.** The author authorizes by name.

| Candidate | Process grounding | Notes |
|---|---|---|
| `postglacial-flood` | Eustasy: form terrain (carve included) at glacial lowstand base level, then raise sea level N metres near the tail | The source generator's core coastal mechanism ("Post-LGM sea level increase"). Subsumes three earlier candidates: shelf-break sharpening, platform mottling, ria-coasts — the break is the fossil lowstand coast. Compatible with the head/tail split; zero = clean ablation |
| `superswells` | Mantle dynamic topography: broad uplift domes (African-superswell style) | The causal story for vast tabular uplands (ref13 west, ref14 left); escarpment dissection then emerges from the existing carve. Replaces the era-roots guess for tablelands |
| `rim-enclosed-plateau` | Crustal thickening between parallel convergent fronts + endorheic sediment fill; hydrology already identifies basins that cannot drain | Tibet-style (source: "Tibetan plateau frequency"). Flat floor *is* deposition; speckle lakes fall out of fill-and-label |
| `terrane-blocks` | Accreted crustal blocks uplifting as units, competing for space with smooth uplands | Source: "Uplands frequency" tooltip. Adds sharp-edged stepped patchwork to interiors alongside (not replacing) smooth province swells |
| `deep-plumes` | Ridge-interacting large plumes (Iceland-style): swell + subaerial volcanic platform; radial arms are elevated ridge segments crossing it | Decodes ref2's star island. Second plume class beside our Hawaiian-style chains (which the source confirms as-is) |
| `inlier-massifs` | Worn roots of ancient orogens — era relics expressed as compact clumps, not only belts | Source's "old mountains / old hills" split suggests degradation *class* as the author-facing control |
| `belt-width-breathing` | Convergence rate/obliquity vary along strike — Euler `vn` already computed, plumbed to amplitude but not width | Plumbing, not painting |
| `belt-asymmetry` | Subduction polarity (known) sets the steep pro-side; foreland flexure under the load builds the retro apron | |
| `flank-dissection` | Carve concentrates on aprons (slope×area); crests stay clean | Mostly retuning visibility of what the channel threshold already does |
| `high-lakes` | Closed crest-zone depressions surviving fill-and-label | Verify emergent before building anything |
| `lowland-incision` | Taper the channel-initiation threshold instead of hard-gating; plains channels mark one band deep | Audit's lowland thread |
| `floodplain-deposition` | Deposit where transport capacity drops; ponding gives river lake chains | Audit candidate B5, canon-backed |
| `plains-grain` | Process-modulated sub-grid relief (see principle) | Currently tectonically gated to ~zero on plains |
| `marginal-sea` | Back-arc basin = young oceanic crust → the existing age-law makes it shallow, smooth, distinct | Re-plumb `backarc_basins` into the age field |
| `remnant-arcs` | Trench rollback strands the extinct arc behind the back-arc basin; subsided by age since extinction | Kyushu–Palau style pale drowned band |
| `trench-narrowing` | Real trench width (~50–100 km) with flexural outer rise; our σ is simply too wide | `outer_rise` already owns the outboard half |
| `calm-abyss` | Pelagic/turbidite burial smooths old floor; abyssal-hill fabric survives only on young thin-sediment crust | Fabric amplitude decays with age/sediment — ties to existing ledger ablations |
| `edifice-anatomy` | Flux → cone **on a broad constructional pedestal/apron that shoals the surrounding floor**; load/age → flexural moat + arch + subsidence; wave base → bench; old edifices become guyots. Author-flagged 2026-08-28: deep bathymetric bands currently pass beneath islands unperturbed (the "plopped" look) — the apron is the piece that makes contours deflect around an island, emergently in render | Rings emerge in render, never drawn |
| `palette-rework` | Render-side half of the banded look (canon 7): nonlinear stops, dark summits, deep-band compression | Render-only; compresses much abyssal texture for free |
| `river-overlay` | Render `flow_acc` lines in hypsometric — *test after* palette + incision; may be unnecessary | Demoted pending that test |
| `coast-accent` | One-cell coastal rim in render | Render-only |
| `plate_speed` | Global kinematic magnitude (cm/yr) scaling all boundary consequences causally, upstream of paint amplitudes | Source: "Average plate speed" |
| `crust_clustering` | Budget nucleus mass by land-cluster size distribution (fraction in largest / second-largest cluster) | Source's Pangaea/New-World dials; the legible home for the A1 supercontinent watch item |

Shelved per author steer: `mountain-scarcity` and all abundance
judgments — composition balance belongs to author knobs, not the canon.

Source-generator reference notes (mechanics only; globes/seasons remain
out of scope): terrain formed at lowstand then flooded; superswells; two
plume classes (shallow static chains / deep ridge-coupled); uplands vs
uplifted terranes competing for interior space; old-orogen frequency
split by degradation class; orogeny severity driven by plate speed; land
composition via crust fraction + clustering fractions; coral reefs and
LGM-anomaly glacial features are climate-era systems (M3/M5 notes).

### Zero-based review (2026-08-28): three keystones

Author-directed reevaluation of *all* prior work against the single goal
"produce example-like results via simulated process, sunk cost ignored."
Verdict: the skeleton (kinematic snapshot → causal fields → footprint
painting → hydrology/erosion → tail) is the right path. Three foundations
are wrong-path; most listed candidates are *expressions* of them and
would be tuned twice if built first.

- **K1 — drowned datum.** The coastline is currently a primary object
  (quantile land mask, sea level 0 before erosion, shelf painted as a
  band); in the canon it is a consequence of flooding a formed landscape.
  Replace: head stages (carve included) operate against a glacial
  lowstand base level; the tail floods by a controlled amount in metres.
  The shelf break becomes a fossil coastline; rias/drowned valleys/island
  fields emerge. Re-keys: erosion grade floor, sediment depth bands.
  Border guarantee unaffected (ring floor is far below any lowstand;
  flooding only adds water).
  **Implemented (0.5.0):** as formation base level — relief assembly
  still builds final-datum elevations; erosion grades to
  −`flood_rise_m` (default 120, author pick) and sediment gains
  cut-only `wave_planation` at the lowstand shore plus canyon mouths at
  the lowstand coastline. This keeps `land_fraction` exact by
  construction and keeps `sea_level_m` as the late-class trim — the
  exact-vs-loose sub-decision dissolved. Gallery verdict: platform seas
  gained real interior structure; full ria intricacy confirmed gated on
  K3's worked lowlands (the dependency the zero-based review predicted).
- **K2 — physical belt/edifice profiles.** Boundary placement and type
  are causal and stay; the belt *shape* is a prescribed symmetric
  Gaussian (σ/amplitude tables) — the concentric-rings failure at orogen
  scale. Replace with a small footprint model: crustal thickening (from
  vn × crust involvement, capped by an isostatic ceiling → rim-enclosed
  plateau floors emerge where fronts are close) + retro-side flexure
  (foreland basin + forebulge) + polarity asymmetry. Same model at small
  scale gives volcanic edifices (flux cone + flexural moat + wave-base
  bench; rings emergent). The FFT splat machinery survives as the
  renderer of profiles. Slated for retirement/subsumption by K2: tanh
  `_POS_CAP` top-end (saturation becomes physical), compensated arc
  halos (flexural moat is the process version), part of massif
  decomposition (clumping should partly emerge from vn variation),
  `outer_rise` as a separate flourish (folds into trench flexure).
  **Implemented (0.7.0):** convergent branches emit profile rows —
  saturating Hs = H_ceil·T/(T+k) per class, plateau fill rows + far rim
  where T exceeds a `plateau_tendency`-gated threshold, apron and
  foreland-flexure rows retro-side (foreland shares the `outer_rise`
  knob — both are plate flexure). Whole-stack tanh replaced by linear
  stacking + isostatic-ceiling knee (4800 m / 0.22); R3 edged-plateau
  shaping and the C2 collision blob retired; R1 feedback coefficient
  retuned 0.75→0.55. Kept pending review evidence: massif decomposition,
  arc/hotspot comp rings (re-grounded as flexural moats, values
  unchanged). Per-edifice wave benches deferred — K1 planation covers
  the lowstand case; age-tracked benches would need edifice birth times.
  Gallery verdict: nested-band anatomy, doubled rims with trapped lakes,
  foreland lake chains — canon qualities 1 and 2 now emergent.
- **K3 — erosion mass balance.** The hard channel-initiation threshold is
  process-false (manufactures dissection by switching incision off; makes
  plains dead-smooth) and eroded mass vanishes (no floodplains, valley
  fill, endorheic plateau floors). Replace: taper incision continuously
  into the hillslope regime; route eroded flux to deposition where
  transport capacity drops (this is also what makes rim-enclosed plateaus
  *flat*). Includes fixing the confirmed coastline-diffusion leak (a
  process violation: diffusion must not move coasts) and process-
  modulated plains grain (deposition smooth / erosional rough).
  **Implemented (0.6.0):** taper exponent from `lowland_dissection`;
  downstream settling pass with conservation findings; no-flux-coast
  Laplacian; `plains_grain` in fixed-km octaves floored at ~2.2·cell.
  Deposition constants (_S_REF 2.6 m/km settling slope, _CAP 3.5 m/step)
  are deliberately conservative — author tunes at review. Amends K1: the
  planation cut is capped at 45 m ravinement thickness (uncapped, the
  bump-vs-blur measure shaved km-deep craters where the band crossed
  cliffs). Gallery verdict: worked lowlands + platform island fields
  arrive; K1's coastal payoff confirmed unlocked.

Scorecard for everything else: kinematics, classification, eras,
hydrology, sediment stage, hotspot chains, border stack, registry/
determinism infrastructure — keep (compatible + compliant). Provinces —
re-ground as superswells. Seafloor fabric — retune per burial (calm-
abyss). Crust nuclei — keep as legitimate initial condition (the source's
own crust dials prove the precedent); clustering + internal structure are
upgrades, not fixes. Hypsometric ramp — rework (render-only, order-free).

Proposed order: **KR (ramp rework, render-only, first — all keystone
galleries get judged through it) → K1 → K3 (needs K1's datum) → K2 →
one combined formal image review** replacing the separately pending
M1/M2 reviews (their galleries would be invalidated by the keystones
anyway). Smaller candidates wait for corrected foundations plus review
evidence. Status: proposed, awaiting author authorization by name.

### Canon-distance comparison (2026-08-28, at 0.7.0)

k_review vs examples/, post-K-series. Verdict: the *structural* gap is
closed (composition, belt grammar, drowned coasts, palette family are in
the refs' language); the remaining distance is *surface*: ours reads
airbrushed where the refs read chunky and worked. Ranked gaps:

1. **Smoothness** — ragged fractal belt edges, mottled bands, chunky
   summit fields in refs vs our wire crests, smooth flanks, snow piping.
   The refs are also visibly quantized into stepped bands (q. 6 does
   real work in the canon look).
2. **Interior bullseye** — refs: green lowlands deep inland, tans as
   localized provinces; ours: green coastal ring, uniform tan dome.
3. **Lake speckle** — refs: rare structural lakes; ours: cyan confetti.
4. **No visible rivers** — refs' lowlands are threaded with dark
   dendritic lines; our incision doesn't read at map scale.
5. **Belt color mass** — refs spend area in the dark summit stops; our
   peaks live on 1-px crestlines.
6. **Abrupt margins** — refs: broad slope/rise transitions, wildly
   varying shelf width; ours: uniform turquoise halo → abyss.

Plus **plate scale**: at review extent (4,096 km) our 10-plate default
gives ~1,450 km plate footprints (microplate mosaic: short belt strokes,
trench commas); refs' frames read as 2–4 plates (long continuous
boundary systems). Confirmed by the plate_count 4/6/8/10 ladder.

Fix program, tiered by cost (author authorizes by name):

- **Tier 1 — default retune, authorized & done (0.7.1):**
  `plate_count` 10→6, `render_quantize` 0→12, `lake_min_depth_m`
  0.8→6, `deposition` 0.6→0.8 (honest: barely visible in 0.6–1.0 —
  constants are the real lever), `plains_grain` 0.5→0.7. Evidence:
  out/tier1_defaults/ sweeps + variety de-risk; smoke 24/24.
- **Tier 2 — belt-and-basin anatomy run, unauthorized** (expanded
  2026-08-28 after the author flagged inland-water anatomy: belt
  mega-lakes read wrong — many, massive, smooth — vs ref7's small
  irregular chains; the rift graben reads as a straight abyssal
  "laser". Diagnosis: K2 troughs are unsegmented along-strike,
  fill-to-spill lakes are maximal by construction, and K3's pit-trap
  cap is too weak to fill closed basins — we render the transient
  state, not the steady state):
  belt raggedness (along-strike shortening variation + differential
  erosion — attacks gap 1 at the source; also segments rim troughs
  into lake chains); **intermontane basin fill** (strengthen K3's
  closed-basin deposition branch so trapped mass fills floors toward
  spill — steady-state basins are sediment plains with small residual
  lakes, per Tarim/Po/Ganges; uses existing `basin_trapped_km3`
  bookkeeping); foreland along-strike modulation (kills the
  dead-straight foreland lake strips); **rift segmentation** —
  en-echelon graben segments linked by transfer zones, depth/width
  pulsing along-strike, sediment-floored young rifts (flooded arms
  read shallow, not abyssal) — East African lake-chain grammar on
  land, irregular flooded arms at coasts; crest-zone mass (summit
  *regions* near the isostatic ceiling, not ridgelines — gap 5);
  lake palette quieting (render-only). Untouched by all of this:
  the drowned-shallows/fragmented-coast look (K1+K3+quantize) the
  author blessed — B1's taper widens it. M3's evaporative lake
  levels later shrink dry-side lakes causally (deferred by design;
  not the primary fix).
- **Tier 3 — new candidates, unauthorized:** provincial interiors
  (strengthen `province_relief` into 500–1,500 km epeirogenic
  undulation + deposition filling the lows — gap 2; superswells/terrane
  blocks extend later), river-overlay test sheet (render read of
  hydrology — gap 4), and **B1 — passive-margin bathymetry** (gap 6,
  author-flagged 2026-08-28: depth plunges after every shoreline;
  canon plunges only at faults/trenches). Two mechanisms, one run:
  *stretched-margin crustal taper* — passive margins extend the crust
  field seaward over a controlled km width before the oceanic regime
  (rifted-crust thinning; buys shallow platforms + gradual basement),
  active margins stay trench-clipped so the plunge survives where
  earned; *exported-sediment rise* — route K3's `exported_km3` (today
  bookkeeping-only) into a low-gradient apron from the slope base,
  scaled by adjacent drainage export (re-grounds `sediment_softening`'s
  apron the way `outer_rise` became flexure). Sequenced after A2 —
  A2 creates the common active margins that make the contrast read.
  Scope note (2026-08-28): the taper applies to *any* continent–ocean
  transition — small crustal islands' perimeters included, so islets
  gain pedestals for free; volcanic islands get theirs from
  `edifice-anatomy` (candidates table), a natural rider on this run.
  Design commitment (second author crop, land "floating" over an
  abyssal basin abutting the shore): the margin/pedestal profile is
  *authoritative within its zone* — shore to slope base is authored by
  the profile, and the regional floor field (age law, basins) is taken
  up only beyond it; a purely additive taper would lose against deep
  structure the floor field places at the coast.
- Nominated perf pass slots before tier 3 (24 s vs 15 s at 1024²).

## Open questions log

| # | Question | Status |
|---|---|---|
| 1 | Eras (history option C) authorized as extension? | open — author call |
| 2 | "Center arid": geometric vs. continental default | open — gates M4 |
| 3 | scipy for depression fill | closed — numpy-only PD fill beat budgets at M2 (2.7 s @2048²); question dead |
| 4 | Export schema details | deferred — co-design at M5 |
| 5 | Contract v0.1 → v1.0 blessing | pending author review |
| 6 | Chunky cell-scale grain (canon quality 8): source artifact or wanted style? | closed — wanted; `render_quantize` default 12 (sqrt-space bands), tier-1 retune 2026-08-28; author may re-tune at the sitting |
| 7 | Keystone path (KR → K1 → K3 → K2 → combined review) proposed at the zero-based review — authorize? K1 sub-decision: exact vs loose `land_fraction` under flood | closed — authorized by name and executed (0.4.1 → 0.7.0; pack at 0.7.1); the K1 sub-decision dissolved (formation base level keeps `land_fraction` exact) |
