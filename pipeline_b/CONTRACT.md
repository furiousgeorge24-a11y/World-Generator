# Contract — output requirements

The promises. Everything here must be true of every delivered map.

This document specifies **what the generator produces**, plus one
standard governing how features may come to exist (§11). It contains no
architecture, no algorithms, and no mechanism design — those are yours
to invent. Where a requirement has a reason, the reason is stated in
terms of what a viewer sees, because that is how results are judged
here.

Real-world earth science is fair game and actively encouraged as a
source of understanding. Other map generators in this repository are
not: see the quarantine note at the repo root.

---

## 1. Scope and deliverable

- The generator produces **terrain form**: elevation, water, and derived
  physical fields. It does not produce settlements, regions, routes,
  names, history, or lore.
- **Ship ingredients, not conclusions.** Biome and terrain
  classification (swamp, savanna, hills, …) is explicitly out of scope.
  Emit physical field data; a downstream consumer makes those calls
  itself. A future consumer may want the fields downsampled; no schema
  for that is fixed yet, and it must never influence generation.
- The user-facing deliverable is a **PNG image** plus a machine-readable
  run report.
- Input is a **seed plus author controls**. The same inputs must always
  give the same map (§4).

## 2. Image, grid, units

- Square lattice, generation-native. Rectangular maps allowed: each axis
  independently **128–2048** cells.
- Edges are bounded. No wrapping — the map is a bounded region of a
  world, not a torus.
- **Physical units throughout.** Elevation in **metres, sea level = 0**.
  Horizontal scale is an author-set real distance per cell (km), so what
  a map *means* — a region, a continent, a world — is a knob rather than
  an ambiguity. Every requirement below that names a distance or a depth
  is in those real units and must hold at any resolution.
- **Structural resolution independence.** The same seed and controls at
  different resolutions must yield *the same world* — same landmasses,
  same ranges, same coastline shape — with finer detail at higher
  resolution. Fine texture may differ; large structure may not.
  Preview-then-final is a supported workflow, not a hope.

## 3. The water border

Two separate requirements. The first is absolute; the second is where
the author has rejected work on sight.

**3a. The outermost ring of cells is water on every delivered map.** No
land touches the frame, at any control setting, on any seed. Every run
reports the nearest-land-to-border distance as regression insurance.

**3b. Land must not crowd, parallel, or mirror the frame.** The
guarantee in 3a must fall out of *where land is allowed to form*, not
from painting the edge blue at the end. Masking or redrawing a border
after the fact produces the signature the author has explicitly
rejected: coastlines that run straight alongside the frame for long
stretches, and right-angle-ish corner features. On a reviewed batch the
author found "land often contours the border of a map very closely,
three of four map corners even have rather right angle-ish features …
terribly unnatural."

Terrain near the border is not itself bad — excess is. The author
accepted a bordering landmass that was "fairly small in size, doesn't
mirror the map border precisely, and curves in and out of it fairly
naturally." That is the bar: a coast near the frame must be as
uncorrelated with the frame as a coast anywhere else.

## 4. Determinism

- Same seed + same controls + same version → **bit-identical** output in
  the same environment, and structurally identical across environments
  (floating-point variance between platforms is acknowledged, not
  promised away).
- **Adjusting one control must not reshuffle unrelated things.**
  Dragging an erosion-related slider must not move the continents. A
  control changes what it promises to change and nothing else.
- Prohibited as inputs to results: wall-clock time, unordered iteration,
  unordered parallel reductions.
- Any change that alters output for the same seed and controls —
  including a changed default — bumps the version stamp.

## 5. Robustness, reporting, provenance

- **Generation never fails.** Every in-range control combination
  produces a map. Internal findings — clamps, anomalies, failed
  invariant checks — ship in the report *beside* the map. They never
  destroy a run and never block delivery.
- **Report per run**: seed, full control echo, timings, and findings
  (land fraction, elevation range, lake count, nearest-land-to-border,
  and whatever else is worth auditing).
- **Every PNG embeds its provenance** (seed, controls, version), so any
  image from any gallery can be regenerated exactly.

## 6. Bathymetry

The author has flagged underwater terrain as a specific, repeated
failure. Read this section as strictly as §3.

**Governing look:** land is rough, the seafloor is quiet. Fine-grained
detail largely dies below the shelf break; what drama the deep has is
**sparse, linear, and deliberate** rather than textural.

**6a. Depth must not plummet at the shoreline.** In the author's words:
"ocean depths don't plummet immediately after the shoreline unless a
fault lies there. Most gradually descend." A steep drop immediately
offshore, everywhere, is the single most-cited defect. The normal case
is a broad, gradual descent over a long distance.

**6b. The descent sequence, shore to deep**, each zone legible:

| Zone | Character | Real-world depth |
|---|---|---|
| Shelf | flat, muted, gently sloping | shore to roughly −150 m |
| Shelf break | a legible edge, crisp, at a consistent depth | roughly −100 to −200 m |
| Slope | the steep part; steepness varies by coast type | down to ~−2,000 m |
| Rise | broad low-gradient apron at the slope's foot | ~−2,000 to −4,000 m |
| Abyssal plain | the flattest thing on the map | ~−3,500 to −5,500 m |

**6c. Shelf width varies enormously between coasts** — very broad on
quiet trailing coasts, nearly absent on steep active ones. A uniform
band of shallow water haloing every landmass is a rejected look; so is
uniform shelf width around a single landmass.

**6d. Sharp plunges must be earned and rare.** Where a trench or fault
does run offshore, the drop should be abrupt — and *sharper* for the
contrast with gradual margins elsewhere. Trenches are narrow, crisp,
linear, sparse, and the deepest features on the map.

**6e. Islands must not float.** An island standing in deep water with
deep contours passing beneath it unperturbed reads as "plopped" —
pasted on rather than part of the seafloor. Islands, island chains, and
seamounts stand on broad shoaled aprons; **depth contours bow around
them**. The apron is wider than the island and the shoaling is gradual.

**6f. The abyss is calm.** Sedimented, smooth, with at most one or two
deeper pools per basin and only faint fine texture. Sheltered basins
behind island chains read at their own distinctly shallower depth,
separate from the open deep.

**6g. Branching detail underwater is legal in exactly two places:**
muted drowned valleys on the shelf, and sparse canyons notching the
slope (opposite major river mouths, ending in fans). Dendritic texture
anywhere else underwater is a defect — worth an automatic check.

## 7. Topography

**7a. Mountain ranges are often coastal.** From the reference review:
"in the reference images, most mountain ranges form along coastlines"
— since softened by the author to "ranges are often coastal," which is
the operative wording. Coastal ranges must be a *common* outcome, not a
lucky one. A generator in which ranges never meet a coast is wrong; so
is one where every range does.

**7b. Ranges are ragged, segmented, and irregular.** Not ruler-straight
ribbons, not smooth pipes, not single-pixel wire crests. Along their
length they pinch and swell in width, rise and fall through saddles and
passes, and wander off a straight axis. They terminate by tapering into
hill country or by running offshore as island chains.

**7c. A range has anatomy** — nested bands from outside in: foothill
apron → flank → high core → crest, each band's edge irregular down to
cell scale. Ranges are **asymmetric**: steeper on one side, with a broad
apron on the other. Flanks read visibly dissected while crests stay
cleaner.

**7d. Summits occupy area.** High country should be broad clusters of
near-peak terrain — a massif region, not a one-cell crest line with
everything falling away immediately. In the references, real area is
spent in the darkest elevation stops.

**7e. Plateaus are first-class**, and the author added a reference image
specifically so plateaus would not be lost. Two kinds:
- **Rim-enclosed plateaus** — a flat high floor ringed by crests, calm
  in the interior, small lakes on the floor, dissection concentrated at
  the rims.
- **Tabular uplands** — vast flat highlands ending in escarpments whose
  edges are gnawed by canyon heads.

**7f. Interiors must not be a bullseye.** A green coastal ring around a
uniform tan dome is a rejected look. The references put green lowlands
deep inland and treat high tan ground as *localized provinces*.
Interiors should carry variety: isolated massifs standing alone in
plains, stepped patchworks of distinct blocks, broad basins, worn
shields.

**7g. Lowlands are worked, not blank.** Fine mottling at cell scale
everywhere; faint incised valley networks converging tree-wise
downstream; lake chains along floodplains; and a distinct dissected
hill-country register that sits *between* flat plains and mountains.

**7h. Rivers should read at map scale.** Lowlands in the references are
threaded with fine dark dendritic lines. Incision that only exists at
extreme zoom does not satisfy this.

**7i. Peak heights stay physically plausible.** Typical high ranges top
out in the low thousands of metres; a few summits may go higher; nothing
should approach absurdity. Very high terrain is *rare and clustered*,
never a general condition of the map.

## 8. Composition and scale

- **Few, large landmasses with long continuous mountain systems.** When
  a batch was generated with many small structural domains, the author's
  verdict was that it "looks a hectic mess": short broken range strokes
  and scattered fragments. Reference frames read as roughly two to four
  large structural domains, with mountain systems running a long way
  unbroken.
- **Water dominates.** Land occupies a minority of the map — roughly a
  third is a good center, with author control over the balance.
- **Highland/lowland balance is deliberately NOT fixed by the reference
  images.** How much of the land is mountainous versus flat stays an
  author knob. Do not over-fit the references on this axis; do over-fit
  them on *formation* — the shapes, anatomy, and textures above.

## 9. Inland water

**Liked, keep:**
- Small, irregular lakes, often in chains. One reference image in
  particular was singled out for its natural-looking mountain water.
- Drowned shallows, fragmented coastline, and flooded lowland water that
  joins the ocean — the author called these out approvingly and they
  should survive any change.

**Rejected:**
- **Mountain mega-lakes.** On review the author wrote that large inland
  lakes near mountains "look… odd… many of them, and they're
  *massive*." Many huge smooth lakes is a defect; a few small irregular
  ones is the target.
- **Straight uniform troughs.** A long rift valley rendering as a
  perfectly straight, uniform-width, abyssally deep line was described
  as "the tectonic abyss that looks like a laser from space." Rift
  valleys should break into offset segments of differing depth and
  width, with shallower ground between them, reading as a chain rather
  than a slot.
- **Lake speckle.** Scattered cyan confetti across the map. The
  references show *rare structural* lakes. Tiny puddles everywhere are
  noise.
- **Flooded valleys rendering abyssally deep.** Where a valley or rift
  takes on water, it should read shallow with a deeper axis, not as a
  slice of open ocean.

## 10. Coastlines

- **Intricate and varied.** Coast character should change along the map:
  some stretches bold and smooth, others heavily indented and broken.
  Uniform coastal character across a whole map is a defect.
- **The shallow platform is drowned landscape.** It should carry banks,
  drowned valleys, island fields, and the land's own texture — flooded
  terrain rather than featureless blue.
- **Estuarine notching.** Drowned valley mouths cut into the coast.
- The shelf break reads as one crisp edge at a consistent depth.

## 11. The process-footprint principle

The author's foundational standard, held as a standing rule and applied
at every review. It governs the rest of this document.

> **Every feature in the map must be the footprint of a natural process
> the generator actually models — never a shape placed because it looks
> right.**

Paint the cause; let the render reveal the consequence. A mountain
range, a basin, a lake chain, an island group, a broad shelf: each
should appear because the physical conditions that produce it are
present at that spot, and should be absent where they are not. Nothing
is drawn *at* a location in order to satisfy an aesthetic goal.

What this rules out:

- Placing a feature because the map "needs one there," or because it
  balances the composition.
- Producing a look by drawing its symptoms — the recognizable shape of
  a landform stamped, tiled, mirrored, or repeated into place.
- Reaching for an appearance-level fix when the output looks wrong.
  The correction belongs in whatever produced the wrong result, not in
  a post-hoc adjustment that covers it up. This is the author's most
  frequently restated preference: improvements should come out of the
  modelled processes rather than out of corrective patches applied to
  their output. A fix that merely suppresses a symptom will be
  identified as such and rejected.

**Texture is parameterization, not decoration.** Sub-grid noise is
legitimate and expected — no generator resolves every scale — but its
amplitude and character must be modulated by local conditions, so the
texture *belongs* to the terrain carrying it. Uniform jitter sprinkled
over a finished surface is decoration, and reads as decoration.

**How you model those processes is deliberately open.** This principle
constrains where features come from, not what machinery produces them.
The only hard limit on that machinery is the performance budget (§15).

### 11a. The naturalness bar (how work gets rejected)

The visible symptom of a violated principle: **nothing may read as
drawn, stamped, tiled, or placed for appearance.** Regularity is the
tell. If a viewer can infer a spacing, a radius, a repeated stamp, or
an alignment to the image frame, the feature is wrong no matter how
pretty it is.

Specific patterns that have been rejected on sight:
- **Concentric rings** around islands. This is the canonical failure and
  gets referred to by name; it is the standard against which later work
  is checked.
- **Evenly spaced, even-width parallel bands.** Caught by the author in
  a difference image, where the give-away was "the fairly even
  distribution and widths."
- **Straight uniform troughs** (§9).
- **Frame-correlated coastlines** (§3b).
- **Floating islands** (§6e).

The last of these is instructive about the standard: the objection was
not that the islands looked bad in isolation, but that the seafloor
around them carried no trace of them existing. A feature whose
surroundings are unaffected by it has not been produced by a process.

## 12. Reference images (`examples/`)

`examples/ref1.png` – `ref14.png` are **author-blessed excellent
outputs** from a separate, external globe-generating program. They are
positive references and may be studied freely and often.

- **All are hypsometric views** — colour is a direct function of
  elevation through an elevation ramp. Early reviews mis-read lowland
  mottling in them as vegetation; it is terrain texture.
- **One image was added specifically so that plateau features would not
  be trained away** (§7e). Treat plateaus as a protected feature class.

**Three artifacts of the source program are explicitly NOT goals** —
do not reproduce them:
1. **Globe distortion** — the references are projections of a sphere;
   ours is a bounded flat region. Ignore the stretching.
2. **Day/night shading.** Several references are darkened by simulated
   sun position. That darkness is not the palette.
3. **Land touching the frame.** The references let land run off the
   edge. Our water border (§3) stands regardless.

**What the author values in them** (formation-focused):
1. Range anatomy — nested ragged bands, asymmetry, width that breathes
   along strike, clumped massif crests, dissected flanks, terminations
   that taper or run out to sea as island chains.
2. Plateaus as a first-class feature, both rim-enclosed and tabular.
3. Interior interest — lone massifs in plains, stepped block
   patchworks, real inland variety.
4. Worked lowlands — mottling, faint valley webs, floodplain lake
   chains, a hill-country register between plains and mountains.
5. Coasts and shelves that read as drowned landscape, with shelf
   breadth swinging from extreme to razor-thin depending on the coast.
6. Bathymetric drama that is linear and sparse — crisp narrow trenches,
   distinct sheltered-basin depths, a calm sedimented abyss.
7. **The ramp carries half the look.** Colour stops are dense near sea
   level on *both* sides, compressed through the middle elevations, with
   dark summits and only sparse pale caps at the very top.
8. **A chunky, stepped, banded look rather than an airbrushed one.**
   The references read visibly quantized into stepped bands, and the
   author has since confirmed a preference for that stepped look.
   Smooth gradient shading was a named deficiency in earlier work; the
   banding is not an artifact to be smoothed away. Banded anatomy comes
   from terrain crossing thresholds *noisily* against a ramp with
   visible stops.

## 13. Controls

- A control is **data**: name, type, range, default, tier, promise. The
  preview UI is generated from that data — see `webui/README.md` at the
  repo root for the interface contract and the expected `hypsometric`
  base view.
- **Aesthetic decisions become controls**, with a stated range and a
  stated promise, rather than hard-coded constants.
- **A promise must hold across the entire stated range**, including at
  both ends.
- **Promises are worded in process terms** — rates, physical
  magnitudes, what the thing *does* — rather than in terms of the
  appearance they are expected to produce (§11).
- Every feature worth having is worth being able to turn **off**, so any
  feature can be shown on/off against the same seed.

## 14. Review

- Review is by **image-batch gallery** — multiple seeds, multiple sizes,
  and same-seed variant comparisons. A single hand-picked example is
  never evidence.
- Contact sheets are first-class output, not a debugging convenience.
- Any layer or field the generator computes should have a view that can
  be rendered and looked at. Work that cannot be seen cannot be judged.

## 15. Performance

Interactivity is a feature: the control panel must feel alive.

| Resolution | Target |
|---|---|
| 256² preview | < 1 s |
| 512² | < 3 s |
| 1024² | < 15 s |
| 2048² final | < 90 s |

Rendering an already-generated result must be cheap — milliseconds —
so that changing a purely visual setting never waits on regeneration.
