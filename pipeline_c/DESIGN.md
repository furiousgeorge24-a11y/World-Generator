# Design note — rebuilding land origin from C03

Written 2026-09-01, before any code, after C02 was judged unfit to build on.
[`CONTRACT.md`](CONTRACT.md) stays normative and is unchanged by this note.
[`AUTHOR_RULINGS.md`](AUTHOR_RULINGS.md) stays authoritative on taste. This
note decides the things the contract deliberately leaves open: the parent
domain, the scales, and the mechanism. Each decision says what it expects to
produce, so the first run can be judged against a prediction rather than a
hope.

Decisions marked **ratify** need the author's yes before code is written
against them. Everything else is a working default the builder may change,
one at a time, with the change and its expected effect stated first.

## 0. What went wrong, in one paragraph

C01 partitioned the world by distance to points and got a honeycomb. C02
partitioned it by shortest path on a four-neighbour lattice and got diamonds,
octagons, and axis-aligned bands. Neither was a process; both were a
partition dressed as one, and the discretization imprinted itself on every
contact. The lesson is not "be more physical". It is: **nothing visible may be
the direct output of a partition, a graph search, a nearest-point rule, or a
closed-form field.** Visible structure must be the residue of a history in
which things moved, thickened, broke, and were consumed. The discretization
must be one that does not draw its own stencil.

## 1. Parent domain — **ratify**

**Decision: stay on the flat torus.** Do not build a sphere.

Reasoning. The honest objection to a torus is that a rigid plate on it can
only translate, so relative motion is constant along an entire boundary and
kinematics alone yields neither curvature nor along-strike regime change.
That objection holds only if plates are rigid bodies with prescribed
velocities. In this design they are not: plates are *emergent* regions of
coherent motion in a deformable lithosphere, and boundaries curve because the
driving field curves and because they inherit old weakness. Trench curvature
on Earth is mostly slab dynamics and inheritance, not Euler geometry, so the
plane loses less than it appears to.

What the sphere would cost: a new discretization, advection on it, a
projection for delivery with visible scale distortion across a
continent-sized window, and months before the first geology. What the torus
keeps: the sampler, boundary-neutrality, windows anywhere, no seams.

**Predeclared tripwire.** If, after the C6 run below, the boundary view across
twelve seeds shows boundaries that are straight, uniform in regime along their
length, or all of one curvature sign, the sphere question reopens. It is not
reopened for any lesser reason.

**Marked for possible future revision** (author, 2026-09-01). The torus is
accepted for now, not permanently. A sphere, or a bounded plane with a
formation-neutral margin, may replace it later. Everything above the domain
(sampler, history rules, vertical stage, selector) is written so that the
domain is one module with a wrap rule, and nothing else knows it is periodic.

**Rendering.** The WebUI shows parent fields as a plain 2D image with the
understanding that the edges wrap. One additional audit view tiles the parent
2 × 2 at half scale so the wrap point sits in the middle of the image, which
is the "view over two periods" that `VIEWS.md` asks for and the only place a
seam could not hide. No pan-with-wrap viewer is planned; it would need changes
to the shared shell for no gain over the tile.

## 2. Scales — **ratify**

**Scale is authorable and never changes with resolution; the simulated
planet is sized to the map** (author rulings, 2026-09-01). Scale is
kilometres per delivered pixel, a slider in the WebUI, range 5–20, default 5.
Resolution sets how many pixels are delivered. Together they fix the window's
physical extent, and the parent world is sized from the window. A lower
resolution at the same scale is therefore a *smaller planet*, with features
the same size on screen, not a coarser view of the same planet.

| Quantity | Rule | At 1024 px, 5 km/px |
|---|---|---|
| Scale | authorable, 5–20 km/px, default 5 | 5 km/px |
| Delivered resolution | 128–2048 px per axis, default 1024 | 1024 px |
| Window | `pixels × scale` | 5,120 km |
| Parent world | `2 × window` per side, floored at 5,120 km | 10,240 km square torus |
| Kinematic cell | `8 × scale` km, i.e. eight delivered pixels | 40 km |
| Kinematic grid | `parent / cell` = `pixels / 4` per side (128² at the floor) | 256 × 256 |
| Velocity solve grid | half the kinematic grid per side; velocity is smooth by construction and is interpolated back | 128 × 128 |
| History duration | ~300 Myr in 75 steps of 4 Myr | 75 steps |

*Revised 2026-09-01 after the first C03 build.* The first build used a
4-pixel cell and 150 steps and cost 110 s per 1024 px world, 92 % of it in
the velocity solve. The kinematic cell was coarsened to 8 pixels, the
velocity is solved on a further-halved grid, and the step was doubled with
an exact damage integrator. Boundary zones now resolve at 40–80 km at the
default scale, against real boundary zones of 50–200 km. The crust stage
rides on markers and is not bound to this grid.

**What follows from the rules.**

- **Cost depends on resolution alone.** The history grid is half the
  delivered resolution per side at every scale, so a 1024 px map costs the
  same at 5 km/px and at 20 km/px. Maps of 512 px and below sit on the
  5,120 km floor and all cost the same few seconds.
- **The floor exists for physics, not for previews.** A parent below about
  5,120 km is smaller than one real plate and the process degenerates. Tiny
  maps see a piece of a floor-sized world.
- **No preview parity.** The same seed at 512 px and at 1024 px is two
  different worlds, the second twice the size of the first. This is chosen,
  not incidental. World identity is seed, resolution, and scale together.
- **The physics stays in kilometres.** Plate speed, arc offset, thickness,
  subsidence, and hotspot spacing are physical constants. Only the grid
  follows scale. *Revised 2026-09-02 (C03.10):* this includes the mantle
  drive's coarsest wavelength and the strength noise's, which the first
  builds had as fractions of the parent — 5,120 km and 1,280 km at 1024 px,
  half that at 512 px. They are now constants in kilometres with those
  values, so a 512-px world is a smaller window on the same mantle rather
  than a world with mantle cells half the size. C03.9's three 512-px
  screen passes against none at 1024 px were the symptom. At 20 km/px the cell is 80 km, which still resolves
  boundaries and arcs at one to three cells, and the delivered map could not
  show finer anyway.
- **The top corner is allowed, not promised.** A 2048 px map at 20 km/px is
  a 40,960 km window on an 81,920 km world, twice Earth's circumference. A
  70 % target there needs a 30,000 km continent, which the history will not
  produce. Expect `NO_VALID_WINDOW` and report it.
- **Scale is world geometry, not a formation control.** It sits beside size
  in the registry and is held fixed, like the seed, by every same-family
  sweep of `target_land_percent` or `landmass_fragmentation`. It may be a
  slider; it may never be swept.

**Why twice the window and not more.** The ring problem in §6 is a scale
problem at every scale. At target 70 the window's land is one body covering
seven tenths of the window with water around it, so the world must contain
an isolated body at roughly window scale: a 500 km island for a 128 px map,
a 4,000 km continent for a 1024 px map. A parent of twice the window holds
four window-areas for the selector and cannot show a feature twice in one
window. A history that produces island families and unequal continents
supplies bodies at every scale; if some scale came out systematically empty,
the fix is at the process level, never by scaling the window to fit.

**Version constants.** The parent-to-window ratio, the floor, and the
cell-to-pixel ratio are version constants; changing any of them reforms
every world. Scale and resolution are inputs.

**Contract amendment required.** `CONTRACT.md` §2 currently says that
supported resolutions sample the same physical window and that resolution
never changes what the world is. Under these rulings resolution and scale
size the world. The same-family definition must list scale beside seed and
size. The contract is amended when the deletion and rewrite in §9 happen,
not silently here.

## 3. The history model

The world is a two-dimensional lithosphere on the torus. It has a driving
field, a strength field, a velocity field solved from the two, and crust that
is carried by the velocity and changed where the velocity converges or
diverges. Plates are not an input. They are what you see when you look at the
velocity field after strain has localized.

### 3.1 Fields on the history grid

| Field | Type | Role |
|---|---|---|
| drive `D` | vector | Basal traction from the mantle: the sum of a few slowly drifting upwelling and downwelling sources plus one low-frequency noise term. Noise is the sanctioned exception; its causal role is mantle heterogeneity and it is stated as such. |
| strength `S` | scalar in (0, 1] | Lithosphere integrity. Starts high with mild noise. Falls where strain rate is high (damage), recovers slowly (healing). Old boundaries persist as weak sutures. |
| velocity `v` | vector | Solved each step: `v` relaxes toward `D` through a variable-coefficient diffusion whose coefficient grows with `S`. Strong lithosphere homogenizes velocity inside it; weak zones let it jump. This is the viscous-sheet-with-weak-zones approximation and it is what makes plates emerge and stay coherent. |
| strain rate `ε̇` | scalar + regime | From `∇v`. Divergence, convergence, and shear classify boundary regime *locally*, so regime varies along a boundary as the drive field turns. |
| plate label `P` | categorical, **derived** | Connected regions of near-uniform velocity. A view, a diagnostic, and an input to nothing. |

The diffusion solve is a few dozen Jacobi sweeps on 512² per step, with a
nine-point stencil so the operator is as isotropic as a square grid allows.
Predeclared check: a rotated drive field must give a rotated result within
tolerance, and the layer audit must not read the velocity or plate view as
grid-locked.

### 3.2 Crust as markers, not as a raster

Crust is carried by Lagrangian markers, one to four per cell, each with
continental thickness `h_c`, oceanic age `a`, and provenance. Markers move
with `v` and are re-gridded each step for the vertical stage and the views.
Advecting a crust raster semi-Lagrangian for 150 steps smears every margin
into a gradient; markers keep edges sharp without any re-sharpening step,
which would otherwise be a pure-math fix. This is standard marker-in-cell and
costs about a million vectorized marker moves per step.

### 3.3 What happens at boundaries

All of these are local rules on the regridded state, applied where `ε̇` says
they apply. None places a shape.

- **Divergence, oceanic.** New markers with `a = 0`. Ridges spread
  symmetrically and therefore migrate at half the plate rate.
- **Divergence, continental.** `h_c` thins. Below a threshold the crust
  breaks, becomes oceanic, and leaves *rifted margins* with thinned
  continental crust on both sides, and sometimes a detached sliver between
  two rifts. A rift that stops before breaking leaves a thinned trough and a
  weak zone that later history may reactivate.
- **Convergence, oceanic on one side.** The older or the oceanic side
  subducts: its markers are consumed at the trench. Arc magmatism adds `h_c`
  on the overriding side at 150–300 km from the trench along the convergence
  normal. The strip between trench and arc, the forearc, is thinned by
  subduction erosion and held down by slab coupling, so it sits low. That is
  what makes a coast hug the toe of an Andean belt instead of drifting
  seaward of it across a plain. Trenches roll back at a rate that grows with
  the age of the subducting crust, which is what bends them into arcs.
- **Convergence, continental both sides.** `h_c` thickens (orogeny). The
  boundary weakens and persists as a suture. Continents weld.
- **Shear.** Damage without thickness change. Transform segments and offsets.
- **Hotspots.** Three to six points fixed in the mantle frame add small
  thickness pulses to whatever passes over them. Tracks on ocean floor,
  occasional isolated islands, and a thickness anomaly where a continent
  drifts across one. Deliberately minor.
- **Age.** Oceanic `a` increases every step. Old floor is deep and subducts
  preferentially.

### 3.4 Initial condition — **ratify**

**Decision: no initial continents and no initial plates.** The world starts
as oceanic lithosphere of uniform age with a mild noise in `S`, the drive
sources, and a handful of *cratonic seed points* of a few cells' radius with
thick continental crust. Everything visible at the end is what 300 Myr of
arc accretion, collision, rifting, and drift did to those seeds.

This is the honest version of "simulate a process": continents are
agglomerations with sutures inside them, active margins on one side and
passive ones on another, and subordinate fragments left over from rifting.
Their outlines were never drawn.

The risk is that continental crust accrues too slowly to reach the land
budget in the history length. The lever is arc-accretion rate and craton
count, both process parameters. If accretion cannot deliver 30–45 % of the
parent as continental crust in a reasonable run, the fallback is a short
*pre-history* run at coarse time steps, not an initial mask.

### 3.5 Randomness

Every stochastic input goes through the existing stateless SHA-256 sampler
keyed by world, stage, process, and physical address: drive sources, craton
seeds, hotspot positions, the `S` noise, marker jitter. Nothing draws from a
stream, so traversal order, chunking, and resolution cannot reroll anything.
The two author controls, when they arrive, consume no random input of their
own.

### 3.6 The seam formulation — **ratified 2026-09-02**

The viscous sheet of §3.1 forms boundaries by diffuse damage: every cell
whose strain exceeds a yield weakens, and plates are what is left between
weak zones. Four runs measured why that cannot pass the plate screen. The
solve length that makes a plate interior rigid is also the distance over
which strain spreads around a weak zone, so every cell within it fails and
a zone widens until it is that length across. Heterogeneity (C03.7), damage
by dissipated work (C03.8), fast damage with fast healing (the corner
search), and a finer solve grid (C03.9) each moved the width, and none
crossed the bound: at production size the frontier sits with two thirds of
its network wider than four cells and nothing passing, and the width-versus-
plate-count trade is measured as physical once the grid's share is removed.

The seam formulation keeps the solve and changes where damage is allowed to
happen. A cell is either **intact**, part of a piece, or a **seam**, one
cell wide by construction. Damage is confined to three places:

- **On a seam**, by the work the slip across it dissipates, the C03.8 law,
  healing at a fixed rate, so a seam that stops slipping seals and its
  neighbours rejoin one piece. That is merging.
- **At a seam's tip**, which advances one cell at a time into the intact
  cell whose would-be seam carries the greatest traction, when that
  traction exceeds the intact strength scaled by the crack's length, the
  Griffith rule. A tip that reaches another seam joins it. A tip that
  closes a loop has split a piece. Propagation speed is a physical rate in
  kilometres per million years, so a crack runs across a piece over
  several steps and the field it runs through is re-solved as it goes.
- **By nucleation**, a few cells per step at most, at the highest-stress
  intact cells not adjacent to an existing seam, when their stress exceeds
  the intact strength. Heterogeneity in the intact strength is where the
  strength noise now acts.

Nothing else weakens. A cell beside a long seam is never damaged however
high its strain, because it is neither a seam, a tip, nor a nucleation
site; its load is what the solve says a seam's neighbour carries, and an
open seam carries little, so it is small. That is the unloading the sheet
lacked, and it is what holds width at one cell. The stiffness, the velocity
solve, the advection of strength with the lithosphere, the views, the
metrics, and the search are unchanged; a seam is a cell with strength below
the weak threshold, which is what every view and metric already reads.

Pieces are not enforced rigid in the first build. They are stiff regions of
the same sheet, and how rigidly they move is what the stiffness dial sets
and the velocity view shows. If the first build's pieces deform too much to
count as plates, the follow-up replaces the velocity solve inside pieces
with a rigid-body solve, three unknowns per piece coupled through seam
tractions, and keeps the sheet solve only for the internal stress that
drives nucleation and tips. That is a second run, taken only if measured.

*Measured 2026-09-03, C04 – C04.4.* Five runs, each removing one thing.
The seam rules on the sheet's solve (C04, C04.1) held width at one cell
but meshed cracks in the drive band. Rigid pieces with the integrated
stress and seams on markers (C04.2) gave exact force balance, edge
fraction 1.0, and cost 1.1 times the sheet, and cracks inside a piece
could not slip, so they healed. Elastic slip from the internal-stress
solve (C04.3) made cracks persist and they still did not run. C04.4 found
the reason: **the stress concentration at a crack tip exists in the
solve only when the solve grid is the kinematic grid.** At the half-grid
solve a four-cell crack is two solve cells and concentrates nothing. On
the full grid the median stress ratio ahead of a tip goes from 0.54 to
3.9, cracks run to 230 – 1,206 cells, six of twelve seeds cut a piece of
655 cells or more, and a 40-cell probe passes plate count in 17.5 % of
worlds, from zero. The fracture toughness, made a dial in the same
order, changes nothing qualitative. Two consequences follow. The cut
pieces flicker, rejoining whenever one seam cell on the loop heals, so
the healing time for a plate boundary must be that of a suture, tens to
hundreds of Myr, not the sheet's corner value of 10. And the full-grid
solve at 1024 px costs 58 s a world in NumPy against 2.5 s at 512 px, so
the search runs at 512 px on the full grid, and production at 1024 px
needs either a 16-pixel kinematic cell (the same 128-cell solve) or the
Rust solver the roadmap already defers. That is a cost decision for the
author, not a physics one.

*Measured 2026-09-03, after C04.4.* Two more things. The eight-direction
tip rule locks crack paths to the lattice: on two seeds at 512 px, two
thirds of eight-cell chords along a crack lie within 7.5° of a lattice
angle, against a third for an isotropic process, and the lock is there at
one advance per step, so it is the eight directions and not the field held
between advances. Because seams already live on markers with continuous
positions, the tip's position is continuous too, and the direction can be
the principal axis of the stress rather than the nearest of eight; that is
C04.5. And on 9,324 search worlds at 512 px on the full grid, plate count
is the term that fails, on 90 % of worlds, while drift alone fails 7 %:
the pieces are cut and rejoin on the next step, and whether the loop cell
that lets them rejoin healed or was vacated by marker motion has not been
counted. C04.5 counts it before anything is changed for it.

*Measured 2026-09-03, C04.5.* The continuous tip direction moved the
count: plate count on the twelve went from one seed in the screen's band
to ten, the largest piece from 82 – 95 % of the world to 21 – 74 %, and
the lattice share of crack chords from 0.65 – 0.74 to 0.42 – 0.43. It also
opened four times as many cells, so the weak share ended at 0.17 – 0.28
and every seed still fails drift. The rejoin count answered the other
question: **99.7 % of the cells that let a cut piece rejoin were vacated,
not healed.** Two markers in one cell translate together, cross a cell
boundary, and round into two different cells, leaving the cell they
shared intact while the seam was slipping at twenty times the yield. The
marker raster is a set of points, and a set of points has holes. A seam
is a curve, so the next change is to carry it as one: markers linked in
order along each crack, and the raster drawn from the segments between
them, which cannot leave a hole. The tips that now spend 700 – 1,500
advances a run re-closing holes are the likely source of the excess
opening, so that change is predicted to bring the weak share and the drift
back down with the count kept.

## 4. Vertical response and first exposure (C8–C9)

At delivered resolution, history fields are interpolated to 10 km cells. That
interpolation is sampling, not smoothing, in the contract's sense: the world
is defined at history resolution and observed more finely.

- **Elevation** is Airy isostasy on crustal thickness plus thermal subsidence
  of ocean floor with age. Reference continental thickness 35 km sits near
  sea level; 65–70 km orogens stand 4–5 km high; rifted margins at 15–25 km
  sit 1–3 km *below* the water reference, which is where shelves and drowned
  platforms come from without any rule for them. Ocean floor deepens as the
  square root of age and flattens after ~70 Myr.
- **Relief** below 40 km is a noise term whose amplitude is set by recent
  strain, thickness gradient, and arc activity. Process-modulated noise is
  permitted by contract §7; its role is stated as sub-grid tectonic
  roughness. Quiet interiors get little, active margins get more. This is
  the only place noise touches elevation.
- **Water reference** is a single global sea level. The land mask is
  `elevation > sea level`, nothing else. Coasts are flooded structure by
  construction; no boundary is ever traced.

## 5. The two controls

- **`target_land_percent` (C11).** Sea level is the solved parameter. Realized
  land in the window is monotone non-increasing in sea level, so the solve is
  a scalar bracket that always converges. Same-seed target sweeps are
  non-decreasing by construction. The solve drives the modeled process (water
  exposure) and is reported in full.
- **`landmass_fragmentation` (C13).** Acts on the *history*, through two
  process parameters that share the same keyed randomness: continental
  lithosphere strength, which sets how readily a continent rifts, and drive
  vigour, which sets how far fragments disperse. At 0, strong continents and
  convergent drive assemble one macro-mass. At 1, weak continents rift and
  disperse. No component is ever counted, cut, or welded. Expected residual
  land change across the sweep is small because sea level re-solves per cell
  of the matrix; the report states it.

A rejected alternative worth recording: fragmentation as the *epoch* at which
the history is read out, since a supercontinent and its dispersed successors
are the same world at different times. It is natural, but it changes the
geology under a fixed window and makes the ring predicate epoch-dependent. Not
adopted.

`AUTHOR_RULINGS.md` says plate count is an internal versioned setting. In this
design plate count is emergent; the versioned settings are drive-source count
and lithosphere strength. The ruling's intent, that plate count never becomes
a hidden fragmentation synonym, is preserved.

## 6. Window selection and the water ring (C12)

The frame determines what is shown, never what forms. The selector observes
finished geography and picks once for the whole control family.

- **Candidates.** Window origins on a lattice of every 8 history cells
  (320 km) across the parent: 4,096 candidates on the torus. Order is a
  sampler-derived permutation of that lattice, declared and versioned.
- **Eligibility.** For a candidate, solve the sea level `s70` that gives
  70 % land inside it. The candidate is eligible if every cell of its outer
  ring has elevation below `s70`. Because sea level for every lower target is
  higher, a ring that is water at `s70` is water for the whole family. This
  is a deliverability predicate, permitted by contract §5; the selector never
  ranks target accuracy, resemblance, or appearance.
- **Choice.** First eligible in the declared order. None eligible reports
  `NO_VALID_WINDOW` with every candidate's observation. No retry, no seed
  change, no geography change.

**Expected behaviour.** A 70 % window with a water ring hugs a continent that
is nearly window-sized with basins around it. With Earth-like continents most
seeds should offer one. Some will not, and that failure rate is the number to
watch and freeze before validation. The ring is therefore mostly a statement
about continent scale, which is §2, not a border operation.

## 7. Views owed

Per [`VIEWS.md`](VIEWS.md), every field above gets a view before the author
sees the stage. The list, so none is forgotten:

drive (hue for direction, value for magnitude) · strength · velocity ·
strain rate (ramp and banded) · regime (categorical) · plate label
(categorical) and its **boundary set** · continental thickness (ramp and
banded) · oceanic age · hotspot input · marker density · elevation (ramp and
contoured) · relief noise on its own · land mask and its outline · window
candidates with eligibility.

And because the mechanism is a history, **time-lapse is a view**: the same
fields at several epochs, so the author can see plates assemble and break
rather than infer it from the end state.

## 8. Predictions to judge the first run against

Stated now so the run is checked against them, not rationalized after.

- Boundaries are long, curve with the drive field, change regime along their
  length, and offset across shear segments. Old sutures show inside
  continents as thickness and strength lines with no coast on them.
- Continents are unequal in size, have an active thickened side and a
  passive thinned side, and carry fragments and slivers near their rifted
  margins.
- **Andean margins are common, not universal.** Where a continent's leading
  edge overrides ocean, a segmented belt runs 100–300 km behind the trench
  with the coast on its seaward toe, a narrow low forearc, a broad shallow
  passive side opposite, and the belt continuing off both ends of the
  continent as island arcs. The author's reference `examples/` image of a
  long north–south continent with an eastern belt is this case. Expected on
  a substantial fraction of continents in a spread of seeds; absent from
  collisional interiors, rifted margins, and mid-ocean windows. The rate is
  emergent and is never a selector criterion.
- Islands appear as arcs offset from trenches, as rifted slivers, as hotspot
  chains, and as drowned shelf remnants at high sea level. Several families
  per seed, never one.
- Interiors are quiet. Detail concentrates at margins and belts.
- **For the seam formulation (C04), stated 2026-09-02.** Edge fraction is
  near one in every world, by construction. Weak fraction is a few percent,
  because a network of seams one cell wide over a 256-cell world is a few
  thousand cells. Plate count is a free outcome for the first time: cracks
  either close loops and cut pieces, or run into each other and stall as
  dead ends. The failure modes are crazing, many parallel cracks in the
  band where the drive's stress is highest, and dead-end networks that
  never split a piece. Both have a view, the plates sheet, and both are
  distinguishable by plate count and network share. Drift and peak ratio
  are unknowns; fast healing is expected to settle a network within the
  history.
- Failure modes to watch for: velocity that never localizes and gives
  diffuse convergence bands instead of boundaries; localization that follows
  the grid; markers clumping into lace; accretion too slow to build
  continents. Each is a process parameter or a stencil choice, and each has
  a view that would show it.

## 9. Build order, replacing the C4–C5 rows of the roadmap

| Run | Builds | Stops for |
|---|---|---|
| C03 | Domain, sampler wiring, history grid, drive, strength, velocity, damage. No crust. Plate label and boundary views over twelve seeds and several epochs. | Whether boundaries curve, segment, and change regime. The §1 tripwire. |
| C03.1–C03.10 | Performance, flux-consistent strain, the exploration lab, the regime search, the work law, the solve divisor, the drive in kilometres. Measured the sheet to its ceiling. | Done. The sheet does not pass the plate screen at production size; §3.6 records why. |
| C04 | The seam formulation of §3.6 behind a switch, production byte-identical: seam damage by slip work, tip propagation, nucleation, healing and merging. The corner search's dials and the existing views. | Whether cracks close loops and cut pieces, or craze or dead-end. Plate count and network share on the twelve seeds. |
| C04.1 | Rigid-body motion inside pieces, only if C04's pieces deform too much to count as plates. | Whether the velocity view shows bodies. |
| C04.2–C04.4 | Rigid pieces with the integrated internal stress, seams on markers, elastic slip inside a piece, the toughness dial, the full-grid solve at 512 px. | Done. Cracks run and cut pieces; §3.6 records the measured state. |
| C04.5 | A continuous tip direction from the principal stress axis, the tip's position carried by its marker; a count of what reconnects a cut piece, healed or vacated. | Done. Plate count in band on ten of twelve; lattice share 0.65 → 0.42; weak share and drift went up; rejoins are 99.7 % vacated cells, not healed. |
| C04.6 | The seam as a curve: markers linked in order along each crack, the raster drawn from the segments between them, so marker motion cannot leave a hole. | Whether the flicker stops and the weak share and drift come back down with the count kept. |
| C6 | Crust markers, ridges, subduction, arcs, collision, rifting, hotspots, age. Thickness and age views. | Whether continents agglomerate with believable margins and fragments. |
| C7–C10 | As in the roadmap, now mostly readouts of the history rather than new mechanisms. | Per roadmap. |
| C11–C13 | Sea level solve, selector and ring, fragmentation through strength and vigour. | Per roadmap. |

Each run carries a determinism test, its views, and the layer audit.
Nothing else: no certificates, cohort manifests, or content-hash records for
intermediate state. Those were built for a review pipeline that no longer
exists and should not return by habit.

## 10. Cost envelope

*Measured, not estimated, 2026-09-01.* The first C03 build, at a 4-pixel
cell (512² for 1024 px) and 150 steps, cost 110 s per 1024 px world, with
92 % in the multigrid-preconditioned velocity solve at about four cycles per
step. The original estimate of 10–25 s was wrong by a factor of four. The
author's target is under 5 s. The revisions in §2 (8-pixel cell, half-grid
solve, looser warm-started tolerance, 75 steps, exact damage integrator, fast
labelling) are expected to reach about 3–4 s at 1024 px and about 15 s at
2048 px, with cost still depending on resolution alone. Crust markers in C6
add a further per-step cost on top of this and will need their own budget.

*Measured again 2026-09-02.* The half-grid solve at 1024 px costs 2.3 to
4.4 s per world on the eight-worker pool. The full-grid solve at 1024 px
costs 95 s per world in NumPy, 22 times the half grid at four times the
cells, and is not a production or a search option; at 512 px it costs
2.5 s. The seam formulation adds raster operations on the order of the
seam count per step and is expected to stay within the half-grid budget.

## 11. Open

- Marker count per cell and regridding kernel. Start at two per cell.
- Drive-source count and drift rate. Start at four to six sources drifting a
  few hundred km per 100 Myr.
- Whether strength recovery should depend on age of the weak zone.
- Arc offset as a constant or as a function of subducting age.
- The exact craton seed count and thickness. Start at four to eight.

None of these are needed to start C03, which has no crust.
