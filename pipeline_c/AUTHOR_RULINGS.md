# Author rulings

What the author wants, preserved so it does not depend on conversation
history. [`CONTRACT.md`](CONTRACT.md) is normative for conformance; this file
records intent and taste, which the contract cannot fully express.

## The look being built toward

A strong hierarchy of scale:

- a **low number of world-scale primary actors**, with broad, comparatively
  quiet interiors — not a dense mosaic of equal cells;
- **long boundary systems** that are coherent at world scale, curve and change
  direction, segment, and carry local raggedness — not straight polygon edges,
  and not uncorrelated noise;
- major landmasses, basins, shelves, margins, peninsulas, detached blocks, and
  island families that **share visible structural causes**;
- **island variety**: boundary arcs, broken continuations of margins or belts,
  detached fragments, drowned platform remnants, pieces of major landmasses,
  and occasional isolated volcanic or structural origins;
- **substantial quiet areas**, so detail is concentrated rather than spread
  evenly.

These are review obligations, not counts. No exact plate, island, or component
number is promised. A quantitative criterion may become a gate only after it
is predeclared and calibrated.

Two attempts have now failed against this description while passing all their
numeric gates. Regularity is the recurring enemy: repeated blobs, equant
cells, parallel bands, rulers, common radii, lace, and grid-axis preference
are the specific tripwires to watch for.

## Boundaries and coastlines are related, not identical

Tectonic boundaries and inherited margins should often exert first-order
influence on where coasts fall. This is statistical and causal, not a tracing
rule.

A boundary may be offshore, submerged beneath an arc, inland behind a coast,
weakly expressed, inactive, or preserved as an interior suture. A coast may
instead be passive, or controlled by flooding of broad structure. The
shoreline must emerge from continuous vertical state and water exposure; it
may never be copied from the boundary network.

## Reference images

`examples/ref1.png` through `ref14.png` are the perceptual canon — outputs
from a separate program that the author rated as good.

Take from them: macro-scale hierarchy, coherent large and intermediate
structure, long curving segmented organization, island-family variety, and
uneven regional expression — active-looking margins beside quiet ones.

Ignore: their finished erosion, drainage, relief rendering, and cartography,
all of which are downstream of this module; and their land touching the image
frame, which this module must never do. Resemblance to a reference is not by
itself evidence of natural causation.

## Frame and border

Restated in full as [`CONTRACT.md`](CONTRACT.md) §6. The one-sentence version:

> The delivered frame may determine what already-formed geography is shown; it
> may not determine what geography forms.

A naturally formed shoreline may parallel, crowd, turn with, or perfectly
contour the rectangle. Visual alignment is not a defect. Frame-aware clipping,
flooding, tapering, masking, moats, or repair are.

## Accepted defaults

- `target_land_percent` — range `0`–`70`, default **`50`**, per-map tolerance
  ±10 percentage points.
- `landmass_fragmentation` — range `0`–`1`, default **`0.5`**, no body-count
  promise.
- Delivered maps are square via the existing `size` interface, default
  **`1024 × 1024`**, at an authorable scale of `5`–`20` km per pixel,
  default **`5`**. Scale never changes with resolution; a lower resolution
  is a smaller world, not smaller features. Internal geometry carries width
  and height independently so rectangles remain possible later.
- Plate count is emergent, not a setting. The internal versioned settings
  are the drive field's wavelength and the lithosphere strength constants.
  None of them may become a hidden synonym for fragmentation.
- NumPy and Pillow are acceptable dependencies. A heavier scientific stack
  needs a specific mechanism to justify it.
- Same-seed control variants reuse keyed latent randomness and one window; a
  control change may not reroll unrelated structure.

## Failure policy

Honest generation failure is permitted during development. If no valid window
or conforming causal solve exists, the attempt fails with its evidence intact
— it may not repair geography, substitute a seed, silently retry, or emit a
failed candidate as a conforming map.

The long-term objective is to make failures very rare or eliminate them. The
acceptable rate must be frozen before final validation, never chosen after
seeing results.

## Who judges

The author decides whether the morphology is right, by looking at it. Passing
checks is not the same as being good: two mechanisms have now passed every
threshold declared for them and were still wrong on sight.

Whatever a stage produces must be visible in the WebUI as the fields that
genuinely exist — never a placeholder standing in for something unbuilt. A
stage looking acceptable says nothing about the stages after it, about
milestone conformance, or about generator readiness.

## Isolation

Pipeline C is developed from its own documentation, code, evidence, and author
rulings, plus the shared root WebUI shell. The reference images are external
perceptual evidence only.

Pipeline A has never been authorized for consultation. One narrowly scoped,
read-only Pipeline B consultation was authorized on 2026-09-01 for a
high-level inventory of systems that can precede land origin; it imported no
algorithm, constant, or result, and that exception is now closed. Future work
may not open, import from, or consult `pipeline_a` or `pipeline_b`.
