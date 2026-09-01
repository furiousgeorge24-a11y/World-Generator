# Terrain map discrimination trials

Each `trial_XX.png` shows two terrain-map tiles, A (left) and B
(right). In every pair, at least one tile comes from a curated
reference set of terrain maps whose formation quality is considered
excellent. The other tile may come from the same reference set or from
a different map source.

Judge which tile is more plausibly reference-grade, based only on the
visible plausibility of the terrain's formation. Do not try to identify
the software or source that produced either tile.

## Validity

A valid trial must show enough visible formation in both tiles to
compare at feature scale. If either tile is predominantly near-black or
featureless, or if the apparent physical scales are so different that
formation cannot be compared fairly, mark the trial `void`; do not
guess. Do not mark a trial void merely because visible terrain is
bland: blandness can itself be quality evidence when enough terrain is
visible to assess it.

The tiles are intended to be compared at similar physical scale.
Apparent zoom, global extent, screenshot scaling, palette, or source
style is not evidence.

## Evidence and confidence

For every valid trial, name at least one specific visible feature and
location on each side, then explain why the selected side has the more
plausible natural formation. A process name or overall impression by
itself is not evidence.

Confidence 1 means an effectively tied forced choice. Confidence 5
requires multiple independent feature-scale diagnostic cues. If both
tiles look reference-grade, still pick one with low confidence.

## Feature-scale literacy

- The maps are hypsometric: colour encodes elevation. Palette or hue
  difference alone is not evidence. Grey-violet plateau fill and green
  lowland mottling are not materials or vegetation.
- One-to-two-pixel shoreline strokes and square boundary stair-steps
  can be screenshot rasterization rather than landform geometry.
- Tiny bright plateau dots may be floor lakes. A single radial or
  star-shaped volcanic island or massif is not evidence of stamping.
- Pale-core islets inside green shelf rings may be drowned-shelf hills,
  not bullseye stamps.
- A repetition or stamp claim is usable only when the evidence gives
  the repeat count, approximate motif size and spacing, and shows that
  the pattern persists above raster scale. A uniform-width claim must
  compare at least two separated locations and their visible widths.
- Lowland texture must likewise be assessed at feature scale, not from
  crop-scale grain.
- Some reference images carry day/night darkening, globe-projection
  stretching, or land running off the image edge. None is evidence.

Return a JSON list with exactly one object per trial, in trial-number
order. Do not add prose or Markdown outside the JSON.

Valid trial:

`{"trial": N, "void": false, "pick": "A"|"B", "confidence": 1-5, "evidence": "location-specific comparative evidence for both sides"}`

Invalid trial:

`{"trial": N, "void": true, "pick": null, "confidence": null, "evidence": "which side is unjudgeable or how scale is mismatched"}`

Do not consult anything outside the trials directory.
