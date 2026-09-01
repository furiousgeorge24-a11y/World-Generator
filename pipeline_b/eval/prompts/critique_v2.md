# Terrain formation critique

Review each `panel_XX.png` independently. Each is a terrain map whose
stepped colour ramp encodes elevation; palettes vary by source and
palette taste is not a terrain defect. Critique **formation quality**:
do the visible landforms read as plausible consequences of natural
processes?

For each panel give exactly three buckets:

- `done_poorly`: up to 5 claims;
- `done_well`: up to 5 claims;
- `cannot_identify`: up to 3 honest uncertainties.

Every entry in every bucket must include:

- `what`: a feature name, or a neutral visual description when it
  cannot be identified;
- `where`: a specific image location;
- `evidence`: the visible morphology supporting the defect, praise, or
  uncertainty.

Praise carries the same evidence burden as criticism. A named process
or an overall impression is not evidence by itself. For example,
"coherent orogen" needs visible support such as alignment, asymmetry,
relief gradation, or associated structures. If the evidence cannot
distinguish a source artifact from a formation defect, use
`cannot_identify` instead of inventing an explanation.

Tag every `done_poorly` claim with one severity:

- `A` — artifact or regularity: repeated identical marks, axis
  alignment, right angles, starbursts, rings or halos, even spacing or
  widths, straight uniform features, frame correlation, seams, dotted
  lines, or other processing artifacts.
- `B` — formation implausibility: features ignore their causes or
  surroundings, such as floating islands, rivers without drainage
  logic, lakes without context, unbroken escarpments, or uniform shelf
  width.
- `C` — character or quality: blandness, weak variety, missing anatomy.
- `D` — rendering or palette obstructs interpretation. This is not a
  terrain-formation defect and is excluded from formation scoring.

## Feature-scale evidence rules

- The maps are hypsometric. Grey-violet plateau fill is high elevation;
  green lowland mottling is terrain texture, not vegetation.
- Pixel-scale square crenellation and one-to-two-pixel shoreline
  strokes can be screenshot rasterization. Do not classify them A or B.
- Tiny bright plateau dots may be floor lakes. A single radial or
  star-shaped volcanic island or massif is not a stamp.
- Pale-core islets inside green shelf rings may be drowned-shelf hills,
  not bullseye stamps.
- For an A claim based on stamping, repetition, bullseyes, mottle,
  crenellation, or uniform width, state the repeat count and approximate
  feature size or spacing, or compare widths at two separated
  locations. Explain why the pattern persists above raster scale.
- Lowland texture must be judged at feature scale, not from crop-scale
  grain.
- Day/night darkening, globe-projection stretching, and land touching
  the frame are source artifacts, not terrain defects.

Return a JSON list with exactly one object per panel, in panel-number
order. Do not add prose or Markdown outside the JSON:

`{"panel": N, "done_poorly": [{"what": "...", "where": "...", "evidence": "...", "severity": "A|B|C|D"}], "done_well": [{"what": "...", "where": "...", "evidence": "..."}], "cannot_identify": [{"what": "...", "where": "...", "evidence": "..."}]}`

Do not consult anything outside the panels directory.
