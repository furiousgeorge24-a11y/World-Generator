# Single-image mechanism audit — layer_audit_v1

Each `panel_XX.png` is a raster visualization of one two-dimensional field —
either a continuous scalar field drawn on a fixed blue-to-warm ramp, or a class
field drawn with a fixed categorical palette. Colour carries no meaning beyond
ordering and identity. Judge each panel **entirely on its own**. Panels are not
variants of each other, and nothing outside the supplied packet is relevant.

Do not assess whether a panel is attractive, well composed, or realistic. The
only subject is **what kind of thing produced it**.

## What to answer for each panel

**1. The generating rule.** State the rule that would reproduce this image.
Write it as you would to someone who must implement it. Then set `closure`:

- `closed` — a short exact rule reproduces the image, up to constants you can
  read off it. Nothing essential is left over.
- `partial` — a short rule captures the dominant structure, but real features
  are unaccounted for.
- `open` — no compact rule reproduces it. An honest description needs
  exceptions, local qualifications, or several scales.

Judge closure by whether your own description is finishable, not by how
complicated the image looks. Busy images can be closed; plain ones can be open.

**2. The off-frame prediction.** Suppose the field continued past the **right
edge** of the panel. Set `predictable` true only if you can say what is there.
If true, describe it in `prediction`, and give `period_px` and
`orientation_deg` when a repeat exists — otherwise leave those null. If false,
`prediction`, `period_px` and `orientation_deg` are all null.

Do not hedge. This answer is checked against the actual adjacent region.

**3. The mechanism.** Choose exactly one label:

- `periodic_waves` — superimposed periodic functions; a repeating motif.
- `distance_or_cost_field` — value grows with distance or accumulated cost
  from points or lines; nested equidistant contours.
- `filtered_noise` — random values correlated over a length scale; fractal or
  multi-octave character, no repeat.
- `iterative_growth` — regions advanced from origins and met; contact
  boundaries, unequal extents, engulfed remnants.
- `thresholded_field` — a continuous field cut into levels or classes; the
  banding follows an underlying smooth field.
- `cannot_determine` — the panel does not settle it.

Give the visible evidence and a confidence from 1 to 5.

**4. Regularities.** At most five, each naming `what`, a specific `where`, and
visible `evidence`, tagged with one `kind`: `grid_locking`, `periodicity`,
`constant_scale`, `constant_curvature`, `symmetry`, `straight_runs`,
`degenerate_morphology`, `seams`. Report only regularities you can point to.
An empty list is a valid and meaningful answer.

**5. The verdict.**

- `formula` — a closed-form rule evaluated per position, or a simple geometric
  construction such as nearest-of-several-points, accounts for essentially the
  whole image.
- `process` — it does not. Structure remains that no compact rule reproduces.
- `undecided` — genuinely cannot be settled from this panel.

**Random and noise-derived fields are `process`, not `formula`.** No short rule
writes down their values. `formula` is reserved for images a compact expression
reproduces exactly.

## Cautions

Irregularity is not automatically process, and regularity is not automatically
formula. A closed form can look busy; a process can produce a locally straight
run, a round body, or a smooth gradient. Decide on whether the structure is
*reproducible by a compact rule*, not on how orderly it appears.

Single-pixel stair-stepping on a diagonal edge is rasterization. Do not report
it unless the pattern persists at a scale far above individual pixels.

Report `cannot_determine` and `undecided` when they are true. An honest
non-answer is more useful than a confident guess.

## Response format

Return only one JSON array, exactly one object per panel, in ascending panel
order. No Markdown and no prose outside the JSON. This is a literal valid
example:

```json
[
  {
    "panel": 1,
    "verdict": "formula",
    "generating_rule": {
      "description": "Two sawtooth waves summed. Each takes a linear combination of the x and y coordinate, divides by a wavelength of roughly 120 pixels, and folds the fractional part into a triangle. One runs near 45 degrees, the other near 20 degrees.",
      "closure": "closed"
    },
    "off_frame_prediction": {
      "predictable": true,
      "prediction": "The same diagonal stripe set continues unchanged, with the bright ridge nearest the right edge repeating about 120 pixels further right.",
      "period_px": 120,
      "orientation_deg": 45.0
    },
    "mechanism": {
      "label": "periodic_waves",
      "confidence": 5,
      "evidence": "Bright ridges are straight across the full panel, evenly spaced, and cross at two constant angles to form identical diamonds everywhere."
    },
    "regularities": [
      {
        "what": "identical diamond cells across the whole field",
        "where": "everywhere, including all four corners",
        "evidence": "Every cell has the same two edge lengths and the same interior brightness profile; no cell differs from its neighbours.",
        "kind": "periodicity"
      }
    ]
  }
]
```

Consult only the supplied judge packet.
