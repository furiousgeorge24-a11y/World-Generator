# Land-control sweep review — land_controls_sweep_v1

Each `panel_XX.png` is one same-family matrix. Rows are labeled requested
`target_land_percent`; columns are labeled `landmass_fragmentation`. All cells must use
the same seed, stable latent randomness, non-tested controls, and delivered
window. Judge authorability and perceptual continuity—not numeric conformance,
which is checked from masks outside this review.

Assess exactly four dimensions:

- `target_continuity`: increasing requested land should reveal or grow a
  recognizably related geography rather than rerolling the world.
- `fragmentation_response`: at 0 there should be a strong tendency toward one
  dominant broad landmass, while small coastal, barrier, volcanic, and other
  secondary islands remain acceptable. Increasing fragmentation should
  reorganize major bodies gradually; it promises no island count.
- `land_amount_leakage`: fragmentation should primarily change organization,
  not visibly act as a second land-quantity control. Numeric land fractions
  remain authoritative.
- `naturalness`: changes should look like consequences of formation rather
  than finished-mask channel cutting, deletion, stamping, tiling, or other
  appearance-level editing.

For each dimension choose `supports`, `concern`, or `cannot_assess`, and cite a
specific matrix location plus visible evidence. A concern is an investigation
lead, not proof of mechanism. A straight, frame-parallel, or frame-near coast
is not evidence of a hack by itself; confirmation requires a causal trace
outside this review. At very low land amounts, drowning may naturally separate
remnants even at fragmentation 0. At zero realized land, fragmentation is not
visually assessable.

Return only one JSON array, with exactly one object per panel in panel-number
order. Do not add Markdown or prose outside the JSON. This is a literal valid
example:

```json
[
  {
    "panel": 1,
    "target_continuity": {
      "assessment": "supports",
      "where": "fragmentation 0.5 column from target 20 through 60",
      "evidence": "The same western and southern bodies persist and expand while smaller connections emerge progressively."
    },
    "fragmentation_response": {
      "assessment": "concern",
      "where": "target 50 row between fragmentation 0.5 and 1.0",
      "evidence": "Several separations appear at similar width simultaneously, which could indicate a common cutting operation."
    },
    "land_amount_leakage": {
      "assessment": "cannot_assess",
      "where": "target 30 row",
      "evidence": "The visual areas look close, but the panel is not a reliable substitute for the recorded mask fractions."
    },
    "naturalness": {
      "assessment": "supports",
      "where": "target 60 row across all fragmentation columns",
      "evidence": "Major interiors stay broad and coastlines vary across scales without repeated stamped outlines."
    }
  }
]
```

Consult only the supplied judge packet.
