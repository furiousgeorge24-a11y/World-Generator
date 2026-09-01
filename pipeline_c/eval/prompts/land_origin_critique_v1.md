# Land-origin diagnostic critique — land_origin_critique_v1

Review each `panel_XX.png` independently. Each panel is a neutral view intended
to expose the origin and organization of land. Critique visible formation
quality, not palette taste and not whether the composition matches a particular
reference.

For every panel return exactly three buckets:

- `done_poorly`: at most 5 prioritized claims;
- `done_well`: at most 5 evidence-supported claims;
- `cannot_identify`: at most 3 honest uncertainties.

Every claim must name `what`, a specific `where`, and visible `evidence`.
Praise has the same evidence burden as criticism. If the view cannot
distinguish a legitimate process outcome from an artifact, use
`cannot_identify` instead of inventing a cause.

Tag each `done_poorly` claim with one severity:

- `A`: suspected construction artifact or unsupported regularity—repeated
  same-scale bodies, tiling, seams, repeated radii, systematic grid locking,
  or apparent post-formation cuts. This is a visual hypothesis requiring a
  separate causal audit, not a veto by itself.
- `B`: formation implausibility—land bodies lack coherent interiors or
  relationships, broad continents collapse into lace/web/ribbon forms, or
  fragmentation visibly ignores surrounding organization.
- `C`: character or quality—blandness, weak multiscale structure, or limited
  variety.
- `D`: presentation obstructs interpretation. This is not a formation defect.

Do not infer an artificial edge treatment merely because land approaches the
frame or a coast is straight, turns near, or parallels it. Naturally formed
geometry remains valid; a frame-caused mechanism must be established outside
this visual review. Do not use raw island count as a quality rule. Pixel-scale
stair steps are rasterization unless a cited pattern persists at larger scale.

Return only one JSON array, with exactly one object per panel in panel-number
order. Do not add Markdown or prose outside the JSON. This is a literal valid
example:

```json
[
  {
    "panel": 1,
    "done_poorly": [
      {
        "what": "three repeated rounded major bodies",
        "where": "northwest, center, and southeast",
        "evidence": "All three have similar diameter and the same two-lobed outline at a scale far above individual pixels.",
        "severity": "A"
      }
    ],
    "done_well": [
      {
        "what": "coherent dominant landmass",
        "where": "western half",
        "evidence": "A broad interior connects several differently shaped coastal provinces without uniform-width bridges."
      }
    ],
    "cannot_identify": [
      {
        "what": "narrow eastern separation",
        "where": "east-center coast",
        "evidence": "The view alone cannot distinguish a naturally flooded passage from a post-formation cut."
      }
    ]
  }
]
```

Consult only the supplied judge packet.
