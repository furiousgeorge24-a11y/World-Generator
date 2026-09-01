# Land-origin morphology discrimination — land_origin_2afc_v1

Each `trial_XX.png` contains two neutral land/water views, A on the left and B
on the right. At least one side comes from the frozen positive reference set;
the other may be another reference or a candidate. Judge which side has the
more plausibly natural **origin and organization of land**. Do not try to
identify a source, model, or implementation.

Use visible evidence about broad two-dimensional land bodies, coastline
variation across scales, relationships among major and secondary bodies,
natural necks and separations, and whether shapes look causally coherent rather
than stamped, tiled, painted, or cut after formation. There is no desired
island count. Small coastal, barrier, volcanic, and other secondary islands can
be entirely natural.

Palette, labels, antialiasing, and hue are not evidence. Exact candidate
outer-ring water and requested-land accuracy are deterministic checks outside
this trial. Do not reward or punish either side for proximity to the image
frame. A straight or frame-parallel coastline is not evidence of an artificial
fix by itself. Repetition or geometric regularity is usable only when you cite
multiple separated examples, their approximate scale, and a pattern that
persists above raster quantization.

Mark a trial `void` rather than guessing if either side lacks enough land/coast
to judge, is illegible, or is at such a different apparent physical scale that
the comparison is unfair. Bland but judgeable morphology is evidence, not an
automatic void.

For every valid trial, identify specific evidence on both A and B and explain
the comparison. Confidence 1 is an effectively tied forced choice; confidence
5 requires multiple independent feature-scale cues. If both sides look
reference-grade, still choose one at low confidence.

Return only one JSON array, with exactly one object per trial in trial-number
order. Do not add Markdown or prose outside the JSON. The following is a
literal valid example showing both response forms:

```json
[
  {
    "trial": 1,
    "void": false,
    "pick": "A",
    "confidence": 2,
    "evidence": {
      "A": "The west coast has several scales of embayment and two broad connected interiors.",
      "B": "The central bodies repeat a similar rounded outline at three separated locations.",
      "comparison": "A has more varied, mutually coherent major-land morphology, although the choice is close."
    },
    "void_reason": null
  },
  {
    "trial": 2,
    "void": true,
    "pick": null,
    "confidence": null,
    "evidence": null,
    "void_reason": "B contains too little visible coastline to assess at the presented scale."
  }
]
```

Consult only the supplied judge packet.
