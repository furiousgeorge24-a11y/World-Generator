# Contract — v0.1 (draft until author blesses)

The promises. Everything here must stay true of every delivered map. How
these are achieved lives in `design.md`; when they arrive lives in
`milestones.md`. This file supersedes `testbed_handoff.md` (removed; its
durable content is absorbed here and in `CLAUDE.md`).

## 1. Scope and deliverable

- The generator produces **terrain form**: elevation, water, and derived
  fields (climate, wetness, relief). It does not produce settlements,
  regions, routes, names, or history.
- **This project ships ingredients; the target system cooks.** Biome/terrain
  classification (swamp, savanna, hills, …) is explicitly out of scope. We
  emit the data layers the target editor needs to make those calls itself.
- The user-facing deliverable is a **PNG** (plus report sidecar). Hex export
  is a downstream projection, never an influence on generation.

## 2. Core object and API shape

```
generate(seed, controls, size) -> World      # expensive
render(World, style)          -> PNG         # cheap, milliseconds
hexify(World, cols, rows)     -> per-hex field aggregates
```

A `World` is a stack of named, same-shape 2D arrays plus scalar metadata.
The generate/render split is load-bearing: it is what makes a slider UI
feel alive.

## 3. Sizes and grid

- Square lattice, simulation-native. Rectangular maps allowed: each axis
  independently **128–2048** cells.
- Hex export dimensions must **never exceed** field resolution
  (hexification is always a downsample).
- Edges are bounded; no wrapping.

## 4. Units

Physical: elevation in **metres**, sea level = 0. Horizontal scale is the
author control `cell_size_km`. What a map *means* (region vs. continent vs.
world) is a knob, not an ambiguity.

## 5. Determinism

- Same seed + same controls + same version → **bit-identical** output on the
  same environment; **structurally identical** across environments (float
  variance across BLAS/OS is acknowledged, not promised away).
- RNG is keyed per stage: changing one control never reshuffles the random
  draws of unrelated stages. **Dragging the erosion slider does not move
  your continents.**
- Prohibited: wall-clock input, unordered iteration feeding results,
  unordered parallel reductions.

## 6. Structural resolution independence

Same seed + controls at different resolutions yield **the same world** —
same continents, ranges, coastline shape — with finer detail at higher
resolution. Fine erosion filigree may differ; large structure may not.
Preview-then-final is a supported workflow, not a hope.

## 7. Border invariant

- The **outermost ring of cells is water** on every delivered map. No land
  touches the frame.
- Achieved **by construction** (border-avoidance is a property of causes:
  where land-building processes may act), never by post-hoc masking —
  redrawn borders produce frame-correlated coastlines and are prohibited.
- Every run reports nearest-land-to-border distance (regression insurance).
- Border **hexes** are water in any export; the guaranteed water band is
  sized to survive the downsample.
- Implementation freedom: the engine may generate on a padded domain
  (sacrificial apron) and crop. All guarantees are stated about the
  delivered map. Apron size, if used, is deterministic and versioned.

## 8. Robustness and reporting

- **Generation never fails.** Every in-range control combination produces a
  map. Internal findings (instability clamps, anomalies, invariant checks)
  ship in the report sidecar alongside the map — they never destroy a run.
- Report sidecar per run: seed, full control echo, per-stage timings,
  findings (land fraction, elevation range, lake count, roughness above vs.
  below the shelf break, nearest-land-to-border, …).
- Every PNG embeds provenance (seed, controls, version): any gallery image
  can be regenerated exactly.

## 9. Controls

- A control is **data**, not a function argument: name, type, range,
  default, stage, promise, invalidation class, tier. UIs are generated from
  the registry.
- Promises hold across the entire stated range.
- Invalidation classes: **render** (instant, no re-sim), **late** (re-run
  from that stage over cached upstream fields), **full** (re-simulate).
- Tiers: **primary** (the sliders) and **advanced** (present, tucked away).

## 10. Performance budgets

| Resolution | Target |
|---|---|
| 256² preview | < 1 s |
| 512² | < 3 s |
| 1024² | < 15 s |
| 2048² final | < 90 s |

## 11. Hex promise

`hexify` emits per-hex **area-weighted aggregates of fields** (not terrain
ids) on pointy-top, odd-r offset layout, addressed `col,row`. Lossy,
deterministic, strictly downstream. Exact schema (JSON shape, per-layer
aggregate choice — mean/dominant/max) is co-designed with the target editor
at M5; it is that system's API. Editor facts: odd rows shift half a hex
right; six neighbors; editor caps at 1000 hexes per axis.

## 12. Review and process

- Review is by **image-batch galleries** (seeds × sizes × variants; contact
  sheets are first-class output).
- Value-ledger rule per `CLAUDE.md`: predicted yields at build time,
  ablation pairs for predicted-marginal features, observed yields at review,
  trims decided only by the author.
- Milestones end with commit recommendations; exit criteria include current
  ledger rows.

## 13. Dependencies and versioning

- numpy + Pillow. scipy (or anything else) only with author sign-off backed
  by benchmarks.
- The engine version stamps every output. Any change that alters output for
  the same seed+controls (algorithm, apron, defaults) bumps it.
