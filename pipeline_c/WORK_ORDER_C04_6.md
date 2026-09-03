# Work order — C04.6: the seam is a curve

Issued 2026-09-03 for Opus 5. Follows [`WORK_ORDER_C04_5.md`](WORK_ORDER_C04_5.md)
and its report at `out/C04_5_BUILD_REPORT.md`. Same rules as C04.5:
isolation from `pipeline_a`, `pipeline_b`, `pipeline_d`; no commit; do not
edit `DESIGN.md`; the search server on port 5004 is `GET /api/status`
only, never started, stopped or posted to; no aesthetic language; one
mechanism change.

## 0. Purpose

C04.5 measured what reconnects a cut piece: of the cells that let a
piece rejoin, 99.65 % on the C04.5 engine and 99.71 % on the C04.4 engine
were **vacated**, not healed. Two markers in one cell translate together,
cross a cell boundary, and round into two different cells; the cell they
shared holds nothing, so it is intact, while the seam through it was
slipping at a median twenty times the yield. The report's §6 traces one
event to the marker. `gaps_closed`, the cells a tip opened that the marker
record had flagged as a hole, ran 706 to 1,547 per run, and advances ran
4,991 to 8,926 against 899 to 2,181 under the C04.4 rule: the tips spend
most of their advances re-closing holes, and the network that results
opens four times as many cells, ends at a weak share of 0.17 to 0.28, and
fails drift on all twelve seeds while passing plate count on ten.

The marker raster is a set of points, and a set of points has holes. A
seam is a curve. **One mechanism change:** the seam becomes a curve
carried by its markers, which are linked in order along each crack, and
the raster is drawn from the segments between linked markers. A segment
cannot be vacated: wherever its two ends go, the cells between them are
drawn. The tip is the end vertex of its chain, its position is that
vertex, and a crack that reaches another crack links to it and stops.
Nothing else changes: the direction rule of C04.5, the Griffith threshold,
nucleation, the damage and healing law per marker, the marker motion, the
rigid solve and the internal-stress solve are as they are.

**Predictions on record**, twelve development seeds at 512 px,
`solve_divisor = 1`, `seams = 2`, the C04.4 §1 dials, against the C04.5
report's tables:

- P1. Vacated bridging cells are zero by construction, and rejoin events
  on the twelve fall from 1,181 to under 50, all through healed cells.
- P2. The flicker stops. After the first cut of a piece of 164 cells or
  more (the floor at 128²), the second piece holds at least half of its
  size at the cut on at least 80 % of the remaining steps, on at least
  eight seeds. (C04.5: `892, 2, 6, 30, 387, 845, ...`.)
- P3. Advances per run fall below 4,000 on every seed, `weak_final` sits
  in 0.02 – 0.25 on at least nine seeds, and `weak_drift` passes on at
  least six, from zero.
- P4. `plate_count` stays in the 3 – 8 band on at least eight seeds, from
  ten.
- P5. `edge_fraction` stays at or above 0.90 on all twelve.
- P6. Cost within 1.2 times C04.5.
- P7. In the paired probe, worlds passing all six terms rise from 2 of
  160 to at least 8, and at least one cell passes on all four seeds.

Failure modes: the eight of C04.5 (the seven C04.4 definitions plus
**bridging**), and two new ones. **Sutures**: at the end of a run the
markers held above `WEAK_THRESHOLD` and not yet removed outnumber the weak
markers three to one, or reactivations, markers crossing back below the
threshold, outnumber nucleations, so remembered curves rather than new
cracks carry the dynamics. **Tangling**: the mean number of segments
covering a seam cell exceeds 2, or more than 1 % of segments are longer
than 1.5 cells after subdivision.

## 1. The mechanism

### 1.1 Links

`Markers` gains an edge list: two integer arrays `a`, `b` of equal length,
marker indices, undirected, no duplicates, no self-edges. Every operation
that removes or reorders markers reindexes the edges, and a test says so.
`markers.empty()` has no edges. A marker's **degree** is its edge count; a
**chain** is a connected component of the edge graph; a marker of degree
0 is a nucleus and of degree 1 a tip. `create` takes, for each new marker,
the index of the marker it links to, or none for a nucleus.

### 1.2 The raster

`markers.raster` draws every edge as a line on the torus, by the minimal
image, and every marker as a point. Sample each segment at a spacing of
at most half a cell including both ends, round each sample to its nearest
cell, and give the cell the strength linearly interpolated along the
segment between the two ends' `s`; a cell's value is the **minimum** over
every sample it receives, as now over markers. Half-cell sampling makes
the drawn cells an 8-connected path, so a segment at any angle is one
cell wide and a segment of one cell length draws its two end cells. The
reduction is a sort, not a scatter with repeated indices, as now. State in
the report the total samples per step at the end of a run.

### 1.3 Tips, direction, advance, meeting

A tip is a marker of degree at most 1; the raster-based `tips` is no
longer read under `seams = 2`. Its position `p` is its own position, so
`multi_marker_tips` is gone; delete the counter. The direction is
C04.5's, from the stress averaged over the intact 8-neighbours of the
tip's cell; the sign points away from the tip's one linked marker, by the
minimal-image vector from that marker to the tip; a nucleus tries both
signs as now. The crack length `L` in the Griffith threshold is the number
of markers in the tip's chain, at least 1, and replaces the raster
`crack_lengths` under `seams = 2`.

The advance walks from `p` along `d` as C04.5 does, to the first cell
whose nearest cell differs; that cell must be intact in the raster and its
traction must reach the threshold, as now. A qualifying advance creates a
marker at `p'` with one edge to the tip. Then, if any marker of a
**different** chain lies within 1.5 cells of `p'`, add a second edge from
the new marker to the nearest such marker: the crack has met another and
the new marker has degree 2, so it is no longer a tip and this crack
stops here. Record per run the meetings so made. A cell already covered by
a segment whose strength is at or above `WEAK_THRESHOLD` is intact and a
candidate like any other, and an advance into it that links to that
segment's chain counts as a meeting.

Nucleation is unchanged: it creates a degree-0 marker at the cell centre.

### 1.4 Motion, damage, healing, removal

Markers move as now, each at its own cell's velocity. After the move,
every edge longer than 1.5 cells is **subdivided**: a marker at its
midpoint with `s` the mean of the ends, the edge replaced by two. Record
subdivisions per run.

Damage and healing per marker are unchanged. Removal moves: a marker is
removed when its strength reaches `SUTURE_STRENGTH = 0.9`, a new constant
in `constants.py` with this order named beside it, not at
`WEAK_THRESHOLD`. Between the two a marker is intact in the raster and
still a vertex, so the curve is remembered and reopens where the slip
returns; that is a suture, and **sutures** above names the mode where it
takes over. On removal, a marker of degree 2 is replaced by one edge
between its two neighbours; any other degree just drops its edges. Record
per run the markers between the two thresholds at the end, and the
reactivations: markers whose `s` crossed from above `WEAK_THRESHOLD` to
below it in a step.

`gap_cells` and `gaps_closed` stay, and must read zero, which is the
construction's own check.

### 1.5 Untouched

`seams = 0` and `seams = 1` are byte-identical to C04.5, with their pins.
Under `seams = 2`, `explore_worker` records the new counters; no view
changes, though the `pieces_motion` sheet is expected to look different
and the report says how the pieces move after a cut.

## 2. Tests

A new `tests/test_curve.py`, or additions to `tests/test_markers.py`:

- A single edge from (10.0, 10.0) to (27.3, 20.0) on a 64-cell torus
  rasters to an 8-connected, one-cell-wide path that contains both end
  cells; the same edge across the wrap does too.
- The C04.5 §6 event as a unit test: two linked markers at (98.5012,
  98.6648) and (99.3770, 99.1475), both displaced by (−0.269, +0.509)
  cells, leave cell (99, 99) drawn.
- A tip advancing into a cell within 1.5 cells of another chain gains a
  second edge and is not a tip afterwards.
- Removal of a degree-2 marker leaves one edge between its neighbours and
  the raster between them unbroken; removal of a degree-3 marker drops
  three edges.
- An edge stretched past 1.5 cells by a move is subdivided once and the
  new marker's `s` is the mean.
- Edges reindex correctly under removal: a random removal set on a random
  graph leaves every surviving edge joining the same two surviving markers.
- Determinism across the pool and the in-process path on `seams = 2`.
- `seams = 0` and `seams = 1` pins unchanged; every `seams = 2` pin that
  moves is re-pinned with a one-line comment naming this order and listed
  in the report.

The suite must pass:

```
py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"
```

## 3. Run it once

1. The twelve seeds as C04.5 §5.2, the C04.5 table's columns plus: rejoin
   events and bridging cells by class (§2 of C04.5, rerun on this engine);
   the flicker series of the second piece after its first cut of 164 cells
   or more, per seed; meetings, subdivisions, suture markers, reactivations,
   samples per step; mean segments per seam cell. Every view sheet to
   `out/c04_6_twelve_<view>.png`. Keep the C04.5 raster reachable behind a
   private flag for the comparison row and remove the flag before the
   report, as C04.5 did.
2. The orientation measurement of C04.5 §5.1 on the same two seeds at
   crack speeds 10 and 40, this engine only, same columns.
3. The probe: 40 cells, 4 seeds, stage 1 only, `search_seed = 11`, at the
   served defaults, paired with the C04.5 probe that ran at the served
   defaults (the one of the two that did **not** pair with C04.4's; the
   report names its directory). Term pass rates, passers, the ten
   failure-mode counts, the paired table.
4. Seconds per world at `seams = 2` on this engine and C04.5's, same pool,
   same session, both orders.

## 4. Documents and report

`STATUS.md` "Now" one sentence; `EXPLORE.md` and `SEARCH.md` where the
matter belongs, one sentence each, including that the search's marker
count and `gaps_closed` mean something different from this order on.
`out/C04_6_BUILD_REPORT.md`: what was built; deviations; check output;
cost; the twelve-seed table against C04.5's; the flicker series; the
rejoin classes; the orientation table; the probe with its pairing;
predictions with the deciding numbers; the named mode; observations with
evidence and no proposed values. If pieces still rejoin, trace one event
as C04.5 did. If the weak share does not fall, say where the advances go
now that no holes need closing.
