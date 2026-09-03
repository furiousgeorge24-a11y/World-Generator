# Work order — C04.5: a crack runs where the stress points

Issued 2026-09-03 for Opus 5. Follows [`WORK_ORDER_C04_4.md`](WORK_ORDER_C04_4.md)
and its report at `out/C04_4_BUILD_REPORT.md`. Same rules as C04.4:
isolation from `pipeline_a`, `pipeline_b`, `pipeline_d`; no commit; do not
edit `DESIGN.md`; do not touch the search server on port 5004 beyond
`GET /api/status`, and do not start or stop it; no aesthetic language; one
mechanism change.

## 0. Purpose

Two things were measured on 2026-09-03 after C04.4, on the author's two
search runs (`out/search/20260903T080901Z-s12`, 1,341 cells, and
`20260903T155731Z-s13`, whose first three rounds re-sampled the first run
and whose fourth is new) and on two development seeds at 512 px.

**The crack paths are locked to the lattice.** The tip rule of C04 §2.4
scores the eight neighbours of a tip by the traction a seam in that
direction would carry and steps into the highest. Under a stress field
that varies over hundreds of cells the winner is whichever of the eight
directions lies nearest the principal axis, and it wins again at the next
advance, so a crack loaded at 20 degrees runs at 0 degrees. Nothing in the
rule can alternate two lattice directions to make an angle between them,
because it has no memory of how far the tip has been pushed off its line.
Measured on seeds `4287772760` and `2075014389` at 512 px, full grid,
`seams = 2`, lab defaults otherwise, over cells with exactly two seam
neighbours and over chords eight cells long on unbranched cracks:

| `crack_speed_km_per_myr` | cells per step | straight : bend | chords within 7.5° of a lattice angle |
|---|---|---|---|
| 10 | 1 | 2.8 : 1 | 0.69 |
| 40 | 4 | 4.3 : 1 | 0.65 |
| 160 | 16 | 4.2 : 1 | too few unbranched chords |

An isotropic process scores about 0.33 on the chord column. The lock is
there at one cell per step, so it is the eight directions and not the
unsolved field between advances. On both seeds horizontal runs were about
twice as common as anti-diagonal ones; two seeds cannot say whether that
is the tie-break order of `DIRECTIONS`.

**Plate count is the term the search fails on, and the pieces flicker.**
Across 9,324 search worlds at 512 px on the full grid, 71 % end with one
plate and 20 % with two; `plate_count` fails on 90 % of worlds, alone on
26 %, while `weak_drift` fails alone on 7 %. The C04.4 report §10 shows
why: pieces of 655 cells or more are cut on six of twelve seeds and then
rejoin on the next step, `759, 670, 122, 747, 62, 728, ...`, and
`gaps_closed` runs 28 to 316 per run. The report reads this as one loop
cell healing. It may instead be one loop cell **vacated**: two markers on
the loop that move differently land in the same cell and leave a hole,
which `markers.py`'s docstring names as the price of the marker raster.
Which of the two it is decides the next order, and nobody has counted.

**One mechanism change**, §1: the tip's direction becomes continuous. The
crack's own position is already continuous, because since C04.2 a seam is
a set of markers with float positions; the tip marker's position is the
tip. **One measurement**, §2, with no engine change beyond a counter: what
reconnects a cut piece. And one bookkeeping fix, §3.

**Predictions on record.**

- P1. On the two seeds above at crack speeds 10 and 40, the share of
  eight-cell chords within 7.5° of a lattice angle falls below 0.45, and
  the largest of the four straight-orientation counts is within 1.5 times
  the smallest. (If the share stays above 0.6, the lock is somewhere else
  and the report says where.)
- P2. `edge_fraction` on the twelve development seeds stays at or above
  0.90: a staircase at 30° is not wider than a diagonal at 45° as the term
  measures width.
- P3. `plate_count` on the twelve moves by at most one seed in either
  direction, and the flicker of C04.4 §10 is still there. This change is
  about angle, not count; the count is §2's question.
- P4. Of the events in §2 where a piece above the floor rejoins, at least
  two thirds reconnect through a **vacated** cell and not a healed one.
- P5. Cost within 1.1 times C04.4 at 512 px, full grid, `seams = 2`.

Failure modes: the eight of C04.4, plus **bridging**: `edge_fraction`
below 0.85 on four or more of the twelve, the staircase written two cells
wide.

## 1. The mechanism: a continuous tip direction

Under `seams = 2` only. `seams = 0` and `seams = 1` stay byte-identical;
`seams = 1` keeps the eight-direction rule as it is, and the pins that
guard it do not move.

**The tip's position.** A tip is still a seam cell with at most one seam
neighbour, found on the raster as now. Its position `p`, in cell units, is
the mean position of the markers the tip cell holds. Record per run how
often a tip cell holds more than one marker.

**The direction.** Read the lifted stress tensor the tip rule already
reads, averaged with equal weight over the tip cell's **intact**
8-neighbours: these are the cells the eight-direction rule scored. Take
`n` as the eigenvector of the largest absolute eigenvalue, which is the
normal that maximises `|sigma . n|` over all directions and is therefore
the same quantity the eight-direction rule maximised over eight. The seam
runs along `d`, `n` turned a quarter turn. Choose the sign of `d` that
points away from the crack: positive dot product with the vector from the
tip's one seam neighbour to the tip. A tip with no seam neighbour, a fresh
nucleus, tries both signs and keeps the one whose candidate carries the
larger traction, tie to the sign with positive `x`, then positive `y`.
Record how many tips per run had the two absolute eigenvalues within 1 %
of each other; for those the direction is whatever the eigensolver
returns, deterministically.

**The advance.** Move from `p` along `d` in steps of one cell length until
the nearest cell, by rounding each coordinate, differs from the tip cell;
that cell is the candidate, it is one of the eight neighbours, and the
point reached is `p'`. The candidate qualifies exactly as now: intact, and
the traction `|sigma . n|` read at the candidate with this `n` reaches
`toughness_fraction * sigma_c_field[candidate] / sqrt(L)`. A qualifying
candidate opens and its marker is created **at `p'`**, not at the cell
centre; `markers.create` gains the position. A tip with no qualifying
candidate stands still, as now. Nucleation is unchanged and still creates
its marker at the cell centre.

That is the whole change. The threshold, the length `L`, the stress the
rule reads, healing, merging, the rigid solve and the internal-stress
solve are untouched. A crack under a field whose principal axis sits at
20° now steps mostly along one lattice direction and sometimes along the
next, in the proportion that keeps `p` on the 20° line, because `p`
remembers where the line is.

## 2. The measurement: what reconnects a cut piece

No engine change beyond bookkeeping. Per step, on the twelve seeds, take
the intact components above the floor `plate_percent` uses. For every
component present at step `t` whose cells at `t + 1` lie mostly inside a
larger component, find the cells that were seam at `t`, intact at `t + 1`,
and 8-adjacent to both the piece and the component it joined. Classify
each such cell:

- **healed**: a marker in it was removed by `damage_and_heal` at that step;
- **vacated**: every marker it held left it through `move` and none was
  removed;
- **both**: some of each.

Report per seed: pieces of 655 cells or more cut; rejoin events; bridging
cells by class; and for the vacated class, the slip rate at that cell on
the step before, against the yield, so the report can say whether the
seam that let go had stopped slipping. Run this on the C04.5 engine; run
it also on the C04.4 engine (`git stash` is not available to you; keep the
old rule reachable behind the `seams = 1` path or a private flag for the
measurement only, and remove the flag before the report) so the two
columns sit side by side.

## 3. Bookkeeping

`explore_adapter.py`: the lab's `damage_time_myr` floor becomes 0.5, the
search's lower bound, so a search cell can be typed into the lab. Nothing
else in the dial table moves.

## 4. Tests

`tests/test_seams.py` or a new `tests/test_tip_direction.py`:

- A 64-cell synthetic world with a uniform stress whose principal axis is
  at 20°, one nucleus, forty advances at `seams = 2`: the chord from the
  first cell to the last is within 3° of the direction the rule predicts,
  and the chain contains at least eight bends. The same at 0° and at 45°:
  the chain is straight in the lattice direction, no bends.
- A marker created by an advance sits at `p'`, not at the cell centre, and
  inside the cell it opened.
- `seams = 0` and `seams = 1`: byte-identical to C04.4 on the pinned
  lines. Every `seams = 2` pin that depends on where a tip went will move;
  re-pin each with a one-line comment naming this order, and list them in
  the report.
- Determinism across the pool and the in-process path, as the adapter's
  test already does, on `seams = 2`.

The suite must pass:

```
py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"
```

## 5. Run it once

1. The orientation measurement of §0 on the same two seeds at crack speeds
   10, 40 and 160, before and after, with the same three columns and the
   four orientation counts. Before means the C04.4 rule; measure it in the
   same session rather than copying §0's numbers.
2. The twelve seeds at 512 px, `scale_km = 5`, `solve_divisor = 1`,
   `seams = 2`, the C04.4 §1 dials: the C04.4 §4.1 table's columns, plus
   §2's counts, plus the multi-marker and degenerate-eigenvalue counts.
   Every view sheet to `out/c04_5_twelve_<view>.png`.
3. The probe: 40 cells, 4 seeds, stage 1 only, through the library, to
   `out/search/`, at the search's served defaults, **at `search_seed = 11`**
   so `tools/pair_runs.py` pairs it cell for cell with the C04.4 probe
   `20260903T015247Z-s11`. Term pass rates, passers, the nine failure-mode
   counts, and the paired table. Note for the record that a run's round
   `r` samples from `search_seed + r`, so a probe must never start at a
   seed within four of a search run's; 11 is safe because no search run
   started below 12.
4. Seconds per world at `seams = 2`, before and after, same pool, same
   session.

## 6. Documents and report

`EXPLORE.md`, `SEARCH.md`, `STATUS.md` "Now": one sentence each; in
`SEARCH.md` say that search runs before C04.5 sample the same dials but
different crack paths, so a pre-C04.5 run pairs with a post-C04.5 run at
the same seed as an ablation pair and not as a replica.
`out/C04_5_BUILD_REPORT.md`: what was built; deviations; check output;
cost; the orientation table before and after; the twelve-seed table; §2's
table for both engines; the probe with its pairing; predictions with the
deciding numbers; the named mode; observations with evidence and no
proposed values. If §2 says vacated, describe exactly how a marker leaves
a cell without another arriving, with one traced event: the two markers'
positions and velocities on the steps before and after. If it says healed,
give the slip-rate history of the healed marker over its last five steps.
