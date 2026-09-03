# Status

Where Pipeline C stands and what has already been ruled out. This is the only
file that tracks state; update it when something changes.

Last updated: 2026-09-03

## Now

The mechanism is still **C03**, the kinematic history: a lithosphere on a
periodic parent world with a mantle drive, a strength field that weakens under
strain and heals slowly, a velocity solved from the two by a geometric
multigrid, and plates and boundaries that emerge from where strain localizes.
It contains no crust, elevation, water, coastline, island, or land, so it is
not a map and nothing in it may be called one. The two runs since C03.4
changed how it is looked at rather than what it does. C03.5 replaced the
initial strength noise with an isotropic spectral generator, which moved the
initial strength field's axis-to-diagonal power from 1.777 to 0.954 on the
audit seed, and put the history's settings behind a `HistoryParams` record
driving a second WebUI on port 5003: eleven development dials, eight seeds per
generation, every view a contact sheet of the eight, in a process pool. C03.6
then rebuilt the multigrid, because C03.5's first sweep of the dial space
reported plate counts and weak fractions read off velocity fields that had
never reached `MG_TOL`. The hierarchy now coarsens edge coefficients instead
of cell coefficients, smooths with red-black Gauss-Seidel, and solves the
coarsest grid exactly. The production default, which used to spend all twenty
cycles on 67 of 75 steps and finish at a worst relative residual of 1.7e-2,
now converges on every step at a worst of 9.9e-4 and costs 2.28 s at 1024 px
against 5.98 s. Above the default the solver still runs out: at
`stiffness_fraction` 1.0 and 2.0 every cell of the rerun sweep spends its
forty cycles without reaching tolerance, and the reason is measured — below
the first coarsening a two-cell weak line is narrower than one coarse cell,
the 2 x 2 aggregate absorbs it, and the coarse operator carries velocity
straight across the barrier. Half the dial space is therefore still unread.
No stable localizing regime has been found in it and none has been ruled out:
`stable_count` is zero in every converged cell of both sweeps, and the author
has not turned a dial yet. C03.7 added a ninth dial, the initial strength
spread, and a regime search on port 5004 that samples the whole dial space,
screens each cell on six measured properties of a plate regime, and reruns
passers on the twelve development seeds automatically; what it produces is
candidates for the author's eyes and not approvals, and it is described in
[`SEARCH.md`](SEARCH.md). C03.8 built the second of the two physics attempts
the stopping rule allows — damage driven by dissipated work rather than by
strain rate, behind a `work_damage` switch whose engine default is 0, so
every production output is byte-identical — and set the search's defaults to
the control run's own `config.json` with that one field flipped. The author
ran it: `20260902T170740Z-s3`, 1460 cells, no passer. It moved one thing.
The share of in-band worlds with two or more plates went from 0.025 under the
strain-rate law to 0.118 under the work law, crossing the 0.10 line the work
order named in advance as the condition under which the law earns further
work; zone width did not follow, with a band median `edge_fraction` of 0.20
against 0.17 and the `edge_fraction`-to-`weak_final` rank correlation still
at −0.82. Seven of those 1460 cells reached three or more plates at a weak
fraction of 0.30 or less, all of them at fast damage with fast healing and a
high yield percentile, and the search's defaults are now that corner at a
fresh seed. The question it is sampled to answer is whether zone width keeps
narrowing as both times shorten or bottoms out near an `edge_fraction` of
0.25; the run is the author's to start and its outcome is not yet on record.
The formal pairing against the control has not been run. That corner run,
`20260902T183110Z-s2`, is now on disk at 2154 cells and 12,492 worlds with no
passer, and the author measured zone widths directly on five of its frontier
cells: 90 to 98 % of weak cells sit in even-aligned 2 x 2 blocks and the share
at least three cells wide equals the share at least four in every world, so
the widths are quantized to the solve cell and the frontier sits at two to
three solve cells against a screen bound of two. C03.9 made the solve divisor
a parameter and reran twenty of those cells, on their own seeds, on the full
kinematic grid: `edge_fraction` rose by a paired median of +0.078 with 140 of
156 worlds rising, and no rerun world passed all six terms, so under the
order's own decision rule — ten per cent passing on one side, a paired rise
below +0.03 on the other — the run landed between the two thresholds and the
reading is the author's. C03.10 then put the mantle drive's coarsest
wavelength and the initial strength noise's into kilometres, so every
resolution now runs the same physics as `DESIGN.md` §2 requires and the
production world at 1024 px and 5 km per pixel is byte-identical: the symptom
that found it was C03.9's 512-px rerun, where three worlds passed the whole
screen and none passed at 1024 px on the same dials, because the drive was
half the parent at both sizes and so 2,560 km at one and 5,120 km at the
other while damage, healing and zone width stayed in kilometres. Those three
passes do not recur once the wavelength is fixed.

**C04, 2026-09-02.** The sheet is done. One length in it sets both things the
screen asks for — the solve length that makes a plate interior rigid is the
length over which strain spreads around a weak zone, so a zone widens until
it is that length across — and four runs moved the width without crossing the
bound; [`DESIGN.md`](DESIGN.md) §3.6 records them and the author ratified the
replacement. The seam formulation is now built behind a `seams` switch whose
engine default is 0, so every production output is byte-identical: a cell is
either intact or a seam, and damage happens on a seam by the work its slip
dissipates, at a seam's tip by the Griffith rule, or at a nucleation site,
and nowhere else. On the twelve development seeds at 1024 px, at the
production defaults with the corner's centre: **P1 failed by one world** —
`edge_fraction` is at or above 0.85 in eleven of twelve and 0.7225 in the
twelfth, seed `287488203`; **P2 failed** — `weak_final` runs 0.0029 to 0.0157
against a predicted floor of 0.005, so six of twelve are below the band and
none is above it; **P3 failed** — every one of the twelve ends with one plate;
**P4 held** — 1.31 s per world against the sheet's 0.93 s on the same pool in
the same session, a ratio of 1.41 against a bound of 1.5; **P5 held** in the
40-cell probe `20260902T225848Z-s11`, where nine of 160 worlds fail exactly
one term and that term is `plate_count` in all nine. **Neither predeclared
failure mode occurred on the twelve seeds**: crazing needs `weak_final` above
0.05 and the highest is 0.0157, dead ends need `network_share` above 0.9 and
the highest is 0.552. What happened instead is that cracks nucleate, run tens
of cells, and heal shut before they meet: the median crack is 4 to 8 cells
and the longest 24 to 47 on a 256-cell world, and no plate count changes at
any epoch of any trace. Both modes do occur in the probe, seven worlds each
of 160, and two of its 160 worlds reached two plates. Width is no longer the
obstacle — the probe's median `edge_fraction` is 0.954 and the term passes in
146 of 160 worlds, against a median of 0.348 and 21 % of worlds in the
sheet's 2154-cell corner run and 0.129 and 9 % in its whole-space run —
and the obstacle is now that a crack does not survive long enough to close a
loop. One measured artifact is on record and is not physics: the nearest-cell
advection carries a per-cell sub-cell remainder, and where the velocity
gradient is steep two neighbouring cells can spend theirs on the same source
cell and duplicate it. On seed `287488203` that puts 59 % of weak cells into
all-weak 2 x 2 blocks and holds `edge_fraction` at 0.72; with advection
frozen the same seed gives 6 % and 0.998. The measurements are in
[`out/C04_BUILD_REPORT.md`](out/C04_BUILD_REPORT.md).

**C04.1, 2026-09-02.** C04's outcome was that a seam damaged by the work its
slip dissipates cannot stay open — an open seam carries almost no traction, so
it dissipates almost nothing however fast it slips, its damage rate is near
zero, and healing shuts it in two or three steps — so cracks healed before
they met, and the twelve seeds opened 1,432 to 3,295 cells and kept a few
hundred. C04.1 makes `work_damage` mean something under `seams` — at 0, now
the default, a seam damages by its slip rate, so a fault that is slipping
stays weak and heals only when it stops — and on the same twelve seeds at the
same dials **P1 held** (persistence 0.85 to 1.26 against a floor of 0.5 in
eight seeds, in all twelve), **P2 held** (the longest crack at the end is
3,298 to 8,613 cells against a floor of 100, in all twelve), **P3 failed**
(every seed still ends with one plate, at every one of the 75 steps),
**P4 failed** (`edge_fraction` 0.426 to 0.617, below the 0.7 the advection
artifact was allowed, in all twelve), and **P5 failed by one world**
(`weak_final` 0.071 to 0.152, with seed `548870008` above the 0.10 ceiling).
None of the three counted failure modes occurred on the twelve: what happens
instead is that loops *do* close, from step 6 on seed `2075014389`, but they
enclose 1 to 200 cells against the screen's 1 % of 655, so the sheet ends as
one piece holding 83 to 91 % of the world plus 114 to 274 fragments, none of
them a plate. With advection frozen the same law gives `edge_fraction`
1.0000 and a longest crack of 74 to 76 cells, so the reach of P2 and the
width of P4 are both dominated by the advection artifact. The measurements
are in [`out/C04_1_BUILD_REPORT.md`](out/C04_1_BUILD_REPORT.md).

**C04.2, 2026-09-02.** C04.1's outcome was that seams persist and loops close
but enclose crumbs of at most 200 cells, so the plate count stayed 1 on all
twelve seeds while one piece held 83 to 91 % of the world. C04.2 builds the
block model of [`DESIGN.md`](DESIGN.md) §3.6's last paragraph behind
`seams = 2`, production and `seams = 1` untouched: pieces are rigid bodies
with three unknowns each, coupled through the tractions their seams transmit
by the sheet's own equation summed over a piece; the stress the tip and
nucleation rules read is a second sheet solve forced by the drag a piece
failed to match rather than by the drive; and the seam network is a marker
set with no advection at all. On the same twelve seeds at the same dials,
**P1 held** — `edge_fraction` is exactly 1.0000 in all twelve, against a
floor of 0.95, and the weak cell count equals the marker count at every step,
so nothing is duplicated; **P2 failed** — every seed ends with one plate and
the largest piece holds 99.73 to 99.93 % of the world, against a prediction
of six seeds at two or more plates and six at 70 % or less; **P3 failed** —
the first loop to close encloses 1 to 23 cells against the screen's 655, and
the largest piece any loop encloses at any step of any seed is 24 cells;
**P4 held** — 6.24 s per world against the sheet's 5.79 s on the same pool in
the same session, a ratio of **1.08** against a bound of 2.0; **P5 failed** —
no world of the 40-cell probe `20260903T004318Z-s11` passes `plate_count`,
against a floor of 10 %. The named failure mode is **heals before meeting**,
in all twelve seeds: `weak_final` is 0.0006 to 0.0024 and persistence 0.030
to 0.080. The mechanism is measured: a seam cell slips only where it links
two distinct pieces, so a crack inside a piece has no slip, no damage and
nothing to oppose healing, and on seed `2075014389` only 212 of 4,256
marker-steps were ever above the yield slip rate while 37 of the 75 steps had
none at all. The width question is closed by construction and the reach
question is open again. The measurements are in
[`out/C04_2_BUILD_REPORT.md`](out/C04_2_BUILD_REPORT.md).

**C04.3, 2026-09-02.** C04.2's outcome was that a seam cell slips only where
it links two distinct pieces, so a crack inside a piece had no slip, no
damage and nothing to oppose its healing, and all twelve seeds ended with one
plate holding 99.7 to 99.9 % of the world. C04.3 makes the slip a seam cell
damages by the rigid jump **plus** the elastic strain-rate invariant of the
internal-stress solution at that cell — the rate a crack's faces displace
under the load its own piece carries — so a crack inside a piece slips too,
and on the same twelve seeds at the same dials the healing question closed
and the reach question did not: **P1 failed** on its second half — persistence
is 0.78 to 0.94 against 0.030 to 0.080 and above the 0.5 floor in all twelve,
but the longest crack is 31 to 121 cells and none of the twelve reaches 200;
**P2 failed** — every seed still ends with one plate and the largest piece
holds 98.95 to 99.24 % at the end and never less than 98.96 % at any of the
900 steps; **P3 failed** — the first loop encloses 1 to 3 cells and the
largest any loop encloses at any step is 16, against the screen's 655;
**P4 failed** — 1.69 to 1.76 s per world against the sheet's 0.86 to 0.94 on
the same pool in the same session, a ratio of 1.88 to 1.97 against a bound of
1.3; **P5 failed** — no world of the 40-cell probe `20260903T011451Z-s11`
passes `plate_count`, and that probe drew exactly the cells C04.2's did.
**None of the six counted failure modes occurred on the twelve**, each by its
own number, and what happens instead is measured: cracks now persist —
16,818 of 16,925 marker-steps are above the yield slip rate on seed
`2075014389` against C04.2's 212 of 4,256, and no step has none against 37 of
75 — but they do not run. Tips per run rose to 1,140 – 2,330 while advances
fell to 438 – 635, and each seed ends with 81 to 108 separate cracks of
median 4 cells that never join, `network_share` 0.05 to 0.18. The reason is
the Griffith threshold: at the end of seed `2075014389` only 9 of 40 tips
have an intact neighbour reaching `sigma_c / sqrt(L)` even on the stress
magnitude, an upper bound on the traction, because the crack at a tip is 2
cells in the median and the mean stress on intact cells is 1.21 against an
intact strength of 2.03 — and that median ratio does not move as the seam
count grows five-fold over the run. The elastic part carried the damage
essentially alone: the per-run mean elastic slip rate over seam cells is
0.017 to 0.035 against a rigid 0.0001 to 0.0020. The measurements are in
[`out/C04_3_BUILD_REPORT.md`](out/C04_3_BUILD_REPORT.md).

**C04.4, 2026-09-03.** C04.3's outcome was that cracks persist and do not
run, because the stress ahead of a tip reaches `sigma_c / sqrt(L)` for only 9
of 40 tips; C04.4 separated the two candidate causes — an unresolved tip
concentration and an implicit toughness — and measured them side by side, and
**the solve grid is what moves the block and the toughness is not**. Binned
by crack length the tip ratio at `solve_divisor` 2 runs 0.488, 0.574, 0.833,
0.941, 3.230 with 354 of 515 tips in the shortest bin, and at
`solve_divisor` 1 it runs 0.858, 0.834, 0.978, 2.624, 7.895 with 216 of 386
tips in the longest bin and 211 of those qualifying; two seeds at 1024 px on
the full grid give the same shape at the same resolution, so it is the grid
and not the world size. On the twelve seeds at 512 px on the full grid the
longest crack is **230 to 1,206** cells against 31 to 121, `network_share`
0.267 to 0.923 against 0.051 to 0.183, `weak_final` 0.044 to 0.080 — inside
the screen's band on every seed for the first time in the series — and six of
the twelve cut a piece of 655 cells or more, at steps 30 to 69, against none
ever; but `plate_count` is still 1 on eight of twelve, because the largest
intact component holds 82 to 95 % at the end and the pieces a loop cuts
flicker, holding 655 cells or more on only 2 to 6 of the steps after the cut.
**P1 held on its first half and failed on its second** (4 of 12 seeds reach
two plates, against six); **P2 failed on both halves** — at
`toughness_fraction` 0.5 the longest crack is 42 to 191 and `plate_count` is
1 on all twelve, and on C04.3's own scale the stress ahead of a short tip
*falls*, median 0.369 against 0.488, because the extra advances unload the
intact cells further; **P3 held** — 17.5 % of the 512-px probe
`20260903T015247Z-s11`'s 160 worlds pass `plate_count` against 0.0 % in
C04.2's and C04.3's probes; **P4 failed** at 1.94 to 1.99 times the 1024-px
sheet against a bound of 1.5, though the 512-px full-grid run costs 1.09
times the 1024-px half-grid run it replaces. The named mode is **crazing on 5
of 12** with nothing named on the other 7, and C04.3's *persists but does not
run* is 12 of 12 at `solve_divisor` 2, 10 of 12 at toughness 0.5 and 0 of 12
at `solve_divisor` 1. Two things got worse: `weak_drift` fails on 5 of the
twelve and has a probe median of 0.0538 against a tolerance of 0.03, and 6 of
the probe's 40 cells are invalid at `max_cycles` 40 where the two earlier
probes had none. The measurements are in
[`out/C04_4_BUILD_REPORT.md`](out/C04_4_BUILD_REPORT.md).

**C04.5, 2026-09-03.** C04.4's outcome was that cracks run at 512 px on the
full grid but the plate count stays 1 on eight of twelve, and two things were
measured after it: the crack paths are locked to the lattice — the tip rule
scored eight neighbours and took the best, so a crack loaded at 20 degrees ran
at 0 — and the pieces a loop cuts flicker on and off from step to step. C04.5
makes the tip's direction continuous under `seams = 2` alone: the direction is
the eigenvector of the stress averaged over the tip cell's intact neighbours,
the advance walks one cell at a time from the tip's own markers, and the
marker it creates sits at the point the walk reached, so the crack's position
between steps is a float and not a cell centre. `seams = 0` and `seams = 1`
are byte-identical and keep the eight-direction rule. On the twelve
development seeds at 512 px on the full grid at C04.4's dials, **P1 held on
its first half and failed on its second** — the share of eight-cell chords
within 7.5 degrees of a lattice angle falls from 0.653 to 0.415 at one cell
per step and from 0.737 to 0.430 at four, against a predicted 0.45, and the
straight cells go from 1.03 : 1 straight-to-bend to 0.47 : 1, but the four
straight orientations are still 3.4 to 3.8 times apart against a predicted
1.5, horizontal and vertical over the two diagonals; **P2 held** —
`edge_fraction` is 0.948 to 0.977 on the twelve, above the predicted 0.90, and
the new **bridging** mode did not occur; **P3 failed** — `plate_count` was
predicted to move by at most one seed and moved on ten, from 1, 1, 1, 1, 1, 1,
1, 1, 2, 2, 2, 4 to 1, 3, 3, 3, 3, 4, 6, 7, 7, 8, 8, 9, so ten of twelve are
now inside the screen's band of 3 to 8 against one before, while the flicker
is still there — the second piece on seed `4287772760` runs 892, 2, 6, 30,
387, 845 from the step of the first cut; **P4 held** — of 1,181 rejoin events
on the twelve, 25,258 of 25,347 bridging cells were **vacated** and 88 healed,
99.6 %, and on the C04.4 engine measured in the same session 3,469 of 3,479,
99.7 %, so the C04.4 report's reading was wrong: a cut piece rejoins because
two markers that shared a cell move the same way and both round out of it,
not because a seam heals; **P5 failed** — 1.39 times C04.4 by pool wall and
1.42 by mean per-world seconds against a bound of 1.1. Two things got worse:
`weak_final` is 0.171 to 0.282 and three seeds are above the screen's 0.25
ceiling, which names **runaway** on 3 of 12 with nothing named on the other 9,
and `weak_drift` is 0.079 to 0.147 against a tolerance of 0.03 on all twelve,
against 5 of 12 failing before. In the 40-cell probe at C04.4's own config,
which pairs cell for cell with `20260903T015247Z-s11`, `plate_count` goes from
17.5 % of worlds to 52.5 % and `network_share` from 48.8 % to 80.6 %, while
`weak_final` falls from 98.1 % to 33.8 % and `weak_drift` from 16.9 % to
2.5 %. The measurements are in
[`out/C04_5_BUILD_REPORT.md`](out/C04_5_BUILD_REPORT.md).

**C04.6, 2026-09-03.** C04.5's outcome was that 99.6 % of the cells through
which a cut piece rejoined a larger one were **vacated** and not healed: two
markers that shared a cell moved the same way and both rounded out of it. C04.6
makes a `seams = 2` seam a **curve**. The markers of a crack are linked in
order, the raster draws the segments between linked markers as well as the
markers themselves, a tip is the end vertex of its chain, a crack that reaches
within one and a half cells of another links to it and stops, a marker stays on
the curve until it heals to `SUTURE_STRENGTH` 0.9 rather than leaving at the
weak threshold, and every edge the motion stretches past one and a half cells is
split at its midpoint. `seams = 0` and `seams = 1` are byte-identical. **The
mechanism does not finish a run at the order's own dials**: the split puts a new
vertex exactly where the velocity field jumps from one cell to the next, and the
vertex count grows by 1.18 to 1.21 per step on every one of the twelve seeds,
reaching 250,000 vertices on a 16,384-cell world at step 44 to 50 of 75. The
ablation is decisive — with the split off the same world finishes in 4.9 s with
2,394 markers, and with the suture threshold put back at the weak threshold and
the split kept it still reaches 176,194 — and the discontinuity it feeds on was
already there: on the C04.5 engine the velocity difference between a seam cell
and a seam neighbour has a median of 0.000 cells per step and a ninetieth
percentile of 0.11 to 0.78. The twelve seeds were therefore measured over
**120 Myr**, thirty steps of the same 4 Myr, which both engines finish, and again
over 300 Myr under a budget. **P1 failed** — vacated bridging cells were
predicted zero and are 2,107 of 2,774 over 120 Myr, every traced one a cell that
held a single marker at the **end** of a chain, where no segment lies beyond it
to draw the cell it leaves; **P2 held** — the flicker stopped, the second piece
holding half its size on 9 of 12 seeds over 120 Myr and 12 of 12 over the
budgeted 300 Myr, against 0 of 12 for C04.5, and a cut piece rejoins on 21 % of
the steps it is cut against 95 to 100 %; **P3 held on two of three** — advances
are 1,304 to 2,050 against a bound of 4,000 and `weak_final` is 0.154 to 0.227
on all twelve, but `weak_drift` passes on none; **P4 held** — `plate_count` is
in the screen's band of 3 to 8 on 8 of 12 against 1 of 12 for C04.5 on the same
120 Myr; **P5 failed** — `edge_fraction` is 0.677 to 0.818 against a predicted
0.90, so the predeclared **bridging** mode fires on all twelve; **P6 failed** —
2.43 and 2.51 times C04.5 by pool wall against a bound of 1.2; **P7 was not
run**, because a 160-world probe of worlds that do not finish cannot be written
into the search's record. The named modes are **tangling** on 12 of 12 — 4.5 to
16.9 segments draw into a seam cell and 4 to 11 % of segments are still over the
bound after the split — **sutures** on 12 of 12 on its reactivation clause, and
**bridging** on 12 of 12. The measurements are in
[`out/C04_6_BUILD_REPORT.md`](out/C04_6_BUILD_REPORT.md).

## The open question

Not a formal gate — just the thing worth deciding next:

**Does a stable localizing regime exist anywhere in the dial space?** A world
counts for the screen when it ends with 3 to 8 plates above 1 % of the parent,
a final weak fraction between 0.02 and 0.25, and a peak weak fraction under
1.5 times the final: strain that localized into boundaries and then stopped
spreading, rather than a sheet that stayed whole or one that failed
everywhere. Two sweeps have now looked at 25 settings of `stiffness_fraction`
against `yield_percentile` on eight seeds each, and `stable_count` is zero in
every cell that converged; eleven of the twenty-five cells did not converge
and have no result to screen at all. Neither sweep moved any of the other nine dials.

Undecided as of 2026-09-02, and it stays undecided until the author explores.
[`EXPLORE.md`](EXPLORE.md) says how to start the lab and what the dials mean;
`out/c03_5_sweep.md` and `out/c03_6_sweep.md` are the two passes over the
`stiffness_fraction` by `yield_percentile` grid, so the search does not start
from a blank panel. The sweeps are a map of that grid, not a search of the
space: they propose nothing, and `stable_count` is a screening number for the
person at the dials rather than a gate or an approval.

The [`DESIGN.md`](DESIGN.md) §8 question underneath it — whether boundaries
curve, segment, and change regime along their length, which §1's torus
decision rides on — cannot be read until something localizes and stays
localized. The blind layer audit is the instrument for the mechanism half of
it, whether these fields are the footprint of a process or the output of a
formula. The macro-scale half stays with the author.

## What has been tried

This is the part worth carrying forward. All three attempts passed every
automated gate they declared; two were rejected on sight anyway. Numeric
diversity gates have so far failed to detect a repeated visual grammar.

### C00 — parent-world foundation (superseded)

A `40,960 × 40,960 km` parent on a flat torus in both axes, addressed in
integer metres, sampled at 512 and 1024. Randomness is a stateless SHA-256
sampler keyed by world, stage, stage version, process, channel, and physical
address, so traversal order, resolution, chunking, cache warmth, and observer
window cannot reroll anything. Twelve development identities, four debug, 32
sealed validation.

This layer works and is still in use. It contains no geology.

### C01 — point-power affiliation (rejected)

Seven actors placed by global maximin, ownership assigned by an additive
weighted power partition over the torus metric.

**Failed because** every one of the twelve worlds came out equant, evenly
packed, and honeycomb-like. Unique hashes and 24–64% pairwise pixel
disagreement did not detect the repetition — the worlds differed in detail
while sharing one grammar.

**Do not** return to maximin point placement or point-distance power
partitions for macro affiliation.

### C02 — connected competitive growth (rejected)

Seven actors grown from 33-cell germ lines by claim-once competitive blocked
growth on a fixed 1024² torus lattice, with four seeded layout families
(`scatter`, `belt`, `dual_focus`, `arc_void`), two low-frequency triangle-wave
resistance modes, and directional step costs.

Better hierarchy and family variation than C01 — unequal actors (4.3%–26.4%),
belts, lobes, junctions, broad interiors. But contacts are largely long
planar runs, several worlds read as horizontal or vertical bands, and D09 has
a narrow vertical tether between two banded halves.

**Suspected cause** if C03 is needed: the step cost is `192` parallel to the
actor axis versus `896` perpendicular, on a four-neighbour cardinal lattice.
That ratio makes axis-aligned growth strongly preferred, so the mechanism
tends to manufacture exactly the banding the gate asks the author to screen
for. Curvature and local raggedness are deferred to C6, but C6 cannot be asked
to rescue an unacceptable macro topology.

**Rejected on sight** 2026-09-01: contacts locked to 0°, 45°, and 90° because
growth was a shortest-path search on a four-neighbour lattice; the directional
cost only added banding on top. **Do not** return to lattice graph search for
any moving front.

## Where the old evidence is

Nothing is sealed any more. Every world is reproducible from its seed, its
resolution, and its scale, which together are its identity. The twelve
development seeds, in order, are:

```
2075014389, 2477733044, 476149591, 151640007, 2697441485, 1504571935,
548870008, 2157195430, 4108373596, 4287772760, 287488203, 1833546021
```

## Verification

Last run 2026-09-02 on this working tree, after C04.1:

| Suite | Result |
|---|---|
| `py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"` | 361 passed (engine, both adapters, the search library and its server, the run-pairing tool, the A/B solve tool, the work-damage law, the seam formulation, the solve divisor, the C03.1 regression checks, noise and strength isotropy, the yield gate, the effective-gradient gate, and the stiff-network solver gate) |
| `py -3.14 pipeline_c/tests/eval_checks.py` | 70/70 passed (eval scaffold) |
| `py -3.14 pipeline_c/tests/layer_audit_checks.py` | 59/59 passed (layer audit) |

The solver's gate is `tests/test_solver.py::StiffNetwork`: a 128 cell grid at
`stiffness_fraction` 2.0, cut by three rows, two columns and a diagonal of
weak cells two cells wide, must reach `MG_TOL` from a cold start inside forty
cycles and a warm restart must cost at most two. It takes **37 cycles** to a
residual of 4.95e-4, and the warm restart costs zero. The Fourier comparison,
the barrier, the symmetry and the adjoint tests are unchanged and pass; the
barrier now costs 5 cycles against 10, and the Fourier comparison 3 against 3
at a fifth of the solution error.

At the production default the whole history now converges: worst relative
residual 9.9e-4 on seed `4287772760` at 1024 px, mean 5.45 cycles per step
against a budget of 20, no step exhausted, 2.28 s per world against 5.98 s
before. Pooled eight-world generations at 1024 px cost 4.71 s at
`stiffness_fraction` 0.125 against 12.34 s before, and 6.63 s at 2.0 against
6.16 s. At 2.0 the worst residual fell from 104 to 0.20 and is still above
tolerance; the measurements are in
[`out/C03_6_BUILD_REPORT.md`](out/C03_6_BUILD_REPORT.md).

The blind layer audit was last run on `20260902T052615Z-seed4287772760`, over
all ten default views at 1024 px, thirty-two panels in two fresh-context
calls, and is reported in
[`out/C03_4_BUILD_REPORT.md`](out/C03_4_BUILD_REPORT.md). It predates the
isotropic noise and the solver rebuild and has not been rerun since.

## Leftovers

`eval/` still holds the three land instruments, which are protocol only: they
need generator output that does not exist yet. The fourth instrument, the
blind layer audit in [`run_layer_audit.py`](run_layer_audit.py), is complete
and is wired to the C03 views.
