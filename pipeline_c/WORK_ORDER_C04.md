# Work order — C04: the seam formulation

Issued 2026-09-02 for Opus 5. Follows [`WORK_ORDER_C03_10.md`](WORK_ORDER_C03_10.md)
and its report at `out/C03_10_BUILD_REPORT.md`; it builds on that order's
engine, with the drive in kilometres. The design is `DESIGN.md` §3.6,
ratified by the author; read it first. Same rules: isolation from
`pipeline_a`, `pipeline_b`, `pipeline_d`; NumPy, Pillow, Flask, and the
standard library; determinism of the engine; no review apparatus; do not
commit; do not edit `DESIGN.md`. Do not describe any field as natural,
plausible, good, or bad, and do not compare anything to the reference
images. **Do not start, stop, or restart the search server on port 5004,
and do not post to its API.** Your runs go through the exploration
adapter's own pool, or the search library, in your process.

## 0. Purpose

The sheet forms boundaries by diffuse damage and cannot hold them narrow;
`DESIGN.md` §3.6 records the four runs that measured that. This order
builds the seam formulation behind a switch: the same drive, the same
solve, the same strength field, the same advection of strength with the
lithosphere, the same views and metrics and search. What changes is
**where damage may happen**: on a seam, at a seam's tip, or at a
nucleation site, and nowhere else. Seams are therefore one cell wide by
construction and the width term of the screen is satisfied by the
formulation rather than searched for. What is *not* known, and what this
run measures, is whether cracks close loops and cut the sheet into pieces,
or craze, or dead-end.

**One mechanism change.** The switch swaps the damage rule. Nothing else
moves; every dial the sheet has keeps its meaning where it applies, and
two new dials cover what the rule needs.

**Predictions on record**, on the twelve development seeds at the
production defaults unless stated:

- P1. `edge_fraction` is at least 0.85 in every world. Whatever is below
  1.0 is advection and healing partials, not widening.
- P2. `weak_final` is between 0.005 and 0.10 in every world.
- P3. At least four of the twelve seeds end with two or more plates.
- P4. Cost per world at 1024 px is at most 1.5 times the sheet's on the
  same pool, measured in the same session.
- P5. In the 40-cell probe of §7, at least one world fails on at most one
  term of the screen.

And the two failure modes, counted so the report can name which one
occurred if P3 fails: **crazing**, `network_share` below 0.5 with
`weak_final` above 0.05; **dead ends**, `plate_count` of 1 with
`network_share` above 0.9.

## 1. Engine: the switch and the dials

In `engine/history/kinematics.py`, `HistoryParams` gains:

| field | default | range | meaning |
|---|---|---|---|
| `seams` | 0 | {0, 1} | 0: the sheet's damage law. 1: the seam formulation. |
| `crack_speed_km_per_myr` | 40.0 | 0 – 400 | how far a tip can advance per million years |
| `nucleations_per_step` | 2 | 0 – 20 | new cracks per step, at most |

All three are recorded by `to_record`. With `seams = 0` every output is
byte-identical to the current engine; `tests/test_regression_c03_1.py` and
the pre-order hash in `tests/test_work_damage.py` are the gates. Under
`seams = 1`, `work_damage` is not consulted — seam damage is always by
dissipated work — and the record still carries it; say so in the
docstring.

Dials that keep their meaning under `seams = 1`: `stiffness_fraction`;
`strength_exponent` (the stiffness of a seam cell relative to intact);
`heal_time_myr` (seam healing); `damage_time_myr` (seam damage by work,
the C03.8 law); `yield_percentile` (now the percentile of the first step's
stress magnitude that sets the intact strength); `strength_spread` (now the
heterogeneity of the intact strength, not of the initial strength field);
`drive_wavelength_km`, `drive_shear`, `solve_divisor`, `history_myr`,
`max_cycles`.

Add the constants the rule needs to `constants.py`: `SEAM_OPEN_STRENGTH =
STRENGTH_MIN` (what a freshly cracked cell's strength becomes) and
`INTACT_SPREAD_CLIP = (0.2, 2.0)` (bounds on the heterogeneity factor).
Nothing else new.

## 2. The rule, step by step

Under `seams = 1`, `run_history` does the following. Keep the code in a
separate module, `engine/history/seams.py`, with pure functions on arrays,
so each piece is testable alone; `run_history` calls them.

### 2.1 Initial state

Strength `S = 1.0` everywhere: an intact sheet, no seams, a stagnant lid.
The strength noise (`initial_strength`'s field at spread 1) becomes the
heterogeneity of the **intact strength** instead: at step 1, after the
solve, read `sigma_c` as the `100 - yield_percentile` percentile of the
solve-grid stress magnitude, and build the kinematic-grid field
`sigma_c_field = sigma_c * clip(1 + strength_spread * noise, *INTACT_SPREAD_CLIP)`.
Read `power_yield` at step 1 as C03.8 does. Both are read once and kept.

### 2.2 Stress

After the solve and `effective_gradients`, on the solve grid:
`sxx = kappa_s * exx`, `syy = kappa_s * eyy`, `sxy = kappa_s * exy`, and
the magnitude `smag = kappa_s * strain_rate_s` (the invariant already
computed). Lift the magnitude to the kinematic grid **in blocks**, as
`strain_rate` is lifted, for damage. Lift the three components and the
magnitude **bilinearly** (`prolong_bilinear`, repeated to `n`) for the tip
and nucleation rules, which pick directions and order cells and must not
see 2 x 2 steps. Keep both lifts; say in the docstring why there are two.

### 2.3 Seam damage and healing

A cell is a seam when `S < WEAK_THRESHOLD`. On seam cells, damage is the
C03.8 law on the block-lifted power (`smag_blocks * strain_rate`), excess
over `power_yield`, squared, at `damage_rate`. On intact cells the damage
rate is zero. Healing is `heal_rate` everywhere, through the same exact
integrator the sheet uses, clipped to `[STRENGTH_MIN, 1]`. A seam cell that
heals to `WEAK_THRESHOLD` or above is intact again, and `label_plates`
merges what it separated without any further code.

### 2.4 Tips

`tips(seam_mask)`: seam cells with at most one seam neighbour in the
8-neighbourhood on the torus. Crack length `L` for a tip is the size of its
8-connected seam component (`label_components(seam, 8)`, sizes by
`bincount`).

For each tip and each of the eight directions `d` to an **intact**
neighbour, the traction the would-be seam would carry is `t = sigma . n`
with `n` the unit normal to `d`, evaluated with the bilinear tensor at the
candidate cell:
`|t| = sqrt((sxx*nx + sxy*ny)**2 + (sxy*nx + syy*ny)**2)`.
The candidate qualifies when `|t| >= sigma_c_field[candidate] / sqrt(L)`,
the Griffith rule with the length in cells and the unit length one cell.
The tip advances into the qualifying candidate with the largest `|t|`;
that cell's strength becomes `SEAM_OPEN_STRENGTH`. Ties break by direction
index, fixed order, so the result is deterministic.

Advances per step: `k = round(crack_speed_km_per_myr * STEP_MYR / cell_km)`,
at least 0. Repeat the tip pass `k` times per step on the same stress
field, recomputing tips, lengths and the seam mask between passes, so a
tip that advanced is the tip that advances next. A tip whose advance
touches another seam component has joined it; nothing special is done,
the labels do it. Vectorize over tips: one pass is a handful of array
operations on the tip list, not a Python loop over cells.

### 2.5 Nucleation

Candidates: intact cells whose bilinear stress magnitude is at least
`sigma_c_field` and that have no seam cell in their 8-neighbourhood. Order
by `smag_bilinear / sigma_c_field`, descending, ties by row then column.
Take the first `nucleations_per_step`; their strength becomes
`SEAM_OPEN_STRENGTH`. A nucleus is a crack of length 1 and is a tip in the
next pass.

### 2.6 Order within a step and advection

Solve; stress; at step 1 the calibrations; damage and healing; `k` tip
passes; nucleation; then advection. Advection of `S` under `seams = 1`
samples the **nearest cell** at the departure point instead of bilinear:
a seam is a discontinuity, and bilinear sampling would turn a one-cell
seam into a two-cell ramp within a few steps, which is widening by
arithmetic. Nearest-cell sampling moves a piece as a body by whole cells
and keeps a seam exactly one cell wide; the error is a jitter of at most
half a cell that does not accumulate. Add `sample_nearest_periodic` to
`engine/domain.py` beside the bilinear sampler. The sheet keeps bilinear.

### 2.7 The record

`History` gains per-step lists `seam_fraction`, `tip_count`,
`nucleation_count` and `advance_count`, and the scalars `sigma_c` and
`power_yield`. `Epoch` gains `stress` (the bilinear magnitude, n x n) and
nothing else. The exploration worker returns `seam_fraction` beside
`weak_fraction` (identical under `seams = 1`, kept for the record).

## 3. Views

Every new layer gets a view. Add to `VIEWS` in both adapters, after
`power`: `stress` (bilinear magnitude at the final epoch, the scalar ramp,
log-scaled as `strain_rate` is if it is) and `intact_strength`
(`sigma_c_field`, scalar ramp). The `plates`, `boundaries`, `weak_*`,
`strength*`, `regime`, `velocity`, `strain_rate`, `power`, `trajectory`
and `drive` views are unchanged and read the seam formulation's fields
without modification.

The blind layer audit is **not** run in this order; the switch is off in
production. If the author keeps the formulation, `stress` and
`intact_strength` go through `run_layer_audit.py` before anything is
frozen. On record as a deviation from the standing rule.

## 4. The lab and the search

- Exploration lab `_DIALS`: `seams` (int, default 1 **in the lab**, lo 0,
  hi 1, primary, promise "0: the sheet, diffuse damage wherever strain
  exceeds yield. 1: seams, damage only on a seam, at its tip, or at a
  nucleation site; boundaries one cell wide by construction.");
  `crack_speed_km_per_myr` (float, default 40, lo 0, hi 400, primary,
  promise "How fast a crack tip runs. A rift propagates at tens of
  kilometres per million years."); `nucleations_per_step` (int, default 2,
  lo 0, hi 20, primary, promise "New cracks per step at the highest-stress
  intact cells away from existing seams."). The lab's default for `seams`
  is 1 because the lab exists to look at the new formulation; the engine's
  default stays 0.
- Search `Space`: fixed `seams: int = 1`; sampled
  `crack_speed_km_per_myr` log-uniform 10 – 200 and `nucleations_per_step`
  uniform over `{1, 2, 4}`; `DIALS` gains both; `params_of` threads all
  three; `search_server.py` offers the knobs with their meanings. The other
  ranges stay as the corner left them. `modernize_dials` fills the three
  new fields with the engine defaults on legacy cells.
- `tools/ab_solve.py` and `tools/pair_runs.py` print the two new dials
  where they print the others.

## 5. Tests

`tests/test_seams.py`, on the pure functions and on short runs at 128 px:

- byte-identity at `seams = 0` on a non-default dial set, against a hash
  taken before the edit, the C03.8 pattern;
- `tips` on a hand-built mask: a straight line has two tips, a loop has
  none, an isolated cell is a tip of length 1;
- the tip rule on a hand-built uniform tensor picks the direction whose
  normal carries the largest traction, and refuses every direction when
  `sigma_c` is set above it; the Griffith threshold at `L = 4` is half the
  threshold at `L = 1`;
- nucleation excludes cells adjacent to a seam, respects the cap, and
  orders by the ratio, with a deterministic tie-break;
- a seam healed to the threshold merges two labels into one;
- nearest-cell advection: a one-cell straight seam carried by a uniform
  velocity of 0.3 cells per step for 75 steps is still exactly one cell
  wide and has moved 22 or 23 cells;
- a 128-px run at `seams = 1` is deterministic, converges below `MG_TOL`
  at every step, and has `edge_fraction >= 0.85` at the end;
- `seams = 2`, `crack_speed = -1`, `nucleations_per_step = 21` are
  refused.

`test_search.py`, `test_search_server.py`, `test_explore_adapter.py`,
`test_adapter.py`: the new dials and views are offered, round-trip, and
render. The full suite must pass:

```
py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"
```

## 6. Cost

Measure and report seconds per world at 1024 px on the eight-worker pool
for the twelve development seeds at `seams = 1` and at `seams = 0`, in the
same session, same pool. The tip passes are `k` array operations on the
tip list per step; if the seam formulation costs more than 1.5 times the
sheet, profile it and say where the time goes. Do not change `k` or the
step to make the number.

## 7. Run it once

1. **The twelve seeds.** One bundle of the twelve development seeds of
   `STATUS.md` at 1024 px, 5 km/px, production defaults with `seams = 1`
   and the corner's centre for the dials that the corner moved
   (`heal_time_myr` 10, `damage_time_myr` 1.5, `yield_percentile` 6,
   `stiffness_fraction` 0.3, `strength_exponent` 2, `drive_wavelength_km`
   5120, `drive_shear` 0.6, `strength_spread` 0.03), through the
   exploration adapter's pool. Every view sheet to
   `out/c04_twelve_<view>.png`. Per world: `weak_final`, `edge_fraction`,
   `plate_count`, `network_share`, `weak_drift`, `peak_ratio`, tip and
   nucleation counts summed over the run, seconds. This decides P1 – P4.
2. **The probe.** A stage-1 of **40 cells at 4 seeds** with the search
   library's defaults (`seams = 1`) through the library, not the server,
   in your process, to `out/search/` like any run so the author can reopen
   it in the gallery. Report term pass rates, any passer, and the count of
   worlds in each failure mode of §0. This decides P5 and names the mode.
   Stop after stage 1; do not run stage 2 or 3 and do not loop.
3. **Sheets for the author.** For the probe's three best cells by soft
   score, the `plates`, `stress` and `trajectory` sheets are already on
   disk; list their paths.

## 8. Documents

- `EXPLORE.md`: the three dials and the two views.
- `SEARCH.md`: the fixed `seams` knob, the two sampled dials with their
  ranges, one sentence that the width term is now satisfied by construction
  so the search's question has become plate count and settling.
- `STATUS.md` "Now": one paragraph. The sheet is done and why, in two
  sentences pointing at `DESIGN.md` §3.6; the seam formulation is built
  behind a switch; the twelve-seed outcome against P1 – P5 in one sentence
  each once measured; the failure mode named if P3 failed.
- `ROADMAP.md`: the C04 and C04.1 rows from `DESIGN.md` §9, if the roadmap
  carries the build table.

## 9. Report

`out/C04_BUILD_REPORT.md`:

1. **What was built**, file by file.
2. **Deviations**, with reasons. The un-run audit is one.
3. **Check output**: the suite's verdict lines, verbatim.
4. **Cost**: the two seconds-per-world lines, and the profile if needed.
5. **The twelve seeds**: the per-world table verbatim; sheet paths.
6. **The probe**: run id, term pass rates, passers, failure-mode counts,
   the three best cells with dials and sheet paths.
7. **The predictions**: P1 – P5 with the deciding numbers and "held" or
   "failed"; the failure mode named and counted.
8. **Observations**, with evidence, no proposed values. If cracks close
   loops, say at what step in the trajectory sheets the plate count
   changes. If they dead-end, say how long they get. If seams widen
   anywhere, say where and by what mechanism, with the numbers.
