# Work order — C04.2: rigid pieces

Issued 2026-09-02 for Opus 5. Follows [`WORK_ORDER_C04_1.md`](WORK_ORDER_C04_1.md)
and its report at `out/C04_1_BUILD_REPORT.md`; the design is `DESIGN.md`
§3.6, whose last paragraph names this run. Same rules: isolation from
`pipeline_a`, `pipeline_b`, `pipeline_d`; NumPy, Pillow, Flask, and the
standard library; determinism; no review apparatus; do not commit; do not
edit `DESIGN.md`; no aesthetic language and no comparison to the reference
images; **do not touch the search server on port 5004 beyond
`GET /api/status`**. Runs go through the exploration adapter's pool or the
search library in your process.

## 0. Purpose

C04 and C04.1 built the seam formulation on the sheet's velocity solve and
measured it. Seams hold at one cell and, with slip-rate damage, persist
while they slip. Plates still did not form: on all twelve seeds the plate
count is 1 at every step, the crack network is a dense mesh inside the
band where the drive's stress is highest, and the loops it closes enclose
crumbs of at most 134 cells while one piece keeps 83 to 91 % of the world.
Two causes, both measured in the C04.1 report:

- **The stress a crack sees is the sheet's.** With mantle drag acting on
  every cell, the stress in the sheet is a local balance and sits in a band
  fixed by the drive's gradients. Every point of that band stays loaded
  however many cracks its neighbours carry, so cracks keep nucleating and
  joining there and never need to cross the quiet regions.
- **Nearest-cell advection duplicates seam cells** where the velocity jumps
  across a seam. With advection frozen, edge fraction is exactly 1.0; with
  it, 0.43 to 0.62.

This order is the block model proper. Pieces are rigid bodies. The stress
inside a piece is the integral of the drag it fails to match, which is
largest along the lines that would cut the piece into halves with opposing
net loads. Seams are carried on markers and cannot duplicate. The seam
rules of C04 and C04.1 — tips, Griffith threshold, nucleation, slip-rate
damage, healing — are unchanged; what changes is the velocity they act
on, the stress they read, and how a seam moves.

**One formulation, behind `seams = 2`.** `seams = 0` is the sheet and
stays production and byte-identical; `seams = 1` stays as C04.1 left it
for comparison.

**Predictions on record**, twelve development seeds at 1024 px, the C04
§7.1 dials with `work_damage = 0`, `seams = 2`:

- P1. `edge_fraction` at least 0.95 in every seed. Markers cannot
  duplicate; whatever is below 1.0 is healing partials and junctions.
- P2. At least six of the twelve seeds end with two or more plates, and on
  at least six the largest piece holds no more than 70 % of the world.
- P3. On at least six seeds the first loop that closes encloses at least
  655 cells, the screen's 1 % threshold. Cracks cut pieces, not crumbs.
- P4. Cost per world at most 2.0 times the sheet's on the same pool.
- P5. In the 40-cell probe, at least 10 % of worlds pass `plate_count`.

Failure modes, counted so the report names one: **crumbs** (largest piece
above 80 % with fifty or more pieces), **crazing**, **dead ends**,
**heals before meeting**, as defined in C04.1, and **locked** (no cell was
ever opened; `weak_final` below 0.001).

## 1. The rigid motion

`engine/history/rigid.py`, pure functions on arrays.

### 1.1 Pieces

Pieces are the 4-connected components of intact cells (`S >= WEAK_THRESHOLD`)
on the torus, from `label_plates`. Seam cells belong to no piece. For each
piece, a reference cell (its first cell in raster order) and each of its
cells' minimal-image offset from it, `dx = ((x - x_ref + n/2) mod n) - n/2`
and likewise `dy`; the centroid is the mean offset plus the reference; `r`
is the offset from the centroid. A piece whose cells cover every column
or every row wraps the torus and has no well-defined rotation: its `omega`
is fixed at zero and the report counts how often that happens.

### 1.2 The balance

The sheet's discrete equation per cell is `u_i - sum_e k_e (u_j - u_i) = D_i`
with `k_e` the harmonic edge stiffness. Summed over a piece with rigid
`u_i = v + omega x r_i`, every interior edge cancels and what remains is
the rigid body's force balance: basal drag over the piece against the
seam tractions on its boundary. The rigid model is that sum, with the
torque balance beside it. No new constant enters: drag is one per cell as
in the sheet, and a seam's coupling is the sheet's own `kappa(S)`.

Per piece `P` with unknowns `v_P` (2) and `omega_P` (1):

- force: `sum_{i in P} (D_i - u_P(x_i)) + sum_{seams s touching P} f_s -> P = 0`
- torque: `sum_{i in P} r_i x (D_i - u_P(x_i)) + sum_s r_s x f_s -> P = 0`

A seam cell `s` with strength `S_s` links each pair of distinct pieces
among its intact 4-neighbours. For the pair `(P, Q)` it transmits
`f = c_s (u_Q(x_s) - u_P(x_s))` to `P` and the opposite to `Q`, with
`c_s = kappa(S_s) / 2`: the two edges in series through the seam cell,
each dominated by the seam's own stiffness. A seam cell whose intact
neighbours all belong to one piece transmits nothing. Seam cells carry no
drag; they are one cell wide.

Assemble the `3N x 3N` system with `bincount` over cells and over seam
links, no Python loop over cells, and solve it dense with `numpy.linalg.solve`.
`N` is the piece count; at a few hundred pieces the dense solve is
milliseconds. Record the maximum force and torque residual per step.

### 1.3 The velocity field

`u(x) = u_P(x)` on intact cells. On a seam cell, the mean of `u_P(x_s)`
over the distinct pieces among its intact 4-neighbours; if it has none
(a seam cell surrounded by seams), the mean over its 8-neighbourhood's
seam cells' values from the previous step, and if that is empty, zero.
This `u` is the velocity every view shows and the one markers move with.

## 2. The internal stress

For each step, form the mismatch `m = D - u` on intact cells and zero on
seam cells, and solve the sheet's operator with it as the forcing:
`w - div(kappa grad w) = m`, with `kappa = KAPPA0 * S ** exponent`, the
same field the sheet uses, so intact cells are stiff and seam cells are
weak, through `to_solve_grid` and `solve` at the run's `solve_divisor`.
`m` has zero net force and torque over every piece by §1.2, so `w` carries
no rigid motion and is the piece's elastic deformation under its own
unbalanced load, screened beyond the stiffness length. The stress is
`kappa_s` times the `effective_gradients` of `w`, as C04 formed it from
`u`; lift the components bilinearly and the magnitude in blocks, exactly
as C04 §2.2, and feed them to the unchanged tip and nucleation rules. At
step 1 there are no seams, the one piece has `v = 0` and `omega = 0`
(the drive has zero mean), `m = D`, and the solve is the sheet's step-1
solve; read `sigma_c` and `yield_strain_per_myr` from it as C04 and the
sheet do. Say in the docstring that a piece larger than the stiffness
length sees its stress partly screened, and that the stiffness dial keeps
that meaning.

## 3. Seam damage on the rigid field

The slip rate at a seam cell is the velocity jump across it divided by
the cell: for the pair `(P, Q)` it links, `|u_Q(x_s) - u_P(x_s)| / cell_km`,
the maximum over its pairs; zero for a seam cell that links no pair. This
is the `strain_rate` the C04.1 slip-rate law reads on seam cells. Intact
cells are rigid and have zero strain rate and no damage. The excess is
over `yield_strain_per_myr` from step 1 as before; the healing, the exact
integrator, and the floor are unchanged. `work_damage = 1` under
`seams = 2` uses the power `kappa(S_s) * slip_rate ** 2` at seam cells
against `power_yield` from step 1, for the record; the default is 0.

## 4. Seams on markers

`engine/history/markers.py`. A seam is a set of markers, each with a
position in cell units (float, periodic) and a strength `s`. The strength
raster is rebuilt every step: `S = 1` everywhere, then for each cell that
holds at least one marker, `S = min(s)` over its markers. There is no
advection of `S` under `seams = 2`.

- **Creation.** Nucleation and tip advance create one marker at the cell's
  centre with `s = SEAM_OPEN_STRENGTH`, using the C04 rules on the raster.
- **Damage and healing** act on each marker through its cell's slip rate,
  so two markers in one cell see the same rate. A marker whose `s` reaches
  `WEAK_THRESHOLD` is removed; its cell is intact again unless another
  marker holds it.
- **Motion.** Each marker moves by `u(cell) * STEP_MYR / cell_km`, with `u`
  the seam-cell velocity of §1.3, and wraps. Markers never duplicate. A gap
  can open at a junction where the two walls' mean velocities differ; the
  gap cell is intact and loaded, the tips on either side see it, and the
  tip rule closes it in the next pass. Count gaps closed per step and
  report them.
- **Order** within a step: raster from markers; pieces; rigid solve;
  mismatch solve and stress; calibrations at step 1; damage and healing on
  markers; tip passes and nucleation (creating markers); marker motion.

Vectorize everything on the marker arrays; a marker count of tens of
thousands is one array.

## 5. The record and the views

`History` gains per-step `piece_count`, `largest_piece_share`,
`force_residual_max`, `marker_count`, `gaps_closed`, `wrapping_pieces`;
`Epoch` gains `mismatch` (n x n magnitude of `m`) beside `stress`. The
exploration worker returns the per-step `piece_count` and
`largest_piece_share` lists so the report can find the step at which a
loop first cut a piece of a given size.

Views, both adapters, after `intact_strength`: `mismatch` (scalar ramp) and
`pieces_motion`: the plates categorical with each piece's `v` drawn as one
arrow from its centroid, the same arrow style the `velocity` view uses,
so the reader sees bodies moving. The layer audit is not run; on record.

## 6. The lab and the search

- Lab: `seams` dial hi becomes 2, default 2 in the lab, promise gains
  "2: rigid pieces; stress is the integral of the unmatched drag over a
  piece; seams on markers."
- Search `Space`: fixed `seams = 2`. Everything else as C04.1 left it.
- `modernize_dials` unchanged; `seams` is a fixed setting, not a dial.

## 7. Tests

`tests/test_rigid.py`:

- one piece under a uniform drive moves at the drive with zero rotation
  and zero residual;
- one disc-shaped piece under a pure rotation field about its centre gets
  `omega` equal to the field's angular rate to 1e-6 and `v` zero;
- two pieces separated by a seam at `S = STRENGTH_MIN` under opposite
  uniform drives move nearly independently (each within 5 % of its own
  drive); the same at `S = 0.49` move together (their velocities within
  1 % of each other), because `kappa(0.49)` is thousands of times
  `kappa(0.05)`;
- a piece that wraps the torus has `omega = 0` and is counted;
- the mismatch has zero net force and torque per piece to 1e-8;
- the assembled system on a 3-piece hand-built mask is symmetric where it
  should be and solves.

`tests/test_markers.py`:

- a straight seam of markers under a uniform velocity of 0.3 cells per step
  for 75 steps is exactly one cell wide at every step and has moved 22 or
  23 cells; the same under a velocity that jumps by 1 cell per step across
  the seam does not duplicate;
- a marker healed past the threshold is removed and its cell is intact;
- two markers in one cell give the cell the lower strength;
- a gap opened by hand between two loaded seam segments is closed by one
  tip pass.

`tests/test_seams.py`: the `seams = 0` gate; `seams = 1` pins unchanged;
`seams = 3` refused. A 128-px run at `seams = 2` is deterministic and
converges. The full suite must pass:

```
py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"
```

## 8. Cost

Seconds per world at 1024 px on the pool for the twelve seeds at
`seams = 2` and `seams = 0`, same session. If the ratio exceeds 2.0,
profile and say where the time goes, and do not change the physics to make
the number.

## 9. Run it once

1. **The twelve seeds**, C04 §7.1 dials, `seams = 2`, `work_damage = 0`.
   Per world: the C04.1 table's columns, plus `piece_count` at the end,
   `largest_piece_share` at the end, the step at which `largest_piece_share`
   first fell below 0.9 and below 0.7, the size of the first piece a loop
   enclosed, the maximum force residual, wrapping-piece count, gaps closed.
   Every view sheet to `out/c04_2_twelve_<view>.png`.
2. **The probe**, 40 cells, 4 seeds, stage 1 only, through the library, to
   `out/search/`. Term pass rates, passers, failure-mode counts under the
   five modes of §0. Stop after stage 1.
3. **The comparison row.** The same twelve seeds at `seams = 1` are in
   `out/C04_1_BUILD_REPORT.md`; put the two tables' `plate_count`,
   `largest_piece_share` (compute it for the C04.1 worlds from their
   final strength if the report lacks it, by rerunning those twelve at
   `seams = 1`, which is a minute) and `edge_fraction` side by side.

## 10. Documents

`EXPLORE.md`, `SEARCH.md`: the lines. `STATUS.md` "Now": one paragraph,
the C04.1 outcome in one sentence, C04.2's against P1 – P5 once measured,
the failure mode named if P2 failed. `ROADMAP.md` if it carries the table.

## 11. Report

`out/C04_2_BUILD_REPORT.md`: what was built; deviations; check output
verbatim; cost; the twelve-seed table verbatim; the comparison row; the
probe; predictions with deciding numbers and "held" or "failed"; the named
failure mode; observations with evidence and no proposed values. If loops
cut pieces, say at which step and whether the pieces then hold, merge, or
fragment further. If cracks still mesh in a band, say whether the stress
field's band moved when the first piece split, with the `stress` sheets at
two steps as the evidence.
