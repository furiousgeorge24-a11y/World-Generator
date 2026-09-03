# Work order — C04.3: a crack inside a piece slips

Issued 2026-09-03 for Opus 5. Follows [`WORK_ORDER_C04_2.md`](WORK_ORDER_C04_2.md)
and its report at `out/C04_2_BUILD_REPORT.md`. Same rules as C04.2: isolation
from `pipeline_a`, `pipeline_b`, `pipeline_d`; no commit; do not edit
`DESIGN.md`; do not touch the search server on port 5004 beyond
`GET /api/status`; no aesthetic language; one mechanism change.

## 0. Purpose

C04.2 built rigid pieces, the integrated internal stress, and seams on
markers. All three did what they were built to do: force residuals of
1e-13, edge fraction exactly 1.0 on every seed, cost 1.1 times the sheet,
and the crumb fragments of C04.1 gone with the advection duplicate. Plates
still did not form. The failure mode was **heals before meeting** on all
twelve seeds, and the report measured the cause exactly:

> a seam cell slips only where it links two distinct pieces, so a crack
> inside a piece has zero slip, zero damage, and nothing opposing healing.

On seed 2075014389, 212 of 4,256 marker-steps ever exceeded the yield slip
rate, and 37 of 75 steps had none. A marker with no excess crosses the
weak threshold in two steps, so every crack gets eight tip advances to
reach another before it seals.

That is the rigid idealization taken one step too far. A crack inside an
elastic plate is not rigid across its faces: the faces displace relative
to each other under the load the plate carries, by an amount that grows
with the crack's length and the stress on it. That is the whole reason a
crack concentrates stress at its tip and runs. C04.2 already computes
that displacement rate. The internal-stress solve of C04.2 §2 returns
`w`, the non-rigid part of the velocity, and at a seam cell the sheet's
own `effective_gradients` of `w` is the slip rate the crack's faces have
relative to each other under the piece's load, in the same units as the
rigid slip. C04.2 threw it away for the damage law and kept only the
rigid jump between distinct pieces.

**One mechanism change.** The slip rate a seam cell's damage law reads
becomes the rigid slip rate of C04.2 §3 **plus** the elastic strain-rate
invariant of `w` at that cell, block-lifted as the sheet lifts
`strain_rate`. A crack between two pieces slips by the rigid jump and the
elastic part; a crack inside one piece slips by the elastic part alone,
and it is loaded exactly where the stress that drives its tip is highest.
Nothing else moves: tips, nucleation, healing, markers, the rigid solve,
the internal-stress solve, the views.

**Predictions on record**, twelve seeds at 1024 px, the C04 §7.1 dials,
`seams = 2`, `work_damage = 0`:

- P1. Persistence rises above 0.5 in at least eight seeds (C04.2: 0.03 to
  0.08) and the longest crack exceeds 200 cells in at least eight (C04.2:
  21 to 90).
- P2. At least six seeds end with two or more plates, and on at least six
  the largest piece holds no more than 70 % of the world.
- P3. On at least six seeds the first loop that closes encloses at least
  655 cells.
- P4. Cost within 1.3 times the sheet.
- P5. In the 40-cell probe, at least 10 % of worlds pass `plate_count`.

Failure modes, counted: **crumbs**, **crazing**, **dead ends**, **heals
before meeting**, **locked**, as C04.2 defines them, and one more the new
law could produce, **runaway** (`weak_final` above 0.25 on the twelve, or
`network_share` above 0.9 with `plate_count` of 1 and `weak_final` above
0.10: a network so dense nothing separates).

## 1. Engine

In `run_history` under `seams = 2`, after the internal-stress solve:

```python
elastic_rate = to_kinematic_blocks(strain_rate_of(w_solved, kappa_s), n)  # the sheet's invariant on w
slip_rate = rigid_slip_rate + elastic_rate * seam_mask
```

and `slip_rate` is what `seams.damage_excess` reads as `strain_rate`.
`yield_strain_per_myr` is read at step 1 from the same invariant as
before, which at step 1 is this one, so the calibration is unchanged.
Record per step the share of seam markers above the yield slip rate, the
mean elastic slip rate over seam cells, and the mean rigid slip rate over
seam cells, so the report can say which part carried the damage.

`seams = 0` and `seams = 1` are untouched; `seams = 2` at
`work_damage = 1` uses `kappa(S_s) * slip_rate ** 2` with the new
`slip_rate`, for the record.

## 2. Tests

`tests/test_rigid.py` gains: a single piece with a straight internal
crack of 20 cells under a shear drive across it has an elastic slip rate
on the crack cells above the yield read from the same field, and zero on
the intact cells beside it; the same crack with its cells set intact has
no seam slip anywhere. `tests/test_seams.py`: the `seams = 0` gate and the
`seams = 1` pin, unchanged. The full suite must pass:

```
py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"
```

## 3. Run it once

1. The twelve seeds, C04.2 §9.1's table with its columns, plus the three
   new per-run means; every view sheet to `out/c04_3_twelve_<view>.png`.
2. The probe: 40 cells, 4 seeds, stage 1 only, through the library, to
   `out/search/`; term pass rates, passers, the six failure-mode counts.
3. Seconds per world at `seams = 2` and `seams = 0`, same pool, same
   session.

## 4. Documents and report

`STATUS.md` "Now": one sentence on C04.2's outcome and one on C04.3's.
`out/C04_3_BUILD_REPORT.md`: what was built; deviations; check output;
cost; the twelve-seed table; the probe; predictions with deciding numbers;
the named failure mode; observations with evidence and no proposed
values. If loops now cut pieces, say at which step, what the pieces then
do over the remaining steps, and whether the plate count holds. If cracks
run away, say at which step the seam share crossed 0.10 and what the
elastic slip rate was doing.
