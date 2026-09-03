# Work order — C04.1: a slipping seam stays weak

Issued 2026-09-02 for Opus 5. Follows [`WORK_ORDER_C04.md`](WORK_ORDER_C04.md)
and its report at `out/C04_BUILD_REPORT.md`. Same rules as C04, including:
isolation from `pipeline_a`, `pipeline_b`, `pipeline_d`; no commit; do not
edit `DESIGN.md`; do not touch the search server on port 5004 beyond
`GET /api/status`; no aesthetic language; one mechanism change.

## 0. Purpose

C04 built the seam formulation and measured it on the twelve development
seeds. Width held by construction: `edge_fraction` 0.93 to 0.997 on
eleven seeds, cost 1.4 times the sheet. Plates did not form, and neither
predeclared failure mode occurred. What happened instead: cracks
nucleated, ran a median of 5 to 8 cells, at most 30 to 47, and **healed
shut before they met**. Between 1,282 and 3,145 cells were opened per run
and only 193 to 1,027 were still open at the end.

The cause is the seam damage law. C04 damages a seam by the work its slip
dissipates, the C03.8 law. A fully open seam has stiffness
`KAPPA0 * STRENGTH_MIN ** exponent`, carries almost no traction, and so
dissipates almost nothing however fast it slips. Its damage rate is near
zero and healing wins in two or three steps at the corner's 10 Myr. The
law cannot keep an open fault open. That was the right law for the sheet,
where it stopped zones widening; on a seam it is the wrong one, because a
seam does not widen and does need to persist while it slips.

The physics the seam needs is the ordinary one for faults: **a fault that
is slipping stays weak, and a fault heals only when it stops.** Rate-and-
state friction says so; so does the sheet's own history, where the strain-
rate law kept slipping zones weak so well that they ran away. On a seam,
running away is impossible: damage cannot leave the seam cell. So this
order makes the seam's damage depend on its **slip rate**, which is the
strain-rate excess law the engine already has, confined to seam cells.

**One mechanism change.** Under `seams = 1`, `work_damage` is honoured:
`0` damages a seam by strain-rate excess over `yield_strain_per_myr`, `1`
by work excess as C04 did. The engine's default `work_damage` stays 0, so
the seam formulation's default becomes the slip-rate law; `seams` stays 0
in production and production is byte-identical.

**Predictions on record**, twelve seeds at 1024 px, the C04 §7.1 dials
with `work_damage = 0`:

- P1. Persistence: cells still open at the end are at least half the cells
  opened during the run, in at least eight seeds (C04: 15 to 33 %).
- P2. Reach: the longest crack at the end is at least 100 cells in at least
  eight seeds (C04: 30 to 47).
- P3. At least four of the twelve seeds end with two or more plates.
- P4. `edge_fraction` at least 0.9 in every seed except where the
  advection artifact of C04 §8 acts, and there at least 0.7.
- P5. `weak_final` between 0.005 and 0.10 in every seed.

Failure modes, counted so the report names one if P3 fails: **crazing**
(`network_share < 0.5` with `weak_final > 0.05`), **dead ends**
(`plate_count == 1` with `network_share > 0.9`), and the C04 mode,
**heals before meeting** (`weak_final < 0.01` with persistence under 0.5).

## 1. Engine

In `engine/history/seams.py` and `run_history`: under `seams = 1`, the
damage rate on seam cells is

```python
if params.work_damage:
    excess = np.maximum(power / power_yield - 1.0, 0.0)          # C04
else:
    excess = np.maximum(strain_rate / yield_strain_per_myr - 1.0, 0.0)
rate = damage_rate * excess * excess * seam_mask
```

with `strain_rate` the block-lifted invariant the sheet uses and
`yield_strain_per_myr` read at step 1 exactly as the sheet reads it (the
same percentile, on the solve grid, before any damage). Intact cells still
never damage. Healing, tips, nucleation, advection: unchanged. The
docstring that said `work_damage` is not consulted under seams is
corrected.

Nothing else moves. The corner's `heal_time_myr` of 10 is kept for the
twelve-seed run so the comparison with C04 is one change; the search's
range is widened in §3 because a persisting seam's healing time is a
different question from the sheet's.

## 2. Tests

`tests/test_seams.py` gains:

- under `seams = 1, work_damage = 0`, a seam cell with strain rate at
  three times yield loses strength at `damage_rate * 4` per Myr before
  healing, and an intact cell with the same strain rate loses none;
- under `seams = 1, work_damage = 1`, the C04 numbers are unchanged (pin
  one 128-px run's `weak_final` from before this edit);
- `seams = 0` byte-identity gate, unchanged.

The full suite must pass:

```
py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"
```

## 3. The lab and the search

- Lab: the `work_damage` dial's promise gains a clause: "Under seams, 0 is
  the slip-rate law that keeps a slipping fault weak; 1 is the work law,
  under which an open fault heals."
- Search `Space`: fixed `work_damage` default becomes **0**;
  `heal_time_myr` range becomes 5 – 200 (log); everything else as C04
  left it. `SEARCH.md`'s corner table follows.

## 4. Run it once

1. **The twelve seeds**, exactly C04 §7.1 with `work_damage = 0`. The same
   per-world table, plus per world: cells opened over the run, cells open
   at the end, persistence (their ratio), longest crack at the end in
   cells, and the step at which `plate_count` first reached 2 if it did.
   Every view sheet to `out/c04_1_twelve_<view>.png`.
2. **The probe**, exactly C04 §7.2 with the new search defaults: 40 cells,
   4 seeds, stage 1 only, through the library, to `out/search/`. Term pass
   rates, passers, failure-mode counts under the three modes above.
3. Seconds per world at `work_damage = 0` and `1` under seams, same pool,
   same session.

## 5. Documents

- `EXPLORE.md`, `SEARCH.md`: the lines above.
- `STATUS.md` "Now": one sentence on C04's outcome (healed before meeting)
  and one on C04.1's against P1 – P5, once measured.

## 6. Report

`out/C04_1_BUILD_REPORT.md`: what was built; deviations; check output
verbatim; the twelve-seed table verbatim with the new columns; the probe;
the predictions with deciding numbers and "held" or "failed"; the named
failure mode if P3 failed; observations with evidence and no proposed
values. If loops close, say at which step and whether the plate count then
holds or the seam heals again. If seams widen anywhere beyond the C04
advection artifact, say where and how.
