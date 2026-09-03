# Work order — C03.2: the reference strain

Issued 2026-09-02 for Opus 5. Follows [`WORK_ORDER_C03_1.md`](WORK_ORDER_C03_1.md)
and its report at `out/C03_1_BUILD_REPORT.md`. Same rules: isolation from
`pipeline_a`, `pipeline_b`, `pipeline_d`; NumPy, Pillow, standard library;
determinism; no review apparatus; do not commit; do not edit `DESIGN.md`.

## 0. Purpose

This run makes **one** change to the mechanism and measures what it does.
The C03 build report §9.1 found that the reference strain rate was defined
at a two-cell length while the velocity solve homogenizes over `parent / 8`,
so the strain the solver permits is about sixteen times smaller than the
reference and damage never fired. The fix is to define the reference at the
length the solver actually uses.

Nothing else changes. Not `DAMAGE_RATE`, not `HEAL_RATE`, not the exponent,
not the drive, not the homogenization length, not any grid or solver
setting. If the result is wrong in some other way, that is the next run's
single change, decided by the author after looking.

## 1. The change

In `engine/history/kinematics.py`, `run_history`, replace

```python
strain_ref = DRIVE_RMS_KM_PER_MYR / (2.0 * cell_km)
```

with

```python
# The strain the solve permits: the drive speed developed over the
# homogenization length, not over two cells.
strain_ref = DRIVE_RMS_KM_PER_MYR / (HOMOG_LENGTH_FRACTION * geometry.parent_km)
```

importing `HOMOG_LENGTH_FRACTION` from constants. At 1024 px and scale 5
that is `40 / 1280 = 0.03125 /Myr`; on the 5,120 km floor it is `0.0625`.
The first build measured a strain-rate maximum of about `0.06` and a mean of
about `0.017` on a comparable world, so the ratio squared now reaches order
one at the strongest strain instead of `4e-3`.

That is the whole change to the engine.

## 2. Prediction, stated before running

Written so the run is judged against it and not rationalized afterwards.

- `weak_fraction` rises from zero within the first 50 Myr, reaches somewhere
  between 0.05 and 0.30, then declines or plateaus between 0.05 and 0.20 as
  plate interiors, whose strain drops once boundaries take the deformation,
  heal while the zones stay weak.
- Three to eight plates with area above 1 % per world at the end.
- Boundaries lie along the drive field's basin edges: where the smoothed
  velocity converges, diverges, or shears most. Because the velocity is
  90 % one wavelength, expect the plates to look like equant convection
  cells of one scale on this first look. That is expected and is the next
  run's subject, not this one's.
- Regime varies along a boundary where the boundary curves relative to the
  drive.
- Boundaries move with the plates and are not pinned; the three epoch weak
  masks differ from each other.

Failure modes the stops catch:

- ⛔ `weak_fraction_final < 0.01` on seed `4287772760` at 1024 px: damage
  still cannot fire. Stop, report the trajectory and the strain statistics.
- ⛔ `weak_fraction` above 0.5 at any step on that seed: runaway. Stop,
  report the trajectory and the epoch at which it crossed.

Either stop is a result; do not tune past it.

## 3. Tests

- `tests/test_regression_c03_1.py` check 4 asserted the pre-fix outcome (no
  localization). Delete that test; it pinned an error. Add in its place a
  test in `tests/test_history.py` that `run_history` uses a reference
  strain equal to `DRIVE_RMS_KM_PER_MYR / (HOMOG_LENGTH_FRACTION * parent_km)`
  (expose it on `History` as `strain_ref` so it can be asserted, and put it
  in the adapter report as `strain_ref_per_myr`).
- Everything else in the suite must pass unchanged. If a test fails because
  the world now localizes (for instance a determinism budget), say which and
  fix the test, not the engine.

## 4. Measurements

Seed set: the twelve in `STATUS.md`. All at 1024 px, scale 5, unless said.

1. **Per-step trajectory** of `weak_fraction` for seeds `4287772760`,
   `2075014389`, `1833546021`, written to `out/c03_2_trajectories.csv`
   (columns `seed, step, t_myr, weak_fraction, strength_mean, strength_min,
   strain_rate_mean, strain_rate_max`). Record these inside `run_history`
   on `History` as lists; it costs nothing.
2. **Per-seed summary** for all twelve: `plate_count`, `plate_area_percent`
   (descending), `weak_fraction_final`, peak `weak_fraction` and the step
   it occurred, boundary cell fraction, and the share of weak cells that are
   divergent, convergent, and shear. Put the shares in the adapter report as
   `regime_share`.
3. **Contact sheets**, twelve seeds each: `plates_1024.png`,
   `boundaries_1024.png`, `regime_1024.png`, `strength_1024.png`,
   `velocity_1024.png`.
4. **Time-lapse** for seed `4287772760`: extend `tools/contact_sheet.py`
   with a `--views` comma list that, given one seed, tiles those views in
   a row. Produce `out/timelapse_4287772760_strength.png` with
   `strength_t25,strength_t50,strength_t75,strength` and
   `out/timelapse_4287772760_weak.png` with
   `boundaries_t25,boundaries_t50,boundaries_t75,boundaries`.
5. **Timing** at 1024 px for seed `4287772760`, generate plus render. Report
   solver cycles mean and max; localization raises the coefficient contrast
   and may raise the cycle count. The 5 s target still applies; if it is
   exceeded, report it and do not change solver settings.

## 5. The audit, run properly

`VIEWS.md` makes this the gate, and for the first time the categorical views
have content. After the measurements:

1. `py -3.14 pipeline_c/run_layer_audit.py build --seed 4287772760` with the
   default view list (all ten, now that `plates`, `boundaries`, and `regime`
   vary). If a view is still constant on this seed, drop only that view and
   say so.
2. Follow the printed judging plan: one fresh-context subagent per call,
   given the prompt file and only that call's panel images. Do not open the
   hidden root. Do not give the judge a directory listing or any source.
3. Concatenate the verdicts, `score`, `verify`.
4. Report the scorer's output verbatim, every candidate finding with the
   judge's reproducing rule and evidence, and whether the batch is void.

A finding is for the author. Do not fix anything it names. One audit run,
no reseeding to get a non-void batch.

## 6. Documents

- `STATUS.md`: "Now" and "The open question" rewritten to say what this run
  found, in the same register as before, no more than a paragraph each;
  "Verification" from your final run. Do not characterize the plates as
  good or bad.
- `README.md`: nothing, unless the view list or timing changed.

## 7. Report

`out/C03_2_BUILD_REPORT.md`:

1. **The change**, as applied, with the resulting `strain_ref` at each
   supported grid.
2. **Deviations**.
3. **Stops**, if either fired, with the evidence §2 asks for.
4. **Check output**, verbatim summary lines of every suite and the audit
   build.
5. **Prediction versus outcome.** Each bullet of §2 restated, then what was
   measured, then one of `met`, `not met`, or `cannot tell`. No
   interpretation beyond that.
6. **Measurements** per §4, tables and file paths.
7. **Audit** per §5.
8. **Observations.** Anything the author should know before looking, with
   evidence. You may name a constant or mechanism a measurement implicates.
   Do not propose values.

Do not describe any field as natural, plausible, good, or bad, and do not
compare anything to the reference images.
