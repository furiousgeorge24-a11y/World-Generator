# Work order — C03.3: a yield threshold in the damage law

Issued 2026-09-02 for Opus 5. Follows [`WORK_ORDER_C03_2.md`](WORK_ORDER_C03_2.md)
and its report at `out/C03_2_BUILD_REPORT.md`. Same rules: isolation from
`pipeline_a`, `pipeline_b`, `pipeline_d`; NumPy, Pillow, standard library;
determinism; no review apparatus; do not commit; do not edit `DESIGN.md`.

## 0. Purpose

Two runs have bracketed the outcome from opposite ends. With the reference
strain at a two-cell length nothing failed (stagnant lid). With it at the
homogenization length everything failed within 50 Myr (mobile lid), because
the law damages a little at any strain and the sub-threshold creep, plus the
loss of stiffness once cells start to fail, carried the whole field down.
Report C03.2 §8.1–8.3 has the measurements.

This run makes **one** mechanism change: damage requires exceeding a yield
strain rate. Below yield there is no damage at all and healing is
unopposed; above it damage grows quadratically with the excess. This is how
brittle lithosphere behaves, and it removes the sub-threshold creep that
made run 2 collapse. The yield is tied to the drive's own characteristic
strain rate, not to the homogenization length, because C03.2 §8.2 showed
that the effective homogenization length depends on the strength being
damaged, so a reference built from it is circular.

Nothing else changes: not `DAMAGE_RATE`, `HEAL_RATE`, `STRENGTH_EXPONENT`,
`HOMOG_LENGTH_FRACTION`, the drive, the noise, any grid or solver setting.
The grid-locking in the initial noise (C03.2 §8.5) and the 2048 px solver
non-convergence (§8.4) are known and are **not** addressed here.

## 1. The change

### 1.1 Constant

In `engine/history/constants.py`, add

```python
YIELD_STRAIN_FRACTION = 0.4      # yield strain rate as a fraction of the drive's characteristic strain rate
```

and remove nothing. `DAMAGE_RATE` keeps its value; its meaning becomes "the
damage rate when the strain rate is twice the yield", since the excess
ratio is then 1.

### 1.2 The yield strain rate

In `kinematics.py`, `run_history`, replace the `strain_ref` line with

```python
# The drive's characteristic strain rate: its RMS speed over its coarsest
# wavelength, as a rate. Lithosphere yields at a fraction of that.
drive_wavelength_km = geometry.parent_km / DRIVE_NODES_COARSEST
drive_strain_per_myr = 2.0 * math.pi * DRIVE_RMS_KM_PER_MYR / drive_wavelength_km
yield_strain_per_myr = YIELD_STRAIN_FRACTION * drive_strain_per_myr
```

At 1024 px and scale 5: wavelength 5,120 km, drive strain 0.0491 /Myr, yield
**0.0196 /Myr**. On the 5,120 km floor: 0.0393. Both scale as `1 / parent`,
as the solved strain does, so the yield sits at the same place in the strain
distribution at every size.

For reference, C03.2 measured the step-1 solved strain on seed `4287772760`
at 1024 px: median 0.0123, 99th percentile 0.0305, maximum 0.0486. The yield
sits near the 85th–90th percentile of that field.

Expose `yield_strain_per_myr` on `History` and in the adapter report as
`yield_strain_per_myr`; drop `strain_ref` from both.

### 1.3 The law

Replace the damage rate with

```python
excess = np.maximum(strain_rate / yield_strain_per_myr - 1.0, 0.0)
rate = DAMAGE_RATE * excess * excess
```

and keep the exact integrator exactly as it is:

```python
total = HEAL_RATE + rate
equilibrium = HEAL_RATE / total
S = clip(equilibrium + (S - equilibrium) * exp(-total * dt), STRENGTH_MIN, 1)
```

Where `rate == 0`, `equilibrium == 1` and the cell heals toward full strength
at `HEAL_RATE`. That is the intended behaviour: interiors recover on their
own.

That is the whole change to the engine.

## 2. Prediction, stated before running

- At step 1 on seed `4287772760` at 1024 px, between **5 and 15 %** of cells
  exceed the yield. Report the exact fraction.
- `weak_fraction` rises over the first 20–40 Myr and **plateaus between 0.05
  and 0.20**. It does not go to zero and it does not run away. Cells whose
  strain is between the yield and about 1.2 × yield have an equilibrium
  strength above the weak threshold and stay strong; only cells clearly above
  yield fail.
- Failed zones sharpen: as their strength falls, the velocity jump
  concentrates in them, their strain rises, and they stay failed. Their
  strong neighbours see a small strain halo from the bilinear prolongation,
  one to two cells wide, so zones settle at roughly **3 to 6 cells** wide
  (120–240 km at the default scale).
- **Three to eight plates** with area above 1 %.
- The weak set is **connected into a network**, not scattered flecks: the
  ridges of the drive's gradient magnitude form closed curves around its
  basins. Report the number of 8-connected weak components on the torus and
  the largest component's share of weak cells; I expect the largest to hold
  more than half.
- On this first look plates will read as equant cells of one scale, and
  some boundary segments will run along the grid axes because the initial
  noise is grid-aligned. Both are expected and are later runs' subjects.
- The three epoch weak masks differ: boundaries move with the plates.

Stops:

- ⛔ `weak_fraction_final < 0.01` on seed `4287772760`: the yield is above
  the field. Stop, report the step-1 exceedance fraction and strain
  percentiles.
- ⛔ `weak_fraction > 0.5` at any step on that seed: the halo cascade is
  real. Stop, report the trajectory and the step it crossed.

## 3. Tests

- `tests/test_regression_c03_1.py` check 3 compares the exact integrator
  against one Euler step of the *old* law. Rewrite it for the new law: same
  bound, same field magnitudes, the excess-squared rate.
- Add to `tests/test_history.py`:
  - a cell with strain below yield and strength 0.3 ends one 4 Myr step
    strictly stronger, and after 400 Myr of steps is above 0.98;
  - a cell with strain at three times yield and strength 1.0 ends one step
    strictly weaker;
  - `History.yield_strain_per_myr` equals the §1.2 formula for
    `WorldGeometry(1, 1024, 5)` to 1e-12, and the report carries it.
- Everything else must pass. If a test fails because worlds now localize,
  say which and fix the test.

## 4. Measurements

All twelve `STATUS.md` seeds at 1024 px, scale 5, unless stated.

1. **Step-1 exceedance.** Fraction of cells above yield at step 1 for the
   three trajectory seeds, plus the 50th, 85th, 90th, 95th, 99th percentiles
   of the step-1 strain rate.
2. **Per-step trajectory** for seeds `4287772760`, `2075014389`,
   `1833546021` to `out/c03_3_trajectories.csv`, same columns as C03.2 plus
   `exceed_fraction`.
3. **Per-seed summary** for all twelve: `plate_count`, `plate_area_percent`,
   `weak_fraction_final`, peak `weak_fraction` and its step,
   `boundary_cell_fraction`, `regime_share`, number of 8-connected weak
   components on the torus, largest component's share of weak cells, and
   median weak-zone width (estimate it as `weak_fraction × n² / boundary
   skeleton length`; if you have no cheap skeleton, report the fraction of
   weak cells that have at least one strong 4-neighbour, which is the zone
   edge fraction, and say what it implies for width).
4. **Contact sheets**, twelve seeds each at 1024 px: `plates_1024.png`,
   `boundaries_1024.png`, `regime_1024.png`, `strength_1024.png`,
   `velocity_1024.png`, `strain_rate_1024.png`.
5. **Time-lapse** for seed `4287772760`: `strength` and `boundaries` across
   the four epochs, as in C03.2.
6. **Timing** at 1024 px, generate plus render; solver cycles mean and max;
   whether any step exhausted `MG_MAX_CYCLES` at 1024 or 2048 px.

## 5. The audit, run properly

As in C03.2 §5, on the full default view list. If `plates`, `boundaries`,
or `regime` is still constant on seed `4287772760`, drop only that view and
say so. Fresh-context judges, panel images only, no hidden root, no source.
Concatenate, `score`, `verify`. Report the scorer output verbatim, every
candidate finding with the judge's rule and evidence, and whether the batch
is void. Findings are for the author; fix nothing they name.

## 6. Documents

- `STATUS.md`: "Now" and "The open question" rewritten for what this run
  found, a paragraph each, no more; "Verification" from the final run.
- `README.md`: nothing unless the report keys or timing changed.

## 7. Report

`out/C03_3_BUILD_REPORT.md`:

1. **The change**, as applied, with the yield at each supported grid.
2. **Deviations**.
3. **Stops**, if either fired, with the evidence.
4. **Check output**, verbatim summary lines.
5. **Prediction versus outcome**: each bullet of §2, what was measured,
   `met` / `not met` / `cannot tell`.
6. **Measurements** per §4.
7. **Audit** per §5.
8. **Observations**, with evidence, naming what a measurement implicates.
   No proposed values.

Do not describe any field as natural, plausible, good, or bad, and do not
compare anything to the reference images.
