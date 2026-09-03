# Work order — C03.1: performance run

Issued 2026-09-01 for Opus 5. Follows [`WORK_ORDER_C03.md`](WORK_ORDER_C03.md)
and the build report at `out/C03_BUILD_REPORT.md`. The same rules apply:
isolation from `pipeline_a`, `pipeline_b`, `pipeline_d`; NumPy, Pillow, and
the standard library only; determinism; no certificates or review apparatus;
do not commit; every constant in `engine/history/constants.py`.

## 0. Purpose, and what this run is not

The C03 build works and takes 110 s for a 1024 px world. This run makes it
take under 5 s. **It changes nothing that shapes the physics.** The
reference-strain error the build report found in §9.1 is real and is fixed
in the *next* run, C03.2, as the single mechanism change of that run. If you
fix it here, the timing evidence and the localization evidence become
inseparable and the author cannot tell which change did what.

So the expected outcome of this run is the same as the last: no
localization, one plate per world, strength rising toward 1. That is the
regression check. The ⛔ localization stop of the first order does **not**
apply to this run; a weak fraction of zero is the expected result.

## 1. Definition of done

1. `webui_adapter.generate` for seed `4287772760` at 1024 px, scale 5,
   completes in **under 5 s** on this machine, measured as the adapter's
   own `elapsed_s` plus the time to render every view.
2. The regression checks in §5 pass.
3. The full test suite and the two kept suites pass.
4. Documents in §6 are updated.
5. The report in §7 is written.

## 2. The changes

Make all of them. Each is a discretization, a grid, a solver setting, or a
rendering detail. None is a mechanism.

### 2.1 Kinematic grid at eight delivered pixels per cell

In `constants.py`: `CELL_PX = 8`, `MIN_HISTORY_N = 128`.

In `geometry.py`, generalize `history_n` so it no longer assumes `CELL_PX`
is four:

```
history_n = max(2 * pixels // CELL_PX, MIN_HISTORY_N)
```

(`2 *` is the parent-to-window ratio; if you prefer, add
`PARENT_WINDOW_RATIO = 2` to constants and use it.) Everything else in
`WorldGeometry` follows. New expected values for `tests/test_geometry.py`:

| pixels | scale | history_n | cell_km | parent_km | window_km | window_cells |
|---|---|---|---|---|---|---|
| 1024 | 5 | 256 | 40 | 10240 | 5120 | 128 |
| 512 | 5 | 128 | 40 | 5120 | 2560 | 64 |
| 128 | 5 | 128 | 40 | 5120 | 640 | 16 |
| 2048 | 5 | 512 | 40 | 20480 | 10240 | 256 |
| 1024 | 20 | 256 | 160 | 40960 | 20480 | 128 |
| 2048 | 20 | 512 | 160 | 81920 | 40960 | 256 |

The parent floor is unchanged at 5,120 km at the default scale. The
sampler's lattice divisibility still holds (`parent_m` is a multiple of
128 for every row above; keep the test that checks it).

Why this is acceptable: at the default scale a kinematic cell is now 40 km,
so boundary zones resolve at 40–80 km. Real plate boundary zones are 50 to
200 km wide. The crust stage (C6) rides on markers and is not bound to this
grid.

### 2.2 Solve velocity on a grid half the kinematic grid

The velocity is smooth by construction (the build report measured 90 % of
its power at one cycle across the parent), so solving it at half resolution
loses nothing the damage law can see.

In `kinematics.py`, per step:

1. `kappa` on the kinematic grid as now.
2. `kappa_s = restrict_kappa(kappa)` — the existing harmonic 2 × 2 mean
   with the quarter factor is exactly the right operator for this.
3. `D_s = restrict(traction)` — the existing 2 × 2 mean.
4. `v_s = solve(D_s, kappa_s, u0=v_s_prev)` — warm start on the solve grid.
5. `velocity = prolong_bilinear(v_s)` — a **new** function, periodic
   bilinear interpolation from the `(2, n/2, n/2)` solve grid to the
   `(2, n, n)` kinematic grid, with cell centres aligned (fine cell `i` sits
   at coarse coordinate `(i + 0.5) / 2 - 0.5`). Put it in `domain.py`. It is
   deliberately **not** the solver's internal `prolong`, which must stay
   piecewise constant for the adjointness reason recorded in the solver's
   docstring. Name and document them so nobody confuses the two.

Strain, damage, advection, and views continue on the kinematic grid.

Add `SOLVE_GRID_DIVISOR = 2` to constants. `MG_COARSEST = 8` still fits:
the smallest solve grid is 64.

### 2.3 Solver tolerance

`MG_TOL = 1e-3`. The forcing changes slowly per step and the solve is
warm-started; a residual of a tenth of a percent is far below anything the
damage law responds to. Expect one to two cycles per step. `MG_MAX_CYCLES`
stays at 20.

### 2.4 Seventy-five steps of four million years

`STEP_MYR = 4.0`, so the default is 75 steps. Epoch indices become 19, 38,
56, 75, distinct. Displacement per step is at most about four cells; the
semi-Lagrangian advection handles that.

With the longer step the explicit Euler damage update is no longer safe, so
replace it with the exact solution of the per-cell linear ODE over the step,
which is stable at any step length and reduces to the current update as
`dt → 0`:

```
rate   = DAMAGE_RATE * (strain_rate / strain_ref) ** 2       # 1/Myr
total  = HEAL_RATE + rate
S_eq   = HEAL_RATE / total
S_new  = S_eq + (S - S_eq) * exp(-total * dt)
S_new  = clip(S_new, STRENGTH_MIN, 1.0)
```

This is the same law, integrated exactly instead of approximately. Leave
`strain_ref` exactly as it is: `DRIVE_RMS_KM_PER_MYR / (2.0 * cell_km)`.
That is the C03.2 change, not this one.

### 2.5 Plate labelling by pointer jumping

`label_plates` in `plates.py` currently propagates the minimum label to
neighbours until nothing changes, which takes a number of rounds
proportional to the plate's diameter. Replace the loop body with two
alternating operations until nothing changes:

1. **Neighbour minimum**, as now: for each of the four rolls, where both
   cells are strong, `labels = minimum(labels, rolled)`.
2. **Pointer jumping**: `labels = flat[labels]` where `flat` is the current
   label array raveled, with the sentinel index mapped to itself. Repeat
   this inner step until `flat[labels] == labels` everywhere. Labels are
   flat cell indices, so this follows each cell to the cell its label names
   and adopts that cell's label; it collapses chains in logarithmic rounds.

Renumbering by area, largest first, is unchanged. The existing
`tests/test_plates.py` cases must pass unchanged; add one more with a
single winding weak line that snakes across the whole grid (a long thin
strong corridor) and assert the labelling agrees with the old
implementation, which you keep as `_label_plates_reference` inside the test
file for that comparison only.

### 2.6 Label the final epoch only

In `webui_adapter.generate`, run `label_plates`, `boundary_mask`, and
`regime` on the **final** epoch only. For the three earlier epochs keep the
weak mask and nothing else.

Views: remove `plates_t25`, `plates_t50`, `plates_t75`. The
`boundaries_tNN` views become the weak mask at that epoch; update their
`view_purposes` text to say so ("Weak lithosphere a quarter of the way
through; plate contacts are labelled at the final epoch only."). `VIEWS`
now has 19 entries. Update `tests/test_adapter.py` and the view table in
`README.md`.

### 2.7 Rendering

`image.save(buffer, format="PNG", optimize=False)`. Nothing else.

### 2.8 The audit runner

`run_layer_audit.py` computes its panel size from the smallest candidate
view. With the 1024 px default now producing 256-cell views, `build` at the
default pixels gives 256 px panels, which is fine. Check that
`py -3.14 pipeline_c/run_layer_audit.py build --seed 4287772760` succeeds
on the seven non-constant views (the three constant views still cannot be
batched; pass `--views` with the same seven the build report used). Do not
run the judge this time; the audited fields have not changed in kind.

## 3. Timing targets

Seed `4287772760`, adapter `generate` plus rendering every view, wall time:

| Delivered | Scale | Target |
|---|---|---|
| 128 px | 5 | ≤ 2.5 s |
| 512 px | 5 | ≤ 2.5 s |
| 1024 px | 5 | **≤ 5 s** |
| 1024 px | 20 | ≤ 5 s |
| 2048 px | 5 | ≤ 20 s (allowed, not promised) |

If 1024 px lands between 5 and 8 s after all of §2, profile and report; do
not add a change this order does not list. If it lands above 8 s, something
in §2 was not applied as written; find it.

## 4. Order of work

1. §2.1 geometry and constants, with the new test table. Run `test_geometry`.
2. §2.5 labelling, with the reference comparison test. Run `test_plates`.
3. §2.2, §2.3, §2.4 in `kinematics.py`, `domain.py`, `constants.py`. Run
   `test_solver` and `test_history`; adjust any test that pinned the old
   step count or grid, and say which in the report.
4. §2.6, §2.7 adapter and views. Run `test_adapter`.
5. §5 regression checks.
6. §2.8, §3, §6, §7.

## 5. Regression checks

These separate "faster" from "different". Put them in
`tests/test_regression_c03_1.py` so they stay runnable.

1. **Half-grid solve.** For seed `4287772760` at 1024 px, take the initial
   strength field, build `kappa`, and solve the drive at `t = 0` both on the
   kinematic grid (old path, `solve(traction, kappa)`) and via §2.2. The
   RMS difference of the two velocity fields on the kinematic grid must be
   below **5 %** of the RMS speed. Report the number.
2. **Tolerance.** Same problem, solved on the solve grid at `MG_TOL = 1e-5`
   and at `1e-3` (pass the tolerance as a keyword parameter to `solve` with
   the constant as default). RMS difference below **1 %** of RMS speed.
   Report the number.
3. **Damage integrator.** For a random strength field and strain field of
   the magnitudes the build report measured (strain rate mean 0.017, max
   0.06 /Myr), one exact step of `dt = 2` must agree with one Euler step of
   `dt = 2` to within **1e-3** in strength everywhere. Report the maximum
   difference.
4. **Qualitative outcome.** Full default runs for seeds `4287772760`,
   `2075014389`, `1833546021` at 1024 px: `weak_fraction_final == 0`,
   final strength mean above 0.97, final strength minimum above 0.9,
   `plate_count == 1`. This is the "same result, faster" check and is
   expected to pass. If any of the three localizes, that is a finding
   about a change in §2, not a success; report it and do not proceed to
   the documents until you have found which change did it.
5. **Determinism.** Two full runs of seed `7` at 512 px are byte-identical
   in every epoch's strength and velocity.

## 6. Documents

- `README.md`: timing sentence, view table (19 views), nothing else.
- `STATUS.md`: "Now" gains one sentence on the timing; "Verification"
  updated from your final run. Nothing else.
- `DESIGN.md` §2 and §10 have already been amended by the author's side for
  this run; do not edit `DESIGN.md`.

## 7. Report

`out/C03_1_BUILD_REPORT.md`, sections:

1. **What changed**, file by file, one line each.
2. **Deviations** from this order, with reasons.
3. **Check output**, verbatim final lines of the full test suite, the two
   kept suites, and the audit `build`.
4. **Timing**, the table of §3 with measured values, plus mean and max
   solver cycles per step at 1024 px.
5. **Regression**, the five numbers or outcomes of §5.
6. **Profile** at 1024 px by phase, one run, the same format as the last
   report's §10.
7. **Contact sheets**: `out/plates_1024.png`, `out/strength_1024.png`, and
   `out/velocity_1024.png`, twelve seeds each at 1024 px. They are now
   cheap. No commentary on how they look.

Do not describe any field as natural, plausible, good, or bad. Do not
touch `strain_ref`, `DAMAGE_RATE`, `HEAL_RATE`, `HOMOG_LENGTH_FRACTION`,
`STRENGTH_EXPONENT`, or the drive constants.
