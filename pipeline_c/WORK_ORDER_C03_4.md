# Work order — C03.4: strain from the stress a cell carries

Issued 2026-09-02 for Opus 5. Follows [`WORK_ORDER_C03_3.md`](WORK_ORDER_C03_3.md)
and its report at `out/C03_3_BUILD_REPORT.md`. Same rules: isolation from
`pipeline_a`, `pipeline_b`, `pipeline_d`; NumPy, Pillow, standard library;
determinism; no review apparatus; do not commit; do not edit `DESIGN.md`.

## 0. Purpose

Run 3 showed the yield threshold working on the first step and then failure
propagating outward from every zone at about one cell per step until the
whole field was weak. The cause is how the strain rate is computed at an
interface. The velocity jumps across a failed cell; the central difference
at the strong cell next door reaches across into the failed cell and charges
the strong cell with a share of that jump, a strain rate ten times the
yield. So the neighbour fails, and the front moves. The half-grid solve's
bilinear interpolation smears the jump further and widens the halo.

That is a discretization artifact, not rock behaviour. Across a material
boundary stress is continuous and strain rate is not; a strong block beside
a weak fault carries almost no stress because the fault carries almost
none. The correct strain rate for a cell is the stress it carries divided by
its own stiffness. The solver already computes exactly those stresses: its
edge fluxes use the harmonic-mean coupling and are continuous by
construction.

This run makes **one** change: the strain rate that drives damage is
computed from the edge fluxes, on the solve grid where the fluxes live, and
mapped to the kinematic grid block by block. In a uniform interior it is
identical to the current strain. At a strong cell beside a failed one it is
nearly zero. Inside the failed cell it is the full jump.

Nothing else changes: not the yield, the rates, the exponent, the
homogenization length, the drive, the noise, the grids, or the solver.

## 1. The change

### 1.1 Effective gradients on the solve grid

Add to `engine/history/solver.py`:

```python
def effective_gradients(u: np.ndarray, kappa: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell x and y gradients of `u` consistent with the edge fluxes.

    The flux across an edge is `k_ij * (u_j - u_i)` with `k_ij` the harmonic
    mean of the two cells' kappa, and it is continuous across a stiffness
    contrast. Dividing the mean of a cell's two opposing edge fluxes by the
    cell's own kappa gives the gradient the cell actually experiences. Where
    kappa is uniform this is the ordinary central difference. Where a strong
    cell touches a weak one, the harmonic mean is dominated by the weak side
    and the strong cell's share of the jump is nearly zero, which is what
    stress continuity requires.
    """
```

With `k_east, k_north = edge_coefficients(kappa)`, `k_west = roll_x(k_east, 1)`,
`k_south = roll_y(k_north, 1)`, and `u` of shape `(2, n, n)`:

```
g_x = 0.5 * ( k_east  * (roll_x(u, -1) - u) + k_west  * (u - roll_x(u, 1)) ) / kappa
g_y = 0.5 * ( k_north * (roll_y(u, -1) - u) + k_south * (u - roll_y(u, 1)) ) / kappa
```

in per-cell units of the grid `u` lives on. `kappa` broadcasts over the
component axis. Guard `kappa` with the same `HARMONIC_FLOOR` the module
already uses.

### 1.2 Strain on the solve grid, mapped to the kinematic grid

In `kinematics.py`, `run_history`, replace the strain block with:

```
g_x, g_y = effective_gradients(solved, kappa_s)         # solve grid, per solve cell
cell_s_km = cell_km * SOLVE_GRID_DIVISOR
exx = g_x[0] / cell_s_km
eyy = g_y[1] / cell_s_km
exy = 0.5 * (g_y[0] + g_x[1]) / cell_s_km
strain_rate_s = sqrt(exx² + eyy² + 2 exy²)
divergence_s  = exx + eyy
strain_rate = prolong(strain_rate_s)                    # piecewise constant, 2 x 2 blocks
divergence  = prolong(divergence_s)
```

`prolong` is the solver's piecewise-constant lift. It must be piecewise
constant here: a bilinear lift would put intermediate strain into the
strong cells next to a zone and reintroduce the halo this run removes.
Apply `prolong` as many times as `to_kinematic_grid` does, so the result is
`(n, n)`.

Damage, healing, and the exact integrator are unchanged and use this
`strain_rate`. Advection still uses the bilinear `velocity`. The epoch
fields `strain_rate` and `divergence` are the block-constant kinematic-grid
arrays; the views will show 2 × 2 blocks, which is the truth of where
damage is now resolved. Say so in the two `view_purposes` strings.

That is the whole change to the engine.

### 1.3 Early snapshots, for the report only

Run 3's kept epochs begin at 76 Myr, after the transient the prediction was
about. Add to `History` a list `early: list[tuple[int, float, np.ndarray]]`
of `(step, t_myr, strength.copy())` at steps **2, 4, 8, 12, 16** (`t` = 8,
16, 32, 48, 64 Myr). Not views; the report's time-lapse uses them. Give
`tools/contact_sheet.py` an `--early` flag that, for one seed, tiles the
five early strengths and the five early weak masks in two rows.

## 2. Prediction, stated before running

Seed `4287772760` at 1024 px, scale 5.

- Step 1 exceedance is the same 12.6 % as run 3, since nothing changes
  before the first strain is computed on a uniform field, apart from block
  quantization. Report it.
- `weak_fraction` rises for a few steps as the initially-exceeding blocks
  fail, **peaks below 0.25**, then **falls** as the interiors of wide initial
  patches, which sit below yield once the jump concentrates on their edges,
  heal, and **settles between 0.03 and 0.12** by 150 Myr. It does not cross
  0.5 at any step.
- Strong interiors heal toward full strength: final mean strength above
  0.85 with the minimum at the floor.
- Zones are **2 to 4 kinematic cells** wide (80–160 km): one solve cell plus
  block quantization.
- **Three to eight plates** above 1 % area.
- The weak set is a network: largest 8-connected weak component holds more
  than half of all weak cells, and the number of components is below 20.
- Plates read as equant cells of one scale, and some segments align with
  the axes because of the noise. Expected.
- Solver cycles per step fall from run 3's mean of 12, because the
  high-contrast area shrinks to the zones; generate time under 4 s.

Stops:

- ⛔ `weak_fraction > 0.5` at any step: the halo survives the change. Stop,
  report the trajectory, and report the strain rate in the strong cells
  adjacent to the weak set at step 8 versus the yield, which decides
  whether the halo is still the cause.
- ⛔ `weak_fraction_final < 0.01`: everything healed. Stop, report the
  trajectory and the strain inside the weak set at step 8 versus the yield.

## 3. Tests

Add to `tests/test_solver.py`:

- **Uniform kappa.** For any `u` and constant `kappa`, `effective_gradients`
  equals `ddx`, `ddy` from `domain.py` to 1e-12.
- **Barrier.** `64²` grid, `kappa = 1000` everywhere except row 32 at
  `1000 * STRENGTH_MIN**4`; `u[0]` is `+1` on rows 33–63 and `-1` on rows
  0–31 (a jump across the weak row on the torus, one more at the wrap).
  `g_y[0]` on row 32 exceeds 1.0; on rows 31 and 33 it is below 1e-3;
  elsewhere it is 0.
- **Symmetry.** Transposing `kappa` and swapping-and-transposing `u` gives
  the transposed, swapped gradients to 1e-12.

Add to `tests/test_history.py`: the strain field on an epoch is
block-constant, every 2 × 2 kinematic block equal, and its shape is `(n, n)`.

Everything else must pass. If the full-length determinism test's budget is
exceeded, report the time; do not weaken the test.

## 4. Measurements

All twelve `STATUS.md` seeds at 1024 px, scale 5, unless stated.

1. **Step-1 exceedance** and strain percentiles as in run 3.
2. **Per-step trajectory** for seeds `4287772760`, `2075014389`,
   `1833546021` to `out/c03_4_trajectories.csv`, same columns as run 3.
3. **Per-seed summary** for all twelve: `plate_count`, `plate_area_percent`,
   `weak_fraction_final`, peak `weak_fraction` and its step,
   `boundary_cell_fraction`, `regime_share`, weak 8-connected component
   count and largest share, zone edge fraction, and final strength mean.
4. **Halo check.** At steps 8 and 40 on seed `4287772760`: the mean and
   95th-percentile strain rate over strong cells that are 4-neighbours of a
   weak cell, over strong cells that are not, and over weak cells, each
   against the yield. This is the direct test of the mechanism this run
   changes.
5. **Contact sheets**, twelve seeds each at 1024 px: `plates_1024.png`,
   `boundaries_1024.png`, `regime_1024.png`, `strength_1024.png`,
   `velocity_1024.png`, `strain_rate_1024.png`.
6. **Time-lapse** for seed `4287772760`: the early sheet from §1.3, and the
   four-epoch `strength` and `boundaries` strips as before.
7. **Timing** at 1024 px, generate plus render; solver cycles mean and max;
   steps that exhausted `MG_MAX_CYCLES` at 1024 and 2048 px.

## 5. The audit, run properly

As in run 3 §5, full default view list, fresh-context judges given only the
prompt and their panel paths, told to write their verdict arrays to a named
file before finishing. Concatenate, `score`, `verify`. Report the scorer
output verbatim, every candidate finding with the judge's rule and
evidence, and whether the batch is void. Findings are for the author.

## 6. Documents

- `STATUS.md`: "Now" and "The open question", a paragraph each;
  "Verification" from the final run.
- `README.md`: only if timing or report keys changed.

## 7. Report

`out/C03_4_BUILD_REPORT.md`:

1. **The change**, as applied.
2. **Deviations**.
3. **Stops**, if either fired, with the evidence.
4. **Check output**, verbatim summary lines.
5. **Prediction versus outcome**: each bullet of §2, what was measured,
   `met` / `not met` / `cannot tell`.
6. **Measurements** per §4, including the halo check as its own table.
7. **Audit** per §5.
8. **Observations**, with evidence, naming what a measurement implicates.
   No proposed values.

Do not describe any field as natural, plausible, good, or bad, and do not
compare anything to the reference images.
