# Work order — C03.6: the solver at high stiffness

Issued 2026-09-02 for Opus 5. Follows [`WORK_ORDER_C03_5.md`](WORK_ORDER_C03_5.md)
and its report at `out/C03_5_BUILD_REPORT.md`. Same rules: isolation from
`pipeline_a`, `pipeline_b`, `pipeline_d`; NumPy, Pillow, standard library;
determinism; no review apparatus; do not commit; do not edit `DESIGN.md`.

## 0. Purpose

The C03.5 sweep is unusable above `stiffness_fraction = 0.125`: mean worst
residual 0.04 at 0.25, 1.3 at 0.5, 11 at 1.0, 40 at 2.0, against a
tolerance of 1e-3. Fifteen of twenty-five cells report on velocity fields
that were not solved. The stiff regime is the one the remaining hypothesis
lives in, so the lab cannot answer its question until the solver converges
there.

This order fixes the solver and nothing else. No physics constant, dial
default, noise, grid, or view changes.

## 1. Diagnosis to confirm first

`kappa0 = (stiffness_fraction · n_solve)²`. The coarsest multigrid level is
`MG_COARSEST = 8` cells, reached by halving from 128 in four steps, and
`restrict_kappa` divides by four per level, so the coarsest coefficient is
`kappa0 / 256`. At stiffness 0.125 that is about 1; at 2.0 it is about 256.
The coarsest level is solved by `MG_COARSE_SWEEPS = 50` weighted Jacobi
sweeps, which converge for a coefficient near 1 and do not for 256, so
every V-cycle carries an unconverged coarse correction.

Confirm before changing anything: on the `256²` barrier problem of
`tests/test_solver.py`, scale `kappa` by 1, 16, 64, 256 and record cycles to
tolerance and the final residual with the current code. Then measure the
coarsest-level residual after 50 sweeps for the same four cases. Put both
tables in the report. If the coarse solve is *not* the stalling component,
say so and find what is before proceeding; §2 still applies, but the
report must name the actual cause.

## 2. The change

### 2.1 Exact coarsest solve

Solve the coarsest level exactly. Assemble the dense `m × m` matrix of the
periodic five-point operator on the `MG_COARSEST²` grid from that level's
`k_east`, `k_north`, and `diag` (`m = 64`), factor it once per `solve`
call (`np.linalg.solve` or an explicit LU; either is deterministic), and
apply it to both velocity components. Cache the factorization on the
`Level` for the life of one `solve` call; levels are rebuilt when `kappa`
changes, which is every step.

Remove `MG_COARSE_SWEEPS` from the V-cycle path. Keep the constant only if
something else uses it; otherwise delete it from `constants.py`.

### 2.2 If exact coarse solve is not enough

Run the §3 convergence test. If it still fails at stiffness 2.0, the next
legitimate steps, in order, each measured before the next is tried:

1. Galerkin coarse operators (`R A P` with the existing mean restriction and
   piecewise-constant prolongation) in place of harmonic-restricted
   coefficients. For this transfer pair the coarse operator stays
   five-point, and the coarse edge coefficient is the sum of the fine edge
   coefficients crossing that coarse edge, divided by four. Keep the
   identity term's restriction consistent (mean of the four fine cells,
   which is 1).
2. Red-black Gauss–Seidel in place of weighted Jacobi, `MG_PRE = MG_POST = 2`.
3. `MG_COARSEST = 4`.

Do not raise `MG_MAX_CYCLES` as a fix, do not loosen `MG_TOL`, and do not
replace the multigrid-preconditioned conjugate gradient with anything
else. If all three steps fail, stop and report the residual history of
each.

## 3. Convergence test

Add to `tests/test_solver.py` a **stiff network** case: `128²` solve grid,
`kappa0 = (2.0 · 128)²`, `kappa = kappa0` everywhere except a weak network
at `kappa0 · STRENGTH_MIN⁴`: rows 20, 70, 110 and columns 30, 90, each two
cells wide, plus a diagonal line from `(0, 0)` to `(127, 127)` two cells
wide, all periodic. `D` is a periodic noise field of unit RMS (use the
engine's `periodic_noise` with a fixed world id). The solve from zero must
reach `MG_TOL = 1e-3` within **40 cycles**, and a second solve warm-started
from the result must take at most 2. Record the cycle count in the report.

Keep every existing solver test. The FFT comparison, barrier, symmetry,
and adjoint tests must pass unchanged.

## 4. Re-run the stiff rows of the sweep

Same protocol as C03.5 §4, only the rows `stiffness_fraction ∈ {0.25, 0.5,
1.0, 2.0}` at yield percentiles `{3, 6, 12, 20, 30}`, `max_cycles = 40`,
seeds `4287772760 … +7`, 1024 px. Write `out/c03_6_sweep.md` and `.csv`
with the same columns plus `mean exhausted_steps`. Every cell must show a
mean worst residual below `MG_TOL`; if any does not, report it as
unconverged rather than as a result. For the cell with the highest
`stable_count` across the *union* of C03.5's converged row and this rerun,
write its `plates`, `weak_t32`, and `trajectory` sheets to `out/` as
`c03_6_best_*.png`. One pass, no second sweep.

## 5. Timing

Report the 8-world generate at 1024 px at stiffness 0.125 and at 2.0,
pooled, before and after. If the exact coarse solve changes the
default-stiffness timing by more than 10 %, say so.

## 6. Documents

- `STATUS.md`: "Now" rewritten in a paragraph for the state after C03.5 and
  this run: the exploration lab exists on port 5003, the noise is
  isotropic, the solver converges at high stiffness (or does not), and no
  stable regime has yet been found or ruled out. "Verification" from the
  final run. "The open question" becomes: does a stable localizing regime
  exist in the dial space; undecided until the author explores.
- `EXPLORE.md`: one added line under "what to look for": the report's
  worst residual must be below 1e-3 or the result is not a solve.

## 7. Report

`out/C03_6_BUILD_REPORT.md`:

1. **Diagnosis** per §1, with both tables.
2. **The change**, as applied, and which of §2.2's steps were needed.
3. **Deviations**.
4. **Check output**, verbatim summary lines, including the stiff-network
   test's cycle count.
5. **Timing** per §5.
6. **The sweep** per §4, table and sheet paths, with the union's best
   cell.
7. **Observations**, with evidence. No proposed dial values; the author
   turns the dials.

Do not describe any field as natural, plausible, good, or bad, and do not
compare anything to the reference images.
