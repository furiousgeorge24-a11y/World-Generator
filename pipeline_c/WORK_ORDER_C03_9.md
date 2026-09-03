# Work order — C03.9: is the width floor the grid?

Issued 2026-09-02 for Opus 5. Follows [`WORK_ORDER_C03_8.md`](WORK_ORDER_C03_8.md)
and its report at `out/C03_8_BUILD_REPORT.md`. Same rules: isolation from
`pipeline_a`, `pipeline_b`, `pipeline_d`; NumPy, Pillow, Flask, and the
standard library; determinism of the engine; one change; no review
apparatus; do not commit; do not edit `DESIGN.md`. Do not describe any field
as natural, plausible, good, or bad, and do not compare anything to the
reference images. **Do not start, stop, or restart the search server on
port 5004, and do not post to its API**: the author's search may be running
on it, and this order's runs go through the exploration adapter's own pool
in your process.

## 0. Purpose

The corner search under the work law, run `out/search/20260902T183110Z-s2/`
(2154 cells, 12,492 worlds, no passer), has 387 worlds that fail the screen
on `edge_fraction` alone: 3 to 5 plates, weak fraction 0.11 to 0.24,
settled, one network. Their edge fraction has a median of 0.30 and a
maximum of 0.44 against a bound of 0.5.

The author measured zone widths directly on the plates sheets of five
frontier cells (`c02126`, `c00212`, `c01018`, `c01852`, `c00177`), reading
the black cells off `plates.png` at the native 256-cell resolution and
eroding on the torus:

- 90 to 98 % of weak cells sit in even-aligned 2 x 2 blocks;
- the share of the weak set at least 3 cells wide equals the share at least
  4 wide, and the share at least 5 equals the share at least 6, in every
  world: widths come in even numbers only;
- in the best worlds 75 to 90 % of the network is at least 4 kinematic cells
  wide and 50 to 80 % at least 6.

That is the solve grid. `SOLVE_GRID_DIVISOR = 2` solves velocity on half the
kinematic grid and lifts strain back as 2 x 2 blocks, so damage happens in
units of one solve cell, 80 km at 5 km/px, and the screen's width bound of
4 kinematic cells is exactly 2 solve cells. The frontier sits at 2 to 3
solve cells. One solve cell of halo on each side of a zone is 160 km of
width, which is the whole bound.

**The question.** Is the widening numerical, a halo of one solve cell per
side that halves when the solve cell halves, or physical, a band set by the
drive's strain field in kilometres that a finer grid only resolves better?
The test is the same worlds, the same dials, the same seeds, solved on the
full kinematic grid, compared cell for cell.

**Prediction on record.** Part of the halo is numerical and part is not.

- P1. At divisor 1 the widths stop being even: the share at least 3 wide
  differs from the share at least 4 wide by more than 0.05 in most worlds.
- P2. Paired `edge_fraction` rises: the median difference (divisor 1 minus
  divisor 2) over paired worlds is between +0.03 and +0.12, with more worlds
  rising than falling.
- P3. It is not enough on its own: fewer than 10 % of the rerun worlds pass
  all six terms at divisor 1.
- P4. Cost: a world at divisor 1 and 1024 px takes 5 to 9 times as long as
  at divisor 2.
- P5. Convergence: at `stiffness_fraction` above 0.4 some divisor-1 worlds
  need more than 40 cycles; at `max_cycles` 80 every rerun world converges
  below `MG_TOL`.

**Decision rule, fixed in advance.** If 10 % or more of the rerun worlds
pass all six at divisor 1, the ceiling was the grid, and what follows is a
resolution-and-cost decision for the author, not another physics step. If
the median paired rise in `edge_fraction` is below +0.03, the widening is
physical and the sheet is finished; the block formulation follows. Between
those, report the numbers and the author decides. Either outcome is a
result.

## 1. Engine: the solve divisor becomes a parameter

In `engine/history/kinematics.py`:

1. `HistoryParams` gains `solve_divisor: int = SOLVE_GRID_DIVISOR`,
   validated as an integer in `{1, 2}`, recorded by `to_record`. The
   production default is unchanged.
2. Every use of the module constant in `run_history` reads
   `params.solve_divisor` instead: `solve_n_for(n, divisor)`, and
   `cell_s_km = cell_km * divisor`. `to_solve_grid`, `to_kinematic_grid` and
   `to_kinematic_blocks` already loop on shapes; check that each is exact at
   divisor 1 (identity, no restriction, no lift) and that
   `to_kinematic_blocks` derives its factor from the shapes rather than the
   constant.
3. Nothing in `kappa0_for` changes. Its length is in kinematic cells and
   `restrict_kappa` carries the quarter factor per coarsening, so the
   stiffness length in kilometres is the same at either divisor. State in
   the report, with the two numbers, that `kappa` on the solve grid at
   divisor 2 equals a quarter of the 2 x 2 mean of `kappa` at divisor 1 for
   the same strength field.

With `solve_divisor = 2` every output must be byte-identical to the current
engine. `tests/test_regression_c03_1.py` and the pre-order hash test in
`tests/test_work_damage.py` are the gates and must pass unchanged.

## 2. The dial, in the lab and in the search

- Exploration lab: `solve_divisor` in `_DIALS`, `int`, default 2, lo 1,
  hi 2, tier **advanced** (it is a discretization control, not a formation
  control), promise "Kinematic cells per solve cell. 2 solves velocity on
  half the grid and lifts strain back in 2 x 2 blocks, so a zone cannot be
  narrower than two cells. 1 solves on the full grid at about six times the
  cost."
- Search: `Space` gains a fixed `solve_divisor: int = 2` beside
  `work_damage`, threaded through `params_of`; `search_server.py` gains the
  knob with lo 1, hi 2 and its meaning. Fixed, not sampled, for the same
  reason `work_damage` is.

No new view. `strain_rate` and `power` become finer at divisor 1 by
themselves.

## 3. The A/B tool

`pipeline_c/tools/ab_solve.py`, importable and runnable:

```
py -3.14 pipeline_c/tools/ab_solve.py <run_id> [--cells 20] [--pixels 1024] [--max-cycles 80]
```

1. **Select** cells from the run's `cells.jsonl`: rank by the count of
   worlds that fail on `edge_fraction` alone under the run's own screen,
   ties by soft score; take the top 12; then add the best cells by soft
   score not already taken until there are `--cells`. Use each cell's own
   seeds.
2. **Rerun** every selected cell at `solve_divisor` 2 and then 1, on its
   own seeds, through `explore_adapter`'s process pool (`parallel=True`,
   the eight-worker spawn pool), at `--pixels`, `scale_km` 5,
   `history_myr` 300, `max_cycles` from the flag, `work_damage` and every
   dial from the cell. The divisor-2 rerun must reproduce the run's logged
   metrics for every world to 1e-9; that is the determinism gate, and the
   tool stops with an error if it does not.
3. **Measure** each world with `search.world_metrics` and
   `search.screen_world` under the run's screen; record `residual_max`,
   solver cycles, and seconds per world. Additionally measure width on the
   final weak mask, computed from the world's strength field and not from a
   PNG: the share of weak cells covered by some fully-weak k x k square on
   the torus for k = 2 … 8, and the share of weak cells whose even-aligned
   2 x 2 block is entirely weak.
4. **Write** `out/ab_solve_<run_id>.md` (and to stdout) with: the selection;
   per world both variants' `edge_fraction`, `weak_final`, `plate_count`,
   `weak_drift`, `network_share`, pass state, cycles, residual; paired
   differences per metric as median, rising, falling; worlds passing all
   six at each divisor; invalid at each; the width shares averaged over
   worlds at each divisor; seconds per world at each divisor, min / mean /
   max. Write the `plates` sheet for both variants of every cell to
   `out/ab_solve/<run_id>/<cell>_d<divisor>_plates.png`, and the
   `strain_rate` and `power` sheets for the first three cells.

A test in `tests/test_ab_solve.py` covers the selection rule on a synthetic
run, the width shares on hand-built masks (a 2-wide torus line gives
share ≥ 2 of 1.0 and ≥ 3 of 0.0; a 3-wide line gives ≥ 3 of 1.0 and ≥ 4 of
0.0; a block-aligned 2-wide line gives alignment 1.0 and one shifted by a
cell gives 0.0), and the determinism gate on a stub.

## 4. Tests

- `test_history.py` or a new file: `solve_n_for(n, 1) == n`;
  `to_solve_grid` is the identity at divisor 1; a 128-px run at divisor 1
  converges below `MG_TOL` and is deterministic; the kappa relation in §1.3
  holds numerically; `solve_divisor` 3, 0 and `True` are refused.
- `test_search.py`: `params_of` carries `solve_divisor`; the hypercube does
  not move with it.
- `test_search_server.py`: the knob is offered, round-trips, and 3 is
  refused.

The full suite must pass:

```
py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"
```

## 5. The A/B run

Run the tool on `20260902T183110Z-s2` with 20 cells at 1024 px and
`max_cycles` 80. Twenty cells at 4 or 8 seeds each is up to 160 worlds per
variant; expect a minute for divisor 2 and ten to fifteen for divisor 1.
The author's search may be running on the other eight cores; note its
`/api/status` at start and end (read only) and say so beside the timings.

If any divisor-1 world fails to converge at 80 cycles, it is invalid and
stays in the table as invalid; do not raise the cycles again and do not
drop it.

**§5b, the same question at 512 px.** Run the tool again with
`--pixels 512` on the same selection. At 512 px the history grid is 128
cells, divisor 1 is a 128-cell solve, the same size as the 1024-px divisor-2
solve, so it costs the same as the search's own worlds and converges where
they do. The parent is half the size, so the worlds are different worlds
and the pairing is by dials, not by world; the width shares are the
comparable numbers. Report it as a second table.

## 6. Documents

- `EXPLORE.md`: one line for the dial.
- `SEARCH.md`: one line for the fixed knob.
- `STATUS.md` "Now": two sentences. One records the width measurement from
  §0, that zone widths in the corner search are quantized to the solve
  cell and the frontier sits at two to three solve cells against a bound of
  two. One records this order's outcome under the decision rule, once
  measured.

## 7. Report

`out/C03_9_BUILD_REPORT.md`:

1. **What was built**, file by file.
2. **Deviations**, with reasons.
3. **Check output**: the suite's verdict lines, verbatim.
4. **The A/B at 1024 px**: the tool's page verbatim.
5. **The A/B at 512 px**: the tool's page verbatim.
6. **The predictions**: P1 to P5 each with the deciding number and "held"
   or "failed"; then the decision rule's two thresholds with their numbers
   and which branch the run landed in.
7. **Cost**: seconds per world at 1024 px for each divisor and at 512 px
   for each, from the reruns, as one table. No recommendation about
   production settings; that is the author's.
8. **Observations**, with evidence, no proposed values. If widths at
   divisor 1 are still even-quantized anywhere, say where. If any world
   passes all six, give its cell id, seed, dials, and sheet paths so the
   author can open it.
