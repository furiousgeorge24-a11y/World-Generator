# Work order — C03.8: work-based damage

Issued 2026-09-02 for Opus 5. Follows [`WORK_ORDER_C03_7.md`](WORK_ORDER_C03_7.md)
and its report at `out/C03_7_BUILD_REPORT.md`. Same rules: isolation from
`pipeline_a`, `pipeline_b`, `pipeline_d`; NumPy, Pillow, Flask, and the
standard library; determinism of the engine; one mechanism change; no review
apparatus; do not commit; do not edit `DESIGN.md`. Do not describe any field
as natural, plausible, good, or bad, and do not compare anything to the
reference images.

## 0. Purpose

The regime search has now read the sheet's dial space. Run
`out/search/20260902T154430Z-s2/` is 786 cells over three rounds, 600 of them
stage-1 Latin-hypercube samples on four seeds. Its result, measured over all
3,912 worlds:

- `edge_fraction` passes in no cell. The median implied zone width is about
  fifteen cells. Zones thinner than four cells occur only in worlds whose weak
  fraction is below 3 %, which is a scatter of isolated cells, not a network.
- `plate_count` passes in 18 cells, and every one of them has a weak fraction
  between 60 and 90 %: the "plates" are strong islands left in failed
  material. All 18 fail `weak_final`, `weak_drift`, and `edge_fraction`.
- Across cells, `edge_fraction` and `weak_final` have a rank correlation of
  −0.87. No dial has a rank correlation with `plate_count` above 0.15 in
  magnitude. `strength_spread`, the heterogeneity attempt, correlates with
  `edge_fraction` at +0.10.

One length sets both things the screen asks for: the solve length that makes
a plate interior rigid is also the distance over which strain spreads around
a weak zone, and every cell inside it fails. The stopping rule in `STATUS.md`
allows the sheet two more physics attempts before the formulation changes.
Heterogeneity was the first and is spent. This order is the second:
**damage driven by dissipated work instead of by strain rate.** It is a one-
line change to the law and a full paired search run to measure it.

**Prediction on record.** For a cell at its initial strength the two laws
have the same threshold, because power at fixed stiffness is a monotone
function of strain rate. Above threshold the work law damages faster, and a
cell that has started to fail damages faster still. So this note predicts:

- P1. Paired stage-1 cells show `weak_final` rising or unchanged under the
  work law: the median paired difference is ≥ 0 and more cells rise than
  fall.
- P2. Zones do not narrow. Among worlds with weak fraction inside the
  screen's band (0.02 to 0.25), the median `edge_fraction` stays within
  ±0.05 of the control's 0.174, and the paired median difference in
  `edge_fraction` is ≤ 0.
- P3. The anticorrelation survives: rank correlation of `edge_fraction`
  with `weak_final` across stage-1 cells stays below −0.7.
- P4. Plate count stays unreachable: more than 90 % of band worlds have one
  plate, and no cell passes the screen.
- P5. Time to failure shortens: for paired cells whose control weak fraction
  ends above 0.1, the step at which the weak fraction first reaches half of
  its final value is earlier or equal under the work law in most pairs.

The prediction fails, and the law earns further work, if the band's median
`edge_fraction` rises above 0.30 or if more than 10 % of band worlds have two
or more plates. Either outcome is a result. Report what was measured.

## 1. Engine: the law

In `engine/history/kinematics.py`:

1. `HistoryParams` gains `work_damage: int = 0`, validated to `0 ≤ v ≤ 1`
   as an integer, recorded by `to_record`. `0` is the strain-rate law as it
   stands; `1` is the work law. The production default is `0`; nothing in
   production changes.
2. In the step loop, after `strain_rate_s` is formed, form the dissipated
   power on the solve grid:

   ```python
   power_s = kappa_s * strain_rate_s * strain_rate_s
   ```

   Stress in this sheet is stiffness times strain rate, so this is stress
   times strain rate up to the constant the calibration absorbs. Lift it to
   the kinematic grid with `to_kinematic_blocks`, exactly as `strain_rate`
   is lifted.
3. At step 1, alongside `yield_strain_per_myr`, read `yield_power` as the
   same percentile of `power_s`, with the same guard against a non-positive
   value. Both are read every run whatever the law, so the record is
   complete.
4. The excess is the only line that changes:

   ```python
   if params.work_damage:
       excess = np.maximum(power / yield_power - 1.0, 0.0)
   else:
       excess = np.maximum(strain_rate / yield_strain_per_myr - 1.0, 0.0)
   ```

   Everything after it — the squared excess, the exact integrator, healing,
   the floor, advection — is untouched.
5. `Epoch` gains `power: np.ndarray` (n, n) and `History` gains
   `yield_power: float`. Nothing else in the record moves.

With `work_damage = 0` every output must be byte-identical to the current
engine. `tests/test_regression_c03_1.py` is the gate on that and must pass
unchanged.

## 2. The view

Every layer gets a view. Add `power` to `VIEWS` in both `explore_adapter.py`
and `webui_adapter.py`, immediately after `strain_rate`, drawn with the same
scalar mapping the `strain_rate` view uses (log-scaled if that one is), from
the final epoch's `power`. Give it the same one-line description pattern
the other views carry. The search's `ALL_SHEETS` follows `VIEWS`, so stage-3
cells and findings gain the sheet without further change. Do not restart the
production server on port 5002 if one is running; the author reloads it.

**Deviation from the standing rule, on record:** this order does not run
the blind layer audit on `power`. The prediction is that the law is not kept,
and auditing a layer that is then discarded is waste. If the author keeps the
law, `power` goes through `run_layer_audit.py` before anything is frozen.

## 3. The dial, in the lab and in the search

- Exploration lab: add `work_damage` to `_DIALS` in `explore_adapter.py` as
  an `int`, default 0, lo 0, hi 1, tier primary, promise "0: damage by
  strain-rate excess. 1: damage by dissipated-work excess, stress times
  strain rate. Above the same threshold the work law fails a cell faster."
- Search: `Space` gains a **fixed** setting `work_damage: int`, beside
  `pixels` and `max_cycles`, threaded through `params_of`, **defaulting to
  `1`** so that the author's next Start is the treatment run (amended, see
  §6; the engine's own default stays `0`); `search_server.py`
  gains the knob in the `space` group with lo 0, hi 1, and its meaning. It is
  deliberately **not** a sampled dial: the Latin hypercube is then the same
  for a given search seed, so a run at `work_damage = 1` samples exactly the
  cells the control run sampled, cell for cell. That is the ablation pair.

## 4. The pairing tool

`pipeline_c/tools/pair_runs.py`, importable and runnable:

```
py -3.14 pipeline_c/tools/pair_runs.py <control_run_id> <treatment_run_id>
```

It loads both runs' `cells.jsonl`, pairs stage-1 cells by their dial values
(equal to 12 decimal places, all dials), and writes a markdown table to
stdout and to `out/pair_<control>_<treatment>.md` with:

1. cell counts: paired, unpaired on each side, invalid on each side;
2. per metric (`weak_final`, `edge_fraction`, `plate_count`,
   `network_share`, `weak_peak`), over cell means: median paired difference
   (treatment minus control), count rising, count falling, count within
   1e-6;
3. rank correlation of `edge_fraction` with `weak_final` across stage-1
   cells, each run separately (Spearman; write the rank correlation by hand
   from `numpy`, no SciPy);
4. band statistics for each run over worlds with weak fraction in the
   screen's `[weak_min, weak_max]` read from the run's `config.json`:
   world count, `edge_fraction` p50 / p90 / max, share of worlds with one
   plate, share with two or more;
5. time to half: for pairs whose control `weak_final` mean ≥ 0.1, the first
   step at which each world's `weak_fraction` trajectory reaches half of its
   final value, averaged per cell; median paired difference and the counts
   earlier / later / equal;
6. passers, findings, and throughput (cells per minute, worlds per second)
   for each run.

A test in `tests/test_pair_runs.py` builds two synthetic runs on disk with
known differences and checks the counts, the medians, the correlation on a
hand-computable case, and that an unpaired cell is reported, not dropped
silently.

## 5. Tests

`tests/test_history.py` (or a new `tests/test_work_damage.py`, your choice):

- at `work_damage = 0` a short run's strength field is byte-identical to the
  same run before this order (the regression test already covers the
  production path; add one at a non-default dial set);
- at step 1 with `strength_spread = 0`, the sets of cells above threshold
  under the two laws are identical, because at uniform stiffness the power
  threshold is the strain threshold squared;
- `power` on the solve grid equals `kappa_s * strain_rate_s ** 2` to
  floating tolerance, is non-negative everywhere, and `yield_power` is the
  stated percentile of it;
- `work_damage = 2` and `work_damage = True` are refused by `HistoryParams`.

`tests/test_search.py`: `params_of` carries `work_damage` from the space; a
`Space` with `work_damage = 1` produces the same Latin hypercube as one with
`0` for the same seed.

`tests/test_search_server.py`: the knob is offered, round-trips through
start, and `2` is refused.

The full suite must pass:

```
py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"
```

## 6. The paired run

**Amended by the author after issue: this order does not start the run.
The author runs it from the gallery.** What the order does instead:

1. Set the search's defaults so that pressing Start with no edits produces
   the paired run: `Space.work_damage` defaults to `1` (the engine's
   `HistoryParams.work_damage` stays `0`; production is untouched) and the
   default `search_seed` becomes `2`, so the Latin hypercube matches the
   control run `out/search/20260902T154430Z-s2/`. If the control's
   `config.json` differs from the current defaults in any other knob, make
   the default match it and say so in the report. Add a test that the
   default `Space` at the default seed reproduces the first three stage-1
   dial sets of the control's `cells.jsonl` to 12 decimal places.
2. The search server on port 5004 is running the code from before this
   order and is idle. Stop it and start it again the same way, detached, so
   it loads the new module:

   ```powershell
   $p = (Get-NetTCPConnection -LocalPort 5004 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1).OwningProcess
   if ($p) { Stop-Process -Id $p -Force -Confirm:$false }
   ```

   ```
   WEBUI_RELOAD=0 py -3.14 pipeline_c/search_server.py --backend explore_adapter --root pipeline_c --port 5004
   ```

   Confirm `/api/status` answers and reports idle. Do not post to
   `/api/start`.
3. Put the pairing command the author runs afterwards into the report and
   into `SEARCH.md`:

   ```
   py -3.14 pipeline_c/tools/pair_runs.py 20260902T154430Z-s2 <treatment_run_id>
   ```

   where the treatment run id is the one the gallery shows.

The author lets the run complete three rounds, presses Stop, and runs the
pairing tool. If a stage-1 cell passes, the run proceeds through stage 2 and
3 on its own.

## 7. Documents

- `SEARCH.md`: one line for the fixed `work_damage` knob and one sentence on
  the pairing tool.
- `EXPLORE.md`: one line for the dial.
- `STATUS.md` "Now": one sentence saying C03.8 tested the work law as the
  second of the two physics attempts, with the outcome in one clause once
  measured.

## 8. Report

`out/C03_8_BUILD_REPORT.md`:

1. **What was built**, file by file.
2. **Deviations** from this order, with reasons.
3. **Check output**: the suite's verdict lines, verbatim.
4. **The paired run**: not run by this order; the author runs it. The
   pairing command, and the defaults that make Start reproduce the control's
   sample.
5. **The predictions**: P1 to P5 each with the metric that decides it and
   the number that would decide it, marked "pending the author's run"; then
   the two failure conditions from §0 in the same form.
6. **Observations** from the build and the tests, with evidence, no
   proposed values.
