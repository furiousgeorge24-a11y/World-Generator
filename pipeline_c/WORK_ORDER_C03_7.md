# Work order — C03.7: the regime search

Issued 2026-09-02 for Opus 5. Follows [`WORK_ORDER_C03_6.md`](WORK_ORDER_C03_6.md)
and its report at `out/C03_6_BUILD_REPORT.md`. Same rules: isolation from
`pipeline_a`, `pipeline_b`, `pipeline_d`; NumPy, Pillow, Flask (already a
dependency of the shared shell), and the standard library; determinism of
the engine; no review apparatus; do not commit; do not edit `DESIGN.md`.

## 0. Purpose

Two sweeps over two dials found no stable regime. Seven dials remain, and a
search over them at five seconds per cell is an hour's work for a machine
and a week's for a person. This order builds that search as a **third tab
on its own port**, with a live gallery so the author can watch it, knobs
for what "stable" means, and defaults set to the screen the author and
this note have agreed on. Its output is candidates for the author's eyes.
It grants nothing.

Three principles the search must keep:

- **The screen is physics, not aesthetics.** Every term is a measured
  property of a plate regime. Nothing in the score says how a field looks.
- **An unconverged cell is invalid, not a result.** Worst residual above
  tolerance means the velocity was not solved; the cell is marked invalid
  and never scored.
- **Eight seeds screen; twelve seeds confirm.** A cell that passes at
  eight is rerun on the twelve development seeds before it is called a
  finding, automatically.

## 1. Engine: one new dial

Add `strength_spread: float = STRENGTH_INIT_SPREAD` to `HistoryParams`
(range `0 ≤ spread ≤ 0.3`) and use it in `initial_strength`. Add it to the
exploration adapter's controls as `strength_spread`, float, default 0.1,
lo 0, hi 0.3, tier primary, promise "Initial heterogeneity of the
lithosphere. Soft spots concentrate strain and may seed failure." The
production default is unchanged.

## 2. The search, as a library

`pipeline_c/search.py`, importable without Flask, deterministic given a
search seed. It reuses the exploration adapter's process-pool worker for
worlds and adds nothing to the engine.

### 2.1 Per-world metrics

From the worker's returned dict, compute for each world:

| metric | definition |
|---|---|
| `weak_final` | weak fraction at the last step |
| `weak_peak` | maximum weak fraction over all steps |
| `weak_drift` | `abs(weak_final − weak fraction at t = history − flat_window_myr)` |
| `plate_count` | plates above 1 % of the parent, as the report already computes |
| `network_share` | largest 8-connected component of the final weak set on the torus, as a share of all weak cells; 0 if no weak cells |
| `edge_fraction` | weak cells with at least one strong 4-neighbour, as a share of all weak cells; 0 if no weak cells |
| `residual_max` | worst solver relative residual over the run |

Add a general `label_components(mask, connectivity)` to
`engine/history/plates.py` for `network_share`, using the same pointer-
jumping propagation as `label_plates`, with 8-connectivity meaning the
four diagonal rolls are included.

### 2.2 The screen

A `Screen` dataclass with these fields and defaults. Every one is a knob in
the UI.

| knob | default | meaning |
|---|---|---|
| `weak_min` | 0.02 | some lithosphere must have failed |
| `weak_max` | 0.25 | most must not have |
| `peak_ratio_max` | 1.5 | `weak_peak / weak_final`; the weak set is not collapsing back or overshooting |
| `flat_window_myr` | 100 | window over which the weak fraction must be settled |
| `flat_tolerance` | 0.03 | `weak_drift` bound; the weak set has stopped moving |
| `plates_min` | 3 | |
| `plates_max` | 8 | |
| `network_share_min` | 0.5 | the weak set is a connected network, not speckle |
| `edge_fraction_min` | 0.5 | zones are thin; for a line of width `w` cells the edge fraction is about `2 / w`, so 0.5 means width four or less |
| `residual_max` | 1e-3 | below this the cell is a solve; above it the cell is invalid |
| `pass_fraction` | 1.0 | share of a cell's worlds that must pass for the cell to pass |

A world **passes** when every term holds. A cell is **invalid** if any
world's `residual_max` exceeds `residual_max`; otherwise it **passes** when
at least `pass_fraction` of its worlds pass.

A **soft score** ranks cells that do not pass: for each term, the
normalized violation (how far outside the bound, divided by the bound's
width, 0 if inside), summed over terms and averaged over worlds. Lower is
closer. It contains no aesthetic term and is used only for ordering
stage-2 candidates.

### 2.3 The dial space

A `Space` dataclass of ranges, every one a knob. Defaults are the
converged region and the ranges the reports point at:

| dial | default range | sampling |
|---|---|---|
| `stiffness_fraction` | 0.05 – 0.6 | log-uniform |
| `yield_percentile` | 1 – 15 | log-uniform |
| `heal_time_myr` | 20 – 500 | log-uniform |
| `damage_time_myr` | 1 – 30 | log-uniform |
| `strength_exponent` | {2, 3, 4} | uniform over the set |
| `strength_spread` | 0 – 0.1 | uniform |
| `drive_nodes` | {1, 2, 3} | uniform over the set |
| `drive_shear` | 0 – 1 | uniform |

Fixed per run, also knobs: `pixels` (default 1024), `scale_km` (5),
`history_myr` (300), `max_cycles` (40), `base_seed` (4287772760).

### 2.4 Stages

All sampling from `numpy.random.default_rng(search_seed)`; the engine
itself stays deterministic and the search is reproducible from
`search_seed`. Knobs: `stage1_cells` 200, `stage1_seeds` 4, `stage2_top`
10, `stage2_perturbations` 5, `stage2_seeds` 8, `stage3_top` 3.

1. **Stage 1, explore.** `stage1_cells` Latin-hypercube samples of the
   space, each run on `stage1_seeds` consecutive seeds from `base_seed`.
   Cells are submitted to the pool in a rolling window of four so results
   arrive continuously; each world is one pool task.
2. **Stage 2, refine.** Take every stage-1 passer plus the `stage2_top`
   best by soft score. For each, run it and `stage2_perturbations`
   Gaussian perturbations (10 % of each range's log-width or width;
   discrete dials resampled with probability 0.2) on `stage2_seeds` seeds.
3. **Stage 3, confirm.** Every stage-2 passer, plus the `stage3_top` best
   if none passed, on the twelve development seeds from `STATUS.md`. A
   cell that passes here is a **finding**. Write every view sheet for it.
4. **Loop.** If stage 3 produced no finding and the run has not been
   stopped, start a new stage 1 with `search_seed + 1`. Stop when a
   finding exists or the author stops it.

### 2.5 Persistence

`out/search/<run_id>/`: `config.json` (screen, space, stage knobs, seeds),
`cells.jsonl` appended per cell (stage, dials, seeds, per-world metrics,
pass/invalid, soft score, seconds), and per cell `plates.png` and
`trajectory.png` sheets; for stage-3 cells and findings, every view sheet.
The gallery reads from disk, so a run can be reopened after the server
restarts.

## 3. The server and the gallery

`pipeline_c/search_server.py`, a small Flask app on port **5004**, launched
by `pipeline_c/search.bat` (copy the launcher pattern; `WEBUI_RELOAD=0`).
Do not modify the shared shell in `webui/`; this is its own page.

Endpoints:

- `GET /` — the page, `pipeline_c/web_search/index.html`, served no-cache.
- `GET /api/config` — current screen, space, and stage knobs with defaults.
- `POST /api/start` — body is the knob values; validates; starts a run in a
  background thread that owns the process pool (eight workers, spawn
  context, created once). Returns `run_id`. Refuses if a run is active.
- `POST /api/stop` — asks the run to stop after in-flight cells finish.
- `GET /api/status` — run id, stage, cells done, passers, invalid count,
  cells per minute, best soft score, elapsed, running or stopped.
- `GET /api/cells?after=<n>&limit=<m>` — cells in completion order, newest
  last, with metrics and pass state, so the page can poll incrementally.
- `GET /api/cell/<id>/<sheet>.png` — a sheet from disk.
- `GET /api/runs` — previous runs on disk, to reopen one.

The page, vanilla JS, one file, no build step:

- a **knob panel**: every `Screen`, `Space`, and stage field, prefilled
  from `/api/config`, with a one-line meaning beside each; Start, Stop,
  Reopen run;
- a **status bar**: stage, cells done, passers, invalid, rate, best score;
- a **gallery**: one card per cell, newest first, passers pinned to the
  top and outlined, invalid cells greyed. Each card shows the `plates`
  sheet, the `trajectory` sheet below it, the dial values, and the per-
  term pass/fail as small labelled chips. Clicking a card opens a panel
  with every available sheet for that cell at full size and the dial
  values as copyable text, so the author can enter them in the
  exploration lab.

Poll `/api/status` and `/api/cells` every two seconds while a run is
active. No analysis rectangles or markers on any sheet; chips and outlines
live in the page, not in the images.

## 4. Defaults, and why

The defaults above are the ones this note thinks most likely to find a
regime if one exists: the converged stiffness range, low yield percentiles
because every sweep failed by too much failure, a heterogeneity range that
reaches zero because soft spots are the current suspect for seeding, and
exponent 2 in the set because it eases both zone self-sustainment and the
solver. The screen defaults are the six-term definition of a plate regime
this note and the author agreed on. If you have a measured reason to move
a default, do it and record the reason; do not move one on taste.

## 5. Tests

`tests/test_search.py`, no Flask, no pool:

- metrics on synthetic worlds: a thin closed loop of weak cells gives
  `network_share` 1.0 and `edge_fraction` 1.0; a filled disc gives a low
  edge fraction; two separated loops give `network_share` 0.5;
- the screen passes a synthetic world built to pass and fails it on each
  term in turn;
- `label_components` with 8-connectivity joins diagonal cells that
  4-connectivity does not;
- Latin-hypercube sampling is deterministic for a search seed and covers
  every range;
- the soft score is zero for a passer and grows monotonically with
  violation.

`tests/test_search_server.py`, Flask test client, pool replaced by a
sequential stub: config round-trip, start/stop, cells endpoint pagination,
sheet endpoint 404 on a bad id.

## 6. Run it once

Start the server as the launcher would, drive it over the API, and let
one stage-1 run of **50 cells** at 4 seeds complete (about five minutes),
then stop it. Report the status, the count of passers and invalid cells,
and the five best cells by soft score with their dial values and metrics.
Do not run stage 2 or 3 in this order unless a stage-1 cell passes, in
which case let the run continue through stage 3 and report the finding
with its sheet paths. Do not loop into a second stage 1.

## 7. Documents

`pipeline_c/SEARCH.md`, one page: how to start it (port 5004), what the
knobs mean, what the gallery shows, that a passing cell at twelve seeds is
a finding for the author's eyes and not an approval, and where runs are
stored. Add one line to `EXPLORE.md` pointing at it. `STATUS.md` "Now"
gains one sentence.

## 8. Report

`out/C03_7_BUILD_REPORT.md`:

1. **What was built**, file by file.
2. **Deviations**.
3. **Check output**, verbatim summary lines.
4. **The trial run** per §6.
5. **Throughput**: cells per minute at 4 and 8 seeds, worlds per second
   across the pool.
6. **Observations**, with evidence, no proposed values.

Do not describe any field as natural, plausible, good, or bad, and do not
compare anything to the reference images.
