# Work order — C03.5: the exploration lab

Issued 2026-09-02 for Opus 5. Follows [`WORK_ORDER_C03_4.md`](WORK_ORDER_C03_4.md)
and its report at `out/C03_4_BUILD_REPORT.md`. Same rules: isolation from
`pipeline_a`, `pipeline_b`, `pipeline_d`; NumPy, Pillow, standard library
(`concurrent.futures` and `multiprocessing` are standard library);
determinism; no review apparatus; do not commit; do not edit `DESIGN.md`.

## 0. Purpose

Four runs have each fixed what they targeted and moved the failure one
layer down. The remaining question is whether a stable localizing regime
exists anywhere in this model's parameter space, and a person with dials
and a five-second loop answers that faster than one change per run. This
order builds that loop as a **separate WebUI on its own port and tab**, so
the production adapter and the layer audit are untouched.

It also makes one engine change that is independent of the dials: the
initial strength noise is grid-aligned (axis bias 1.78 against 1.2 for an
isotropic control, in every run's audit) and enters the stiffness at the
fourth power. A dial sweep on top of that artifact would tune toward or
away from the grid. The noise generator is replaced by an isotropic one
before the dials exist.

The exploration dials are **development instruments**. They are not author
controls, they never appear in the production adapter, and whatever is
found with them gets frozen back into `constants.py` afterwards.

## 1. Engine: parameters as a record, constants as defaults

### 1.1 `HistoryParams`

Add to `engine/history/kinematics.py`:

```python
@dataclass(frozen=True, slots=True)
class HistoryParams:
    stiffness_fraction: float = HOMOG_LENGTH_FRACTION   # homogenization length / parent
    yield_percentile: float = 12.0   # percent of the step-1 strain field above yield
    heal_time_myr: float = 1.0 / HEAL_RATE
    damage_time_myr: float = 1.0 / DAMAGE_RATE           # time to fail at twice yield
    strength_exponent: int = STRENGTH_EXPONENT
    drive_nodes: int = DRIVE_NODES_COARSEST
    drive_shear: float = DRIVE_ROT_RATIO
    history_myr: float = HISTORY_MYR
    max_cycles: int = MG_MAX_CYCLES
```

`run_history(geometry, *, params=None, steps=None)` uses `params` or the
defaults. Every place the loop, `build_drive`, `kappa0_for`, and `solve`
read one of these constants now reads the field instead. `constants.py`
keeps every value; it is the source of the defaults and nothing else
changes there. The production adapter passes no `params`.

Validate ranges in `__post_init__`: `0.02 ≤ stiffness_fraction ≤ 4`,
`0.5 ≤ yield_percentile ≤ 50`, `5 ≤ heal_time_myr ≤ 2000`,
`0.5 ≤ damage_time_myr ≤ 200`, `1 ≤ strength_exponent ≤ 8`,
`1 ≤ drive_nodes ≤ 6`, `0 ≤ drive_shear ≤ 2`, `50 ≤ history_myr ≤ 1000`,
`5 ≤ max_cycles ≤ 200`.

### 1.2 Yield from a percentile of the step-1 strain

Replace the `YIELD_STRAIN_FRACTION` derivation. At step 1, after the first
strain field is computed on the solve grid and before damage:

```
yield_strain_per_myr = percentile(strain_rate_s, 100 - params.yield_percentile)
```

with `np.percentile(..., method="linear")` on the solve-grid field. Store it
on `History` as before. At the default percentile of 12 on seed
`4287772760` at 1024 px this reproduces run 4's yield to within a few
percent (run 4 measured 12.7 % above a yield of 0.01964); report the
exact default-yield value so the two can be compared.

This is a calibration convenience: it makes the yield dial mean the same
thing at every stiffness, because a stiffer sheet has smaller strains
everywhere. It is a statistical self-reference, not a physical law, and it
is **not** to survive into production: once a setting is chosen, the
equivalent physical fraction of the drive's characteristic strain is
frozen as a constant and the percentile goes. Say this in the docstring.
Remove `YIELD_STRAIN_FRACTION` from `constants.py`; the production default
is the percentile default.

### 1.3 Rates from times

`HEAL_RATE = 1 / heal_time_myr`, `DAMAGE_RATE = 1 / damage_time_myr`,
computed once per run from the params. Defaults reproduce 0.01 and 0.2.

### 1.4 Steps

`steps = round(params.history_myr / STEP_MYR)`, `STEP_MYR` unchanged at 4.
Epochs at the same fractions.

## 2. Engine: isotropic noise

Replace the body of `periodic_noise` in `engine/noise.py`. Signature and
normalization stay; the lattice goes.

1. **White noise per cell from the sampler.** For the `n × n` grid, one
   `unit_float` per cell at that cell's centre in metres, `channel` as
   given, `index = 0`, mapped to a standard normal by the inverse error
   function (`math.erf` has no inverse in the standard library; use a
   Box–Muller pair from two uniforms, drawing the second uniform at
   `index = 1`, and keep only the first normal so each cell costs two
   hashes). At 256² that is 131k hashes, well under a second; cache the
   white field per `(world_id, stage, process, channel, n)` with
   `functools.lru_cache` on a helper, since drive and strength fields ask
   for it repeatedly.
2. **Radial spectral envelope.** `F = fft2(white)`. With `kx, ky` the
   integer cycle counts per parent axis, `k = sqrt(kx² + ky²)`, the
   envelope is `k ** -1` for `nodes_coarsest ≤ k ≤ nodes_coarsest ·
   2 ** (octaves - 1)` and `0` elsewhere, including `k = 0`. This matches
   the old value noise's amplitude halving per octave over the same band,
   so the two share a spectrum and differ only in isotropy. Multiply, take
   `ifft2`, keep the real part.
3. Normalize to zero mean and unit standard deviation as before.

The result is periodic by construction, isotropic by construction, and
deterministic (NumPy's FFT is deterministic on one machine; the
determinism tests will say so).

Tests in `tests/test_noise.py`: keep determinism, channel independence,
seamlessness, normalization. Replace the lattice-divisibility test with an
**isotropy** test: mean power density in ±15° wedges about the k-axes over
the same about the diagonals, `4 ≤ |k| ≤ n/2`, on the raw noise field at
`n = 256`, must be within `0.9`–`1.1`. Add the same statistic for the
initial strength field in `tests/test_history.py` with bound `1.15`.

Record in the report the statistic on the initial strength before and
after, on seed `4287772760` at 1024 px, alongside the 1.777 every audit has
measured.

## 3. The exploration adapter

### 3.1 `pipeline_c/explore_adapter.py`

A second adapter module for the same shared shell, on its own port. It
imports the engine and the production adapter's view functions; the
production `webui_adapter.py` is **not modified** beyond what §1 and §2
require to keep it running with defaults.

`meta()`:

- `name`: `"pipeline_c exploration lab"`, `version`, `ready: True`,
  `stage: STAGE_ID`, `status`: one sentence saying these are development
  dials, not author controls, and that eight seeds are shown per setting.
- `default_size`, `supported_sizes` as production.
- `controls`, in this order, all `invalidates: "full"`:

| name | ctype | default | lo | hi | tier | promise (one line) |
|---|---|---|---|---|---|---|
| `scale_km` | int | 5 | 5 | 20 | advanced | as production |
| `seeds_per_view` | int | 8 | 1 | 8 | primary | Worlds per generation, seeds `seed`…`seed+n-1`, shown side by side |
| `stiffness_fraction` | float | 0.125 | 0.05 | 2.0 | primary | Fraction of the world over which a plate holds together. Below a plate's size the plate deforms internally |
| `yield_percentile` | float | 12 | 1 | 40 | primary | Percent of the initial strain field above yield. What breaks first |
| `heal_time_myr` | float | 100 | 10 | 1000 | primary | Time for a fault to seal once it stops moving |
| `damage_time_myr` | float | 5 | 1 | 100 | primary | Time for intact rock at twice yield to fail |
| `strength_exponent` | int | 4 | 2 | 6 | advanced | How steeply stiffness falls with damage |
| `drive_nodes` | int | 2 | 1 | 4 | advanced | Coarsest mantle wavelength, world over this |
| `drive_shear` | float | 0.5 | 0 | 1 | advanced | Rotational drive relative to pushing drive |
| `history_myr` | int | 300 | 100 | 600 | advanced | How long the history runs |
| `max_cycles` | int | 40 | 10 | 100 | advanced | Solver effort per step; the report shows the worst residual |

`generate(seed, controls, size)`: validate, build `HistoryParams`, and run
`seeds_per_view` worlds for seeds `seed, seed+1, …` (wrapping at 2³²) in
parallel. Return a bundle.

### 3.2 Parallelism

A `concurrent.futures.ProcessPoolExecutor` with `max_workers = 8`, created
lazily on the first generate and kept for the life of the server process,
`mp_context = multiprocessing.get_context("spawn")`. The worker function is
a module-level function taking `(seed, pixels, scale_km, params_dict)` and
returning a plain dict of the arrays and numbers the views and report need
(strength epochs, weak masks at the early steps and epochs, labels, regime,
velocity, strain rate, trajectory lists, solver stats). Nothing but NumPy
arrays, floats, ints, and lists crosses the process boundary.

If the pool cannot be created (the shell's reloader is on, or any
`OSError`), fall back to sequential generation and say so in the report as
`parallel: false`. Determinism must not depend on which path ran: add a
test that a 2-world bundle from the pool equals the same bundle generated
sequentially, byte for byte.

### 3.3 Views

Every view is a contact sheet of the bundle: panels in reading order,
`ceil(n/4)` rows of up to 4, native resolution, a 4 px black gutter, no
text. With one seed it is one panel with no gutter.

| view | panel content |
|---|---|
| `plates` | final labels |
| `boundaries` | final boundary mask |
| `weak_t16`, `weak_t32`, `weak_t64` | weak mask at 16, 32, 64 Myr (from the early snapshots; add 64 if it is not already stored) |
| `weak_t25`, `weak_t50`, `weak_t75` | weak mask at the epochs |
| `strength` | final strength |
| `strength_t25`, `strength_t50`, `strength_t75` | strength at the epochs |
| `regime` | final regime |
| `velocity` | final velocity |
| `strain_rate` | final strain rate |
| `trajectory` | see below |
| `drive` | drive at the end |

`trajectory`: per world, a strip `steps` pixels wide and 64 pixels tall,
weak fraction against time drawn as a filled column per step (height =
fraction × 64) in the palette's third colour on black, with a one-pixel
line at the 50 % height in the palette's first colour. The eight strips
stack vertically with a 2 px gutter. No axes, no text. It is the one view
that separates a stable regime from a slow collapse.

Put `plates` first. There is no `hypsometric`.

### 3.4 Report

```
{
  "dials": {…every control value…},
  "yield_strain_per_myr": [per world],
  "parallel": true/false,
  "generation_seconds": total wall,
  "worlds": [
    {"seed": …, "plate_count": …, "plate_area_percent": […],
     "weak_final": …, "weak_peak": …, "weak_peak_myr": …,
     "weak_at_100_myr": …, "strength_mean_strong": …,
     "solver_cycles_mean": …, "solver_residual_max": …,
     "exhausted_steps": …}
  ],
  "summary": {"plate_count_min": …, "plate_count_max": …,
              "weak_final_mean": …, "stable_count": …}
}
```

`stable` for one world means: `3 ≤ plate_count ≤ 8`, `0.02 ≤ weak_final ≤
0.25`, and `weak_peak / weak_final < 1.5` (the weak set is not still
growing at the end or collapsing). `stable_count` is how many of the
bundle meet it. This is a screening number for the person at the dials,
not a gate and not an approval; say so in the key's name or a `note`.

### 3.5 Launcher

`pipeline_c/explore.bat`, a copy of `run.bat` with port `5003`, backend
`explore_adapter`, and `set WEBUI_RELOAD=0` before the server starts (the
process pool and the reloader do not mix). It reuses `prepare_webui.ps1`
with `-Port 5003 -Backend explore_adapter`; check that script's
registry-name check does not reject the exploration name, and if it does,
pass the name through rather than weakening the check.

## 4. A first map of the space

Before handing over, run a coarse sweep so the author starts with a map
rather than a blank panel. Seeds `4287772760 … +7`, 1024 px, scale 5, all
other dials at default, `max_cycles = 40`:

- `stiffness_fraction` ∈ {0.125, 0.25, 0.5, 1.0, 2.0}
- `yield_percentile` ∈ {3, 6, 12, 20, 30}

Twenty-five cells, eight worlds each, in the pool. For each cell report
`stable_count`, mean `weak_final`, mean `plate_count`, mean
`solver_residual_max`, and mean seconds per world. Write it as a table to
`out/c03_5_sweep.md` and to `out/c03_5_sweep.csv`. Then, for the cell with
the highest `stable_count` (ties: lowest stiffness), write its eight
`plates` and `trajectory` sheets to `out/`. Do not run a second sweep and
do not adjust any other dial; this is a map, not a search.

## 5. Guide

`pipeline_c/EXPLORE.md`, one page:

- how to start it and that it is a separate tab on port 5003;
- what each dial means physically, one line each, in the order of §3.1;
- what to look for: a flat `trajectory` strip after the first 50 Myr; a
  `plates` sheet where all eight worlds have several plates; `weak_t16`
  showing lines rather than blobs;
- that stability on eight seeds is a screen, and any setting worth keeping
  is then run on the twelve development seeds and the layer audit before
  it is frozen into `constants.py`;
- that the dials are development instruments, not author controls, and the
  percentile yield in particular is a calibration convenience that does
  not survive into production.

No more than that.

## 6. Tests

- `HistoryParams` validation and defaults; `run_history` with default
  params is byte-identical to `run_history` with no params.
- Percentile yield: at the default it reproduces run 4's yield on seed
  `4287772760` at 1024 px within 5 %; at percentile 50 it equals the
  median of the step-1 solve-grid strain.
- Noise isotropy and initial-strength isotropy, per §2.
- Explore adapter: meta shape; a 2-world bundle at 128 px renders every view
  at the expected sheet size; report keys; pool equals sequential;
  determinism across two generates.
- Production adapter: still passes its suite, with the expected changes
  from the noise (say which tests changed and why).

## 7. Report

`out/C03_5_BUILD_REPORT.md`:

1. **What changed**, file by file.
2. **Deviations**.
3. **Noise**: the isotropy statistic on the raw field and on the initial
   strength, before and after.
4. **Default reproduction**: the default-dial yield versus run 4's, and
   the default-dial `weak_final` and `plate_count` for seed `4287772760`
   versus run 4's (they will differ because the noise changed; report
   both).
5. **Check output**, verbatim summary lines.
6. **Timing**: 8-world generate at 1024 px, parallel and sequential; 1-world
   at 1024 px.
7. **The sweep** per §4, with the table and the two sheets' paths.
8. **Observations**, with evidence. No proposed values; the dials exist so
   the author can turn them.

Do not describe any field as natural, plausible, good, or bad, and do not
compare anything to the reference images.
