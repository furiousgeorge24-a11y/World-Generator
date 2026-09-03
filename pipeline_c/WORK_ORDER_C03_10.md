# Work order — C03.10: the drive in kilometres

Issued 2026-09-02 for Opus 5. Follows [`WORK_ORDER_C03_9.md`](WORK_ORDER_C03_9.md)
and its report at `out/C03_9_BUILD_REPORT.md`. Same rules: isolation from
`pipeline_a`, `pipeline_b`, `pipeline_d`; NumPy, Pillow, Flask, and the
standard library; determinism of the engine; one change; no review
apparatus; do not commit; do not edit `DESIGN.md`. Do not describe any field
as natural, plausible, good, or bad, and do not compare anything to the
reference images. **Do not start, stop, or restart the search server on
port 5004, and do not post to its API.** Your runs go through the
exploration adapter's own pool in your process.

## 0. Purpose

`DESIGN.md` §2, ratified by the author, says that scale never changes with
resolution: a lower resolution at the same scale is a smaller planet with
features the same size, and the physics stays in kilometres while only the
grid follows scale. The engine breaks that rule in two places.

The mantle drive's coarsest wavelength is `parent / DRIVE_NODES_COARSEST`,
and the initial strength noise's is `parent / STRENGTH_NODES_COARSEST`. Both
are fractions of the parent world, so both halve when the parent halves. At
1024 px and 5 km/px the coarsest drive wavelength is 5,120 km; at 512 px it
is 2,560 km. Every driving feature at 512 px is half its 1024-px size in
kilometres, while damage, healing, and zone width stay in kilometres. C03.9
saw the consequence: three worlds passed the whole screen at 512 px and
none at 1024 px, on the same dials, because the plates shrank and the
boundaries did not. Until this is fixed, no measurement at any resolution
other than 1024 px says anything about the physics, and the cheap 512-px
search is not a search of the same world.

This order puts both wavelengths in kilometres. The production world at
1024 px and 5 km/px is byte-identical afterwards, because the new defaults
are exactly what that world had. Every other resolution and scale changes
by design, and that is the point.

**Prediction on record.** At 512 px with the drive at 5,120 km, the
frontier cells of `20260902T183110Z-s2` rerun on their own dials show
plate counts and weak fractions closer to their 1024-px values than the
512-px rerun in `out/ab_solve_20260902T183110Z-s2_512.md` showed, and the
three 512-px passes there do not recur. That is a prediction, not a goal; a
world at 512 px is a different world and the pairing is by dials.

## 1. Engine

### 1.1 Constants

In `engine/history/constants.py`, replace

```python
DRIVE_NODES_COARSEST = 2         # coarsest wavelength = parent / 2
STRENGTH_NODES_COARSEST = 8
```

with

```python
DRIVE_WAVELENGTH_KM = 5120.0     # coarsest mantle wavelength; parent / 2 at 1024 px, 5 km/px
STRENGTH_WAVELENGTH_KM = 1280.0  # coarsest strength-noise wavelength; parent / 8 there
```

`DRIVE_OCTAVES` and `STRENGTH_OCTAVES` stay. The finest wavelength of each
band is the coarsest over `2 ** (octaves - 1)`: 640 km for the drive and
80 km for the strength noise, both now fixed in kilometres too. Where a
grid cannot hold an octave — its wavelength under two cells — that octave
is simply absent, because the envelope has no mode to put it on; say so in
the docstring rather than raising.

### 1.2 Noise

`engine/noise.py`: `periodic_noise` accepts **either** `nodes_coarsest`
(kept, and now a float is allowed) **or** a new keyword `wavelength_km`,
exactly one of the two, and converts the wavelength to a cycle count as
`geometry.parent_km / wavelength_km`, a float. `_radial_envelope` takes the
cycle count as a float; `low = nodes` and `high = nodes * 2 ** (octaves - 1)`
as before, no clamping. `k = 0` is outside the band whenever `low > 0`, which
the validation guarantees. The envelope cache keys on the float.

A wavelength longer than the parent gives `nodes < 1`: the band then holds
only the lowest modes the torus has, and the world sees a piece of a larger
mantle cell. That is the intended behaviour at small worlds and needs no
special case; document it.

### 1.3 Drive and strength

`engine/history/drive.py`: `build_drive(geometry, *, wavelength_km=
DRIVE_WAVELENGTH_KM, rot_ratio, history_myr)` passes `wavelength_km` to
`periodic_noise` for every keyframe channel. `initial_strength` in
`kinematics.py` passes `wavelength_km=STRENGTH_WAVELENGTH_KM`.

### 1.4 Params

`HistoryParams.drive_nodes: int` becomes `drive_wavelength_km: float =
DRIVE_WAVELENGTH_KM`, validated as a number in `[100.0, 100000.0]`, recorded
by `to_record` under its new name. `run_history` passes it to `build_drive`.
There is no strength-wavelength dial; `strength_spread` stays the strength
noise's only dial.

### 1.5 Byte-identity

At 1024 px and 5 km/px the cycle counts are `10240 / 5120 = 2.0` and
`10240 / 1280 = 8.0`, exact in floating point, so the envelopes are the
arrays they were and every production output is byte-identical.
`tests/test_regression_c03_1.py` and the pre-order hash in
`tests/test_work_damage.py` are the gates and must pass unchanged. The
regression test's own noise calls use `nodes_coarsest=8` directly and are
untouched by design; they pin an integrator, not the drive.

## 2. Legacy dials

Every run under `out/search/` records `drive_nodes` in its dials. Add
`search.modernize_dials(dials, pixels, scale_km)`: if `drive_nodes` is
present and `drive_wavelength_km` is not, set
`drive_wavelength_km = WorldGeometry(0, pixels, scale_km).parent_km /
drive_nodes` and drop `drive_nodes`; otherwise return the dials unchanged.
`params_of`, `tools/pair_runs.py`, and `tools/ab_solve.py` call it on every
cell they read, so the runs on disk stay rerunnable and pairable. At 1024 px
the conversion is exact and a rerun reproduces the logged metrics.

## 3. The dial, in the lab and in the search

- Exploration lab: `drive_nodes` in `_DIALS` becomes `drive_wavelength_km`,
  float, default 5120, lo 640, hi 40960, tier primary, promise "Coarsest
  mantle wavelength in kilometres, the same at every resolution and scale.
  It sets how many mantle cells the world holds and so how many plates can
  form; 5,120 km is two cells across the default 1024-px world."
- Search: `Space.drive_nodes_set` becomes `drive_wavelength_km_lo /
  _hi`, sampled log-uniform; `DIALS` gains `("drive_wavelength_km", "log")`
  in place of the set entry; `params_of` threads it. The corner's default
  set `{1, 2}` at 1024 px becomes the range 5120 – 10240 km. `search_server.py`
  offers the two knobs with lo 640 and hi 40960 and their meanings.
  `SEARCH.md`'s whole-space table gets the equivalent range for `{1, 2, 3}`,
  3413 – 10240 km.
- `tools/ab_solve.py` prints the wavelength where it printed the node count.

## 4. Tests

- `test_noise.py`: `periodic_noise(wavelength_km=parent / 2)` is bit-identical
  to `periodic_noise(nodes_coarsest=2)` at 1024 px; passing both keywords or
  neither raises; a non-integer cycle count produces a field whose radial
  power spectrum, via `tools/spectrum.py`, has its low edge at that count.
- Scale invariance: the drive potential's dominant wavelength in kilometres
  is the same, to within one spectral bin, at 512 px and at 1024 px for the
  default wavelength, using `tools/spectrum.py` on the first keyframe.
- `test_history.py`: `HistoryParams` refuses 0, a negative, and `True`; the
  record carries `drive_wavelength_km` and not `drive_nodes`.
- `test_search.py`: `modernize_dials` converts a 1024-px `drive_nodes = 2`
  to exactly 5120.0 and leaves modern dials alone; `params_of` on a legacy
  cell's dials works; the hypercube covers the log range.
- `test_search_server.py`: the two knobs are offered and round-trip.
- `test_ab_solve.py` / `test_pair_runs.py`: a synthetic legacy run is read
  and modernized.

The full suite must pass:

```
py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"
```

## 5. Run it once

1. **The gate at production size.** Rerun the first three cells of
   `20260902T183110Z-s2` on their own seeds at 1024 px through the legacy
   conversion (`tools/ab_solve.py` with a new `--divisors 2` option that
   runs only the listed divisors) and show every logged metric reproduced
   to 1e-9. This is the byte-identity proof on the production path through
   real dials.
2. **512 px, the prediction.** Rerun the same twenty-cell selection C03.9
   used, at 512 px, divisor 2, on the cells' own seeds, with the dials
   modernized (so the drive is at 5,120 km, not 2,560). Report per cell the
   mean plate count, weak fraction and edge fraction beside the 1024-px
   divisor-2 values from `out/ab_solve_20260902T183110Z-s2.md` and the old
   512-px values from `out/ab_solve_20260902T183110Z-s2_512.md`, and say
   whether the three passes recur. A small script under `tools/` or a flag
   on `ab_solve.py` is fine; keep it deterministic and write the page to
   `out/c03_10_512_rerun.md`.
3. **Sheets.** One eight-seed bundle at 512 px and one at 1024 px at the
   production defaults, `drive` and `plates` sheets, to
   `out/c03_10_<pixels>_<view>.png`. No analysis marks on them.

## 6. Documents

- `EXPLORE.md`: the dial's line.
- `SEARCH.md`: the two knobs, the corner table and the whole-space table
  in kilometres, and one sentence on legacy runs being modernized on read.
- `STATUS.md` "Now": one sentence that the drive and strength wavelengths
  are now in kilometres so every resolution is the same physics, with
  C03.9's 512-px passes named as the symptom that found it.
- `CONTRACT.md`: if it states the drive's wavelength as a parent fraction
  anywhere, amend that line and nothing else.

## 7. Report

`out/C03_10_BUILD_REPORT.md`:

1. **What was built**, file by file.
2. **Deviations**, with reasons.
3. **Check output**: the suite's verdict lines, verbatim.
4. **The gate**: the three-cell reproduction, as a table of maximum
   absolute difference per metric.
5. **512 px**: the page from §5.2 verbatim, and the prediction marked held
   or failed with the numbers that decide it.
6. **Observations**, with evidence, no proposed values.
