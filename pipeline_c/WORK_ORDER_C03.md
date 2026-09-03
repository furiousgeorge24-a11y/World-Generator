# Work order — C03: kinematic history, first build

Issued 2026-09-01 for Opus 5. The decisions that belong to the author or to
the design are fixed here: the domain, the scales, the mechanism, the
constants, the views, and what counts as done. Implementation choices inside
those are yours. Where this document is silent, use your judgement and
record the choice in the report; where it is explicit, follow it even if you
would have chosen differently, and say so in the report if you would have.

The design this implements is [`DESIGN.md`](DESIGN.md). Read it once before
starting, then work from this document. On conflict, this document wins for
the scope of C03; `CONTRACT.md` wins over both.

## 0. How to use this document

- Work the phases in order. Each phase ends with a check you run yourself.
  Do not start a phase until the previous phase's check passes.
- **Stop conditions** are marked ⛔. When one triggers, stop, write the
  report in §12, and do not work around it. A stop is a correct outcome.
- Never write code that touches, imports, reads, or references
  `pipeline_a`, `pipeline_b`, or `pipeline_d`. Not for ideas, not for
  comparison, not for a helper function. This is absolute.
- Do not commit. The author commits.
- Do not add documentation beyond what §3 names. Do not add certificates,
  content-hash records, snapshots, baselines, deltas, review packets, or
  approval workflows of any kind. If you feel a stage needs one, it does not.
- Dependencies are NumPy, Pillow, and the standard library. No SciPy, no
  Numba, nothing else. If something seems to need more, that is a ⛔.
- Determinism is not optional. The same inputs must produce byte-identical
  outputs on the same machine. No wall clock, no `random`, no
  `np.random`, no hash-seed dependence, no dict-order dependence on sets,
  no threading with non-deterministic reductions.
- Python is `py -3.14`. Run every check from the repository root
  (`c:\Users\kstre\Desktop\mapgen`).

## 1. Goal and definition of done

Pipeline C returns to the level of completion it had before its C02
mechanism was judged unfit: a foundation, one generating stage visible in
the WebUI for any seed, a test suite, and the layer audit wired to the new
stage. The stage is different. It is **C03, the kinematic history**: a
lithosphere on a periodic grid with a mantle drive, a strength field that
weakens under strain and heals slowly, a velocity solved from the two, and
plates that emerge from where strain localizes. There is no crust, no
elevation, no water, and no land. It is not a map, and nothing may call it
one.

Done means all of the following are true:

1. The superseded code in §3.1 is deleted and the documents in §3.2 say what
   is true.
2. `pipeline_c\run.bat` starts the WebUI on port 5002, a seed generates a
   world at any supported size and scale, and every view in §9.3 renders.
3. Every test in §11 passes, and the two kept suites
   (`tests/eval_checks.py`, `tests/layer_audit_checks.py`) still pass.
4. `run_layer_audit.py build` publishes a batch from the new views.
5. `tools/contact_sheet.py` writes a 12-seed contact sheet for the `plates`
   view at 512 px, and the report in §12 is written.

Done does **not** mean the plates look right. That judgement is the
author's and is made by looking. Your report states what was built and what
the numbers were; it does not say the result is good, natural, or plausible.

## 2. Hard rules for the implementation

- **Units.** Horizontal distance in kilometres. Time in millions of years
  (Myr). Velocity in km/Myr. Sampler addresses in integer metres. Grid
  indices in cells. Never mix these silently; name variables with their unit
  suffix (`_km`, `_myr`, `_m`, `_cells`).
- **Array convention.** Every field is a NumPy array indexed `[row, col]` =
  `[y, x]`. Row 0 is the minimum-y row of the world. Views flip vertically
  at render time so north is up on screen. Vector fields are shape
  `(2, n, n)` with component 0 = x, component 1 = y.
- **Periodicity.** Every field on the history grid wraps in both axes.
  Every neighbour access is `np.roll` or explicit modulo. No `np.pad`, no
  edge modes, no clamping.
- **No graph searches, no nearest-point rules, no closed-form spatial
  formulas** in anything that produces a visible field. The only permitted
  stochastic input is the periodic noise of §6, sourced from the sampler.
- **No resampling of views.** Views render at native history resolution
  with nothing drawn on top. No text, no borders, no legends, no markers.
- **Every constant lives in one file**, `engine/history/constants.py`, with
  the values in §10. Nothing else defines a number that shapes a field.

## 3. Phase 0 — deletion and documents

### 3.1 Delete

Delete these paths entirely:

```
pipeline_c/engine/tectonic_fabric_c02/
pipeline_c/engine/foundation/audit.py
pipeline_c/engine/foundation/cache.py
pipeline_c/engine/foundation/cohorts.py
pipeline_c/engine/foundation/context.py
pipeline_c/engine/foundation/identity.py
pipeline_c/engine/foundation/state.py
pipeline_c/engine/foundation/geometry.py
pipeline_c/engine/foundation/constants.py
pipeline_c/engine/foundation/__init__.py
pipeline_c/engine/registry.py
pipeline_c/artifacts.py
pipeline_c/tests/adapter_checks.py
pipeline_c/tests/test_c4_foundation_engine.py
pipeline_c/tests/test_c5_c02_cohort.py
pipeline_c/tests/test_c5_c02_engine.py
pipeline_c/tests/test_c5_c02_support.py
```

Before deleting `tests/test_c4_foundation_engine.py`, copy every test in it
that exercises `StageSampler` or `SampleAddress` (digest values, address
encoding, range checks) into the new `tests/test_sampler.py` of §4.4,
adjusting imports only. Those tests pin the sampler's bytes and must not be
lost.

Then move `pipeline_c/engine/foundation/prf.py` to
`pipeline_c/engine/sampler.py`. It imports `require_hash`, `require_id`,
`require_int`, and `FoundationRecordError` from `foundation/_util.py`; move
those four names into a new `pipeline_c/engine/_util.py` (rename the error
class to `EngineRecordError`), delete `foundation/_util.py`, and remove the
now-empty `foundation/` directory. Change nothing else in the sampler. Keep
`KEY_SCHEDULE_ID = "pipeline-c-sha256-address-prf-v1"` in `sampler.py`.

Check `eval/` does not import anything deleted (it does not, as of this
writing; verify with a grep for `engine` and `artifacts` under `eval/`).
`eval/bundle.py` has its own copies of the artifact helpers.

Add `out/` to `pipeline_c/.gitignore`. Remove the `.review_store/` line; the
directory no longer exists.

### 3.2 Documents

Make exactly these edits. Do not restructure the files.

**`CONTRACT.md` §2.** Replace the bullet

> Horizontal geometry is stated in physical world units. Resolution changes
> sample the same world more finely rather than changing what the world is.

with

> Horizontal geometry is stated in physical world units. Scale, in
> kilometres per delivered pixel, is an author input with a fixed default.
> Resolution and scale together size the delivered window, and the simulated
> parent world is sized from that window, so a different resolution or scale
> is a different world. Features keep their physical size and their on-screen
> size at every resolution; a smaller map is a smaller piece of a smaller
> world, never a coarser sampling of the same one.

Replace the final paragraph of §2, from "The initial delivery target" to
"rather than selecting a new one.", with

> The delivery target is a square `size × size` raster with default
> `size = 1024` at a default scale of `5 km/px`. Every material attempt
> predeclares its supported sizes, its scale range, the parent-to-window
> ratio, and its sampling convention. Internal geometry carries width and
> height independently so square delivery is not a hidden architectural
> assumption. Rectangular delivery is not yet claimed. Scale is world
> geometry, not a formation control: it is held fixed, like the seed, by
> every same-family sweep of the author controls, and it is never swept.

**`CONTRACT.md` §5.** In the opening line "For a fixed seed, physical
delivered geometry, and implementation version:" leave the text as is; it
already covers scale under "physical delivered geometry". In the second
bullet, replace "at every target, fragmentation value, and supported
resolution" with "at every target and fragmentation value". Add nothing
else.

**`AUTHOR_RULINGS.md`, "Accepted defaults".** Replace the bullet beginning
"Delivered maps are square via the existing `size` interface" with

> - Delivered maps are square via the existing `size` interface, default
>   **`1024 × 1024`**, at an authorable scale of `5`–`20` km per pixel,
>   default **`5`**. Scale never changes with resolution; a lower resolution
>   is a smaller world, not smaller features. Internal geometry carries width
>   and height independently so rectangles remain possible later.

Replace the bullet beginning "Plate count is an **internal versioned
setting**" with

> - Plate count is emergent, not a setting. The internal versioned settings
>   are the drive field's wavelength and the lithosphere strength constants.
>   None of them may become a hidden synonym for fragmentation.

**`ROADMAP.md`.** Replace the paragraph "C4 (foundation) and C5 (tectonic
fabric) exist. The rest do not." with "C03 (foundation and kinematic
history) exists. The rest do not." In the build-order table, insert this row
above C6:

> | C03 | Foundation and kinematic history — periodic domain, sampler, mantle drive, strength and damage, velocity solve, emergent plates and boundaries. No crust. | Boundaries that curve, segment, and change regime along their length, on a spread of seeds |

Change the C6 row's "Responsibility" to "Crust on markers — creation,
transport, thickening, subduction, arcs, rifting, hotspots, age" and leave
its last column.

**`STATUS.md`.** Rewrite in full. Keep the same headings: Now, The open
question, What has been tried, Verification, Leftovers. Under "What has been
tried", keep the C00, C01, and C02 subsections verbatim except: change the
C00 heading to "C00 — parent-world foundation (superseded)", change the C02
heading to "C02 — connected competitive growth (rejected)", and append one
paragraph to C02: "**Rejected on sight** 2026-09-01: contacts locked to 0°,
45°, and 90° because growth was a shortest-path search on a four-neighbour
lattice; the directional cost only added banding on top. **Do not** return
to lattice graph search for any moving front." Under "Now", describe C03 in
three sentences: what exists, that it contains no crust, water, or land, and
that the author has not judged it. "The open question" becomes: whether
C03's boundaries curve, segment, and change regime along their length, per
`DESIGN.md` §1 and §8; undecided until the author looks. Replace the seed
list's sentence about D09 with nothing; keep the twelve seeds. "Leftovers"
becomes two sentences: `eval/` still holds the land instruments, which need
output that does not exist; the layer audit is wired to the C03 views.
Fill "Verification" from your actual final run.

**`README.md`.** Rewrite the "Running it" section's view table to list the
C03 views of §9.3 with one-line descriptions, change "about 3 seconds per
world" to the timing you measure, and change the paragraph beginning "What
is generated is the **tectonic fabric** stage only" to say the kinematic
history stage only, with no crust, elevation, water, coastline, island, or
land. Update the "Checks" block to the commands in §11. Change nothing else.

**Check:** `git status` shows only deletions, the moved sampler, the
document edits, and new files. `py -3.14 pipeline_c/tests/eval_checks.py`
and `py -3.14 pipeline_c/tests/layer_audit_checks.py` still pass.

## 4. Phase 1 — geometry and world identity

### 4.1 Layout

Create this tree. Empty `__init__.py` files where shown.

```
pipeline_c/engine/__init__.py          VERSION = "0.4.0-c03-kinematic-history"
pipeline_c/engine/_util.py             (from §3.1)
pipeline_c/engine/sampler.py           (from §3.1)
pipeline_c/engine/geometry.py
pipeline_c/engine/domain.py
pipeline_c/engine/noise.py
pipeline_c/engine/history/__init__.py
pipeline_c/engine/history/constants.py
pipeline_c/engine/history/drive.py
pipeline_c/engine/history/solver.py
pipeline_c/engine/history/kinematics.py
pipeline_c/engine/history/plates.py
pipeline_c/engine/views.py
pipeline_c/webui_adapter.py            (rewritten)
pipeline_c/run_layer_audit.py          (edited)
pipeline_c/tools/__init__.py
pipeline_c/tools/contact_sheet.py
pipeline_c/tests/test_sampler.py
pipeline_c/tests/test_geometry.py
pipeline_c/tests/test_noise.py
pipeline_c/tests/test_solver.py
pipeline_c/tests/test_history.py
pipeline_c/tests/test_plates.py
pipeline_c/tests/test_adapter.py
```

### 4.2 `engine/geometry.py`

```python
@dataclass(frozen=True, slots=True)
class WorldGeometry:
    seed: int          # 0 <= seed <= 2**32 - 1
    pixels: int        # in SUPPORTED_SIZES
    scale_km: int      # SCALE_MIN <= scale_km <= SCALE_MAX, integer
```

Validate in `__post_init__`: reject bools, non-ints, out-of-range values with
`ValueError`/`TypeError`. Derived properties, all integers:

| Property | Formula |
|---|---|
| `history_n` | `max(pixels // 2, MIN_HISTORY_N)` |
| `cell_km` | `CELL_PX * scale_km` |
| `parent_km` | `history_n * cell_km` |
| `window_km` | `pixels * scale_km` |
| `window_cells` | `pixels // CELL_PX` |
| `cell_m` | `cell_km * 1000` |
| `parent_m` | `parent_km * 1000` |

`world_id` property: SHA-256 hex of the canonical JSON bytes of
`{"pixels": pixels, "scale_km": scale_km, "schema": "pipeline-c-world-id:v2", "seed": seed}`
using `json.dumps(..., sort_keys=True, separators=(",", ":"))` encoded
UTF-8. `to_record()` returns a dict of every field and property above.

`cell_centre_m(index) -> int`: `cell_m * index + cell_m // 2`. `cell_m` is
always even (`cell_km * 1000`), so this is exact.

### 4.3 Expected values (put these in `tests/test_geometry.py`)

| pixels | scale | history_n | cell_km | parent_km | window_km | window_cells |
|---|---|---|---|---|---|---|
| 1024 | 5 | 512 | 20 | 10240 | 5120 | 256 |
| 512 | 5 | 256 | 20 | 5120 | 2560 | 128 |
| 128 | 5 | 256 | 20 | 5120 | 640 | 32 |
| 2048 | 5 | 1024 | 20 | 20480 | 10240 | 512 |
| 1024 | 20 | 512 | 80 | 40960 | 20480 | 256 |
| 2048 | 20 | 1024 | 80 | 81920 | 40960 | 512 |

Also test: `pixels=1000` rejected, `scale_km=4` and `21` rejected,
`scale_km=5.0` rejected (must be `int`), `seed=-1` and `2**32` rejected,
`world_id` differs between `(seed, 1024, 5)` and `(seed, 512, 5)` and between
`(seed, 1024, 5)` and `(seed, 1024, 6)`, and is stable across calls.

### 4.4 `tests/test_sampler.py`

The tests moved from the old foundation suite, imports adjusted to
`engine.sampler`. Plus one new test: `StageSampler(world_id, "kinematic_history.v1", "1", "x").uint64(0, 0)` for a fixed `world_id` string returns a value you record in the test after first computing it. That pins the sampler for the future.

Every new test file begins the way the existing suites do:

```python
from pathlib import Path
import sys
import unittest

PIPELINE_C = Path(__file__).resolve().parents[1]
if str(PIPELINE_C) not in sys.path:
    sys.path.insert(0, str(PIPELINE_C))

from engine.geometry import WorldGeometry  # noqa: E402
```

Top-level imports (`engine...`, `webui_adapter`) everywhere, in tests, in
the adapter, and in `run_layer_audit.py`. No `if __package__:` dual-import
blocks and no `tests/__init__.py`.

**Check:** `py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"`
passes with the two new files (the deleted suites are gone, so nothing else
is discovered yet).

## 5. Phase 2 — periodic domain utilities

### 5.1 `engine/domain.py`

All functions take and return NumPy arrays on an `n × n` periodic grid and
never allocate more than a few temporaries.

- `roll_x(a, k)`, `roll_y(a, k)`: `np.roll` along axis 1 and axis 0.
- `ddx(a)`, `ddy(a)`: periodic central differences,
  `(roll_x(a, -1) - roll_x(a, 1)) / 2`, per cell. Same for y.
- `grad(a) -> (2, n, n)`: `[ddx(a), ddy(a)]`.
- `perp_grad(a) -> (2, n, n)`: `[-ddy(a), ddx(a)]`.
- `div(v)`: `ddx(v[0]) + ddy(v[1])`.
- `sample_bilinear_periodic(a, x, y)`: `a` is `(n, n)` or `(k, n, n)`; `x`,
  `y` are float arrays of fractional cell coordinates of any shape (cell
  centre `i` is at coordinate `i`). Wrap with `np.mod`, take `floor`, blend
  the four neighbours. Vectorized, no loops.
- `tile2x2(a)`: `np.tile` for `(n, n)` or `(n, n, 3)` arrays, returning the
  field repeated twice in each axis.

### 5.2 `engine/noise.py`

One function:

```python
def periodic_noise(sampler: StageSampler, geometry: WorldGeometry, *,
                   channel: int, nodes_coarsest: int, octaves: int) -> np.ndarray
```

Returns an `(n, n)` float64 field with zero mean and unit standard deviation,
periodic on the grid by construction, fully determined by the sampler's key
and the arguments.

Algorithm, for octave `o` in `0 .. octaves-1`:

1. `L = nodes_coarsest * 2**o` lattice nodes per axis. Node `(j, i)` (row,
   col) sits at metres `x_m = (i * parent_m) // L`, `y_m = (j * parent_m) // L`.
   If `parent_m % L != 0`, that is a configuration error: raise. (It never is
   for the values in §10 and §4.3; the test checks.)
2. Node value: `sampler.unit_float(x_m, y_m, channel=channel, index=o)`.
   Build the `(L, L)` array with a double loop or a vectorized list
   comprehension; it is at most `nodes_coarsest² · 4^(octaves-1)` hashes.
3. Interpolate onto the grid: for cell centre coordinate `c = index + 0.5`
   in cells, lattice coordinate `u = c * L / n`; `k = floor(u) mod L`;
   `f = u - floor(u)`; `s = f*f*(3 - 2*f)` (smoothstep); bilinear blend of
   the four nodes `(k, k+1 mod L)` in each axis. Vectorized with
   `np.ix_`-style indexing.
4. Accumulate `amplitude * field` with `amplitude = 0.5**o`.

Finally subtract the mean and divide by the standard deviation (guard a zero
standard deviation by raising; it cannot happen with real hashes).

### 5.3 `tests/test_noise.py`

- Determinism: two calls with the same arguments are byte-identical.
- Independence: changing `channel` changes the field; changing the
  sampler's `process_id` changes the field.
- Seamlessness: for `geometry = WorldGeometry(1, 512, 5)`, the maximum
  absolute difference across the wrap edge (`a[:, 0] - a[:, -1]` and
  `a[0, :] - a[-1, :]`) is no larger than the maximum absolute interior
  neighbour difference. If it were, the wrap would be a seam.
- Normalization: mean within `1e-9` of 0, standard deviation within `1e-9`
  of 1.
- Scale invariance of the lattice: the node values at octave 0 for
  `(seed, 1024, 5)` and `(seed, 512, 5)` differ, because `parent_m`
  differs. Assert they differ. (This documents that different resolutions
  are different worlds.)

**Check:** `test_noise` passes.

## 6. Phase 3 — the drive field

### 6.1 `engine/history/drive.py`

The drive is the mantle's basal traction on the lithosphere, in km/Myr. It
is the sum of a curl-free part and a divergence-free part, each from a
periodic noise potential, and it evolves through three keyframes over the
history.

```python
@dataclass(frozen=True, slots=True)
class Drive:
    geometry: WorldGeometry
    phi: np.ndarray    # (KEYFRAMES, n, n) potentials for the curl-free part
    psi: np.ndarray    # (KEYFRAMES, n, n) potentials for the rotational part
    scale: float       # multiplier that sets the RMS speed

    def field(self, t_myr: float) -> np.ndarray:   # (2, n, n), km/Myr
    def potentials(self, t_myr: float) -> tuple[np.ndarray, np.ndarray]
```

Construction, `build_drive(geometry) -> Drive`:

- `sampler = StageSampler(geometry.world_id, STAGE_ID, STAGE_VERSION, "mantle-drive")`.
- For keyframe `k` in `0 .. DRIVE_KEYFRAMES-1`:
  `phi[k] = periodic_noise(sampler, geometry, channel=2*k, nodes_coarsest=DRIVE_NODES_COARSEST, octaves=DRIVE_OCTAVES)`,
  `psi[k] = periodic_noise(..., channel=2*k+1, ...)`.
- Raw field at keyframe 0: `D0 = grad(phi[0]) + DRIVE_ROT_RATIO * perp_grad(psi[0])`,
  in per-cell units. `scale = DRIVE_RMS_KM_PER_MYR / rms(|D0|)` where
  `rms(|D|) = sqrt(mean(D[0]**2 + D[1]**2))`. One scalar, fixed for the
  world, computed from keyframe 0 only.

`potentials(t)`: keyframes sit at `t_k = k * HISTORY_MYR / (KEYFRAMES - 1)`.
For `t` between `t_a` and `t_b`, `s = (t - t_a) / (t_b - t_a)`,
`w = 0.5 * (1 - cos(pi * s))`, and the potential is `(1 - w) * P[a] + w * P[b]`.
Clamp `t` to `[0, HISTORY_MYR]`.

`field(t)`: `scale * (grad(phi_t) + DRIVE_ROT_RATIO * perp_grad(psi_t))`.

### 6.2 Test (`tests/test_history.py`, first cases)

- `field(0)` has RMS speed within `1e-9` of `DRIVE_RMS_KM_PER_MYR`.
- `field(t)` is periodic-seamless by the same criterion as §5.3.
- `field(HISTORY_MYR / 2)` equals the field built from `phi[1], psi[1]`
  alone, within `1e-9` (the cosine weight is exactly 1 at a keyframe).
- Determinism.

## 7. Phase 4 — the velocity solver

### 7.1 The equation

For each velocity component `u` (x and y solved together as a stacked
`(2, n, n)` array with the same coefficients):

```
u - div( kappa * grad(u) ) = D
```

on the periodic grid, with `kappa = KAPPA0 * S**STRENGTH_EXPONENT`, where
`KAPPA0 = (HOMOG_LENGTH_FRACTION * history_n) ** 2` in cell² units. This
homogenizes velocity over a length `HOMOG_LENGTH_FRACTION * parent` inside
strong lithosphere and lets it jump across weak zones.

Discrete operator, cell `i` with its four neighbours `j`:

```
(A u)_i = u_i - sum_j k_ij * (u_j - u_i)
k_ij = 2 * kappa_i * kappa_j / (kappa_i + kappa_j + 1e-30)     # harmonic mean
diag_i = 1 + sum_j k_ij
```

Implement `apply_A(u, kappa)` and `diagonal(kappa)` with `np.roll`. Compute
the four edge coefficients once per solve (`k_east`, `k_north`; west and
south are rolls of those).

### 7.2 Geometric multigrid V-cycle

`solve(D, kappa, u0=None) -> (u, cycles, residual)`.

Levels: `n, n/2, n/4, ..., MG_COARSEST`. `n` is always a power of two
(§4.3), and `MG_COARSEST = 8`.

Per level, precompute `kappa_level`. Restriction of `kappa` to the next
coarser level: **harmonic mean over each 2×2 block, then divided by 4** (a
coarser cell is twice as wide, so the same physical diffusivity is a quarter
in cell² units; harmonic so a thin weak line stays a barrier).

Smoother: weighted Jacobi, `u <- u + MG_OMEGA * (rhs - A u) / diag`, or
red-black Gauss–Seidel with checkerboard masks if you prefer; either is
acceptable and deterministic.

Restriction of residuals and right-hand sides: mean over each 2×2 block.
Prolongation of corrections: piecewise constant (`np.repeat` along both
axes) or periodic bilinear, your choice. Bilinear converges faster on this
problem and is worth the extra twenty lines.

V-cycle at a level:

1. `MG_PRE` smoother sweeps on `u`.
2. `r = rhs - A u`; `rhs_c = restrict(r)`; `e_c = zeros`.
3. If the next level is the coarsest, do `MG_COARSE_SWEEPS` smoother sweeps
   on `e_c`; else recurse.
4. `u += prolong(e_c)`.
5. `MG_POST` smoother sweeps on `u`.

Driver: start from `u0` if given, else zeros. Repeat V-cycles until
`norm(rhs - A u) / norm(rhs) < MG_TOL` or `MG_MAX_CYCLES` cycles. Return
`u`, the cycle count, and the final relative residual.

⛔ If the driver cannot meet `MG_TOL` within `MG_MAX_CYCLES` on the barrier
test in §7.3 after you have tried the smoother and prolongation variants
above, stop and report the residual history. Do not loosen `MG_TOL`, do not
raise `MG_MAX_CYCLES` above 40, and do not substitute a solver that is not
a geometric multigrid on this operator. A multigrid-preconditioned conjugate
gradient is within bounds if you judge it necessary; note it as a deviation.

### 7.3 `tests/test_solver.py`

- **Constant coefficient versus FFT.** With `kappa` constant `= 100` on a
  `128²` grid and `D` a periodic noise field, the FFT solution is
  `u_hat = D_hat / (1 + kappa * (4 - 2 cos(kx) - 2 cos(ky)))` with
  `kx = 2π m / n`. The multigrid result must agree to relative L2 error
  below `1e-4`.
- **Barrier.** `256²` grid. `kappa = KAPPA0` everywhere except rows
  `100..101` and rows `200..201`, which get `KAPPA0 * STRENGTH_MIN**4`.
  `D[0] = +1` for rows `102..199`, `-1` elsewhere (two bands on the torus).
  The solve must converge within `MG_MAX_CYCLES`; the mean of `u[0]` over
  rows `120..180` must be within `0.05` of `+1` and over rows `220..255`
  and `0..80` within `0.05` of `-1`.
- **Lattice symmetry.** For arbitrary `kappa` and `D`, solving the transposed
  problem (`kappa.T`, `D` components swapped and transposed) gives the
  transposed, swapped solution to `1e-9`. Same for a 180° rotation
  (`[::-1, ::-1]` with both components negated). This is the check that the
  operator does not prefer an axis.
- **Warm start.** Solving again from the returned `u` converges in 1 cycle.

**Check:** `test_solver` passes. Record the barrier test's cycle count in
your report.

## 8. Phase 5 — the history loop, plates, boundaries

### 8.1 `engine/history/kinematics.py`

```python
@dataclass(slots=True)
class Epoch:
    t_myr: float
    strength: np.ndarray       # (n, n)
    velocity: np.ndarray       # (2, n, n) km/Myr
    strain_rate: np.ndarray    # (n, n) 1/Myr
    divergence: np.ndarray     # (n, n) 1/Myr

@dataclass(slots=True)
class History:
    geometry: WorldGeometry
    drive: Drive
    strength_initial: np.ndarray
    epochs: list[Epoch]          # at EPOCH_FRACTIONS of HISTORY_MYR; last is final
    weak_fraction: list[float]   # per step, fraction of cells with S < WEAK_THRESHOLD
    solver_cycles: list[int]     # per step
    solver_residual: list[float] # per step
    steps: int
    step_myr: float

def run_history(geometry: WorldGeometry, *, steps: int | None = None) -> History
```

`steps` defaults to `HISTORY_MYR // STEP_MYR`; tests may pass a smaller
number. `step_myr = HISTORY_MYR / steps` so a short run still spans the
whole drive schedule.

Initial strength:

```
sampler = StageSampler(world_id, STAGE_ID, STAGE_VERSION, "strength-initial")
noise = periodic_noise(sampler, geometry, channel=0,
                       nodes_coarsest=STRENGTH_NODES_COARSEST, octaves=STRENGTH_OCTAVES)
S = clip(STRENGTH_INIT_MEAN + STRENGTH_INIT_SPREAD * noise, STRENGTH_MIN, 1.0)
```

Per step, with `dt = step_myr`, `t` the time at the start of the step,
`cell = geometry.cell_km`:

1. `D = drive.field(t)`.
2. `kappa = KAPPA0 * S**STRENGTH_EXPONENT`; `v = solve(D, kappa, u0=v_prev)`.
   Record cycles and residual.
3. Strain, all in 1/Myr: `exx = ddx(v[0]) / cell`, `eyy = ddy(v[1]) / cell`,
   `exy = 0.5 * (ddy(v[0]) + ddx(v[1])) / cell`.
   `strain_rate = sqrt(exx² + eyy² + 2 exy²)`; `divergence = exx + eyy`.
4. Damage and healing:
   `S = S - dt * DAMAGE_RATE * (strain_rate / strain_ref)**2 * S + dt * HEAL_RATE * (1 - S)`,
   then clip to `[STRENGTH_MIN, 1]`, where
   `strain_ref = DRIVE_RMS_KM_PER_MYR / (2 * cell)`.
5. Advect strength with the lithosphere (semi-Lagrangian): the value at cell
   `(row, col)` after the step is `S` sampled at
   `(col - v[0] * dt / cell, row - v[1] * dt / cell)` with
   `sample_bilinear_periodic`. Strength belongs to the plate and moves with
   it; weak zones must not stay pinned to the mantle frame.
6. Record `weak_fraction = mean(S < WEAK_THRESHOLD)`.
7. Epochs are stored at step indices `round(f * steps)` for `f` in
   `EPOCH_FRACTIONS`, using the 1-based index of the step just completed.
   Compute those indices once before the loop, as a sorted set, so a short
   run whose fractions collide stores each step at most once. The last
   fraction is `1.0`, so the final step always stores the last epoch.
   `run_history` raises `ValueError` if `steps` is not a multiple of 4, which
   keeps the four epochs distinct for every caller; tests pass `steps=4`,
   `8`, or `12`.
8. `t += dt`.

Order matters: damage uses the strain of this step's velocity; advection
moves the damaged field. Keep exactly this order.

⛔ **Localization stop.** After a full default-length run at
`WorldGeometry(4287772760, 512, 5)`: if the final `weak_fraction` is below
`0.01` (nothing localized) or above `0.5` (everything failed), stop and
report the `weak_fraction` trajectory and the `strength` and `strain_rate`
views at every epoch. Do not tune constants to get past this; the author
decides what to change.

### 8.2 `engine/history/plates.py`

```python
def weak_mask(strength) -> np.ndarray            # bool, S < WEAK_THRESHOLD
def label_plates(strength) -> np.ndarray         # int32 (n, n), -1 on weak cells
def boundary_mask(labels, weak) -> np.ndarray    # bool
def regime(divergence, strain_rate, weak) -> np.ndarray   # int8: -1 none, 0 shear, 1 divergent, 2 convergent
def plate_areas(labels) -> np.ndarray            # cell counts by label, descending
```

`label_plates`: connected components of the strong cells (`~weak`) under
4-connectivity **on the torus**. Implement by label propagation in NumPy:
start with `labels = arange(n*n).reshape(n, n)`, set weak cells to a
sentinel, then repeat until no change: for each of the four rolls, where
both the cell and its rolled neighbour are strong, `labels = minimum(labels,
rolled_labels)`. Renumber components by area, largest first, starting at 0.
Weak cells are `-1`. This loop can take several hundred iterations on a big
plate; that is acceptable. It must be deterministic, and it is.

`boundary_mask`: a cell is boundary if it is weak, or if any of its four
neighbours has a different label.

`regime`: only on weak cells. `ratio = divergence / max(strain_rate, 1e-12)`.
`ratio > REGIME_RATIO` → 1 (divergent); `ratio < -REGIME_RATIO` → 2
(convergent); otherwise 0 (shear). Strong cells → −1.

### 8.3 `tests/test_plates.py`

- Two horizontal weak bands (rows `10..11` and `100..101`) on a `128²`
  strong field give exactly two plates on the torus, both touching the wrap
  edge, with areas `88*128` and `36*128`.
- One weak band gives one plate (the torus makes both sides the same plate).
- An isolated weak square gives one plate and the square's cells are `-1`.
- `boundary_mask` on the two-band case marks exactly the weak rows plus the
  strong rows adjacent to them.
- `regime` classifies a synthetic `divergence = +strain_rate` as 1 and
  `-strain_rate` as 2 and `0` as 0.

### 8.4 `tests/test_history.py` (remaining cases)

- **Determinism, full length.** `run_history(WorldGeometry(7, 128, 5))`
  twice (this is the `256²` floor grid); every epoch's `strength` and
  `velocity` are byte-identical (`np.array_equal` and equal `tobytes()`).
  Budget: this test may take up to 90 s. Print its elapsed time.
- **Short-run sanity.** `run_history(WorldGeometry(7, 128, 5), steps=8)`:
  every `strength` stays within `[STRENGTH_MIN, 1]`, `weak_fraction` has
  length 8, solver residuals all below `MG_TOL`, exactly four epochs are
  stored at steps 2, 4, 6, and 8. `steps=5` raises `ValueError`.
- **Strength moves.** With `steps=4`, the initial strength and the final
  epoch's strength differ (advection and damage happened).

**Check:** `test_history` and `test_plates` pass. Then run the localization
check of §8.1 once at `(4287772760, 512, 5)` and record the final
`weak_fraction`, the elapsed seconds, and the mean and maximum solver
cycles per step.

## 9. Phase 6 — views and the WebUI adapter

### 9.1 Palette

Append nine colours to `CATEGORY_COLORS` in `eval/palette.py`, after the
existing seven, so plates beyond seven are distinguishable:

```
(88, 160, 80), (200, 120, 60), (60, 140, 200), (170, 60, 60),
(120, 190, 160), (190, 170, 220), (150, 110, 40), (40, 90, 130), (220, 130, 130)
```

Do not change the first seven or `SCALAR_RAMP`; the audit controls depend on
them. `tests/layer_audit_checks.py` must still pass after this edit.

### 9.2 `engine/views.py`

Pure functions from arrays to `(n, n, 3)` uint8. No text or overlays.

- `categorical(labels)`: `labels % len(CATEGORY_COLORS)` through
  `categorical_rgb`; cells with label `-1` are `(0, 0, 0)`.
- `scalar(field)`: `scalar_rgb`.
- `banded(field, bands=8)`: quantize to `bands` equal-width levels between
  the field's min and max, then `scalar_rgb` of the level index. This is the
  contour companion `VIEWS.md` asks for.
- `mask(bool_field)`: white `(255, 255, 255)` where true, black elsewhere.
- `regime_rgb(regime)`: −1 → black; 0 → `(226, 201, 79)`; 1 →
  `(25, 187, 190)`; 2 → `(213, 73, 91)`.
- `vector(v)`: hue from direction, value from magnitude. `h = (atan2(v[1], v[0]) / (2π)) mod 1`,
  `s = 1`, `val = 0.15 + 0.85 * |v| / max|v|`. Convert HSV to RGB
  vectorized (the standard six-sector formula on arrays). No arrow glyphs.

Every view is flipped vertically (`[::-1]`) by the adapter at render time,
not here.

### 9.3 `webui_adapter.py`

Rewrite the module. Interface the shared shell expects, from
`webui/README.md`:

```python
def meta() -> dict
def generate(seed: int, controls: dict | None = None, size: int | None = None) -> World
def views(world) -> list[str]
def render_png(world, view: str) -> bytes
def report(world) -> dict
```

Keep the module-level `VIEWS` tuple and the `World` attributes `seed`,
`world_id`, `pixels`, `scale_km`; `run_layer_audit.py` uses them.

`meta()` returns:

```python
{
  "name": "pipeline_c land-origin lab",
  "version": VERSION,
  "ready": True,
  "stage": STAGE_ID,
  "status": "Kinematic history only: emergent plates and boundaries over a periodic parent world. No crust, elevation, water, coastline, island, or land.",
  "controls": [
    {"name": "scale_km", "ctype": "int", "default": SCALE_DEFAULT, "lo": SCALE_MIN, "hi": SCALE_MAX,
     "tier": "primary", "invalidates": "full",
     "promise": "Kilometres per delivered pixel. World geometry, not a formation control: it sizes the simulated planet and is never swept."}
  ],
  "default_size": DEFAULT_SIZE,
  "supported_sizes": list(SUPPORTED_SIZES),
  "views": list(VIEWS),
  "view_purposes": {...one sentence per view...},
}
```

There is no `hypsometric` view. Nothing here is elevation, and a fake one
would be a placeholder. The shell falls back to the first view; put `plates`
first.

`generate`: validate `seed` (uint32), `controls` (only `scale_km`, integer in
range, `bool` rejected, unknown names rejected with `ValueError`), `size`
(in `SUPPORTED_SIZES`, default `DEFAULT_SIZE`). Build `WorldGeometry`, call
`run_history`, compute the final epoch's labels, boundary mask, and regime,
and the same three for each earlier epoch. Time the whole thing with
`time.perf_counter`; that number is reported, never used.

`VIEWS`, in this order:

| View | Content | Source |
|---|---|---|
| `plates` | categorical labels, final epoch | §9.2 `categorical` |
| `boundaries` | boundary mask, final | `mask` |
| `regime` | divergent / convergent / shear on weak cells, final | `regime_rgb` |
| `strength` | strength ramp, final | `scalar` |
| `strength_banded` | 8 bands, final | `banded` |
| `velocity` | direction hue, magnitude value, final | `vector` |
| `strain_rate` | ramp, final | `scalar` |
| `strain_rate_banded` | 8 bands, final | `banded` |
| `drive` | drive field at `HISTORY_MYR` | `vector` |
| `drive_phi` | curl-free potential at `HISTORY_MYR` | `scalar` |
| `drive_psi` | rotational potential at `HISTORY_MYR` | `scalar` |
| `strength_initial` | the initial strength field | `scalar` |
| `plates_t25`, `plates_t50`, `plates_t75` | plates at the earlier epochs | `categorical` |
| `boundaries_t25`, `boundaries_t50`, `boundaries_t75` | boundaries at the earlier epochs | `mask` |
| `strength_t25`, `strength_t50`, `strength_t75` | strength at the earlier epochs | `scalar` |
| `plates_tiled` | `plates` tiled 2 × 2 at native resolution | `tile2x2` |

`render_png`: build the RGB, flip vertically, encode PNG with Pillow. Never
resize. A 1024 px world renders 512 × 512 views (1024 × 1024 for the tiled
one); that is correct and expected.

`report(world)` returns a JSON-serializable dict with: `seed`, `pixels`,
`scale_km`, `window_km`, `parent_km`, `history_n`, `cell_km`, `steps`,
`step_myr`, `history_myr`, `generation_seconds`, `world_id`, `stage`,
`plate_count` (labels with area ≥ 1 % of cells), `plate_area_percent`
(list, descending, of those plates), `weak_fraction_final`,
`weak_fraction_by_epoch`, `solver_cycles_mean`, `solver_cycles_max`,
`solver_residual_max`, `velocity_rms_km_per_myr`,
`contains: "emergent plate labels, boundaries, and kinematic fields only"`,
`does_not_contain: "crust, elevation, water, coastline, islands, land, or a map"`.
Cast every NumPy scalar to a Python `int`/`float`.

### 9.4 `tests/test_adapter.py`

- `meta()` has the keys above; the one control is `scale_km` with the
  stated range and type; `supported_sizes` equals `SUPPORTED_SIZES`.
- `generate(1, {"scale_km": 4})`, `{"scale_km": 21}`, `{"scale_km": True}`,
  `{"scale_km": 5.5}`, `{"unknown": 1}`, `size=1000`, `seed=-1` all raise.
- `generate(1, None, 128)` (floor grid) returns a world; `views(world)`
  equals `VIEWS`; every view's `render_png` decodes with Pillow to
  `(256, 256)` RGB, except `plates_tiled` at `(512, 512)`. Keep this test
  short by passing `_steps=8` to `webui_adapter.generate(seed, controls,
  size, *, _steps=None)`, a keyword-only parameter the shell never passes.
  It forwards to `run_history(steps=...)`, so the multiple-of-4 rule applies.
- `report(world)` is `json.dumps`-able and contains every key listed.
- Determinism: two `generate` calls with the same inputs give identical
  `render_png` bytes for `plates` and `strength`.

**Check:** `test_adapter` passes. Then `pipeline_c\run.bat`, open
`http://127.0.0.1:5002/`, generate seed `4287772760` at 1024 and 512, and
confirm every view renders. Report the wall time each took.

## 10. Constants — `engine/history/constants.py`

Exactly these names and values. Every number in this order that shapes a
field is here and nowhere else.

```python
STAGE_ID = "kinematic_history.v1"
STAGE_VERSION = "1"

SCALE_MIN = 5                    # km per delivered pixel
SCALE_MAX = 20
SCALE_DEFAULT = 5
SUPPORTED_SIZES = (128, 256, 512, 1024, 2048)
DEFAULT_SIZE = 1024
MIN_HISTORY_N = 256              # floor grid; parent never below 256 cells
CELL_PX = 4                      # one history cell is four delivered pixels

HISTORY_MYR = 300.0
STEP_MYR = 2.0
EPOCH_FRACTIONS = (0.25, 0.5, 0.75, 1.0)

DRIVE_RMS_KM_PER_MYR = 40.0      # 4 cm/yr
DRIVE_ROT_RATIO = 0.5            # rotational part relative to curl-free part
DRIVE_KEYFRAMES = 3
DRIVE_NODES_COARSEST = 2         # coarsest wavelength = parent / 2
DRIVE_OCTAVES = 4

STRENGTH_MIN = 0.05
STRENGTH_INIT_MEAN = 0.9
STRENGTH_INIT_SPREAD = 0.1
STRENGTH_NODES_COARSEST = 8
STRENGTH_OCTAVES = 5
STRENGTH_EXPONENT = 4            # kappa = KAPPA0 * S**4
HOMOG_LENGTH_FRACTION = 0.125    # homogenization length = parent / 8

DAMAGE_RATE = 0.2                # 1/Myr at the reference strain rate
HEAL_RATE = 0.01                 # 1/Myr
WEAK_THRESHOLD = 0.5
REGIME_RATIO = 0.3

MG_COARSEST = 8
MG_PRE = 3
MG_POST = 3
MG_OMEGA = 2.0 / 3.0
MG_COARSE_SWEEPS = 50
MG_TOL = 1e-5
MG_MAX_CYCLES = 20
```

`KAPPA0` is derived per world in code as
`(HOMOG_LENGTH_FRACTION * history_n) ** 2`; it is not a constant here
because it depends on the grid.

You may not change any value above to make a test pass or a stop condition
go away. If a value is wrong, that is a report item.

## 11. Phase 7 — audit runner, tools, and the full check

### 11.1 `run_layer_audit.py`

- `build` gains `--pixels` (default 1024) and `--scale` (default 5);
  `_candidate_sources` calls `webui_adapter.generate(seed, {"scale_km": scale}, pixels)`
  and records `pixels` and `scale_km` in `provenance["world"]`. `verify`
  regenerates with all three.
- Default `--views` becomes
  `plates,boundaries,regime,strength,strength_banded,velocity,strain_rate,strain_rate_banded,drive,strength_initial`.
  Epoch and tiled views are not audited by default.
- `DECLARED_MECHANISM`: `drive`, `drive_phi`, `drive_psi`,
  `strength_initial` → `"filtered_noise"`. Every other view → `None`. Do
  not declare a mechanism you cannot defend from the code.
- Nothing else in the runner changes. `eval/` is untouched apart from §9.1.

### 11.2 `tools/contact_sheet.py`

```
py -3.14 pipeline_c/tools/contact_sheet.py --view plates --pixels 512 --scale 5 --out pipeline_c/out/plates_512.png
```

Generates the twelve seeds in `STATUS.md` in that order, renders the named
view for each, and tiles them 4 across by 3 down with a 4-pixel black gutter.
Nothing drawn on the panels. Also accept `--seeds` as a comma list. Print
per-seed generation seconds and the report's `plate_count` and
`weak_fraction_final`.

### 11.3 The full check

```powershell
py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"
py -3.14 pipeline_c/tests/eval_checks.py
py -3.14 pipeline_c/tests/layer_audit_checks.py
py -3.14 pipeline_c/run_layer_audit.py build --seed 4287772760 --pixels 512
py -3.14 pipeline_c/tools/contact_sheet.py --view plates --pixels 512 --out pipeline_c/out/plates_512.png
py -3.14 pipeline_c/tools/contact_sheet.py --view boundaries --pixels 512 --out pipeline_c/out/boundaries_512.png
```

All must succeed.

### 11.4 Run the audit properly

`VIEWS.md` makes the blind layer audit the standing gate on new code, and
you are able to run it as designed. After `build`:

1. Read the judging plan it prints. Do **not** open anything under the
   hidden root.
2. For each call in the plan, spawn a fresh-context subagent, give it the
   prompt file and only the panel images assigned to that call (the images,
   not a directory listing), and collect its JSON verdict.
3. Concatenate the verdict arrays, run `score`, then `verify`.
4. Put the scorer's output and every candidate finding, with the judge's
   stated reproducing rule, in the report under a new section **Audit**.

If a candidate view is called formulaic, that is a finding for the author,
not a bug for you to fix: the mechanism and its constants are fixed by this
order. Report it with the evidence and move on. If the batch is void, say
so and include the calibration failure; do not rerun with a different seed
to get a non-void batch. One audit run, reported honestly, is what is
wanted.

The judge shares a model family with you. `VIEWS.md` explains why the
controls bound that weakness without removing it; note it in the report and
do not overstate what a clean batch means.

## 12. Report

Write `pipeline_c/out/C03_BUILD_REPORT.md` (gitignored; the author reads it
from disk) with exactly these sections:

1. **What was built.** File list with one line each.
2. **Deviations.** Every place you did something this order did not say, or
   did not do something it said, with the reason. "None" is a valid entry
   only if it is true.
3. **Stops.** Any ⛔ that triggered, with the evidence it asked for.
4. **Check output.** Verbatim final lines of every command in §11.3.
5. **Timing.** Seconds per world at 128, 512, 1024, and 2048 px at scale 5,
   and at 1024 px at scale 20, for seed `4287772760`. Mean and max solver
   cycles per step at 1024 px.
6. **Trajectories.** `weak_fraction` at each epoch for seeds `4287772760`,
   `2075014389`, and `1833546021` at 512 px. `plate_count` for all twelve.
7. **Contact sheets.** Paths to the two PNGs. No commentary on how they
   look. The author looks.
8. **Audit.** Per §11.4.
9. **Observations.** Anything you noticed that the author should know
   before looking: a constant that behaves as if it were at the wrong order
   of magnitude, a view that fails to show what it was meant to show, a
   place where the design's prediction in `DESIGN.md` §8 is visibly not
   met by a field you can measure. State the observation and the evidence.
   You may name the constant or mechanism it implicates. Do not propose a
   value; the author decides what to change and by how much.
10. **Profile.** Where the time goes at 1024 px, by phase, from one run.
    No optimization; this is a note for later.

Do not describe the plates as natural, plausible, good, or bad, and do not
compare them to the reference images. Those judgements are the author's,
by looking, and a report that pre-empts them is worth less than one that
does not.

## 13. Errata, recorded after the build

Found by the implementer and resolved in the build; kept here so the next
reader is not misled by the text above.

- **§8.1.7 / §8.4.** The default of 150 steps is not a multiple of four, so
  the multiple-of-four rule applies only to an explicitly passed `steps`.
  The epoch indices at 150 steps are distinct anyway.
- **§7.2.** The remark that bilinear prolongation "converges faster and is
  worth the extra twenty lines" is wrong for this operator. With a 2 × 2
  mean restriction, piecewise-constant prolongation is the exact adjoint,
  which makes one V-cycle symmetric and a valid conjugate-gradient
  preconditioner; bilinear breaks that and interpolates straight through a
  high-contrast barrier. The standalone V-cycle diverges on the barrier test
  in every variant; the working solver is a multigrid-preconditioned
  conjugate gradient, as §7.2 permits.
- **§11.1 / §11.3.** `PANEL_PX = 512` could not crop a 256-cell view, so the
  `--pixels 512` command could never have run. The runner now sizes panels
  from the smallest candidate view and passes the same size to the controls.
- **§10.** `strain_ref = DRIVE_RMS / (2 · cell_km)` sets the reference strain
  at a two-cell length while the solver homogenizes over `parent / 8`. The
  strain the solver permits is about sixteen times smaller, so damage never
  fired. This is an author error, corrected in C03.2 as that run's single
  mechanism change.
