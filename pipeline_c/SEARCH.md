# The regime search

A third WebUI, on its own port and its own tab, that turns the exploration
lab's dials by itself. It samples the dial space, runs several seeds per
setting, measures each world, and screens the result on six measured
properties of a plate regime. The production lab on port 5002 and the
exploration lab on port 5003 are untouched.

```powershell
pipeline_c\search.bat
```

That starts it on port `5004`. Set the knobs, press Start, and watch the
gallery fill. Press Stop and the run finishes the cells already in flight and
ends. Nothing else ends it: a run loops through rounds until it is stopped or
the server goes down, and a finding does not stop it.

## What it does

Three stages, then a loop.

1. **Stage 1, explore.** `stage1_cells` Latin-hypercube samples of the dial
   space, each run on `stage1_seeds` consecutive seeds from `base_seed`. Cells
   go to the pool in a rolling window, so results arrive continuously rather
   than in stage-sized lumps.
2. **Stage 2, refine.** Every stage-1 passer, plus the `stage2_top` best by
   soft score, each run again and with `stage2_perturbations` Gaussian
   perturbations, on `stage2_seeds` seeds.
3. **Stage 3, confirm.** Every stage-2 passer, on the twelve development seeds
   of [`STATUS.md`](STATUS.md). A cell that passes here is a **finding**, and
   every view sheet is written for it. If nothing passed, the `stage3_top`
   best go through anyway, so there is always something to look at.
4. **Loop.** Not stopped: a new stage 1 at `search_seed + 1`. Every round is
   a blind restart. A finding is recorded and pinned at the top of the
   gallery, and the run goes on; more findings are more candidates.

The whole search is reproducible from `search_seed`; the engine itself is
deterministic and nothing about the pool changes a world.

## The knobs

Every field of the screen, the space, and the stages is a knob on the page,
with its meaning printed beside it. The ones worth knowing before you start:

### Screen — what counts as a plate regime

Six terms decide whether one world passes. Every one is a measured property
of the plate regime. **None of them says how a field looks**, and none of them
is a comparison to anything.

| Knob | Default | Term |
|---|---|---|
| `weak_min`, `weak_max` | 0.02, 0.25 | Final weak fraction: some lithosphere failed, most did not. |
| `peak_ratio_max` | 1.5 | Peak weak fraction over final: the weak set is not collapsing back or overshooting. |
| `flat_window_myr`, `flat_tolerance` | 100, 0.03 | The weak fraction has stopped moving over the last 100 Myr. |
| `plates_min`, `plates_max` | 3, 8 | Plates above 1 % of the parent. |
| `network_share_min` | 0.5 | Largest 8-connected component of the weak set, as a share of it: a connected network, not speckle. |
| `edge_fraction_min` | 0.5 | Weak cells with a strong neighbour, as a share of the weak set. A line `w` cells wide gives about `2 / w`, so 0.5 means width four or less. |

Two more knobs are not terms:

- **`residual_max`**, default `1e-3`. Any world above it and the whole cell is
  **invalid**: its velocity fields were never solved, so its plates, weak
  fraction and trajectory are readings off an unfinished iterate. An invalid
  cell is greyed in the gallery and is never scored.
- **`pass_fraction`**, default 1.0. The share of a cell's worlds that must
  pass every term for the cell to pass.

A cell that does not pass gets a **soft score**: each term's violation, in
units of that term's own interval width, summed and averaged over the cell's
worlds. Zero means a passer; lower is closer. It orders the stage-2
candidates and nothing else, and it contains no aesthetic term.

### Space — the dial ranges sampled

Ten dials, sampled per cell: `stiffness_fraction`, `yield_percentile`,
`heal_time_myr`, `damage_time_myr` and `drive_wavelength_km` log-uniform over
their ranges; `strength_spread` and `drive_shear` uniform; `strength_exponent`
uniform over a set you type as a list. Their meanings are the ones in
[`EXPLORE.md`](EXPLORE.md), and `strength_spread` is the ninth dial the lab
gained with this search: the initial heterogeneity of the lithosphere, whose
soft spots concentrate strain.

`drive_wavelength_km` is two knobs, `drive_wavelength_km_lo` and `_hi`, each
between 640 and 40,960 km. It is the coarsest mantle wavelength as a length,
so the same number means the same size of mantle cell at every resolution and
scale; it replaced the `drive_nodes` set, which was a count of cycles across
whatever parent the run's resolution produced.

**Runs written before either change are modernized on read.** Every cell
written before `WORK_ORDER_C04.md` also lacks `seams`,
`crack_speed_km_per_myr` and `nucleations_per_step`; `search.modernize_dials`
fills them with the engine's own defaults, which is what those runs ran on.

**Runs written before the kilometre change are modernized on read too.** Every cell under
`out/search/` records `drive_nodes`; `search.modernize_dials` converts it to
the wavelength it meant at that run's own resolution and scale, so those runs
stay rerunnable and pairable and a rerun at the run's own size reproduces the
metrics it logged.

`crack_speed_km_per_myr` and `nucleations_per_step` are the two the seam
formulations added: how fast a crack tip runs, log-uniform over **10 to
200 km per million years**, and how many new cracks a step may open, uniform
over the set **{1, 2, 4}**. Both are read only under the seam formulation and
both are recorded either way.

`toughness_fraction` is the third, added by
[`WORK_ORDER_C04_4.md`](WORK_ORDER_C04_4.md) §2 and sampled log-uniform over
**0.1 to 1.0**: the fracture toughness as a fraction of the intact strength,
so a tip propagates at `toughness_fraction * sigma_c / sqrt(L)` and
nucleation still needs the full intact strength. It is the last column of the
hypercube, so a run written before it existed redraws every earlier column
exactly; `search.modernize_dials` fills a legacy cell with **1.0**, the
constant the tip rule carried implicitly before the dial existed.

**`seams` is fixed for the run and its default is 2.** `0` is the sheet,
diffuse damage wherever strain exceeds yield; `1` is the seam formulation of
[`DESIGN.md`](DESIGN.md) §3.6, damage only on a seam, at its tip, or at a
nucleation site, on the sheet's own velocity solve; `2` is the block model of
that section's last paragraph — pieces are rigid bodies coupled through seam
tractions, the stress the seam rules read is the sheet solve of the drag a
piece failed to match rather than the sheet's own, and the seam network is
carried on markers that cannot duplicate. It is fixed rather than sampled for
the reason `work_damage` is: a given `search_seed` then draws the same Latin
hypercube whichever it is, and two runs differing only in it are an ablation
pair. **The width term of the screen is now satisfied by construction rather
than searched for** — a seam is one cell wide because nothing beside it is
allowed to weaken, and at `2` because a marker cannot be written twice — so
the question the search is left with is plate count and settling.

`pixels`, `scale_km`, `history_myr`, `max_cycles`, `work_damage`, `seams`,
`solve_divisor` and `base_seed` are held fixed for the whole run.
**`work_damage`** picks the damage law — `0` compares the strain rate with
its own percentile, `1` compares the dissipated work, stiffness times the
square of the strain rate, with the same percentile of its own field — and it
is deliberately fixed rather than sampled, so a given `search_seed` draws the
same Latin hypercube either way and two runs that differ only in it are an
ablation pair. **Its default is now 0.** Under the seam formulation the two
laws differ in kind: at 0 a seam damages by its slip rate, so a slipping
fault stays weak and heals when it stops; at 1 an open seam dissipates almost
nothing, damages almost not at all, and heals shut, which is what C04
measured on the twelve development seeds. **`solve_divisor`** is fixed for the same reason: it is the
number of kinematic cells per solve cell, `2` solving the velocity on half
the grid and lifting strain back in 2 x 2 blocks, `1` solving on the full
grid at about six times the cost. **`pixels` and `solve_divisor` are the
pair `WORK_ORDER_C04_4.md` §1 measured.** At 1024 px and `solve_divisor` 2 a
crack of four kinematic cells is two solve cells, and the stress ahead of it
is what the intact sheet carried; at `solve_divisor` 1 the tip concentration
is resolved and the same measurement rises with crack length. C03.10 made a
512-px world the same physics on a smaller parent, and C03.9 measured the
full-grid solve at about 2.5 s per world there against 22 times the cost at
1024 px, so **a search that needs the resolution runs at `pixels = 512` with
`solve_divisor = 1`**, which is what C04.4's probe did. Neither default was
moved by that order: they are still 1024 and 2, and a run sets them in its
own config.

## What the defaults sample now: the corner

Run `20260902T170740Z-s3` sampled the whole dial space under the work law:
1460 cells, no passer. Seven of those cells had a weak fraction of 0.30 or
less **and** three or more plates, and those seven share a setting the run's
medians do not:

| dial | the seven | all 1460 |
|---|---|---|
| `heal_time_myr` | 27 | 95 |
| `damage_time_myr` | 1.4 | 6.6 |
| `yield_percentile` | 8.9 | 3.4 |
| `drive_wavelength_km` | 10,240 km in five of seven | evenly spread |
| `strength_exponent` | 2 in five of seven | evenly spread |

Their `edge_fraction` reaches 0.25 against the run's in-band median of 0.20:
zones about eight cells across rather than twelve. Fast damage with fast
healing is the only setting the sheet has found that both closes a network
and slows its widening.

**The defaults are that corner**, with both times opened downward to the
engine's own floors — `heal_time_myr` to 5, `damage_time_myr` to 0.5 — because
the question the corner is sampled to answer is whether zone width keeps
narrowing as damage and healing both get faster, or bottoms out near 0.25.
`search_seed` is 11.

**`heal_time_myr` now reaches 200, not 60.** That is the one range C04.1
widened, and it is widened because the question changed with the rule: under
the sheet, healing had to be fast or a diffuse zone kept widening, and under
the seam formulation a seam cannot widen and healing decides instead how long
a fault that has stopped slipping is still there when the next crack reaches
it. Nothing else in the corner moved.

The two drive rows are in kilometres at the run's 1024 px and 5 km per
pixel, where the parent is 10,240 km: the old corner set `{1, 2}` is
10,240 km and 5,120 km, and the old whole-space set `{1, 2, 3}` reaches
3,413 km.

| dial | corner | whole space |
|---|---|---|
| `stiffness_fraction` | 0.08 – 0.5 | 0.05 – 0.6 |
| `yield_percentile` | 2 – 15 | 1 – 15 |
| `heal_time_myr` | 5 – 200 | 20 – 500 |
| `damage_time_myr` | 0.5 – 5 | 1 – 30 |
| `strength_exponent` | 2, 3 | 2, 3, 4 |
| `drive_wavelength_km` | 5120 – 10240 | 3413 – 10240 |
| `strength_spread` | 0 – 0.1 | 0 – 0.1 |
| `drive_shear` | 0 – 1 | 0 – 1 |

Type the right-hand column into the panel to search broadly again. The screen
never moves with the question: its eleven fields are the agreed definition of
a plate regime and their defaults are unchanged.

## Pairing a run against one on disk

The search is reproducible from a config, so an ablation pair is a run whose
config is another run's with one field changed. The control at
`work_damage = 0` is `out/search/20260902T154430Z-s2`; to run its treatment
half, load that run's `config.json` into the panel and set `work_damage` to
1. Then:

```powershell
py -3.14 pipeline_c/tools/pair_runs.py 20260902T154430Z-s2 <treatment_run_id>
```

**A run made before `WORK_ORDER_C04_5.md` and one made after it are an
ablation pair, not a replica**: that order changed how a crack tip picks its
direction at `seams = 2`, so the two runs sample the same dial values at the
same `search_seed` but grow different crack paths from them, and every
difference between them is the tip rule. `20260903T015247Z-s11` is the
pre-order half at 40 cells and `20260903T190345Z-s11` is the post-order half
at the same config.

**A run made before `WORK_ORDER_C04_6.md` and one made after it are an
ablation pair too, and two of the numbers change their meaning across it**:
that order makes a `seams = 2` seam a curve carried on linked markers, so from
it on the marker count is the number of *vertices* of that curve and not the
number of seam cells — a chain of `k` markers can draw more cells than `k`, and
the motion adds vertices by splitting any segment it stretches past one and a
half cells — while `gaps_closed`, which counted the holes a set of points left
behind, has nothing left to count and reads at or near zero by construction.
No run written before that order is on disk at `seams = 2` with a comparable
marker count.

The treatment run id is the one the gallery shows, and it is also the newest
directory under `out/search/`. The tool pairs the two runs' stage-1 cells by
dial values and writes `out/pair_<control>_<treatment>.md` — and the same
page to stdout — with paired differences per metric, the rank correlation of
`edge_fraction` with `weak_final` in each run, band statistics, time to half
the final weak fraction, and throughput. Pairing needs the two runs to have
drawn the same cells, so it applies to runs that share a space and a seed.

### Stages

`stage1_cells`, `stage1_seeds`, `stage2_top`, `stage2_perturbations`,
`stage2_seeds`, `stage3_top`, plus `search_seed` and `window` (cells in flight
at once).

## The gallery

One card per cell, newest first, with passers pinned to the top and outlined
and invalid cells greyed. Each card carries the cell's `plates` sheet, its
`trajectory` sheet below it, the six terms as small chips (green passed, red
failed), and the dial values. Clicking a card opens every sheet that cell has
at full size, a per-world table of the seven metrics, and the dial values as
copyable text, so a setting can be typed straight into the exploration lab on
port 5003.

Nothing is drawn on any sheet. The chips and the outlines live in the page.

## What a finding is worth

**A passing cell at twelve seeds is a candidate for the author's eyes. It is
not an approval and it grants nothing.** The screen is a physical filter: it
says a regime localized, settled, and left a thin connected network with a few
plates. It says nothing at all about whether the result is worth keeping.
That judgement is the author's, at the exploration lab, in front of the
sheets.

A setting the author does keep still goes through the blind layer audit before
anything is frozen into `engine/history/constants.py`. The search's dials, like
the lab's, are development instruments and never appear in the production
adapter.

## Where runs are stored

`pipeline_c/out/search/<run_id>/`, one directory per run:

- `config.json` — the screen, the space, the stage knobs and the seeds, so a
  run can be repeated exactly;
- `cells.jsonl` — one line per cell in completion order: stage, dials, seeds,
  per-world metrics, pass and invalid state, soft score, seconds;
- `cells/<cell_id>/*.png` — the `plates`, `stress` and `trajectory` sheets for
  every cell, and every view sheet for stage-3 cells and findings. `stress`
  joined the per-cell set with the seam formulation, because under it the
  stress field is what decides where a crack starts and which way a tip runs.

The gallery reads from disk, so a run can be reopened after the server has
restarted: pick it from the dropdown beside the buttons and press **Reopen
run**. A reopened run shows at most 1,000 cells: every passed cell, and the
newest of the rest to fill the room. The full record stays in `cells.jsonl`.
