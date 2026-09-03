# Pipeline C — land-origin laboratory

One problem: generate plausibly natural land origins while giving an author
reliable control over how much land is delivered and how strongly it
fragments.

This is the tectonic, crustal, vertical, and exposure machinery needed to
*originate* land — not a terrain pipeline. Mountains, bathymetry, erosion,
rivers, lakes, climate, biomes, and cartography are all downstream and out of
scope. Success is a small documented interface that could later be ported.

**Current state, and what has already been ruled out: [`STATUS.md`](STATUS.md).**

## The two controls to be solved

- `target_land_percent` — `0`–`70`, default `50`. Every delivered map within
  10 percentage points of the request.
- `landmass_fragmentation` — `0`–`1`, default `0.5`. At `0`, a strong tendency
  toward one dominant macro-landmass; higher values move the same land budget
  into more separated bodies. Never a promise about island count.

Neither is implemented yet, so neither is advertised by the WebUI; they are
added back when C11 and C13 give them a causal stage. Neither may ever get its
result by rerolling seeds, moving the crop, editing a finished mask, or letting
the frame shape terrain.

## Running it

From the repository root:

```powershell
pipeline_c\run.bat
```

That starts the WebUI on port `5002`. Enter a seed, press generate, and look
at the result — about six and a half seconds per world at the default
1024 px, and under two seconds at 512 px and below, which share one
floor-sized world.
Every view is the raw raster at native history resolution with nothing drawn
over it, so a 1024 px world shows 256 × 256 views:

| View | Shows |
|---|---|
| `plates` | Emergent plate labels at the end of the history — connected regions of strong lithosphere |
| `boundaries` | The weak cells and the strong cells that touch a different plate |
| `regime` | Divergent, convergent, or shear on each weak cell, from the local strain |
| `strength` | Lithosphere integrity at the end of the history |
| `strength_banded` | The same field in eight bands, so gradients read as contours |
| `velocity` | Solved lithosphere velocity: hue is direction, brightness is speed |
| `strain_rate` | Second invariant of the strain-rate tensor |
| `strain_rate_banded` | The same field in eight bands |
| `drive` | The mantle's basal traction at the end of the history |
| `drive_phi` | The curl-free potential the drive's gradient part comes from |
| `drive_psi` | The rotational potential the drive's perpendicular part comes from |
| `strength_initial` | The strength field before any history ran |
| `boundaries_t25`, `boundaries_t50`, `boundaries_t75` | Weak lithosphere at a quarter, a half, and three quarters of the history; plate contacts are labelled at the final epoch only |
| `strength_t25`, `strength_t50`, `strength_t75` | Strength at those epochs |
| `plates_tiled` | `plates` repeated 2 × 2, so the wrap point sits mid-image |

There is a second, separate WebUI for development: `pipeline_c\explore.bat`
starts the **exploration lab** on port `5003`, which exposes the history's
settings as dials and shows eight seeds side by side per setting. Its dials
are development instruments, never author controls, and the production lab
above is unchanged by it. See [`EXPLORE.md`](EXPLORE.md).

A third one, `pipeline_c\search.bat`, starts the **regime search** on port
`5004`: it turns those dials by itself, screens each setting on six measured
properties of a plate regime, and shows the result in a live gallery. What it
produces is candidates for the author's eyes and not approvals. See
[`SEARCH.md`](SEARCH.md).

Any seed works, not just a frozen cohort, and the same seed always gives the
same world. Scale, in kilometres per delivered pixel, is the one control; it
sizes the world rather than shaping it. There is no review, baseline, delta, or
approval workflow: look at what you get, and discuss it.

What is generated is the **kinematic history** stage only — a drive field, a
strength field, a velocity, and the plates and boundaries that emerge from
them. No crust, elevation, water, coastline, island, or land exists yet, so
this is not a map.

## Auditing a layer

Before a newly implemented stage reaches the author, its views go through a
blind agentic audit that asks what kind of rule produced them. It mixes hidden
calibration panels into the batch, so a judge that misses a planted lattice
voids its own verdict. Full design in [`VIEWS.md`](VIEWS.md).

```powershell
py -3.14 pipeline_c/run_layer_audit.py build --seed 4287772760
py -3.14 pipeline_c/run_layer_audit.py score --run <id> --verdict verdict.json
```

This is a work-approval gate on new code, not a per-generation filter, and a
clean audit is never an approval — it means nothing was caught.

Checks:

```powershell
py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"
py -3.14 pipeline_c/tests/eval_checks.py
py -3.14 pipeline_c/tests/layer_audit_checks.py
py -3.14 pipeline_c/run_layer_audit.py build --seed 4287772760 --pixels 512
py -3.14 pipeline_c/tools/contact_sheet.py --view plates --pixels 512 --out pipeline_c/out/plates_512.png
py -3.14 pipeline_c/tools/contact_sheet.py --view boundaries --pixels 512 --out pipeline_c/out/boundaries_512.png
```

## Documents

| File | Contents |
|---|---|
| [`CONTRACT.md`](CONTRACT.md) | Normative. What a conforming result must satisfy, and which causal shortcuts are banned. |
| [`STATUS.md`](STATUS.md) | Where the work is, what is waiting on the author, and what each attempt got wrong. |
| [`ROADMAP.md`](ROADMAP.md) | What is left to build, in order; open design questions; working rules. |
| [`VIEWS.md`](VIEWS.md) | Every layer gets a view; how to draw one; the blind single-image audit that screens them. |
| [`EXPLORE.md`](EXPLORE.md) | The development exploration lab on port 5003: what each dial means and what a finding there is worth. |
| [`SEARCH.md`](SEARCH.md) | The regime search on port 5004: the stages, the screen's terms, the gallery, and where runs are stored. |
| [`AUTHOR_RULINGS.md`](AUTHOR_RULINGS.md) | The look being built toward, accepted defaults, review authority, isolation. |
| [`eval/README.md`](eval/README.md) | Judging prompts, verdict schemas, and scoring for critiquing output later. |

On conflict, the contract wins.

## Isolation

Pipeline C uses only its own code and evidence plus the shared root WebUI
shell. It may not open, import from, or consult `pipeline_a` or `pipeline_b`.
