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
at the result — about 3 seconds per world. Three views, each the raw raster
with nothing drawn over it:

| View | Shows |
|---|---|
| `affiliation` | Which primary actor owns each cell — the world itself |
| `arrival` | When each cell was claimed during growth; banding here means the growth is axis-locked |
| `resistance` | The low-frequency field growth had to push through |

Any seed works, not just a frozen cohort, and the same seed always gives the
same world. There is no review, baseline, delta, or approval workflow: look at
what you get, and discuss it.

What is generated is the **tectonic fabric** stage only — a categorical actor
field over the parent world. No elevation, water, coastline, island, or land
exists yet, so this is not a map.

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
py -3.14 pipeline_c/tests/adapter_checks.py
py -3.14 pipeline_c/tests/eval_checks.py
py -3.14 pipeline_c/tests/layer_audit_checks.py
```

## Documents

| File | Contents |
|---|---|
| [`CONTRACT.md`](CONTRACT.md) | Normative. What a conforming result must satisfy, and which causal shortcuts are banned. |
| [`STATUS.md`](STATUS.md) | Where the work is, what is waiting on the author, and what each attempt got wrong. |
| [`ROADMAP.md`](ROADMAP.md) | What is left to build, in order; open design questions; working rules. |
| [`VIEWS.md`](VIEWS.md) | Every layer gets a view; how to draw one; the blind single-image audit that screens them. |
| [`AUTHOR_RULINGS.md`](AUTHOR_RULINGS.md) | The look being built toward, accepted defaults, review authority, isolation. |
| [`eval/README.md`](eval/README.md) | Judging prompts, verdict schemas, and scoring for critiquing output later. |

On conflict, the contract wins.

## Isolation

Pipeline C uses only its own code and evidence plus the shared root WebUI
shell. It may not open, import from, or consult `pipeline_a` or `pipeline_b`.
