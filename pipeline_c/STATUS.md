# Status

Where Pipeline C stands and what has already been ruled out. This is the only
file that tracks state; update it when something changes.

Last updated: 2026-09-01

## Now

The current mechanism is **C02**, connected competitive growth. Nothing
downstream of the tectonic fabric has been built: there is no land, water,
coastline, elevation, or map anywhere in this module yet.

The WebUI generates the tectonic fabric for any seed in about three seconds.
Neither author control is implemented, so neither is advertised.

## The open question

Not a formal gate — just the thing worth deciding next:

**Is C02's macro organization good enough to build C6 on, or does it need a
third attempt?** Generate a spread of seeds and look. The specific worry is
below under C02: long planar contacts and axis-aligned banding. If it needs
revising, the mechanism to change is the directional step cost.

Undecided as of 2026-09-01, pending investigation. The blind layer audit is
the instrument for the mechanism half of that question — whether these fields
are the footprint of a process or the output of a formula. The macro-scale
half stays with the author.

## What has been tried

This is the part worth carrying forward. All three attempts passed every
automated gate they declared; two were rejected on sight anyway. Numeric
diversity gates have so far failed to detect a repeated visual grammar.

### C00 — parent-world foundation (accepted, retained)

A `40,960 × 40,960 km` parent on a flat torus in both axes, addressed in
integer metres, sampled at 512 and 1024. Randomness is a stateless SHA-256
sampler keyed by world, stage, stage version, process, channel, and physical
address, so traversal order, resolution, chunking, cache warmth, and observer
window cannot reroll anything. Twelve development identities, four debug, 32
sealed validation.

This layer works and is still in use. It contains no geology.

### C01 — point-power affiliation (rejected)

Seven actors placed by global maximin, ownership assigned by an additive
weighted power partition over the torus metric.

**Failed because** every one of the twelve worlds came out equant, evenly
packed, and honeycomb-like. Unique hashes and 24–64% pairwise pixel
disagreement did not detect the repetition — the worlds differed in detail
while sharing one grammar.

**Do not** return to maximin point placement or point-distance power
partitions for macro affiliation.

### C02 — connected competitive growth (current)

Seven actors grown from 33-cell germ lines by claim-once competitive blocked
growth on a fixed 1024² torus lattice, with four seeded layout families
(`scatter`, `belt`, `dual_focus`, `arc_void`), two low-frequency triangle-wave
resistance modes, and directional step costs.

Better hierarchy and family variation than C01 — unequal actors (4.3%–26.4%),
belts, lobes, junctions, broad interiors. But contacts are largely long
planar runs, several worlds read as horizontal or vertical bands, and D09 has
a narrow vertical tether between two banded halves.

**Suspected cause** if C03 is needed: the step cost is `192` parallel to the
actor axis versus `896` perpendicular, on a four-neighbour cardinal lattice.
That ratio makes axis-aligned growth strongly preferred, so the mechanism
tends to manufacture exactly the banding the gate asks the author to screen
for. Curvature and local raggedness are deferred to C6, but C6 cannot be asked
to rescue an unacceptable macro topology.

## Where the old evidence is

The three sealed runs are still under `.review_store/` as content-addressed
snapshots, but nothing reads them any more. Every world in them is
reproducible from its seed, so they are a historical curiosity rather than a
dependency. The twelve C02 development seeds are:

```
2075014389, 2477733044, 476149591, 151640007, 2697441485, 1504571935,
548870008, 2157195430, 4108373596, 4287772760, 287488203, 1833546021
```

D09 — the banded world with the narrow vertical tether — is seed `4287772760`.

## Verification

Last run 2026-09-01 on this working tree:

| Suite | Result |
|---|---|
| `py -3.14 -B -m unittest discover -s pipeline_c/tests -p "test_*.py"` | 35 passed (engine) |
| `py -3.14 pipeline_c/tests/adapter_checks.py` | 9 passed (adapter + quarantine) |
| `py -3.14 pipeline_c/tests/eval_checks.py` | 70 passed (eval scaffold) |
| `py -3.14 pipeline_c/tests/layer_audit_checks.py` | 59 passed (layer audit) |

## Leftovers

`.review_store/` (156 MB of sealed snapshots from the retired review pipeline)
is orphaned. Nothing reads it, and every world in it is reproducible from its
seed. It is gitignored, so it costs only disk.

`eval/` holds the frozen judging prompts, the verdict schemas, and the
mechanical scoring. Its three land instruments are still protocol only — they
need generator output that does not exist yet.

The fourth, `layer_audit_v1`, is complete and runnable via
[`run_layer_audit.py`](run_layer_audit.py): it builds a blind batch of native
window panels seeded with hidden calibration controls, and voids itself if the
judge misses a planted lattice. See [`VIEWS.md`](VIEWS.md). It has no provider
client — the judge is a fresh-context subagent, which means it shares a model
family with whoever wrote the code under review. The controls bound that
weakness; they do not remove it.
