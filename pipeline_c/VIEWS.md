# Views and layer review

Every layer, mask, and process that participates in forming a map gets a view.
An agentic judge looks at it first; what survives goes to the author's eyes.

This is the standing method for the rest of the pipeline's development, not a
one-off for the current stage.

## Why

Two mechanisms have now been built, passed every numeric threshold declared
for them, and been rejected on sight — C01 for honeycomb cells, C02 for
axis-aligned banding. Aggregate statistics did not catch either, because the
defect was a *structural signature*, not a distributional one. Spacing
coefficients of variation and pairwise disagreement cannot see a plaid.

The C02 resistance field is the sharpest example. It is
`T(k₁x + k₂y + φ₁) + T(k₃x + k₄y + φ₂)` — the sum of two triangle waves along
integer lattice directions. It cannot be anything but a rigid diagonal
lattice, and it is the field the growth had to push through, so it shaped
every world in the cohort. Nobody saw it for the life of the attempt, because
there was no view for it. The moment it got one, the problem was obvious in a
second.

That is the entire argument for this document: **an unviewed field is an
unexamined assumption.**

## What gets a view

Anything with spatial extent that influences the result. Not just the outputs
the stage is proud of:

- the stage's own output field;
- every intermediate or scratch field it consumes or produces, including ones
  considered "just noise" or "just a helper";
- any mask, weight, cost, resistance, or bias field;
- any field inherited from an upstream stage and modified;
- the boundaries or contours implied by a categorical or scalar field, where
  the adjacency is what carries the structure.

A field exempt from a view is a field nobody has to defend. There should be
very few, and the reason belongs in the stage's notes.

## How to represent one

The goal of a view is to make unnatural geometry *visible*, not to make the
output look finished. Bias every choice toward exposure.

- **Base image only.** No headers, sidebars, captions, legends, markers, or
  analysis rectangles drawn over the data. Chrome hides exactly the kind of
  small regularity worth catching. Metadata travels in PNG text fields.
- **Native resolution, nearest-neighbour.** Resampling and smoothing are
  themselves geometric operations; they can both hide a defect and invent one.
- **Categorical fields** (actor, plate, material class): distinct hues, flat
  fill. Give the *boundary set* its own view — straightness, junction angles,
  and repeated contact motifs live there, not in the fill.
- **Scalar fields** (elevation, cost, arrival, resistance, distance): a
  monotone ramp for magnitude, plus a **quantized or contoured** companion
  view. Continuous ramps flatter smooth data; iso-bands make lattice
  alignment, constant curvature, and repeated wavelengths jump out. The
  attached arrival field reads as soft diamonds on a ramp and as unmistakable
  concentric rhombi once banded.
- **Boolean masks**: the mask, and separately its outline.
- **Direction or vector fields**: hue for direction, value for magnitude. Not
  arrow glyphs — glyph spacing imposes its own lattice.
- **Anything periodic by construction** should be viewed over at least two
  full periods, so the repeat is visible rather than cropped out of frame.

When a field has no honest visual representation, say so rather than
inventing a flattering one.

## What the review is looking for

The question is always the same: *does this look like the footprint of a
process, or like the output of a formula?* Concretely, the signatures that
have already bitten this project or are likely to:

- **grid locking** — features preferring 0°, 45°, or 90°, or growth fronts
  shaped like the neighbourhood stencil;
- **periodicity and tiling** — a motif that repeats at a fixed interval;
- **constant scale** — one characteristic feature size everywhere, no
  hierarchy;
- **constant curvature** — repeated radii, arcs that look struck with a
  compass;
- **mirror or rotational symmetry** that no process would produce;
- **straight runs** — long unbroken linear boundaries;
- **degenerate morphology** — lace, webs, ribbons, or isolated speckle where
  broad coherent bodies belong;
- **seams** — discontinuities at domain edges, chunk boundaries, or wrap
  points.

Regularity is a reason to investigate, never proof of a defect on its own. A
real process can produce a locally straight coast or a circular feature. The
judgement is whether the regularity is *systematic* across the field and
across seeds.

## Order of defense

1. **Agentic review.** The judge sees the view panels and returns an
   evidence-anchored critique. A construction-artifact finding stops the layer
   here and it gets fixed before anyone's time is spent on it.
2. **Author's eyes.** What survives goes up for the judgement that actually
   decides: does this look natural, and is it worth building on.

The judge is a filter, not an authority. It never approves anything — it only
catches the obvious before the author has to. A clean automated pass is not
evidence that a layer is good, and the author overrides it in both directions.

## How this uses `eval/`

The scaffold in [`eval/`](eval/README.md) already has the right shape for
this, and the critique rubric is very nearly the right rubric already.

`land_origin_critique_v1` asks for three buckets per panel — `done_poorly`
(≤5), `done_well` (≤5), `cannot_identify` (≤3) — where every claim must name
`what`, a specific `where`, and visible `evidence`, and praise carries the
same evidence burden as criticism. Its severity ladder is:

- **A** — suspected construction artifact or unsupported regularity: repeated
  same-scale bodies, tiling, seams, repeated radii, systematic grid locking;
- **B** — formation implausibility;
- **C** — character or quality weakness;
- **D** — presentation obstructs interpretation.

Severity **A** is precisely the layer-review question, and **D** is unusually
valuable here: it lets the judge report that *the view design* is failing
rather than the layer, which is the difference between fixing a renderer and
rewriting a mechanism.

The existing machinery that carries over unchanged:

| Piece | Role |
|---|---|
| `eval/verdicts.py` | Strict validation of returned JSON against the versioned schema |
| `eval/keys.py` | Hidden panel identity and declared duplicate groups |
| `eval/score.py` | Mechanical tallies plus duplicate-reliability probes |
| `eval/bundle.py` | Append-only bundle → submissions → results stages |
| `eval/schemas/` | The frozen response schemas |

### What was added for it

| Piece | Where |
|---|---|
| The audit prompt | `eval/prompts/layer_audit_v1.md` |
| Verdict and hidden-key schemas | `eval/schemas/layer_audit_v1.schema.json`, `layer_panel_key_v1.schema.json` |
| Strict validation and dispatch | `eval/verdicts.py`, `eval/keys.py` |
| Synthetic calibration controls | `eval/controls.py` |
| Panel assembly, hidden key, judging plan | `eval/stimulus.py` |
| Calibration and consistency scoring | `eval/audit.py` |
| The one fixed palette | `eval/palette.py` |
| The gate itself | [`run_layer_audit.py`](run_layer_audit.py) |

`land_origin_critique_v1` was left alone rather than adapted. It is written for
land panels — it opens "each panel is a neutral view intended to expose the
origin and organization of land" — and a resistance field is not land; asking a
judge to critique it as land invites invented findings. The frozen prompts may
not be silently edited, so the audit took its own ID.

**2AFC does not apply to intermediate layers.** It is a reference-versus-
candidate discrimination, and there is no reference image for a resistance
field. `land_controls_sweep_v1` becomes applicable once author controls exist
and a layer can be compared across settings.

## The single-image audit

The hard version of the question: given one view, with no reference to compare
against, decide whether it is the footprint of a process or the output of a
formula.

Asking a judge that directly does not work. "Does this look natural?" invites
rationalizing almost anything; "find the mathematical artifacts" finds them
everywhere. Both failure modes are worse than useless, because they produce
confident answers either way. The design below replaces the aesthetic question
with tasks that have checkable answers.

### 1. Ask for the generating rule, not a verdict

The discriminating prompt is **"state the rule that would reproduce this
image."**

A triangle lattice invites a short, closable answer: *two superimposed linear
gradients folded into a sawtooth, period about N pixels, oriented near 45°.*
A process-shaped field does not close — the honest description keeps needing
exceptions, local qualifications, and multiple scales.

The **closability of the description is the signal.** This converts an
unreliable aesthetic judgement into a recall-and-describe task, which vision
models are markedly better at, and it produces an artifact a human can check.

### 2. Make the judge commit to a falsifiable prediction

Follow with: *if this field continued past the frame, what would be there?
State the repeat period and orientation if one exists.*

Then render the actual adjacent region and check the prediction. A formula is
predictable by construction — a correct off-frame prediction is strong
evidence of formulaic structure, and a confidently wrong one is evidence
against. This is self-verifying without any measurement instrument, and it is
the single most valuable signal in the set.

### 3. Classify the mechanism — this task is supervised

For an intermediate view **the true mechanism is always known**, because the
code that produced it was just written. So ask the judge to name it from a
fixed list rather than to rate it:

- superimposed periodic waves;
- distance, cost, or arrival field from point or line sources;
- filtered or fractal noise;
- iterative growth, accretion, or erosion;
- thresholded or quantized version of another field;
- cannot determine.

Because the answer is known, the judge can be **scored**, not merely trusted.
A judge that cannot name a triangle lattice as periodic waves has no standing
to clear anything else.

### 4. Seed calibration panels into every batch

Each audit batch mixes the new views with hidden controls: known-formulaic
panels (a triangle lattice, a radial cost field) and known-acceptable ones.
Filtered noise is a legitimate positive control here — a field that reads as
noise passes, by the standard this project actually cares about.

If the judge fails to flag the known-formulaic controls, **the batch is void**
and its verdict on the new views is discarded.

This is not comparison in the sense the audit forbids. No panel is judged
against a reference of what it should look like; each is still judged on its
own. The controls measure whether the judge is working at all.

`panel_key_v1` is frozen and its `hidden_kind` admits only
`candidate`/`reference`, so the audit got a sibling, `layer_panel_key_v1`. It
adds the two control kinds, the `true_mechanism` that makes the batch
supervised, and the `crop_factor` and `window` that make an off-frame
prediction checkable after the fact. A control with no declared mechanism is
rejected, as is a batch missing either kind of control: a key that cannot
calibrate its own batch will not validate.

### 5. Repeat and vary the framing

- **Multi-scale crops.** The same field at several zooms and offsets. A
  formula looks the same at every scale; a process has different structure at
  different scales. Ask which crops are zoomed in — inability to tell is
  itself a finding.
- **Duplicate panels, fresh context.** The same panel more than once under
  different numbering. Agreement is the confidence measure, and
  `eval/score.py` already scores duplicate reliability.
- **Fixed palette.** Colormap changes apparent structure, so the audit fixes
  one ramp and one categorical palette.

### When it runs

This is a **work-approval gate on newly implemented code**, not a per-run
filter.

```powershell
py -3.14 pipeline_c/run_layer_audit.py build --seed 4287772760
py -3.14 pipeline_c/run_layer_audit.py score  --run <id> --verdict verdict.json
py -3.14 pipeline_c/run_layer_audit.py verify --run <id> --verdict verdict.json
```

`build` renders every declared view, mixes in the hidden controls, cuts native
windows at two magnifications, repeats two panels byte-for-byte, shuffles, and
publishes an immutable batch. The judge packet is a directory of numbered PNGs
and the prompt; the key, the provenance, and the judging plan sit under a
disjoint hidden root. Whoever runs the judge must not read the key first —
that is a protocol obligation, not an enforced sandbox.

`build` also prints a **judging plan**: which panels go to which fresh-context
call. Duplicate-group members are deliberately split across calls, so their
agreement measures independent re-judgement rather than one reader's internal
consistency, and every call carries its own formulaic and process control.
Concatenate the returned arrays before scoring.

`score` voids the batch if a planted formula went uncaught or a known process
panel was condemned, reports mechanism accuracy where the truth is declared,
and lists every candidate the judge called formulaic. It exits non-zero on a
void batch. `verify` renders what is actually past the right edge of each
panel the judge claimed to predict.

Then: fix anything that draws a construction-artifact finding before the author
spends time on it, and send what survives up with the judge's own transcript.
Two fix-and-rerun cycles is the budget; after that it goes to the author
regardless. The author sees the raw verdicts either way, and a clean audit is
not an approval and never appears as one.

### What this will not do

- It catches blatant construction, not subtle bias. A field can pass every
  question here and still be quietly wrong.
- It will sometimes flag genuine regularity. A real process can produce a
  straight run or a circular feature; the follow-up is a causal question, not
  another look.
- Period and orientation estimates from a vision model are approximate. They
  are useful as claims to verify, not as measurements.
- A judge fluent in the domain may infer the mechanism from context rather
  than from the image. Panels carry no stage names, parameters, or filenames.
- **It judges mechanism from native windows, not world-scale composition.** A
  512-pixel window of a 1024-pixel field cannot speak to actor hierarchy or
  macro organization. That stays with the author, who has the WebUI for it.
- **The judge is the same model family as the implementer**, so its blind
  spots may be the implementer's. The controls bound this — an instrument that
  cannot see a planted lattice is reported as broken — but they do not remove
  it. An independent provider would.
- **Panel file sizes vary with content**, so a judge with shell access could in
  principle infer something from byte counts rather than from the image. Give
  the judge the images, not the directory listing.

## Scope note

None of this asserts that a layer passing review is natural, or that the
current C02 fields pass. The resistance and arrival views are known-bad
examples that motivated the policy; they have not been fixed, and fixing them
is a separate decision about the growth mechanism.
