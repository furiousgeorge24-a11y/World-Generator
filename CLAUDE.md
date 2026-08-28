# mapgen — testbed for world-map terrain generation

Standalone testbed. Generates terrain-form worlds (elevation, water, climate
fields) from a seed + author controls, presented as top-down PNG renders.
Downstream, fields export to a hex-map editor. Read these before working:

- `docs/contract.md` — the promises. What must always be true.
- `docs/design.md` — the architecture and accumulated design decisions.
- `docs/milestones.md` — sequencing, scope, exit criteria, status.
- `docs/value_ledger.md` — per-feature cost/yield tracking (see rule below).

## The one hard rule — quarantine

Prior map-generation pipelines exist in a parent project. They are
**quarantined**: never ask to see them, never mine them for design, never
treat any fragment that surfaces as a design source. When the author shares
an image from a prior attempt, it is a *failure spec* — analyze what went
wrong, design fresh. Public/textbook knowledge is fair game; this project's
ancestors are not. (`out/` at repo root contains prior-attempt renders —
leave unexamined.)

## Standing working agreements

- **The author drives design.** Bug reports and review findings get
  investigation and *named* design candidates — never reactive patches.
  Implement only what is explicitly authorized, by name.
- **Aesthetic decisions become controls** — registry entries with a stated
  range and a stated promise, never hard-coded. Tiered primary/advanced.
  Elicit the author's aesthetic standards at image reviews; don't assume.
- **User-facing generation never fails.** Findings ship as a report beside
  the delivered map; they never destroy a run.
- **Seeded determinism.** Same seed + settings + version → identical map.
- **Review is by image batch** — galleries across seeds and sizes, never a
  single hand-picked example.
- **Value ledger** (see `docs/value_ledger.md`): the project is deliberately
  over-engineered for now; trimming comes later and must be evidence-based.
  Every feature gets a ledger row with *predicted* yield written at
  implementation time. Features predicted marginal get same-seed on/off
  ablation pairs in the next gallery (every feature has a control, so
  ablation = knob at zero). Observed yield is recorded at review. The ledger
  nominates trims; only the author decides them. A milestone is not done
  until its ledger rows are current.
- **Milestones end with commit recommendations**; the author handles git.
  Confirm scope before any large deletion or rework.

## Tech constraints

- Python 3.12 + numpy + Pillow. **scipy only with author sign-off**, backed
  by benchmark numbers of the numpy-only alternative.
- Eventual integration target: the parent editor is Python (Flask) +
  vanilla JS. This testbed's future webui matches that stack.

## Code conventions

- Physical units everywhere: elevation in metres (sea level = 0), horizontal
  scale via `cell_size_km`.
- Per-stage RNG keying — `hash(seed, stage_name)` — never one running
  stream. Dragging one control must not reshuffle unrelated stages.
- Determinism hygiene: no wall clock, no set/dict iteration order feeding
  results, no unordered parallel reductions.
- Sample noise and place features in **world-space km**, not cell space —
  structural resolution independence depends on it.
- `generate(...)` / `render(...)` / `hexify(...)` stay strictly separated;
  rendering a World must remain cheap.
- Every PNG carries provenance metadata (seed, controls, version); every run
  writes a report sidecar. Render output goes to `out/` (gitignored).
