# Work order — C04.4: why a persisting crack does not run

Issued 2026-09-03 for Opus 5. Follows [`WORK_ORDER_C04_3.md`](WORK_ORDER_C04_3.md)
and its report at `out/C04_3_BUILD_REPORT.md`. Same rules as C04.3.

## 0. Purpose

C04.3 made cracks persist: 98 % of marker-steps above the yield slip
rate, persistence 0.78 to 0.94. They do not run. Each seed ends with 81 to
108 separate cracks of median length 4 cells, tips rose to 1,140 to 2,330
while advances fell to 438 to 635, and the report measured the block: at
a tip, the stress ahead reaches `sigma_c / sqrt(L)` for only 9 of 40 tips,
the median ratio is 0.61, and **the ratio does not move as the seam count
grows five-fold**. A crack of two cells sees the same stress ahead of it
as the intact sheet did.

Two candidate causes, one numerical and one a constant of mine.

- **The tip concentration is not resolved.** In an elastic plate the
  stress ahead of a crack grows with the crack's length, which is why the
  Griffith rule is written as it is. In the solve it can only appear if
  the crack is several solve cells long. At `solve_divisor = 2` a crack of
  four kinematic cells is two solve cells, and a barrier two cells long
  concentrates almost nothing at its ends. C03.9 found that the full-grid
  solve costs 22 times at 1024 px but 2.5 s per world at 512 px, and
  C03.10 made a 512-px world the same physics on a smaller parent.
- **The toughness is an implicit constant.** C04 §2.4 wrote the threshold
  as `sigma_c / sqrt(L)` with the unit length one cell, which fixes the
  fracture toughness at the intact strength times the square root of one
  cell. Intact strength and toughness are different material properties;
  a material's propagation stress for a crack of a few cells is normally
  well below its nucleation stress. That constant was never a decision.
  It should be a parameter with a physical name.

**Two things, separated.** §1 is a measurement with no new physics; §2 is
one new dial. Their runs are reported side by side so the author can see
which one moves the block.

**Predictions on record.**

- P1. At 512 px with `solve_divisor = 1`, the median tip ratio rises above
  1.0 at crack lengths of eight cells or more, and at least six of the
  twelve seeds end with two or more plates. (If it does not rise, the tip
  concentration is not what is missing.)
- P2. At 1024 px with `solve_divisor = 2` and `toughness_fraction = 0.5`,
  cracks run: longest crack above 200 cells on at least eight seeds, and
  at least six seeds end with two or more plates.
- P3. In the 512-px probe with the dial sampled, at least 10 % of worlds
  pass `plate_count`.
- P4. Cost at 512 px, divisor 1, `seams = 2` is within 1.5 times the
  1024-px, divisor 2 sheet.

Failure modes: the seven of C04.3 (its seventh, **persists but does not
run**: `network_share < 0.3` with persistence above 0.5 and longest crack
under 150), and **runaway** as C04.3 defines it.

## 1. The measurement: resolution

No engine change. Rerun the twelve seeds of C04.3 §3.1 at
`pixels = 512`, `scale_km = 5`, `solve_divisor = 1`, `seams = 2`, the
same dials, through the pool. Also run two seeds (4287772760,
2075014389) at 1024 px with `solve_divisor = 1`, which is about 100 s
each. Per world, the C04.3 table's columns plus the **tip ratio
diagnostic** the C04.3 report computed (share of tips with a qualifying
neighbour, median ratio of the best neighbour's traction to the
threshold), binned by crack length: 1–2, 3–4, 5–8, 9–16, 17+ cells. The
same diagnostic on the C04.3 worlds at divisor 2 is the comparison row;
recompute it from their final state in the same session rather than
copying it.

## 2. The dial: toughness

`HistoryParams.toughness_fraction: float = 1.0`, range `[0.05, 1.0]`,
recorded. The tip rule's threshold becomes
`toughness_fraction * sigma_c_field[candidate] / sqrt(L)`. Nucleation is
unchanged and still needs the full intact strength. At 1.0 every output
is byte-identical to C04.3; the `seams = 0` gate and the `seams = 1` pins
are untouched. Lab dial: float, default 1.0, lo 0.05, hi 1.0, primary,
promise "Fracture toughness as a fraction of intact strength. Cracks
propagate at this fraction of the stress it takes to nucleate one, for a
crack one cell long; longer cracks propagate at less." Search `Space`:
sampled log-uniform 0.1 – 1.0; `DIALS`, `params_of`, server knob,
`modernize_dials` filling 1.0 on legacy cells.

Rerun the twelve seeds at 1024 px, `solve_divisor = 2`, C04.3 dials with
`toughness_fraction = 0.5`, the same table.

## 3. The probe

40 cells, 4 seeds, stage 1 only, through the library, to `out/search/`,
**at `pixels = 512`, `solve_divisor = 1`** with the toughness dial
sampled and everything else as the search's defaults. Term pass rates,
passers, the eight failure-mode counts. A 512-px world is a smaller world
on the same physics since C03.10, and this is what the author's search
will run if the resolution is what moves the block; say so in
`SEARCH.md` beside the fixed `pixels` and `solve_divisor` knobs, without
changing their defaults.

## 4. Tests, documents, report

`tests/test_seams.py`: `toughness_fraction = 1.0` reproduces a pinned
C04.3 128-px line; at 0.5 a hand-built tip advances where at 1.0 it did
not; 0 and 1.5 refused. The suite must pass. `EXPLORE.md`, `SEARCH.md`,
`STATUS.md` "Now" one sentence each. `out/C04_4_BUILD_REPORT.md`: what
was built; deviations; check output; the three twelve-seed tables (512 px
divisor 1; 1024 px toughness 0.5; the two 1024-px divisor-1 seeds) with
the tip-ratio diagnostic binned by length beside the C04.3 comparison
row; cost; the probe; predictions with numbers; the named mode;
observations with evidence and no proposed values. If either change makes
cracks run, say which, at what step loops first cut a piece of 655 cells
or more, and what the pieces do afterwards.
