# The exploration lab

A second WebUI, on its own port and its own tab, with the kinematic history's
settings exposed as dials and eight seeds shown side by side per setting. The
production lab on port 5002 is untouched and still has one control.

```powershell
pipeline_c\explore.bat
```

That starts it on port `5003`. Enter a seed, turn a dial, generate: eight
worlds run at once, about twelve seconds at 1024 px. Every view is a contact
sheet of the eight, panels four across at native history resolution on a black
gutter, with nothing drawn on them.

## The dials

In the order they appear.

| Dial | What it means |
|---|---|
| `scale_km` | Kilometres per delivered pixel. World geometry, not a formation control: it sizes the simulated planet. |
| `seeds_per_view` | Worlds per generation, seeds `seed` to `seed + n - 1`, shown side by side. |
| `stiffness_fraction` | The fraction of the world over which a plate holds together. Below a plate's size the plate deforms internally instead of moving as one. |
| `yield_percentile` | The percent of the initial strain field that sits above yield: what breaks first. |
| `heal_time_myr` | How long a fault takes to seal once it stops moving. |
| `damage_time_myr` | How long intact rock at twice yield takes to fail. Its floor is 0.5 Myr, the search's own lower bound, so a cell the search drew can be typed in as it stands. |
| `work_damage` | Which damage law runs: 0 the excess of the strain rate over its yield, 1 the excess of the dissipated work, stress times strain rate, over the same percentile of its own field. Under seams, 0 is the slip-rate law that keeps a slipping fault weak; 1 is the work law, under which an open fault heals. |
| `seams` | 0: the sheet, diffuse damage wherever strain exceeds yield. 1: seams, damage only on a seam, at its tip, or at a nucleation site; boundaries one cell wide by construction. 2: rigid pieces; stress is the integral of the unmatched drag over a piece; seams on markers. **The lab defaults to 2** because that is the formulation it exists to look at; the engine's own default is 0 and production is the sheet. |
| `crack_speed_km_per_myr` | How fast a crack tip runs. A rift propagates at tens of kilometres per million years. Read only at `seams = 1` and `2`. |
| `nucleations_per_step` | New cracks per step at the highest-stress intact cells away from existing seams. Read only at `seams = 1` and `2`. |
| `toughness_fraction` | Fracture toughness as a fraction of intact strength. Cracks propagate at this fraction of the stress it takes to nucleate one, for a crack one cell long; longer cracks propagate at less. Read only at `seams = 1` and `2`. |
| `strength_spread` | Initial heterogeneity of the lithosphere. Soft spots concentrate strain and may seed failure. Under seams it is the heterogeneity of the *intact strength* instead: where a crack finds it easier to start and easier to run. |
| `strength_exponent` | How steeply stiffness falls with damage. |
| `drive_wavelength_km` | The coarsest mantle wavelength in kilometres, the same at every resolution and scale. It sets how many mantle cells the world holds and so how many plates can form; 5,120 km is two cells across the default 1024-px world. |
| `drive_shear` | Rotational drive relative to pushing drive. |
| `history_myr` | How long the history runs. |
| `max_cycles` | Solver effort per step. The report shows the worst residual it reached. |
| `solve_divisor` | Kinematic cells per solve cell. 2 solves the velocity on half the grid and lifts strain back in 2 x 2 blocks, so a zone cannot be narrower than two cells; 1 solves on the full grid at about six times the cost. |

Under `seams = 1` the sheet starts intact at strength 1 everywhere,
`yield_percentile` sets the intact strength as a percentile of the first
step's stress rather than of its strain, and `work_damage` picks between two
laws that differ in kind on a seam. At **0**, the default, a seam damages by
its slip rate, so a fault that is slipping stays weak and heals only when it
stops. At **1** it damages by the work its slip dissipates, which is what
C04 ran: an open seam is so weak that it carries almost no traction, so it
dissipates almost nothing however fast it slips, and healing shuts it in two
or three steps at a 10 Myr healing time.

Under `seams = 2` the same rules run on a different velocity. A piece is a
rigid body with three unknowns, two of velocity and one of rotation; the
pieces are coupled through the tractions their seams transmit, and the
balance is the sheet's own equation summed over a piece, so no constant is
added. The stress the tip and nucleation rules read is a second solve, of
the sheet's operator forced by the drag a piece failed to match rather than
by the drive, which is the piece's own elastic deformation and is screened
beyond the stiffness length — so `stiffness_fraction` keeps its meaning and a
piece larger than that length sees its stress partly screened. The strain
rate a seam cell reads is its own slip rate: the velocity jump across it
divided by the cell, plus the elastic strain rate of that second solve at the
same cell, which is the rate a crack's faces displace relative to each other
under the load its own piece carries — so a crack inside a piece, which links
no pair of pieces and has no rigid jump, slips by the elastic part alone. An
intact cell is rigid, has no strain rate of its own, and never damages. And the seam network is a marker set rather than a raster:
the strength field is rebuilt from the markers every step, there is no
advection at all, and a seam cannot be duplicated. **A tip's direction at
`seams = 2` is continuous**, not one of eight: the direction is the
eigenvector of the stress averaged over the tip cell's intact neighbours, the
advance walks from the tip's own markers rather than from its cell's centre,
and the marker it creates sits at the point the walk reached, so a crack
loaded between two lattice directions alternates between them instead of
locking onto one. **And since `WORK_ORDER_C04_6.md` the markers of a crack are
linked in order and the seam is a curve**: the raster draws the segments
between linked markers as well as the markers themselves, so wherever a
segment's two ends go the cells between them are drawn and no cell the network
spans can be left empty by rounding; a tip is the end vertex of its chain, a
crack that reaches within one and a half cells of another links to it and
stops, a marker stays on the curve until it heals to `SUTURE_STRENGTH` rather
than leaving at the weak threshold, and every edge the motion stretches past
one and a half cells is split at its midpoint. `seams = 1` keeps the
eight-direction rule and the point raster.

## The four views the seam formulations added

| View | What it shows |
|---|---|
| `stress` | Magnitude of the stress tensor at the end, stiffness times the strain rate, interpolated to the kinematic grid rather than lifted in 2 x 2 blocks. It is the field the seam rules choose from: which way a tip runs, and which intact cell nucleates next. |
| `intact_strength` | The stress an intact cell carries before it cracks: a percentile of the first step's stress, scaled cell by cell by the strength noise and clipped. Flat black at `seams = 0`, which has no intact strength. |
| `mismatch` | The drag a piece failed to match, `D - u`, zero on seam cells: what the internal-stress solve is forced with at `seams = 2`. Flat black under the sheet and under `seams = 1`, which solve the drive itself. |
| `pieces_motion` | The rigid bodies the last step's solve moved, categorical, with each body's velocity drawn as one arrow from its centroid, hue for direction and brightness for speed as the `velocity` view colours a cell. It is the one view that draws on top of a raster, because a body's velocity is three numbers and not a field. No arrows under the sheet or `seams = 1`, which have no rigid bodies. |

## What to look for

- **`trajectory`** is the view that separates a settled regime from a slow
  collapse. One strip per seed, one filled column per step, height is the weak
  fraction, with a line at the half mark. After the first 50 Myr a settled
  regime is flat. A strip that keeps climbing to the right is still failing at
  the end of the run, whatever the final image looks like.
- **`plates`** with several coloured regions in all eight panels, rather than
  several in one panel and none in the next. A setting that works on one seed
  and not on the next has not been found.
- **`weak_t16`** showing lines rather than blobs: what fails first, and in
  what shape.
- **The report's `solver_residual_max`**, which must be below `1e-3` or the
  result is not a solve: above that the velocity fields are unfinished
  iterates, and the plates, the weak fraction, the trajectory and
  `stable_count` are readings off them rather than readings off a world.
- The report's `stable_count`, which counts the worlds of a generation with 3
  to 8 plates above 1 % of the parent, a final weak fraction between 0.02 and
  0.25, and a peak weak fraction under 1.5 times the final. It is a screening
  number for the person at the dials — **not a gate and not an approval**.

`out/c03_5_sweep.md` is one pass over `stiffness_fraction` against
`yield_percentile` at every other default, so the search does not start from a
blank panel.

[`SEARCH.md`](SEARCH.md) is the regime search on port 5004, which turns these
dials by itself and hands back candidates for these sheets.

## What a finding here is worth

Stability on eight seeds is a screen, not a result. Any setting worth keeping
is then run on the twelve development seeds of [`STATUS.md`](STATUS.md) and
through the blind layer audit before it is frozen into
`engine/history/constants.py`.

These dials are **development instruments**. They are not author controls,
they are not promised to anyone, and they never appear in the production
adapter. The percentile yield in particular is a calibration convenience: it
makes the yield dial mean the same thing at every stiffness, because a stiffer
sheet has smaller strains everywhere. It is a statistical self-reference, not
a physical law, and it does not survive into production. Once a setting is
chosen, the equivalent physical fraction of the drive's characteristic strain
is frozen as a constant and the percentile goes.
