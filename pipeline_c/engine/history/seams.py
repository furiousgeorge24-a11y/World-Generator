"""The seam formulation: the three places a cell is allowed to weaken.

`DESIGN.md` §3.6. The viscous sheet weakens every cell whose load exceeds a
yield, and the length over which the solve makes a plate interior rigid is
also the length over which strain spreads around a weak zone, so a zone
widens until it is that length across. This module keeps the sheet's solve
and changes *where* damage may happen. A cell is either **intact**, part of a
piece, or a **seam**, and damage happens on a seam, at a seam's tip, or at a
nucleation site, and nowhere else. A seam is therefore one cell wide by
construction: a cell beside a long seam is none of those three things however
high its strain.

Every function here is pure — arrays in, arrays out, no engine state — so
each piece is testable on a hand-built field. `kinematics.run_history` calls
them in the order `WORK_ORDER_C04.md` §2.6 fixes: damage and healing, `k` tip
passes, nucleation, advection.

A seam is a cell whose strength is below `WEAK_THRESHOLD`, which is exactly
what `plates.label_plates` and every view and metric already read, so nothing
downstream needs to know this module exists.
"""

from __future__ import annotations

import math

import numpy as np

from ..domain import sample_nearest_periodic
from .constants import SEAM_OPEN_STRENGTH, WEAK_THRESHOLD
from .plates import NEIGHBOURS_8, label_components

#: The eight directions a tip may advance in, as `(dy, dx)` offsets to the
#: neighbour at `(y + dy, x + dx)`. The set is `NEIGHBOURS_8` and so is the
#: order, which is what breaks a tie between two candidates carrying the same
#: traction: `np.argmax` takes the first maximum, so the winner is the one
#: earliest in this tuple and the result is deterministic.
DIRECTIONS: tuple[tuple[int, int], ...] = NEIGHBOURS_8


def seam_mask(strength: np.ndarray) -> np.ndarray:
    """Cells that are seams: strength below the weak threshold."""
    return np.asarray(strength) < WEAK_THRESHOLD


def neighbour_count(mask: np.ndarray) -> np.ndarray:
    """How many of the eight neighbours on the torus are inside `mask`."""
    flags = np.asarray(mask, dtype=bool)
    total = np.zeros(flags.shape, dtype=np.int16)
    for dy, dx in DIRECTIONS:
        total += np.roll(flags, (-dy, -dx), axis=(-2, -1))
    return total


def damage_excess(strength: np.ndarray, power: np.ndarray,
                  power_yield: float, strain_rate: np.ndarray,
                  yield_strain_per_myr: float,
                  work_damage: int) -> np.ndarray:
    """The excess over yield a seam cell damages by. Zero off the seam set.

    Two laws, picked by `work_damage`, and on a seam the choice matters.

    At `1` the excess is the dissipated power over its own yield, the C03.8
    law C04 ran. That law was written for the sheet, where its job was to
    stop a weak zone widening: a cell that has already failed carries almost
    no traction, so it dissipates almost nothing however fast it slips, its
    damage rate falls to nothing, and healing takes it back. On a sheet that
    is the brake. On a seam it is the wrong law, because a seam does not
    widen and does have to persist while it slips: an open seam's stiffness
    is `KAPPA0 * STRENGTH_MIN ** exponent`, so its power is near zero, its
    damage is near zero, and healing shuts it in two or three steps.

    At `0`, the default, the excess is the strain rate over the same
    percentile of the same first-step field the sheet reads it at. That is
    the ordinary friction of a fault: **a fault that is slipping stays weak,
    and heals only when it stops.** An open seam carrying the whole velocity
    jump has a large strain rate however small its traction, so it damages
    while it slips and heals when the slip moves elsewhere. On the sheet this
    law ran away, because damage could spread to the neighbour that the
    weakened cell loaded; on a seam it cannot, since an intact cell's damage
    rate is zero wherever it sits.

    Intact cells never damage under either law: that is the seam formulation,
    and it is what holds a seam one cell wide.
    """
    strength = np.asarray(strength, dtype=np.float64)
    if work_damage:
        excess = np.maximum(power / power_yield - 1.0, 0.0)
    else:
        excess = np.maximum(strain_rate / yield_strain_per_myr - 1.0, 0.0)
    return np.where(seam_mask(strength), excess, 0.0)


def tips(seam: np.ndarray) -> np.ndarray:
    """Seam cells with at most one seam neighbour in the 8-neighbourhood.

    An isolated cell has none and is a tip; the two ends of a line have one
    each; every cell of a closed loop has two, so a loop has no tips and
    stops growing, which is what makes a loop the thing that cuts a piece in
    two rather than a thing that keeps eating the sheet.
    """
    flags = np.asarray(seam, dtype=bool)
    return flags & (neighbour_count(flags) <= 1)


def crack_lengths(seam: np.ndarray) -> np.ndarray:
    """Size in cells of the 8-connected seam component each cell belongs to.

    Zero off the seam set. This is the `L` of the Griffith rule, in cells,
    with the unit length one cell.
    """
    flags = np.asarray(seam, dtype=bool)
    lengths = np.zeros(flags.shape, dtype=np.int64)
    if not flags.any():
        return lengths
    labels = label_components(flags, 8)
    inside = labels[flags]
    lengths[flags] = np.bincount(inside)[inside]
    return lengths


def traction_magnitude(sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray,
                       dy: int, dx: int) -> np.ndarray:
    """`|sigma . n|` for the seam that would run along `(dy, dx)`.

    A seam along `d` carries the traction on its own face, so the normal is
    `d` turned a quarter turn: `n = (-dy, dx) / |d|` in `(x, y)`. The sign of
    the normal does not survive the magnitude, so which quarter turn does not
    matter.
    """
    norm = math.hypot(float(dx), float(dy))
    nx = -float(dy) / norm
    ny = float(dx) / norm
    tx = sxx * nx + sxy * ny
    ty = sxy * nx + syy * ny
    return np.sqrt(tx * tx + ty * ty)


def tip_pass(strength: np.ndarray, sxx: np.ndarray, syy: np.ndarray,
             sxy: np.ndarray, sigma_c_field: np.ndarray,
             toughness_fraction: float = 1.0
             ) -> tuple[np.ndarray, int, int]:
    """One pass over every tip, in eight directions. The `seams = 1` rule.

    Returns the strength, the tips, the advances. Since
    `WORK_ORDER_C04_5.md` §1 the block model at `seams = 2` runs
    `tip_pass_continuous` instead, whose direction is not one of eight; this
    rule and the pins that guard it are untouched by that order.

    For each tip and each of the eight directions to an **intact** neighbour,
    the candidate qualifies when the traction the would-be seam would carry
    reaches `toughness_fraction * sigma_c_field[candidate] / sqrt(L)`, with
    `L` the length in cells of the tip's own crack. That is the Griffith
    rule: a long crack concentrates stress at its tip, so it runs on into
    rock that a short one could not open. The tip advances into the
    qualifying candidate carrying the largest traction and that cell's
    strength becomes `SEAM_OPEN_STRENGTH`; a tip with no qualifying candidate
    stands still.

    **Why the toughness is a fraction and not the intact strength.** The
    threshold was written with the unit length one cell and no coefficient,
    which fixed the fracture toughness at the intact strength times the
    square root of one cell: a crack one cell long propagated at exactly the
    stress it takes to nucleate one. Intact strength and toughness are
    different material properties, so `toughness_fraction` names the ratio
    and makes it a setting. At `1.0`, the default, the threshold is what it
    was, bit for bit — multiplying by one is exact — and nucleation, which
    still reads the full `sigma_c_field`, is untouched at every value.

    The stress tensor is read at the candidate cell, on the bilinear lift, so
    a direction is chosen from a field that varies cell by cell rather than
    in 2 x 2 steps.

    Two tips may open the same cell, and a tip may open a cell belonging to
    another crack: nothing special is done either way, since the seam mask
    and `label_components` are recomputed from the strength field and merge
    what has met.
    """
    strength = np.asarray(strength, dtype=np.float64)
    toughness = float(toughness_fraction)
    seam = seam_mask(strength)
    tip = tips(seam)
    count = int(tip.sum())
    if count == 0:
        return strength, 0, 0

    n = strength.shape[-1]
    lengths = crack_lengths(seam)
    ty, tx = np.nonzero(tip)
    root_length = np.sqrt(lengths[ty, tx].astype(np.float64))

    scores = np.empty((len(DIRECTIONS), ty.size), dtype=np.float64)
    rows = np.empty((len(DIRECTIONS), ty.size), dtype=np.int64)
    columns = np.empty((len(DIRECTIONS), ty.size), dtype=np.int64)
    for index, (dy, dx) in enumerate(DIRECTIONS):
        cy = (ty + dy) % n
        cx = (tx + dx) % n
        rows[index] = cy
        columns[index] = cx
        magnitude = traction_magnitude(sxx[cy, cx], syy[cy, cx], sxy[cy, cx],
                                       dy, dx)
        qualifies = (~seam[cy, cx]) & (
            magnitude >= (toughness * sigma_c_field[cy, cx]) / root_length)
        scores[index] = np.where(qualifies, magnitude, -np.inf)

    # First maximum wins, so a tie between two directions carrying the same
    # traction is broken by the fixed order of `DIRECTIONS`.
    best = np.argmax(scores, axis=0)
    lane = np.arange(ty.size)
    advanced = np.isfinite(scores[best, lane])
    if not advanced.any():
        return strength, count, 0
    strength = strength.copy()
    strength[rows[best, lane][advanced], columns[best, lane][advanced]] = \
        SEAM_OPEN_STRENGTH
    return strength, count, int(advanced.sum())


def principal_normal(sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
    """The unit normal `n` maximising `|sigma . n|`, and the two `|lambda|`.

    Returns `(nx, ny, larger, smaller)`: the eigenvector of the eigenvalue of
    largest absolute value, in `(x, y)`, and the two absolute eigenvalues,
    larger first. `|sigma . n|` over unit normals is maximised by exactly that
    eigenvector, so this is the continuous form of the quantity the
    eight-direction tip rule maximised over eight offsets.

    For the symmetric tensor `[[sxx, sxy], [sxy, syy]]` the eigenvalues are
    the mean of the diagonal plus and minus the radius of Mohr's circle. Both
    `(sxy, lambda - sxx)` and `(lambda - syy, sxy)` are eigenvectors of
    `lambda` by the characteristic equation; the longer of the two is taken,
    so the answer is stable wherever one of them collapses. Where both
    collapse the tensor is isotropic, every direction is an eigenvector, and
    the answer is `(1, 0)` — deterministic, and the run counts how often the
    two absolute eigenvalues came within a hundredth of each other.
    """
    sxx = np.asarray(sxx, dtype=np.float64)
    syy = np.asarray(syy, dtype=np.float64)
    sxy = np.asarray(sxy, dtype=np.float64)
    mean = 0.5 * (sxx + syy)
    radius = np.hypot(0.5 * (sxx - syy), sxy)
    high = mean + radius
    low = mean - radius
    take_high = np.abs(high) >= np.abs(low)
    lam = np.where(take_high, high, low)
    ax = sxy
    ay = lam - sxx
    bx = lam - syy
    by = sxy
    first = np.hypot(ax, ay) >= np.hypot(bx, by)
    vx = np.where(first, ax, bx)
    vy = np.where(first, ay, by)
    norm = np.hypot(vx, vy)
    flat = norm <= 0.0
    safe = np.where(flat, 1.0, norm)
    nx = np.where(flat, 1.0, vx / safe)
    ny = np.where(flat, 0.0, vy / safe)
    larger = np.maximum(np.abs(high), np.abs(low))
    smaller = np.minimum(np.abs(high), np.abs(low))
    return nx, ny, larger, smaller


def traction_on(sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray,
                nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """`|sigma . n|` for a continuous unit normal, elementwise.

    `traction_magnitude` is this for one of the eight lattice directions,
    written with the direction rather than the normal; this one takes the
    normal itself, because the continuous rule has already turned a direction
    into it.
    """
    tx = sxx * nx + sxy * ny
    ty = sxy * nx + syy * ny
    return np.hypot(tx, ty)


#: How far a tip may be carried along its direction, in whole cell lengths,
#: before it must have left its own cell. A step is one cell long and a
#: marker sits within half a cell of its own centre, so by the third step the
#: point is at least 2.29 cells out and one of its coordinates is at least
#: 1.6, which cannot round back to the cell it started in.
_MAX_ADVANCE_STEPS = 3


def step_out(offset_x: np.ndarray, offset_y: np.ndarray,
              ex: np.ndarray, ey: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray]:
    """Walk from a tip along `(ex, ey)` until the nearest cell changes.

    Everything is an offset from the tip's own cell centre, in cell units, so
    there is no wrapping to do here. Returns the candidate cell's offset from
    the tip cell as `(dx, dy)`, where inside the candidate the point landed as
    `(px, py)`, and whether the walk found a candidate in the eight
    neighbours at all.

    A step is one cell long, so a coordinate can grow by at most one per step
    from a start within half a cell of the centre: the first point that
    rounds away from the tip cell rounds to an offset of at most one in each
    coordinate and the candidate is one of the eight neighbours. The single
    exception is a coordinate landing on exactly 1.5, which `np.rint` rounds
    to 2 as it rounds every half to the even integer; the walk reports that
    tip as having no candidate rather than reaching past its neighbours.
    """
    shape = np.shape(offset_x)
    dx = np.zeros(shape, dtype=np.float64)
    dy = np.zeros(shape, dtype=np.float64)
    px = np.zeros(shape, dtype=np.float64)
    py = np.zeros(shape, dtype=np.float64)
    found = np.zeros(shape, dtype=bool)
    for step in range(1, _MAX_ADVANCE_STEPS + 1):
        walk_x = offset_x + step * ex
        walk_y = offset_y + step * ey
        round_x = np.rint(walk_x)
        round_y = np.rint(walk_y)
        moved = (round_x != 0.0) | (round_y != 0.0)
        take = moved & ~found
        dx = np.where(take, round_x, dx)
        dy = np.where(take, round_y, dy)
        px = np.where(take, walk_x - round_x, px)
        py = np.where(take, walk_y - round_y, py)
        found |= moved
        if bool(found.all()):
            break
    inside = found & (np.abs(dx) <= 1.0) & (np.abs(dy) <= 1.0)
    return dx, dy, px, py, inside


def tip_pass_continuous(strength: np.ndarray, sxx: np.ndarray,
                        syy: np.ndarray, sxy: np.ndarray,
                        sigma_c_field: np.ndarray, offsets: np.ndarray,
                        holds: np.ndarray, toughness_fraction: float = 1.0
                        ) -> tuple[np.ndarray, int, int, np.ndarray, int, int]:
    """One pass over every tip, with the tip's direction continuous.

    `WORK_ORDER_C04_5.md` §1. `tip_pass` scores the eight neighbours of a tip
    and steps into the best of them, so under a field that varies over
    hundreds of cells the winner is whichever of the eight lies nearest the
    principal axis and it wins again at the next advance: a crack loaded at
    20 degrees runs at 0. Nothing in that rule can alternate two lattice
    directions to make an angle between them, because it has no memory of how
    far the tip has been pushed off its line. This one has: the tip's position
    is the position of the markers in its cell, which is a float, and the
    advance starts from that position rather than from the cell's centre.

    - **The position.** A tip is still a seam cell with at most one seam
      neighbour. `offsets` is `markers.cell_offsets`, so `p` is the mean
      position of the markers the tip cell holds. A cell with none — which
      cannot happen under the block model, where every seam cell is a
      marker — is read as its own centre.
    - **The direction.** The lifted stress tensor is averaged with equal
      weight over the tip cell's **intact** 8-neighbours, which are the cells
      the eight-direction rule scored; `n` is the eigenvector of the largest
      absolute eigenvalue of that average, which is the same quantity the old
      rule maximised over eight offsets, and the seam runs along `n` turned a
      quarter turn. The sign is the one that points away from the crack: a
      positive dot product with the vector from the tip's one seam neighbour
      to the tip. A tip with no seam neighbour is a fresh nucleus and tries
      both signs, keeping the one whose candidate carries the larger
      traction; ties, and a direction exactly along the crack, take the sign
      with positive `x`, then positive `y`.
    - **The advance.** The walk of `step_out`: one cell length at a time
      from `p` along the direction until the nearest cell differs from the
      tip's. That cell is the candidate and the point reached is `p'`. The
      candidate qualifies exactly as it did — intact, and `|sigma . n|` read
      at the candidate with this `n` reaching
      `toughness_fraction * sigma_c_field[candidate] / sqrt(L)` — and a tip
      with no qualifying candidate stands still.

    Returns the strength, the tips, the advances, the `(2, n, n)` offsets
    where inside its cell each opened cell's marker goes (`NaN` off the
    opened set), how many tips stood on a cell holding more than one marker,
    and how many tips read a tensor whose two absolute eigenvalues were
    within one per cent of each other — for those the direction is whatever
    the eigensolver returns, deterministically.

    Two tips may open the same cell. The marker goes at the `p'` of the first
    of them in row-major order of tips, which is fixed, so the pass is
    deterministic however NumPy orders a scatter.
    """
    strength = np.asarray(strength, dtype=np.float64)
    toughness = float(toughness_fraction)
    n = strength.shape[-1]
    opened_offsets = np.full((2, n, n), np.nan, dtype=np.float64)
    seam = seam_mask(strength)
    tip = tips(seam)
    count = int(tip.sum())
    if count == 0:
        return strength, 0, 0, opened_offsets, 0, 0

    lengths = crack_lengths(seam)
    ty, tx = np.nonzero(tip)
    root_length = np.sqrt(lengths[ty, tx].astype(np.float64))
    intact = ~seam

    offsets = np.asarray(offsets, dtype=np.float64)
    offset_x = offsets[0][ty, tx]
    offset_y = offsets[1][ty, tx]
    offset_x = np.where(np.isnan(offset_x), 0.0, offset_x)
    offset_y = np.where(np.isnan(offset_y), 0.0, offset_y)
    held = np.asarray(holds)[ty, tx]
    multi = int((held > 1).sum())

    # The tensor the direction is read from: the mean over the intact
    # 8-neighbours, which are the cells the eight-direction rule scored. A
    # tip whose eight neighbours are all seams has nowhere to go.
    total_xx = np.zeros(ty.size, dtype=np.float64)
    total_yy = np.zeros(ty.size, dtype=np.float64)
    total_xy = np.zeros(ty.size, dtype=np.float64)
    neighbours = np.zeros(ty.size, dtype=np.float64)
    seam_dx = np.zeros(ty.size, dtype=np.float64)
    seam_dy = np.zeros(ty.size, dtype=np.float64)
    has_seam = np.zeros(ty.size, dtype=bool)
    for dy, dx in DIRECTIONS:
        cy = (ty + dy) % n
        cx = (tx + dx) % n
        free = intact[cy, cx]
        weight = free.astype(np.float64)
        total_xx += weight * sxx[cy, cx]
        total_yy += weight * syy[cy, cx]
        total_xy += weight * sxy[cy, cx]
        neighbours += weight
        # A tip has at most one seam neighbour, so the first found is it.
        take = seam[cy, cx] & ~has_seam
        seam_dx = np.where(take, float(dx), seam_dx)
        seam_dy = np.where(take, float(dy), seam_dy)
        has_seam |= seam[cy, cx]
    live = neighbours > 0.0
    divisor = np.where(live, neighbours, 1.0)
    nx, ny, larger, smaller = principal_normal(total_xx / divisor,
                                               total_yy / divisor,
                                               total_xy / divisor)
    degenerate = int((live & (larger - smaller <= 0.01 * larger)
                      & (larger > 0.0)).sum())

    # The seam runs along the normal turned a quarter turn.
    ex = ny
    ey = -nx
    # Away from the crack: the vector from the tip's seam neighbour to the
    # tip is minus that neighbour's offset.
    away = ex * (-seam_dx) + ey * (-seam_dy)
    # The sign a tip takes when nothing else decides: positive `x`, then
    # positive `y`.
    default = np.where((ex > 0.0) | ((ex == 0.0) & (ey > 0.0)), 1.0, -1.0)

    trials = []
    for sign in (1.0, -1.0):
        dx, dy, px, py, inside = step_out(offset_x, offset_y,
                                           sign * ex, sign * ey)
        cy = (ty + dy.astype(np.int64)) % n
        cx = (tx + dx.astype(np.int64)) % n
        pull = traction_on(sxx[cy, cx], syy[cy, cx], sxy[cy, cx], nx, ny)
        usable = inside & live
        trials.append({
            "rows": cy, "columns": cx, "px": px, "py": py,
            "traction": np.where(usable, pull, -np.inf),
            "qualifies": usable & (~seam[cy, cx]) & (
                pull >= (toughness * sigma_c_field[cy, cx]) / root_length),
        })

    positive, negative = trials
    # A tip with a seam neighbour takes the sign that points away from it; a
    # fresh nucleus takes the sign whose candidate carries more traction, and
    # a tie — or a direction exactly along the crack — takes the default.
    by_traction = np.where(positive["traction"] > negative["traction"], 1.0,
                           np.where(positive["traction"] < negative["traction"],
                                    -1.0, default))
    sign = np.where(has_seam & (away > 0.0), 1.0,
                    np.where(has_seam & (away < 0.0), -1.0, by_traction))
    take_positive = sign > 0.0
    rows = np.where(take_positive, positive["rows"], negative["rows"])
    columns = np.where(take_positive, positive["columns"], negative["columns"])
    px = np.where(take_positive, positive["px"], negative["px"])
    py = np.where(take_positive, positive["py"], negative["py"])
    advanced = np.where(take_positive, positive["qualifies"],
                        negative["qualifies"])

    if not advanced.any():
        return strength, count, 0, opened_offsets, multi, degenerate
    strength = strength.copy()
    strength[rows[advanced], columns[advanced]] = SEAM_OPEN_STRENGTH
    # One marker per opened cell, at the first advancing tip's `p'`.
    flat = (rows[advanced] * n + columns[advanced])
    _cell, first = np.unique(flat, return_index=True)
    opened_offsets[0].reshape(-1)[flat[first]] = px[advanced][first]
    opened_offsets[1].reshape(-1)[flat[first]] = py[advanced][first]
    return (strength, count, int(advanced.sum()), opened_offsets, multi,
            degenerate)


def nucleate(strength: np.ndarray, smag: np.ndarray,
             sigma_c_field: np.ndarray, cap: int) -> tuple[np.ndarray, int]:
    """Open at most `cap` new cracks at the highest-stress intact cells.

    A candidate is intact, has no seam anywhere in its 8-neighbourhood, and
    carries a stress magnitude at least its own intact strength. Candidates
    are ordered by `smag / sigma_c_field` descending and ties by row then
    column, which is the order the flat index already has, so the pick is
    deterministic. A nucleus is a crack of length 1 and is a tip on the next
    pass.

    Excluding the neighbourhood of an existing seam is what keeps nucleation
    from thickening a seam: a new crack starts where there is no crack.
    """
    strength = np.asarray(strength, dtype=np.float64)
    cap = int(cap)
    if cap <= 0:
        return strength, 0
    seam = seam_mask(strength)
    candidates = (~seam) & (neighbour_count(seam) == 0) & (smag >= sigma_c_field)
    flat = np.nonzero(candidates.ravel())[0]
    if flat.size == 0:
        return strength, 0
    ratio = (smag / sigma_c_field).ravel()[flat]
    # A stable sort of the negated ratio: descending by ratio, and among
    # equal ratios in flat-index order, which is row then column.
    order = np.argsort(-ratio, kind="stable")
    picked = flat[order[:cap]]
    strength = strength.copy()
    strength.reshape(-1)[picked] = SEAM_OPEN_STRENGTH
    return strength, int(picked.size)


def advect_nearest(strength: np.ndarray, displacement: np.ndarray,
                   offset: np.ndarray, columns: np.ndarray,
                   rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Carry `strength` with the lithosphere by whole cells.

    `displacement` is `(2, n, n)` in cells, the offset from a cell to the
    point its material came from; `offset` is the sub-cell remainder the last
    call could not spend, in the same units and shape. The two are added, the
    sum is rounded to whole cells, the field is sampled there, and what is
    left over is returned to be carried into the next step.

    **Why the remainder is carried.** A seam is a discontinuity and bilinear
    sampling of it is a ramp: a one-cell line resampled at a fractional
    offset puts a share of its value into the neighbour, and within a few
    steps the line is two or three cells across. That is widening by
    arithmetic and it would defeat the whole formulation. Nearest-cell
    sampling keeps the line one cell wide, but on its own it also rounds
    every displacement below half a cell to nothing, so a sheet moving at a
    third of a cell per step would never move at all. Carrying the remainder
    spends it: the field steps a whole cell whenever the arrears reach one,
    the mean speed is right, and the position error stays inside half a cell
    instead of accumulating.

    The remainder is held per cell in the mantle frame rather than carried
    with the material, which is what makes it a fixed-size array and not a
    marker set; the error that costs is bounded by the same half cell.
    """
    total = np.asarray(displacement, dtype=np.float64) + offset
    whole = np.rint(total)
    moved = sample_nearest_periodic(strength,
                                    columns + whole[0], rows + whole[1])
    return moved, total - whole


def intact_strength_field(sigma_c: float, noise: np.ndarray, spread: float,
                          clip: tuple[float, float]) -> np.ndarray:
    """The heterogeneity of the intact sheet, as a field of yield stresses.

    Under the sheet the strength noise was the initial strength itself. Here
    the sheet starts intact and uniform, and the noise is where a crack finds
    it easier to start and easier to run: `sigma_c` scaled cell by cell and
    clipped, so no cell is arbitrarily weak or arbitrarily strong however the
    spread is set.
    """
    factor = np.clip(1.0 + float(spread) * np.asarray(noise, dtype=np.float64),
                     clip[0], clip[1])
    return float(sigma_c) * factor


__all__ = [
    "DIRECTIONS",
    "advect_nearest",
    "crack_lengths",
    "damage_excess",
    "intact_strength_field",
    "neighbour_count",
    "nucleate",
    "principal_normal",
    "seam_mask",
    "step_out",
    "tip_pass",
    "tip_pass_continuous",
    "tips",
    "traction_magnitude",
    "traction_on",
]
