"""Rigid pieces: the block model of `DESIGN.md` §3.6's last paragraph.

Under `seams = 1` a piece is a stiff region of the same viscous sheet, and how
rigidly it moves is what the stiffness dial sets. C04.1 measured what that
costs: the stress a crack sees is the sheet's, a local balance sitting in a
band fixed by the drive's gradients, so every point of that band stays loaded
however many cracks its neighbours already carry and the cracks never leave
it. This module replaces the velocity solve inside a piece with a rigid-body
solve — three unknowns per piece, coupled through seam tractions — and leaves
the sheet solve to carry the *internal* stress, which is a different field
with a different shape.

**No new constant enters.** The sheet's discrete equation per cell is

    u_i - sum_e k_e (u_j - u_i) = D_i

with `k_e` the harmonic edge stiffness. Sum that over a piece whose velocity
is rigid, `u_i = v + omega x r_i`: every edge with both ends inside the piece
cancels, and what is left is the rigid body's own force balance — basal drag
over the piece against the seam tractions on its boundary. The drag
coefficient is the sheet's identity term, one per cell; a seam's coupling is
the sheet's own `kappa(S)`. Nothing here is fitted.

Every function is pure: arrays in, arrays out, no engine state.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np

from .constants import WEAK_THRESHOLD
from .plates import NEIGHBOURS_4, NEIGHBOURS_8, label_plates

#: The six unordered pairs of the four cardinal neighbour slots. A seam cell
#: links each pair of *distinct* pieces among its intact 4-neighbours, and
#: four slots can hold at most four distinct pieces, so six pairs cover it.
_PAIRS: tuple[tuple[int, int], ...] = tuple(itertools.combinations(range(4), 2))


def minimal_image(delta: np.ndarray | float, n: int) -> np.ndarray:
    """The shortest signed offset `delta` on a torus of `n` cells.

    `((d + n/2) mod n) - n/2`, so the result lies in `[-n/2, n/2)`. An offset
    of exactly half the world is reported as `-n/2`; on a torus the two are
    the same distance apart and the choice only has to be consistent.
    """
    half = 0.5 * n
    return np.mod(np.asarray(delta, dtype=np.float64) + half, n) - half


@dataclass(frozen=True, slots=True)
class Pieces:
    """The rigid bodies of one step, and the frame each one rotates about.

    `labels` is `label_plates`: the 4-connected components of the intact
    cells, `-1` on seam cells, renumbered largest first. `centroid` is the
    mean of each piece's cells' minimal-image offsets from its own reference
    cell — the first of its cells in raster order — added back to that
    reference, so it is a position in cell units that may sit outside
    `[0, n)`. `rx` and `ry` are each intact cell's offset from its own
    piece's centroid, zero on seam cells; they sum to zero over a piece by
    construction, which is what makes the drag's contribution to the torque
    independent of where the reference cell happened to fall.

    `wrapping` marks a piece whose cells cover every column or every row.
    Such a piece goes all the way round the torus, its minimal-image offsets
    are not the offsets of any planar body, and it has no well-defined
    rotation: `rigid_motion` pins its `omega` at zero and the run counts how
    many there were. A single-cell piece has no rotation either — its inertia
    is exactly zero — and is pinned the same way.

    `folded` marks the same failure one step earlier. A minimal image folds
    at half the world: a piece that reaches more than `n / 2` across an axis
    without covering it has cells whose true offset from the reference is
    read back with the wrong sign, so its centroid, its inertia and its body
    offsets are not a planar body's either. It is pinned exactly as a
    wrapping piece is — and a wrapping piece is always folded, so the pinned
    set is `folded` plus the single cells. Pinning makes every one of those
    quantities inert: a pinned piece's `omega` is zero, and its `rx` and `ry`
    enter the assembled system only through its own torque row and column,
    which the pin clears.
    """

    labels: np.ndarray       # (n, n) int32; -1 on seam cells
    count: int               # number of pieces, N
    cells: np.ndarray        # (N,) int64, cells per piece
    centroid: np.ndarray     # (2, N) float, x then y, in cell units
    rx: np.ndarray           # (n, n) float, 0 on seam cells
    ry: np.ndarray
    inertia: np.ndarray      # (N,) sum of rx**2 + ry**2 over the piece
    wrapping: np.ndarray     # (N,) bool
    folded: np.ndarray       # (N,) bool, reaches more than half the torus

    @property
    def pinned(self) -> np.ndarray:
        """Pieces whose `omega` is fixed at zero."""
        return self.folded | self.wrapping | (self.inertia <= 0.0)

    @property
    def largest_share(self) -> float:
        """Cells of the largest piece as a share of the whole grid."""
        if self.count == 0:
            return 0.0
        n = self.labels.shape[-1]
        return float(int(self.cells.max())) / float(n * n)

    @property
    def second_size(self) -> int:
        """Cells of the second largest piece: what a closed loop enclosed.

        `label_plates` renumbers by area, largest first, so this is piece 1.
        Zero while there is only one piece.
        """
        return int(self.cells[1]) if self.count > 1 else 0


def piece_frames(labels: np.ndarray) -> Pieces:
    """Centroids, body offsets, inertias and the wrapping flag, per piece."""
    labels = np.asarray(labels)
    n = labels.shape[-1]
    intact = labels >= 0
    count = int(labels.max()) + 1 if intact.any() else 0
    if count == 0:
        return Pieces(labels=labels, count=0,
                      cells=np.zeros(0, dtype=np.int64),
                      centroid=np.zeros((2, 0), dtype=np.float64),
                      rx=np.zeros((n, n), dtype=np.float64),
                      ry=np.zeros((n, n), dtype=np.float64),
                      inertia=np.zeros(0, dtype=np.float64),
                      wrapping=np.zeros(0, dtype=bool),
                      folded=np.zeros(0, dtype=bool))

    rows, columns = np.nonzero(intact)
    lab = labels[intact].astype(np.int64)
    flat = rows.astype(np.int64) * n + columns.astype(np.int64)
    # The reference cell is each piece's first cell in raster order.
    # `np.nonzero` returns raster order, so assigning in reverse leaves the
    # smallest flat index as the last write and therefore the survivor.
    reference = np.zeros(count, dtype=np.int64)
    reference[lab[::-1]] = flat[::-1]
    ref_x = (reference % n).astype(np.float64)
    ref_y = (reference // n).astype(np.float64)

    dx = minimal_image(columns - ref_x[lab], n)
    dy = minimal_image(rows - ref_y[lab], n)
    cells = np.bincount(lab, minlength=count).astype(np.int64)
    mean_x = np.bincount(lab, dx, minlength=count) / cells
    mean_y = np.bincount(lab, dy, minlength=count) / cells

    body_x = dx - mean_x[lab]
    body_y = dy - mean_y[lab]
    rx = np.zeros((n, n), dtype=np.float64)
    ry = np.zeros((n, n), dtype=np.float64)
    rx[intact] = body_x
    ry[intact] = body_y
    inertia = np.bincount(lab, body_x * body_x + body_y * body_y,
                          minlength=count)

    # Occupancy in the *offset* frame, one row per piece: covering every
    # offset is covering every column, because the offsets are the columns
    # shifted by the reference, and the span in that frame is what says
    # whether the minimal image folded.
    half = n // 2
    occupied_x = np.zeros((count, n), dtype=bool)
    occupied_y = np.zeros((count, n), dtype=bool)
    occupied_x[lab, np.rint(dx).astype(np.int64) + half] = True
    occupied_y[lab, np.rint(dy).astype(np.int64) + half] = True
    wrapping = occupied_x.all(axis=1) | occupied_y.all(axis=1)
    span_x = _span(occupied_x)
    span_y = _span(occupied_y)
    folded = (span_x >= half) | (span_y >= half)

    centroid = np.stack((ref_x + mean_x, ref_y + mean_y))
    return Pieces(labels=labels, count=count, cells=cells, centroid=centroid,
                  rx=rx, ry=ry, inertia=inertia, wrapping=wrapping,
                  folded=folded)


def _span(occupied: np.ndarray) -> np.ndarray:
    """Last occupied index minus the first, per row. `(N,)`."""
    width = occupied.shape[1]
    first = np.argmax(occupied, axis=1)
    last = width - 1 - np.argmax(occupied[:, ::-1], axis=1)
    return last - first


@dataclass(frozen=True, slots=True)
class SeamFrame:
    """Every seam cell, the pieces it touches, and what it transmits.

    A seam cell is one cell wide and carries no drag. It links each pair of
    *distinct* pieces among its intact 4-neighbours; for the pair `(P, Q)` it
    transmits `c (u_Q(x_s) - u_P(x_s))` to `P` and the opposite to `Q`, with
    `c = kappa(S_s) / 2`: two edges in series through the seam cell, each
    dominated by the seam's own stiffness. A seam cell whose intact
    neighbours all belong to one piece transmits nothing, and one with no
    intact neighbour at all touches nothing.

    `neighbour` is `(4, S)`: the piece label at each of the four cardinal
    neighbours, `-1` where that neighbour is itself a seam. `first` marks the
    slots holding a piece not already seen in an earlier slot, so
    `first.sum(axis=0)` is the number of distinct pieces the seam cell
    touches and the pairs of first slots are exactly the links, counted once
    each. `rx` and `ry`, also `(4, S)`, are the seam cell's offset from that
    slot's piece's centroid, which is where the traction is applied and about
    which its torque is taken.
    """

    rows: np.ndarray         # (S,) int64
    columns: np.ndarray      # (S,) int64
    neighbour: np.ndarray    # (4, S) int64, -1 off a piece
    first: np.ndarray        # (4, S) bool
    rx: np.ndarray           # (4, S) float
    ry: np.ndarray
    coupling: np.ndarray     # (S,) float, c_s
    pair: np.ndarray         # (6, S) bool, the valid links

    @property
    def size(self) -> int:
        return int(self.rows.size)


def _first_occurrence(neighbour: np.ndarray) -> np.ndarray:
    """Slots holding a piece label no earlier slot holds. `(4, S)` bool."""
    first = neighbour >= 0
    slots = neighbour.shape[0]
    for later in range(1, slots):
        for earlier in range(later):
            first[later] &= neighbour[later] != neighbour[earlier]
    return first


def seam_frame(pieces: Pieces, strength: np.ndarray, kappa0: float,
               exponent: int) -> SeamFrame:
    """The seam cells of one step with their links and their couplings."""
    labels = pieces.labels
    n = labels.shape[-1]
    seam = labels < 0
    rows, columns = np.nonzero(seam)
    size = rows.size
    neighbour = np.full((4, size), -1, dtype=np.int64)
    rx = np.zeros((4, size), dtype=np.float64)
    ry = np.zeros((4, size), dtype=np.float64)
    if size and pieces.count:
        for slot, (dy, dx) in enumerate(NEIGHBOURS_4):
            found = labels[(rows + dy) % n, (columns + dx) % n]
            neighbour[slot] = found
            safe = np.where(found >= 0, found, 0)
            rx[slot] = minimal_image(columns - pieces.centroid[0][safe], n)
            ry[slot] = minimal_image(rows - pieces.centroid[1][safe], n)
    first = _first_occurrence(neighbour)
    pair = np.zeros((len(_PAIRS), size), dtype=bool)
    for index, (a, b) in enumerate(_PAIRS):
        pair[index] = first[a] & first[b] & (neighbour[a] != neighbour[b])
    coupling = 0.5 * float(kappa0) * (
        np.asarray(strength, dtype=np.float64)[rows, columns] ** int(exponent))
    return SeamFrame(rows=rows, columns=columns, neighbour=neighbour,
                     first=first, rx=rx, ry=ry, coupling=coupling, pair=pair)


def _ordered_links(frame: SeamFrame) -> tuple[np.ndarray, ...]:
    """Each link twice, once from each end's point of view.

    The force a link puts on `P` is `c (u_Q - u_P)` and the one it puts on
    `Q` is the same expression with the two swapped, so one formula covers
    both ends and the assembled matrix comes out symmetric.
    """
    here: list[np.ndarray] = []
    there: list[np.ndarray] = []
    here_x: list[np.ndarray] = []
    here_y: list[np.ndarray] = []
    there_x: list[np.ndarray] = []
    there_y: list[np.ndarray] = []
    strength: list[np.ndarray] = []
    for index, (a, b) in enumerate(_PAIRS):
        mask = frame.pair[index]
        if not mask.any():
            continue
        for one, other in ((a, b), (b, a)):
            here.append(frame.neighbour[one][mask])
            there.append(frame.neighbour[other][mask])
            here_x.append(frame.rx[one][mask])
            here_y.append(frame.ry[one][mask])
            there_x.append(frame.rx[other][mask])
            there_y.append(frame.ry[other][mask])
            strength.append(frame.coupling[mask])
    if not here:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return (empty_i, empty_i, empty_f, empty_f, empty_f, empty_f, empty_f)
    return (np.concatenate(here), np.concatenate(there),
            np.concatenate(here_x), np.concatenate(here_y),
            np.concatenate(there_x), np.concatenate(there_y),
            np.concatenate(strength))


def assemble(pieces: Pieces, frame: SeamFrame,
             drive: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The `3N x 3N` force and torque balance, and its right-hand side.

    Unknowns are `(vx, vy, omega)` per piece, in that order. Row `3P` is the
    x-force balance of piece `P`, `3P + 1` the y-force, `3P + 2` the torque
    about `P`'s centroid. Both are assembled with `bincount` over cells and
    over links; there is no Python loop over cells.

    The drag part is exact and needs no assembly beyond a bincount: because
    the body offsets sum to zero over a piece, the rigid velocity's drag
    contributes `-m_P v_P` to the force and `-I_P omega_P` to the torque, and
    the drive contributes its own sums.

    A piece with no well-defined rotation — one that wraps or folds round the
    torus, or a single cell, whose inertia is exactly zero — has its `omega`
    pinned at zero by clearing that row and column and putting a one on the
    diagonal.
    Clearing the column as well as the row keeps the matrix symmetric, which
    is what it is: the balance is the gradient of a quadratic form in the
    unknowns, so `A` equals its own transpose to the last bit.
    """
    count = pieces.count
    size = 3 * count
    if size == 0:
        return (np.zeros((0, 0), dtype=np.float64),
                np.zeros(0, dtype=np.float64))

    labels = pieces.labels
    intact = labels >= 0
    lab = labels[intact].astype(np.int64)
    drive = np.asarray(drive, dtype=np.float64)
    drive_x = drive[0][intact]
    drive_y = drive[1][intact]
    body_x = pieces.rx[intact]
    body_y = pieces.ry[intact]
    sum_x = np.bincount(lab, drive_x, minlength=count)
    sum_y = np.bincount(lab, drive_y, minlength=count)
    sum_t = np.bincount(lab, body_x * drive_y - body_y * drive_x,
                        minlength=count)

    index = np.arange(count, dtype=np.int64)
    rows = [3 * index, 3 * index + 1, 3 * index + 2]
    columns = [3 * index, 3 * index + 1, 3 * index + 2]
    values = [-pieces.cells.astype(np.float64),
              -pieces.cells.astype(np.float64),
              -pieces.inertia]

    (here, there, here_x, here_y,
     there_x, there_y, c) = _ordered_links(frame)
    if here.size:
        p3 = 3 * here
        q3 = 3 * there
        # Force on `here` from the link: `c (u_there - u_here)` at the seam
        # cell, with `u(x) = v + omega x r` evaluated in each piece's frame.
        entries = (
            (p3, q3, c),
            (p3, q3 + 2, -c * there_y),
            (p3, p3, -c),
            (p3, p3 + 2, c * here_y),
            (p3 + 1, q3 + 1, c),
            (p3 + 1, q3 + 2, c * there_x),
            (p3 + 1, p3 + 1, -c),
            (p3 + 1, p3 + 2, -c * here_x),
            # Torque of that force about `here`'s centroid: `r x f`.
            (p3 + 2, q3, -c * here_y),
            (p3 + 2, q3 + 1, c * here_x),
            (p3 + 2, q3 + 2, c * (here_x * there_x + here_y * there_y)),
            (p3 + 2, p3, c * here_y),
            (p3 + 2, p3 + 1, -c * here_x),
            (p3 + 2, p3 + 2, -c * (here_x * here_x + here_y * here_y)),
        )
        for row, column, value in entries:
            rows.append(row)
            columns.append(column)
            values.append(value)

    flat = np.concatenate(rows) * size + np.concatenate(columns)
    matrix = np.bincount(flat, weights=np.concatenate(values),
                         minlength=size * size).reshape(size, size)
    right = np.empty(size, dtype=np.float64)
    right[0::3] = -sum_x
    right[1::3] = -sum_y
    right[2::3] = -sum_t

    pinned = np.nonzero(pieces.pinned)[0]
    if pinned.size:
        spin = 3 * pinned + 2
        matrix[spin, :] = 0.0
        matrix[:, spin] = 0.0
        matrix[spin, spin] = 1.0
        right[spin] = 0.0
    return matrix, right


def rigid_motion(pieces: Pieces, frame: SeamFrame, drive: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Solve the balance. Returns `v (2, N)`, `omega (N,)`, and two residuals.

    `N` is the piece count; a dense `numpy.linalg.solve` on `3N x 3N` is
    milliseconds at a few hundred pieces. The residuals are the largest
    absolute force and torque left standing in the assembled system, in the
    units of the drag sum, and are what the run records per step.
    """
    matrix, right = assemble(pieces, frame, drive)
    if right.size == 0:
        zero = np.zeros((2, 0), dtype=np.float64)
        return zero, np.zeros(0, dtype=np.float64), 0.0, 0.0
    try:
        answer = np.linalg.solve(matrix, right)
    except np.linalg.LinAlgError:
        # A degenerate assembly is not worth losing a world over: the least
        # squares solution is the same answer wherever the system is
        # non-singular, and it is deterministic.
        answer = np.linalg.lstsq(matrix, right, rcond=None)[0]
    residual = matrix @ answer - right
    force = float(np.abs(residual.reshape(-1, 3)[:, :2]).max())
    torque = float(np.abs(residual[2::3]).max())
    velocity = np.stack((answer[0::3], answer[1::3]))
    return velocity, answer[2::3], force, torque


def _at(velocity: np.ndarray, omega: np.ndarray, label: np.ndarray,
        rx: np.ndarray, ry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`u_P(x) = v_P + omega_P x r`, for the piece named by `label`."""
    return (velocity[0][label] - omega[label] * ry,
            velocity[1][label] + omega[label] * rx)


def velocity_field(pieces: Pieces, frame: SeamFrame, velocity: np.ndarray,
                   omega: np.ndarray, previous_u: np.ndarray | None,
                   previous_seam: np.ndarray | None) -> np.ndarray:
    """The `(2, n, n)` velocity every view shows and markers move with.

    On an intact cell it is the piece's rigid velocity there. On a seam cell
    it is the mean of `u_P(x_s)` over the distinct pieces among its intact
    4-neighbours — the seam is between them and moves with neither. A seam
    cell with no intact neighbour at all is inside a knot of seams and has no
    piece to take a velocity from: it takes the mean of its 8-neighbourhood's
    seam cells' values from the previous step, and zero if that is empty too.
    """
    labels = pieces.labels
    n = labels.shape[-1]
    field = np.zeros((2, n, n), dtype=np.float64)
    intact = labels >= 0
    if pieces.count:
        lab = labels[intact].astype(np.int64)
        u_x, u_y = _at(velocity, omega, lab, pieces.rx[intact],
                       pieces.ry[intact])
        field[0][intact] = u_x
        field[1][intact] = u_y

    size = frame.size
    if size == 0:
        return field
    total_x = np.zeros(size, dtype=np.float64)
    total_y = np.zeros(size, dtype=np.float64)
    share = frame.first.sum(axis=0).astype(np.float64)
    if pieces.count:
        for slot in range(4):
            take = frame.first[slot]
            label = np.where(take, frame.neighbour[slot], 0)
            u_x, u_y = _at(velocity, omega, label, frame.rx[slot],
                           frame.ry[slot])
            total_x += np.where(take, u_x, 0.0)
            total_y += np.where(take, u_y, 0.0)
    touching = share > 0.0
    field[0][frame.rows[touching], frame.columns[touching]] = (
        total_x[touching] / share[touching])
    field[1][frame.rows[touching], frame.columns[touching]] = (
        total_y[touching] / share[touching])

    orphan = ~touching
    if orphan.any() and previous_u is not None and previous_seam is not None:
        was = np.asarray(previous_seam, dtype=bool)
        carried = np.asarray(previous_u, dtype=np.float64) * was
        summed = np.zeros((2, n, n), dtype=np.float64)
        counted = np.zeros((n, n), dtype=np.float64)
        for dy, dx in NEIGHBOURS_8:
            summed += np.roll(carried, (-dy, -dx), axis=(-2, -1))
            counted += np.roll(was, (-dy, -dx), axis=(-2, -1))
        rows = frame.rows[orphan]
        columns = frame.columns[orphan]
        found = counted[rows, columns] > 0.0
        if found.any():
            rows = rows[found]
            columns = columns[found]
            field[0][rows, columns] = summed[0][rows, columns] / counted[rows, columns]
            field[1][rows, columns] = summed[1][rows, columns] / counted[rows, columns]
    return field


def slip_rate(pieces: Pieces, frame: SeamFrame, velocity: np.ndarray,
              omega: np.ndarray, cell_km: float, shape: tuple[int, int]
              ) -> np.ndarray:
    """The velocity jump across each seam cell, per kilometre. `(n, n)`.

    For the pair `(P, Q)` a seam cell links, the jump is
    `|u_Q(x_s) - u_P(x_s)| / cell_km`, and the cell's slip rate is the largest
    over its pairs. A seam cell that links no pair has no jump and no slip.
    Intact cells are rigid: their strain rate is exactly zero and they never
    damage, which is what confines damage to the seam set.
    """
    field = np.zeros(shape, dtype=np.float64)
    if frame.size == 0 or pieces.count == 0:
        return field
    best = np.zeros(frame.size, dtype=np.float64)
    for index, (a, b) in enumerate(_PAIRS):
        mask = frame.pair[index]
        if not mask.any():
            continue
        label_a = np.where(mask, frame.neighbour[a], 0)
        label_b = np.where(mask, frame.neighbour[b], 0)
        ax, ay = _at(velocity, omega, label_a, frame.rx[a], frame.ry[a])
        bx, by = _at(velocity, omega, label_b, frame.rx[b], frame.ry[b])
        jump = np.hypot(bx - ax, by - ay)
        best = np.maximum(best, np.where(mask, jump, 0.0))
    field[frame.rows, frame.columns] = best / float(cell_km)
    return field


def seam_slip_rate(rigid_slip: np.ndarray, elastic_rate: np.ndarray,
                   strength: np.ndarray) -> np.ndarray:
    """The rate a seam cell's faces displace relative to each other. `(n, n)`.

    `slip_rate` above is the *rigid* part: the jump between the two pieces a
    seam cell links, which is zero for a crack that has not yet cut a piece,
    because the same piece is on both sides of it. That is the rigid
    idealization taken one step too far. A crack inside an elastic plate is
    not rigid across its faces: the faces displace relative to each other
    under the load the plate carries, which is the whole reason a crack
    concentrates stress at its tip.

    The block model already computes that rate. The internal-stress solve is
    forced by the mismatch, so its solution `w` is the non-rigid part of the
    velocity, and the sheet's own strain-rate invariant of `w` at a seam cell
    — `elastic_rate`, block-lifted exactly as the sheet lifts `strain_rate` —
    is the elastic slip rate of that cell's faces. It is in the same units as
    the rigid jump: a velocity difference over a cell is `1 / Myr` and so is a
    strain rate.

    So a crack between two pieces slips by the rigid jump **and** the elastic
    part, and a crack inside one piece slips by the elastic part alone, at
    exactly the place where the stress that drives its tip is highest. Off
    the seam set nothing is added: an intact cell is rigid, has no slip and
    never damages, which is what holds a seam one cell wide.
    """
    rigid_slip = np.asarray(rigid_slip, dtype=np.float64)
    elastic_rate = np.asarray(elastic_rate, dtype=np.float64)
    seam = np.asarray(strength, dtype=np.float64) < WEAK_THRESHOLD
    return rigid_slip + elastic_rate * seam


def mismatch(drive: np.ndarray, velocity: np.ndarray,
             labels: np.ndarray) -> np.ndarray:
    """`D - u` on intact cells, zero on seam cells. `(2, n, n)`.

    The forcing of the internal-stress solve. Over a piece with no seam
    links its net force and torque are zero to rounding, because that is the
    balance `rigid_motion` solved; over a piece that a seam links to another,
    its net force is minus the traction that seam transmits, which is the
    same balance written the other way round, and over the whole world the
    tractions cancel in pairs and the net force is zero. Either way `m`
    carries no rigid motion of its own at the scale of the world.
    """
    intact = np.asarray(labels) >= 0
    return np.asarray(drive, dtype=np.float64) * intact - (
        np.asarray(velocity, dtype=np.float64) * intact)


def net_load(labels: np.ndarray, field: np.ndarray, pieces: Pieces
             ) -> tuple[np.ndarray, np.ndarray]:
    """Net force `(2, N)` and torque `(N,)` of a cell field over each piece.

    A measurement, used by the tests and by nothing in the loop.
    """
    intact = np.asarray(labels) >= 0
    lab = np.asarray(labels)[intact].astype(np.int64)
    count = pieces.count
    field = np.asarray(field, dtype=np.float64)
    fx = np.bincount(lab, field[0][intact], minlength=count)
    fy = np.bincount(lab, field[1][intact], minlength=count)
    torque = np.bincount(
        lab,
        pieces.rx[intact] * field[1][intact]
        - pieces.ry[intact] * field[0][intact],
        minlength=count)
    return np.stack((fx, fy)), torque


def seam_traction(pieces: Pieces, frame: SeamFrame, velocity: np.ndarray,
                  omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Net seam force `(2, N)` and torque `(N,)` on each piece.

    A measurement beside `net_load`: the two together are the balance of
    §1.2, and the tests check that they cancel.
    """
    count = pieces.count
    force = np.zeros((2, count), dtype=np.float64)
    torque = np.zeros(count, dtype=np.float64)
    (here, there, here_x, here_y,
     there_x, there_y, c) = _ordered_links(frame)
    if here.size == 0:
        return force, torque
    ax, ay = _at(velocity, omega, here, here_x, here_y)
    bx, by = _at(velocity, omega, there, there_x, there_y)
    fx = c * (bx - ax)
    fy = c * (by - ay)
    force[0] = np.bincount(here, fx, minlength=count)
    force[1] = np.bincount(here, fy, minlength=count)
    torque[:] = np.bincount(here, here_x * fy - here_y * fx, minlength=count)
    return force, torque


def piece_state(strength: np.ndarray, drive: np.ndarray, kappa0: float,
                exponent: int, cell_km: float,
                previous_u: np.ndarray | None = None,
                previous_seam: np.ndarray | None = None) -> dict:
    """One step of the block model: pieces, motion, velocity, slip, mismatch.

    `strength` is the raster the markers built. A cell is intact when its
    strength is at least `WEAK_THRESHOLD` and a seam otherwise, which is what
    every view and metric in the engine already reads.
    """
    labels = label_plates(strength)
    pieces = piece_frames(labels)
    frame = seam_frame(pieces, strength, kappa0, exponent)
    velocity, omega, force_residual, torque_residual = rigid_motion(
        pieces, frame, drive)
    field = velocity_field(pieces, frame, velocity, omega, previous_u,
                           previous_seam)
    return {
        "labels": labels,
        "pieces": pieces,
        "frame": frame,
        "piece_velocity": velocity,
        "omega": omega,
        "velocity": field,
        "slip_rate": slip_rate(pieces, frame, velocity, omega, cell_km,
                               strength.shape),
        "mismatch": mismatch(drive, field, labels),
        "force_residual": force_residual,
        "torque_residual": torque_residual,
        "piece_count": pieces.count,
        "largest_piece_share": pieces.largest_share,
        "second_piece_cells": pieces.second_size,
        "wrapping_pieces": int(pieces.wrapping.sum()),
        "folded_pieces": int(pieces.folded.sum()),
    }


__all__ = [
    "Pieces",
    "SeamFrame",
    "WEAK_THRESHOLD",
    "assemble",
    "minimal_image",
    "mismatch",
    "net_load",
    "piece_frames",
    "piece_state",
    "rigid_motion",
    "seam_frame",
    "seam_slip_rate",
    "seam_traction",
    "slip_rate",
    "velocity_field",
]
