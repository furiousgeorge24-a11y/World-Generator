"""Seams carried on markers linked in order along each crack: a curve.

C04 and C04.1 carried the seam network in the strength raster and moved it
with the lithosphere by resampling. Nearest-cell resampling keeps a one-cell
line one cell wide only where the velocity is smooth; where it jumps — which
is across a seam, which is everywhere the network is — two cells can draw
their material from the same departure cell and the seam is written twice.
C04.1 measured the size of that: with advection frozen `edge_fraction` is
exactly 1.0, and with it, 0.43 to 0.62.

A marker cannot be written twice. C04.2 therefore made a seam a **set** of
markers, each with a position in cell units — a float, periodic — and a
strength `s`, and rebuilt the raster from them every step. A set of points has
holes: C04.5 §2 measured that 99.65 % of the cells through which a cut piece
rejoined a larger one were **vacated**, not healed. Two markers in one cell
translate together, cross a cell boundary, round into two different cells, and
the cell they shared holds nothing, so it is intact while the seam through it
is still slipping at twenty times the yield.

`WORK_ORDER_C04_6.md` §1 makes the seam a **curve**. The markers of a crack
are linked in order by an edge list, `a` and `b`, and the raster is drawn from
the segments between linked markers as well as from the markers themselves. A
segment cannot be vacated: wherever its two ends go, the cells between them
are drawn. A tip is the end vertex of a chain and its position is that
vertex's own position, not a mean over a cell; a crack that reaches another
chain links to it and stops there.

Markers are created by the C04 rules: a nucleus at the centre of its cell with
no edge, and a tip advance at the point the advance reached with one edge back
to the tip. They damage and heal through their own cell's slip rate, so two
markers in one cell see the same rate. A marker is removed when it heals to
`SUTURE_STRENGTH`, not at `WEAK_THRESHOLD`: between the two it is intact in
the raster and still a vertex, so the curve is remembered and reopens where
the slip returns. On removal a marker of degree 2 is replaced by one edge
between its two neighbours, so the curve does not break where a vertex leaves.
They move at the seam-cell velocity of their own cell and they wrap, and every
edge the move stretches past `SEGMENT_MAX_CELLS` is subdivided at its
midpoint.

`gap_cells` stays and is the construction's own check: it counts the cells the
network held last step, lost this step, and still surrounds, and under the
curve it has nothing to count.

Every function here is pure. `Markers` is a frozen record of five arrays; each
operation returns a new one. The edge list is undirected, carries no
duplicates and no self-edges, and every operation that removes or reorders a
marker reindexes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import (
    MEETING_RADIUS_CELLS,
    SEAM_OPEN_STRENGTH,
    SEGMENT_MAX_CELLS,
    SEGMENT_SAMPLE_CELLS,
    STRENGTH_MIN,
    SUTURE_STRENGTH,
    WEAK_THRESHOLD,
)
from .seams import (
    DIRECTIONS,
    neighbour_count,
    principal_normal,
    seam_mask,
    step_out,
    traction_on,
)


def _no_edges() -> np.ndarray:
    return np.zeros(0, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class Markers:
    """A seam curve. `x` and `y` are cell units, periodic; `s` is strength.

    `x`, `y` and `s` are one dimensional and the same length. `a` and `b` are
    the edge list: marker indices, the same length as each other, undirected,
    without duplicates and without self-edges. A marker's **degree** is its
    edge count, a **chain** is a connected component of the edge graph, a
    marker of degree 0 is a nucleus and one of degree 1 is a tip.

    A marker count of tens of thousands is one array and every operation on it
    is one pass.
    """

    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    a: np.ndarray = field(default_factory=_no_edges)
    b: np.ndarray = field(default_factory=_no_edges)

    @property
    def size(self) -> int:
        return int(self.x.size)

    @property
    def edge_count(self) -> int:
        return int(self.a.size)


def empty() -> Markers:
    """No seams: an intact sheet, which is where every run starts."""
    zero = np.zeros(0, dtype=np.float64)
    return Markers(x=zero, y=zero.copy(), s=zero.copy(),
                   a=_no_edges(), b=_no_edges())


def minimal_offset(delta: np.ndarray, n: int) -> np.ndarray:
    """The shortest signed offset on a torus of `n` cells, as `rigid` has it."""
    half = 0.5 * n
    return np.mod(np.asarray(delta, dtype=np.float64) + half, n) - half


def canonical_edges(a: np.ndarray, b: np.ndarray, size: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """The edge list with self-edges dropped and duplicates merged, sorted.

    Undirected, so `(i, j)` and `(j, i)` are one edge and the smaller index is
    written first; the pair is packed into one integer and `np.unique` does
    the rest, which fixes both the order and the invariant in one pass.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.size == 0 or size <= 0:
        return _no_edges(), _no_edges()
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    real = lo != hi
    lo = lo[real]
    hi = hi[real]
    if lo.size == 0:
        return _no_edges(), _no_edges()
    packed = np.unique(lo * np.int64(size) + hi)
    return packed // np.int64(size), packed % np.int64(size)


def degrees(markers: Markers) -> np.ndarray:
    """How many edges each marker carries."""
    if markers.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.bincount(np.concatenate((markers.a, markers.b)),
                       minlength=markers.size).astype(np.int64)


def _adjacency(markers: Markers, degree: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """Every edge in both directions, grouped by its first marker.

    Returns `(starts, others)`: `others` holds each marker's neighbours
    contiguously and `starts[i]` is where marker `i`'s block begins, so the
    one neighbour of a tip is `others[starts[i]]` and the two neighbours of a
    degree-2 marker are that and the next.
    """
    nodes = np.concatenate((markers.a, markers.b))
    others = np.concatenate((markers.b, markers.a))
    order = np.argsort(nodes, kind="stable")
    starts = np.zeros(markers.size, dtype=np.int64)
    if markers.size > 1:
        starts[1:] = np.cumsum(degree)[:-1]
    return starts, others[order]


def chain_labels(markers: Markers) -> np.ndarray:
    """The chain each marker belongs to, as the smallest index in it.

    Label propagation with pointer jumping, the scheme
    `plates.label_components` uses on a raster, written on an edge list
    instead: every marker takes the smallest label among itself and its
    neighbours, then follows the pointer its label names until the chain
    collapses, and both repeat until nothing moves. A marker of degree 0 is
    its own chain.
    """
    if markers.size == 0:
        return np.zeros(0, dtype=np.int64)
    labels = np.arange(markers.size, dtype=np.int64)
    if markers.edge_count == 0:
        return labels
    # Each edge in both directions, grouped by its first marker. The grouping
    # does not change while the labels do, so the sort is taken once and each
    # round is a gather, a segment minimum and a scatter — `np.minimum.at`
    # would redo the grouping on every round.
    nodes = np.concatenate((markers.a, markers.b))
    others = np.concatenate((markers.b, markers.a))
    order = np.argsort(nodes, kind="stable")
    nodes = nodes[order]
    others = others[order]
    heads = np.empty(nodes.size, dtype=bool)
    heads[0] = True
    heads[1:] = nodes[1:] != nodes[:-1]
    starts = np.nonzero(heads)[0]
    owner = nodes[starts]
    while True:
        nxt = labels.copy()
        nxt[owner] = np.minimum(labels[owner],
                                np.minimum.reduceat(labels[others], starts))
        while True:
            jumped = nxt[nxt]
            if np.array_equal(jumped, nxt):
                break
            nxt = jumped
        if np.array_equal(nxt, labels):
            return labels
        labels = nxt


def chain_sizes(markers: Markers) -> np.ndarray:
    """How many markers sit in each marker's own chain, at least 1."""
    labels = chain_labels(markers)
    if labels.size == 0:
        return labels
    return np.bincount(labels, minlength=markers.size)[labels]


def cells(markers: Markers, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The cell each marker sits in, as `(rows, columns)`.

    Cell centre `i` is at coordinate `i`, the convention every sampler in
    `domain.py` uses, so the cell is the nearest integer.
    """
    rows = np.mod(np.rint(markers.y).astype(np.int64), n)
    columns = np.mod(np.rint(markers.x).astype(np.int64), n)
    return rows, columns


def segment_lengths(markers: Markers, n: int) -> np.ndarray:
    """Each edge's length in cells, by the minimal image on the torus."""
    if markers.edge_count == 0:
        return np.zeros(0, dtype=np.float64)
    dx = minimal_offset(markers.x[markers.b] - markers.x[markers.a], n)
    dy = minimal_offset(markers.y[markers.b] - markers.y[markers.a], n)
    return np.hypot(dx, dy)


def samples(markers: Markers, n: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every point the raster draws, and which edge drew it.

    `WORK_ORDER_C04_6.md` §1.2. Each marker is one sample at its own position
    and strength, with an edge index of `-1`; each edge is sampled at a
    spacing of at most `SEGMENT_SAMPLE_CELLS` including both ends, along the
    minimal image, with the strength interpolated linearly between its two
    ends. Half-cell spacing is what makes the cells an edge draws an
    8-connected path: two consecutive samples differ by at most half a cell in
    each coordinate, so their nearest cells differ by at most one.

    Returns `(x, y, s, edge)`, all one dimensional and the same length.
    """
    marker_edge = np.full(markers.size, -1, dtype=np.int64)
    if markers.edge_count == 0:
        return markers.x, markers.y, markers.s, marker_edge
    a = markers.a
    b = markers.b
    dx = minimal_offset(markers.x[b] - markers.x[a], n)
    dy = minimal_offset(markers.y[b] - markers.y[a], n)
    length = np.hypot(dx, dy)
    intervals = np.maximum(
        1, np.ceil(length / SEGMENT_SAMPLE_CELLS).astype(np.int64))
    counts = intervals + 1
    total = int(counts.sum())
    edge = np.repeat(np.arange(a.size, dtype=np.int64), counts)
    starts = np.zeros(a.size, dtype=np.int64)
    if a.size > 1:
        starts[1:] = np.cumsum(counts)[:-1]
    t = (np.arange(total, dtype=np.float64) - starts[edge]) / intervals[edge]
    sx = np.mod(markers.x[a][edge] + t * dx[edge], n)
    sy = np.mod(markers.y[a][edge] + t * dy[edge], n)
    ss = markers.s[a][edge] + t * (markers.s[b][edge] - markers.s[a][edge])
    return (np.concatenate((markers.x, sx)),
            np.concatenate((markers.y, sy)),
            np.concatenate((markers.s, ss)),
            np.concatenate((marker_edge, edge)))


def sample_total(markers: Markers, n: int) -> int:
    """How many samples the raster takes, without taking them."""
    if markers.edge_count == 0:
        return markers.size
    length = segment_lengths(markers, n)
    intervals = np.maximum(
        1, np.ceil(length / SEGMENT_SAMPLE_CELLS).astype(np.int64))
    return int(markers.size + (intervals + 1).sum())


def raster(markers: Markers, n: int) -> np.ndarray:
    """The strength field the curve makes: 1 everywhere, `min(s)` per cell.

    Every marker is drawn as a point and every edge as a line, and a cell's
    value is the minimum over every sample it receives — not the last write
    and not the mean: a cell holding a fresh crack and a nearly healed one is
    as weak as the crack in it. The reduction is a lexicographic sort rather
    than a scatter with repeated indices, so the answer does not depend on how
    NumPy orders duplicate writes.
    """
    field = np.ones((n, n), dtype=np.float64)
    if markers.size == 0:
        return field
    sx, sy, ss, _edge = samples(markers, n)
    rows = np.mod(np.rint(sy).astype(np.int64), n)
    columns = np.mod(np.rint(sx).astype(np.int64), n)
    flat = rows * n + columns
    order = np.lexsort((ss, flat))
    ordered_flat = flat[order]
    ordered_s = ss[order]
    first = np.empty(ordered_flat.size, dtype=bool)
    first[0] = True
    first[1:] = ordered_flat[1:] != ordered_flat[:-1]
    field.reshape(-1)[ordered_flat[first]] = ordered_s[first]
    return field


def segments_per_cell(markers: Markers, n: int) -> np.ndarray:
    """How many distinct edges draw into each cell. Zero where none does."""
    counts = np.zeros((n, n), dtype=np.int64)
    if markers.edge_count == 0:
        return counts
    sx, sy, _ss, edge = samples(markers, n)
    keep = edge >= 0
    rows = np.mod(np.rint(sy[keep]).astype(np.int64), n)
    columns = np.mod(np.rint(sx[keep]).astype(np.int64), n)
    flat = rows * n + columns
    width = np.int64(markers.edge_count)
    pairs = np.unique(flat * width + edge[keep])
    counts.reshape(-1)[:] = np.bincount(pairs // width, minlength=n * n)
    return counts


def drawn_cells(markers: Markers, n: int) -> np.ndarray:
    """Cells the raster draws into at all, whatever strength it gives them."""
    mask = np.zeros((n, n), dtype=bool)
    if markers.size == 0:
        return mask
    sx, sy, _ss, _edge = samples(markers, n)
    mask[np.mod(np.rint(sy).astype(np.int64), n),
         np.mod(np.rint(sx).astype(np.int64), n)] = True
    return mask


def cell_offsets(markers: Markers, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Where in its cell the markers of each cell sit, and how many there are.

    Returns `(offsets, holds)`: `offsets` is `(2, n, n)`, the mean position of
    the markers a cell holds measured from that cell's centre, in cell units
    and in `(x, y)` order; `holds` is `(n, n)`, how many markers the cell
    holds. A cell holding none has an offset of `NaN` and a hold of 0.

    The mean is taken on the offsets and not on the positions, because a cell
    at the edge of the grid can hold a marker at `n - 0.4` and one at `0.4`:
    the two are a fifth of a cell apart on the torus and their positions
    average to the far side of the world.

    Since `WORK_ORDER_C04_6.md` §1.3 a tip's position is its own vertex's and
    not this mean, so the loop no longer reads it; it stays because it is the
    honest answer to where a cell's markers are and the C04.5 rule's unit
    tests are written on it.
    """
    offsets = np.full((2, n, n), np.nan, dtype=np.float64)
    holds = np.zeros((n, n), dtype=np.int64)
    if markers.size == 0:
        return offsets, holds
    rows, columns = cells(markers, n)
    flat = rows * n + columns
    counts = np.bincount(flat, minlength=n * n)
    dx = minimal_offset(markers.x - columns, n)
    dy = minimal_offset(markers.y - rows, n)
    sum_x = np.bincount(flat, weights=dx, minlength=n * n)
    sum_y = np.bincount(flat, weights=dy, minlength=n * n)
    held = counts > 0
    mean_x = np.where(held, sum_x / np.maximum(counts, 1), np.nan)
    mean_y = np.where(held, sum_y / np.maximum(counts, 1), np.nan)
    offsets[0] = mean_x.reshape(n, n)
    offsets[1] = mean_y.reshape(n, n)
    holds = counts.reshape(n, n)
    return offsets, holds


def create(markers: Markers, opened: np.ndarray) -> Markers:
    """One marker of degree 0 at the centre of every cell `opened` marks.

    `opened` is the set of cells a nucleation just cracked. A nucleus is a
    crack with no direction yet, so it carries no edge and is a tip on the
    next pass, which is what it has always been. Existing edges are untouched:
    markers are appended, so no index moves.
    """
    opened = np.asarray(opened, dtype=bool)
    if not opened.any():
        return markers
    rows, columns = np.nonzero(opened)
    fresh = np.full(rows.size, float(SEAM_OPEN_STRENGTH), dtype=np.float64)
    return Markers(
        x=np.concatenate((markers.x, columns.astype(np.float64))),
        y=np.concatenate((markers.y, rows.astype(np.float64))),
        s=np.concatenate((markers.s, fresh)),
        a=markers.a, b=markers.b)


def add(markers: Markers, x: np.ndarray, y: np.ndarray, s: np.ndarray,
        link_to: np.ndarray | None = None) -> Markers:
    """Append markers, each with one edge to the marker `link_to` names.

    `link_to` is one index per new marker, or `-1` for a nucleus with no edge.
    New markers take the indices after the last existing one, in the order
    given, so the caller fixes the order and nothing here can reorder it.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return markers
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    base = markers.size
    fresh = base + np.arange(x.size, dtype=np.int64)
    a = markers.a
    b = markers.b
    if link_to is not None:
        link_to = np.asarray(link_to, dtype=np.int64)
        linked = link_to >= 0
        a = np.concatenate((a, link_to[linked]))
        b = np.concatenate((b, fresh[linked]))
    a, b = canonical_edges(a, b, base + x.size)
    return Markers(x=np.concatenate((markers.x, x)),
                   y=np.concatenate((markers.y, y)),
                   s=np.concatenate((markers.s, s)), a=a, b=b)


def link(markers: Markers, a: np.ndarray, b: np.ndarray) -> Markers:
    """The same markers with these edges added, canonically."""
    joined_a = np.concatenate((markers.a, np.asarray(a, dtype=np.int64)))
    joined_b = np.concatenate((markers.b, np.asarray(b, dtype=np.int64)))
    new_a, new_b = canonical_edges(joined_a, joined_b, markers.size)
    return Markers(x=markers.x, y=markers.y, s=markers.s, a=new_a, b=new_b)


def remove(markers: Markers, keep: np.ndarray) -> Markers:
    """Drop the markers `keep` is false at and reindex the edges.

    `WORK_ORDER_C04_6.md` §1.4. A marker of degree 2 is replaced by one edge
    between its two neighbours, so the curve does not break where a vertex
    leaves; a marker of any other degree just drops its edges. The rule is
    applied to the whole removal set at once, so where two neighbouring
    degree-2 markers go in the same step each one's replacement edge reaches
    the other and is dropped with it, and the chain parts there rather than
    being spliced across both. Every surviving edge is then renumbered by the
    running count of survivors, which is the only reindexing there is.
    """
    keep = np.asarray(keep, dtype=bool)
    if bool(keep.all()):
        return markers
    a = markers.a
    b = markers.b
    if a.size:
        degree = degrees(markers)
        bridged = np.nonzero((~keep) & (degree == 2))[0]
        if bridged.size:
            starts, others = _adjacency(markers, degree)
            a = np.concatenate((a, others[starts[bridged]]))
            b = np.concatenate((b, others[starts[bridged] + 1]))
        alive = keep[a] & keep[b]
        index = np.cumsum(keep) - 1
        a, b = canonical_edges(index[a[alive]], index[b[alive]],
                               int(keep.sum()))
    return Markers(x=markers.x[keep], y=markers.y[keep], s=markers.s[keep],
                   a=a, b=b)


def healed_strength(markers: Markers, excess: np.ndarray, heal_rate: float,
                    damage_rate: float, step_myr: float,
                    n: int) -> np.ndarray:
    """The strength every marker reaches over one step. Pure, and no removal.

    The integrator is the sheet's, exact over the step: per marker the law is
    linear in `s`, so it is stable at any step length and reduces to the
    explicit update as `dt -> 0`. Two markers in one cell see the same excess,
    because the excess is a field on cells and slip is a property of the cell.
    """
    rows, columns = cells(markers, n)
    here = np.asarray(excess, dtype=np.float64)[rows, columns]
    rate = float(damage_rate) * here * here
    total = float(heal_rate) + rate
    equilibrium = float(heal_rate) / total
    return np.clip(
        equilibrium + (markers.s - equilibrium) * np.exp(-total * float(step_myr)),
        STRENGTH_MIN, 1.0)


def damage_and_heal(markers: Markers, excess: np.ndarray, heal_rate: float,
                    damage_rate: float, step_myr: float, n: int
                    ) -> tuple[Markers, int, int]:
    """Damage and heal every marker, and drop the ones that have sutured.

    `healed_strength` is the law and it has not moved. What moved with
    `WORK_ORDER_C04_6.md` §1.4 is where a marker leaves: at
    `SUTURE_STRENGTH`, not at `WEAK_THRESHOLD`. Between the two the marker is
    intact in the raster and still a vertex of the curve, so the crack is
    remembered and reopens where the slip returns.

    Returns the survivors, how many were removed, and how many **reactivated**
    — crossed back from at or above `WEAK_THRESHOLD` to below it in this step,
    which is a remembered curve reopening rather than a new crack.
    """
    if markers.size == 0:
        return markers, 0, 0
    strength = healed_strength(markers, excess, heal_rate, damage_rate,
                               step_myr, n)
    keep = strength < SUTURE_STRENGTH
    removed = int(strength.size - keep.sum())
    reactivated = int(((markers.s >= WEAK_THRESHOLD)
                       & (strength < WEAK_THRESHOLD)).sum())
    healed = Markers(x=markers.x, y=markers.y, s=strength,
                     a=markers.a, b=markers.b)
    return remove(healed, keep), removed, reactivated


def move(markers: Markers, velocity: np.ndarray, step_myr: float,
         cell_km: float, n: int) -> Markers:
    """Carry every marker at its own cell's velocity, and wrap.

    `velocity` is the `(2, n, n)` field of `rigid.velocity_field`, in km/Myr;
    a marker's cell is a seam cell, so the velocity it takes is the mean of
    the pieces on either side of it. The displacement is in cells and it is
    not rounded: a marker holds its own sub-cell position, so there is no
    remainder to carry and no jitter to bound. Nothing here touches the edge
    list: the curve moves, it is not resampled.
    """
    if markers.size == 0:
        return markers
    rows, columns = cells(markers, n)
    scale = float(step_myr) / float(cell_km)
    velocity = np.asarray(velocity, dtype=np.float64)
    return Markers(
        x=np.mod(markers.x + velocity[0][rows, columns] * scale, n),
        y=np.mod(markers.y + velocity[1][rows, columns] * scale, n),
        s=markers.s, a=markers.a, b=markers.b)


def subdivide(markers: Markers, n: int) -> tuple[Markers, int]:
    """Split every edge longer than `SEGMENT_MAX_CELLS` at its midpoint.

    `WORK_ORDER_C04_6.md` §1.4, run after the move, which is what stretches an
    edge: the two ends belong to two cells and take two velocities. The new
    marker sits at the minimal-image midpoint with the mean of the two ends'
    strengths, and the edge becomes two. One pass, so an edge more than twice
    the bound long is still over it afterwards; the run counts how many
    segments that leaves.
    """
    if markers.edge_count == 0:
        return markers, 0
    a = markers.a
    b = markers.b
    dx = minimal_offset(markers.x[b] - markers.x[a], n)
    dy = minimal_offset(markers.y[b] - markers.y[a], n)
    long_edge = np.hypot(dx, dy) > SEGMENT_MAX_CELLS
    count = int(long_edge.sum())
    if count == 0:
        return markers, 0
    mid_x = np.mod(markers.x[a[long_edge]] + 0.5 * dx[long_edge], n)
    mid_y = np.mod(markers.y[a[long_edge]] + 0.5 * dy[long_edge], n)
    mid_s = 0.5 * (markers.s[a[long_edge]] + markers.s[b[long_edge]])
    fresh = markers.size + np.arange(count, dtype=np.int64)
    keep = ~long_edge
    new_a, new_b = canonical_edges(
        np.concatenate((a[keep], a[long_edge], fresh)),
        np.concatenate((b[keep], fresh, b[long_edge])),
        markers.size + count)
    return Markers(x=np.concatenate((markers.x, mid_x)),
                   y=np.concatenate((markers.y, mid_y)),
                   s=np.concatenate((markers.s, mid_s)),
                   a=new_a, b=new_b), count


def _cell_table(markers: Markers, n: int) -> np.ndarray:
    """Every cell's markers as an `(n * n, k)` table of indices, `-1` for none.

    `k` is the most markers any one cell holds. The table is what makes the
    meeting test of §1.3 a fixed number of vectorized comparisons rather than
    a search: a marker within `MEETING_RADIUS_CELLS` of a point that sits
    within half a cell of its own centre lies in one of the 25 cells whose
    offsets are at most two, so the test is 25 by `k` comparisons.
    """
    if markers.size == 0:
        return np.full((n * n, 1), -1, dtype=np.int64)
    rows, columns = cells(markers, n)
    flat = rows * n + columns
    counts = np.bincount(flat, minlength=n * n)
    table = np.full((n * n, max(int(counts.max()), 1)), -1, dtype=np.int64)
    order = np.argsort(flat, kind="stable")
    ordered = flat[order]
    starts = np.zeros(n * n, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    rank = np.arange(ordered.size, dtype=np.int64) - starts[ordered]
    table[ordered, rank] = order
    return table


def advance_tips(markers: Markers, strength: np.ndarray, sxx: np.ndarray,
                 syy: np.ndarray, sxy: np.ndarray, sigma_c_field: np.ndarray,
                 toughness_fraction: float = 1.0
                 ) -> tuple[Markers, int, int, int, int]:
    """One pass over every tip of the curve. The `seams = 2` rule.

    `WORK_ORDER_C04_6.md` §1.3, which keeps C04.5's direction rule and moves
    the tip from the raster onto the curve.

    - **The tip.** A marker of degree at most 1. Its position `p` is its own,
      so nothing averages two markers into one starting point, and the
      raster's `tips` is not read at all.
    - **The direction.** The lifted stress tensor averaged with equal weight
      over the tip cell's **intact** 8-neighbours, `n` the eigenvector of its
      largest absolute eigenvalue, and the seam running along `n` turned a
      quarter turn. The sign is the one pointing away from the tip's one
      linked marker, by the minimal-image vector from that marker to the tip;
      a nucleus of degree 0 tries both signs and keeps the one whose candidate
      carries the larger traction, ties taking positive `x` then positive `y`.
    - **The length.** `L` in the Griffith threshold is the number of markers
      in the tip's own chain, at least 1.
    - **The advance.** `seams.step_out`'s walk from `p` along the direction to
      the first cell whose nearest cell differs from the tip's. That cell must
      be **intact in the raster** — a cell a segment covers at or above
      `WEAK_THRESHOLD` is intact and a candidate like any other — and its
      traction must reach `toughness_fraction * sigma_c_field / sqrt(L)`. The
      advance creates a marker at the point reached with one edge to the tip.
    - **The meeting.** If a marker of a **different** chain lies within
      `MEETING_RADIUS_CELLS` of the new marker, the nearest such marker gains
      an edge to it: the crack has met another, the new marker has degree 2,
      and this crack stops here. Ties take the lower marker index. The
      candidates are the markers on the record when the pass began, so two
      tips that advance towards each other in one pass meet on the next.

    Returns the markers, the tips, the advances, the meetings, and how many
    tips read a tensor whose two absolute eigenvalues were within one per cent
    of each other — for those the direction is whatever the eigensolver
    returns, deterministically.
    """
    n = strength.shape[-1]
    if markers.size == 0:
        return markers, 0, 0, 0, 0
    toughness = float(toughness_fraction)
    intact = ~seam_mask(strength)

    degree = degrees(markers)
    tip = np.nonzero(degree <= 1)[0]
    count = int(tip.size)
    if count == 0:
        return markers, 0, 0, 0, 0

    label = chain_labels(markers)
    size = np.bincount(label, minlength=markers.size)[label]
    root_length = np.sqrt(np.maximum(size[tip], 1).astype(np.float64))

    px = markers.x[tip]
    py = markers.y[tip]
    tx = np.mod(np.rint(px).astype(np.int64), n)
    ty = np.mod(np.rint(py).astype(np.int64), n)
    offset_x = minimal_offset(px - tx, n)
    offset_y = minimal_offset(py - ty, n)

    # The tensor the direction is read from: the mean over the intact
    # 8-neighbours, which are the cells the eight-direction rule scored. A
    # tip whose eight neighbours are all seams has nowhere to go.
    total_xx = np.zeros(count, dtype=np.float64)
    total_yy = np.zeros(count, dtype=np.float64)
    total_xy = np.zeros(count, dtype=np.float64)
    neighbours = np.zeros(count, dtype=np.float64)
    for dy, dx in DIRECTIONS:
        cy = (ty + dy) % n
        cx = (tx + dx) % n
        weight = intact[cy, cx].astype(np.float64)
        total_xx += weight * sxx[cy, cx]
        total_yy += weight * syy[cy, cx]
        total_xy += weight * sxy[cy, cx]
        neighbours += weight
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
    # Away from the crack: the minimal-image vector from the tip's one linked
    # marker to the tip itself.
    linked = degree[tip] == 1
    away = np.zeros(count, dtype=np.float64)
    if markers.edge_count:
        starts, others = _adjacency(markers, degree)
        # A tip of degree 0 has no block of its own, and its `starts` entry is
        # the next marker's; the read is clipped and then thrown away by the
        # `linked` test rather than guarded with a branch.
        here = np.minimum(starts[tip], others.size - 1)
        neighbour = np.where(linked, others[here], tip)
        away = (ex * minimal_offset(px - markers.x[neighbour], n)
                + ey * minimal_offset(py - markers.y[neighbour], n))
    # The sign a nucleus takes when nothing else decides: positive `x`, then
    # positive `y`.
    default = np.where((ex > 0.0) | ((ex == 0.0) & (ey > 0.0)), 1.0, -1.0)

    trials = []
    for way in (1.0, -1.0):
        dx, dy, wx, wy, inside = step_out(offset_x, offset_y,
                                          way * ex, way * ey)
        cy = (ty + dy.astype(np.int64)) % n
        cx = (tx + dx.astype(np.int64)) % n
        pull = traction_on(sxx[cy, cx], syy[cy, cx], sxy[cy, cx], nx, ny)
        usable = inside & live
        trials.append({
            "dx": dx, "dy": dy, "wx": wx, "wy": wy,
            "traction": np.where(usable, pull, -np.inf),
            "qualifies": usable & intact[cy, cx] & (
                pull >= (toughness * sigma_c_field[cy, cx]) / root_length),
        })

    positive, negative = trials
    by_traction = np.where(positive["traction"] > negative["traction"], 1.0,
                           np.where(positive["traction"] < negative["traction"],
                                    -1.0, default))
    sign = np.where(linked & (away > 0.0), 1.0,
                    np.where(linked & (away < 0.0), -1.0, by_traction))
    take = sign > 0.0
    advanced = np.where(take, positive["qualifies"], negative["qualifies"])
    if not advanced.any():
        return markers, count, 0, 0, degenerate

    dx = np.where(take, positive["dx"], negative["dx"])[advanced]
    dy = np.where(take, positive["dy"], negative["dy"])[advanced]
    wx = np.where(take, positive["wx"], negative["wx"])[advanced]
    wy = np.where(take, positive["wy"], negative["wy"])[advanced]
    parent = tip[advanced]
    new_x = np.mod(tx[advanced] + dx + wx, n)
    new_y = np.mod(ty[advanced] + dy + wy, n)
    fresh = np.full(parent.size, float(SEAM_OPEN_STRENGTH), dtype=np.float64)

    # The meeting test, against the markers on the record when the pass began.
    table = _cell_table(markers, n)
    best = np.full(parent.size, np.inf, dtype=np.float64)
    partner = np.full(parent.size, -1, dtype=np.int64)
    own = label[parent]
    ncy = np.mod(np.rint(new_y).astype(np.int64), n)
    ncx = np.mod(np.rint(new_x).astype(np.int64), n)
    for oy in (-2, -1, 0, 1, 2):
        for ox in (-2, -1, 0, 1, 2):
            flat = ((ncy + oy) % n) * n + ((ncx + ox) % n)
            for slot in range(table.shape[1]):
                candidate = table[flat, slot]
                here = candidate >= 0
                if not here.any():
                    continue
                safe = np.where(here, candidate, 0)
                gap = np.hypot(
                    minimal_offset(new_x - markers.x[safe], n),
                    minimal_offset(new_y - markers.y[safe], n))
                closer = (gap < best) | ((gap == best) & (safe < partner))
                ok = (here & (label[safe] != own)
                      & (gap <= MEETING_RADIUS_CELLS) & closer)
                best = np.where(ok, gap, best)
                partner = np.where(ok, safe, partner)

    grown = add(markers, new_x, new_y, fresh, parent)
    met = partner >= 0
    meetings = int(met.sum())
    if meetings:
        made = markers.size + np.arange(parent.size, dtype=np.int64)
        grown = link(grown, partner[met], made[met])
    return grown, count, int(parent.size), meetings, degenerate


def opened_cells(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Cells that were intact before a pass and are seams after it."""
    return seam_mask(after) & ~seam_mask(before)


def gap_cells(previous_seam: np.ndarray, current_seam: np.ndarray,
              minimum: int = 2) -> np.ndarray:
    """Cells the network held last step, lost this step, and still surrounds.

    A gap is what marker motion leaves behind when a seam is a set of points:
    two markers land in one cell and the cell one of them came from goes
    intact. Under the curve of `WORK_ORDER_C04_6.md` the cells between two
    linked markers are drawn wherever the two ends go, so there is nothing
    here to count, and that is the construction's own check.
    """
    previous_seam = np.asarray(previous_seam, dtype=bool)
    current_seam = np.asarray(current_seam, dtype=bool)
    lost = previous_seam & ~current_seam
    return lost & (neighbour_count(current_seam) >= int(minimum))


__all__ = [
    "Markers",
    "add",
    "advance_tips",
    "canonical_edges",
    "cell_offsets",
    "cells",
    "chain_labels",
    "chain_sizes",
    "create",
    "damage_and_heal",
    "degrees",
    "drawn_cells",
    "empty",
    "gap_cells",
    "healed_strength",
    "link",
    "minimal_offset",
    "move",
    "opened_cells",
    "raster",
    "remove",
    "sample_total",
    "samples",
    "segment_lengths",
    "segments_per_cell",
    "subdivide",
]
