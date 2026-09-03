"""Plates read off the strength field, and the boundaries between them.

Nothing here places, counts, or shapes a plate. A plate is what a connected
region of strong lithosphere looks like once strain has localized somewhere
else, and it is an input to nothing.
"""

from __future__ import annotations

import numpy as np

from .constants import REGIME_RATIO, WEAK_THRESHOLD

REGIME_NONE = -1
REGIME_SHEAR = 0
REGIME_DIVERGENT = 1
REGIME_CONVERGENT = 2


def weak_mask(strength: np.ndarray) -> np.ndarray:
    """Cells whose lithosphere has failed."""
    return np.asarray(strength) < WEAK_THRESHOLD


#: Neighbour offsets as `(dy, dx)`, in the order the propagation applies
#: them. Four-connectivity is the cardinal set; eight adds the diagonals.
#: The order does not change the result — the fixed point of "take the
#: smallest label among yourself and your neighbours" is the smallest flat
#: index in the component however the rolls are ordered — but it is fixed so
#: the arithmetic is too.
NEIGHBOURS_4 = ((0, -1), (0, 1), (-1, 0), (1, 0))
NEIGHBOURS_8 = NEIGHBOURS_4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def label_components(mask: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """Connected components of a boolean mask on the torus, largest first.

    Label propagation, not a graph search: every cell takes the smallest label
    among itself and its neighbours inside the mask, then follows the pointer
    its label names until the chain collapses, and both repeat until nothing
    changes. Components are renumbered by area, largest first. Cells outside
    the mask are -1.

    A label is the flat index of a cell, so `labels[labels]` reads "the label
    of the cell my label names". Neighbour minima alone move a label one cell
    per round, which costs rounds proportional to the component's diameter;
    the pointer jump halves every chain it walks, so the pair converges in
    logarithmic rounds instead.

    `connectivity` is 4 (cardinal neighbours only) or 8 (the four diagonals as
    well). Eight-connectivity joins two cells that touch at a corner; four
    does not.
    """
    if connectivity == 4:
        offsets = NEIGHBOURS_4
    elif connectivity == 8:
        offsets = NEIGHBOURS_8
    else:
        raise ValueError("connectivity must be 4 or 8")
    inside = np.asarray(mask, dtype=bool)
    if inside.ndim != 2 or inside.shape[0] != inside.shape[1]:
        raise ValueError("mask must be a square 2-D array")
    n = inside.shape[0]
    labels = np.arange(n * n, dtype=np.int64).reshape(n, n)
    sentinel = n * n
    labels = np.where(inside, labels, sentinel)
    pointers = np.empty(n * n + 1, dtype=np.int64)
    pointers[sentinel] = sentinel

    while True:
        previous = labels
        for offset in offsets:
            rolled = np.roll(labels, offset, axis=(-2, -1))
            both = inside & np.roll(inside, offset, axis=(-2, -1))
            labels = np.where(both, np.minimum(labels, rolled), labels)
        while True:
            pointers[:sentinel] = labels.reshape(-1)
            jumped = pointers[labels]
            if np.array_equal(jumped, labels):
                break
            labels = jumped
        if np.array_equal(labels, previous):
            break

    result = np.full((n, n), -1, dtype=np.int32)
    if not inside.any():
        return result
    roots, inverse, counts = np.unique(
        labels[inside], return_inverse=True, return_counts=True)
    # Largest first; ties broken by the root index, which is deterministic.
    order = np.lexsort((roots, -counts))
    rank = np.empty(order.size, dtype=np.int32)
    rank[order] = np.arange(order.size, dtype=np.int32)
    result[inside] = rank[inverse]
    return result


def label_plates(strength: np.ndarray) -> np.ndarray:
    """Connected components of the strong cells, four-connected on the torus.

    A plate is what a connected region of strong lithosphere looks like once
    strain has localized somewhere else. `label_components` does the work;
    weak cells are -1.
    """
    return label_components(~weak_mask(strength), 4)


def boundary_mask(labels: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Weak cells, plus strong cells that touch a different plate."""
    boundary = np.asarray(weak).copy()
    for shift, axis in ((-1, -1), (1, -1), (-1, -2), (1, -2)):
        boundary |= labels != np.roll(labels, shift, axis=axis)
    return boundary


def regime(divergence: np.ndarray, strain_rate: np.ndarray,
           weak: np.ndarray) -> np.ndarray:
    """Divergent, convergent, or shear, decided cell by cell on weak ground."""
    ratio = np.asarray(divergence) / np.maximum(np.asarray(strain_rate), 1e-12)
    result = np.full(ratio.shape, REGIME_SHEAR, dtype=np.int8)
    result[ratio > REGIME_RATIO] = REGIME_DIVERGENT
    result[ratio < -REGIME_RATIO] = REGIME_CONVERGENT
    result[~weak] = REGIME_NONE
    return result


def plate_areas(labels: np.ndarray) -> np.ndarray:
    """Cell counts by label, descending."""
    strong = labels >= 0
    if not strong.any():
        return np.zeros(0, dtype=np.int64)
    return np.bincount(labels[strong].ravel())


__all__ = [
    "NEIGHBOURS_4",
    "NEIGHBOURS_8",
    "REGIME_CONVERGENT",
    "REGIME_DIVERGENT",
    "REGIME_NONE",
    "REGIME_SHEAR",
    "boundary_mask",
    "label_components",
    "label_plates",
    "plate_areas",
    "regime",
    "weak_mask",
]
