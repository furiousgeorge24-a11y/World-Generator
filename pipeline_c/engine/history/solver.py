"""Geometric multigrid for the velocity of a viscous sheet with weak zones.

For each velocity component `u`,

    u - div( kappa * grad(u) ) = D

on the periodic grid, with `kappa` growing steeply with lithosphere strength.
Strong lithosphere homogenizes velocity over a length `sqrt(kappa)`; weak
zones let it jump. Both components are solved together with the same
coefficients, so the operator has no preferred axis beyond the square grid's
own five-point stencil.

The solver is a conjugate gradient preconditioned by one V-cycle: red-black
Gauss-Seidel on every grid, a 2 x 2 mean down and a piecewise-constant lift
back, edge coefficients rediscretized level by level, and an exact solve on
the coarsest grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..domain import roll_x, roll_y
from .constants import (
    HOMOG_LENGTH_FRACTION,
    MG_COARSEST,
    MG_MAX_CYCLES,
    MG_POST,
    MG_PRE,
    MG_TOL,
)

HARMONIC_FLOOR = 1e-30


def kappa0_for(history_n: int,
               stiffness_fraction: float = HOMOG_LENGTH_FRACTION) -> float:
    """Diffusivity of unit-strength lithosphere, in cell² units.

    `stiffness_fraction` is the homogenization length as a fraction of the
    parent: the distance over which strong lithosphere carries velocity
    without deforming internally. `HOMOG_LENGTH_FRACTION` is its default and
    the only value production uses.
    """
    return (stiffness_fraction * history_n) ** 2


def edge_coefficients(kappa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic-mean coefficients on the east and north edges of each cell.

    Harmonic, not arithmetic, so one thin weak line stays a barrier instead of
    being averaged away by its strong neighbours.
    """
    east_neighbour = roll_x(kappa, -1)
    north_neighbour = roll_y(kappa, -1)
    k_east = 2.0 * kappa * east_neighbour / (kappa + east_neighbour + HARMONIC_FLOOR)
    k_north = 2.0 * kappa * north_neighbour / (kappa + north_neighbour + HARMONIC_FLOOR)
    return k_east, k_north


def effective_gradients(u: np.ndarray, kappa: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell x and y gradients of `u` consistent with the edge fluxes.

    The flux across an edge is `k_ij * (u_j - u_i)` with `k_ij` the harmonic
    mean of the two cells' kappa, and it is continuous across a stiffness
    contrast. Dividing the mean of a cell's two opposing edge fluxes by the
    cell's own kappa gives the gradient the cell actually experiences. Where
    kappa is uniform this is the ordinary central difference. Where a strong
    cell touches a weak one, the harmonic mean is dominated by the weak side
    and the strong cell's share of the jump is nearly zero, which is what
    stress continuity requires.

    `u` may carry a leading component axis; `kappa` broadcasts over it. The
    result is in per-cell units of the grid `u` lives on.
    """
    k_east, k_north = edge_coefficients(kappa)
    k_west = roll_x(k_east, 1)
    k_south = roll_y(k_north, 1)
    guarded = kappa + HARMONIC_FLOOR
    g_x = 0.5 * (k_east * (roll_x(u, -1) - u)
                 + k_west * (u - roll_x(u, 1))) / guarded
    g_y = 0.5 * (k_north * (roll_y(u, -1) - u)
                 + k_south * (u - roll_y(u, 1))) / guarded
    return g_x, g_y


def diagonal(kappa: np.ndarray) -> np.ndarray:
    """`1 + sum_j k_ij`, the diagonal of the discrete operator."""
    k_east, k_north = edge_coefficients(kappa)
    return _diagonal_from_edges(k_east, k_north)


def _diagonal_from_edges(k_east: np.ndarray, k_north: np.ndarray) -> np.ndarray:
    return (1.0 + k_east + roll_x(k_east, 1) + k_north + roll_y(k_north, 1))


def _apply_from_edges(u: np.ndarray, k_east: np.ndarray, k_north: np.ndarray,
                      diag: np.ndarray) -> np.ndarray:
    k_west = roll_x(k_east, 1)
    k_south = roll_y(k_north, 1)
    return (
        diag * u
        - k_east * roll_x(u, -1)
        - k_west * roll_x(u, 1)
        - k_north * roll_y(u, -1)
        - k_south * roll_y(u, 1)
    )


def apply_A(u: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """`u - div(kappa grad u)` on the periodic grid."""
    k_east, k_north = edge_coefficients(kappa)
    return _apply_from_edges(u, k_east, k_north,
                             _diagonal_from_edges(k_east, k_north))


def restrict(a: np.ndarray) -> np.ndarray:
    """Mean over each 2 × 2 block."""
    coarse = a.shape[-1] // 2
    blocks = a.reshape(a.shape[:-2] + (coarse, 2, coarse, 2))
    return blocks.mean(axis=(-3, -1))


def restrict_kappa(kappa: np.ndarray) -> np.ndarray:
    """Harmonic mean over each 2 × 2 block, then a quarter for the wider cell.

    This is the *grid* coarsening: it rediscretizes the same physical
    coefficient on a grid of twice the cell size, and it is what carries the
    kinematic grid's strength down to the coarser grid the velocity is solved
    on. The multigrid hierarchy no longer uses it; see `coarsen_edges`.
    """
    coarse = kappa.shape[-1] // 2
    blocks = (1.0 / (kappa + HARMONIC_FLOOR)).reshape(
        kappa.shape[:-2] + (coarse, 2, coarse, 2))
    return 1.0 / blocks.sum(axis=(-3, -1))


def prolong(e: np.ndarray) -> np.ndarray:
    """Piecewise-constant lift of a coarse correction onto the fine grid.

    This is the exact adjoint of `restrict` up to the factor four, which makes
    one V-cycle a symmetric operator and therefore usable as a conjugate
    gradient preconditioner. Bilinear interpolation is smoother but is not
    adjoint to a 2 x 2 mean, and across a strength contrast of `STRENGTH_MIN`
    to the fourth power it interpolates straight through the barrier it is
    meant to respect.
    """
    return np.repeat(np.repeat(e, 2, axis=-1), 2, axis=-2)


def coarsen_edges(k_east: np.ndarray, k_north: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Edge coefficients one grid coarser: the operator rediscretized there.

    A coarse edge is crossed by exactly two fine edges. Its coefficient is
    their mean, halved twice more for the doubled cell size, so a coarse
    coefficient is a quarter of a uniform fine one, which is the same
    per-level quarter `restrict_kappa` gives and the scale the operator has on
    a grid of twice the cell size. The identity term needs no coarsening: the
    mean of four ones is one, so `_diagonal_from_edges` still gives the coarse
    diagonal, and the coarse operator stays five-point.

    Twice this is the Galerkin operator `prolong.T A prolong / 4` for this
    transfer pair, exactly; piecewise-constant interpolation inflates a smooth
    coarse function's energy by that factor, so the Galerkin coarse operator is
    twice as stiff as the grid it sits on and its correction falls short.
    Measured both ways in `out/C03_6_BUILD_REPORT.md`.

    The hierarchy used to coarsen `kappa` per cell with `restrict_kappa`
    instead of coarsening the edges. That gave a 2 x 2 block the stiffness of
    its weakest cell even when the weak cell sat in the block's interior and
    no coarse edge crossed it, which understates a mostly-strong block by
    orders of magnitude; the coarse correction then overshoots and the cycle
    stops being a contraction. Coarsening the edges asks only about the two
    fine edges that the coarse edge is actually made of.
    """
    east = 0.125 * (k_east[..., 0::2, 1::2] + k_east[..., 1::2, 1::2])
    north = 0.125 * (k_north[..., 1::2, 0::2] + k_north[..., 1::2, 1::2])
    return east, north


@lru_cache(maxsize=8)
def _colours(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The red and black halves of an `n x n` checkerboard, as 0/1 floats.

    Cached by size: they depend on nothing else, every level of every solve of
    that size wants the same pair, and they are never written to.
    """
    rows, columns = np.indices((n, n))
    red = ((rows + columns) % 2 == 0).astype(np.float64)
    return red, 1.0 - red


@dataclass(slots=True)
class Level:
    """One grid of the hierarchy: its edge coefficients and its diagonal.

    Not frozen, because the coarsest level caches the inverse of its own dense
    operator the first time it is asked to solve. Levels are rebuilt whenever
    `kappa` changes, which is every history step, so that cache lives exactly
    as long as the coefficients it belongs to.
    """

    n: int
    k_east: np.ndarray
    k_north: np.ndarray
    diag: np.ndarray
    _inverse: np.ndarray | None = field(default=None, repr=False, compare=False)

    def apply(self, u: np.ndarray) -> np.ndarray:
        return _apply_from_edges(u, self.k_east, self.k_north, self.diag)

    def smooth(self, u: np.ndarray, rhs: np.ndarray, sweeps: int,
               *, reverse: bool = False) -> np.ndarray:
        """Red-black Gauss-Seidel sweeps.

        Within one colour no two cells are five-point neighbours, so a colour's
        update is the exact Gauss-Seidel step for those cells given the other
        colour. `reverse` runs black before red: smoothing forward on the way
        down and reversed on the way up is what keeps one cycle a symmetric
        operator, which is what lets it precondition a conjugate gradient.
        """
        first, second = _colours(self.n)
        if reverse:
            first, second = second, first
        for _ in range(sweeps):
            u = u + first * (rhs - self.apply(u)) / self.diag
            u = u + second * (rhs - self.apply(u)) / self.diag
        return u

    def dense(self) -> np.ndarray:
        """The `n**2 x n**2` matrix of this level's periodic five-point operator."""
        m = self.n * self.n
        matrix = np.zeros((m, m), dtype=np.float64)
        index = np.arange(m).reshape(self.n, self.n)
        rows = index.ravel()
        matrix[rows, rows] = self.diag.ravel()
        east = np.roll(index, -1, axis=1).ravel()
        north = np.roll(index, -1, axis=0).ravel()
        np.add.at(matrix, (rows, east), -self.k_east.ravel())
        np.add.at(matrix, (east, rows), -self.k_east.ravel())
        np.add.at(matrix, (rows, north), -self.k_north.ravel())
        np.add.at(matrix, (north, rows), -self.k_north.ravel())
        return matrix

    def solve_exact(self, rhs: np.ndarray) -> np.ndarray:
        """Solve `A x = rhs` on this level exactly, for every leading component.

        Used on the coarsest grid, where the matrix is 64 x 64. Its inverse is
        formed once per `solve` call and cached on the level, so a cycle costs
        one small matrix product per component rather than a factorization.
        """
        if self._inverse is None:
            self._inverse = np.linalg.inv(self.dense())
        rhs = np.asarray(rhs, dtype=np.float64)
        flat = rhs.reshape(-1, self.n * self.n)
        return (flat @ self._inverse.T).reshape(rhs.shape)


def build_levels(kappa: np.ndarray) -> list[Level]:
    """Coefficient hierarchy from the finest grid down to `MG_COARSEST`."""
    n = kappa.shape[-1]
    if kappa.shape[-2] != n:
        raise ValueError("kappa must be square")
    if n & (n - 1):
        raise ValueError("the history grid must be a power of two")
    if n < MG_COARSEST:
        raise ValueError(f"grid is finer than {MG_COARSEST} cells")

    levels: list[Level] = []
    k_east, k_north = edge_coefficients(np.asarray(kappa, dtype=np.float64))
    size = n
    while True:
        levels.append(Level(size, k_east, k_north,
                            _diagonal_from_edges(k_east, k_north)))
        if size == MG_COARSEST:
            break
        k_east, k_north = coarsen_edges(k_east, k_north)
        size //= 2
    return levels


def v_cycle(levels: list[Level], u: np.ndarray, rhs: np.ndarray,
            depth: int = 0) -> np.ndarray:
    """One V-cycle: smooth, correct from the next grid down, smooth again.

    The coarsest grid is solved exactly rather than smoothed. Weighted Jacobi
    there converged only while the coarsest coefficient was near one; at the
    top of the stiffness dial it is 256 and fifty sweeps left 94 % of the
    coarse residual standing, so every cycle carried an unfinished correction.
    """
    level = levels[depth]
    u = level.smooth(u, rhs, MG_PRE)

    coarse_rhs = restrict(rhs - level.apply(u))
    coarse = levels[depth + 1]
    if depth + 2 == len(levels):
        correction = coarse.solve_exact(coarse_rhs)
    else:
        correction = v_cycle(levels, np.zeros_like(coarse_rhs), coarse_rhs,
                             depth + 1)

    return level.smooth(u + prolong(correction), rhs, MG_POST, reverse=True)


def solve(D: np.ndarray, kappa: np.ndarray, u0: np.ndarray | None = None,
          *, levels: list[Level] | None = None, tol: float = MG_TOL,
          max_cycles: int = MG_MAX_CYCLES) -> tuple[np.ndarray, int, float]:
    """Solve `u - div(kappa grad u) = D`. Returns `(u, cycles, residual)`.

    `tol` is the relative residual the driver stops at, `MG_TOL` by default.
    It is a parameter so a caller can measure what a tighter one would have
    changed; nothing in the history passes anything but the default.

    `max_cycles` is the effort budget per solve, `MG_MAX_CYCLES` by default.
    A solve that spends it without reaching `tol` returns what it has and
    reports the residual it reached, which is what the callers record.

    The V-cycle above is used as the preconditioner of a conjugate gradient
    rather than as a standalone iteration. The operator is symmetric positive
    definite and so is one V-cycle, so this is the same multigrid on the same
    operator with a Krylov acceleration around it. The standalone V-cycle is
    not a contraction on a stiff weak network: below the first coarsening a
    two-cell weak line is narrower than one coarse cell, the aggregate absorbs
    it, and the coarse operator welds across the barrier. Conjugate gradient
    absorbs the error modes the coarse grids get wrong, and how many cycles
    that takes is what the callers record.
    """
    rhs = np.asarray(D, dtype=np.float64)
    if levels is None:
        levels = build_levels(kappa)
    finest = levels[0]
    single_grid = len(levels) == 1

    def precondition(r: np.ndarray) -> np.ndarray:
        if single_grid:
            return finest.solve_exact(r)
        return v_cycle(levels, np.zeros_like(r), r)

    scale = float(np.linalg.norm(rhs))
    if scale == 0.0:
        return np.zeros_like(rhs), 0, 0.0

    u = np.zeros_like(rhs) if u0 is None else np.array(u0, dtype=np.float64)
    residual = rhs - finest.apply(u)
    relative = float(np.linalg.norm(residual)) / scale
    if relative < tol:
        return u, 0, relative

    z = precondition(residual)
    direction = z.copy()
    rz = float(np.sum(residual * z))
    cycles = 0
    while relative >= tol and cycles < max_cycles:
        curvature = finest.apply(direction)
        denominator = float(np.sum(direction * curvature))
        if denominator <= 0.0:
            break
        step = rz / denominator
        u = u + step * direction
        residual = residual - step * curvature
        cycles += 1
        relative = float(np.linalg.norm(residual)) / scale
        if relative < tol:
            break
        z = precondition(residual)
        rz_next = float(np.sum(residual * z))
        direction = z + (rz_next / rz) * direction
        rz = rz_next
    return u, cycles, relative


__all__ = [
    "Level",
    "apply_A",
    "build_levels",
    "coarsen_edges",
    "diagonal",
    "edge_coefficients",
    "effective_gradients",
    "kappa0_for",
    "prolong",
    "restrict",
    "restrict_kappa",
    "solve",
    "v_cycle",
]
