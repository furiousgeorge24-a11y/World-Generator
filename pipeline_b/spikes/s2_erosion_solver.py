"""S2 spike: erosion solver feasibility.

Question: can flow routing + implicit stream-power incision run inside
the contract's §15 budgets at 512/1024/2048 in numpy-only Python, and
do the resulting valley webs read at map scale?

Throwaway de-risking code. The terrain here is synthetic noise + a
broad uplift field — NOT pipeline output and NOT a design precedent;
only the solver mechanics and timings matter.

    py -3.14 spikes\\s2_erosion_solver.py [--sizes 256 512 1024] [--steps 8]

Mechanics: depression-fill by vectorized morphological reconstruction
(alternating directional sweeps with an epsilon gradient), D8 steepest
receivers on the filled surface, topological ordering by vectorized
Kahn batches, O(n) flow accumulation over the batches, and the
implicit stream-power update processed downstream-first (receiver's new
elevation before donor's), which is unconditionally stable so a few
large geologic steps reach a steady-ish state.
"""

import argparse
import os
import time

import numpy as np
from PIL import Image

# Fill gradient so routed water always has a way out. Routing plumbing
# only: it must stay far below physical magnitudes at ANY grid size.
# Was 1e-3, whose accumulated floor (EPS * G/2 = 0.512 at G=1024)
# crossed the 0.5 lake-view threshold at map center and painted a
# dotted column of phantom lakes — see MILESTONES.md S2 diagnosis.
EPS = 1e-5
M_EXP = 0.5     # stream-power area exponent
K_SPL = 0.03    # erodibility
DT = 2.0        # geologic step
UPLIFT = 60.0   # peak uplift per step
HILL_ALPHA = 0.2     # hillslope diffusion (soil creep) per substep
HILL_SUBSTEPS = 3    # substeps per geologic step (explicit, stable <=0.25)


def upsample_bilinear(coarse, size):
    n = coarse.shape[0] - 1
    idx = np.linspace(0.0, n, size)
    i0 = np.minimum(idx.astype(int), n - 1)
    f = idx - i0
    rows = coarse[i0, :] * (1 - f)[:, None] + coarse[i0 + 1, :] * f[:, None]
    return rows[:, i0] * (1 - f)[None, :] + rows[:, i0 + 1] * f[None, :]


def fbm(rng, size, base=4, octaves=7, gain=0.55):
    out = np.zeros((size, size))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = min(base * (2 ** o), size)
        out += amp * upsample_bilinear(rng.standard_normal((cells + 1,
                                                            cells + 1)), size)
        total += amp
        amp *= gain
    return out / total


# ------------------------------------------------------------- fill

def fill_depressions(h):
    """Morphological reconstruction with epsilon, by alternating sweeps.

    In-row propagation uses the min-plus trick: min over k<=j of
    (F[k] + (j-k)*eps) computed with a single minimum.accumulate.
    """
    G = h.shape[0]
    F = np.full_like(h, np.inf)
    F[0, :] = h[0, :]
    F[-1, :] = h[-1, :]
    F[:, 0] = h[:, 0]
    F[:, -1] = h[:, -1]
    ramp = EPS * np.arange(G)

    def sweep_rows(rows):
        for i in rows:
            prev = F[i - 1] if rows.step == 1 else F[i + 1]
            cand = np.minimum(prev,
                              np.minimum(np.r_[np.inf, prev[:-1]],
                                         np.r_[prev[1:], np.inf])) + EPS
            row = np.minimum(F[i], cand)
            row = np.maximum(row, h[i])
            # in-row left-to-right then right-to-left propagation
            row = np.maximum(h[i], np.minimum(
                row, np.minimum.accumulate(row - ramp) + ramp + EPS))
            rr = row[::-1]
            row = np.maximum(h[i], np.minimum(
                row, (np.minimum.accumulate(rr - ramp) + ramp + EPS)[::-1]))
            F[i] = row

    for _ in range(6):
        before = F.sum()
        sweep_rows(range(1, G - 1, 1))
        sweep_rows(range(G - 2, 0, -1))
        if abs(F.sum() - before) < 1e-6:
            break
    return F


# ------------------------------------------------------------- routing

NBR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
NBR_D = np.array([np.hypot(dy, dx) for dy, dx in NBR])


def neighbor_shift(F, dy, dx, fill):
    G = F.shape[0]
    sh = np.full_like(F, fill)
    y0s, y1s = max(0, -dy), G + min(0, -dy)
    x0s, x1s = max(0, -dx), G + min(0, -dx)
    y0d, y1d = max(0, dy), G + min(0, dy)
    x0d, x1d = max(0, dx), G + min(0, dx)
    sh[y0d:y1d, x0d:x1d] = F[y0s:y1s, x0s:x1s]
    return sh


def receivers(F):
    """D8 steepest-descent receiver index per cell (flattened), plus
    slope-weighted multiple-flow-direction weights for accumulation.

    MFD keeps drainage from locking onto the 8 compass directions on
    low-gradient ground (the straight-comb artifact)."""
    G = F.shape[0]
    n = G * G
    best_drop = np.zeros((G, G))
    rcv = np.arange(n).reshape(G, G)
    idx = np.arange(n).reshape(G, G)
    targets = np.empty((8, G, G), np.int64)
    weights = np.zeros((8, G, G))
    for k, ((dy, dx), dist) in enumerate(zip(NBR, NBR_D)):
        sh = neighbor_shift(F, dy, dx, np.inf)
        drop = (F - sh) / dist
        src = np.full_like(rcv, -1)
        src[max(0, dy):G + min(0, dy), max(0, dx):G + min(0, dx)] = \
            idx[max(0, -dy):G + min(0, -dy), max(0, -dx):G + min(0, -dx)]
        targets[k] = src
        weights[k] = np.clip(drop, 0.0, None) ** 1.1
        take = drop > best_drop
        best_drop[take] = drop[take]
        rcv[take] = src[take]
    border = np.zeros((G, G), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    rcv[border] = idx[border]
    weights[:, border] = 0.0
    wsum = weights.sum(axis=0)
    flat = wsum <= 0.0
    weights /= np.where(wsum > 0, wsum, 1.0)[None]
    return rcv.ravel(), targets.reshape(8, n), weights.reshape(8, n), \
        flat.ravel()


def topo_batches(rcv, targets, weights, flat):
    """Vectorized Kahn over the full MFD edge set (D8 fallback edges for
    flat cells included), so both MFD accumulation and the D8 implicit
    solve can run on the same batches — D8 edges are a subset of MFD
    edges. Per-batch cost stays O(batch): the in-degree array persists
    and new frontier cells are discovered among the batch's edge
    targets only, never by scanning the grid."""
    n = rcv.size
    indeg = np.zeros(n, np.int64)
    for k in range(8):
        e = weights[k] > 0
        np.add.at(indeg, targets[k, e], 1)
    fb = flat & (rcv != np.arange(n))
    np.add.at(indeg, rcv[fb], 1)

    frontier = np.nonzero(indeg == 0)[0]
    batches = []
    seen = 0
    while frontier.size:
        batches.append(frontier)
        seen += frontier.size
        outs = []
        for k in range(8):
            e = weights[k, frontier] > 0
            outs.append(targets[k, frontier[e]])
        ff = frontier[flat[frontier]]
        ff = ff[rcv[ff] != ff]
        outs.append(rcv[ff])
        tgt = np.concatenate(outs) if outs else np.empty(0, np.int64)
        np.subtract.at(indeg, tgt, 1)
        cand = np.unique(tgt)
        frontier = cand[indeg[cand] == 0]
    assert seen == n, f"topo order covered {seen}/{n}"
    return batches


def flow_accumulation(rcv, batches, n, targets, weights):
    """MFD accumulation over the D8-topological batches. Every MFD
    target is strictly lower, and batch order sorts by the filled
    surface, so all of a cell's mass is settled before it distributes.
    Cells with no downslope weight (flats) fall back to their D8
    receiver so mass still leaves them."""
    A = np.ones(n)
    for b in batches:
        w = weights[:, b]
        has = w.sum(axis=0) > 0
        bb = b[has]
        if bb.size:
            for k in range(8):
                wk = weights[k, bb]
                nz = wk > 0
                if nz.any():
                    np.add.at(A, targets[k, bb[nz]], A[bb[nz]] * wk[nz])
        fb = b[~has]
        keep = rcv[fb] != fb
        if keep.any():
            np.add.at(A, rcv[fb[keep]], A[fb[keep]])
    # concentrated D8 accumulation over the same batches (valid: D8
    # edges are a subset of the batch edge set) — the channel view
    A8 = np.ones(n)
    for b in batches:
        r = rcv[b]
        keep = r != b
        np.add.at(A8, r[keep], A8[b[keep]])
    return A, A8


def hillslope_diffuse(h):
    """Soil creep: explicit diffusion substeps. The cross-slope process
    stream power lacks — keeps graded slopes curved instead of planar
    (see MILESTONES.md rectangular-lake diagnosis) and relaxes
    cell-scale spikes."""
    for _ in range(HILL_SUBSTEPS):
        p = np.pad(h, 1, mode="edge")
        lap = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
               - 4.0 * h)
        h = h + HILL_ALPHA * lap
        h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = 0.0
    return h


def spl_implicit(h, U, rcv, batches, A):
    hf = h.ravel()
    Uf = U.ravel()
    f = K_SPL * DT * A ** M_EXP
    for b in reversed(batches):
        r = rcv[b]
        keep = r != b
        bb, rr = b[keep], r[keep]
        hf[bb] = (hf[bb] + DT * Uf[bb] + f[bb] * hf[rr]) / (1.0 + f[bb])
        out = b[~keep]
        hf[out] = 0.0  # border/base level
    return hf.reshape(h.shape)


# ------------------------------------------------------------- driver

def run(seed, G, steps):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5B52]))
    U = UPLIFT * np.clip(fbm(rng, G, base=3, octaves=4) * 1.6 + 0.3, 0, None) ** 1.5
    h = U * 4.0 + 30.0 * fbm(rng, G, base=6, octaves=10, gain=0.62)
    h[0, :] = h[-1, :] = h[:, 0] = h[:, -1] = 0.0

    t = {"fill": 0.0, "route": 0.0, "topo": 0.0, "accum": 0.0,
         "solve": 0.0, "creep": 0.0}
    A = None
    for _ in range(steps):
        t0 = time.perf_counter()
        F = fill_depressions(h)
        t1 = time.perf_counter()
        rcv, targets, weights, flat = receivers(F)
        t2 = time.perf_counter()
        batches = topo_batches(rcv, targets, weights, flat)
        t3 = time.perf_counter()
        A, A8 = flow_accumulation(rcv, batches, G * G, targets, weights)
        t4 = time.perf_counter()
        h = spl_implicit(h, U, rcv, batches, A)
        t5 = time.perf_counter()
        h = hillslope_diffuse(h)
        t6 = time.perf_counter()
        t["fill"] += t1 - t0
        t["route"] += t2 - t1
        t["topo"] += t3 - t2
        t["accum"] += t4 - t3
        t["solve"] += t5 - t4
        t["creep"] += t6 - t5
    return h, fill_depressions(h), A8, t


def render(h, F, A8, path):
    G = h.shape[0]
    t = np.clip(h / max(h.max(), 1.0), 0, 1) ** 0.7
    stops = np.array([
        [88, 138, 88], [122, 162, 96], [168, 186, 112], [205, 200, 138],
        [196, 174, 118], [172, 142, 94], [146, 110, 74], [120, 86, 58],
    ])
    band = np.minimum((t * len(stops)).astype(int), len(stops) - 1)
    img = stops[band].astype(float)
    lakes = (F - h) > 0.5
    rivers = (A8.reshape(G, G) > (G * G) * 0.0003) & ~lakes
    img[rivers] = (40, 70, 100)
    img[lakes] = (95, 140, 175)
    Image.fromarray(img.astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "out", "spikes")
    os.makedirs(outdir, exist_ok=True)
    for G in args.sizes:
        t0 = time.perf_counter()
        h, F, A8, t = run(args.seed, G, args.steps)
        total = time.perf_counter() - t0
        parts = "  ".join(f"{k}={v:.2f}s" for k, v in t.items())
        print(f"{G:5d}²  total={total:6.2f}s  ({parts})")
        render(h, F, A8, os.path.join(outdir, f"s2_erosion_{G}.png"))
    print("renders in", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
