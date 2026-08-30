"""S1 spike: coarse time-stepped plate history.

Question: does a simple raster plate simulation at a fixed coarse scale
produce diverse, believable structure — few large domains, long
connected belt systems, margins that vary along one continent — in
negligible time?

Throwaway de-risking code (see MILESTONES.md): no controls, no
provenance, no architecture precedent. Judged on structure views in a
contact sheet across seeds.

    py -3.14 spikes\\s1_tectonic_history.py [--seeds 12] [--size 128] [--eras 14]

Mechanics, spike-grade: plates partition the world via noise-warped
nearest-seed regions; continental nuclei seed old crust; each era every
plate moves rigidly (translation + slow rotation, wandering per era).
Where plates overlap, crust converges: continental overrides oceanic
(subduction), continent-on-continent marks collision; every overlap
stamps belt intensity and a belt age. Where plates separate, the gap
fills with new age-zero oceanic crust (ridges). Margins are classified
at the end: continental crust adjacent to ocean is "active" near
recent convergence, else "passive".
"""

import argparse
import os
import time

import numpy as np
from PIL import Image, ImageDraw

DT_MYR = 50.0          # crust aging per era
NUCLEUS_AGE = 1800.0   # starting age of continental crust


# ---------------------------------------------------------------- helpers

def shift(a, dy, dx, fill):
    """Shift a 2D array without wrapping (the world is bounded)."""
    G0, G1 = a.shape
    out = np.full_like(a, fill)
    y0s, y1s = max(0, -dy), min(G0, G0 - dy)
    x0s, x1s = max(0, -dx), min(G1, G1 - dx)
    y0d, y1d = max(0, dy), min(G0, G0 + dy)
    x0d, x1d = max(0, dx), min(G1, G1 + dx)
    out[y0d:y1d, x0d:x1d] = a[y0s:y1s, x0s:x1s]
    return out


def dilate(mask, r):
    out = mask.copy()
    for _ in range(r):
        grown = out.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            grown |= shift(out, dy, dx, False)
        out = grown
    return out


def upsample_bilinear(coarse, size):
    n = coarse.shape[0] - 1
    idx = np.linspace(0.0, n, size)
    i0 = np.minimum(idx.astype(int), n - 1)
    f = idx - i0
    rows = coarse[i0, :] * (1 - f)[:, None] + coarse[i0 + 1, :] * f[:, None]
    return rows[:, i0] * (1 - f)[None, :] + rows[:, i0 + 1] * f[None, :]


def noise_field(rng, size, cells, amp):
    return amp * upsample_bilinear(
        rng.standard_normal((cells + 1, cells + 1)), size)


# ---------------------------------------------------------------- sim

class W:
    """World state at the structural scale.

    belt/belt_age are crust properties: they advect with the plate that
    carries them (an orogen rides its continent), never staying behind
    in world coordinates.
    """

    def __init__(self, G):
        self.G = G
        self.label = np.zeros((G, G), np.int32)
        self.cont = np.zeros((G, G), bool)
        self.age = np.zeros((G, G))
        self.belt = np.zeros((G, G))
        self.belt_age = np.full((G, G), -1.0)


def partition_plates(rng, G, P):
    yy, xx = np.indices((G, G)).astype(float)
    pts = rng.uniform(0.05 * G, 0.95 * G, (P, 2))
    best = np.full((G, G), np.inf)
    label = np.zeros((G, G), np.int32)
    for p in range(P):
        d = np.hypot(yy - pts[p, 0], xx - pts[p, 1])
        warp = (1.0 + noise_field(rng, G, 8, 0.55)).clip(0.25)
        cost = d * warp
        take = cost < best
        best[take] = cost[take]
        label[take] = p
    return label


DEFAULTS = {
    "nuc_box": 0.15,      # nuclei centers kept this fraction in from the rim
    "nuc_r": (0.11, 0.20),  # nucleus radius range, fraction of world
    "speed": (0.6, 2.2),    # plate speed range, cells per era
    "speed_cap": 2.5,
    # candidate process, NOT ratified: weak mantle-circulation bias
    # pulling plate drift toward the world center (supercontinent
    # clustering). 0 = off.
    "center_pull": 0.0,
}


def seed_crust(rng, G, K, label, P, prm):
    """One nucleus per chosen plate, centered well inside it, so a
    craton rides one plate instead of straddling a boundary and being
    torn apart from era one."""
    yy, xx = np.indices((G, G)).astype(float)
    interior = ~plate_edges_wide(label, 4)
    sizes = np.bincount(label.ravel(), minlength=P).astype(float)
    order = np.argsort(-sizes)
    hosts = order[:K]
    cont = np.zeros((G, G), bool)
    box = prm["nuc_box"]
    for p in hosts:
        cells = np.nonzero((label == p) & interior
                           & (yy > box * G) & (yy < (1 - box) * G)
                           & (xx > box * G) & (xx < (1 - box) * G))
        if not cells[0].size:
            cells = np.nonzero(label == p)
        i = int(rng.integers(0, cells[0].size))
        cy, cx = float(cells[0][i]), float(cells[1][i])
        r = rng.uniform(*prm["nuc_r"]) * G
        # irregular outline with BOUNDED reach: radius varies by at most
        # +-28%, so a craton's maximum extent is a real invariant
        wob = np.clip(noise_field(rng, G, 6, 1.0), -0.8, 0.8)
        cont |= np.hypot(yy - cy, xx - cx) < r * (1.0 + 0.35 * wob)
    age = np.where(cont, NUCLEUS_AGE, rng.uniform(0.0, 180.0, (G, G)))
    return cont, age


def plate_edges_wide(label, r):
    e = np.zeros(label.shape, bool)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        e |= label != shift(label, dy, dx, -1)
    return dilate(e, r)


def centroids(label, P):
    cents = np.zeros((P, 2))
    for p in range(P):
        ys, xs = np.nonzero(label == p)
        if ys.size:
            cents[p] = ys.mean(), xs.mean()
    return cents


def fill_new_ocean(label):
    lab = label.copy()
    empty = lab < 0
    for _ in range(lab.shape[0]):
        if not empty.any():
            break
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            neighbor = shift(lab, dy, dx, -1)
            take = empty & (neighbor >= 0)
            lab[take] = neighbor[take]
            empty = lab < 0
    return lab


def step(w, vel, omg, cents, era):
    G, P = w.G, len(vel)
    yy, xx = np.indices((G, G)).astype(float)
    new_label = np.full((G, G), -1, np.int32)
    new_cont = np.zeros((G, G), bool)
    new_age = np.zeros((G, G))
    new_belt = np.zeros((G, G))
    new_belt_age = np.full((G, G), -1.0)
    win_cont = np.zeros((G, G), bool)
    overlap = np.zeros((G, G), bool)
    cont_clash = np.zeros((G, G), bool)

    for p in range(P):
        vy, vx = vel[p]
        cy, cx = cents[p]
        c, s = np.cos(-omg[p]), np.sin(-omg[p])
        sy = yy - vy - cy
        sx = xx - vx - cx
        iy = np.rint(cy + sy * c - sx * s).astype(int)
        ix = np.rint(cx + sy * s + sx * c).astype(int)
        inside = (iy >= 0) & (iy < G) & (ix >= 0) & (ix < G)
        iyc = iy.clip(0, G - 1)
        ixc = ix.clip(0, G - 1)
        mask = inside & (w.label[iyc, ixc] == p)
        if not mask.any():
            continue
        m = np.nonzero(mask)
        my, mx = iyc[m], ixc[m]
        src_cont = np.zeros((G, G), bool)
        src_cont[m] = w.cont[my, mx]
        src_age = np.zeros((G, G))
        src_age[m] = w.age[my, mx]
        src_belt = np.zeros((G, G))
        src_belt[m] = w.belt[my, mx]
        src_belt_age = np.full((G, G), -1.0)
        src_belt_age[m] = w.belt_age[my, mx]

        occupied = new_label >= 0
        collide = mask & occupied
        overlap |= collide
        cont_clash |= collide & src_cont & win_cont
        win = mask & (~occupied | (src_cont & ~win_cont))
        new_label[win] = p
        new_cont[win] = src_cont[win]
        new_age[win] = src_age[win]
        new_belt[win] = src_belt[win]
        new_belt_age[win] = src_belt_age[win]
        win_cont[win] = src_cont[win]

    # divergence gaps: new oceanic crust at the ridge, owned by a neighbor
    gap = new_label < 0
    new_label = fill_new_ocean(new_label)
    new_cont[gap] = False
    new_age[gap] = 0.0
    new_belt[gap] = 0.0
    new_belt_age[gap] = -1.0

    w.label = new_label
    w.cont = new_cont
    w.age = new_age + DT_MYR
    # convergence stamps belt onto the surviving (overriding) crust
    w.belt = new_belt + np.where(cont_clash, 2.0, 0.0) \
        + np.where(overlap, 1.0, 0.0)
    w.belt_age = np.where(overlap, float(era), new_belt_age)


def simulate(seed, G, eras, prm=None):
    prm = {**DEFAULTS, **(prm or {})}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5B51]))
    P = int(rng.integers(5, 9))
    K = int(min(rng.integers(2, 6), P))
    w = W(G)
    w.label = partition_plates(rng, G, P)
    w.cont, w.age = seed_crust(rng, G, K, w.label, P, prm)

    # coherent kinematics: plates keep direction for many eras, wander slowly
    speed = rng.uniform(*prm["speed"], P)
    ang = rng.uniform(0.0, 2 * np.pi, P)
    omg = rng.normal(0.0, 0.006, P)
    for era in range(eras):
        ang = ang + rng.normal(0.0, 0.08, P)
        speed = np.clip(speed + rng.normal(0.0, 0.06, P), 0.4,
                        prm["speed_cap"])
        vel = np.stack([speed * np.sin(ang), speed * np.cos(ang)], axis=1)
        cents = centroids(w.label, P)
        pull = prm["center_pull"]
        if pull > 0.0:
            to_c = (G / 2.0) - cents
            dist = np.hypot(to_c[:, 0], to_c[:, 1]) + 1e-9
            gain = np.minimum(1.0, dist / (0.3 * G))
            vel += pull * gain[:, None] * (to_c / dist[:, None])
        step(w, vel, omg, cents, era)
    return w, P, K


# ---------------------------------------------------------------- views

def lerp(c0, c1, t):
    t = np.clip(t, 0.0, 1.0)[..., None]
    return (np.array(c0) * (1 - t) + np.array(c1) * t)


def plate_edges(label):
    e = np.zeros(label.shape, bool)
    for dy, dx in ((0, 1), (1, 0)):
        e |= label != shift(label, dy, dx, -1)
    e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = False
    return e


def view_crust(w):
    img = np.zeros((w.G, w.G, 3))
    t_oc = np.clip(w.age / 400.0, 0, 1)
    img[:] = lerp((110, 165, 215), (16, 34, 76), t_oc)
    t_ct = np.clip(w.age / 2600.0, 0, 1)
    img[w.cont] = lerp((208, 193, 158), (162, 143, 105), t_ct)[w.cont]
    s = np.clip(w.belt / 4.0, 0, 1)
    belt_land = lerp((208, 193, 158), (122, 58, 30), s)
    belt_sea = lerp((16, 34, 76), (6, 10, 30), s)
    b = w.belt > 0
    img[b & w.cont] = belt_land[b & w.cont]
    img[b & ~w.cont] = belt_sea[b & ~w.cont]
    img[plate_edges(w.label)] *= 0.55
    return img.astype(np.uint8)


def coast_and_margins(w, eras):
    ocean = ~w.cont
    coast = w.cont & (shift(ocean, 0, 1, True) | shift(ocean, 0, -1, True)
                      | shift(ocean, 1, 0, True) | shift(ocean, -1, 0, True))
    recent = (w.belt > 0) & (w.belt_age >= eras - 4)
    near = dilate(recent, 5)
    return coast & near, coast & ~near


def view_margins(w, eras):
    img = np.zeros((w.G, w.G, 3))
    img[:] = (25, 35, 60)
    img[w.cont] = (185, 185, 185)
    active, passive = coast_and_margins(w, eras)
    img[dilate(passive, 1)] = (70, 180, 90)
    img[dilate(active, 1)] = (225, 60, 45)
    return img.astype(np.uint8)


def view_belt_age(w, eras):
    img = np.zeros((w.G, w.G, 3))
    img[:] = (28, 32, 48)
    img[w.cont] = (78, 78, 78)
    b = w.belt > 0
    t = np.clip(w.belt_age / max(eras - 1, 1), 0, 1)
    img[b] = lerp((96, 42, 130), (250, 220, 80), t)[b]
    return img.astype(np.uint8)


def upscale(img, k):
    return np.kron(img, np.ones((k, k, 1), np.uint8))


def contact_sheet(results, G, eras, k, path):
    tiles = []
    for seed, w, P, K, ms in results:
        row = [upscale(view_crust(w), k),
               upscale(view_margins(w, eras), k),
               upscale(view_belt_age(w, eras), k)]
        strip = np.concatenate(row, axis=1)
        im = Image.fromarray(strip)
        d = ImageDraw.Draw(im)
        inset = int(0.18 * G) * k
        for cx0 in range(3):
            x0 = cx0 * G * k + inset
            d.rectangle([x0, inset, cx0 * G * k + G * k - inset,
                         G * k - inset], outline=(245, 245, 245), width=1)
        d.text((4, 2), f"seed {seed}  plates {P}  nuclei {K}  {ms:.0f} ms",
               fill=(255, 255, 255))
        tiles.append(np.asarray(im))
    sheet = np.concatenate(tiles, axis=0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(sheet).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--eras", type=int, default=14)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "out", "spikes",
        "s1_contact.png"))
    args = ap.parse_args()

    results = []
    for seed in range(1, args.seeds + 1):
        t0 = time.perf_counter()
        w, P, K = simulate(seed, args.size, args.eras)
        ms = (time.perf_counter() - t0) * 1000
        results.append((seed, w, P, K, ms))
        print(f"seed {seed}: plates={P} nuclei={K} {ms:.1f} ms")
    contact_sheet(results, args.size, args.eras, args.scale, args.out)
    print("sheet:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
