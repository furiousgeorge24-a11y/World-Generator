"""S3 spike: border-mechanism tail statistics.

Question: with the world larger than the frame and the causes of land
(continental nuclei) confined toward the world interior, how often does
continental crust reach the frame ring anyway — and which parameter
regime drives that tail to zero without strangling in-frame land?

Continental crust is a conservative land proxy (real land is a subset:
margins flood). The frame is the delivered map window; the world rim
outside it is discarded at crop time.

    py -3.14 spikes\\s3_border_stats.py [--seeds 300] [--size 128]

Reports per configuration: distribution of the nearest
continental-crust-to-frame-edge distance (negative = crust crosses the
frame boundary into the outermost ring), violation and touch rates,
in-frame land fraction (composition health), and a frame-hug score
(occupancy of the band just inside the frame edge). Worst offenders go
to a gallery sheet.
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "s1", os.path.join(_here, "s1_tectonic_history.py"))
s1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1)

CONFIGS = {
    # current S1 defaults: no deliberate confinement beyond nucleus box
    "A_current": {},
    # tempered: tighter nucleus box, smaller cratons, slower plates
    "B_tempered": {"nuc_box": 0.30, "nuc_r": (0.09, 0.15),
                   "speed": (0.5, 1.6), "speed_cap": 1.8},
    # tempered + wider discarded rim (frame inset grows 0.18 -> 0.24)
    "C_wide_rim": {"nuc_box": 0.32, "nuc_r": (0.09, 0.15),
                   "speed": (0.5, 1.6), "speed_cap": 1.8},
    # D: same as B, relying on the now-bounded nucleus outlines
    "D_bounded": {"nuc_box": 0.30, "nuc_r": (0.09, 0.15),
                  "speed": (0.5, 1.6), "speed_cap": 1.8},
    # E: D + candidate mantle-circulation clustering (NOT ratified)
    "E_mantle": {"nuc_box": 0.30, "nuc_r": (0.09, 0.15),
                 "speed": (0.5, 1.6), "speed_cap": 1.8,
                 "center_pull": 0.55},
    # F: free drift, frame derived from the structural result
    # (cartographic framing): window chosen post-history, from the
    # tectonic stage only, to minimize ring land then frame-hugging
    "F_window": {"nuc_box": 0.22, "nuc_r": (0.09, 0.15)},
}
FRAME_INSET = {"A_current": 0.18, "B_tempered": 0.18, "C_wide_rim": 0.24,
               "D_bounded": 0.18, "E_mantle": 0.18, "F_window": None}
WINDOW_FRAC = 0.60  # F: frame edge length as a fraction of the world


def dist_to_mask(mask):
    """Vectorized two-pass chamfer distance to the nearest True cell."""
    G0, G1 = mask.shape
    INF = 1e9
    d = np.where(mask, 0.0, INF)
    # forward pass (row loop, vector ops per row)
    for i in range(G0):
        if i > 0:
            up = d[i - 1]
            cand = np.minimum(up + 1.0,
                              np.minimum(np.r_[INF, up[:-1]] + 1.414,
                                         np.r_[up[1:], INF] + 1.414))
            d[i] = np.minimum(d[i], cand)
        row = d[i]
        acc = np.minimum.accumulate(row - np.arange(G1))
        d[i] = np.minimum(row, acc + np.arange(G1))
        rr = row[::-1]
        acc = np.minimum.accumulate(rr - np.arange(G1))
        d[i] = np.minimum(d[i], (acc + np.arange(G1))[::-1])
    for i in range(G0 - 2, -1, -1):
        dn = d[i + 1]
        cand = np.minimum(dn + 1.0,
                          np.minimum(np.r_[INF, dn[:-1]] + 1.414,
                                     np.r_[dn[1:], INF] + 1.414))
        d[i] = np.minimum(d[i], cand)
        row = d[i]
        acc = np.minimum.accumulate(row - np.arange(G1))
        d[i] = np.minimum(row, acc + np.arange(G1))
        rr = row[::-1]
        acc = np.minimum.accumulate(rr - np.arange(G1))
        d[i] = np.minimum(d[i], (acc + np.arange(G1))[::-1])
    return d


def best_window(cont, G, wlen):
    """Cartographic framing: among all window positions (coarse step),
    pick the one whose outer ring holds the least continental crust,
    breaking ties by least land near the ring. Uses the structural
    stage only — nothing downstream can move the frame."""
    best = None
    for oy in range(0, G - wlen + 1, 4):
        for ox in range(0, G - wlen + 1, 4):
            win = cont[oy:oy + wlen, ox:ox + wlen]
            ring = np.concatenate([win[0, :], win[-1, :],
                                   win[1:-1, 0], win[1:-1, -1]])
            band = win[:8, :].sum() + win[-8:, :].sum() \
                + win[8:-8, :8].sum() + win[8:-8, -8:].sum()
            key = (int(ring.sum()), int(band))
            if best is None or key < best[0]:
                best = (key, oy, ox)
    return best[1], best[2]


def measure(w, inset, window=None):
    G = w.G
    if window is not None:
        a_y, a_x, wlen = window
        a, b = 0, 0  # unused in window mode
        frame = np.zeros((G, G), bool)
        frame[a_y:a_y + wlen, a_x:a_x + wlen] = True
        ring = frame.copy()
        ring[a_y + 1:a_y + wlen - 1, a_x + 1:a_x + wlen - 1] = False
        band = np.zeros((G, G), bool)
        wband = max(2, int(0.05 * wlen))
        band |= frame
        band[a_y + wband:a_y + wlen - wband,
             a_x + wband:a_x + wlen - wband] = False
        cont_in_frame = w.cont & frame
        if not cont_in_frame.any():
            return None
        d = dist_to_mask(w.cont)
        return {"edge_d": float(d[ring].min()),
                "ring_land": bool((w.cont & ring).any()),
                "hug": float((w.cont & band).sum()) / float(band.sum()),
                "land_frac": float(cont_in_frame.sum()) / float(frame.sum())}
    a = int(round(inset * G))
    b = G - a
    frame = np.zeros((G, G), bool)
    frame[a:b, a:b] = True
    ring = frame.copy()
    ring[a + 1:b - 1, a + 1:b - 1] = False

    cont_in_frame = w.cont & frame
    if not cont_in_frame.any():
        return None  # no land in map at all — composition failure

    d = dist_to_mask(w.cont)
    # distance from the frame edge inward to nearest crust; if crust is
    # in the ring itself the distance is 0 -> report negative reach
    edge_d = d[ring].min()
    ring_land = bool((w.cont & ring).any())
    band = np.zeros((G, G), bool)
    wband = max(2, int(0.05 * (b - a)))
    band[a:b, a:b] = True
    band[a + wband:b - wband, a + wband:b - wband] = False
    hug = float((w.cont & band).sum()) / float(band.sum())
    land_frac = float(cont_in_frame.sum()) / float(frame.sum())
    return {"edge_d": float(edge_d), "ring_land": ring_land,
            "hug": hug, "land_frac": land_frac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--eras", type=int, default=20)
    args = ap.parse_args()

    outdir = os.path.join(_here, "..", "out", "spikes")
    os.makedirs(outdir, exist_ok=True)

    for name, prm in CONFIGS.items():
        inset = FRAME_INSET[name]
        t0 = time.perf_counter()
        rows, worst, empty = [], [], 0
        wlen = int(WINDOW_FRAC * args.size)
        for seed in range(1, args.seeds + 1):
            w, P, K = s1.simulate(seed, args.size, args.eras, prm)
            if inset is None:
                oy, ox = best_window(w.cont, args.size, wlen)
                m = measure(w, None, window=(oy, ox, wlen))
            else:
                m = measure(w, inset)
            if m is None:
                empty += 1
                continue
            rows.append(m)
            worst.append((m["edge_d"], seed))
        el = time.perf_counter() - t0

        ed = np.array([r["edge_d"] for r in rows])
        ring_hits = sum(r["ring_land"] for r in rows)
        hug = np.array([r["hug"] for r in rows])
        lf = np.array([r["land_frac"] for r in rows])
        n = len(rows)
        mode = "derived window" if inset is None else f"inset {inset:.2f}"
        print(f"\n{name}  ({mode}, {n} valid seeds, "
              f"{empty} empty-frame, {el:.0f}s)")
        print(f"  ring-land violations : {ring_hits}/{n}")
        print(f"  edge distance (cells): min={ed.min():.1f} "
              f"p1={np.percentile(ed, 1):.1f} p5={np.percentile(ed, 5):.1f} "
              f"median={np.median(ed):.1f}")
        print(f"  frame-hug occupancy  : mean={hug.mean():.3f} "
              f"p95={np.percentile(hug, 95):.3f}")
        print(f"  in-frame land frac   : mean={lf.mean():.3f} "
              f"p5={np.percentile(lf, 5):.3f} p95={np.percentile(lf, 95):.3f}")

        worst.sort(key=lambda t: t[0])
        tiles = []
        for edge_d, seed in worst[:6]:
            w, _, _ = s1.simulate(seed, args.size, args.eras, prm)
            img = s1.view_crust(w)
            img = np.kron(img, np.ones((3, 3, 1), np.uint8))
            im = Image.fromarray(img)
            d2 = ImageDraw.Draw(im)
            if inset is None:
                oy, ox = best_window(w.cont, args.size, wlen)
                a_y, a_x, b_y, b_x = (oy * 3, ox * 3,
                                      (oy + wlen) * 3, (ox + wlen) * 3)
                d2.rectangle([a_x, a_y, b_x, b_y],
                             outline=(255, 80, 80), width=2)
            else:
                a = int(round(inset * args.size)) * 3
                b = (args.size - int(round(inset * args.size))) * 3
                d2.rectangle([a, a, b, b], outline=(255, 80, 80), width=2)
            d2.text((4, 2), f"seed {seed} edge_d={edge_d:.0f}",
                    fill=(255, 255, 255))
            tiles.append(np.asarray(im))
        sheet = np.concatenate(tiles, axis=1)
        Image.fromarray(sheet).save(
            os.path.join(outdir, f"s3_worst_{name}.png"))
    print("\nworst-case sheets in", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
