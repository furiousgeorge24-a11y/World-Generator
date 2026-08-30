"""Decision-support gallery for the two open M2 author calls.

THROWAWAY EVALUATION CODE — nothing here is wired into the engine.
The derived-window sampler below is a DEMO of S3 candidate (a): it
re-views already-simulated worlds through a different crop window.
Candidate (b) (mantle clustering) is NOT implemented; its block shows
seeds whose natural drift history happened to keep continental cores
interior — the outcome the clustering process would make typical.

Outputs:
  out/m2/m2_decision_border.png  — §3a closure candidates
  out/m2/m2_decision_budget.png  — continental_budget default retune
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import noise
from engine.elevation import coarse_elevation
from engine.registry import make_config
from engine.render_map import render_map_view
from engine.rng import stage_salt
from engine.surface import (BASE_LAM_KM, DETAIL_GAIN, FULL_OCTAVES,
                            _bicubic, _smooth01, sample_map)
from engine.tectonics import build_structure

OUT = ROOT / "out" / "m2"
OUT.mkdir(parents=True, exist_ok=True)
PAD = 8
FOOT = 16


def gen(seed, **overrides):
    cfg = make_config(overrides)
    s = build_structure(seed, cfg)
    ce = coarse_elevation(s, cfg, seed)
    return s, ce, cfg


def sample_window(s, ce, cfg, seed, size, oy_km, ox_km):
    """Script-local copy of engine.surface.sample_map with a free
    window origin (the engine's frame is fixed; a shifted window is
    just different km coordinates into the same world)."""
    ck = s.world_km / s.n
    f0, f1 = s.frame_slice
    frame_km = (f1 - f0) * ck
    km_px = frame_km / size
    q = (np.arange(size) + 0.5) * km_px
    x_km = (ox_km + q)[None, :]
    y_km = (oy_km + q)[:, None]
    hc = _bicubic(ce["h"], y_km, x_km, ck)
    oro = _bicubic(ce["oro"], y_km, x_km, ck)
    land_amp = 80.0 + 0.12 * np.maximum(hc, 0.0) + 0.05 * oro
    ocean_amp = 18.0 + 27.0 * _smooth01((hc + 2500.0) / 2250.0)
    amp = np.where(hc >= 0.0, land_amp, ocean_amp)
    lam = BASE_LAM_KM
    kept = 0
    for _ in range(FULL_OCTAVES):
        if lam < km_px:
            break
        kept += 1
        lam /= 2.0
    kept = max(kept, 4)
    det = noise.fbm(x_km, y_km, BASE_LAM_KM, kept,
                    stage_salt(seed, "surface-detail"),
                    gain=DETAIL_GAIN, norm_octaves=FULL_OCTAVES)
    h = hc + cfg.detail_amplitude * amp * det
    water = (h < 0.0) & (hc < 0.0)
    return {"h": h.astype(np.float32), "hc": hc.astype(np.float32),
            "water": water, "km_per_px": km_px, "size": size}


def ring_land_count(land):
    ring = np.zeros_like(land)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    return int((land & ring).sum())


def nearest_land_km(m):
    land = ~m["water"]
    ys, xs = np.nonzero(land)
    if not ys.size:
        return None
    n = land.shape[0]
    d = np.minimum(np.minimum(ys, n - 1 - ys),
                   np.minimum(xs, n - 1 - xs)).min()
    return float(d * m["km_per_px"])


def scan_windows(s, ce, cfg, seed, verify_size, max_try=8):
    """Deterministic scan for a ring-clean window in the simulated
    world. Score prefers content and penalizes frame-hug; the winner
    is verified at output resolution (noise islands can breach a
    shallow ring shelf). Returns (oy_km, ox_km, tried) or None."""
    ck = s.world_km / s.n
    f0, f1 = s.frame_slice
    W = f1 - f0
    Wat = ce["water"]
    max_off = f0 - 8          # stay clear of the world-rim band
    band = max(2, int(0.05 * W))
    cands = []
    for oy in range(-max_off, max_off + 1, 3):
        for ox in range(-max_off, max_off + 1, 3):
            L = ~Wat[f0 + oy:f1 + oy, f0 + ox:f1 + ox]
            if L[0, :].any() or L[-1, :].any() \
                    or L[:, 0].any() or L[:, -1].any():
                continue
            lf = float(L.mean())
            if lf < 0.04:
                continue
            hug = float(np.concatenate([
                L[:band, :].ravel(), L[-band:, :].ravel(),
                L[:, :band].ravel(), L[:, -band:].ravel()]).mean())
            cands.append((lf - 0.6 * hug, oy, ox, lf))
    cands.sort(key=lambda t: (-t[0], t[1], t[2]))
    for k, (score, oy, ox, lf) in enumerate(cands[:max_try]):
        oy_km = (f0 + oy) * ck
        ox_km = (f0 + ox) * ck
        m = sample_window(s, ce, cfg, seed, verify_size, oy_km, ox_km)
        if ring_land_count(~m["water"]) == 0:
            return oy_km, ox_km, k + 1
    return None


def mark_ring_land(im, land):
    """Red ticks where land touches the frame ring."""
    d = ImageDraw.Draw(im)
    n = land.shape[0]
    for x in np.nonzero(land[0, :])[0]:
        d.rectangle([int(x) - 1, 0, int(x) + 1, 7], fill=(255, 40, 40))
    for x in np.nonzero(land[-1, :])[0]:
        d.rectangle([int(x) - 1, n - 8, int(x) + 1, n - 1],
                    fill=(255, 40, 40))
    for y in np.nonzero(land[:, 0])[0]:
        d.rectangle([0, int(y) - 1, 7, int(y) + 1], fill=(255, 40, 40))
    for y in np.nonzero(land[:, -1])[0]:
        d.rectangle([n - 8, int(y) - 1, n - 1, int(y) + 1],
                    fill=(255, 40, 40))
    return im


def tile(im, caption, w=512):
    canvas = Image.new("RGB", (w, im.height + FOOT), (14, 14, 18))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((3, im.height + 2), caption, fill=(215, 215, 215))
    return canvas


def sheet(tiles, cols, path, title, footer_lines=()):
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    fh = 14 * len(footer_lines) + (10 if footer_lines else 0)
    img = Image.new("RGB", (cols * w + (cols + 1) * PAD,
                            rows * h + (rows + 1) * PAD + 22 + fh),
                    (10, 10, 12))
    d = ImageDraw.Draw(img)
    d.text((PAD, 5), title, fill=(235, 235, 235))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        img.paste(t, (PAD + c * (w + PAD), 22 + PAD + r * (h + PAD)))
    y0 = 22 + rows * (h + PAD) + PAD
    for j, line in enumerate(footer_lines):
        d.text((PAD, y0 + 4 + 14 * j), line, fill=(180, 180, 185))
    img.save(path)
    print(f"wrote {path}")


def centered_origin(s):
    ck = s.world_km / s.n
    f0 = s.frame_slice[0]
    return f0 * ck, f0 * ck


def main():
    t0 = time.perf_counter()

    # ---------------- feasibility + clean-seed sweep (60 seeds)
    print("== sweep: derived-window feasibility + naturally-clean seeds ==")
    feasible = 0
    clean_centered = []
    centered_viol = 0
    for sd in range(60):
        s, ce, cfg = gen(sd)
        oy, ox = centered_origin(s)
        m = sample_window(s, ce, cfg, sd, 192, oy, ox)
        if ring_land_count(~m["water"]) == 0:
            clean_centered.append(sd)
        else:
            centered_viol += 1
        if scan_windows(s, ce, cfg, sd, verify_size=192) is not None:
            feasible += 1
    print(f"  centered violations: {centered_viol}/60; "
          f"clean-by-history seeds: {clean_centered}; "
          f"derived window feasible: {feasible}/60")

    # ---------------- page 1: border decision
    print("== page 1: border candidates ==")
    tiles = []
    for sd in (3, 19, 51, 63):
        s, ce, cfg = gen(sd)
        oy, ox = centered_origin(s)
        mc = sample_window(s, ce, cfg, sd, 512, oy, ox)
        landc = ~mc["water"]
        imc = mark_ring_land(render_map_view(mc, "hypsometric"), landc)
        tiles.append(tile(imc,
                          f"seed {sd}  STATUS QUO (centered)  ring "
                          f"{ring_land_count(landc)}px  land "
                          f"{landc.mean():.2f}"))
        got = scan_windows(s, ce, cfg, sd, verify_size=512)
        if got is None:
            tiles.append(tile(
                Image.new("RGB", (512, 512), (30, 30, 34)),
                f"seed {sd}  no clean window found (report as such)"))
            continue
        wy, wx, tried = got
        md = sample_window(s, ce, cfg, sd, 512, wy, wx)
        landd = ~md["water"]
        dy, dx = wy - oy, wx - ox
        nk = nearest_land_km(md)
        tiles.append(tile(render_map_view(md, "hypsometric"),
                          f"seed {sd}  (a) DERIVED WINDOW demo  moved "
                          f"({dx:+.0f},{dy:+.0f}) km  ring 0  land "
                          f"{landd.mean():.2f}  near {nk:.0f} km"))
    footer1 = [
        "Status quo: land is decided by elevation+sea level; continental cores drift over the frame line and cannot flood "
        f"(centered violations {centered_viol}/60 at 192px check).",
        f"Candidate (a) derived cartographic window - DEMO ONLY, not implemented: same simulated world, crop moved; a ring-clean window was found in {feasible}/60 worlds "
        "(deterministic scan, content-seeking score, verified vs sub-grid islands).",
        "  caveat: (a) is selection, not formation - it correlates composition with the frame by construction; judge whether these tiles still pass the par-3b look bar.",
        "Candidate (b) mantle-circulation clustering - NOT implemented: bottom row shows seeds whose natural history kept cores interior, the outcome (b) would make typical",
        "  (it would also gather cratons, addressing the M1 scatter observation). Captions carry land fractions so the composition cost/gain of each option is visible.",
    ]
    # candidate (b) target-outcome proxy row
    proxy = clean_centered[:4]
    for sd in proxy:
        s, ce, cfg = gen(sd)
        oy, ox = centered_origin(s)
        m = sample_window(s, ce, cfg, sd, 512, oy, ox)
        land = ~m["water"]
        rl = ring_land_count(land)
        nk = nearest_land_km(m)
        near = f"{nk:.0f} km" if nk is not None else "n/a"
        tiles.append(tile(render_map_view(m, "hypsometric"),
                          f"seed {sd}  (b) TARGET-OUTCOME proxy "
                          f"(natural history)  ring {rl}px  land "
                          f"{land.mean():.2f}  near {near}"))
    sheet(tiles, 2, OUT / "m2_decision_border.png",
          "M2 decision aid 1 - par.3a hard border: status quo vs candidate (a) derived window (DEMO) "
          "vs candidate (b) target-outcome proxy",
          footer1)

    # ---------------- page 2: budget decision
    print("== page 2: budget retune ==")
    budgets = (0.30, 0.38, 0.45)
    print("  sweep 20 seeds x 3 budgets at 192...")
    stats = {}
    for b in budgets:
        lfs, viol = [], 0
        for sd in range(20):
            s, ce, cfg = gen(sd, continental_budget=b)
            oy, ox = centered_origin(s)
            m = sample_window(s, ce, cfg, sd, 192, oy, ox)
            land = ~m["water"]
            lfs.append(float(land.mean()))
            if ring_land_count(land):
                viol += 1
        stats[b] = (float(np.mean(lfs)), float(np.min(lfs)),
                    float(np.max(lfs)), viol)
        print(f"    budget {b}: land mean {stats[b][0]:.2f} "
              f"[{stats[b][1]:.2f}..{stats[b][2]:.2f}]  ring-violating "
              f"{viol}/20")

    tiles = []
    for sd in (7, 40, 77):
        for b in budgets:
            s, ce, cfg = gen(sd, continental_budget=b)
            oy, ox = centered_origin(s)
            m = sample_window(s, ce, cfg, sd, 512, oy, ox)
            land = ~m["water"]
            im = mark_ring_land(render_map_view(m, "hypsometric"), land)
            tiles.append(tile(im,
                              f"seed {sd}  budget {b:.2f}  land "
                              f"{land.mean():.2f}  L{ce['sea_level']:+.0f}m"))
    footer2 = [
        "Same seed left-to-right at continental_budget 0.30 (current default) / 0.38 / 0.45 (range top). Nuclei positions are unchanged - only crust area grows (par-4).",
        f"20-seed sweep, land fraction mean [min..max] and ring-violating seeds:  "
        f"0.30: {stats[0.30][0]:.2f} [{stats[0.30][1]:.2f}..{stats[0.30][2]:.2f}], {stats[0.30][3]}/20   "
        f"0.38: {stats[0.38][0]:.2f} [{stats[0.38][1]:.2f}..{stats[0.38][2]:.2f}], {stats[0.38][3]}/20   "
        f"0.45: {stats[0.45][0]:.2f} [{stats[0.45][1]:.2f}..{stats[0.45][2]:.2f}], {stats[0.45][3]}/20",
        "Par-8 center is ~0.33 land. Sea level rises with displaced crust automatically (eustatic feedback), so land does not scale linearly with budget.",
        "Coupling: a higher budget default presses harder on the par-3a border - the two decisions land together. Red ticks mark ring-land.",
    ]
    sheet(tiles, 3, OUT / "m2_decision_budget.png",
          "M2 decision aid 2 - continental_budget default retune "
          "(observed land 0.05-0.17 at 0.30 vs ~1/3 target)",
          footer2)

    print(f"total {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
