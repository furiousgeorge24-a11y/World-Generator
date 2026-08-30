"""M2 checks: determinism, §2 resolution independence, §4 control
isolation, elevation sanity, §15 perf, and the §3a border-evidence
sweep (evidence printed, never blocking — closure is the author's
call on the sweep numbers).
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.elevation import coarse_elevation
from engine.erosion import run_erosion
from engine.registry import make_config
from engine.render_map import render_map_view
from engine.surface import sample_map
from engine.tectonics import build_structure

PASS = []
FAIL = []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def gen(seed, size, **overrides):
    cfg = make_config(overrides)
    s = build_structure(seed, cfg)
    ce = coarse_elevation(s, cfg, seed)
    er = run_erosion(s, ce, cfg, seed)
    m = sample_map(s, ce, er, cfg, seed, size)
    return s, ce, m


def main():
    print("== determinism ==")
    _, ce1, m1 = gen(7, 256)
    _, ce2, m2 = gen(7, 256)
    check("bit-identical h", np.array_equal(m1["h"], m2["h"]))
    check("bit-identical water", np.array_equal(m1["water"], m2["water"]))
    check("identical sea level", ce1["sea_level"] == ce2["sea_level"])

    print("== resolution independence (§2) ==")
    sA, ceA, mA = gen(11, 128)
    sB, ceB, mB = gen(11, 512)
    check("sea level identical across sizes",
          ceA["sea_level"] == ceB["sea_level"])
    pooled = mB["h"].reshape(128, 4, 128, 4).mean(axis=(1, 3))
    c = np.corrcoef(pooled.ravel(), mA["h"].astype(np.float64).ravel())[0, 1]
    check("elevation structure corr > 0.985", c > 0.985, f"corr={c:.4f}")
    landA = ~mA["water"]
    landB_frac = (~mB["water"]).reshape(128, 4, 128, 4).mean(axis=(1, 3))
    agree = float((landA == (landB_frac > 0.5)).mean())
    check("land-mask block agreement > 0.97", agree > 0.97,
          f"agree={agree:.4f}")

    print("== control isolation (§4) ==")
    s1, ce1, m1 = gen(19, 192, hydrosphere_depth=4700.0)
    s2, ce2, m2 = gen(19, 192, hydrosphere_depth=5200.0)
    check("hydrosphere: structure untouched",
          np.array_equal(s1.label, s2.label)
          and np.array_equal(s1.cont, s2.cont)
          and np.array_equal(s1.belt, s2.belt))
    # the CRUSTAL surface shifts by a pure constant (eustatic datum);
    # the detailed surface legitimately re-erodes against the new base
    # level, so the sharp invariant lives at the coarse stage
    dh = ce2["h"] - ce1["h"]
    check("hydrosphere: crustal surface pure constant offset",
          float(dh.max() - dh.min()) < 1e-3,
          f"spread={dh.max() - dh.min():.2e}, shift={dh.mean():.1f} m")
    check("hydrosphere: absolute crust datum untouched",
          np.array_equal(ce1["z"], ce2["z"]))

    s3, ce3, m3 = gen(19, 192, detail_amplitude=0.0)
    check("detail off: h == process surface exactly",
          np.allclose(m3["h"], m3["hc"], atol=1e-4))
    s4, ce4, m4 = gen(19, 192, detail_amplitude=1.6)
    check("detail: coarse crustal stage identical",
          np.array_equal(ce3["z"], ce4["z"]))

    s5, ce5, m5 = gen(19, 192, orogeny_height=0.0)
    s6, ce6, m6 = gen(19, 192)
    # off-belt crust is bit-identical on the absolute datum; the small
    # world-wide eustatic shift (flattening belts redistributes the
    # shared ocean) is expected physics, reported for the record
    flat = ce6["oro"] == 0.0
    check("orogeny: off-belt crust untouched (absolute datum)",
          bool(flat.any())
          and np.array_equal(ce5["z"][flat], ce6["z"][flat]),
          f"eustatic shift {ce5['sea_level'] - ce6['sea_level']:+.0f} m")
    check("orogeny: structure untouched",
          np.array_equal(s5.label, s6.label))

    s7, _, _ = gen(19, 128, passive_shelf_km=200.0)
    check("shelf width: structure untouched",
          np.array_equal(s7.label, s6.label)
          and np.array_equal(s7.age_myr, s6.age_myr))

    print("== elevation sanity ==")
    s, ce, m = gen(23, 256)
    sel = (~s.cont) & (s.belt == 0) & (ce["d_cont"] > 280.0)
    t = np.maximum(s.age_myr[sel], 0.0)
    law = -np.where(t < 20.0, 2600.0 + 365.0 * np.sqrt(t),
                    5651.0 - 2473.0 * np.exp(-t / 62.8))
    check("GDH1 subsidence wired exactly (open-ocean cells)",
          bool(sel.any()) and np.allclose(ce["z"][sel], law, atol=1e-6))
    check("|sea-level datum| < 800 m at defaults",
          abs(ce["sea_level"]) < 800.0, f"L={ce['sea_level']:.0f} m")
    maxes, mins = [], []
    for sd in (3, 9, 23, 40, 51, 77):
        _, _, mm = gen(sd, 192)
        maxes.append(float(mm["h"].max()))
        mins.append(float(mm["h"].min()))
    check("peak heights plausible (§7i)", max(maxes) < 7500.0,
          f"max peak {max(maxes):.0f} m")
    check("floor plausible", min(mins) > -11000.0,
          f"deepest {min(mins):.0f} m")
    check("water rule: ocean implies process-surface submergence",
          bool((~m["ocean"] | (m["hc"] < 0)).all()))

    print("== §3a border evidence sweep (never blocking) ==")
    viol, viol_crust, viol_noise, nearest = 0, 0, 0, []
    t0 = time.perf_counter()
    for sd in range(30):
        _, _, mm = gen(sd, 256)
        land = ~mm["water"]
        ring = np.zeros_like(land)
        ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
        rl = land & ring
        if rl.any():
            viol += 1
            # provenance: crustal emergence at the frame vs sub-grid
            # bump breaching a submerged shelf
            if (rl & (mm["hc"] >= 0)).any():
                viol_crust += 1
            if (rl & (mm["hc"] < 0)).any():
                viol_noise += 1
        ys, xs = np.nonzero(land)
        if ys.size:
            d = np.minimum(np.minimum(ys, 255 - ys),
                           np.minimum(xs, 255 - xs)).min()
            nearest.append(d * mm["km_per_px"])
    nearest = np.array(nearest)
    print(f"  seeds=30  ring-land seeds={viol}"
          f" (crustal-emergence {viol_crust}, noise-island {viol_noise};"
          f" a seed can be both)"
          f"  nearest-land km: min={nearest.min():.0f}"
          f" p10={np.percentile(nearest, 10):.0f}"
          f" med={np.median(nearest):.0f}"
          f"  ({time.perf_counter() - t0:.1f} s)")
    check("sweep completed (evidence for author)", True)

    print("== perf (§15) ==")
    budgets = {256: 1.0, 512: 3.0, 1024: 15.0}
    for size, budget in budgets.items():
        t0 = time.perf_counter()
        _, _, mm = gen(101, size)
        dt = time.perf_counter() - t0
        check(f"{size}² generate < {budget}s", dt < budget, f"{dt:.2f}s")
    t0 = time.perf_counter()
    render_map_view(mm, "hypsometric")
    dt = time.perf_counter() - t0
    check("re-render 1024² < 0.15s", dt < 0.15, f"{dt * 1000:.0f} ms")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
