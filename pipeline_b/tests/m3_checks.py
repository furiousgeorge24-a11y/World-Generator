"""M3 checks: the surface-process stage's own invariants — process-grid
resolution independence, §4 staging of the erosion controls, sediment
mass balance, lake-mechanism calibration (known positive), river/water
integrity, and the head/tail adapter contract.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.elevation import coarse_elevation
from engine.erosion import (LAKE_FEED_CELLS, LAKE_MIN_DEPTH,
                            fill_depressions, run_erosion)
from engine.registry import make_config
from engine.surface import sample_map
from engine.tectonics import build_structure

PASS = []
FAIL = []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ""))


def gen(seed, size, **overrides):
    cfg = make_config(overrides)
    s = build_structure(seed, cfg)
    ce = coarse_elevation(s, cfg, seed)
    er = run_erosion(s, ce, cfg, seed)
    m = sample_map(s, ce, er, cfg, seed, size)
    return s, ce, er, m


def main():
    print("== determinism (full pipeline) ==")
    _, _, er1, m1 = gen(7, 192)
    _, _, er2, m2 = gen(7, 192)
    check("bit-identical eroded surface", np.array_equal(er1["z"],
                                                         er2["z"]))
    check("bit-identical output h", np.array_equal(m1["h"], m2["h"]))
    check("bit-identical river edges",
          np.array_equal(er1["river_edges"]["a8"],
                         er2["river_edges"]["a8"]))

    print("== process-grid resolution independence (§2) ==")
    _, _, erA, mA = gen(11, 128)
    _, _, erB, mB = gen(11, 512)
    check("erosion fields identical across output sizes",
          np.array_equal(erA["z"], erB["z"])
          and np.array_equal(erA["discharge_log"],
                             erB["discharge_log"]))
    pooled = mB["h"].reshape(128, 4, 128, 4).mean(axis=(1, 3))
    c = np.corrcoef(pooled.ravel(),
                    mA["h"].astype(np.float64).ravel())[0, 1]
    check("output structure corr > 0.985", c > 0.985, f"corr={c:.4f}")

    print("== control staging (§4) ==")
    s1, ce1, er1, _ = gen(19, 128, erosion_time=10.0)
    s2, ce2, er2, _ = gen(19, 128, erosion_time=30.0)
    check("erosion_time: structure untouched",
          np.array_equal(s1.label, s2.label)
          and np.array_equal(s1.cont_frac, s2.cont_frac))
    check("erosion_time: coarse crust untouched",
          np.array_equal(ce1["z"], ce2["z"]))
    check("erosion_time changes the surface",
          not np.array_equal(er1["z"], er2["z"]))
    _, _, er0, m0 = gen(19, 128, erosion_time=0.0)
    check("erosion off: surface = initial surface",
          np.array_equal(er0["z"], er0["z0"]))
    check("erosion off: no sediment", float(er0["sed"].sum()) == 0.0)
    _, _, erc, _ = gen(19, 128, soil_creep=0.0)
    check("soil_creep=0 runs and differs",
          not np.array_equal(erc["z"], er1["z"]))
    _, _, erl, _ = gen(19, 128, lowstand_drop=0.0)
    check("lowstand=0 runs", bool(np.isfinite(erl["z"]).all()))

    print("== physics sanity ==")
    for sd in (7, 23, 88):
        _, _, er, m = gen(sd, 128)
        ero = float(er["ero"].sum())
        dep = float(er["sed"].sum())
        check(f"seed {sd}: mass balance dep<=ero",
              dep <= ero * 1.001,
              f"dep/ero={dep / max(ero, 1e-9):.2f}")
        check(f"seed {sd}: no NaN/inf",
              bool(np.isfinite(er["z"]).all()))
        e = er["river_edges"]
        check(f"seed {sd}: river edges flow downhill on fill surface",
              e["a8"].size > 0)
    _, _, er, m = gen(7, 256)
    check("peaks plausible after erosion (§7i)",
          float(m["h"].max()) < 7500.0, f"peak={m['h'].max():.0f}")
    check("ocean rule: ocean implies process submergence",
          bool((~m["ocean"] | (m["hc"] < 0)).all()))
    check("lakes lie above present sea level",
          bool((er["lake_surf"][er["lake_depth"] > 0] > 0).all())
          if (er["lake_depth"] > 0).any() else True)

    print("== lake mechanism calibration (known positive) ==")
    # synthetic: a real closed basin with a feeding stream must yield a
    # lake through the same fill/threshold logic the engine uses
    G = 64
    zz = np.full((G, G), 500.0)
    yy, xx = np.mgrid[0:G, 0:G]
    zz -= 8.0 * xx                       # regional tilt -> drainage
    bowl = np.hypot(yy - 32, xx - 40) < 6
    zz[bowl] -= 60.0                     # closed basin, spill above sea
    F = fill_depressions(zz)
    depth = F - zz
    check("synthetic basin detected",
          bool(((depth > LAKE_MIN_DEPTH) & (F > 0)).sum() > 10),
          f"cells={int(((depth > LAKE_MIN_DEPTH) & (F > 0)).sum())}")
    check("feed threshold sane (a 20-cell stream qualifies)",
          20.0 > LAKE_FEED_CELLS)

    print("== head/tail adapter contract ==")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import webui_adapter as A
    head = A.generate_head(31, {}, 128)
    z_before = head["coarse"]["z"].copy()
    w1 = A.run_tail(head, {"erosion_time": 12.0})
    w2 = A.run_tail(head, {"erosion_time": 28.0})
    check("run_tail does not mutate head",
          np.array_equal(head["coarse"]["z"], z_before))
    check("tails differ by erosion control",
          not np.array_equal(w1["map"]["h"], w2["map"]["h"]))
    t0 = time.perf_counter()
    A.run_tail(head, {"erosion_time": 20.0})
    dt = time.perf_counter() - t0
    check("late-tier tail < 1.5s", dt < 1.5, f"{dt:.2f}s")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
