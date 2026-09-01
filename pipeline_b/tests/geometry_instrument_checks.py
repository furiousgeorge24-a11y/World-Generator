"""Calibration for eval/geometry_instruments.py (EVAL.md: probes are
instruments and get calibrated before their numbers are trusted).

Arms:
  A. sealed-semantics equivalence — vectorized legacy mode must match a
     direct port of the sealed `_maximum_ruler_run` on random blobs;
  B. digital-line detection efficiency by angle — quantifies the sealed
     flat-density bias; angle_fair mode must detect straightness
     uniformly across angles;
  C. known negative (axis rectangles + plus shape) — D4 gate MUST fire;
  D. rotated-rectangle control (17/29 deg) — straightness at non-D4
     angles must be detected and must NOT fire the D4 gate;
  E. isotropic known positive (thresholded engine fBm) — the gate must
     not condemn isotropic formation; measured behavior is the
     calibrated null for future runs;
  F. approved-production arm — M1 default cont masks scored on the same
     gate that condemned B16/B17 (decision evidence, never blocking);
  G. roundness sanity — circle flagged, elongated blob not.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import noise
from engine.registry import make_config
from engine.rng import stage_salt
from engine.tectonics import build_structure
from eval.geometry_instruments import (
    boundary_mask, component_roundness, d4_distance_degrees,
    label_components, max_ruler_run, rulers_for_mask,
    seed_blocked_d4_test,
)

PASS = []
FAIL = []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ""))


# --- direct port of the sealed `_maximum_ruler_run` (finite frame) ----

def sealed_reference_ruler(by, bx):
    y = np.asarray(by, np.float64)
    x = np.asarray(bx, np.float64)
    best_length = 0.0
    best_angle = None
    for degrees in range(180):
        angle = np.radians(float(degrees))
        cosine = np.cos(angle)
        sine = np.sin(angle)
        tangent = x * cosine + y * sine
        normal = -x * sine + y * cosine
        groups = {}
        normal_bin = np.rint(normal).astype(np.int32)
        for index, centre in enumerate(normal_bin):
            for key in (int(centre - 1), int(centre), int(centre + 1)):
                if abs(normal[index] - key) <= 1.0:
                    groups.setdefault(key, []).append((
                        float(tangent[index]),
                        float(y[index]),
                        float(x[index])))
        for values in groups.values():
            if len(values) < 2:
                continue
            ordered = sorted(values)
            start = 0
            for stop in range(1, len(ordered) + 1):
                at_end = stop == len(ordered)
                gap = (np.inf if at_end else
                       ordered[stop][0] - ordered[stop - 1][0])
                if gap <= 2.0:
                    continue
                segment = ordered[start:stop]
                span = float(segment[-1][0] - segment[0][0] + 1.0)
                density = float(len(segment) / max(span, 1.0))
                if density >= 0.70 and span > best_length:
                    best_length = span
                    best_angle = float(degrees)
                start = stop
    return best_length, best_angle


def strip_mask(n, angle_deg, half_len=100.0, half_width=6.0):
    ys, xs = np.mgrid[0:n, 0:n]
    cy = cx = (n - 1) / 2.0
    a = np.radians(angle_deg)
    dy = ys - cy
    dx = xs - cx
    along = dx * np.cos(a) + dy * np.sin(a)
    across = -dx * np.sin(a) + dy * np.cos(a)
    return (np.abs(along) <= half_len) & (np.abs(across) <= half_width)


def mask_rulers(mask, *, angle_fair, min_cells=10.0):
    out = []
    for ys, xs in label_components(mask):
        comp = np.zeros_like(mask)
        comp[ys, xs] = True
        b = boundary_mask(comp)
        byy, bxx = np.nonzero(b)
        run = max_ruler_run(byy, bxx, angle_fair=angle_fair)
        if run["angle_degrees"] is not None and run["cells"] >= min_cells:
            out.append(run)
    return out


def main():
    t_all = time.perf_counter()

    print("== A. sealed-semantics equivalence (legacy mode) ==")
    rng = np.random.default_rng(20260831)
    mismatches = 0
    for trial in range(12):
        m = np.zeros((48, 48), bool)
        for _ in range(rng.integers(2, 5)):
            cy, cx = rng.integers(8, 40, 2)
            r = rng.integers(3, 9)
            ys, xs = np.mgrid[0:48, 0:48]
            wob = rng.uniform(0.7, 1.3)
            m |= ((ys - cy) ** 2 * wob + (xs - cx) ** 2 / wob) < r ** 2
        for ys, xs in label_components(m):
            comp = np.zeros_like(m)
            comp[ys, xs] = True
            byy, bxx = np.nonzero(boundary_mask(comp))
            if byy.size < 2:
                continue
            ref_len, ref_ang = sealed_reference_ruler(byy, bxx)
            got = max_ruler_run(byy, bxx, angle_fair=False)
            if (abs(got["cells"] - ref_len) > 1e-9
                    or got["angle_degrees"] != ref_ang):
                mismatches += 1
    check("vectorized legacy == sealed reference", mismatches == 0,
          f"{mismatches} mismatches")

    print("== B. detection efficiency by angle ==")
    legacy_eff = {}
    fair_eff = {}
    for ang in (0, 15, 30, 45):
        m = strip_mask(256, ang)
        true_len = 200.0
        rl = mask_rulers(m, angle_fair=False)
        rf = mask_rulers(m, angle_fair=True)
        legacy_eff[ang] = max((r["cells"] for r in rl), default=0.0) / true_len
        fair_eff[ang] = max((r["cells"] for r in rf), default=0.0) / true_len
    print(f"  legacy efficiency: " + "  ".join(
        f"{a}°={legacy_eff[a]:.2f}" for a in legacy_eff))
    print(f"  fair   efficiency: " + "  ".join(
        f"{a}°={fair_eff[a]:.2f}" for a in fair_eff))
    check("angle_fair detects >=0.9 of a straight edge at every angle",
          min(fair_eff.values()) >= 0.9,
          f"min={min(fair_eff.values()):.2f}")
    check("bias documented: legacy at 45° below fair at 45°",
          legacy_eff[45] <= fair_eff[45] + 1e-9,
          f"legacy={legacy_eff[45]:.2f} fair={fair_eff[45]:.2f}")

    print("== C. known negative: axis-aligned shapes must fire ==")
    # >=5 independent blocks: with one ruler per block the blocked null
    # gives (11/45)^k, so k=3 can never reach p<0.01 (power floor found
    # by this suite's first run).
    rulers = []
    shapes = ((30, 90), (80, 26), (44, 44), (24, 70), (66, 22))
    for i, (h, w) in enumerate(shapes):
        m = np.zeros((160, 160), bool)
        m[20:20 + h, 30:30 + w] = True
        for r in mask_rulers(m, angle_fair=True, min_cells=20.0):
            rulers.append({"seed": i, "angle_degrees": r["angle_degrees"]})
    res = seed_blocked_d4_test(rulers, trials=20000)
    check("axis shapes fire the D4 gate", res[
        "randomization_upper_tail_p"] < 0.01,
        f"near {res['observed_near_d4_count']}/{res['ruler_count']}, "
        f"p={res['randomization_upper_tail_p']:.5f}")

    print("== D. rotated rectangles must NOT fire ==")
    rulers = []
    detected = 0
    for i, ang in enumerate((17.0, 29.0, 62.0)):
        m = strip_mask(256, ang, half_len=70, half_width=14)
        found = mask_rulers(m, angle_fair=True, min_cells=25.0)
        detected += len(found)
        for r in found:
            rulers.append({"seed": i, "angle_degrees": r["angle_degrees"]})
    res = seed_blocked_d4_test(rulers, trials=20000)
    near = [d4_distance_degrees(r["angle_degrees"]) <= 5.0
            for r in rulers]
    check("rotated straight edges are detected", detected >= 3,
          f"{detected} rulers")
    check("rotated straight edges read at non-D4 angles",
          sum(near) == 0, f"near-D4 {sum(near)}/{len(rulers)}")
    check("rotated shapes do not fire the D4 gate",
          res["randomization_upper_tail_p"] >= 0.05,
          f"p={res['randomization_upper_tail_p']:.4f}")

    print("== E. isotropic fBm masks (calibrated null) ==")
    CK = 40.0
    N = 614
    xs = (np.arange(N) + 0.5) * CK
    X, Y = np.meshgrid(xs, xs)
    world = N * CK
    for mode_name, fair in (("legacy", False), ("angle_fair", True)):
        rulers = []
        for mid in range(8):
            f = noise.fbm(X, Y, world / 3.0, 4,
                          stage_salt(9000 + mid, "geometry-calib"))
            m = f > np.quantile(f, 0.65)
            for r in rulers_for_mask(m, CK, 1024.0, angle_fair=fair,
                                     min_component_cells=200):
                rulers.append({"seed": mid,
                               "angle_degrees": r["angle_degrees"]})
        res = seed_blocked_d4_test(rulers, trials=50000)
        print(f"  {mode_name}: rulers={res['ruler_count']} "
              f"near={res['observed_near_d4_count']} "
              f"({res['near_fraction']:.3f} vs 0.244 analytic) "
              f"p={res['randomization_upper_tail_p']:.4f}")
        if fair:
            # CALIBRATED FINDING (2026-08-31): rasterized boundaries of
            # perfectly isotropic fields carry a near-D4 long-ruler
            # fraction FAR above the analytic 11/45 rotation null —
            # curved digital boundaries locally quantize into axis or
            # diagonal runs. The analytic-null D4 gate therefore fires
            # on ANY raster formation and is NOT evidence-grade for
            # formation-caused grid lock on its own. Future gates must
            # compare against this matched isotropic baseline (or use
            # manual/rectangle evidence). These asserts pin the
            # documented instrument property so a silent change to it
            # is caught.
            check("analytic-null gate fires on isotropic raster input "
                  "(documented instrument property)",
                  res["randomization_upper_tail_p"] < 0.01,
                  f"p={res['randomization_upper_tail_p']:.4f}")
            check("isotropic near-D4 baseline recorded in sanity band",
                  0.45 <= res["near_fraction"] <= 0.90,
                  f"near_fraction={res['near_fraction']:.3f}")
            check("isotropic arm has enough rulers to be meaningful",
                  res["ruler_count"] >= 20,
                  f"{res['ruler_count']} rulers")

    print("== F. approved-production arm (M1 defaults; evidence only) ==")
    for run_km in (512.0, 1024.0):
        for mode_name, fair in (("legacy", False), ("angle_fair", True)):
            rulers = []
            for sd in (3, 7, 11, 19, 23, 31, 40, 51):
                cfg = make_config({})
                s = build_structure(sd, cfg)
                ck = s.world_km / s.n
                for r in rulers_for_mask(s.cont, ck, run_km,
                                         angle_fair=fair,
                                         min_component_cells=30):
                    rulers.append({"seed": sd,
                                   "angle_degrees": r["angle_degrees"]})
            res = seed_blocked_d4_test(rulers, trials=50000)
            print(f"  M1 cont, run>={run_km:.0f} km, {mode_name}: "
                  f"rulers={res['ruler_count']} "
                  f"near={res['observed_near_d4_count']} "
                  f"({res['near_fraction']:.3f}) "
                  f"p={res['randomization_upper_tail_p']:.4f}")
    check("approved-production arm executed (evidence recorded above)",
          True)

    print("== G. roundness sanity ==")
    n = 120
    ys, xs = np.mgrid[0:n, 0:n]
    circle = (ys - 60) ** 2 + (xs - 60) ** 2 < 30 ** 2
    (cy, cx), = [(a, b) for a, b in [tuple(map(
        np.asarray, np.nonzero(circle)))]]
    met = component_roundness(cy, cx, circle.shape)
    check("circle flagged rounded", met["rounded"],
          f"compactness={met['compactness']:.3f} "
          f"solidity={met['solidity']:.3f}")
    wob = noise.fbm(xs * 30.0, ys * 30.0, 800.0, 4,
                    stage_salt(4, "geometry-calib-blob"))
    ridge = (np.abs(ys - 60 - 18 * np.sin(xs / 12.0)) < 5 + 3 * wob)
    ridge &= (xs > 8) & (xs < 112)
    comps = label_components(ridge)
    big = max(comps, key=lambda c: c[0].size)
    met2 = component_roundness(big[0], big[1], ridge.shape)
    check("sinuous ridge not flagged rounded", not met2["rounded"],
          f"compactness={met2['compactness']:.3f} "
          f"solidity={met2['solidity']:.3f}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed "
          f"({time.perf_counter() - t_all:.1f} s)")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
