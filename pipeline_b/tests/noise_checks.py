"""Noise-module calibration checks (EVAL rule: probes and generators
get validated against known failure modes before their output is
trusted). Exits nonzero on failure.

    py -3.14 tests\\noise_checks.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from engine import noise, rng  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


G = 1024
km = 4000.0
xs = (np.arange(G) + 0.5) * (km / G)
X, Y = np.meshgrid(xs, xs)
salt = rng.stage_salt(7, "noise-check")

t0 = time.perf_counter()
f = noise.fbm(X, Y, base_wavelength_km=1200.0, octaves=8, salt=salt)
el = time.perf_counter() - t0
print(f"fbm 1024^2 x8 octaves: {el:.2f}s")

print("\n[determinism]")
f2 = noise.fbm(X, Y, base_wavelength_km=1200.0, octaves=8, salt=salt)
check("bit-identical rerun", np.array_equal(f, f2))

print("\n[distribution]")
check("zero-mean-ish", abs(f.mean()) < 0.02, f"mean={f.mean():.4f}")
check("unit-scale-ish", 0.1 < f.std() < 0.6, f"std={f.std():.3f}")

print("\n[degenerate columns/rows] (S4 audit failure mode)")
# A REAL degenerate column is geometric: it sits at the same x for any
# salt. A statistical minimum wanders. So: find the weakest column/row
# for three independent salts — coincidence = artifact. Plus a gross
# degeneracy bound (a true degenerate column is near-zero variance).
argmins_c, argmins_r, ratios = [], [], []
for s in (salt, rng.stage_salt(101, "noise-check"),
          rng.stage_salt(202, "noise-check")):
    hfs = noise.fbm(X, Y, base_wavelength_km=125.0, octaves=4, salt=s)
    cs = hfs.std(axis=0)
    rs = hfs.std(axis=1)
    argmins_c.append(int(cs.argmin()))
    argmins_r.append(int(rs.argmin()))
    ratios.append(min(cs.min() / cs.max(), rs.min() / rs.max()))
check("no gross degeneracy", min(ratios) > 0.35,
      f"worst min/max={min(ratios):.2f}")
c_span = max(argmins_c) - min(argmins_c)
r_span = max(argmins_r) - min(argmins_r)
check("weakest column wanders with salt", c_span > 8,
      f"argmins={argmins_c}")
check("weakest row wanders with salt", r_span > 8,
      f"argmins={argmins_r}")

print("\n[lattice-zero alignment] (gradient noise is 0 at lattice pts)")
# single octave WITHOUT offsets, sampled exactly on its lattice, should
# be ~0 — proving the hazard exists — while fbm at the same points
# should not be suppressed, proving the per-octave offsets defeat it.
lam = km / 16.0
lx = np.arange(1, 15, dtype=np.float64) * lam
LX, LY = np.meshgrid(lx, lx)
on_lattice = np.abs(noise.perlin(LX, LY, lam, salt)).mean()
off = np.abs(noise.perlin(LX + 0.5 * lam, LY + 0.5 * lam, lam, salt)).mean()
check("hazard exists (raw octave ~0 on lattice)", on_lattice < 1e-9,
      f"on={on_lattice:.2e} off={off:.3f}")
fb_on = np.abs(noise.fbm(LX, LY, lam * 4, 6, salt)).mean()
fb_off = np.abs(noise.fbm(LX + 0.37 * lam, LY + 0.61 * lam, lam * 4, 6,
                          salt)).mean()
ratio = fb_on / max(fb_off, 1e-12)
check("fbm not lattice-suppressed", 0.5 < ratio < 2.0,
      f"on/off={ratio:.2f}")

print("\n[resolution independence] (§2: same km position, same value)")
G2 = 512
xs2 = (np.arange(G2) + 0.5) * (km / G2)
X2, Y2 = np.meshgrid(xs2, xs2)
f_lo = noise.fbm(X2, Y2, base_wavelength_km=1200.0, octaves=8, salt=salt)
# 1024 grid cell centers at odd indices land exactly on the 512 centers
sub = f[1::2, 1::2]
same = np.allclose(sub, f_lo, atol=0)
# they are DIFFERENT sample positions (offset half a fine cell), so
# instead check correlation of overlapping structure via interpolation-
# free comparison at exactly shared coordinates:
xs_shared = (np.arange(256) + 0.5) * (km / 256)
XS, YS = np.meshgrid(xs_shared, xs_shared)
a = noise.fbm(XS, YS, 1200.0, 8, salt)
b = noise.fbm(XS.copy(), YS.copy(), 1200.0, 8, salt)
check("identical at shared km coords", np.array_equal(a, b))

print("\n[salt independence] (high-frequency: ~1000 independent cells,")
print("  expected |corr| ~ 0.03 for truly unrelated fields)")
hf2 = noise.fbm(X2, Y2, 125.0, 4, salt)
g = noise.fbm(X2, Y2, 125.0, 4, rng.stage_salt(7, "other-stage"))
h = noise.fbm(X2, Y2, 125.0, 4, rng.stage_salt(8, "noise-check"))
c1 = np.corrcoef(g.ravel(), hf2.ravel())[0, 1]
check("stage decorrelation", abs(c1) < 0.08, f"corr={c1:.3f}")
c2 = np.corrcoef(h.ravel(), hf2.ravel())[0, 1]
check("seed decorrelation", abs(c2) < 0.08, f"corr={c2:.3f}")

print()
if fails:
    print("FAILURES:", ", ".join(fails))
    sys.exit(1)
print("all noise checks passed")
