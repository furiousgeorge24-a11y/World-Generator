"""Determinism + contract smoke tests. Run: py -3.14 tests/smoke.py"""

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mapgen import pipeline, registry, render  # noqa: E402
from mapgen.rng import rng_for  # noqa: E402


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        sys.exit(1)


# 1. bit-identical repeat (contract section 5)
w1 = pipeline.generate(7, {}, 256)
w2 = pipeline.generate(7, {}, 256)
check("same seed -> identical elevation",
      np.array_equal(w1["elevation"], w2["elevation"]))

b1, b2 = io.BytesIO(), io.BytesIO()
render.save_png(render.hypsometric(w1), b1, w1)
render.save_png(render.hypsometric(w2), b2, w2)
check("same seed -> identical PNG bytes", b1.getvalue() == b2.getvalue())

# 2. seeds differ
w3 = pipeline.generate(8, {}, 256)
check("different seed -> different map",
      not np.array_equal(w1["elevation"], w3["elevation"]))

# 3. per-stage RNG isolation (contract section 5)
a = rng_for(7, "stage_a").integers(0, 2**32, 8)
b = rng_for(7, "stage_b").integers(0, 2**32, 8)
a2 = rng_for(7, "stage_a").integers(0, 2**32, 8)
check("stage keying: streams differ across stages", not np.array_equal(a, b))
check("stage keying: stream stable for same stage", np.array_equal(a, a2))

# 4. never-fail resolve: clamp + unknown -> findings, not exceptions
vals, finds = registry.resolve({"plate_count": 999, "nope": 3})
check("out-of-range clamps with finding",
      vals["plate_count"] == 24
      and any("clamped" in f["msg"] for f in finds))
check("unknown control ignored with finding",
      any("unknown" in f["msg"] for f in finds))

# 5. structural resolution independence (contract section 6):
# same 1024 km extent at 128 cells (8 km) and 256 cells (4 km).
lo = pipeline.generate(7, {"cell_size_km": 8.0}, 128)["elevation"]
hi = pipeline.generate(7, {"cell_size_km": 4.0}, 256)["elevation"]
hi_ds = hi.reshape(128, 2, 128, 2).mean(axis=(1, 3))
r = np.corrcoef(lo.ravel(), hi_ds.ravel())[0, 1]
check(f"same extent across resolutions correlates (r={r:.3f})", r > 0.9)

# 6. border invariant HOLDS (contract section 7) — green from C1 onward
bf = next(f for f in w1.findings if f["check"] == "border_ring")
check("border invariant holds (outer ring water)", bf["ok"])

# 7. plates/crust sanity
check("plate_id covers multiple plates",
      len(np.unique(w1["plate_id"])) >= 3)
lf = float((w1["elevation"] >= 0).mean())
check(f"land fraction near target (got {lf:.3f})", 0.15 < lf < 0.6)

# 8. crust-plate affinity (A1): at full affinity every non-fallback core
# sits on a continent-flagged plate, and the border invariant still holds
wa = pipeline.generate(11, {"crust_plate_affinity": 1.0}, 192)
am = wa.meta["crust"]["affinity"]
anchored = all(p in am["cont_plates"]
               for p, fb in zip(am["cluster_plates"], am["cluster_fallback"])
               if not fb)
check("affinity=1 anchors cores to continental plates", anchored)
check("border invariant holds at affinity=1",
      next(f for f in wa.findings if f["check"] == "border_ring")["ok"])

# 8b. leading-edge bias (A2): dragging the knob never moves unshifted
# clusters; shifts exist, are convergent-only, and respect the
# placement margin; border invariant holds at full bias
wl0 = pipeline.generate(7, {"active_margin_bias": 0.0}, 192)
wl1 = pipeline.generate(7, {"active_margin_bias": 1.0}, 192)
cl0 = wl0.meta["crust"]["leading"]["clusters"]
cl1 = wl1.meta["crust"]["leading"]["clusters"]
check("bias drag leaves unshifted clusters in place",
      all(a["center_km"] == b["center_km"]
          for a, b in zip(cl0, cl1) if not b["applied"]))
applied = [li for li in cl1 if li["applied"]]
for s in (11, 3, 5):
    if applied:
        break
    wl = pipeline.generate(s, {"active_margin_bias": 1.0}, 192)
    applied = [li for li in wl.meta["crust"]["leading"]["clusters"]
               if li["applied"]]
check("bias=1 produces a leading-edge shift on some seed", len(applied) > 0)
check("shifts are convergent-only",
      all(li["vn"] is not None and li["vn"] > 0.05 for li in applied))
min_l = 192 * wl1.cell_km
qmargin = min(0.13 * min_l, 0.35 * min_l)
check("shifted centers respect the placement margin",
      all(qmargin - 0.1 <= v <= min_l - qmargin + 0.1
          for li in applied for v in li["center_km"]))
check("border invariant holds at bias=1",
      next(f for f in wl1.findings if f["check"] == "border_ring")["ok"])

# 8c. passive-margin bathymetry (B1): the taper only shallows ocean
# (land can gain banks, never lose coast); border holds at B1 extremes;
# the rise never shoals above its depth gate; pedestals only add
_B1OFF = {"margin_width_km": 0.0, "edifice_pedestal": 0.0,
          "rise_feed": 0.0, "canyon_depth": 0.0, "fan_size": 0.0}
wb0 = pipeline.generate(9, dict(_B1OFF), 192)
wb1 = pipeline.generate(9, dict(_B1OFF, margin_width_km=300.0), 192)
l0 = wb0["elevation"] >= 0
l1 = wb1["elevation"] >= 0
check("margin taper leaves coastlines essentially alone",
      float((l0 != l1).mean()) < 0.004)
wbx = pipeline.generate(9, {"margin_width_km": 500.0,
                            "edifice_pedestal": 2.0, "rise_feed": 2.0}, 192)
check("border invariant holds at B1 extremes",
      next(f for f in wbx.findings if f["check"] == "border_ring")["ok"])
wr = pipeline.generate(9, dict(_B1OFF, rise_feed=2.0), 192)
dr = wr["elevation"].astype(float) - wb0["elevation"].astype(float)
shallow0 = wb0["elevation"].astype(float) >= -499.0
check("rise builds only below its depth gate",
      float(np.abs(dr[shallow0]).max()) < 1e-3 and float(dr.max()) > 0.0)
_RAW = dict(_B1OFF, erosion_strength=0.0, wave_planation=0.0,
            sediment_softening=0.0)          # pure construction, no carve
wq0 = pipeline.generate(9, dict(_RAW), 192)
wp = pipeline.generate(9, dict(_RAW, edifice_pedestal=1.0), 192)
# judged on the uplift field: downstream noise/zone reassignment may
# legally move individual elevation cells, but the pedestal's own
# contribution must be pure added mass
dp_ = wp["uplift"].astype(float) - wq0["uplift"].astype(float)
check("edifice pedestal only adds tectonic mass (and does add)",
      float(dp_.min()) > -1e-3 and float(dp_.max()) > 60.0)

# 8d. tier 2 (belt-and-basin anatomy): basin fill only raises land and
# never moves a coastline (fill is capped below the spill); it shrinks
# lake volume; rift segmentation and belt raggedness ablate cleanly at
# zero and reshape anatomy above it; crest zones are pure added mass
wt0 = pipeline.generate(3, {"basin_fill": 0.0}, 256)
wt1 = pipeline.generate(3, {"basin_fill": 1.0}, 256)
lm0 = wt0["elevation"] >= 0
check("basin fill never moves a coastline",
      np.array_equal(lm0, wt1["elevation"] >= 0))
df_ = wt1["elevation"].astype(float) - wt0["elevation"].astype(float)
check("basin fill only raises land (and does raise)",
      float(df_[lm0].min()) > -1e-3 and float(df_.max()) > 1.0)
fv0 = next(f for f in wt0.findings if f["check"] == "lakes")["fill_volume_km3"]
fv1 = next(f for f in wt1.findings if f["check"] == "lakes")["fill_volume_km3"]
check(f"basin fill shrinks lake volume ({fv0} -> {fv1} km3)", fv1 < fv0)
seg_diff = False
for s in (5, 7, 9, 11):
    ws0 = pipeline.generate(s, {"rift_segmentation": 0.0}, 192)
    ws1 = pipeline.generate(s, {"rift_segmentation": 1.0}, 192)
    if not np.array_equal(ws0["elevation"], ws1["elevation"]):
        seg_diff = True
        break
check("rift segmentation changes rift anatomy on some seed", seg_diff)
check("border invariant holds at segmentation=1",
      next(f for f in ws1.findings if f["check"] == "border_ring")["ok"])
wg0 = pipeline.generate(7, {"belt_raggedness": 0.0}, 192)
wg1 = pipeline.generate(7, {"belt_raggedness": 1.0}, 192)
check("belt raggedness reshapes belts",
      not np.array_equal(wg0["elevation"], wg1["elevation"]))
check("border invariant holds at raggedness=1",
      next(f for f in wg1.findings if f["check"] == "border_ring")["ok"])
wz0 = pipeline.generate(7, {"crest_zone": 0.0}, 192)
wz1 = pipeline.generate(7, {"crest_zone": 1.0}, 192)
dz_ = wz1["uplift"].astype(float) - wz0["uplift"].astype(float)
check("crest zones only add summit mass (and do add)",
      float(dz_.min()) > -1e-3 and float(dz_.max()) > 100.0)

# 9. KR palettes: every palette renders; canon differs from classic
imgs = []
for p in range(4):
    w1.controls["render_palette"] = p
    imgs.append(render.hypsometric(w1).tobytes())
w1.controls["render_palette"] = 1
check("all render palettes produce output", all(len(b) > 0 for b in imgs))
check("canon palette differs from classic", imgs[0] != imgs[1])

# 10. K1 drowned datum: border holds at flood extremes; shelf is carved;
# planation never moves today's coastline
wf = pipeline.generate(13, {"flood_rise_m": 250.0}, 192)
check("border invariant holds at flood=250",
      next(f for f in wf.findings if f["check"] == "border_ring")["ok"])
w0 = pipeline.generate(13, {"flood_rise_m": 0.0}, 192)
check("border invariant holds at flood=0",
      next(f for f in w0.findings if f["check"] == "border_ring")["ok"])
ef = next(f for f in w1.findings if f["check"] == "erosion")
check(f"shelf carved while exposed (got {ef['shelf_incision_km3']} km3)",
      ef["shelf_incision_km3"] > 0.0)
wp0 = pipeline.generate(13, {"wave_planation": 0.0}, 192)
wp1 = pipeline.generate(13, {"wave_planation": 1.0}, 192)
check("planation never moves today's coastline",
      np.array_equal(wp0["elevation"] >= 0, wp1["elevation"] >= 0))

# 11. K3 mass balance: deposition is real but conservative; the diffusion
# coastal leak is fixed; plains grain adds lowland texture
ek = next(f for f in w1.findings if f["check"] == "erosion")
check(f"deposition settles carved mass (got {ek['deposited_km3']} km3)",
      0.0 < ek["deposited_km3"] < ek["eroded_volume_km3"])
lf0 = float((pipeline.generate(17, {"hillslope_smoothing": 0.0},
                               192)["elevation"] >= 0).mean())
lf1 = float((pipeline.generate(17, {"hillslope_smoothing": 1.0},
                               192)["elevation"] >= 0).mean())
check(f"diffusion no longer eats coasts (delta {abs(lf1 - lf0):.4f})",
      abs(lf1 - lf0) < 0.008)
def _lowland_texture(e):
    """Mean high-pass magnitude strictly inside land below 300 m (whole
    neighborhood in-band, so coast and mountain edges don't dominate)."""
    nb = (e[:-2, 1:-1] + e[2:, 1:-1] + e[1:-1, :-2] + e[1:-1, 2:]) / 4.0
    hp = e[1:-1, 1:-1] - nb
    band = (e >= 0) & (e < 300)
    m = (band[1:-1, 1:-1] & band[:-2, 1:-1] & band[2:, 1:-1]
         & band[1:-1, :-2] & band[1:-1, 2:])
    return float(np.abs(hp[m]).mean())


_PIN = {"active_margin_bias": 0.0, "margin_width_km": 0.0,
        "edifice_pedestal": 0.0, "rise_feed": 0.0}   # measure grain on the
g0 = pipeline.generate(17, dict(_PIN, plains_grain=0.0),  # K3-era world
                       192)["elevation"].astype(float)
g1 = pipeline.generate(17, dict(_PIN, plains_grain=1.0),
                       192)["elevation"].astype(float)
low0, low1 = _lowland_texture(g0), _lowland_texture(g1)
check(f"plains grain textures lowlands ({low0:.2f} -> {low1:.2f} m/cell)",
      low1 > low0 + 0.4)

# 12. K2 profiles: plateau tendency actually gates plateau formation, and
# the isostatic knee keeps stacked orogens bounded
wp0 = pipeline.generate(3, {"plateau_tendency": 0.0}, 256)
wp1 = pipeline.generate(3, {"plateau_tendency": 1.0}, 256)
s0 = float(wp0["tect_plateau"].sum())
s1 = float(wp1["tect_plateau"].sum())
check(f"plateau_tendency gates plateau mass ({s0:.0f} -> {s1:.0f})",
      s1 > max(s0 * 1.5, 1.0))
emax = max(float(wp0["elevation"].max()), float(wp1["elevation"].max()),
           float(w1["elevation"].max()))
check(f"isostatic knee bounds peaks (max {emax:.0f} m)", emax < 8500.0)

print("all smoke tests pass")
