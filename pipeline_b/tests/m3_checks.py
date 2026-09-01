"""M3 checks: the surface-process stage's own invariants — process-grid
resolution independence, §4 staging, time-scaled conservative creep,
actual-incision sediment sourcing and boundary-export closure,
final-terrain hydrology, flat lake levels, river/water integrity, and
the head/tail/provenance adapter contract.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.elevation import _chamfer_km, coarse_elevation
from engine.erosion import (CREEP_DIFFUSIVITY_KM2_MYR,
                            LAKE_FEED_CELLS, LAKE_MIN_DEPTH,
                            L_MOIST_KM, _balance_lakes,
                            fill_depressions, flow_accumulation_d8,
                            receivers, route_sediment, run_erosion,
                            soil_creep, spl_implicit, topo_batches)
from engine.registry import make_config
from engine.surface import _bilinear, _masked_bilinear, sample_map
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


def final_route(er):
    """Independently reconstruct delivered D8 hydrology from final z."""
    F = fill_depressions(er["z"])
    rcv, targets, weights, flat = receivers(F)
    batches = topo_batches(rcv, targets, weights, flat)
    runoff = np.exp(-_chamfer_km(er["z0"] < 0.0, er["e_km"])
                    / L_MOIST_KM)
    A8 = flow_accumulation_d8(rcv, batches, er["z"].size, runoff)
    return F, rcv, A8


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
    check("bit-identical sediment export",
          er1["sediment_export_m3"] == er2["sediment_export_m3"])

    print("== process-grid resolution independence (§2) ==")
    _, _, erA, mA = gen(11, 128)
    _, _, erB, mB = gen(11, 512)
    check("erosion fields identical across output sizes",
          np.array_equal(erA["z"], erB["z"])
          and np.array_equal(erA["discharge_log"],
                             erB["discharge_log"])
          and erA["sediment_export_m3"]
          == erB["sediment_export_m3"])
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
    check("erosion off: no process products",
          float(er0["ero"].sum()) == 0.0
          and float(er0["sed"].sum()) == 0.0
          and er0["sediment_export_m3"] == 0.0
          and not (er0["lake_depth"] > 0).any()
          and er0["river_edges"]["a8"].size == 0)
    _, _, erc, _ = gen(19, 128, erosion_time=10.0, soil_creep=0.0)
    check("soil_creep=0 runs and differs",
          not np.array_equal(erc["z"], er1["z"]))
    _, _, erl, _ = gen(19, 128, lowstand_drop=0.0)
    check("lowstand=0 runs", bool(np.isfinite(erl["z"]).all()))

    print("== creep time, exposure, and source accounting ==")
    zz = np.full((17, 17), -500.0)
    zz[4:13, 4:13] = 100.0
    zz[8, 8] = 500.0
    inactive = zz < -80.0
    z_tiny = soil_creep(zz.copy(), CREEP_DIFFUSIVITY_KM2_MYR,
                        1e-6, 20.0, -80.0)
    z_short = soil_creep(zz.copy(), CREEP_DIFFUSIVITY_KM2_MYR,
                         1.0, 20.0, -80.0)
    z_long = soil_creep(zz.copy(), CREEP_DIFFUSIVITY_KM2_MYR,
                        20.0, 20.0, -80.0)
    check("creep conserves redistributed mass",
          bool(np.isclose(z_long.sum(), zz.sum(), rtol=0.0, atol=1e-9)))
    check("creep leaves submerged terrain unchanged",
          np.array_equal(z_long[inactive], zz[inactive]))
    check("creep response scales continuously with time",
          zz[8, 8] > z_tiny[8, 8] > z_short[8, 8] > z_long[8, 8])
    z_max = soil_creep(zz.copy(), 2.0 * CREEP_DIFFUSIVITY_KM2_MYR,
                       20.0, 20.0, -80.0)
    check("maximum creep step is finite and conservative",
          bool(np.isfinite(z_max).all())
          and bool(np.isclose(z_max.sum(), zz.sum(),
                              rtol=0.0, atol=1e-9))
          and float(z_max.min()) >= float(zz.min())
          and float(z_max.max()) <= float(zz.max()))

    slope = 500.0 - 5.0 * np.indices((9, 9))[1]
    Fs = fill_depressions(slope)
    rs, ts, ws, fls = receivers(Fs)
    bs = topo_batches(rs, ts, ws, fls)
    z_up, cut = spl_implicit(slope, np.ones_like(slope),
                             np.zeros_like(slope), rs, bs,
                             np.ones(slope.size), 10.0, 20.0,
                             np.zeros_like(slope, bool))
    check("zero erodibility creates no sediment source",
          float(cut.sum()) == 0.0
          and np.array_equal(z_up, slope + 10.0))

    cfg0 = make_config({"erosion_time": 20.0, "soil_creep": 1.0})
    cfg0.erodibility = 0.0
    s0 = build_structure(19, cfg0)
    ce0 = coarse_elevation(s0, cfg0, 19)
    ce0 = dict(ce0)
    ce0["uplift"] = np.zeros_like(ce0["uplift"])
    er_iso = run_erosion(s0, ce0, cfg0, 19)
    check("pure creep is redistribution, not routed sediment source",
          float(er_iso["ero"].sum()) == 0.0
          and float(er_iso["sed"].sum()) == 0.0
          and er_iso["sediment_export_m3"] == 0.0
          and er_iso["sediment_terminal_residual_m3"] == 0.0)

    print("== sediment outlet accounting (analytic) ==")
    zs = np.full((3, 3), -1000.0)
    es = np.zeros_like(zs)
    es[1, 1] = 100.0
    rs = np.arange(zs.size)
    rs[4] = 5
    half_length = 20.0 / np.log(2.0)
    _, dsd, out, residual = route_sediment(
        zs, es, rs, [np.array([4]), np.array([5])],
        np.ones(zs.size), 0.0, half_length, 20.0)
    check("boundary outlet exports only post-settlement remainder",
          bool(np.isclose(dsd.sum(), 75.0))
          and bool(np.isclose(out, 25.0)) and residual == 0.0)
    rs[4] = 4
    _, dsi, out_i, residual_i = route_sediment(
        zs, es, rs, [np.array([4])], np.ones(zs.size),
        0.0, half_length, 20.0)
    check("interior terminal flux is not mislabeled as export",
          bool(np.isclose(dsi.sum(), 50.0)) and out_i == 0.0
          and bool(np.isclose(residual_i, 50.0)))

    print("== physics sanity ==")
    for sd in (7, 23, 88):
        _, _, er, m = gen(sd, 128)
        area_m2 = (er["e_km"] * 1000.0) ** 2
        source_m3 = float(er["ero"].sum()) * area_m2
        deposited_m3 = float(er["sed"].sum()) * area_m2
        accounted_m3 = (deposited_m3 + er["sediment_export_m3"]
                         + er["sediment_terminal_residual_m3"])
        check(f"seed {sd}: sediment budget closes",
              bool(np.isclose(accounted_m3, source_m3,
                              rtol=1e-10, atol=1.0)),
              f"dep/export={deposited_m3 / max(source_m3, 1e-9):.2f}/"
              f"{er['sediment_export_m3'] / max(source_m3, 1e-9):.2f}")
        check(f"seed {sd}: no interior terminal sediment",
              er["sediment_terminal_residual_m3"] == 0.0)
        check(f"seed {sd}: no NaN/inf",
              bool(np.isfinite(er["z"]).all()))
        Ff, rcvf, A8f = final_route(er)
        check(f"seed {sd}: delivered discharge uses final terrain",
              np.array_equal(er["discharge_log"],
                             np.log1p(A8f).reshape(er["z"].shape)))
        e = er["river_edges"]
        iy0 = np.rint(e["y0"] / er["e_km"] - 0.5).astype(np.int64)
        ix0 = np.rint(e["x0"] / er["e_km"] - 0.5).astype(np.int64)
        iy1 = np.rint(e["y1"] / er["e_km"] - 0.5).astype(np.int64)
        ix1 = np.rint(e["x1"] / er["e_km"] - 0.5).astype(np.int64)
        src = iy0 * er["n_e"] + ix0
        dst = iy1 * er["n_e"] + ix1
        check(f"seed {sd}: river edges use final receivers/discharge",
              e["a8"].size > 0 and np.array_equal(rcvf[src], dst)
              and np.array_equal(e["a8"], A8f[src]))
        check(f"seed {sd}: river edges flow downhill on final fill",
              bool((Ff.ravel()[src] > Ff.ravel()[dst]).all()))
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
    ld, ls = _balance_lakes(zz, F, np.full_like(zz, 100.0))
    levels = ls[ld > 0.0]
    check("balanced basin has one horizontal water level",
          levels.size > 0 and float(np.ptp(levels)) == 0.0)

    lf = np.zeros((4, 4))
    lv = np.zeros((4, 4), bool)
    lf[1:3, 1:3] = 100.0
    lv[1:3, 1:3] = True
    q = (np.arange(33) + 0.5) * (80.0 / 33.0)
    xxq, yyq = q[None, :], q[:, None]
    sampled = _masked_bilinear(lf, lv, yyq, xxq, 20.0)
    support = _bilinear(lv.astype(float), yyq, xxq, 20.0) > 1e-12
    check("sampled flat lake is not tapered by dry zero backing",
          bool(np.allclose(sampled[support], 100.0,
                           rtol=0.0, atol=1e-12)))

    from types import SimpleNamespace
    empty_edges = {name: np.empty(0) for name in
                   ("x0", "y0", "x1", "y1", "xd", "yd", "a8")}
    er_syn = {
        "e_km": 20.0,
        "z": np.full((4, 4), 50.0),
        "lake_depth": np.where(lv, 50.0, 0.0),
        "lake_surf": lf,
        "discharge_log": np.zeros((4, 4)),
        "sed": np.zeros((4, 4)),
        "river_edges": empty_edges,
    }
    m_syn = sample_map(
        SimpleNamespace(world_km=80.0, n=4, frame_slice=(0, 4)),
        {}, er_syn, SimpleNamespace(detail_amplitude=0.0), 1, 33)
    check("sample_map wires the flat level through lake pixels",
          bool(m_syn["lake"].any())
          and bool(np.allclose(m_syn["lake_level"][m_syn["lake"]],
                               100.0, rtol=0.0, atol=1e-5)))

    print("== head/tail adapter contract ==")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import webui_adapter as A
    head = A.generate_head(31, {"plates": 9}, 128)
    z_before = head["coarse"]["z"].copy()
    w1 = A.run_tail(head, {"plates": 4, "erosion_time": 12.0})
    w2 = A.run_tail(head, {"erosion_time": 28.0})
    check("run_tail does not mutate head",
          np.array_equal(head["coarse"]["z"], z_before))
    check("tails differ by erosion control",
          not np.array_equal(w1["map"]["h"], w2["map"]["h"]))
    from engine.registry import CONTROLS
    check("world stores truthful complete effective control echo",
          len(w1["controls"]) == len(CONTROLS)
          and w1["controls"]["plates"] == 9
          and w1["controls"]["erosion_time"] == 12.0)

    import io
    import json
    from PIL import Image
    A.set_render_controls(w1, {"river_density": 0.2})
    meta = Image.open(io.BytesIO(A.render_png(w1, "hypsometric"))).text
    png_controls = json.loads(meta["pipeline_b:controls"])
    check("render-control provenance updates with the pixels",
          png_controls == w1["controls"]
          and png_controls["river_density"] == 0.2)
    rep = A.report(w1)
    finding_names = {f["name"] for f in rep["findings"]}
    check("report includes timings and explicit boundary export",
          bool(rep.get("timings"))
          and "sediment_boundary_export_fraction" in finding_names)
    t0 = time.perf_counter()
    A.run_tail(head, {"erosion_time": 20.0})
    dt = time.perf_counter() - t0
    check("late-tier tail < 1.5s", dt < 1.5, f"{dt:.2f}s")

    # §15 contract table on the FULL generation path (a seed change is
    # always a cold run; cached-tail interactivity does not license
    # dropping the tier budgets — restored 2026-08-31 after they were
    # found enforced only in m2_checks, and failing there).
    print("== §15 full-generation tiers ==")
    for size, budget in ((256, 1.0), (512, 3.0), (1024, 15.0),
                         (2048, 90.0)):
        t0 = time.perf_counter()
        A.generate(101, {}, size)
        dt = time.perf_counter() - t0
        check(f"{size}² full generate < {budget}s", dt < budget,
              f"{dt:.2f}s")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
