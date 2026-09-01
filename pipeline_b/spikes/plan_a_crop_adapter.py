"""Plan A step 2: crop-window adapter — periodic parent to finite tail.

Takes the B18 periodic parents (spikes/plan_a_periodic_m1.py), selects
census-qualified windows (crop-last), extracts a finite sub-Structure
with PRODUCTION-EQUIVALENT margins, and runs the UNMODIFIED M2+M3 tail
(`coarse_elevation` -> `run_erosion` -> `sample_map` -> render) to
deliver final-resolution maps. This is the program's first end-to-end
test of the exact delivered ring (§3a) on the periodic-parent
architecture, and the first author-reviewable final imagery from it.

Geometry:
- Census window: 103 cells (4120 km) at 40 km; the exact 4096-km frame
  is centred inside it, so a census ring-water window is a conservative
  structural guarantee for the frame ring (the FINAL ring is still
  decided by M2 sea level + M3 + fine detail — that is what this spike
  tests).
- Extraction: 195 cells (7800 km) with the frame 1852 km from every
  extraction edge — the same rim distance the approved production world
  gives the frame (0.45 x 4096 = 1843.2 km), i.e. the geometry already
  accepted at M1/M2. All per-cell Structure fields are torus-true
  (extracted with wrap), so extraction-edge cells carry exact parent
  values.
- `sample_map` is called through its `_frame_window_km` seam with the
  exact 4096-km window; the public centred-frame path is untouched.

Known open limitation (recorded, not solved here): M2/M3 noise and the
eustatic sea-level solve are anchored to the EXTRACTION's local km
coordinates and land inventory. A shifted or enlarged extraction of the
same frame would therefore differ by noise phase and sea-level offset
by construction, so a nested-domain causality control is NOT run here;
it needs either a global-km noise anchor seam or the nested-chronology
protocol (§4 band-semantics decision, pending with the author).

Nothing in engine/ is modified. Failure of a window's final ring is a
RESULT (it measures the structural-census-to-final gap), not an
implementation failure.

Usage:
  python spikes/plan_a_crop_adapter.py --debug   # seeds 3, 7
  python spikes/plan_a_crop_adapter.py           # cohort 151-158
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.elevation import coarse_elevation
from engine.erosion import run_erosion
from engine.render_map import hypsometric_rgb
from engine.surface import sample_map
from engine.tectonics import FRAME_KM, Config, Structure
from spikes.plan_a_periodic_m1 import (CK, COHORT, N, WINDOW_CELLS,
                                       build_parent, window_fields)

EXT_CELLS = 195                    # 7800 km: production-equivalent world
EXT_MARGIN_CELLS = 46              # frame sits 1852 km from every edge
FRAME_LOCAL_KM = 1852.0            # (46 - 0.3) cells * 40 + 12 exactly
OUT_SIZE = 1024

BANDS = (("low_15_25", 0.15, 0.25),
         ("med_30_40", 0.30, 0.40),
         ("high_45_50", 0.45, 0.4999))


def select_windows(cont):
    """Best ring-water origin per composition band (highest land
    fraction inside the band), plus the richest ring-water origin
    anywhere as a fallback exhibit when no band is populated."""
    frac, ring_water = window_fields(cont)
    picks = []
    for name, lo, hi in BANDS:
        sel = ring_water & (frac >= lo) & (frac <= hi)
        if sel.any():
            flat = np.where(sel, frac, -1.0)
            oy, ox = np.unravel_index(np.argmax(flat), flat.shape)
            picks.append({"band": name, "oy": int(oy), "ox": int(ox),
                          "census_land_frac": float(frac[oy, ox])})
    if not picks and ring_water.any():
        flat = np.where(ring_water, frac, -1.0)
        oy, ox = np.unravel_index(np.argmax(flat), flat.shape)
        picks.append({"band": "best_ring_water", "oy": int(oy),
                      "ox": int(ox),
                      "census_land_frac": float(frac[oy, ox])})
    return picks


def extract_structure(parent, oy, ox):
    """Finite production-geometry sub-Structure around census origin
    (oy, ox); all fields wrap-extracted so every cell is torus-true."""
    e0y = oy - EXT_MARGIN_CELLS
    e0x = ox - EXT_MARGIN_CELLS
    iy = (np.arange(EXT_CELLS) + e0y) % N
    ix = (np.arange(EXT_CELLS) + e0x) % N
    grid = np.ix_(iy, ix)

    def cut(a):
        return np.ascontiguousarray(a[grid])

    return Structure(
        n=EXT_CELLS, world_km=EXT_CELLS * CK,
        frame_slice=(EXT_MARGIN_CELLS, EXT_MARGIN_CELLS + WINDOW_CELLS),
        label=cut(parent.label), cont=cut(parent.cont),
        cont_frac=cut(parent.cont_frac), age_myr=cut(parent.age_myr),
        belt=cut(parent.belt), belt_age_era=cut(parent.belt_age_era),
        conv_recent=cut(parent.conv_recent),
        div_recent=cut(parent.div_recent), coast=cut(parent.coast),
        active_margin=cut(parent.active_margin),
        passive_margin=cut(parent.passive_margin),
        initial_label=cut(parent.initial_label),
        alive_plates=parent.alive_plates, eras=parent.eras,
        timings={})


def run_window(parent, seed, pick, outdir):
    cfg = Config()
    t0 = time.perf_counter()
    s = extract_structure(parent, pick["oy"], pick["ox"])
    ce = coarse_elevation(s, cfg, seed)
    er = run_erosion(s, ce, cfg, seed)
    window = (FRAME_LOCAL_KM, FRAME_LOCAL_KM, float(FRAME_KM))
    m = sample_map(s, ce, er, cfg, seed, OUT_SIZE,
                   _frame_window_km=window)
    tail_s = time.perf_counter() - t0

    water = m["water"]
    ring = np.zeros_like(water)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    ring_land_px = int(np.count_nonzero(ring & ~water))
    land_frac = float(1.0 - water.mean())

    rgb = hypsometric_rgb(m, river_density=cfg.river_density)
    img = Image.fromarray(np.asarray(rgb, np.uint8), "RGB")
    name = f"final_seed{seed}_{pick['band']}_oy{pick['oy']}_ox{pick['ox']}"
    img.save(outdir / f"{name}.png")

    return {
        "seed": seed, **pick,
        "ring_is_all_water": ring_land_px == 0,
        "ring_land_px": ring_land_px,
        "ring_px_total": int(ring.sum()),
        "delivered_land_frac": land_frac,
        "tail_s": round(tail_s, 2),
        "image": f"{name}.png",
    }, img


def main(argv):
    debug = "--debug" in argv
    seeds = [3, 7] if debug else list(COHORT)
    outdir = ROOT / "out" / ("plan_a_crop_debug" if debug
                             else "plan_a_crop_run1")
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    thumbs = []
    for seed in seeds:
        parent, _ = build_parent(seed)
        picks = select_windows(parent.cont)
        if not picks:
            results.append({"seed": seed, "band": "NONE",
                            "note": "no ring-water window"})
            print(f"seed {seed}: no ring-water window")
            continue
        for pick in picks:
            rec, img = run_window(parent, seed, pick, outdir)
            results.append(rec)
            thumbs.append((rec, img))
            print(f"seed {seed} {pick['band']}: census "
                  f"{pick['census_land_frac']:.3f} -> delivered "
                  f"{rec['delivered_land_frac']:.3f}, ring "
                  f"{'WATER' if rec['ring_is_all_water'] else str(rec['ring_land_px']) + ' land px'}"
                  f", tail {rec['tail_s']}s")

    if thumbs:
        tw = 512
        cols = 3
        rows = (len(thumbs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, rows * tw), (12, 12, 12))
        for i, (_, img) in enumerate(thumbs):
            sheet.paste(img.resize((tw, tw), Image.LANCZOS),
                        ((i % cols) * tw, (i // cols) * tw))
        sheet.save(outdir / "montage_final.png")

    report = {
        "spike": "plan_a_crop_adapter",
        "date": "2026-09-01",
        "ext_cells": EXT_CELLS, "ext_margin_cells": EXT_MARGIN_CELLS,
        "frame_local_km": FRAME_LOCAL_KM, "out_size": OUT_SIZE,
        "seeds": seeds,
        "results": results,
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=1),
                                        encoding="utf-8")

    lines = ["# Plan A crop-adapter report", "",
             "| seed | band | census land | delivered land | ring | "
             "tail s | image |", "|" + "---|" * 7]
    for r in results:
        if r.get("band") == "NONE":
            lines.append(f"| {r['seed']} | none | - | - | - | - | - |")
            continue
        ring = ("all water" if r["ring_is_all_water"]
                else f"{r['ring_land_px']}/{r['ring_px_total']} land")
        lines.append(
            f"| {r['seed']} | {r['band']} | {r['census_land_frac']:.3f} "
            f"| {r['delivered_land_frac']:.3f} | {ring} | {r['tail_s']} "
            f"| {r['image']} |")
    (outdir / "report.md").write_text("\n".join(lines) + "\n",
                                      encoding="utf-8")
    print(f"\nwrote {outdir}")


if __name__ == "__main__":
    main(sys.argv[1:])
