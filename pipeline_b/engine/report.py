"""Run report (§5): seed, control echo, timings, findings. Findings are
tripwire results — they veto nothing and never block delivery; levels:
"info" (audit trail) and "warn" (surfaced as a badge in the webui).
"""

from collections import deque

import numpy as np

from . import VERSION


def _components(mask):
    """Connected components (4-neighborhood); returns list of sizes."""
    seen = np.zeros_like(mask)
    sizes = []
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0]:
            continue
        n = 0
        dq = deque([(int(y0), int(x0))])
        seen[y0, x0] = True
        while dq:
            y, x = dq.popleft()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] \
                        and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    dq.append((ny, nx))
        sizes.append(n)
    return sorted(sizes, reverse=True)


def _structure_findings(s):
    f0, f1 = s.frame_slice
    win = np.s_[f0:f1, f0:f1]
    cont_f = s.cont[win]
    frame_cells = cont_f.size

    findings = []

    def add(level, name, value, note=""):
        findings.append({"level": level, "name": name,
                         "value": value, "note": note})

    cf = float(cont_f.mean())
    add("info" if 0.12 <= cf <= 0.50 else "warn",
        "cont_fraction_in_frame", round(cf, 3),
        "conservative upper bound of land (§8: water dominates)")

    comps = _components(cont_f)
    big = [c for c in comps if c >= 0.015 * frame_cells]
    add("info" if 2 <= len(big) <= 5 else "warn",
        "structural_domains", len(big),
        f"cont components >=1.5% of frame; sizes {big[:6]}")

    ring = np.zeros_like(cont_f)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    ring_land = int((cont_f & ring).sum())
    add("info", "ring_cont_cells", ring_land,
        "crust-in-ring proxy; submerged shelf at the frame is legal "
        "(land is decided by elevation + sea level)")

    band = max(2, int(0.05 * (f1 - f0)))
    hug = np.zeros_like(cont_f)
    hug[:band, :] = hug[-band:, :] = True
    hug[:, :band] = hug[:, -band:] = True
    add("info", "frame_hug_occupancy",
        round(float(cont_f[hug].mean()), 3),
        "cont occupancy of the 5% band inside the frame edge")

    add("info", "alive_plates", int(s.alive_plates))
    add("info", "active_margin_cells", int(s.active_margin.sum()))
    add("info", "passive_margin_cells", int(s.passive_margin.sum()))
    add("info", "belt_cells", int((s.belt > 0).sum()))
    add("info", "ocean_age_range_myr",
        [round(float(s.age_myr[~s.cont].min()), 0),
         round(float(s.age_myr[~s.cont].max()), 0)])
    return findings


def map_report(s, ce, er, m, seed, controls, elapsed_s):
    """M3 report: structure findings plus elevation/sea-level and
    surface-process tripwires on the delivered map (§5 names land
    fraction, elevation range, lake count, nearest-land-to-border
    explicitly)."""
    findings = _structure_findings(s)

    def add(level, name, value, note=""):
        findings.append({"level": level, "name": name,
                         "value": value, "note": note})

    h = m["h"]
    water = m["water"]
    land = ~water
    km_px = m["km_per_px"]
    size = h.shape[0]

    add("info", "sea_level_datum_m", round(float(ce["sea_level"]), 1),
        "eustatic stand on the absolute crust datum (output h is "
        "already rebased to sea level = 0)")

    lf = float(land.mean())
    add("info" if 0.15 <= lf <= 0.55 else "warn", "land_fraction",
        round(lf, 3), "§8: water dominates, ~1/3 land is the center")

    # §3a HARD: outermost ring must be water
    ring = np.zeros_like(land)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    ring_land = int((land & ring).sum())
    add("info" if ring_land == 0 else "fail", "ring_land_px", ring_land,
        "§3a hard requirement: no land touches the frame")

    ys, xs = np.nonzero(land)
    if ys.size:
        d_px = np.minimum(np.minimum(ys, size - 1 - ys),
                          np.minimum(xs, size - 1 - xs)).min()
        add("info", "nearest_land_to_border_km",
            round(float(d_px * km_px), 1),
            "§3a regression insurance (reported every run)")
    else:
        add("warn", "nearest_land_to_border_km", None, "no land at all")

    add("info" if -100.0 <= float(h.max()) <= 7500.0 or lf == 0.0
        else "warn", "elev_max_m", round(float(h.max()), 0),
        "§7i: high ground plausible, rare, clustered")
    add("info", "elev_min_m", round(float(h.min()), 0))

    if water.any():
        wet = h[water]
        shelf = float((wet > -200.0).mean())
        add("info" if shelf >= 0.06 else "warn",
            "shelf_band_fraction", round(shelf, 3),
            "§6b: fraction of water in the 0..-200 m band; no mass = "
            "no shelves")
        # §6a plummet tripwire: depth right at the coast
        shore = water & (_shift_bool(land, 0, 1) | _shift_bool(land, 0, -1)
                         | _shift_bool(land, 1, 0) | _shift_bool(land, -1, 0))
        if shore.any():
            med = float(np.median(h[shore]))
            add("info" if med > -300.0 else "warn",
                "median_coastal_depth_m", round(med, 1),
                "§6a: depth immediately offshore; plummet reads as a "
                "large negative")

    # enclosed seas at the crustal scale (M3 lakes handle the rest)
    f0, f1 = s.frame_slice
    wc = ce["water"][f0:f1, f0:f1]
    comps = _components(wc)
    ring_w = int(wc[0, :].sum() + wc[-1, :].sum()
                 + wc[:, 0].sum() + wc[:, -1].sum())
    n_open = 1 if ring_w else 0
    add("info", "enclosed_seas_coarse",
        max(len(comps) - n_open, 0),
        "crustal water bodies in-frame not counting the open ocean")

    # --- M3 surface-process findings
    e_km = er["e_km"]
    fk0 = f0 * (s.world_km / s.n)
    i0 = int(fk0 / e_km)
    i1 = int((f1 * (s.world_km / s.n)) / e_km)
    lk = er["lake_depth"][i0:i1, i0:i1] > 0
    lake_sizes = _components(lk)
    add("info" if len(lake_sizes) <= 12 else "warn",
        "lake_count", len(lake_sizes),
        "drainage-fed depressions in-frame (§9: rare structural lakes; "
        "speckle warns)")
    if lake_sizes:
        add("info", "largest_lake_km2",
            round(lake_sizes[0] * e_km * e_km, 0),
            "§9 mega-lake watch")
    ero_sum = float(er["ero"].sum())
    dep_sum = float(er["sed"].sum())
    cell_area_m2 = (e_km * 1000.0) ** 2
    source_m3 = ero_sum * cell_area_m2
    deposited_m3 = dep_sum * cell_area_m2
    exported_m3 = float(er.get("sediment_export_m3", 0.0))
    terminal_m3 = float(er.get("sediment_terminal_residual_m3", 0.0))
    if source_m3 > 0.0:
        dep_frac = deposited_m3 / source_m3
        export_frac = exported_m3 / source_m3
        terminal_frac = terminal_m3 / source_m3
        closure = (deposited_m3 + exported_m3 + terminal_m3) / source_m3
    else:
        dep_frac = export_frac = terminal_frac = 0.0
        closure = 1.0
    add("info" if abs(closure - 1.0) <= 1e-9 else "warn",
        "sediment_mass_balance", round(closure, 9),
        "(deposited + boundary export + terminal residual) / actual "
        "fluvial source; must close to 1")
    add("info", "sediment_deposited_fraction", round(dep_frac, 3),
        "fraction of actual fluvial source deposited inside the world")
    add("info", "sediment_boundary_export_fraction",
        round(export_frac, 3),
        "fraction of actual fluvial source crossing the process-world "
        "boundary")
    add("info" if terminal_frac <= 1e-9 else "warn",
        "sediment_terminal_residual_fraction", round(terminal_frac, 9),
        "flux trapped at an interior self-receiver; zero is expected")
    add("info", "sediment_boundary_export_km3",
        round(exported_m3 / 1e9, 3))
    add("info", "max_sediment_m", round(float(er["sed"].max()), 0))
    riv_frac = float(((m["riv_log"] > np.log1p(200.0))
                      & ~m["water"]).mean())
    add("info" if riv_frac > 0.0005 else "warn",
        "river_coverage", round(riv_frac, 4),
        "§7h: fraction of frame carrying major-river discharge; ~0 "
        "means rivers will not read at map scale")

    add("info", "km_per_px", round(km_px, 3))

    return {
        "name": "pipeline_b",
        "version": VERSION,
        "stage": "map (M3)",
        "seed": int(seed),
        "controls": dict(controls),
        "world_km": round(s.world_km, 1),
        "coarse_n": int(s.n),
        "eras": int(s.eras),
        "size": int(size),
        "elapsed_s": round(elapsed_s, 3),
        "timings": {k: round(float(v), 6)
                    for k, v in er.get("timings", {}).items()},
        "findings": findings,
    }


def _shift_bool(a, dy, dx):
    out = np.zeros_like(a)
    G0, G1 = a.shape
    ys, ye = max(0, -dy), G0 + min(0, -dy)
    xs, xe = max(0, -dx), G1 + min(0, -dx)
    yd, ye2 = max(0, dy), G0 + min(0, dy)
    xd, xe2 = max(0, dx), G1 + min(0, dx)
    out[yd:ye2, xd:xe2] = a[ys:ye, xs:xe]
    return out


def structure_report(s, seed, controls, elapsed_s):
    findings = _structure_findings(s)
    return {
        "name": "pipeline_b",
        "version": VERSION,
        "stage": "structure (M1)",
        "seed": int(seed),
        "controls": dict(controls),
        "world_km": round(s.world_km, 1),
        "coarse_n": int(s.n),
        "eras": int(s.eras),
        "elapsed_s": round(elapsed_s, 3),
        "findings": findings,
    }
