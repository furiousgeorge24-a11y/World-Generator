"""Stage: boundaries — classification and tectonic feature painting.

Per boundary cell: relative Euler velocity (smoothed along the boundary),
boundary normal, and crust on each side decide the interaction (design.md
grammar). Features are painted by splatting point sources — offset along
the normal where the feature stands away from the boundary — then
convolving each feature grid with its radial Gaussian via FFT. Everything
is sized in km, so structure survives resolution changes.

C3: the four audit bugs are fixed here (km-based side sampling,
orientation-density weighting, fixed reference speed, along-boundary
smoothing); ranges decompose into massifs (width/amp co-modulation +
perpendicular jitter); volcanism placement (arc marks + hotspot chains);
flourishes: jigsaw seas via rift maturity, failed rifts, back-arc basins.
"""

import numpy as np

from .noise import fbm
from .rng import rng_for, salts_for
from .world import World

# feature -> (sigma_km, extent cap as fraction of min extent)
_SIGMA = {
    "trench": (30.0, 0.05),
    "cordillera": (65.0, 0.08),
    "cordillera_n": (36.0, 0.05),
    "crest": (22.0, 0.03),
    "plateau": (60.0, 0.07),
    "apron": (100.0, 0.10),
    "foreland": (85.0, 0.09),
    "arc_oo": (48.0, 0.06),
    "arc_comp": (150.0, 0.15),
    "graben": (20.0, 0.03),
    "graben_w": (55.0, 0.06),
    "shoulder": (46.0, 0.05),
    "axial": (22.0, 0.03),
    "fracture": (26.0, 0.03),
    "rise": (55.0, 0.06),
    "backarc": (130.0, 0.12),
    "hotspot": (30.0, 0.04),
    "hotspot_comp": (120.0, 0.12),
    "hotspot_y": (30.0, 0.04),
    "hotspot_y_comp": (120.0, 0.12),
    "arc_ped": (120.0, 0.12),
    "hotspot_ped": (110.0, 0.11),
    "hotspot_y_ped": (110.0, 0.11),
    "volcanic": (40.0, 0.05),
    "orient_c": (65.0, 0.08),
    "orient_s": (65.0, 0.08),
}

_VN_REF = 0.85   # fixed reference closing speed: same slider, same metres,
                 # in every world (audit bug 3)


def _boxblur(a: np.ndarray, r: int) -> np.ndarray:
    """Separable box blur x2 (rough Gaussian), pure numpy."""
    for _ in range(2):
        c = np.cumsum(np.pad(a, ((r + 1, r), (0, 0)), mode="edge"), axis=0)
        a = (c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)
        c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r)), mode="edge"), axis=1)
        a = (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)
    return a


def _fft_gauss(src: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Convolve a sparse source grid with a unit-peak Gaussian, padded."""
    h, w = src.shape
    pad = int(min(max(4 * sigma_cells, 8), 512))
    ph, pw = h + pad, w + pad
    y = np.arange(ph, dtype=np.float64)
    x = np.arange(pw, dtype=np.float64)
    y = np.minimum(y, ph - y)
    x = np.minimum(x, pw - x)
    ky = np.exp(-(y ** 2) / (sigma_cells ** 2))
    kx = np.exp(-(x ** 2) / (sigma_cells ** 2))
    kern = np.outer(ky, kx)
    out = np.fft.irfft2(np.fft.rfft2(src, (ph, pw)) * np.fft.rfft2(kern, (ph, pw)),
                        (ph, pw))
    return out[:h, :w]


def _smooth_along(t: np.ndarray, vn: np.ndarray, win: int) -> np.ndarray:
    """Moving average of vn in principal-axis order (audit bug 4: no
    conv/transform flicker along one boundary)."""
    order = np.argsort(t, kind="stable")
    v = vn[order]
    k = max(win | 1, 3)
    csum = np.cumsum(np.pad(v, (k // 2 + 1, k // 2), mode="edge"))
    vs = (csum[k:] - csum[:-k]) / k
    out = np.empty_like(vn)
    out[order] = vs
    return out


def stage_boundaries(world: World) -> None:
    c = world.controls
    pid = world["plate_id"]
    pot = world["crust_potential"].astype(np.float64)
    thr = float(world.meta["crust"]["threshold"])
    h, w = world.shape
    cell = world.cell_km
    meta = world.meta["plates"]
    poles = np.array(meta["poles_km"])
    omega = np.array(meta["omega"])
    eh, ew = world.extent_km
    min_e = min(eh, ew)

    # --- collect boundary cells per ordered plate pair ------------------
    pairs: dict[tuple[int, int], list[np.ndarray]] = {}
    for dr, dc in ((0, 1), (1, 0)):
        a = pid[: h - dr, : w - dc]
        b = pid[dr:, dc:]
        m = a != b
        rr, cc = np.nonzero(m)
        lo = np.minimum(a[m], b[m])
        hi = np.maximum(a[m], b[m])
        for key in np.unique(np.stack([lo, hi], 1), axis=0):
            k = (int(key[0]), int(key[1]))
            sel = (lo == k[0]) & (hi == k[1])
            pairs.setdefault(k, []).append(
                np.stack([rr[sel] + dr * 0.5, cc[sel] + dc * 0.5], 1))

    grids = {name: np.zeros((h, w)) for name in _SIGMA}
    mem = {name: np.zeros((h, w)) for name in ("youth1", "youth2", "activity")}
    stats = {"convergent": 0, "divergent": 0, "transform": 0}
    all_rows = []

    for (a, b), chunks in sorted(pairs.items()):
        pts = np.concatenate(chunks, 0)
        if len(pts) < 4:
            continue
        r0 = max(int(pts[:, 0].min()) - 14, 0)
        r1 = min(int(pts[:, 0].max()) + 15, h)
        c0 = max(int(pts[:, 1].min()) - 14, 0)
        c1 = min(int(pts[:, 1].max()) + 15, w)
        ind = ((pid[r0:r1, c0:c1] == b).astype(np.float64)
               - (pid[r0:r1, c0:c1] == a))
        ind = _boxblur(ind, 3)
        gy, gx = np.gradient(ind)
        pr = np.clip(pts[:, 0].astype(int) - r0, 0, r1 - r0 - 1)
        pc = np.clip(pts[:, 1].astype(int) - c0, 0, c1 - c0 - 1)
        nx, ny = gx[pr, pc], gy[pr, pc]           # points a -> b
        norm = np.hypot(nx, ny)
        ok = norm > 1e-12
        pts, nx, ny = pts[ok], nx[ok] / norm[ok], ny[ok] / norm[ok]
        if not len(pts):
            continue
        px = (pts[:, 1] + 0.5) * cell             # km
        py = (pts[:, 0] + 0.5) * cell
        va = np.stack([-omega[a] * (py - poles[a, 1]),
                       omega[a] * (px - poles[a, 0])], 1)
        vb = np.stack([-omega[b] * (py - poles[b, 1]),
                       omega[b] * (px - poles[b, 0])], 1)
        dv = va - vb
        vn = dv[:, 0] * nx + dv[:, 1] * ny        # >0 closing (a toward b)
        all_rows.append((a, b, pts, nx, ny, vn))

    orog = float(c["orogeny_strength"])
    pt_ctl = float(c["plateau_tendency"])
    gap = float(c["arc_gap_km"])
    arc_curv = float(c["arc_curvature"])
    seg_ctl = float(c["ridge_segmentation"])
    crest_ctl = float(c["crest_sharpness"])
    rise_ctl = float(c["outer_rise"])
    fail_ctl = float(c["failed_rifts"])
    back_ctl = float(c["backarc_basins"])
    mat_bias = float(c["rift_maturity"])
    nsalts = salts_for(world.seed, "boundaries:strike", 4)
    msalts = salts_for(world.seed, "boundaries:massif", 6)

    def splat(grid, rr, cc, amp, sigma_km):
        sig_c = max(sigma_km / cell, 1.2)
        ri = np.clip(np.round(rr).astype(int), 0, h - 1)
        ci = np.clip(np.round(cc).astype(int), 0, w - 1)
        np.add.at(grid, (ri, ci), amp / (np.sqrt(np.pi) * sig_c))

    def sample_pot(rr, cc):
        ri = np.clip(np.round(rr).astype(int), 0, h - 1)
        ci = np.clip(np.round(cc).astype(int), 0, w - 1)
        return pot[ri, ci]

    # km-based side sampling with a coarse-cell floor (audit bug 1)
    side_cells = max(45.0, 4.5 * cell) / cell

    for a, b, pts, nx, ny, vn in all_rows:
        rr, cc = pts[:, 0], pts[:, 1]
        px = (cc + 0.5) * cell
        py = (rr + 0.5) * cell
        # orientation-density weight (audit bug 2): uniform line density
        # whichever way the boundary runs
        dens = 1.0 / np.maximum(np.abs(nx) + np.abs(ny), 0.5)

        # pair geometry
        cen = pts.mean(0)
        d0 = pts - cen
        cov = d0.T @ d0 / max(len(pts), 1)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, int(np.argmax(evals))]
        t = d0 @ axis
        big_t = max(float(np.abs(t).max()), 1e-6)
        length_km = 2.0 * big_t * cell
        bulge_n = 1.0 - (t / big_t) ** 2

        vns = _smooth_along(t, vn, max(3, int(140.0 / cell)))
        v = vns / _VN_REF
        pa = sample_pot(rr - ny * side_cells, cc - nx * side_cells)
        pb = sample_pot(rr + ny * side_cells, cc + nx * side_cells)
        ca = pa >= 0.55 * thr
        cb = pb >= 0.55 * thr

        conv = v > 0.25
        div = v < -0.25
        stats["convergent"] += int(conv.sum())
        stats["divergent"] += int(div.sum())
        stats["transform"] += int((~conv & ~div).sum())
        amp = np.clip(np.abs(v), 0.0, 1.8) * dens

        # along-strike modulation + massif decomposition fields
        cmod = 0.72 + 0.5 * fbm(px, py, 420.0, nsalts[0:2])
        amod = np.clip(
            0.42 + 1.1 * (0.5 + 0.5 * fbm(px, py, 260.0, nsalts[2:4])) ** 1.6,
            0.30, 1.5)
        wmod = 0.5 + 0.5 * np.clip(0.5 + 0.5 * fbm(px, py, 300.0, msalts[0:2]),
                                   0.0, 1.0)
        narrow_frac = np.clip(1.35 * (1.0 - wmod), 0.15, 0.85)
        jit = 26.0 * fbm(px, py, 180.0, msalts[2:4]) / cell
        kmod = 0.6 + 0.7 * np.clip(0.5 + 0.5 * fbm(px, py, 240.0, msalts[4:6]),
                                   0.0, 1.0)

        tx, ty = -ny, nx                          # boundary tangent

        # --- convergent (K2: profile model) ----------------------------
        # Thickening intensity T (from closing rate) sets a *saturating*
        # crest height — the isostatic ceiling — and past a threshold the
        # orogen spreads instead of rising: fill rows build a plateau
        # floor, deformation concentrates at its rims (far rim), and the
        # load flexes the retro plate into a foreland basin. Apron rows
        # carry the wide retro flank of an unsaturated belt.
        for m, ovr in ((conv & ca & ~cb, -1.0), (conv & cb & ~ca, +1.0)):
            if m.any():
                g = gap / cell
                bow = arc_curv * 0.5 * min(0.35 * length_km, 260.0) / cell
                dr = -ovr * ny[m] * bow * bulge_n[m]
                dc = -ovr * nx[m] * bow * bulge_n[m]
                splat(grids["trench"], rr[m] + dr, cc[m] + dc,
                      -3400.0 * amp[m] * float(c["trench_depth"]),
                      _SIGMA["trench"][0])
                splat(mem["activity"], rr[m] + dr, cc[m] + dc, dens[m], 280.0)
                ro = 2.2 * _SIGMA["trench"][0] / cell
                splat(grids["rise"], rr[m] + dr - ovr * ny[m] * ro,
                      cc[m] + dc - ovr * nx[m] * ro,
                      170.0 * amp[m] * rise_ctl, _SIGMA["rise"][0])
                ar = rr[m] + dr + ovr * ny[m] * g + ny[m] * jit[m]
                ac = cc[m] + dc + ovr * nx[m] * g + nx[m] * jit[m]
                T = amp[m] * orog * cmod[m]
                Hs = 6000.0 * T / (T + 0.75)          # saturating height
                pw = np.clip((T - (1.55 - 0.95 * pt_ctl)) / 0.55, 0.0, 1.0)
                Wp = (110.0 + 250.0 * pw) / cell      # plateau span (retro)
                splat(grids["cordillera"], ar, ac,
                      0.44 * Hs * wmod[m], _SIGMA["cordillera"][0])
                splat(grids["cordillera_n"], ar, ac,
                      0.48 * Hs * narrow_frac[m], _SIGMA["cordillera_n"][0])
                splat(grids["crest"], ar, ac,
                      0.22 * Hs * kmod[m] * crest_ctl, _SIGMA["crest"][0])
                splat(grids["volcanic"], ar, ac, amp[m], _SIGMA["volcanic"][0])
                oy = ovr * ny[m]
                ox = ovr * nx[m]
                for k in (1, 2, 3):                   # plateau fill rows
                    dk = k * 75.0 / cell
                    mk = (dk <= Wp) & (pw > 0.05)
                    if mk.any():
                        splat(grids["plateau"], (ar + oy * dk)[mk],
                              (ac + ox * dk)[mk], (0.36 * Hs * pw)[mk],
                              _SIGMA["plateau"][0])
                m2 = pw > 0.05
                rim2 = Wp + 40.0 / cell               # far rim encloses it
                if m2.any():
                    splat(grids["cordillera_n"], (ar + oy * rim2)[m2],
                          (ac + ox * rim2)[m2], (0.36 * Hs * pw)[m2],
                          _SIGMA["cordillera_n"][0])
                    splat(grids["crest"], (ar + oy * rim2)[m2],
                          (ac + ox * rim2)[m2],
                          (0.16 * Hs * pw * crest_ctl)[m2],
                          _SIGMA["crest"][0])
                d_ap = np.where(m2, rim2 + 110.0 / cell, 120.0 / cell)
                splat(grids["apron"], ar + oy * d_ap, ac + ox * d_ap,
                      0.28 * Hs * (1.0 - 0.6 * pw), _SIGMA["apron"][0])
                d_fb = d_ap + 120.0 / cell            # foreland flexure
                splat(grids["foreland"], ar + oy * d_fb, ac + ox * d_fb,
                      -0.05 * Hs * rise_ctl, _SIGMA["foreland"][0])
                splat(grids["orient_c"], rr[m], cc[m],
                      (tx[m] ** 2 - ty[m] ** 2) * amp[m], _SIGMA["orient_c"][0])
                splat(grids["orient_s"], rr[m], cc[m],
                      2.0 * tx[m] * ty[m] * amp[m], _SIGMA["orient_s"][0])
        m = conv & ca & cb                          # collision (K2 profile)
        if m.any():
            splat(mem["activity"], rr[m], cc[m], dens[m], 280.0)
            jr = ny[m] * jit[m]
            jc = nx[m] * jit[m]
            T = amp[m] * orog * cmod[m]
            Hs = 6300.0 * T / (T + 0.70)            # collisions run higher
            pw = np.clip((T - (1.15 - 0.85 * pt_ctl)) / 0.5, 0.0, 1.0)
            Wp = (90.0 + 230.0 * pw) / cell         # spreads on both sides
            splat(grids["cordillera"], rr[m] + jr, cc[m] + jc,
                  0.44 * Hs * wmod[m], _SIGMA["cordillera"][0])
            splat(grids["cordillera_n"], rr[m] + jr, cc[m] + jc,
                  0.48 * Hs * narrow_frac[m], _SIGMA["cordillera_n"][0])
            splat(grids["crest"], rr[m] + jr, cc[m] + jc,
                  0.22 * Hs * kmod[m] * crest_ctl, _SIGMA["crest"][0])
            m2 = pw > 0.05
            rim2 = Wp + 40.0 / cell
            d_ap = np.where(m2, rim2 + 105.0 / cell, 115.0 / cell)
            d_fb = d_ap + 115.0 / cell
            for s in (-1.0, 1.0):
                sy = s * ny[m]
                sx = s * nx[m]
                for k in (1, 2, 3):                 # plateau fill rows
                    dk = k * 70.0 / cell
                    mk = (dk <= Wp) & m2
                    if mk.any():
                        splat(grids["plateau"], (rr[m] + jr + sy * dk)[mk],
                              (cc[m] + jc + sx * dk)[mk],
                              (0.34 * Hs * pw)[mk], _SIGMA["plateau"][0])
                if m2.any():                        # enclosing far rims
                    splat(grids["cordillera_n"], (rr[m] + sy * rim2)[m2],
                          (cc[m] + sx * rim2)[m2], (0.34 * Hs * pw)[m2],
                          _SIGMA["cordillera_n"][0])
                    splat(grids["crest"], (rr[m] + sy * rim2)[m2],
                          (cc[m] + sx * rim2)[m2],
                          (0.14 * Hs * pw * crest_ctl)[m2],
                          _SIGMA["crest"][0])
                splat(grids["apron"], rr[m] + sy * d_ap, cc[m] + sx * d_ap,
                      0.24 * Hs * (1.0 - 0.6 * pw), _SIGMA["apron"][0])
                splat(grids["foreland"], rr[m] + sy * d_fb,
                      cc[m] + sx * d_fb, -0.045 * Hs * rise_ctl,
                      _SIGMA["foreland"][0])
            splat(grids["orient_c"], rr[m], cc[m],
                  (tx[m] ** 2 - ty[m] ** 2) * amp[m], _SIGMA["orient_c"][0])
            splat(grids["orient_s"], rr[m], cc[m],
                  2.0 * tx[m] * ty[m] * amp[m], _SIGMA["orient_s"][0])
        m = conv & ~ca & ~cb                        # ocean-ocean arc
        if m.any():
            ovr = np.where(pa >= pb, -1.0, 1.0)     # higher side overrides
            g = 0.8 * gap / cell
            bow = arc_curv * min(0.35 * length_km, 260.0) / cell
            dr = -ovr[m] * ny[m] * bow * bulge_n[m]
            dc = -ovr[m] * nx[m] * bow * bulge_n[m]
            splat(grids["trench"], rr[m] + dr, cc[m] + dc,
                  -3400.0 * amp[m] * float(c["trench_depth"]),
                  _SIGMA["trench"][0])
            splat(mem["activity"], rr[m] + dr, cc[m] + dc, dens[m], 280.0)
            ro = 2.2 * _SIGMA["trench"][0] / cell
            splat(grids["rise"], rr[m] + dr - ovr[m] * ny[m] * ro,
                  cc[m] + dc - ovr[m] * nx[m] * ro,
                  170.0 * amp[m] * rise_ctl, _SIGMA["rise"][0])
            ar = rr[m] + dr + ovr[m] * ny[m] * g
            ac = cc[m] + dc + ovr[m] * nx[m] * g
            splat(grids["arc_oo"], ar, ac,
                  3450.0 * amp[m] * orog * amod[m], _SIGMA["arc_oo"][0])
            splat(grids["arc_comp"], ar, ac,
                  -930.0 * amp[m] * orog * amod[m], _SIGMA["arc_comp"][0])
            # B1 edifice anatomy: constructional pedestal — the pile's
            # submarine flanks shoal the floor around the arc line
            splat(grids["arc_ped"], ar, ac,
                  520.0 * amp[m] * orog * amod[m]
                  * float(c["edifice_pedestal"]), _SIGMA["arc_ped"][0])
            splat(grids["volcanic"], ar, ac, amp[m], _SIGMA["volcanic"][0])
            if back_ctl > 0.0:
                bk = 2.6 * g
                splat(grids["backarc"], rr[m] + dr + ovr[m] * ny[m] * bk,
                      cc[m] + dc + ovr[m] * nx[m] * bk,
                      -700.0 * amp[m] * back_ctl, _SIGMA["backarc"][0])
                splat(mem["youth1"], rr[m] + dr + ovr[m] * ny[m] * bk,
                      cc[m] + dc + ovr[m] * nx[m] * bk,
                      0.35 * dens[m] * back_ctl, 300.0)
            splat(grids["orient_c"], rr[m], cc[m],
                  (tx[m] ** 2 - ty[m] ** 2) * amp[m], _SIGMA["orient_c"][0])
            splat(grids["orient_s"], rr[m], cc[m],
                  2.0 * tx[m] * ty[m] * amp[m], _SIGMA["orient_s"][0])

        # --- divergent -------------------------------------------------
        m = div & ca & cb                           # continental rift
        if m.any():
            # maturity ladder (jigsaw seas): valley -> lake chain -> narrow
            # sea whose flooded strip follows one centerline + smooth width
            # (both coasts correlated)
            prng = rng_for(world.seed, "rift:%d:%d" % (a, b))
            stagev = prng.random() + 0.55 * (mat_bias - 0.5)
            if stagev < 0.45:                       # young valley
                splat(grids["graben"], rr[m], cc[m], -680.0 * amp[m],
                      _SIGMA["graben"][0])
                sh_amp = 430.0
            elif stagev < 0.75:                     # lake chain
                splat(grids["graben"], rr[m], cc[m], -1050.0 * amp[m],
                      _SIGMA["graben"][0])
                sh_amp = 300.0
            else:                                   # narrow sea
                splat(grids["graben"], rr[m], cc[m], -700.0 * amp[m],
                      _SIGMA["graben"][0])
                splat(grids["graben_w"], rr[m], cc[m],
                      -1500.0 * amp[m] * (0.7 + 0.6 * cmod[m]),
                      _SIGMA["graben_w"][0])
                sh_amp = 0.0
            if sh_amp > 0.0:
                g = 42.0 / cell
                for s in (-1.0, 1.0):
                    splat(grids["shoulder"], rr[m] + s * ny[m] * g,
                          cc[m] + s * nx[m] * g, sh_amp * amp[m] * orog,
                          _SIGMA["shoulder"][0])
            splat(grids["volcanic"], rr[m], cc[m], 0.5 * amp[m],
                  _SIGMA["volcanic"][0])
            splat(grids["orient_c"], rr[m], cc[m],
                  0.6 * (tx[m] ** 2 - ty[m] ** 2) * amp[m],
                  _SIGMA["orient_c"][0])
            splat(grids["orient_s"], rr[m], cc[m],
                  0.6 * 2.0 * tx[m] * ty[m] * amp[m], _SIGMA["orient_s"][0])
        m = div & ~ca & ~cb                         # ocean ridge
        if m.any():
            nm = int(m.sum())
            drr = np.zeros(nm)
            dcc = np.zeros(nm)
            if seg_ctl > 0.0 and length_km > 500.0:
                prng = rng_for(world.seed, "bnd_seg:%d:%d" % (a, b))
                seg_len = 380.0 + 270.0 * prng.random()
                n_seg = max(1, int(np.ceil(length_km / seg_len)))
                offs = ((55.0 + 70.0 * prng.random(n_seg))
                        * np.where(np.arange(n_seg) % 2 == 0, 1.0, -1.0)
                        * seg_ctl / cell)
                k = np.clip(((t[m] + big_t) / (2 * big_t) * n_seg).astype(int),
                            0, n_seg - 1)
                nmx, nmy = float(nx[m].mean()), float(ny[m].mean())
                nn = max(np.hypot(nmx, nmy), 1e-9)
                nmx, nmy = nmx / nn, nmy / nn
                drr = nmy * offs[k]
                dcc = nmx * offs[k]
                if n_seg > 1:
                    span = min(900.0, 0.35 * min_e) * seg_ctl
                    npts = max(int(span / (3.0 * cell)), 2)
                    j = np.arange(1, npts + 1) * 3.0
                    for kk in range(1, n_seg):
                        tk = -big_t + (2.0 * big_t / n_seg) * kk
                        fr = np.concatenate([cen[0] + axis[0] * tk + nmy * j,
                                             cen[0] + axis[0] * tk - nmy * j])
                        fc = np.concatenate([cen[1] + axis[1] * tk + nmx * j,
                                             cen[1] + axis[1] * tk - nmx * j])
                        splat(grids["fracture"], fr, fc,
                              np.full(fr.shape, -110.0 * seg_ctl),
                              _SIGMA["fracture"][0])
            splat(mem["youth1"], rr[m] + drr, cc[m] + dcc, dens[m], 300.0)
            splat(mem["youth2"], rr[m] + drr, cc[m] + dcc, dens[m], 900.0)
            slowm = (np.abs(v) < 0.9)[m]
            if slowm.any():
                splat(grids["axial"], (rr[m] + drr)[slowm],
                      (cc[m] + dcc)[slowm], -420.0 * amp[m][slowm],
                      _SIGMA["axial"][0])

        # --- transform on continental crust: failed-rift scars ----------
        m = (~conv & ~div) & ca & cb
        if m.any() and fail_ctl > 0.0:
            splat(grids["graben"], rr[m], cc[m],
                  -240.0 * np.clip(amp[m], 0.2, 1.0) * fail_ctl,
                  _SIGMA["graben"][0])

    # --- hotspot chains (volcanism placement) ---------------------------
    hn = int(c["hotspot_count"])
    if hn > 0:
        hrng = rng_for(world.seed, "hotspots")
        n_young = int(round(float(c["volcano_youth"]) * 3))
        for _ in range(hn):
            hx = hrng.random() * ew
            hy = hrng.random() * eh
            n_ed = 4 + int(hrng.random() * 5)
            step = 90.0 + 60.0 * hrng.random()
            amp0 = 3300.0 + 900.0 * hrng.random()
            for j in range(n_ed):
                ri = int(np.clip(hy / cell - 0.5, 0, h - 1))
                ci = int(np.clip(hx / cell - 0.5, 0, w - 1))
                i = int(pid[ri, ci])
                vx = -omega[i] * (hy - poles[i, 1])
                vy = omega[i] * (hx - poles[i, 0])
                vv = max(np.hypot(vx, vy), 1e-9)
                aj = np.array([amp0 * 0.78 ** j])
                rrj = np.array([hy / cell - 0.5])
                ccj = np.array([hx / cell - 0.5])
                # youngest cones of each chain are placed after erosion
                tgt = "hotspot_y" if j < n_young else "hotspot"
                splat(grids[tgt], rrj, ccj, aj, _SIGMA[tgt][0])
                splat(grids[tgt + "_comp"], rrj, ccj, -0.30 * aj,
                      _SIGMA[tgt + "_comp"][0])
                splat(grids[tgt + "_ped"], rrj, ccj,
                      0.16 * aj * float(c["edifice_pedestal"]),
                      _SIGMA[tgt + "_ped"][0])
                splat(grids["volcanic"], rrj, ccj, np.array([1.0]),
                      _SIGMA["volcanic"][0])
                hx += (vx / vv) * step          # older cones carried away
                hy += (vy / vv) * step

    # --- convolve ------------------------------------------------------
    fields = {}
    for name, grid in grids.items():
        sig_km, cap = _SIGMA[name]
        sig = min(sig_km, cap * min_e)
        fields[name] = (_fft_gauss(grid, max(sig / cell, 1.2))
                        if np.any(grid) else grid)

    for name, f in fields.items():
        world[f"tect_{name}"] = f.astype(np.float32)
    vol = fields["volcanic"]
    ref = float(np.percentile(vol, 99.5)) if np.any(vol) else 1.0
    world["volcanic"] = np.clip(vol / max(ref, 1e-9), 0.0, 1.0).astype(np.float32)
    pos = (fields["cordillera"] + fields["cordillera_n"] + fields["crest"]
           + fields["plateau"] + fields["apron"] + fields["arc_oo"]
           + fields["arc_comp"] + fields["arc_ped"] + fields["shoulder"]
           + fields["rise"] + fields["hotspot"] + fields["hotspot_comp"]
           + fields["hotspot_ped"])
    neg = (fields["trench"] + fields["graben"] + fields["graben_w"]
           + fields["axial"] + fields["fracture"] + fields["backarc"]
           + fields["foreland"])
    world["uplift"] = (pos + neg).astype(np.float32)

    # --- memory layers: margin activity + ocean-crust age ---------------
    def _mem_blur(grid, sig_km, cap):
        sig = min(sig_km, cap * min_e)
        return (_fft_gauss(grid, max(sig / cell, 1.2))
                if np.any(grid) else grid)

    act = np.clip(_mem_blur(mem["activity"], 280.0, 0.20), 0.0, 1.0)
    youth = (0.75 * _mem_blur(mem["youth1"], 300.0, 0.22)
             + 0.55 * _mem_blur(mem["youth2"], 900.0, 0.45))
    age = 1.0 - np.clip(youth, 0.0, 1.0)
    world["margin_activity"] = act.astype(np.float32)
    world["crust_age"] = age.astype(np.float32)

    total = max(sum(stats.values()), 1)
    world.meta["boundaries"] = {k: round(v / total, 3) for k, v in stats.items()}
    world.findings.append({"check": "boundary_shares", "level": "info",
                           "value": world.meta["boundaries"]})
