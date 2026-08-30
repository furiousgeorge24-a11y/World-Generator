"""Stage: eras — pseudo-history (design.md, tectonic option C).

Each ancient era is an independent plate configuration whose *convergent*
boundaries imprint blunted, broadened orogenic belts — but only where
today's continental crust exists (the oceans of a dead era are gone).
Deeper eras are more worn. Option B (time-stepped simulation) stays
rejected; this buys overprinted, aged interior ranges at snapshot prices.

The pair-collection here is a deliberately light copy of the boundaries
stage's (positions, normals, closing rate only — no features).
"""

import numpy as np

from .boundaries import _boxblur, _fft_gauss
from .noise import fbm
from .rng import salts_for
from .tectonics import make_plates
from .world import World

_BELT_SIGMA_KM = 150.0


def stage_eras(world: World) -> None:
    c = world.controls
    n_eras = int(c["era_count"])
    h, w = world.shape
    if n_eras <= 1:
        world["tect_era_belt"] = np.zeros(world.shape, dtype=np.float32)
        return

    pot = world["crust_potential"].astype(np.float64)
    thr = float(world.meta["crust"]["threshold"])
    cell = world.cell_km
    min_e = min(world.extent_km)
    orog = float(c["orogeny_strength"])
    sig_c = max(min(_BELT_SIGMA_KM, 0.10 * min_e) / cell, 1.2)
    belt = np.zeros((h, w))

    for era in range(1, n_eras):
        pid, seeds, poles, omega, _, _ = make_plates(world, f"era{era}")

        pairs: dict[tuple[int, int], list[np.ndarray]] = {}
        for dr, dc in ((0, 1), (1, 0)):
            a = pid[: h - dr, : w - dc]
            b = pid[dr:, dc:]
            m = a != b
            rr, cc2 = np.nonzero(m)
            lo = np.minimum(a[m], b[m])
            hi = np.maximum(a[m], b[m])
            for key in np.unique(np.stack([lo, hi], 1), axis=0):
                k = (int(key[0]), int(key[1]))
                sel = (lo == k[0]) & (hi == k[1])
                pairs.setdefault(k, []).append(
                    np.stack([rr[sel] + dr * 0.5, cc2[sel] + dc * 0.5], 1))

        rows = []
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
            nx, ny = gx[pr, pc], gy[pr, pc]
            norm = np.hypot(nx, ny)
            ok = norm > 1e-12
            pts, nx, ny = pts[ok], nx[ok] / norm[ok], ny[ok] / norm[ok]
            if not len(pts):
                continue
            px = (pts[:, 1] + 0.5) * cell
            py = (pts[:, 0] + 0.5) * cell
            va = np.stack([-omega[a] * (py - poles[a, 1]),
                           omega[a] * (px - poles[a, 0])], 1)
            vb = np.stack([-omega[b] * (py - poles[b, 1]),
                           omega[b] * (px - poles[b, 0])], 1)
            dv = va - vb
            vn = dv[:, 0] * nx + dv[:, 1] * ny
            dens = 1.0 / np.maximum(np.abs(nx) + np.abs(ny), 0.5)
            rows.append((pts, px, py, vn, dens))

        if not rows:
            continue
        salts = salts_for(world.seed, f"eras:{era}", 4)
        blunt = 0.62 ** era
        rag = float(c["belt_raggedness"])

        for pts, px, py, vn, dens in rows:
            v = vn / 0.85                       # fixed reference speed
            m = v > 0.25
            if not m.any():
                continue
            ri = np.clip(np.round(pts[:, 0]).astype(int), 0, h - 1)
            ci = np.clip(np.round(pts[:, 1]).astype(int), 0, w - 1)
            m &= pot[ri, ci] >= 0.75 * thr      # today's continents only
            if not m.any():
                continue
            cmod = 0.55 + 0.75 * fbm(px, py, 470.0, salts[0:2])
            # tier 2: ancient shortening varied along strike too — worn
            # belts fragment into saddles and massifs, not even ribbons
            tvar = np.clip(1.0 + 1.0 * rag
                           * fbm(px, py, 520.0, salts[2:4]), 0.05, 2.2)
            amps = (780.0 * np.clip(v, 0.0, 1.8) * dens * orog * blunt
                    * cmod * tvar / (np.sqrt(np.pi) * sig_c))
            np.add.at(belt, (ri[m], ci[m]), amps[m])

    world["tect_era_belt"] = (_fft_gauss(belt, sig_c)
                              if np.any(belt) else belt).astype(np.float32)
