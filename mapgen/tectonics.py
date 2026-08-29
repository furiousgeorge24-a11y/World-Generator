"""Stage: plates — partition + Euler motion.

Plates are an anisotropic, multi-site, noise-warped weighted-Voronoi
partition in world-space km. Motion is Euler-style: each plate rotates
about a pole, so relative velocity varies along every boundary.

Coastlines and plate boundaries are *correlated, not coincident* (design.md
decision, amended by the A1 run): the crust stage anchors continental cores
to plate interiors through `sample_plate_affinity`, an analytic point
evaluation of the same warped metric that built the raster partition — so
placement is resolution-independent by construction, and continents still
spill across boundaries because only cluster centers are biased.

`make_plates` builds one full configuration for a given key — the current
world uses key "now"; the eras stage builds ancient configurations with
their own keys (design option C).
"""

import numpy as np

from .noise import fbm
from .rng import rng_for, salts_for
from .world import World


def _score_sites(sites, lam, theta, rates, xw, yw):
    """Owner plate, best score, and best score from any *other* plate,
    under the anisotropic weighted metric. Works on grids and on 1-D
    candidate batches alike — the analytic core shared by the raster
    partition and point sampling."""
    best = np.full(np.shape(xw), np.inf)
    second = np.full(np.shape(xw), np.inf)
    pid = np.zeros(np.shape(xw), dtype=np.int16)
    for i, sx, sy in sites:
        i = int(i)
        dx, dy = xw - sx, yw - sy
        ct, st = np.cos(theta[i]), np.sin(theta[i])
        u = ct * dx + st * dy
        v = -st * dx + ct * dy
        score = ((u / lam[i]) ** 2 + (v * lam[i]) ** 2) / rates[i] ** 2
        take = score < best
        demote = take & (pid != i)          # old best belongs to another plate
        second[demote] = best[demote]
        best[take] = score[take]
        m2 = ~take & (score < second) & (pid != i)
        second[m2] = score[m2]
        pid[take] = i
    return pid, best, second


def _interior_of(best, second, spacing: float) -> np.ndarray:
    """Interiority weight in [0, 1): 0 at plate boundaries, saturating
    toward 1 deep inside a plate. Metric margin, not raster distance."""
    margin = np.sqrt(second) - np.sqrt(best)   # inf second -> weight 1
    return 1.0 - np.exp(-margin / (0.30 * spacing))


def make_plates(world: World, key: str):
    """Partition + Euler motion for one plate configuration.

    Returns (pid, seeds, poles, omega, interior, params); params carries
    the full site metric so points can be scored analytically later."""
    c = world.controls
    n = int(c["plate_count"])
    rng = rng_for(world.seed, "plates:" + key)
    xkm, ykm = world.coords_km()
    eh, ew = world.extent_km
    min_e = min(eh, ew)
    spacing = min_e / max(np.sqrt(n), 1.0)

    seeds = rng.random((n, 2)) * np.array([ew, eh])          # (x, y)
    rates = rng.lognormal(0.0, 0.35, n)                      # size variety
    ang = rng.random(n) * 2.0 * np.pi
    pole_d = min_e * (0.3 + 2.7 * rng.random(n))             # near=rotational
    poles = seeds + np.stack([np.cos(ang), np.sin(ang)], 1) * pole_d[:, None]
    omega = (np.where(rng.random(n) < 0.5, -1.0, 1.0)
             * rng.lognormal(0.0, 0.3, n) / pole_d)          # ~unit speeds

    # anisotropic, multi-site plates: elongated metrics and extra sites
    # kill the Apollonius-circle microplates
    aniso = float(c["plate_anisotropy"])
    lam = 1.0 + (0.3 + 1.3 * rng.random(n)) * aniso
    theta = rng.random(n) * np.pi
    n_sites = rng.integers(1, 4, n)
    sites: list[tuple[int, float, float]] = []
    for i in range(n):
        sites.append((i, float(seeds[i, 0]), float(seeds[i, 1])))
        for _ in range(int(n_sites[i]) - 1):
            a2 = rng.random() * 2.0 * np.pi
            d2 = 0.45 * spacing * (0.4 + rng.random())
            sites.append((i, float(seeds[i, 0] + np.cos(a2) * d2),
                          float(seeds[i, 1] + np.sin(a2) * d2)))

    # boundary raggedness: domain-warp the partition query point
    s = salts_for(world.seed, "plates:" + key, 6)
    amp = float(c["plate_raggedness"]) * 0.35 * spacing
    wscale = 1.2 * spacing
    xw = xkm + amp * fbm(xkm, ykm, wscale, s[0:3])
    yw = ykm + amp * fbm(xkm, ykm, wscale, s[3:6])

    pid, best, second = _score_sites(sites, lam, theta, rates, xw, yw)
    interior = _interior_of(best, second, spacing)

    params = {"sites": sites, "lam": lam.tolist(), "theta": theta.tolist(),
              "rates": rates.tolist(), "warp_amp": amp, "warp_scale": wscale,
              "salts": s, "spacing": spacing}
    return pid, seeds, poles, omega, interior, params


def sample_plate_affinity(world: World, xq, yq):
    """Plate id + interiority weight at world-km query points, for the
    "now" configuration. Same warp, same metric as the raster partition,
    evaluated analytically — identical at every raster resolution."""
    p = world.meta["plates"]["sampling"]
    salts = p["salts"]
    xw = xq + p["warp_amp"] * fbm(xq, yq, p["warp_scale"], salts[0:3])
    yw = yq + p["warp_amp"] * fbm(xq, yq, p["warp_scale"], salts[3:6])
    lam = np.asarray(p["lam"])
    theta = np.asarray(p["theta"])
    rates = np.asarray(p["rates"])
    pid, best, second = _score_sites(p["sites"], lam, theta, rates, xw, yw)
    return pid, _interior_of(best, second, p["spacing"])


def stage_plates(world: World) -> None:
    pid, seeds, poles, omega, interior, params = make_plates(world, "now")
    world["plate_id"] = pid
    world["plate_interior"] = interior.astype(np.float32)
    xkm, ykm = world.coords_km()
    n = len(seeds)

    # per-cell velocity: v = omega x (p - pole)
    vx = np.zeros(world.shape, dtype=np.float32)
    vy = np.zeros(world.shape, dtype=np.float32)
    for i in range(n):
        m = pid == i
        vx[m] = (-omega[i] * (ykm[m] - poles[i, 1])).astype(np.float32)
        vy[m] = (omega[i] * (xkm[m] - poles[i, 0])).astype(np.float32)
    world["plate_vx"] = vx
    world["plate_vy"] = vy

    world.meta["plates"] = {
        "count": n,
        "seeds_km": seeds.tolist(),
        "poles_km": poles.tolist(),
        "omega": omega.tolist(),
        "rates": params["rates"],
        "sampling": params,
    }
