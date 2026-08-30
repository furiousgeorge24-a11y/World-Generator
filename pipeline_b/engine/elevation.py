"""M2 coarse elevation: steady-state crustal surface from structure.

Every term is the footprint of a named physical process (§11), computed
on the structural lattice — output resolution never touches this stage
(§2). Sea level is eustatic: a planetary water inventory poured into
the world's hypsometry (exact solve). The S3 border architecture lives
here: the stretched outer margin of every continent subsides below sea
level, so crust legally crosses the frame as flooded shelf while only
emergent cores must stay interior.

Terms:
- Oceanic floor — plate-cooling subsidence (GDH1): depth from crust
  age. One law yields ridges, age-banded basins, and flat old abyss.
- Continental freeboard — isostatic base level of cratonic crust.
- Margins — stretched-crust subsidence profile across a margin-type
  width (passive broad, active narrow — §6c shelf variety from the M1
  margin classification, not from styling).
- Continental rise — sediment-apron shoaling of deep floor toward
  passive margins (standing in for M3's routed sediment; same process,
  crude steady-state form).
- Trenches — subduction flexure offshore active margins (§6d: sharp,
  narrow, earned by a fault).
- Orogeny — belt thickening -> isostatic elevation, saturating with
  accumulated shortening, decaying with belt age (erosional decay of
  orogens). Oceanic belts stand as arc ridges/island chains.
"""

import numpy as np

from . import noise
from .rng import stage_salt
from .tectonics import DT_MYR, _dilate

# process constants (m, km, Myr)
BASE_FREEBOARD = 450.0     # undisturbed craton surface above datum
MARGIN_DROP = 3650.0       # crustal surface drop across a stretched margin
W_ACTIVE_KM = 90.0         # stretched-zone width at active margins
RISE_KM = 260.0            # sediment-apron reach into the basin
RISE_Z = -3400.0           # apron surface it builds toward
TRENCH_DEPTH = 2600.0      # extra flexural deepening at the trench axis
TRENCH_KM = 90.0           # trench-zone reach from the margin
ARC_SCALE = 4200.0         # oceanic-belt (arc) elevation scale
BELT_SAT = 2.5             # belt intensity at which thickening saturates
OROGEN_TAU_MYR = 300.0     # erosional decay time of orogenic elevation
ARC_TAU_MYR = 110.0        # remnant arcs subside fast (thin, thermal
                           # crust) — M2 eval #5: inactive arc rings
                           # must sink, not stand as freestanding loops
UPLIFT_TAU_MYR = 60.0      # recency window of ACTIVE rock uplift (the
                           # erosion solver's U field)
Z_FLOOR = -9500.0          # plausibility clamps (§7i)
Z_CEIL = 9000.0


def _smooth01(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _chamfer_km(source, ck):
    """Distance (km) to the nearest True cell. Two-pass chamfer(1, √2):
    near-isotropic, deterministic, no axis-aligned distance contours
    (an L1 transform would stamp diamond isobands — §11a)."""
    n0, n1 = source.shape
    INF = 1.0e18
    D = np.where(source, 0.0, INF)
    R2 = np.sqrt(2.0)
    j = np.arange(n1, dtype=np.float64)

    def relax_row(v):
        # in-row unit-cost propagation both directions via accumulate
        lr = np.minimum.accumulate(v - j) + j
        rl = np.minimum.accumulate((v + j)[::-1])[::-1] - j
        return np.minimum(lr, rl)

    def from_prev(prev):
        c = prev + 1.0
        c = np.minimum(c, np.concatenate(([INF], prev[:-1])) + R2)
        c = np.minimum(c, np.concatenate((prev[1:], [INF])) + R2)
        return c

    for i in range(n0):
        r = D[i]
        if i:
            r = np.minimum(r, from_prev(D[i - 1]))
        D[i] = relax_row(r)
    for i in range(n0 - 2, -1, -1):
        D[i] = relax_row(np.minimum(D[i], from_prev(D[i + 1])))
    return D * ck


def _sea_level(z, hydrosphere_depth_m):
    """Eustatic level L with total water volume = hydrosphere_depth ×
    world area: solve sum(max(L - z, 0)) = depth × ncells exactly on the
    sorted hypsometry (piecewise-linear, closed form — no iteration)."""
    zs = np.sort(z.ravel())
    target = float(hydrosphere_depth_m) * zs.size
    cs = np.cumsum(zs)
    idx = np.arange(1, zs.size + 1, dtype=np.float64)
    vol_at = zs * idx - cs          # held volume when L == zs[k]
    k = int(np.searchsorted(vol_at, target))
    if k == 0:
        return float(zs[0])
    if k >= zs.size:
        return float((target + cs[-1]) / zs.size)
    return float((target + cs[k - 1]) / k)


def coarse_elevation(s, cfg, seed):
    """Structure -> dict of coarse world fields:
    z (absolute datum), sea_level, h (= z - sea_level; sea level 0),
    oro (orogenic contribution, for texture modulation downstream),
    d_ocean/d_cont (chamfer km), water (h < 0)."""
    cont = s.cont
    ocean = ~cont
    ck = s.world_km / s.n

    # oceanic thermal subsidence (GDH1)
    t = np.maximum(s.age_myr, 0.0)
    with np.errstate(invalid="ignore"):
        d_young = 2600.0 + 365.0 * np.sqrt(t)
    d_old = 5651.0 - 2473.0 * np.exp(-t / 62.8)
    z = -np.where(t < 20.0, d_young, d_old)

    # margin geometry
    d_ocean = _chamfer_km(ocean, ck)     # 0 on ocean cells
    d_cont = _chamfer_km(cont, ck)       # 0 on continental cells
    near_active = _dilate(s.active_margin, 4)
    W = np.where(near_active, W_ACTIVE_KM, float(cfg.passive_shelf_km))
    # margin segmentation: crustal stretching is heterogeneous along
    # strike (inherited structure, rift segmentation), so the stretched
    # width varies along the margin — bounded km-space noise as process
    # parameterization (the standing exception). Without it the width
    # is binary and the slope instrument shows uniform halos (§6c).
    xs = (np.arange(s.n) + 0.5) * ck
    Xc, Yc = np.meshgrid(xs, xs)
    seg = np.clip(noise.fbm(Xc, Yc, 900.0, 4,
                            stage_salt(seed, "margin-segmentation")),
                  -0.9, 0.9)
    W = W * (1.0 + 0.5 * seg)

    # stretched-margin subsidence profile on the continent side:
    # x = 1 deep interior, 0 at the crust edge; quadratic drop gives a
    # gently-sloping outer shelf rolling into a steeper slope (§6a/§6b).
    # Edge cells blend by sub-cell continental fraction — transitional
    # crust at the continent-ocean boundary (and the cure for the
    # lattice-aligned coast scarps: the crust edge now lands at its
    # continuous sub-cell position).
    x = _smooth01(d_ocean / W)
    zc = BASE_FREEBOARD - MARGIN_DROP * (1.0 - x) ** 2
    cf = np.clip(s.cont_frac, 0.0, 1.0)
    z = cf * zc + (1.0 - cf) * z

    # continental rise: apron shoaling toward passive margins only
    ap = _smooth01(1.0 - d_cont / RISE_KM)
    ap = np.where(near_active, 0.0, ap)
    apron = z * (1.0 - ap) + RISE_Z * ap
    z = np.where(ocean, np.maximum(z, apron), z)

    # subduction trench offshore active margins
    tr = _smooth01(1.0 - d_cont / TRENCH_KM) * near_active * ocean
    z = z - TRENCH_DEPTH * tr

    # orogeny: thickened crust stands high, decays with belt age —
    # continental roots slowly (erosional decay), oceanic arcs fast
    # (thermal subsidence of remnant arcs)
    has_belt = s.belt > 0
    age_b = np.where(has_belt,
                     (s.eras - s.belt_age_era.astype(np.float64)) * DT_MYR,
                     0.0)
    thick = 1.0 - np.exp(-s.belt / BELT_SAT)
    tau = np.where(cont, OROGEN_TAU_MYR, ARC_TAU_MYR)
    lift = thick * np.exp(-age_b / tau)
    scale = np.where(cont, cfg.orogeny_height,
                     ARC_SCALE * (cfg.orogeny_height / 4000.0))
    oro = np.where(has_belt, scale * lift, 0.0)
    z = np.clip(z + oro, Z_FLOOR, Z_CEIL)

    # active rock-uplift rate (m/Myr) for the erosion solver: only
    # recently-building belts still rise; everything else just erodes
    uplift = np.where(
        has_belt,
        (scale * thick / OROGEN_TAU_MYR) * 3.0
        * np.exp(-age_b / UPLIFT_TAU_MYR),
        0.0)

    L = _sea_level(z, cfg.hydrosphere_depth)
    h = z - L
    return {
        "z": z, "sea_level": L, "h": h, "oro": oro, "uplift": uplift,
        "d_ocean": d_ocean, "d_cont": d_cont,
        "water": h < 0.0,
    }
