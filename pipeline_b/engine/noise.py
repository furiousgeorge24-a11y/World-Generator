"""Seeded gradient noise sampled in world-space kilometres.

Design constraints, each traceable to a recorded failure:
- Gradient (Perlin-class), NOT value noise: bilinear value interpolation
  stamped diamond/x marks at lattice points (judge-confirmed Class A,
  s4b/s4d).
- Sampled at arbitrary km coordinates — there is no upsampling step, so
  the degenerate-column pathology of lattice resampling cannot exist
  (S4 audit), and structural resolution independence (§2) holds by
  construction: the same km position yields the same value at any grid.
- Each octave's lattice is offset by a seeded fraction of its
  wavelength, so sample grids never systematically align with lattice
  zeros (gradient noise is exactly 0 at lattice points).
- All randomness flows from an explicit 64-bit salt (see rng.stage_salt);
  no global state.
"""

import numpy as np

_M1 = np.uint64(0xff51afd7ed558ccd)
_M2 = np.uint64(0xc4ceb9fe1a85ec53)
_PX = np.uint64(0x9e3779b97f4a7c15)
_PY = np.uint64(0xc2b2ae3d27d4eb4f)
_S33 = np.uint64(33)
_TWO_PI = 2.0 * np.pi
_INV64 = 1.0 / float(1 << 64)


def _mix(h):
    # uint64 wraparound is the intended hashing behavior
    with np.errstate(over="ignore"):
        h = h ^ (h >> _S33)
        h = h * _M1
        h = h ^ (h >> _S33)
        h = h * _M2
        h = h ^ (h >> _S33)
    return h


def _lattice_hash(ix, iy, salt):
    h = ix.astype(np.uint64) * _PX
    h = h ^ (iy.astype(np.uint64) * _PY)
    h = h ^ np.uint64(salt)
    return _mix(h)


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin(x_km, y_km, wavelength_km, salt):
    """Gradient noise, roughly in [-1, 1], zero-mean, at arbitrary
    km coordinates. x_km/y_km are same-shape arrays."""
    x = np.asarray(x_km, np.float64) / wavelength_km
    y = np.asarray(y_km, np.float64) / wavelength_km
    ix0 = np.floor(x).astype(np.int64)
    iy0 = np.floor(y).astype(np.int64)
    fx = x - ix0
    fy = y - iy0
    ux = _fade(fx)
    uy = _fade(fy)

    def corner(dx, dy):
        h = _lattice_hash(ix0 + dx, iy0 + dy, salt)
        ang = h.astype(np.float64) * _INV64 * _TWO_PI
        return np.cos(ang) * (fx - dx) + np.sin(ang) * (fy - dy)

    n00 = corner(0, 0)
    n10 = corner(1, 0)
    n01 = corner(0, 1)
    n11 = corner(1, 1)
    nx0 = n00 + ux * (n10 - n00)
    nx1 = n01 + ux * (n11 - n01)
    return np.sqrt(2.0) * (nx0 + uy * (nx1 - nx0))


def fbm(x_km, y_km, base_wavelength_km, octaves, salt,
        gain=0.55, lacunarity=2.0, norm_octaves=None, first_octave=0):
    """Fractional Brownian sum of gradient octaves. Each octave gets its
    own decorrelated salt and a seeded lattice offset (fraction of its
    wavelength) so no two octaves share lattice geometry.

    norm_octaves: normalize as if this many octaves were summed. Callers
    that trim sub-pixel octaves at low resolution pass the full-stack
    count here so the octaves they DO share keep identical amplitude at
    every resolution (§2) — only invisible sub-pixel energy is lost.

    first_octave: start partway down one conceptual stack — octave
    salts, offsets, wavelengths, and amplitudes are those the full
    stack would use, so a stack may be split across stages (e.g. the
    mid band rides through the erosion solve while the fine band is
    added at output resolution) without changing what any octave is.
    """
    total = np.zeros(np.broadcast(x_km, y_km).shape, np.float64)
    amp = gain ** first_octave
    lam = float(base_wavelength_km) / (lacunarity ** first_octave)
    for o in range(first_octave, first_octave + octaves):
        osalt = int(_mix(np.uint64((salt + 0x9e37 * (o + 1)) & ((1 << 64) - 1))))
        off_x = (osalt & 0xffffffff) / 0xffffffff * lam
        off_y = ((osalt >> 32) & 0xffffffff) / 0xffffffff * lam
        total += amp * perlin(x_km + off_x, y_km + off_y, lam, osalt)
        amp *= gain
        lam /= lacunarity
    norm = 0.0
    if norm_octaves is None:
        # self-normalize over the octaves actually summed
        amp = gain ** first_octave
        for _ in range(octaves):
            norm += amp
            amp *= gain
    else:
        # normalize as the full conceptual stack (split-stack callers)
        amp = 1.0
        for _ in range(norm_octaves):
            norm += amp
            amp *= gain
    return total / norm
