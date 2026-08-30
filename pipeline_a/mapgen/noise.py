"""Seeded lattice value noise sampled in world-space km (contract section 6).

Noise is a function of world coordinates, never of cell indices: the same
seed yields the same large-scale world at any resolution. Lattice values
come from a splitmix64-style integer hash, so there is no permutation
table and no state — evaluation order can never matter.
"""

import numpy as np

_MX = np.uint64(0x9E3779B97F4A7C15)
_MY = np.uint64(0xC2B2AE3D27D4EB4F)


def _mix(h: np.ndarray) -> np.ndarray:
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def _lattice01(ix: np.ndarray, iy: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic value in [0, 1) per integer lattice point. float32
    output: noise feeds rendering-scale fields; hash math stays exact."""
    h = ix.astype(np.int64).astype(np.uint64) * _MX
    h = h + iy.astype(np.int64).astype(np.uint64) * _MY
    h = h ^ np.uint64(salt)
    return (_mix(h) >> np.uint64(40)).astype(np.float32) * np.float32(1.0 / (1 << 24))


def value_noise(xkm, ykm, scale_km: float, salt: int) -> np.ndarray:
    """Smoothstep-interpolated lattice noise in [-1, 1], wavelength scale_km."""
    px = np.asarray(xkm, dtype=np.float64) / scale_km
    py = np.asarray(ykm, dtype=np.float64) / scale_km
    ix, iy = np.floor(px), np.floor(py)
    fx = (px - ix).astype(np.float32)
    fy = (py - iy).astype(np.float32)
    ux = fx * fx * (np.float32(3.0) - np.float32(2.0) * fx)
    uy = fy * fy * (np.float32(3.0) - np.float32(2.0) * fy)
    v00 = _lattice01(ix, iy, salt)
    v10 = _lattice01(ix + 1, iy, salt)
    v01 = _lattice01(ix, iy + 1, salt)
    v11 = _lattice01(ix + 1, iy + 1, salt)
    one = np.float32(1.0)
    v = (v00 * (one - ux) + v10 * ux) * (one - uy)         + (v01 * (one - ux) + v11 * ux) * uy
    return v * np.float32(2.0) - one


def fbm(xkm, ykm, scale_km: float, salts: list[int], gain: float = 0.5,
        lacunarity: float = 2.0) -> np.ndarray:
    """Fractal sum, one salt per octave. Output roughly in [-1, 1]."""
    total = np.zeros(np.asarray(xkm).shape, dtype=np.float32)
    amp, wl, norm = 1.0, float(scale_km), 0.0
    for salt in salts:
        total += np.float32(amp) * value_noise(xkm, ykm, wl, salt)
        norm += amp
        amp *= gain
        wl /= lacunarity
    return total / np.float32(norm)
