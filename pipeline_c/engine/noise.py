"""Periodic isotropic noise sourced from the stateless address sampler.

This is the one sanctioned stochastic input. Its causal role is mantle and
lithosphere heterogeneity, and nothing else in the engine draws a random
number. Cell values are addressed by their physical position in metres, so
traversal order, chunking, and array shape cannot reroll anything.

The field is built as white noise shaped by a **radial** spectral envelope
rather than as a sum of interpolated lattices. A lattice of `nodes` nodes
carries its own square grid: its power sits on the axes at multiples of the
node spacing and its smoothstep interpolation reinforces them, so every field
drawn from it is biased toward the world's axes and everything downstream
inherits that bias. An envelope that depends only on `|k|` has no axis to
prefer. The band and the amplitude fall-off are the same as the lattice sum's
— `k**-1` over `nodes_coarsest` to `nodes_coarsest * 2**(octaves - 1)`, which
is the old amplitude halving per octave — so the two share a spectrum and
differ only in isotropy.

**The band may be given in kilometres.** `nodes_coarsest` is a cycle count
per parent axis and so is a fraction of whatever world asks for it; a
`wavelength_km` is a length, and the cycle count it implies,
`geometry.parent_km / wavelength_km`, follows the world's size. `DESIGN.md`
§2 says the physics stays in kilometres while only the grid follows scale, so
every caller inside the history now passes a wavelength and the cycle count
comes out a float.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from .geometry import WorldGeometry
from .sampler import StageSampler

#: Guards the logarithm of the first Box–Muller uniform. `unit_float` returns
#: a value in `[0, 1)`, so exactly zero is possible and would give an infinite
#: normal. One unit in the last place of the 53-bit draw is the smallest
#: non-zero value it can return, so this replaces zero with it and nothing
#: else.
_UNIFORM_FLOOR = 2.0**-53


@lru_cache(maxsize=32)
def _white_normal(world_id: str, stage_id: str, stage_version: str,
                  process_id: str, channel: int, n: int,
                  cell_m: int) -> np.ndarray:
    """Standard normal white noise, one value per cell, from the sampler.

    Two hashes per cell: a Box–Muller pair of uniforms at `index = 0` and
    `index = 1` on the same address, of which the first normal is kept. The
    result is cached because a world asks for the same white field repeatedly
    — the drive draws six channels and the strength one, and a sweep reruns
    the same seeds — and re-deriving it is pure repeated hashing.

    The returned array is not writeable: it is shared with every later caller.
    """
    sampler = StageSampler(world_id, stage_id, stage_version, process_id)
    centres = [cell_m * index + cell_m // 2 for index in range(n)]
    first = sampler.unit_float_lattice(centres, centres, channel=channel,
                                       index=0)
    second = sampler.unit_float_lattice(centres, centres, channel=channel,
                                        index=1)
    radius = np.sqrt(-2.0 * np.log(np.maximum(first, _UNIFORM_FLOOR)))
    white = radius * np.cos(2.0 * math.pi * second)
    white.flags.writeable = False
    return white


@lru_cache(maxsize=16)
def _radial_envelope(n: int, nodes_coarsest: float, octaves: int) -> np.ndarray:
    """`k**-1` inside the band the octaves span, zero outside and at `k = 0`.

    `kx` and `ky` are whole cycle counts per parent axis, so `k` is a radius
    in cycles per world and the envelope is a function of that radius alone.

    `nodes_coarsest` is the band's low edge as a cycle count per parent axis
    and is a float: a wavelength in kilometres rarely divides a world a whole
    number of times. Neither edge is clamped to the grid.

    - **An octave the grid cannot hold is simply absent.** `high` may sit
      above the Nyquist radius `n / 2`; the modes it names do not exist on
      this grid, so the envelope has nowhere to put them and the band ends at
      whatever the torus has. That is not an error: it is a world too coarse
      to carry the finest octave, and refusing it would make the same physics
      illegal at one resolution and legal at another.
    - **A band below one cycle is a piece of a larger cell.** A wavelength
      longer than the parent gives `low < 1`, so the band holds only the
      lowest modes the torus has and the world sees part of one mantle cell
      rather than a whole one. That is the intended behaviour at small
      worlds.
    - **`k = 0` is always outside.** The validation in `periodic_noise`
      guarantees `low > 0`, so the mean mode never enters the band and the
      field has no constant part to remove.
    """
    cycles = np.fft.fftfreq(n, d=1.0 / n)
    k = np.sqrt(cycles[:, None] ** 2 + cycles[None, :] ** 2)
    low = float(nodes_coarsest)
    high = float(nodes_coarsest * 2 ** (octaves - 1))
    inside = (k >= low) & (k <= high)
    envelope = np.zeros((n, n), dtype=np.float64)
    np.divide(1.0, k, out=envelope, where=inside)
    envelope.flags.writeable = False
    return envelope


def band_cycles(geometry: WorldGeometry, *, nodes_coarsest: float | None = None,
                wavelength_km: float | None = None) -> float:
    """The band's low edge as a cycle count per parent axis.

    Exactly one of the two keywords is given. `nodes_coarsest` is that count
    already and may be a float; `wavelength_km` is a length and becomes
    `geometry.parent_km / wavelength_km`, which is what makes the band a
    physical size rather than a fraction of whatever world asks for it.
    """
    if (nodes_coarsest is None) == (wavelength_km is None):
        raise ValueError(
            "give exactly one of nodes_coarsest and wavelength_km")
    if wavelength_km is not None:
        if isinstance(wavelength_km, bool):
            raise ValueError("wavelength_km must be a number, not a bool")
        wavelength_km = float(wavelength_km)
        if not wavelength_km > 0.0:
            raise ValueError("wavelength_km must be positive")
        return float(geometry.parent_km) / wavelength_km
    if isinstance(nodes_coarsest, bool):
        raise ValueError("nodes_coarsest must be a number, not a bool")
    return float(nodes_coarsest)


def periodic_noise(sampler: StageSampler, geometry: WorldGeometry, *,
                   channel: int, nodes_coarsest: float | None = None,
                   wavelength_km: float | None = None,
                   octaves: int) -> np.ndarray:
    """A seamless `(n, n)` field with zero mean and unit standard deviation.

    Periodic by construction (it is an inverse FFT of a discrete spectrum),
    isotropic by construction (the envelope depends only on `|k|`), and
    deterministic (the white field is addressed in metres and NumPy's FFT is
    deterministic on one machine).

    The band's low edge is given **either** as `nodes_coarsest`, a cycle count
    per parent axis, **or** as `wavelength_km`, a length; exactly one of the
    two, and passing both or neither is an error. A wavelength is converted to
    a cycle count by `band_cycles` and the count is a float from there on.
    """
    nodes = band_cycles(geometry, nodes_coarsest=nodes_coarsest,
                        wavelength_km=wavelength_km)
    if nodes <= 0.0 or octaves < 1:
        raise ValueError("the band's low edge and octaves must be positive")
    n = geometry.history_n
    white = _white_normal(sampler.world_id, sampler.stage_id,
                          sampler.stage_version, sampler.process_id,
                          int(channel), n, geometry.cell_m)
    envelope = _radial_envelope(n, nodes, octaves)
    field = np.fft.ifft2(np.fft.fft2(white) * envelope).real

    field -= field.mean()
    spread = float(field.std())
    if spread <= 0.0:
        raise ValueError("noise field has no variation")
    return field / spread


__all__ = ["band_cycles", "periodic_noise"]
