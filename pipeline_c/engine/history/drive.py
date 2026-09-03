"""The mantle's basal traction on the lithosphere.

The drive is the sum of a curl-free part and a divergence-free part, each the
derivative of a periodic noise potential, and it drifts through a few
keyframes over the history. It is the only field in the stage that is written
rather than solved, and its causal role — mantle heterogeneity — is stated in
`DESIGN.md` §3.1.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..domain import grad, perp_grad
from ..geometry import WorldGeometry
from ..noise import periodic_noise
from ..sampler import StageSampler
from .constants import (
    DRIVE_KEYFRAMES,
    DRIVE_OCTAVES,
    DRIVE_RMS_KM_PER_MYR,
    DRIVE_ROT_RATIO,
    DRIVE_WAVELENGTH_KM,
    HISTORY_MYR,
    STAGE_ID,
    STAGE_VERSION,
)

DRIVE_PROCESS_ID = "mantle-drive"


def _raw_field(phi: np.ndarray, psi: np.ndarray,
               rot_ratio: float = DRIVE_ROT_RATIO) -> np.ndarray:
    """Traction in per-cell potential units, before the world's scaling."""
    return grad(phi) + rot_ratio * perp_grad(psi)


def _rms_speed(field: np.ndarray) -> float:
    return float(np.sqrt(np.mean(field[0] ** 2 + field[1] ** 2)))


@dataclass(frozen=True, slots=True)
class Drive:
    geometry: WorldGeometry
    phi: np.ndarray    # (DRIVE_KEYFRAMES, n, n) curl-free potentials
    psi: np.ndarray    # (DRIVE_KEYFRAMES, n, n) rotational potentials
    scale: float       # one scalar per world, fixed by keyframe 0
    rot_ratio: float = DRIVE_ROT_RATIO      # rotational part, relative
    history_myr: float = HISTORY_MYR        # the span the keyframes cover

    def potentials(self, t_myr: float) -> tuple[np.ndarray, np.ndarray]:
        """The two potentials at `t_myr`, cosine-blended between keyframes."""
        span = self.history_myr / (DRIVE_KEYFRAMES - 1)
        t = min(max(float(t_myr), 0.0), self.history_myr)
        first = min(int(t / span), DRIVE_KEYFRAMES - 2)
        second = first + 1
        s = (t - first * span) / span
        w = 0.5 * (1.0 - math.cos(math.pi * s))
        return (
            (1.0 - w) * self.phi[first] + w * self.phi[second],
            (1.0 - w) * self.psi[first] + w * self.psi[second],
        )

    def field(self, t_myr: float) -> np.ndarray:
        """(2, n, n) basal traction in km/Myr."""
        phi_t, psi_t = self.potentials(t_myr)
        return self.scale * _raw_field(phi_t, psi_t, self.rot_ratio)


def build_drive(geometry: WorldGeometry, *,
                wavelength_km: float = DRIVE_WAVELENGTH_KM,
                rot_ratio: float = DRIVE_ROT_RATIO,
                history_myr: float = HISTORY_MYR) -> Drive:
    """The mantle drive for one world.

    The three keyword arguments are the drive's share of `HistoryParams`;
    their defaults are the constants and are what production uses.

    `wavelength_km` is the coarsest mantle wavelength as a length, so the same
    number means the same size of mantle cell at every resolution and scale
    and the drive obeys `DESIGN.md` §2. A smaller world holds fewer cells of
    that size, not smaller cells.
    """
    sampler = StageSampler(geometry.world_id, STAGE_ID, STAGE_VERSION,
                           DRIVE_PROCESS_ID)
    n = geometry.history_n
    phi = np.empty((DRIVE_KEYFRAMES, n, n), dtype=np.float64)
    psi = np.empty((DRIVE_KEYFRAMES, n, n), dtype=np.float64)
    for keyframe in range(DRIVE_KEYFRAMES):
        phi[keyframe] = periodic_noise(
            sampler, geometry, channel=2 * keyframe,
            wavelength_km=wavelength_km, octaves=DRIVE_OCTAVES)
        psi[keyframe] = periodic_noise(
            sampler, geometry, channel=2 * keyframe + 1,
            wavelength_km=wavelength_km, octaves=DRIVE_OCTAVES)

    reference = _rms_speed(_raw_field(phi[0], psi[0], rot_ratio))
    if reference <= 0.0:
        raise ValueError("drive potentials produced a motionless field")
    return Drive(geometry=geometry, phi=phi, psi=psi,
                 scale=DRIVE_RMS_KM_PER_MYR / reference,
                 rot_ratio=rot_ratio, history_myr=float(history_myr))


__all__ = ["DRIVE_PROCESS_ID", "Drive", "build_drive"]
