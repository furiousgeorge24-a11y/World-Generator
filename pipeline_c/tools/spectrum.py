"""The spectral statistics the build reports and the tests share.

Two questions about a square periodic field, and nothing else. Does its power
prefer the world's axes over its diagonals — `axis_to_diagonal`. Where does
its power sit as a function of `|k|` alone — `radial_power`, and the two
readings off it, `low_edge` and `dominant_cycles`. Nothing here knows what a
field means, and nothing in the engine imports it.

`|k|` is counted in cycles per parent axis throughout, so a radius `k` is a
wavelength of `parent_km / k` and the second question can be asked in
kilometres.
"""

from __future__ import annotations

import numpy as np

#: Half-width of each wedge, in degrees.
WEDGE_DEGREES = 15.0
#: Lowest radius counted, in cycles per parent axis. Below this a square grid
#: has too few bins per wedge for a mean to say anything.
K_MIN = 4.0


def axis_to_diagonal(field: np.ndarray) -> float:
    """Mean power density about the k-axes over the same about the diagonals.

    Wedges are +-`WEDGE_DEGREES` about 0, 90, 180 and 270 degrees for the axes
    and about 45, 135, 225 and 315 for the diagonals, over the annulus
    `K_MIN <= |k| <= n/2`. One is isotropic; above one the field's power
    leans on the axes.
    """
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("axis_to_diagonal expects a square field")
    n = values.shape[0]
    power = np.abs(np.fft.fft2(values)) ** 2
    cycles = np.fft.fftfreq(n, d=1.0 / n)
    kx = cycles[None, :]
    ky = cycles[:, None]
    k = np.sqrt(kx * kx + ky * ky)
    # Angular distance to the nearest multiple of 90 degrees, in [0, 45].
    to_axis = np.abs(((np.degrees(np.arctan2(ky, kx)) + 45.0) % 90.0) - 45.0)
    band = (k >= K_MIN) & (k <= n / 2.0)
    axis = band & (to_axis <= WEDGE_DEGREES)
    diagonal = band & (np.abs(to_axis - 45.0) <= WEDGE_DEGREES)
    if not axis.any() or not diagonal.any():
        raise ValueError("the field is too small for the measured band")
    return float(power[axis].mean() / power[diagonal].mean())


def _power_and_radius(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("a spectrum is read off a square field")
    n = values.shape[0]
    power = np.abs(np.fft.fft2(values)) ** 2
    cycles = np.fft.fftfreq(n, d=1.0 / n)
    radius = np.sqrt(cycles[:, None] ** 2 + cycles[None, :] ** 2)
    return power, radius


def radial_power(field: np.ndarray) -> np.ndarray:
    """Mean power density in each unit-wide annulus of `|k|`.

    `out[r]` is the mean of `|F|**2` over the modes whose radius rounds to
    `r`, for `r = 0 … n // 2`; an empty annulus is zero. A mean rather than a
    sum, so the reading does not simply follow the number of modes at each
    radius.
    """
    power, radius = _power_and_radius(field)
    n = power.shape[0]
    bins = np.rint(radius).astype(np.int64)
    keep = bins <= n // 2
    counts = np.bincount(bins[keep], minlength=n // 2 + 1)
    totals = np.bincount(bins[keep], weights=power[keep],
                         minlength=n // 2 + 1)
    out = np.zeros(n // 2 + 1, dtype=np.float64)
    np.divide(totals, counts, out=out, where=counts > 0)
    return out


def low_edge(field: np.ndarray, *, floor: float = 1e-12) -> float:
    """The smallest `|k|` carrying more than `floor` of the field's peak power.

    A band-limited field's modes below its band are zero up to the rounding of
    one forward and one inverse transform, which lands some thirty orders of
    magnitude below the peak in power, so any floor between the two separates
    them. This is the band's low edge as the field itself shows it.
    """
    power, radius = _power_and_radius(field)
    peak = float(power.max())
    if peak <= 0.0:
        raise ValueError("the field has no power")
    inside = power > floor * peak
    if not inside.any():
        raise ValueError("no mode is above the floor")
    return float(radius[inside].min())


def dominant_cycles(field: np.ndarray) -> int:
    """The radius, in cycles per parent axis, of the strongest annulus.

    `k = 0` is excluded: it is the field's mean, which every normalized field
    has removed anyway.
    """
    spectrum = radial_power(field)
    if spectrum.size < 2:
        raise ValueError("the field is too small for a radial spectrum")
    return int(np.argmax(spectrum[1:]) + 1)


__all__ = ["K_MIN", "WEDGE_DEGREES", "axis_to_diagonal", "dominant_cycles",
           "low_edge", "radial_power"]
