"""The one fixed palette every audit panel is rendered through.

Colormap changes apparent structure, so candidate views and synthetic
calibration controls must use an identical ramp. If controls were rendered
differently a judge could separate them by colour alone and the calibration
would measure nothing.

This module is deliberately engine-independent: it maps numbers to pixels and
knows nothing about what the numbers mean.
"""

from __future__ import annotations

import numpy as np

# Perceptually ordered deep-blue to warm-white. Five stops, linearly blended.
SCALAR_RAMP = np.asarray(
    (
        (12, 16, 42),
        (36, 68, 130),
        (28, 150, 158),
        (214, 190, 92),
        (245, 238, 220),
    ),
    dtype=np.float64,
)


# One colour per categorical class. Categorical calibration controls draw from
# the same table as real class fields, for the same blinding reason.
CATEGORY_COLORS = np.asarray(
    (
        (213, 73, 91),
        (136, 76, 196),
        (242, 157, 74),
        (25, 187, 190),
        (239, 85, 181),
        (226, 201, 79),
        (100, 105, 220),
        (88, 160, 80),
        (200, 120, 60),
        (60, 140, 200),
        (170, 60, 60),
        (120, 190, 160),
        (190, 170, 220),
        (150, 110, 40),
        (40, 90, 130),
        (220, 130, 130),
    ),
    dtype=np.uint8,
)


def categorical_rgb(field: np.ndarray) -> np.ndarray:
    """Map small non-negative integer class labels onto `CATEGORY_COLORS`."""
    labels = np.asarray(field)
    if labels.min() < 0 or labels.max() >= len(CATEGORY_COLORS):
        raise ValueError("categorical labels must index CATEGORY_COLORS")
    return CATEGORY_COLORS[labels]


def scalar_rgb(field: np.ndarray) -> np.ndarray:
    """Map a scalar field onto `SCALAR_RAMP`, normalized to its own extremes.

    Per-field normalization is intentional: absolute magnitude is not the
    subject of review, spatial organization is.
    """
    values = np.asarray(field, dtype=np.float64)
    low, high = float(values.min()), float(values.max())
    unit = np.zeros_like(values) if high <= low else (values - low) / (high - low)
    position = unit * (len(SCALAR_RAMP) - 1)
    index = np.clip(position.astype(np.int64), 0, len(SCALAR_RAMP) - 2)
    blend = (position - index)[..., None]
    return (
        SCALAR_RAMP[index] * (1 - blend) + SCALAR_RAMP[index + 1] * blend
    ).astype(np.uint8)


__all__ = ["CATEGORY_COLORS", "SCALAR_RAMP", "categorical_rgb", "scalar_rgb"]
