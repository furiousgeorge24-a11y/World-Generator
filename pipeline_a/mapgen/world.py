"""The World object (contract section 2): named same-shape 2D arrays + metadata."""

import numpy as np

from . import VERSION


class World:
    def __init__(self, seed: int, shape: tuple[int, int], controls: dict):
        self.seed = int(seed)
        self.shape = (int(shape[0]), int(shape[1]))  # (H, W)
        self.controls = dict(controls)
        self.version = VERSION
        self.layers: dict[str, np.ndarray] = {}
        self.meta: dict[str, object] = {}  # per-stage scalars (poles, thresholds)
        self.timings: dict[str, float] = {}
        self.findings: list[dict] = []
        self._coords: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def cell_km(self) -> float:
        return float(self.controls["cell_size_km"])

    @property
    def extent_km(self) -> tuple[float, float]:
        h, w = self.shape
        return (h * self.cell_km, w * self.cell_km)

    def coords_km(self) -> tuple[np.ndarray, np.ndarray]:
        """(xkm, ykm) world-space cell-center coordinates, cached."""
        if self._coords is None:
            h, w = self.shape
            y, x = np.mgrid[0:h, 0:w].astype(np.float64)
            self._coords = ((x + 0.5) * self.cell_km, (y + 0.5) * self.cell_km)
        return self._coords

    def __getitem__(self, name: str) -> np.ndarray:
        return self.layers[name]

    def __setitem__(self, name: str, arr: np.ndarray) -> None:
        if arr.shape != self.shape:
            raise ValueError(f"layer {name!r} shape {arr.shape} != {self.shape}")
        self.layers[name] = arr
