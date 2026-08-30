"""Stage-keyed randomness.

Every stage derives its own generator from (world seed, stage name), so
dragging one control never reshuffles an unrelated stage (§4). Noise
salts come from the same keying so field noise is per-stage stable too.
"""

import numpy as np

_FNV_OFFSET = 0xcbf29ce484222325
_FNV_PRIME = 0x100000001b3
_MASK64 = (1 << 64) - 1


def fnv1a64(text: str) -> int:
    h = _FNV_OFFSET
    for b in text.encode("utf-8"):
        h = ((h ^ b) * _FNV_PRIME) & _MASK64
    return h


def stage_rng(seed: int, stage: str) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(seed) & _MASK64, fnv1a64(stage)]))


def stage_salt(seed: int, stage: str) -> int:
    """64-bit salt for the noise lattice hash."""
    h = fnv1a64(f"{stage}:{int(seed)}")
    # one extra avalanche round so nearby seeds decorrelate
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & _MASK64
    h ^= h >> 33
    return h
