"""Per-stage seeded RNG (contract section 5).

Every stage draws from its own generator keyed by (seed, stage name), so
changing one stage's controls never reshuffles the random draws of any
other stage. sha256 is used because Python's hash() is salted per-process
and would break bit-identical reproduction.
"""

import hashlib

import numpy as np


def stage_key(seed: int, stage: str) -> list[int]:
    digest = hashlib.sha256(f"{seed}:{stage}".encode("utf-8")).digest()
    return [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]


def rng_for(seed: int, stage: str) -> np.random.Generator:
    """A fresh, deterministic Generator for one stage of one run."""
    return np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(stage_key(seed, stage)))
    )


def salts_for(seed: int, stage: str, n: int = 1) -> list[int]:
    """Deterministic uint64 salts for hash-lattice noise, keyed per stage."""
    r = rng_for(seed, stage + ":salt")
    return [int(v) for v in r.integers(0, 2**63, size=n, dtype=np.uint64)]
