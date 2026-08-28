"""Determinism + contract smoke tests. Run: py -3.14 tests/smoke.py"""

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mapgen import pipeline, registry, render  # noqa: E402
from mapgen.rng import rng_for  # noqa: E402


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        sys.exit(1)


# 1. bit-identical repeat (contract section 5)
w1 = pipeline.generate(7, {}, 256)
w2 = pipeline.generate(7, {}, 256)
check("same seed -> identical elevation",
      np.array_equal(w1["elevation"], w2["elevation"]))

b1, b2 = io.BytesIO(), io.BytesIO()
render.save_png(render.hypsometric(w1), b1, w1)
render.save_png(render.hypsometric(w2), b2, w2)
check("same seed -> identical PNG bytes", b1.getvalue() == b2.getvalue())

# 2. seeds differ
w3 = pipeline.generate(8, {}, 256)
check("different seed -> different map",
      not np.array_equal(w1["elevation"], w3["elevation"]))

# 3. per-stage RNG isolation (contract section 5)
a = rng_for(7, "stage_a").integers(0, 2**32, 8)
b = rng_for(7, "stage_b").integers(0, 2**32, 8)
a2 = rng_for(7, "stage_a").integers(0, 2**32, 8)
check("stage keying: streams differ across stages", not np.array_equal(a, b))
check("stage keying: stream stable for same stage", np.array_equal(a, a2))

# 4. never-fail resolve: clamp + unknown -> findings, not exceptions
vals, finds = registry.resolve({"stub_relief_amp_m": 99999, "nope": 3})
check("out-of-range clamps with finding",
      vals["stub_relief_amp_m"] == 8000.0
      and any("clamped" in f["msg"] for f in finds))
check("unknown control ignored with finding",
      any("unknown" in f["msg"] for f in finds))

# 5. structural resolution independence (contract section 6):
# same 1024 km extent at 128 cells (8 km) and 256 cells (4 km).
lo = pipeline.generate(7, {"cell_size_km": 8.0}, 128)["elevation"]
hi = pipeline.generate(7, {"cell_size_km": 4.0}, 256)["elevation"]
hi_ds = hi.reshape(128, 2, 128, 2).mean(axis=(1, 3))
r = np.corrcoef(lo.ravel(), hi_ds.ravel())[0, 1]
check(f"same extent across resolutions correlates (r={r:.3f})", r > 0.9)

# 6. border finding present (contract section 7 regression hook)
check("border_ring finding reported",
      any(f["check"] == "border_ring" for f in w1.findings))

print("all smoke tests pass")
