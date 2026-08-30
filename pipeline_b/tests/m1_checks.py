"""M1 structural-stage checks. Exits nonzero on failure.

    py -3.14 tests\\m1_checks.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from engine.tectonics import Config, build_structure  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


print("[determinism] same seed+config -> bit-identical fields")
t0 = time.perf_counter()
a = build_structure(11)
el = time.perf_counter() - t0
b = build_structure(11)
same = all(np.array_equal(getattr(a, f), getattr(b, f))
           for f in ("label", "cont", "age_myr", "belt", "belt_age_era",
                     "coast", "active_margin", "passive_margin"))
check("bit-identical", same)
print(f"  (build time {el:.2f}s, n={a.n}, world={a.world_km:.0f} km)")

print("\n[perf] structural stage must stay interactive-cheap")
check("under 2 s", el < 2.0, f"{el:.2f}s")

print("\n[static world invariance] plate_speed=0: 20 eras of identity")
print("  transforms must reproduce the initial state exactly — the")
print("  composed-transform guarantee (no resampling accumulation)")
s = build_structure(11, Config(plate_speed=0.0, wander=0.0))
check("label == initial partition",
      np.array_equal(s.label, s.initial_label))
check("no belts formed", float(s.belt.sum()) == 0.0,
      f"belt sum={s.belt.sum():.1f}")
check("no convergent events", not s.conv_recent.any())

print("\n[control isolation] eras must not reshuffle partition/nuclei")
e1 = build_structure(11, Config(eras=8))
e2 = build_structure(11, Config(eras=24))
check("same initial partition across eras settings",
      np.array_equal(e1.initial_label, e2.initial_label))
w1 = build_structure(11, Config(wander=0.02))
check("same initial partition across wander settings",
      np.array_equal(w1.initial_label, e1.initial_label))

print("\n[activity sanity] a moving world must actually converge/diverge")
check("belts formed", float(a.belt.sum()) > 50.0,
      f"belt sum={a.belt.sum():.0f}")
check("recent divergence exists", bool(a.div_recent.any()))
check("continental crust survives", 0.05 < a.cont.mean() < 0.6,
      f"cont fraction(world)={a.cont.mean():.2f}")
check("both margin classes occur",
      bool(a.active_margin.any()) and bool(a.passive_margin.any()),
      f"active={int(a.active_margin.sum())} "
      f"passive={int(a.passive_margin.sum())}")

print("\n[seed variety] different seeds -> different worlds")
c = build_structure(12)
check("labels differ", not np.array_equal(a.label, c.label))
check("cont differs", not np.array_equal(a.cont, c.cont))

print()
if fails:
    print("FAILURES:", ", ".join(fails))
    sys.exit(1)
print("all m1 checks passed")
