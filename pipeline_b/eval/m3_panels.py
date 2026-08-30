"""M3 critique panels (EVAL.md layer 3, type 2): 3 candidates + 2
canon defendants + 1 DELIBERATE DUPLICATE of a candidate (test-retest
reliability probe, EVAL pending item).
  out/m3/eval/panels/… + key
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.m2_trials import REFS, bland
from eval.m3_trials import candidate_png

OUT = ROOT / "out" / "m3" / "eval"
PANELS = OUT / "panels"
KEY = OUT / "key"
SIZE = 1024
CAND_SEEDS = [19, 40, 101]
CANON = [1, 6]
DUP_OF = 40


def main():
    PANELS.mkdir(parents=True, exist_ok=True)
    KEY.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(30082026)

    entries = [("cand", sd) for sd in CAND_SEEDS]
    entries += [("canon", r) for r in CANON]
    entries.append(("dup", DUP_OF))
    order = rng.permutation(len(entries))

    cache = {}
    key = []
    for slot, ei in enumerate(order):
        kind, ident = entries[ei]
        if kind in ("cand", "dup"):
            if ident not in cache:
                cache[ident] = candidate_png(ident)
            im = cache[ident]
            key.append({"panel": slot + 1,
                        "kind": "candidate" if kind == "cand"
                        else "duplicate_of_candidate",
                        "seed": ident})
        else:
            ref = Image.open(REFS / f"ref{ident}.png").convert("RGB")
            w, h = ref.size
            for _ in range(40):
                x = int(rng.integers(0, w - SIZE + 1))
                y = int(rng.integers(0, h - SIZE + 1))
                im = ref.crop((x, y, x + SIZE, y + SIZE))
                if not bland(im):
                    break
            key.append({"panel": slot + 1, "kind": "canon",
                        "ref": f"ref{ident}@{x},{y}"})
        im.save(PANELS / f"panel_{slot + 1:02d}.png")

    (KEY / "m3_panels_key.json").write_text(json.dumps(key, indent=1))
    src = (ROOT / "out" / "m2" / "eval" / "panels"
           / "rubric.md").read_text()
    (PANELS / "rubric.md").write_text(src.replace(
        "each `panel_XX.png`", "each `panel_XX.png` (there are 6)"))
    print(f"built {len(entries)} panels -> {PANELS}")


if __name__ == "__main__":
    main()
