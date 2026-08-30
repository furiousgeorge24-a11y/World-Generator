"""S4c: diagnostic critique panels — trial type 2 of the evaluation
harness (author-adopted 2026-08-29).

Each panel is ONE map image reviewed on its own against the formation
rubric (never "diff vs a reference"): the judge returns three buckets —
done poorly / done well / cannot identify — with per-claim evidence
anchors. Praise carries the same evidence burden as defects. The
cannot-identify bucket is honest-unknown space, not failure.

Calibration: some panels secretly hold reference crops (canon in the
defendant's seat). A judge's harshness on those calibrates their
baseline; provenance framing in the judge prompt stays neutral.

    py -3.14 spikes\\s4c_critique.py
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "s4c"
spec = importlib.util.spec_from_file_location(
    "s4b", os.path.join(_here, "s4b_imposter_1024.py"))
s4b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s4b)

from PIL import Image  # noqa: E402  (after s4b import for consistency)

N_CAND = 4
N_CANON = 2
CAND_SEEDS = [31, 32, 33, 34]


def main():
    rng = np.random.default_rng(np.random.SeedSequence([13, 0x5B54]))
    panels_dir = os.path.join(_here, "..", "out", "spikes", f"{TAG}_panels")
    os.makedirs(panels_dir, exist_ok=True)
    for f in os.listdir(panels_dir):
        os.remove(os.path.join(panels_dir, f))

    entries = []
    for seed in CAND_SEEDS:
        t0 = time.perf_counter()
        h, F, A8, _ = s4b.s2.run(seed, s4b.TILE, 24)
        entries.append(("cand", f"s2 seed {seed}",
                        s4b.render_candidate(h, F, A8)))
        print(f"candidate seed {seed}: {time.perf_counter() - t0:.0f}s")

    refs = [Image.open(os.path.join(s4b.REFS_DIR, f)).convert("RGB")
            for f in s4b.REF_OK]
    names = list(s4b.REF_OK)
    # dedupe canon picks: distinct refs per run (EVAL.md panel hygiene;
    # the first run drew overlapping crops of one ref)
    picks = rng.permutation(len(refs))[:N_CANON]
    for i in picks:
        entries.append(("canon", names[int(i)],
                        s4b.crop_scored(refs[int(i)], rng)))

    order = rng.permutation(len(entries))
    answers = []
    for slot, idx in enumerate(order, start=1):
        kind, src, im = entries[int(idx)]
        im.save(os.path.join(panels_dir, f"panel_{slot:02d}.png"))
        answers.append({"panel": slot, "kind": kind, "source": src})
        print(f"panel {slot:02d}: {kind}")

    with open(os.path.join(_here, "..", "out", "spikes",
                           f"{TAG}_answers.json"), "w") as f:
        json.dump(answers, f, indent=1)
    print("panels :", os.path.abspath(panels_dir))


if __name__ == "__main__":
    main()
