"""M2 diagnostic critique panels (EVAL.md layer 3, type 2).

Single images judged blind against the formation rubric in three
buckets (done poorly / done well / cannot identify), every claim
evidence-anchored and severity-tagged (A/B/C/D). Canon sits in the
defendant's seat: two panels are reference crops presented identically
(deduped: distinct refs, disjoint regions). Output:
  out/m2/eval/panels/panel_XX.png + rubric.md   (judges see this)
  out/m2/eval/key/m2_panels_key.json            (they never do)
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.m2_trials import bland, candidate_png, REFS

OUT = ROOT / "out" / "m2" / "eval"
PANELS = OUT / "panels"
KEY = OUT / "key"
SIZE = 1024
CAND_SEEDS = [19, 40, 101]     # disjoint from the 2AFC candidate pool
CANON = [(2, None), (10, None)]


def main():
    PANELS.mkdir(parents=True, exist_ok=True)
    KEY.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9282026)

    entries = [("cand", sd) for sd in CAND_SEEDS]
    for rid, _ in CANON:
        entries.append(("canon", rid))
    order = rng.permutation(len(entries))

    key = []
    for slot, ei in enumerate(order):
        kind, ident = entries[ei]
        if kind == "cand":
            im = candidate_png(ident)
            key.append({"panel": slot + 1, "kind": "candidate",
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

    (KEY / "m2_panels_key.json").write_text(json.dumps(key, indent=1))

    (PANELS / "rubric.md").write_text("""# Terrain formation critique

Review each `panel_XX.png` independently. Each is a terrain map
(colour encodes elevation through a stepped ramp; palettes vary by
source — palette taste is NOT a defect). Critique **formation
quality**: are the landforms the plausible footprint of natural
processes?

For each panel give exactly three buckets:
- `done_poorly`: up to 5 claims,
- `done_well`: up to 5 claims,
- `cannot_identify`: up to 3 features you cannot name or explain
  (honest unknowns beat invented stories).

Every claim must carry: `what` (the feature), `where` (location in the
image, e.g. "large island, NE quadrant"), `evidence` (what you see
that supports the claim). Praise carries the same evidence burden as
defects.

Additionally, tag every `done_poorly` claim with a severity class:
- "A" — artifact/regularity: repeated identical marks, axis alignment,
  right angles, rings/halos, even spacing/widths, straight uniform
  features, frame-correlated structure, processing artifacts (seams,
  dotted lines).
- "B" — formation implausibility: features ignoring their causes or
  surroundings (islands with no seafloor footprint, uniform shelf
  width everywhere, depth plunging at every shore, lakes/rivers with
  no drainage logic).
- "C" — character/quality: blandness, weak variety, missing anatomy.
- "D" — render/palette: colour-choice issues, not terrain defects.

Context notes (to prevent false positives): tiny bright dots on high
plateaus can be floor lakes; small volcanic islands read as
cross/star marks at pixel scale; some panels may carry day/night
darkening, projection stretching, or land running off the frame —
ignore those as source artifacts, and pixel-scale square stair-steps
on boundaries are raster quantization, judged at feature scale.

Answer in JSON: a list with one object per panel:
`{"panel": N, "done_poorly": [{"what":..., "where":..., "evidence":...,
"severity":"A|B|C|D"}], "done_well": [{"what":..., "where":...,
"evidence":...}], "cannot_identify": [{"what":..., "where":...}]}`
Do not consult anything outside the panels directory.
""")
    print(f"built {len(entries)} panels -> {PANELS}")


if __name__ == "__main__":
    main()
