"""S4 spike: evaluation-harness seed — the blind imposter protocol.

Question: does blind two-alternative forced-choice (2AFC) discrimination
against the canon work end-to-end — trials buildable, fresh judges able
to run them, scores meaningful?

Builds side-by-side trial images: each pair is either canon-vs-candidate
(candidate = the S2 erosion spike render, the only terrain pipeline_b
can produce today) or canon-vs-canon (calibration: no signal, so judge
choices should split ~50/50 and confidence claims can be audited).
Tiles are grayscaled and autocontrast-normalized so judgments rest on
formation, not palette or the references' day/night lighting.

Answers are written OUTSIDE the trials directory; judges get only the
trials. Scoring happens back in the orchestrating session.

    py -3.14 spikes\\s4_imposter.py [--trials 12] [--tile 160]
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageOps

_here = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_here, "..", ".."))
REFS = os.path.join(REPO, "examples")
CAND = os.path.join(_here, "..", "out", "spikes",
                    "s2_erosion_512_24steps.png")


def load_gray(path):
    im = Image.open(path).convert("L")
    return ImageOps.autocontrast(im, cutoff=1)


def crop(im, rng, tile, margin_frac=0.08):
    w, h = im.size
    mx, my = int(w * margin_frac), int(h * margin_frac)
    x = int(rng.integers(mx, max(mx + 1, w - mx - tile)))
    y = int(rng.integers(my, max(my + 1, h - my - tile)))
    return im.crop((x, y, x + tile, y + tile))


def make_trial(left, right, tile, gap=10):
    im = Image.new("L", (tile * 2 + gap, tile + 18), 235)
    im.paste(left, (0, 18))
    im.paste(right, (tile + gap, 18))
    d = ImageDraw.Draw(im)
    d.text((4, 2), "L", fill=0)
    d.text((tile + gap + 4, 2), "R", fill=0)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--tile", type=int, default=160)
    ap.add_argument("--calib_every", type=int, default=3,
                    help="every Nth trial is canon-vs-canon calibration")
    args = ap.parse_args()

    rng = np.random.default_rng(np.random.SeedSequence([9, 0x5B54]))
    refs = sorted(os.listdir(REFS))
    refs = [os.path.join(REFS, f) for f in refs if f.lower().endswith(".png")]
    assert refs, "no reference images found"
    assert os.path.exists(CAND), "run s2 first (candidate render missing)"
    cand_im = load_gray(CAND)

    trials_dir = os.path.join(_here, "..", "out", "spikes", "s4_trials")
    os.makedirs(trials_dir, exist_ok=True)
    for f in os.listdir(trials_dir):
        os.remove(os.path.join(trials_dir, f))

    answers = []
    for t in range(1, args.trials + 1):
        calib = (t % args.calib_every) == 0
        ref_im = load_gray(refs[int(rng.integers(0, len(refs)))])
        a = crop(ref_im, rng, args.tile)
        if calib:
            ref2 = load_gray(refs[int(rng.integers(0, len(refs)))])
            b = crop(ref2, rng, args.tile)
            kinds = ["ref", "ref"]
        else:
            b = crop(cand_im, rng, args.tile)
            kinds = ["ref", "cand"]
        if rng.random() < 0.5:
            a, b = b, a
            kinds.reverse()
        make_trial(a, b, args.tile).save(
            os.path.join(trials_dir, f"trial_{t:02d}.png"))
        answers.append({"trial": t, "left": kinds[0], "right": kinds[1],
                        "type": "ref_vs_ref" if calib else "ref_vs_cand"})

    with open(os.path.join(_here, "..", "out", "spikes",
                           "s4_answers.json"), "w") as f:
        json.dump(answers, f, indent=1)
    print("trials :", os.path.abspath(trials_dir))
    print("answers:", os.path.abspath(os.path.join(
        _here, "..", "out", "spikes", "s4_answers.json")))
    print(f"{args.trials} trials "
          f"({sum(1 for a in answers if a['type'] == 'ref_vs_ref')} calibration)")


if __name__ == "__main__":
    main()
