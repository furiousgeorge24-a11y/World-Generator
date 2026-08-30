"""S4b: imposter trial at 1024x1024, applying the S4 calibration lessons.

Differences from the first (invalidated) trial build:
- Tiles are 1024x1024: whole candidate maps (S2 erosion spike, 24
  geologic steps) vs full-size reference crops — enough area to judge
  formation seriously.
- Palette preserved: the candidate renders through a reference-like
  stepped hypsometric ramp (dense stops low, tan/brown high) instead of
  grayscaling the look away. Palettes still differ; judges are told to
  judge formation, not hue.
- Bland-tile filter: reference crops must carry structure (luminance
  variance + gradient content, near-black rejection) or be resampled.
- Calibration arms kept: every third trial is ref-vs-ref.

    py -3.14 spikes\\s4b_imposter_1024.py
"""

import importlib.util
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

_here = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_here, "..", ".."))
REFS_DIR = os.path.join(REPO, "examples")
TILE = 1024
# references whose both dimensions allow a full 1024 crop
REF_OK = ["ref1.png", "ref2.png", "ref6.png", "ref9.png",
          "ref10.png", "ref14.png"]

spec = importlib.util.spec_from_file_location(
    "s2", os.path.join(_here, "s2_erosion_solver.py"))
s2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s2)

# stepped hypsometric ramp eyeballed from the references: dense green
# stops low, compressed pale middle, tan/brown high
LAND_STOPS = np.array([
    [70, 105, 78], [95, 125, 88], [118, 143, 100], [140, 158, 110],
    [165, 175, 122], [185, 188, 135], [200, 196, 148], [208, 196, 155],
    [200, 180, 135], [185, 158, 112], [165, 132, 92], [140, 105, 72],
])


def render_candidate(h, F, A8):
    G = h.shape[0]
    t = np.clip(h / max(h.max(), 1.0), 0, 1) ** 0.55
    band = np.minimum((t * len(LAND_STOPS)).astype(int), len(LAND_STOPS) - 1)
    img = LAND_STOPS[band].astype(float)
    depth = F - h
    lakes = depth > 0.5
    img[lakes & (depth <= 8)] = (150, 178, 188)
    img[lakes & (depth > 8) & (depth <= 25)] = (118, 152, 172)
    img[lakes & (depth > 25)] = (88, 126, 158)
    rivers = (A8.reshape(G, G) > G * G * 0.0003) & ~lakes
    img[rivers] = (52, 84, 120)
    return Image.fromarray(img.astype(np.uint8))


def crop_scored(im, rng, tries=20):
    """Random crop that must carry structure; returns the best-scoring
    crop if none passes the bar."""
    w, h = im.size
    best, best_score = None, -1.0
    for _ in range(tries):
        x = int(rng.integers(0, w - TILE + 1))
        y = int(rng.integers(0, h - TILE + 1))
        c = im.crop((x, y, x + TILE, y + TILE))
        g = np.asarray(c.convert("L"), float)
        lum_std = g.std()
        grad = (np.abs(np.diff(g, axis=0)).mean()
                + np.abs(np.diff(g, axis=1)).mean())
        black = (g < 12).mean()
        score = lum_std + 8 * grad - 500 * max(0.0, black - 0.03)
        if lum_std > 18 and grad > 2.0 and black < 0.03:
            return c
        if score > best_score:
            best, best_score = c, score
    return best


def make_trial(left, right, gap=12, header=22):
    im = Image.new("RGB", (TILE * 2 + gap, TILE + header), (240, 240, 240))
    im.paste(left, (0, header))
    im.paste(right, (TILE + gap, header))
    d = ImageDraw.Draw(im)
    d.text((6, 4), "L", fill=(0, 0, 0))
    d.text((TILE + gap + 6, 4), "R", fill=(0, 0, 0))
    return im


def main():
    rng = np.random.default_rng(np.random.SeedSequence([11, 0x5B54]))
    trials_dir = os.path.join(_here, "..", "out", "spikes", "s4b_trials")
    os.makedirs(trials_dir, exist_ok=True)
    for f in os.listdir(trials_dir):
        os.remove(os.path.join(trials_dir, f))

    refs = [Image.open(os.path.join(REFS_DIR, f)).convert("RGB")
            for f in REF_OK]

    n_cand = 8
    cands = []
    for i in range(n_cand):
        seed = 21 + i
        t0 = time.perf_counter()
        h, F, A8, _ = s2.run(seed, TILE, 24)
        cands.append(render_candidate(h, F, A8))
        print(f"candidate seed {seed}: {time.perf_counter() - t0:.0f}s")

    answers = []
    ci = 0
    for t in range(1, 13):
        calib = (t % 3) == 0
        a = crop_scored(refs[int(rng.integers(0, len(refs)))], rng)
        if calib:
            b = crop_scored(refs[int(rng.integers(0, len(refs)))], rng)
            kinds = ["ref", "ref"]
        else:
            b = cands[ci]
            ci += 1
            kinds = ["ref", "cand"]
        if rng.random() < 0.5:
            a, b = b, a
            kinds.reverse()
        make_trial(a, b).save(os.path.join(trials_dir, f"trial_{t:02d}.png"))
        answers.append({"trial": t, "left": kinds[0], "right": kinds[1],
                        "type": "ref_vs_ref" if calib else "ref_vs_cand"})
        print(f"trial {t:02d} built ({'calib' if calib else 'cand'})")

    with open(os.path.join(_here, "..", "out", "spikes",
                           "s4b_answers.json"), "w") as f:
        json.dump(answers, f, indent=1)
    print("trials :", os.path.abspath(trials_dir))


if __name__ == "__main__":
    main()
