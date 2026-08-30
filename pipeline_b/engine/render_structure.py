"""Structure-stage views (§14: every layer visible, every stage
judged). All views draw the delivered-frame rectangle as an annotation;
the frame never participates in generation.
"""

import numpy as np
from PIL import Image, ImageDraw

VIEWS = ["crust", "margins", "belt_age", "boundaries", "plates"]


def _lerp(c0, c1, t):
    t = np.clip(t, 0.0, 1.0)[..., None]
    return np.asarray(c0, float) * (1 - t) + np.asarray(c1, float) * t


def _edges(label):
    e = np.zeros(label.shape, bool)
    e[:, 1:] |= label[:, 1:] != label[:, :-1]
    e[1:, :] |= label[1:, :] != label[:-1, :]
    return e


def _dilate1(m):
    g = m.copy()
    g[:, 1:] |= m[:, :-1]
    g[:, :-1] |= m[:, 1:]
    g[1:, :] |= m[:-1, :]
    g[:-1, :] |= m[1:, :]
    return g


def _base(s, view):
    n = s.n
    img = np.zeros((n, n, 3))
    if view == "crust":
        t_oc = np.clip(s.age_myr / 400.0, 0, 1)
        img[:] = _lerp((110, 165, 215), (16, 34, 76), t_oc)
        t_ct = np.clip(s.age_myr / 2000.0, 0, 1)
        cont_col = _lerp((208, 193, 158), (162, 143, 105), t_ct)
        img[s.cont] = cont_col[s.cont]
        b = np.clip(s.belt / 6.0, 0, 1)
        belt_land = _lerp(cont_col, np.broadcast_to((122, 58, 30),
                                                    cont_col.shape), b)
        img[s.cont & (s.belt > 0)] = belt_land[s.cont & (s.belt > 0)]
        arc = (~s.cont) & (s.belt > 0)
        img[arc] = _lerp(img, np.broadcast_to((80, 120, 160),
                                              img.shape), b)[arc]
        img[_edges(s.label)] *= 0.6
    elif view == "margins":
        img[:] = (25, 35, 60)
        img[s.cont] = (185, 185, 185)
        img[_dilate1(s.passive_margin)] = (70, 180, 90)
        img[_dilate1(s.active_margin)] = (225, 60, 45)
    elif view == "belt_age":
        img[:] = (28, 32, 48)
        img[s.cont] = (78, 78, 78)
        b = s.belt > 0
        t = np.clip(s.belt_age_era / max(s.eras - 1, 1), 0, 1)
        img[b] = _lerp((96, 42, 130), (250, 220, 80), t)[b]
    elif view == "boundaries":
        img[:] = (18, 20, 30)
        img[s.cont] = (60, 60, 60)
        img[_dilate1(s.coast)] = (110, 110, 110)
        img[s.div_recent] = (90, 200, 220)
        img[s.conv_recent] = (225, 70, 50)
    elif view == "plates":
        rngc = np.random.default_rng(4)
        pal = rngc.uniform(70, 200, (int(s.label.max()) + 2, 3))
        img = pal[s.label.clip(0)]
        img[s.cont] = img[s.cont] * 0.6 + np.array((90, 80, 60)) * 0.4
        img[_edges(s.label)] *= 0.45
    else:
        raise ValueError(f"unknown view {view!r}")
    return img.astype(np.uint8)


def render_view(s, view, size):
    img = _base(s, view)
    im = Image.fromarray(img).resize((size, size), Image.NEAREST)
    d = ImageDraw.Draw(im)
    k = size / s.n
    f0, f1 = s.frame_slice
    d.rectangle([f0 * k, f0 * k, f1 * k - 1, f1 * k - 1],
                outline=(245, 245, 245), width=1)
    return im
