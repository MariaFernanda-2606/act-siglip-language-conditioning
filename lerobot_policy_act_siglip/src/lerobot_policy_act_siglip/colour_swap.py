"""Cube colour-permutation augmentation (training only).

The cube layout is shuffled per episode, so during the pick phase only the instruction tells which cube is the
target. With ~160 training episodes the network learned to recognise each scene and recall where the target was,
instead of reading the colour. Here the colours of the cubes are permuted in the images (e.g. red <-> green) and the
pick instruction is changed accordingly, while the actions are kept: the robot still goes to the same cube, which
now has the instructed colour. The drawers never move, so they are protected (fixed boxes) and keep their colours.

Pixels are classified by hue (red / yellow / green) with hysteresis: strict thresholds find confident seeds, which
then grow within looser thresholds (shadowed or dark faces of the same object). Recolouring maps the hue, saturation
and value of each class onto those of the target colour. Thresholds and colour statistics come from dataset frames
(OpenCV HSV medians: red H0 S236 V145, yellow H22 S242 V140, green H69 S167 V47).
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from .planner import COLORS

# ImageNet statistics: the normaliser used for the camera images (dataset.use_imagenet_stats, lerobot's default).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Per colour (same order as COLORS = red, green, yellow): hue range in degrees, strict / loose (s, v) minima,
# and median (hue, s, v) of the class.
# The green cube is dark (V down to ~0.09 in the low camera) but very saturated and of very stable hue (130-140 deg),
# so its seeds need high S and low V, and its hue range is kept narrow (dark backgrounds have noisy greenish hues).
HUE_RANGES = {"red": (336.0, 12.0), "green": (112.0, 160.0), "yellow": (28.0, 64.0)}
STRICT = {"red": (0.59, 0.235), "green": (0.60, 0.07), "yellow": (0.59, 0.235)}
LOOSE = {"red": (0.31, 0.118), "green": (0.30, 0.05), "yellow": (0.35, 0.157)}
MEDIAN = {"red": (0.0, 0.925, 0.569), "green": (138.0, 0.655, 0.184), "yellow": (44.0, 0.949, 0.549)}
HUE_SPREAD = 0.5  # deviations from the class hue are compressed (avoids e.g. green -> red tinting magenta)
V_GAMMA = 0.6     # compression of brightness deviations from the class median
GROW_STEPS = 40   # px the seeds may grow into the loose region

# The 5 non-identity permutations of the 3 colours, as index maps (new colour of colour i = PERMS[k][i]).
PERMS = torch.tensor([p for p in itertools.permutations(range(len(COLORS))) if list(p) != list(range(len(COLORS)))])


def rgb_to_hsv(rgb: Tensor) -> Tensor:
    """(B, 3, H, W) in [0, 1] -> (B, 3, H, W) with hue in degrees [0, 360) and s, v in [0, 1]."""
    r, g, b = rgb.unbind(1)
    v, _ = rgb.max(1)
    c = v - rgb.min(1).values
    s = torch.where(v > 0, c / v.clamp_min(1e-8), torch.zeros_like(v))
    cc = c.clamp_min(1e-8)
    h = torch.where(v == r, ((g - b) / cc) % 6, torch.where(v == g, (b - r) / cc + 2, (r - g) / cc + 4))
    h = torch.where(c > 0, h * 60, torch.zeros_like(h))
    return torch.stack([h, s, v], 1)


def hsv_to_rgb(hsv: Tensor) -> Tensor:
    h, s, v = hsv.unbind(1)
    k = lambda n: (n + h / 60) % 6  # noqa: E731
    f = lambda n: v - v * s * (torch.minimum(k(n), 4 - k(n)).clamp(0, 1))  # noqa: E731
    return torch.stack([f(5), f(3), f(1)], 1)


def _in_hue(h: Tensor, lo: float, hi: float) -> Tensor:
    return (h >= lo) | (h <= hi) if lo > hi else (h >= lo) & (h <= hi)


def _dilate(m: Tensor, k: int) -> Tensor:
    return F.max_pool2d(m, k, stride=1, padding=k // 2)


def _erode(m: Tensor, k: int) -> Tensor:
    return 1 - F.max_pool2d(1 - m, k, stride=1, padding=k // 2)


def colour_masks(hsv: Tensor, protect: Tensor | None = None) -> Tensor:
    """(B, 3, H, W) HSV -> (B, n_colours, H, W) bool masks of the cube colours.

    protect: (n_colours, H, W) bool, pixels that can't seed that colour (the drawer of that colour). Growth from seeds
        outside may still enter them: a cube in front of the drawer of its own colour, which often overlaps it in the
        low camera, is recoloured whole (at the cost of the touching part of the drawer).
    """
    h, s, v = hsv.unbind(1)
    masks = []
    for i, colour in enumerate(COLORS):
        hue = _in_hue(h, *HUE_RANGES[colour])
        seedable = hue if protect is None else hue & ~protect[i]
        core = (seedable & (s >= STRICT[colour][0]) & (v >= STRICT[colour][1])).float()[:, None]
        loose = (hue & (s >= LOOSE[colour][0]) & (v >= LOOSE[colour][1])).float()[:, None]
        loose = _erode(_dilate(loose, 5), 5)              # close small gaps (texture, specular spots)
        region = _dilate(_erode(core, 3), 3) * loose      # drop isolated seed pixels (noise on dark objects)
        for _ in range(GROW_STEPS):
            region = _dilate(region, 3) * loose
        masks.append(region[:, 0] > 0.5)
    return torch.stack(masks, 1)


def recolour(rgb: Tensor, perm: Tensor, protect: Tensor | None = None) -> Tensor:
    """Permute the cube colours of (B, 3, H, W) images in [0, 1]; perm (B, n_colours) gives each colour's new index."""
    hsv = rgb_to_hsv(rgb)
    masks = colour_masks(hsv, protect)
    h, s, v = hsv.unbind(1)
    new_h, new_s, new_v = h.clone(), s.clone(), v.clone()
    for i, src in enumerate(COLORS):
        src_h, src_s, src_v = MEDIAN[src]
        dev = h - src_h
        dev = torch.where(dev > 180, dev - 360, torch.where(dev < -180, dev + 360, dev))
        for j, dst in enumerate(COLORS):
            if i == j:
                continue
            m = masks[:, i] & (perm[:, i] == j)[:, None, None]
            if not m.any():
                continue
            dst_h, dst_s, dst_v = MEDIAN[dst]
            new_h = torch.where(m, (dst_h + HUE_SPREAD * dev) % 360, new_h)
            new_s = torch.where(m, (s * dst_s / src_s).clamp(0, 1), new_s)
            # Brightness relative to the class median, compressed (gamma) so that e.g. the dark faces of the green
            # cube don't become brown when mapped to yellow.
            new_v = torch.where(m, (dst_v * (v / src_v).clamp_min(1e-6) ** V_GAMMA).clamp(0, 1), new_v)
    return hsv_to_rgb(torch.stack([new_h, new_s, new_v], 1))


def protect_mask(boxes: list[list[float]], h: int, w: int, device) -> Tensor | None:
    """(n_colours, H, W) bool from boxes [x0, y0, x1, y1] (all colours) or [x0, y0, x1, y1, colour index]
    (only that colour, e.g. the band of one drawer: a cube of another colour in front of it is still recoloured)."""
    if not boxes:
        return None
    m = torch.zeros(len(COLORS), h, w, dtype=torch.bool, device=device)
    for box in boxes:
        x0, y0, x1, y1 = box[:4]
        colours = [int(box[4])] if len(box) == 5 else slice(None)
        m[colours, round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)] = True
    return m


def recolour_normalised(img: Tensor, perm: Tensor, protect: Tensor | None = None) -> Tensor:
    """Same as `recolour` for ImageNet-normalised images."""
    mean = img.new_tensor(IMAGENET_MEAN)[None, :, None, None]
    std = img.new_tensor(IMAGENET_STD)[None, :, None, None]
    rgb = (img.float() * std + mean).clamp(0, 1)
    return ((recolour(rgb, perm, protect) - mean) / std).to(img.dtype)
