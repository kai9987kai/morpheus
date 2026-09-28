"""Target anatomy, wounds and anatomical measurements.

The target is a planarian-like body: an elongated, tapering body with a head, a
trunk and a tail region. Only the anatomy is specified. The voltage channel has no
target, so any bioelectric prepattern a trained tissue uses is its own invention.
"""

from __future__ import annotations

import numpy as np

from .tissue import ALPHA, TYPES

SIZE = 32
HEAD, TRUNK, TAIL = 0, 1, 2
REGION_NAMES = ("head", "trunk", "tail")


def target(size: int = SIZE) -> np.ndarray:
    """(H,W,4): [alpha, head, trunk, tail] with alpha 1 inside the body and a one-hot region."""
    t = np.zeros((size, size, 4), np.float32)
    cy, cx = (size - 1) / 2, (size - 1) / 2
    rx = 0.41 * size
    for y in range(size):
        for x in range(size):
            u = (x - cx) / rx
            ry = 0.19 * size * (1 - 0.45 * max(0.0, u))  # the tail tapers
            if u * u + ((y - cy) / ry) ** 2 <= 1:
                region = HEAD if u < -0.38 else (TRUNK if u < 0.3 else TAIL)
                t[y, x, 0] = 1
                t[y, x, 1 + region] = 1
    return t


def visible(s: np.ndarray) -> np.ndarray:
    """(...,C) state -> (...,4) [alpha, head, trunk, tail] comparable with ``target``."""
    return np.concatenate([s[..., ALPHA:ALPHA + 1], s[..., TYPES]], axis=-1)


def loss(s: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Per-tissue anatomical error: mean squared error of alpha and the three identity channels."""
    return ((visible(s) - tgt) ** 2).mean(axis=(1, 2, 3))


def fidelity(s: np.ndarray, tgt: np.ndarray) -> dict:
    """Descriptive measures per tissue: body IoU and region accuracy inside the target body."""
    body = s[..., ALPHA] > 0.1
    tbody = tgt[..., 0] > 0.5
    inter = (body & tbody).sum(axis=(1, 2))
    union = (body | tbody).sum(axis=(1, 2))
    region = np.argmax(s[..., TYPES], axis=-1)
    tregion = np.argmax(tgt[..., 1:], axis=-1)
    correct = (region == tregion) & body & tbody
    return {
        "iou": inter / np.maximum(union, 1),
        "region_accuracy": correct.sum(axis=(1, 2)) / max(int(tbody.sum()), 1),
    }


WOUNDS = ("head_amputation", "tail_amputation", "disc", "lateral")


def wound(kind: str, rng: np.random.Generator, size: int = SIZE) -> np.ndarray:
    """(H,W) bool mask of cells removed by a wound of the given kind (randomised extent)."""
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2
    if kind == "head_amputation":
        return xx < c - size * rng.uniform(0.12, 0.22)
    if kind == "tail_amputation":
        return xx > c + size * rng.uniform(0.08, 0.2)
    if kind == "disc":
        cy = c + rng.uniform(-2, 2)
        cx = c + rng.uniform(-0.3, 0.3) * size
        r = rng.uniform(3.0, 5.5)
        return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    if kind == "lateral":
        side = rng.choice([-1, 1])
        return side * (yy - c) > rng.uniform(0.5, 2.5)
    raise ValueError(f"unknown wound {kind!r}")


def random_wounds(n: int, rng: np.random.Generator, size: int = SIZE, kinds=WOUNDS):
    kinds_drawn = [kinds[i] for i in rng.integers(0, len(kinds), n)]
    return kinds_drawn, np.stack([wound(k, rng, size) for k in kinds_drawn])
