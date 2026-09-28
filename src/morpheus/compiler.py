"""Anatomical compiler: design a bioelectric intervention that reprograms what regrows.

Given a frozen rule, find a voltage pattern P (one value per grid position, bounded by
tanh to +-4) such that injecting P into every cell's voltage for the first ``steps``
steps after a tail amputation makes the tissue regrow a *head* where the tail was:
a two-headed body, the synthetic counterpart of the octanol-induced two-headed
planaria (Durant et al. 2017). The design uses exact gradients through the whole
regeneration (grad.loss_and_grad with ``v_inject``), on training tissues only.

The confirmatory tests (experiments.h5_compiler) apply the finished pattern to
held-out tissues and compare it with the same values spatially shuffled, so they ask
whether the *spatial code* matters, not merely that voltage was perturbed.
"""

from __future__ import annotations

import time

import numpy as np

from . import anatomy
from .life import Protocol, live
from .tissue import Physics

INJECT_STEPS = 12
AMPLITUDE = 4.0


def two_headed_target(size: int = anatomy.SIZE) -> np.ndarray:
    """The normal body's anterior half mirrored onto the posterior: head | trunk | trunk | head."""
    t = anatomy.target(size)
    out = t.copy()
    half = size // 2
    out[:, half:] = t[:, :size - half][:, ::-1]
    return out


def posterior_head_index(s: np.ndarray, size: int = anatomy.SIZE) -> np.ndarray:
    """In the normal body's tail region: fraction of cells typed head minus fraction typed tail.

    About -1 for a normal tail, about +1 when a head has grown there.
    """
    tail_region = anatomy.target(size)[..., 3] > 0.5
    alive = s[..., 0] > 0.1
    typ = np.argmax(s[..., 2:5], -1)
    head = ((typ == anatomy.HEAD) & alive & tail_region).sum((1, 2))
    tail = ((typ == anatomy.TAIL) & alive & tail_region).sum((1, 2))
    return (head - tail) / tail_region.sum()


def tail_wounds(n: int, rng: np.random.Generator, size: int = anatomy.SIZE) -> np.ndarray:
    return np.stack([anatomy.wound("tail_amputation", rng, size) for _ in range(n)])


def grown_pool(theta, physics: Physics, n: int, seed: int):
    """n grown tissues (64 steps from a founder) and their comparator errors."""
    dummy = np.zeros((n, anatomy.SIZE, anatomy.SIZE), bool)
    out = live(theta, dummy, physics, Protocol(regen=1), seed)
    return out["state"], out["eps"]


def pattern_of(q: np.ndarray) -> np.ndarray:
    return (AMPLITUDE * np.tanh(q)).astype(np.float32)


def _chunk(args):
    from .grad import loss_and_grad
    theta, wounds, physics, seed, s0, e0, pattern, steps, tgt2, mode = args
    loss, _, ex = loss_and_grad(theta, wounds, physics, Protocol(grow=0, regen=48), seed,
                                self_model_weight=0.0, init_state=s0, init_eps=e0,
                                regen_target=tgt2, v_inject=pattern, inject_steps=steps,
                                inject_mode=mode)
    return loss * len(wounds), ex["g_inject"] * len(wounds)


def design(theta, physics: Physics, iterations=600, batch=16, lr=0.1, seed=0, workers=4,
           steps=INJECT_STEPS, reg=1e-3, mode="clamp", log=None) -> dict:
    """Optimise the injected pattern with Adam on training tissues. Returns pattern and history."""
    from .train import Adam, _pool
    rng = np.random.default_rng([seed, 77])
    pool_s, pool_e = grown_pool(theta, physics, 256, int(rng.integers(1 << 31)))
    tgt2 = two_headed_target()
    q = np.zeros((anatomy.SIZE, anatomy.SIZE))
    opt = Adam(q.size, lr)
    mp = _pool(workers)
    hist, t0 = [], time.time()
    for it in range(iterations):
        idx = rng.choice(len(pool_s), batch, replace=False)
        wounds = tail_wounds(batch, rng)
        pattern = pattern_of(q)
        chunks = [c for c in np.array_split(np.arange(batch), max(workers, 1)) if len(c)]
        jobs = [(theta, wounds[c], physics, int(rng.integers(1 << 31)), pool_s[idx[c]], pool_e[idx[c]],
                 pattern, steps, tgt2, mode) for c in chunks]
        parts = mp.map(_chunk, jobs) if mp else [_chunk(j) for j in jobs]
        loss = sum(p[0] for p in parts) / batch
        g_pat = sum(p[1] for p in parts) / batch + reg * 2 * pattern / pattern.size
        g_q = g_pat * AMPLITUDE * (1 - np.tanh(q) ** 2)
        q = q - opt.update(g_q.ravel()).reshape(q.shape)
        hist.append(loss)
        if log and (it % 20 == 0 or it == iterations - 1):
            log(f"design iter {it:4d}  loss vs two-headed target {np.mean(hist[-20:]):.4f}  ({time.time() - t0:.0f}s)")
    return {"pattern": pattern_of(q), "steps": steps, "mode": mode, "history": hist, "iterations": iterations,
            "batch": batch, "lr": lr, "seed": seed, "reg": reg}


def shuffled(pattern: np.ndarray, seed: int) -> np.ndarray:
    """Same values, spatially permuted within the body region: keeps the dose, destroys the code."""
    body = anatomy.target(pattern.shape[0])[..., 0] > 0.5
    out = pattern.copy()
    idx = np.flatnonzero(body)
    out.flat[idx] = pattern.flat[np.random.default_rng(seed).permutation(idx)]
    return out
