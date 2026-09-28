"""Evolution-strategies training of the tissue's update rule (no autodiff needed).

Antithetic Gaussian perturbations, centred-rank fitness shaping and Adam
(Salimans et al. 2017, "Evolution Strategies as a Scalable Alternative to RL").
Every population member in a generation is evaluated on the same founders, wounds
and fire masks (common random numbers), so fitness differences come from the
parameters alone.

Objective per tissue (lower is better):

    grow loss   anatomical error over the last steps of growth
  + regen loss  anatomical error over the last steps of regeneration from a random wound
  + lambda * self-model error (mean squared comparator error over the whole life)

A quarter of tissues are trained with the comparator silenced, so that the "zero"
ablation in the experiments is not out of distribution.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import anatomy
from .life import Protocol, live
from .tissue import N_PARAMS, Physics, init_params

SELF_MODEL_WEIGHT = 0.05


def tune_malloc():
    """Keep freed buffers in the heap (glibc). Backprop allocates and frees many ~1 MB arrays per
    step; returning each to the OS makes every step page-fault, which costs ~40% of run time."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        for param, value in ((-3, 1 << 30), (-1, 1 << 30), (-2, 256 << 20)):  # MMAP_THRESHOLD, TRIM, TOP_PAD
            libc.mallopt(param, value)
    except (OSError, AttributeError):
        pass


tune_malloc()


def objective(theta_batch, wounds, physics, proto, seed, comparator_mask, tgt,
              self_model_weight=SELF_MODEL_WEIGHT, crn_group=None):
    """Per-tissue loss for (N,P) parameters on N tissues."""
    p = Protocol(**{**proto.__dict__, "comparator_mask": comparator_mask})
    out = live(theta_batch, wounds, physics, p, seed, tgt=tgt, crn_group=crn_group)
    return out["grow_loss"] + out["regen_loss"] + self_model_weight * out["self_model_error"], out


def _member_losses(args):
    members, wounds, cm, physics, proto, seed, tgt = args
    t = len(wounds)
    batch = np.repeat(members, t, axis=0)
    loss, _ = objective(batch, np.tile(wounds, (len(members), 1, 1)), physics, proto, seed,
                        np.tile(cm, len(members)), tgt, crn_group=t)
    return loss.reshape(len(members), t).mean(axis=1)


_POOL = None


def _pool(workers):
    global _POOL
    if _POOL is None and workers > 1:
        import multiprocessing as mp
        _POOL = mp.get_context("fork").Pool(workers)
    return _POOL


def evaluate_members(members, wounds, cm, physics, proto, seed, tgt, workers=1):
    """Mean loss of each parameter vector in ``members`` on the same tissues and fire masks."""
    pool = _pool(workers)
    if pool is None:
        return _member_losses((members, wounds, cm, physics, proto, seed, tgt))
    chunks = np.array_split(members, workers)
    parts = pool.map(_member_losses, [(c, wounds, cm, physics, proto, seed, tgt) for c in chunks if len(c)])
    return np.concatenate(parts)


class Adam:
    def __init__(self, n, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def update(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + self.eps)


def centred_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def es_gradient(theta, rng, pairs, sigma, tissues_per_member, physics, proto, seed, tgt, workers=1):
    """Estimate the gradient of the mean loss with antithetic sampling. Returns (grad, mean loss)."""
    noise = rng.standard_normal((pairs, theta.size)).astype(np.float32)
    members = np.concatenate([theta + sigma * noise, theta - sigma * noise])       # (2K, P)
    _, wounds = anatomy.random_wounds(tissues_per_member, rng)
    cm = rng.random(tissues_per_member) > 0.25
    fit = evaluate_members(members, wounds, cm, physics, proto, seed, tgt, workers)
    r = centred_ranks(fit)
    g = ((r[:pairs] - r[pairs:])[:, None] * noise).sum(0) / (2 * pairs * sigma)
    return g, float(fit.mean())


def train(generations=300, pairs=16, sigma=0.02, lr=0.01, tissues_per_member=2, seed=0,
          physics=Physics(), proto=Protocol(), theta=None, log=None, weight_decay=0.0005, workers=1):
    rng = np.random.default_rng(seed)
    theta = init_params(rng) if theta is None else theta.astype(np.float32).copy()
    opt = Adam(N_PARAMS, lr)
    tgt = anatomy.target()
    history = []
    t0 = time.time()
    for gen in range(generations):
        g, mean_loss = es_gradient(theta, rng, pairs, sigma, tissues_per_member, physics, proto,
                                   int(rng.integers(1 << 31)), tgt, workers)
        theta = (theta - opt.update(g) - lr * weight_decay * theta).astype(np.float32)
        history.append(mean_loss)
        if log and (gen % 10 == 0 or gen == generations - 1):
            log(f"gen {gen:4d}  population loss {mean_loss:.4f}  ({time.time() - t0:.0f}s)")
    return theta, history


def save_weights(path, theta, physics: Physics, meta: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": 1, "n_params": int(theta.size), "physics": physics.__dict__,
           "meta": meta, "theta": [round(float(v), 7) for v in theta]}
    Path(path).write_text(json.dumps(doc), encoding="utf-8")


def load_weights(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    theta = np.asarray(doc["theta"], np.float32)
    if theta.size != N_PARAMS:
        raise ValueError(f"{path}: {theta.size} parameters, model expects {N_PARAMS}")
    return theta, Physics(**doc["physics"]), doc.get("meta", {})


# --------------------------------------------------------------------------------------
# Backpropagation through time (the default trainer; ES above is kept for the RSI proposer)

def _bptt_chunk(args):
    from .grad import loss_and_grad
    theta, wounds, cm, physics, proto, seed = args
    loss, g, _ = loss_and_grad(theta, wounds, physics, proto, seed, comparator_mask=cm)
    return loss * len(wounds), g * len(wounds)


def _normalise(g: np.ndarray) -> np.ndarray:
    """Per-tensor gradient normalisation (as in Mordvintsev et al. 2020, "Growing NCA")."""
    from .tissue import SHAPES
    out, i = np.empty_like(g), 0
    for _, shape in SHAPES:
        k = int(np.prod(shape))
        out[i:i + k] = g[i:i + k] / (np.linalg.norm(g[i:i + k]) + 1e-8)
        i += k
    return out


def train_bptt(iterations=2000, batch=16, lr=2e-3, seed=0, physics=Physics(), proto=Protocol(),
               theta=None, workers=4, log=None, dropout=0.25):
    rng = np.random.default_rng(seed)
    theta = init_params(rng).astype(np.float64) if theta is None else theta.astype(np.float64).copy()
    opt = Adam(N_PARAMS, lr)
    pool = _pool(workers)
    history, t0 = [], time.time()
    for it in range(iterations):
        opt.lr = lr * (0.1 if it >= 0.7 * iterations else 1.0)
        _, wounds = anatomy.random_wounds(batch, rng)
        cm = rng.random(batch) > dropout
        chunks = np.array_split(np.arange(batch), max(workers, 1))
        jobs = [(theta, wounds[c], cm[c], physics, proto, int(rng.integers(1 << 31))) for c in chunks if len(c)]
        parts = pool.map(_bptt_chunk, jobs) if pool else [_bptt_chunk(j) for j in jobs]
        loss = sum(p[0] for p in parts) / batch
        g = sum(p[1] for p in parts) / batch
        theta = theta - opt.update(_normalise(g))
        history.append(loss)
        if log and (it % 20 == 0 or it == iterations - 1):
            log(f"iter {it:5d}  loss {np.mean(history[-20:]):.4f}  ({time.time() - t0:.0f}s)")
    return theta.astype(np.float32), history
