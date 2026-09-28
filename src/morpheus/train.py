"""Training of the tissue's update rule by backpropagation through time (grad.py).

(An evolution-strategies trainer was tried first; with 2,450 parameters and 112-step
lives it plateaued near loss 0.17 while BPTT reaches ~0.02, so it was removed.)

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


_POOL = None


def _pool(workers):
    global _POOL
    if _POOL is None and workers > 1:
        import multiprocessing as mp
        _POOL = mp.get_context("fork").Pool(workers)
    return _POOL


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


def _bptt_chunk(args):
    from .grad import loss_and_grad
    theta, wounds, cm, physics, proto, seed, init_state, init_eps = args
    loss, g, ex = loss_and_grad(theta, wounds, physics, proto, seed, comparator_mask=cm,
                                init_state=init_state, init_eps=init_eps)
    return loss * len(wounds), g * len(wounds), ex["state"], ex["eps"]


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
               theta=None, workers=4, log=None, dropout=0.25, pool_size=128, persist_steps=48):
    """Each iteration trains on two half-batches and averages their gradients:

    life         founder cells grow (64 steps), are wounded and regenerate (48 steps), as in
                 the experiments
    persistence  tissues sampled from a pool of old tissues (ages 112 steps and up, since each
                 pass returns them to the pool) are wounded with probability 1/2 and must hold
                 or regrow the body for another 48 steps. This trains anatomical homeostasis
                 beyond the length of one life (sample-pool training, Mordvintsev et al. 2020).
    """
    rng = np.random.default_rng(seed)
    theta = init_params(rng).astype(np.float64) if theta is None else theta.astype(np.float64).copy()
    opt = Adam(N_PARAMS, lr)
    mp_pool = _pool(workers)
    persist_proto = Protocol(grow=0, regen=persist_steps, tail_window=proto.tail_window)
    pool_s, pool_e = [], []
    history, t0 = [], time.time()
    half = batch // 2

    def run(jobs):
        return mp_pool.map(_bptt_chunk, jobs) if mp_pool else [_bptt_chunk(j) for j in jobs]

    def split(n):
        return [c for c in np.array_split(np.arange(n), max(workers, 1)) if len(c)]

    for it in range(iterations):
        opt.lr = lr * (0.1 if it >= 0.7 * iterations else 1.0)
        n_life = batch if len(pool_s) < 2 * half else half
        _, wounds = anatomy.random_wounds(n_life, rng)
        cm = rng.random(n_life) > dropout
        jobs = [(theta, wounds[c], cm[c], physics, proto, int(rng.integers(1 << 31)), None, None) for c in split(n_life)]
        n_persist = batch - n_life
        if n_persist:
            idx = rng.choice(len(pool_s), n_persist, replace=False)
            _, pw = anatomy.random_wounds(n_persist, rng)
            pw &= (rng.random(n_persist) < 0.5)[:, None, None]
            pcm = rng.random(n_persist) > dropout
            ps, pe = np.stack([pool_s[i] for i in idx]), np.stack([pool_e[i] for i in idx])
            jobs += [(theta, pw[c], pcm[c], physics, persist_proto, int(rng.integers(1 << 31)), ps[c], pe[c])
                     for c in split(n_persist)]
        parts = run(jobs)
        loss = sum(p[0] for p in parts) / batch
        g = sum(p[1] for p in parts) / batch
        finals_s = np.concatenate([p[2] for p in parts])
        finals_e = np.concatenate([p[3] for p in parts])
        if n_persist:
            for k, i in enumerate(idx):          # persistence tissues go back, older
                pool_s[i], pool_e[i] = finals_s[n_life + k], finals_e[n_life + k]
        for k in range(n_life):                  # fresh lives join the pool, replacing random members
            if len(pool_s) < pool_size:
                pool_s.append(finals_s[k]); pool_e.append(finals_e[k])
            else:
                j = int(rng.integers(pool_size))
                pool_s[j], pool_e[j] = finals_s[k], finals_e[k]
        theta = theta - opt.update(_normalise(g))
        history.append(loss)
        if log and (it % 20 == 0 or it == iterations - 1):
            log(f"iter {it:5d}  loss {np.mean(history[-20:]):.4f}  ({time.time() - t0:.0f}s)")
    return theta.astype(np.float32), history
