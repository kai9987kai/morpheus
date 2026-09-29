"""Exact backpropagation through time for the tissue, in NumPy.

``loss_and_grad`` runs the same life cycle as ``life.live`` (author comparator,
growth then wound then regeneration, same fire masks for the same seed) and returns
the training objective of ``train.objective`` together with its gradient with respect
to the shared parameter vector. Alive masks are treated as constants (they are step
functions), which is the standard treatment for neural cellular automata.

tests/test_grad.py checks the gradient against central finite differences.
"""

from __future__ import annotations

import numpy as np

from . import anatomy
from .life import Protocol, fire_masks
from .tissue import (C, N_IN, STATE_CLIP, VIS, VOLT, Physics, alive_mask, gap_junction_flux,
                     perceive, perceive_adjoint, seed_state, unpack)

SELF_MODEL_WEIGHT = 0.05


def loss_and_grad(theta, wounds, physics: Physics, proto: Protocol, seed: int,
                  comparator_mask=None, self_model_weight=SELF_MODEL_WEIGHT, crn_group=None,
                  dtype=np.float32, init_state=None, init_eps=None, regen_target=None,
                  v_inject=None, inject_steps=0, inject_mode="add", regen_weight=None):
    """Mean over tissues of grow loss + regen loss + weight * self-model error, and its gradient.

    With ``init_state`` (and ``init_eps``) the life starts from those tissues instead of a
    founder cell; with ``proto.grow == 0`` the wound is applied to them at once. This is the
    persistence phase of training: old tissues from a pool are wounded and must regrow.

    ``regen_target`` replaces the anatomy scored at the end of regeneration. ``v_inject`` (H,W)
    is added to every cell's voltage on each of the first ``inject_steps`` steps after the wound
    (a designed bioelectric intervention); with ``inject_mode="clamp"`` voltage is instead held at
    ``v_inject`` on those steps. ``v_inject`` may also be (P,H,W): a program of P phases that
    split the ``inject_steps`` evenly. Its gradient is returned as extras["g_inject"] (same shape).
    ``regen_weight`` (H,W), mean 1, reweights the regeneration loss spatially.

    Returns (mean loss, gradient, extras) with extras = per-tissue loss, final state and error.
    """
    assert theta.ndim == 1, "shared parameters only"
    n, size = wounds.shape[0], wounds.shape[1]
    tgt = anatomy.target(size)
    p = unpack(theta.astype(dtype))
    W1, b1, W2, b2 = p["W1"], p["b1"], p["W2"], p["b2"]
    rng = np.random.default_rng(seed)
    fires = fire_masks(rng, n, size, physics.fire_rate, crn_group)
    cm = np.ones((n, 1, 1, 1), dtype) if comparator_mask is None else comparator_mask[:, None, None, None].astype(dtype)
    keep = ~wounds[..., None]
    G, R, Wn = proto.grow, proto.regen, proto.tail_window
    T = G + R
    hw = size * size
    D, gain = physics.diffusion, physics.gain

    tgt = tgt.astype(dtype)
    rtgt = tgt if regen_target is None else regen_target.astype(dtype)
    inj = None if v_inject is None else np.asarray(v_inject, dtype).reshape(-1, size, size)
    g_inject = None if inj is None else np.zeros(inj.shape)
    rw = None if regen_weight is None else np.asarray(regen_weight, dtype)[None, :, :, None]

    def phase(t):
        return (t - G) * inj.shape[0] // inject_steps
    s = (seed_state(n, size, size) if init_state is None else init_state).astype(dtype)
    eps = (np.zeros((n, size, size, VIS)) if init_eps is None else init_eps).astype(dtype)
    per_tissue = np.zeros(n)
    tape = []
    for t in range(T):
        if t == G:
            s = s * keep
        e_in = eps * cm
        f = next(fires).astype(dtype)
        pre = alive_mask(s)
        x = np.concatenate([perceive(s), gain * e_in], -1).reshape(n, hw, N_IN)
        a = x @ W1 + b1
        o = (np.maximum(a, 0) @ W2 + b2).reshape(n, size, size, C + VIS)
        ds, pred = o[..., :C], o[..., C:]
        u = s + ds * f
        if D:
            u[..., VOLT:VOLT + 1] += D * gap_junction_flux(s[..., VOLT:VOLT + 1], pre)
        if inj is not None and G <= t < G + inject_steps:
            if inject_mode == "clamp":
                u[..., VOLT] = inj[phase(t)]
            else:
                u[..., VOLT] += inj[phase(t)]
        clipm = np.abs(u) < STATE_CLIP
        c = np.clip(u, -STATE_CLIP, STATE_CLIP)
        alive = pre & alive_mask(c)
        s2 = c * alive
        eps = ((s2 - s)[..., :VIS] - f * pred) * alive
        w = 1.0 / Wn if (G - Wn <= t < G) or (t >= T - Wn) else 0.0
        if w:
            if rw is not None and t >= G:
                per_tissue += w * (rw * (anatomy.visible(s2) - rtgt) ** 2).mean(axis=(1, 2, 3))
            else:
                per_tissue += w * anatomy.loss(s2, tgt if t < G else rtgt)
        per_tissue += self_model_weight * (eps.astype(np.float64) ** 2).mean(axis=(1, 2, 3)) / T
        tape.append((s, x, a, f, pre, alive, clipm, eps, w, s2 if w else None))
        s = s2
    final_state, final_eps = s, eps

    gW1, gb1 = np.zeros_like(W1), np.zeros_like(b1)
    gW2, gb2 = np.zeros_like(W2), np.zeros_like(b2)
    gs = np.zeros((n, size, size, C), dtype)
    ge = np.zeros((n, size, size, VIS), dtype)
    for t in reversed(range(T)):
        s_in, x, a, f, pre, alive, clipm, eps, w, s2 = tape[t]
        tape[t] = None
        gs2 = gs
        if w:
            vis = anatomy.visible(s2)
            g_vis = w * 2 * (vis - (tgt if t < G else rtgt)) / (hw * 4) / n
            if rw is not None and t >= G:
                g_vis = g_vis * rw
            gs2[..., 0:1] += g_vis[..., 0:1]
            gs2[..., 2:5] += g_vis[..., 1:4]
        ge_t = (ge + self_model_weight * 2 * eps / (hw * VIS) / T / n) * alive
        gs2[..., :VIS] += ge_t
        gs_in = np.zeros_like(gs2)
        gs_in[..., :VIS] -= ge_t
        gpred = -ge_t * f
        gu = gs2 * alive * clipm
        if inj is not None and G <= t < G + inject_steps:
            g_inject[phase(t)] += gu[..., VOLT].sum(0)
            if inject_mode == "clamp":
                gu = gu.copy()
                gu[..., VOLT] = 0
        gs_in += gu
        gds = gu * f
        if D:
            gs_in[..., VOLT:VOLT + 1] += D * gap_junction_flux(gu[..., VOLT:VOLT + 1], pre)
        h = np.maximum(a, 0).reshape(n * hw, -1)
        go = np.concatenate([gds, gpred], -1).reshape(n * hw, C + VIS)
        gW2 += h.T @ go
        gb2 += go.sum(0)
        ga = (go @ W2.T) * (a.reshape(n * hw, -1) > 0)
        gW1 += x.reshape(n * hw, N_IN).T @ ga
        gb1 += ga.sum(0)
        gx = (ga @ W1.T).reshape(n, size, size, N_IN)
        gs_in += perceive_adjoint(gx[..., :4 * C])
        ge = gx[..., 4 * C:] * gain * cm
        if t == G:
            gs_in *= keep
        gs = gs_in
    grad = np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2]).astype(np.float64)
    return float(per_tissue.mean()), grad, {"per_tissue": per_tissue, "state": final_state, "eps": final_eps,
                                             "g_inject": None if g_inject is None else
                                             (g_inject[0] if np.ndim(v_inject) == 2 else g_inject)}
