"""Life cycle of a batch of tissues: grow from a founder cell, get wounded, regenerate.

Growth is identical across experimental conditions (same parameters, same founder,
same fire masks drawn from the same seed). Manipulations apply only after the wound,
so every condition is a paired comparison against the same pre-wound tissue.

Comparator conditions during regeneration:

    author      each cell receives its own self-model error from the previous step
    zero        comparator input silenced (ablation)
    transplant  receives the error stream recorded, at the same step, from a *different*
                tissue regenerating from a different wound under the author condition
    delayed     receives its own error, but ``delay`` steps late
    shuffled    receives its own error field, spatially permuted across the tissue's own
                alive cells (same values and timing, wrong cells)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import anatomy
from .tissue import VIS, VOLT, Physics, seed_state, step

COMPARATORS = ("author", "zero", "transplant", "delayed", "shuffled")


@dataclass
class Protocol:
    grow: int = 64
    regen: int = 48
    comparator: str = "author"
    delay: int = 8
    gap_block: bool = False              # diffusion = 0 during regeneration
    voltage_pulse: tuple | None = None   # (value, steps): clamp V of every cell for the first steps after the wound
    voltage_inject: tuple | None = None  # (pattern (H,W), steps[, "add"|"clamp"]): designed V for the first steps after the wound
    second_wound_after: int = 0          # >0: re-wound (same masks) and regrow this many more steps
    record_eps: bool = False
    snapshots: tuple = ()                # absolute step indices at which to keep full states
    tail_window: int = 8                 # losses are averaged over the last steps of each phase
    comparator_mask: np.ndarray | None = field(default=None, repr=False)  # (N,) False = silenced in every phase


def fire_masks(rng: np.random.Generator, n: int, size: int, rate: float, group: int | None = None):
    """Asynchronous-update masks. With ``group`` < n, masks are drawn for ``group`` tissues and
    tiled, so tissue i and tissue i + group share every mask (common random numbers)."""
    g = group or n
    while True:
        m = rng.random((g, size, size, 1), dtype=np.float32) < rate
        yield m if g == n else np.tile(m, (n // g, 1, 1, 1))


def live(theta, wounds: np.ndarray, physics: Physics, proto: Protocol, seed: int,
         donor_eps: np.ndarray | None = None, tgt: np.ndarray | None = None,
         crn_group: int | None = None, on_regen_step=None) -> dict:
    """Run N tissues (N = len(wounds)) through growth, wound and regeneration.

    Returns final state, mean anatomical loss over the tail of growth and regeneration,
    per-step loss traces, and optionally the regeneration error streams and snapshots.
    ``on_regen_step(k, state, eps_in)`` is called before every regeneration step.
    """
    n, size = wounds.shape[0], wounds.shape[1]
    tgt = anatomy.target(size) if tgt is None else tgt
    rng = np.random.default_rng(seed)
    fires = fire_masks(rng, n, size, physics.fire_rate, crn_group)
    shuffle_rng = np.random.default_rng(seed + 7919)
    s = seed_state(n, size, size)
    eps = np.zeros((n, size, size, VIS), np.float32)
    cmask = None if proto.comparator_mask is None else proto.comparator_mask[:, None, None, None].astype(np.float32)
    history = deque(maxlen=max(proto.delay, 1))
    trace, eps_rec, snaps = [], [], {}
    sm_err = np.zeros(n)

    def tick(t, eps_in, diffusion=None, v_clamp=None, v_add=None, v_set=None):
        nonlocal s, eps, sm_err
        if cmask is not None:
            eps_in = eps_in * cmask
        s, eps, _, _ = step(s, eps_in, theta, physics, next(fires), diffusion=diffusion, v_clamp=v_clamp, v_add=v_add, v_set=v_set)
        trace.append(anatomy.loss(s, tgt))
        sm_err += np.einsum("nhwc,nhwc->n", eps, eps) / eps[0].size
        if t in proto.snapshots:
            snaps[t] = s.copy()

    t = 0
    for _ in range(proto.grow):
        history.append(eps)
        tick(t, eps)
        t += 1
    grow_loss = np.mean(trace[-proto.tail_window:], axis=0)
    grown = s.copy()

    def regenerate(steps, phase_offset, pulse=None, inject=None):
        nonlocal t
        for k in range(steps):
            own = eps
            if proto.record_eps:
                eps_rec.append(own.copy())
            if proto.comparator == "author":
                eps_in = own
            elif proto.comparator == "zero":
                eps_in = np.zeros_like(own)
            elif proto.comparator == "transplant":
                eps_in = donor_eps[phase_offset + k]
            elif proto.comparator == "delayed":
                eps_in = history[0] if len(history) == history.maxlen else np.zeros_like(own)
            elif proto.comparator == "shuffled":
                eps_in = _shuffle_within_body(own, s, shuffle_rng)
            else:
                raise ValueError(proto.comparator)
            history.append(own)
            if on_regen_step is not None:
                on_regen_step(phase_offset + k, s, eps_in)
            clamp = None
            if pulse is not None and k < pulse[1]:
                clamp = (np.ones(s[..., :1].shape, bool), pulse[0])
            add = vset = None
            if inject is not None and k < inject[1]:
                if len(inject) > 2 and inject[2] == "clamp":
                    vset = inject[0]
                else:
                    add = inject[0]
            tick(t, eps_in, diffusion=0.0 if proto.gap_block else None, v_clamp=clamp, v_add=add, v_set=vset)
            t += 1

    s = s * ~wounds[..., None]
    regenerate(proto.regen, 0, proto.voltage_pulse, proto.voltage_inject)
    regen_loss = np.mean(trace[-proto.tail_window:], axis=0)
    out = {"state": s, "eps": eps, "grown": grown, "self_model_error": sm_err / len(trace), "grow_loss": grow_loss, "regen_loss": regen_loss,
           "trace": np.stack(trace, axis=1)}
    if proto.second_wound_after:
        s = s * ~wounds[..., None]
        regenerate(proto.second_wound_after, proto.regen)
        out["second_regen_loss"] = np.mean(trace[-proto.tail_window:], axis=0)
        out["state"], out["eps"] = s, eps
    if proto.record_eps:
        out["eps_stream"] = np.stack(eps_rec)
    if snaps:
        out["snapshots"] = snaps
    return out


def _shuffle_within_body(e: np.ndarray, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(e)
    alive = np.abs(s).sum(-1) > 0
    for i in range(e.shape[0]):
        idx = np.flatnonzero(alive[i])
        flat = e[i].reshape(-1, e.shape[-1])
        o = out[i].reshape(-1, e.shape[-1])
        o[idx] = flat[rng.permutation(idx)]
    return out


def voltage(s: np.ndarray) -> np.ndarray:
    return s[..., VOLT]


def continue_life(theta, state, eps, wounds, physics: Physics, steps: int, seed: int,
                  tgt: np.ndarray | None = None, tail_window: int = 8) -> dict:
    """Wound existing tissues (author comparator) and let them regenerate for ``steps`` steps."""
    n, size = state.shape[0], state.shape[1]
    tgt = anatomy.target(size) if tgt is None else tgt
    fires = fire_masks(np.random.default_rng(seed), n, size, physics.fire_rate)
    s = state * ~wounds[..., None]
    e = eps * ~wounds[..., None]
    trace = []
    for _ in range(steps):
        s, e, _, _ = step(s, e, theta, physics, next(fires))
        trace.append(anatomy.loss(s, tgt))
    return {"state": s, "eps": e, "regen_loss": np.mean(trace[-tail_window:], axis=0)}
