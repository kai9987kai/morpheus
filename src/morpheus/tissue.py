"""Tissue: a neural cellular automaton with bioelectric coupling and a per-cell self-model.

Every cell on an H x W grid carries a 13-channel state:

    0      alpha      "aliveness"; a cell is alive while it or a neighbour has alpha > 0.1
    1      V          membrane voltage, coupled to alive neighbours through gap junctions
    2..4   types      head / trunk / tail identity
    5..12  hidden     private state

Each step every cell perceives its 3x3 neighbourhood (identity, Sobel x/y, Laplacian
for every channel) plus its comparator input, and a shared two-layer network emits

    ds    a proposed change to its own 13 channels (its *action*), and
    pred  a forward-model prediction of how its 5 visible channels (alpha, V, types)
          will actually change this step, given that it fires.

Only a random half of the cells fire per step (asynchronous update). Voltage then
diffuses between alive neighbours (a fixed physical law, not learned). The actual
change of the visible channels therefore mixes the cell's own action with its
neighbours' actions and the physics. The self-model error

    eps = actual_visible_change - fire * pred

is the comparator signal. In the *author* condition a cell receives its own eps on the
next step; the experiments replace it with other streams (see ``experiments.py``).

The model is batched over N tissues. Parameters are a flat vector (shared) or an
(N, P) array (one parameter vector per tissue, used by evolution strategies).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C = 13            # channels per cell
VIS = 5           # visible channels the self-model predicts: alpha, V, head, trunk, tail
HID = 32          # hidden units of the update network
N_IN = 4 * C + VIS
N_OUT = C + VIS
ALPHA, VOLT = 0, 1
TYPES = slice(2, 5)
HIDDEN = slice(5, 13)
ALIVE_THRESHOLD = 0.1
STATE_CLIP = 4.0

SHAPES = (("W1", (N_IN, HID)), ("b1", (HID,)), ("W2", (HID, N_OUT)), ("b2", (N_OUT,)))
N_PARAMS = sum(int(np.prod(s)) for _, s in SHAPES)


@dataclass(frozen=True)
class Physics:
    """Fixed laws of the tissue. These are not learned by training (the RSI loop may edit them)."""

    diffusion: float = 0.1    # gap-junction coupling of voltage between alive neighbours
    fire_rate: float = 0.5    # probability that a cell updates on a given step
    gain: float = 1.0         # comparator gain on the self-model error input


def init_params(rng: np.random.Generator) -> np.ndarray:
    """Small random first layer, zero action head (a fresh tissue does nothing), small prediction head."""
    W1 = rng.normal(0, 1 / np.sqrt(N_IN), (N_IN, HID))
    W2 = np.zeros((HID, N_OUT))
    W2[:, C:] = rng.normal(0, 0.01, (HID, VIS))
    return np.concatenate([W1.ravel(), np.zeros(HID), W2.ravel(), np.zeros(N_OUT)]).astype(np.float32)


def unpack(theta: np.ndarray):
    """Flat (P,) or batched (N, P) parameters -> dict of arrays with an optional leading batch dim."""
    out, i = {}, 0
    lead = theta.shape[:-1]
    for name, shape in SHAPES:
        n = int(np.prod(shape))
        out[name] = theta[..., i:i + n].reshape(lead + shape)
        i += n
    return out


def _pad(x: np.ndarray) -> np.ndarray:
    N, H, W, K = x.shape
    p = np.zeros((N, H + 2, W + 2, K), x.dtype)
    p[:, 1:-1, 1:-1] = x
    return p


def perceive(s: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """(N,H,W,C) -> (N,H,W,4C): identity, Sobel-x, Sobel-y, Laplacian (zero padding).

    Uses the separable forms: Sobel-x = [1,2,1]^T (x) [-1,0,1] / 8, and the Laplacian
    kernel [[1,2,1],[2,-12,2],[1,2,1]] / 16 = ([1,2,1]^T (x) [1,2,1] - 16 delta) / 16.
    """
    N, H, W, K = s.shape
    if out is None:
        out = np.empty((N, H, W, 4 * K), s.dtype)
    p = _pad(s)
    vs = p[:, :-2] + 2 * p[:, 1:-1] + p[:, 2:]           # (N,H,W+2,K) vertical smoothing
    hs = p[:, :, :-2] + 2 * p[:, :, 1:-1] + p[:, :, 2:]   # (N,H+2,W,K) horizontal smoothing
    out[..., :K] = s
    np.subtract(vs[:, :, 2:], vs[:, :, :-2], out=out[..., K:2 * K])
    out[..., K:2 * K] *= 1 / 8
    np.subtract(hs[:, 2:], hs[:, :-2], out=out[..., 2 * K:3 * K])
    out[..., 2 * K:3 * K] *= 1 / 8
    lap = vs[:, :, :-2] + 2 * vs[:, :, 1:-1] + vs[:, :, 2:] - 16 * s
    np.multiply(lap, 1 / 16, out=out[..., 3 * K:])
    return out


def perceive_adjoint(g: np.ndarray) -> np.ndarray:
    """Adjoint of ``perceive``: (N,H,W,4C) -> (N,H,W,C).

    With zero padding, the adjoint of a 'same' correlation is the correlation with the
    kernel rotated by 180 degrees: Sobel-x and Sobel-y flip sign, the Laplacian is symmetric.
    """
    K = g.shape[-1] // 4
    gx, gy, gl = g[..., K:2 * K], g[..., 2 * K:3 * K], g[..., 3 * K:]
    px, py, pl = _pad(gx), _pad(gy), _pad(gl)
    vx = px[:, :-2] + 2 * px[:, 1:-1] + px[:, 2:]
    hy = py[:, :, :-2] + 2 * py[:, :, 1:-1] + py[:, :, 2:]
    vl = pl[:, :-2] + 2 * pl[:, 1:-1] + pl[:, 2:]
    out = g[..., :K].copy()
    out -= (vx[:, :, 2:] - vx[:, :, :-2]) / 8
    out -= (hy[:, 2:] - hy[:, :-2]) / 8
    out += (vl[:, :, :-2] + 2 * vl[:, :, 1:-1] + vl[:, :, 2:] - 16 * gl) / 16
    return out


def _neighbours(x: np.ndarray):
    H, W = x.shape[1:3]
    p = _pad(x)
    return lambda dy, dx: p[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]


def alive_mask(s: np.ndarray) -> np.ndarray:
    """(N,H,W,C) -> (N,H,W,1) bool: max-pooled alpha over the 3x3 neighbourhood above threshold."""
    n = _neighbours(s[..., ALPHA:ALPHA + 1])
    m = n(0, 0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                m = np.maximum(m, n(dy, dx))
    return m > ALIVE_THRESHOLD


def gap_junction_flux(v: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Sum over the 4 neighbours of (V_n - V) for pairs of alive cells. v, alive: (N,H,W,1)."""
    m = alive.astype(v.dtype)
    nv, nm = _neighbours(v * m), _neighbours(m)
    flux = np.zeros_like(v)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        flux += nv(dy, dx) - nm(dy, dx) * v
    return flux * m


def seed_state(n: int, h: int, w: int, at=None) -> np.ndarray:
    """A single founder cell (all channels 1 except the identity types) at ``at`` (default: centre)."""
    s = np.zeros((n, h, w, C), np.float32)
    y, x = at if at is not None else (h // 2, w // 2)
    s[:, y, x, :] = 1.0
    s[:, y, x, TYPES] = 0.0
    return s


def step(s, eps_in, theta, physics: Physics, fire, diffusion=None, v_clamp=None, v_add=None, v_set=None):
    """One synchronous tick of N tissues.

    s        (N,H,W,C) state
    eps_in   (N,H,W,VIS) comparator input for this step (own error in the author condition)
    theta    (P,) or (N,P) parameters
    fire     (N,H,W,1) bool: which cells update this step
    diffusion  overrides physics.diffusion (0 models a gap-junction blocker)
    v_clamp  optional (mask (N,H,W,1) bool, value) holding voltage fixed where mask is set
    v_add    optional (H,W) voltage injected into every cell this step (a designed intervention)
    v_set    optional (H,W) voltage every cell is clamped to this step (a designed clamp)

    Returns (new_state, eps_out, pred, alive).
    """
    p = unpack(theta)
    N, H, W, _ = s.shape
    pre = alive_mask(s)
    x = np.empty((N, H, W, N_IN), s.dtype)
    perceive(s, x[..., :4 * C])
    np.multiply(eps_in, physics.gain, out=x[..., 4 * C:])
    x = x.reshape(N, H * W, N_IN)
    W1, W2 = p["W1"], p["W2"]
    b1 = p["b1"] if p["b1"].ndim == 1 else p["b1"][:, None, :]
    b2 = p["b2"] if p["b2"].ndim == 1 else p["b2"][:, None, :]
    hdn = np.maximum(x @ W1 + b1, 0)
    out = (hdn @ W2 + b2).reshape(N, H, W, N_OUT)
    ds, pred = out[..., :C], out[..., C:]
    f = fire.astype(s.dtype)
    new = s + ds * f
    d = physics.diffusion if diffusion is None else diffusion
    if d:
        new[..., VOLT:VOLT + 1] += d * gap_junction_flux(s[..., VOLT:VOLT + 1], pre)
    if v_add is not None:
        new[..., VOLT] += v_add
    if v_set is not None:
        new[..., VOLT] = v_set
    if v_clamp is not None:
        mask, value = v_clamp
        new[..., VOLT:VOLT + 1] = np.where(mask, value, new[..., VOLT:VOLT + 1])
    np.clip(new, -STATE_CLIP, STATE_CLIP, out=new)
    alive = pre & alive_mask(new)
    new *= alive
    eps = ((new - s)[..., :VIS] - f * pred) * alive
    return new.astype(np.float32), eps.astype(np.float32), pred, alive
