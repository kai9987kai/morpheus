"""Backpropagation through time must match finite differences and the forward simulator."""

import numpy as np

from morpheus import anatomy, grad, train
from morpheus.life import Protocol
from morpheus.tissue import N_PARAMS, Physics, init_params


def _setup():
    rng = np.random.default_rng(3)
    theta = init_params(rng).astype(np.float64)
    theta[-(32 * 18 + 18):] += rng.normal(0, 0.05, 32 * 18 + 18)
    _, wounds = anatomy.random_wounds(3, rng, size=12)
    return rng, theta, wounds, Protocol(grow=8, regen=6, tail_window=3), Physics(), np.array([True, False, True])


def test_loss_matches_forward_simulator():
    _, theta, wounds, proto, physics, cm = _setup()
    loss, _, _ = grad.loss_and_grad(theta, wounds, physics, proto, 5, comparator_mask=cm, dtype=np.float64)
    fwd, _ = train.objective(theta.astype(np.float32), wounds, physics, proto, 5, cm, anatomy.target(12))
    assert abs(loss - fwd.mean()) < 1e-6


def test_gradient_matches_finite_differences():
    rng, theta, wounds, proto, physics, cm = _setup()
    _, g, _ = grad.loss_and_grad(theta, wounds, physics, proto, 5, comparator_mask=cm, dtype=np.float64)
    h = 1e-5
    for i in rng.choice(N_PARAMS, 12, replace=False):
        e = np.zeros(N_PARAMS)
        e[i] = h
        lp, _, _ = grad.loss_and_grad(theta + e, wounds, physics, proto, 5, comparator_mask=cm, dtype=np.float64)
        lm, _, _ = grad.loss_and_grad(theta - e, wounds, physics, proto, 5, comparator_mask=cm, dtype=np.float64)
        fd = (lp - lm) / (2 * h)
        assert abs(g[i] - fd) <= 1e-5 * max(1e-3, abs(fd)), (i, g[i], fd)


def test_gradient_from_initial_state_matches_finite_differences():
    rng, theta, wounds, _, physics, cm = _setup()
    proto = Protocol(grow=0, regen=6, tail_window=3)
    s0 = np.abs(rng.normal(0, 0.5, (3, 12, 12, 13)))
    e0 = rng.normal(0, 0.1, (3, 12, 12, 5))
    kw = dict(comparator_mask=cm, dtype=np.float64, init_state=s0, init_eps=e0)
    _, g, ex = grad.loss_and_grad(theta, wounds, physics, proto, 5, **kw)
    assert ex["state"].shape == s0.shape
    h = 1e-5
    for i in rng.choice(N_PARAMS, 6, replace=False):
        e = np.zeros(N_PARAMS)
        e[i] = h
        lp, _, _ = grad.loss_and_grad(theta + e, wounds, physics, proto, 5, **kw)
        lm, _, _ = grad.loss_and_grad(theta - e, wounds, physics, proto, 5, **kw)
        fd = (lp - lm) / (2 * h)
        assert abs(g[i] - fd) <= 1e-5 * max(1e-3, abs(fd)), (i, g[i], fd)
