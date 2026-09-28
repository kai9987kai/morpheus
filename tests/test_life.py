import numpy as np

from morpheus import anatomy
from morpheus.life import Protocol, live
from morpheus.tissue import Physics, init_params, perceive, perceive_adjoint


def _theta():
    rng = np.random.default_rng(0)
    th = init_params(rng)
    th[-(32 * 18 + 18):] += rng.normal(0, 0.05, 32 * 18 + 18).astype(np.float32)
    return th


def test_perceive_adjoint():
    rng = np.random.default_rng(1)
    s, g = rng.standard_normal((2, 6, 7, 13)), rng.standard_normal((2, 6, 7, 52))
    assert abs((perceive(s) * g).sum() - (s * perceive_adjoint(g)).sum()) < 1e-9


def test_deterministic_and_paired():
    th, ph = _theta(), Physics()
    _, w = anatomy.random_wounds(4, np.random.default_rng(2), size=16)
    p = Protocol(grow=10, regen=8, record_eps=True)
    a = live(th, w, ph, p, 7)
    b = live(th, w, ph, p, 7)
    np.testing.assert_array_equal(a["state"], b["state"])
    # Conditions differ only after the wound: growth is identical.
    z = live(th, w, ph, Protocol(grow=10, regen=8, comparator="zero"), 7)
    np.testing.assert_array_equal(a["grown"], z["grown"])
    # Transplanting a tissue's own stream back into itself reproduces the author condition.
    t = live(th, w, ph, Protocol(grow=10, regen=8, comparator="transplant"), 7, donor_eps=a["eps"])
    np.testing.assert_allclose(t["state"], a["state"], atol=1e-6)


def test_voltage_clamp_changes_voltage():
    th, ph = _theta(), Physics()
    _, w = anatomy.random_wounds(2, np.random.default_rng(3), size=16)
    a = live(th, w, ph, Protocol(grow=10, regen=6), 1)
    b = live(th, w, ph, Protocol(grow=10, regen=6, voltage_pulse=(0.9, 3)), 1)
    assert not np.allclose(a["state"][..., 1], b["state"][..., 1])


def test_target_has_three_regions():
    t = anatomy.target()
    assert t[..., 0].sum() > 150
    assert all(t[..., 1 + r].sum() > 30 for r in range(3))
    assert np.all(t[..., 1:].sum(-1) == t[..., 0])
