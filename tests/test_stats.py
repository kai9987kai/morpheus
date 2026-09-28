import numpy as np

from morpheus import stats


def test_sign_flip_exact_extremes():
    assert stats.sign_flip_p(np.ones(6)) == 1 / 64
    assert stats.sign_flip_p(-np.ones(6), alternative="less") == 1 / 64
    assert stats.sign_flip_p(np.array([1.0, -1.0])) == 0.75


def test_paired_ci_contains_mean():
    d = np.random.default_rng(0).normal(0.5, 1, 40)
    r = stats.paired(d)
    assert r["ci95"][0] < r["mean"] < r["ci95"][1]
    assert r["p"] < 0.05


def test_holm():
    out = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03}, alpha=0.05)
    assert out["a"]["reject"] and not out["b"]["reject"] and not out["c"]["reject"]


def test_lord_controls_false_discoveries_under_global_null():
    rng = np.random.default_rng(0)
    runs, any_false = 200, 0
    for _ in range(runs):
        lord = stats.LordPlusPlus(alpha=0.1)
        any_false += any(lord.test(p)[0] for p in rng.random(50))
    # Under the global null every rejection is false, so FDR = P(any rejection).
    assert any_false / runs <= 0.15


def test_lord_level_is_positive_and_below_alpha():
    lord = stats.LordPlusPlus(alpha=0.1)
    assert 0 < lord.level() < 0.1
    lord.test(0.0)
    assert 0 < lord.level() < 0.1


def test_stouffer():
    z, p = stats.stouffer([0.5, 0.5, 0.5])
    assert abs(z) < 1e-6 and abs(p - 0.5) < 1e-6
    z, p = stats.stouffer([0.05])
    assert abs(z - 1.6449) < 1e-3 and abs(p - 0.05) < 1e-6
