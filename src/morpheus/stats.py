"""Statistics: paired effect sizes, resampling tests and online false-discovery control."""

from __future__ import annotations

import math

import numpy as np


def paired(diff, n_boot=10000, n_perm=10000, seed=0, alternative="greater") -> dict:
    """Summary of a paired difference vector (treatment - control, per pair).

    mean, bootstrap 95% CI of the mean, Cohen's d_z, fraction positive, and a sign-flip
    permutation p-value (exact when n <= 16, Monte Carlo otherwise).
    """
    d = np.asarray(diff, float)
    n = d.size
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, n), replace=True).mean(axis=1)
    sd = d.std(ddof=1) if n > 1 else float("nan")
    return {
        "n": int(n),
        "mean": float(d.mean()),
        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        "d_z": float(d.mean() / sd) if sd and sd > 0 else 0.0,
        "frac_positive": float((d > 0).mean()),
        "p": sign_flip_p(d, n_perm, rng, alternative),
        "alternative": alternative,
    }


def sign_flip_p(d, n_perm=10000, rng=None, alternative="greater") -> float:
    d = np.asarray(d, float)
    n = d.size
    obs = d.mean()
    if n <= 16:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        rng = rng or np.random.default_rng(0)
        signs = rng.choice([-1, 1], (n_perm, n))
    null = (signs * d).mean(axis=1)
    if alternative == "greater":
        k = (null >= obs - 1e-12).sum()
    elif alternative == "less":
        k = (null <= obs + 1e-12).sum()
    else:
        k = (np.abs(null) >= abs(obs) - 1e-12).sum()
    if n <= 16:
        return float(k / len(null))
    return float((k + 1) / (len(null) + 1))


def holm(pvals: dict, alpha=0.05) -> dict:
    """Holm-Bonferroni step-down. Returns {name: {"p": p, "p_adj": ..., "reject": bool}}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running, stop = len(items), {}, 0.0, False
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        reject = not stop and p <= alpha / (m - i)
        stop = stop or not reject
        out[k] = {"p": p, "p_adj": running, "reject": reject}
    return out


class LordPlusPlus:
    """LORD++ online FDR control (Ramdas, Yang, Wainwright & Jordan, NeurIPS 2017).

    Hypotheses arrive one at a time; test t is run at level alpha_t which depends only on
    past decisions. The false discovery rate over the whole (unbounded) sequence is
    controlled at ``alpha`` for independent p-values. Each self-modification the RSI loop
    considers is one hypothesis ("this edit improves the organism"), so the loop can run
    forever without its accepted edits becoming mostly noise.
    """

    def __init__(self, alpha=0.1, w0=None, horizon=100000):
        self.alpha = alpha
        self.w0 = alpha / 2 if w0 is None else w0
        j = np.arange(1, horizon + 1, dtype=float)
        g = np.log(np.maximum(j, 2)) / (j * np.exp(np.sqrt(np.log(j))))
        self.gamma = g / g.sum()
        self.t = 0
        self.rejections: list[int] = []

    def _g(self, k):
        return self.gamma[k - 1] if 1 <= k <= len(self.gamma) else 0.0

    def level(self) -> float:
        t = self.t + 1
        a = self._g(t) * self.w0
        if self.rejections:
            a += (self.alpha - self.w0) * self._g(t - self.rejections[0])
            a += self.alpha * sum(self._g(t - r) for r in self.rejections[1:])
        return float(a)

    def test(self, p: float) -> tuple[bool, float]:
        a = self.level()
        self.t += 1
        reject = p <= a
        if reject:
            self.rejections.append(self.t)
        return reject, a


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF by bisection on math.erf (no SciPy)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def stouffer(pvals) -> tuple[float, float]:
    """Combine one-sided p-values with equal weights: Z = sum(Phi^-1(1 - p_i)) / sqrt(k)."""
    zs = [_norm_ppf(1 - min(max(p, 1e-15), 1 - 1e-15)) for p in pvals]
    z = sum(zs) / math.sqrt(len(zs))
    return float(z), float(0.5 * math.erfc(z / math.sqrt(2)))


def mean_ci(x, seed=0, n_boot=10000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n_boot, x.size)).mean(axis=1)
    return {"mean": float(x.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))], "n": int(x.size)}


def fmt_p(p: float) -> str:
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def isfinite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


class BettingEProcess:
    """Anytime-valid test of H0: E[x] <= 0 for x in [-1, 1] (Waudby-Smith & Ramdas 2024).

    Wealth K_n = prod(1 + lambda_i x_i) with a predictable bet lambda_i in [0, 1/2] (aGRAPA-style
    plug-in from the observations before i). Under H0, K is a nonnegative supermartingale, so
    P(sup K >= 1/a) <= a at any stopping time (Ville's inequality).
    """

    def __init__(self):
        self.wealth, self.n, self.s, self.ss = 1.0, 0, 0.0, 0.0

    def update(self, x: float) -> float:
        mu = self.s / (self.n + 1)
        var = (self.ss - self.s ** 2 / max(self.n, 1) + 0.25) / (self.n + 1) if self.n else 0.25
        lam = min(max(mu / (var + mu * mu), 0.0), 0.5) if mu > 0 else 0.0
        if self.n == 0:
            lam = 0.1
        self.wealth *= 1 + lam * x
        self.n += 1
        self.s += x
        self.ss += x * x
        return self.wealth


class ELond:
    """e-LOND online FDR control with e-values (Xu & Ramdas 2024): test t rejects when its e-value
    reaches 1/alpha_t with alpha_t = alpha * gamma_t * (discoveries so far + 1). Valid under
    arbitrary dependence between tests."""

    def __init__(self, alpha=0.1, horizon=100000, uniform=False):
        """With ``uniform=True`` the spending sequence is gamma_t = 1/horizon for a known, finite
        number of tests (still sums to 1, so the guarantee holds), instead of a decaying one."""
        self.alpha = alpha
        j = np.arange(1, horizon + 1, dtype=float)
        g = np.ones_like(j) if uniform else np.log(np.maximum(j, 2)) / (j * np.exp(np.sqrt(np.log(j))))
        self.gamma = g / g.sum()
        self.t, self.discoveries = 0, 0

    def level(self) -> float:
        return float(self.alpha * self.gamma[min(self.t, len(self.gamma) - 1)] * (self.discoveries + 1))

    def record(self, rejected: bool):
        self.t += 1
        self.discoveries += int(rejected)
