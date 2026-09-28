"""Preregistered experiments (see prereg/PREREGISTRATION.json).

Every experiment is paired: each condition reuses the same founders, wounds and fire
masks, so a per-tissue difference isolates the manipulation. Signs are arranged so a
positive difference supports the hypothesis.

H1  authorship      regeneration loss is higher when the comparator is fed someone
                    else's self-model error (transplant) than the tissue's own (author)
H2  gap junctions   blocking voltage coupling during regeneration increases loss
H3  voltage memory  a transient voltage clamp after a head amputation changes the
                    regenerated anatomy, and the change persists through a second
                    amputation with no clamp (two-sided)
CAL null calibration  A/A tests (author vs author under different fire noise) give
                    the false-positive rate of the H1 pipeline
H4  audited RSI     see rsi.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__, anatomy, stats
from .life import Protocol, continue_life, live
from .tissue import C, TYPES, VIS, VOLT

ROOT = Path(__file__).resolve().parents[2]


class RidgeR2:
    """Accumulates sufficient statistics to ask: how much of the comparator input a cell
    receives is linearly predictable from that cell's own current state?"""

    def __init__(self, lam=1e-3):
        k = C + 1
        self.xtx = np.zeros((k, k))
        self.xty = np.zeros((k, VIS))
        self.ysum = np.zeros(VIS)
        self.yy = np.zeros(VIS)
        self.n = 0
        self.lam = lam

    def __call__(self, k, s, eps_in):
        alive = np.abs(s).sum(-1) > 0
        x = s[alive].astype(float)
        y = eps_in[alive].astype(float)
        x = np.concatenate([x, np.ones((len(x), 1))], 1)
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.ysum += y.sum(0)
        self.yy += (y * y).sum(0)
        self.n += len(y)

    def r2(self):
        """Pooled R^2 over the 5 channels, or None when the input is identically zero (zero condition)."""
        if self.yy.sum() < 1e-12:
            return None
        k = self.xtx.shape[0]
        beta = np.linalg.solve(self.xtx + self.lam * np.eye(k), self.xty)
        sse = self.yy - 2 * (beta * self.xty).sum(0) + np.einsum("ij,ik,kj->j", beta, self.xtx, beta)
        sst = self.yy - self.ysum ** 2 / max(self.n, 1)
        return float(1 - sse.sum() / max(sst.sum(), 1e-12))


def _suite(n, seed, kinds=anatomy.WOUNDS):
    rng = np.random.default_rng(seed)
    k, w = anatomy.random_wounds(n, rng, kinds=kinds)
    return k, w, int(rng.integers(1 << 31))


def _summ(out, tgt):
    f = anatomy.fidelity(out["state"], tgt)
    return {"regen_loss": out["regen_loss"], "iou": f["iou"], "region_accuracy": f["region_accuracy"]}


def h1_authorship(theta, physics, n=96, seed=1, log=print) -> dict:
    kinds, wounds, fseed = _suite(n, seed)
    tgt = anatomy.target()
    res, r2 = {}, {}
    hook = RidgeR2()
    author = live(theta, wounds, physics, Protocol(comparator="author", record_eps=True), fseed, on_regen_step=hook)
    res["author"], r2["author"] = _summ(author, tgt), hook.r2()
    donor = np.roll(author["eps_stream"], 1, axis=1)  # tissue i receives tissue i-1's own error stream
    for cond in ("transplant", "zero", "delayed", "shuffled"):
        hook = RidgeR2()
        out = live(theta, wounds, physics, Protocol(comparator=cond), fseed,
                   donor_eps=donor if cond == "transplant" else None, on_regen_step=hook)
        res[cond], r2[cond] = _summ(out, tgt), hook.r2()
        log(f"H1 {cond:10s} regen loss {out['regen_loss'].mean():.4f}  (author {author['regen_loss'].mean():.4f})")
    contrasts = {c: stats.paired(res[c]["regen_loss"] - res["author"]["regen_loss"], seed=seed)
                 for c in ("transplant", "zero", "delayed", "shuffled")}
    return {
        "n_tissues": n, "wound_kinds": {k: kinds.count(k) for k in anatomy.WOUNDS},
        "grow_loss": stats.mean_ci(author["grow_loss"]),
        "grown_fidelity": {k: float(v.mean()) for k, v in anatomy.fidelity(author["grown"], tgt).items()},
        "conditions": {c: {k: stats.mean_ci(v) for k, v in r.items()} for c, r in res.items()},
        "primary": {"H1_transplant_minus_author": contrasts["transplant"]},
        "secondary_holm": stats.holm({c: v["p"] for c, v in contrasts.items()}),
        "contrasts": contrasts,
        "comparator_input_predictable_from_own_state_r2": r2,
        "per_tissue_transplant_minus_author": (res["transplant"]["regen_loss"] - res["author"]["regen_loss"]).tolist(),
    }


def h2_gap_junctions(theta, physics, n=96, seed=2, log=print) -> dict:
    _, wounds, fseed = _suite(n, seed)
    tgt = anatomy.target()
    normal = live(theta, wounds, physics, Protocol(), fseed)
    blocked = live(theta, wounds, physics, Protocol(gap_block=True), fseed)
    log(f"H2 regen loss normal {normal['regen_loss'].mean():.4f}  gap-blocked {blocked['regen_loss'].mean():.4f}")
    return {
        "n_tissues": n,
        "conditions": {"normal": {k: stats.mean_ci(v) for k, v in _summ(normal, tgt).items()},
                       "gap_block": {k: stats.mean_ci(v) for k, v in _summ(blocked, tgt).items()}},
        "primary": {"H2_block_minus_normal": stats.paired(blocked["regen_loss"] - normal["regen_loss"], seed=seed)},
    }


def tail_voltage(theta, physics, n=32, seed=99) -> float:
    """Preregistered clamp value: mean voltage of grown tissues' tail-region cells (calibration suite)."""
    _, wounds, fseed = _suite(n, seed)
    out = live(theta, wounds, physics, Protocol(regen=0), fseed)
    tgt = anatomy.target()
    tail = (tgt[..., 3] > 0.5)[None] & (out["grown"][..., 0] > 0.1)
    return float(out["grown"][..., VOLT][tail].mean())


def _anterior_identity(s, tgt):
    """In the target head region: fraction of cells typed head minus fraction typed tail."""
    head_region = tgt[..., 1] > 0.5
    alive = s[..., 0] > 0.1
    typ = np.argmax(s[..., TYPES], -1)
    denom = head_region.sum()
    return (((typ == anatomy.HEAD) & alive & head_region).sum((1, 2))
            - ((typ == anatomy.TAIL) & alive & head_region).sum((1, 2))) / denom


def h3_voltage_memory(theta, physics, n=96, seed=3, pulse_steps=12, log=print) -> dict:
    _, wounds, fseed = _suite(n, seed, kinds=("head_amputation",))
    tgt = anatomy.target()
    v_star = tail_voltage(theta, physics)
    base = dict(second_wound_after=48)
    ctrl = live(theta, wounds, physics, Protocol(**base), fseed, crn_group=None)
    pulse = live(theta, wounds, physics, Protocol(**base, voltage_pulse=(v_star, pulse_steps)), fseed)
    log(f"H3 v*={v_star:+.3f}  1st regen {ctrl['regen_loss'].mean():.4f} -> {pulse['regen_loss'].mean():.4f}; "
        f"2nd regen {ctrl['second_regen_loss'].mean():.4f} -> {pulse['second_regen_loss'].mean():.4f}")
    # First-regeneration anatomy needs the state at the end of phase 1: rerun without the second wound.
    ctrl1 = live(theta, wounds, physics, Protocol(), fseed)
    pulse1 = live(theta, wounds, physics, Protocol(voltage_pulse=(v_star, pulse_steps)), fseed)
    return {
        "n_tissues": n, "clamp_voltage": v_star, "pulse_steps": pulse_steps,
        "primary": {
            "H3a_pulse_minus_control_first_regen": stats.paired(pulse["regen_loss"] - ctrl["regen_loss"], seed=seed, alternative="two-sided"),
            "H3b_pulse_minus_control_second_regen": stats.paired(pulse["second_regen_loss"] - ctrl["second_regen_loss"], seed=seed, alternative="two-sided"),
        },
        "anterior_identity": {
            "control_first": stats.mean_ci(_anterior_identity(ctrl1["state"], tgt)),
            "pulse_first": stats.mean_ci(_anterior_identity(pulse1["state"], tgt)),
            "control_second": stats.mean_ci(_anterior_identity(ctrl["state"], tgt)),
            "pulse_second": stats.mean_ci(_anterior_identity(pulse["state"], tgt)),
        },
    }


def h3_memory_locus(theta, physics, n=128, seed=3, pulse_steps=12, log=print) -> dict:
    """EXPLORATORY (not preregistered). Where does the clamp-induced change live?

    After the first regeneration (clamped vs control), copy one channel group of the control
    tissue into the clamped tissue, then amputate the head again and regenerate 48 steps.
    If restoring a group abolishes the persistent deficit, that group carries the memory.
    """
    _, wounds, fseed = _suite(n, seed, kinds=("head_amputation",))
    v_star = tail_voltage(theta, physics)
    ctrl = live(theta, wounds, physics, Protocol(), fseed)
    pulse = live(theta, wounds, physics, Protocol(voltage_pulse=(v_star, pulse_steps)), fseed)
    groups = {"none": [], "voltage": [VOLT], "identity": [2, 3, 4], "hidden": list(range(5, C)),
              "all_but_voltage": [0, 2, 3, 4] + list(range(5, C))}
    base = continue_life(theta, ctrl["state"], ctrl["eps"], wounds, physics, 48, fseed + 11)["regen_loss"]
    out = {}
    for name, chans in groups.items():
        s = pulse["state"].copy()
        s[..., chans] = ctrl["state"][..., chans]
        r = continue_life(theta, s, pulse["eps"], wounds, physics, 48, fseed + 11)["regen_loss"]
        out[name] = stats.paired(r - base, seed=seed, alternative="two-sided")
        log(f"H3-locus restore {name:16s} deficit {out[name]['mean']:+.5f}")
    return {"exploratory": True, "n_tissues": n, "restored_group_deficit_vs_control": out}


def calibration(theta, physics, reps=20, n=96, seed=4, log=print) -> dict:
    """A/A tests: the H1 pipeline applied to two author runs that differ only in fire noise."""
    ps = []
    for r in range(reps):
        _, wounds, fseed = _suite(n, seed * 1000 + r)
        a = live(theta, wounds, physics, Protocol(), fseed)
        b = live(theta, wounds, physics, Protocol(), fseed + 1)
        ps.append(stats.paired(b["regen_loss"] - a["regen_loss"], seed=r)["p"])
    ps = np.array(ps)
    log(f"CAL A/A false-positive rate at 0.05: {(ps <= 0.05).mean():.2f} over {reps}")
    return {"reps": reps, "n_tissues": n, "p_values": ps.tolist(), "false_positive_rate_05": float((ps <= 0.05).mean())}


def provenance(weights_path: Path, prereg_path: Path | None) -> dict:
    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest() if p and Path(p).exists() else None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    return {"morpheus": __version__, "python": sys.version.split()[0], "numpy": np.__version__,
            "platform": platform.platform(), "git_commit": commit or None,
            "weights_sha256": sha(weights_path), "prereg_sha256": sha(prereg_path),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items() if k != "theta"}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write(path: Path, doc: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(doc), indent=1), encoding="utf-8")


# ------------------------------------------------------------------------------------------
# v0.2 (prereg/PREREGISTRATION_v2.json)

def h5_compiler(theta, physics, pattern, steps=12, mode="clamp", n=128, seed=5, log=print) -> dict:
    """H5: a designed voltage pattern makes held-out tail stumps grow head tissue where the tail
    was, more than the same values spatially shuffled. H6: the change persists through a second
    tail amputation with no intervention."""
    from . import compiler
    rng = np.random.default_rng(seed)
    wounds = compiler.tail_wounds(n, rng)
    fseed = int(rng.integers(1 << 31))
    shuf = compiler.shuffled(pattern, seed)
    conds = {"none": None, "designed": pattern, "shuffled": shuf}
    first, second = {}, {}
    for name, pat in conds.items():
        proto = Protocol() if pat is None else Protocol(voltage_inject=(pat, steps, mode))
        out = live(theta, wounds, physics, proto, fseed)
        again = continue_life(theta, out["state"], out["eps"], wounds, physics, 48, fseed + 13)
        first[name] = compiler.posterior_head_index(out["state"])
        second[name] = compiler.posterior_head_index(again["state"])
        log(f"H5 {name:9s} posterior head index {first[name].mean():+.3f}   after re-amputation {second[name].mean():+.3f}")
    return {
        "n_tissues": n, "steps": steps, "mode": mode,
        "posterior_head_index": {k: stats.mean_ci(v) for k, v in first.items()},
        "posterior_head_index_after_reamputation": {k: stats.mean_ci(v) for k, v in second.items()},
        "primary": {
            "H5_designed_minus_shuffled": stats.paired(first["designed"] - first["shuffled"], seed=seed),
            "H6_persistence_designed_minus_none": stats.paired(second["designed"] - second["none"], seed=seed),
        },
        "secondary": {"H5_designed_minus_none": stats.paired(first["designed"] - first["none"], seed=seed)},
        "frac_tissues_with_posterior_head_majority": float((first["designed"] > 0).mean()),
    }


def h1_pooled(docs: dict) -> dict:
    """H1 across independently evolved rules: Stouffer combination of the per-rule one-sided
    p-values (equal weights), and how many rules are individually significant."""
    per = {k: d["primary"]["H1_transplant_minus_author"] for k, d in docs.items()}
    z, p = stats.stouffer([v["p"] for v in per.values()])
    return {"rules": list(docs), "per_rule": {k: {x: v[x] for x in ("mean", "d_z", "p", "frac_positive")} for k, v in per.items()},
            "stouffer_z": z, "pooled_p": p, "rules_significant": int(sum(v["p"] < 0.05 for v in per.values()))}
