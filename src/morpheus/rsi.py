"""Recursive self-improvement under statistical audit.

The organism repeatedly proposes edits to its own developmental rule (its network
weights) and to its own physics (comparator gain, gap-junction coupling), and decides
whether to adopt each edit. Proposals are "practice" (a few steps of its own learning
algorithm on fresh wounds), random mutation, or a physics edit. Two ways of deciding are compared on the *same* proposer:

    naive  adopt when the edit lowers mean loss on a small evaluation suite that is
           reused for every decision (the usual "benchmark hill-climb")
    gated  test every edit on fresh tissues never used before (paired, common random
           numbers), adopt only when the one-sided sign-flip p-value clears the current
           LORD++ online-FDR level, and never regress on a frozen anchor suite by more
           than a tolerance (anti-forgetting)

    eaudit (v0.3) test every edit *sequentially* on fresh tissues with an anytime-valid betting
           e-process, spending more tissues only while the edit looks promising, and adopt
           when the e-value clears the e-LOND online-FDR level (Xu & Ramdas 2024), with the
           same anchor check. Stopping early for futility or success keeps validity.

Each loop keeps a ledger of what it *believes* it gained (the estimate at the moment
of adoption) and an auditor, which the loop never sees, measures what it *actually*
gained on a large held-out test suite. Their difference is the loop's
self-deception: how much a self-improving system overstates its own progress.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import anatomy, grad, stats, train
from .life import Protocol, live
from .tissue import Physics


@dataclasses.dataclass(frozen=True)
class Config:
    proposals: int = 40
    alpha: float = 0.1               # LORD++ target FDR for adopted edits
    gated_eval: int = 24             # fresh tissues per gated decision
    naive_eval: int = 8              # reused tissues for every naive decision
    anchor: int = 16                 # frozen anchor suite (gated only)
    anchor_tolerance: float = 0.02   # relative anchor regression allowed
    e_batch: int = 8                 # eaudit: fresh tissues per sequential look
    e_max: int = 96                  # eaudit: most tissues spent on one edit
    e_bound: float = 0.001           # eaudit: paired differences are clipped to +-e_bound (fixed a priori)
    e_futility_after: int = 16       # eaudit: stop an edit once its mean difference is <= 0 after this many tissues
    test: int = 128                  # auditor's held-out suite
    practice_steps: int = 2          # "practice": a few Adam steps of backprop on fresh tissues
    practice_batch: int = 8
    practice_lr: float = 2e-4
    mutation_sd: float = 0.001
    physics_sd: float = 0.3
    p_practice: float = 0.4
    p_mutation: float = 0.4          # remainder: physics edits


def suite(n: int, seed: int):
    rng = np.random.default_rng(seed)
    kinds, wounds = anatomy.random_wounds(n, rng)
    return {"kinds": kinds, "wounds": wounds, "seed": int(rng.integers(1 << 31))}


def losses(theta, physics, s) -> np.ndarray:
    out = live(theta, s["wounds"], physics, Protocol(), s["seed"])
    return out["grow_loss"] + out["regen_loss"]


def propose(theta, physics: Physics, rng, cfg: Config):
    u = rng.random()
    if u < cfg.p_practice:
        th = theta.astype(np.float64)
        opt = train.Adam(th.size, cfg.practice_lr)
        for _ in range(cfg.practice_steps):
            _, wounds = anatomy.random_wounds(cfg.practice_batch, rng)
            _, g, _ = grad.loss_and_grad(th, wounds, physics, Protocol(), int(rng.integers(1 << 31)))
            th = th - opt.update(train._normalise(g))
        return "practice", th.astype(np.float32), physics
    if u < cfg.p_practice + cfg.p_mutation:
        return "mutation", (theta + cfg.mutation_sd * rng.standard_normal(theta.size)).astype(np.float32), physics
    field = "gain" if rng.random() < 0.5 else "diffusion"
    new = dataclasses.replace(physics, **{field: float(getattr(physics, field) * math.exp(cfg.physics_sd * rng.standard_normal()))})
    return f"physics:{field}", theta, new


def run_loop(theta0, physics0: Physics, mode: str, seed: int, cfg: Config = Config(), log=None) -> dict:
    """Run one self-improvement loop. ``seed`` fixes the proposer; suites are derived from it."""
    assert mode in ("naive", "gated", "eaudit")
    prop_rng = np.random.default_rng([seed, 1])        # same proposer stream for both modes
    fresh_seeds = np.random.default_rng([seed, 2])
    naive_suite = suite(cfg.naive_eval, 10_000 + seed)
    anchor_suite = suite(cfg.anchor, 20_000 + seed)
    test_suite = suite(cfg.test, 30_000 + seed)        # the auditor's, never used for decisions
    lord = stats.LordPlusPlus(cfg.alpha)

    theta, physics = theta0.copy(), physics0
    test_now = losses(theta, physics, test_suite)
    test_start = float(test_now.mean())
    naive_now = losses(theta, physics, naive_suite) if mode == "naive" else None
    anchor_now = losses(theta, physics, anchor_suite) if mode in ("gated", "eaudit") else None
    elond = stats.ELond(cfg.alpha, horizon=cfg.proposals, uniform=True)
    spent = 0
    believed, ledger = 0.0, []

    for t in range(cfg.proposals):
        kind, cand_theta, cand_phys = propose(theta, physics, prop_rng, cfg)
        row = {"t": t, "kind": kind}
        if mode == "naive":
            cand = losses(cand_theta, cand_phys, naive_suite)
            est = float((naive_now - cand).mean())
            accept = est > 0
            row.update(estimate=est, accepted=bool(accept))
        elif mode == "eaudit":
            ep = stats.BettingEProcess()
            level = elond.level()
            ds = []
            while len(ds) < cfg.e_max:
                fresh = suite(cfg.e_batch, int(fresh_seeds.integers(1 << 31)))
                d = losses(theta, physics, fresh) - losses(cand_theta, cand_phys, fresh)
                ds.extend(d.tolist())
                for x in np.clip(d / cfg.e_bound, -1, 1):
                    ep.update(float(x))
                if ep.wealth >= 1 / level or (len(ds) >= cfg.e_futility_after and np.mean(ds) <= 0):
                    break
            spent += len(ds)
            passed = ep.wealth >= 1 / level
            elond.record(passed)
            est = float(np.mean(ds))
            anchor_ok = True
            if passed:
                anchor_cand = losses(cand_theta, cand_phys, anchor_suite)
                anchor_ok = anchor_cand.mean() <= anchor_now.mean() * (1 + cfg.anchor_tolerance)
            accept = passed and anchor_ok
            row.update(estimate=est, e_value=float(ep.wealth), level=level, tissues=len(ds),
                       anchor_ok=bool(anchor_ok), accepted=bool(accept))
        else:
            fresh = suite(cfg.gated_eval, int(fresh_seeds.integers(1 << 31)))
            d = losses(theta, physics, fresh) - losses(cand_theta, cand_phys, fresh)
            p = stats.sign_flip_p(d, rng=np.random.default_rng(t), alternative="greater")
            passed, level = lord.test(p)
            est = float(d.mean())
            anchor_cand = losses(cand_theta, cand_phys, anchor_suite)
            anchor_ok = anchor_cand.mean() <= anchor_now.mean() * (1 + cfg.anchor_tolerance)
            accept = passed and anchor_ok
            row.update(estimate=est, p=p, level=level, anchor_ok=bool(anchor_ok), accepted=bool(accept))
        if accept:
            test_cand = losses(cand_theta, cand_phys, test_suite)
            row["true_gain"] = float((test_now - test_cand).mean())
            believed += est
            theta, physics, test_now = cand_theta, cand_phys, test_cand
            if mode == "naive":
                naive_now = cand
            else:
                anchor_now = anchor_cand
        row["believed_total"] = believed
        row["true_total"] = test_start - float(test_now.mean())
        ledger.append(row)
        if log:
            log(f"[{mode} seed {seed}] {t:3d} {kind:18s} est {row['estimate']:+.5f} "
                f"{'ADOPT' if accept else 'reject'}  believed {believed:+.4f}  true {row['true_total']:+.4f}")

    adopted = [r for r in ledger if r["accepted"]]
    return {
        "mode": mode, "seed": seed,
        "adopted": len(adopted),
        "false_adoptions": sum(r["true_gain"] <= 0 for r in adopted),
        "believed_gain": believed,
        "true_gain": test_start - float(test_now.mean()),
        "self_deception": believed - (test_start - float(test_now.mean())),
        "test_loss_start": test_start, "test_loss_end": float(test_now.mean()),
        "final_physics": dataclasses.asdict(physics),
        "evaluation_tissues": spent if mode == "eaudit" else (cfg.gated_eval * cfg.proposals if mode == "gated" else cfg.naive_eval),
        "ledger": ledger,
        "theta": theta,
    }
