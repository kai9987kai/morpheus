"""Command line: train rules, run the preregistered experiments, render tissues, export the web lab."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import anatomy, experiments, rsi, train
from .life import Protocol, live

ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "prereg" / "PREREGISTRATION.json"
PREREG_V2 = ROOT / "prereg" / "PREREGISTRATION_v2.json"


def _log(msg):
    print(msg, flush=True)


def cmd_train(a):
    t0 = time.time()
    theta, hist = train.train_bptt(iterations=a.iterations, seed=a.seed, workers=a.workers, log=_log)
    physics = train.Physics()
    train.save_weights(a.out, theta, physics, {
        "trainer": "bptt+persistence-pool", "iterations": a.iterations, "batch": 16, "lr": 2e-3, "seed": a.seed,
        "final_loss_last50": float(np.mean(hist[-50:])), "seconds": round(time.time() - t0)})
    _log(f"saved {a.out}")


def _rsi_job(args):
    weights, mode, seed, proposals = args
    theta, physics, _ = train.load_weights(weights)
    cfg = rsi.Config(proposals=proposals)
    return rsi.run_loop(theta, physics, mode, seed, cfg, log=_log)


def run_rsi(weights, seeds, proposals, workers):
    jobs = [(weights, mode, s, proposals) for s in seeds for mode in ("naive", "gated")]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            loops = pool.map(_rsi_job, jobs)
    else:
        loops = [_rsi_job(j) for j in jobs]
    from . import stats
    by = {(r["mode"], r["seed"]): r for r in loops}
    dec = np.array([by[("naive", s)]["self_deception"] - by[("gated", s)]["self_deception"] for s in seeds])
    summary = {}
    for mode in ("naive", "gated"):
        rs = [by[(mode, s)] for s in seeds]
        summary[mode] = {k: stats.mean_ci([r[k] for r in rs]) for k in
                         ("adopted", "false_adoptions", "believed_gain", "true_gain", "self_deception")}
        summary[mode]["false_adoption_fraction"] = float(
            sum(r["false_adoptions"] for r in rs) / max(1, sum(r["adopted"] for r in rs)))
    return {"seeds": list(seeds), "proposals": proposals, "summary": summary,
            "primary": {"H4_naive_minus_gated_self_deception": stats.paired(dec, alternative="greater")},
            "loops": loops}


def run_eaudit(weights, seeds, proposals, workers, baseline):
    """H8: e-audited loops on the same seeds as the v0.2 naive and gated loops, compared per seed."""
    jobs = [(weights, "eaudit", s, proposals) for s in seeds]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            loops = pool.map(_rsi_job, jobs)
    else:
        loops = [_rsi_job(j) for j in jobs]
    from . import stats
    base = json.loads(Path(baseline).read_text(encoding="utf-8"))["H4"]["loops"]
    by = {(r["mode"], r["seed"]): r for r in base + loops}
    col = lambda mode, k: np.array([by[(mode, s)][k] for s in seeds])  # noqa: E731
    summary = {}
    for mode in ("naive", "gated", "eaudit"):
        summary[mode] = {k: stats.mean_ci(col(mode, k)) for k in ("adopted", "false_adoptions", "believed_gain", "true_gain", "self_deception")}
        if mode != "naive":   # v0.2 loops predate the field; a gated loop always spends 24 per proposal
            spent = [by[(mode, s)].get("evaluation_tissues", 24 * proposals) for s in seeds]
            summary[mode]["evaluation_tissues_per_loop"] = float(np.mean(spent))
    return {"seeds": seeds, "proposals": proposals, "baseline": baseline, "summary": summary,
            "primary": {"H8a_eaudit_minus_gated_true_gain": stats.paired(col("eaudit", "true_gain") - col("gated", "true_gain"), alternative="greater"),
                        "H8b_naive_minus_eaudit_self_deception": stats.paired(col("naive", "self_deception") - col("eaudit", "self_deception"), alternative="greater")},
            "loops": loops}


def cmd_run(a):
    if a.experiment == "h1pool":
        docs = {Path(f).stem: json.loads(Path(f).read_text(encoding="utf-8"))["H1"] for f in a.files}
        experiments.write(Path(a.out), {"H1_pooled": experiments.h1_pooled(docs),
                                        "files": a.files, "provenance": experiments.provenance(Path(a.files[0]), PREREG_V2)})
        _log(f"wrote {a.out}")
        return
    theta, physics, meta = train.load_weights(a.weights)
    out = Path(a.out)
    which = a.experiment
    doc = {"weights": str(a.weights), "weights_meta": meta,
           "provenance": experiments.provenance(Path(a.weights), PREREG)}
    doc["provenance"]["prereg_v2_sha256"] = experiments.provenance(Path(a.weights), PREREG_V2)["prereg_sha256"]
    t0 = time.time()
    if which in ("h1", "all"):
        doc["H1"] = experiments.h1_authorship(theta, physics, n=a.n, log=_log)
    if which in ("h2", "all"):
        doc["H2"] = experiments.h2_gap_junctions(theta, physics, n=a.n, log=_log)
    if which in ("h3", "all"):
        doc["H3"] = experiments.h3_voltage_memory(theta, physics, n=a.n, log=_log)
    if which == "design":
        from . import compiler
        r = compiler.design(theta, physics, iterations=a.iterations, seed=a.seed, workers=a.workers, log=_log)
        doc["design"] = {k: v for k, v in r.items() if k != "history"}
        doc["design"]["loss_first20"] = float(np.mean(r["history"][:20]))
        doc["design"]["loss_last20"] = float(np.mean(r["history"][-20:]))
    if which == "design3":
        from . import compiler
        r = compiler.design_v3(theta, physics, seed=a.seed, workers=a.workers, log=_log)
        doc["design"] = {k: v for k, v in r.items() if k != "history"}
        doc["design"]["loss_last20"] = float(np.mean(r["history"][-20:]))
    if which == "h7":
        p3 = json.loads(Path(a.pattern).read_text(encoding="utf-8"))["design"]
        p2 = json.loads(Path(a.pattern_v2).read_text(encoding="utf-8"))["design"]
        doc["pattern_files"] = [a.pattern, a.pattern_v2]
        doc["H7"] = experiments.h7_compiler_v3(theta, physics, np.asarray(p3["pattern"], np.float32), p3["steps"],
                                               np.asarray(p2["pattern"], np.float32), p2["steps"], n=a.n, log=_log)
    if which == "h5":
        pdoc = json.loads(Path(a.pattern).read_text(encoding="utf-8"))["design"]
        doc["pattern_file"] = a.pattern
        doc["H5"] = experiments.h5_compiler(theta, physics, np.asarray(pdoc["pattern"], np.float32),
                                            steps=pdoc["steps"], mode=pdoc["mode"], n=a.n, log=_log)
    if which == "h3locus":
        doc["H3_locus_exploratory"] = experiments.h3_memory_locus(theta, physics, n=a.n, log=_log)
    if which in ("cal", "all"):
        doc["CAL"] = experiments.calibration(theta, physics, reps=a.reps, n=a.n, log=_log)
    if which == "eaudit":
        doc["H8"] = run_eaudit(a.weights, list(range(a.seeds)), a.proposals, a.workers, a.baseline)
    if which in ("h4", "rsi"):
        doc["H4"] = run_rsi(a.weights, list(range(a.seeds)), a.proposals, a.workers)
    doc["seconds"] = round(time.time() - t0)
    experiments.write(out, doc)
    _log(f"wrote {out}")


def render(s) -> list[str]:
    rows = []
    for y in range(s.shape[0]):
        r = ""
        for x in range(s.shape[1]):
            r += "HTt"[int(np.argmax(s[y, x, 2:5]))] if s[y, x, 0] > 0.1 else "."
        rows.append(r)
    return rows


def cmd_show(a):
    theta, physics, _ = train.load_weights(a.weights)
    rng = np.random.default_rng(a.seed)
    wounds = np.stack([anatomy.wound(a.wound, rng)])
    out = live(theta, wounds, physics, Protocol(), a.seed)
    tgt = anatomy.target()
    panels = [render(tgt[..., [0, 0, 1, 2, 3]]), render(out["grown"][0]),
              render(out["grown"][0] * ~wounds[0][..., None]), render(out["state"][0])]
    print("   target".ljust(35) + "grown".ljust(34) + f"wounded ({a.wound})".ljust(34) + "regenerated")
    for rows in zip(*panels):
        print("   ".join(rows))
    print(f"grow loss {out['grow_loss'][0]:.4f}   regen loss {out['regen_loss'][0]:.4f}")


def cmd_export_web(a):
    theta, physics, meta = train.load_weights(a.weights)
    data = {"weights": {"theta": [round(float(v), 6) for v in theta], "physics": physics.__dict__, "meta": meta},
            "target": anatomy.target().tolist(), "results": {}, "claims": None}
    data["designs"] = {}
    for p in sorted((ROOT / "results").glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        if "design" in doc:
            data["designs"][p.stem] = {"pattern": [[round(v, 3) for v in row] for row in doc["design"]["pattern"]],
                                       "steps": doc["design"]["steps"], "mode": doc["design"]["mode"],
                                       "weights": doc["weights"]}
            continue
        data["results"][p.stem] = _slim(doc)
    ledger = ROOT / "claims" / "claims.json"
    if ledger.exists():
        from . import claims
        data["claims"] = claims.rendered_rows()
    Path(a.out).write_text("window.MORPHEUS_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    _log(f"wrote {a.out}")


def _slim(doc):
    """Drop bulky per-tissue arrays from results before embedding them in the web page."""
    if isinstance(doc, dict):
        return {k: _slim(v) for k, v in doc.items() if k not in ("p_values",)}
    if isinstance(doc, list):
        return [_slim(v) for v in doc]
    return doc


def cmd_claims(a):
    from . import claims
    sys.exit(claims.main(a.action))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="morpheus", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train a rule by backpropagation through time")
    t.add_argument("--iterations", type=int, default=3000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--out", default="weights/rule_a.json")
    t.set_defaults(fn=cmd_train)
    r = sub.add_parser("run", help="run preregistered experiments")
    r.add_argument("experiment", choices=["h1", "h2", "h3", "cal", "all", "h4", "rsi", "h3locus", "design", "h5", "h1pool", "design3", "h7", "eaudit"])
    r.add_argument("--weights", default="weights/rule_a.json")
    r.add_argument("--n", type=int, default=128)
    r.add_argument("--reps", type=int, default=40)
    r.add_argument("--seeds", type=int, default=6)
    r.add_argument("--proposals", type=int, default=40)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--iterations", type=int, default=600, help="design iterations")
    r.add_argument("--seed", type=int, default=0, help="design seed")
    r.add_argument("--pattern", help="results file holding a designed pattern (for h5)")
    r.add_argument("--pattern-v2", help="v0.2 design results file (for h7)")
    r.add_argument("--baseline", help="v0.2 RSI results with the naive and gated loops (for eaudit)")
    r.add_argument("--files", nargs="*", default=["results/rule_a.json", "results/rule_b.json", "results/rule_c.json"],
                   help="per-rule results files (for h1pool)")
    r.add_argument("--out", required=True)
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("show", help="print a tissue growing, being wounded and regenerating")
    s.add_argument("--weights", default="weights/rule_a.json")
    s.add_argument("--wound", default="head_amputation", choices=anatomy.WOUNDS)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_show)
    w = sub.add_parser("export-web", help="write web/data.js for the interactive lab")
    w.add_argument("--weights", default="weights/rule_a.json")
    w.add_argument("--out", default=str(ROOT / "web" / "data.js"))
    w.set_defaults(fn=cmd_export_web)
    c = sub.add_parser("claims", help="check or render the claims ledger")
    c.add_argument("action", choices=["check", "render"])
    c.set_defaults(fn=cmd_claims)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
