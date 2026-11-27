"""Reproducible evaluation for problem 5. Run: python scripts/eval_steering.py

Writes results/steering.json and results/steering.md. Fixed seeds everywhere.
Every reported probability comes from *evaluation probes* trained on a split the method never sees.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from steer import data, fidelity  # noqa: E402
from steer.encoder import embed  # noqa: E402
from steer.engine import METHODS, Contract, Steerer, apply, fit_probe, monotonicity, prob, sts_quality  # noqa: E402

GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
OUT = ROOT / "results"


def ci(x, n=1000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(x, (n, len(x))).mean(1)
    return [float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def embed_concepts(concepts):
    return {c: {side: {sp: embed(t) for sp, t in v[side].items()} for side in ("pos", "neg")}
            for c, v in concepts.items()}


def main():
    t0 = time.time()
    concepts = data.load_concepts()
    emb = embed_concepts(concepts)
    s1, s2, gold = data.load_stsb()
    E1, E2 = embed(s1), embed(s2)
    perm = np.random.default_rng(0).permutation(len(gold))
    cal, test = perm[: len(perm) // 2], perm[len(perm) // 2:]
    sts_cal, sts_test = (E1[cal], E2[cal], gold[cal]), (E1[test], E2[test], gold[test])
    q0 = sts_quality(*sts_test)

    st = Steerer(emb)
    evalp = {c: fit_probe(emb[c]["pos"]["probe"], emb[c]["neg"]["probe"], seed=1) for c in st.concepts}
    res = {"model": "BAAI/bge-small-en-v1.5", "concepts": st.concepts, "grid": GRID,
           "n_test_per_concept": {c: len(emb[c]["neg"]["test"]) for c in st.concepts},
           "sts_test_pairs": int(len(test)), "sts_base_spearman": float(q0)}

    # Eval probe sanity: they must actually read the concept, or nothing below means anything.
    res["eval_probe_acc"] = {c: float(np.mean(np.r_[prob(evalp[c], emb[c]["pos"]["test"]) > .5,
                                                     prob(evalp[c], emb[c]["neg"]["test"]) <= .5]))
                             for c in st.concepts}

    # 1) Steering sweep: steer test negatives toward the concept.
    sweep = {}
    for m in METHODS:
        sweep[m] = {}
        for c in st.concepts:
            d, E = st.d(c, m), emb[c]["neg"]["test"]
            base = {o: prob(evalp[o], E) for o in st.concepts}
            rows = []
            for a in GRID:
                S = apply(E, d, a)
                g = prob(evalp[c], S) - base[c]
                leak = np.mean([np.abs(prob(evalp[o], S) - base[o]) for o in st.concepts if o != c], 0)
                rows.append({"alpha": a, "gain": ci(g), "success": float(np.mean(prob(evalp[c], S) > .5)),
                             "leakage": ci(leak), "sts_drop": float(q0 - sts_quality(*sts_test, d, a)),
                             "cos_to_original": float((S * E).sum(1).mean())})
            sweep[m][c] = {"rows": rows, "monotonicity": monotonicity(evalp[c], E, d, np.array(GRID))}
    res["sweep"] = sweep

    # 2) Pairwise geometry of the learned directions.
    res["direction_cosines"] = {m: np.round(np.abs(st.dirs[m] @ st.dirs[m].T), 3).tolist() for m in METHODS}

    # 3) Steering contract vs open-loop baselines, judged by eval probes on test.
    contract = Contract(st, emb, sts_cal)
    reqs = [(t, qb, lb) for t in (0.2, 0.4, 0.6) for qb, lb in ((0.01, 0.03), (0.03, 0.08))]
    crows = []
    for c in st.concepts:
        E = emb[c]["neg"]["test"]
        base = {o: prob(evalp[o], E) for o in st.concepts}
        d = st.d(c)
        for t, qb, lb in reqs:
            out = contract.run(c, E, target=t, max_qdrop=qb, max_leak=lb)
            # baseline A: one fixed alpha calibrated on the mean (open loop)
            cur = contract.curves[c]
            hit = np.where(cur[:, 1] >= t)[0]
            a_fixed = float(cur[hit[0], 0]) if len(hit) else float(cur[-1, 0])
            # baseline B: uncalibrated default slider value
            variants = {"contract": out, "open_loop_calibrated": {"alphas": np.full(len(E), a_fixed)},
                        "uncalibrated_0.3": {"alphas": np.full(len(E), 0.3)}}
            for name, v in variants.items():
                row = {"concept": c, "target": t, "qdrop_budget": qb, "leak_budget": lb, "method": name}
                if v.get("status") == "refused":
                    row.update(status="refused", reason=v["reason"])
                    crows.append(row)
                    continue
                S = E + v["alphas"][:, None] * d
                S /= np.linalg.norm(S, axis=1, keepdims=True)
                g = prob(evalp[c], S) - base[c]
                leak = np.mean([np.abs(prob(evalp[o], S) - base[o]) for o in st.concepts if o != c], 0)
                a_eff = float(np.max(v["alphas"]))
                qd = float(q0 - sts_quality(*sts_test, d, a_eff))
                if "verified" in v:  # contract only: per-input claims, checked by the independent probe
                    vm = v["verified"]
                    row.update(claimed_ok=float(vm.mean()),
                               claim_precision=float(np.mean(g[vm] >= t)) if vm.any() else float("nan"),
                               margin=v["margin"])
                row.update(status="ok", satisfied=float(np.mean(g >= t)), mean_gain=float(g.mean()),
                           mean_leak=float(leak.mean()), sts_drop_at_max_alpha=qd,
                           within_budget=bool(qd <= qb and leak.mean() <= lb),
                           mean_alpha=float(v["alphas"].mean()))
                crows.append(row)
    res["contract"] = crows

    # 4) Visualizer faithfulness on a mixed sample.
    rng = np.random.default_rng(0)
    X = np.vstack([emb[c][s]["test"][rng.choice(len(emb[c][s]["test"]), 125, replace=False)]
                   for c in st.concepts for s in ("pos", "neg")])
    fid = {}
    for m in ("random", "pca", "umap"):
        Y, red = fidelity.project(X, m)
        fid[m] = fidelity.report(X, Y)
        if m != "random":
            steered = apply(emb["sentiment"]["neg"]["test"][:100], st.d("sentiment"), 0.4)
            fid[m]["steered_knn_preservation"], _ = fidelity.out_of_sample_fidelity(red, X, Y, steered)
    res["fidelity"] = {"n_points": int(len(X)), "k": 10, "by_projection": fid}
    res["runtime_sec"] = round(time.time() - t0, 1)

    OUT.mkdir(exist_ok=True)
    (OUT / "steering.json").write_text(json.dumps(res, indent=1))
    (OUT / "steering.md").write_text(to_markdown(res))
    print(to_markdown(res))


def to_markdown(r):
    L = [f"# Steering evaluation ({r['model']})", "",
         f"Test negatives per concept: {r['n_test_per_concept']}. STS-B test pairs: {r['sts_test_pairs']} "
         f"(base Spearman {r['sts_base_spearman']:.3f}). Runtime {r.get('runtime_sec')}s.", "",
         "Eval-probe accuracy (held-out readers, never used by the method): "
         + ", ".join(f"{c} {v:.3f}" for c, v in r["eval_probe_acc"].items()), "",
         "## 1. Sweep at alpha = 0.3 (mean over concepts)", "",
         "| method | gain (target prob) | success rate | leakage (other concepts) | selectivity = leak/gain | STS-B drop | monotonicity |",
         "|---|---|---|---|---|---|---|"]
    i = r["grid"].index(0.3)
    for m, per in r["sweep"].items():
        g = np.mean([v["rows"][i]["gain"][0] for v in per.values()])
        s = np.mean([v["rows"][i]["success"] for v in per.values()])
        lk = np.mean([v["rows"][i]["leakage"][0] for v in per.values()])
        q = np.mean([v["rows"][i]["sts_drop"] for v in per.values()])
        mo = np.mean([v["monotonicity"] for v in per.values()])
        L.append(f"| {m} | {g:.3f} | {s:.3f} | {lk:.3f} | {lk / max(g, 1e-9):.3f} | {q:.4f} | {mo:.3f} |")
    L += ["", "## 2. Per concept, nullspace vs naive, alpha = 0.3 (95% bootstrap CI)", "",
          "| concept | method | gain [CI] | leakage [CI] |", "|---|---|---|---|"]
    for c in r["concepts"]:
        for m in ("naive", "nullspace"):
            row = r["sweep"][m][c]["rows"][i]
            g, lk = row["gain"], row["leakage"]
            L.append(f"| {c} | {m} | {g[0]:.3f} [{g[1]:.3f}, {g[2]:.3f}] | {lk[0]:.3f} [{lk[1]:.3f}, {lk[2]:.3f}] |")
    L += ["", "## 3. Steering contract vs baselines (judged by held-out eval probes)", "",
          "| method | requests | refused | per-input target met | within budget |", "|---|---|---|---|---|"]
    for m in ("contract", "open_loop_calibrated", "uncalibrated_0.3"):
        rows = [x for x in r["contract"] if x["method"] == m]
        ok = [x for x in rows if x["status"] == "ok"]
        sat = np.mean([x["satisfied"] for x in ok]) if ok else float("nan")
        wb = np.mean([x["within_budget"] for x in ok]) if ok else float("nan")
        L.append(f"| {m} | {len(rows)} | {len(rows) - len(ok)} | {sat:.3f} | {wb:.3f} |")
    ok = [x for x in r["contract"] if x["method"] == "contract" and x["status"] == "ok"]
    if ok:
        L += ["", "Contract per-input claims: says 'verified' for "
              f"{np.mean([x['claimed_ok'] for x in ok]):.3f} of inputs; when it says so, the independent probe "
              f"agrees {np.nanmean([x['claim_precision'] for x in ok]):.3f} of the time "
              "(baselines make no per-input claim)."]
    L += ["", f"## 4. Visualizer faithfulness (n={r['fidelity']['n_points']}, k={r['fidelity']['k']})", "",
          "| projection | trustworthiness | continuity | kNN preservation | distance Spearman | steered-point kNN |",
          "|---|---|---|---|---|---|"]
    for m, f in r["fidelity"]["by_projection"].items():
        sk = f.get("steered_knn_preservation")
        L.append(f"| {m} | {f['trustworthiness']:.3f} | {f['continuity']:.3f} | {f['knn_preservation']:.3f} | "
                 f"{f['distance_spearman']:.3f} | {'-' if sk is None else f'{sk:.3f}'} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
