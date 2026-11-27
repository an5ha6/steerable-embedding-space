"""Two attacks on the main result. Run: python scripts/eval_robustness.py

A) "Linear steering read by linear probes is circular." Re-measure gain and leakage with NONLINEAR
   evaluation readers (MLP, kNN) trained on the probe split, which the method never sees.
B) "Why not just prepend a text prefix?" Compare against prefix steering ("Business news: " + text):
   gain / leakage under all readers, STS-B drop, and steered search on AG News test.

Writes results/robustness.json and results/robustness.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from steer import data  # noqa: E402
from steer.encoder import MODEL_NAME, embed  # noqa: E402
from steer.engine import METHODS, Steerer, apply, fit_probe, sts_quality  # noqa: E402

ALPHA = 0.3
PREFIX = {"sentiment": "Positive review: ", "sports": "Sports news: ", "business": "Business news: ",
          "scitech": "Science and technology news: "}
AG = {0: "world", 1: "sports", 2: "business", 3: "scitech"}
K = 10


def readers(emb, concepts):
    """Three independent reader families per concept, all trained on the probe split only."""
    out = {}
    for c in concepts:
        P, N = emb[c]["pos"]["probe"], emb[c]["neg"]["probe"]
        X, y = np.vstack([P, N]), np.r_[np.ones(len(P)), np.zeros(len(N))]
        out[c] = {"linear": fit_probe(P, N, seed=1),
                  "mlp": MLPClassifier((256,), alpha=1e-3, max_iter=1000, random_state=0).fit(X, y),
                  "knn": KNeighborsClassifier(15, metric="cosine").fit(X, y)}
    return out


def score(R, c, E, S, concepts):
    """Mean gain on c and mean |change| on the other concepts, per reader family."""
    res = {}
    for r in ("linear", "mlp", "knn"):
        p = {o: (R[o][r].predict_proba(E)[:, 1], R[o][r].predict_proba(S)[:, 1]) for o in concepts}
        res[r] = {"gain": float((p[c][1] - p[c][0]).mean()),
                  "leakage": float(np.mean([np.abs(p[o][1] - p[o][0]).mean() for o in concepts if o != c]))}
    return res


def retrieval(st, method_vecs):
    from datasets import load_dataset
    d = load_dataset("fancyzhx/ag_news", split="test")
    texts, y = [t.strip() for t in d["text"]], np.array(d["label"])
    C = embed(texts)
    rng = np.random.default_rng(0)  # same query draw as scripts/eval_extras.py
    out = {}
    for tgt_id, tgt in AG.items():
        if tgt not in st.concepts:
            continue
        q_idx = np.concatenate([rng.choice(np.where(y == c)[0], 100, replace=False) for c in AG if c != tgt_id])
        Q = C[q_idx]
        variants = {"unsteered": Q, "prefix": embed([PREFIX[tgt] + texts[i] for i in q_idx])}
        variants.update({m: apply(Q, st.d(tgt, m), ALPHA) for m in method_vecs})
        out[tgt] = {}
        for name, S in variants.items():
            sims = S @ C.T
            sims[np.arange(len(q_idx)), q_idx] = -9
            top = np.argsort(-sims, 1)[:, :K]
            out[tgt][name] = {"precision_target": float((y[top] == tgt_id).mean()),
                              "relevance_to_original_query": float(np.mean([(C[t] @ Q[i]).mean()
                                                                           for i, t in enumerate(top)]))}
    return out


def main():
    concepts = data.load_concepts()
    emb = {c: {s: {sp: embed(t) for sp, t in v[s].items()} for s in ("pos", "neg")} for c, v in concepts.items()}
    st = Steerer(emb)
    R = readers(emb, st.concepts)
    acc = {c: {r: float(np.mean(np.r_[R[c][r].predict(emb[c]["pos"]["test"]) == 1,
                                       R[c][r].predict(emb[c]["neg"]["test"]) == 0]))
               for r in ("linear", "mlp", "knn")} for c in st.concepts}

    s1, s2, gold = data.load_stsb()
    perm = np.random.default_rng(0).permutation(len(gold))
    test = perm[len(perm) // 2:]  # same STS-B test half as scripts/eval_steering.py
    E1, E2, g = embed([s1[i] for i in test]), embed([s2[i] for i in test]), gold[test]
    q0 = sts_quality(E1, E2, g)

    res = {"model": MODEL_NAME, "alpha": ALPHA, "prefix": PREFIX, "reader_accuracy": acc, "steering": {},
           "sts_drop": {}}
    for c in st.concepts:
        E = emb[c]["neg"]["test"]
        res["steering"][c] = {m: score(R, c, E, apply(E, st.d(c, m), ALPHA), st.concepts) for m in METHODS}
        res["steering"][c]["prefix"] = score(R, c, E, embed([PREFIX[c] + t for t in concepts[c]["neg"]["test"]]),
                                             st.concepts)
        res["sts_drop"][c] = {m: float(q0 - sts_quality(E1, E2, g, st.d(c, m), ALPHA)) for m in METHODS}
        res["sts_drop"][c]["prefix"] = float(q0 - sts_quality(embed([PREFIX[c] + s1[i] for i in test]), E2, g))
    res["retrieval"] = retrieval(st, ("naive", "nullspace"))

    (ROOT / "results" / "robustness.json").write_text(json.dumps(res, indent=1))
    md = to_markdown(res)
    (ROOT / "results" / "robustness.md").write_text(md)
    print(md)


def to_markdown(r):
    cs = list(r["steering"])
    meths = list(r["steering"][cs[0]])
    avg = lambda f: np.mean([f(c) for c in cs])  # noqa: E731
    L = [f"# Robustness checks ({r['model']}, alpha = {r['alpha']}, mean over {len(cs)} concepts)", "",
         "Reader accuracy on held-out test (trained on the probe split, never seen by the method):", "",
         "| concept | linear | MLP | kNN |", "|---|---|---|---|"]
    for c, a in r["reader_accuracy"].items():
        L.append(f"| {c} | {a['linear']:.3f} | {a['mlp']:.3f} | {a['knn']:.3f} |")
    L += ["", "## A + B. Gain and leakage under three reader families", "",
          "| method | linear gain | linear leak | MLP gain | MLP leak | kNN gain | kNN leak | STS-B drop |",
          "|---|---|---|---|---|---|---|---|"]
    for m in meths:
        cells = [f"{avg(lambda c: r['steering'][c][m][rd][k]):.3f}"
                 for rd in ("linear", "mlp", "knn") for k in ("gain", "leakage")]
        L.append(f"| {m} | " + " | ".join(cells) + f" | {avg(lambda c: r['sts_drop'][c][m]):.4f} |")
    L += ["", "## B. Steered search on AG News test (300 queries per target, top-10)", "",
          "| method | precision@10 of target | relevance to original query |", "|---|---|---|"]
    tg = list(r["retrieval"])
    for m in r["retrieval"][tg[0]]:
        p = np.mean([r["retrieval"][t][m]["precision_target"] for t in tg])
        rel = np.mean([r["retrieval"][t][m]["relevance_to_original_query"] for t in tg])
        L.append(f"| {m} | {p:.3f} | {rel:.3f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
