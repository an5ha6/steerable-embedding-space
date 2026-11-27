"""Two extra claims, measured. Run: python scripts/eval_extras.py -> results/extras.json, results/extras.md

A) Downstream retrieval: does steering a search query actually change what you get back, as intended?
   Corpus = AG News *test* split (7,600 articles, never used to learn directions).
B) Concept erasure: can we remove a concept so that no freshly trained linear reader can recover it,
   while general semantic similarity (STS-B) and the other concepts survive?
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from steer import data  # noqa: E402
from steer.encoder import MODEL_NAME, embed  # noqa: E402
from steer.engine import METHODS, Steerer, apply, erase, erase_basis, fit_probe, prob, sts_quality  # noqa: E402

AG = {0: "world", 1: "sports", 2: "business", 3: "scitech"}
ALPHAS = [0.0, 0.1, 0.2, 0.3]
K = 10


def retrieval(st):
    from datasets import load_dataset
    d = load_dataset("fancyzhx/ag_news", split="test")
    texts, y = [t.strip() for t in d["text"]], np.array(d["label"])
    C = embed(texts)
    rng = np.random.default_rng(0)
    out = {}
    for tgt_id, tgt in AG.items():
        if tgt not in st.concepts:
            continue
        q_idx = np.concatenate([rng.choice(np.where(y == c)[0], 100, replace=False) for c in AG if c != tgt_id])
        Q = C[q_idx]
        out[tgt] = {}
        for m in METHODS:
            rows = []
            for a in ALPHAS:
                S = apply(Q, st.d(tgt, m), a)
                sims = S @ C.T
                sims[np.arange(len(q_idx)), q_idx] = -9  # never retrieve the query itself
                top = np.argsort(-sims, 1)[:, :K]
                lab = y[top]
                own = y[q_idx][:, None]
                rows.append({"alpha": a,
                             "precision_target": float((lab == tgt_id).mean()),
                             "drift_to_other_classes": float(((lab != tgt_id) & (lab != own)).mean()),
                             "relevance_to_original_query": float(np.mean([(C[t] @ Q[i]).mean() for i, t in enumerate(top)]))})
            out[tgt][m] = rows
    return out


def erasure(emb, sts, st):
    E1, E2, gold = sts
    q0 = sts_quality(E1, E2, gold)
    rng = np.random.default_rng(0)
    out = {}
    for c in st.concepts:
        Qfull = erase_basis(emb[c]["pos"]["fit"], emb[c]["neg"]["fit"], iters=16)
        rows = []
        for k in (0, 1, 2, 4, 8, 16):
            res = {"k": k}
            for name, Q in (("ours", Qfull[:, :k]),
                            ("random", np.linalg.qr(rng.standard_normal((Qfull.shape[0], max(k, 1))))[0][:, :k])):
                f = (lambda E: E) if k == 0 else (lambda E, Q=Q: erase(E, Q))
                acc = {}
                for o in st.concepts:  # fresh reader trained AFTER erasure, on a split erasure never saw
                    p = fit_probe(f(emb[o]["pos"]["probe"]), f(emb[o]["neg"]["probe"]), seed=7)
                    acc[o] = float(np.mean(np.r_[prob(p, f(emb[o]["pos"]["test"])) > .5,
                                                 prob(p, f(emb[o]["neg"]["test"])) <= .5]))
                res[name] = {"target_acc": acc[c],
                             "other_acc_mean": float(np.mean([v for o, v in acc.items() if o != c])),
                             "sts_drop": float(q0 - sts_quality(f(E1), f(E2), gold))}
            rows.append(res)
        out[c] = rows
    return out


def main():
    concepts = data.load_concepts()
    emb = {c: {s: {sp: embed(t) for sp, t in v[s].items()} for s in ("pos", "neg")} for c, v in concepts.items()}
    st = Steerer(emb)
    s1, s2, gold = data.load_stsb()
    res = {"model": MODEL_NAME, "retrieval": retrieval(st), "erasure": erasure(emb, (embed(s1), embed(s2), gold), st)}
    (ROOT / "results" / "extras.json").write_text(json.dumps(res, indent=1))
    md = to_markdown(res)
    (ROOT / "results" / "extras.md").write_text(md)
    print(md)


def to_markdown(r):
    L = [f"# Extras ({r['model']})", "", "## A. Steered search on AG News test (300 queries per target, top-10)", "",
         "Mean over targets sports / business / sci-tech. Queries come from the other 3 classes.", "",
         "| method | alpha | precision@10 of target | drift to unrelated classes | relevance to original query |",
         "|---|---|---|---|---|"]
    tg = list(r["retrieval"])
    for m in METHODS:
        for i, a in enumerate(ALPHAS):
            if a == 0 and m != "naive":
                continue
            v = lambda key: np.mean([r["retrieval"][t][m][i][key] for t in tg])  # noqa: E731
            L.append(f"| {'unsteered' if a == 0 else m} | {a} | {v('precision_target'):.3f} | "
                     f"{v('drift_to_other_classes'):.3f} | {v('relevance_to_original_query'):.3f} |")
    L += ["", "## B. Concept erasure (fresh linear reader trained after erasure; chance = 0.5)", "",
          "| concept | k removed dims | target acc (ours) | target acc (random dims) | other concepts acc (ours) | STS-B drop (ours) |",
          "|---|---|---|---|---|---|"]
    for c, rows in r["erasure"].items():
        for x in rows:
            L.append(f"| {c} | {x['k']} | {x['ours']['target_acc']:.3f} | {x['random']['target_acc']:.3f} | "
                     f"{x['ours']['other_acc_mean']:.3f} | {x['ours']['sts_drop']:.4f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
