"""Bonus (problem 1) evaluation. Run: python scripts/eval_papers.py

Claim tested: the embedding scope map recovers the real structure of a 1,200-paper literature
(8 arXiv CS areas, labels never shown to the method) better than a bag-of-words baseline, and the
2D map is faithful. It does NOT measure human understanding; that needs a user study.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from steer import fidelity, papers  # noqa: E402
from steer.encoder import embed  # noqa: E402


def main():
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    P = papers.load_papers()
    texts = [papers.text(p) for p in P]
    y = [p["cat"] for p in P]
    E = embed(texts)
    k = len(papers.CATS)
    out = {"n_papers": len(P), "per_category": {c: y.count(c) for c in papers.CATS}, "seeds": [0, 1, 2, 3, 4]}

    def score(X):
        nmi, ari = [], []
        for s in out["seeds"]:
            lab = KMeans(k, n_init=10, random_state=s).fit_predict(X)
            nmi.append(normalized_mutual_info_score(y, lab))
            ari.append(adjusted_rand_score(y, lab))
        return {"nmi": [float(np.mean(nmi)), float(np.std(nmi))], "ari": [float(np.mean(ari)), float(np.std(ari))]}

    tfidf = TfidfVectorizer(stop_words="english", max_df=0.5, min_df=2, sublinear_tf=True).fit_transform(texts)
    lsa = TruncatedSVD(100, random_state=0).fit_transform(tfidf)
    lsa /= np.linalg.norm(lsa, axis=1, keepdims=True)
    out["clustering"] = {"ours_embedding": score(E), "baseline_tfidf_lsa": score(lsa)}

    labels, names, start = papers.scope_map(E, texts, k)
    out["clusters"] = [{"name": names[j], "size": int((labels == j).sum()), "start_here": P[start[j]]["title"],
                        "majority_category": max(set(np.array(y)[labels == j]), key=list(np.array(y)[labels == j]).count)}
                       for j in range(k)]
    Y, _ = fidelity.project(E, "umap")
    out["map_fidelity_umap"] = fidelity.report(E, Y)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "papers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
