"""Bonus for problem 1: research scope map over hundreds of papers, built on the same embedding space."""
import numpy as np

CATS = ["cs.CV", "cs.CL", "cs.LG", "cs.CR", "cs.RO", "cs.DB", "cs.NI", "cs.SE"]


def load_papers(per_cat=150, seed=0, scan=200_000):
    """arXiv abstracts, balanced over 8 CS areas. Label = primary (first-listed) category."""
    from datasets import load_dataset
    ds = load_dataset("gfissore/arxiv-abstracts-2021", split="train", streaming=True)
    got = {c: [] for c in CATS}
    for i, r in enumerate(ds):
        if i >= scan or all(len(v) >= per_cat for v in got.values()):
            break
        cats = r["categories"]
        primary = (cats[0] if isinstance(cats, list) else cats).split()[0]
        if primary in got and len(got[primary]) < per_cat:
            got[primary].append({"title": " ".join(r["title"].split()), "abstract": " ".join(r["abstract"].split()),
                                 "cat": primary})
    papers = [p for c in CATS for p in got[c]]
    return papers


def text(p):
    return f"{p['title']}. {p['abstract'][:1200]}"


def cluster_labels(texts, assign, k, top=4):
    """Name each cluster by its most distinctive terms (class-based TF-IDF)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = [" ".join(t for t, a in zip(texts, assign) if a == j) for j in range(k)]
    v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.6, min_df=1, sublinear_tf=True)
    M = v.fit_transform(docs).toarray()
    vocab = np.array(v.get_feature_names_out())
    return [", ".join(vocab[np.argsort(-M[j])[:top]]) for j in range(k)]


def scope_map(E, texts, k=8, seed=0):
    """Clusters, names, and a 'start here' paper (closest to centroid) per cluster."""
    from sklearn.cluster import KMeans
    km = KMeans(k, n_init=10, random_state=seed).fit(E)
    names = cluster_labels(texts, km.labels_, k)
    start = []
    for j in range(k):
        idx = np.where(km.labels_ == j)[0]
        start.append(int(idx[np.argmax(E[idx] @ km.cluster_centers_[j])]))
    return km.labels_, names, start
