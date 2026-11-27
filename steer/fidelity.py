"""How faithful is a 2D map to the 384-D space? Standard, citable metrics."""
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors


def knn_sets(X, k, metric):
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def report(X_high, X_2d, k=10):
    """trustworthiness: are 2D neighbours real neighbours? (penalises false clusters)
    continuity: are real neighbours kept close in 2D? (penalises torn clusters)
    knn_preservation: share of the k true neighbours also among 2D neighbours
    distance_spearman: global rank agreement of all pairwise distances"""
    hi = knn_sets(X_high, k, "cosine")
    lo = knn_sets(X_2d, k, "euclidean")
    overlap = np.mean([len(set(a) & set(b)) / k for a, b in zip(hi, lo)])
    rho = spearmanr(pdist(X_high, "cosine"), pdist(X_2d)).statistic
    return {
        "trustworthiness": float(trustworthiness(X_high, X_2d, n_neighbors=k, metric="cosine")),
        "continuity": float(trustworthiness(X_2d, X_high, n_neighbors=k)),
        "knn_preservation": float(overlap),
        "distance_spearman": float(rho),
    }


def project(X, method="umap", seed=0):
    """Returns (2D coords, fitted reducer with .transform for steered points)."""
    if method == "pca":
        from sklearn.decomposition import PCA
        r = PCA(2, random_state=seed)
    elif method == "random":
        from sklearn.random_projection import GaussianRandomProjection
        r = GaussianRandomProjection(2, random_state=seed)
    else:
        import umap
        r = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed)
    return r.fit_transform(X), r


def out_of_sample_fidelity(reducer, X_ref, Y_ref, X_new, k=10):
    """For points placed after fitting (steered points): kNN overlap between their true
    neighbours in X_ref and their 2D neighbours in Y_ref."""
    Y_new = reducer.transform(X_new)
    hi = NearestNeighbors(n_neighbors=k, metric="cosine").fit(X_ref).kneighbors(X_new, return_distance=False)
    lo = NearestNeighbors(n_neighbors=k).fit(Y_ref).kneighbors(Y_new, return_distance=False)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(hi, lo)])), Y_new
