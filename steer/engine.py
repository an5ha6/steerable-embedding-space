"""Steerable embedding adapter over a frozen encoder.

e' = normalize(e + sum_c alpha_c * d_c)

Direction methods (all unit-norm, so alpha means the same thing for each):
  random    : random direction (control)
  naive     : difference of class means
  lowdin    : naive directions, symmetrically orthogonalized to each other
  nullspace : ours - difference of means with every *other* concept's linear readout projected out,
              so steering concept A leaves linear readouts of B, C, D unchanged.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

METHODS = ["random", "naive", "lowdin", "nullspace"]


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def apply(E, d, alpha):
    return unit(E + alpha * d)


def fit_probe(pos, neg, seed=0, n=300):
    """Every probe sees exactly n examples per side, so verifier, calibration reader and evaluation probe
    share one probability scale (probe confidence grows with training size)."""
    rng = np.random.default_rng(seed)
    pos = pos[rng.choice(len(pos), min(n, len(pos)), replace=False)]
    neg = neg[rng.choice(len(neg), min(n, len(neg)), replace=False)]
    X = np.vstack([pos, neg])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return LogisticRegression(C=1.0, max_iter=3000, random_state=seed).fit(X, y)


def prob(probe, E):
    return probe.predict_proba(E)[:, 1]


def lowdin(D):
    """Symmetric orthogonalization: closest orthonormal set to rows of D (order-independent)."""
    U, _, Vt = np.linalg.svd(D, full_matrices=False)
    return U @ Vt


class Steerer:
    def __init__(self, emb, seed=0):
        """emb: {concept: {"pos": {split: E}, "neg": {split: E}}} with unit embeddings."""
        self.concepts = list(emb)
        rng = np.random.default_rng(seed)
        raw = np.stack([emb[c]["pos"]["fit"].mean(0) - emb[c]["neg"]["fit"].mean(0) for c in self.concepts])
        # verifier probes: trained on the fit split, used by the method itself (closed loop)
        self.verifier = {c: fit_probe(emb[c]["pos"]["fit"], emb[c]["neg"]["fit"]) for c in self.concepts}
        dim = raw.shape[1]
        self.dirs = {
            "random": unit(rng.standard_normal((len(self.concepts), dim))),
            "naive": unit(raw),
            "lowdin": lowdin(unit(raw)),
        }
        ns = []
        for i, c in enumerate(self.concepts):
            others = [unit(self.verifier[o].coef_[0]) for o in self.concepts if o != c]
            others += [unit(raw[j]) for j in range(len(self.concepts)) if j != i]
            Q, _ = np.linalg.qr(np.stack(others).T)  # orthonormal basis of the readouts to protect
            ns.append(raw[i] - Q @ (Q.T @ raw[i]))
        self.dirs["nullspace"] = unit(np.stack(ns))
        self.add_custom = {}

    def d(self, concept, method="nullspace"):
        if concept in self.add_custom:
            return self.add_custom[concept]
        return self.dirs[method][self.concepts.index(concept)]

    def add_concept(self, name, pos_E, neg_E):
        """User-defined concept from example sentences, protected against the built-in concepts."""
        raw = pos_E.mean(0) - neg_E.mean(0)
        others = [unit(p.coef_[0]) for p in self.verifier.values()]
        Q, _ = np.linalg.qr(np.stack(others).T)
        self.add_custom[name] = unit(raw - Q @ (Q.T @ raw))
        self.verifier[name] = fit_probe(pos_E, neg_E)
        return self.add_custom[name]

    def steer(self, E, alphas, method="nullspace"):
        """alphas: {concept: alpha}. Multiple concepts compose additively."""
        v = sum(a * self.d(c, method) for c, a in alphas.items() if a)
        return E.copy() if isinstance(v, int) else apply(E, v, 1.0)


def sts_quality(E1, E2, gold, d=None, alpha=0.0):
    """Spearman(cos(steered s1, s2), gold). Asymmetric: only the 'query' side is steered."""
    A = E1 if d is None or alpha == 0 else apply(E1, d, alpha)
    return spearmanr((A * E2).sum(1), gold).statistic


class Contract:
    """Calibrated, budgeted steering.

    Request: concept, wanted probability gain `target`, max STS-B quality drop, max leakage.
    The contract (a) picks alpha_max from calibration curves so the budgets hold on average,
    (b) per input, searches the smallest alpha <= alpha_max whose verifier-probe gain reaches `target`
    (closed loop), (c) refuses when no alpha inside the budget can reach the target.
    """
    GRID = np.round(np.linspace(0, 1.0, 21), 3)

    def __init__(self, steerer, cal_emb, sts_cal, method="nullspace", coverage=0.9):
        """Curves and the safety margin come from *calibration readers*: probes trained on the cal split,
        independent of the verifier. The verifier is overconfident on its own training distribution, so
        the per-input target is raised by a conformal margin: the `coverage` quantile of
        (verifier gain - independent gain) on calibration data."""
        self.s, self.method = steerer, method
        E1, E2, gold = sts_cal
        half = {c: len(cal_emb[c]["neg"]["cal"]) // 2 for c in steerer.concepts}
        # calibration readers use the first half of the cal split; curves and margins use the second half
        pool = {c: (np.vstack([cal_emb[c]["pos"]["cal"], cal_emb[c]["pos"]["fit"][::2]]),
                    np.vstack([cal_emb[c]["neg"]["cal"][: half[c]], cal_emb[c]["neg"]["fit"][::2]]))
                for c in steerer.concepts}
        # 5 readers on different subsamples: the margin must cover disagreement between readers too
        readers = {c: [fit_probe(*pool[c], seed=s) for s in range(2, 7)] for c in steerer.concepts}
        self.reader = {c: r[0] for c, r in readers.items()}
        self.curves, self.margin = {}, {}
        q0 = sts_quality(E1, E2, gold)
        for c in steerer.concepts:
            d = steerer.d(c, method)
            src = cal_emb[c]["neg"]["cal"][half[c]:]
            base = {o: prob(self.reader[o], src) for o in steerer.concepts}
            vbase = prob(steerer.verifier[c], src)
            rbase = [prob(r, src) for r in readers[c]]
            rows, resid = [], []
            for a in self.GRID:
                S = apply(src, d, a)
                g = prob(self.reader[c], S) - base[c]
                if a > 0:
                    vg = prob(steerer.verifier[c], S) - vbase
                    resid += [vg - (prob(r, S) - b) for r, b in zip(readers[c], rbase)]
                leak = float(np.mean([np.abs(prob(self.reader[o], S) - base[o]).mean()
                                      for o in steerer.concepts if o != c]))
                qdrop = float(q0 - sts_quality(E1, E2, gold, d, a))
                rows.append((a, float(g.mean()), leak, qdrop))
            self.curves[c] = np.array(rows)
            self.margin[c] = float(max(0.0, np.quantile(np.concatenate(resid), coverage)))

    def alpha_max(self, concept, max_qdrop, max_leak):
        cur = self.curves[concept]
        ok = (cur[:, 3] <= max_qdrop) & (cur[:, 2] <= max_leak)
        # budgets must hold for every alpha up to the chosen one
        bad = np.where(~ok)[0]
        last = (bad[0] - 1) if len(bad) else len(cur) - 1
        return float(cur[max(last, 0), 0]), cur[max(last, 0)]

    def run(self, concept, E, target=0.4, max_qdrop=0.02, max_leak=0.05, iters=12):
        """Returns per-input alphas, status ('ok' | 'refused'), and the calibrated prediction."""
        amax, row = self.alpha_max(concept, max_qdrop, max_leak)
        d, v = self.s.d(concept, self.method), self.s.verifier[concept]
        if row[1] < target * 0.5:  # even alpha_max gains < half the target on average: refuse up front
            return {"status": "refused", "alpha_max": amax, "expected_gain_at_max": float(row[1]),
                    "reason": f"budget allows alpha<={amax:.2f}, which gains only {row[1]:.2f} on average"}
        p0 = prob(v, E)
        need = target + self.margin.get(concept, 0.0)
        lo, hi = np.zeros(len(E)), np.full(len(E), amax)
        for _ in range(iters):  # per-input binary search, closed loop on the verifier probe
            mid = (lo + hi) / 2
            reached = prob(v, unit(E + mid[:, None] * d)) - p0 >= need
            hi = np.where(reached, mid, hi)
            lo = np.where(reached, lo, mid)
        alphas = hi
        S = unit(E + alphas[:, None] * d)
        gain = prob(v, S) - p0
        return {"status": "ok", "alpha_max": amax, "alphas": alphas, "steered": S,
                "verified": gain >= need - 1e-6, "verifier_gain": gain, "margin": self.margin.get(concept, 0.0)}


def erase_basis(pos, neg, iters=8, seed=0):
    """Iterative null-space projection: repeatedly fit a linear probe and project its direction out,
    until no linear reader can recover the concept. Returns an orthonormal basis Q (dim, iters)."""
    Q = np.zeros((pos.shape[1], 0))
    for i in range(iters):
        P, N = pos - (pos @ Q) @ Q.T, neg - (neg @ Q) @ Q.T
        w = fit_probe(P, N, seed=seed + i, n=len(pos)).coef_[0]
        w = w - Q @ (Q.T @ w)
        Q = np.column_stack([Q, unit(w)])
    return Q


def erase(E, Q):
    return unit(E - (E @ Q) @ Q.T)


def monotonicity(probe, E, d, grid):
    """Mean per-sentence Spearman(alpha, probe prob)."""
    P = np.stack([prob(probe, apply(E, d, a)) for a in grid], 1)
    r = [spearmanr(grid, p).statistic for p in P]
    return float(np.nanmean(r))
