"""Fast, deterministic tests using synthetic embeddings only."""
import numpy as np
import pytest

from steer.engine import Steerer, apply, erase, erase_basis, fit_probe, lowdin, prob, unit
from steer.fidelity import report


def _synthetic(seed=17, n=240, dim=384):
    """Four linear concepts; A and B share two signal dimensions."""
    rng = np.random.default_rng(seed)
    supports = ((0, 1, 2, 3), (2, 3, 4, 5), (8, 9, 10, 11), (16, 17, 18, 19))
    data = {}
    for ci, name in enumerate("ABCD"):
        sides = {"pos": {}, "neg": {}}
        for side, label in (("pos", 1.0), ("neg", -1.0)):
            X = rng.normal(0, 0.65, (n, dim))
            # Each concept has a strong, independently planted signal. Shared A/B
            # coordinates create deliberate overlap for the selectivity test.
            X[:, supports[ci]] += label * 1.35
            sides[side]["fit"] = unit(X[: n * 2 // 3]).astype(np.float32)
            sides[side]["test"] = unit(X[n * 2 // 3 :]).astype(np.float32)
        data[name] = sides
    return data


def test_apply_normalizes_and_zero_alpha_preserves_input():
    rng = np.random.default_rng(1)
    E = unit(rng.normal(size=(12, 384)))
    d = unit(rng.normal(size=384))
    assert np.allclose(np.linalg.norm(apply(E, d, 0.4), axis=1), 1.0, atol=1e-6)
    assert np.allclose(apply(E, d, 0), E, atol=1e-6)


def test_lowdin_rows_are_orthonormal():
    D = np.random.default_rng(2).normal(size=(4, 384))
    Q = lowdin(D)
    assert np.allclose(Q @ Q.T, np.eye(4), atol=1e-10)


def test_nullspace_reduces_leakage_and_target_probability_rises():
    emb = _synthetic()
    st = Steerer(emb, seed=3)
    # Probe B/C/D on held-out negatives, steering A across alpha values.
    src = np.vstack([emb["A"]["neg"]["test"]])
    others = [fit_probe(emb[c]["pos"]["fit"], emb[c]["neg"]["fit"], seed=4)
              for c in "BCD"]
    leak = {}
    for method in ("naive", "nullspace"):
        steered = apply(src, st.d("A", method), 0.8)
        leak[method] = np.mean([np.abs(prob(p, steered) - prob(p, src)).mean() for p in others])
    assert leak["nullspace"] < leak["naive"]

    p = st.verifier["A"]
    means = [prob(p, apply(src, st.d("A", "nullspace"), a)).mean() for a in (0, .25, .5, .75)]
    assert np.all(np.diff(means) > 0)


def test_erasure_hurts_fresh_probe_but_random_projection_does_not():
    emb = _synthetic(seed=8, n=900)
    pos_fit, neg_fit = emb["A"]["pos"]["fit"], emb["A"]["neg"]["fit"]
    pos_test, neg_test = emb["A"]["pos"]["test"], emb["A"]["neg"]["test"]
    Q = erase_basis(pos_fit, neg_fit, iters=4, seed=9)
    # Train fresh probes after transformation on held-out data, then evaluate fresh
    # held-out examples to avoid reusing the erasure learner's probe.
    rng = np.random.default_rng(10)
    pos_train, neg_train = pos_test[:50], neg_test[:50]
    pos_eval, neg_eval = pos_test[50:], neg_test[50:]
    erased_train = (erase(pos_train, Q), erase(neg_train, Q))
    erased_eval = (erase(pos_eval, Q), erase(neg_eval, Q))
    p_erased = fit_probe(*erased_train, seed=11)
    acc_erased = np.mean(np.r_[prob(p_erased, erased_eval[0]) > .5,
                              prob(p_erased, erased_eval[1]) <= .5])
    random_Q = np.linalg.qr(rng.normal(size=(384, 4)))[0]
    p_random = fit_probe(erase(pos_train, random_Q), erase(neg_train, random_Q), seed=11)
    acc_random = np.mean(np.r_[prob(p_random, erase(pos_eval, random_Q)) > .5,
                              prob(p_random, erase(neg_eval, random_Q)) <= .5])
    assert acc_erased < 0.65
    assert acc_random > 0.8


def test_fidelity_identity_like_projection():
    rng = np.random.default_rng(12)
    latent = rng.normal(size=(120, 2))
    X = np.zeros((120, 384))
    X[:, :2] = latent
    X += rng.normal(0, 0.005, X.shape)
    Y = X[:, :2].copy()
    scores = report(X, Y, k=8)
    assert scores["trustworthiness"] > 0.95
