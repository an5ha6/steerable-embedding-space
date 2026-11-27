"""Expensive demo state, built once and stored in .cache/demo so the app starts in seconds.
Build ahead of time with: python scripts/build_demo.py"""
import pickle
from pathlib import Path

import numpy as np

from . import data, fidelity, papers
from .encoder import embed
from .engine import Contract, Steerer

DEMO_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "demo"


def _on_disk(name, build):
    # Only loads files this project wrote itself; pickle is not safe for untrusted files.
    path = DEMO_CACHE / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    out = build()
    DEMO_CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def _build_core():
    concepts = data.load_concepts()
    emb = {c: {s: {sp: embed(t) for sp, t in v[s].items()} for s in ("pos", "neg")} for c, v in concepts.items()}
    stx = Steerer(emb)
    s1, s2, gold = data.load_stsb()
    half = np.random.default_rng(0).permutation(len(gold))[: len(gold) // 2]
    contract = Contract(stx, emb, (embed(s1)[half], embed(s2)[half], gold[half]))
    texts, labels, E = [], [], []
    for c, v in concepts.items():
        for s in ("pos", "neg"):
            texts += v[s]["test"]
            labels += [f"{c}{'+' if s == 'pos' else '-'}"] * len(v[s]["test"])
            E.append(emb[c][s]["test"])
    E = np.vstack(E)
    Y, red = fidelity.project(E, "umap")
    red.transform(E[:2])  # compile numba paths now, not during the demo
    fid = fidelity.report(E[::3], Y[::3])
    return stx, contract, texts, np.array(labels), E, Y, red, fid


def _build_papers():
    P = papers.load_papers()
    texts = [papers.text(p) for p in P]
    E = embed(texts)
    lab, names, start = papers.scope_map(E, texts, len(papers.CATS))
    Y, _ = fidelity.project(E, "umap")
    return P, E, lab, names, start, Y


def core():
    return _on_disk("core", _build_core)


def paper_space():
    return _on_disk("papers", _build_papers)
