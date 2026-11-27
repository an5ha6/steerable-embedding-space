"""Public datasets -> binary concept splits. Fixed seed, so every run sees identical data."""
import numpy as np

SEED = 0
PER_CLASS = 1500  # sentences per side of each concept
SPLITS = {"fit": 0.4, "cal": 0.2, "probe": 0.2, "test": 0.2}
AG_LABELS = {0: "world", 1: "sports", 2: "business", 3: "scitech"}


def _split(texts, rng):
    idx = rng.permutation(len(texts))
    out, start = {}, 0
    for name, frac in SPLITS.items():
        n = int(round(frac * len(texts)))
        out[name] = [texts[i] for i in idx[start:start + n]]
        start += n
    return out


def load_concepts():
    """{concept: {"pos": {split: [text]}, "neg": {split: [text]}}}.

    fit   -> learn steering directions (+ null-space probes)
    cal   -> calibrate the steering contract
    probe -> train independent *evaluation* probes (never seen by the method)
    test  -> report numbers
    """
    from datasets import load_dataset
    rng = np.random.default_rng(SEED)
    concepts = {}

    sst = load_dataset("stanfordnlp/sst2", split="train")
    # SST-2 train holds phrases; keep full-ish sentences only.
    rows = [(r["sentence"].strip(), r["label"]) for r in sst if len(r["sentence"].split()) >= 8]
    pos = [t for t, l in rows if l == 1]
    neg = [t for t, l in rows if l == 0]
    pos = [pos[i] for i in rng.choice(len(pos), PER_CLASS, replace=False)]
    neg = [neg[i] for i in rng.choice(len(neg), PER_CLASS, replace=False)]
    concepts["sentiment"] = {"pos": _split(pos, rng), "neg": _split(neg, rng)}

    ag = load_dataset("fancyzhx/ag_news", split="train")
    by = {k: [] for k in AG_LABELS}
    for r in ag.shuffle(seed=SEED).select(range(40000)):
        by[r["label"]].append(r["text"].strip())
    for lab, name in AG_LABELS.items():
        if name == "world":
            continue  # world is the shared negative pool; 3 topic concepts keep the demo readable
        p = by[lab][:PER_CLASS]
        others = [t for l, ts in by.items() if l != lab for t in ts[PER_CLASS:PER_CLASS * 2]]
        n = [others[i] for i in rng.choice(len(others), PER_CLASS, replace=False)]
        concepts[name] = {"pos": _split(p, rng), "neg": _split(n, rng)}
    return concepts


def load_stsb():
    """STS-B test pairs, gold similarity in [0, 1]."""
    from datasets import load_dataset
    d = load_dataset("sentence-transformers/stsb", split="test")
    return list(d["sentence1"]), list(d["sentence2"]), np.array(d["score"], dtype=np.float32)
