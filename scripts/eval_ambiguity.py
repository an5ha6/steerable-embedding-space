"""Bonus (problem 6) evaluation. Run: python scripts/eval_ambiguity.py

Definitions (operational, stated up front):
  * Target words: nouns that have >= 2 coarse senses (WordNet supersenses), each covering >= 20% of the
    word's sense-tagged SemCor occurrences ("balanced" words, so neither sense is a safe default).
  * NOT ambiguous: a real SemCor sentence using the word; human annotators assigned one sense, so the
    context disambiguates. Correct resolution = predicted coarse sense equals the annotated one.
  * AMBIGUOUS: the same word in a context-free request ("Tell me about the bank."), where both senses
    are plausible by construction. Correct handling = flag it and ask.
Words are split dev/test *by word*, so the threshold never sees a test word.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from steer.ambiguity import Disambiguator, coarse_senses  # noqa: E402

TEMPLATES = ["{w}", "the {w}", "Tell me about the {w}.", "Where is the {w}?", "I need information on the {w}.",
             "What do you think of the {w}?"]
MAX_CTX_PER_WORD = 8


def semcor_items(n_files=120):
    cache = ROOT / ".cache" / f"semcor_{n_files}.json"
    if cache.exists():
        return [tuple(x) for x in json.loads(cache.read_text())]
    items = _semcor_items(n_files)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(items))
    return items


def _semcor_items(n_files):
    import nltk
    try:
        nltk.data.find("corpora/semcor")
    except LookupError:
        nltk.download("semcor", quiet=True)
    from nltk.corpus import semcor
    from nltk.tree import Tree

    from steer.ambiguity import _wn
    wn = _wn()  # ensures WordNet is downloaded; without it every label lookup fails silently
    items = []
    for sent in semcor.tagged_sents(fileids=semcor.fileids()[:n_files], tag="sem"):
        toks, tg = [], []
        for ch in sent:
            if isinstance(ch, Tree):
                w = ch.leaves()
                lab = ch.label()
                try:  # label is a Lemma object or a synset name like 'group.n.01', depending on nltk version
                    syn = lab.synset() if hasattr(lab, "synset") else wn.synset(lab)
                except Exception:
                    syn = None
                if syn is not None and syn.pos() == "n" and len(w) == 1 and w[0].isalpha():
                    tg.append((w[0].lower(), syn.lexname()))
                toks.extend(w)
            else:
                toks.extend(ch)
        text = " ".join(toks)
        if 6 <= len(toks) <= 60:
            items += [(text, w, lex) for w, lex in tg]
    return items


def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


SUPERSENSE_WORD = {"Tops": "general concept", "act": "action", "artifact": "man-made object",
                   "cognition": "idea", "communication": "message", "possession": "money", "body": "body part",
                   "group": "group or organization", "location": "place", "person": "person", "state": "condition",
                   "attribute": "quality", "event": "event", "time": "time period", "quantity": "amount",
                   "object": "natural object", "substance": "material", "feeling": "emotion", "food": "food",
                   "plant": "plant", "animal": "animal", "process": "process", "phenomenon": "phenomenon",
                   "relation": "relation", "shape": "shape", "motive": "motive"}


NEUTRAL = ["one I mentioned", "usual one", "thing we discussed", "one from before"]


def build_data():
    """Returns items (text, word, gold_lex or None, kind) and the dev/test word split.
    kind: 'context' (real sentence, clear), 'short_clear' (short request + sense hint, clear),
          'ambiguous' (short request, no hint). short_clear controls for 'the detector just flags short text'."""
    items = semcor_items()
    freq = defaultdict(Counter)
    for _, w, lex in items:
        freq[w][lex] += 1
    words = []
    for w, c in freq.items():
        tot = sum(c.values())
        big = [l for l, n in c.items() if n / tot >= 0.2]
        if tot >= 8 and len(big) >= 2 and set(big) <= set(coarse_senses(w)):
            words.append(w)
    words = sorted(words)
    rng = np.random.default_rng(0)
    rng.shuffle(words)
    dev_w, test_w = set(words[: len(words) // 2]), set(words[len(words) // 2:])

    data = []
    per = Counter()
    for text, w, lex in items:
        if w in dev_w | test_w and per[w] < MAX_CTX_PER_WORD:
            per[w] += 1
            data.append((text, w, lex, "context"))
    for w in words:
        data += [(t.format(w=w), w, None, "ambiguous") for t in TEMPLATES]
        for j, lex in enumerate(sorted(l for l, n in freq[w].items() if n / sum(freq[w].values()) >= 0.2)):
            hint = SUPERSENSE_WORD.get(lex.split(".")[1], lex.split(".")[1])
            stem = TEMPLATES[2 + j % 4].format(w=w).rstrip(".?")
            data.append((stem + f" (the {hint}).", w, lex, "short_clear"))
            # length/format-matched twin with a neutral, non-disambiguating hint
            data.append((stem + f" (the {NEUTRAL[j % len(NEUTRAL)]}).", w, None, "ambiguous_matched"))
    return data, dev_w, test_w


def main():
    data, dev_w, test_w = build_data()
    dis = Disambiguator()
    res = [dis.analyze(t, w) for t, w, _, _ in data]
    split = np.array(["dev" if w in dev_w else "test" for _, w, _, _ in data])
    kind = np.array([k for *_, k in data])
    amb = np.isin(kind, ["ambiguous", "ambiguous_matched"])
    margin = np.array([r["margin"] for r in res], dtype=float)  # context evidence; low = ambiguous
    dev_f1 = dis.calibrate(margin[split == "dev"], amb[split == "dev"])
    rng = np.random.default_rng(0)

    T = np.asarray(split == "test", dtype=bool)
    flag = np.asarray(margin <= dis.threshold, dtype=bool)
    short = T & np.isin(kind, ["short_clear", "ambiguous_matched"])  # identical format, differ only in the hint
    n_tok = np.array([len(t.split()) for t, *_ in data], dtype=float)
    tp, fp = (flag & amb & T).sum(), (flag & ~amb & T).sum()
    fn, tn = (~flag & amb & T).sum(), (~flag & ~amb & T).sum()
    # resolution on non-ambiguous test items
    na = [i for i in np.where(T & ~amb)[0]]
    correct = np.array([res[i]["best"] == data[i][2] for i in na])
    wn_first = []
    from nltk.corpus import wordnet as wn
    for i in na:
        w, gold = data[i][1], data[i][2]
        first = next(s.lexname() for s in wn.synsets(w, pos="n") if s.lexname() in coarse_senses(w))
        wn_first.append(first == gold)
    rand = np.mean([1 / len(res[i]["senses"]) for i in na])

    # end-to-end "correct handling": ambiguous -> ask; not ambiguous -> don't ask AND resolve right
    ours = np.r_[flag[T & amb], [(not flag[i]) and c for i, c in zip(na, correct)]]
    always_guess = np.r_[np.zeros((T & amb).sum()), wn_first]  # first-sense, never asks
    always_ask = np.r_[np.ones((T & amb).sum()), np.zeros(len(na))]

    out = {
        "n_words": {"dev": len(dev_w), "test": len(test_w)},
        "n_test_items": {"ambiguous": int((T & amb).sum()), "clear_in_context": int((T & (kind == "context")).sum()),
                         "clear_short_with_hint": int((T & (kind == "short_clear")).sum())},
        "threshold_margin": dis.threshold, "dev_f1": dev_f1,
        "detection_test": {"auroc": auroc(-margin[T], amb[T]),
                           "auroc_length_matched": auroc(-margin[short], amb[short]),
                           "baseline_shortness_auroc": auroc(-n_tok[T], amb[T]),
                           "baseline_shortness_auroc_length_matched": auroc(-n_tok[short], amb[short]),
                           "precision": float(tp / max(tp + fp, 1)),
                           "recall": float(tp / max(tp + fn, 1)), "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
                           "false_positive_rate_in_context": float(np.mean(flag[T & (kind == "context")])),
                           "false_positive_rate_short_with_hint": float(np.mean(flag[T & (kind == "short_clear")]))},
        "resolution_test_on_clear_text": {"ours_accuracy": float(correct.mean()),
                                          "wordnet_first_sense": float(np.mean(wn_first)),
                                          "random": float(rand)},
        "correct_handling_rate": {"ours": float(ours.mean()), "always_guess_first_sense": float(always_guess.mean()),
                                  "always_ask": float(always_ask.mean())},
        "examples": [{"text": data[i][0][:160], "word": data[i][1], "gold": data[i][2], "ambiguous": bool(amb[i]),
                      "flagged": bool(flag[i]), "pred": res[i]["best"], "margin": round(float(margin[i]), 3)}
                     for i in rng.choice(np.where(T)[0], 12, replace=False)],
        "test_words": sorted(test_w),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "ambiguity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("examples", "test_words")}, indent=1))


if __name__ == "__main__":
    main()
