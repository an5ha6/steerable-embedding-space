"""Bonus for problem 6: ambiguity detection and resolution in the same embedding space.

Idea: a word's coarse senses (WordNet supersenses, e.g. noun.group vs noun.object for "bank") are
prototype points in embedding space. A text is ambiguous for that word when its embedding sits
between two sense prototypes (small margin). Resolution = nearest sense, or, when ambiguous,
a clarification question; the chosen sense becomes a steering direction (problem 5 machinery).
"""
import re

import numpy as np

from .encoder import embed


def _wn():
    import nltk
    for pkg in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import wordnet
    return wordnet


def coarse_senses(word, max_senses=6):
    """{supersense: prototype text} for the noun `word`, grouping fine synsets by lexname."""
    wn = _wn()
    groups = {}
    for syn in wn.synsets(word, pos="n")[:max_senses * 3]:
        if word.lower() not in [l.name().lower() for l in syn.lemmas()]:
            continue
        g = groups.setdefault(syn.lexname(), [])
        if len(g) < 2:
            g.append(f"{syn.definition()}. " + " ".join(syn.examples()[:1]))
    return {lex: f"{word}: " + " ".join(defs) for lex, defs in list(groups.items())[:max_senses]}


class Disambiguator:
    def __init__(self, tau=0.02):
        self.tau = tau  # softmax temperature over cosine scores
        self.threshold = 0.5  # margin threshold, set by calibrate()
        self._cache = {}

    def senses(self, word):
        if word not in self._cache:
            s = coarse_senses(word)
            self._cache[word] = (list(s), list(s.values()), embed(list(s.values())) if s else None)
        return self._cache[word]

    def analyze(self, text, word):
        """Ambiguity evidence = how far the context moves the text's sense profile away from the bare word's.
        Context that adds no sense information -> small shift -> ambiguous. (Absolute margins between
        senses failed: they mostly reflect the encoder's prior for the word, AUROC 0.56 on dev.)"""
        labels, glosses, P = self.senses(word)
        if P is None or len(labels) < 2:
            return {"word": word, "ambiguous": False, "margin": 1.0, "senses": labels, "probs": [1.0] * len(labels)}
        e, w = embed([window(text, word), word])
        st = P @ e
        shift = st - P @ w
        evidence = float(np.linalg.norm(shift - shift.mean()))
        z = st / self.tau
        p = np.exp(z - z.max())
        p /= p.sum()
        return {"word": word, "senses": labels, "glosses": glosses, "probs": p.tolist(), "margin": evidence,
                "ambiguous": evidence <= self.threshold, "best": labels[int(np.argmax(st))]}

    def calibrate(self, margins, is_amb):
        """Pick the margin threshold maximizing F1 on a dev split."""
        margins, is_amb = np.asarray(margins), np.asarray(is_amb, bool)
        best = (0, self.threshold)
        for t in np.unique(margins):
            pred = margins <= t
            tp, fp, fn = (pred & is_amb).sum(), (pred & ~is_amb).sum(), (~pred & is_amb).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            best = max(best, (f1, float(t)))
        self.threshold = best[1] + 1e-9
        return best[0]

    def clarify(self, res):
        """Clarifying question listing the two most likely senses."""
        order = np.argsort(res["probs"])[::-1][:2]
        opts = [res["glosses"][i].split(": ", 1)[1].split(".")[0] for i in order]
        return f'By "{res["word"]}", do you mean (a) {opts[0]} or (b) {opts[1]}?'

    def resolve_direction(self, word, sense_label):
        """Steering direction toward one sense and away from the others (for problem-5 retrieval)."""
        labels, _, P = self.senses(word)
        i = labels.index(sense_label)
        d = P[i] - np.delete(P, i, 0).mean(0)
        return d / np.linalg.norm(d)


def window(text, word, size=12):
    """Context window around the first occurrence of `word` (whole text if absent)."""
    toks = text.split()
    for i, t in enumerate(toks):
        if re.sub(r"\W", "", t.lower()) == word.lower():
            return " ".join(toks[max(0, i - size): i + size + 1])
    return text
