"""Frozen sentence encoder with an on-disk embedding cache."""
import hashlib
import os
from pathlib import Path

import numpy as np

# Override to test that the method is encoder-agnostic, e.g. STEER_MODEL=sentence-transformers/all-MiniLM-L6-v2
MODEL_NAME = os.environ.get("STEER_MODEL", "BAAI/bge-small-en-v1.5")
CACHE = Path(__file__).resolve().parent.parent / ".cache" / "emb"
_model = None


def model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed(texts, use_cache=True):
    """Unit-normalized embeddings, shape (n, 384)."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, model().get_sentence_embedding_dimension()), dtype=np.float32)
    key = hashlib.sha1((MODEL_NAME + "\x00" + "\x01".join(texts)).encode()).hexdigest()
    path = CACHE / f"{key}.npy"
    if use_cache and path.exists():
        return np.load(path)
    E = model().encode(texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=len(texts) > 500).astype(np.float32)
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, E)
    return E
