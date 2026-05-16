"""Embedding generation using sentence-transformers."""

import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def get_model(name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load (and cache) the sentence-transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(name)
    return _model


def embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Return (N, D) float32 embedding matrix for a list of texts."""
    return get_model(model_name).encode(
        texts, convert_to_numpy=True, show_progress_bar=False
    )
