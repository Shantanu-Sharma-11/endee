"""Embedding utils using sentence-transformers (all-MiniLM-L6-v2)."""

import logging
import os
from typing import List, Optional

from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)

_model: Optional[SentenceTransformer] = None
_dimension: Optional[int] = None


def _load_model():
    global _model, _dimension
    if _model is None:
        log.info("Loading model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        _dimension = _model.get_sentence_embedding_dimension()
        log.info("Model ready (dim=%d)", _dimension)
    return _model


def get_embedding(text):
    """Embed a single string."""
    model = _load_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def get_embeddings(texts):
    """Embed a list of strings."""
    if not texts:
        return []
    model = _load_model()
    vecs = model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=True, batch_size=64)
    return vecs.tolist()


def get_dimension():
    """Get embedding dimension (loads model if needed)."""
    if _dimension is None:
        _load_model()
    return _dimension


def get_model_name():
    return MODEL_NAME
