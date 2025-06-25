from __future__ import annotations

"""Shared TF-IDF vectorisation helper (Phase-16 FEATURE-02/03)

This module centralises TF-IDF embedding so that capacity and pre-processing
can be tuned globally via `optimization_config.json`.
"""

from typing import List
import os
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from core.text_cleaning import clean_text, STOP_WORDS


__all__ = ["tfidf_embeddings", "contextual_embeddings"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tfidf_embeddings(
    texts: List[str], *, max_features: int = 500, clean: bool = False
) -> np.ndarray:
    """Compute TF-IDF embeddings for a list of texts.

    Args:
        texts: Sequence of raw strings.
        max_features: Vocabulary size upper bound.
        clean: Whether to apply HTML / URL stripping and stop-word removal prior
            to vectorisation.

    Returns:
        2-D `np.ndarray` of shape (len(texts), v).
    """
    if len(texts) == 0:
        return np.empty((0, 0))
    if len(texts) < 2:
        # Avoid scikit-learn complaining about single sample; return zeros so
        # downstream cosine similarity logic degrades gracefully.
        return np.zeros((len(texts), 1))

    processed = [clean_text(t) if clean else t for t in texts]

    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    return vec.fit_transform(processed).toarray()


# ---------------------------------------------------------------------------
# Contextual Embeddings (FEATURE-04)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _get_sentence_transformer(model_name: str, cache_dir: str, device: str):
    """Load and cache SentenceTransformer model. LRU cache prevents reloading."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence_transformers not installed. Run: pip install sentence-transformers"
        )
    
    # Auto-detect device
    if device == "auto":
        device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") and os.getenv("CUDA_VISIBLE_DEVICES") != "" else "cpu"
    
    # Create cache directory if needed
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load model with fail-fast behavior
    model = SentenceTransformer(model_name, cache_folder=cache_dir, device=device)
    return model


def contextual_embeddings(
    texts: List[str], *, model_name: str = "all-mpnet-base-v2", 
    cache_dir: str = ".hf_cache", device: str = "auto", clean: bool = False
) -> np.ndarray:
    """Compute contextual embeddings for a list of texts using SentenceTransformers.

    Args:
        texts: Sequence of raw strings.
        model_name: HuggingFace model identifier. Options:
            - "all-mpnet-base-v2" (768-dim, general purpose, ~420MB)
            - "allenai/specter2" (768-dim, scientific papers, ~1.3GB) 
            - "all-MiniLM-L6-v2" (384-dim, lightweight, ~110MB)
        cache_dir: Directory to store downloaded models.
        device: "auto", "cpu", or "cuda". Auto-detects based on CUDA_VISIBLE_DEVICES.
        clean: Whether to apply text cleaning before embedding.

    Returns:
        2-D `np.ndarray` of shape (len(texts), embedding_dim).
        
    Raises:
        ImportError: If sentence_transformers not installed.
        Exception: If model loading fails (fail-fast behavior).
    """
    if len(texts) == 0:
        return np.empty((0, 0))
    
    processed = [clean_text(t) if clean else t for t in texts]
    
    # Handle empty or very short texts
    processed = [t if t.strip() else "[EMPTY]" for t in processed]
    
    model = _get_sentence_transformer(model_name, cache_dir, device)
    embeddings = model.encode(processed, convert_to_numpy=True)
    
    return embeddings