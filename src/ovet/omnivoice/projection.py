"""Language projection removal.

Given an emotion direction vector ``v_emo`` and a language direction
vector ``v_lang`` (both typically per-layer), strip the component of
``v_emo`` that is parallel to ``v_lang``. The result is a v_emo_clean
that is orthogonal to v_lang.

Used in Phase 4 to keep cross-lingual emotion injection from carrying
language-specific bias.
"""
from __future__ import annotations
import numpy as np


def remove_projection(
    v_emo: np.ndarray,
    v_lang: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Return v_emo with the v_lang-aligned component removed.

    The result satisfies ``np.dot(out, v_lang) ≈ 0`` modulo eps.
    """
    coef = float(np.sum(v_emo * v_lang) / (np.sum(v_lang * v_lang) + eps))
    return v_emo - coef * v_lang


def remove_projection_per_layer(
    emotion_vectors: dict[int, np.ndarray],
    language_vectors: dict[int, np.ndarray],
    eps: float = 1e-8,
) -> dict[int, np.ndarray]:
    """Apply projection removal layer-wise.

    Layers absent from ``language_vectors`` are passed through unchanged.
    """
    out: dict[int, np.ndarray] = {}
    for layer_id, v_emo in emotion_vectors.items():
        v_lang = language_vectors.get(layer_id)
        out[layer_id] = remove_projection(v_emo, v_lang, eps) if v_lang is not None else v_emo
    return out
