"""Geometric validation primitives for Phase 4a.

Given a population of per-layer hidden vectors with (speaker, emotion,
language) labels, we ask:

1. Is language linearly separable from these hiddens?
   (probably yes — the question is *how* separable.)
2. Is emotion linearly separable?
3. After projecting out the *language direction*, does language-prob drop
   significantly while emotion-prob is mostly preserved?

If yes → Phase 4b (language projection removal) is justified.
If no → keep Phase 3 alone.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np


# ---------------------------------------------------------------------
# Linear probe (logistic regression with leave-one-out CV)
# ---------------------------------------------------------------------

def linear_probe(
    X: np.ndarray,
    y: Sequence,
    *,
    cv: str = "loo",
    seed: int = 0,
    C: float = 1.0,
) -> float:
    """Return cross-validated accuracy of a logistic regression on (X, y).

    ``cv='loo'`` does leave-one-out (good for tiny datasets).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_score

    if cv != "loo":
        raise NotImplementedError(cv)
    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    scores = cross_val_score(clf, X, list(y), cv=LeaveOneOut())
    return float(np.mean(scores))


# ---------------------------------------------------------------------
# v_lang construction from a labeled hidden set
# ---------------------------------------------------------------------

def v_lang_from_means(
    hiddens: np.ndarray,
    languages: Sequence[str],
    l1_tag: str,
    l2_tag: str,
) -> np.ndarray:
    """Return ``mean(hiddens[l1]) - mean(hiddens[l2])`` (a 1-D vector)."""
    languages = np.asarray(languages)
    h_l1 = hiddens[languages == l1_tag].mean(axis=0)
    h_l2 = hiddens[languages == l2_tag].mean(axis=0)
    return (h_l1 - h_l2).astype(np.float32)


def project_out(
    X: np.ndarray,
    direction: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """For each row x in X, return ``x - ((x·d)/(d·d)) * d``."""
    direction = np.asarray(direction, dtype=np.float32).reshape(-1)
    denom = float((direction * direction).sum()) + eps
    coef = (X @ direction) / denom         # shape: (N,)
    return X - np.outer(coef, direction)


# ---------------------------------------------------------------------
# Main analysis container
# ---------------------------------------------------------------------

@dataclass
class LayerAnalysis:
    layer_id: int
    n_samples: int
    languages: list[str]
    emotions: list[str]
    speakers: list[str]
    # Pre-projection
    lang_acc_pre: float
    emo_acc_pre:  float
    spk_acc_pre:  float
    # Post-projection
    lang_acc_post: float
    emo_acc_post:  float
    spk_acc_post:  float
    # Magnitudes
    v_lang_norm:   float
    cos_lang_emo:  float          # |cos(v_lang, v_emo_mean)|
    cos_lang_spk:  float

    def as_dict(self) -> dict:
        return {
            "layer_id": self.layer_id,
            "n_samples": self.n_samples,
            "lang_acc_pre":  self.lang_acc_pre,
            "lang_acc_post": self.lang_acc_post,
            "emo_acc_pre":   self.emo_acc_pre,
            "emo_acc_post":  self.emo_acc_post,
            "spk_acc_pre":   self.spk_acc_pre,
            "spk_acc_post":  self.spk_acc_post,
            "v_lang_norm":   self.v_lang_norm,
            "cos_lang_emo":  self.cos_lang_emo,
            "cos_lang_spk":  self.cos_lang_spk,
        }


def analyze_layer(
    hiddens: np.ndarray,           # [N, D]
    speakers: Sequence[str],
    emotions: Sequence[str],
    languages: Sequence[str],
    layer_id: int,
    l1_tag: str = "Japanese",
    l2_tag: str = "English",
) -> LayerAnalysis:
    """Compute pre/post linear-probe accuracies for one layer."""
    speakers  = np.asarray(speakers)
    emotions  = np.asarray(emotions)
    languages = np.asarray(languages)

    # Pre-projection accuracies
    lang_pre = linear_probe(hiddens, languages)
    emo_pre  = linear_probe(hiddens, emotions)
    spk_pre  = linear_probe(hiddens, speakers)

    # v_lang from group means
    v_lang = v_lang_from_means(hiddens, languages, l1_tag, l2_tag)
    v_lang_norm = float(np.linalg.norm(v_lang))

    # Project out v_lang
    H_clean = project_out(hiddens, v_lang)

    lang_post = linear_probe(H_clean, languages)
    emo_post  = linear_probe(H_clean, emotions)
    spk_post  = linear_probe(H_clean, speakers)

    # Geometry: cosine between v_lang and emotion / speaker centroid difference
    def _cos(a, b):
        a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
        return float(abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # Build a representative emotion direction (anger - sad mean)
    h_anger = hiddens[emotions == "anger"].mean(axis=0) if (emotions == "anger").any() else hiddens.mean(0)
    h_sad   = hiddens[emotions == "sad"].mean(axis=0)   if (emotions == "sad").any()   else hiddens.mean(0)
    v_emo   = h_anger - h_sad

    # Speaker direction (F vs M centroid)
    is_f = np.array([s.startswith("F") for s in speakers])
    h_f  = hiddens[is_f].mean(axis=0)  if is_f.any()    else hiddens.mean(0)
    h_m  = hiddens[~is_f].mean(axis=0) if (~is_f).any() else hiddens.mean(0)
    v_spk = h_f - h_m

    return LayerAnalysis(
        layer_id=layer_id,
        n_samples=len(hiddens),
        languages=list(set(languages.tolist())),
        emotions=list(set(emotions.tolist())),
        speakers=list(set(speakers.tolist())),
        lang_acc_pre=lang_pre, emo_acc_pre=emo_pre, spk_acc_pre=spk_pre,
        lang_acc_post=lang_post, emo_acc_post=emo_post, spk_acc_post=spk_post,
        v_lang_norm=v_lang_norm,
        cos_lang_emo=_cos(v_lang, v_emo),
        cos_lang_spk=_cos(v_lang, v_spk),
    )
