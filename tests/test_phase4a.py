"""Pure-logic tests for the Phase 4a spike (no model loads required)."""
from __future__ import annotations
import numpy as np
import pytest

from ovet.analysis.phase4a import (
    linear_probe, v_lang_from_means, project_out, analyze_layer,
)


# ---------------------------------------------------------------------
# linear_probe
# ---------------------------------------------------------------------

def test_linear_probe_perfect_when_separable():
    rng = np.random.default_rng(0)
    # Two clusters along axis 0
    X1 = rng.standard_normal((10, 4)) + np.array([5, 0, 0, 0])
    X2 = rng.standard_normal((10, 4)) + np.array([-5, 0, 0, 0])
    X = np.concatenate([X1, X2])
    y = ["A"] * 10 + ["B"] * 10
    acc = linear_probe(X, y)
    assert acc > 0.95


def test_linear_probe_chance_when_random():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 4))
    y = ["A", "B"] * 10
    acc = linear_probe(X, y)
    # LOO accuracy on iid noise should be near chance (0.5)
    assert acc < 0.75


# ---------------------------------------------------------------------
# v_lang_from_means + project_out
# ---------------------------------------------------------------------

def test_v_lang_centroid_difference():
    rng = np.random.default_rng(2)
    direction = np.array([1.0, 0.0, 0.0, 0.0])
    H_l1 = rng.standard_normal((5, 4)) * 0.1 + 3 * direction
    H_l2 = rng.standard_normal((5, 4)) * 0.1 - 3 * direction
    H = np.concatenate([H_l1, H_l2])
    langs = ["JA"] * 5 + ["EN"] * 5
    v = v_lang_from_means(H, langs, "JA", "EN")
    # First component should dominate
    assert abs(v[0]) > 4.0
    assert np.linalg.norm(v[1:]) < 1.0


def test_project_out_orthogonal():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((20, 8)).astype(np.float32)
    d = rng.standard_normal(8).astype(np.float32)
    X_clean = project_out(X, d)
    inner = X_clean @ d
    assert np.max(np.abs(inner)) < 1e-3


def test_project_out_kills_lang_classification():
    """If language differs only along v_lang, projecting kills lang acc."""
    rng = np.random.default_rng(4)
    direction = np.array([1.0] + [0.0] * 7)
    H_ja = rng.standard_normal((10, 8)).astype(np.float32) * 0.05 + 5 * direction
    H_en = rng.standard_normal((10, 8)).astype(np.float32) * 0.05 - 5 * direction
    H = np.concatenate([H_ja, H_en]).astype(np.float32)
    langs = ["JA"] * 10 + ["EN"] * 10

    # Pre-projection: language is trivially separable
    pre = linear_probe(H, langs)
    assert pre > 0.95

    v = v_lang_from_means(H, langs, "JA", "EN")
    H_clean = project_out(H, v)
    post = linear_probe(H_clean, langs)
    # After removing v_lang, the residual is ~iid noise → near chance
    assert post < 0.7


# ---------------------------------------------------------------------
# analyze_layer end-to-end with synthetic data
# ---------------------------------------------------------------------

def test_analyze_layer_synthetic_clean_signal():
    """Construct hiddens where:
       - language information lives along axis 0 (orthogonal to other axes)
       - emotion information lives along axis 1
       - speaker information lives along axis 2
    Expect projection to wipe lang while preserving emo/spk.
    """
    rng = np.random.default_rng(5)

    # Build 4 speakers × 3 emotions × 2 languages = 24 samples
    samples = []
    speakers = []; emotions = []; languages = []
    for sp_idx, sp in enumerate(["F1", "F2", "M1", "M2"]):
        for emo_idx, emo in enumerate(["anger", "sad", "fear"]):
            for lang in ["Japanese", "English"]:
                v = rng.standard_normal(8).astype(np.float32) * 0.05
                # axis 0: language
                v[0] += 5.0 if lang == "Japanese" else -5.0
                # axis 1: emotion (3 values)
                v[1] += {"anger": 5.0, "sad": -5.0, "fear": 0.0}[emo]
                # axis 2: speaker
                v[2] += {"F1": 3.0, "F2": 1.0, "M1": -1.0, "M2": -3.0}[sp]
                samples.append(v)
                speakers.append(sp); emotions.append(emo); languages.append(lang)

    H = np.stack(samples)
    ana = analyze_layer(H, speakers, emotions, languages, layer_id=0)

    # Pre-projection: all three should be separable
    assert ana.lang_acc_pre > 0.9
    assert ana.emo_acc_pre  > 0.6
    assert ana.spk_acc_pre  > 0.6

    # Post-projection: language should drop a lot, emo/spk preserved
    assert ana.lang_acc_post < 0.7
    assert ana.emo_acc_post  > 0.6
    assert ana.spk_acc_post  > 0.6
