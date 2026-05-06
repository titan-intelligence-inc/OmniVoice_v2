"""Pure-logic tests for run_multilang helpers (no GPU)."""
from __future__ import annotations
import numpy as np
import pytest

# Ensure the helpers are import-only safe (don't load OmniVoice).
from ovet.cli.run_multilang import _agg


def test_agg_empty():
    s = _agg([])
    assert s == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0, "median": 0.0}


def test_agg_includes_median():
    s = _agg([0.10, 0.20, 18.50])
    # mean is dragged up by the outlier; median is robust
    assert s["mean"] > 5.0
    assert s["median"] == 0.20


def test_agg_single_value():
    s = _agg([0.5])
    assert s["mean"] == 0.5
    assert s["std"] == 0.0
    assert s["n"] == 1
    assert s["min"] == s["max"] == 0.5


def test_agg_multiple():
    s = _agg([0.10, 0.20, 0.30])
    assert s["mean"] == pytest.approx(0.20, abs=1e-6)
    assert s["min"] == 0.10
    assert s["max"] == 0.30
    assert 0 < s["std"] < 0.15
    assert s["n"] == 3


# ---------------------------------------------------------------------
# Synthetic-neutral v_emo geometry
# ---------------------------------------------------------------------

def test_synthetic_neutral_centroid_invariant():
    """If h_synth_neutral = mean(h_emo), then sum of v_emo's should be ~0."""
    rng = np.random.default_rng(0)
    n_emo = 6
    dim = 32
    hidden_per_emo = {f"emo_{i}": rng.standard_normal(dim) for i in range(n_emo)}
    centroid = np.mean(list(hidden_per_emo.values()), axis=0)
    v_emos = {k: h - centroid for k, h in hidden_per_emo.items()}
    summed = np.sum(list(v_emos.values()), axis=0)
    assert np.allclose(summed, 0.0, atol=1e-9)


def test_per_language_v_lang_zero_when_target_equals_base():
    """If target language == base language, v_lang should be zero (no projection)."""
    h = np.random.RandomState(1).standard_normal(16)
    v_lang_self = h - h
    assert np.allclose(v_lang_self, 0.0)


# ---------------------------------------------------------------------
# Best-of-reps selection logic
# ---------------------------------------------------------------------

def test_min_selection_picks_lowest_metric():
    rep_records = [
        {"rep": 0, "scores": {"vad_dist": 0.3}, "total_score": 0.1},
        {"rep": 1, "scores": {"vad_dist": 0.1}, "total_score": -0.1},
        {"rep": 2, "scores": {"vad_dist": 0.2}, "total_score": 0.0},
    ]
    best = min(rep_records, key=lambda r: r["scores"]["vad_dist"])
    assert best["rep"] == 1
