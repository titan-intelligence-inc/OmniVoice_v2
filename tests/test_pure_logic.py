"""Pure-logic tests for ovet (no GPU / no model loads required)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from ovet.types import (
    Candidate, CandidateScores, Thresholds, ScoringWeights,
)
from ovet.prompts.instruct_proxy import (
    InstructProxyComposer, SpeakerAttrs, VALID_TOKENS,
)
from ovet.omnivoice.projection import (
    remove_projection, remove_projection_per_layer,
)
from ovet.evaluation.scoring import compute_total_score
from ovet.evaluation.selector import select_best, filter_valid


# -----------------------------------------------------------------------
# instruct_proxy
# -----------------------------------------------------------------------

def test_proxy_returns_only_valid_tokens():
    c = InstructProxyComposer()
    for emo in ["anger", "sad", "happy", "fear", "disgust", "surprise", "calm"]:
        s = c.compose(emo)
        assert s is None or InstructProxyComposer.is_valid(s), f"invalid for {emo}: {s!r}"


def test_proxy_unknown_emotion_returns_none_or_attrs_only():
    c = InstructProxyComposer()
    assert c.compose("not_an_emotion") is None
    # With prepend_attrs, gender/age can still be returned even for unknown emo.
    c2 = InstructProxyComposer(prepend_attrs=True)
    s = c2.compose("not_an_emotion", SpeakerAttrs(gender="female"))
    assert s == "female"


def test_proxy_prepends_attrs():
    c = InstructProxyComposer(prepend_attrs=True)
    s = c.compose("anger", SpeakerAttrs(gender="female", age="young adult"))
    parts = [p.strip() for p in s.split(",")]
    assert parts[0] == "female"
    assert parts[1] == "young adult"
    # emotion proxy "high pitch" should be appended.
    assert "high pitch" in parts


def test_proxy_no_duplicate_tokens():
    c = InstructProxyComposer(
        mapping={"anger": ["high pitch", "high pitch"]},
        prepend_attrs=False,
    )
    s = c.compose("anger")
    assert s == "high pitch"


# -----------------------------------------------------------------------
# projection
# -----------------------------------------------------------------------

def test_remove_projection_orthogonality():
    rng = np.random.default_rng(0)
    v_emo  = rng.standard_normal(64).astype(np.float32)
    v_lang = rng.standard_normal(64).astype(np.float32)
    cleaned = remove_projection(v_emo, v_lang)
    assert abs(float(np.dot(cleaned, v_lang))) < 1e-4


def test_remove_projection_idempotent():
    rng = np.random.default_rng(1)
    v_emo  = rng.standard_normal(32).astype(np.float32)
    v_lang = rng.standard_normal(32).astype(np.float32)
    once  = remove_projection(v_emo, v_lang)
    twice = remove_projection(once, v_lang)
    assert np.allclose(once, twice, atol=1e-5)


def test_remove_projection_per_layer_pass_through_missing():
    rng = np.random.default_rng(2)
    emo_vecs = {0: rng.standard_normal(8).astype(np.float32),
                1: rng.standard_normal(8).astype(np.float32)}
    lang_vecs = {0: rng.standard_normal(8).astype(np.float32)}
    out = remove_projection_per_layer(emo_vecs, lang_vecs)
    assert np.allclose(out[1], emo_vecs[1])
    # Layer 0 should be orthogonal to lang
    assert abs(float(np.dot(out[0], lang_vecs[0]))) < 1e-4


# -----------------------------------------------------------------------
# scoring
# -----------------------------------------------------------------------

def _make_scores(**override):
    base = dict(
        vad_dist=0.20, valence_diff=0.10, arousal_diff=0.05, dominance_diff=0.05,
        f0_std_ratio=1.05, energy_std_ratio=0.90,
        e2v_cos=0.80, e2v_p_target=0.50, e2v_top="angry",
        speaker_sim=0.95, content_error=0.10, audio_quality=4.0,
    )
    base.update(override)
    return CandidateScores(**base)


def test_scoring_higher_is_better_for_e2v_cos():
    w = ScoringWeights()
    low = compute_total_score(_make_scores(e2v_cos=0.30), w)
    high = compute_total_score(_make_scores(e2v_cos=0.95), w)
    assert high > low


def test_scoring_lower_is_better_for_vad_dist():
    w = ScoringWeights()
    near = compute_total_score(_make_scores(vad_dist=0.05), w)
    far  = compute_total_score(_make_scores(vad_dist=0.50), w)
    assert near > far


def test_scoring_lower_is_better_for_content_error():
    w = ScoringWeights()
    clean = compute_total_score(_make_scores(content_error=0.02), w)
    bad   = compute_total_score(_make_scores(content_error=0.30), w)
    assert clean > bad


def test_scoring_f0_ratio_penalizes_deviation_from_one():
    w = ScoringWeights()
    perfect = compute_total_score(_make_scores(f0_std_ratio=1.00), w)
    over    = compute_total_score(_make_scores(f0_std_ratio=1.50), w)
    under   = compute_total_score(_make_scores(f0_std_ratio=0.50), w)
    assert perfect > over
    assert perfect > under


# -----------------------------------------------------------------------
# selector
# -----------------------------------------------------------------------

def _make_candidate(tag: str, total: float, **score_overrides) -> Candidate:
    return Candidate(
        wav_path=Path(f"/tmp/{tag}.wav"),
        instruct=None,
        alpha=0.0,
        layer_ids=[],
        projection_removal_language=False,
        scores=_make_scores(**score_overrides),
        total_score=total,
        meta={"tag": tag},
    )


def test_select_best_picks_max_score_among_valid():
    cs = [
        _make_candidate("a", total=0.5),
        _make_candidate("b", total=0.9),
        _make_candidate("c", total=0.7),
    ]
    out = select_best(cs, Thresholds())
    assert out.meta["tag"] == "b"


def test_select_best_filters_constraints():
    cs = [
        # b has the best total but violates content_error
        _make_candidate("a", total=0.6),
        _make_candidate("b", total=0.9, content_error=0.5),
        _make_candidate("c", total=0.7),
    ]
    out = select_best(cs, Thresholds(content_error=0.20))
    assert out.meta["tag"] == "c"


def test_select_best_fallback_when_none_valid():
    cs = [
        _make_candidate("a", total=0.6, content_error=0.30),
        _make_candidate("b", total=0.9, content_error=0.40),
    ]
    out = select_best(cs, Thresholds(content_error=0.20))
    # Both invalid; fallback prefers lower content_error
    assert out.meta["tag"] == "a"


def test_filter_valid_drops_below_threshold():
    cs = [
        _make_candidate("a", 0.5, speaker_sim=0.99),
        _make_candidate("b", 0.5, speaker_sim=0.40),
        _make_candidate("c", 0.5, audio_quality=0.10),
    ]
    valid = filter_valid(cs, Thresholds(speaker=0.85, quality=0.50))
    tags = {c.meta["tag"] for c in valid}
    assert tags == {"a"}
