"""Unit tests for Phase 2-specific pure-logic modules."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from ovet.config import load_config, OvetConfig
from ovet.prompts.proxy_grid import ProxyGrid
from ovet.prompts.instruct_proxy import InstructProxyComposer
from ovet.evaluation.stability import _stats, is_metric_reliable, MetricStats
from ovet.evaluation.report import render_report
from ovet.types import Thresholds


# -----------------------------------------------------------------------
# config loader
# -----------------------------------------------------------------------

def test_default_config_loads():
    cfg = load_config()
    assert isinstance(cfg, OvetConfig)
    # Sanity values from the YAML
    assert cfg.thresholds.content_error <= 0.5
    assert "anger" in cfg.proxy_grid
    assert any(p == "high pitch" for p in cfg.proxy_grid["anger"] if p)


def test_config_to_dict_roundtrip():
    cfg = load_config()
    d = cfg.to_dict()
    assert "scoring" in d
    assert "vad" in d["scoring"]


# -----------------------------------------------------------------------
# proxy_grid
# -----------------------------------------------------------------------

def test_proxy_grid_baseline_always_present():
    pg = ProxyGrid({"anger": ["high pitch", "low pitch"]})  # no None entry
    out = pg.for_emotion("anger")
    tags = {t for t, _ in out}
    assert "baseline" in tags


def test_proxy_grid_dedupes():
    pg = ProxyGrid({"anger": [None, "high pitch", "high pitch"]})
    out = pg.for_emotion("anger")
    instructs = [i for _, i in out]
    assert instructs.count("high pitch") == 1
    assert None in instructs


def test_proxy_grid_rejects_invalid_token():
    with pytest.raises(ValueError):
        ProxyGrid({"anger": ["totally made up phrase"]}).for_emotion("anger")


def test_proxy_grid_unknown_emotion_falls_back():
    # ProxyGrid composes from default mapping when key is absent
    pg = ProxyGrid({})
    out = pg.for_emotion("anger")
    instructs = [i for _, i in out]
    # baseline present + at least one valid proxy from default mapping
    assert None in instructs
    assert any(i is not None and InstructProxyComposer.is_valid(i) for i in instructs)


def test_proxy_grid_filename_safe_tags():
    pg = ProxyGrid({"sad": [None, "low pitch", "very low pitch"]})
    out = pg.for_emotion("sad")
    for tag, _ in out:
        assert "/" not in tag
        assert " " not in tag


# -----------------------------------------------------------------------
# stability stats
# -----------------------------------------------------------------------

def test_stats_single_value_zero_std():
    s = _stats([0.5])
    assert s.std == 0.0
    assert s.mean == 0.5
    assert s.n == 1


def test_stats_multiple_values():
    s = _stats([0.10, 0.20, 0.30])
    assert s.mean == pytest.approx(0.20, abs=1e-6)
    assert 0.0 < s.std < 0.15
    assert s.minv == pytest.approx(0.10)
    assert s.maxv == pytest.approx(0.30)
    assert s.n == 3


def test_is_metric_reliable():
    s_low  = MetricStats(mean=0.5, std=0.03, minv=0.4, maxv=0.6, n=3)
    s_high = MetricStats(mean=0.5, std=0.10, minv=0.3, maxv=0.7, n=3)
    assert is_metric_reliable(s_low, threshold=0.05)
    assert not is_metric_reliable(s_high, threshold=0.05)


# -----------------------------------------------------------------------
# report rendering
# -----------------------------------------------------------------------

def test_render_report_basic_structure():
    from ovet.types import Candidate, CandidateScores

    sc = CandidateScores(
        vad_dist=0.10, valence_diff=0.05, arousal_diff=0.03, dominance_diff=0.04,
        f0_std_ratio=1.05, energy_std_ratio=0.95,
        e2v_cos=0.80, e2v_p_target=0.40, e2v_top="angry",
        speaker_sim=0.78, content_error=0.05, audio_quality=4.0,
    )
    cand = Candidate(
        wav_path=Path("/tmp/x.wav"), instruct="high pitch",
        alpha=0.0, layer_ids=[], projection_removal_language=False,
        scores=sc, total_score=0.5,
        meta={"tag": "proxy"},
    )
    md = render_report(
        title="test",
        request_meta={"text": "hi", "language": "English"},
        ref_summary={"emotion": "angry"},
        candidates=[cand],
        best_tag="proxy",
        thresholds=Thresholds(),
    )
    assert "# test" in md
    assert "proxy" in md
    assert "high pitch" in md
    assert "Selected" in md
