"""Tests for the CER cap + Whisper-hallucination guard."""
from __future__ import annotations
import pytest

from ovet.analyzers.asr_analyzer import _cer, _detect_hallucination


# ---------------------------------------------------------------------
# _detect_hallucination
# ---------------------------------------------------------------------

def test_detects_single_char_loop():
    assert _detect_hallucination("娃娃娃娃娃娃娃娃娃")
    assert _detect_hallucination("aaaaaaaaaaaaa")


def test_detects_short_cycle_loop():
    assert _detect_hallucination("abcabcabcabcabc", min_repeats=5)
    assert _detect_hallucination("123412341234123412341234")


def test_does_not_flag_normal_text():
    assert not _detect_hallucination("Hello, this is a normal sentence.")
    assert not _detect_hallucination("明天的天气预报是多云转晴。")
    assert not _detect_hallucination("Большое спасибо, что пришли сегодня.")


def test_short_string_not_flagged():
    # ≥ 5 repeats of a 1-char cycle = 5 chars minimum
    assert not _detect_hallucination("abcd")
    # Just under threshold
    assert not _detect_hallucination("aaaa", min_repeats=5)


# ---------------------------------------------------------------------
# _cer with cap
# ---------------------------------------------------------------------

def test_cer_normal_no_cap_needed():
    assert _cer("hello", "hello") == 0.0
    assert _cer("hello", "world") == pytest.approx(0.8)
    assert _cer("hello", "helo") == pytest.approx(0.2)


def test_cer_caps_long_hallucination():
    ref = "明天的天气预报是多云转晴"          # 12 chars
    hyp = "明天的展示" + "娃" * 220           # huge hallucination loop
    capped = _cer(ref, hyp, cap=2.0)
    assert capped == 2.0
    raw = _cer(ref, hyp, cap=None)
    assert raw > 5.0


def test_cer_caps_long_text_without_loop():
    ref = "abc"
    hyp = "abc" + "x" * 100   # no repetition loop, but very long
    capped = _cer(ref, hyp, cap=2.0)
    assert capped == 2.0


def test_cer_cap_disabled():
    ref = "abc"
    hyp = "x" * 100
    # cap=None → may exceed 1.0
    raw = _cer(ref, hyp, cap=None)
    assert raw > 30.0


def test_cer_empty_ref():
    assert _cer("", "") == 0.0
    assert _cer("", "x") == 1.0
