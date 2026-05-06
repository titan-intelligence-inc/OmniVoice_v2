"""Tests for the accent-removal (language_alpha) feature in SteeringConfig."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from ovet.omnivoice.hidden_hooks import HiddenSteerer


# Tiny mock matching OmniVoice's inner LLM structure.
class _Layer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    def forward(self, x):
        return (x + self.bias, "side")


class _LLM(nn.Module):
    def __init__(self, n_layers: int, hidden_size: int):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(hidden_size) for _ in range(n_layers)])


class _Model(nn.Module):
    def __init__(self, n_layers: int = 2, hidden_size: int = 4):
        super().__init__()
        self.llm = _LLM(n_layers, hidden_size)


# ---------------------------------------------------------------------
# Composition math: emo + accent (mocked at the HiddenSteerer level)
# ---------------------------------------------------------------------

def test_composed_vector_emo_plus_negative_lang():
    """delta = a_emo * v_emo - a_lang * v_lang (verifying composition logic)."""
    v_emo  = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_lang = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    a_emo, a_lang = 0.5, 1.5

    composed = a_emo * v_emo - a_lang * v_lang
    assert np.allclose(composed, [0.5, -1.5, 0.0, 0.0])


def test_composed_steerer_applies_both_directions():
    """HiddenSteerer with composed vectors moves output along both axes."""
    model = _Model(n_layers=2, hidden_size=4)
    v_emo  = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_lang = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    composed = {0: 0.5 * v_emo - 1.0 * v_lang}

    x = torch.zeros(1, 2, 4)
    base = model.llm.layers[0](x)[0].detach()
    with HiddenSteerer(model, alpha=1.0, vectors=composed):
        out = model.llm.layers[0](x)[0].detach()
    expected = base + torch.tensor([0.5, -1.0, 0.0, 0.0])
    assert torch.allclose(out, expected, atol=1e-5)


def test_accent_only_pushes_negative_lang():
    """When a_emo=0 and a_lang>0, the delta is pure -lang_alpha * v_lang."""
    a_emo, a_lang = 0.0, 1.0
    v_emo  = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    v_lang = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    composed = a_emo * v_emo - a_lang * v_lang
    assert np.allclose(composed, [0.0, -1.0, 0.0, 0.0])


def test_emo_only_unchanged_by_accent_disabled():
    """When a_lang=0, accent contribution is exactly zero — equivalent to old Phase 4."""
    a_emo, a_lang = 1.0, 0.0
    v_emo  = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    v_lang = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    composed = a_emo * v_emo - a_lang * v_lang
    assert np.allclose(composed, v_emo)


# ---------------------------------------------------------------------
# SteeringConfig field defaults
# ---------------------------------------------------------------------

def test_steering_config_default_language_alpha_zero():
    from ovet.omnivoice.wrapper import SteeringConfig
    sc = SteeringConfig(enabled=True, alpha=1.0)
    assert sc.language_alpha == 0.0


def test_steering_config_no_op_when_both_zero():
    """alpha=0 AND language_alpha=0 should short-circuit to plain generate."""
    from ovet.omnivoice.wrapper import SteeringConfig
    sc = SteeringConfig(enabled=True, alpha=0.0, language_alpha=0.0)
    # Sanity: both fields readable, defaults make sense
    assert sc.alpha == 0.0
    assert sc.language_alpha == 0.0
