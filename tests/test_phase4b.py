"""Tests for Phase 4b wrapper integration (no GPU)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from ovet.omnivoice.hidden_hooks import HiddenSteerer
from ovet.omnivoice.projection import remove_projection_per_layer


# Tiny mock matching the structure of OmniVoice's inner LLM.
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
    def __init__(self, n_layers: int = 4, hidden_size: int = 8):
        super().__init__()
        self.llm = _LLM(n_layers, hidden_size)


def test_steerer_with_projected_vectors():
    """Steering with v_emo' (projection-cleaned) is equivalent to applying
    remove_projection_per_layer first then HiddenSteerer."""
    model = _Model(n_layers=3, hidden_size=4)
    rng = np.random.default_rng(0)

    layers = [0, 1]
    v_emo  = {i: rng.standard_normal(4).astype(np.float32) for i in layers}
    v_lang = {i: rng.standard_normal(4).astype(np.float32) for i in layers}

    cleaned = remove_projection_per_layer(v_emo, v_lang)
    # Each cleaned should be orthogonal to v_lang
    for i in layers:
        assert abs(float(np.dot(cleaned[i], v_lang[i]))) < 1e-4

    x = torch.zeros(1, 3, 4)
    with HiddenSteerer(model, alpha=1.0, vectors=cleaned):
        out = model.llm.layers[0](x)[0].detach()

    # Manually compute expected output
    expected = torch.from_numpy(cleaned[0]).float()  # bias=0 so output is just alpha*v
    assert torch.allclose(out[0, 0], expected, atol=1e-5)


def test_projection_when_emo_already_orthogonal():
    """If v_emo is already orthogonal to v_lang, projection is a no-op."""
    v_lang = {0: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}
    # v_emo has zero component along v_lang
    v_emo  = {0: np.array([0.0, 1.0, 0.5, -0.5], dtype=np.float32)}
    cleaned = remove_projection_per_layer(v_emo, v_lang)
    assert np.allclose(cleaned[0], v_emo[0], atol=1e-6)


def test_projection_when_emo_parallel_to_lang():
    """If v_emo is fully parallel to v_lang, projection zeroes it."""
    v_lang = {0: np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)}
    v_emo  = {0: np.array([2.0, 2.0, 0.0, 0.0], dtype=np.float32)}
    cleaned = remove_projection_per_layer(v_emo, v_lang)
    assert np.allclose(cleaned[0], 0.0, atol=1e-5)


def test_steerer_partial_lang_subset():
    """Layers in v_emo but missing from v_lang should pass through unchanged."""
    v_emo  = {0: np.ones(4, dtype=np.float32), 1: np.ones(4, dtype=np.float32) * 2}
    v_lang = {0: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)}  # only layer 0
    cleaned = remove_projection_per_layer(v_emo, v_lang)
    # Layer 0: orthogonal to v_lang
    assert abs(float(np.dot(cleaned[0], v_lang[0]))) < 1e-4
    # Layer 1: passed through unchanged
    assert np.allclose(cleaned[1], v_emo[1])
