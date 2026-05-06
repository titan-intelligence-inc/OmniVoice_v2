"""Tests for position_mask: delta is applied only where audio_mask=True."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from ovet.omnivoice.hidden_hooks import HiddenSteerer


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
    """Mock matching OmniVoice's wrapper.

    forward(input_ids, audio_mask) — we ignore input_ids but stash audio_mask
    when HiddenSteerer's position_mask is on.
    """
    def __init__(self, n_layers: int = 2, hidden_size: int = 4):
        super().__init__()
        self.llm = _LLM(n_layers, hidden_size)

    def forward(self, input_ids, audio_mask):  # noqa: ARG002 — kept for API parity
        # Run the LLM on a derived input
        x = torch.zeros(*audio_mask.shape, self.llm.layers[0].bias.numel())
        for ly in self.llm.layers:
            x, _ = ly(x)
        return x


# ---------------------------------------------------------------------
# Position mask behavior
# ---------------------------------------------------------------------

def test_position_mask_applies_only_at_audio_positions():
    model = _Model(n_layers=1, hidden_size=4)
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # batch=1, seq=4. audio at positions [1, 2]; text at [0, 3]
    audio_mask = torch.tensor([[False, True, True, False]])
    input_ids = torch.zeros_like(audio_mask, dtype=torch.long)

    with HiddenSteerer(model, alpha=1.0, vectors={0: v}, position_mask=True):
        out = model(input_ids, audio_mask)

    # delta = [1,0,0,0]; only positions 1, 2 should have it added
    # h was zero everywhere; after layer with bias=0; out[..., 0] should be 1 at audio pos
    expected_axis0 = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
    assert torch.allclose(out[0, :, 0], expected_axis0[0], atol=1e-5)


def test_position_mask_off_applies_everywhere():
    """Without position_mask=True, delta is added uniformly."""
    model = _Model(n_layers=1, hidden_size=4)
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    audio_mask = torch.tensor([[False, True, True, False]])
    input_ids = torch.zeros_like(audio_mask, dtype=torch.long)

    with HiddenSteerer(model, alpha=1.0, vectors={0: v}, position_mask=False):
        out = model(input_ids, audio_mask)

    # delta added at every position
    assert torch.allclose(out[0, :, 0], torch.ones(4), atol=1e-5)


def test_position_mask_stashes_audio_mask_and_cleans_up():
    """Inside the context, audio_mask is stashed on the model. After exit it's removed."""
    model = _Model(n_layers=1, hidden_size=4)
    v = np.ones(4, dtype=np.float32)
    audio_mask = torch.tensor([[True, True]])
    input_ids = torch.zeros_like(audio_mask, dtype=torch.long)

    with HiddenSteerer(model, alpha=1.0, vectors={0: v}, position_mask=True):
        _ = model(input_ids, audio_mask)
        assert getattr(model, "_ovet_audio_mask", None) is not None
        assert torch.equal(model._ovet_audio_mask, audio_mask)

    # After exit: stashed mask removed
    assert not hasattr(model, "_ovet_audio_mask")
