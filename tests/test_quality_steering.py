"""Tests for the quality-tuning extensions on HiddenSteerer:

- step_window: apply only on diffusion steps in [start, end)
- norm_clip_factor: cap |delta| relative to mean(|h|)
"""
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
    def forward(self, x):
        for ly in self.layers:
            x, _ = ly(x)
        return x


class _Model(nn.Module):
    def __init__(self, n_layers: int = 3, hidden_size: int = 4):
        super().__init__()
        self.llm = _LLM(n_layers, hidden_size)
    def forward(self, x):
        return self.llm(x)


# ---------------------------------------------------------------------
# step_window
# ---------------------------------------------------------------------

def test_step_window_skips_outside():
    """delta only applied when step counter is inside [start, end)."""
    model = _Model(n_layers=3, hidden_size=4)
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    x = torch.zeros(1, 2, 4)

    # Window (1, 3): step 0 should NOT steer; steps 1, 2 should; step 3+ should NOT
    with HiddenSteerer(model, alpha=1.0, vectors={2: v}, step_window=(1, 3)):
        # step 0 — not steered
        out0 = model(x).detach().clone()
        # step 1 — steered
        out1 = model(x).detach().clone()
        # step 2 — steered
        out2 = model(x).detach().clone()
        # step 3 — not steered
        out3 = model(x).detach().clone()

    base = torch.zeros_like(out0)
    assert torch.allclose(out0, base, atol=1e-5)
    expected_steered = base + torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(out1, expected_steered, atol=1e-5)
    assert torch.allclose(out2, expected_steered, atol=1e-5)
    assert torch.allclose(out3, base, atol=1e-5)


def test_step_window_open_end():
    """end=None means apply from start indefinitely."""
    model = _Model(n_layers=3, hidden_size=4)
    v = np.ones(4, dtype=np.float32)
    x = torch.zeros(1, 1, 4)

    with HiddenSteerer(model, alpha=1.0, vectors={2: v}, step_window=(2, None)):
        out0 = model(x).detach().clone()  # not yet
        out1 = model(x).detach().clone()  # not yet
        out2 = model(x).detach().clone()  # now in
        out3 = model(x).detach().clone()  # still in

    assert torch.allclose(out0, torch.zeros_like(out0), atol=1e-5)
    assert torch.allclose(out1, torch.zeros_like(out0), atol=1e-5)
    assert torch.allclose(out2, torch.ones_like(out0),  atol=1e-5)
    assert torch.allclose(out3, torch.ones_like(out0),  atol=1e-5)


def test_step_counter_advances_per_forward_not_per_layer():
    """When multiple layers are hooked, step counter ticks once per LLM forward."""
    model = _Model(n_layers=3, hidden_size=4)
    v = np.ones(4, dtype=np.float32)
    x = torch.zeros(1, 1, 4)

    with HiddenSteerer(model, alpha=1.0, vectors={0: v, 1: v, 2: v},
                       step_window=(0, 2)) as st:
        _ = model(x)  # step 0 — should steer
        # All 3 layers fired in this single forward; counter should be 1 now.
        assert st._step_counter == 1
        _ = model(x)  # step 1 — should steer
        assert st._step_counter == 2
        # step 2 onward — outside window
        out = model(x).detach().clone()
        assert st._step_counter == 3

    # Last forward (step 2) was outside the window, so net effect at step 2
    # is zero added vector at any layer; final hidden = bias = 0.
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


# ---------------------------------------------------------------------
# norm_clip_factor
# ---------------------------------------------------------------------

def test_norm_clip_caps_delta():
    """If alpha*v has magnitude > β * mean(|h|), it's scaled down."""
    model = _Model(n_layers=2, hidden_size=4)
    # Inputs with known norm
    x = torch.ones(1, 2, 4)  # |h_t| per token = sqrt(4) = 2.0 → mean 2.0
    # Steering vector with large norm
    v = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)  # |v| = 20.0

    # Without clip: huge delta
    with HiddenSteerer(model, alpha=1.0, vectors={1: v}):
        out_no_clip = model(x).detach().clone()
    # With clip factor 0.5: delta capped at 0.5 * 2.0 = 1.0
    with HiddenSteerer(model, alpha=1.0, vectors={1: v}, norm_clip_factor=0.5):
        out_clip = model(x).detach().clone()

    # Clipped output should be much closer to base
    base = x  # no steering = h unchanged
    diff_no = (out_no_clip - base).norm().item()
    diff_clip = (out_clip - base).norm().item()
    assert diff_clip < diff_no
    # The capped delta-norm per position is ~1.0; total over (1,2,4) shape = sqrt(2) ≈ 1.41
    assert diff_clip < 2.5


def test_norm_clip_preserves_direction():
    """Clipping reduces magnitude but keeps direction."""
    model = _Model(n_layers=2, hidden_size=4)
    x = torch.ones(1, 1, 4)
    v = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)  # along axis 0
    base = model(x).detach().clone()

    with HiddenSteerer(model, alpha=1.0, vectors={1: v}, norm_clip_factor=0.5):
        out = model(x).detach().clone()

    delta = (out - base)[0, 0]  # along axis 0 only
    # Direction preserved: positive on axis 0, zero on others
    assert delta[0] > 0
    assert torch.allclose(delta[1:], torch.zeros(3), atol=1e-5)


def test_norm_clip_disabled_when_factor_none():
    """norm_clip_factor=None means no clipping."""
    model = _Model(n_layers=2, hidden_size=4)
    x = torch.zeros(1, 1, 4)
    v = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float32)

    with HiddenSteerer(model, alpha=1.0, vectors={1: v}, norm_clip_factor=None):
        out = model(x)
    # delta = 100 along axis 0
    assert out[0, 0, 0].item() == 100.0


# ---------------------------------------------------------------------
# combined: window + clip
# ---------------------------------------------------------------------

def test_window_and_clip_compose():
    model = _Model(n_layers=2, hidden_size=4)
    x = torch.ones(1, 1, 4)
    v = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)

    with HiddenSteerer(model, alpha=1.0, vectors={1: v},
                       step_window=(0, 1), norm_clip_factor=0.25):
        in_window = model(x).detach().clone()
        out_window = model(x).detach().clone()

    # Step 0: in window, clip applied → small positive on axis 0
    delta = (in_window - x)[0, 0]
    assert 0.0 < delta[0].item() < 1.0   # |h|=2.0, cap=0.5 → max delta=0.5
    # Step 1: outside window → no steering at all
    assert torch.allclose(out_window, x, atol=1e-5)
