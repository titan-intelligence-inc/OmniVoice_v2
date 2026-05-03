"""Pure-logic tests for Phase 3 hooks/steering (no GPU required)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import pytest

from ovet.omnivoice.hidden_hooks import HiddenCapturer, HiddenSteerer
from ovet.omnivoice.projection import remove_projection_per_layer
from ovet.omnivoice.steering import make_clean_vectors


# ---------------------------------------------------------------------
# Tiny mock model with the same .llm.layers structure that the real
# wrapper uses. Each "layer" returns a (hidden, mask) tuple to mimic
# Qwen3DecoderLayer.
# ---------------------------------------------------------------------

class _FakeLayer(nn.Module):
    """Outputs a tuple (hidden, side_info) — like Qwen3DecoderLayer."""
    def __init__(self, hidden_size: int, layer_id: int):
        super().__init__()
        self.bias = nn.Parameter(torch.full((hidden_size,), float(layer_id) * 0.1))
    def forward(self, x):
        return (x + self.bias, "side")  # tuple output


class _FakeLLM(nn.Module):
    def __init__(self, n_layers: int = 4, hidden_size: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(hidden_size, i) for i in range(n_layers)])
    def forward(self, x):
        for ly in self.layers:
            x, _ = ly(x)
        return x


class _FakeModel(nn.Module):
    def __init__(self, n_layers: int = 4, hidden_size: int = 8):
        super().__init__()
        self.llm = _FakeLLM(n_layers, hidden_size)
    def forward(self, x):
        return self.llm(x)


# ---------------------------------------------------------------------
# HiddenCapturer
# ---------------------------------------------------------------------

def test_capturer_records_means_per_layer():
    model = _FakeModel(n_layers=4, hidden_size=8)
    x = torch.zeros(1, 6, 8)  # [batch, seq, hidden]

    with HiddenCapturer(model, layer_ids=[0, 2]) as cap:
        _ = model(x)

    # After layer 0, hidden = 0 + 0.0 = 0; mean = 0
    assert np.allclose(cap.means[0], 0.0)
    # After layer 2, hidden = 0 + 0.0 + 0.1 + 0.2 = 0.3; mean = 0.3
    assert np.allclose(cap.means[2], 0.3, atol=1e-5)


def test_capturer_running_mean_across_calls():
    model = _FakeModel(n_layers=2, hidden_size=4)
    with HiddenCapturer(model, layer_ids=[0]) as cap:
        _ = model(torch.zeros(1, 3, 4))
        _ = model(torch.ones(1, 3, 4))
    # First call mean=0.0, second call mean=1.0; running mean=0.5
    assert np.allclose(cap.means[0], 0.5, atol=1e-5)


def test_capturer_first_only():
    model = _FakeModel(n_layers=2, hidden_size=4)
    with HiddenCapturer(model, layer_ids=[0], first_only=True) as cap:
        _ = model(torch.zeros(1, 3, 4))
        _ = model(torch.ones(1, 3, 4))
    # Only first call captured: mean=0.0
    assert np.allclose(cap.means[0], 0.0, atol=1e-5)


def test_capturer_handles_removed_after_context():
    model = _FakeModel(n_layers=2, hidden_size=4)
    with HiddenCapturer(model, layer_ids=[0]) as cap:
        _ = model(torch.zeros(1, 3, 4))
    # After context, no more capture should happen
    state = dict(cap.means)
    _ = model(torch.ones(1, 3, 4))
    assert np.allclose(cap.means[0], state[0], atol=1e-7)


# ---------------------------------------------------------------------
# HiddenSteerer
# ---------------------------------------------------------------------

def test_steerer_adds_alpha_v():
    model = _FakeModel(n_layers=3, hidden_size=4)
    v = np.ones(4, dtype=np.float32) * 0.5
    x = torch.zeros(1, 2, 4)

    # Use module(x) so __call__ triggers forward hooks.
    out_no = model.llm.layers[0](x)[0].detach()
    with HiddenSteerer(model, alpha=2.0, vectors={0: v}):
        out_st = model.llm.layers[0](x)[0].detach()
    # Steering adds 2.0 * 0.5 = 1.0 to every element
    assert torch.allclose(out_st, out_no + 1.0, atol=1e-5)


def test_steerer_alpha_zero_is_noop():
    model = _FakeModel(n_layers=3, hidden_size=4)
    v = np.ones(4, dtype=np.float32)
    x = torch.zeros(1, 2, 4)

    with HiddenSteerer(model, alpha=0.0, vectors={0: v}):
        out = model.llm.layers[0](x)[0]
    base = model.llm.layers[0](x)[0]
    assert torch.allclose(out, base, atol=1e-7)


def test_steerer_only_affects_specified_layers():
    model = _FakeModel(n_layers=3, hidden_size=4)
    v = np.ones(4, dtype=np.float32)
    x = torch.zeros(1, 2, 4)

    base = [model.llm.layers[i](x)[0].detach().clone() for i in range(3)]
    with HiddenSteerer(model, alpha=1.0, vectors={1: v}):
        for i in range(3):
            out = model.llm.layers[i](x)[0]
            if i == 1:
                assert torch.allclose(out, base[i] + 1.0, atol=1e-5)
            else:
                assert torch.allclose(out, base[i], atol=1e-7)


# ---------------------------------------------------------------------
# steering.make_clean_vectors
# ---------------------------------------------------------------------

def test_make_clean_vectors_passthrough_when_disabled():
    rng = np.random.default_rng(0)
    v_emo = {0: rng.standard_normal(8).astype(np.float32),
             1: rng.standard_normal(8).astype(np.float32)}
    v_lang = {0: rng.standard_normal(8).astype(np.float32)}
    out = make_clean_vectors(v_emo, v_lang, projection_removal=False)
    for i, v in v_emo.items():
        assert np.allclose(out[i], v)


def test_make_clean_vectors_strips_lang_when_enabled():
    rng = np.random.default_rng(1)
    v_emo  = {0: rng.standard_normal(8).astype(np.float32)}
    v_lang = {0: rng.standard_normal(8).astype(np.float32)}
    out = make_clean_vectors(v_emo, v_lang, projection_removal=True)
    # Result should be orthogonal to v_lang
    assert abs(float(np.dot(out[0], v_lang[0]))) < 1e-4


# ---------------------------------------------------------------------
# steering.save / load round-trip
# ---------------------------------------------------------------------

def test_save_load_vectors_roundtrip(tmp_path):
    from ovet.omnivoice.steering import save_vectors, load_vectors
    v = {3: np.array([0.1, 0.2, 0.3], dtype=np.float32),
         7: np.array([1.0, -1.0, 0.5], dtype=np.float32)}
    meta = {"alpha": 0.4, "source": "test"}
    save_vectors(tmp_path / "vec.npz", v, meta)
    loaded, m = load_vectors(tmp_path / "vec.npz")
    assert set(loaded.keys()) == {3, 7}
    assert np.allclose(loaded[3], v[3])
    assert np.allclose(loaded[7], v[7])
    assert m == meta
