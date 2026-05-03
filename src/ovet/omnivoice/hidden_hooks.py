"""Forward-hook helpers for Qwen3 layers inside OmniVoice.

Two primitives:

- :class:`HiddenCapturer` — context manager that records each target
  layer's output (mean-over-positions) on every forward call.
- :class:`HiddenSteerer` — context manager that adds ``alpha * v`` to
  each target layer's output on every forward call.

Both treat tuple-output layers (Qwen3DecoderLayer returns
``(hidden_states, ...)``) and tensor-output layers transparently.
"""
from __future__ import annotations
from typing import Iterable
import numpy as np
import torch


def _split_output(outputs):
    """Return (hidden_state_tensor, rest_or_None) for a layer output."""
    if isinstance(outputs, tuple):
        return outputs[0], outputs[1:]
    return outputs, None


def _rebuild_output(new_h, rest):
    if rest is None:
        return new_h
    return (new_h, *rest)


class HiddenCapturer:
    """Capture per-layer hidden-state means during forward passes.

    After a forward, ``capturer.means[layer_id]`` holds a 1-D numpy array
    of shape ``[hidden_size]`` averaged over the (batch, sequence) axes.
    Multiple calls accumulate as a running mean over **all** invocations
    (e.g. across diffusion steps) — set ``first_only=True`` to only keep
    the first call's snapshot.

    Usage::

        with HiddenCapturer(model, layer_ids=[8, 12, 16]) as cap:
            ...  # run forward / generate
        v8 = cap.means[8]
    """

    def __init__(
        self,
        model,
        layer_ids: Iterable[int],
        first_only: bool = False,
    ):
        self.model = model
        self.layer_ids = list(layer_ids)
        self.first_only = first_only

        self._handles = []
        self._sums:  dict[int, np.ndarray] = {}
        self._counts: dict[int, int]       = {}
        self._seen:  dict[int, bool]       = {}

    # ------------------------------------------------------------------
    def __enter__(self):
        for i in self.layer_ids:
            layer = self.model.llm.layers[i]
            self._handles.append(
                layer.register_forward_hook(self._make_hook(i))
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    def _make_hook(self, layer_id: int):
        def hook(_module, _inputs, outputs):
            if self.first_only and self._seen.get(layer_id):
                return outputs
            h, _ = _split_output(outputs)
            # h shape: [batch, seq, hidden]; mean over (batch, seq) → [hidden]
            mean = h.detach().to(torch.float32).mean(dim=(0, 1)).cpu().numpy()
            if layer_id in self._sums:
                self._sums[layer_id]   += mean
                self._counts[layer_id] += 1
            else:
                self._sums[layer_id]   = mean.astype(np.float32, copy=True)
                self._counts[layer_id] = 1
            self._seen[layer_id] = True
            return outputs
        return hook

    # ------------------------------------------------------------------
    @property
    def means(self) -> dict[int, np.ndarray]:
        """Per-layer running mean of captured hidden states."""
        return {
            i: self._sums[i] / max(self._counts[i], 1)
            for i in self._sums
        }


class HiddenSteerer:
    """Add a per-layer steering vector to each target layer's output.

    ``vectors[layer_id]`` is a 1-D numpy array of shape ``[hidden_size]``.
    The hook adds ``alpha * vectors[i]`` to the layer-i hidden state on
    every forward call.

    Usage::

        with HiddenSteerer(model, alpha=0.4, vectors={8: v8, 12: v12}):
            audio = wrapper.generate(...)
    """

    def __init__(
        self,
        model,
        alpha: float,
        vectors: dict[int, np.ndarray],
    ):
        self.model   = model
        self.alpha   = float(alpha)
        self.vectors = vectors
        self._handles = []
        self._tensors: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def __enter__(self):
        if self.alpha == 0.0 or not self.vectors:
            return self
        try:
            param = next(self.model.parameters())
            device, dtype = param.device, param.dtype
        except StopIteration:  # pragma: no cover
            device = torch.device("cpu")
            dtype  = torch.float32

        for layer_id, v in self.vectors.items():
            t = torch.from_numpy(np.asarray(v, dtype=np.float32)).to(device=device, dtype=dtype)
            self._tensors[layer_id] = t
            layer = self.model.llm.layers[layer_id]
            self._handles.append(
                layer.register_forward_hook(self._make_hook(layer_id))
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._tensors.clear()

    # ------------------------------------------------------------------
    def _make_hook(self, layer_id: int):
        v = self._tensors[layer_id]
        a = self.alpha

        def hook(_module, _inputs, outputs):
            h, rest = _split_output(outputs)
            # h: [batch, seq, hidden]; v: [hidden] broadcasts over (batch, seq)
            return _rebuild_output(h + a * v, rest)
        return hook
