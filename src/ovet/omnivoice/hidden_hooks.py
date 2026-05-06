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

    Optional refinements
    --------------------
    * ``step_window=(start, end)`` — only apply steering when the diffusion
      step (counted internally by counting forward calls of the smallest-id
      hooked layer) is in ``[start, end)``. ``end=None`` means "open".
    * ``norm_clip_factor=β`` — cap the per-layer ``alpha * v`` magnitude at
      ``β * mean(|h_t|)`` per call. Prevents distribution-shift artefacts at
      high alpha while keeping the same direction.

    Usage::

        with HiddenSteerer(model, alpha=0.4, vectors={8: v8, 12: v12},
                           step_window=(0, 8), norm_clip_factor=0.3):
            audio = wrapper.generate(...)
    """

    def __init__(
        self,
        model,
        alpha: float,
        vectors: dict[int, np.ndarray],
        *,
        step_window: tuple[int, int | None] | None = None,
        norm_clip_factor: float | None = None,
        position_mask: bool = False,
    ):
        self.model   = model
        self.alpha   = float(alpha)
        self.vectors = vectors
        self.step_window      = step_window
        self.norm_clip_factor = norm_clip_factor
        self.position_mask    = bool(position_mask)

        self._handles = []
        self._tensors: dict[int, torch.Tensor] = {}
        self._original_forward = None      # for monkey-patching when position_mask=True

        sorted_ids = sorted(self.vectors.keys()) if self.vectors else []
        # We tick the step counter once per LLM forward — increment after the
        # *last* hooked layer fires so all hooks within a step see the same value.
        self._first_layer_id = sorted_ids[0] if sorted_ids else None
        self._last_layer_id  = sorted_ids[-1] if sorted_ids else None
        self._step_counter   = 0

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
        self._step_counter = 0

        if self.position_mask:
            # Capture audio_mask via a wrapped forward — hooks read it from
            # ``model._ovet_audio_mask``.
            self._original_forward = self.model.forward
            steerer = self
            orig = self._original_forward

            def _wrapped(*args, **kwargs):
                am = kwargs.get("audio_mask")
                if am is None and len(args) > 1:
                    am = args[1]
                steerer.model._ovet_audio_mask = am
                return orig(*args, **kwargs)

            self.model.forward = _wrapped

        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._tensors.clear()
        if self._original_forward is not None:
            self.model.forward = self._original_forward
            self._original_forward = None
        if hasattr(self.model, "_ovet_audio_mask"):
            del self.model._ovet_audio_mask

    # ------------------------------------------------------------------
    def _step_in_window(self) -> bool:
        if self.step_window is None:
            return True
        start, end = self.step_window
        s = self._step_counter
        if s < start:
            return False
        if end is not None and s >= end:
            return False
        return True

    def _maybe_clip_delta(self, delta: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Cap ``|delta| <= norm_clip_factor * mean(|h_t|)`` if requested."""
        if self.norm_clip_factor is None:
            return delta
        # h shape [batch, seq, hidden]; per-position L2 then mean
        h_norm = h.detach().to(torch.float32).norm(dim=-1).mean().item()
        cap    = float(self.norm_clip_factor) * h_norm
        d_norm = float(torch.norm(delta).item())
        if d_norm <= 1e-12 or d_norm <= cap:
            return delta
        return delta * (cap / d_norm)

    def _make_hook(self, layer_id: int):
        v = self._tensors[layer_id]
        a = self.alpha

        def hook(_module, _inputs, outputs):
            apply = self._step_in_window()

            if not apply:
                if layer_id == self._last_layer_id:
                    self._step_counter += 1
                return outputs

            h, rest = _split_output(outputs)
            delta = a * v
            delta = self._maybe_clip_delta(delta, h)

            if self.position_mask:
                am = getattr(self.model, "_ovet_audio_mask", None)
                if am is not None:
                    # am: [batch, seq] bool/int → match h's batch+seq shape
                    mask = am.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
                    new_h = h + delta.view(1, 1, -1) * mask
                else:
                    new_h = h + delta
            else:
                new_h = h + delta

            if layer_id == self._last_layer_id:
                self._step_counter += 1
            return _rebuild_output(new_h, rest)
        return hook
