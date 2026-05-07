"""Language grafting hook: substitute the language component of the
hidden state with a target-language signature extracted from a native
reference. (User-proposed Phase 6 direction.)

Operation per layer L::

    h' = h - α · proj(h, Q_L) + β · lang_target_L

where:
  * ``Q_L``: orthonormal basis of the language subspace at layer L,
    fitted from labeled probe data (see ``compute_lang_subspace``).
  * ``lang_target_L``: the language component of a *native* target-
    language reference, projected onto Q_L.
  * ``α``: how much of the current language signal to remove.
  * ``β``: how much native-target signal to inject.

This is a runtime-dependent operation (depends on incoming h), so it
gets its own hook class rather than reusing the additive
``HiddenSteerer``.

Disentanglement was empirically validated at L8/L12 on a 48-clone
4-speaker × 4-language dataset (see
``scripts/disentanglement_validation.py``):

  * removing the lang subspace drops lang_acc 0.85→0.00 while spk_acc
    stays 0.94 and emo_acc stays 0.96-0.98.
  * substituting another speaker's same-lang component preserves
    spk_acc (=0.94) and emo_acc (=0.96).

Layers 20+ entangle lang and speaker too tightly for this op.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .hidden_hooks import _split_output, _rebuild_output


# ----------------------------------------------------------------------
# Subspace + target construction utilities (offline, numpy)
# ----------------------------------------------------------------------

def compute_lang_subspace(
    H: np.ndarray, langs: Sequence[str],
) -> np.ndarray:
    """Return orthonormal basis ``Q ∈ R^{D × K}`` for the language subspace.

    K = number of distinct languages. ``Q``'s columns span the
    centered class-mean directions (each language's deviation from the
    overall centroid). For the projection operations this is what we
    need — the rank-(K-1) effective subspace lives inside this span.

    Args:
        H:     [N, D] hidden vectors.
        langs: length-N language labels.

    Returns:
        Q: [D, K] orthonormal columns.
    """
    H = np.asarray(H, dtype=np.float32)
    unique = sorted(set(langs))
    means  = np.stack([H[np.array(langs) == l].mean(axis=0) for l in unique])
    overall = H.mean(axis=0, keepdims=True)
    M = (means - overall).astype(np.float32)              # [K, D]
    Q, _ = np.linalg.qr(M.T)                               # [D, K]
    return Q.astype(np.float32)


def project_onto_subspace(
    H: np.ndarray, Q: np.ndarray,
) -> np.ndarray:
    """Project rows of H onto col-space of Q. Returns same shape as H."""
    coef = H @ Q
    return coef @ Q.T


def compute_lang_target_from_hiddens(
    H_native: np.ndarray, Q: np.ndarray,
) -> np.ndarray:
    """Extract the language component from a native-target hidden mean.

    Returns a 1-D vector ``[D]`` that lives in col-space of Q.
    """
    if H_native.ndim == 1:
        H_native = H_native[None, :]
    proj = project_onto_subspace(H_native, Q)
    return proj.mean(axis=0).astype(np.float32)


# ----------------------------------------------------------------------
# Hook: LanguageGrafter
# ----------------------------------------------------------------------

class LanguageGrafter:
    """Forward-hook context manager that grafts a target-language
    component onto each chosen layer's hidden state.

    Args:
        model:                the OmniVoice ``model`` whose
                              ``llm.layers[i]`` we hook.
        subspace_per_layer:   ``{layer_id: Q [D, K]}`` orthonormal.
        target_per_layer:     ``{layer_id: lang_target [D]}``.
        remove_alpha:         α  (default 1.0 — fully remove old lang).
        inject_beta:          β  (default 1.0 — fully inject native lang).
        step_window, position_mask: same conventions as
                              ``HiddenSteerer``.
    """

    def __init__(
        self,
        model,
        subspace_per_layer: dict[int, np.ndarray],
        target_per_layer:   dict[int, np.ndarray],
        *,
        remove_alpha: float = 1.0,
        inject_beta:  float = 1.0,
        step_window: tuple[int, int | None] | None = None,
        position_mask: bool = False,
    ):
        self.model = model
        self.subspace_per_layer = subspace_per_layer
        self.target_per_layer   = target_per_layer
        self.remove_alpha = float(remove_alpha)
        self.inject_beta  = float(inject_beta)
        self.step_window  = step_window
        self.position_mask = bool(position_mask)

        self._handles = []
        self._Q_t:      dict[int, torch.Tensor] = {}
        self._target_t: dict[int, torch.Tensor] = {}
        self._original_forward = None

        layer_ids = sorted(set(subspace_per_layer.keys()) & set(target_per_layer.keys()))
        if not layer_ids:
            raise ValueError(
                "subspace_per_layer and target_per_layer share no layer ids. "
                f"subspace={list(subspace_per_layer.keys())} "
                f"target={list(target_per_layer.keys())}"
            )
        self._layer_ids        = layer_ids
        self._first_layer_id   = layer_ids[0]
        self._last_layer_id    = layer_ids[-1]
        self._step_counter     = 0

    # ------------------------------------------------------------------
    def __enter__(self):
        if self.remove_alpha == 0.0 and self.inject_beta == 0.0:
            return self
        try:
            param = next(self.model.parameters())
            device, dtype = param.device, param.dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype  = torch.float32

        for layer_id in self._layer_ids:
            Q = np.asarray(self.subspace_per_layer[layer_id], dtype=np.float32)
            t = np.asarray(self.target_per_layer[layer_id], dtype=np.float32)
            self._Q_t[layer_id]      = torch.from_numpy(Q).to(device=device, dtype=dtype)
            self._target_t[layer_id] = torch.from_numpy(t).to(device=device, dtype=dtype)
            layer = self.model.llm.layers[layer_id]
            self._handles.append(
                layer.register_forward_hook(self._make_hook(layer_id))
            )
        self._step_counter = 0

        if self.position_mask:
            self._original_forward = self.model.forward
            grafter = self
            orig = self._original_forward

            def _wrapped(*args, **kwargs):
                am = kwargs.get("audio_mask")
                if am is None and len(args) > 1:
                    am = args[1]
                grafter.model._ovet_audio_mask = am
                return orig(*args, **kwargs)

            self.model.forward = _wrapped

        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._Q_t.clear()
        self._target_t.clear()
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

    def _make_hook(self, layer_id: int):
        Q      = self._Q_t[layer_id]                # [D, K]
        target = self._target_t[layer_id]           # [D]
        a      = self.remove_alpha
        b      = self.inject_beta

        def hook(_module, _inputs, outputs):
            apply = self._step_in_window()
            if not apply:
                if layer_id == self._last_layer_id:
                    self._step_counter += 1
                return outputs

            h, rest = _split_output(outputs)         # [B, S, D]
            # Project h onto subspace: coef = h @ Q  -> [B, S, K]
            # lang_component = coef @ Q.T            -> [B, S, D]
            coef = torch.matmul(h, Q)
            lang_component = torch.matmul(coef, Q.transpose(0, 1))

            # Build delta: -α·lang_component + β·target  (broadcast target)
            target_b = target.view(1, 1, -1)         # broadcast over [B, S]
            delta = -a * lang_component + b * target_b

            if self.position_mask:
                am = getattr(self.model, "_ovet_audio_mask", None)
                if am is not None:
                    mask = am.to(device=h.device, dtype=h.dtype).unsqueeze(-1)
                    new_h = h + delta * mask
                else:
                    new_h = h + delta
            else:
                new_h = h + delta

            if layer_id == self._last_layer_id:
                self._step_counter += 1
            return _rebuild_output(new_h, rest)
        return hook


# ----------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------

def save_graft_artifacts(
    path: Path,
    subspace_per_layer: dict[int, np.ndarray],
    target_per_layer:   dict[int, np.ndarray],
    *,
    meta: dict | None = None,
) -> None:
    """Save subspace + target as a single .npz with structured keys.

    Keys:  Q_<layer>   (D, K)
           T_<layer>   (D,)
           layers      (np int array)
           __meta__    (object array of meta dict, if provided)
    """
    payload: dict = {}
    layer_ids = sorted(set(subspace_per_layer.keys()) & set(target_per_layer.keys()))
    for L in layer_ids:
        payload[f"Q_{L}"] = np.asarray(subspace_per_layer[L], dtype=np.float32)
        payload[f"T_{L}"] = np.asarray(target_per_layer[L], dtype=np.float32)
    payload["layers"] = np.asarray(layer_ids, dtype=np.int32)
    if meta is not None:
        payload["__meta__"] = np.array([meta], dtype=object)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def load_graft_artifacts(
    path: Path,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict | None]:
    """Inverse of ``save_graft_artifacts``. Returns
    ``(subspace_per_layer, target_per_layer, meta)``.
    """
    d = np.load(path, allow_pickle=True)
    layers = [int(x) for x in d["layers"]]
    Q_per = {L: np.asarray(d[f"Q_{L}"], dtype=np.float32) for L in layers}
    T_per = {L: np.asarray(d[f"T_{L}"], dtype=np.float32) for L in layers}
    meta = None
    if "__meta__" in d.files:
        meta = d["__meta__"][0] if d["__meta__"].size else None
    return Q_per, T_per, meta
