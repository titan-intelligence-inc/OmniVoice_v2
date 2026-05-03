"""High-level steering helpers built on top of HiddenCapturer / HiddenSteerer.

Workflow:

1. ``extract_layer_vectors(wrapper, audio_path, ref_text, layer_ids)``
   runs a single short generate() pass with ``audio_path`` as the
   reference and captures per-layer hidden-state means.

2. ``compute_v_emo(emotional_audio, neutral_audio, layer_ids, ...)``
   subtracts the neutral capture from the emotional one to yield a
   per-layer emotion direction.

3. ``make_clean_vectors(v_emo, v_lang)`` (Phase 4) applies language
   projection removal layer-wise.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from .hidden_hooks import HiddenCapturer, HiddenSteerer
from .projection import remove_projection_per_layer


# A short, language-neutral target text used purely to trigger a forward
# pass. We discard the output audio.
_PROBE_TEXT = "This is a probe."


def extract_layer_vectors(
    wrapper,
    audio_path: str | Path,
    layer_ids: list[int],
    ref_text: str | None = None,
    probe_text: str = _PROBE_TEXT,
    probe_language: str = "English",
    num_step: int = 4,
    seed: int | None = 0,
) -> dict[int, np.ndarray]:
    """Capture mean hidden states at ``layer_ids`` for ``audio_path``.

    The function runs a minimal generate pass to drive the LLM through
    its forward; we keep ``num_step`` low for speed (we only need the
    hiddens, not the audio).

    Determinism: pass ``seed`` to fix torch's RNG so two captures over
    the same audio give the same hiddens.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = wrapper.model
    if ref_text is None:
        ref_text = wrapper.transcribe(audio_path, language=None)

    with HiddenCapturer(model, layer_ids=layer_ids, first_only=False) as cap:
        # Drive the LLM once. We keep num_step small but still > 0 so
        # diffusion runs at least a few times.
        _ = model.generate(
            text=probe_text,
            language=probe_language,
            ref_audio=str(audio_path),
            ref_text=ref_text,
            num_step=num_step,
        )
    return cap.means


def compute_v_emo(
    wrapper,
    emotional_audio: str | Path,
    neutral_audio: str | Path,
    layer_ids: list[int],
    *,
    emotional_text: str | None = None,
    neutral_text: str | None = None,
    seed: int | None = 0,
    num_step: int = 4,
) -> dict[int, np.ndarray]:
    """Per-layer ``v_emo = mean_hidden(emotional) - mean_hidden(neutral)``."""
    h_emo = extract_layer_vectors(
        wrapper, emotional_audio, layer_ids,
        ref_text=emotional_text, seed=seed, num_step=num_step,
    )
    h_neu = extract_layer_vectors(
        wrapper, neutral_audio,   layer_ids,
        ref_text=neutral_text,   seed=seed, num_step=num_step,
    )
    return {i: (h_emo[i] - h_neu[i]).astype(np.float32) for i in layer_ids}


def compute_v_lang(
    wrapper,
    speaker_l1_audio: str | Path,
    speaker_l2_audio: str | Path,
    layer_ids: list[int],
    *,
    l1_text: str | None = None,
    l2_text: str | None = None,
    seed: int | None = 0,
    num_step: int = 4,
) -> dict[int, np.ndarray]:
    """Per-layer ``v_lang = mean_hidden(L1) - mean_hidden(L2)`` (Phase 4 prep)."""
    h_l1 = extract_layer_vectors(
        wrapper, speaker_l1_audio, layer_ids,
        ref_text=l1_text, seed=seed, num_step=num_step,
    )
    h_l2 = extract_layer_vectors(
        wrapper, speaker_l2_audio, layer_ids,
        ref_text=l2_text, seed=seed, num_step=num_step,
    )
    return {i: (h_l1[i] - h_l2[i]).astype(np.float32) for i in layer_ids}


def make_clean_vectors(
    v_emo: dict[int, np.ndarray],
    v_lang: dict[int, np.ndarray] | None,
    projection_removal: bool = True,
) -> dict[int, np.ndarray]:
    """Optionally strip ``v_lang`` direction from ``v_emo`` per layer."""
    if not projection_removal or v_lang is None:
        return dict(v_emo)
    return remove_projection_per_layer(v_emo, v_lang)


def save_vectors(path: str | Path, vectors: dict[int, np.ndarray], meta: dict | None = None):
    """Save a per-layer vector dict as a single .npz file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"layer_{i}": v.astype(np.float32) for i, v in vectors.items()}
    if meta:
        # Persist meta as a JSON string in the npz
        import json
        arrays["__meta__"] = np.array(json.dumps(meta), dtype=object)
    np.savez(p, **arrays)
    return p


def load_vectors(path: str | Path) -> tuple[dict[int, np.ndarray], dict | None]:
    """Inverse of save_vectors. Returns (vectors, meta)."""
    p = Path(path)
    data = np.load(p, allow_pickle=True)
    out = {}
    meta = None
    for k in data.files:
        if k == "__meta__":
            import json
            try:
                meta = json.loads(str(data[k]))
            except Exception:
                meta = None
        elif k.startswith("layer_"):
            out[int(k.split("_", 1)[1])] = np.asarray(data[k]).astype(np.float32)
    return out, meta
