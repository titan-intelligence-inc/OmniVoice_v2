"""OmniVoiceWrapper: voice cloning with optional ref_text auto-transcription.

Currently supports Phase 1 (no steering). Phase 3+ will extend this
class to attach forward hooks to ``model.llm.layers[i]``.
"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import torch


@dataclass
class SteeringConfig:
    """Configuration for Phase 3+ activation steering. Phase 1 leaves enabled=False.

    The composed per-layer steering vector applied during generation is::

        delta_layer = alpha * v_emo_clean[layer]
                    - language_alpha * v_lang[layer]

    where ``v_emo_clean = v_emo - proj(v_emo, v_lang)`` if
    ``projection_removal=True``. ``language_alpha > 0`` pushes hidden
    state *away* from the v_lang direction (e.g. away from
    Japanese-acoustic-features toward target-language-acoustic-features
    when v_lang = h(JP) − h(target)).

    Quality-tuning knobs (Phase 4+):
      * ``step_window`` — apply only on diffusion steps
        ``[start, end)``. Useful to keep prosody refinement uncorrupted.
        ``None`` = apply on every step.
      * ``norm_clip_factor`` — cap ``|alpha * v_layer|`` to this fraction
        of the hidden state's mean per-position L2 norm. Prevents
        distribution-shift artefacts at high alpha. ``None`` = no clip.
    """
    enabled: bool = False
    alpha: float = 0.0
    layer_ids: list[int] = field(default_factory=list)
    emotion_vector:   dict[int, np.ndarray] | None = None
    language_vector:  dict[int, np.ndarray] | None = None
    projection_removal: bool = True
    language_alpha:   float = 0.0   # accent-removal weight (0=disabled)
    step_window:      tuple[int, int | None] | None = None
    norm_clip_factor: float | None = None
    position_mask:    bool = False    # apply delta only at audio token positions


class OmniVoiceWrapper:
    """Thin wrapper around k2-fsa/OmniVoice with our pipeline conventions."""

    SAMPLING_RATE = 24000

    def __init__(
        self,
        model_id: str = "k2-fsa/OmniVoice",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        hf_home: str | None = None,
    ):
        if hf_home:
            os.environ.setdefault("HF_HOME", hf_home)
        from omnivoice import OmniVoice  # type: ignore
        self.model = OmniVoice.from_pretrained(model_id, device_map=device, dtype=dtype)
        self.model_id = model_id
        self.device   = device
        # Lazy-load ASR (we maintain our own due to torchcodec issues).
        self._asr = None

    # ------------------------------------------------------------------
    # Architecture introspection (Phase 3 prep)
    # ------------------------------------------------------------------
    @property
    def llm_num_layers(self) -> int:
        return int(self.model.llm.config.num_hidden_layers)

    @property
    def llm_hidden_size(self) -> int:
        return int(self.model.llm.config.hidden_size)

    def get_layer(self, idx: int):
        return self.model.llm.layers[idx]

    # ------------------------------------------------------------------
    # ref_text auto-transcription
    # ------------------------------------------------------------------
    def _get_asr(self):
        if self._asr is None:
            from ..analyzers.asr_analyzer import ASRAnalyzer
            self._asr = ASRAnalyzer(device=self.device.split(":")[0])
        return self._asr

    def transcribe(self, wav_path: str | Path, language: str | None = None) -> str:
        return self._get_asr().transcribe(wav_path, language=language)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        text: str,
        language: str,
        ref_audio: str | Path,
        ref_text: str | None = None,
        instruct: str | None = None,
        steering: SteeringConfig | None = None,
    ) -> np.ndarray:
        """Run voice cloning and return mono float32 waveform @ 24kHz.

        If ``ref_text`` is None, Whisper is used to auto-transcribe.
        If ``steering`` is enabled, Qwen3 forward hooks add ``alpha *
        v_emo[layer]`` to each target layer's output during generation.
        """
        if ref_text is None:
            ref_text = self.transcribe(ref_audio, language=language)

        kwargs = dict(
            text=text,
            language=language,
            ref_audio=str(ref_audio),
            ref_text=ref_text,
        )
        if instruct:
            kwargs["instruct"] = instruct

        # Steering off: short-circuit when nothing to do.
        no_emo  = (steering is None or not steering.enabled or steering.alpha == 0.0)
        no_lang = (steering is None or steering.language_alpha == 0.0
                   or not steering.language_vector)
        if no_emo and no_lang:
            audios = self.model.generate(**kwargs)
            return audios[0]

        # ------------------------------------------------------------------
        # Activation steering: install hooks for the duration of generation.
        # Compose the per-layer delta from emotion + language components.
        # ------------------------------------------------------------------
        from .hidden_hooks import HiddenSteerer
        from .projection import remove_projection_per_layer

        layer_ids: list[int] = list(steering.layer_ids) if steering else []
        if not layer_ids:
            # Fall back to whatever layers any vector dict provides.
            if steering and steering.emotion_vector:
                layer_ids = list(steering.emotion_vector.keys())
            elif steering and steering.language_vector:
                layer_ids = list(steering.language_vector.keys())

        # Emotion side
        emo_vecs: dict[int, np.ndarray] = {}
        if not no_emo and steering.emotion_vector is not None:
            emo_vecs = {
                i: steering.emotion_vector[i]
                for i in layer_ids if i in steering.emotion_vector
            }
            if steering.projection_removal and steering.language_vector:
                v_lang_subset = {
                    i: steering.language_vector[i]
                    for i in emo_vecs if i in steering.language_vector
                }
                if v_lang_subset:
                    emo_vecs = remove_projection_per_layer(emo_vecs, v_lang_subset)

        # Language side (push away from v_lang → −language_alpha * v_lang)
        lang_vecs: dict[int, np.ndarray] = {}
        if not no_lang:
            lang_vecs = {
                i: steering.language_vector[i]
                for i in layer_ids if i in steering.language_vector
            }

        # Compose final delta per layer.
        a_emo  = steering.alpha          if not no_emo  else 0.0
        a_lang = steering.language_alpha if not no_lang else 0.0
        composed: dict[int, np.ndarray] = {}
        for i in set(emo_vecs) | set(lang_vecs):
            v = np.zeros_like(emo_vecs[i] if i in emo_vecs else lang_vecs[i])
            if i in emo_vecs:
                v = v + a_emo * emo_vecs[i]
            if i in lang_vecs:
                v = v - a_lang * lang_vecs[i]   # subtract → away from L1
            composed[i] = v.astype(np.float32, copy=False)

        # HiddenSteerer multiplies its `alpha` by `vectors` — we already baked
        # both magnitudes into `composed`, so use alpha=1.0.
        with HiddenSteerer(
            self.model, alpha=1.0, vectors=composed,
            step_window=steering.step_window,
            norm_clip_factor=steering.norm_clip_factor,
            position_mask=steering.position_mask,
        ):
            audios = self.model.generate(**kwargs)
        return audios[0]

    def generate_batch(
        self,
        texts: list[str],
        language: str,
        ref_audio: str | Path,
        ref_text: str | None = None,
    ) -> list[np.ndarray]:
        """Native-batched cloning: same ref for all items, varying text.

        Uses OmniVoice's list-form ``generate()`` to share KV cache /
        feature extraction work where possible. No steering path —
        steering hooks would need per-batch composition which the
        upstream model does not currently support.
        """
        if ref_text is None:
            ref_text = self.transcribe(ref_audio, language=language)
        n = len(texts)
        audios = self.model.generate(
            text=texts,
            language=[language] * n,
            ref_audio=[str(ref_audio)] * n,
            ref_text=[ref_text] * n,
        )
        return list(audios)
