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
    """Configuration for Phase 3+ activation steering. Phase 1 leaves enabled=False."""
    enabled: bool = False
    alpha: float = 0.0
    layer_ids: list[int] = field(default_factory=list)
    emotion_vector:   dict[int, np.ndarray] | None = None
    language_vector:  dict[int, np.ndarray] | None = None
    projection_removal: bool = True


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

        if steering is None or not steering.enabled or steering.alpha == 0.0:
            audios = self.model.generate(**kwargs)
            return audios[0]

        # ------------------------------------------------------------------
        # Activation steering: install hooks for the duration of generation.
        # If v_lang is provided and projection_removal is True, strip the
        # language direction from v_emo per layer (Phase 4b).
        # ------------------------------------------------------------------
        from .hidden_hooks import HiddenSteerer
        from .projection import remove_projection_per_layer

        if steering.emotion_vector is None:
            raise ValueError("SteeringConfig.enabled but emotion_vector is None")
        layer_ids = list(steering.layer_ids) or list(steering.emotion_vector.keys())
        vectors = {
            i: steering.emotion_vector[i]
            for i in layer_ids if i in steering.emotion_vector
        }

        if steering.projection_removal and steering.language_vector:
            v_lang_subset = {
                i: steering.language_vector[i]
                for i in vectors if i in steering.language_vector
            }
            if v_lang_subset:
                vectors = remove_projection_per_layer(vectors, v_lang_subset)

        with HiddenSteerer(self.model, alpha=steering.alpha, vectors=vectors):
            audios = self.model.generate(**kwargs)
        return audios[0]
