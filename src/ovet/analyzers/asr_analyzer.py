"""Whisper-based ASR for ref_text auto-transcription and CER on output.

Note: torchcodec was incompatible with torch 2.8 in this venv. We bypass
it by feeding raw numpy arrays to the transformers ``pipeline``, which
goes through soundfile + librosa rather than torchcodec.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from ..utils.io import load_wav

_DEFAULT_MODEL = "openai/whisper-large-v3-turbo"


def _normalize_text(s: str) -> str:
    """Cheap normalization for CER: lowercase, strip ws/punct."""
    import re
    s = s.lower()
    s = re.sub(r"[\s\.,!?。、！？・…]+", "", s)
    return s.strip()


def _cer(ref: str, hyp: str) -> float:
    """Character error rate (Levenshtein / len(ref)). 0.0..>1.0."""
    ref_n = _normalize_text(ref)
    hyp_n = _normalize_text(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    # Levenshtein distance
    m, n = len(ref_n), len(hyp_n)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if ref_n[i - 1] == hyp_n[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n] / m


class ASRAnalyzer:
    """Whisper wrapper. Lazy-load on first transcribe call."""

    _LANG_HINT = {
        "Japanese": "japanese",
        "English":  "english",
        "Chinese":  "chinese",
        "ja": "japanese", "en": "english", "zh": "chinese",
    }

    def __init__(self, model_id: str = _DEFAULT_MODEL, device: str = "cuda"):
        self.model_id = model_id
        self.device   = device if torch.cuda.is_available() else "cpu"
        self._pipe = None

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        from transformers import pipeline
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=self.device,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

    def transcribe(self, wav_path: str | Path, language: str | None = None) -> str:
        self._ensure_loaded()
        wav, _ = load_wav(wav_path, target_sr=16000)
        lang = self._LANG_HINT.get(language) if language else None
        kwargs = {}
        if lang:
            kwargs["generate_kwargs"] = {"language": lang}
        res = self._pipe(
            {"array": wav.astype(np.float32), "sampling_rate": 16000},
            **kwargs,
        )
        return res["text"].strip()

    def content_error(
        self,
        wav_path: str | Path,
        reference_text: str,
        language: str | None = None,
    ) -> float:
        """CER between hypothesis and reference. Returns float in [0, ~1+]."""
        hyp = self.transcribe(wav_path, language=language)
        return _cer(reference_text, hyp)
