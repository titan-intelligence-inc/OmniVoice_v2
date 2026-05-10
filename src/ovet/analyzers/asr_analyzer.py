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


def _normalize_text(s: str, *, zh: bool = False) -> str:
    """Cheap normalization for CER: lowercase, strip ws/punct.

    When ``zh=True``, also fold Traditional Chinese to Simplified via
    ``zhconv``. Use this for Mandarin evaluation: Whisper sometimes
    emits Traditional characters even when the audio is Mandarin /
    written reference is Simplified, inflating CER spuriously
    (e.g. 东 vs 東 or 岛 vs 島).
    """
    import re
    s = s.lower()
    s = re.sub(r"[\s\.,!?。、！？・…“”‘’\"'(){}\[\]\(\)（）「」『』]+", "", s)
    if zh:
        try:
            import zhconv
            s = zhconv.convert(s, "zh-cn")
        except Exception:
            pass
    return s.strip()


def _detect_hallucination(s: str, *, max_cycle: int = 4, min_repeats: int = 5) -> bool:
    """True if ``s`` contains a Whisper-style repetition loop.

    Catches both single-char runs (``"娃娃娃娃娃..."``) and short-cycle
    repetitions (``"abcabcabc..."``). Run-length cutoff is
    ``cycle_len * min_repeats`` characters.
    """
    if not s:
        return False
    n = len(s)
    for cycle_len in range(1, max_cycle + 1):
        if n < cycle_len * min_repeats:
            continue
        # Slide; only need to check positions that allow min_repeats fits
        for start in range(n - cycle_len * min_repeats + 1):
            chunk = s[start:start + cycle_len]
            ok = True
            for k in range(1, min_repeats):
                if s[start + k * cycle_len : start + (k + 1) * cycle_len] != chunk:
                    ok = False
                    break
            if ok:
                return True
    return False


def _cer(ref: str, hyp: str, *, cap: float | None = 2.0,
         zh: bool = False) -> float:
    """Character error rate (Levenshtein / len(ref)).

    If ``hyp`` contains a Whisper repetition loop or CER would exceed
    ``cap``, the returned value is clipped at ``cap``. This prevents a
    single hallucinated rep from dominating downstream means.

    When ``zh=True``, both ref and hyp are Simplified-folded before
    comparison (use this for Mandarin to avoid Trad/Simp inflation).

    Set ``cap=None`` to disable capping (returns raw CER).
    """
    ref_n = _normalize_text(ref, zh=zh)
    hyp_n = _normalize_text(hyp, zh=zh)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    if _detect_hallucination(hyp_n) and cap is not None:
        return cap
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
    raw = dp[n] / m
    if cap is not None and raw > cap:
        return cap
    return raw


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

    def transcribe_batch(
        self,
        wav_paths: list[str | Path],
        language: str | None = None,
        *,
        batch_size: int = 8,
    ) -> list[str]:
        """Batched Whisper transcription via HF pipeline.

        Loads each wav to 16 kHz numpy and feeds the whole list to the
        pipeline with ``batch_size``. The pipeline pads internally and
        runs a single forward over the chunk-grouped batch.
        """
        self._ensure_loaded()
        lang = self._LANG_HINT.get(language) if language else None
        kwargs = {}
        if lang:
            kwargs["generate_kwargs"] = {"language": lang}
        inputs = []
        for p in wav_paths:
            w, _ = load_wav(p, target_sr=16000)
            inputs.append({"array": w.astype(np.float32), "sampling_rate": 16000})
        results = self._pipe(inputs, batch_size=batch_size, **kwargs)
        return [r["text"].strip() for r in results]

    def content_error(
        self,
        wav_path: str | Path,
        reference_text: str,
        language: str | None = None,
        *,
        zh_normalize: bool | None = None,
    ) -> float:
        """CER between hypothesis and reference. Returns float in [0, ~1+].

        For Mandarin (``language`` resolves to ``"chinese"``),
        ``zh_normalize`` defaults to True — fold Trad/Simp before
        comparison, since Whisper-large-v3-turbo often emits Traditional
        characters even when the reference text is Simplified, which
        otherwise inflates CER spuriously.
        """
        hyp = self.transcribe(wav_path, language=language)
        is_zh = self._LANG_HINT.get(language) == "chinese"
        if zh_normalize is None:
            zh_normalize = is_zh
        return _cer(reference_text, hyp, zh=zh_normalize)
