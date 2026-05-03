"""Prosodic feature extraction via librosa.

Reproduces the formulas used in baseline v3, surfaced as a class.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import librosa

from ..types import ProsodyFeatures
from ..utils.io import load_wav


class ProsodyAnalyzer:
    def __init__(
        self,
        target_sr: int = 16000,
        f0_min:    int = 50,
        f0_max:    int = 500,
    ):
        self.target_sr = target_sr
        self.f0_min    = f0_min
        self.f0_max    = f0_max

    def analyze(self, wav_path: str | Path) -> ProsodyFeatures:
        wav, sr = load_wav(wav_path, target_sr=self.target_sr)
        return self._compute(wav, sr)

    def _compute(self, wav: np.ndarray, sr: int) -> ProsodyFeatures:
        f0 = librosa.yin(wav, fmin=self.f0_min, fmax=self.f0_max, sr=sr)
        voiced = f0[~np.isnan(f0) & (f0 > self.f0_min)]
        rms = librosa.feature.rms(y=wav)[0]

        f0_mean  = float(np.mean(voiced)) if len(voiced) else 0.0
        f0_std   = float(np.std(voiced))  if len(voiced) else 0.0
        f0_range = (
            float(np.percentile(voiced, 95) - np.percentile(voiced, 5))
            if len(voiced) else 0.0
        )
        return ProsodyFeatures(
            f0_mean=f0_mean,
            f0_std=f0_std,
            f0_range=f0_range,
            energy_mean=float(np.mean(rms)),
            energy_std=float(np.std(rms)),
            duration=float(len(wav) / sr),
            voiced_ratio=float(len(voiced) / max(len(f0), 1)),
        )
