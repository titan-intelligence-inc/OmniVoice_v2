"""Audio I/O and resampling helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa


def load_wav(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load mono float32 wav. Optionally resample to target_sr."""
    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if target_sr is not None and sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return wav, sr


def save_wav(path: str | Path, wav: np.ndarray, sr: int) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), wav, sr)
    return p


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    """Cosine similarity between two 1-D vectors. Returns float."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))
