"""F0 statistics transfer for emotion conditioning.

Pre-conditions a content-donor audio with the F0 statistics of an
emotional reference, so that downstream VC (SeedVC etc.) sees an
emotional source rather than a neutral one.

Operation
---------
  WORLD-decompose content_donor into (f0, sp, ap).
  Compute statistics (mean, std) of the voiced F0 from both the
  content donor and the emotional reference.
  Renormalize the content donor's voiced F0 to match the emotional
  reference's distribution::

      f0_new = (f0 - mean_content) / std_content * std_emo + mean_emo

  Recompose the audio with WORLD using ``(f0_new, sp_content,
  ap_content)``. Spectral envelope (= phoneme identity) is untouched;
  only the pitch contour is rescaled to the emotional range.

This is a *statistics* transfer, not a contour-shape transfer — the
content donor's voicing pattern is preserved (so timing/syllabification
stays correct), only its pitch range and variability are rescaled.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class F0Stats:
    """Per-utterance pitch statistics."""
    mean_hz:    float
    std_hz:     float
    median_hz:  float
    voiced_frac: float


def compute_f0_stats(
    audio: np.ndarray, sr: int, *,
    f0_floor: float = 70.0, f0_ceil: float = 600.0,
    frame_period_ms: float = 5.0,
) -> F0Stats:
    """Extract voiced F0 statistics via WORLD harvest+stonemask."""
    import pyworld as pw
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    f0, t = pw.harvest(
        audio, sr, f0_floor=f0_floor, f0_ceil=f0_ceil,
        frame_period=frame_period_ms,
    )
    f0 = pw.stonemask(audio, f0, t, sr)
    voiced = f0[f0 > 0]
    if len(voiced) == 0:
        return F0Stats(mean_hz=0.0, std_hz=0.0, median_hz=0.0, voiced_frac=0.0)
    return F0Stats(
        mean_hz   = float(voiced.mean()),
        std_hz    = float(voiced.std() + 1e-6),
        median_hz = float(np.median(voiced)),
        voiced_frac = float((f0 > 0).mean()),
    )


def transfer_f0_emotion(
    content_audio: np.ndarray, content_sr: int,
    emotion_audio: np.ndarray, emotion_sr: int,
    *,
    blend: float = 1.0,
    f0_floor: float = 70.0, f0_ceil: float = 600.0,
    frame_period_ms: float = 5.0,
) -> np.ndarray:
    """Return an audio identical in content/timbre to ``content_audio``
    but with its F0 distribution rescaled to ``emotion_audio``'s
    statistics.

    Args:
        content_audio:  the audio whose phonemes / voice timbre we keep.
        content_sr:     its sample rate.
        emotion_audio:  the emotional reference.
        emotion_sr:     its sample rate.
        blend:          0.0 = no transfer (pass-through), 1.0 = full
                        renormalisation toward emotion_audio's stats.
                        Intermediate values do partial blending.

    Returns:
        ndarray at ``content_sr`` (the WORLD reconstruction may differ
        slightly in length due to frame quantisation; we trim/pad to
        match the input length).
    """
    import pyworld as pw

    content_audio = np.asarray(content_audio, dtype=np.float64)
    if content_audio.ndim > 1:
        content_audio = content_audio.mean(axis=1)
    emotion_audio = np.asarray(emotion_audio, dtype=np.float64)
    if emotion_audio.ndim > 1:
        emotion_audio = emotion_audio.mean(axis=1)

    # Emotion stats
    emo_stats = compute_f0_stats(
        emotion_audio, emotion_sr,
        f0_floor=f0_floor, f0_ceil=f0_ceil,
        frame_period_ms=frame_period_ms,
    )

    # Content WORLD decomposition
    f0_c, t_c = pw.harvest(content_audio, content_sr,
                           f0_floor=f0_floor, f0_ceil=f0_ceil,
                           frame_period=frame_period_ms)
    f0_c = pw.stonemask(content_audio, f0_c, t_c, content_sr)
    sp_c = pw.cheaptrick(content_audio, f0_c, t_c, content_sr)
    ap_c = pw.d4c(content_audio, f0_c, t_c, content_sr)

    voiced_mask = f0_c > 0
    if voiced_mask.any() and emo_stats.std_hz > 0:
        voiced_c = f0_c[voiced_mask]
        mean_c = float(voiced_c.mean())
        std_c  = float(voiced_c.std() + 1e-6)
        # Z-normalise to content stats, rescale to emotion stats
        renorm = (voiced_c - mean_c) / std_c * emo_stats.std_hz + emo_stats.mean_hz
        # Clip to reasonable range so WORLD doesn't blow up
        renorm = np.clip(renorm, f0_floor, f0_ceil)
        # Blend
        f0_new = f0_c.copy()
        f0_new[voiced_mask] = (1.0 - blend) * f0_c[voiced_mask] + blend * renorm

        out = pw.synthesize(f0_new, sp_c, ap_c, content_sr,
                            frame_period=frame_period_ms)
    else:
        out = pw.synthesize(f0_c, sp_c, ap_c, content_sr,
                            frame_period=frame_period_ms)

    out = out.astype(np.float32)
    # Trim/pad to original length so downstream length-aware ops behave.
    n = len(content_audio)
    if len(out) < n:
        out = np.pad(out, (0, n - len(out)))
    elif len(out) > n:
        out = out[:n]
    return out


def transfer_f0_emotion_from_paths(
    content_path: Path, emotion_path: Path, *,
    blend: float = 1.0, f0_ceil: float = 600.0,
) -> tuple[np.ndarray, int]:
    """File-based wrapper: returns ``(audio, sr)`` matching content_path's sr."""
    import soundfile as sf
    content, sr_c = sf.read(content_path)
    emotion, sr_e = sf.read(emotion_path)
    out = transfer_f0_emotion(content, sr_c, emotion, sr_e,
                              blend=blend, f0_ceil=f0_ceil)
    return out, sr_c
