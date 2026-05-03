"""CandidateEvaluator: compute the 4-axis score against a reference audio.

Designed so analyzers can be injected (cheap to swap or stub for tests).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from ..types import (
    CandidateScores, EmotionAnalysis, VADFeatures,
    ProsodyFeatures, SpeakerFeatures,
)
from ..utils.io import cosine


_LABEL_MAP = {  # ref-side label → emotion2vec class name (out side)
    "anger":   "angry",   "angry":   "angry",
    "sad":     "sad",     "sadness": "sad",
    "fear":    "fearful", "fearful": "fearful",
    "happy":   "happy",   "happiness": "happy", "joy": "happy",
    "surprise":"surprised","surprised":"surprised",
    "disgust": "disgusted","disgusted":"disgusted",
    "neutral": "neutral",
}


def _vad_dist(a: VADFeatures, b: VADFeatures) -> float:
    return float(np.sqrt(
        (a.valence   - b.valence)   ** 2
        + (a.arousal - b.arousal)   ** 2
        + (a.dominance - b.dominance) ** 2
    ))


def _safe_ratio(num: float, denom: float, eps: float = 1e-6) -> float:
    return float(num / max(denom, eps))


class CandidateEvaluator:
    """Score one or more output candidates against a single reference."""

    def __init__(
        self,
        emotion_analyzer,
        vad_analyzer,
        prosody_analyzer,
        speaker_analyzer,
        asr_analyzer,
    ):
        self.emo = emotion_analyzer
        self.vad = vad_analyzer
        self.pros = prosody_analyzer
        self.spk = speaker_analyzer
        self.asr = asr_analyzer

    # ------------------------------------------------------------------
    def analyze_reference(self, ref_audio: str | Path) -> dict:
        """Pre-compute reference features once and reuse across candidates."""
        return {
            "emo":  self.emo.analyze(ref_audio),
            "vad":  self.vad.analyze(ref_audio),
            "pros": self.pros.analyze(ref_audio),
            "spk":  self.spk.analyze(ref_audio),
        }

    # ------------------------------------------------------------------
    def evaluate(
        self,
        candidate_wav: str | Path,
        ref_features: dict,
        target_text: str,
        target_language: str | None = None,
        target_emotion_label: str | None = None,
    ) -> CandidateScores:
        out_emo:  EmotionAnalysis  = self.emo.analyze(candidate_wav)
        out_vad:  VADFeatures      = self.vad.analyze(candidate_wav)
        out_pros: ProsodyFeatures  = self.pros.analyze(candidate_wav)
        out_spk:  SpeakerFeatures  = self.spk.analyze(candidate_wav)

        ref_emo:  EmotionAnalysis  = ref_features["emo"]
        ref_vad:  VADFeatures      = ref_features["vad"]
        ref_pros: ProsodyFeatures  = ref_features["pros"]
        ref_spk:  SpeakerFeatures  = ref_features["spk"]

        target_label = _LABEL_MAP.get((target_emotion_label or ref_emo.label).lower(), "neutral")
        e2v_p_target = float(out_emo.logits.get(target_label, 0.0))

        e2v_cos = cosine(ref_emo.embedding, out_emo.embedding)
        spk_sim = cosine(ref_spk.embedding, out_spk.embedding)

        f0_ratio  = _safe_ratio(out_pros.f0_std,     ref_pros.f0_std)
        e_ratio   = _safe_ratio(out_pros.energy_std, ref_pros.energy_std)

        cer = self.asr.content_error(candidate_wav, target_text, language=target_language)

        return CandidateScores(
            vad_dist        = _vad_dist(out_vad, ref_vad),
            valence_diff    = abs(out_vad.valence   - ref_vad.valence),
            arousal_diff    = abs(out_vad.arousal   - ref_vad.arousal),
            dominance_diff  = abs(out_vad.dominance - ref_vad.dominance),
            f0_std_ratio    = f0_ratio,
            energy_std_ratio = e_ratio,
            e2v_cos         = e2v_cos,
            e2v_p_target    = e2v_p_target,
            e2v_top         = out_emo.label,
            speaker_sim     = spk_sim,
            content_error   = cer,
            audio_quality   = 1.0,   # placeholder until UTMOS/DNSMOS wired in
        )
