"""Shared dataclasses for the ovet pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class GenerationRequest:
    text: str
    language: str
    ref_audio: Path
    ref_text: str | None = None
    neutral_ref_audio: Path | None = None
    lang_pair_ref_l1: Path | None = None
    lang_pair_ref_l2: Path | None = None
    output_dir: Path = Path("outputs")
    emotion_label_hint: str | None = None


@dataclass
class EmotionAnalysis:
    label: str                              # top-1 from emotion2vec
    confidence: float                        # top-1 prob
    embedding: np.ndarray                    # utterance-level emotion2vec emb
    logits: dict[str, float]                 # full 9-class probability map


@dataclass
class VADFeatures:
    valence:   float
    arousal:   float
    dominance: float


@dataclass
class ProsodyFeatures:
    f0_mean:    float
    f0_std:     float
    f0_range:   float
    energy_mean: float
    energy_std:  float
    duration:    float
    voiced_ratio: float


@dataclass
class SpeakerFeatures:
    embedding: np.ndarray   # ECAPA-TDNN utterance-level


@dataclass
class CandidateScores:
    """Per-candidate evaluation metrics."""
    vad_dist:           float   # ‖(V,A,D)_out - (V,A,D)_ref‖₂
    valence_diff:       float
    arousal_diff:       float
    dominance_diff:     float
    f0_std_ratio:       float   # 1.0 = identical to ref
    energy_std_ratio:   float
    e2v_cos:            float   # cosine similarity in emotion2vec space
    e2v_p_target:       float   # only meaningful for ZH outputs
    e2v_top:            str
    speaker_sim:        float
    content_error:      float   # CER (normalized [0,1])
    audio_quality:      float   # MOS-like score, [0,5]


@dataclass
class Candidate:
    wav_path: Path
    instruct: str | None
    alpha: float
    layer_ids: list[int]
    projection_removal_language: bool
    scores: CandidateScores
    total_score: float
    meta: dict = field(default_factory=dict)


@dataclass
class Thresholds:
    content_error: float = 0.20
    quality:       float = 0.50
    speaker:       float = 0.85


@dataclass
class ScoringWeights:
    vad:        float = 1.0
    valence:    float = 0.8
    arousal:    float = 0.4
    f0_dev:     float = 0.3
    energy_dev: float = 0.5
    e2v_cos:    float = 0.3
    content:    float = 0.8
    quality:    float = 0.5
