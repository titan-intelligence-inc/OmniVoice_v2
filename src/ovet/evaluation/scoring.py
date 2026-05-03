"""Total-score computation from CandidateScores.

The sign convention matches §15.3 of the design doc: distance-style
metrics enter with a negative sign (smaller is better), similarity-style
metrics enter positively.
"""
from __future__ import annotations

from ..types import CandidateScores, ScoringWeights


def compute_total_score(scores: CandidateScores, weights: ScoringWeights) -> float:
    """Linear combination per design §15.3."""
    return (
        - weights.vad        * scores.vad_dist
        - weights.valence    * scores.valence_diff
        - weights.arousal    * scores.arousal_diff
        - weights.f0_dev     * abs(1.0 - scores.f0_std_ratio)
        - weights.energy_dev * abs(1.0 - scores.energy_std_ratio)
        + weights.e2v_cos    * scores.e2v_cos
        - weights.content    * scores.content_error
        + weights.quality    * scores.audio_quality
    )
