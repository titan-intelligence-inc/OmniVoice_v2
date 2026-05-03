"""Best-candidate selection with hard constraints + fallback."""
from __future__ import annotations

from ..types import Candidate, Thresholds


def filter_valid(candidates: list[Candidate], thresholds: Thresholds) -> list[Candidate]:
    out = []
    for c in candidates:
        s = c.scores
        if s.content_error > thresholds.content_error:
            continue
        if s.audio_quality < thresholds.quality:
            continue
        if s.speaker_sim < thresholds.speaker:
            continue
        out.append(c)
    return out


def select_best(candidates: list[Candidate], thresholds: Thresholds) -> Candidate:
    """Return the candidate with max total_score among the constraint-valid set.

    Fallback (no candidate satisfies constraints): pick the one that minimizes
    content_error first, then maximizes audio_quality, then minimizes vad_dist,
    then maximizes e2v_cos.
    """
    if not candidates:
        raise ValueError("select_best: no candidates")

    valid = filter_valid(candidates, thresholds)
    if valid:
        return max(valid, key=lambda c: c.total_score)

    return max(
        candidates,
        key=lambda c: (
            -c.scores.content_error,
            c.scores.audio_quality,
            -c.scores.vad_dist,
            c.scores.e2v_cos,
        ),
    )
