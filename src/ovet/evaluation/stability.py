"""Run a fixed CandidateSpec multiple times, measure metric variance.

Used to validate that observed metric differences (across alpha values
in Phase 3, across proxies in Phase 2) exceed the noise floor of the
generation+evaluation pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics

import numpy as np

from ..types import GenerationRequest, CandidateScores
from ..generation.candidate_generator import CandidateGenerator, CandidateSpec
from ..evaluation.evaluator import CandidateEvaluator


_METRICS = (
    "vad_dist", "valence_diff", "arousal_diff", "dominance_diff",
    "f0_std_ratio", "energy_std_ratio",
    "e2v_cos", "e2v_p_target", "speaker_sim", "content_error", "audio_quality",
)


@dataclass
class MetricStats:
    mean: float
    std: float
    minv: float
    maxv: float
    n: int


@dataclass
class StabilityReport:
    spec_tag: str
    instruct: str | None
    reps: int
    per_metric: dict[str, MetricStats]
    runs: list[dict[str, float]]   # raw scores per repetition

    def to_dict(self) -> dict:
        return {
            "spec_tag": self.spec_tag,
            "instruct": self.instruct,
            "reps":     self.reps,
            "per_metric": {k: asdict(v) for k, v in self.per_metric.items()},
            "runs": self.runs,
        }


def _stats(values: list[float]) -> MetricStats:
    if not values:
        return MetricStats(0.0, 0.0, 0.0, 0.0, 0)
    return MetricStats(
        mean=float(statistics.fmean(values)),
        std=float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        minv=float(min(values)),
        maxv=float(max(values)),
        n=len(values),
    )


def run_stability(
    request: GenerationRequest,
    spec: CandidateSpec,
    reps: int,
    generator: CandidateGenerator,
    evaluator: CandidateEvaluator,
    ref_features: dict,
    target_text: str,
    target_language: str,
    target_emotion_label: str | None = None,
    output_dir: Path | None = None,
) -> StabilityReport:
    """Generate ``reps`` outputs for ``spec`` and aggregate metric stats.

    Each repetition writes its wav into a per-rep subdirectory so they
    are not overwritten.
    """
    runs: list[dict[str, float]] = []
    base = output_dir or (Path(request.output_dir) / "stability" / spec.tag)
    base.mkdir(parents=True, exist_ok=True)

    for i in range(reps):
        rep_request = GenerationRequest(**{**request.__dict__, "output_dir": base / f"rep{i:02d}"})
        gens = generator.generate(rep_request, [spec], ref_text=request.ref_text)
        g = gens[0]
        sc = evaluator.evaluate(
            g.wav_path, ref_features,
            target_text=target_text,
            target_language=target_language,
            target_emotion_label=target_emotion_label,
        )
        # Convert dataclass to plain dict (skip non-numeric fields)
        d = {k: getattr(sc, k) for k in _METRICS}
        d["e2v_top"] = sc.e2v_top  # keep label for inspection but exclude from stats
        runs.append(d)

    per_metric: dict[str, MetricStats] = {}
    for k in _METRICS:
        values = [r[k] for r in runs if isinstance(r[k], (int, float))]
        per_metric[k] = _stats(values)

    return StabilityReport(
        spec_tag=spec.tag,
        instruct=spec.instruct,
        reps=reps,
        per_metric=per_metric,
        runs=runs,
    )


def is_metric_reliable(stats: MetricStats, threshold: float = 0.05) -> bool:
    """Convenience: a metric is 'reliable' if its std stays within ``threshold``."""
    return stats.std <= threshold
