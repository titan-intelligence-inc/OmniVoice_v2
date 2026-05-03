"""YAML config loader → typed config dataclasses.

Use ``load_config(path)`` to get an ``OvetConfig`` for a run. Falls back
to ``configs/default.yaml`` when no path is given.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import yaml

from .types import Thresholds, ScoringWeights


_DEFAULT = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"


@dataclass
class GenerationConfig:
    reps_per_spec: int = 1
    alpha_grid: list[float] = field(default_factory=lambda: [0.0])
    layer_sets: list[list[int]] = field(default_factory=lambda: [[]])
    projection_removal_language: bool = False


@dataclass
class StabilityConfig:
    enabled: bool = False
    reps: int = 3


@dataclass
class OvetConfig:
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    thresholds: Thresholds = field(default_factory=Thresholds)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    proxy_grid: dict[str, list[str | None]] = field(default_factory=dict)
    stability: StabilityConfig = field(default_factory=StabilityConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": asdict(self.generation),
            "thresholds": asdict(self.thresholds),
            "scoring":    asdict(self.scoring),
            "proxy_grid": self.proxy_grid,
            "stability":  asdict(self.stability),
        }


def load_config(path: str | Path | None = None) -> OvetConfig:
    p = Path(path) if path is not None else _DEFAULT
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return OvetConfig(
        generation=GenerationConfig(**(raw.get("generation") or {})),
        thresholds=Thresholds(**(raw.get("thresholds") or {})),
        scoring=ScoringWeights(**(raw.get("scoring") or {})),
        proxy_grid=raw.get("proxy_grid") or {},
        stability=StabilityConfig(**(raw.get("stability") or {})),
    )
