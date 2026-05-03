"""Phase 1/2 candidate generator: vary instruct proxy across runs.

Phase 3+ will additionally vary alpha/layer for activation steering.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from ..types import GenerationRequest
from ..omnivoice.wrapper import OmniVoiceWrapper
from ..utils.io import save_wav


@dataclass
class CandidateSpec:
    """Generation parameter set for one candidate."""
    instruct: str | None
    alpha: float = 0.0
    layer_ids: tuple[int, ...] = ()
    projection_removal_language: bool = False
    tag: str = ""


@dataclass
class GeneratedCandidate:
    spec: CandidateSpec
    wav_path: Path
    duration: float


class CandidateGenerator:
    """Generate one or more candidates for a single GenerationRequest."""

    def __init__(self, wrapper: OmniVoiceWrapper):
        self.wrapper = wrapper

    def generate(
        self,
        req: GenerationRequest,
        specs: list[CandidateSpec],
        ref_text: str | None = None,
    ) -> list[GeneratedCandidate]:
        out_dir = Path(req.output_dir) / "candidates"
        out_dir.mkdir(parents=True, exist_ok=True)
        results: list[GeneratedCandidate] = []

        for i, spec in enumerate(specs):
            tag = spec.tag or f"cand{i:02d}"
            audio = self.wrapper.generate(
                text=req.text,
                language=req.language,
                ref_audio=req.ref_audio,
                ref_text=ref_text or req.ref_text,
                instruct=spec.instruct,
            )
            outp = save_wav(out_dir / f"{tag}.wav", audio, self.wrapper.SAMPLING_RATE)
            results.append(GeneratedCandidate(
                spec=spec,
                wav_path=outp,
                duration=float(len(audio) / self.wrapper.SAMPLING_RATE),
            ))
        return results
