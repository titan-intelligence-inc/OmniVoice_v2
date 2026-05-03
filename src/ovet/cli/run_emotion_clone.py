"""End-to-end CLI for Phase 1: voice cloning + (optional) instruct proxy + evaluation.

Example:
    python -m ovet.cli.run_emotion_clone \
        --text "Thank you so much for coming today." \
        --language English \
        --ref-audio baseline/jvnv_samples/jvnv_F1_anger.wav \
        --emotion-hint anger \
        --output-dir outputs/test_anger_en
"""
from __future__ import annotations
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from ..types import (
    GenerationRequest, Thresholds, ScoringWeights, Candidate,
)
from ..prompts.instruct_proxy import InstructProxyComposer
from ..omnivoice.wrapper import OmniVoiceWrapper
from ..analyzers.emotion_analyzer import EmotionAnalyzer
from ..analyzers.vad_analyzer import VADAnalyzer
from ..analyzers.prosody_analyzer import ProsodyAnalyzer
from ..analyzers.speaker_analyzer import SpeakerAnalyzer
from ..analyzers.asr_analyzer import ASRAnalyzer
from ..evaluation.evaluator import CandidateEvaluator
from ..evaluation.scoring import compute_total_score
from ..evaluation.selector import select_best
from ..generation.candidate_generator import CandidateGenerator, CandidateSpec
from ..utils.io import save_wav, load_wav


def _scores_to_dict(scores) -> dict:
    return asdict(scores)


def main():
    parser = argparse.ArgumentParser(description="Voice cloning + emotion preservation (Phase 1)")
    parser.add_argument("--text",        required=True)
    parser.add_argument("--language",    required=True)
    parser.add_argument("--ref-audio",   required=True, type=Path)
    parser.add_argument("--ref-text",    default=None)
    parser.add_argument("--emotion-hint", default=None,
                        help="Optional emotion label to drive instruct proxy "
                             "(e.g. 'anger'). Defaults to ref's emotion2vec top label.")
    parser.add_argument("--output-dir",  default="outputs/run", type=Path)
    parser.add_argument("--also-baseline", action="store_true",
                        help="Generate a no-instruct baseline candidate alongside the proxy one.")
    parser.add_argument("--hf-home",     default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Reference analysis (drives both instruct proxy and evaluation)
    # ------------------------------------------------------------------
    print("[ovet] Loading analyzers ...", flush=True)
    emo_an  = EmotionAnalyzer()
    vad_an  = VADAnalyzer()
    pros_an = ProsodyAnalyzer()
    spk_an  = SpeakerAnalyzer()
    asr_an  = ASRAnalyzer()

    print(f"[ovet] Analyzing reference: {args.ref_audio}", flush=True)
    ref_emo  = emo_an.analyze(args.ref_audio)
    ref_vad  = vad_an.analyze(args.ref_audio)
    ref_pros = pros_an.analyze(args.ref_audio)
    ref_spk  = spk_an.analyze(args.ref_audio)
    print(f"[ovet] ref emotion: {ref_emo.label} ({ref_emo.confidence:.3f}), "
          f"V/A/D=({ref_vad.valence:.2f},{ref_vad.arousal:.2f},{ref_vad.dominance:.2f}), "
          f"f0_std={ref_pros.f0_std:.1f}, energy_std={ref_pros.energy_std:.3f}",
          flush=True)

    emotion_for_proxy = (args.emotion_hint or ref_emo.label)
    proxy = InstructProxyComposer().compose(emotion_for_proxy)
    print(f"[ovet] instruct proxy: {proxy!r}", flush=True)

    # ------------------------------------------------------------------
    # OmniVoice load + ref_text auto-transcription
    # ------------------------------------------------------------------
    print("[ovet] Loading OmniVoice ...", flush=True)
    wrapper = OmniVoiceWrapper(hf_home=args.hf_home)

    if args.ref_text:
        ref_text = args.ref_text
    else:
        print("[ovet] Auto-transcribing ref_text via Whisper ...", flush=True)
        # Heuristic: ref language can differ from target language; here we let
        # Whisper auto-detect by passing None.
        ref_text = wrapper.transcribe(args.ref_audio, language=None)
        print(f"[ovet] ref_text (auto): {ref_text!r}", flush=True)

    # ------------------------------------------------------------------
    # Generate candidates
    # ------------------------------------------------------------------
    specs: list[CandidateSpec] = []
    if args.also_baseline:
        specs.append(CandidateSpec(instruct=None, tag="baseline"))
    if proxy:
        specs.append(CandidateSpec(instruct=proxy, tag="proxy"))
    elif not specs:
        specs.append(CandidateSpec(instruct=None, tag="baseline"))

    req = GenerationRequest(
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=ref_text,
        output_dir=args.output_dir,
        emotion_label_hint=args.emotion_hint,
    )
    gen = CandidateGenerator(wrapper)
    print(f"[ovet] Generating {len(specs)} candidate(s) ...", flush=True)
    generated = gen.generate(req, specs)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print("[ovet] Evaluating candidates ...", flush=True)
    ev = CandidateEvaluator(emo_an, vad_an, pros_an, spk_an, asr_an)
    ref_features = {"emo": ref_emo, "vad": ref_vad, "pros": ref_pros, "spk": ref_spk}

    weights = ScoringWeights()
    thresholds = Thresholds()
    candidates: list[Candidate] = []
    for g in generated:
        sc = ev.evaluate(
            g.wav_path, ref_features,
            target_text=args.text,
            target_language=args.language,
            target_emotion_label=args.emotion_hint or ref_emo.label,
        )
        total = compute_total_score(sc, weights)
        c = Candidate(
            wav_path=g.wav_path,
            instruct=g.spec.instruct,
            alpha=g.spec.alpha,
            layer_ids=list(g.spec.layer_ids),
            projection_removal_language=g.spec.projection_removal_language,
            scores=sc,
            total_score=total,
            meta={"tag": g.spec.tag, "duration": g.duration},
        )
        candidates.append(c)
        print(f"  [{g.spec.tag}] vad_dist={sc.vad_dist:.3f}  valence_diff={sc.valence_diff:.3f}  "
              f"e2v_cos={sc.e2v_cos:.3f}  spk={sc.speaker_sim:.3f}  CER={sc.content_error:.3f}  "
              f"score={total:.3f}", flush=True)

    best = select_best(candidates, thresholds)
    print(f"[ovet] best: {best.meta.get('tag')} score={best.total_score:.3f}", flush=True)

    # Save final.wav as a copy of the best candidate
    final_path = args.output_dir / "final.wav"
    wav, sr = load_wav(best.wav_path)
    save_wav(final_path, wav, sr)

    result = {
        "request": {
            "text": args.text,
            "language": args.language,
            "ref_audio": str(args.ref_audio),
            "ref_text": ref_text,
            "emotion_hint": args.emotion_hint,
        },
        "reference_analysis": {
            "emotion_label": ref_emo.label,
            "emotion_confidence": ref_emo.confidence,
            "vad": asdict(ref_vad),
            "prosody": asdict(ref_pros),
        },
        "best_tag": best.meta.get("tag"),
        "best_path": str(best.wav_path),
        "candidates": [
            {
                "tag": c.meta.get("tag"),
                "wav": str(c.wav_path),
                "instruct": c.instruct,
                "scores": _scores_to_dict(c.scores),
                "total_score": c.total_score,
            }
            for c in candidates
        ],
    }
    with open(args.output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[ovet] Saved: {args.output_dir/'final.wav'}, {args.output_dir/'result.json'}",
          flush=True)


if __name__ == "__main__":
    main()
