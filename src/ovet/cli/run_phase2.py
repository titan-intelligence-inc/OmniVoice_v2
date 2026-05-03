"""Phase 2 CLI: multi-candidate run with optional stability check + report.md.

Differences vs Phase 1 CLI:
- Reads ``configs/default.yaml`` (override with --config) for thresholds,
  scoring weights, and the proxy grid.
- Generates one candidate per (proxy, rep) — instead of just baseline+proxy.
- Optionally runs a stability sweep (--stability) for one chosen spec.
- Emits ``result.json`` *and* ``report.md`` *and* (if applicable) ``stability.json``.

Example:
    python -m ovet.cli.run_phase2 \
        --text "Thank you so much for coming today." \
        --language English \
        --ref-audio baseline/jvnv_samples/jvnv_F1_anger.wav \
        --emotion-hint anger \
        --output-dir outputs/phase2_anger_en \
        --stability --stability-reps 3
"""
from __future__ import annotations
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from ..config import load_config
from ..types import (
    GenerationRequest, Candidate,
)
from ..prompts.proxy_grid import ProxyGrid
from ..omnivoice.wrapper import OmniVoiceWrapper
from ..analyzers.emotion_analyzer import EmotionAnalyzer
from ..analyzers.vad_analyzer import VADAnalyzer
from ..analyzers.prosody_analyzer import ProsodyAnalyzer
from ..analyzers.speaker_analyzer import SpeakerAnalyzer
from ..analyzers.asr_analyzer import ASRAnalyzer
from ..evaluation.evaluator import CandidateEvaluator
from ..evaluation.scoring import compute_total_score
from ..evaluation.selector import select_best
from ..evaluation.stability import run_stability
from ..evaluation.report import render_report, write_report
from ..generation.candidate_generator import CandidateGenerator, CandidateSpec
from ..utils.io import load_wav, save_wav


def _candidate_to_dict(c: Candidate) -> dict:
    return {
        "tag":      c.meta.get("tag"),
        "instruct": c.instruct,
        "wav":      str(c.wav_path),
        "scores":   asdict(c.scores),
        "total":    c.total_score,
        "alpha":    c.alpha,
        "layer_ids": c.layer_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2: multi-candidate + stability")
    parser.add_argument("--text",         required=True)
    parser.add_argument("--language",     required=True)
    parser.add_argument("--ref-audio",    required=True, type=Path)
    parser.add_argument("--ref-text",     default=None)
    parser.add_argument("--emotion-hint", default=None)
    parser.add_argument("--output-dir",   default="outputs/phase2_run", type=Path)
    parser.add_argument("--config",       default=None, type=Path)
    parser.add_argument("--stability",    action="store_true",
                        help="Also run a repeated-generation stability check.")
    parser.add_argument("--stability-reps", type=int, default=None,
                        help="Override config's stability.reps")
    parser.add_argument("--stability-spec", default="baseline",
                        help="Which spec tag to repeat for stability (default: baseline)")
    parser.add_argument("--hf-home",      default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)

    # ------------------------------------------------------------------
    # Analyzers + reference features (single set of evaluator instances)
    # ------------------------------------------------------------------
    print("[ovet] Loading analyzers ...", flush=True)
    emo_an  = EmotionAnalyzer()
    vad_an  = VADAnalyzer()
    pros_an = ProsodyAnalyzer()
    spk_an  = SpeakerAnalyzer()
    asr_an  = ASRAnalyzer()
    evaluator = CandidateEvaluator(emo_an, vad_an, pros_an, spk_an, asr_an)

    print(f"[ovet] Analyzing reference: {args.ref_audio}", flush=True)
    ref_features = evaluator.analyze_reference(args.ref_audio)
    ref_emo  = ref_features["emo"]
    ref_vad  = ref_features["vad"]
    ref_pros = ref_features["pros"]
    print(f"[ovet] ref emotion: {ref_emo.label} ({ref_emo.confidence:.3f}), "
          f"V/A/D=({ref_vad.valence:.2f},{ref_vad.arousal:.2f},{ref_vad.dominance:.2f}), "
          f"f0_std={ref_pros.f0_std:.1f}, energy_std={ref_pros.energy_std:.3f}",
          flush=True)

    # ------------------------------------------------------------------
    # OmniVoice + ref_text auto-transcription
    # ------------------------------------------------------------------
    print("[ovet] Loading OmniVoice ...", flush=True)
    wrapper = OmniVoiceWrapper(hf_home=args.hf_home)
    if args.ref_text:
        ref_text = args.ref_text
    else:
        print("[ovet] Auto-transcribing ref_text via Whisper ...", flush=True)
        ref_text = wrapper.transcribe(args.ref_audio, language=None)
        print(f"[ovet] ref_text (auto): {ref_text!r}", flush=True)

    generator = CandidateGenerator(wrapper)

    # ------------------------------------------------------------------
    # Proxy grid → CandidateSpecs
    # ------------------------------------------------------------------
    emo_for_proxy = (args.emotion_hint or ref_emo.label)
    pg = ProxyGrid(cfg.proxy_grid)
    grid_entries = pg.for_emotion(emo_for_proxy)
    specs: list[CandidateSpec] = [
        CandidateSpec(instruct=instruct, tag=tag) for tag, instruct in grid_entries
    ]
    print(f"[ovet] Proxy grid for {emo_for_proxy!r}: " +
          ", ".join(f"{s.tag}={s.instruct!r}" for s in specs), flush=True)

    req = GenerationRequest(
        text=args.text, language=args.language,
        ref_audio=args.ref_audio, ref_text=ref_text,
        output_dir=args.output_dir, emotion_label_hint=args.emotion_hint,
    )

    # ------------------------------------------------------------------
    # Generate + evaluate all candidates
    # ------------------------------------------------------------------
    print(f"[ovet] Generating {len(specs)} candidates ...", flush=True)
    generated = generator.generate(req, specs, ref_text=ref_text)

    candidates: list[Candidate] = []
    for g in generated:
        sc = evaluator.evaluate(
            g.wav_path, ref_features,
            target_text=args.text,
            target_language=args.language,
            target_emotion_label=args.emotion_hint or ref_emo.label,
        )
        total = compute_total_score(sc, cfg.scoring)
        c = Candidate(
            wav_path=g.wav_path,
            instruct=g.spec.instruct,
            alpha=g.spec.alpha,
            layer_ids=list(g.spec.layer_ids),
            projection_removal_language=g.spec.projection_removal_language,
            scores=sc, total_score=total,
            meta={"tag": g.spec.tag, "duration": g.duration},
        )
        candidates.append(c)
        print(f"  [{g.spec.tag:<20}] vad={sc.vad_dist:.3f}  V_diff={sc.valence_diff:.3f}  "
              f"e2v_cos={sc.e2v_cos:.3f}  spk={sc.speaker_sim:.3f}  CER={sc.content_error:.3f}  "
              f"score={total:.3f}", flush=True)

    best = select_best(candidates, cfg.thresholds)
    print(f"[ovet] best: {best.meta.get('tag')} score={best.total_score:.3f}", flush=True)

    # ------------------------------------------------------------------
    # Stability check (optional)
    # ------------------------------------------------------------------
    stability_reports = []
    if args.stability or cfg.stability.enabled:
        reps = args.stability_reps or cfg.stability.reps
        target_spec = next((s for s in specs if s.tag == args.stability_spec), specs[0])
        print(f"[ovet] Running stability ({reps} reps) on spec={target_spec.tag} ...", flush=True)
        rep = run_stability(
            request=req, spec=target_spec, reps=reps,
            generator=generator, evaluator=evaluator,
            ref_features=ref_features,
            target_text=args.text, target_language=args.language,
            target_emotion_label=args.emotion_hint or ref_emo.label,
        )
        stability_reports.append(rep)
        print("[ovet] Stability per-metric std:", flush=True)
        for k in ("vad_dist", "valence_diff", "energy_std_ratio", "e2v_cos", "speaker_sim"):
            st = rep.per_metric[k]
            print(f"  {k:<18} mean={st.mean:.3f}  std={st.std:.3f}", flush=True)

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    final_path = args.output_dir / "final.wav"
    wav, sr = load_wav(best.wav_path)
    save_wav(final_path, wav, sr)

    result = {
        "request": {
            "text": args.text, "language": args.language,
            "ref_audio": str(args.ref_audio), "ref_text": ref_text,
            "emotion_hint": args.emotion_hint,
        },
        "config": cfg.to_dict(),
        "reference_analysis": {
            "emotion_label": ref_emo.label,
            "emotion_confidence": ref_emo.confidence,
            "vad": asdict(ref_vad),
            "prosody": asdict(ref_pros),
        },
        "best_tag":  best.meta.get("tag"),
        "best_path": str(best.wav_path),
        "candidates": [_candidate_to_dict(c) for c in candidates],
    }
    with open(args.output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if stability_reports:
        with open(args.output_dir / "stability.json", "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in stability_reports], f, indent=2, ensure_ascii=False)

    md = render_report(
        title=f"Phase 2 run — {args.emotion_hint or ref_emo.label} → {args.language}",
        request_meta={
            "text": args.text, "language": args.language,
            "ref_audio": str(args.ref_audio), "ref_text": ref_text,
        },
        ref_summary={
            "emotion": ref_emo.label,
            "vad": {"V": ref_vad.valence, "A": ref_vad.arousal, "D": ref_vad.dominance},
            "prosody": {"f0_std": ref_pros.f0_std, "energy_std": ref_pros.energy_std},
        },
        candidates=candidates,
        best_tag=best.meta.get("tag"),
        thresholds=cfg.thresholds,
        stability=stability_reports or None,
    )
    write_report(args.output_dir / "report.md", md)
    print(f"[ovet] Saved: {args.output_dir}/{{final.wav, result.json, report.md"
          + (', stability.json' if stability_reports else '') + "}}", flush=True)


if __name__ == "__main__":
    main()
