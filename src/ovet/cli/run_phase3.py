"""Phase 3 CLI: alpha-grid sweep with multi-rep averaging.

Workflow:

1. Build a per-layer ``v_emo`` from (emotional_ref, neutral_ref).
2. For each ``alpha`` in the grid, generate ``reps`` outputs with
   activation steering enabled (alpha=alpha, layers=layer_ids).
3. Evaluate all outputs against the emotional_ref features.
4. Aggregate per-alpha mean ± std for each metric and emit
   result.json + report.md + alpha_curves.csv.

Example:
    python -m ovet.cli.run_phase3 \
        --text "Thank you so much for coming today." \
        --language English \
        --emotional-ref baseline/jvnv_samples/jvnv_F1_anger.wav \
        --neutral-ref   baseline/jvnv_samples/jvnv_F1_sad.wav \
        --layers 8,12,16 \
        --alpha-grid 0.0,0.5,1.0,2.0 \
        --reps 3 \
        --output-dir outputs/phase3_anger_en
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..config import load_config
from ..types import GenerationRequest, Candidate
from ..omnivoice.wrapper import OmniVoiceWrapper, SteeringConfig
from ..omnivoice.steering import compute_v_emo, save_vectors
from ..analyzers.emotion_analyzer import EmotionAnalyzer
from ..analyzers.vad_analyzer import VADAnalyzer
from ..analyzers.prosody_analyzer import ProsodyAnalyzer
from ..analyzers.speaker_analyzer import SpeakerAnalyzer
from ..analyzers.asr_analyzer import ASRAnalyzer
from ..evaluation.evaluator import CandidateEvaluator
from ..evaluation.scoring import compute_total_score
from ..utils.io import save_wav, load_wav


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_layers(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Phase 3: activation-steering alpha sweep")
    ap.add_argument("--text",             required=True)
    ap.add_argument("--language",         required=True)
    ap.add_argument("--emotional-ref",    required=True, type=Path,
                    help="Reference audio carrying the desired emotion (also used as ref_audio for cloning).")
    ap.add_argument("--neutral-ref",      required=True, type=Path,
                    help="Same-speaker neutral audio used to compute v_emo.")
    ap.add_argument("--emotional-text",   default=None)
    ap.add_argument("--neutral-text",     default=None)
    ap.add_argument("--layers",           required=True, type=str,
                    help="Comma-separated Qwen3 layer ids (e.g. 8,12,16).")
    ap.add_argument("--alpha-grid",       default="0.0,0.5,1.0,2.0")
    ap.add_argument("--reps",             type=int, default=3,
                    help="Reps per alpha (multi-rep averaging).")
    ap.add_argument("--instruct",         default=None,
                    help="Optional fixed instruct proxy across the whole sweep.")
    ap.add_argument("--probe-num-step",   type=int, default=4)
    ap.add_argument("--seed",             type=int, default=0)
    ap.add_argument("--config",           default=None, type=Path)
    ap.add_argument("--output-dir",       default="outputs/phase3_run", type=Path)
    ap.add_argument("--hf-home",          default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_ids = _parse_layers(args.layers)
    alphas    = _parse_floats(args.alpha_grid)
    cfg       = load_config(args.config)

    # ------------------------------------------------------------------
    print("[ovet] Loading analyzers ...", flush=True)
    emo_an, vad_an, pros_an = EmotionAnalyzer(), VADAnalyzer(), ProsodyAnalyzer()
    spk_an, asr_an          = SpeakerAnalyzer(), ASRAnalyzer()
    evaluator = CandidateEvaluator(emo_an, vad_an, pros_an, spk_an, asr_an)

    print(f"[ovet] Analyzing reference: {args.emotional_ref}", flush=True)
    ref_features = evaluator.analyze_reference(args.emotional_ref)
    ref_emo = ref_features["emo"]
    ref_vad = ref_features["vad"]
    ref_pros = ref_features["pros"]
    print(f"[ovet] ref emotion: {ref_emo.label} ({ref_emo.confidence:.3f}), "
          f"V/A/D=({ref_vad.valence:.2f},{ref_vad.arousal:.2f},{ref_vad.dominance:.2f}), "
          f"f0_std={ref_pros.f0_std:.1f}, energy_std={ref_pros.energy_std:.3f}",
          flush=True)

    # ------------------------------------------------------------------
    print("[ovet] Loading OmniVoice ...", flush=True)
    w = OmniVoiceWrapper(hf_home=args.hf_home)

    emotional_text = args.emotional_text or w.transcribe(args.emotional_ref, language=None)
    neutral_text   = args.neutral_text   or w.transcribe(args.neutral_ref,   language=None)
    print(f"[ovet] emotional_text: {emotional_text!r}", flush=True)
    print(f"[ovet] neutral_text:   {neutral_text!r}",   flush=True)

    # ------------------------------------------------------------------
    print(f"[ovet] Computing v_emo at layers {layer_ids} ...", flush=True)
    v_emo = compute_v_emo(
        w,
        emotional_audio=args.emotional_ref,
        neutral_audio  =args.neutral_ref,
        layer_ids      =layer_ids,
        emotional_text =emotional_text,
        neutral_text   =neutral_text,
        seed           =args.seed,
        num_step       =args.probe_num_step,
    )
    save_vectors(args.output_dir / "v_emo.npz", v_emo, meta={
        "emotional_ref": str(args.emotional_ref),
        "neutral_ref":   str(args.neutral_ref),
        "layer_ids":     layer_ids,
        "seed":          args.seed,
    })
    for i in layer_ids:
        v = v_emo[i]
        print(f"  v_emo[{i:>2}]: shape={v.shape} std={v.std():.3f} |v|₂={float((v**2).sum()**0.5):.3f}",
              flush=True)

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    candidates_per_alpha: dict[float, list[Candidate]] = {a: [] for a in alphas}
    target_label = (args.instruct and ref_emo.label) or ref_emo.label

    for alpha in alphas:
        for rep in range(args.reps):
            tag = f"alpha={alpha:.2f}_rep{rep}"
            sc = SteeringConfig(
                enabled       = True,
                alpha         = alpha,
                layer_ids     = layer_ids,
                emotion_vector= v_emo,
                projection_removal=False,
            )
            torch.manual_seed(args.seed + rep)  # different seeds across reps but same across alphas
            audio = w.generate(
                text=args.text, language=args.language,
                ref_audio=args.emotional_ref, ref_text=emotional_text,
                instruct=args.instruct, steering=sc,
            )
            wav_path = args.output_dir / f"candidates/alpha{alpha:.2f}_rep{rep}.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            save_wav(wav_path, audio, w.SAMPLING_RATE)

            scores = evaluator.evaluate(
                wav_path, ref_features,
                target_text=args.text, target_language=args.language,
                target_emotion_label=target_label,
            )
            total = compute_total_score(scores, cfg.scoring)
            cand = Candidate(
                wav_path=wav_path, instruct=args.instruct,
                alpha=alpha, layer_ids=layer_ids,
                projection_removal_language=False,
                scores=scores, total_score=total,
                meta={"tag": tag, "rep": rep},
            )
            candidates_per_alpha[alpha].append(cand)
            print(f"  [{tag:<24}] vad={scores.vad_dist:.3f}  V_diff={scores.valence_diff:.3f}  "
                  f"E_ratio={scores.energy_std_ratio:.3f}  e2v_cos={scores.e2v_cos:.3f}  "
                  f"spk={scores.speaker_sim:.3f}  CER={scores.content_error:.3f}  "
                  f"score={total:.3f}", flush=True)

    # ------------------------------------------------------------------
    # Aggregate per-alpha statistics
    # ------------------------------------------------------------------
    metrics = ("vad_dist", "valence_diff", "arousal_diff",
               "f0_std_ratio", "energy_std_ratio",
               "e2v_cos", "speaker_sim", "content_error", "audio_quality")
    summary = []
    print("\n=== alpha sweep summary (mean ± std across reps) ===")
    print(f"{'alpha':<7} {'vad_dist':<14} {'val_diff':<14} {'E_ratio':<14} {'e2v_cos':<14} {'spk':<14}")
    for a in alphas:
        cands = candidates_per_alpha[a]
        per_metric = {}
        for m in metrics:
            vals = [getattr(c.scores, m) for c in cands]
            per_metric[m] = {
                "mean": float(statistics.fmean(vals)),
                "std":  float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
                "n":    len(vals),
            }
        totals = [c.total_score for c in cands]
        per_metric["total_score"] = {
            "mean": float(statistics.fmean(totals)),
            "std":  float(statistics.pstdev(totals)) if len(totals) > 1 else 0.0,
            "n":    len(totals),
        }
        summary.append({"alpha": a, "metrics": per_metric})
        print(f"{a:<7.2f}",
              f"{per_metric['vad_dist']['mean']:.3f}±{per_metric['vad_dist']['std']:.3f}".ljust(14),
              f"{per_metric['valence_diff']['mean']:.3f}±{per_metric['valence_diff']['std']:.3f}".ljust(14),
              f"{per_metric['energy_std_ratio']['mean']:.3f}±{per_metric['energy_std_ratio']['std']:.3f}".ljust(14),
              f"{per_metric['e2v_cos']['mean']:.3f}±{per_metric['e2v_cos']['std']:.3f}".ljust(14),
              f"{per_metric['speaker_sim']['mean']:.3f}±{per_metric['speaker_sim']['std']:.3f}".ljust(14))

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    result = {
        "request": {
            "text": args.text, "language": args.language,
            "emotional_ref": str(args.emotional_ref),
            "neutral_ref":   str(args.neutral_ref),
            "emotional_text": emotional_text,
            "neutral_text":   neutral_text,
            "instruct": args.instruct,
        },
        "config": {
            "layers": layer_ids, "alpha_grid": alphas, "reps": args.reps,
            "seed": args.seed,
        },
        "reference_analysis": {
            "emotion_label": ref_emo.label,
            "vad": asdict(ref_vad),
            "prosody": asdict(ref_pros),
        },
        "summary_per_alpha": summary,
        "candidates": [
            {
                "tag": c.meta["tag"], "alpha": c.alpha, "rep": c.meta["rep"],
                "wav": str(c.wav_path), "scores": asdict(c.scores),
                "total": c.total_score,
            }
            for cands in candidates_per_alpha.values() for c in cands
        ],
    }
    with open(args.output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # CSV
    import csv
    with open(args.output_dir / "alpha_curves.csv", "w", newline="") as f:
        w_csv = csv.writer(f)
        w_csv.writerow(["alpha", "metric", "mean", "std", "n"])
        for s in summary:
            for m, st in s["metrics"].items():
                w_csv.writerow([s["alpha"], m, st["mean"], st["std"], st["n"]])

    print(f"\n[ovet] Saved: {args.output_dir}/{{result.json, alpha_curves.csv, v_emo.npz, candidates/}}",
          flush=True)


if __name__ == "__main__":
    main()
