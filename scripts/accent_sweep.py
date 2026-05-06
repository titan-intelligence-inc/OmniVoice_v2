"""Quick sweep: language_alpha grid on a single (lang, emotion) pair to test
the accent-removal hypothesis (subtracting v_lang during generation pushes
the output away from the JP-acoustic basin).

Reuses the v_emo and v_lang vectors saved by the multilang sweep, so it
does not need to recompute them.

Example:
    python scripts/accent_sweep.py \
        --multilang-dir outputs/multilang_F1_anger_sad_fear_happy_surprise_disgust \
        --lang-code zh \
        --emotion sad \
        --alpha 1.0 \
        --accent-grid 0.0,0.5,1.0,1.5,2.0 \
        --reps 3
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def _load_npz(path: Path) -> dict[int, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    out = {}
    for k in data.files:
        if k.startswith("layer_"):
            out[int(k.split("_", 1)[1])] = np.asarray(data[k]).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multilang-dir", required=True, type=Path,
                    help="Output dir from a previous run_multilang sweep "
                         "(must contain v_emo_<emo>.npz and v_lang_<lang>.npz)")
    ap.add_argument("--lang-code",     required=True)
    ap.add_argument("--emotion",       required=True)
    ap.add_argument("--alpha",         type=float, default=1.0,
                    help="Emotion steering weight")
    ap.add_argument("--accent-grid",   default="0.0,0.5,1.0,1.5,2.0")
    ap.add_argument("--layers",        default="8,12,16")
    ap.add_argument("--reps",          type=int, default=3)
    ap.add_argument("--output-dir",    default=None, type=Path)
    ap.add_argument("--config-yaml",   default="configs/multilang.yaml", type=Path)
    ap.add_argument("--seed",          type=int, default=0)
    ap.add_argument("--hf-home",       default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    out_dir = args.output_dir or (args.multilang_dir.parent
                                  / f"accent_sweep_{args.lang_code}_{args.emotion}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config and vectors
    with open(args.config_yaml, encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)
    target_text = cfg_yaml["target_text"][args.lang_code]
    lang_full   = cfg_yaml["language_full_name"][args.lang_code]
    ref_audio   = Path(cfg_yaml["ref_audio_dir"]) / f"jvnv_{cfg_yaml['ref_speaker']}_{args.emotion}.wav"
    layers      = [int(x) for x in args.layers.split(",")]
    accent_grid = [float(x) for x in args.accent_grid.split(",")]

    v_emo  = _load_npz(args.multilang_dir / f"v_emo_{args.emotion}.npz")
    v_lang = _load_npz(args.multilang_dir / f"v_lang_{args.lang_code}.npz")
    print(f"[ovet] v_emo  layers: {sorted(v_emo.keys())}", flush=True)
    print(f"[ovet] v_lang layers: {sorted(v_lang.keys())}", flush=True)

    # Load analyzers + OmniVoice
    sys.path.insert(0, "src")
    from ovet.config import load_config
    from ovet.types import Candidate
    from ovet.omnivoice.wrapper import OmniVoiceWrapper, SteeringConfig
    from ovet.analyzers.emotion_analyzer import EmotionAnalyzer
    from ovet.analyzers.vad_analyzer import VADAnalyzer
    from ovet.analyzers.prosody_analyzer import ProsodyAnalyzer
    from ovet.analyzers.speaker_analyzer import SpeakerAnalyzer
    from ovet.analyzers.asr_analyzer import ASRAnalyzer
    from ovet.evaluation.evaluator import CandidateEvaluator
    from ovet.evaluation.scoring import compute_total_score
    from ovet.utils.io import save_wav

    print("[ovet] Loading analyzers ...", flush=True)
    emo_an, vad_an, pros_an = EmotionAnalyzer(), VADAnalyzer(), ProsodyAnalyzer()
    spk_an, asr_an          = SpeakerAnalyzer(), ASRAnalyzer()
    evaluator = CandidateEvaluator(emo_an, vad_an, pros_an, spk_an, asr_an)
    ref_features = evaluator.analyze_reference(ref_audio)

    print("[ovet] Loading OmniVoice ...", flush=True)
    w = OmniVoiceWrapper(hf_home=args.hf_home)
    ref_text = w.transcribe(ref_audio, language=None)
    cfg = load_config()

    METRICS = ("vad_dist", "valence_diff", "energy_std_ratio",
               "e2v_cos", "speaker_sim", "content_error")

    print(f"\n=== {args.lang_code}/{args.emotion}  ref={ref_audio.name} "
          f"alpha={args.alpha} layers={layers} reps={args.reps} ===", flush=True)
    rows = []
    for accent_alpha in accent_grid:
        per_rep_scores: list[dict] = []
        for rep in range(args.reps):
            sc = SteeringConfig(
                enabled=True,
                alpha=args.alpha if (args.alpha > 0 or accent_alpha == 0) else args.alpha,
                layer_ids=layers,
                emotion_vector=v_emo,
                language_vector=v_lang,
                projection_removal=True,
                language_alpha=accent_alpha,
            )
            torch.manual_seed(args.seed + rep)
            audio = w.generate(
                text=target_text, language=lang_full,
                ref_audio=str(ref_audio), ref_text=ref_text,
                steering=sc,
            )
            wav_path = out_dir / f"acc{accent_alpha:.2f}_rep{rep}.wav"
            save_wav(wav_path, audio, w.SAMPLING_RATE)
            scores = evaluator.evaluate(
                wav_path, ref_features,
                target_text=target_text, target_language=lang_full,
                target_emotion_label=args.emotion,
            )
            d = {k: getattr(scores, k) for k in METRICS}
            d["wav"] = str(wav_path)
            per_rep_scores.append(d)

        # Aggregate
        agg = {}
        for k in METRICS:
            vals = [r[k] for r in per_rep_scores]
            agg[k] = {
                "mean":   statistics.fmean(vals),
                "median": statistics.median(vals),
                "std":    statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "min":    min(vals),
                "max":    max(vals),
            }
        rows.append({"accent_alpha": accent_alpha, "metrics": agg,
                     "per_rep": per_rep_scores})
        print(f"  acc={accent_alpha:.2f}  "
              f"vad={agg['vad_dist']['mean']:.3f}  "
              f"V_d={agg['valence_diff']['mean']:.3f}  "
              f"E_r={agg['energy_std_ratio']['mean']:.3f}  "
              f"spk={agg['speaker_sim']['mean']:.3f}  "
              f"CER mean={agg['content_error']['mean']:.3f} "
              f"(med={agg['content_error']['median']:.3f})",
              flush=True)

    # Persist
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "lang": args.lang_code, "emotion": args.emotion,
            "alpha": args.alpha, "accent_grid": accent_grid,
            "layers": layers, "reps": args.reps,
            "rows": rows,
        }, f, indent=2, ensure_ascii=False)

    md = [f"# accent sweep — {args.lang_code}/{args.emotion}\n",
          f"- ref: `{ref_audio}`",
          f"- emotion alpha = {args.alpha}, accent_grid = {accent_grid}",
          f"- layers = {layers}, reps = {args.reps}\n",
          "| accent_α | vad_dist | val_diff | E_ratio | spk | CER mean | CER median |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["metrics"]
        md.append(f"| {r['accent_alpha']:.2f} | "
                  f"{m['vad_dist']['mean']:.3f}±{m['vad_dist']['std']:.3f} | "
                  f"{m['valence_diff']['mean']:.3f} | "
                  f"{m['energy_std_ratio']['mean']:.3f} | "
                  f"{m['speaker_sim']['mean']:.3f} | "
                  f"{m['content_error']['mean']:.3f} | "
                  f"{m['content_error']['median']:.3f} |")
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[ovet] Saved: {out_dir}/{{result.json, report.md, *.wav}}", flush=True)


if __name__ == "__main__":
    main()
