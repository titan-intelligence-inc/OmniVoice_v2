"""Quick quality-tuning test on a single (lang, emotion) cell.

Compares 4 variants:
  - baseline                (alpha=0)
  - phase4                  (alpha=1, accent=0)
  - phase4_accent_strong    (alpha=1, accent=1.0, layers=[8,12,16], no quality knobs)
  - phase4_accent_quality   (alpha=1, accent=0.5, layers=[12], step_window=(0, 16),
                             norm_clip_factor=0.3)

The "quality" variant is the result of applying the four interventions
(A: lower alpha + single layer; B: step window; D: norm clip).

Example:
    python scripts/quality_test.py \
        --multilang-dir outputs/multilang_F1_3strat \
        --lang-code ko --emotion sad --reps 3
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
    return {int(k.split("_", 1)[1]): np.asarray(data[k]).astype(np.float32)
            for k in data.files if k.startswith("layer_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multilang-dir", required=True, type=Path)
    ap.add_argument("--lang-code",     required=True)
    ap.add_argument("--emotion",       required=True)
    ap.add_argument("--reps",          type=int, default=3)
    ap.add_argument("--output-dir",    default=None, type=Path)
    ap.add_argument("--config-yaml",   default="configs/multilang.yaml", type=Path)
    ap.add_argument("--seed",          type=int, default=0)
    ap.add_argument("--hf-home",       default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", args.hf_home)

    out_dir = args.output_dir or (args.multilang_dir.parent
                                  / f"quality_test_{args.lang_code}_{args.emotion}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config_yaml, encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)
    target_text = cfg_yaml["target_text"][args.lang_code]
    lang_full   = cfg_yaml["language_full_name"][args.lang_code]
    ref_audio   = Path(cfg_yaml["ref_audio_dir"]) / f"jvnv_{cfg_yaml['ref_speaker']}_{args.emotion}.wav"

    v_emo  = _load_npz(args.multilang_dir / f"v_emo_{args.emotion}.npz")
    v_lang = _load_npz(args.multilang_dir / f"v_lang_{args.lang_code}.npz")

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

    # Round 3: keep all 32 diffusion steps active (so accent stays "anchored"
    # throughout) but cap the per-step delta magnitude with a tighter
    # norm_clip. Round 2 showed clip=0.5 didn't bite — try 0.2 / 0.3.
    # Also try "overdrive + clip": raise accent_α above 1.0 but clip hard.
    full_layers = [8, 12, 16]
    base_kwargs = dict(
        enabled=True, alpha=1.0, layer_ids=full_layers,
        emotion_vector=v_emo, language_vector=v_lang,
        projection_removal=True, language_alpha=1.0,
    )

    def cfg(**override):
        return SteeringConfig(**{**base_kwargs, **override})

    # Round 4: localize push in time AND amplify it. step_window=(0,8)
    # gave clean audio but lost ~30% of accent. Hypothesis: overdriving
    # accent_α inside that small window restores accent without bringing
    # back artifacts (because the remaining 24 diffusion steps refine
    # the prosody freely).
    # Round 5: take step_8th_a20 (round 4 winner on prosody+accent) and add
    # position_mask. Hypothesis: applying delta only at audio token positions
    # (not text/ref) reduces audio-side disruption → cleaner audio.
    variants = [
        ("baseline", lambda: None),
        ("phase4_accent_strong", lambda: cfg()),                                 # anchor
        ("strong_step_8th_a20",  lambda: cfg(step_window=(0, 4),
                                              language_alpha=2.0)),              # round 4 winner
        # NEW: position mask on top of round 4 winner
        ("strong_8th_a20_pmask", lambda: cfg(step_window=(0, 4),
                                              language_alpha=2.0,
                                              position_mask=True)),
        # NEW: position mask on the original anchor (accent=1.0, all 32 steps)
        ("strong_pmask",         lambda: cfg(position_mask=True)),
        # NEW: position mask + heavier overdrive (could afford more push at fewer positions)
        ("strong_8th_a30_pmask", lambda: cfg(step_window=(0, 4),
                                              language_alpha=3.0,
                                              position_mask=True)),
    ]

    METRICS = ("vad_dist", "valence_diff", "energy_std_ratio",
               "e2v_cos", "speaker_sim", "content_error")

    print(f"\n=== {args.lang_code}/{args.emotion}  reps={args.reps} ===", flush=True)
    rows = []
    for name, build_sc in variants:
        per_rep = []
        for rep in range(args.reps):
            sc = build_sc()
            torch.manual_seed(args.seed + rep)
            try:
                audio = w.generate(
                    text=target_text, language=lang_full,
                    ref_audio=str(ref_audio), ref_text=ref_text,
                    steering=sc,
                )
                wav_path = out_dir / f"{name}_rep{rep}.wav"
                save_wav(wav_path, audio, w.SAMPLING_RATE)
                scores = evaluator.evaluate(
                    wav_path, ref_features,
                    target_text=target_text, target_language=lang_full,
                    target_emotion_label=args.emotion,
                )
                per_rep.append({"wav": str(wav_path),
                                **{k: getattr(scores, k) for k in METRICS}})
            except Exception as e:
                print(f"  ⚠ {name} rep{rep} FAILED: {type(e).__name__}: {e}", flush=True)
                # Use placeholder values so aggregation still works
                per_rep.append({"wav": None,
                                **{k: 0.0 for k in METRICS}})
        agg = {k: {
            "mean":   statistics.fmean(r[k] for r in per_rep),
            "median": statistics.median(r[k] for r in per_rep),
            "std":    statistics.pstdev(r[k] for r in per_rep) if args.reps > 1 else 0.0,
        } for k in METRICS}
        rows.append({"name": name, "metrics": agg, "per_rep": per_rep})
        print(f"  {name:<26}  vad={agg['vad_dist']['mean']:.3f}±{agg['vad_dist']['std']:.3f}  "
              f"V_d={agg['valence_diff']['mean']:.3f}  "
              f"E_r={agg['energy_std_ratio']['mean']:.3f}  "
              f"spk={agg['speaker_sim']['mean']:.3f}  "
              f"CER med={agg['content_error']['median']:.3f}",
              flush=True)

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump({"lang": args.lang_code, "emotion": args.emotion,
                   "reps": args.reps, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\n[ovet] Saved: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
