"""Phase 4 CLI: alpha sweep × language-projection on/off.

Workflow:

1. Synthesize "neutral" JP and EN clones of the target speaker via
   OmniVoice itself, using the emotional reference as the cloning ref.
   These are the "OmniVoice self-generated" v_lang basis (Phase 4a
   showed this works).
2. Build ``v_lang`` per layer from those synthesized clones.
3. Build ``v_emo`` per layer from (emotional_ref, neutral_ref).
4. For each (alpha, projection_removal) combination, run ``reps``
   generations. Evaluate against the emotional reference.
5. Aggregate per-cell mean ± std, write CSV / JSON / report.md.

Example:
    python -m ovet.cli.run_phase4 \
        --text "Thank you so much for coming today." \
        --language English \
        --emotional-ref baseline/jvnv_samples/jvnv_F1_anger.wav \
        --neutral-ref   baseline/jvnv_samples/jvnv_F1_sad.wav \
        --layers 8,12,16 \
        --alpha-grid 0.0,0.5,1.0,2.0 \
        --reps 3 \
        --output-dir outputs/phase4_anger_en
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
from ..types import Candidate
from ..omnivoice.wrapper import OmniVoiceWrapper, SteeringConfig
from ..omnivoice.steering import (
    compute_v_emo, extract_layer_vectors, save_vectors,
)
from ..analyzers.emotion_analyzer import EmotionAnalyzer
from ..analyzers.vad_analyzer import VADAnalyzer
from ..analyzers.prosody_analyzer import ProsodyAnalyzer
from ..analyzers.speaker_analyzer import SpeakerAnalyzer
from ..analyzers.asr_analyzer import ASRAnalyzer
from ..evaluation.evaluator import CandidateEvaluator
from ..evaluation.scoring import compute_total_score
from ..utils.io import save_wav


_NEUTRAL_TEXTS = {
    "Japanese": "今日の会議は午後三時から始まります。",
    "English":  "The meeting starts at three in the afternoon today.",
    "Chinese":  "今天的会议下午三点开始。",
}


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_layers(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _synth_lang_pair_clones(
    wrapper: OmniVoiceWrapper,
    target_speaker_audio: Path,
    ref_text: str,
    out_dir: Path,
    seed: int,
    num_step: int = 8,
    l1: str = "Japanese",
    l2: str = "English",
) -> tuple[Path, Path]:
    """Self-generate neutral L1/L2 clones of the target speaker."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for lang in (l1, l2):
        outp = out_dir / f"selfgen_{lang[:2].lower()}.wav"
        if not outp.exists():
            torch.manual_seed(seed)
            audio = wrapper.model.generate(
                text=_NEUTRAL_TEXTS[lang],
                language=lang,
                ref_audio=str(target_speaker_audio),
                ref_text=ref_text,
                num_step=num_step,
            )[0]
            save_wav(outp, audio, wrapper.SAMPLING_RATE)
        paths[lang] = outp
    return paths[l1], paths[l2]


def _aggregate(cands: list[Candidate], metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    out = {}
    totals = [c.total_score for c in cands]
    out["total_score"] = {
        "mean": float(statistics.fmean(totals)),
        "std":  float(statistics.pstdev(totals)) if len(totals) > 1 else 0.0,
        "n":    len(totals),
    }
    for m in metrics:
        vals = [getattr(c.scores, m) for c in cands]
        out[m] = {
            "mean": float(statistics.fmean(vals)),
            "std":  float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
            "n":    len(vals),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Phase 4: alpha × projection on/off sweep")
    ap.add_argument("--text",             required=True)
    ap.add_argument("--language",         required=True)
    ap.add_argument("--emotional-ref",    required=True, type=Path)
    ap.add_argument("--neutral-ref",      required=True, type=Path)
    ap.add_argument("--emotional-text",   default=None)
    ap.add_argument("--neutral-text",     default=None)

    # Optional: pre-existing language pair refs. If not given, we
    # synthesize neutral clones of the target speaker.
    ap.add_argument("--lang-pair-l1-audio", default=None, type=Path)
    ap.add_argument("--lang-pair-l2-audio", default=None, type=Path)
    ap.add_argument("--lang-pair-l1",       default="Japanese")
    ap.add_argument("--lang-pair-l2",       default="English")

    ap.add_argument("--layers",           required=True, type=str)
    ap.add_argument("--alpha-grid",       default="0.0,0.5,1.0,2.0")
    ap.add_argument("--reps",             type=int, default=3)
    ap.add_argument("--probe-num-step",   type=int, default=4)
    ap.add_argument("--seed",             type=int, default=0)
    ap.add_argument("--config",           default=None, type=Path)
    ap.add_argument("--output-dir",       default="outputs/phase4_run", type=Path)
    ap.add_argument("--hf-home",          default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = _parse_layers(args.layers)
    alphas = _parse_floats(args.alpha_grid)
    cfg    = load_config(args.config)

    # ------------------------------------------------------------------
    print("[ovet] Loading analyzers ...", flush=True)
    emo_an, vad_an, pros_an = EmotionAnalyzer(), VADAnalyzer(), ProsodyAnalyzer()
    spk_an, asr_an          = SpeakerAnalyzer(), ASRAnalyzer()
    evaluator = CandidateEvaluator(emo_an, vad_an, pros_an, spk_an, asr_an)

    print(f"[ovet] Analyzing reference: {args.emotional_ref}", flush=True)
    ref_features = evaluator.analyze_reference(args.emotional_ref)
    ref_emo, ref_vad, ref_pros = ref_features["emo"], ref_features["vad"], ref_features["pros"]
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
    # v_emo
    # ------------------------------------------------------------------
    print(f"[ovet] Computing v_emo at layers {layers} ...", flush=True)
    v_emo = compute_v_emo(
        w,
        emotional_audio=args.emotional_ref, neutral_audio=args.neutral_ref,
        layer_ids=layers,
        emotional_text=emotional_text, neutral_text=neutral_text,
        seed=args.seed, num_step=args.probe_num_step,
    )
    save_vectors(args.output_dir / "v_emo.npz", v_emo, meta={
        "emotional_ref": str(args.emotional_ref), "neutral_ref": str(args.neutral_ref),
        "layer_ids": layers, "seed": args.seed,
    })

    # ------------------------------------------------------------------
    # v_lang (synthesize neutral L1/L2 clones if not provided)
    # ------------------------------------------------------------------
    if args.lang_pair_l1_audio and args.lang_pair_l2_audio:
        l1_audio = args.lang_pair_l1_audio
        l2_audio = args.lang_pair_l2_audio
        print(f"[ovet] Using provided lang pair: L1={l1_audio.name}, L2={l2_audio.name}", flush=True)
    else:
        print(f"[ovet] Synthesizing neutral {args.lang_pair_l1}/{args.lang_pair_l2} "
              f"clones of target speaker ...", flush=True)
        l1_audio, l2_audio = _synth_lang_pair_clones(
            w, args.emotional_ref, emotional_text,
            out_dir=args.output_dir / "lang_pair_clones",
            seed=args.seed,
            l1=args.lang_pair_l1, l2=args.lang_pair_l2,
        )

    print(f"[ovet] Computing v_lang at layers {layers} ...", flush=True)
    h_l1 = extract_layer_vectors(w, l1_audio, layers, ref_text=None,
                                 num_step=args.probe_num_step, seed=args.seed)
    h_l2 = extract_layer_vectors(w, l2_audio, layers, ref_text=None,
                                 num_step=args.probe_num_step, seed=args.seed)
    v_lang = {i: (h_l1[i] - h_l2[i]).astype(np.float32) for i in layers}
    save_vectors(args.output_dir / "v_lang.npz", v_lang, meta={
        "l1_audio": str(l1_audio), "l2_audio": str(l2_audio),
        "l1": args.lang_pair_l1, "l2": args.lang_pair_l2,
        "layer_ids": layers, "seed": args.seed,
    })
    for i in layers:
        ve, vl = v_emo[i], v_lang[i]
        cos = float(abs(np.dot(ve, vl)) / (np.linalg.norm(ve) * np.linalg.norm(vl) + 1e-9))
        print(f"  layer {i:>2}: |v_emo|={np.linalg.norm(ve):.2f} "
              f"|v_lang|={np.linalg.norm(vl):.2f} |cos(emo,lang)|={cos:.3f}",
              flush=True)

    # ------------------------------------------------------------------
    # Sweep: alpha × projection on/off
    # ------------------------------------------------------------------
    METRICS = ("vad_dist", "valence_diff", "arousal_diff",
               "f0_std_ratio", "energy_std_ratio",
               "e2v_cos", "speaker_sim", "content_error", "audio_quality")
    cells: dict[tuple[float, bool], list[Candidate]] = {}

    for alpha in alphas:
        for proj in (False, True):
            cell_key = (alpha, proj)
            cells[cell_key] = []
            for rep in range(args.reps):
                tag = f"alpha{alpha:.2f}_proj{int(proj)}_rep{rep}"
                sc = SteeringConfig(
                    enabled        = True,
                    alpha          = alpha,
                    layer_ids      = layers,
                    emotion_vector = v_emo,
                    language_vector= v_lang,
                    projection_removal=proj,
                )
                torch.manual_seed(args.seed + rep)
                audio = w.generate(
                    text=args.text, language=args.language,
                    ref_audio=args.emotional_ref, ref_text=emotional_text,
                    steering=sc,
                )
                wav_path = args.output_dir / "candidates" / f"{tag}.wav"
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                save_wav(wav_path, audio, w.SAMPLING_RATE)

                scores = evaluator.evaluate(
                    wav_path, ref_features,
                    target_text=args.text, target_language=args.language,
                    target_emotion_label=ref_emo.label,
                )
                total = compute_total_score(scores, cfg.scoring)
                c = Candidate(
                    wav_path=wav_path, instruct=None,
                    alpha=alpha, layer_ids=layers,
                    projection_removal_language=proj,
                    scores=scores, total_score=total,
                    meta={"tag": tag, "rep": rep, "alpha": alpha, "proj": proj},
                )
                cells[cell_key].append(c)
                print(f"  [{tag:<24}] vad={scores.vad_dist:.3f}  V_diff={scores.valence_diff:.3f}  "
                      f"E_ratio={scores.energy_std_ratio:.3f}  e2v_cos={scores.e2v_cos:.3f}  "
                      f"spk={scores.speaker_sim:.3f}  CER={scores.content_error:.3f}  "
                      f"score={total:.3f}", flush=True)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    print("\n=== Phase 4 sweep summary (mean ± std across reps) ===")
    print(f"{'alpha':<6} {'proj':<5} {'vad_dist':<14} {'val_diff':<14} {'E_ratio':<14} "
          f"{'e2v_cos':<14} {'spk':<14} {'score':<14}")
    summary = []
    for alpha in alphas:
        for proj in (False, True):
            agg = _aggregate(cells[(alpha, proj)], METRICS)
            summary.append({"alpha": alpha, "projection_removal_language": proj,
                            "metrics": agg})
            def f(k):
                return f"{agg[k]['mean']:.3f}±{agg[k]['std']:.3f}".ljust(14)
            print(f"{alpha:<6.2f} {str(proj):<5} {f('vad_dist')} {f('valence_diff')} "
                  f"{f('energy_std_ratio')} {f('e2v_cos')} {f('speaker_sim')} {f('total_score')}")

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    payload = {
        "request": {
            "text": args.text, "language": args.language,
            "emotional_ref": str(args.emotional_ref),
            "neutral_ref":   str(args.neutral_ref),
            "lang_pair":     {"l1": args.lang_pair_l1, "l2": args.lang_pair_l2,
                              "l1_audio": str(l1_audio), "l2_audio": str(l2_audio)},
        },
        "config": {"layers": layers, "alpha_grid": alphas,
                   "reps": args.reps, "seed": args.seed},
        "reference_analysis": {
            "emotion_label": ref_emo.label,
            "vad": asdict(ref_vad),
            "prosody": asdict(ref_pros),
        },
        "summary": summary,
        "candidates": [
            {"alpha": c.alpha, "proj": c.projection_removal_language,
             "rep": c.meta["rep"], "wav": str(c.wav_path),
             "scores": asdict(c.scores), "total": c.total_score}
            for cands in cells.values() for c in cands
        ],
    }
    with open(args.output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # CSV
    import csv
    with open(args.output_dir / "alpha_curves.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["alpha", "projection_removal_language", "metric", "mean", "std", "n"])
        for s in summary:
            for m, st in s["metrics"].items():
                wr.writerow([s["alpha"], int(s["projection_removal_language"]),
                             m, st["mean"], st["std"], st["n"]])

    # Markdown delta table (proj=on minus proj=off, per alpha)
    md = ["# Phase 4 sweep: alpha × language projection on/off\n"]
    md.append(f"- Emotional ref: `{args.emotional_ref}`")
    md.append(f"- Neutral ref:   `{args.neutral_ref}`")
    md.append(f"- Language pair: L1={args.lang_pair_l1}, L2={args.lang_pair_l2}")
    md.append(f"- Layers: {layers}")
    md.append(f"- alpha grid: {alphas}, reps: {args.reps}\n")
    md.append("## Headline table\n")
    md.append("| alpha | proj | vad_dist | val_diff | E_ratio | e2v_cos | spk_sim | score |")
    md.append("|---|---|---|---|---|---|---|---|")
    for s in summary:
        a, p, m = s["alpha"], s["projection_removal_language"], s["metrics"]
        md.append(f"| {a:.2f} | {p} | {m['vad_dist']['mean']:.3f}±{m['vad_dist']['std']:.3f} | "
                  f"{m['valence_diff']['mean']:.3f}±{m['valence_diff']['std']:.3f} | "
                  f"{m['energy_std_ratio']['mean']:.3f}±{m['energy_std_ratio']['std']:.3f} | "
                  f"{m['e2v_cos']['mean']:.3f}±{m['e2v_cos']['std']:.3f} | "
                  f"{m['speaker_sim']['mean']:.3f}±{m['speaker_sim']['std']:.3f} | "
                  f"{m['total_score']['mean']:.3f}±{m['total_score']['std']:.3f} |")

    md.append("\n## Δ(proj=on − proj=off) per alpha\n")
    md.append("(Positive Δspk_sim = projection helps speaker preservation; "
              "negative Δvad_dist = projection helps emotion preservation.)\n")
    md.append("| alpha | Δvad_dist | Δval_diff | ΔE_ratio | Δspk_sim | Δscore |")
    md.append("|---|---|---|---|---|---|")
    for alpha in alphas:
        on  = next(s for s in summary if s["alpha"] == alpha and s["projection_removal_language"])
        off = next(s for s in summary if s["alpha"] == alpha and not s["projection_removal_language"])
        d_vad = on["metrics"]["vad_dist"]["mean"]    - off["metrics"]["vad_dist"]["mean"]
        d_val = on["metrics"]["valence_diff"]["mean"] - off["metrics"]["valence_diff"]["mean"]
        d_e   = on["metrics"]["energy_std_ratio"]["mean"] - off["metrics"]["energy_std_ratio"]["mean"]
        d_spk = on["metrics"]["speaker_sim"]["mean"]  - off["metrics"]["speaker_sim"]["mean"]
        d_sc  = on["metrics"]["total_score"]["mean"]  - off["metrics"]["total_score"]["mean"]
        md.append(f"| {alpha:.2f} | {d_vad:+.3f} | {d_val:+.3f} | {d_e:+.3f} | "
                  f"{d_spk:+.3f} | {d_sc:+.3f} |")

    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[ovet] Saved: {args.output_dir}/{{result.json, alpha_curves.csv, "
          f"report.md, v_emo.npz, v_lang.npz, candidates/}}", flush=True)


if __name__ == "__main__":
    main()
