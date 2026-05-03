"""Phase 5: black-box optimisation across alpha × projection × layers × proxy.

Pipeline:

1. Build v_emo (and v_lang, when projection-removal is in the grid) once.
2. For each grid point, run ``reps`` generations.
3. Aggregate per-cell mean / std.
4. Compute Pareto front along (emotion fidelity, speaker preservation).
5. Apply constrained selection (max emotion fidelity subject to
   speaker_sim ≥ threshold).
6. Emit result.json + report.md + grid.csv + pareto.csv.

Example:
    python -m ovet.cli.run_phase5 \
        --text "Thank you so much for coming today." \
        --language English \
        --emotional-ref baseline/jvnv_samples/jvnv_F1_anger.wav \
        --neutral-ref   baseline/jvnv_samples/jvnv_F1_sad.wav \
        --alphas 0.0,0.5,1.0,1.5 \
        --projections off,on \
        --layer-sets "8,12,16;12;8,12" \
        --instructs ",high pitch,low pitch" \
        --reps 2 \
        --speaker-threshold 0.30 \
        --output-dir outputs/phase5_anger_en
"""
from __future__ import annotations
import argparse
import csv
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
from ..optimization.grid import (
    GridSpec, GridPoint, ParetoPoint, pareto_front,
    constrained_best, Constraint, make_objectives,
)
from ..utils.io import save_wav


_NEUTRAL_TEXTS = {
    "Japanese": "今日の会議は午後三時から始まります。",
    "English":  "The meeting starts at three in the afternoon today.",
}


# ---------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------

def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_layer_sets(s: str) -> list[tuple[int, ...]]:
    out = []
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(tuple(int(x) for x in chunk.split(",") if x.strip()))
    return out


def _parse_projections(s: str) -> list[bool]:
    out = []
    for tok in s.split(","):
        t = tok.strip().lower()
        if t in ("on", "true", "1", "yes"):
            out.append(True)
        elif t in ("off", "false", "0", "no", ""):
            out.append(False)
        else:
            raise ValueError(f"unknown projection token: {tok!r}")
    return out


def _parse_instructs(s: str) -> list[str | None]:
    out = []
    for tok in s.split(","):
        t = tok.strip()
        out.append(None if t == "" or t.lower() in ("none", "null") else t)
    return out


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Phase 5: black-box optimisation")
    ap.add_argument("--text",             required=True)
    ap.add_argument("--language",         required=True)
    ap.add_argument("--emotional-ref",    required=True, type=Path)
    ap.add_argument("--neutral-ref",      required=True, type=Path)
    ap.add_argument("--emotional-text",   default=None)
    ap.add_argument("--neutral-text",     default=None)
    # grid
    ap.add_argument("--alphas",           default="0.0,0.5,1.0,1.5")
    ap.add_argument("--projections",      default="off,on")
    ap.add_argument("--layer-sets",       default="8,12,16;12")
    ap.add_argument("--instructs",        default=",high pitch")
    ap.add_argument("--reps",             type=int, default=2)
    # constrained selection
    ap.add_argument("--speaker-threshold", type=float, default=0.30,
                    help="speaker_sim hard constraint for constrained_best")
    ap.add_argument("--objective",         default="vad_dist")
    ap.add_argument("--objective-direction", default="min", choices=["min", "max"])
    # i/o
    ap.add_argument("--probe-num-step",   type=int, default=4)
    ap.add_argument("--seed",             type=int, default=0)
    ap.add_argument("--config",           default=None, type=Path)
    ap.add_argument("--output-dir",       default="outputs/phase5_run", type=Path)
    ap.add_argument("--lang-pair-l1",     default="Japanese")
    ap.add_argument("--lang-pair-l2",     default="English")
    ap.add_argument("--hf-home",          default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    spec = GridSpec(
        alphas      = _parse_floats(args.alphas),
        projections = _parse_projections(args.projections),
        layer_sets  = _parse_layer_sets(args.layer_sets),
        instructs   = _parse_instructs(args.instructs),
    )
    grid = spec.enumerate()
    print(f"[ovet] grid size: {len(grid)} cells × {args.reps} reps = "
          f"{len(grid)*args.reps} generations", flush=True)

    # ------------------------------------------------------------------
    print("[ovet] Loading analyzers ...", flush=True)
    emo_an, vad_an, pros_an = EmotionAnalyzer(), VADAnalyzer(), ProsodyAnalyzer()
    spk_an, asr_an          = SpeakerAnalyzer(), ASRAnalyzer()
    evaluator = CandidateEvaluator(emo_an, vad_an, pros_an, spk_an, asr_an)

    print(f"[ovet] Analyzing reference: {args.emotional_ref}", flush=True)
    ref_features = evaluator.analyze_reference(args.emotional_ref)
    ref_emo, ref_vad, ref_pros = (
        ref_features["emo"], ref_features["vad"], ref_features["pros"]
    )
    print(f"[ovet] ref emotion: {ref_emo.label} ({ref_emo.confidence:.3f}), "
          f"V/A/D=({ref_vad.valence:.2f},{ref_vad.arousal:.2f},{ref_vad.dominance:.2f}), "
          f"f0_std={ref_pros.f0_std:.1f}, energy_std={ref_pros.energy_std:.3f}",
          flush=True)

    # ------------------------------------------------------------------
    print("[ovet] Loading OmniVoice ...", flush=True)
    w = OmniVoiceWrapper(hf_home=args.hf_home)

    emotional_text = args.emotional_text or w.transcribe(args.emotional_ref, language=None)
    neutral_text   = args.neutral_text   or w.transcribe(args.neutral_ref,   language=None)

    # ------------------------------------------------------------------
    # Build vectors once for *all* layers used in the grid.
    # ------------------------------------------------------------------
    union_layers = sorted({l for ls in spec.layer_sets for l in ls})
    print(f"[ovet] Computing v_emo at layers {union_layers} ...", flush=True)
    v_emo = compute_v_emo(
        w, emotional_audio=args.emotional_ref, neutral_audio=args.neutral_ref,
        layer_ids=union_layers,
        emotional_text=emotional_text, neutral_text=neutral_text,
        seed=args.seed, num_step=args.probe_num_step,
    )
    save_vectors(args.output_dir / "v_emo.npz", v_emo,
                 meta={"layers": union_layers, "seed": args.seed,
                       "emotional_ref": str(args.emotional_ref),
                       "neutral_ref": str(args.neutral_ref)})

    needs_v_lang = any(p.projection_removal for p in grid)
    v_lang = None
    if needs_v_lang:
        # Self-generate neutral L1/L2 clones of the target speaker.
        clones_dir = args.output_dir / "lang_pair_clones"
        clones_dir.mkdir(parents=True, exist_ok=True)
        l1_path = clones_dir / "selfgen_l1.wav"
        l2_path = clones_dir / "selfgen_l2.wav"
        if not l1_path.exists():
            torch.manual_seed(args.seed)
            audio = w.model.generate(
                text=_NEUTRAL_TEXTS[args.lang_pair_l1], language=args.lang_pair_l1,
                ref_audio=str(args.emotional_ref), ref_text=emotional_text, num_step=8,
            )[0]
            save_wav(l1_path, audio, w.SAMPLING_RATE)
        if not l2_path.exists():
            torch.manual_seed(args.seed)
            audio = w.model.generate(
                text=_NEUTRAL_TEXTS[args.lang_pair_l2], language=args.lang_pair_l2,
                ref_audio=str(args.emotional_ref), ref_text=emotional_text, num_step=8,
            )[0]
            save_wav(l2_path, audio, w.SAMPLING_RATE)
        print("[ovet] Computing v_lang from self-generated clones ...", flush=True)
        h_l1 = extract_layer_vectors(w, l1_path, union_layers, ref_text=None,
                                     num_step=args.probe_num_step, seed=args.seed)
        h_l2 = extract_layer_vectors(w, l2_path, union_layers, ref_text=None,
                                     num_step=args.probe_num_step, seed=args.seed)
        v_lang = {i: (h_l1[i] - h_l2[i]).astype(np.float32) for i in union_layers}
        save_vectors(args.output_dir / "v_lang.npz", v_lang,
                     meta={"l1": args.lang_pair_l1, "l2": args.lang_pair_l2,
                           "seed": args.seed})

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    METRICS = ("vad_dist", "valence_diff", "arousal_diff",
               "f0_std_ratio", "energy_std_ratio",
               "e2v_cos", "speaker_sim", "content_error", "audio_quality")

    cells: list[dict] = []
    cands_dir = args.output_dir / "candidates"
    cands_dir.mkdir(parents=True, exist_ok=True)

    for gp in grid:
        cell_cands: list[Candidate] = []
        for rep in range(args.reps):
            tag = f"{gp.tag}_rep{rep}"
            sc_layers = list(gp.layer_ids)
            if gp.alpha == 0.0 or not sc_layers:
                sc = None  # plain cloning; no hooks installed
            else:
                vecs = {i: v_emo[i] for i in sc_layers if i in v_emo}
                lang_vecs = ({i: v_lang[i] for i in sc_layers if v_lang and i in v_lang}
                             if (gp.projection_removal and v_lang) else None)
                sc = SteeringConfig(
                    enabled=True, alpha=gp.alpha, layer_ids=sc_layers,
                    emotion_vector=vecs, language_vector=lang_vecs,
                    projection_removal=gp.projection_removal,
                )
            torch.manual_seed(args.seed + rep)
            audio = w.generate(
                text=args.text, language=args.language,
                ref_audio=args.emotional_ref, ref_text=emotional_text,
                instruct=gp.instruct, steering=sc,
            )
            wav_path = cands_dir / f"{tag}.wav"
            save_wav(wav_path, audio, w.SAMPLING_RATE)

            scores = evaluator.evaluate(
                wav_path, ref_features,
                target_text=args.text, target_language=args.language,
                target_emotion_label=ref_emo.label,
            )
            total = compute_total_score(scores, cfg.scoring)
            cell_cands.append(Candidate(
                wav_path=wav_path, instruct=gp.instruct,
                alpha=gp.alpha, layer_ids=list(gp.layer_ids),
                projection_removal_language=gp.projection_removal,
                scores=scores, total_score=total,
                meta={"tag": tag, "rep": rep, "grid_tag": gp.tag},
            ))

        # Aggregate per cell
        agg = {}
        totals = [c.total_score for c in cell_cands]
        agg["total_score"] = {
            "mean": float(statistics.fmean(totals)),
            "std":  float(statistics.pstdev(totals)) if len(totals) > 1 else 0.0,
        }
        for m in METRICS:
            vals = [getattr(c.scores, m) for c in cell_cands]
            agg[m] = {
                "mean": float(statistics.fmean(vals)),
                "std":  float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
            }
        # Flat metric mean dict for constraint / Pareto consumption
        flat_metrics = {k: v["mean"] for k, v in agg.items()}

        cells.append({
            "tag": gp.tag,
            "alpha": gp.alpha,
            "projection_removal": gp.projection_removal,
            "layer_ids": list(gp.layer_ids),
            "instruct": gp.instruct,
            "metrics": flat_metrics,
            "agg": agg,
            "candidates": [
                {"wav": str(c.wav_path), "rep": c.meta["rep"],
                 "scores": asdict(c.scores), "total_score": c.total_score}
                for c in cell_cands
            ],
        })
        print(f"[{gp.tag:<48}] vad={flat_metrics['vad_dist']:.3f}  "
              f"V_diff={flat_metrics['valence_diff']:.3f}  "
              f"E_ratio={flat_metrics['energy_std_ratio']:.3f}  "
              f"e2v_cos={flat_metrics['e2v_cos']:.3f}  "
              f"spk={flat_metrics['speaker_sim']:.3f}  "
              f"score={flat_metrics['total_score']:.3f}",
              flush=True)

    # ------------------------------------------------------------------
    # Pareto front along (emotion fidelity, speaker preservation)
    # ------------------------------------------------------------------
    pareto_axes = (("vad_dist", "min"), ("speaker_sim", "max"))
    pareto_pts = [
        ParetoPoint(
            objectives=make_objectives(c["metrics"], pareto_axes),
            meta={"tag": c["tag"]},
        )
        for c in cells
    ]
    front_tags = {p.meta["tag"] for p in pareto_front(pareto_pts)}
    for c in cells:
        c["pareto"] = c["tag"] in front_tags

    # ------------------------------------------------------------------
    # Constrained best
    # ------------------------------------------------------------------
    constraints = [
        Constraint("speaker_sim",  ">=", args.speaker_threshold),
        Constraint("content_error","<=", cfg.thresholds.content_error),
    ]
    best_constrained = constrained_best(
        cells, objective=args.objective,
        objective_direction=args.objective_direction,
        constraints=constraints,
    )

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    payload = {
        "request": {
            "text": args.text, "language": args.language,
            "emotional_ref": str(args.emotional_ref),
            "neutral_ref":   str(args.neutral_ref),
        },
        "grid": {
            "alphas": spec.alphas, "projections": spec.projections,
            "layer_sets": [list(L) for L in spec.layer_sets],
            "instructs":  spec.instructs, "reps": args.reps,
        },
        "reference": {
            "emotion_label": ref_emo.label,
            "vad": asdict(ref_vad),
            "prosody": asdict(ref_pros),
        },
        "cells": cells,
        "pareto_tags": sorted(front_tags),
        "best_constrained": (
            {"tag": best_constrained["tag"], "metrics": best_constrained["metrics"]}
            if best_constrained else None
        ),
        "constraints": [
            {"metric": k.metric, "op": k.op, "value": k.value} for k in constraints
        ],
        "objective": {"metric": args.objective, "direction": args.objective_direction},
    }
    with open(args.output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # CSV for easy plotting
    with open(args.output_dir / "grid.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow([
            "tag", "alpha", "projection", "layers", "instruct",
            "vad_dist", "valence_diff", "energy_std_ratio",
            "e2v_cos", "speaker_sim", "content_error", "total_score",
            "pareto",
        ])
        for c in cells:
            m = c["metrics"]
            wr.writerow([
                c["tag"], c["alpha"], int(c["projection_removal"]),
                "|".join(str(x) for x in c["layer_ids"]),
                c["instruct"] or "",
                f"{m['vad_dist']:.4f}", f"{m['valence_diff']:.4f}",
                f"{m['energy_std_ratio']:.4f}",
                f"{m['e2v_cos']:.4f}", f"{m['speaker_sim']:.4f}",
                f"{m['content_error']:.4f}", f"{m['total_score']:.4f}",
                int(c["pareto"]),
            ])

    # Markdown report
    md = ["# Phase 5 black-box optimisation report\n"]
    md.append(f"- Emotional ref: `{args.emotional_ref}`")
    md.append(f"- Neutral ref:   `{args.neutral_ref}`")
    md.append(f"- Target text:   `{args.text}` ({args.language})")
    md.append(f"- Grid: {len(grid)} cells × {args.reps} reps "
              f"= {len(grid)*args.reps} generations\n")

    md.append("## Constrained best\n")
    md.append(f"Objective: **{args.objective_direction}** of `{args.objective}`")
    md.append("Constraints: " + ", ".join(f"`{k.metric} {k.op} {k.value}`" for k in constraints))
    if best_constrained:
        bm = best_constrained["metrics"]
        md.append(f"\n**Winner**: `{best_constrained['tag']}`\n")
        md.append("| metric | value |")
        md.append("|---|---|")
        for k in ("vad_dist", "valence_diff", "energy_std_ratio",
                  "e2v_cos", "speaker_sim", "content_error", "total_score"):
            md.append(f"| {k} | {bm[k]:.3f} |")
    else:
        md.append("\n⚠️ No cell satisfies the constraints.")

    md.append("\n## Pareto front (vad_dist ↓ × speaker_sim ↑)\n")
    md.append("| tag | vad_dist | speaker_sim | score |")
    md.append("|---|---|---|---|")
    sorted_front = sorted(
        (c for c in cells if c["pareto"]),
        key=lambda c: c["metrics"]["vad_dist"],
    )
    for c in sorted_front:
        m = c["metrics"]
        md.append(f"| {c['tag']} | {m['vad_dist']:.3f} | "
                  f"{m['speaker_sim']:.3f} | {m['total_score']:.3f} |")

    md.append("\n## Full grid\n")
    md.append("(⭐ = on Pareto front)\n")
    md.append("| tag | vad_dist | val_diff | E_ratio | e2v_cos | spk | score | pareto |")
    md.append("|---|---|---|---|---|---|---|---|")
    for c in sorted(cells, key=lambda c: c["metrics"]["vad_dist"]):
        m = c["metrics"]
        marker = " ⭐" if c["pareto"] else ""
        md.append(f"| {c['tag']} | {m['vad_dist']:.3f} | {m['valence_diff']:.3f} | "
                  f"{m['energy_std_ratio']:.3f} | {m['e2v_cos']:.3f} | "
                  f"{m['speaker_sim']:.3f} | {m['total_score']:.3f} |{marker}")

    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[ovet] Saved: {args.output_dir}/{{result.json, grid.csv, report.md, "
          f"v_emo.npz, v_lang.npz, candidates/}}", flush=True)


if __name__ == "__main__":
    main()
