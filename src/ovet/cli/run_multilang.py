"""Multi-language production sweep:

  6 emotions × 10 languages × {baseline, phase4} × N reps

Per cell we keep the best wav by ``vad_dist`` and aggregate metric stats
across reps. The end result is a per-language and per-emotion listening
matrix plus an index.html with embedded ``<audio>`` players.

Pipeline:

1. Capture hidden means for each emotional ref → build a synthetic
   neutral as the **mean across emotions**, then derive v_emo per
   emotion as ``h(emotion) − h(synth_neutral)``. Same speaker only.

2. For each target language, self-generate neutral L1 (Japanese) and
   target-language clones of the speaker, capture their hiddens, and
   build a per-language ``v_lang = h(L1) − h(target)``.

3. Sweep: for each (lang, emotion, rep) generate baseline (alpha=0) and
   phase4 (alpha=1.0 + projection on with that language's v_lang).

4. Pick best wav per cell by ``vad_dist``. Emit index.html, report.md,
   per-language and per-emotion mirror directories.

Example:
    python -m ovet.cli.run_multilang --output-dir outputs/multilang_F1
"""
from __future__ import annotations
import argparse
import csv
import html
import json
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from ..config import load_config
from ..types import Candidate
from ..omnivoice.wrapper import OmniVoiceWrapper, SteeringConfig
from ..omnivoice.steering import (
    extract_layer_vectors, save_vectors,
)
from ..analyzers.emotion_analyzer import EmotionAnalyzer
from ..analyzers.vad_analyzer import VADAnalyzer
from ..analyzers.prosody_analyzer import ProsodyAnalyzer
from ..analyzers.speaker_analyzer import SpeakerAnalyzer
from ..analyzers.asr_analyzer import ASRAnalyzer
from ..evaluation.evaluator import CandidateEvaluator
from ..evaluation.scoring import compute_total_score
from ..utils.io import save_wav, load_wav


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

METRICS = ("vad_dist", "valence_diff", "arousal_diff",
           "f0_std_ratio", "energy_std_ratio",
           "e2v_cos", "speaker_sim", "content_error", "audio_quality")


def _agg(vals: list[float]) -> dict:
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0, "median": 0.0}
    return {
        "mean":   float(statistics.fmean(vals)),
        "std":    float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "min":    float(min(vals)),
        "max":    float(max(vals)),
        "n":      len(vals),
        "median": float(statistics.median(vals)),
    }


def _build_synth_neutral_v_emo(
    wrapper: OmniVoiceWrapper,
    emo_refs: dict[str, tuple[Path, str]],   # emotion → (audio_path, ref_text)
    layers: list[int],
    seed: int,
    num_step: int,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[int, np.ndarray]]:
    """Capture per-emotion hidden means, then build:

        synth_neutral[L]  = mean over emotions of h_emo[L]
        v_emo[emo][L]     = h_emo[L] − synth_neutral[L]
    """
    h_per_emo: dict[str, dict[int, np.ndarray]] = {}
    for emo, (audio, text) in emo_refs.items():
        print(f"[ovet] probing hidden for emotion={emo} ({audio.name}) ...", flush=True)
        h_per_emo[emo] = extract_layer_vectors(
            wrapper, audio, layers, ref_text=text,
            num_step=num_step, seed=seed,
        )

    synth_neutral = {}
    for L in layers:
        stacked = np.stack([h_per_emo[emo][L] for emo in h_per_emo])
        synth_neutral[L] = stacked.mean(axis=0).astype(np.float32)

    v_emo = {}
    for emo, hL in h_per_emo.items():
        v_emo[emo] = {L: (hL[L] - synth_neutral[L]).astype(np.float32) for L in layers}

    return v_emo, synth_neutral


def _build_per_language_v_lang(
    wrapper: OmniVoiceWrapper,
    cloning_ref_audio: Path,
    cloning_ref_text: str,
    base_lang_code: str,
    base_lang_full: str,
    base_lang_text: str,
    targets: list[tuple[str, str, str]],     # [(code, full, text), ...]
    layers: list[int],
    seed: int,
    num_step: int,
    out_dir: Path,
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, Path]]:
    """For each target language T, self-generate neutral L1 (= base) and
    neutral T clones of the cloning speaker, then return
    ``v_lang[T][L] = h(L1)[L] − h(T)[L]``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # L1 base clone (one across all targets)
    l1_path = out_dir / f"selfgen_{base_lang_code}.wav"
    if not l1_path.exists():
        torch.manual_seed(seed)
        audio = wrapper.model.generate(
            text=base_lang_text, language=base_lang_full,
            ref_audio=str(cloning_ref_audio),
            ref_text=cloning_ref_text, num_step=8,
        )[0]
        save_wav(l1_path, audio, wrapper.SAMPLING_RATE)
    h_l1 = extract_layer_vectors(
        wrapper, l1_path, layers, ref_text=None,
        num_step=num_step, seed=seed,
    )

    v_lang: dict[str, dict[int, np.ndarray]] = {}
    target_paths: dict[str, Path] = {base_lang_code: l1_path}

    for code, full, text in targets:
        if code == base_lang_code:
            v_lang[code] = {L: np.zeros_like(h_l1[L], dtype=np.float32) for L in layers}
            continue
        target_path = out_dir / f"selfgen_{code}.wav"
        if not target_path.exists():
            print(f"[ovet] self-gen neutral clone: {full}", flush=True)
            torch.manual_seed(seed)
            audio = wrapper.model.generate(
                text=text, language=full,
                ref_audio=str(cloning_ref_audio),
                ref_text=cloning_ref_text, num_step=8,
            )[0]
            save_wav(target_path, audio, wrapper.SAMPLING_RATE)
        h_t = extract_layer_vectors(
            wrapper, target_path, layers, ref_text=None,
            num_step=num_step, seed=seed,
        )
        v_lang[code] = {L: (h_l1[L] - h_t[L]).astype(np.float32) for L in layers}
        target_paths[code] = target_path

    return v_lang, target_paths


def _build_multispeaker_v_lang(
    wrapper: OmniVoiceWrapper,
    cloning_refs: list[tuple[str, Path, str]],   # [(speaker_tag, audio, ref_text)]
    base_lang_code: str,
    base_lang_full: str,
    base_lang_text: str,
    targets: list[tuple[str, str, str]],
    layers: list[int],
    seed: int,
    num_step: int,
    out_dir: Path,
) -> tuple[
    dict[str, dict[int, np.ndarray]],                       # averaged v_lang[code][L]
    dict[str, dict[str, dict[int, np.ndarray]]],            # per_speaker[spk][code][L]
]:
    """Build per-language v_lang averaged across multiple speaker refs.

    Each speaker's clone pair is generated independently (under
    ``out_dir/<speaker_tag>/``). Per-layer per-language averaging happens
    after all speakers are processed. The averaged dict has the same
    shape as ``_build_per_language_v_lang`` so it's a drop-in
    replacement at the call site.

    Rationale: with a single ref speaker (e.g. JVNV F1), v_lang carries
    that speaker's idiosyncratic JP coloring. Averaging across multiple
    JP speakers cancels per-speaker noise while leaving the shared
    language-axis signal intact, yielding a less F1-flavoured accent
    direction.
    """
    per_speaker: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for spk_tag, ref_audio, ref_text in cloning_refs:
        print(f"[ovet] v_lang for speaker={spk_tag} ({ref_audio.name})", flush=True)
        v_lang_spk, _ = _build_per_language_v_lang(
            wrapper,
            cloning_ref_audio=ref_audio,
            cloning_ref_text=ref_text,
            base_lang_code=base_lang_code,
            base_lang_full=base_lang_full,
            base_lang_text=base_lang_text,
            targets=targets,
            layers=layers,
            seed=seed,
            num_step=num_step,
            out_dir=out_dir / spk_tag,
        )
        per_speaker[spk_tag] = v_lang_spk

    averaged: dict[str, dict[int, np.ndarray]] = {}
    for code, _, _ in targets:
        averaged[code] = {}
        for L in layers:
            stacked = np.stack(
                [per_speaker[spk][code][L] for spk in per_speaker]
            )
            averaged[code][L] = stacked.mean(axis=0).astype(np.float32)

    return averaged, per_speaker


# ------------------------------------------------------------------
# HTML & Markdown report generators
# ------------------------------------------------------------------

def _audio_tag(wav_rel: str) -> str:
    return (
        f'<audio controls preload="metadata" '
        f'style="width:240px;height:30px"><source src="{html.escape(wav_rel)}" '
        f'type="audio/wav">audio</audio>'
    )


def _render_html(
    out_dir: Path,
    cells: list[dict],
    languages: list[tuple[str, str, str]],
    emotions: list[str],
    strategy_names: list[str],
    title: str,
):
    """Emit a single ``index.html`` with an emotions × languages matrix.

    Each cell stacks one audio player + metric line per strategy.
    """
    by_key = {(c["lang_code"], c["emotion"], c["strategy"]): c for c in cells}

    head = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; margin: 16px; color: #222; }}
h1 {{ font-size: 18px; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; font-size: 12px; }}
th {{ background: #f5f5f5; position: sticky; top: 0; }}
.cell {{ min-width: 240px; }}
.metric {{ color: #666; font-family: monospace; font-size: 11px; }}
.label {{ font-weight: 600; color: #444; margin-right: 4px; }}
.s0 {{ background: #fafafa; }}   /* baseline */
.s1 {{ background: #f0f7ff; }}   /* phase4 */
.s2 {{ background: #f5fff0; }}   /* phase4_accent */
.s3 {{ background: #fffaf0; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Strategies: {', '.join(html.escape(s) for s in strategy_names)}.
Wav files are best-of-reps by vad_dist.</p>
"""
    rows = ['<table><thead><tr><th>emotion ↓ / language →</th>']
    for code, full, _ in languages:
        rows.append(f'<th>{html.escape(full)} ({code})</th>')
    rows.append('</tr></thead><tbody>')

    for emo in emotions:
        rows.append(f'<tr><th>{html.escape(emo)}</th>')
        for code, _, _ in languages:
            cell_html = []
            for idx, strat in enumerate(strategy_names):
                css = f"s{min(idx, 3)}"
                c = by_key.get((code, emo, strat))
                if not c:
                    cell_html.append(f'<div class="{css}"><i>missing {strat}</i></div>')
                    continue
                wav_rel = os.path.relpath(c["best_wav"], out_dir)
                m = c["agg_metrics"]
                metric_line = (
                    f'<div class="metric">'
                    f'<span class="label">{strat}</span>'
                    f'vad={m["vad_dist"]["mean"]:.3f}±{m["vad_dist"]["std"]:.3f} '
                    f'V_diff={m["valence_diff"]["mean"]:.3f} '
                    f'spk={m["speaker_sim"]["mean"]:.3f} '
                    f'CER={m["content_error"]["median"]:.3f}'
                    f'</div>'
                )
                cell_html.append(
                    f'<div class="{css}">{_audio_tag(wav_rel)}{metric_line}</div>'
                )
            rows.append(f'<td class="cell">{"".join(cell_html)}</td>')
        rows.append('</tr>')

    rows.append('</tbody></table>')
    rows.append('</body></html>')

    (out_dir / "index.html").write_text(head + "\n".join(rows), encoding="utf-8")


def _render_markdown(
    out_dir: Path,
    cells: list[dict],
    languages: list[tuple[str, str, str]],
    emotions: list[str],
    strategy_names: list[str],
    title: str,
):
    """Per-emotion table: one row per language, one (vad, spk, CER) triple per strategy."""
    by_key = {(c["lang_code"], c["emotion"], c["strategy"]): c for c in cells}
    md = [f"# {title}\n",
          f"Strategies: {', '.join('`'+s+'`' for s in strategy_names)}\n",
          "Wav files linked are best-of-reps by `vad_dist`. CER values are medians "
          "(robust to single-rep Whisper hallucinations).\n"]

    for emo in emotions:
        md.append(f"\n## {emo}\n")
        header = ["language"] + sum(
            ([f"{s} vad", f"{s} spk", f"{s} CER"] for s in strategy_names), []
        )
        md.append("| " + " | ".join(header) + " |")
        md.append("|" + "|".join(["---"] * len(header)) + "|")
        for code, full, _ in languages:
            row = [f"{full} ({code})"]
            for s in strategy_names:
                c = by_key.get((code, emo, s))
                if not c:
                    row.extend(["–", "–", "–"])
                    continue
                m = c["agg_metrics"]
                row.append(f'{m["vad_dist"]["mean"]:.3f}±{m["vad_dist"]["std"]:.3f}')
                row.append(f'{m["speaker_sim"]["mean"]:.3f}')
                row.append(f'{m["content_error"]["median"]:.3f}')
            md.append("| " + " | ".join(row) + " |")

    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Multi-language production sweep")
    ap.add_argument("--config",     default="configs/multilang.yaml", type=Path)
    ap.add_argument("--output-dir", default="outputs/multilang_sweep", type=Path)
    ap.add_argument("--reps",       type=int, default=None,
                    help="Override config reps")
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--scoring-config", default=None, type=Path,
                    help="Optional ovet config (default: configs/default.yaml)")
    ap.add_argument("--hf-home",    default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, encoding="utf-8") as f:
        mlc = yaml.safe_load(f)

    emotions       = mlc["emotions"]
    lang_order     = mlc["language_order"]
    lang_full_map  = mlc["language_full_name"]
    target_text    = mlc["target_text"]
    ref_audio_dir  = Path(mlc["ref_audio_dir"])
    ref_speaker    = mlc["ref_speaker"]
    base_lang_code = mlc["v_lang_base_language"]
    layers         = list(mlc["steering"]["layers"])
    probe_steps    = int(mlc["steering"]["num_step_probe"])
    reps           = int(args.reps if args.reps is not None else mlc["reps"])
    best_metric    = mlc["best_metric"]
    best_direction = mlc["best_direction"]

    # Strategies: list of {name, alpha, accent_alpha, projection_removal}.
    # Backwards-compat: if absent, use the legacy 2-cell setup.
    strategies = mlc.get("strategies") or [
        {"name": "baseline", "alpha": 0.0, "accent_alpha": 0.0,
         "projection_removal": False},
        {"name": "phase4",   "alpha": float(mlc.get("steering", {}).get("alpha", 1.0)),
         "accent_alpha": 0.0,
         "projection_removal": bool(mlc.get("steering", {}).get("projection_removal", True))},
    ]
    print(f"[ovet] strategies: {[s['name'] for s in strategies]}", flush=True)

    languages = [(code, lang_full_map[code], target_text[code]) for code in lang_order]
    base_lang_full = lang_full_map[base_lang_code]
    base_lang_text = target_text[base_lang_code]

    cfg = load_config(args.scoring_config)

    # ------------------------------------------------------------------
    print("[ovet] Loading analyzers ...", flush=True)
    emo_an, vad_an, pros_an = EmotionAnalyzer(), VADAnalyzer(), ProsodyAnalyzer()
    spk_an, asr_an          = SpeakerAnalyzer(), ASRAnalyzer()
    evaluator = CandidateEvaluator(emo_an, vad_an, pros_an, spk_an, asr_an)

    print("[ovet] Loading OmniVoice ...", flush=True)
    w = OmniVoiceWrapper(hf_home=args.hf_home)

    # ------------------------------------------------------------------
    # Reference audio per emotion + Whisper transcription
    # ------------------------------------------------------------------
    emo_refs: dict[str, tuple[Path, str]] = {}
    for emo in emotions:
        ref_path = ref_audio_dir / f"jvnv_{ref_speaker}_{emo}.wav"
        if not ref_path.exists():
            raise FileNotFoundError(f"missing JVNV ref: {ref_path}")
        ref_text = w.transcribe(ref_path, language=None)
        emo_refs[emo] = (ref_path, ref_text)
        print(f"[ovet] {emo}: {ref_path.name}  text={ref_text[:48]}...", flush=True)

    # Cache reference analyses (for evaluation against per-emotion ref).
    print("[ovet] Pre-computing reference features ...", flush=True)
    ref_features_per_emo = {emo: evaluator.analyze_reference(p)
                            for emo, (p, _) in emo_refs.items()}

    # ------------------------------------------------------------------
    # Build v_emo using synthetic-neutral (mean over emotions)
    # ------------------------------------------------------------------
    print(f"[ovet] Building v_emo at layers {layers} (synthetic neutral) ...", flush=True)
    v_emo_per_emo, synth_neutral = _build_synth_neutral_v_emo(
        w, emo_refs, layers, args.seed, probe_steps,
    )
    save_vectors(args.output_dir / "synth_neutral.npz", synth_neutral, meta={"layers": layers})
    for emo in emotions:
        save_vectors(args.output_dir / f"v_emo_{emo}.npz", v_emo_per_emo[emo],
                     meta={"emotion": emo, "neutral": "synthetic_mean", "layers": layers})

    # ------------------------------------------------------------------
    # Build per-language v_lang via OmniVoice self-generation
    # ------------------------------------------------------------------
    # Optional multi-speaker averaging: if v_lang_speakers lists 2+ JVNV
    # speakers, build per-speaker v_lang then average. This dilutes the
    # ref speaker's idiosyncratic JP coloring while preserving the
    # shared language-axis signal.
    v_lang_speaker_codes: list[str] = mlc.get("v_lang_speakers") or [ref_speaker]
    v_lang_clone_emo: str = mlc.get("v_lang_clone_emotion", "anger")

    cloning_refs: list[tuple[str, Path, str]] = []
    for spk in v_lang_speaker_codes:
        spk_ref_path = ref_audio_dir / f"jvnv_{spk}_{v_lang_clone_emo}.wav"
        if not spk_ref_path.exists():
            # Fall back to multi-speaker JVNV dir if main dir doesn't have this speaker
            alt = Path("baseline/jvnv_samples_multi") / f"jvnv_{spk}_{v_lang_clone_emo}.wav"
            if alt.exists():
                spk_ref_path = alt
            else:
                raise FileNotFoundError(
                    f"missing JVNV ref for v_lang speaker={spk} emo={v_lang_clone_emo}: "
                    f"tried {spk_ref_path} and {alt}"
                )
        spk_ref_text = w.transcribe(spk_ref_path, language=None)
        cloning_refs.append((spk, spk_ref_path, spk_ref_text))

    if len(cloning_refs) == 1:
        # Single-speaker (legacy) path — keep on-disk layout identical.
        spk_tag, cloning_ref_audio, cloning_ref_text = cloning_refs[0]
        print(f"[ovet] Building v_lang single-speaker={spk_tag} (base={base_lang_code}) ...", flush=True)
        v_lang_per_lang, lang_pair_paths = _build_per_language_v_lang(
            w,
            cloning_ref_audio=cloning_ref_audio,
            cloning_ref_text=cloning_ref_text,
            base_lang_code=base_lang_code,
            base_lang_full=base_lang_full,
            base_lang_text=base_lang_text,
            targets=languages,
            layers=layers,
            seed=args.seed,
            num_step=probe_steps,
            out_dir=args.output_dir / "lang_pair_clones",
        )
        v_lang_per_speaker = None
    else:
        print(
            f"[ovet] Building v_lang multi-speaker={v_lang_speaker_codes} "
            f"(base={base_lang_code}) ...", flush=True,
        )
        v_lang_per_lang, v_lang_per_speaker = _build_multispeaker_v_lang(
            w,
            cloning_refs=cloning_refs,
            base_lang_code=base_lang_code,
            base_lang_full=base_lang_full,
            base_lang_text=base_lang_text,
            targets=languages,
            layers=layers,
            seed=args.seed,
            num_step=probe_steps,
            out_dir=args.output_dir / "lang_pair_clones",
        )
        # Save per-speaker artifacts for inspection
        for spk_tag, vl_dict in v_lang_per_speaker.items():
            for code, vl in vl_dict.items():
                save_vectors(
                    args.output_dir / f"v_lang_{code}_{spk_tag}.npz", vl,
                    meta={"target": code, "base": base_lang_code,
                          "speaker": spk_tag, "layers": layers},
                )

    for code, vl in v_lang_per_lang.items():
        save_vectors(args.output_dir / f"v_lang_{code}.npz", vl,
                     meta={"target": code, "base": base_lang_code,
                           "speakers": v_lang_speaker_codes,
                           "clone_emotion": v_lang_clone_emo,
                           "layers": layers})

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------
    cells: list[dict] = []
    cands_dir = args.output_dir / "_all_candidates"
    cands_dir.mkdir(parents=True, exist_ok=True)
    per_lang_dir    = args.output_dir / "per_language"
    per_emotion_dir = args.output_dir / "per_emotion"
    for d in (per_lang_dir, per_emotion_dir):
        d.mkdir(parents=True, exist_ok=True)

    total_cells = len(languages) * len(emotions) * len(strategies)
    cell_idx = 0
    t0 = time.time()

    for lang_code, lang_full, lang_text in languages:
        for emo in emotions:
            ref_audio_path, ref_text = emo_refs[emo]
            ref_features = ref_features_per_emo[emo]
            v_emo_layers = v_emo_per_emo[emo]
            v_lang_layers = v_lang_per_lang[lang_code]

            for strat in strategies:
                strategy = strat["name"]
                s_alpha = float(strat.get("alpha", 0.0))
                s_accent = float(strat.get("accent_alpha", 0.0))
                s_proj = bool(strat.get("projection_removal", False))
                # Quality knobs (Phase 4+ for production)
                s_window_raw   = strat.get("step_window")
                s_step_window  = (tuple(s_window_raw)
                                  if s_window_raw is not None else None)
                s_norm_clip    = strat.get("norm_clip_factor")
                s_position_mask = bool(strat.get("position_mask", False))
                cell_idx += 1
                rep_records: list[dict] = []
                for rep in range(reps):
                    if s_alpha == 0.0 and s_accent == 0.0:
                        sc = None
                    else:
                        sc = SteeringConfig(
                            enabled=True, alpha=s_alpha,
                            layer_ids=layers,
                            emotion_vector=v_emo_layers,
                            language_vector=v_lang_layers,
                            projection_removal=s_proj,
                            language_alpha=s_accent,
                            step_window=s_step_window,
                            norm_clip_factor=s_norm_clip,
                            position_mask=s_position_mask,
                        )
                    torch.manual_seed(args.seed + rep)
                    audio = w.generate(
                        text=lang_text, language=lang_full,
                        ref_audio=str(ref_audio_path), ref_text=ref_text,
                        steering=sc,
                    )
                    wav_path = (
                        cands_dir
                        / f"{lang_code}__{emo}__{strategy}__rep{rep}.wav"
                    )
                    save_wav(wav_path, audio, w.SAMPLING_RATE)

                    scores = evaluator.evaluate(
                        wav_path, ref_features,
                        target_text=lang_text, target_language=lang_full,
                        target_emotion_label=emo,
                    )
                    total = compute_total_score(scores, cfg.scoring)
                    rep_records.append({
                        "rep": rep,
                        "wav_path": wav_path,
                        "scores": asdict(scores),
                        "total_score": total,
                    })

                # Pick best-of-reps
                if best_direction == "min":
                    best = min(rep_records, key=lambda r: r["scores"][best_metric])
                else:
                    best = max(rep_records, key=lambda r: r["scores"][best_metric])

                # Aggregate metrics across reps
                agg = {}
                for m in METRICS:
                    agg[m] = _agg([r["scores"][m] for r in rep_records])
                agg["total_score"] = _agg([r["total_score"] for r in rep_records])

                # Mirror best wav into per_language/<lang>/ and per_emotion/<emo>/
                lang_subdir = per_lang_dir / f"{lang_code}_{lang_full}"
                emo_subdir  = per_emotion_dir / emo
                lang_subdir.mkdir(parents=True, exist_ok=True)
                emo_subdir.mkdir(parents=True, exist_ok=True)
                lang_dst = lang_subdir / f"{emo}_{strategy}.wav"
                emo_dst  = emo_subdir  / f"{lang_code}_{strategy}.wav"
                if not lang_dst.exists():
                    wav, sr = load_wav(best["wav_path"])
                    save_wav(lang_dst, wav, sr)
                if not emo_dst.exists():
                    wav, sr = load_wav(best["wav_path"])
                    save_wav(emo_dst, wav, sr)

                cells.append({
                    "lang_code": lang_code, "lang_full": lang_full,
                    "emotion": emo, "strategy": strategy,
                    "best_wav": str(lang_dst),
                    "best_rep": best["rep"],
                    "agg_metrics": agg,
                    "rep_records": [
                        {"rep": r["rep"], "wav": str(r["wav_path"]),
                         "scores": r["scores"], "total": r["total_score"]}
                        for r in rep_records
                    ],
                })

                elapsed = time.time() - t0
                eta = elapsed / cell_idx * (total_cells - cell_idx)
                print(f"  [{cell_idx:>3}/{total_cells}] {lang_code}/{emo}/{strategy:<8}  "
                      f"vad={agg['vad_dist']['mean']:.3f}±{agg['vad_dist']['std']:.3f}  "
                      f"V_d={agg['valence_diff']['mean']:.3f}  "
                      f"spk={agg['speaker_sim']['mean']:.3f}  "
                      f"CER={agg['content_error']['mean']:.3f}  "
                      f"(elapsed {elapsed/60:.1f}m, eta {eta/60:.1f}m)",
                      flush=True)

    # ------------------------------------------------------------------
    # Persist artefacts
    # ------------------------------------------------------------------
    payload = {
        "config": mlc,
        "reps": reps,
        "seed": args.seed,
        "cells": cells,
        "ref_texts": {emo: t for emo, (_, t) in emo_refs.items()},
    }
    with open(args.output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(args.output_dir / "grid.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["lang_code", "emotion", "strategy", "best_rep",
                     "vad_mean", "vad_std",
                     "valence_diff_mean", "energy_std_ratio_mean",
                     "e2v_cos_mean", "speaker_sim_mean",
                     "content_error_mean", "total_score_mean"])
        for c in cells:
            m = c["agg_metrics"]
            wr.writerow([c["lang_code"], c["emotion"], c["strategy"], c["best_rep"],
                         f'{m["vad_dist"]["mean"]:.4f}',
                         f'{m["vad_dist"]["std"]:.4f}',
                         f'{m["valence_diff"]["mean"]:.4f}',
                         f'{m["energy_std_ratio"]["mean"]:.4f}',
                         f'{m["e2v_cos"]["mean"]:.4f}',
                         f'{m["speaker_sim"]["mean"]:.4f}',
                         f'{m["content_error"]["mean"]:.4f}',
                         f'{m["total_score"]["mean"]:.4f}'])

    title = f"Multi-language sweep — JVNV {ref_speaker}, reps={reps}"
    strategy_names = [s["name"] for s in strategies]
    _render_html(args.output_dir, cells, languages, emotions, strategy_names, title=title)
    _render_markdown(args.output_dir, cells, languages, emotions, strategy_names, title=title)

    elapsed = time.time() - t0
    print(f"\n[ovet] Done in {elapsed/60:.1f} min", flush=True)
    print(f"[ovet] Listening UI: {args.output_dir/'index.html'}", flush=True)
    print(f"[ovet] Per-language wavs: {per_lang_dir}/", flush=True)
    print(f"[ovet] Per-emotion wavs:  {per_emotion_dir}/", flush=True)


if __name__ == "__main__":
    main()
