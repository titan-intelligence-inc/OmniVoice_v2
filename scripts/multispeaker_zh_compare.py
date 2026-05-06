"""A/B compare phase4_accent_v2 with single-speaker (F1 only) vs
multi-speaker ([F1, F2, M1, M2] averaged) v_lang on zh outputs.

Reuses already-computed v_lang from:
  * single: outputs/multilang_F1_v2/v_lang_zh.npz
  * multi:  outputs/multispeaker_smoke/lang_pair_clones/<spk>/...
            (rebuilds the average from the per-speaker artifacts on
             demand if multispeaker_zh.npz isn't yet saved)

Generation: zh × {sad, happy, anger} × {single_v_lang, multi_v_lang} × N reps.
Reports CER and saves WAVs side-by-side for listening.

Usage:
    venv/bin/python scripts/multispeaker_zh_compare.py
"""
from __future__ import annotations
import os
import sys
import json
import statistics
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper, SteeringConfig    # noqa: E402
from ovet.cli.run_multilang import (                                    # noqa: E402
    _build_multispeaker_v_lang,
)
from ovet.analyzers.asr_analyzer import ASRAnalyzer                    # noqa: E402
from ovet.utils.io import save_wav                                     # noqa: E402


SPEAKERS  = ["F1", "F2", "M1", "M2"]
EMOTIONS  = ["sad", "happy", "anger"]
LAYERS    = [8, 12, 16]
TARGETS   = [
    ("ja", "Japanese", "今日の天気予報は曇り時々晴れです。"),
    ("zh", "Chinese",  "明天的天气预报是多云转晴。"),
]
ZH_TEXT   = TARGETS[1][2]
ZH_FULL   = TARGETS[1][1]
REPS      = 3


def _load_or_build_multispeaker_v_lang(w, refs, out_dir):
    """Load multispeaker v_lang for zh from disk if available, else build."""
    cache = out_dir / "v_lang_multi_zh.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        v = {int(k.split("_")[1]): np.asarray(d[k]) for k in d.files
             if k.startswith("layer_")}
        return v
    averaged, per_spk = _build_multispeaker_v_lang(
        w, cloning_refs=refs,
        base_lang_code="ja", base_lang_full="Japanese",
        base_lang_text=TARGETS[0][2],
        targets=TARGETS, layers=LAYERS, seed=0, num_step=4,
        out_dir=out_dir / "lang_pair_clones",
    )
    arr = {f"layer_{L}": averaged["zh"][L] for L in LAYERS}
    np.savez(cache, **arr)
    return averaged["zh"]


def _build_v_emo(w, refs, layers):
    """Quick v_emo build: synth_neutral = mean(h(emo refs)).
    For zh sweep we only need v_emo per emotion at the steering layers.
    """
    from ovet.omnivoice.steering import extract_layer_vectors
    h_per_emo = {}
    for emo in EMOTIONS:
        ref = next((p for s, p, _ in refs if s == "F1" and emo in p.name), None)
        if ref is None:
            for d in (Path("baseline/jvnv_samples"), Path("baseline/jvnv_samples_multi")):
                p = d / f"jvnv_F1_{emo}.wav"
                if p.exists(): ref = p; break
        ref_text = w.transcribe(ref, language=None)
        print(f"[zhcmp] probe v_emo for {emo}: {ref.name}", flush=True)
        h_per_emo[emo] = extract_layer_vectors(
            w, ref, layers, ref_text=ref_text, num_step=4, seed=0)
    synth_neutral = {L: np.stack([h_per_emo[e][L] for e in EMOTIONS]).mean(axis=0)
                     for L in layers}
    v_emo = {emo: {L: (h_per_emo[emo][L] - synth_neutral[L]).astype(np.float32)
                    for L in layers}
             for emo in EMOTIONS}
    return v_emo


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    out_dir = Path("outputs/multispeaker_zh_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[zhcmp] loading OmniVoice ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()

    # Build cloning refs (F1, F2, M1, M2 anger)
    refs = []
    for spk in SPEAKERS:
        for d in (Path("baseline/jvnv_samples_multi"), Path("baseline/jvnv_samples")):
            p = d / f"jvnv_{spk}_anger.wav"
            if p.exists():
                refs.append((spk, p, w.transcribe(p, language=None)))
                break

    # ---------- v_lang single (F1 only) ----------
    on_disk = Path("outputs/multilang_F1_v2/v_lang_zh.npz")
    if on_disk.exists():
        d = np.load(on_disk)
        v_lang_single = {int(k.split("_")[1]): np.asarray(d[k]) for k in d.files
                         if k.startswith("layer_")}
        # Only keep layers we're using; drop unsupported.
        v_lang_single = {L: v_lang_single[L] for L in LAYERS if L in v_lang_single}
        print(f"[zhcmp] loaded v_lang_single from disk: layers={list(v_lang_single.keys())}", flush=True)
    else:
        # Rebuild F1-only via multispeaker function with single ref
        averaged_f1, _ = _build_multispeaker_v_lang(
            w, cloning_refs=[refs[0]],
            base_lang_code="ja", base_lang_full="Japanese",
            base_lang_text=TARGETS[0][2],
            targets=TARGETS, layers=LAYERS, seed=0, num_step=4,
            out_dir=out_dir / "lang_pair_clones_single",
        )
        v_lang_single = averaged_f1["zh"]

    # ---------- v_lang multi ----------
    v_lang_multi = _load_or_build_multispeaker_v_lang(w, refs, out_dir)
    print(f"[zhcmp] loaded v_lang_multi: layers={list(v_lang_multi.keys())}", flush=True)

    # ---------- v_emo ----------
    v_emo = _build_v_emo(w, refs, LAYERS)

    # ---------- Sweep ----------
    rows = []
    for emo in EMOTIONS:
        ref = next(p for s, p, _ in refs if s == "F1")
        # Use F1's emotion ref (anger only available from refs list, build others)
        for d in (Path("baseline/jvnv_samples"), Path("baseline/jvnv_samples_multi")):
            cand = d / f"jvnv_F1_{emo}.wav"
            if cand.exists():
                ref = cand; break
        ref_text = w.transcribe(ref, language=None)
        print(f"\n[zhcmp] === emotion={emo} ===", flush=True)

        for variant_name, v_lang in [("single", v_lang_single), ("multi", v_lang_multi)]:
            cers = []
            for rep in range(REPS):
                sc = SteeringConfig(
                    enabled=True, alpha=1.0, layer_ids=LAYERS,
                    emotion_vector=v_emo[emo],
                    language_vector=v_lang,
                    projection_removal=True,
                    language_alpha=2.0,         # phase4_accent_v2
                    step_window=(0, 4),         # phase4_accent_v2
                )
                torch.manual_seed(rep)
                audio = w.generate(
                    text=ZH_TEXT, language=ZH_FULL,
                    ref_audio=str(ref), ref_text=ref_text,
                    steering=sc,
                )
                wav_path = out_dir / f"zh_{emo}_{variant_name}_rep{rep}.wav"
                save_wav(wav_path, audio, w.SAMPLING_RATE)
                cer = asr.content_error(wav_path, ZH_TEXT, language="chinese")
                cers.append(cer)
                print(f"  {variant_name} rep{rep}: CER={cer:.3f}  -> {wav_path.name}",
                      flush=True)
            rows.append({"emotion": emo, "variant": variant_name,
                         "cer_median": statistics.median(cers),
                         "cer_mean":   statistics.fmean(cers),
                         "cer_std":    statistics.pstdev(cers) if len(cers) > 1 else 0.0,
                         "cer_per_rep": cers})

    print("\n=== summary ===", flush=True)
    print(f"{'emotion':<8} {'variant':<8} {'CER med':>9} {'CER mean':>9} {'CER std':>8}")
    for r in rows:
        print(f"{r['emotion']:<8} {r['variant']:<8} "
              f"{r['cer_median']:>9.3f} {r['cer_mean']:>9.3f} {r['cer_std']:>8.3f}")

    # Aggregate by variant (across emotions)
    print("\n=== aggregate by variant (median over all emotions × reps) ===", flush=True)
    for variant_name in ("single", "multi"):
        all_cers = [c for r in rows if r["variant"] == variant_name
                    for c in r["cer_per_rep"]]
        print(f"  {variant_name}: median={statistics.median(all_cers):.3f}  "
              f"mean={statistics.fmean(all_cers):.3f}  "
              f"n={len(all_cers)}")

    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n[zhcmp] saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
