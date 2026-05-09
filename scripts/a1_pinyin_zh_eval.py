"""A1 (lazy Pinyin input) zh CER measurement under the new eval recipe.

Measures the gap from current best (Pinyin input) to the 20% CER target,
using:
  * Median CER as the primary metric (robust to Whisper hallucination tail).
  * Trad/Simp-folded normalization (Whisper sometimes emits Traditional).
  * reps = 8 per cell (matches RECOMMENDED_REPS — σ on native zh ≈ 14%pt
    means smaller rep counts cannot discriminate sub-5%pt effects).
  * 6 emotions × 8 reps = 48 reps per variant.

Variants:
  hanzi_baseline   raw OmniVoice with Hanzi input (current production)
  pinyin_x         OmniVoice with lazy Pinyin input (A1)

Output:
  * Per-(variant, emotion) median + IQR.
  * Aggregate median across all 48 reps per variant.
  * Gap-to-20% in percentage points.
"""
from __future__ import annotations
import os, sys, json, statistics
from pathlib import Path

import numpy as np
import pypinyin
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper                            # noqa: E402
from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402
from ovet.evaluation.zh_eval import (                                          # noqa: E402
    aggregate_zh_cell, RECOMMENDED_REPS, format_cell_row,
)


JVNV_DIR     = Path("baseline/jvnv_samples")
EMOTIONS     = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
ZH_FULL      = "Chinese"
ZH_HANZI     = "明天的天气预报是多云转晴。"
ZH_PINYIN_X  = " ".join(pypinyin.lazy_pinyin(ZH_HANZI))
REPS         = RECOMMENDED_REPS         # = 8
SEED         = 0
OUT_DIR      = Path("outputs/a1_pinyin_zh_eval")
TARGET_CER   = 0.20


VARIANTS = [
    ("hanzi_baseline", ZH_HANZI),
    ("pinyin_x",       ZH_PINYIN_X),
]


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[a1-eval] loading OmniVoice + ASR ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()

    print(f"\nHanzi target  : {ZH_HANZI}")
    print(f"Pinyin target : {ZH_PINYIN_X}")
    print(f"emotions={len(EMOTIONS)}  reps={REPS}  total_reps_per_variant="
          f"{len(EMOTIONS) * REPS}", flush=True)

    cells: list[dict] = []
    flat_cers: dict[str, list[float]] = {v: [] for v, _ in VARIANTS}

    for vname, target_text in VARIANTS:
        print(f"\n[a1-eval] === variant={vname} ===", flush=True)
        for emo in EMOTIONS:
            ref = JVNV_DIR / f"jvnv_F1_{emo}.wav"
            ref_text = w.transcribe(ref, language=None)
            hyps: list[str] = []
            wav_paths: list[Path] = []
            for rep in range(REPS):
                torch.manual_seed(SEED + rep)
                audio = w.model.generate(
                    text=target_text, language=ZH_FULL,
                    ref_audio=str(ref), ref_text=ref_text,
                )[0]
                wav_path = OUT_DIR / f"{vname}_{emo}_rep{rep}.wav"
                save_wav(wav_path, audio, w.SAMPLING_RATE)
                hyp = asr.transcribe(wav_path, language="zh")
                hyps.append(hyp)
                wav_paths.append(wav_path)
            stats = aggregate_zh_cell(hyps, ZH_HANZI)
            cells.append({
                "variant": vname, "emotion": emo,
                "stats": stats.__dict__,
            })
            # Accumulate per-rep CERs for the variant-level aggregate.
            from ovet.evaluation.zh_eval import cer_zh
            flat_cers[vname].extend([cer_zh(ZH_HANZI, h) for h in hyps])

            print(f"  {emo:<9} {format_cell_row('', stats)}", flush=True)

    print("\n=== per-(variant, emotion) ===", flush=True)
    print(f"{'variant':<18} {'emotion':<9} {'cer_med':>8} "
          f"{'cer_q1':>8} {'cer_q3':>8} {'σ':>7} {'hallu':>6} {'pfx':>5}",
          flush=True)
    for c in cells:
        s = c["stats"]
        cers = [None]    # placeholder; reconstructed from raw CER list below
        # We don't have per-rep CER list in stats; recompute from hyps.
        from ovet.evaluation.zh_eval import cer_zh
        per_rep = [cer_zh(ZH_HANZI, h) for h in s["hyps"]]
        q1 = statistics.quantiles(per_rep, n=4)[0] if len(per_rep) >= 4 else min(per_rep)
        q3 = statistics.quantiles(per_rep, n=4)[2] if len(per_rep) >= 4 else max(per_rep)
        print(f"{c['variant']:<18} {c['emotion']:<9} "
              f"{s['cer_median']:>8.3f} {q1:>8.3f} {q3:>8.3f} "
              f"{s['cer_std']:>7.3f} {s['hallu_rate']:>5.0%} "
              f"{s['prefix_mean']:>5.1f}", flush=True)

    print("\n=== variant aggregate (48 reps each) ===", flush=True)
    print(f"{'variant':<18} {'cer_med':>8} {'cer_q1':>8} {'cer_q3':>8} "
          f"{'cer_mean':>9} {'σ':>7} {'gap_to_20%':>11}",
          flush=True)
    summary = []
    for v, _ in VARIANTS:
        cers = flat_cers[v]
        med  = statistics.median(cers)
        q1   = statistics.quantiles(cers, n=4)[0]
        q3   = statistics.quantiles(cers, n=4)[2]
        mean = statistics.fmean(cers)
        std  = statistics.pstdev(cers)
        gap  = (med - TARGET_CER) * 100
        summary.append({
            "variant": v, "n": len(cers),
            "cer_median": med, "cer_q1": q1, "cer_q3": q3,
            "cer_mean": mean, "cer_std": std, "gap_pp": gap,
        })
        print(f"{v:<18} {med:>8.3f} {q1:>8.3f} {q3:>8.3f} "
              f"{mean:>9.3f} {std:>7.3f} {gap:>+10.1f}pp",
              flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "emotions": EMOTIONS, "reps": REPS, "target_cer": TARGET_CER,
                "hanzi": ZH_HANZI, "pinyin": ZH_PINYIN_X,
            },
            "cells": cells,
            "variant_summary": summary,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[a1-eval] saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
