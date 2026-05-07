"""α/β sweep for the 1D LanguageAxisGrafter on zh.

Mirrors graft_alpha_beta_sweep.py but uses the rank-1 axis operator
instead of the rank-K subspace operator. Same metrics so we can put
the two side-by-side.

Comparison anchors at the end:
  * subspace results read from outputs/graft_alpha_beta_sweep/result.json
  * axis    results computed here
"""
from __future__ import annotations
import os
import re
import sys
import json
import statistics
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper                           # noqa: E402
from ovet.omnivoice.lang_graft import (                                       # noqa: E402
    LanguageAxisGrafter, load_axis_artifacts,
)
from ovet.analyzers.asr_analyzer import ASRAnalyzer                           # noqa: E402
from ovet.utils.io import save_wav                                            # noqa: E402


AXIS_NPZ      = Path("outputs/graft/zh_axis.npz")
JVNV_DIR      = Path("baseline/jvnv_samples")
LAYERS        = [8, 12]
STEP_WINDOW   = (0, 8)
EMOTION       = "happy"        # same focus emotion as subspace sweep
ZH_TEXT       = "明天的天气预报是多云转晴。"
ZH_FULL       = "Chinese"
REPS          = 3
SEED          = 0
OUT_DIR       = Path("outputs/graft_axis_sweep")

ALPHAS = [0.0, 0.3, 0.6, 1.0, 1.5]    # widened: axis op is less invasive
BETAS  = [0.3, 0.6, 1.0, 1.5, 2.0]    # widened upper end


KATAKANA_RE   = re.compile(r"[゠-ヿ]")
HIRAGANA_RE   = re.compile(r"[぀-ゟ]")
HALLU_PATTERN = re.compile(r"(.{1,4})\1{8,}")


def _strip_punct(s: str) -> str:
    return re.sub(r"[\s\.,!?。、！？・…]+", "", s).lower()


def prefix_match(hyp: str, target: str) -> int:
    a, b = _strip_punct(hyp), _strip_punct(target)
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def hallucination_score(hyp: str) -> bool:
    return bool(HALLU_PATTERN.search(hyp))


def katakana_count(hyp: str) -> int:
    return len(KATAKANA_RE.findall(hyp)) + len(HIRAGANA_RE.findall(hyp))


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[axis-sweep] loading OmniVoice + ASR ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()

    axis, target_c, meta = load_axis_artifacts(AXIS_NPZ)
    axis_subset = {L: axis[L] for L in LAYERS}
    target_subset = {L: target_c[L] for L in LAYERS}
    print(f"[axis-sweep] artifacts: layers={list(axis_subset.keys())} "
          f"target_c={ {L: f'{target_subset[L]:+.2f}' for L in LAYERS} }", flush=True)

    ref = JVNV_DIR / f"jvnv_F1_{EMOTION}.wav"
    ref_text = w.transcribe(ref, language=None)
    print(f"[axis-sweep] ref={ref.name}  text='{ref_text[:48]}...'", flush=True)
    target_chars = len(_strip_punct(ZH_TEXT))

    variants: list[tuple[str, dict]] = [("baseline", {"kind": "raw"})]
    for a in ALPHAS:
        for b in BETAS:
            if a == 0.0 and b == 0.0:
                continue
            variants.append((f"a{a:.1f}_b{b:.1f}",
                             {"kind": "graft", "a": a, "b": b}))
    print(f"[axis-sweep] variants: {len(variants)}  reps: {REPS}", flush=True)

    rows = []
    for vname, vcfg in variants:
        cers, prefixes, hallus, kanas, hyps = [], [], [], [], []
        for rep in range(REPS):
            torch.manual_seed(SEED + rep)
            if vcfg["kind"] == "raw":
                audio = w.model.generate(
                    text=ZH_TEXT, language=ZH_FULL,
                    ref_audio=str(ref), ref_text=ref_text,
                )[0]
            else:
                with LanguageAxisGrafter(
                    w.model,
                    axis_per_layer=axis_subset,
                    target_c_per_layer=target_subset,
                    remove_alpha=vcfg["a"], inject_beta=vcfg["b"],
                    step_window=STEP_WINDOW,
                ):
                    audio = w.model.generate(
                        text=ZH_TEXT, language=ZH_FULL,
                        ref_audio=str(ref), ref_text=ref_text,
                    )[0]
            wav_path = OUT_DIR / f"{vname}_rep{rep}.wav"
            save_wav(wav_path, audio, w.SAMPLING_RATE)
            hyp = asr.transcribe(wav_path, language="zh")
            cer = asr.content_error(wav_path, ZH_TEXT, language="zh")
            pm  = prefix_match(hyp, ZH_TEXT)
            hl  = hallucination_score(hyp)
            kk  = katakana_count(hyp)
            cers.append(cer); prefixes.append(pm); hallus.append(hl)
            kanas.append(kk); hyps.append(hyp)

        row = {
            "variant": vname,
            "cer_med":      statistics.median(cers),
            "cer_min":      min(cers),
            "prefix_max":   max(prefixes),
            "prefix_mean":  statistics.fmean(prefixes),
            "hallu_rate":   sum(hallus) / len(hallus),
            "kana_med":     statistics.median(kanas),
            "hyps":         hyps,
        }
        rows.append(row)
        flag = "🎯" if row["prefix_max"] >= 4 else (
            "·" if row["prefix_max"] >= 2 else (
                "⚠️" if row["hallu_rate"] >= 0.5 else "—"))
        print(f" {flag} {vname:<14} "
              f"cer_med={row['cer_med']:.2f}  "
              f"pfx_max={row['prefix_max']:>2}/{target_chars}  "
              f"pfx_mean={row['prefix_mean']:.1f}  "
              f"hallu={row['hallu_rate']:.0%}  "
              f"kana={row['kana_med']:>3}",
              flush=True)

    # Sort + display
    rows_sorted = sorted(rows, key=lambda r: (-r["prefix_max"], -r["prefix_mean"],
                                              r["hallu_rate"], r["cer_med"]))
    print("\n=== axis 1D grid (sorted by prefix_max desc, prefix_mean desc) ===", flush=True)
    print(f"{'variant':<16} {'pfx_max':>7} {'pfx_mean':>9} {'hallu':>6} "
          f"{'kana':>5} {'cer_med':>7} {'cer_min':>7}")
    for r in rows_sorted:
        print(f"{r['variant']:<16} "
              f"{r['prefix_max']:>7} {r['prefix_mean']:>9.1f} "
              f"{r['hallu_rate']:>5.0%} {r['kana_med']:>5} "
              f"{r['cer_med']:>7.2f} {r['cer_min']:>7.2f}")

    print("\n=== top-3 axis transcripts ===", flush=True)
    for r in rows_sorted[:3]:
        print(f"--- {r['variant']} (pfx_max={r['prefix_max']}, "
              f"pfx_mean={r['prefix_mean']:.1f}, hallu={r['hallu_rate']:.0%}, "
              f"kana={r['kana_med']}) ---")
        for h in r["hyps"]:
            print(f"  {h[:80]}")

    # Cross-experiment comparison
    sub_path = Path("outputs/graft_alpha_beta_sweep/result.json")
    if sub_path.exists():
        with open(sub_path) as f:
            sub_data = json.load(f)
        sub_top = sorted(sub_data["rows"],
                         key=lambda r: (-r["prefix_max"], -r["prefix_mean"]))[:3]
        print("\n=== compared to subspace top-3 (same emotion=happy) ===")
        print(f"{'op':<10} {'variant':<14} {'pfx_max':>7} {'pfx_mean':>9} "
              f"{'hallu':>6} {'kana':>5} {'cer_med':>7}")
        for r in sub_top:
            print(f"{'subspace':<10} {r['variant']:<14} "
                  f"{r['prefix_max']:>7} {r['prefix_mean']:>9.1f} "
                  f"{r['hallu_rate']:>5.0%} {r['kana_med']:>5} "
                  f"{r['cer_med']:>7.2f}")
        for r in rows_sorted[:3]:
            print(f"{'axis(1D)':<10} {r['variant']:<14} "
                  f"{r['prefix_max']:>7} {r['prefix_mean']:>9.1f} "
                  f"{r['hallu_rate']:>5.0%} {r['kana_med']:>5} "
                  f"{r['cer_med']:>7.2f}")

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"emotion": EMOTION, "step_window": list(STEP_WINDOW),
                   "layers": LAYERS, "reps": REPS,
                   "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\n[axis-sweep] saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
