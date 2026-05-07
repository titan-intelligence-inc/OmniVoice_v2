"""Validate the axis-graft optimum on sad + anger.

Drops to the best 2 axis configs found on happy + a baseline + the best
subspace config for cross-comparison. 3 reps per cell.
"""
from __future__ import annotations
import os, re, sys, json, statistics
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper                        # noqa: E402
from ovet.omnivoice.lang_graft import (                                    # noqa: E402
    LanguageAxisGrafter, load_axis_artifacts,
    LanguageGrafter, load_graft_artifacts,
)
from ovet.analyzers.asr_analyzer import ASRAnalyzer                        # noqa: E402
from ovet.utils.io import save_wav                                         # noqa: E402


JVNV_DIR     = Path("baseline/jvnv_samples")
LAYERS       = [8, 12]
STEP_WINDOW  = (0, 8)
EMOTIONS     = ["sad", "anger"]
ZH_TEXT      = "明天的天气预报是多云转晴。"
ZH_FULL      = "Chinese"
REPS         = 3
SEED         = 0
OUT_DIR      = Path("outputs/graft_axis_validate")

KATAKANA_RE   = re.compile(r"[゠-ヿ]")
HIRAGANA_RE   = re.compile(r"[぀-ゟ]")
HALLU_PATTERN = re.compile(r"(.{1,4})\1{8,}")


def _strip(s):  return re.sub(r"[\s\.,!?。、！？・…]+", "", s).lower()
def prefix_match(h, t):
    a, b = _strip(h), _strip(t); n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]: return i
    return n
def hallu(h): return bool(HALLU_PATTERN.search(h))
def kana(h):  return len(KATAKANA_RE.findall(h)) + len(HIRAGANA_RE.findall(h))


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[validate] loading OmniVoice + ASR ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()

    axis, target_c, _ = load_axis_artifacts(Path("outputs/graft/zh_axis.npz"))
    Q_per, T_per, _   = load_graft_artifacts(Path("outputs/graft/zh.npz"))
    axis_ss = {L: axis[L] for L in LAYERS}; tc_ss = {L: target_c[L] for L in LAYERS}
    Q_ss = {L: Q_per[L] for L in LAYERS}; T_ss = {L: T_per[L] for L in LAYERS}

    # Best from sweeps + baseline.
    variants = [
        ("baseline",                  {"kind": "raw"}),
        ("subspace_a0.0_b0.6",        {"kind": "subspace", "a": 0.0, "b": 0.6}),
        ("axis_a1.0_b1.0",            {"kind": "axis", "a": 1.0, "b": 1.0}),
        ("axis_a1.5_b0.6",            {"kind": "axis", "a": 1.5, "b": 0.6}),
    ]

    rows = []
    for emo in EMOTIONS:
        ref = JVNV_DIR / f"jvnv_F1_{emo}.wav"
        ref_text = w.transcribe(ref, language=None)
        print(f"\n[validate] === {emo} ref={ref.name} ===", flush=True)
        for vname, vcfg in variants:
            cers, prefixes, hallus, kanas, hyps = [], [], [], [], []
            for rep in range(REPS):
                torch.manual_seed(SEED + rep)
                kwargs = dict(text=ZH_TEXT, language=ZH_FULL,
                              ref_audio=str(ref), ref_text=ref_text)
                if vcfg["kind"] == "raw":
                    audio = w.model.generate(**kwargs)[0]
                elif vcfg["kind"] == "subspace":
                    with LanguageGrafter(
                        w.model, subspace_per_layer=Q_ss,
                        target_per_layer=T_ss,
                        remove_alpha=vcfg["a"], inject_beta=vcfg["b"],
                        step_window=STEP_WINDOW,
                    ):
                        audio = w.model.generate(**kwargs)[0]
                else:
                    with LanguageAxisGrafter(
                        w.model, axis_per_layer=axis_ss,
                        target_c_per_layer=tc_ss,
                        remove_alpha=vcfg["a"], inject_beta=vcfg["b"],
                        step_window=STEP_WINDOW,
                    ):
                        audio = w.model.generate(**kwargs)[0]
                wav_path = OUT_DIR / f"{emo}_{vname}_rep{rep}.wav"
                save_wav(wav_path, audio, w.SAMPLING_RATE)
                hyp = asr.transcribe(wav_path, language="zh")
                cer = asr.content_error(wav_path, ZH_TEXT, language="zh")
                cers.append(cer); prefixes.append(prefix_match(hyp, ZH_TEXT))
                hallus.append(hallu(hyp)); kanas.append(kana(hyp))
                hyps.append(hyp)
            row = {
                "emotion": emo, "variant": vname,
                "cer_med": statistics.median(cers),
                "prefix_max": max(prefixes),
                "prefix_mean": statistics.fmean(prefixes),
                "hallu_rate": sum(hallus) / len(hallus),
                "kana_med": statistics.median(kanas),
                "hyps": hyps,
            }
            rows.append(row)
            flag = "🎯" if row["prefix_max"] >= 4 else (
                "·" if row["prefix_max"] >= 2 else (
                    "⚠️" if row["hallu_rate"] >= 0.5 else "—"))
            print(f"  {flag} {vname:<22} pfx={row['prefix_max']}/{row['prefix_mean']:.1f}  "
                  f"hallu={row['hallu_rate']:.0%}  kana={row['kana_med']}  "
                  f"cer_med={row['cer_med']:.2f}",
                  flush=True)
            for h in hyps:
                print(f"      {h[:70]}")

    print("\n=== aggregate (mean over both emotions, 6 reps each) ===", flush=True)
    by_v: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        d = by_v.setdefault(r["variant"], {"pfx": [], "kana": [], "cer": [], "hallu": []})
        d["pfx"].append(r["prefix_mean"])
        d["kana"].append(r["kana_med"])
        d["cer"].append(r["cer_med"])
        d["hallu"].append(r["hallu_rate"])
    print(f"{'variant':<24} {'pfx_mean':>9} {'kana_med':>9} {'cer_med':>9} {'hallu':>6}")
    for v in sorted(by_v.keys()):
        print(f"{v:<24} "
              f"{statistics.fmean(by_v[v]['pfx']):>9.2f} "
              f"{statistics.fmean(by_v[v]['kana']):>9.1f} "
              f"{statistics.fmean(by_v[v]['cer']):>9.2f} "
              f"{statistics.fmean(by_v[v]['hallu']):>5.0%}")

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "config": {
            "emotions": EMOTIONS, "reps": REPS, "layers": LAYERS,
            "step_window": list(STEP_WINDOW),
        }}, f, indent=2, ensure_ascii=False)
    print(f"\n[validate] saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
