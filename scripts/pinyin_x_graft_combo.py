"""Combine lazy Pinyin (A1 winner) with axis graft.

Question: does graft still help when input is pinyin_x, or has pinyin_x
already captured most of the gain we were getting from graft?

Variants (3 emotions × 3 reps each):
  hanzi_baseline    Hanzi input,  no graft       (= original baseline)
  hanzi_graft       Hanzi input,  graft a=b=1
  pinyin_baseline   pinyin_x input, no graft     (= A1 winner)
  pinyin_graft      pinyin_x input, graft a=b=1  (combo)
"""
from __future__ import annotations
import os, re, sys, json, statistics
from pathlib import Path

import numpy as np
import pypinyin
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper                            # noqa: E402
from ovet.omnivoice.lang_graft import (                                        # noqa: E402
    LanguageAxisGrafter, load_axis_artifacts,
)
from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402


JVNV_DIR     = Path("baseline/jvnv_samples")
EMOTIONS     = ["sad", "happy", "anger"]
LAYERS       = [8, 12]
STEP_WINDOW  = (0, 8)
ZH_FULL      = "Chinese"
ZH_HANZI     = "明天的天气预报是多云转晴。"
ZH_PINYIN_X  = " ".join(pypinyin.lazy_pinyin(ZH_HANZI))
REPS         = 3
SEED         = 0
OUT_DIR      = Path("outputs/pinyin_x_graft_combo")
AXIS_NPZ     = Path("outputs/graft/zh_axis.npz")


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


VARIANTS = [
    ("hanzi_baseline",  {"text": ZH_HANZI,    "graft": False}),
    ("hanzi_graft",     {"text": ZH_HANZI,    "graft": True}),
    ("pinyin_baseline", {"text": ZH_PINYIN_X, "graft": False}),
    ("pinyin_graft",    {"text": ZH_PINYIN_X, "graft": True}),
]


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[combo] loading OmniVoice + ASR ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()

    axis, target_c, _ = load_axis_artifacts(AXIS_NPZ)
    axis_subset = {L: axis[L] for L in LAYERS}
    target_subset = {L: target_c[L] for L in LAYERS}

    print(f"\nHanzi:  {ZH_HANZI}")
    print(f"Pinyin: {ZH_PINYIN_X}")

    rows = []
    for emo in EMOTIONS:
        ref = JVNV_DIR / f"jvnv_F1_{emo}.wav"
        ref_text = w.transcribe(ref, language=None)
        print(f"\n[combo] === emo={emo} ===", flush=True)
        for vname, vcfg in VARIANTS:
            cers, prefixes, hallus, kanas, hyps = [], [], [], [], []
            for rep in range(REPS):
                torch.manual_seed(SEED + rep)
                kwargs = dict(text=vcfg["text"], language=ZH_FULL,
                              ref_audio=str(ref), ref_text=ref_text)
                if vcfg["graft"]:
                    with LanguageAxisGrafter(
                        w.model,
                        axis_per_layer=axis_subset,
                        target_c_per_layer=target_subset,
                        remove_alpha=1.0, inject_beta=1.0,
                        step_window=STEP_WINDOW,
                    ):
                        audio = w.model.generate(**kwargs)[0]
                else:
                    audio = w.model.generate(**kwargs)[0]
                wav_path = OUT_DIR / f"{emo}_{vname}_rep{rep}.wav"
                save_wav(wav_path, audio, w.SAMPLING_RATE)
                hyp = asr.transcribe(wav_path, language="zh")
                cer = asr.content_error(wav_path, ZH_HANZI, language="zh")
                cers.append(cer); prefixes.append(prefix_match(hyp, ZH_HANZI))
                hallus.append(hallu(hyp)); kanas.append(kana(hyp))
                hyps.append(hyp)
            rows.append({
                "emotion": emo, "variant": vname,
                "cer_med": statistics.median(cers),
                "prefix_max": max(prefixes),
                "prefix_mean": statistics.fmean(prefixes),
                "hallu_rate": sum(hallus) / len(hallus),
                "kana_med": statistics.median(kanas),
                "hyps": hyps,
            })
            r = rows[-1]
            flag = "🎯" if r["prefix_max"] >= 4 else (
                "·" if r["prefix_max"] >= 2 else (
                    "⚠️" if r["hallu_rate"] >= 0.5 else "—"))
            print(f"  {flag} {vname:<18} pfx={r['prefix_max']}/{r['prefix_mean']:.1f}  "
                  f"hallu={r['hallu_rate']:.0%}  kana={r['kana_med']}  "
                  f"cer_med={r['cer_med']:.2f}", flush=True)
            for h in hyps:
                print(f"      {h[:75]}")

    by_v: dict[str, dict] = {}
    for r in rows:
        d = by_v.setdefault(r["variant"], {"pfx": [], "kana": [], "cer": [], "hallu": []})
        d["pfx"].append(r["prefix_mean"])
        d["kana"].append(r["kana_med"])
        d["cer"].append(r["cer_med"])
        d["hallu"].append(r["hallu_rate"])

    print("\n=== aggregate (3 emotions × 3 reps each) ===", flush=True)
    print(f"{'variant':<18} {'pfx_mean':>9} {'kana_med':>9} {'cer_med':>9} {'hallu':>6}")
    for v in sorted(by_v.keys(), key=lambda k: -statistics.fmean(by_v[k]["pfx"])):
        print(f"{v:<18} "
              f"{statistics.fmean(by_v[v]['pfx']):>9.2f} "
              f"{statistics.fmean(by_v[v]['kana']):>9.1f} "
              f"{statistics.fmean(by_v[v]['cer']):>9.2f} "
              f"{statistics.fmean(by_v[v]['hallu']):>5.0%}")

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "config": {
            "emotions": EMOTIONS, "reps": REPS,
            "hanzi": ZH_HANZI, "pinyin_x": ZH_PINYIN_X,
        }}, f, indent=2, ensure_ascii=False)
    print(f"\n[combo] saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
