"""A1 — test Pinyin input variants for zh on OmniVoice.

Hypothesis: providing tone information in the input text directly
(via Pinyin diacritics or numeric tone markers) may help the LLM
encode tones explicitly into the hidden state at text-token positions,
relieving the F1-pitch-accent-overrides-tone problem.

Variants:
  hanzi      "明天的天气预报是多云转晴。"
  pinyin_d   "míng tiān de tiān qì yù bào shì duō yún zhuǎn qíng 。"
  pinyin_n   "ming2 tian1 de5 tian1 qi4 yu4 bao4 shi4 duo1 yun2 zhuan3 qing2 。"
  pinyin_x   lazy pinyin (no tone)  -- control: shows whether *tone*
                                      info specifically helps, vs just
                                      "input is now Latin alphabet"

Tested on F1 ref × 2 emotions × 3 reps. Whisper auto-detect language
(no language hint) so it can identify whatever the model produces
correctly.

Cost: 4 variants × 2 emo × 3 reps = 24 generations, ~3 min.
"""
from __future__ import annotations
import os, re, sys, json, statistics
from pathlib import Path

import numpy as np
import pypinyin
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper                            # noqa: E402
from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402


JVNV_DIR     = Path("baseline/jvnv_samples")
EMOTIONS     = ["sad", "anger"]
ZH_FULL      = "Chinese"
ZH_HANZI     = "明天的天气预报是多云转晴。"
REPS         = 3
SEED         = 0
OUT_DIR      = Path("outputs/pinyin_input_test")


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


def to_pinyin_diacritic(text: str) -> str:
    return " ".join(pypinyin.lazy_pinyin(text, style=pypinyin.Style.TONE))

def to_pinyin_numeric(text: str) -> str:
    return " ".join(pypinyin.lazy_pinyin(
        text, style=pypinyin.Style.TONE3, neutral_tone_with_five=True))

def to_pinyin_lazy(text: str) -> str:
    return " ".join(pypinyin.lazy_pinyin(text))


VARIANTS = [
    ("hanzi",     ZH_HANZI),
    ("pinyin_d",  to_pinyin_diacritic(ZH_HANZI)),
    ("pinyin_n",  to_pinyin_numeric(ZH_HANZI)),
    ("pinyin_x",  to_pinyin_lazy(ZH_HANZI)),
]


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[py] loading OmniVoice + ASR ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()

    print("\n=== input variants (target_text fed to OmniVoice) ===")
    for name, txt in VARIANTS:
        print(f"  {name}:   {txt}")

    rows = []
    for emo in EMOTIONS:
        ref = JVNV_DIR / f"jvnv_F1_{emo}.wav"
        ref_text = w.transcribe(ref, language=None)
        print(f"\n[py] === emo={emo} ref={ref.name} ===", flush=True)
        for vname, target_text in VARIANTS:
            cers, prefixes, hallus, kanas, hyps_zh, hyps_auto = [], [], [], [], [], []
            for rep in range(REPS):
                torch.manual_seed(SEED + rep)
                audio = w.model.generate(
                    text=target_text, language=ZH_FULL,
                    ref_audio=str(ref), ref_text=ref_text,
                )[0]
                wav_path = OUT_DIR / f"{emo}_{vname}_rep{rep}.wav"
                save_wav(wav_path, audio, w.SAMPLING_RATE)
                # Two transcriptions: forced-zh and auto-detect.
                hyp_zh   = asr.transcribe(wav_path, language="zh")
                hyp_auto = asr.transcribe(wav_path, language=None)
                # CER vs the canonical Hanzi (always — that's the
                # "intended" text regardless of how we encoded it).
                cer = asr.content_error(wav_path, ZH_HANZI, language="zh")
                cers.append(cer); prefixes.append(prefix_match(hyp_zh, ZH_HANZI))
                hallus.append(hallu(hyp_zh)); kanas.append(kana(hyp_zh))
                hyps_zh.append(hyp_zh); hyps_auto.append(hyp_auto)
            rows.append({
                "emotion": emo, "variant": vname,
                "cer_med": statistics.median(cers),
                "prefix_max": max(prefixes),
                "prefix_mean": statistics.fmean(prefixes),
                "hallu_rate": sum(hallus) / len(hallus),
                "kana_med": statistics.median(kanas),
                "hyps_zh": hyps_zh, "hyps_auto": hyps_auto,
            })
            r = rows[-1]
            flag = "🎯" if r["prefix_max"] >= 4 else (
                "·" if r["prefix_max"] >= 2 else (
                    "⚠️" if r["hallu_rate"] >= 0.5 else "—"))
            print(f"  {flag} {vname:<12} pfx={r['prefix_max']}/{r['prefix_mean']:.1f}  "
                  f"hallu={r['hallu_rate']:.0%}  kana={r['kana_med']}  "
                  f"cer_med={r['cer_med']:.2f}", flush=True)
            for h_zh, h_au in zip(hyps_zh, hyps_auto):
                print(f"      zh-hint:  {h_zh[:75]}")
                print(f"      auto:     {h_au[:75]}")

    by_v: dict[str, dict] = {}
    for r in rows:
        d = by_v.setdefault(r["variant"], {"pfx": [], "kana": [], "cer": [], "hallu": []})
        d["pfx"].append(r["prefix_mean"])
        d["kana"].append(r["kana_med"])
        d["cer"].append(r["cer_med"])
        d["hallu"].append(r["hallu_rate"])

    print("\n=== aggregate (2 emotions × 3 reps each) ===", flush=True)
    print(f"{'variant':<14} {'pfx_mean':>9} {'kana_med':>9} {'cer_med':>9} {'hallu':>6}")
    for v in sorted(by_v.keys(), key=lambda k: -statistics.fmean(by_v[k]["pfx"])):
        print(f"{v:<14} "
              f"{statistics.fmean(by_v[v]['pfx']):>9.2f} "
              f"{statistics.fmean(by_v[v]['kana']):>9.1f} "
              f"{statistics.fmean(by_v[v]['cer']):>9.2f} "
              f"{statistics.fmean(by_v[v]['hallu']):>5.0%}")

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "config": {
            "emotions": EMOTIONS, "reps": REPS,
            "target_hanzi": ZH_HANZI,
            "variants": [{"name": n, "text": t} for n, t in VARIANTS],
        }}, f, indent=2, ensure_ascii=False)
    print(f"\n[py] saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
