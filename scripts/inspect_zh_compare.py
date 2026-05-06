"""Inspect Whisper transcriptions of the zh single vs multi v_lang outputs.

The CER comparison saturated at the cap (~2.0), so visual transcripts are
more informative for "which variant produces something more Chinese-like".
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from collections import defaultdict

import statistics

sys.path.insert(0, "src")
from ovet.analyzers.asr_analyzer import ASRAnalyzer        # noqa: E402


WAV_DIR = Path("outputs/multispeaker_zh_compare")
TARGET = "明天的天气预报是多云转晴。"


def main():
    asr = ASRAnalyzer()
    rows: list[dict] = []
    grouped = defaultdict(list)
    for wav in sorted(WAV_DIR.glob("zh_*_*.wav")):
        # zh_{emo}_{variant}_rep{N}.wav
        parts = wav.stem.split("_")
        emo, variant, rep_str = parts[1], parts[2], parts[3]
        # Force Chinese language hint
        hyp_zh = asr.transcribe(wav, language="zh")
        hyp_auto = asr.transcribe(wav, language=None)
        rows.append({
            "wav": wav.name, "emotion": emo, "variant": variant,
            "rep": int(rep_str.replace("rep", "")),
            "transcribed_zh_hint": hyp_zh,
            "transcribed_autodetect": hyp_auto,
        })
        grouped[(emo, variant)].append(hyp_zh)
        print(f"{emo:<6} {variant:<7} rep{rep_str.replace('rep','')}: ", flush=True)
        print(f"  zh-hint:  {hyp_zh}", flush=True)
        print(f"  auto:     {hyp_auto}", flush=True)
        print(flush=True)

    # Save raw transcripts
    out_json = WAV_DIR / "transcripts.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"target": TARGET, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"saved -> {out_json}")

    # Cross-rep similarity: how stable are transcriptions per cell?
    print("\n=== sample length per (emotion, variant) ===")
    for (emo, variant), hyps in sorted(grouped.items()):
        lens = [len(h) for h in hyps]
        print(f"  {emo:<7} {variant:<7}  hyp lens={lens}  median={statistics.median(lens)}")


if __name__ == "__main__":
    main()
