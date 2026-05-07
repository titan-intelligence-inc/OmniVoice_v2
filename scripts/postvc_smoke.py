"""Smoke test for OpenVoice v2 ToneColorConverter post-processing.

Loads ToneColorConverter, runs all 3 modes on an existing OmniVoice zh
output, and reports:
  * shape / sample-rate sanity
  * Whisper transcription before/after (proxy for accent reduction)
  * audio file sizes (sanity)

Reads from outputs/multispeaker_zh_compare/ (created by the previous
multispeaker A/B sweep).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "src")
from ovet.postprocessing.openvoice_vc import OpenVoicePostVC, VCConfig    # noqa: E402
from ovet.analyzers.asr_analyzer import ASRAnalyzer                       # noqa: E402


REF_AUDIO = Path("baseline/jvnv_samples/jvnv_F1_anger.wav")
INPUT_DIR = Path("outputs/multispeaker_zh_compare")
OUT_DIR   = Path("outputs/postvc_smoke")
TARGET_TEXT = "明天的天气预报是多云转晴。"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[postvc] loading ToneColorConverter ...", flush=True)
    pvc = OpenVoicePostVC()

    # Pick a few cells: prefer non-hallucinated single-speaker reps.
    cells = sorted(INPUT_DIR.glob("zh_anger_single_rep[12].wav"))
    cells += sorted(INPUT_DIR.glob("zh_sad_multi_rep[12].wav"))
    print(f"[postvc] testing on {len(cells)} input files", flush=True)

    asr = ASRAnalyzer()
    rows = []
    for src in cells:
        wav, sr = sf.read(src)
        assert sr == 24000, f"expected 24kHz, got {sr}"
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        hyp_orig = asr.transcribe(src, language="zh")
        print(f"\n[postvc] === {src.name} ===")
        print(f"  orig:           {hyp_orig}")

        for mode in ("identity_pass", "F1_to_native", "native_to_F1"):
            cfg = VCConfig(
                enabled=True,
                mode=mode,
                target_language_se="zh.pth",
                speaker_ref_audio=REF_AUDIO,
                tau=0.3,
            )
            try:
                out_audio = pvc.convert(wav, cfg)
                out_path = OUT_DIR / f"{src.stem}__{mode}.wav"
                sf.write(out_path, out_audio, 24000)
                hyp = asr.transcribe(out_path, language="zh")
                print(f"  {mode:<14} {hyp}")
                rows.append({"src": src.name, "mode": mode,
                             "out": out_path.name, "hyp": hyp})
            except Exception as e:
                print(f"  {mode:<14} FAILED: {type(e).__name__}: {e}")
                rows.append({"src": src.name, "mode": mode, "error": str(e)})

    # Save raw transcripts
    import json
    with open(OUT_DIR / "transcripts.json", "w", encoding="utf-8") as f:
        json.dump({"target": TARGET_TEXT, "rows": rows}, f,
                  indent=2, ensure_ascii=False)
    print(f"\n[postvc] saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
