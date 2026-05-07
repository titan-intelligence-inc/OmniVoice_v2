"""Quick smoke test: per-language axis-graft on a single (lang, emotion).

For each non-zh target language, generate baseline vs axis-graft (a=b=1)
and report Whisper transcription. Confirms the per-language artifact
loads cleanly and produces audio without crashing.

Cost: ~9 langs × 2 variants × 2 reps = 36 generations, ~3 min.
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper                            # noqa: E402
from ovet.omnivoice.lang_graft import (                                        # noqa: E402
    LanguageAxisGrafter, load_axis_artifacts,
)
from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402


JVNV_DIR     = Path("baseline/jvnv_samples")
LAYERS       = [8, 12]
STEP_WINDOW  = (0, 8)
EMOTION      = "anger"
REPS         = 2
SEED         = 0
OUT_DIR      = Path("outputs/graft_smoke_per_lang")

LANG_TEXT = {
    "en": ("English",    "The weather forecast for tomorrow is partly cloudy."),
    "ko": ("Korean",     "내일 일기 예보는 가끔 구름이 끼겠습니다."),
    "es": ("Spanish",    "El pronóstico del tiempo para mañana es parcialmente nublado."),
    "fr": ("French",     "Les prévisions météo pour demain annoncent un temps partiellement nuageux."),
    "de": ("German",     "Die Wettervorhersage für morgen ist teilweise bewölkt."),
    "pt": ("Portuguese", "A previsão do tempo para amanhã é parcialmente nublado."),
    "ru": ("Russian",    "Прогноз погоды на завтра — переменная облачность."),
    "vi": ("Vietnamese", "Dự báo thời tiết ngày mai là có mây rải rác."),
    "zh": ("Chinese",    "明天的天气预报是多云转晴。"),
}


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[smoke] loading OmniVoice + ASR ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()

    ref_path = JVNV_DIR / f"jvnv_F1_{EMOTION}.wav"
    ref_text = w.transcribe(ref_path, language=None)

    rows = []
    for lang_code in sorted(LANG_TEXT.keys()):
        lang_full, target_text = LANG_TEXT[lang_code]
        artifact = Path(f"outputs/graft/{lang_code}_axis.npz")
        if not artifact.exists():
            print(f"\n[{lang_code}] no artifact, skip", flush=True)
            continue
        axis, target_c, _ = load_axis_artifacts(artifact)
        axis = {L: axis[L] for L in LAYERS}
        target_c = {L: target_c[L] for L in LAYERS}
        print(f"\n[{lang_code}] {lang_full}  target='{target_text[:40]}...'", flush=True)

        for kind in ("baseline", "graft"):
            for rep in range(REPS):
                torch.manual_seed(SEED + rep)
                kwargs = dict(text=target_text, language=lang_full,
                              ref_audio=str(ref_path), ref_text=ref_text)
                if kind == "baseline":
                    audio = w.model.generate(**kwargs)[0]
                else:
                    with LanguageAxisGrafter(
                        w.model, axis_per_layer=axis,
                        target_c_per_layer=target_c,
                        remove_alpha=1.0, inject_beta=1.0,
                        step_window=STEP_WINDOW,
                    ):
                        audio = w.model.generate(**kwargs)[0]
                wav_path = OUT_DIR / f"{lang_code}_{kind}_rep{rep}.wav"
                save_wav(wav_path, audio, w.SAMPLING_RATE)
                hyp = asr.transcribe(wav_path, language=lang_code)
                rows.append({"lang": lang_code, "kind": kind, "rep": rep,
                             "hyp": hyp, "wav": wav_path.name})
                print(f"  {kind:<9} rep{rep}: {hyp[:80]}", flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n[smoke] saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
