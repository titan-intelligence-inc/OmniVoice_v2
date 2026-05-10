"""Multi-language CER for the latest ε-3 + emotion architecture.

Pipeline per language (same recipe as the zh production path):
  1. OmniVoice (batched) — clone N native refs, generate target text.
  2. F0-stats transfer (Approach 1) toward F1 anger ref.
  3. SeedVC (batched) with emotion2vec-blended style (Approach 2,
     β=0.6) — target speaker = F1 anger ref, so output is "F1 anger
     voice saying <target text>" in the target language.
  4. Whisper ASR (batched) → CER vs the target text.

Languages: ja, en, de, es, fr, ko, pt, ru, vi  (the 10 majors minus zh).

Usage::
    HF_HOME=/workspace/hf_cache MODELSCOPE_CACHE=/workspace/hf_cache/modelscope \\
    venv/bin/python scripts/multilang_cer.py
"""
from __future__ import annotations
import os
import re
import sys
import json
import statistics
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "upstream/OmniVoice")

from ovet.analyzers.asr_analyzer import ASRAnalyzer, _normalize_text             # noqa: E402
from ovet.analyzers.emotion_analyzer import EmotionAnalyzer                      # noqa: E402
from ovet.omnivoice.wrapper import OmniVoiceWrapper                              # noqa: E402
from ovet.postprocessing.seedvc import load_seedvc                               # noqa: E402
from ovet.postprocessing.seedvc_batched import convert_voice_batch               # noqa: E402
from ovet.postprocessing.f0_emotion_transfer import transfer_f0_emotion          # noqa: E402
from ovet.utils.io import save_wav                                               # noqa: E402


JVNV_DIR    = Path("baseline/jvnv_samples")
JVNV_MULTI  = Path("baseline/jvnv_samples_multi")
NATIVE_DIR  = Path("baseline/native_refs")
OUT_DIR     = Path("outputs/multilang_cer")
EMOTION     = "anger"
F1_REF      = JVNV_DIR / f"jvnv_F1_{EMOTION}.wav"
N           = 8
STEPS       = 100
CFG         = 0.7
BETA        = 0.6
F0_BLEND    = 1.0
F0_CEIL     = 600.0
SR_OMNI     = 24000


# Ref text per language. The semantic content is roughly equivalent
# across all 9 ("Tomorrow's forecast: cloudy, then sun") so CER stays
# comparable. Pinyin/punctuation removed at scoring time.
LANGS: list[dict] = [
    {"id": "ja", "ov": "Japanese",   "wh": "japanese",
     "text": "明日の天気予報は曇り時々晴れです。",
     "refs_glob": None,  # use JVNV non-F1 samples
     "fold_zh": False},
    {"id": "en", "ov": "English",    "wh": "english",
     "text": "Tomorrow's weather forecast is partly cloudy with sunny intervals.",
     "refs_glob": ("en", "fleurs_en_dev_*.wav"),
     "fold_zh": False},
    {"id": "de", "ov": "German",     "wh": "german",
     "text": "Die Wettervorhersage für morgen ist wolkig mit sonnigen Abschnitten.",
     "refs_glob": ("de", "fleurs_de_dev_*.wav"),
     "fold_zh": False},
    {"id": "es", "ov": "Spanish",    "wh": "spanish",
     "text": "El pronóstico del tiempo para mañana es nublado con claros.",
     "refs_glob": ("es", "fleurs_es_dev_*.wav"),
     "fold_zh": False},
    {"id": "fr", "ov": "French",     "wh": "french",
     "text": "Les prévisions météo pour demain sont nuageuses avec des éclaircies.",
     "refs_glob": ("fr", "fleurs_fr_dev_*.wav"),
     "fold_zh": False},
    {"id": "ko", "ov": "Korean",     "wh": "korean",
     "text": "내일의 일기 예보는 구름이 많고 가끔 맑겠습니다.",
     "refs_glob": ("ko", "fleurs_ko_dev_*.wav"),
     "fold_zh": False},
    {"id": "pt", "ov": "Portuguese", "wh": "portuguese",
     "text": "A previsão do tempo para amanhã é nublado com abertas de sol.",
     "refs_glob": ("pt", "fleurs_pt_dev_*.wav"),
     "fold_zh": False},
    {"id": "ru", "ov": "Russian",    "wh": "russian",
     "text": "Прогноз погоды на завтра — облачно с прояснениями.",
     "refs_glob": ("ru", "fleurs_ru_dev_*.wav"),
     "fold_zh": False},
    {"id": "vi", "ov": "Vietnamese", "wh": "vietnamese",
     "text": "Dự báo thời tiết ngày mai có mây và có lúc nắng.",
     "refs_glob": ("vi", "fleurs_vi_dev_*.wav"),
     "fold_zh": False},
]


def _resolve_refs(lang: dict) -> list[Path]:
    """Pick N content refs for this language, cycling if pool < N."""
    if lang["id"] == "ja":
        # JVNV non-F1 emotional samples → ja phonetic content donor.
        pool = sorted(JVNV_MULTI.glob("jvnv_[FM][2]_*.wav")) \
             + sorted(JVNV_MULTI.glob("jvnv_M1_*.wav"))
    else:
        sub, glob = lang["refs_glob"]
        pool = sorted((NATIVE_DIR / sub).glob(glob))
    assert pool, f"no refs found for {lang['id']}"
    return [pool[i % len(pool)] for i in range(N)]


def _cer(ref: str, hyp: str, *, cap: float | None = 2.0,
         fold_zh: bool = False) -> float:
    """CER (Levenshtein / |ref|), language-agnostic. Strips punct +
    case-folds before comparing. Same hallucination cap as zh path."""
    from ovet.analyzers.asr_analyzer import _detect_hallucination
    rn = _normalize_text(ref, zh=fold_zh)
    hn = _normalize_text(hyp, zh=fold_zh)
    if not rn:
        return 0.0 if not hn else 1.0
    if _detect_hallucination(hn) and cap is not None:
        return cap
    m, n = len(rn), len(hn)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if rn[i - 1] == hn[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    raw = dp[n] / m
    return min(raw, cap) if cap is not None else raw


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("MODELSCOPE_CACHE", "/workspace/hf_cache/modelscope")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[mlc] loading models ...", flush=True)
    w   = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()
    svc = load_seedvc()
    emo_an = EmotionAnalyzer()
    asr._ensure_loaded()

    # Pre-compute emotion embedding for F1 anger ref (shared across langs).
    emo_emb = np.array(emo_an.analyze(F1_REF).embedding,
                       dtype=np.float32, copy=True)
    print(f"[mlc] F1 emo ref  : {F1_REF}", flush=True)
    print(f"[mlc] target text per lang, β={BETA}, F0 blend={F0_BLEND}, "
          f"steps={STEPS}", flush=True)

    rows = []
    print(f"\n{'lang':<4} {'cer_med':>8} {'q1':>8} {'q3':>8} "
          f"{'best/N':>8} {'mean':>8}", flush=True)
    print("-" * 50, flush=True)

    for lang in LANGS:
        lid = lang["id"]
        refs = _resolve_refs(lang)
        out_lang = OUT_DIR / lid
        out_lang.mkdir(parents=True, exist_ok=True)

        # Pre-transcribe each ref so OmniVoice batched call has ref_texts.
        ref_texts = [w.transcribe(p, language=None) for p in refs]

        # ---- 1. OmniVoice content donors (batched) ----
        donors = w.model.generate(
            text=[lang["text"]] * N,
            language=[lang["ov"]] * N,
            ref_audio=[str(p) for p in refs],
            ref_text=ref_texts,
        )
        donor_paths = []
        for i, a in enumerate(donors):
            p = out_lang / f"01_donor_{i}.wav"
            save_wav(p, a, SR_OMNI)
            donor_paths.append(p)

        # ---- 2. F0-stats transfer toward F1 anger ref ----
        emo_audio, emo_sr = sf.read(F1_REF)
        if emo_audio.ndim > 1:
            emo_audio = emo_audio.mean(axis=1)
        f0t_paths = []
        for i, dp in enumerate(donor_paths):
            cd, sr_c = sf.read(dp)
            if cd.ndim > 1:
                cd = cd.mean(axis=1)
            shifted = transfer_f0_emotion(
                cd, sr_c, emo_audio, emo_sr,
                blend=F0_BLEND, f0_ceil=F0_CEIL,
            )
            p = out_lang / f"02_f0t_{i}.wav"
            save_wav(p, shifted.astype(np.float32), sr_c)
            f0t_paths.append(p)

        # ---- 3. SeedVC batched with emo blend ----
        with torch.no_grad():
            sr_b, outs_b = convert_voice_batch(
                svc, sources=f0t_paths, target_path=F1_REF,
                emotion_audio_path=F1_REF, emotion_embedding=emo_emb,
                alpha=1.0, beta=BETA,
                diffusion_steps=STEPS, inference_cfg_rate=CFG,
            )
        conv_paths = []
        import librosa
        for i, conv in enumerate(outs_b):
            conv = np.asarray(conv, dtype=np.float32)
            conv_24k = librosa.resample(conv, orig_sr=sr_b, target_sr=SR_OMNI)
            p = out_lang / f"03_converted_{i}.wav"
            save_wav(p, conv_24k, SR_OMNI)
            conv_paths.append(p)

        # ---- 4. ASR batched + CER ----
        hyps = asr.transcribe_batch(conv_paths, language=lang["wh"], batch_size=N)
        cers = [_cer(lang["text"], h, fold_zh=lang["fold_zh"]) for h in hyps]
        med = statistics.median(cers)
        q1  = statistics.quantiles(cers, n=4)[0]
        q3  = statistics.quantiles(cers, n=4)[2]
        bo  = min(cers)
        mean = statistics.fmean(cers)

        rows.append({
            "lang": lid, "ov_lang": lang["ov"], "text": lang["text"],
            "cers": cers, "hyps": hyps,
            "cer_med": med, "cer_q1": q1, "cer_q3": q3, "best": bo, "mean": mean,
        })
        print(f"{lid:<4} {med:8.3f} {q1:8.3f} {q3:8.3f} "
              f"{bo:8.3f} {mean:8.3f}", flush=True)

    # Aggregate
    print("\n=== summary (median across reps, anger emo, β=0.6, N=8) ===",
          flush=True)
    print(f"{'lang':<4} {'cer_med':>8} {'best/N':>8}   ref → top hyp",
          flush=True)
    for r in rows:
        # show first hyp as a sample
        sample = r["hyps"][0][:40] if r["hyps"] else ""
        print(f"{r['lang']:<4} {r['cer_med']:8.3f} {r['best']:8.3f}   "
              f"{r['text'][:30]:<30} → {sample}", flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "rows": rows,
            "config": {
                "emotion": EMOTION, "f1_ref": str(F1_REF),
                "beta": BETA, "f0_blend": F0_BLEND, "steps": STEPS, "cfg": CFG,
                "n": N,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[mlc] result.json -> {OUT_DIR / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
