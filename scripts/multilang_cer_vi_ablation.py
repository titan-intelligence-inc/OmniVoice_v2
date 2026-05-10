"""Vietnamese-only ablation: which stage of the latest pipeline is
killing vi CER?

4 variants, each with N=8 reps:
  A. plain OmniVoice (no SeedVC, no F0 transfer) — measures whether
     OmniVoice itself can pronounce vi correctly.
  B. OmniVoice + SeedVC β=0  (no emo blend, no F0) — does plain VC
     onto F1 voice destroy vi?
  C. OmniVoice + SeedVC β=0.6 (emo blend, no F0) — does the emotion
     style projection corrupt vi tones?
  D. OmniVoice + F0 transfer (blend=1.0) + SeedVC β=0.6 — the
     production pipeline (= multilang_cer.py for vi).

Output per variant: cer_med, Q1, Q3, best/N, plus a sample hyp so
we can see whether Whisper is hallucinating English vs. actually
mistranscribing vi.
"""
from __future__ import annotations
import os
import sys
import json
import statistics
from pathlib import Path

import librosa
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
NATIVE_DIR  = Path("baseline/native_refs/vi")
OUT_DIR     = Path("outputs/multilang_cer_vi_ablation")

VI_TEXT     = "Dự báo thời tiết ngày mai có mây và có lúc nắng."
F1_REF      = JVNV_DIR / "jvnv_F1_anger.wav"
N           = 8
STEPS       = 100
CFG         = 0.7
SR_OMNI     = 24000
F0_CEIL     = 600.0


def _cer(ref: str, hyp: str, *, cap: float | None = 2.0) -> float:
    from ovet.analyzers.asr_analyzer import _detect_hallucination
    rn = _normalize_text(ref); hn = _normalize_text(hyp)
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

    print("[vi-abl] loading models ...", flush=True)
    w   = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()
    svc = load_seedvc()
    emo_an = EmotionAnalyzer()
    asr._ensure_loaded()

    pool = sorted(NATIVE_DIR.glob("fleurs_vi_dev_*.wav"))
    refs = [pool[i % len(pool)] for i in range(N)]
    print(f"[vi-abl] vi refs: {[p.name for p in refs]}", flush=True)
    print(f"[vi-abl] target: {VI_TEXT}", flush=True)

    ref_texts = [w.transcribe(p, language=None) for p in refs]
    print("[vi-abl] vi ref_texts (auto-transcribed by Whisper):", flush=True)
    for p, t in zip(refs, ref_texts):
        print(f"   {p.name}: {t[:60]}", flush=True)

    emo_emb = np.array(emo_an.analyze(F1_REF).embedding,
                       dtype=np.float32, copy=True)

    # ---- Stage 1: OmniVoice content donors (shared across variants) ----
    print("\n[vi-abl] OmniVoice batched generate ...", flush=True)
    donors = w.model.generate(
        text=[VI_TEXT] * N, language=["Vietnamese"] * N,
        ref_audio=[str(p) for p in refs], ref_text=ref_texts,
    )
    donor_paths = []
    for i, a in enumerate(donors):
        p = OUT_DIR / f"01_donor_{i}.wav"
        save_wav(p, a, SR_OMNI)
        donor_paths.append(p)

    # ---- Stage 2: F0-stats transfer (used only by D) ----
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
            blend=1.0, f0_ceil=F0_CEIL,
        )
        p = OUT_DIR / f"02_f0t_{i}.wav"
        save_wav(p, shifted.astype(np.float32), sr_c)
        f0t_paths.append(p)

    variants: list[dict] = []

    # ---- A. plain OmniVoice ----
    print("\n[vi-abl] === A: plain OmniVoice ===", flush=True)
    hyps_A = asr.transcribe_batch(donor_paths, language="vietnamese", batch_size=N)
    cers_A = [_cer(VI_TEXT, h) for h in hyps_A]
    variants.append({"id": "A", "name": "OmniVoice only",
                     "cers": cers_A, "hyps": hyps_A})

    # ---- B. OmniVoice + SeedVC β=0 (no emo, no F0) ----
    print("[vi-abl] === B: + SeedVC β=0 ===", flush=True)
    with torch.no_grad():
        sr_b, outs_b = convert_voice_batch(
            svc, sources=donor_paths, target_path=F1_REF,
            emotion_audio_path=F1_REF, emotion_embedding=emo_emb,
            alpha=1.0, beta=0.0,
            diffusion_steps=STEPS, inference_cfg_rate=CFG,
        )
    pB = []
    for i, conv in enumerate(outs_b):
        conv = np.asarray(conv, dtype=np.float32)
        conv_24k = librosa.resample(conv, orig_sr=sr_b, target_sr=SR_OMNI)
        p = OUT_DIR / f"B_converted_{i}.wav"
        save_wav(p, conv_24k, SR_OMNI); pB.append(p)
    hyps_B = asr.transcribe_batch(pB, language="vietnamese", batch_size=N)
    cers_B = [_cer(VI_TEXT, h) for h in hyps_B]
    variants.append({"id": "B", "name": "+ SeedVC β=0 (no emo, no F0)",
                     "cers": cers_B, "hyps": hyps_B})

    # ---- C. OmniVoice + SeedVC β=0.6 (emo, no F0) ----
    print("[vi-abl] === C: + SeedVC β=0.6 ===", flush=True)
    with torch.no_grad():
        sr_c, outs_c = convert_voice_batch(
            svc, sources=donor_paths, target_path=F1_REF,
            emotion_audio_path=F1_REF, emotion_embedding=emo_emb,
            alpha=1.0, beta=0.6,
            diffusion_steps=STEPS, inference_cfg_rate=CFG,
        )
    pC = []
    for i, conv in enumerate(outs_c):
        conv = np.asarray(conv, dtype=np.float32)
        conv_24k = librosa.resample(conv, orig_sr=sr_c, target_sr=SR_OMNI)
        p = OUT_DIR / f"C_converted_{i}.wav"
        save_wav(p, conv_24k, SR_OMNI); pC.append(p)
    hyps_C = asr.transcribe_batch(pC, language="vietnamese", batch_size=N)
    cers_C = [_cer(VI_TEXT, h) for h in hyps_C]
    variants.append({"id": "C", "name": "+ SeedVC β=0.6 (emo, no F0)",
                     "cers": cers_C, "hyps": hyps_C})

    # ---- D. + F0 transfer + SeedVC β=0.6 (= production) ----
    print("[vi-abl] === D: + F0 transfer + SeedVC β=0.6 (production) ===",
          flush=True)
    with torch.no_grad():
        sr_d, outs_d = convert_voice_batch(
            svc, sources=f0t_paths, target_path=F1_REF,
            emotion_audio_path=F1_REF, emotion_embedding=emo_emb,
            alpha=1.0, beta=0.6,
            diffusion_steps=STEPS, inference_cfg_rate=CFG,
        )
    pD = []
    for i, conv in enumerate(outs_d):
        conv = np.asarray(conv, dtype=np.float32)
        conv_24k = librosa.resample(conv, orig_sr=sr_d, target_sr=SR_OMNI)
        p = OUT_DIR / f"D_converted_{i}.wav"
        save_wav(p, conv_24k, SR_OMNI); pD.append(p)
    hyps_D = asr.transcribe_batch(pD, language="vietnamese", batch_size=N)
    cers_D = [_cer(VI_TEXT, h) for h in hyps_D]
    variants.append({"id": "D", "name": "+ F0 + SeedVC β=0.6 (production)",
                     "cers": cers_D, "hyps": hyps_D})

    # ---- summary ----
    print("\n=== vi ablation summary (target_lang=vi forced into Whisper) ===",
          flush=True)
    print(f"{'var':<3} {'name':<36} {'cer_med':>8} {'q1':>8} {'q3':>8} "
          f"{'best/N':>8} {'mean':>8}", flush=True)
    for v in variants:
        c = v["cers"]
        print(f"{v['id']:<3} {v['name']:<36} "
              f"{statistics.median(c):8.3f} "
              f"{statistics.quantiles(c, n=4)[0]:8.3f} "
              f"{statistics.quantiles(c, n=4)[2]:8.3f} "
              f"{min(c):8.3f} {statistics.fmean(c):8.3f}", flush=True)

    print("\n=== sample hyps (rep0, all variants) ===", flush=True)
    print(f"  ref: {VI_TEXT}")
    for v in variants:
        print(f"  {v['id']}: {v['hyps'][0][:100]}", flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "ref_text": VI_TEXT, "f1_ref": str(F1_REF),
            "vi_refs": [p.name for p in refs],
            "vi_ref_texts": ref_texts,
            "variants": variants,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[vi-abl] result -> {OUT_DIR / 'result.json'}")


if __name__ == "__main__":
    main()
