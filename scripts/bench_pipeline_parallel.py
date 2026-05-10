"""Bench: serial (per-sample loop) vs batched (native list inputs).

Runs the eps3 pipeline for ONE emotion and times each stage:
  * OmniVoice content donor generation     (×N)
  * SeedVC convert with emotion blend      (×N)
  * Whisper ASR on the converted output    (×N)

Two paths:
  * SERIAL:   each stage runs N independent calls.
  * BATCHED:  each stage runs ONE call with list inputs of size N.

Reports per-stage timings, total wall-clock, and speedup factors.

Usage::
    HF_HOME=/workspace/hf_cache \\
    MODELSCOPE_CACHE=/workspace/hf_cache/modelscope \\
    python scripts/bench_pipeline_parallel.py
"""
from __future__ import annotations
import os
import sys
import time
import statistics
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "src")
sys.path.insert(0, "upstream/OmniVoice")

from ovet.analyzers.asr_analyzer import ASRAnalyzer                              # noqa: E402
from ovet.analyzers.emotion_analyzer import EmotionAnalyzer                      # noqa: E402
from ovet.omnivoice.wrapper import OmniVoiceWrapper                              # noqa: E402
from ovet.postprocessing.seedvc import load_seedvc                               # noqa: E402
from ovet.postprocessing.seedvc_emotion import convert_voice_with_emotion        # noqa: E402
from ovet.postprocessing.seedvc_batched import convert_voice_batch               # noqa: E402
from ovet.utils.io import save_wav                                               # noqa: E402


JVNV_DIR    = Path("baseline/jvnv_samples")
NATIVE_DIR  = Path("baseline/native_refs")
OUT_DIR     = Path("outputs/bench_pipeline_parallel")
ZH_HANZI    = "明天的天气预报是多云转晴。"
EMOTION     = "anger"
N           = 8
STEPS       = 100
CFG         = 0.7
BETA        = 0.6
SEED        = 1234


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench(label: str, fn):
    """Run fn once after a warm CUDA sync, return elapsed seconds + result."""
    _sync()
    t0 = time.perf_counter()
    out = fn()
    _sync()
    dt = time.perf_counter() - t0
    print(f"  {label:<28} {dt:7.2f}s", flush=True)
    return dt, out


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("MODELSCOPE_CACHE", "/workspace/hf_cache/modelscope")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[bench] loading OmniVoice + SeedVC + ASR + emotion2vec ...", flush=True)
    w   = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()
    svc = load_seedvc()
    emo_an = EmotionAnalyzer()
    asr._ensure_loaded()        # bring Whisper online before timing

    f1_ref = JVNV_DIR / f"jvnv_F1_{EMOTION}.wav"
    native_pool = sorted(NATIVE_DIR.glob("fleurs_zh_dev_*.wav"))
    assert native_pool, "no native zh refs found"
    # cycle pool to length N (we need N refs even if pool is smaller)
    native_refs = [native_pool[i % len(native_pool)] for i in range(N)]

    # Pre-extract emotion embedding (shared across both paths).
    emo_emb = np.array(emo_an.analyze(f1_ref).embedding, dtype=np.float32, copy=True)

    # Pre-transcribe ref_texts (so neither path is charged for ASR-on-refs).
    print("[bench] pre-transcribing refs ...", flush=True)
    f1_ref_text = w.transcribe(f1_ref, language=None)
    native_ref_texts = [w.transcribe(p, language="zh") for p in native_refs]

    # ------------------------------------------------------------------
    # WARMUP: one full single-sample pass through each model so compile /
    # autotune costs aren't billed to the timed runs.
    # ------------------------------------------------------------------
    print("\n[bench] warmup ...", flush=True)
    torch.manual_seed(SEED)
    _ = w.model.generate(
        text=ZH_HANZI, language="Chinese",
        ref_audio=str(native_refs[0]), ref_text=native_ref_texts[0],
    )[0]
    warm_src = OUT_DIR / "_warm_src.wav"
    save_wav(warm_src, _, 24000)
    with torch.no_grad():
        _ = convert_voice_with_emotion(
            svc, source_path=warm_src, target_path=f1_ref,
            emotion_audio_path=f1_ref, emotion_embedding=emo_emb,
            alpha=1.0, beta=BETA,
            diffusion_steps=10, inference_cfg_rate=CFG,
        )
    _ = asr.transcribe(warm_src, language="zh")

    # ------------------------------------------------------------------
    # SERIAL path
    # ------------------------------------------------------------------
    print("\n[bench] === SERIAL (per-sample loop, N=8) ===", flush=True)
    serial_t = {}

    def _serial_omni():
        outs = []
        for i in range(N):
            torch.manual_seed(SEED + i)
            a = w.model.generate(
                text=ZH_HANZI, language="Chinese",
                ref_audio=str(native_refs[i]), ref_text=native_ref_texts[i],
            )[0]
            outs.append(a)
        return outs
    serial_t["omnivoice"], donors_serial = _bench("OmniVoice ×N", _serial_omni)

    serial_src_paths = []
    for i, a in enumerate(donors_serial):
        p = OUT_DIR / f"serial_src_{i}.wav"
        save_wav(p, a, 24000)
        serial_src_paths.append(p)

    def _serial_svc():
        outs = []
        with torch.no_grad():
            for src in serial_src_paths:
                sr_o, conv = convert_voice_with_emotion(
                    svc, source_path=src, target_path=f1_ref,
                    emotion_audio_path=f1_ref, emotion_embedding=emo_emb,
                    alpha=1.0, beta=BETA,
                    diffusion_steps=STEPS, inference_cfg_rate=CFG,
                )
                outs.append((sr_o, conv))
        return outs
    serial_t["seedvc"], svc_serial = _bench("SeedVC ×N", _serial_svc)

    serial_conv_paths = []
    for i, (sr_o, conv) in enumerate(svc_serial):
        p = OUT_DIR / f"serial_conv_{i}.wav"
        save_wav(p, np.asarray(conv, dtype=np.float32), sr_o)
        serial_conv_paths.append(p)

    def _serial_asr():
        return [asr.transcribe(p, language="zh") for p in serial_conv_paths]
    serial_t["asr"], hyps_serial = _bench("ASR ×N", _serial_asr)

    serial_total = sum(serial_t.values())
    print(f"  {'TOTAL':<28} {serial_total:7.2f}s")

    # ------------------------------------------------------------------
    # BATCHED path
    # ------------------------------------------------------------------
    print("\n[bench] === BATCHED (single list-form call per stage, N=8) ===",
          flush=True)
    batch_t = {}

    def _batch_omni():
        return w.model.generate(
            text=[ZH_HANZI] * N,
            language=["Chinese"] * N,
            ref_audio=[str(p) for p in native_refs],
            ref_text=native_ref_texts,
        )
    batch_t["omnivoice"], donors_batch = _bench("OmniVoice ×1 (B=N)", _batch_omni)

    batch_src_paths = []
    for i, a in enumerate(donors_batch):
        p = OUT_DIR / f"batch_src_{i}.wav"
        save_wav(p, a, 24000)
        batch_src_paths.append(p)

    def _batch_svc():
        with torch.no_grad():
            sr_o, outs = convert_voice_batch(
                svc, sources=batch_src_paths, target_path=f1_ref,
                emotion_audio_path=f1_ref, emotion_embedding=emo_emb,
                alpha=1.0, beta=BETA,
                diffusion_steps=STEPS, inference_cfg_rate=CFG,
            )
        return sr_o, outs
    batch_t["seedvc"], (sr_b, outs_b) = _bench("SeedVC ×1 (B=N)", _batch_svc)

    batch_conv_paths = []
    for i, conv in enumerate(outs_b):
        p = OUT_DIR / f"batch_conv_{i}.wav"
        save_wav(p, np.asarray(conv, dtype=np.float32), sr_b)
        batch_conv_paths.append(p)

    def _batch_asr():
        return asr.transcribe_batch(batch_conv_paths, language="zh", batch_size=N)
    batch_t["asr"], hyps_batch = _bench("ASR ×1 (B=N)", _batch_asr)

    batch_total = sum(batch_t.values())
    print(f"  {'TOTAL':<28} {batch_total:7.2f}s")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n=== summary (N=8) ===", flush=True)
    print(f"{'stage':<12} {'serial':>10} {'batched':>10} {'speedup':>10}")
    for k in ("omnivoice", "seedvc", "asr"):
        s = serial_t[k]; b = batch_t[k]
        sp = s / b if b > 0 else float("inf")
        print(f"{k:<12} {s:>9.2f}s {b:>9.2f}s {sp:>9.2f}x")
    sp_tot = serial_total / batch_total if batch_total > 0 else float("inf")
    print(f"{'TOTAL':<12} {serial_total:>9.2f}s {batch_total:>9.2f}s "
          f"{sp_tot:>9.2f}x")

    print(f"\n[bench] sample serial hyps:  {hyps_serial[:2]}")
    print(f"[bench] sample batched hyps: {hyps_batch[:2]}")
    print(f"[bench] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
