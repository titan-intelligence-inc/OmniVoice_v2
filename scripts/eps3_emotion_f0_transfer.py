"""Approach 1: pre-condition the SeedVC source with F1's emotional F0 stats.

For each emotion × rep:
  1. Take the (cleanest) content donor wav (native voice, neutral
     prosody).
  2. Apply F0-statistics renormalisation toward the F1 JVNV ref of
     matching emotion → "emotional source" audio.
  3. Run SeedVC: source = emotional source, target = F1 JVNV ref.
  4. Compare with the no-transfer baseline (current production).

Sweep blend ∈ {0.0 (= no transfer = baseline), 0.5, 1.0} and a couple
of f0_ceil values to handle high-pitched emotional refs.
"""
from __future__ import annotations
import os, sys, json, statistics, shutil
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "src")

from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402
from ovet.evaluation.zh_eval import cer_zh                                     # noqa: E402
from ovet.postprocessing.seedvc import load_seedvc, convert_voice_full         # noqa: E402
from ovet.postprocessing.f0_emotion_transfer import (                          # noqa: E402
    transfer_f0_emotion, compute_f0_stats,
)


JVNV_DIR    = Path("baseline/jvnv_samples")
EPS3_DIR    = Path("outputs/eps3_semantic_vc")
OUT_DIR     = Path("outputs/eps3_emotion_f0_transfer")
ZH_HANZI    = "明天的天气预报是多云转晴。"
EMOTIONS    = ["sad", "happy", "anger"]
REPS        = 8
STEPS       = 100
CFG         = 0.7
SR_OMNI     = 24000
BLENDS      = [0.0, 0.5, 1.0]


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[emo-f0] loading ASR + SeedVC ...", flush=True)
    asr = ASRAnalyzer()
    svc = load_seedvc()

    # Diagnostic: print F1 emotional ref's F0 stats
    print("\n[emo-f0] F1 emotional F0 stats:", flush=True)
    for emo in EMOTIONS:
        wav, sr = sf.read(JVNV_DIR / f"jvnv_F1_{emo}.wav")
        if wav.ndim > 1: wav = wav.mean(axis=1)
        s = compute_f0_stats(wav, sr, f0_ceil=600.0)
        print(f"  {emo:<7} mean={s.mean_hz:6.1f}Hz  std={s.std_hz:5.1f}  "
              f"median={s.median_hz:6.1f}  voiced={s.voiced_frac:.0%}",
              flush=True)

    rows = []
    flat: dict[str, list[float]] = {}
    best_per_var: dict[str, dict[str, tuple[Path, str, float]]] = {}

    for blend in BLENDS:
        vname = f"blend{blend:.1f}"
        print(f"\n[emo-f0] === {vname} ===", flush=True)
        cers_cell = []
        best_per_var[vname] = {}
        for emo in EMOTIONS:
            target_path = JVNV_DIR / f"jvnv_F1_{emo}.wav"
            cers_emo, hyps_emo, wavs_emo = [], [], []
            for rep in range(REPS):
                content_path = EPS3_DIR / f"content_donor_{emo}_rep{rep}.wav"
                # Pre-condition source with emotion F0 stats
                content_audio, sr_c = sf.read(content_path)
                emotion_audio, sr_e = sf.read(target_path)
                if blend > 0.0:
                    src_audio = transfer_f0_emotion(
                        content_audio, sr_c, emotion_audio, sr_e,
                        blend=blend, f0_ceil=600.0,
                    )
                else:
                    src_audio = content_audio if content_audio.ndim == 1 else content_audio.mean(axis=1)
                    src_audio = src_audio.astype(np.float32)
                # Save the pre-conditioned source for downstream debugging
                src_wav = OUT_DIR / f"_source_{vname}_{emo}_rep{rep}.wav"
                save_wav(src_wav, src_audio, sr_c)
                # SeedVC
                out_sr, conv = convert_voice_full(
                    svc,
                    source=src_wav, target=target_path,
                    diffusion_steps=STEPS, inference_cfg_rate=CFG,
                )
                conv = np.asarray(conv, dtype=np.float32)
                import librosa
                conv_24k = librosa.resample(conv, orig_sr=out_sr, target_sr=SR_OMNI)
                out_wav = OUT_DIR / f"{vname}_{emo}_rep{rep}.wav"
                save_wav(out_wav, conv_24k, SR_OMNI)
                hyp = asr.transcribe(out_wav, language="zh")
                cer = cer_zh(ZH_HANZI, hyp)
                cers_emo.append(cer); hyps_emo.append(hyp); wavs_emo.append(out_wav)
            cers_cell.extend(cers_emo)
            best_idx = int(np.argmin(cers_emo))
            best_per_var[vname][emo] = (
                wavs_emo[best_idx], hyps_emo[best_idx], cers_emo[best_idx]
            )
            rows.append({"variant": vname, "emotion": emo,
                         "cers": cers_emo, "hyps": hyps_emo})
            print(f"  {emo:<7}  cer_med={statistics.median(cers_emo):.3f}  "
                  f"best={cers_emo[best_idx]:.3f}", flush=True)
        flat[vname] = cers_cell

    print("\n=== summary ===", flush=True)
    print(f"{'variant':<12} {'cer_med':>8} {'q1':>8} {'q3':>8} {'best-of-8':>10}",
          flush=True)
    for vname, cers in flat.items():
        med = statistics.median(cers)
        q1  = statistics.quantiles(cers, n=4)[0]
        q3  = statistics.quantiles(cers, n=4)[2]
        bo  = statistics.median([best_per_var[vname][e][2] for e in EMOTIONS])
        print(f"{vname:<12} {med:>8.3f} {q1:>8.3f} {q3:>8.3f} {bo:>10.3f}",
              flush=True)

    pkt = OUT_DIR / "_listening"
    pkt.mkdir(exist_ok=True)
    for emo in EMOTIONS:
        shutil.copy2(JVNV_DIR / f"jvnv_F1_{emo}.wav",
                     pkt / f"{emo}__01_F1_ref.wav")
        for vname in flat:
            src, _, cer = best_per_var[vname][emo]
            shutil.copy2(src, pkt / f"{emo}__{vname}__cer{cer:.3f}.wav")
            # Also copy the pre-conditioned source for diagnostic
            best_rep = next(r for r in range(REPS)
                            if (OUT_DIR / f"{vname}_{emo}_rep{r}.wav") == src)
            src_pre = OUT_DIR / f"_source_{vname}_{emo}_rep{best_rep}.wav"
            if src_pre.exists():
                shutil.copy2(src_pre, pkt / f"{emo}__{vname}__source.wav")
    print(f"\n[emo-f0] listening packet -> {pkt}", flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "rows": rows,
            "best_per_variant": {
                v: {e: {"src": str(t[0]), "hyp": t[1], "cer": t[2]}
                    for e, t in d.items()}
                for v, d in best_per_var.items()},
            "config": {"cfg": CFG, "steps": STEPS, "blends": BLENDS,
                       "emotions": EMOTIONS, "reps": REPS},
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
