"""Approach 2: emotion2vec embedding injected into SeedVC's style.

Sweep β (emotion-style weight) keeping α=1.0 (full speaker style).
For β=0 we get the SeedVC baseline; β>0 progressively injects the
projected emotion2vec embedding into the cfm conditioning.

Per-emotion target = matching JVNV F1 ref. Emotion ref = same JVNV
ref (could be different in production).

3 emotions × 8 reps × 4 beta = 96 generations. Steps=100, cfg=0.7.
"""
from __future__ import annotations
import os, sys, json, statistics, shutil
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "src")

from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.analyzers.emotion_analyzer import EmotionAnalyzer                    # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402
from ovet.evaluation.zh_eval import cer_zh                                     # noqa: E402
from ovet.postprocessing.seedvc import load_seedvc                             # noqa: E402
from ovet.postprocessing.seedvc_emotion import convert_voice_with_emotion      # noqa: E402


JVNV_DIR    = Path("baseline/jvnv_samples")
EPS3_DIR    = Path("outputs/eps3_semantic_vc")
OUT_DIR     = Path("outputs/eps3_emotion2vec_inject")
ZH_HANZI    = "明天的天气预报是多云转晴。"
EMOTIONS    = ["sad", "happy", "anger"]
REPS        = 8
STEPS       = 100
CFG         = 0.7
BETAS       = [0.0, 0.3, 0.6, 1.0]


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("MODELSCOPE_CACHE", "/workspace/hf_cache/modelscope")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[emo2v] loading ASR + emotion2vec + SeedVC ...", flush=True)
    asr = ASRAnalyzer()
    emo_analyzer = EmotionAnalyzer()
    svc = load_seedvc()

    # Pre-compute emotion2vec embeddings for each F1 emotional ref —
    # so we don't reload emotion2vec on every call.
    print("[emo2v] extracting emotion2vec embeddings for F1 refs ...", flush=True)
    emo_embeddings: dict[str, np.ndarray] = {}
    for emo in EMOTIONS:
        ref = JVNV_DIR / f"jvnv_F1_{emo}.wav"
        e = emo_analyzer.analyze(ref)
        emo_embeddings[emo] = np.array(e.embedding, dtype=np.float32, copy=True)
        print(f"  {emo:<7}  label={e.label:<10} confidence={e.confidence:.3f}  "
              f"emb dim={emo_embeddings[emo].shape}", flush=True)

    rows = []
    flat: dict[str, list[float]] = {}
    best_per_var: dict[str, dict[str, tuple[Path, str, float]]] = {}

    for beta in BETAS:
        vname = f"beta{beta:.1f}"
        print(f"\n[emo2v] === {vname} ===", flush=True)
        cers_cell = []
        best_per_var[vname] = {}
        for emo in EMOTIONS:
            target_path = JVNV_DIR / f"jvnv_F1_{emo}.wav"
            cers_emo, hyps_emo, wavs_emo = [], [], []
            for rep in range(REPS):
                source = EPS3_DIR / f"content_donor_{emo}_rep{rep}.wav"
                with torch.no_grad():
                    out_sr, conv = convert_voice_with_emotion(
                        svc,
                        source_path=source,
                        target_path=target_path,
                        emotion_audio_path=target_path,
                        emotion_embedding=emo_embeddings[emo],
                        alpha=1.0, beta=beta,
                        diffusion_steps=STEPS,
                        inference_cfg_rate=CFG,
                    )
                conv = np.asarray(conv, dtype=np.float32)
                import librosa
                conv_24k = librosa.resample(conv, orig_sr=out_sr, target_sr=24000)
                out_wav = OUT_DIR / f"{vname}_{emo}_rep{rep}.wav"
                save_wav(out_wav, conv_24k, 24000)
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
    print(f"{'variant':<10} {'cer_med':>8} {'q1':>8} {'q3':>8} {'best-of-8':>10}",
          flush=True)
    for vname, cers in flat.items():
        med = statistics.median(cers)
        q1  = statistics.quantiles(cers, n=4)[0]
        q3  = statistics.quantiles(cers, n=4)[2]
        bo  = statistics.median([best_per_var[vname][e][2] for e in EMOTIONS])
        print(f"{vname:<10} {med:>8.3f} {q1:>8.3f} {q3:>8.3f} {bo:>10.3f}",
              flush=True)

    pkt = OUT_DIR / "_listening"
    pkt.mkdir(exist_ok=True)
    for emo in EMOTIONS:
        shutil.copy2(JVNV_DIR / f"jvnv_F1_{emo}.wav",
                     pkt / f"{emo}__01_F1_ref.wav")
        for vname in flat:
            src, _, cer = best_per_var[vname][emo]
            shutil.copy2(src, pkt / f"{emo}__{vname}__cer{cer:.3f}.wav")
    print(f"\n[emo2v] listening packet -> {pkt}", flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "rows": rows,
            "best_per_variant": {
                v: {e: {"src": str(t[0]), "hyp": t[1], "cer": t[2]}
                    for e, t in d.items()}
                for v, d in best_per_var.items()},
            "config": {"cfg": CFG, "steps": STEPS, "betas": BETAS,
                       "emotions": EMOTIONS, "reps": REPS},
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
