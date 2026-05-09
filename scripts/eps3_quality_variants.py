"""ε-3 quality variants — fight the kNN-VC HiFiGAN spectral blur.

CER-optimal config (xl_pm0_k8) sounds muffled. Try variants that trade
some CER for sharper / less averaged output.

Variants:
  prod_pm0_k8        production CER winner (baseline for comparison)
  pm1_k8             prematched HiFiGAN (cleaner vocoder distribution)
  pm0_k1             no averaging (each frame uses its single nearest)
  pm0_k2             minimal averaging
  pm1_k1             prematched + no averaging
  pm1_k2             prematched + minimal averaging
  pm0_k8_eq          prod + 4-8kHz EQ shelf boost
  pm1_k4             prematched + middle ground

For each variant, generate the 8 reps × 3 emotions and evaluate CER
(median + best-of-8). The user listens to ``best-of-8`` per variant.
"""
from __future__ import annotations
import os, sys, json, statistics, shutil
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import scipy.signal as sps

sys.path.insert(0, "src")
sys.path.insert(0, "upstream/knn-vc")

from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402
from ovet.evaluation.zh_eval import cer_zh, aggregate_zh_cell                  # noqa: E402


JVNV_DIR_FLAT     = Path("baseline/jvnv_samples")
JVNV_DIR_MULTI    = Path("baseline/jvnv_samples_multi")
DISENT_CLONES_DIR = Path("outputs/disentanglement_v2/clones")
EPS3_DIR          = Path("outputs/eps3_semantic_vc")
LIST_DIR          = Path("outputs/eps3_listening")
OUT_DIR           = Path("outputs/eps3_quality_variants")
ZH_HANZI          = "明天的天气预报是多云转晴。"
SR_OMNI           = 24000
SR_KNN            = 16000
EMOTIONS          = ["sad", "happy", "anger"]
ALL_EMOTIONS      = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
REPS              = 8


VARIANTS = [
    # name,        prematched, topk, eq_boost
    ("pm0_k8",     False, 8,  False),   # current production (CER winner)
    ("pm1_k8",     True,  8,  False),
    ("pm0_k1",     False, 1,  False),
    ("pm0_k2",     False, 2,  False),
    ("pm1_k1",     True,  1,  False),
    ("pm1_k2",     True,  2,  False),
    ("pm1_k4",     True,  4,  False),
    ("pm0_k8_eq",  False, 8,  True),
]


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    import librosa
    return librosa.resample(audio.astype(np.float32),
                            orig_sr=src_sr, target_sr=dst_sr)


def _to_16k(src: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"_16k_{src.stem}.wav"
    if out.exists():
        return out
    audio, sr = sf.read(src)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_16k = _resample(audio, sr, SR_KNN)
    sf.write(out, audio_16k, SR_KNN)
    return out


def _build_xl_set(cache_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for d in (JVNV_DIR_FLAT, JVNV_DIR_MULTI):
        for e in ALL_EMOTIONS:
            p = d / f"jvnv_F1_{e}.wav"
            if p.exists() and p not in paths:
                paths.append(p)
    for e in EMOTIONS:
        for rep in range(REPS):
            p = EPS3_DIR / f"speaker_donor_{e}_rep{rep}.wav"
            if p.exists():
                paths.append(p)
    for p in sorted(DISENT_CLONES_DIR.glob("F1_*.wav")):
        paths.append(p)
    seen, dedup = set(), []
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            dedup.append(p)
    return [_to_16k(p, cache_dir) for p in dedup]


def _highshelf_eq(audio: np.ndarray, sr: int, gain_db: float = 6.0,
                  fc: float = 4000.0) -> np.ndarray:
    """Apply a high-shelf filter to boost frequencies above fc.

    Restores some perceived clarity lost in kNN-VC spectral averaging.
    """
    # Biquad high-shelf, RBJ cookbook
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / sr
    cosw = np.cos(w0)
    sinw = np.sin(w0)
    S = 1.0
    alpha = sinw / 2 * np.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    sqrtA = np.sqrt(A)

    b0 =     A * ((A + 1) + (A - 1) * cosw + 2 * sqrtA * alpha)
    b1 =-2 * A * ((A - 1) + (A + 1) * cosw)
    b2 =     A * ((A + 1) + (A - 1) * cosw - 2 * sqrtA * alpha)
    a0 =         (A + 1) - (A - 1) * cosw + 2 * sqrtA * alpha
    a1 = 2 *    ((A - 1) - (A + 1) * cosw)
    a2 =         (A + 1) - (A - 1) * cosw - 2 * sqrtA * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return sps.lfilter(b, a, audio).astype(np.float32)


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("TORCH_HOME", "/workspace/hf_cache/torch")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache16 = OUT_DIR / "_cache_16k"

    print("[ε3-q] loading ASR + knn-VC (both vocoders) ...", flush=True)
    asr = ASRAnalyzer()
    from hubconf import knn_vc                                                 # noqa: E402
    knnvcs = {pm: knn_vc(pretrained=True, prematched=pm, device="cuda")
              for pm in (False, True)}

    print("[ε3-q] building xl matching set ...", flush=True)
    xl_paths = _build_xl_set(cache16)
    print(f"  {len(xl_paths)} F1 wavs", flush=True)
    matching_sets = {pm: knnvcs[pm].get_matching_set([str(p) for p in xl_paths])
                     for pm in (False, True)}

    rows = []
    flat_cers: dict[str, list[float]] = {}
    best_per_variant: dict[str, dict[str, tuple[Path, str, float]]] = {}

    for vname, pm, topk, eq in VARIANTS:
        print(f"\n[ε3-q] === variant={vname}  pm={pm} topk={topk} eq={eq} ===", flush=True)
        cers_cell = []
        best_per_variant[vname] = {}
        for emo in EMOTIONS:
            wavs = []
            cers_emo = []
            hyps_emo = []
            for rep in range(REPS):
                content_path = EPS3_DIR / f"content_donor_{emo}_rep{rep}.wav"
                content_16k = _to_16k(content_path, cache16)
                query = knnvcs[pm].get_features(str(content_16k))
                converted = knnvcs[pm].match(
                    query, matching_sets[pm], topk=topk,
                ).cpu().numpy()
                converted_24k = _resample(converted, SR_KNN, SR_OMNI)
                if eq:
                    converted_24k = _highshelf_eq(converted_24k, SR_OMNI,
                                                  gain_db=6.0, fc=4000.0)
                wav_path = OUT_DIR / f"{vname}_{emo}_rep{rep}.wav"
                save_wav(wav_path, converted_24k, SR_OMNI)
                hyp = asr.transcribe(wav_path, language="zh")
                cer = cer_zh(ZH_HANZI, hyp)
                wavs.append(wav_path); cers_emo.append(cer); hyps_emo.append(hyp)
            stats = aggregate_zh_cell(hyps_emo, ZH_HANZI)
            rows.append({"variant": vname, "emotion": emo, "stats": stats.__dict__})
            cers_cell.extend(cers_emo)
            # remember best rep per (variant, emotion)
            best_idx = int(np.argmin(cers_emo))
            best_per_variant[vname][emo] = (wavs[best_idx], hyps_emo[best_idx],
                                            cers_emo[best_idx])
            print(f"  {emo:<7}  cer_med={statistics.median(cers_emo):.3f}  "
                  f"best={cers_emo[best_idx]:.3f}  best_hyp='{hyps_emo[best_idx][:50]}'",
                  flush=True)
        flat_cers[vname] = cers_cell

    print("\n=== variant summary ===", flush=True)
    print(f"{'variant':<12} {'cer_med':>8} {'q1':>8} {'q3':>8} "
          f"{'best-of-8 med':>14}",
          flush=True)
    summary = []
    for vname, _, _, _ in VARIANTS:
        cers = flat_cers[vname]
        med  = statistics.median(cers)
        q1   = statistics.quantiles(cers, n=4)[0]
        q3   = statistics.quantiles(cers, n=4)[2]
        bo   = statistics.median([best_per_variant[vname][e][2] for e in EMOTIONS])
        summary.append({"variant": vname, "cer_med": med, "q1": q1, "q3": q3,
                        "best_of_8_median": bo})
        print(f"{vname:<12} {med:>8.3f} {q1:>8.3f} {q3:>8.3f} {bo:>14.3f}",
              flush=True)

    # Listening packet: copy best-of-8 of each variant into a flat dir.
    pkt = OUT_DIR / "_listening"
    pkt.mkdir(exist_ok=True)
    for vname, _, _, _ in VARIANTS:
        for emo in EMOTIONS:
            src_wav, hyp, cer = best_per_variant[vname][emo]
            dst = pkt / f"{emo}__{vname}__cer{cer:.3f}.wav"
            shutil.copy2(src_wav, dst)
    print(f"\n[ε3-q] listening packet -> {pkt}", flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary,
                   "best_per_variant": {
                       v: {e: {"src": str(t[0]), "hyp": t[1], "cer": t[2]}
                           for e, t in best_per_variant[v].items()}
                       for v in best_per_variant
                   }}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
