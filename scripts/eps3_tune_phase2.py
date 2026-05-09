"""ε-3 phase-2 tuning — push from 0.292 toward < 0.20.

Building on phase-1 winner (large pool + pm=False + topk=16):

  * Extend the F1 matching set further: 60 → ~100+ wavs by including
    all language clones we have under outputs/disentanglement_v2/clones.
  * Sweep topk ∈ {8, 16, 32, 64, 128} on the extended set.
  * Multi-content-donor ensembling: for each emotion, run knn-VC over
    each of the 8 native-ref attempts, ASR-rank, take the best.
  * Combined: extended set × best topk × multi-content best-of.
"""
from __future__ import annotations
import os, sys, json, statistics
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "src")
sys.path.insert(0, "upstream/knn-vc")

from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402
from ovet.evaluation.zh_eval import (                                          # noqa: E402
    aggregate_zh_cell, format_cell_row, cer_zh,
)


JVNV_DIR_FLAT     = Path("baseline/jvnv_samples")
JVNV_DIR_MULTI    = Path("baseline/jvnv_samples_multi")
DISENT_CLONES_DIR = Path("outputs/disentanglement_v2/clones")
EPS3_DIR          = Path("outputs/eps3_semantic_vc")
TUNE1_DIR         = Path("outputs/eps3_tune")
OUT_DIR           = Path("outputs/eps3_tune_phase2")
ZH_HANZI          = "明天的天气预报是多云转晴。"
SR_OMNI           = 24000
SR_KNN            = 16000
EMOTIONS          = ["sad", "happy", "anger"]
ALL_EMOTIONS      = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
REPS              = 8
TOPKS             = [8, 16, 32, 64, 128]


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


def _build_xlarge_matching_set(emo: str, cache_dir: Path) -> list[Path]:
    """Largest available F1 pool: refs + zh attempts + every cross-lang
    self-clone we have on disk."""
    paths: list[Path] = []
    # JVNV refs (all 6 emotions, both flat and multi dirs)
    for d in (JVNV_DIR_FLAT, JVNV_DIR_MULTI):
        for e in ALL_EMOTIONS:
            p = d / f"jvnv_F1_{e}.wav"
            if p.exists() and p not in paths:
                paths.append(p)
    # zh attempts from phase-1 (3 emotions × 8 reps = 24 wavs)
    for e in EMOTIONS:
        for rep in range(REPS):
            p = EPS3_DIR / f"speaker_donor_{e}_rep{rep}.wav"
            if p.exists():
                paths.append(p)
    # All cross-language F1 self-clones (anger/sad/fear × 4 langs = 12)
    for p in sorted(DISENT_CLONES_DIR.glob("F1_*.wav")):
        paths.append(p)
    # Resolve symlinks and dedupe
    seen: set[str] = set()
    deduped: list[Path] = []
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            deduped.append(p)
    return [_to_16k(p, cache_dir) for p in deduped]


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("TORCH_HOME", "/workspace/hf_cache/torch")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache16 = OUT_DIR / "_cache_16k"

    print("[ε3-tune2] loading ASR + knn-VC ...", flush=True)
    asr = ASRAnalyzer()
    from hubconf import knn_vc                                                 # noqa: E402
    knnvc = knn_vc(pretrained=True, prematched=False, device="cuda")    # phase-1 winner

    # Build XL matching set (one F1 pool, shared across emotions since
    # adding more F1 ref of any emotion only helps).
    print("[ε3-tune2] building xlarge matching set ...", flush=True)
    xl_paths = _build_xlarge_matching_set(EMOTIONS[0], cache16)
    print(f"  {len(xl_paths)} F1 wavs", flush=True)
    xl_match = knnvc.get_matching_set([str(p) for p in xl_paths])

    # Score content donors to identify the best per emotion.
    print("\n[ε3-tune2] scoring content donors per emo ...", flush=True)
    content_cers: dict[str, dict[int, float]] = {}
    for emo in EMOTIONS:
        cers = {}
        for rep in range(REPS):
            p = EPS3_DIR / f"content_donor_{emo}_rep{rep}.wav"
            hyp = asr.transcribe(p, language="zh")
            cers[rep] = cer_zh(ZH_HANZI, hyp)
        content_cers[emo] = cers
        print(f"  {emo}: {sorted(cers.values())} "
              f"(best rep={min(cers, key=cers.get)})", flush=True)

    # Sweep topk on extended pool.
    print(f"\n[ε3-tune2] sweeping {len(TOPKS)} × {len(EMOTIONS)} × "
          f"{REPS} reps = {len(TOPKS) * len(EMOTIONS) * REPS} VC calls",
          flush=True)
    rows = []
    flat = {}
    for topk in TOPKS:
        cers_cell = []
        for emo in EMOTIONS:
            hyps = []
            for rep in range(REPS):
                content_path = EPS3_DIR / f"content_donor_{emo}_rep{rep}.wav"
                content_16k = _to_16k(content_path, cache16)
                query_seq = knnvc.get_features(str(content_16k))
                converted = knnvc.match(query_seq, xl_match, topk=topk).cpu().numpy()
                converted_24k = _resample(converted, SR_KNN, SR_OMNI)
                out_wav = OUT_DIR / f"xl_k{topk}_{emo}_rep{rep}.wav"
                save_wav(out_wav, converted_24k, SR_OMNI)
                hyps.append(asr.transcribe(out_wav, language="zh"))
            stats = aggregate_zh_cell(hyps, ZH_HANZI)
            rows.append({"cell": f"xl_k{topk}", "emotion": emo,
                         "stats": stats.__dict__})
            cers_cell.extend([cer_zh(ZH_HANZI, h) for h in hyps])
        flat[f"xl_k{topk}"] = cers_cell
        print(f"  xl_k{topk:<3}  cer_med={statistics.median(cers_cell):.3f}  "
              f"q1={statistics.quantiles(cers_cell, n=4)[0]:.3f}  "
              f"q3={statistics.quantiles(cers_cell, n=4)[2]:.3f}",
              flush=True)

    # Multi-content best-of: for each topk, ASR-rank the 8 reps and
    # report the best per emotion.
    print("\n=== best-of-8 per topk × emotion (lowest CER among 8 content donors) ===",
          flush=True)
    best_per_topk: dict[str, list[float]] = {}
    for topk in TOPKS:
        cell = f"xl_k{topk}"
        best_cers = []
        for emo in EMOTIONS:
            emo_cers = []
            for rep in range(REPS):
                wav = OUT_DIR / f"{cell}_{emo}_rep{rep}.wav"
                hyp = asr.transcribe(wav, language="zh")
                emo_cers.append(cer_zh(ZH_HANZI, hyp))
            best_cers.append(min(emo_cers))
            print(f"  {cell:<10} {emo:<7}  best={min(emo_cers):.3f}  "
                  f"all={[f'{c:.2f}' for c in sorted(emo_cers)]}",
                  flush=True)
        best_per_topk[cell] = best_cers
        med = statistics.median(best_cers)
        print(f"  {cell}: best-of-8 median across emos = {med:.3f}", flush=True)

    print("\n=== final ranking (median across emotions) ===", flush=True)
    print(f"{'cell':<10} {'cer_med (8 reps)':>16} {'cer_med (best-of-8)':>20} {'gap_to_20%':>12}")
    final = []
    for cell, cers in flat.items():
        med_all = statistics.median(cers)
        bo = statistics.median(best_per_topk[cell])
        gap = (bo - 0.20) * 100
        final.append({"cell": cell, "cer_med_all": med_all,
                      "cer_med_best_of_8": bo, "gap_pp": gap})
        print(f"{cell:<10} {med_all:>16.3f} {bo:>20.3f} {gap:>+11.1f}pp",
              flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "rows": rows, "best_per_topk": best_per_topk,
            "final": final,
            "config": {"topks": TOPKS, "emotions": EMOTIONS, "reps": REPS,
                       "matching_set_size": len(xl_paths),
                       "matching_set_files": [p.name for p in xl_paths]},
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[ε3-tune2] saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
