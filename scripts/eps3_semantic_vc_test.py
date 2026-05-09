"""ε-3: semantic codec phoneme rewrite via OmniVoice ×2 + knn-VC.

Pipeline per (text, F1_ref):

  Stage 1  Content donor: feed a NATIVE zh ref into OmniVoice. Output
           audio has the donor's voice but native zh phonetic content
           and proper Mandarin tones — because the LLM is conditioned
           on a native-speaking reference.

  Stage 2  Speaker donor:  feed F1's JVNV ref into OmniVoice. Output
           audio has F1's voice but JP-accented zh content (= our
           original problem).

  Stage 3  knn-VC voice transfer: WavLM-encode the content donor as
           the source utterance and the F1-ref / OmniVoice-F1 audio as
           the matching set. The kNN matcher replaces each source
           feature with the average of its k-nearest neighbours from
           the F1 set, then HiFiGAN vocodes back to waveform.

  Result   F1's voice + native pronunciation, in principle.

The 646-language story stays intact: for any target language, we just
need a small pool of native ref audios (we already pulled FLEURS
samples for 9 languages in build_all_axes).
"""
from __future__ import annotations
import os, sys, json, statistics
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "src")
sys.path.insert(0, "upstream/knn-vc")

from ovet.omnivoice.wrapper import OmniVoiceWrapper                            # noqa: E402
from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402
from ovet.evaluation.zh_eval import (                                          # noqa: E402
    aggregate_zh_cell, format_cell_row, RECOMMENDED_REPS, cer_zh,
)


JVNV_DIR     = Path("baseline/jvnv_samples")
NATIVE_REFS  = sorted(Path("baseline/native_refs/zh").glob("fleurs_zh_dev_*.wav"))
EMOTIONS     = ["sad", "happy", "anger"]      # subset for the smoke test
ZH_FULL      = "Chinese"
ZH_HANZI     = "明天的天气预报是多云转晴。"
REPS         = RECOMMENDED_REPS                # = 8
SEED         = 0
OUT_DIR      = Path("outputs/eps3_semantic_vc")
SR_OMNI      = 24000
SR_KNN       = 16000


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    import librosa
    return librosa.resample(audio.astype(np.float32),
                            orig_sr=src_sr, target_sr=dst_sr)


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("TORCH_HOME", "/workspace/hf_cache/torch")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not NATIVE_REFS:
        # Fall back to the legacy flat layout
        legacy = sorted(Path("baseline/native_refs").glob("fleurs_zh_dev_*.wav"))
        if not legacy:
            raise FileNotFoundError("no native zh refs found")
        native_refs = legacy
    else:
        native_refs = NATIVE_REFS
    print(f"[ε3] native refs: {[p.name for p in native_refs]}", flush=True)

    print("[ε3] loading OmniVoice + ASR + knn-VC ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()

    from hubconf import knn_vc                                                  # noqa: E402
    knnvc = knn_vc(pretrained=True, prematched=True, device="cuda")

    # Pre-compute the F1 matching set per emotion. Use the JVNV ref +
    # the OmniVoice "F1 attempts at zh" generated below — both contain
    # F1 voice, so kNN will match toward those features.
    rows = []
    flat_cers: dict[str, list[float]] = {}

    for emo in EMOTIONS:
        f1_ref_path = JVNV_DIR / f"jvnv_F1_{emo}.wav"
        f1_ref_text = w.transcribe(f1_ref_path, language=None)
        print(f"\n[ε3] === emo={emo}  F1 ref={f1_ref_path.name} ===", flush=True)

        # ---- Speaker donor (F1 attempts at zh) ----
        f1_attempts: list[Path] = []
        for rep in range(REPS):
            torch.manual_seed(SEED + rep)
            audio = w.model.generate(
                text=ZH_HANZI, language=ZH_FULL,
                ref_audio=str(f1_ref_path), ref_text=f1_ref_text,
            )[0]
            wp = OUT_DIR / f"speaker_donor_{emo}_rep{rep}.wav"
            save_wav(wp, audio, SR_OMNI)
            f1_attempts.append(wp)

        # F1 matching set = JVNV ref + 8 F1 attempts (~all in F1's voice).
        f1_set_paths = [f1_ref_path, *f1_attempts]
        # knn-VC expects 16k; resample on disk via temp wavs.
        f1_set_16k_paths = []
        for src in f1_set_paths:
            audio, src_sr = sf.read(src)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio_16k = _resample(audio, src_sr, SR_KNN)
            tp = OUT_DIR / f"_f1_set_16k_{emo}_{src.name}"
            sf.write(tp, audio_16k, SR_KNN)
            f1_set_16k_paths.append(tp)
        matching_set = knnvc.get_matching_set(
            [str(p) for p in f1_set_16k_paths]
        )

        # ---- Content donor (NATIVE ref attempts at zh) ----
        # One attempt per native ref. The native ref's voice is some
        # Chinese narrator; OmniVoice clones that voice, so the content
        # is in native zh phonetic style.
        for rep in range(REPS):
            native_ref = native_refs[rep % len(native_refs)]
            native_ref_text = w.transcribe(native_ref, language="zh")
            torch.manual_seed(SEED + rep)
            audio_native = w.model.generate(
                text=ZH_HANZI, language=ZH_FULL,
                ref_audio=str(native_ref), ref_text=native_ref_text,
            )[0]
            content_path = OUT_DIR / f"content_donor_{emo}_rep{rep}.wav"
            save_wav(content_path, audio_native, SR_OMNI)

            # Resample content donor to 16k for knn-VC
            audio_native_16k = _resample(audio_native, SR_OMNI, SR_KNN)
            content_16k_path = OUT_DIR / f"_content_16k_{emo}_rep{rep}.wav"
            sf.write(content_16k_path, audio_native_16k, SR_KNN)

            # ---- knn-VC swap ----
            query_seq = knnvc.get_features(str(content_16k_path))
            converted = knnvc.match(
                query_seq, matching_set, topk=4,
            ).cpu().numpy()
            # knn-VC outputs at 16k; up-resample to 24k for evaluation
            converted_24k = _resample(converted, SR_KNN, SR_OMNI)
            out_path = OUT_DIR / f"converted_{emo}_rep{rep}.wav"
            save_wav(out_path, converted_24k, SR_OMNI)

        # ---- Evaluate the 3 streams: speaker, content, converted ----
        for stream in ("speaker_donor", "content_donor", "converted"):
            hyps = [
                asr.transcribe(OUT_DIR / f"{stream}_{emo}_rep{rep}.wav",
                               language="zh")
                for rep in range(REPS)
            ]
            stats = aggregate_zh_cell(hyps, ZH_HANZI)
            rows.append({
                "emotion": emo, "stream": stream,
                "stats": stats.__dict__,
            })
            flat_cers.setdefault(stream, []).extend(
                [cer_zh(ZH_HANZI, h) for h in hyps]
            )
            print(f"  {stream:<15} {format_cell_row('', stats)}", flush=True)
            for h in stats.hyps[:3]:
                print(f"      {h[:65]}")

    print("\n=== aggregate (across emotions) ===", flush=True)
    print(f"{'stream':<18} {'cer_med':>8} {'cer_q1':>8} {'cer_q3':>8} "
          f"{'cer_mean':>9} {'σ':>7} {'gap_to_20%':>11}",
          flush=True)
    for stream, cers in flat_cers.items():
        med  = statistics.median(cers)
        q1   = statistics.quantiles(cers, n=4)[0] if len(cers) >= 4 else min(cers)
        q3   = statistics.quantiles(cers, n=4)[2] if len(cers) >= 4 else max(cers)
        mean = statistics.fmean(cers)
        std  = statistics.pstdev(cers)
        gap  = (med - 0.20) * 100
        print(f"{stream:<18} {med:>8.3f} {q1:>8.3f} {q3:>8.3f} "
              f"{mean:>9.3f} {std:>7.3f} {gap:>+10.1f}pp",
              flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "config": {
            "emotions": EMOTIONS, "reps": REPS, "n_native": len(native_refs),
        }}, f, indent=2, ensure_ascii=False)
    print(f"\n[ε3] saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
