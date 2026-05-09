"""ε-3 with OpenVoice ToneColorConverter as the VC backbone.

knn-VC's WavLM averaging blurs spectral detail and discards F1-specific
timbre cues. OpenVoice's ToneColorConverter is purpose-built for
timbre transfer — it explicitly extracts per-utterance "tone color"
embeddings and uses a flow-based decoder that should preserve speaker
identity better.

Pipeline (per emotion × rep):
  1. Use the same content donor wav (native ref → OmniVoice clean zh)
     that won the knn-VC sweep. Cer ≈ 0.0–0.17.
  2. Extract src_se from the content donor itself (= the donor speaker).
  3. Extract tgt_se from F1's audio pool (refs + zh attempts) — average
     several SE so the converter gets a robust F1 timbre target.
  4. ToneColorConverter.convert(content_donor, src_se, tgt_se)
     → audio with F1 timbre + native zh content/tones.
  5. ASR + CER.

Compared to the knn-VC variant we keep:
  * The same OmniVoice 2× generation idea.
  * The same best-of-8 selection.
What changes: the VC step.
"""
from __future__ import annotations
import os, sys, json, statistics, shutil
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "src")
sys.path.insert(0, "upstream/OpenVoice")

# Monkey-patch ToneColorConverter to fix the kwarg leak that breaks
# enable_watermark=False (current OpenVoice main branch passes kwargs
# down to OpenVoiceBaseClass which doesn't accept 'enable_watermark').
from openvoice.api import ToneColorConverter, OpenVoiceBaseClass               # noqa: E402
def _patched_tcc_init(self, *args, **kwargs):
    enable_watermark = kwargs.pop("enable_watermark", True)
    OpenVoiceBaseClass.__init__(self, *args, **kwargs)
    if enable_watermark:
        import wavmark
        self.watermark_model = wavmark.load_model().to(self.device)
    else:
        self.watermark_model = None
    self.version = getattr(self.hps, "_version_", "v1")
ToneColorConverter.__init__ = _patched_tcc_init

from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402
from ovet.evaluation.zh_eval import cer_zh, aggregate_zh_cell                  # noqa: E402
from ovet.postprocessing.openvoice_vc import (                                 # noqa: E402
    OpenVoicePostVC,
)


JVNV_DIR_FLAT     = Path("baseline/jvnv_samples")
EPS3_DIR          = Path("outputs/eps3_semantic_vc")
OUT_DIR           = Path("outputs/eps3_openvoice_tcc")
ZH_HANZI          = "明天的天气预报是多云转晴。"
EMOTIONS          = ["sad", "happy", "anger"]
REPS              = 8
TAUS              = [0.1, 0.3, 0.5]   # ToneColorConverter sampling temperature


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[ovc] loading ASR + OpenVoice TCC ...", flush=True)
    asr = ASRAnalyzer()
    pvc = OpenVoicePostVC()

    # Build target SE: average over several F1 utterances.
    print("[ovc] extracting F1 tgt_se ...", flush=True)
    f1_paths: list[Path] = []
    for e in ("anger", "sad", "fear", "happy", "disgust", "surprise"):
        p = JVNV_DIR_FLAT / f"jvnv_F1_{e}.wav"
        if p.exists(): f1_paths.append(p)
    # Also add a few zh-attempt wavs (longer = better SE estimate)
    for e in EMOTIONS:
        for rep in range(min(REPS, 4)):
            p = EPS3_DIR / f"speaker_donor_{e}_rep{rep}.wav"
            if p.exists(): f1_paths.append(p)
    print(f"  F1 pool: {len(f1_paths)} wavs", flush=True)

    # extract_se accepts a list and averages them.
    tgt_se = pvc.converter.extract_se(
        [str(p) for p in f1_paths]
    ).to(pvc.device)
    print(f"  tgt_se shape: {tuple(tgt_se.shape)}", flush=True)

    # Sweep tau (= temperature). Lower tau = more conservative timbre swap.
    rows = []
    flat: dict[str, list[float]] = {}
    best_per_var: dict[str, dict[str, tuple[Path, str, float]]] = {}

    for tau in TAUS:
        vname = f"tau{tau:.1f}"
        print(f"\n[ovc] === {vname} ===", flush=True)
        cers_cell = []
        best_per_var[vname] = {}
        for emo in EMOTIONS:
            cers_emo = []; hyps_emo = []; wavs_emo = []
            for rep in range(REPS):
                content_path = EPS3_DIR / f"content_donor_{emo}_rep{rep}.wav"
                # Extract src_se from the content donor itself.
                src_se = pvc.converter.extract_se([str(content_path)]).to(pvc.device)
                # The OpenVoice converter expects src/tgt in its native
                # 22.05k space. Read content donor (24k), resample, hand
                # over to converter.
                audio, sr = sf.read(content_path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                # Resample 24k → 22.05k via librosa for the converter
                import librosa
                audio_22k = librosa.resample(audio.astype(np.float32),
                                             orig_sr=sr, target_sr=pvc.SAMPLING_RATE)
                # Write resampled temp, run convert (which reads from disk)
                tmp_in = OUT_DIR / f"_tmp_in_{vname}_{emo}_rep{rep}.wav"
                sf.write(tmp_in, audio_22k, pvc.SAMPLING_RATE)
                converted_22k = pvc.converter.convert(
                    audio_src_path=str(tmp_in),
                    src_se=src_se, tgt_se=tgt_se, output_path=None, tau=tau,
                )
                tmp_in.unlink(missing_ok=True)
                # Resample back to 24k for evaluation
                converted_24k = librosa.resample(
                    converted_22k.astype(np.float32),
                    orig_sr=pvc.SAMPLING_RATE, target_sr=24000,
                )
                wav_path = OUT_DIR / f"{vname}_{emo}_rep{rep}.wav"
                save_wav(wav_path, converted_24k, 24000)
                hyp = asr.transcribe(wav_path, language="zh")
                cer = cer_zh(ZH_HANZI, hyp)
                cers_emo.append(cer); hyps_emo.append(hyp); wavs_emo.append(wav_path)

            stats = aggregate_zh_cell(hyps_emo, ZH_HANZI)
            rows.append({"variant": vname, "emotion": emo,
                         "stats": stats.__dict__})
            cers_cell.extend(cers_emo)
            best_idx = int(np.argmin(cers_emo))
            best_per_var[vname][emo] = (wavs_emo[best_idx], hyps_emo[best_idx],
                                        cers_emo[best_idx])
            print(f"  {emo:<7}  cer_med={statistics.median(cers_emo):.3f}  "
                  f"best={cers_emo[best_idx]:.3f}  "
                  f"best_hyp='{hyps_emo[best_idx][:55]}'", flush=True)
        flat[vname] = cers_cell

    print("\n=== summary ===", flush=True)
    print(f"{'variant':<10} {'cer_med':>8} {'q1':>8} {'q3':>8} {'best-of-8':>10}",
          flush=True)
    summary = []
    for vname in [f"tau{t:.1f}" for t in TAUS]:
        cers = flat[vname]
        med = statistics.median(cers)
        q1  = statistics.quantiles(cers, n=4)[0]
        q3  = statistics.quantiles(cers, n=4)[2]
        bo  = statistics.median([best_per_var[vname][e][2] for e in EMOTIONS])
        summary.append({"variant": vname, "cer_med": med, "q1": q1, "q3": q3,
                        "best_of_8_median": bo})
        print(f"{vname:<10} {med:>8.3f} {q1:>8.3f} {q3:>8.3f} {bo:>10.3f}",
              flush=True)

    # Listening packet
    pkt = OUT_DIR / "_listening"
    pkt.mkdir(exist_ok=True)
    for vname, _, _ in [(f"tau{t:.1f}", None, None) for t in TAUS]:
        for emo in EMOTIONS:
            src_wav, _, cer = best_per_var[vname][emo]
            shutil.copy2(src_wav, pkt / f"{emo}__ovTCC_{vname}__cer{cer:.3f}.wav")
    print(f"\n[ovc] listening packet -> {pkt}", flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary,
                   "best_per_variant": {
                       v: {e: {"src": str(t[0]), "hyp": t[1], "cer": t[2]}
                           for e, t in d.items()}
                       for v, d in best_per_var.items()},
                   "config": {"taus": TAUS, "emotions": EMOTIONS, "reps": REPS,
                              "f1_pool_size": len(f1_paths)}},
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
