"""ε-3 OpenVoice TCC — push tau higher and try cleaner F1 SE.

Two axes:
  tau    ∈ {0.5, 0.7, 0.9, 1.0}            stronger timbre push
  pool   ∈ {jvnv_only, all_18, jvnv_long}  cleaner F1 source

  jvnv_only:  6 JVNV refs only (cleanest F1 — natural JP speech, no
              accent contamination from OmniVoice zh attempts)
  all_18:     prior pool (6 JVNV refs + 12 zh attempts)
  jvnv_long:  concat the 6 JVNV refs into one long utterance, extract
              SE from that — gives the converter more material to
              average over for a robust F1 timbre estimate
"""
from __future__ import annotations
import os, sys, json, statistics, shutil
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "src")
sys.path.insert(0, "upstream/OpenVoice")

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
from ovet.evaluation.zh_eval import cer_zh                                     # noqa: E402
from ovet.postprocessing.openvoice_vc import OpenVoicePostVC                   # noqa: E402


JVNV_DIR    = Path("baseline/jvnv_samples")
EPS3_DIR    = Path("outputs/eps3_semantic_vc")
OUT_DIR     = Path("outputs/eps3_ovc_strong_tau")
ZH_HANZI    = "明天的天气预报是多云转晴。"
EMOTIONS    = ["sad", "happy", "anger"]
ALL_EMOS    = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
REPS        = 8
TAUS        = [0.5, 0.7, 0.9, 1.0]


def _build_pool(kind: str, scratch: Path) -> list[Path]:
    if kind == "jvnv_only":
        return [JVNV_DIR / f"jvnv_F1_{e}.wav" for e in ALL_EMOS
                if (JVNV_DIR / f"jvnv_F1_{e}.wav").exists()]
    elif kind == "all_18":
        paths = [JVNV_DIR / f"jvnv_F1_{e}.wav" for e in ALL_EMOS
                 if (JVNV_DIR / f"jvnv_F1_{e}.wav").exists()]
        for e in EMOTIONS:
            for r in range(min(REPS, 4)):
                p = EPS3_DIR / f"speaker_donor_{e}_rep{r}.wav"
                if p.exists(): paths.append(p)
        return paths
    elif kind == "jvnv_long":
        # Concatenate all 6 JVNV refs into one long wav for SE extraction.
        scratch.mkdir(parents=True, exist_ok=True)
        cat_path = scratch / "_F1_jvnv_concat.wav"
        if not cat_path.exists():
            chunks = []
            target_sr = None
            for e in ALL_EMOS:
                p = JVNV_DIR / f"jvnv_F1_{e}.wav"
                if not p.exists(): continue
                wav, sr = sf.read(p)
                if wav.ndim > 1: wav = wav.mean(axis=1)
                if target_sr is None: target_sr = sr
                chunks.append(wav.astype(np.float32))
            cat = np.concatenate(chunks)
            sf.write(cat_path, cat, target_sr)
        return [cat_path]
    else:
        raise ValueError(kind)


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[ovc-tau] loading ASR + OpenVoice TCC ...", flush=True)
    asr = ASRAnalyzer()
    pvc = OpenVoicePostVC(enable_watermark=False)

    # Build target SE per pool kind once.
    print("[ovc-tau] building F1 target SEs ...", flush=True)
    tgt_ses = {}
    for kind in ("jvnv_only", "all_18", "jvnv_long"):
        paths = _build_pool(kind, OUT_DIR / "_scratch")
        se = pvc.converter.extract_se([str(p) for p in paths]).to(pvc.device)
        tgt_ses[kind] = se
        print(f"  {kind}: pool size={len(paths)}  se shape={tuple(se.shape)}", flush=True)

    rows = []
    flat = {}
    best_per_var: dict[str, dict[str, tuple[Path, str, float]]] = {}

    for kind in tgt_ses:
        for tau in TAUS:
            vname = f"{kind}_tau{tau:.1f}"
            print(f"\n[ovc-tau] === {vname} ===", flush=True)
            cers_cell = []
            best_per_var[vname] = {}
            for emo in EMOTIONS:
                cers_emo, hyps_emo, wavs_emo = [], [], []
                for rep in range(REPS):
                    content_path = EPS3_DIR / f"content_donor_{emo}_rep{rep}.wav"
                    src_se = pvc.converter.extract_se(
                        [str(content_path)]).to(pvc.device)
                    audio, sr = sf.read(content_path)
                    if audio.ndim > 1: audio = audio.mean(axis=1)
                    import librosa
                    audio_22k = librosa.resample(
                        audio.astype(np.float32),
                        orig_sr=sr, target_sr=pvc.SAMPLING_RATE)
                    tmp = OUT_DIR / f"_tmp_{vname}_{emo}_rep{rep}.wav"
                    sf.write(tmp, audio_22k, pvc.SAMPLING_RATE)
                    converted_22k = pvc.converter.convert(
                        audio_src_path=str(tmp),
                        src_se=src_se, tgt_se=tgt_ses[kind],
                        output_path=None, tau=tau,
                    )
                    tmp.unlink(missing_ok=True)
                    converted_24k = librosa.resample(
                        converted_22k.astype(np.float32),
                        orig_sr=pvc.SAMPLING_RATE, target_sr=24000)
                    out_wav = OUT_DIR / f"{vname}_{emo}_rep{rep}.wav"
                    save_wav(out_wav, converted_24k, 24000)
                    hyp = asr.transcribe(out_wav, language="zh")
                    cer = cer_zh(ZH_HANZI, hyp)
                    cers_emo.append(cer); hyps_emo.append(hyp); wavs_emo.append(out_wav)
                rows.append({"variant": vname, "emotion": emo,
                             "cers": cers_emo, "hyps": hyps_emo})
                cers_cell.extend(cers_emo)
                best_idx = int(np.argmin(cers_emo))
                best_per_var[vname][emo] = (
                    wavs_emo[best_idx], hyps_emo[best_idx], cers_emo[best_idx]
                )
                print(f"  {emo:<7}  cer_med={statistics.median(cers_emo):.3f}  "
                      f"best={cers_emo[best_idx]:.3f}  "
                      f"best_hyp='{hyps_emo[best_idx][:55]}'",
                      flush=True)
            flat[vname] = cers_cell

    # Summary
    print("\n=== summary (sorted by CER best-of-8 median, then cer_med) ===",
          flush=True)
    print(f"{'variant':<22} {'cer_med':>8} {'q1':>8} {'q3':>8} {'best-of-8':>10}",
          flush=True)
    summary = []
    for vname, cers in flat.items():
        med  = statistics.median(cers)
        q1   = statistics.quantiles(cers, n=4)[0]
        q3   = statistics.quantiles(cers, n=4)[2]
        bo   = statistics.median([best_per_var[vname][e][2] for e in EMOTIONS])
        summary.append({"variant": vname, "cer_med": med, "q1": q1, "q3": q3,
                        "best_of_8_median": bo})
    for r in sorted(summary, key=lambda r: (r["best_of_8_median"], r["cer_med"])):
        print(f"{r['variant']:<22} {r['cer_med']:>8.3f} {r['q1']:>8.3f} "
              f"{r['q3']:>8.3f} {r['best_of_8_median']:>10.3f}",
              flush=True)

    # Listening packet
    pkt = OUT_DIR / "_listening"
    pkt.mkdir(exist_ok=True)
    for vname in flat:
        for emo in EMOTIONS:
            src, hyp, cer = best_per_var[vname][emo]
            shutil.copy2(src, pkt / f"{emo}__{vname}__cer{cer:.3f}.wav")
    print(f"\n[ovc-tau] listening packet -> {pkt}", flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary,
                   "best": {v: {e: {"src": str(t[0]), "hyp": t[1], "cer": t[2]}
                                 for e, t in d.items()}
                            for v, d in best_per_var.items()},
                   "config": {"taus": TAUS, "emotions": EMOTIONS, "reps": REPS}},
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
