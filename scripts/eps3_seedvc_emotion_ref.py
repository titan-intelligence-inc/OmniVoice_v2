"""SeedVC reference-strategy sweep for emotion preservation.

User feedback: sad samples sound angry — emotional prosody isn't
transferring through SeedVC. Hypothesis: the F1 jvnv ref (e.g.
"はあ、人生の意味が…") gives speaker timbre but the source content
donor's neutral prosody dominates. Try richer target refs that include
emotional F1 zh attempts — the style embedding may then carry
emotional spectral cues.

Variants (target ref strategy), with cfg=0.7, steps=100:

  jvnv_emo            single F1 JVNV ref of matching emotion (current)
  zh_attempt          all 8 F1 zh attempts of matching emotion
                      (= speaker_donor wavs — F1 emotional voice +
                       JP-accented zh content)
  jvnv_plus_zh        JVNV ref + 8 zh attempts (richest of all)
  jvnv_long           concat of all 6 JVNV emotions (emotion-mixed F1)

Plus f0_condition=True experiments with the best ref strategy from
above to see if F0 conditioning helps emotional contour transfer.
"""
from __future__ import annotations
import os, sys, json, statistics, shutil, time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "src")
sys.path.insert(0, "upstream/seed-vc")

from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402
from ovet.evaluation.zh_eval import cer_zh                                     # noqa: E402


JVNV_DIR    = Path("baseline/jvnv_samples")
EPS3_DIR    = Path("outputs/eps3_semantic_vc")
OUT_DIR     = Path("outputs/eps3_seedvc_emotion_ref")
ZH_HANZI    = "明天的天气预报是多云转晴。"
EMOTIONS    = ["sad", "happy", "anger"]
ALL_EMOS    = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
REPS        = 8
STEPS       = 100
CFG         = 0.7

REF_KINDS   = ["jvnv_emo", "zh_attempt", "jvnv_plus_zh", "jvnv_long"]
F0_VARIANTS = [(False, True), (True, True), (True, False)]  # (f0_cond, auto_f0_adjust)


def _build_target(emo: str, kind: str, scratch: Path) -> Path:
    """Return a single wav path to feed as SeedVC target. Concatenate
    multiple sources into one file so the speaker embedding sees them
    all (SeedVC takes a single ref path, not a list)."""
    scratch.mkdir(parents=True, exist_ok=True)
    cache_path = scratch / f"_ref_{emo}_{kind}.wav"
    if cache_path.exists():
        return cache_path

    sources: list[Path] = []
    if kind == "jvnv_emo":
        sources = [JVNV_DIR / f"jvnv_F1_{emo}.wav"]
    elif kind == "zh_attempt":
        sources = [EPS3_DIR / f"speaker_donor_{emo}_rep{r}.wav"
                   for r in range(REPS)]
    elif kind == "jvnv_plus_zh":
        sources = [JVNV_DIR / f"jvnv_F1_{emo}.wav"] + [
            EPS3_DIR / f"speaker_donor_{emo}_rep{r}.wav"
            for r in range(REPS)
        ]
    elif kind == "jvnv_long":
        sources = [JVNV_DIR / f"jvnv_F1_{e}.wav" for e in ALL_EMOS]
    else:
        raise ValueError(kind)

    sources = [p for p in sources if p.exists()]
    if not sources:
        return None

    chunks, target_sr = [], None
    for p in sources:
        wav, sr = sf.read(p)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if target_sr is None: target_sr = sr
        chunks.append(wav.astype(np.float32))
    sf.write(cache_path, np.concatenate(chunks), target_sr)
    return cache_path


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[svc-emo] loading ASR + SeedVC ...", flush=True)
    asr = ASRAnalyzer()

    from seed_vc_wrapper import SeedVCWrapper
    svc = SeedVCWrapper(device=torch.device("cuda"))

    rows = []
    flat: dict[str, list[float]] = {}
    best_per_var: dict[str, dict[str, tuple[Path, str, float]]] = {}

    # Phase 1: target ref strategy sweep at f0_condition=False (default).
    print("\n[svc-emo] === Phase 1: ref strategies (f0_condition=False) ===",
          flush=True)
    for ref_kind in REF_KINDS:
        vname = f"f0F_{ref_kind}"
        print(f"\n[svc-emo] --- {vname} ---", flush=True)
        cers_cell = []
        best_per_var[vname] = {}
        for emo in EMOTIONS:
            target = _build_target(emo, ref_kind, OUT_DIR / "_refs")
            if target is None: continue
            cers_emo, hyps_emo, wavs_emo = [], [], []
            for rep in range(REPS):
                source = EPS3_DIR / f"content_donor_{emo}_rep{rep}.wav"
                last = None
                for item in svc.convert_voice(
                    source=str(source), target=str(target),
                    diffusion_steps=STEPS, length_adjust=1.0,
                    inference_cfg_rate=CFG, f0_condition=False,
                    stream_output=True,
                ):
                    last = item
                if last is None: continue
                out_sr, conv = last[1]
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

    # Phase 2: with the (likely) best ref strategy, sweep f0_condition.
    # We pick "jvnv_plus_zh" as the most informative ref (richest emotional
    # F1 content) — the user can compare with other phase-1 variants.
    best_kind = "jvnv_plus_zh"
    print(f"\n[svc-emo] === Phase 2: f0_condition × auto_f0 (ref={best_kind}) ===",
          flush=True)
    for f0_cond, auto_adj in F0_VARIANTS:
        vname = f"f0{int(f0_cond)}_auto{int(auto_adj)}_{best_kind}"
        if not f0_cond and not auto_adj: continue   # auto_adj only matters when f0_cond=True
        print(f"\n[svc-emo] --- {vname} ---", flush=True)
        cers_cell = []
        best_per_var[vname] = {}
        for emo in EMOTIONS:
            target = _build_target(emo, best_kind, OUT_DIR / "_refs")
            cers_emo, hyps_emo, wavs_emo = [], [], []
            for rep in range(REPS):
                source = EPS3_DIR / f"content_donor_{emo}_rep{rep}.wav"
                last = None
                for item in svc.convert_voice(
                    source=str(source), target=str(target),
                    diffusion_steps=STEPS, length_adjust=1.0,
                    inference_cfg_rate=CFG, f0_condition=f0_cond,
                    auto_f0_adjust=auto_adj, stream_output=True,
                ):
                    last = item
                if last is None: continue
                out_sr, conv = last[1]
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
    print(f"{'variant':<32} {'cer_med':>8} {'q1':>8} {'q3':>8} {'best-of-8':>10}",
          flush=True)
    for vname, cers in flat.items():
        med = statistics.median(cers)
        q1  = statistics.quantiles(cers, n=4)[0]
        q3  = statistics.quantiles(cers, n=4)[2]
        bo  = statistics.median([best_per_var[vname][e][2] for e in EMOTIONS])
        print(f"{vname:<32} {med:>8.3f} {q1:>8.3f} {q3:>8.3f} {bo:>10.3f}",
              flush=True)

    # Listening packet
    pkt = OUT_DIR / "_listening"
    pkt.mkdir(exist_ok=True)
    for emo in EMOTIONS:
        shutil.copy2(JVNV_DIR / f"jvnv_F1_{emo}.wav",
                     pkt / f"{emo}__01_F1_ref.wav")
        for vname in flat:
            src, _, cer = best_per_var[vname][emo]
            shutil.copy2(src, pkt / f"{emo}__{vname}__cer{cer:.3f}.wav")
    print(f"\n[svc-emo] listening packet -> {pkt}", flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "rows": rows,
            "best_per_variant": {
                v: {e: {"src": str(t[0]), "hyp": t[1], "cer": t[2]}
                    for e, t in d.items()}
                for v, d in best_per_var.items()},
            "config": {"cfg": CFG, "steps": STEPS, "emotions": EMOTIONS,
                       "reps": REPS, "ref_kinds": REF_KINDS},
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
