"""ε-3 with SeedVC as the VC backbone.

OpenVoice ToneColorConverter is structurally limited (= flow anchored
to source); even at tau=1.0 F1's voice doesn't come back. SeedVC is a
DiT + flow-matching zero-shot VC purpose-built for speaker swap.

Pipeline (per emotion × rep):
  1. Source audio = content donor (native voice + clean zh).
  2. Target audio = F1 ref (JVNV recording).
  3. SeedVCWrapper.convert_voice(source, target) → audio with F1
     voice + native zh content.

Sweep dimensions:
  diffusion_steps    ∈ {10, 25}
  inference_cfg_rate ∈ {0.5, 0.7, 0.9}
  ref selection      ∈ {single F1 emo ref, F1 jvnv concat}

Compare with OpenVoice TCC's best result (which had F1 voice missing)
and prior knn-VC (which was muffled).
"""
from __future__ import annotations
import os, sys, json, statistics, shutil
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, "src")
sys.path.insert(0, "upstream/seed-vc")

from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402
from ovet.evaluation.zh_eval import cer_zh                                     # noqa: E402


JVNV_DIR     = Path("baseline/jvnv_samples")
EPS3_DIR     = Path("outputs/eps3_semantic_vc")
OUT_DIR      = Path("outputs/eps3_seedvc")
ZH_HANZI     = "明天的天气预报是多云转晴。"
EMOTIONS     = ["sad", "happy", "anger"]
ALL_EMOS     = ["anger", "disgust", "fear", "happy", "sad", "surprise"]
REPS         = 8
CFGS         = [0.5, 0.7, 0.9]
DIFF_STEPS   = [10, 25]
SR_OUT       = 22050        # SeedVC v1 output is 22050 Hz


def _build_jvnv_concat(out: Path):
    if out.exists(): return out
    chunks, target_sr = [], None
    for e in ALL_EMOS:
        p = JVNV_DIR / f"jvnv_F1_{e}.wav"
        if not p.exists(): continue
        wav, sr = sf.read(p)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if target_sr is None: target_sr = sr
        chunks.append(wav.astype(np.float32))
    sf.write(out, np.concatenate(chunks), target_sr)
    return out


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[svc] loading SeedVC + ASR ...", flush=True)
    asr = ASRAnalyzer()

    from seed_vc_wrapper import SeedVCWrapper
    svc = SeedVCWrapper(device=torch.device("cuda"))

    # Build the long F1 ref once.
    f1_long = _build_jvnv_concat(OUT_DIR / "_F1_jvnv_concat.wav")

    rows = []
    flat: dict[str, list[float]] = {}
    best_per_var: dict[str, dict[str, tuple[Path, str, float]]] = {}

    for cfg in CFGS:
        for steps in DIFF_STEPS:
            for ref_kind in ("emo_ref", "long_ref"):
                vname = f"cfg{cfg:.1f}_steps{steps}_{ref_kind}"
                print(f"\n[svc] === {vname} ===", flush=True)
                cers_cell = []
                best_per_var[vname] = {}
                for emo in EMOTIONS:
                    target = (JVNV_DIR / f"jvnv_F1_{emo}.wav") if ref_kind == "emo_ref" else f1_long
                    cers_emo, hyps_emo, wavs_emo = [], [], []
                    for rep in range(REPS):
                        source = EPS3_DIR / f"content_donor_{emo}_rep{rep}.wav"
                        # SeedVC yields (mp3_bytes, (sr_int, audio_ndarray))
                        # — only the last yield holds the complete output.
                        last = None
                        for item in svc.convert_voice(
                            source=str(source), target=str(target),
                            diffusion_steps=steps, length_adjust=1.0,
                            inference_cfg_rate=cfg, f0_condition=False,
                            stream_output=True,
                        ):
                            last = item
                        if last is None:
                            continue
                        # last[1] = (sample_rate, audio_array)
                        out_sr, converted = last[1]
                        converted = np.asarray(converted, dtype=np.float32)
                        # Resample to 24k for evaluation parity.
                        import librosa
                        converted_24k = librosa.resample(
                            converted, orig_sr=out_sr, target_sr=24000)
                        out_wav = OUT_DIR / f"{vname}_{emo}_rep{rep}.wav"
                        save_wav(out_wav, converted_24k, 24000)
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
                          f"best={cers_emo[best_idx]:.3f}  "
                          f"best_hyp='{hyps_emo[best_idx][:55]}'",
                          flush=True)
                flat[vname] = cers_cell

    print("\n=== summary (sorted by best-of-8 then cer_med) ===", flush=True)
    print(f"{'variant':<28} {'cer_med':>8} {'q1':>8} {'q3':>8} {'best-of-8':>10}",
          flush=True)
    summary = []
    for v, cers in flat.items():
        med = statistics.median(cers)
        q1  = statistics.quantiles(cers, n=4)[0]
        q3  = statistics.quantiles(cers, n=4)[2]
        bo  = statistics.median([best_per_var[v][e][2] for e in EMOTIONS])
        summary.append({"variant": v, "cer_med": med, "q1": q1, "q3": q3,
                        "best_of_8_median": bo})
    for r in sorted(summary, key=lambda r: (r["best_of_8_median"], r["cer_med"])):
        print(f"{r['variant']:<28} {r['cer_med']:>8.3f} {r['q1']:>8.3f} "
              f"{r['q3']:>8.3f} {r['best_of_8_median']:>10.3f}",
              flush=True)

    pkt = OUT_DIR / "_listening"
    pkt.mkdir(exist_ok=True)
    for v in flat:
        for e in EMOTIONS:
            src, _, cer = best_per_var[v][e]
            shutil.copy2(src, pkt / f"{e}__{v}__cer{cer:.3f}.wav")
    # Also copy F1 ref + content donor for direct comparison
    for e in EMOTIONS:
        shutil.copy2(JVNV_DIR / f"jvnv_F1_{e}.wav", pkt / f"{e}__01_F1_ref.wav")
        # best content donor
        content_cands = [(p, asr.transcribe(p, language="zh"))
                         for p in [EPS3_DIR / f"content_donor_{e}_rep{r}.wav"
                                   for r in range(REPS)]]
        best_ct = min(content_cands, key=lambda x: cer_zh(ZH_HANZI, x[1]))
        shutil.copy2(best_ct[0], pkt / f"{e}__03_content_native.wav")
    print(f"\n[svc] listening packet -> {pkt}", flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary,
                   "best_per_variant": {
                       v: {e: {"src": str(t[0]), "hyp": t[1], "cer": t[2]}
                           for e, t in d.items()}
                       for v, d in best_per_var.items()},
                   "config": {"cfgs": CFGS, "steps": DIFF_STEPS,
                              "emotions": EMOTIONS, "reps": REPS}},
                  f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
