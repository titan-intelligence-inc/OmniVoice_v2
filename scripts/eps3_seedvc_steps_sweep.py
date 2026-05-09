"""SeedVC diffusion-steps sweep — higher quality push.

Holds cfg=0.7, emo_ref (= F1 jvnv ref of matching emotion). Sweep
diffusion_steps ∈ {25, 50, 100, 200}.

Higher steps = more iterative refinement of the diffusion sampler =
typically better quality at the cost of inference time.

Output: outputs/eps3_seedvc_steps/_listening with best-of-8 per
(emotion, steps).
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
OUT_DIR     = Path("outputs/eps3_seedvc_steps")
ZH_HANZI    = "明天的天气预报是多云转晴。"
EMOTIONS    = ["sad", "happy", "anger"]
REPS        = 8
CFG         = 0.7
STEPS_LIST  = [25, 50, 100, 200]


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[svc-steps] loading ASR + SeedVC ...", flush=True)
    asr = ASRAnalyzer()

    from seed_vc_wrapper import SeedVCWrapper
    svc = SeedVCWrapper(device=torch.device("cuda"))

    rows = []
    flat: dict[str, list[float]] = {}
    best_per_var: dict[str, dict[str, tuple[Path, str, float]]] = {}
    timings: dict[int, list[float]] = {}

    for steps in STEPS_LIST:
        vname = f"steps{steps:03d}"
        print(f"\n[svc-steps] === {vname} (cfg={CFG}, emo_ref) ===", flush=True)
        cers_cell = []
        best_per_var[vname] = {}
        timings[steps] = []
        for emo in EMOTIONS:
            target = JVNV_DIR / f"jvnv_F1_{emo}.wav"
            cers_emo, hyps_emo, wavs_emo = [], [], []
            for rep in range(REPS):
                source = EPS3_DIR / f"content_donor_{emo}_rep{rep}.wav"
                t0 = time.time()
                last = None
                for item in svc.convert_voice(
                    source=str(source), target=str(target),
                    diffusion_steps=steps, length_adjust=1.0,
                    inference_cfg_rate=CFG, f0_condition=False,
                    stream_output=True,
                ):
                    last = item
                t_elapsed = time.time() - t0
                timings[steps].append(t_elapsed)
                if last is None:
                    continue
                out_sr, converted = last[1]
                converted = np.asarray(converted, dtype=np.float32)
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
        avg_t = sum(timings[steps]) / len(timings[steps])
        print(f"  steps={steps}  avg time/sample={avg_t:.1f}s  "
              f"total={sum(timings[steps]):.1f}s", flush=True)

    print("\n=== summary ===", flush=True)
    print(f"{'variant':<14} {'cer_med':>8} {'q1':>8} {'q3':>8} "
          f"{'best-of-8':>10} {'sec/gen':>8}",
          flush=True)
    for steps in STEPS_LIST:
        v = f"steps{steps:03d}"
        cers = flat[v]
        med = statistics.median(cers)
        q1  = statistics.quantiles(cers, n=4)[0]
        q3  = statistics.quantiles(cers, n=4)[2]
        bo  = statistics.median([best_per_var[v][e][2] for e in EMOTIONS])
        avg_t = sum(timings[steps]) / len(timings[steps])
        print(f"{v:<14} {med:>8.3f} {q1:>8.3f} {q3:>8.3f} "
              f"{bo:>10.3f} {avg_t:>7.1f}s",
              flush=True)

    # Listening packet
    pkt = OUT_DIR / "_listening"
    pkt.mkdir(exist_ok=True)
    for emo in EMOTIONS:
        # Add F1 ref for comparison
        shutil.copy2(JVNV_DIR / f"jvnv_F1_{emo}.wav",
                     pkt / f"{emo}__01_F1_ref.wav")
        for steps in STEPS_LIST:
            v = f"steps{steps:03d}"
            src, _, cer = best_per_var[v][emo]
            shutil.copy2(src, pkt / f"{emo}__{v}__cer{cer:.3f}.wav")
    print(f"\n[svc-steps] listening packet -> {pkt}", flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "rows": rows,
            "best_per_variant": {
                v: {e: {"src": str(t[0]), "hyp": t[1], "cer": t[2]}
                    for e, t in d.items()}
                for v, d in best_per_var.items()},
            "timings_seconds_per_sample": {
                str(s): timings[s] for s in STEPS_LIST},
            "config": {"cfg": CFG, "emotions": EMOTIONS, "reps": REPS,
                       "steps": STEPS_LIST},
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
