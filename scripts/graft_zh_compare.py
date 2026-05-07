"""A/B compare zh outputs with vs without LanguageGrafter.

Variants:

  baseline                  no steering
  phase4_accent_v2          current production (multi-speaker v_lang)
  graft_a1_b1               LanguageGrafter remove=1.0, inject=1.0
  graft_a1_b1_window        same, step_window=(0,4) (echoing v2's window)
  graft_a05_b05             half-strength (sanity)
  graft_a0_b1               inject only, no removal (additive)
  graft_a1_b0               removal only, no inject (= projection removal but on lang subspace not v_lang)

For each: emotion ∈ {sad, happy, anger}, reps=2, F1 ref.

Reports CER + transcripts.
"""
from __future__ import annotations
import os
import sys
import json
import statistics
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper, SteeringConfig            # noqa: E402
from ovet.omnivoice.lang_graft import LanguageGrafter, load_graft_artifacts   # noqa: E402
from ovet.omnivoice.steering import extract_layer_vectors                     # noqa: E402
from ovet.analyzers.asr_analyzer import ASRAnalyzer                           # noqa: E402
from ovet.utils.io import save_wav                                            # noqa: E402


GRAFT_NPZ      = Path("outputs/graft/zh.npz")
JVNV_DIR       = Path("baseline/jvnv_samples")
EMOTIONS       = ["sad", "happy", "anger"]
ZH_TEXT        = "明天的天气预报是多云转晴。"
ZH_FULL        = "Chinese"
JA_TEXT        = "今日の天気予報は曇り時々晴れです。"
JA_FULL        = "Japanese"
LAYERS         = [8, 12]                       # graft sweet spot
REPS           = 2
SEED           = 0
OUT_DIR        = Path("outputs/graft_zh_compare")


def _generate_with_graft(
    w: OmniVoiceWrapper, text: str, language: str,
    ref_audio: Path, ref_text: str,
    Q_per_layer: dict[int, np.ndarray],
    T_per_layer: dict[int, np.ndarray],
    *, remove_alpha: float, inject_beta: float,
    step_window: tuple[int, int | None] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Run OmniVoice generate() while LanguageGrafter is active."""
    torch.manual_seed(seed)
    Q_subset = {L: Q_per_layer[L] for L in LAYERS if L in Q_per_layer}
    T_subset = {L: T_per_layer[L] for L in LAYERS if L in T_per_layer}
    with LanguageGrafter(
        w.model,
        subspace_per_layer=Q_subset, target_per_layer=T_subset,
        remove_alpha=remove_alpha, inject_beta=inject_beta,
        step_window=step_window,
    ):
        audios = w.model.generate(
            text=text, language=language,
            ref_audio=str(ref_audio), ref_text=ref_text,
        )
    return audios[0]


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[zhgr] loading OmniVoice + ASR ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()

    # ---- artifacts ----
    Q_per, T_per, meta = load_graft_artifacts(GRAFT_NPZ)
    print(f"[zhgr] artifacts loaded: layers={list(Q_per.keys())} "
          f"meta={meta}", flush=True)

    # ---- variants definition ----
    variants: list[tuple[str, dict]] = [
        ("baseline",         {"kind": "raw"}),
        ("graft_a1_b0",      {"kind": "graft", "a": 1.0, "b": 0.0}),
        ("graft_a0_b1",      {"kind": "graft", "a": 0.0, "b": 1.0}),
        ("graft_a1_b1",      {"kind": "graft", "a": 1.0, "b": 1.0}),
        ("graft_a05_b05",    {"kind": "graft", "a": 0.5, "b": 0.5}),
        ("graft_a1_b1_w0_4", {"kind": "graft", "a": 1.0, "b": 1.0,
                              "step_window": (0, 4)}),
        ("graft_a1_b1_w0_8", {"kind": "graft", "a": 1.0, "b": 1.0,
                              "step_window": (0, 8)}),
    ]

    # ---- sweep ----
    rows: list[dict] = []
    for emo in EMOTIONS:
        ref_path = JVNV_DIR / f"jvnv_F1_{emo}.wav"
        ref_text = w.transcribe(ref_path, language=None)
        print(f"\n[zhgr] === emo={emo} ref={ref_path.name} ===", flush=True)

        for vname, vcfg in variants:
            cers = []
            transcripts = []
            for rep in range(REPS):
                if vcfg["kind"] == "raw":
                    torch.manual_seed(SEED + rep)
                    audio = w.model.generate(
                        text=ZH_TEXT, language=ZH_FULL,
                        ref_audio=str(ref_path), ref_text=ref_text,
                    )[0]
                else:
                    audio = _generate_with_graft(
                        w, text=ZH_TEXT, language=ZH_FULL,
                        ref_audio=ref_path, ref_text=ref_text,
                        Q_per_layer=Q_per, T_per_layer=T_per,
                        remove_alpha=vcfg["a"], inject_beta=vcfg["b"],
                        step_window=vcfg.get("step_window"),
                        seed=SEED + rep,
                    )
                wav_path = OUT_DIR / f"zh_{emo}_{vname}_rep{rep}.wav"
                save_wav(wav_path, audio, w.SAMPLING_RATE)
                hyp = asr.transcribe(wav_path, language="zh")
                cer = asr.content_error(wav_path, ZH_TEXT, language="zh")
                cers.append(cer)
                transcripts.append(hyp)

            agg = {
                "median": statistics.median(cers),
                "mean":   statistics.fmean(cers),
            }
            rows.append({
                "emotion": emo, "variant": vname,
                "cer": agg, "cers": cers,
                "transcripts": transcripts,
            })
            print(f"  {vname:<22} CER={agg['median']:.3f} (median)  hyps:")
            for t in transcripts:
                print(f"      {t[:80]}")

    print("\n=== summary (median CER per emotion × variant) ===", flush=True)
    print(f"{'variant':<22} " + " ".join(f"{e:<10}" for e in EMOTIONS) + "  overall")
    by_var: dict[str, dict[str, float]] = {}
    for r in rows:
        by_var.setdefault(r["variant"], {})[r["emotion"]] = r["cer"]["median"]
    overall_med: dict[str, float] = {}
    for v in by_var:
        per_emo = [by_var[v][e] for e in EMOTIONS if e in by_var[v]]
        overall_med[v] = statistics.median(per_emo)
    for v in sorted(by_var.keys(), key=lambda x: overall_med[x]):
        line = f"{v:<22} " + " ".join(
            f"{by_var[v].get(e, float('nan')):>10.3f}" for e in EMOTIONS
        ) + f"  {overall_med[v]:.3f}"
        print(line)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "config": {
            "layers": LAYERS, "reps": REPS, "seed": SEED,
            "graft_meta": meta,
        }}, f, indent=2, ensure_ascii=False)
    print(f"\n[zhgr] saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
