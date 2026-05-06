"""'Reverse' accent test — ref is a non-JP voice, target is Japanese.

The user is a Japanese native speaker, so they can directly judge whether
each output sounds like natural Japanese vs accented Japanese.

Setup:
  ref_audio  = selfgen_en.wav  (OmniVoice-synthesized "F1 speaks English")
  target text = Japanese
  v_lang      = h(JP_clone) - h(EN_clone)   (existing v_lang_en)
  accent_α    = swept negative-to-positive

  delta = -accent_α * v_lang
        = -accent_α * (h(JP) - h(EN))
        = +accent_α * (h(EN) - h(JP))

So with accent_α < 0 we push hidden TOWARD h(JP), which should make the
output sound more natively Japanese.

Example:
    python scripts/accent_jp_target_test.py --reps 2
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch


def _load_npz(path: Path) -> dict[int, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {int(k.split("_", 1)[1]): np.asarray(data[k]).astype(np.float32)
            for k in data.files if k.startswith("layer_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multilang-dir",
                    default="outputs/multilang_F1_anger_sad_fear_happy_surprise_disgust",
                    type=Path)
    ap.add_argument("--ref-audio", default=None, type=Path,
                    help="Defaults to lang_pair_clones/selfgen_en.wav")
    ap.add_argument("--target-text",
                    default="明日の天気予報は曇り時々晴れです。",
                    help="Japanese text — user will judge naturalness")
    ap.add_argument("--accent-grid", default="-2.0,-1.0,-0.5,0.0,0.5",
                    help="negative pushes toward JP, positive away from JP")
    ap.add_argument("--layers", default="8,12,16")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--output-dir", default=None, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", args.hf_home)

    out_dir = args.output_dir or (args.multilang_dir.parent / "accent_jp_target")
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    grid   = [float(x) for x in args.accent_grid.split(",")]
    ref    = args.ref_audio or (args.multilang_dir / "lang_pair_clones/selfgen_en.wav")
    if not ref.exists():
        raise FileNotFoundError(ref)

    # Load v_lang_en — direction we want to push along
    v_lang = _load_npz(args.multilang_dir / "v_lang_en.npz")
    print(f"[ovet] v_lang layers: {sorted(v_lang)}", flush=True)
    print(f"[ovet] ref: {ref}", flush=True)
    print(f"[ovet] target text (JP): {args.target_text}", flush=True)

    sys.path.insert(0, "src")
    from ovet.omnivoice.wrapper import OmniVoiceWrapper, SteeringConfig
    from ovet.utils.io import save_wav

    print("[ovet] Loading OmniVoice ...", flush=True)
    w = OmniVoiceWrapper(hf_home=args.hf_home)
    ref_text = w.transcribe(ref, language=None)
    print(f"[ovet] ref_text (auto): {ref_text!r}", flush=True)

    print("\n=== JP target, EN-y ref, accent_α sweep ===", flush=True)
    for accent_alpha in grid:
        for rep in range(args.reps):
            sc = SteeringConfig(
                enabled=True,
                alpha=0.0,                   # no emotion steering
                layer_ids=layers,
                emotion_vector=None,
                language_vector=v_lang,
                projection_removal=False,
                language_alpha=accent_alpha,
            )
            torch.manual_seed(args.seed + rep)
            audio = w.generate(
                text=args.target_text, language="Japanese",
                ref_audio=str(ref), ref_text=ref_text, steering=sc,
            )
            outp = out_dir / f"acc{accent_alpha:+.1f}_rep{rep}.wav"
            save_wav(outp, audio, w.SAMPLING_RATE)
            print(f"  acc={accent_alpha:+.1f}  rep={rep}  → {outp}", flush=True)

    print(f"\n[ovet] Listen to: {out_dir}/*.wav", flush=True)
    print("\nINTERPRETATION:")
    print("  acc= 0.0  → baseline (no steering, may sound EN-accented JP)")
    print("  acc<-0.0  → pushed toward JP-natural; should sound MORE natural Japanese")
    print("  acc>+0.0  → pushed away from JP; should sound LESS natural Japanese")


if __name__ == "__main__":
    main()
