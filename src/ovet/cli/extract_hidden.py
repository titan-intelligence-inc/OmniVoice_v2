"""Extract per-layer hidden-state means from a reference audio and save to npz.

Useful for offline construction of v_emo / v_lang vectors.

Example:
    python -m ovet.cli.extract_hidden \
        --ref-audio baseline/jvnv_samples/jvnv_F1_anger.wav \
        --layers 8,12,16,20 \
        --output outputs/vectors/anger_F1.npz
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

from ..omnivoice.wrapper import OmniVoiceWrapper
from ..omnivoice.steering import (
    extract_layer_vectors, compute_v_emo, compute_v_lang, save_vectors,
)


def _parse_layers(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Extract Qwen3 layer hidden vectors")
    ap.add_argument("--ref-audio",       required=True, type=Path,
                    help="Reference audio whose hiddens we extract.")
    ap.add_argument("--ref-text",        default=None,
                    help="Optional ref_text; auto-transcribed if absent.")
    ap.add_argument("--layers",          required=True, type=str,
                    help="Comma-separated layer ids (e.g. 8,12,16).")
    ap.add_argument("--output",          required=True, type=Path,
                    help="Where to save the .npz file.")
    ap.add_argument("--num-step",        type=int, default=4,
                    help="Diffusion steps for the probe forward (>=1).")
    ap.add_argument("--seed",            type=int, default=0)
    ap.add_argument("--probe-text",      default="This is a probe.",
                    help="Throwaway target text used to drive forward.")
    ap.add_argument("--probe-language",  default="English")

    # Optional: produce a difference vector instead of a raw mean.
    ap.add_argument("--minus-audio",     default=None, type=Path,
                    help="If given, save (mean(ref) − mean(minus)) per layer "
                         "instead of the raw mean. Used for v_emo / v_lang.")
    ap.add_argument("--minus-text",      default=None,
                    help="ref_text for --minus-audio.")
    ap.add_argument("--mode",            default="raw",
                    choices=["raw", "v_emo", "v_lang"],
                    help="Tagging only — informs --output's metadata.")
    ap.add_argument("--hf-home",         default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    layer_ids = _parse_layers(args.layers)

    print("[ovet] Loading OmniVoice ...", flush=True)
    w = OmniVoiceWrapper(hf_home=args.hf_home)

    if args.minus_audio is None:
        print(f"[ovet] Extracting layer means: {args.ref_audio}", flush=True)
        vectors = extract_layer_vectors(
            wrapper=w, audio_path=args.ref_audio,
            layer_ids=layer_ids, ref_text=args.ref_text,
            probe_text=args.probe_text, probe_language=args.probe_language,
            num_step=args.num_step, seed=args.seed,
        )
        meta = {
            "mode": args.mode,
            "ref_audio": str(args.ref_audio),
            "ref_text": args.ref_text,
            "layer_ids": layer_ids,
            "num_step": args.num_step,
            "seed": args.seed,
        }
    else:
        print(f"[ovet] Computing diff: ({args.ref_audio}) − ({args.minus_audio})", flush=True)
        if args.mode == "v_emo":
            vectors = compute_v_emo(
                w, args.ref_audio, args.minus_audio, layer_ids,
                emotional_text=args.ref_text, neutral_text=args.minus_text,
                seed=args.seed, num_step=args.num_step,
            )
        elif args.mode == "v_lang":
            vectors = compute_v_lang(
                w, args.ref_audio, args.minus_audio, layer_ids,
                l1_text=args.ref_text, l2_text=args.minus_text,
                seed=args.seed, num_step=args.num_step,
            )
        else:
            # default: same arithmetic as v_emo
            vectors = compute_v_emo(
                w, args.ref_audio, args.minus_audio, layer_ids,
                emotional_text=args.ref_text, neutral_text=args.minus_text,
                seed=args.seed, num_step=args.num_step,
            )
        meta = {
            "mode": args.mode,
            "ref_audio": str(args.ref_audio),
            "ref_text": args.ref_text,
            "minus_audio": str(args.minus_audio),
            "minus_text": args.minus_text,
            "layer_ids": layer_ids,
            "num_step": args.num_step,
            "seed": args.seed,
        }

    save_vectors(args.output, vectors, meta)
    print(f"[ovet] Saved {len(vectors)} vectors → {args.output}", flush=True)
    for i, v in vectors.items():
        print(f"  layer {i:>2}: shape={v.shape} mean={v.mean():.4f} std={v.std():.4f} "
              f"|v|₂={float((v ** 2).sum() ** 0.5):.4f}", flush=True)


if __name__ == "__main__":
    main()
