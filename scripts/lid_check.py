"""Whisper-based Language Identification (LID) on the accent sweep wavs.

For each wav, compute P(zh), P(ja), P(en) and the top-1 detected
language. This gives an objective measure of "does this audio sound
Chinese" without requiring a native listener.

Whisper's first decoder step emits a language token. We extract the
language-token logits and softmax them.

Example:
    python scripts/lid_check.py \
        --dir outputs/accent_sweep_zh_sad \
        --target zh \
        --extra ja,en
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",     required=True, type=Path,
                    help="Directory of wav files to score.")
    ap.add_argument("--target",  required=True,
                    help="Target language code (zh / ja / en / ...) — ranked first.")
    ap.add_argument("--extra",   default="ja,en",
                    help="Comma-separated additional language codes to report.")
    ap.add_argument("--model",   default="openai/whisper-large-v3-turbo")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", args.hf_home)

    import soundfile as sf
    import librosa
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print(f"[lid] Loading {args.model} ...", flush=True)
    proc = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to("cuda").eval()
    model = model.to(torch.float16)

    # Build language-token id table — Whisper uses tokens of the form "<|xx|>".
    all_codes = [args.target] + [c for c in args.extra.split(",") if c.strip() and c != args.target]
    lang_token_ids = {}
    for code in all_codes:
        tok = f"<|{code}|>"
        tid = proc.tokenizer.convert_tokens_to_ids(tok)
        if tid is None or tid == proc.tokenizer.unk_token_id:
            raise ValueError(f"Whisper token not found: {tok}")
        lang_token_ids[code] = tid
    print(f"[lid] tracking codes: {list(lang_token_ids)}", flush=True)

    # Special tokens
    sot_id = proc.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    decoder_input = torch.tensor([[sot_id]], dtype=torch.long, device="cuda")

    files = sorted(args.dir.glob("*.wav"))
    print(f"[lid] {len(files)} wav files in {args.dir}\n", flush=True)
    print(f"{'file':<28} {'top':<6} ", end="")
    for c in all_codes:
        print(f"P({c}):  ", end="")
    print()

    @torch.inference_mode()
    def lid(wav_path: Path):
        wav, sr = sf.read(wav_path)
        if sr != 16000:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        feat = proc(wav.astype(np.float32), sampling_rate=16000, return_tensors="pt"
                    ).input_features.to("cuda").to(torch.float16)
        # Encoder pass
        enc = model.model.encoder(feat)
        # First decoder step on SOT
        out = model(decoder_input_ids=decoder_input, encoder_outputs=enc)
        logits = out.logits[0, 0]
        # Softmax over the **language tokens only** (not all 51k tokens)
        ids = list(lang_token_ids.values())
        sub = logits[ids].float()
        probs = torch.softmax(sub, dim=-1).cpu().numpy()
        # Map back to codes
        return {c: float(probs[i]) for i, c in enumerate(lang_token_ids)}

    rows = []
    for f in files:
        ps = lid(f)
        top = max(ps.items(), key=lambda x: x[1])
        rows.append((f.name, top[0], ps))
        print(f"{f.name:<28} {top[0]:<6} ", end="")
        for c in all_codes:
            print(f"{ps[c]:.3f}    ", end="")
        print()

    # Compact summary by accent_alpha if file pattern matches "accNN"
    print("\n=== summary by accent_alpha (mean over reps) ===", flush=True)
    grouped: dict[str, list[dict]] = {}
    for name, _, ps in rows:
        # match acc0.50_rep0.wav style
        if name.startswith("acc"):
            key = name.split("_")[0]   # e.g. "acc0.50"
            grouped.setdefault(key, []).append(ps)
    if grouped:
        print(f"{'acc':<8} ", end="")
        for c in all_codes:
            print(f"P({c})_mean  ", end="")
        print()
        for key in sorted(grouped):
            ps_list = grouped[key]
            avg = {c: float(np.mean([p[c] for p in ps_list])) for c in all_codes}
            print(f"{key:<8} ", end="")
            for c in all_codes:
                print(f"  {avg[c]:.3f}     ", end="")
            print()


if __name__ == "__main__":
    main()
