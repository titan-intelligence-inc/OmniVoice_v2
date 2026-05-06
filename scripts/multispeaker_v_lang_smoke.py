"""Smoke test for multi-speaker v_lang averaging.

Exercises ``_build_multispeaker_v_lang`` on a tiny case:
  * speakers: [F1, F2] (or whatever is on disk in baseline/jvnv_samples_multi)
  * targets:  [en, zh] (+ ja base)
  * 1 layer  (12)

Reports:
  * |v_lang| per (speaker, target, layer)
  * cosine(F1, F2) per target — should be 0.5–0.9 if the language axis
    is real but speaker-specific noise is non-trivial
  * |averaged| / |single F1| — averaging shouldn't shrink the norm a lot
    if axes line up
  * cosine(averaged, F1-only on disk) — quick reality check

Usage:
    python scripts/multispeaker_v_lang_smoke.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper                   # noqa: E402
from ovet.cli.run_multilang import _build_multispeaker_v_lang         # noqa: E402


SPEAKERS  = ["F1", "F2"]
EMOTION   = "anger"
LAYERS    = [12]
TARGETS   = [
    ("ja", "Japanese", "今日の天気予報は曇り時々晴れです。"),
    ("en", "English",  "The weather forecast for tomorrow is partly cloudy."),
    ("zh", "Chinese",  "明天的天气预报是多云转晴。"),
]
BASE      = "ja"


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    out_dir = Path("outputs/multispeaker_smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[smoke] loading OmniVoice ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])

    # Build cloning_refs by checking both candidate dirs
    refs: list[tuple[str, Path, str]] = []
    for spk in SPEAKERS:
        for d in (Path("baseline/jvnv_samples_multi"), Path("baseline/jvnv_samples")):
            p = d / f"jvnv_{spk}_{EMOTION}.wav"
            if p.exists():
                ref_text = w.transcribe(p, language=None)
                refs.append((spk, p, ref_text))
                print(f"[smoke] {spk}: {p.name} ({ref_text[:36]}...)", flush=True)
                break
        else:
            raise FileNotFoundError(f"no JVNV ref found for {spk}/{EMOTION}")

    base_full = next(f for c, f, _ in TARGETS if c == BASE)
    base_text = next(t for c, _, t in TARGETS if c == BASE)

    print(f"\n[smoke] building multispeaker v_lang on layer {LAYERS} ...", flush=True)
    averaged, per_spk = _build_multispeaker_v_lang(
        w, cloning_refs=refs,
        base_lang_code=BASE, base_lang_full=base_full, base_lang_text=base_text,
        targets=TARGETS, layers=LAYERS, seed=0, num_step=4,
        out_dir=out_dir / "lang_pair_clones",
    )

    print("\n=== per-speaker v_lang norms ===", flush=True)
    L = LAYERS[0]
    for code, _, _ in TARGETS:
        if code == BASE: continue
        for spk in [s for s, _, _ in refs]:
            v = per_spk[spk][code][L]
            print(f"  spk={spk}  target={code}  L={L}  |v|={np.linalg.norm(v):.3f}")

    print("\n=== cosine similarity across speakers (per target) ===", flush=True)
    for code, _, _ in TARGETS:
        if code == BASE: continue
        spks = [s for s, _, _ in refs]
        if len(spks) < 2: continue
        v_a = per_spk[spks[0]][code][L]
        v_b = per_spk[spks[1]][code][L]
        cos = float(np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b) + 1e-9))
        print(f"  target={code}  L={L}  cos({spks[0]},{spks[1]})={cos:+.3f}")

    print("\n=== averaged vs single-speaker (F1) ===", flush=True)
    for code, _, _ in TARGETS:
        if code == BASE: continue
        v_avg = averaged[code][L]
        v_f1  = per_spk["F1"][code][L]
        cos = float(np.dot(v_avg, v_f1) / (np.linalg.norm(v_avg) * np.linalg.norm(v_f1) + 1e-9))
        ratio = np.linalg.norm(v_avg) / (np.linalg.norm(v_f1) + 1e-9)
        print(f"  target={code}  L={L}  cos(avg,F1)={cos:+.3f}  |avg|/|F1|={ratio:.3f}")

    # Compare against existing on-disk F1-only v_lang (if it has matching layer).
    on_disk = Path("outputs/multilang_F1_v2/v_lang_en.npz")
    if on_disk.exists():
        d = np.load(on_disk)
        if f"layer_{L}" in d.files:
            v_disk = d[f"layer_{L}"]
            v_new  = per_spk["F1"]["en"][L]
            cos = float(np.dot(v_disk, v_new) / (
                np.linalg.norm(v_disk) * np.linalg.norm(v_new) + 1e-9))
            print(f"\n[smoke] reproducibility: cos(this-run F1 v_lang_en[L=12], on-disk)={cos:+.3f}")

    print("\n[smoke] DONE", flush=True)


if __name__ == "__main__":
    main()
