"""Build language-graft artifacts (subspace Q + native target) for zh.

  1. Re-extract per-layer hiddens from the 48 disentanglement clones at
     ``outputs/disentanglement_v2/clones/``.
  2. Compute per-layer ``Q_lang`` from the (speaker, lang) labels.
  3. Extract hiddens from native zh refs at ``baseline/native_refs/``,
     project onto Q_lang, average → ``lang_target_zh``.
  4. Save to ``outputs/graft/zh.npz`` for downstream consumption by
     ``LanguageGrafter``.

Run:
    venv/bin/python scripts/build_graft_artifacts.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper                     # noqa: E402
from ovet.omnivoice.steering import extract_layer_vectors               # noqa: E402
from ovet.omnivoice.lang_graft import (                                 # noqa: E402
    compute_lang_subspace, compute_lang_target_from_hiddens,
    save_graft_artifacts,
)


CLONES_DIR     = Path("outputs/disentanglement_v2/clones")
NATIVE_REFS    = sorted(Path("baseline/native_refs").glob("fleurs_zh_dev_*.wav"))
LAYERS         = [4, 8, 12, 16]    # confine to where disentanglement is clean
NUM_STEP       = 4
SEED           = 0
OUT_PATH       = Path("outputs/graft/zh.npz")


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    print("[graft] loading OmniVoice ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])

    # --------------------------------------------------------------
    # 1. Hiddens from existing 48 clones
    # --------------------------------------------------------------
    clones = sorted(CLONES_DIR.glob("*.wav"))
    if not clones:
        raise FileNotFoundError(f"no clones in {CLONES_DIR}")
    print(f"[graft] {len(clones)} clones found", flush=True)

    # Filename: {speaker}_{emotion}_{lang}.wav
    def parse(p: Path) -> tuple[str, str, str]:
        spk, emo, lang_code = p.stem.split("_")
        return spk, emo, lang_code

    H_per_layer: dict[int, list[np.ndarray]] = {l: [] for l in LAYERS}
    langs: list[str] = []
    for p in clones:
        _, _, lang = parse(p)
        vecs = extract_layer_vectors(
            w, p, LAYERS, ref_text=None, num_step=NUM_STEP, seed=SEED,
        )
        for l in LAYERS:
            H_per_layer[l].append(vecs[l])
        langs.append(lang)

    H_clones = {l: np.stack(H_per_layer[l]).astype(np.float32) for l in LAYERS}
    print(f"[graft] clone hiddens: shape={H_clones[LAYERS[0]].shape}", flush=True)

    # --------------------------------------------------------------
    # 2. Compute Q per layer (rank-K, K=4 unique langs)
    # --------------------------------------------------------------
    Q_per: dict[int, np.ndarray] = {}
    for l in LAYERS:
        Q = compute_lang_subspace(H_clones[l], langs)
        Q_per[l] = Q
        print(f"[graft] L{l}: Q shape={Q.shape}", flush=True)

    # --------------------------------------------------------------
    # 3. Hiddens from native zh refs (FLEURS), average over refs
    # --------------------------------------------------------------
    if not NATIVE_REFS:
        raise FileNotFoundError("no native zh refs; run the FLEURS pull first.")
    print(f"[graft] {len(NATIVE_REFS)} native zh refs:", flush=True)
    H_native_per_layer: dict[int, list[np.ndarray]] = {l: [] for l in LAYERS}
    for p in NATIVE_REFS:
        print(f"  {p.name}", flush=True)
        # Pass language=None so Whisper will auto-transcribe;
        # extract_layer_vectors uses ref_text only for prompt
        # text-encoding, which we don't need to be perfect here —
        # we just want the hidden state when the model "sees" this audio.
        try:
            vecs = extract_layer_vectors(
                w, p, LAYERS, ref_text=None, num_step=NUM_STEP, seed=SEED,
            )
        except Exception as e:
            print(f"  WARN extraction failed for {p.name}: {e}", flush=True)
            continue
        for l in LAYERS:
            H_native_per_layer[l].append(vecs[l])
    H_native = {l: np.stack(H_native_per_layer[l]).astype(np.float32) for l in LAYERS}
    print(f"[graft] native hiddens: shape={H_native[LAYERS[0]].shape}", flush=True)

    # --------------------------------------------------------------
    # 4. Project + average → lang_target per layer
    # --------------------------------------------------------------
    T_per: dict[int, np.ndarray] = {}
    for l in LAYERS:
        # Per-ref project, then mean
        T_per[l] = compute_lang_target_from_hiddens(H_native[l], Q_per[l])
        print(f"[graft] L{l}: |T|={np.linalg.norm(T_per[l]):.3f}", flush=True)

    # --------------------------------------------------------------
    # 5. Save
    # --------------------------------------------------------------
    meta = {
        "lang_code":   "zh",
        "n_clones":    len(clones),
        "n_native":    len(NATIVE_REFS),
        "layers":      LAYERS,
        "num_step":    NUM_STEP,
        "native_paths": [str(p) for p in NATIVE_REFS],
    }
    save_graft_artifacts(OUT_PATH, Q_per, T_per, meta=meta)
    print(f"[graft] saved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
