"""Build 1D ja→zh axis artifacts for the LanguageAxisGrafter.

Re-extracts hiddens from:
  * outputs/disentanglement_v2/clones/  (48 clones, 4 spk × 4 lang × 3 emo)
  * baseline/native_refs/fleurs_zh_dev_*.wav  (5 native zh refs)

Then computes:
  * d_L = unit(mean(H | lang=zh) - mean(H | lang=ja))   per layer
  * target_c_L = mean(H_native_zh @ d_L)                 per layer

Saves to outputs/graft/zh_axis.npz.

For diagnostic, also reports:
  * c(ja_clones), c(zh_clones), c(en_clones), c(ko_clones)  — coords
    of the existing clones along d (sanity: ja << zh, en/ko mid)
  * |d|  (always 1, but printed)
  * target_c relative to clone-zh mean (how much further is FLEURS?)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper                            # noqa: E402
from ovet.omnivoice.steering import extract_layer_vectors                      # noqa: E402
from ovet.omnivoice.lang_graft import (                                        # noqa: E402
    compute_lang_axis, compute_axis_coord, save_axis_artifacts,
)


CLONES_DIR  = Path("outputs/disentanglement_v2/clones")
NATIVE_REFS = sorted(Path("baseline/native_refs").glob("fleurs_zh_dev_*.wav"))
LAYERS      = [4, 8, 12, 16]
NUM_STEP    = 4
SEED        = 0
OUT_PATH    = Path("outputs/graft/zh_axis.npz")
HIDDEN_CACHE = Path("outputs/graft/hidden_cache.npz")


def _parse_clone(p: Path) -> tuple[str, str, str]:
    spk, emo, lang = p.stem.split("_")
    return spk, emo, lang


def _extract_or_load(w: OmniVoiceWrapper, paths: list[Path]) -> dict[int, np.ndarray]:
    """Cache-aware extraction. Cache key = sorted file list + layers."""
    key = "|".join(p.name for p in paths) + f"|L={LAYERS}"
    if HIDDEN_CACHE.exists():
        d = np.load(HIDDEN_CACHE, allow_pickle=True)
        cached_key = str(d.get("__key__", "")).strip("[]'\"")
        if cached_key == key:
            print("[axis] using hidden cache", flush=True)
            return {L: np.asarray(d[f"L{L}"]) for L in LAYERS}
    print(f"[axis] extracting hiddens for {len(paths)} wavs ...", flush=True)
    H_per_layer: dict[int, list[np.ndarray]] = {l: [] for l in LAYERS}
    for p in paths:
        vecs = extract_layer_vectors(
            w, p, LAYERS, ref_text=None, num_step=NUM_STEP, seed=SEED,
        )
        for l in LAYERS:
            H_per_layer[l].append(vecs[l])
    H = {L: np.stack(H_per_layer[L]).astype(np.float32) for L in LAYERS}
    HIDDEN_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(HIDDEN_CACHE, **{f"L{L}": H[L] for L in LAYERS}, __key__=key)
    return H


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    print("[axis] loading OmniVoice ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])

    clones = sorted(CLONES_DIR.glob("*.wav"))
    print(f"[axis] {len(clones)} clones, {len(NATIVE_REFS)} native refs", flush=True)
    if not clones:
        raise FileNotFoundError(CLONES_DIR)
    if not NATIVE_REFS:
        raise FileNotFoundError("baseline/native_refs/")

    all_paths = clones + NATIVE_REFS
    H_all = _extract_or_load(w, all_paths)
    H_clones = {L: H_all[L][:len(clones)] for L in LAYERS}
    H_native = {L: H_all[L][len(clones):] for L in LAYERS}

    langs = [_parse_clone(p)[2] for p in clones]
    print(f"[axis] clone lang distribution: "
          f"{ {l: langs.count(l) for l in set(langs)} }", flush=True)

    axis: dict[int, np.ndarray] = {}
    target_c: dict[int, float] = {}
    for L in LAYERS:
        d = compute_lang_axis(H_clones[L], langs,
                              source_lang="ja", target_lang="zh")
        axis[L] = d
        # Coord stats for sanity
        c_clones = compute_axis_coord(H_clones[L], d)
        c_native = compute_axis_coord(H_native[L], d)
        c_per_lang = {l: float(np.mean([c_clones[i] for i, ll in enumerate(langs)
                                        if ll == l]))
                      for l in sorted(set(langs))}
        c_native_mean = float(np.mean(c_native))
        target_c[L] = c_native_mean

        print(f"[axis] L{L}: |d|={np.linalg.norm(d):.4f}  "
              f"c(ja)={c_per_lang['ja']:+.2f}  c(en)={c_per_lang['en']:+.2f}  "
              f"c(ko)={c_per_lang['ko']:+.2f}  c(zh)={c_per_lang['zh']:+.2f}  "
              f"c(FLEURS)={c_native_mean:+.2f}",
              flush=True)
        # Sanity: ja should be most negative (source), zh most positive
        # (target). FLEURS should be at or beyond zh.

    save_axis_artifacts(OUT_PATH, axis, target_c, meta={
        "lang_code": "zh", "source_lang": "ja",
        "n_clones": len(clones), "n_native": len(NATIVE_REFS),
        "layers": LAYERS, "num_step": NUM_STEP,
    })
    print(f"\n[axis] saved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
