"""End-to-end multi-language axis builder.

For each non-ja target language T in {en, ko, es, fr, de, pt, ru, vi}
(zh already done):

  1. Generate JVNV self-clones in T  (4 spk × 3 emo × 1 lang = 12 wavs)
     under outputs/disentanglement_v2/clones/  (extends the dataset
     used to build the language axis).
  2. Pull native T audio from FLEURS  (5 wavs / lang)
     under baseline/native_refs/{T}/.
  3. Extract per-layer hiddens for everything (cache-aware).
  4. Compute axis d_T (= unit(mean(H | lang=T) - mean(H | lang=ja))) and
     target_c_T (= mean(H_native_T @ d_T))  for each layer in [4,8,12,16].
  5. Save outputs/graft/{T}_axis.npz.

Existing zh artifacts are left untouched.

Run:
    venv/bin/python scripts/build_all_axes.py
"""
from __future__ import annotations
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper                            # noqa: E402
from ovet.omnivoice.steering import extract_layer_vectors                      # noqa: E402
from ovet.omnivoice.lang_graft import (                                        # noqa: E402
    compute_lang_axis, compute_axis_coord, save_axis_artifacts,
)
from ovet.utils.io import save_wav                                             # noqa: E402


SPEAKERS  = ["F1", "F2", "M1", "M2"]
EMOTIONS  = ["anger", "sad", "fear"]
JVNV_DIR  = Path("baseline/jvnv_samples_multi")
CLONES_DIR = Path("outputs/disentanglement_v2/clones")
NATIVE_ROOT = Path("baseline/native_refs")
OUT_DIR    = Path("outputs/graft")
LAYERS    = [4, 8, 12, 16]
NUM_STEP  = 4
SEED      = 0

# (lang_code, lang_full, target_text, fleurs_config)
LANGS: list[tuple[str, str, str, str]] = [
    ("en", "English",    "The weather forecast for tomorrow is partly cloudy.",
                         "en_us"),
    ("ko", "Korean",     "내일 일기 예보는 가끔 구름이 끼겠습니다.",
                         "ko_kr"),
    ("es", "Spanish",    "El pronóstico del tiempo para mañana es parcialmente nublado.",
                         "es_419"),
    ("fr", "French",     "Les prévisions météo pour demain annoncent un temps partiellement nuageux.",
                         "fr_fr"),
    ("de", "German",     "Die Wettervorhersage für morgen ist teilweise bewölkt.",
                         "de_de"),
    ("pt", "Portuguese", "A previsão do tempo para amanhã é parcialmente nublado.",
                         "pt_br"),
    ("ru", "Russian",    "Прогноз погоды на завтра — переменная облачность.",
                         "ru_ru"),
    ("vi", "Vietnamese", "Dự báo thời tiết ngày mai là có mây rải rác.",
                         "vi_vn"),
]
JA_TEXT = "今日の天気予報は曇り時々晴れです。"


# ---------------------------------------------------------------- 1. clones

def ensure_clones(w: OmniVoiceWrapper) -> None:
    """Generate all needed JVNV self-clones (no-op if already on disk)."""
    CLONES_DIR.mkdir(parents=True, exist_ok=True)
    todo = []
    # ja clones (always needed as source)
    for sp in SPEAKERS:
        for emo in EMOTIONS:
            ref = JVNV_DIR / f"jvnv_{sp}_{emo}.wav"
            out = CLONES_DIR / f"{sp}_{emo}_ja.wav"
            if ref.exists() and not out.exists():
                todo.append((sp, emo, "ja", "Japanese", JA_TEXT, ref, out))
    for lang_code, lang_full, txt, _ in LANGS:
        for sp in SPEAKERS:
            for emo in EMOTIONS:
                ref = JVNV_DIR / f"jvnv_{sp}_{emo}.wav"
                out = CLONES_DIR / f"{sp}_{emo}_{lang_code}.wav"
                if ref.exists() and not out.exists():
                    todo.append((sp, emo, lang_code, lang_full, txt, ref, out))

    print(f"[axes] {len(todo)} new JVNV clones to generate", flush=True)
    for sp, emo, code, full, txt, ref, out in todo:
        ref_text = w.transcribe(ref, language=None)
        torch.manual_seed(SEED)
        audio = w.model.generate(
            text=txt, language=full,
            ref_audio=str(ref), ref_text=ref_text, num_step=8,
        )[0]
        save_wav(out, audio, w.SAMPLING_RATE)
        print(f"  generated {out.name}", flush=True)


# ---------------------------------------------------------------- 2. natives

def ensure_native_refs(n_per_lang: int = 5) -> dict[str, list[Path]]:
    """Pull n_per_lang FLEURS samples per language into baseline/native_refs/{T}/.
    Already-existing files are skipped.
    """
    from huggingface_hub import hf_hub_download
    out: dict[str, list[Path]] = {}
    for lang_code, _, _, fleurs_cfg in LANGS:
        outdir = NATIVE_ROOT / lang_code
        outdir.mkdir(parents=True, exist_ok=True)
        existing = sorted(outdir.glob("fleurs_*_dev_*.wav"))
        if len(existing) >= n_per_lang:
            out[lang_code] = existing[:n_per_lang]
            print(f"[axes] {lang_code}: {len(existing)} native refs already on disk", flush=True)
            continue
        print(f"[axes] {lang_code}: pulling FLEURS {fleurs_cfg} dev tarball ...", flush=True)
        path = hf_hub_download(
            repo_id="google/fleurs",
            filename=f"data/{fleurs_cfg}/audio/dev.tar.gz",
            repo_type="dataset",
        )
        wavs_pulled = []
        with tarfile.open(path) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".wav")][:n_per_lang]
            for i, m in enumerate(members):
                wpath = outdir / f"fleurs_{lang_code}_dev_{i:03d}.wav"
                if not wpath.exists():
                    f = tf.extractfile(m)
                    with open(wpath, "wb") as o:
                        o.write(f.read())
                wavs_pulled.append(wpath)
        # Older zh refs sit at baseline/native_refs/fleurs_zh_dev_*.wav (no
        # subdir). Don't touch them — zh has its own artifact already.
        out[lang_code] = wavs_pulled
    return out


# ---------------------------------------------------------------- 3. hiddens

def extract_hiddens(
    w: OmniVoiceWrapper, paths: list[Path],
) -> dict[int, np.ndarray]:
    """Plain extraction (no cache; the cache logic lives in the per-call
    wrapper if needed). Returns {layer: [N, D]}."""
    H_per: dict[int, list[np.ndarray]] = {l: [] for l in LAYERS}
    for i, p in enumerate(paths):
        if i and i % 10 == 0:
            print(f"  extracted {i}/{len(paths)}", flush=True)
        vecs = extract_layer_vectors(
            w, p, LAYERS, ref_text=None, num_step=NUM_STEP, seed=SEED,
        )
        for L in LAYERS:
            H_per[L].append(vecs[L])
    return {L: np.stack(H_per[L]).astype(np.float32) for L in LAYERS}


# ---------------------------------------------------------------- 4. axes

def build_one_axis(
    H_clones: dict[int, np.ndarray], langs_per_clone: list[str],
    H_native: dict[int, np.ndarray], target_lang: str,
    out_path: Path, fleurs_cfg: str, n_clones: int, n_native: int,
) -> None:
    axis: dict[int, np.ndarray] = {}
    target_c: dict[int, float] = {}
    for L in LAYERS:
        d = compute_lang_axis(H_clones[L], langs_per_clone,
                              source_lang="ja", target_lang=target_lang)
        axis[L] = d
        target_c[L] = float(np.mean(compute_axis_coord(H_native[L], d)))
        # Diagnostic per layer
        c_per_lang = {}
        for ll in sorted(set(langs_per_clone)):
            mask = np.array([l == ll for l in langs_per_clone])
            c_per_lang[ll] = float(H_clones[L][mask].mean(axis=0) @ d)
        print(f"  L{L}: c(ja)={c_per_lang.get('ja', 0):+.2f}  "
              f"c({target_lang})={c_per_lang.get(target_lang, 0):+.2f}  "
              f"c(FLEURS)={target_c[L]:+.2f}",
              flush=True)
    save_axis_artifacts(out_path, axis, target_c, meta={
        "lang_code": target_lang, "source_lang": "ja",
        "n_clones": n_clones, "n_native": n_native,
        "layers": LAYERS, "num_step": NUM_STEP, "fleurs_config": fleurs_cfg,
    })
    print(f"[axes] saved -> {out_path}", flush=True)


# ---------------------------------------------------------------- main

def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[axes] loading OmniVoice ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])

    # 1. JVNV self-clones (extends disentanglement_v2/clones/)
    ensure_clones(w)

    # 2. FLEURS native refs (per-lang dirs under baseline/native_refs/)
    natives = ensure_native_refs(n_per_lang=5)

    # 3. Extract hiddens for ALL clones (single pass — same set serves
    #    as source for every per-language axis build).
    all_clones = sorted(CLONES_DIR.glob("*.wav"))
    langs_per_clone = [p.stem.split("_")[2] for p in all_clones]
    print(f"\n[axes] extracting clone hiddens (n={len(all_clones)}) ...", flush=True)
    H_clones = extract_hiddens(w, all_clones)

    # 4. Per-language axis build
    for lang_code, _, _, fleurs_cfg in LANGS:
        out_path = OUT_DIR / f"{lang_code}_axis.npz"
        if out_path.exists():
            print(f"\n[axes] {lang_code}: artifact exists, skipping ({out_path})", flush=True)
            continue
        ref_paths = natives[lang_code]
        if not ref_paths:
            print(f"[axes] {lang_code}: no native refs found, skip", flush=True)
            continue
        print(f"\n[axes] === build axis for {lang_code} ===", flush=True)
        H_native = extract_hiddens(w, ref_paths)
        build_one_axis(
            H_clones=H_clones, langs_per_clone=langs_per_clone,
            H_native=H_native, target_lang=lang_code,
            out_path=out_path, fleurs_cfg=fleurs_cfg,
            n_clones=len(all_clones), n_native=len(ref_paths),
        )

    print("\n[axes] All done. Artifacts in:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
