"""Disentanglement validation for the language-grafting plan.

Question: at which layers can we cleanly extract a "language component"
from a hidden state, then substitute it from a different sample, while
preserving speaker and emotion?

Expanded over Phase 4a (which only tested JP vs EN binary):
  * languages: ja, en, zh, ko  (4-way)
  * speakers:  F1, F2, M1, M2
  * emotions:  anger, sad, fear

Total: 48 self-clones (reuses phase4a_spike's 24 ja/en, generates 24
zh/ko on top). Per-layer hidden vectors are extracted with
``extract_layer_vectors`` (num_step=4).

Diagnostics, per layer:

  A. **Probe accuracy (LOO multi-class)**
       - lang_acc, spk_acc, emo_acc  -- baseline separability
  B. **Subspace decomposition**
       - Train multi-class logistic regression for each axis. Use the
         classifier weight matrix as the axis subspace.
       - Compute orthogonality: cos angle between top components of
         (lang ⊥ spk) and (lang ⊥ emo).
  C. **Removal test**: project out lang subspace, then re-probe
       - lang_acc_post  (should drop near chance: 0.25 for 4-way)
       - spk_acc_post   (should stay high if disentangled)
       - emo_acc_post   (should stay high if disentangled)
  D. **Substitution test (the critical one)**
       For each sample ``i`` with (spk_i, lang_i):
         pick a *donor* sample ``j`` with same lang_i but different spk_j
         h_subst = h_i - lang_proj(h_i) + lang_proj(h_j)
         Predict spk(h_subst). Does it still predict spk_i?
       Reports spk_subst_acc — high means graft preserves speaker.

Run:
    venv/bin/python scripts/disentanglement_validation.py
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper          # noqa: E402
from ovet.omnivoice.steering import extract_layer_vectors    # noqa: E402
from ovet.utils.io import save_wav                           # noqa: E402


SPEAKERS  = ["F1", "F2", "M1", "M2"]
EMOTIONS  = ["anger", "sad", "fear"]
# Reuse the texts from configs/multilang.yaml. We only need a single
# neutral text per language for hidden capture.
LANGUAGES: list[tuple[str, str, str]] = [
    ("ja", "Japanese", "今日の天気予報は曇り時々晴れです。"),
    ("en", "English",  "The weather forecast for tomorrow is partly cloudy."),
    ("zh", "Chinese",  "明天的天气预报是多云转晴。"),
    ("ko", "Korean",   "내일 일기 예보는 가끔 구름이 끼겠습니다."),
]
LAYERS = [4, 8, 12, 16, 20, 24]
REF_DIR_PRIMARY = Path("baseline/jvnv_samples_multi")
EXISTING_CLONES = Path("outputs/phase4a_spike/clones")     # ja/en already here
OUT_DIR  = Path("outputs/disentanglement_v2")
NUM_STEP = 4
SEED     = 0


# ----------------------------------------------------------------------
# Stage 1: ensure 48 wav clones exist on disk
# ----------------------------------------------------------------------

def ensure_clones(w: OmniVoiceWrapper) -> list[dict]:
    """Return list of {speaker, emotion, lang_code, lang_full, wav, ref_text}.
    Generate any missing clones; reuse phase4a_spike for ja/en where available.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clones_dir = OUT_DIR / "clones"
    clones_dir.mkdir(exist_ok=True)
    samples = []
    for sp in SPEAKERS:
        for emo in EMOTIONS:
            ref = REF_DIR_PRIMARY / f"jvnv_{sp}_{emo}.wav"
            if not ref.exists():
                print(f"[disent] missing ref {ref}, skipping {sp}/{emo}", flush=True)
                continue
            ref_text = w.transcribe(ref, language=None)
            for code, lang_full, target_text in LANGUAGES:
                # ja/en: reuse from phase4a_spike if present
                phase4a_path = EXISTING_CLONES / f"{sp}_{emo}_{code}.wav"
                wav_path = clones_dir / f"{sp}_{emo}_{code}.wav"
                if phase4a_path.exists() and not wav_path.exists():
                    # Symlink to avoid duplication
                    wav_path.symlink_to(phase4a_path.resolve())
                if not wav_path.exists():
                    print(f"[disent] generating {sp}_{emo}_{code} ({lang_full}) ...",
                          flush=True)
                    torch.manual_seed(SEED)
                    audio = w.model.generate(
                        text=target_text, language=lang_full,
                        ref_audio=str(ref), ref_text=ref_text, num_step=8,
                    )[0]
                    save_wav(wav_path, audio, w.SAMPLING_RATE)
                samples.append({
                    "speaker": sp, "emotion": emo,
                    "lang_code": code, "lang_full": lang_full,
                    "wav": str(wav_path), "ref_text": ref_text,
                })
    print(f"[disent] total clones: {len(samples)}", flush=True)
    return samples


# ----------------------------------------------------------------------
# Stage 2: extract hiddens
# ----------------------------------------------------------------------

def extract_all_hiddens(w: OmniVoiceWrapper, samples: list[dict]) -> dict:
    """Returns dict[layer_id] -> np.ndarray [N, D]."""
    H_per_layer: dict[int, list[np.ndarray]] = {l: [] for l in LAYERS}
    for s in samples:
        vecs = extract_layer_vectors(
            w, Path(s["wav"]), LAYERS, ref_text=None,
            num_step=NUM_STEP, seed=SEED,
        )
        for l in LAYERS:
            H_per_layer[l].append(vecs[l])
    return {l: np.stack(H_per_layer[l]).astype(np.float32) for l in LAYERS}


# ----------------------------------------------------------------------
# Stage 3: per-layer diagnostics
# ----------------------------------------------------------------------

def loo_probe_accuracy(X: np.ndarray, y: list, *, C: float = 1.0) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_score
    # Suppress convergence warnings — they don't affect the metric.
    import warnings
    warnings.filterwarnings("ignore")
    clf = LogisticRegression(C=C, max_iter=2000, random_state=0)
    return float(np.mean(cross_val_score(clf, X, y, cv=LeaveOneOut())))


def lang_subspace_via_class_means(
    H: np.ndarray, langs: list[str]
) -> np.ndarray:
    """Build a low-rank language subspace from class means.

    Take the per-language mean hidden vectors as anchors, center them
    (subtract overall mean), and use them as a basis. Returns
    orthonormal columns spanning the subspace.
    """
    unique_langs = sorted(set(langs))
    means = np.stack([H[np.array(langs) == l].mean(axis=0) for l in unique_langs])
    overall = H.mean(axis=0, keepdims=True)
    M = means - overall                       # [K, D]
    # Orthonormalize via QR. For K << D, this gives at most K-1
    # meaningful orthogonal directions (mean of M is ~0 after centering
    # only if we centered by means' own mean, which we didn't fully —
    # acceptable, the rank-K span still contains the language axis).
    Q, _ = np.linalg.qr(M.T)                  # [D, K]
    return Q.astype(np.float32)


def project_out_subspace(H: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Remove the column-space of Q from each row of H.
    H: [N, D], Q: [D, K] orthonormal.  Returns H_perp: [N, D].
    """
    coef = H @ Q                              # [N, K]
    return H - coef @ Q.T


def project_onto_subspace(H: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Project each row of H onto column-space of Q.
    Returns shape [N, D].
    """
    coef = H @ Q                              # [N, K]
    return coef @ Q.T


def variance_captured(H: np.ndarray, Q: np.ndarray) -> float:
    """% of total variance captured by the subspace."""
    Hc = H - H.mean(axis=0, keepdims=True)
    total = (Hc * Hc).sum()
    proj  = project_onto_subspace(Hc, Q)
    cap   = (proj * proj).sum()
    return float(cap / (total + 1e-9))


def substitution_test(
    H: np.ndarray, langs: list[str], spks: list[str], emos: list[str],
    Q_lang: np.ndarray, rng: np.random.Generator,
) -> dict:
    """For each sample, swap its lang component with a same-lang
    different-speaker sample's lang component. Then probe spk/emo.

    Returns dict with subst spk_acc / emo_acc and pre-substitution
    baseline for comparison.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_score

    N = H.shape[0]
    H_subst = H.copy()
    swapped = 0
    for i in range(N):
        candidates = [j for j in range(N)
                      if j != i and langs[j] == langs[i] and spks[j] != spks[i]]
        if not candidates:
            continue
        j = int(rng.choice(candidates))
        h_lang_i = project_onto_subspace(H[i:i+1], Q_lang)[0]
        h_lang_j = project_onto_subspace(H[j:j+1], Q_lang)[0]
        H_subst[i] = H[i] - h_lang_i + h_lang_j
        swapped += 1

    spk_acc_subst = loo_probe_accuracy(H_subst, spks)
    emo_acc_subst = loo_probe_accuracy(H_subst, emos)
    lang_acc_subst = loo_probe_accuracy(H_subst, langs)
    return {
        "n_swapped": swapped,
        "spk_acc_subst":  spk_acc_subst,
        "emo_acc_subst":  emo_acc_subst,
        "lang_acc_subst": lang_acc_subst,
    }


def cos_same_lang_diff_spk(H: np.ndarray, langs: list[str], spks: list[str],
                            Q_lang: np.ndarray) -> float:
    """Cosine similarity of lang projections across different speakers
    of the same language. High means lang_proj is speaker-invariant."""
    proj = project_onto_subspace(H, Q_lang)
    sims = []
    for i in range(len(H)):
        for j in range(i+1, len(H)):
            if langs[i] == langs[j] and spks[i] != spks[j]:
                a, b = proj[i], proj[j]
                den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
                sims.append(float(np.dot(a, b) / den))
    return float(np.mean(sims)) if sims else float("nan")


def cos_same_spk_diff_lang(H: np.ndarray, langs: list[str], spks: list[str],
                            Q_lang: np.ndarray) -> float:
    """Cosine similarity of lang projections within same speaker
    different languages. Low means lang_proj is language-discriminating."""
    proj = project_onto_subspace(H, Q_lang)
    sims = []
    for i in range(len(H)):
        for j in range(i+1, len(H)):
            if spks[i] == spks[j] and langs[i] != langs[j]:
                a, b = proj[i], proj[j]
                den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
                sims.append(float(np.dot(a, b) / den))
    return float(np.mean(sims)) if sims else float("nan")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[disent] loading OmniVoice ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])

    samples = ensure_clones(w)
    spks  = [s["speaker"]   for s in samples]
    emos  = [s["emotion"]   for s in samples]
    langs = [s["lang_code"] for s in samples]
    print(f"[disent] languages: {sorted(set(langs))}", flush=True)
    print(f"[disent] speakers:  {sorted(set(spks))}", flush=True)
    print(f"[disent] emotions:  {sorted(set(emos))}", flush=True)

    print("[disent] extracting per-layer hiddens ...", flush=True)
    Hmat = extract_all_hiddens(w, samples)

    rng = np.random.default_rng(SEED)
    rows = []
    print("\n=== per-layer diagnostics ===", flush=True)
    print(f"{'L':<4} {'lang':<5} {'spk':<5} {'emo':<5}  | {'lang/spk-rem':<13} {'spk/spk-rem':<13} {'emo/spk-rem':<13}  | {'cos(SL,DS)':<10} {'cos(SS,DL)':<10}  | {'subst:spk':<11} {'subst:emo':<11} {'subst:lang':<11}", flush=True)
    for l in LAYERS:
        H = Hmat[l]
        lang_acc = loo_probe_accuracy(H, langs)
        spk_acc  = loo_probe_accuracy(H, spks)
        emo_acc  = loo_probe_accuracy(H, emos)

        Q_lang = lang_subspace_via_class_means(H, langs)
        var_cap = variance_captured(H, Q_lang)

        H_no_lang = project_out_subspace(H, Q_lang)
        lang_acc_post = loo_probe_accuracy(H_no_lang, langs)
        spk_acc_post  = loo_probe_accuracy(H_no_lang, spks)
        emo_acc_post  = loo_probe_accuracy(H_no_lang, emos)

        cos_SL_DS = cos_same_lang_diff_spk(H, langs, spks, Q_lang)
        cos_SS_DL = cos_same_spk_diff_lang(H, langs, spks, Q_lang)

        sub = substitution_test(H, langs, spks, emos, Q_lang, rng)

        row = {
            "layer": l, "var_cap": var_cap,
            "lang_acc": lang_acc, "spk_acc": spk_acc, "emo_acc": emo_acc,
            "lang_acc_post": lang_acc_post, "spk_acc_post": spk_acc_post,
            "emo_acc_post": emo_acc_post,
            "cos_same_lang_diff_spk": cos_SL_DS,
            "cos_same_spk_diff_lang": cos_SS_DL,
            **sub,
        }
        rows.append(row)
        print(
            f"{l:<4} "
            f"{lang_acc:.2f}  {spk_acc:.2f}  {emo_acc:.2f}  | "
            f"{lang_acc:.2f}→{lang_acc_post:.2f}    "
            f"{spk_acc:.2f}→{spk_acc_post:.2f}    "
            f"{emo_acc:.2f}→{emo_acc_post:.2f}    | "
            f"{cos_SL_DS:+.3f}    {cos_SS_DL:+.3f}    | "
            f"{sub['spk_acc_subst']:.2f}        "
            f"{sub['emo_acc_subst']:.2f}        "
            f"{sub['lang_acc_subst']:.2f}",
            flush=True,
        )

    out_json = OUT_DIR / "result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "samples": samples,
                   "config": {"layers": LAYERS, "num_step": NUM_STEP, "seed": SEED}},
                  f, indent=2, ensure_ascii=False)
    print(f"\n[disent] saved -> {out_json}", flush=True)

    # Quick verdict
    print("\n=== verdict per layer ===")
    for r in rows:
        ok_remove = (r["lang_acc"] - r["lang_acc_post"] >= 0.25 and
                     r["spk_acc"] - r["spk_acc_post"] <= 0.15)
        ok_subst  = (r["spk_acc_subst"] >= 0.6 and
                     r["lang_acc_subst"] >= 0.6)
        v = "✅ GRAFT-VIABLE" if (ok_remove and ok_subst) else (
            "△ removal-ok-graft-poor" if ok_remove else "❌")
        print(f"  L{r['layer']:<3}  {v}")


if __name__ == "__main__":
    main()
