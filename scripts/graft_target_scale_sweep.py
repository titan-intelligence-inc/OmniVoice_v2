"""target_c-scale sweep for the 1D LanguageAxisGrafter.

Holds α=1.0, β=1.0 (pure substitution — robust point from previous
sweep) and varies the target_c value:

    target_c_scaled = scale_factor · c_FLEURS

scale_factor candidates:
  0.5x  : pull only halfway to FLEURS    (gentler than current)
  1.0x  : current (= c_FLEURS)            -- the existing axis_a1.0_b1.0
  1.5x  : push past FLEURS into stronger-zh territory
  2.0x  : push to ~2x FLEURS coord
  3.0x  : aggressive — beyond JVNV-zh-clone (which is ~1.2x FLEURS)

Optionally also try absolute targets:
  abs_zhclone : target_c = mean coord of JVNV zh-clones
                (a different anchor — "make hidden look like a JVNV
                speaker doing zh", not "look like a native zh speaker")

Test on 3 emotions × 3 reps = 9 reps per cell. Compare against the
previously-validated axis_a1.0_b1.0 (= scale=1.0) and baseline.
"""
from __future__ import annotations
import os, re, sys, json, statistics
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper                            # noqa: E402
from ovet.omnivoice.lang_graft import (                                        # noqa: E402
    LanguageAxisGrafter, load_axis_artifacts, compute_axis_coord,
)
from ovet.omnivoice.steering import extract_layer_vectors                      # noqa: E402
from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402


AXIS_NPZ      = Path("outputs/graft/zh_axis.npz")
HIDDEN_CACHE  = Path("outputs/graft/hidden_cache.npz")
JVNV_DIR      = Path("baseline/jvnv_samples")
EMOTIONS      = ["sad", "happy", "anger"]
LAYERS        = [8, 12]
STEP_WINDOW   = (0, 8)
ZH_TEXT       = "明天的天气预报是多云转晴。"
ZH_FULL       = "Chinese"
REPS          = 3
SEED          = 0
OUT_DIR       = Path("outputs/graft_target_scale_sweep")
SCALES        = [0.5, 1.0, 1.5, 2.0, 3.0]


KATAKANA_RE   = re.compile(r"[゠-ヿ]")
HIRAGANA_RE   = re.compile(r"[぀-ゟ]")
HALLU_PATTERN = re.compile(r"(.{1,4})\1{8,}")


def _strip(s):  return re.sub(r"[\s\.,!?。、！？・…]+", "", s).lower()
def prefix_match(h, t):
    a, b = _strip(h), _strip(t); n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]: return i
    return n
def hallu(h): return bool(HALLU_PATTERN.search(h))
def kana(h):  return len(KATAKANA_RE.findall(h)) + len(HIRAGANA_RE.findall(h))


def _load_zhclone_coords(axis: dict[int, np.ndarray]) -> dict[int, float]:
    """Compute mean coord of JVNV zh-clones along the axis (cached)."""
    if not HIDDEN_CACHE.exists():
        return {}
    d = np.load(HIDDEN_CACHE, allow_pickle=True)
    # Cache contains 48 clones + 5 native, in alphabetical order.
    # Re-derive language labels from filename metadata.
    clones_dir = Path("outputs/disentanglement_v2/clones")
    files = sorted(clones_dir.glob("*.wav"))
    langs = [p.stem.split("_")[2] for p in files]
    is_zh = np.array([l == "zh" for l in langs])
    out = {}
    for L in axis:
        H = np.asarray(d[f"L{L}"])[: len(files)]    # clones only
        zh_mean = H[is_zh].mean(axis=0)
        out[L] = float(zh_mean @ axis[L])
    return out


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[scale] loading OmniVoice + ASR ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()

    axis, target_c, meta = load_axis_artifacts(AXIS_NPZ)
    axis_subset = {L: axis[L] for L in LAYERS}
    base_target = {L: target_c[L] for L in LAYERS}
    print(f"[scale] base target_c (= c_FLEURS): "
          f"{ {L: f'{base_target[L]:+.2f}' for L in LAYERS} }", flush=True)

    # JVNV zh-clone coords as an alternative anchor.
    zhclone_c = _load_zhclone_coords(axis_subset)
    if zhclone_c:
        print(f"[scale] c(zh-clone): "
              f"{ {L: f'{zhclone_c[L]:+.2f}' for L in LAYERS} }", flush=True)

    # Variants
    variants: list[tuple[str, dict]] = [("baseline", {"kind": "raw"})]
    for s in SCALES:
        target = {L: base_target[L] * s for L in LAYERS}
        variants.append((f"scale_{s:g}x", {"kind": "axis", "target_c": target}))
    if zhclone_c:
        variants.append(("anchor_zhclone", {"kind": "axis", "target_c": zhclone_c}))
    print(f"[scale] {len(variants)} variants × {len(EMOTIONS)} emo × "
          f"{REPS} reps = {len(variants)*len(EMOTIONS)*REPS} generations", flush=True)

    rows = []
    for emo in EMOTIONS:
        ref = JVNV_DIR / f"jvnv_F1_{emo}.wav"
        ref_text = w.transcribe(ref, language=None)
        print(f"\n[scale] === emo={emo} ===", flush=True)
        for vname, vcfg in variants:
            cers, prefixes, hallus, kanas, hyps = [], [], [], [], []
            for rep in range(REPS):
                torch.manual_seed(SEED + rep)
                kwargs = dict(text=ZH_TEXT, language=ZH_FULL,
                              ref_audio=str(ref), ref_text=ref_text)
                if vcfg["kind"] == "raw":
                    audio = w.model.generate(**kwargs)[0]
                else:
                    with LanguageAxisGrafter(
                        w.model,
                        axis_per_layer=axis_subset,
                        target_c_per_layer=vcfg["target_c"],
                        remove_alpha=1.0, inject_beta=1.0,
                        step_window=STEP_WINDOW,
                    ):
                        audio = w.model.generate(**kwargs)[0]
                wav_path = OUT_DIR / f"{emo}_{vname}_rep{rep}.wav"
                save_wav(wav_path, audio, w.SAMPLING_RATE)
                hyp = asr.transcribe(wav_path, language="zh")
                cer = asr.content_error(wav_path, ZH_TEXT, language="zh")
                cers.append(cer); prefixes.append(prefix_match(hyp, ZH_TEXT))
                hallus.append(hallu(hyp)); kanas.append(kana(hyp))
                hyps.append(hyp)
            rows.append({
                "emotion": emo, "variant": vname,
                "cer_med": statistics.median(cers),
                "prefix_max": max(prefixes),
                "prefix_mean": statistics.fmean(prefixes),
                "hallu_rate": sum(hallus) / len(hallus),
                "kana_med": statistics.median(kanas),
                "hyps": hyps,
            })
            r = rows[-1]
            flag = "🎯" if r["prefix_max"] >= 4 else (
                "·" if r["prefix_max"] >= 2 else (
                    "⚠️" if r["hallu_rate"] >= 0.5 else "—"))
            print(f"  {flag} {vname:<18} pfx={r['prefix_max']}/{r['prefix_mean']:.1f}  "
                  f"hallu={r['hallu_rate']:.0%}  kana={r['kana_med']}  "
                  f"cer_med={r['cer_med']:.2f}", flush=True)

    # Aggregate
    by_v: dict[str, dict] = {}
    for r in rows:
        d = by_v.setdefault(r["variant"], {"pfx": [], "kana": [], "cer": [], "hallu": []})
        d["pfx"].append(r["prefix_mean"])
        d["kana"].append(r["kana_med"])
        d["cer"].append(r["cer_med"])
        d["hallu"].append(r["hallu_rate"])

    print("\n=== aggregate (mean over 3 emotions × 3 reps each) ===", flush=True)
    print(f"{'variant':<20} {'pfx_mean':>9} {'kana_med':>9} {'cer_med':>9} {'hallu':>6}")
    rows_agg = []
    for v in by_v:
        rows_agg.append({
            "variant": v,
            "pfx_mean":  statistics.fmean(by_v[v]["pfx"]),
            "kana_med":  statistics.fmean(by_v[v]["kana"]),
            "cer_med":   statistics.fmean(by_v[v]["cer"]),
            "hallu":     statistics.fmean(by_v[v]["hallu"]),
        })
    for r in sorted(rows_agg, key=lambda x: (-x["pfx_mean"], x["kana_med"], x["cer_med"])):
        print(f"{r['variant']:<20} "
              f"{r['pfx_mean']:>9.2f} {r['kana_med']:>9.1f} "
              f"{r['cer_med']:>9.2f} {r['hallu']:>5.0%}")

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "config": {
            "scales": SCALES, "emotions": EMOTIONS, "reps": REPS,
            "layers": LAYERS, "step_window": list(STEP_WINDOW),
            "alpha": 1.0, "beta": 1.0,
        }}, f, indent=2, ensure_ascii=False)
    print(f"\n[scale] saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
