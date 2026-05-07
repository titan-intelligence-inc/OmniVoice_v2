"""step_window × position_mask sweep on top of α=β=1, scale=1x.

Holds the previously-validated robust point (axis_a1.0_b1.0,
target_c = c_FLEURS) and varies:

  step_window   ∈ {(0,4), (0,8), (0,16), (0,32)} or None=all-steps
  position_mask ∈ {False, True}

Hypothesis: longer windows let the graft keep pushing through the
diffusion → may recover content past the prefix. position_mask
restricts the perturbation to audio-token positions, hopefully
preserving text-prompt content alignment.

Test on 3 emotions × 3 reps = 9 reps per cell. Compare against
scale_1x baseline (= step_window=(0,8), position_mask=False).
"""
from __future__ import annotations
import os, re, sys, json, statistics
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper                            # noqa: E402
from ovet.omnivoice.lang_graft import (                                        # noqa: E402
    LanguageAxisGrafter, load_axis_artifacts,
)
from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402


AXIS_NPZ      = Path("outputs/graft/zh_axis.npz")
JVNV_DIR      = Path("baseline/jvnv_samples")
EMOTIONS      = ["sad", "happy", "anger"]
LAYERS        = [8, 12]
ZH_TEXT       = "明天的天气预报是多云转晴。"
ZH_FULL       = "Chinese"
REPS          = 3
SEED          = 0
OUT_DIR       = Path("outputs/graft_window_pmask_sweep")

WINDOWS       = [(0, 4), (0, 8), (0, 16), (0, 32), None]
PMASKS        = [False, True]


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


def _window_label(w):
    if w is None: return "all"
    return f"{w[0]}_{w[1]}"


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[wp] loading OmniVoice + ASR ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()

    axis, target_c, _ = load_axis_artifacts(AXIS_NPZ)
    axis_subset   = {L: axis[L] for L in LAYERS}
    target_subset = {L: target_c[L] for L in LAYERS}

    variants: list[tuple[str, dict]] = [("baseline", {"kind": "raw"})]
    for win in WINDOWS:
        for pm in PMASKS:
            name = f"w{_window_label(win)}_pm{int(pm)}"
            variants.append((name, {"kind": "axis", "win": win, "pm": pm}))
    print(f"[wp] {len(variants)} variants × {len(EMOTIONS)} emo × "
          f"{REPS} reps = {len(variants)*len(EMOTIONS)*REPS} generations", flush=True)

    rows = []
    for emo in EMOTIONS:
        ref = JVNV_DIR / f"jvnv_F1_{emo}.wav"
        ref_text = w.transcribe(ref, language=None)
        print(f"\n[wp] === emo={emo} ===", flush=True)
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
                        target_c_per_layer=target_subset,
                        remove_alpha=1.0, inject_beta=1.0,
                        step_window=vcfg["win"],
                        position_mask=vcfg["pm"],
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
            print(f"  {flag} {vname:<14} pfx={r['prefix_max']}/{r['prefix_mean']:.1f}  "
                  f"hallu={r['hallu_rate']:.0%}  kana={r['kana_med']}  "
                  f"cer_med={r['cer_med']:.2f}", flush=True)

    by_v: dict[str, dict] = {}
    for r in rows:
        d = by_v.setdefault(r["variant"], {"pfx": [], "kana": [], "cer": [], "hallu": []})
        d["pfx"].append(r["prefix_mean"])
        d["kana"].append(r["kana_med"])
        d["cer"].append(r["cer_med"])
        d["hallu"].append(r["hallu_rate"])

    print("\n=== aggregate (3 emotions × 3 reps each) ===", flush=True)
    print(f"{'variant':<16} {'pfx_mean':>9} {'kana_med':>9} {'cer_med':>9} {'hallu':>6}")
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
        print(f"{r['variant']:<16} "
              f"{r['pfx_mean']:>9.2f} {r['kana_med']:>9.1f} "
              f"{r['cer_med']:>9.2f} {r['hallu']:>5.0%}")

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "config": {
            "windows": [list(w) if w else None for w in WINDOWS],
            "pmasks": PMASKS, "emotions": EMOTIONS, "reps": REPS,
            "layers": LAYERS, "alpha": 1.0, "beta": 1.0,
        }}, f, indent=2, ensure_ascii=False)
    print(f"\n[wp] saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
