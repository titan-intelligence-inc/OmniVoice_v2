"""Layer-specific α/β tuning for the axis grafter.

Holds step_window=(0,8), pm=False (current optimum) and varies which
layer(s) get grafted and at what strength.

Variants:
  baseline        — no graft
  L8L12_a1b1      — current optimum (= scale_1x = w0_8_pm0)
  L8only_a1b1     — only L8 grafted (L12 untouched)
  L12only_a1b1    — only L12 grafted (L8 untouched)
  L8s_L12n        — L8 strong (α=β=1.5), L12 normal (α=β=1.0)
  L8n_L12s        — L8 normal, L12 strong
  L8only_a15b15   — L8 only, doubled
  L12only_a15b15  — L12 only, doubled
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
STEP_WINDOW   = (0, 8)
ZH_TEXT       = "明天的天气预报是多云转晴。"
ZH_FULL       = "Chinese"
REPS          = 3
SEED          = 0
OUT_DIR       = Path("outputs/graft_layer_tuning")


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


class _PerLayerAxisGrafter:
    """Wraps LanguageAxisGrafter to allow per-layer α/β.

    Stack two single-layer LanguageAxisGrafter contexts.
    """
    def __init__(self, model, full_axis, full_target, per_layer):
        """per_layer = {layer_id: {'a': float, 'b': float}}"""
        self.children = []
        for L, params in per_layer.items():
            if params['a'] == 0 and params['b'] == 0:
                continue
            self.children.append(LanguageAxisGrafter(
                model,
                axis_per_layer={L: full_axis[L]},
                target_c_per_layer={L: full_target[L]},
                remove_alpha=params['a'], inject_beta=params['b'],
                step_window=STEP_WINDOW,
            ))

    def __enter__(self):
        for c in self.children:
            c.__enter__()
        return self

    def __exit__(self, *exc):
        for c in reversed(self.children):
            c.__exit__(*exc)


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[ltune] loading OmniVoice + ASR ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()

    axis, target_c, _ = load_axis_artifacts(AXIS_NPZ)

    # Variants: each has per-layer {a, b}.
    variants: list[tuple[str, dict | None]] = [
        ("baseline",        None),
        ("L8L12_a1b1",      {8: {"a": 1.0, "b": 1.0}, 12: {"a": 1.0, "b": 1.0}}),
        ("L8only_a1b1",     {8: {"a": 1.0, "b": 1.0}, 12: {"a": 0.0, "b": 0.0}}),
        ("L12only_a1b1",    {8: {"a": 0.0, "b": 0.0}, 12: {"a": 1.0, "b": 1.0}}),
        ("L8s_L12n",        {8: {"a": 1.5, "b": 1.5}, 12: {"a": 1.0, "b": 1.0}}),
        ("L8n_L12s",        {8: {"a": 1.0, "b": 1.0}, 12: {"a": 1.5, "b": 1.5}}),
        ("L8only_a15b15",   {8: {"a": 1.5, "b": 1.5}, 12: {"a": 0.0, "b": 0.0}}),
        ("L12only_a15b15",  {8: {"a": 0.0, "b": 0.0}, 12: {"a": 1.5, "b": 1.5}}),
    ]
    print(f"[ltune] {len(variants)} variants × {len(EMOTIONS)} × "
          f"{REPS} reps = {len(variants)*len(EMOTIONS)*REPS} gens", flush=True)

    rows = []
    for emo in EMOTIONS:
        ref = JVNV_DIR / f"jvnv_F1_{emo}.wav"
        ref_text = w.transcribe(ref, language=None)
        print(f"\n[ltune] === emo={emo} ===", flush=True)
        for vname, per_layer in variants:
            cers, prefixes, hallus, kanas, hyps = [], [], [], [], []
            for rep in range(REPS):
                torch.manual_seed(SEED + rep)
                kwargs = dict(text=ZH_TEXT, language=ZH_FULL,
                              ref_audio=str(ref), ref_text=ref_text)
                if per_layer is None:
                    audio = w.model.generate(**kwargs)[0]
                else:
                    with _PerLayerAxisGrafter(w.model, axis, target_c, per_layer):
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

    by_v: dict[str, dict] = {}
    for r in rows:
        d = by_v.setdefault(r["variant"], {"pfx": [], "kana": [], "cer": [], "hallu": []})
        d["pfx"].append(r["prefix_mean"])
        d["kana"].append(r["kana_med"])
        d["cer"].append(r["cer_med"])
        d["hallu"].append(r["hallu_rate"])

    print("\n=== aggregate ===", flush=True)
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
            "emotions": EMOTIONS, "reps": REPS, "step_window": list(STEP_WINDOW),
        }}, f, indent=2, ensure_ascii=False)
    print(f"\n[ltune] saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
