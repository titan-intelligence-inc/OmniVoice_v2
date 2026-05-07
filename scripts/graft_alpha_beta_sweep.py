"""α/β grid search for LanguageGrafter on zh.

Sweeps remove_alpha × inject_beta on a small grid, with the
``step_window=(0, 8)`` setting fixed (best from prior coarse run).

Per-cell metrics:

  cer            Whisper CER vs target text  (cap=2.0)
  prefix_match   chars matching target prefix from char 0 (max 14)
  prefix_ratio   = prefix_match / len(target) — 0..1
  hallucination  hyp contains a >=12-char repeated unit (catches the
                 ``娃娃娃娃...`` and ``不不不不...`` failure modes)
  katakana_chars number of katakana chars in the hyp (proxy for JP
                 leakage — should drop with successful zh injection)

We pick **happy** as the focus emotion (the prior run produced the
breakthrough hit there) and run 3 reps per cell.

Sweep dim:  4 × 4 = 16 graft cells + 1 baseline = 17 variants
            × 3 reps = 51 generations on a single emotion.
"""
from __future__ import annotations
import os
import re
import sys
import json
import statistics
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
from ovet.omnivoice.wrapper import OmniVoiceWrapper                            # noqa: E402
from ovet.omnivoice.lang_graft import LanguageGrafter, load_graft_artifacts   # noqa: E402
from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.utils.io import save_wav                                             # noqa: E402


GRAFT_NPZ      = Path("outputs/graft/zh.npz")
JVNV_DIR       = Path("baseline/jvnv_samples")
LAYERS         = [8, 12]
STEP_WINDOW    = (0, 8)
EMOTION        = "happy"        # focus emotion (prior breakthrough rep was here)
ZH_TEXT        = "明天的天气预报是多云转晴。"
ZH_FULL        = "Chinese"
REPS           = 3
SEED           = 0
OUT_DIR        = Path("outputs/graft_alpha_beta_sweep")

# Grid -- excludes (α=1, β=0) which is known catastrophic.
ALPHAS = [0.0, 0.3, 0.6, 1.0]
BETAS  = [0.3, 0.6, 1.0, 1.5]


# ----------------------------------------------------------------------
# Custom metrics
# ----------------------------------------------------------------------

KATAKANA_RE     = re.compile(r"[゠-ヿ]")
HIRAGANA_RE     = re.compile(r"[぀-ゟ]")
HALLU_PATTERN   = re.compile(r"(.{1,4})\1{8,}")     # any 1-4 char unit repeated 9+ times


def _strip_punct(s: str) -> str:
    return re.sub(r"[\s\.,!?。、！？・…]+", "", s).lower()


def prefix_match(hyp: str, target: str) -> int:
    """Longest common prefix length between hyp and target (after
    stripping whitespace/punct, case-insensitive)."""
    a, b = _strip_punct(hyp), _strip_punct(target)
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def hallucination_score(hyp: str) -> bool:
    return bool(HALLU_PATTERN.search(hyp))


def katakana_count(hyp: str) -> int:
    return len(KATAKANA_RE.findall(hyp)) + len(HIRAGANA_RE.findall(hyp))


# ----------------------------------------------------------------------

def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[αβ] loading OmniVoice + ASR ...", flush=True)
    w = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()

    Q_per, T_per, meta = load_graft_artifacts(GRAFT_NPZ)
    Q_subset = {L: Q_per[L] for L in LAYERS}
    T_subset = {L: T_per[L] for L in LAYERS}
    print(f"[αβ] artifacts loaded: layers={list(Q_subset.keys())}", flush=True)

    ref = JVNV_DIR / f"jvnv_F1_{EMOTION}.wav"
    ref_text = w.transcribe(ref, language=None)
    print(f"[αβ] ref={ref.name}", flush=True)
    print(f"[αβ] target text: {ZH_TEXT}", flush=True)
    target_chars = len(_strip_punct(ZH_TEXT))

    variants: list[tuple[str, dict]] = [("baseline", {"kind": "raw"})]
    for a in ALPHAS:
        for b in BETAS:
            if a == 0.0 and b == 0.0:
                continue
            variants.append((f"a{a:.1f}_b{b:.1f}", {"kind": "graft", "a": a, "b": b}))
    print(f"[αβ] variants: {len(variants)}  reps: {REPS}", flush=True)

    rows = []
    for vname, vcfg in variants:
        cers, prefixes, hallus, kanas, hyps = [], [], [], [], []
        for rep in range(REPS):
            torch.manual_seed(SEED + rep)
            if vcfg["kind"] == "raw":
                audio = w.model.generate(
                    text=ZH_TEXT, language=ZH_FULL,
                    ref_audio=str(ref), ref_text=ref_text,
                )[0]
            else:
                with LanguageGrafter(
                    w.model,
                    subspace_per_layer=Q_subset, target_per_layer=T_subset,
                    remove_alpha=vcfg["a"], inject_beta=vcfg["b"],
                    step_window=STEP_WINDOW,
                ):
                    audio = w.model.generate(
                        text=ZH_TEXT, language=ZH_FULL,
                        ref_audio=str(ref), ref_text=ref_text,
                    )[0]
            wav_path = OUT_DIR / f"{vname}_rep{rep}.wav"
            save_wav(wav_path, audio, w.SAMPLING_RATE)
            hyp = asr.transcribe(wav_path, language="zh")
            cer = asr.content_error(wav_path, ZH_TEXT, language="zh")
            pm  = prefix_match(hyp, ZH_TEXT)
            hl  = hallucination_score(hyp)
            kk  = katakana_count(hyp)
            cers.append(cer); prefixes.append(pm); hallus.append(hl)
            kanas.append(kk); hyps.append(hyp)

        row = {
            "variant": vname,
            "cer_med":      statistics.median(cers),
            "cer_min":      min(cers),
            "prefix_max":   max(prefixes),
            "prefix_mean":  statistics.fmean(prefixes),
            "hallu_rate":   sum(hallus) / len(hallus),
            "kana_med":     statistics.median(kanas),
            "hyps":         hyps,
        }
        rows.append(row)
        flag = "🎯" if row["prefix_max"] >= 2 else (
            "⚠️" if row["hallu_rate"] >= 0.5 else "·")
        print(
            f" {flag} {vname:<14} "
            f"cer_med={row['cer_med']:.2f}  "
            f"prefix_max={row['prefix_max']:>2}/{target_chars}  "
            f"prefix_mean={row['prefix_mean']:.1f}  "
            f"hallu={row['hallu_rate']:.0%}  "
            f"kana_med={row['kana_med']:>3}",
            flush=True,
        )

    # ----- summary table -----
    print("\n=== α/β grid (sorted by prefix_max desc, then hallu_rate asc) ===", flush=True)
    rows_sorted = sorted(rows, key=lambda r: (-r["prefix_max"], r["hallu_rate"], r["cer_med"]))
    print(f"{'variant':<16} {'pfx_max':>7} {'pfx_mean':>9} {'hallu':>6} "
          f"{'kana_med':>8} {'cer_med':>7} {'cer_min':>7}")
    for r in rows_sorted:
        print(f"{r['variant']:<16} "
              f"{r['prefix_max']:>7} {r['prefix_mean']:>9.1f} "
              f"{r['hallu_rate']:>5.0%} {r['kana_med']:>8} "
              f"{r['cer_med']:>7.2f} {r['cer_min']:>7.2f}")

    print("\n=== top-3 transcripts ===", flush=True)
    for r in rows_sorted[:3]:
        print(f"--- {r['variant']} (prefix_max={r['prefix_max']}, hallu={r['hallu_rate']:.0%}) ---")
        for h in r["hyps"]:
            print(f"  {h[:80]}")

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"emotion": EMOTION, "step_window": list(STEP_WINDOW),
                   "layers": LAYERS, "reps": REPS,
                   "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\n[αβ] saved -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
