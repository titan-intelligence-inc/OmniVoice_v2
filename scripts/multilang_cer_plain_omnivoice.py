"""Plain OmniVoice CER across the 9 non-zh languages.

Runs ONLY stage 1 of the pipeline (OmniVoice batched generate) — no
SeedVC, no F0 transfer. Measures the raw cross-lingual TTS capability
of OmniVoice when given a native reference + the target text.

The result is the "lower bound" against which the post-VC stack (zh-
tuned SeedVC + F0 transfer toward F1 ja anger) must be compared. If a
language has CER≈0 at this stage, all the damage is downstream; if
it's already high, OmniVoice itself is the bottleneck.

Reads target text + ref pools from the same config as
``multilang_cer.py`` so the two reports are directly comparable.
"""
from __future__ import annotations
import os
import sys
import json
import statistics
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "upstream/OmniVoice")

from ovet.analyzers.asr_analyzer import ASRAnalyzer, _normalize_text             # noqa: E402
from ovet.omnivoice.wrapper import OmniVoiceWrapper                              # noqa: E402
from ovet.utils.io import save_wav                                               # noqa: E402

# Reuse the per-language config defined in multilang_cer.py
from multilang_cer import LANGS, _resolve_refs, N, SR_OMNI                        # noqa: E402


OUT_DIR = Path("outputs/multilang_cer_plain_omnivoice")


def _cer(ref: str, hyp: str, *, fold_zh: bool = False,
         cap: float | None = 2.0) -> float:
    from ovet.analyzers.asr_analyzer import _detect_hallucination
    rn = _normalize_text(ref, zh=fold_zh)
    hn = _normalize_text(hyp, zh=fold_zh)
    if not rn:
        return 0.0 if not hn else 1.0
    if _detect_hallucination(hn) and cap is not None:
        return cap
    m, n = len(rn), len(hn)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if rn[i - 1] == hn[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    raw = dp[n] / m
    return min(raw, cap) if cap is not None else raw


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    os.environ.setdefault("MODELSCOPE_CACHE", "/workspace/hf_cache/modelscope")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[plain] loading OmniVoice + ASR ...", flush=True)
    w   = OmniVoiceWrapper(hf_home=os.environ["HF_HOME"])
    asr = ASRAnalyzer()
    asr._ensure_loaded()

    # Optional: load production result.json for side-by-side display.
    prod_path = Path("outputs/multilang_cer/result.json")
    prod = {}
    if prod_path.exists():
        prod = {r["lang"]: r["cer_med"]
                for r in json.load(open(prod_path))["rows"]}

    print(f"\n{'lang':<4} {'cer_med':>8} {'q1':>8} {'q3':>8} "
          f"{'best/N':>8} {'mean':>8}    {'prod_med':>9}  {'Δ':>7}",
          flush=True)
    print("-" * 75, flush=True)

    rows = []
    for lang in LANGS:
        lid = lang["id"]
        refs = _resolve_refs(lang)
        out_lang = OUT_DIR / lid
        out_lang.mkdir(parents=True, exist_ok=True)

        ref_texts = [w.transcribe(p, language=None) for p in refs]

        # Batched OmniVoice generate.
        outs = w.model.generate(
            text=[lang["text"]] * N,
            language=[lang["ov"]] * N,
            ref_audio=[str(p) for p in refs],
            ref_text=ref_texts,
        )
        out_paths = []
        for i, a in enumerate(outs):
            p = out_lang / f"plain_{i}.wav"
            save_wav(p, a, SR_OMNI)
            out_paths.append(p)

        hyps = asr.transcribe_batch(out_paths, language=lang["wh"], batch_size=N)
        cers = [_cer(lang["text"], h, fold_zh=lang["fold_zh"]) for h in hyps]

        med = statistics.median(cers)
        q1  = statistics.quantiles(cers, n=4)[0]
        q3  = statistics.quantiles(cers, n=4)[2]
        bo  = min(cers)
        mean = statistics.fmean(cers)

        prod_med = prod.get(lid)
        if prod_med is not None:
            delta = prod_med - med
            tail = f"   {prod_med:9.3f}  {delta:+7.3f}"
        else:
            tail = f"   {'-':>9}  {'-':>7}"

        rows.append({
            "lang": lid, "ov_lang": lang["ov"], "text": lang["text"],
            "cers": cers, "hyps": hyps,
            "cer_med": med, "cer_q1": q1, "cer_q3": q3,
            "best": bo, "mean": mean,
            "production_cer_med": prod_med,
        })
        print(f"{lid:<4} {med:8.3f} {q1:8.3f} {q3:8.3f} "
              f"{bo:8.3f} {mean:8.3f}{tail}", flush=True)

    # Sample hyp per lang
    print("\n=== sample hyps (rep0) ===", flush=True)
    for r in rows:
        print(f"  {r['lang']}: ref={r['text'][:35]}", flush=True)
        print(f"      hyp={r['hyps'][0][:60]}", flush=True)

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "n": N}, f, indent=2, ensure_ascii=False)
    print(f"\n[plain] result -> {OUT_DIR / 'result.json'}")


if __name__ == "__main__":
    main()
