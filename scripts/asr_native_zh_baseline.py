"""ASR baseline measurement on native zh audio.

Goal: establish the noise floor of our evaluator (whisper-large-v3-
turbo) on clean native Chinese speech, under the **production CER
recipe** (Trad/Simp folded, hallucination-capped, median-primary).

Pipeline:
  1. Pull FLEURS cmn_hans_cn dev.tar.gz + dev.tsv  (transcriptions)
  2. Match wavs to transcriptions, run Whisper on each
  3. Compute CER three ways for diagnostics:
       cer_raw_strict      — vs raw transcription, no Trad/Simp folding
       cer_raw_zhfolded    — vs raw transcription, Trad/Simp folded
       cer_norm_zhfolded   — vs normalized transcription, Trad/Simp folded
                             (this is the "evaluation recipe" we use
                             on OmniVoice output)
  4. Report median / mean / std / min / max for each, plus
     hallucination rate (Whisper looping on clean audio).

Per-cell stats use median as primary (robust to the long tail of
Whisper hallucinations on clean native audio).

Output: outputs/asr_native_zh_baseline/result.json + console summary.
"""
from __future__ import annotations
import os
import sys
import csv
import json
import statistics
import tarfile
import tempfile
from pathlib import Path

import soundfile as sf

sys.path.insert(0, "src")
from ovet.analyzers.asr_analyzer import (                                    # noqa: E402
    ASRAnalyzer, _normalize_text, _cer, _detect_hallucination,
)


N_SAMPLES   = 100        # sample size for CER stats (100 = a few minutes wall clock)
OUT_DIR     = Path("outputs/asr_native_zh_baseline")
HF_HOME     = "/workspace/hf_cache"


def _pull_fleurs_zh_dev() -> tuple[Path, Path]:
    """Returns (dev.tar.gz, dev.tsv) paths in HF cache."""
    from huggingface_hub import hf_hub_download
    tar_path = Path(hf_hub_download(
        repo_id="google/fleurs",
        filename="data/cmn_hans_cn/audio/dev.tar.gz",
        repo_type="dataset",
    ))
    tsv_path = Path(hf_hub_download(
        repo_id="google/fleurs",
        filename="data/cmn_hans_cn/dev.tsv",
        repo_type="dataset",
    ))
    return tar_path, tsv_path


def _read_tsv(tsv_path: Path) -> list[dict]:
    """FLEURS tsv columns (tab-separated, no header):
       id, filename, raw_transcription, transcription, n_samples,
       speaker_id, gender."""
    rows = []
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            rows.append({
                "id":       parts[0],
                "filename": parts[1],
                "raw":      parts[2],
                "norm":     parts[3],
            })
    return rows


def main():
    os.environ.setdefault("HF_HOME", HF_HOME)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[asr-base] pulling FLEURS cmn_hans_cn dev metadata + audio ...", flush=True)
    tar_path, tsv_path = _pull_fleurs_zh_dev()
    rows = _read_tsv(tsv_path)
    print(f"[asr-base] {len(rows)} rows in dev.tsv", flush=True)

    # Index transcriptions by filename. Note: tsv filename may be
    # like '10116334326946811387.wav' matching the wav inside the tarball.
    by_fname = {r["filename"]: r for r in rows}

    print(f"[asr-base] extracting first {N_SAMPLES} wavs from tarball ...", flush=True)
    extract_dir = OUT_DIR / "wavs"
    extract_dir.mkdir(exist_ok=True)
    samples: list[dict] = []
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if m.name.endswith(".wav")]
        for m in members[:N_SAMPLES]:
            base = Path(m.name).name
            if base not in by_fname:
                continue
            out_wav = extract_dir / base
            if not out_wav.exists():
                with open(out_wav, "wb") as o:
                    o.write(tf.extractfile(m).read())
            samples.append({
                "wav": out_wav,
                "ref_raw": by_fname[base]["raw"],
                "ref_norm": by_fname[base]["norm"],
                "id": by_fname[base]["id"],
            })
            if len(samples) >= N_SAMPLES:
                break
    print(f"[asr-base] {len(samples)} wavs ready for ASR", flush=True)

    print("[asr-base] loading whisper-large-v3-turbo via ASRAnalyzer ...", flush=True)
    asr = ASRAnalyzer()

    # 3-way CER diagnostic per sample.
    results = []
    print("\n[asr-base] running ASR (3-way CER per sample) ...", flush=True)
    for i, s in enumerate(samples):
        hyp = asr.transcribe(s["wav"], language="zh")
        results.append({
            "id":       s["id"],
            "wav":      s["wav"].name,
            "ref_raw":  s["ref_raw"],
            "ref_norm": s["ref_norm"],
            "hyp":      hyp,
            "cer_raw_strict":   _cer(s["ref_raw"],  hyp, zh=False),
            "cer_raw_zhfolded": _cer(s["ref_raw"],  hyp, zh=True),
            "cer_norm_zhfolded": _cer(s["ref_norm"], hyp, zh=True),
            "hallucinated":     _detect_hallucination(_normalize_text(hyp, zh=True)),
            "len_ref_norm": len(_normalize_text(s["ref_norm"], zh=True)),
        })
        if (i + 1) % 10 == 0:
            running = [r["cer_norm_zhfolded"] for r in results]
            print(f"  {i+1}/{len(samples)}: running median CER={statistics.median(running):.3f} "
                  f"(zhfolded vs norm)", flush=True)

    # Aggregate
    print("\n=== Whisper CER on FLEURS native zh dev (median primary) ===", flush=True)
    print(f"  N samples: {len(results)}", flush=True)
    print(f"  hallucination rate: "
          f"{sum(int(r['hallucinated']) for r in results) / len(results):.1%}",
          flush=True)
    print()
    for label, key in [
        ("CER vs raw, NO Trad/Simp fold     ", "cer_raw_strict"),
        ("CER vs raw, Trad/Simp folded      ", "cer_raw_zhfolded"),
        ("CER vs norm, Trad/Simp folded     ", "cer_norm_zhfolded"),
    ]:
        vals = [r[key] for r in results]
        med = statistics.median(vals)
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals)
        # Quartiles
        q1 = statistics.quantiles(vals, n=4)[0]
        q3 = statistics.quantiles(vals, n=4)[2]
        print(f"{label}: median={med:.4f}  mean={mean:.4f}  σ={std:.4f}  "
              f"Q1={q1:.4f}  Q3={q3:.4f}  min={min(vals):.4f}  max={max(vals):.4f}",
              flush=True)

    # Worst / best diagnostics use the folded-norm metric (= our
    # production recipe).
    primary_key = "cer_norm_zhfolded"
    print(f"\n=== worst-5 by {primary_key} ===", flush=True)
    for r in sorted(results, key=lambda x: -x[primary_key])[:5]:
        print(f"  cer={r[primary_key]:.3f}  hallu={r['hallucinated']}  ref='{r['ref_raw'][:55]}'")
        print(f"                              hyp='{r['hyp'][:55]}'")

    print(f"\n=== best-5 by {primary_key} ===", flush=True)
    for r in sorted(results, key=lambda x: x[primary_key])[:5]:
        print(f"  cer={r[primary_key]:.3f}  ref='{r['ref_raw'][:55]}'")
        print(f"                  hyp='{r['hyp'][:55]}'")

    def _stats(vals):
        return {
            "median": statistics.median(vals),
            "mean":   statistics.fmean(vals),
            "std":    statistics.pstdev(vals),
            "q1":     statistics.quantiles(vals, n=4)[0],
            "q3":     statistics.quantiles(vals, n=4)[2],
            "min":    min(vals),
            "max":    max(vals),
        }

    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_id":    asr.model_id,
            "n_samples":   len(results),
            "rows":        results,
            "stats_raw_strict":   _stats([r["cer_raw_strict"]    for r in results]),
            "stats_raw_zhfolded": _stats([r["cer_raw_zhfolded"]  for r in results]),
            "stats_norm_zhfolded":_stats([r["cer_norm_zhfolded"] for r in results]),
            "hallucination_rate": sum(int(r["hallucinated"]) for r in results) / len(results),
            "primary_metric":     "cer_norm_zhfolded (median)",
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[asr-base] saved -> {OUT_DIR / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
