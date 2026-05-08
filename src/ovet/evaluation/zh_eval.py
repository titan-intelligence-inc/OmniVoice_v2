"""Centralized zh evaluation helpers using the recommended design.

Recommendations from the FLEURS native-zh ASR baseline measurement:

  1. **Median > Mean** as the primary CER metric. The Whisper noise
     floor distribution has a long tail (max=1.12 on 100 native
     samples; std ≈ 14 percentage points). Median is robust to the
     occasional hallucination rep.
  2. **reps ≥ 8** per cell. Per-rep std on native zh is ~14%pt; need
     enough reps to discriminate small effect sizes.
  3. **Trad/Simp normalization** before CER. Whisper-large-v3-turbo
     occasionally emits Traditional characters; folding to Simplified
     removes a significant amount of spurious CER. (Already wired
     through ``ASRAnalyzer.content_error(language="zh")`` — this
     helper makes it explicit.)

Use this module's helpers instead of hand-rolling CER aggregation in
sweep scripts so the choice of stat is consistent across experiments.
"""
from __future__ import annotations
import re
import statistics
from dataclasses import dataclass

from ..analyzers.asr_analyzer import ASRAnalyzer, _cer, _normalize_text


# Default reps tied to the FLEURS std observation.
RECOMMENDED_REPS = 8


@dataclass
class ZhCellStats:
    """Per-cell aggregated stats for zh evaluation.

    ``cer_median`` is the recommended primary metric.
    ``cer_mean`` is reported for legacy compatibility.
    ``hallu_rate`` is a separate stability indicator (a single
    hallucinated rep already pegs CER at the cap=2.0).
    """
    n_reps:        int
    cer_median:    float
    cer_mean:      float
    cer_std:       float
    cer_min:       float
    cer_max:       float
    hallu_rate:    float
    prefix_mean:   float
    prefix_max:    int
    kana_median:   float
    hyps:          list[str]


_KATAKANA_RE   = re.compile(r"[゠-ヿ]")
_HIRAGANA_RE   = re.compile(r"[぀-ゟ]")
_HALLU_PATTERN = re.compile(r"(.{1,4})\1{8,}")


def _strip_punct(s: str) -> str:
    return re.sub(r"[\s\.,!?。、！？・…]+", "", s).lower()


def _has_hallucination(s: str) -> bool:
    return bool(_HALLU_PATTERN.search(s))


def _kana_count(s: str) -> int:
    return len(_KATAKANA_RE.findall(s)) + len(_HIRAGANA_RE.findall(s))


def _prefix_match(hyp: str, target: str) -> int:
    a, b = _strip_punct(hyp), _strip_punct(target)
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def cer_zh(reference: str, hyp: str, *, cap: float | None = 2.0) -> float:
    """Trad-Simp-normalized CER. Use this for all zh comparisons."""
    return _cer(reference, hyp, cap=cap, zh=True)


def aggregate_zh_cell(
    hyps_per_rep: list[str], reference: str,
) -> ZhCellStats:
    """Aggregate one cell's reps into ZhCellStats.

    The CERs here use the Trad-Simp normalization. Pass the raw
    Whisper hypothesis strings (no pre-normalization needed).
    """
    cers      = [cer_zh(reference, h) for h in hyps_per_rep]
    prefixes  = [_prefix_match(h, reference) for h in hyps_per_rep]
    hallus    = [int(_has_hallucination(h)) for h in hyps_per_rep]
    kanas     = [_kana_count(h) for h in hyps_per_rep]
    n = len(cers)
    return ZhCellStats(
        n_reps      = n,
        cer_median  = statistics.median(cers),
        cer_mean    = statistics.fmean(cers),
        cer_std     = statistics.pstdev(cers) if n > 1 else 0.0,
        cer_min     = min(cers),
        cer_max     = max(cers),
        hallu_rate  = sum(hallus) / n,
        prefix_mean = statistics.fmean(prefixes),
        prefix_max  = max(prefixes),
        kana_median = statistics.median(kanas),
        hyps        = list(hyps_per_rep),
    )


def transcribe_and_aggregate(
    asr: ASRAnalyzer, wav_paths: list, reference: str,
) -> ZhCellStats:
    """Transcribe a list of wav paths with zh language hint, then
    aggregate. Convenience wrapper for sweep scripts."""
    hyps = [asr.transcribe(p, language="zh") for p in wav_paths]
    return aggregate_zh_cell(hyps, reference)


def format_cell_row(name: str, stats: ZhCellStats) -> str:
    """One-line summary suitable for table rows."""
    return (
        f"{name:<22} "
        f"cer_med={stats.cer_median:.3f} "
        f"cer_mean={stats.cer_mean:.3f} "
        f"σ={stats.cer_std:.3f} "
        f"hallu={stats.hallu_rate:.0%} "
        f"pfx={stats.prefix_mean:.1f}/{stats.prefix_max} "
        f"kana={stats.kana_median:.0f}"
    )
