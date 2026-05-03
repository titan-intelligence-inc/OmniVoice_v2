"""Human-readable Markdown report for a Phase 2 run.

Inputs come from ``cli/run_phase2.py``: the per-candidate scores plus the
optional stability data. The output is a single ``report.md``.
"""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ..types import Candidate, Thresholds
from .stability import StabilityReport


def _fmt(x: float, prec: int = 3) -> str:
    return f"{x:.{prec}f}"


def _candidate_row(c: Candidate) -> list[str]:
    s = c.scores
    return [
        c.meta.get("tag", ""),
        c.instruct or "",
        _fmt(c.total_score),
        _fmt(s.vad_dist),
        _fmt(s.valence_diff),
        _fmt(s.arousal_diff),
        _fmt(s.f0_std_ratio),
        _fmt(s.energy_std_ratio),
        _fmt(s.e2v_cos),
        s.e2v_top,
        _fmt(s.speaker_sim),
        _fmt(s.content_error),
    ]


def render_report(
    title: str,
    request_meta: dict,
    ref_summary: dict,
    candidates: list[Candidate],
    best_tag: str,
    thresholds: Thresholds,
    stability: list[StabilityReport] | None = None,
) -> str:
    out: list[str] = []
    out.append(f"# {title}\n")

    out.append("## Request\n")
    for k, v in request_meta.items():
        out.append(f"- **{k}**: `{v}`")
    out.append("")

    out.append("## Reference summary\n")
    for k, v in ref_summary.items():
        if isinstance(v, dict):
            inner = ", ".join(f"{kk}={vv:.3f}" if isinstance(vv, (int, float)) else f"{kk}={vv}"
                              for kk, vv in v.items())
            out.append(f"- **{k}**: {inner}")
        else:
            out.append(f"- **{k}**: {v}")
    out.append("")

    out.append("## Candidates\n")
    out.append("| tag | instruct | score | vad_dist | val_diff | aro_diff | f0_ratio | E_ratio | e2v_cos | e2v_top | spk | CER |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in candidates:
        row = _candidate_row(c)
        marker = " ⭐" if c.meta.get("tag") == best_tag else ""
        out.append("| " + " | ".join(row) + " |" + marker)
    out.append("")

    out.append("## Constraints\n")
    out.append(f"- content_error ≤ {thresholds.content_error}")
    out.append(f"- audio_quality ≥ {thresholds.quality}")
    out.append(f"- speaker_sim ≥ {thresholds.speaker}")
    out.append("")

    if stability:
        out.append("## Stability (per-spec, std across reps)\n")
        out.append("| tag | reps | vad_dist (mean ± std) | val_diff (mean ± std) | E_ratio (mean ± std) | e2v_cos (mean ± std) | spk (mean ± std) | CER (mean ± std) |")
        out.append("|---|---|---|---|---|---|---|---|")
        for s in stability:
            pm = s.per_metric
            out.append("| " + " | ".join([
                s.spec_tag, str(s.reps),
                f"{pm['vad_dist'].mean:.3f} ± {pm['vad_dist'].std:.3f}",
                f"{pm['valence_diff'].mean:.3f} ± {pm['valence_diff'].std:.3f}",
                f"{pm['energy_std_ratio'].mean:.3f} ± {pm['energy_std_ratio'].std:.3f}",
                f"{pm['e2v_cos'].mean:.3f} ± {pm['e2v_cos'].std:.3f}",
                f"{pm['speaker_sim'].mean:.3f} ± {pm['speaker_sim'].std:.3f}",
                f"{pm['content_error'].mean:.3f} ± {pm['content_error'].std:.3f}",
            ]) + " |")
        out.append("")

        out.append("**Reliability gate** (Phase 5 acceptance): all metric std ≤ 0.05.\n")
        violations = []
        for s in stability:
            for name, st in s.per_metric.items():
                if st.std > 0.05:
                    violations.append((s.spec_tag, name, st.std))
        if violations:
            out.append("⚠️ Metrics exceeding 0.05 std:\n")
            for tag, name, std in violations:
                out.append(f"- `{tag}` / `{name}` → std={std:.3f}")
        else:
            out.append("✅ All metrics within reliability threshold.")
        out.append("")

    out.append(f"## Selected\n\n**{best_tag}**\n")
    return "\n".join(out)


def write_report(path: str | Path, content: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
