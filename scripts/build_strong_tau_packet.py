"""Build a focused listening packet for the high-tau OpenVoice TCC sweep.

For each emotion, gather:
  01_F1_ref.wav                                  — target speaker
  02_baseline_F1_zh.wav                          — F1 + JP-accented zh
  03_content_native.wav                          — native voice + clean zh
  04_ovTCC_jvnv_only_tau0.5.wav                  — TCC, mid push, jvnv-only SE
  05_ovTCC_jvnv_only_tau0.7.wav                  — TCC, stronger push
  06_ovTCC_jvnv_only_tau1.0.wav                  — TCC, max push (sampling limit)
  07_ovTCC_jvnv_long_tau1.0.wav                  — same with concat-jvnv SE
  08_ovTCC_all_18_tau1.0.wav                     — same with mixed SE

The question we want to settle: does any tau / SE combination get
audible F1 timbre back? If not, ToneColorConverter is the wrong tool
and we move to SeedVC.
"""
from __future__ import annotations
import os, sys, html, json, shutil
from pathlib import Path

sys.path.insert(0, "src")
from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.evaluation.zh_eval import cer_zh                                     # noqa: E402


JVNV_DIR  = Path("baseline/jvnv_samples")
EPS3_DIR  = Path("outputs/eps3_semantic_vc")
TAU_DIR   = Path("outputs/eps3_ovc_strong_tau")
OUT_DIR   = Path("outputs/strong_tau_listening")
ZH_HANZI  = "明天的天气预报是多云转晴。"
EMOTIONS  = ["sad", "happy", "anger"]
REPS      = 8


def _pick_best(asr, candidates):
    best = None
    for p in candidates:
        if not p.exists(): continue
        hyp = asr.transcribe(p, language="zh")
        cer = cer_zh(ZH_HANZI, hyp)
        if best is None or cer < best[2]:
            best = (p, hyp, cer)
    return best


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    asr = ASRAnalyzer()

    streams = [
        ("F1 reference", "01", lambda emo: [(JVNV_DIR / f"jvnv_F1_{emo}.wav")]),
        ("baseline (F1 + JP zh)", "02",
            lambda emo: [EPS3_DIR / f"speaker_donor_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
        ("content donor (native voice)", "03",
            lambda emo: [EPS3_DIR / f"content_donor_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
        ("TCC jvnv_only tau=0.5", "04",
            lambda emo: [TAU_DIR / f"jvnv_only_tau0.5_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
        ("TCC jvnv_only tau=0.7", "05",
            lambda emo: [TAU_DIR / f"jvnv_only_tau0.7_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
        ("TCC jvnv_only tau=1.0", "06",
            lambda emo: [TAU_DIR / f"jvnv_only_tau1.0_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
        ("TCC jvnv_long tau=1.0", "07",
            lambda emo: [TAU_DIR / f"jvnv_long_tau1.0_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
        ("TCC all_18 tau=1.0", "08",
            lambda emo: [TAU_DIR / f"all_18_tau1.0_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
    ]

    rows = []
    for emo in EMOTIONS:
        emo_dir = OUT_DIR / emo
        emo_dir.mkdir(exist_ok=True)
        cells = []
        for label, idx, pickfn in streams:
            cands = pickfn(emo)
            best = _pick_best(asr, cands)
            if best is None: continue
            fname = f"{idx}_{label.replace(' ', '_').replace('=', '').replace('(', '').replace(')', '').replace('+', '').replace('.', '')}.wav"
            shutil.copy2(best[0], emo_dir / fname)
            cells.append({"label": label, "fname": fname,
                          "cer": best[2], "hyp": best[1]})
        rows.append({"emotion": emo, "cells": cells})
        print(f"[pkt] {emo}: {len(cells)} streams")
        for c in cells:
            print(f"  {c['label']:<32} cer={c['cer']:.3f}  hyp={c['hyp'][:50]}")

    h = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
         "<title>OpenVoice TCC strong-tau listening</title>",
         "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}",
         "table{border-collapse:collapse;width:100%}",
         "td,th{border:1px solid #ccc;padding:8px;vertical-align:top}",
         "audio{width:280px}",
         "code{background:#f4f4f4;padding:2px 6px;border-radius:3px;font-size:0.9em}",
         "h2{margin-top:2em;border-bottom:2px solid #333;padding-bottom:4px}",
         ".target{background:#ffffd0;padding:4px 8px;border-radius:4px}",
         "</style></head><body>",
         f"<h1>OpenVoice TCC strong-tau listening</h1>",
         f"<p>Target: <code class='target'>{ZH_HANZI}</code></p>",
         "<p>Question: does <b>any</b> tau / SE combination get F1's "
         "timbre back? Compare the converted streams (04-08) with "
         "F1 reference (01).</p>"]
    for r in rows:
        h.append(f"<h2>{r['emotion']}</h2><table>")
        h.append("<tr><th>variant</th><th>audio</th><th>CER</th><th>Whisper hyp</th></tr>")
        for c in r["cells"]:
            cer = "—" if c["cer"] is None else f"{c['cer']:.3f}"
            h.append(f"<tr><td>{html.escape(c['label'])}</td>"
                     f"<td><audio controls src='{r['emotion']}/{c['fname']}'></audio></td>"
                     f"<td>{cer}</td><td><code>{html.escape(c['hyp'])}</code></td></tr>")
        h.append("</table>")
    h.append("</body></html>")
    (OUT_DIR / "index.html").write_text("\n".join(h), encoding="utf-8")
    with open(OUT_DIR / "rows.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n[pkt] saved -> {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
