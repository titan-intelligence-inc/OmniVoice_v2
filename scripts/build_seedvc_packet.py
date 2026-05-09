"""Final VC backbone comparison packet — SeedVC vs OpenVoice TCC vs knn-VC.

For each emotion, gather:
  01_F1_ref.wav                          — target speaker
  02_baseline_F1_zh.wav                  — F1 + JP-accented zh
  03_content_native.wav                  — native voice + clean zh
  04_knn_pm0_k8.wav                      — knn-VC (muffled, F1 lost)
  05_OpenVoice_TCC_tau1.0.wav            — OV TCC max push (F1 still lost)
  06_SeedVC_cfg0.7_steps10_emo_ref.wav   — SeedVC moderate
  07_SeedVC_cfg0.9_steps10_long_ref.wav  — SeedVC strong + concat ref
  08_SeedVC_cfg0.9_steps25_emo_ref.wav   — SeedVC strong + more steps

Key question: does SeedVC bring F1's voice back where TCC failed?
"""
from __future__ import annotations
import os, sys, html, json, shutil
from pathlib import Path

sys.path.insert(0, "src")
from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.evaluation.zh_eval import cer_zh                                     # noqa: E402


JVNV_DIR  = Path("baseline/jvnv_samples")
EPS3_DIR  = Path("outputs/eps3_semantic_vc")
KNN_DIR   = Path("outputs/eps3_tune_phase2")
TCC_DIR   = Path("outputs/eps3_ovc_strong_tau")
SVC_DIR   = Path("outputs/eps3_seedvc")
OUT_DIR   = Path("outputs/seedvc_listening")
ZH_HANZI  = "明天的天气预报是多云转晴。"
EMOTIONS  = ["sad", "happy", "anger"]
REPS      = 8


def _pick_best(asr, cands):
    best = None
    for p in cands:
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
        ("01 F1 reference", lambda emo: [JVNV_DIR / f"jvnv_F1_{emo}.wav"]),
        ("02 baseline (F1 + JP zh)",
            lambda emo: [EPS3_DIR / f"speaker_donor_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
        ("03 content donor (native voice)",
            lambda emo: [EPS3_DIR / f"content_donor_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
        ("04 knn-VC pm0_k8",
            lambda emo: [KNN_DIR / f"xl_k8_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
        ("05 OpenVoice TCC tau=1.0 (F1 not recovered)",
            lambda emo: [TCC_DIR / f"jvnv_only_tau1.0_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
        ("06 SeedVC cfg=0.7 / steps=10 / emo_ref",
            lambda emo: [SVC_DIR / f"cfg0.7_steps10_emo_ref_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
        ("07 SeedVC cfg=0.9 / steps=10 / long_ref",
            lambda emo: [SVC_DIR / f"cfg0.9_steps10_long_ref_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
        ("08 SeedVC cfg=0.9 / steps=25 / emo_ref",
            lambda emo: [SVC_DIR / f"cfg0.9_steps25_emo_ref_{emo}_rep{r}.wav"
                         for r in range(REPS)]),
    ]

    rows = []
    for emo in EMOTIONS:
        emo_dir = OUT_DIR / emo
        emo_dir.mkdir(exist_ok=True)
        cells = []
        for label, pickfn in streams:
            best = _pick_best(asr, pickfn(emo))
            if best is None:
                continue
            idx = label.split()[0]
            fname = f"{idx}_{label.split(' ', 1)[1].replace(' ', '_').replace('/', '').replace('=', '').replace('(', '').replace(')', '').replace('.', '')}.wav"
            shutil.copy2(best[0], emo_dir / fname)
            cells.append({"label": label, "fname": fname,
                          "cer": best[2], "hyp": best[1]})
        rows.append({"emotion": emo, "cells": cells})
        print(f"[pkt] {emo}:")
        for c in cells:
            print(f"  {c['label']:<48} cer={c['cer']:.3f}  hyp={c['hyp'][:55]}")

    h = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
         "<title>SeedVC final comparison</title>",
         "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}",
         "table{border-collapse:collapse;width:100%}",
         "td,th{border:1px solid #ccc;padding:8px;vertical-align:top}",
         "audio{width:280px}",
         "code{background:#f4f4f4;padding:2px 6px;border-radius:3px;font-size:0.9em}",
         "h2{margin-top:2em;border-bottom:2px solid #333;padding-bottom:4px}",
         ".target{background:#ffffd0;padding:4px 8px;border-radius:4px}",
         "tr.svc{background:#e0ffe0}</style></head><body>",
         f"<h1>SeedVC vs prior VC backbones</h1>",
         f"<p>Target: <code class='target'>{ZH_HANZI}</code></p>",
         "<p>Listening checklist:</p><ol>"
         "<li><b>F1 voice</b>: streams 06-08 should sound like 01 (F1 ref)</li>"
         "<li><b>Audio quality</b>: streams 06-08 should not be muffled</li>"
         "<li><b>Content correctness</b>: CER should be near 0</li></ol>"]
    for r in rows:
        h.append(f"<h2>{r['emotion']}</h2><table>")
        h.append("<tr><th>variant</th><th>audio</th><th>CER</th><th>Whisper hyp</th></tr>")
        for c in r["cells"]:
            cls = " class='svc'" if c["label"].startswith(("06", "07", "08")) else ""
            h.append(f"<tr{cls}><td>{html.escape(c['label'])}</td>"
                     f"<td><audio controls src='{r['emotion']}/{c['fname']}'></audio></td>"
                     f"<td>{c['cer']:.3f}</td><td><code>{html.escape(c['hyp'])}</code></td></tr>")
        h.append("</table>")
    h.append("</body></html>")
    (OUT_DIR / "index.html").write_text("\n".join(h), encoding="utf-8")
    print(f"\n[pkt] saved -> {OUT_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
