"""Build a listening packet that compares all VC backbones side by side.

For each emotion, gather:
  01_F1_ref.wav                 — target speaker
  02_baseline_F1_zh.wav         — current production (no VC, JP-accented)
  03_content_native.wav         — content donor (native voice, clean zh)
  04_knn_pm0_k8.wav             — knn-VC original (muffled, F1 lost)
  05_knn_pm1_k2.wav             — knn-VC tuned (less averaging)
  06_openvoice_tau0.1.wav       — OpenVoice TCC, weakest timbre push
  07_openvoice_tau0.3.wav       — OpenVoice TCC, mid
  08_openvoice_tau0.5.wav       — OpenVoice TCC, strongest

Pick the lowest-CER rep from each pool. Generates index.html with
embedded audio players + transcripts + CER per cell.
"""
from __future__ import annotations
import os, sys, html, json, shutil
from pathlib import Path

sys.path.insert(0, "src")
from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.evaluation.zh_eval import cer_zh                                     # noqa: E402


JVNV_DIR     = Path("baseline/jvnv_samples")
EPS3_DIR     = Path("outputs/eps3_semantic_vc")
KNN_TUNE2    = Path("outputs/eps3_tune_phase2")
KNN_QV       = Path("outputs/eps3_quality_variants")
OV_DIR       = Path("outputs/eps3_openvoice_tcc")
OUT_DIR      = Path("outputs/vc_comparison_listening")
ZH_HANZI     = "明天的天气预报是多云转晴。"
EMOTIONS     = ["sad", "happy", "anger"]
REPS         = 8


def _pick_best(asr, candidates):
    best = None
    for p in candidates:
        if not p.exists():
            continue
        hyp = asr.transcribe(p, language="zh")
        cer = cer_zh(ZH_HANZI, hyp)
        if best is None or cer < best[2]:
            best = (p, hyp, cer)
    return best


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[pkt-vc] loading ASR ...", flush=True)
    asr = ASRAnalyzer()

    rows = []
    for emo in EMOTIONS:
        emo_dir = OUT_DIR / emo
        emo_dir.mkdir(exist_ok=True)
        cells = []

        # 01 F1 ref
        ref = JVNV_DIR / f"jvnv_F1_{emo}.wav"
        ref_hyp = asr.transcribe(ref, language=None)
        shutil.copy2(ref, emo_dir / "01_F1_ref.wav")
        cells.append(("01 F1 reference", "01_F1_ref.wav", "—", ref_hyp))

        # 02 baseline (speaker_donor)
        sp_best = _pick_best(asr, [EPS3_DIR / f"speaker_donor_{emo}_rep{r}.wav"
                                    for r in range(REPS)])
        shutil.copy2(sp_best[0], emo_dir / "02_baseline_F1_zh.wav")
        cells.append(("02 baseline (F1 + JP-accented zh)",
                      "02_baseline_F1_zh.wav", f"{sp_best[2]:.3f}", sp_best[1]))

        # 03 content_donor
        ct_best = _pick_best(asr, [EPS3_DIR / f"content_donor_{emo}_rep{r}.wav"
                                    for r in range(REPS)])
        shutil.copy2(ct_best[0], emo_dir / "03_content_native.wav")
        cells.append(("03 content donor (native voice)",
                      "03_content_native.wav", f"{ct_best[2]:.3f}", ct_best[1]))

        # 04 knn pm0_k8
        knn1 = _pick_best(asr, [KNN_TUNE2 / f"xl_k8_{emo}_rep{r}.wav"
                                 for r in range(REPS)])
        if knn1:
            shutil.copy2(knn1[0], emo_dir / "04_knn_pm0_k8.wav")
            cells.append(("04 knn-VC pm0_k8 (muffled, F1 lost)",
                          "04_knn_pm0_k8.wav", f"{knn1[2]:.3f}", knn1[1]))

        # 05 knn pm1_k2 (tuned)
        knn2 = _pick_best(asr, [KNN_QV / f"pm1_k2_{emo}_rep{r}.wav"
                                 for r in range(REPS)])
        if knn2:
            shutil.copy2(knn2[0], emo_dir / "05_knn_pm1_k2.wav")
            cells.append(("05 knn-VC pm1_k2 (less averaging)",
                          "05_knn_pm1_k2.wav", f"{knn2[2]:.3f}", knn2[1]))

        # 06-08 OpenVoice taus
        for tau in (0.1, 0.3, 0.5):
            ov = _pick_best(asr, [OV_DIR / f"tau{tau:.1f}_{emo}_rep{r}.wav"
                                   for r in range(REPS)])
            if ov:
                idx = {0.1: "06", 0.3: "07", 0.5: "08"}[tau]
                shutil.copy2(ov[0], emo_dir / f"{idx}_openvoice_tau{tau:.1f}.wav")
                cells.append((f"{idx} OpenVoice TCC tau={tau:.1f}",
                              f"{idx}_openvoice_tau{tau:.1f}.wav",
                              f"{ov[2]:.3f}", ov[1]))

        rows.append({"emotion": emo, "cells": cells})
        print(f"[pkt-vc] {emo}: {len(cells)} streams", flush=True)
        for label, _, cer, hyp in cells:
            print(f"  {label:<42}  cer={cer}  hyp={hyp[:55]}")

    # HTML
    h = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
         "<title>VC backbone comparison</title>",
         "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}",
         "table{border-collapse:collapse;width:100%}",
         "td,th{border:1px solid #ccc;padding:8px;vertical-align:top}",
         "audio{width:280px}",
         "code{background:#f4f4f4;padding:2px 6px;border-radius:3px;font-size:0.9em}",
         "h2{margin-top:2em;border-bottom:2px solid #333;padding-bottom:4px}",
         ".target{background:#ffffd0;padding:4px 8px;border-radius:4px}",
         "tr.win{background:#e0ffe0}</style></head><body>",
         f"<h1>VC backbone comparison</h1>",
         f"<p>Target: <code class='target'>{ZH_HANZI}</code> (12 chars)</p>",
         "<p>Listening checklist:</p><ol>"
         "<li>Is the voice color recognizable as F1 (compare with 01)?</li>"
         "<li>Is the audio clear (not muffled)?</li>"
         "<li>Is the content correct (low CER, transcript right)?</li></ol>"]
    for r in rows:
        h.append(f"<h2>{r['emotion']}</h2><table>")
        h.append("<tr><th>variant</th><th>audio</th><th>CER</th>"
                 "<th>Whisper hyp</th></tr>")
        for label, fname, cer, hyp in r["cells"]:
            highlight = " class='win'" if label.startswith(("06", "07", "08")) else ""
            h.append(f"<tr{highlight}><td>{html.escape(label)}</td>"
                     f"<td><audio controls src='{r['emotion']}/{fname}'></audio></td>"
                     f"<td>{cer}</td><td><code>{html.escape(hyp)}</code></td></tr>")
        h.append("</table>")
    h.append("</body></html>")
    (OUT_DIR / "index.html").write_text("\n".join(h), encoding="utf-8")
    with open(OUT_DIR / "rows.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n[pkt-vc] saved -> {OUT_DIR / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
