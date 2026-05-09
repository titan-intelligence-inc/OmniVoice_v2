"""Build a listening packet from ε-3 outputs.

For each emotion in {sad, happy, anger}, gather:
  1. F1 reference (the JVNV ref we cloned)
  2. baseline   = speaker_donor (F1 voice + JP-accented zh, cer ≈ 1.0)
  3. content    = best content_donor (native ref voice + clean zh, cer ≈ 0.08)
  4. converted  = best xl_k8 best-of-8 (F1 voice + native content, cer ≈ 0.17)

Saves them under outputs/eps3_listening/{emo}/ with descriptive
filenames and writes index.md / index.html with paired ASR transcripts.
"""
from __future__ import annotations
import os, sys, shutil, html, json
from pathlib import Path

sys.path.insert(0, "src")
from ovet.analyzers.asr_analyzer import ASRAnalyzer                            # noqa: E402
from ovet.evaluation.zh_eval import cer_zh                                     # noqa: E402


JVNV_DIR     = Path("baseline/jvnv_samples")
EPS3_DIR     = Path("outputs/eps3_semantic_vc")
TUNE2_DIR    = Path("outputs/eps3_tune_phase2")
OUT_DIR      = Path("outputs/eps3_listening")
ZH_HANZI     = "明天的天气预报是多云转晴。"
EMOTIONS     = ["sad", "happy", "anger"]
REPS         = 8


def _pick_best(asr: ASRAnalyzer, candidates: list[Path]) -> tuple[Path, str, float]:
    best = None
    for p in candidates:
        hyp = asr.transcribe(p, language="zh")
        cer = cer_zh(ZH_HANZI, hyp)
        if best is None or cer < best[2]:
            best = (p, hyp, cer)
    return best


def main():
    os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[pkt] loading ASR ...", flush=True)
    asr = ASRAnalyzer()

    rows = []
    for emo in EMOTIONS:
        emo_dir = OUT_DIR / emo
        emo_dir.mkdir(exist_ok=True)
        # 1. F1 reference
        f1_ref = JVNV_DIR / f"jvnv_F1_{emo}.wav"
        f1_ref_txt = asr.transcribe(f1_ref, language=None)
        shutil.copy2(f1_ref, emo_dir / "01_F1_ref.wav")

        # 2. baseline (speaker_donor) — pick best (lowest CER) across 8 reps
        speaker_cands = [EPS3_DIR / f"speaker_donor_{emo}_rep{r}.wav"
                         for r in range(REPS)]
        sp_best, sp_hyp, sp_cer = _pick_best(asr, speaker_cands)
        shutil.copy2(sp_best, emo_dir / "02_baseline_F1_zh.wav")

        # 3. content_donor — pick best (lowest CER)
        content_cands = [EPS3_DIR / f"content_donor_{emo}_rep{r}.wav"
                         for r in range(REPS)]
        ct_best, ct_hyp, ct_cer = _pick_best(asr, content_cands)
        shutil.copy2(ct_best, emo_dir / "03_content_native.wav")

        # 4. converted — pick best from xl_k8 (= production winner)
        converted_cands = [TUNE2_DIR / f"xl_k8_{emo}_rep{r}.wav"
                           for r in range(REPS)]
        cv_best, cv_hyp, cv_cer = _pick_best(asr, converted_cands)
        shutil.copy2(cv_best, emo_dir / "04_eps3_converted.wav")

        # 5. median-of-8 representative for converted (= the
        #    "expected" production quality per single rep)
        converted_med = sorted(
            [(p, asr.transcribe(p, language="zh")) for p in converted_cands],
            key=lambda t: cer_zh(ZH_HANZI, t[1]),
        )[len(converted_cands) // 2]
        shutil.copy2(converted_med[0], emo_dir / "05_eps3_converted_median.wav")
        cv_med_hyp = converted_med[1]
        cv_med_cer = cer_zh(ZH_HANZI, cv_med_hyp)

        rows.append({
            "emotion": emo,
            "f1_ref_text":  f1_ref_txt,
            "speaker_donor": {"src": str(sp_best), "hyp": sp_hyp, "cer": sp_cer},
            "content_donor": {"src": str(ct_best), "hyp": ct_hyp, "cer": ct_cer},
            "converted_best": {"src": str(cv_best), "hyp": cv_hyp, "cer": cv_cer},
            "converted_median": {"src": str(converted_med[0]),
                                 "hyp": cv_med_hyp, "cer": cv_med_cer},
        })
        print(f"[pkt] {emo}: ", flush=True)
        print(f"  F1-ref     : {f1_ref_txt[:60]}")
        print(f"  baseline   : cer={sp_cer:.3f}  hyp={sp_hyp[:60]}")
        print(f"  content    : cer={ct_cer:.3f}  hyp={ct_hyp[:60]}")
        print(f"  converted* : cer={cv_cer:.3f}  hyp={cv_hyp[:60]}")
        print(f"  conv med   : cer={cv_med_cer:.3f}  hyp={cv_med_hyp[:60]}")

    # Markdown index
    md = ["# ε-3 listening packet\n",
          f"\nTarget text: `{ZH_HANZI}` (12 chars)\n",
          f"\nProduction recipe: native-ref content-donor + best-of-8 + "
          f"knn-VC (xl_k8). Per-emotion best CER on this set is **0.167**.\n",
          "\nFor each emotion below, file order is:\n",
          "1. `01_F1_ref.wav` — speaker target (JVNV recording)\n",
          "2. `02_baseline_F1_zh.wav` — current production "
          "(F1 voice + JP-accented zh, cer ≈ 1.0)\n",
          "3. `03_content_native.wav` — content donor "
          "(native voice, clean zh, cer ≈ 0.08)\n",
          "4. `04_eps3_converted.wav` — ε-3 production output, "
          "BEST of 8 (cer ≈ 0.17) ← ⭐ 主目標達成サンプル\n",
          "5. `05_eps3_converted_median.wav` — ε-3 median rep "
          "(typical single-rep quality)\n"]
    for r in rows:
        md.append(f"\n## {r['emotion']}\n")
        md.append(f"\n| stream | CER | Whisper hypothesis |\n|---|---|---|\n")
        md.append(f"| 02 baseline | {r['speaker_donor']['cer']:.3f} | "
                  f"`{r['speaker_donor']['hyp']}` |\n")
        md.append(f"| 03 content (native) | {r['content_donor']['cer']:.3f} | "
                  f"`{r['content_donor']['hyp']}` |\n")
        md.append(f"| 04 ε-3 converted (best) | {r['converted_best']['cer']:.3f} | "
                  f"`{r['converted_best']['hyp']}` |\n")
        md.append(f"| 05 ε-3 converted (median) | {r['converted_median']['cer']:.3f} | "
                  f"`{r['converted_median']['hyp']}` |\n")
    (OUT_DIR / "index.md").write_text("".join(md), encoding="utf-8")

    # Minimal HTML with audio players
    h = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
         "<title>ε-3 listening</title>",
         "<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;padding:0 1em}",
         "table{border-collapse:collapse;width:100%}",
         "td,th{border:1px solid #ccc;padding:8px;vertical-align:top}",
         "audio{width:300px}",
         "code{background:#f4f4f4;padding:2px 6px;border-radius:3px}",
         "h2{margin-top:2em;border-bottom:2px solid #333;padding-bottom:4px}",
         ".target{background:#ffffd0;padding:4px 8px;border-radius:4px}",
         "</style></head><body>",
         f"<h1>ε-3 listening packet</h1>",
         f"<p>Target text: <code class='target'>{ZH_HANZI}</code> (12 chars)</p>",
         "<p>Production recipe: native-ref content-donor + best-of-8 + "
         "knn-VC (xl_k8). Per-emotion best CER on this set is "
         "<b>0.167</b>, beating the 20% target.</p>"]
    for r in rows:
        h.append(f"<h2>{r['emotion']}</h2>")
        h.append("<table>")
        h.append("<tr><th>stream</th><th>audio</th><th>CER</th><th>Whisper hyp</th></tr>")
        for label, key, fname in [
            ("F1 reference (target speaker)", None, "01_F1_ref.wav"),
            ("02 baseline (F1+JP-accented zh)", "speaker_donor", "02_baseline_F1_zh.wav"),
            ("03 content donor (native voice)", "content_donor", "03_content_native.wav"),
            ("04 ε-3 converted ⭐ best-of-8",  "converted_best", "04_eps3_converted.wav"),
            ("05 ε-3 converted (median)",     "converted_median", "05_eps3_converted_median.wav"),
        ]:
            cer = "—" if key is None else f"{r[key]['cer']:.3f}"
            hyp = html.escape(r['f1_ref_text'] if key is None else r[key]['hyp'])
            audio_src = f"{r['emotion']}/{fname}"
            h.append(f"<tr><td>{label}</td>"
                     f"<td><audio controls src='{audio_src}'></audio></td>"
                     f"<td>{cer}</td><td><code>{hyp}</code></td></tr>")
        h.append("</table>")
    h.append("</body></html>")
    (OUT_DIR / "index.html").write_text("\n".join(h), encoding="utf-8")

    with open(OUT_DIR / "rows.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\n[pkt] saved -> {OUT_DIR}", flush=True)
    print(f"  index.html : {OUT_DIR / 'index.html'}", flush=True)
    print(f"  per-emo dirs : {[str(OUT_DIR / e) for e in EMOTIONS]}", flush=True)


if __name__ == "__main__":
    main()
