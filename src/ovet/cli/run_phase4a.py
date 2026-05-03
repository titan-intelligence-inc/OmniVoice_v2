"""Phase 4a Geometric Validation Spike runner.

Pipeline:

1. For each (speaker, emotion) JVNV ref, generate one short clone in JP
   and one in EN — same content, same speaker, different language. The
   audio is OmniVoice-synthesized (zero-data).
2. Extract per-layer hidden means for each generated clip.
3. Per layer: linear probes for {language, emotion, speaker}, before
   and after projecting out the language direction.
4. Emit a Markdown report and the raw numbers as JSON.

The decision rule is summarized at the end of the report:

    PROJECTION REMOVAL IS JUSTIFIED  iff
        (lang_acc_pre  → lang_acc_post)  drops a lot
      AND
        (emo_acc_pre   → emo_acc_post)   stays roughly the same
"""
from __future__ import annotations
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..omnivoice.wrapper import OmniVoiceWrapper
from ..omnivoice.steering import extract_layer_vectors
from ..analysis.phase4a import analyze_layer
from ..utils.io import save_wav


SPEAKERS = ["F1", "F2", "M1", "M2"]
EMOTIONS = ["anger", "sad", "fear"]
LANGUAGES = [("Japanese", "今日の会議は午後三時から始まります。"),
             ("English",  "The meeting starts at three in the afternoon today.")]


def _parse_layers(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Phase 4a Geometric Validation Spike")
    ap.add_argument("--ref-dir",      default="baseline/jvnv_samples_multi", type=Path)
    ap.add_argument("--layers",       default="4,8,12,16,20,24")
    ap.add_argument("--output-dir",   default="outputs/phase4a_spike", type=Path)
    ap.add_argument("--num-step",     type=int, default=4)
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--hf-home",      default=os.environ.get("HF_HOME", "/workspace/hf_cache"))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", args.hf_home)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = _parse_layers(args.layers)

    print("[ovet] Loading OmniVoice ...", flush=True)
    w = OmniVoiceWrapper(hf_home=args.hf_home)

    # ------------------------------------------------------------------
    # Stage 1: synthesize JP+EN clones for every (speaker, emotion).
    # ------------------------------------------------------------------
    clones_dir = args.output_dir / "clones"
    clones_dir.mkdir(parents=True, exist_ok=True)

    samples = []   # list of (speaker, emotion, language_tag, language_full, wav_path)
    for sp in SPEAKERS:
        for emo in EMOTIONS:
            ref = args.ref_dir / f"jvnv_{sp}_{emo}.wav"
            if not ref.exists():
                print(f"[ovet] missing {ref}, skipping {sp}/{emo}", flush=True)
                continue
            ref_text = w.transcribe(ref, language=None)
            for lang_full, target_text in LANGUAGES:
                tag = lang_full[:2].lower()  # ja / en
                outp = clones_dir / f"{sp}_{emo}_{tag}.wav"
                if not outp.exists():
                    torch.manual_seed(args.seed)
                    audio = w.model.generate(
                        text=target_text, language=lang_full,
                        ref_audio=str(ref), ref_text=ref_text,
                        num_step=8,
                    )[0]
                    save_wav(outp, audio, w.SAMPLING_RATE)
                samples.append((sp, emo, tag, lang_full, outp, ref_text))
                print(f"[ovet] clone {sp}/{emo}/{tag}: {outp.name}", flush=True)

    print(f"[ovet] total clones: {len(samples)}", flush=True)

    # ------------------------------------------------------------------
    # Stage 2: extract per-layer hidden means for each clone.
    # ------------------------------------------------------------------
    H = {l: [] for l in layers}
    speakers, emotions, languages = [], [], []
    for (sp, emo, lang_tag, lang_full, wav, ref_text) in samples:
        # We use the clone itself as the "ref_audio" for the probe forward.
        # The clone embodies the (speaker, emotion, language) we want to
        # measure — same regime as production cloning.
        vecs = extract_layer_vectors(
            w, wav, layers, ref_text=None, num_step=args.num_step, seed=args.seed,
        )
        for l in layers:
            H[l].append(vecs[l])
        speakers.append(sp); emotions.append(emo); languages.append(lang_full)

    # Stack
    Hmat = {l: np.stack(H[l]).astype(np.float32) for l in layers}
    speakers, emotions, languages = list(speakers), list(emotions), list(languages)

    # ------------------------------------------------------------------
    # Stage 3: per-layer linear probes pre/post projection.
    # ------------------------------------------------------------------
    print("\n=== Phase 4a layer analyses ===", flush=True)
    print(f"{'layer':<6} {'lang_acc':<14} {'emo_acc':<14} {'spk_acc':<14} "
          f"{'|v_lang|':<10} {'cos(L,E)':<10} {'cos(L,S)':<10}",
          flush=True)
    layer_results = []
    for l in layers:
        ana = analyze_layer(
            hiddens=Hmat[l],
            speakers=speakers, emotions=emotions, languages=languages,
            layer_id=l,
        )
        layer_results.append(ana)
        print(
            f"{l:<6}",
            f"{ana.lang_acc_pre:.2f}→{ana.lang_acc_post:.2f}".ljust(14),
            f"{ana.emo_acc_pre:.2f}→{ana.emo_acc_post:.2f}".ljust(14),
            f"{ana.spk_acc_pre:.2f}→{ana.spk_acc_post:.2f}".ljust(14),
            f"{ana.v_lang_norm:.2f}".ljust(10),
            f"{ana.cos_lang_emo:.3f}".ljust(10),
            f"{ana.cos_lang_spk:.3f}".ljust(10),
            flush=True,
        )

    # ------------------------------------------------------------------
    # Stage 4: persist artifacts and write report
    # ------------------------------------------------------------------
    payload = {
        "samples": [
            {"speaker": s, "emotion": e, "language": lf, "wav": str(wav)}
            for (s, e, _, lf, wav, _) in samples
        ],
        "layers":  layers,
        "layer_results": [r.as_dict() for r in layer_results],
        "config":  {"seed": args.seed, "num_step": args.num_step},
    }
    with open(args.output_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # Markdown summary
    md = ["# Phase 4a Geometric Validation Spike\n"]
    md.append(f"- Samples: **{len(samples)}** clones "
              f"({len(set(speakers))} speakers × {len(set(emotions))} emotions × "
              f"{len(set(languages))} languages)\n")
    md.append(f"- Languages: {', '.join(sorted(set(languages)))}\n")
    md.append(f"- Speakers:  {', '.join(sorted(set(speakers)))}\n")
    md.append(f"- Emotions:  {', '.join(sorted(set(emotions)))}\n")
    md.append("\n## Layer-wise linear probe (LOO accuracy)\n")
    md.append("| layer | lang pre→post | emo pre→post | spk pre→post | |v_lang| | cos(L,E) | cos(L,S) |")
    md.append("|---|---|---|---|---|---|---|")
    for r in layer_results:
        md.append("| " + " | ".join([
            str(r.layer_id),
            f"{r.lang_acc_pre:.2f} → {r.lang_acc_post:.2f}",
            f"{r.emo_acc_pre:.2f} → {r.emo_acc_post:.2f}",
            f"{r.spk_acc_pre:.2f} → {r.spk_acc_post:.2f}",
            f"{r.v_lang_norm:.2f}",
            f"{r.cos_lang_emo:.3f}",
            f"{r.cos_lang_spk:.3f}",
        ]) + " |")

    md.append("\n## Decision rule\n")
    md.append("**Phase 4b (language projection removal) is justified iff:**\n")
    md.append("- `lang_acc_post` drops well below `lang_acc_pre` (language information is captured by v_lang)\n")
    md.append("- `emo_acc_post` is close to `emo_acc_pre` (emotion is preserved after projection)\n")

    # Auto verdict — call it green if at least one layer satisfies both
    green_layers = [
        r.layer_id for r in layer_results
        if (r.lang_acc_pre - r.lang_acc_post) >= 0.20
        and (r.emo_acc_pre  - r.emo_acc_post) <= 0.15
    ]
    md.append("\n**Verdict:** ")
    if green_layers:
        md.append(f"✅ **GO** — Phase 4b justified. Layers satisfying the rule: "
                  f"{green_layers}\n")
    else:
        md.append("⚠️ **MARGINAL / NO-GO** — projection removal does not cleanly preserve emotion\n"
                  "while removing language. Consider sticking with Phase 3 alone, "
                  "or refining v_lang construction (multi-speaker average, alternative bases).\n")

    (args.output_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[ovet] Saved: {args.output_dir}/{{result.json, report.md, clones/}}", flush=True)


if __name__ == "__main__":
    main()
