"""Phase 1 batch: run the new pipeline across the JVNV baseline matrix.

For each (emotion, target_language) pair, generate baseline + proxy
candidates and dump per-pair score deltas. Output mirrors baseline v3
so we can compare directly.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from dataclasses import asdict

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

ROOT = Path("/workspace/OmniVoice_v2")
sys.path.insert(0, str(ROOT / "src"))

from ovet.types import GenerationRequest, Thresholds, ScoringWeights, Candidate
from ovet.prompts.instruct_proxy import InstructProxyComposer
from ovet.omnivoice.wrapper import OmniVoiceWrapper
from ovet.analyzers.emotion_analyzer import EmotionAnalyzer
from ovet.analyzers.vad_analyzer import VADAnalyzer
from ovet.analyzers.prosody_analyzer import ProsodyAnalyzer
from ovet.analyzers.speaker_analyzer import SpeakerAnalyzer
from ovet.analyzers.asr_analyzer import ASRAnalyzer
from ovet.evaluation.evaluator import CandidateEvaluator
from ovet.evaluation.scoring import compute_total_score
from ovet.generation.candidate_generator import CandidateGenerator, CandidateSpec


REF_DIR = ROOT / "baseline/jvnv_samples"
OUT     = ROOT / "outputs/phase1_batch"
OUT.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["anger", "sad", "fear"]
TARGETS = [
    ("ja", "今日の会議は午後三時から始まります。", "Japanese"),
    ("en", "The meeting starts at three in the afternoon today.", "English"),
]


def main():
    print("[batch] Loading analyzers ...", flush=True)
    emo_an  = EmotionAnalyzer()
    vad_an  = VADAnalyzer()
    pros_an = ProsodyAnalyzer()
    spk_an  = SpeakerAnalyzer()
    asr_an  = ASRAnalyzer()
    proxy   = InstructProxyComposer()
    wrapper = OmniVoiceWrapper()
    gen     = CandidateGenerator(wrapper)
    ev      = CandidateEvaluator(emo_an, vad_an, pros_an, spk_an, asr_an)
    weights = ScoringWeights()

    rows = []
    for emo in EMOTIONS:
        ref_path = REF_DIR / f"jvnv_F1_{emo}.wav"
        ref = ev.analyze_reference(ref_path)
        ref_text = wrapper.transcribe(ref_path, language=None)
        proxy_str = proxy.compose(emo)
        for tag, text, lang in TARGETS:
            run_dir = OUT / f"{emo}__{tag}"
            run_dir.mkdir(parents=True, exist_ok=True)
            req = GenerationRequest(
                text=text, language=lang, ref_audio=ref_path,
                ref_text=ref_text, output_dir=run_dir,
                emotion_label_hint=emo,
            )
            specs = [
                CandidateSpec(instruct=None,       tag="baseline"),
                CandidateSpec(instruct=proxy_str,  tag="proxy"),
            ]
            generated = gen.generate(req, specs, ref_text=ref_text)
            for g in generated:
                sc = ev.evaluate(g.wav_path, ref, target_text=text,
                                 target_language=lang, target_emotion_label=emo)
                total = compute_total_score(sc, weights)
                rows.append({
                    "emo": emo, "tag": tag, "lang": lang, "variant": g.spec.tag,
                    "instruct": g.spec.instruct,
                    "scores": asdict(sc), "total": total,
                })
                print(f"[{emo}|{tag}|{g.spec.tag:<8}] vad={sc.vad_dist:.3f}  "
                      f"V_diff={sc.valence_diff:.3f}  e2v_cos={sc.e2v_cos:.3f}  "
                      f"E_ratio={sc.energy_std_ratio:.3f}  spk={sc.speaker_sim:.3f}  "
                      f"score={total:.3f}", flush=True)

    with open(OUT / "summary.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    # Print compact table comparing baseline vs proxy
    print("\n=== Baseline vs Proxy delta (proxy − baseline) ===")
    print(f"{'emo':<8} {'tag':<5} {'Δvad_dist':<11} {'Δvalence_diff':<14} {'ΔE_ratio→1':<13} {'Δspk_sim':<10}")
    by_key = {(r["emo"], r["tag"], r["variant"]): r for r in rows}
    for emo in EMOTIONS:
        for tag, _, _ in TARGETS:
            b = by_key.get((emo, tag, "baseline"))
            p = by_key.get((emo, tag, "proxy"))
            if not (b and p):
                continue
            d_vad = p["scores"]["vad_dist"] - b["scores"]["vad_dist"]
            d_val = p["scores"]["valence_diff"] - b["scores"]["valence_diff"]
            d_e   = abs(1 - p["scores"]["energy_std_ratio"]) - abs(1 - b["scores"]["energy_std_ratio"])
            d_spk = p["scores"]["speaker_sim"] - b["scores"]["speaker_sim"]
            print(f"{emo:<8} {tag:<5} {d_vad:+.3f}      {d_val:+.3f}         {d_e:+.3f}        {d_spk:+.3f}")
    print(f"\nResults: {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
