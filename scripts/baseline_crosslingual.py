"""Baseline experiment: measure cross-lingual emotion attenuation.

For each (emotion, ref_audio) pair:
  - Clone in JP (control) — same language as ref
  - Clone in EN (test)    — cross-lingual
  - Measure emotion2vec scores on each output

Hypothesis: cross-lingual outputs show weaker emotion signal than monolingual.
"""
from __future__ import annotations
import os, json, traceback
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import torch
import numpy as np
import soundfile as sf
from omnivoice import OmniVoice
from funasr import AutoModel as FunAutoModel

ROOT = Path("/workspace/OmniVoice_v2")
JVNV = ROOT / "baseline/jvnv_samples"
OUT  = ROOT / "baseline/outputs"
OUT.mkdir(parents=True, exist_ok=True)

EMOTIONS = ["anger", "sad", "fear"]

# Reference transcripts (Whisper-transcribed from JVNV samples)
REF_TEXTS = {
    "anger":    "うわぁ何度も言わなきゃわからないのこの機械を潰してしまったら自分の首を絞めることになるんだよ",
    "sad":      "はあ、人生の意味がわからなくなって、私はただ呆然としたままです。",
    "happy":    "この小説は奇妙な出来事が次々に起こってワクワク感が止まらないおお",
    "fear":     "あの森に入るのは怖い。そこに住む奇妙な生き物がいると聞いたからだ。え!?",
    "surprise": "あああの試合で彼のサブスティテューションが決まっていたなんてまさかの展開でしたね",
    "disgust":  "くぅ、不要意な言動で人を傷つけるなんて最低だと思わないか?",
}

# Target generation texts (neutral content - emotion should come from ref)
TARGET_TEXT_JP = "今日は来てくれて、本当にありがとう。"
TARGET_TEXT_EN = "Thank you so much for coming today."

print("Loading OmniVoice ...", flush=True)
ov = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16,
)
print("Loading emotion2vec...", flush=True)
em = FunAutoModel(model="iic/emotion2vec_plus_base", hub="hf", disable_update=True)

def emotion_scores(wav_path):
    out = em.generate(str(wav_path), granularity="utterance", extract_embedding=True)[0]
    labels = out["labels"]
    scores = out["scores"]
    emb = out.get("feats", None)
    score_map = {l.split("/")[-1]: float(s) for l, s in zip(labels, scores)}
    return score_map, emb

def emo2vec_emb(wav_path):
    out = em.generate(str(wav_path), granularity="utterance", extract_embedding=True)[0]
    return np.asarray(out["feats"]).astype(np.float32).reshape(-1)

LABEL_MAP = {  # JVNV→emotion2vec class
    "anger":   "angry",
    "sad":     "sad",
    "fear":    "fearful",
    "happy":   "happy",
    "surprise":"surprised",
    "disgust": "disgusted",
}

results = []
for emo in EMOTIONS:
    ref = JVNV / f"jvnv_F1_{emo}.wav"
    print(f"\n=== {emo}: ref={ref.name} ===", flush=True)

    # Reference emotion signal & embedding (gold)
    ref_scores, _ = emotion_scores(ref)
    ref_emb = emo2vec_emb(ref)
    target_label = LABEL_MAP[emo]
    print(f"REF  scores: {ref_scores}", flush=True)

    for lang_tag, text, lang in [("ja", TARGET_TEXT_JP, "Japanese"),
                                  ("en", TARGET_TEXT_EN, "English")]:
        try:
            audio = ov.generate(
                text=text, language=lang,
                ref_audio=str(ref), ref_text=REF_TEXTS[emo],
            )
            outp = OUT / f"{emo}__{lang_tag}.wav"
            sf.write(outp, audio[0], 24000)

            sc, _ = emotion_scores(outp)
            emb = emo2vec_emb(outp)
            cos = float(np.dot(ref_emb, emb) / (np.linalg.norm(ref_emb)*np.linalg.norm(emb) + 1e-9))
            print(f"  [{lang_tag}] target_p({target_label})={sc.get(target_label, 0):.3f}  cos(ref)={cos:.3f}  top={max(sc.items(), key=lambda x:x[1])}", flush=True)
            results.append({
                "emotion": emo,
                "lang": lang_tag,
                "ref": str(ref),
                "out": str(outp),
                "ref_p_target": ref_scores.get(target_label, 0),
                "out_p_target": sc.get(target_label, 0),
                "out_top": max(sc.items(), key=lambda x:x[1])[0],
                "out_top_p": max(sc.values()),
                "cos_to_ref": cos,
                "out_scores": sc,
            })
        except Exception as e:
            traceback.print_exc()
            results.append({"emotion": emo, "lang": lang_tag, "error": str(e)})

with open(OUT / "baseline_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n=== SUMMARY ===")
print(f"{'emo':<8} {'lang':<5} {'p(target)':<12} {'cos(ref)':<10} {'top':<12}")
for r in results:
    if "error" in r: continue
    print(f"{r['emotion']:<8} {r['lang']:<5} {r['out_p_target']:.3f}        {r['cos_to_ref']:.3f}      {r['out_top']}")
print(f"\nResults saved: {OUT/'baseline_results.json'}")
