"""Baseline v3 with hardened evaluators.

Three-axis evaluation:
  (1) emotion2vec_plus_base — label probability + embedding cosine
  (2) audeering V/A/D       — language-agnostic dimensional emotion
  (3) prosodic features     — F0/energy variance, duration

Target: quantify cross-lingual emotion attenuation with reliable metrics
across JP/EN/ZH outputs.
"""
from __future__ import annotations
import os, json, sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
sys.path.insert(0, str(Path(__file__).parent))

import torch, numpy as np, soundfile as sf
from omnivoice import OmniVoice
from funasr import AutoModel as FunAutoModel

from probe_evaluators import (
    load_audeering, run_audeering, prosody_features,
)

ROOT = Path("/workspace/OmniVoice_v2")
JVNV = ROOT / "baseline/jvnv_samples"
OUT  = ROOT / "baseline/outputs_v3"
OUT.mkdir(parents=True, exist_ok=True)

REF_TEXTS = {
    "anger":    "うわぁ何度も言わなきゃわからないのこの機械を潰してしまったら自分の首を絞めることになるんだよ",
    "sad":      "はあ、人生の意味がわからなくなって、私はただ呆然としたままです。",
    "happy":    "この小説は奇妙な出来事が次々に起こってワクワク感が止まらないおお",
    "fear":     "あの森に入るのは怖い。そこに住む奇妙な生き物がいると聞いたからだ。え!?",
    "surprise": "あああの試合で彼のサブスティテューションが決まっていたなんてまさかの展開でしたね",
    "disgust":  "くぅ、不要意な言動で人を傷つけるなんて最低だと思わないか?",
}

# Two semantically neutral sentences per language
TARGETS = [
    ("ja_a", "今日の会議は午後三時から始まります。",                   "Japanese"),
    ("ja_b", "明日の予定を確認しておきましょう。",                     "Japanese"),
    ("en_a", "The meeting starts at three in the afternoon today.",   "English"),
    ("en_b", "Let us go over the schedule for tomorrow.",             "English"),
    ("zh_a", "今天的会议下午三点开始。",                              "Chinese"),
    ("zh_b", "我们来确认一下明天的安排。",                            "Chinese"),
]

# Only emotions whose ref is reliably classified by emotion2vec_base
EMOTIONS = ["anger", "sad", "fear"]
LABEL_MAP = {"anger":"angry","sad":"sad","fear":"fearful",
             "happy":"happy","surprise":"surprised","disgust":"disgusted"}


def emotion2vec_eval(em, p):
    out = em.generate(str(p), granularity="utterance", extract_embedding=True)[0]
    sc = {l.split("/")[-1]: float(s) for l, s in zip(out["labels"], out["scores"])}
    emb = np.asarray(out["feats"]).astype(np.float32).reshape(-1)
    return sc, emb


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def evaluate(p, em, aud_fe, aud_m):
    sc, emb = emotion2vec_eval(em, p)
    wav, sr = sf.read(p)
    vad = run_audeering(aud_fe, aud_m, wav, sr)
    pros = prosody_features(wav, sr)
    return {"e2v_scores": sc, "e2v_emb": emb, "vad": vad, "pros": pros}


print("Loading OmniVoice ...", flush=True)
ov = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
print("Loading emotion2vec_plus_base ...", flush=True)
em = FunAutoModel(model="iic/emotion2vec_plus_base", hub="hf", disable_update=True)
print("Loading audeering V/A/D ...", flush=True)
aud_fe, aud_m = load_audeering()
print("All loaded.", flush=True)

results = []
for emo in EMOTIONS:
    ref = JVNV / f"jvnv_F1_{emo}.wav"
    ref_eval = evaluate(ref, em, aud_fe, aud_m)
    target_label = LABEL_MAP[emo]
    print(f"\n=== {emo} ref ===", flush=True)
    print(f"  e2v p({target_label})={ref_eval['e2v_scores'].get(target_label,0):.3f}  "
          f"V/A/D=({ref_eval['vad']['valence']:.2f}, {ref_eval['vad']['arousal']:.2f}, {ref_eval['vad']['dominance']:.2f})  "
          f"F0_std={ref_eval['pros']['f0_std']:.1f}  E_std={ref_eval['pros']['energy_std']:.3f}",
          flush=True)

    for tag, text, lang in TARGETS:
        try:
            audio = ov.generate(text=text, language=lang,
                                ref_audio=str(ref), ref_text=REF_TEXTS[emo])
            outp = OUT / f"{emo}__{tag}.wav"
            sf.write(outp, audio[0], 24000)
            ev = evaluate(outp, em, aud_fe, aud_m)
            entry = {
                "emo": emo, "tag": tag, "lang_tag": tag.split("_")[0],
                "ref_text": REF_TEXTS[emo], "tgt_text": text,
                "out": str(outp),
                "e2v_p_target":   ev["e2v_scores"].get(target_label, 0),
                "e2v_top":        max(ev["e2v_scores"].items(), key=lambda x:x[1])[0],
                "e2v_top_p":      max(ev["e2v_scores"].values()),
                "e2v_cos_to_ref": cos(ev["e2v_emb"], ref_eval["e2v_emb"]),
                "vad":            ev["vad"],
                "vad_dist_to_ref":{
                    "arousal_diff":   abs(ev["vad"]["arousal"]   - ref_eval["vad"]["arousal"]),
                    "valence_diff":   abs(ev["vad"]["valence"]   - ref_eval["vad"]["valence"]),
                    "dominance_diff": abs(ev["vad"]["dominance"] - ref_eval["vad"]["dominance"]),
                },
                "pros":           ev["pros"],
                "pros_ratio_to_ref":{
                    "f0_std_ratio":     ev["pros"]["f0_std"]     / max(ref_eval["pros"]["f0_std"],     1e-6),
                    "energy_std_ratio": ev["pros"]["energy_std"] / max(ref_eval["pros"]["energy_std"], 1e-6),
                },
            }
            results.append(entry)
            print(f"  [{tag}] e2v_p={entry['e2v_p_target']:.3f}({entry['e2v_top']})  "
                  f"cos={entry['e2v_cos_to_ref']:.3f}  "
                  f"V={ev['vad']['valence']:.2f} A={ev['vad']['arousal']:.2f} D={ev['vad']['dominance']:.2f}  "
                  f"F0r={entry['pros_ratio_to_ref']['f0_std_ratio']:.2f}  "
                  f"Er={entry['pros_ratio_to_ref']['energy_std_ratio']:.2f}",
                  flush=True)
        except Exception as e:
            print(f"  [{tag}] ERROR: {e}", flush=True)
            results.append({"emo":emo, "tag":tag, "error":str(e)})

with open(OUT/"results.json", "w") as f:
    # Drop e2v_emb (too large) before saving
    json.dump(results, f, indent=2, ensure_ascii=False)

# Summary tables
print("\n\n=== TABLE 1: emotion2vec p(target) ===")
print(f"{'emo':<8} {'ja_a':<14} {'ja_b':<14} {'en_a':<14} {'en_b':<14} {'zh_a':<14} {'zh_b':<14}")
for emo in EMOTIONS:
    row = []
    for tag,_,_ in TARGETS:
        m = next((r for r in results if r.get('emo')==emo and r.get('tag')==tag and 'e2v_p_target' in r), None)
        row.append(f"{m['e2v_p_target']:.2f}({m['e2v_top'][:4]})" if m else "ERR")
    print(f"{emo:<8}", " ".join(f"{c:<14}" for c in row))

print("\n=== TABLE 2: V/A/D distance to ref (lower=better) ===")
print(f"{'emo':<8} {'metric':<10} {'ja_a':<8} {'ja_b':<8} {'en_a':<8} {'en_b':<8} {'zh_a':<8} {'zh_b':<8}")
for emo in EMOTIONS:
    for met in ['arousal_diff','valence_diff','dominance_diff']:
        row = []
        for tag,_,_ in TARGETS:
            m = next((r for r in results if r.get('emo')==emo and r.get('tag')==tag and 'vad_dist_to_ref' in r), None)
            row.append(f"{m['vad_dist_to_ref'][met]:.2f}" if m else "ERR")
        print(f"{emo:<8} {met:<10}", " ".join(f"{c:<8}" for c in row))

print("\n=== TABLE 3: prosodic ratio to ref (1.0=identical) ===")
print(f"{'emo':<8} {'metric':<10} {'ja_a':<8} {'ja_b':<8} {'en_a':<8} {'en_b':<8} {'zh_a':<8} {'zh_b':<8}")
for emo in EMOTIONS:
    for met in ['f0_std_ratio','energy_std_ratio']:
        row = []
        for tag,_,_ in TARGETS:
            m = next((r for r in results if r.get('emo')==emo and r.get('tag')==tag and 'pros_ratio_to_ref' in r), None)
            row.append(f"{m['pros_ratio_to_ref'][met]:.2f}" if m else "ERR")
        print(f"{emo:<8} {met:<10}", " ".join(f"{c:<8}" for c in row))

print("\n=== TABLE 4: cos(emotion2vec embedding to ref) ===")
print(f"{'emo':<8} {'ja_a':<8} {'ja_b':<8} {'en_a':<8} {'en_b':<8} {'zh_a':<8} {'zh_b':<8}")
for emo in EMOTIONS:
    row = []
    for tag,_,_ in TARGETS:
        m = next((r for r in results if r.get('emo')==emo and r.get('tag')==tag and 'e2v_cos_to_ref' in r), None)
        row.append(f"{m['e2v_cos_to_ref']:.2f}" if m else "ERR")
    print(f"{emo:<8}", " ".join(f"{c:<8}" for c in row))

print(f"\nResults: {OUT/'results.json'}")
