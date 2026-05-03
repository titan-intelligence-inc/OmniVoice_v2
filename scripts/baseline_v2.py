"""Baseline v2: control for target text semantics.

Use semantically neutral target texts to isolate prosodic emotion transfer
from textual emotion bias.
"""
from __future__ import annotations
import os, json
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import torch, numpy as np, soundfile as sf
from omnivoice import OmniVoice
from funasr import AutoModel as FunAutoModel

ROOT = Path("/workspace/OmniVoice_v2")
JVNV = ROOT / "baseline/jvnv_samples"
OUT  = ROOT / "baseline/outputs_v2"
OUT.mkdir(parents=True, exist_ok=True)

REF_TEXTS = {
    "anger":    "うわぁ何度も言わなきゃわからないのこの機械を潰してしまったら自分の首を絞めることになるんだよ",
    "sad":      "はあ、人生の意味がわからなくなって、私はただ呆然としたままです。",
    "happy":    "この小説は奇妙な出来事が次々に起こってワクワク感が止まらないおお",
    "fear":     "あの森に入るのは怖い。そこに住む奇妙な生き物がいると聞いたからだ。え!?",
    "surprise": "あああの試合で彼のサブスティテューションが決まっていたなんてまさかの展開でしたね",
    "disgust":  "くぅ、不要意な言動で人を傷つけるなんて最低だと思わないか?",
}

# Semantically neutral target texts
TARGETS = [
    ("ja_neutral_a", "今日の会議は午後三時から始まります。", "Japanese"),
    ("ja_neutral_b", "明日の予定を確認しておきましょう。",  "Japanese"),
    ("en_neutral_a", "The meeting starts at three in the afternoon today.", "English"),
    ("en_neutral_b", "Let us go over the schedule for tomorrow.", "English"),
    ("zh_neutral_a", "今天的会议下午三点开始。", "Chinese"),
    ("zh_neutral_b", "我们来确认一下明天的安排。", "Chinese"),
]

EMOTIONS = ["anger", "sad", "happy", "fear", "surprise", "disgust"]
LABEL_MAP = {"anger":"angry","sad":"sad","fear":"fearful","happy":"happy","surprise":"surprised","disgust":"disgusted"}

print("Loading OmniVoice ...", flush=True)
ov = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
print("Loading emotion2vec ...", flush=True)
em = FunAutoModel(model="iic/emotion2vec_plus_base", hub="hf", disable_update=True)

def emo_eval(p):
    out = em.generate(str(p), granularity="utterance", extract_embedding=True)[0]
    sc = {l.split("/")[-1]: float(s) for l, s in zip(out["labels"], out["scores"])}
    emb = np.asarray(out["feats"]).astype(np.float32).reshape(-1)
    return sc, emb

results = []
for emo in EMOTIONS:
    ref = JVNV / f"jvnv_F1_{emo}.wav"
    ref_sc, ref_emb = emo_eval(ref)
    target_label = LABEL_MAP[emo]
    print(f"\n=== {emo} (ref P={ref_sc.get(target_label,0):.3f}) ===", flush=True)
    for tag, text, lang in TARGETS:
        try:
            audio = ov.generate(text=text, language=lang,
                                 ref_audio=str(ref), ref_text=REF_TEXTS[emo])
            outp = OUT / f"{emo}__{tag}.wav"
            sf.write(outp, audio[0], 24000)
            sc, emb = emo_eval(outp)
            cos = float(np.dot(ref_emb, emb)/(np.linalg.norm(ref_emb)*np.linalg.norm(emb)+1e-9))
            top = max(sc.items(), key=lambda x:x[1])
            print(f"  {tag:<14} p({target_label})={sc.get(target_label,0):.3f}  cos={cos:.3f}  top={top[0]}({top[1]:.2f})", flush=True)
            results.append({"emo":emo,"tag":tag,"lang":lang,"p_target":sc.get(target_label,0),
                            "cos":cos,"top":top[0],"top_p":top[1],"scores":sc})
        except Exception as e:
            print(f"  {tag} ERR: {e}", flush=True)
            results.append({"emo":emo,"tag":tag,"error":str(e)})

with open(OUT/"results.json","w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n=== SUMMARY: p(target) per emotion×lang_tag ===")
emos_sorted = sorted({r['emo'] for r in results if 'p_target' in r})
tags_sorted = [t for t,_,_ in TARGETS]
print(f"{'emo':<10}", *[f"{t:<16}" for t in tags_sorted])
for emo in emos_sorted:
    row = []
    for tag in tags_sorted:
        m = next((r for r in results if r.get('emo')==emo and r.get('tag')==tag), None)
        row.append(f"{m['p_target']:.3f}({m['top'][:4]})" if m and 'p_target' in m else "ERR")
    print(f"{emo:<10}", *[f"{c:<16}" for c in row])

print(f"\nFiles: {OUT}")
