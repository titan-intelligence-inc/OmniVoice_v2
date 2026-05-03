"""Probe candidate evaluators on the JVNV reference samples.

Targets:
1. emotion2vec_plus_base (existing)
2. emotion2vec_plus_large (second opinion)
3. audeering wav2vec2-large-robust-12-ft-emotion-msp-dim (V/A/D)
4. Prosodic features via librosa
"""
from __future__ import annotations
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

import torch
import numpy as np
import soundfile as sf
import librosa
import torch.nn as nn
from transformers import (
    AutoConfig, AutoFeatureExtractor,
    Wav2Vec2PreTrainedModel, Wav2Vec2Model,
)

JVNV = Path("/workspace/OmniVoice_v2/baseline/jvnv_samples")
EMOTIONS = ["anger", "sad", "happy", "fear", "surprise", "disgust"]


# === audeering V/A/D model (custom head, written in file context) ===
class _RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class _AudeeringEmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden = outputs[0]
        hidden = torch.mean(hidden, dim=1)
        return self.classifier(hidden)


def load_audeering():
    """Manually load audeering V/A/D model.

    Sidesteps transformers 5.x's hard requirements on PreTrainedModel
    by loading the wav2vec2 backbone via the standard API and applying
    the head weights manually.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    mid = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    fe = AutoFeatureExtractor.from_pretrained(mid)
    cfg = AutoConfig.from_pretrained(mid)

    backbone = Wav2Vec2Model.from_pretrained(mid).cuda().eval()
    head = _RegressionHead(cfg).cuda().eval()

    # Load the full checkpoint and pull the classifier weights into the head
    weights_path = hf_hub_download(repo_id=mid, filename="pytorch_model.bin")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    head_state = {
        "dense.weight":    state["classifier.dense.weight"],
        "dense.bias":      state["classifier.dense.bias"],
        "out_proj.weight": state["classifier.out_proj.weight"],
        "out_proj.bias":   state["classifier.out_proj.bias"],
    }
    head.load_state_dict(head_state)

    class _Combo:
        def __init__(self, backbone, head):
            self.backbone, self.head = backbone, head
        def __call__(self, input_values):
            with torch.no_grad():
                hidden = self.backbone(input_values)[0]
                hidden = torch.mean(hidden, dim=1)
                return self.head(hidden)

    return fe, _Combo(backbone, head)


def run_audeering(fe, model, wav, sr):
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
    inp = fe(wav, sampling_rate=16000, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(inp.input_values)[0].cpu().numpy()
    # Order per audeering README: arousal, dominance, valence (range 0..1 approximately)
    return {"arousal": float(out[0]), "dominance": float(out[1]), "valence": float(out[2])}


def emotion2vec_eval(em, p):
    out = em.generate(str(p), granularity="utterance", extract_embedding=True)[0]
    sc = {l.split("/")[-1]: float(s) for l, s in zip(out["labels"], out["scores"])}
    emb = np.asarray(out["feats"]).astype(np.float32).reshape(-1)
    return sc, emb


def prosody_features(wav, sr):
    if sr != 16000:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    f0 = librosa.yin(wav, fmin=50, fmax=500, sr=sr)
    voiced = f0[~np.isnan(f0) & (f0 > 50)]
    rms = librosa.feature.rms(y=wav)[0]
    return {
        "f0_mean":  float(np.mean(voiced)) if len(voiced) > 0 else 0.0,
        "f0_std":   float(np.std(voiced)) if len(voiced) > 0 else 0.0,
        "f0_range": float(np.percentile(voiced, 95) - np.percentile(voiced, 5)) if len(voiced) > 0 else 0.0,
        "energy_mean": float(np.mean(rms)),
        "energy_std":  float(np.std(rms)),
        "duration":    float(len(wav)/sr),
    }


if __name__ == "__main__":
    from funasr import AutoModel as FunAutoModel
    print("Loading emotion2vec_plus_base ...", flush=True)
    em_base = FunAutoModel(model="iic/emotion2vec_plus_base", hub="hf", disable_update=True)
    print("Loading emotion2vec_plus_large ...", flush=True)
    em_large = FunAutoModel(model="iic/emotion2vec_plus_large", hub="hf", disable_update=True)
    print("Loading audeering V/A/D ...", flush=True)
    aud_fe, aud_m = load_audeering()
    print("All loaded.", flush=True)

    print(f"\n{'sample':<10} {'em2v_base_top':<25} {'em2v_lg_top':<25} {'A':<6} {'D':<6} {'V':<6} {'F0std':<7} {'Estd':<6}")
    for emo in EMOTIONS:
        p = JVNV / f"jvnv_F1_{emo}.wav"
        wav, sr = sf.read(p)
        sc_b, _ = emotion2vec_eval(em_base, p)
        sc_l, _ = emotion2vec_eval(em_large, p)
        vad     = run_audeering(aud_fe, aud_m, wav, sr)
        pros    = prosody_features(wav, sr)
        top_b = max(sc_b.items(), key=lambda x:x[1])
        top_l = max(sc_l.items(), key=lambda x:x[1])
        print(f"{emo:<10} {top_b[0]+f'({top_b[1]:.2f})':<25} {top_l[0]+f'({top_l[1]:.2f})':<25} {vad['arousal']:.2f}   {vad['dominance']:.2f}   {vad['valence']:.2f}   {pros['f0_std']:.1f}   {pros['energy_std']:.3f}")
