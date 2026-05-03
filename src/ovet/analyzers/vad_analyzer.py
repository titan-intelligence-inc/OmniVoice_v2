"""Dimensional emotion (Valence / Arousal / Dominance) via audeering.

Uses ``audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim``. The
checkpoint has a custom regression head that doesn't fit transformers
5.x's ``PreTrainedModel`` subclass discovery, so we load the backbone
via the standard API and apply the head weights manually (validated in
``scripts/probe_evaluators.py``).

Output ranges: roughly 0..1 for each axis.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from ..types import VADFeatures
from ..utils.io import load_wav

_AUD_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


class _RegressionHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, final_dropout: float):
        super().__init__()
        self.dense    = nn.Linear(hidden_size, hidden_size)
        self.dropout  = nn.Dropout(final_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class VADAnalyzer:
    def __init__(self, device: str = "cuda", model_id: str = _AUD_MODEL_ID):
        from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2Model
        from huggingface_hub import hf_hub_download

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.fe  = AutoFeatureExtractor.from_pretrained(model_id)
        cfg      = AutoConfig.from_pretrained(model_id)
        self.backbone = Wav2Vec2Model.from_pretrained(model_id).to(self.device).eval()
        self.head     = _RegressionHead(
            hidden_size=cfg.hidden_size,
            num_labels=cfg.num_labels,
            final_dropout=cfg.final_dropout,
        ).to(self.device).eval()

        ckpt_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        state     = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.head.load_state_dict({
            "dense.weight":    state["classifier.dense.weight"],
            "dense.bias":      state["classifier.dense.bias"],
            "out_proj.weight": state["classifier.out_proj.weight"],
            "out_proj.bias":   state["classifier.out_proj.bias"],
        })

    @torch.inference_mode()
    def analyze(self, wav_path: str | Path) -> VADFeatures:
        wav, _ = load_wav(wav_path, target_sr=16000)
        inp = self.fe(wav, sampling_rate=16000, return_tensors="pt").to(self.device)
        hidden = self.backbone(inp.input_values)[0]
        hidden = torch.mean(hidden, dim=1)
        out = self.head(hidden)[0].cpu().numpy()
        # audeering README: order is arousal, dominance, valence
        return VADFeatures(
            arousal=float(out[0]),
            dominance=float(out[1]),
            valence=float(out[2]),
        )
