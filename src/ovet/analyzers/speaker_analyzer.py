"""Speaker embedding via SpeechBrain ECAPA-TDNN."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from ..types import SpeakerFeatures
from ..utils.io import load_wav

_DEFAULT_ID = "speechbrain/spkrec-ecapa-voxceleb"


class SpeakerAnalyzer:
    def __init__(self, model_id: str = _DEFAULT_ID, device: str = "cuda"):
        from speechbrain.inference.speaker import EncoderClassifier
        self.device = device if torch.cuda.is_available() else "cpu"
        savedir = f"/tmp/sb_{model_id.replace('/', '_')}"
        self._model = EncoderClassifier.from_hparams(
            source=model_id,
            savedir=savedir,
            run_opts={"device": self.device},
        )

    @torch.inference_mode()
    def analyze(self, wav_path: str | Path) -> SpeakerFeatures:
        wav, _sr = load_wav(wav_path, target_sr=16000)
        wav_t = torch.from_numpy(wav).unsqueeze(0).to(self.device)
        emb = self._model.encode_batch(wav_t).squeeze().cpu().numpy().astype(np.float32)
        return SpeakerFeatures(embedding=emb)
