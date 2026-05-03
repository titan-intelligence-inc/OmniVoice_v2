"""emotion2vec-based emotion classification + embedding extraction.

Uses ``iic/emotion2vec_plus_base`` via funasr. The _large variant
performed worse on JVNV F1 in baseline v3, so we stick with _base.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from ..types import EmotionAnalysis


class EmotionAnalyzer:
    """Wrapper around emotion2vec_plus_base."""

    def __init__(self, model_id: str = "iic/emotion2vec_plus_base"):
        from funasr import AutoModel
        self._model = AutoModel(model=model_id, hub="hf", disable_update=True)

    def analyze(self, wav_path: str | Path) -> EmotionAnalysis:
        out = self._model.generate(
            str(wav_path), granularity="utterance", extract_embedding=True
        )[0]
        labels = out["labels"]
        scores = out["scores"]
        # emotion2vec returns labels like "生气/angry" — keep the english side.
        logits = {l.split("/")[-1]: float(s) for l, s in zip(labels, scores)}
        top_label, top_p = max(logits.items(), key=lambda x: x[1])
        emb = np.asarray(out["feats"]).astype(np.float32).reshape(-1)
        return EmotionAnalysis(
            label=top_label,
            confidence=float(top_p),
            embedding=emb,
            logits=logits,
        )
