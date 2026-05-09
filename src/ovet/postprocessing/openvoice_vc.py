"""OpenVoice v2 ToneColorConverter wrapper for Phase 6 post-processing.

Use case: feed OmniVoice output (F1 voice + JP-flavored target language)
through ToneColorConverter to attempt accent reduction. ToneColorConverter
is designed for timbre swap, not accent removal — but its flow model
operates on multilingual data and may regularize non-native articulation
as a side effect. We measure empirically.

Three configurations of interest:

  identity_pass   src_se = tgt_se = F1   (just passes through the flow)
  F1_to_native    src_se = F1, tgt_se = native target-lang base speaker
  native_to_F1    src_se = native, tgt_se = F1  (rare, but completes the matrix)

OmniVoice outputs are 24 kHz. ToneColorConverter v2 operates at 22.05 kHz.
We resample on the way in and on the way back out to keep downstream
metrics consistent with the rest of the pipeline.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import librosa
import soundfile as sf

# OpenVoice was installed in editable mode so this import works.
from openvoice.api import ToneColorConverter                # type: ignore


# Default checkpoint layout matching the snapshot we downloaded into
# upstream/OpenVoice_checkpoints/v2/.
_DEFAULT_V2_ROOT = Path("upstream/OpenVoice_checkpoints/v2")


@dataclass
class VCConfig:
    """Phase 6 post-processing knobs.

    enabled               toggle the whole post-VC pipeline
    mode                  which (src_se, tgt_se) configuration to use
    target_language_se    pre-extracted base-speaker SE filename in
                          ``base_speakers/ses/`` (e.g. ``zh.pth``).
                          Required when mode involves "native".
    speaker_ref_audio     audio file used to extract the reference
                          speaker's SE (e.g. JVNV F1 anger). Required
                          for any mode that touches "F1".
    tau                   ToneColorConverter sampling temperature.
                          Lower = more conservative timbre swap. Default
                          0.3 matches OpenVoice demo notebook.
    """
    enabled: bool = False
    mode: Literal["identity_pass", "F1_to_native", "native_to_F1"] = "F1_to_native"
    target_language_se: str = "zh.pth"
    speaker_ref_audio: Path | None = None
    tau: float = 0.3


class OpenVoicePostVC:
    """Lazy-loaded ToneColorConverter wrapper.

    Holds:
      * the converter model (loaded once)
      * a cached SE for ``speaker_ref_audio`` (extracted on first use)
      * pre-extracted target language SEs (loaded from base_speakers/ses)
    """

    SAMPLING_RATE = 22050    # ToneColorConverter v2 native rate

    def __init__(
        self,
        ckpt_root: Path = _DEFAULT_V2_ROOT,
        device: str = "cuda:0",
        enable_watermark: bool = False,
    ):
        ckpt_root = Path(ckpt_root)
        config_path = ckpt_root / "converter" / "config.json"
        ckpt_path   = ckpt_root / "converter" / "checkpoint.pth"
        if not config_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(
                f"OpenVoice v2 converter not found at {ckpt_root}. "
                f"Expected: {config_path} and {ckpt_path}."
            )
        # ToneColorConverter signature differs slightly between v1 and
        # v2 — older constructor accepts ``enable_watermark`` kwarg,
        # newer one doesn't. Handle both.
        try:
            self.converter = ToneColorConverter(
                str(config_path), device=device,
                enable_watermark=enable_watermark,
            )
        except TypeError:
            self.converter = ToneColorConverter(
                str(config_path), device=device,
            )
        self.converter.load_ckpt(str(ckpt_path))
        self.ckpt_root = ckpt_root
        self.device = device
        self._spk_ref_se_cache: dict[str, torch.Tensor] = {}

    def _get_ref_se(self, ref_audio: Path) -> torch.Tensor:
        """Extract & cache speaker SE for a reference audio file."""
        key = str(Path(ref_audio).resolve())
        if key not in self._spk_ref_se_cache:
            self._spk_ref_se_cache[key] = self.converter.extract_se([str(ref_audio)])
        return self._spk_ref_se_cache[key]

    def _get_lang_se(self, name: str) -> torch.Tensor:
        """Load pre-extracted base-speaker SE (e.g. zh.pth)."""
        path = self.ckpt_root / "base_speakers" / "ses" / name
        if not path.exists():
            available = list((self.ckpt_root / "base_speakers" / "ses").glob("*.pth"))
            raise FileNotFoundError(
                f"Base speaker SE not found: {path}. Available: "
                f"{[p.name for p in available]}"
            )
        return torch.load(path, map_location=self.device)

    def convert(
        self,
        audio_24khz: np.ndarray,
        cfg: VCConfig,
        out_path: Path | None = None,
    ) -> np.ndarray:
        """Run ToneColorConverter on a 24kHz mono numpy waveform.

        Returns the converted audio resampled back to 24 kHz so it can
        flow through the rest of the OmniVoice evaluation pipeline
        unchanged.
        """
        if not cfg.enabled:
            return audio_24khz

        # Resample 24 kHz -> 22.05 kHz, write to temp file (extract_se /
        # convert both expect on-disk paths internally).
        tmp_in = Path("/tmp") / f"_postvc_in_{id(audio_24khz)}.wav"
        try:
            audio_22k = librosa.resample(
                audio_24khz.astype(np.float32),
                orig_sr=24000, target_sr=self.SAMPLING_RATE,
            )
            sf.write(tmp_in, audio_22k, self.SAMPLING_RATE)

            # Resolve src / tgt SE per mode
            if cfg.mode == "identity_pass":
                if cfg.speaker_ref_audio is None:
                    raise ValueError("identity_pass needs speaker_ref_audio")
                ref_se = self._get_ref_se(cfg.speaker_ref_audio)
                src_se, tgt_se = ref_se, ref_se
            elif cfg.mode == "F1_to_native":
                if cfg.speaker_ref_audio is None:
                    raise ValueError("F1_to_native needs speaker_ref_audio")
                # src_se must reflect the audio's actual speaker, so
                # extract it from audio itself rather than the ref. This
                # matches OpenVoice demo's recommendation.
                src_se = self.converter.extract_se([str(tmp_in)])
                tgt_se = self._get_lang_se(cfg.target_language_se)
            elif cfg.mode == "native_to_F1":
                if cfg.speaker_ref_audio is None:
                    raise ValueError("native_to_F1 needs speaker_ref_audio")
                src_se = self._get_lang_se(cfg.target_language_se)
                tgt_se = self._get_ref_se(cfg.speaker_ref_audio)
            else:
                raise ValueError(f"unknown VC mode: {cfg.mode!r}")

            converted_22k = self.converter.convert(
                audio_src_path=str(tmp_in),
                src_se=src_se, tgt_se=tgt_se,
                output_path=None, tau=cfg.tau,
            )
        finally:
            if tmp_in.exists():
                tmp_in.unlink()

        # 22.05 kHz -> 24 kHz so downstream code sees the same rate as
        # raw OmniVoice output.
        converted_24k = librosa.resample(
            converted_22k.astype(np.float32),
            orig_sr=self.SAMPLING_RATE, target_sr=24000,
        )

        if out_path is not None:
            sf.write(out_path, converted_24k, 24000)

        return converted_24k
