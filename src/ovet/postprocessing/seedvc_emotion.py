"""Approach 2: emotion2vec-injected style for SeedVC.

SeedVC's flow conditions on a 192-d speaker style vector (campplus
output of the target audio). We extract an emotion2vec embedding from
an emotional reference, project it to 192-d via a fixed random
orthogonal matrix (preserves geometry, no training), L2-normalise it,
and blend with the campplus style::

    style' = α · style_speaker + β · proj_192(emotion2vec(emo_ref))

The blend is fed to ``cfm.inference`` in place of the original style2,
so the rest of the SeedVC pipeline is untouched.

Expected behaviour: speaker identity comes from α (target = F1 ref);
emotional spectral cues come from β. If emotion2vec captures emotion
information that survives the random projection, β > 0 should add
emotional colour to the output without disturbing the F1 voice.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio


# ----------------------------------------------------------------------
# emotion2vec → 192-d projection (deterministic, training-free)
# ----------------------------------------------------------------------

_PROJECTION_CACHE: dict[tuple[int, int, int], np.ndarray] = {}


def _orthogonal_projection(d_in: int, d_out: int, seed: int = 42) -> np.ndarray:
    """Return a fixed (d_in × d_out) orthogonal matrix.

    Uses QR decomposition of a Gaussian matrix. Same seed → same
    matrix across calls so the projection is reproducible.
    """
    key = (d_in, d_out, seed)
    if key in _PROJECTION_CACHE:
        return _PROJECTION_CACHE[key]
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d_in, d_out)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    _PROJECTION_CACHE[key] = Q
    return Q


def emotion_to_style(
    emotion_embedding: np.ndarray,
    target_dim: int = 192,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Project an emotion2vec utterance embedding to ``target_dim``.

    Uses a fixed random orthogonal projection. The result is L2-
    normalised so its scale is comparable across utterances.
    """
    e = np.asarray(emotion_embedding, dtype=np.float32).reshape(-1)
    if e.shape[0] >= target_dim:
        # Just project down
        Q = _orthogonal_projection(e.shape[0], target_dim, seed=seed)
        out = e @ Q
    else:
        # Project up — pad with zeros first then project (rare path)
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[: e.shape[0]] = e
        out = padded
    n = np.linalg.norm(out) + 1e-9
    return (out / n).astype(np.float32)


# ----------------------------------------------------------------------
# SeedVC convert with style blending
# ----------------------------------------------------------------------

def convert_voice_with_emotion(
    svc,
    source_path: str | Path,
    target_path: str | Path,
    emotion_audio_path: str | Path,
    *,
    emotion_embedding: Optional[np.ndarray] = None,
    alpha: float = 1.0,           # speaker style weight
    beta: float = 0.5,            # emotion style weight
    diffusion_steps: int = 100,
    inference_cfg_rate: float = 0.7,
    length_adjust: float = 1.0,
    f0_condition: bool = False,
    auto_f0_adjust: bool = True,
) -> tuple[int, np.ndarray]:
    """Run SeedVC convert_voice with emotion-blended style.

    Re-implements the wrapper's convert_voice body so we can patch
    style2 mid-flight. Returns ``(sample_rate, audio_array)``.

    Args:
        svc:                 ``SeedVCWrapper`` from ``load_seedvc()``.
        source_path:         audio with the content we want.
        target_path:         audio with the speaker timbre we want.
        emotion_audio_path:  audio whose emotion2vec embedding is
                             projected into the style.
        emotion_embedding:   optional pre-computed emotion2vec
                             embedding (skip extraction).
        alpha:               speaker style weight (≥ 0).
        beta:                emotion style weight (≥ 0).
    """
    import librosa
    # Load emotion embedding if not pre-computed
    if emotion_embedding is None:
        from ..analyzers.emotion_analyzer import EmotionAnalyzer
        analyzer = EmotionAnalyzer()
        emotion_embedding = analyzer.analyze(str(emotion_audio_path)).embedding
    # Defensive: emotion2vec may emit inference-mode tensors, copy to plain np
    emotion_embedding = np.array(emotion_embedding, dtype=np.float32, copy=True)

    # Project to 192-d with the same seed every time
    emo_style = emotion_to_style(emotion_embedding, target_dim=192)
    emo_style_t = torch.from_numpy(emo_style).to(svc.device).unsqueeze(0)

    # ----- copy of svc.convert_voice body, with style2 patched -----
    # (matches Plachta/seed-vc seed_vc_wrapper.py @ commit cloned in
    #  upstream/seed-vc; modify carefully if the upstream API drifts.)
    inference_module = svc.model if not f0_condition else svc.model_f0
    mel_fn = svc.to_mel if not f0_condition else svc.to_mel_f0
    bigvgan_fn = svc.bigvgan_model if not f0_condition else svc.bigvgan_44k_model
    sr = 22050 if not f0_condition else 44100
    hop_length = 256 if not f0_condition else 512
    max_context_window = sr // hop_length * 30
    overlap_wave_len = svc.overlap_frame_len * hop_length

    src_audio_np = librosa.load(str(source_path), sr=sr)[0]
    ref_audio_np = librosa.load(str(target_path), sr=sr)[0]
    src_audio = torch.tensor(src_audio_np).unsqueeze(0).float().to(svc.device)
    ref_audio = torch.tensor(ref_audio_np[:sr * 25]).unsqueeze(0).float().to(svc.device)

    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    converted_waves_16k = torchaudio.functional.resample(src_audio, sr, 16000)

    S_alt = svc._process_whisper_features(converted_waves_16k, is_source=True)
    S_ori = svc._process_whisper_features(ref_waves_16k, is_source=False)

    mel = mel_fn(src_audio.float())
    mel2 = mel_fn(ref_audio.float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2_speaker = svc.campplus_model(feat2.unsqueeze(0))     # [1, 192]

    # Match emotion style to speaker style scale, then blend.
    # Clone everything to escape any inference-mode tensors that may
    # have leaked in from the emotion2vec / fbank computation.
    speaker_norm = float(style2_speaker.norm().item()) + 1e-6
    emo_style_t = (emo_style_t * speaker_norm).detach().clone()
    style2 = (alpha * style2_speaker + beta * emo_style_t).detach().clone()

    # F0 (kept for completeness — disabled by default)
    if f0_condition:
        F0_ori = svc.rmvpe.infer_from_audio(ref_waves_16k[0], thred=0.03)
        F0_alt = svc.rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)
        F0_ori = torch.from_numpy(F0_ori).to(svc.device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(svc.device)[None]
        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]
        median_log_f0_ori = torch.median(torch.log(voiced_F0_ori + 1e-5))
        median_log_f0_alt = torch.median(torch.log(voiced_F0_alt + 1e-5))
        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_f0_adjust:
            shifted_log_f0_alt[F0_alt > 1] = (
                log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            )
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
    else:
        F0_ori = None
        shifted_f0_alt = None

    cond, _, _, _, _ = inference_module.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
    prompt_condition, _, _, _, _ = inference_module.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

    # Single-pass (no streaming) for the full output
    chunks = []
    max_source_window = max_context_window - mel2.size(2)
    processed = 0
    previous_chunk = None
    generated = []
    while processed < cond.size(1):
        chunk_cond = cond[:, processed:processed + max_source_window]
        is_last = processed + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(device_type=svc.device.type, dtype=torch.float16):
            vc_target = inference_module.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2, style2, None, diffusion_steps,
                inference_cfg_rate=inference_cfg_rate)
            vc_target = vc_target[:, :, mel2.size(-1):]
        vc_wave = bigvgan_fn(vc_target.float())[0]
        # Use the wrapper's own chunking logic for cross-fades.
        processed_frames, previous_chunk, should_break, _, full_audio = (
            svc._stream_wave_chunks(
                vc_wave, processed, vc_target, overlap_wave_len,
                generated, previous_chunk, is_last,
                stream_output=True, sr=sr,
            )
        )
        processed = processed_frames
        if full_audio is not None:
            chunks.append(full_audio)
        if should_break:
            break

    # `full_audio` is (sr, audio_array) per the wrapper's contract
    last_sr, last_audio = chunks[-1] if chunks else (sr, np.zeros(0, dtype=np.float32))
    return int(last_sr), np.asarray(last_audio, dtype=np.float32)
