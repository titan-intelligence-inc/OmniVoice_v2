"""Batched SeedVC convert with optional emotion-blended style.

Runs the full SeedVC inner loop for ``B`` source utterances at once,
all sharing the same target reference and (optionally) the same
emotion blend. Padding-aware: ``cfm.inference`` already accepts an
``x_lens`` tensor that gates attention, so we pad the conditioning
batch to the max length and pass true lengths.

Restrictions vs the single-utterance path:
  * Single-shot only. Assumes total context (prompt + content) fits
    inside ``max_context_window`` (~30 s at 22.05 kHz / hop 256).
    Adequate for 5-10 s clips. For long inputs, fall back to the
    streaming version.
  * F0 conditioning not implemented (we don't use it in the eps3
    pipeline).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torchaudio


def _resolve_proj(svc):
    """Lazy import + cache the emotion->style projection helper."""
    from .seedvc_emotion import emotion_to_style
    return emotion_to_style


def _batched_solve_euler(cfm, mu, x_lens, prompt, style, n_timesteps,
                         inference_cfg_rate, temperature=1.0):
    """Batched re-implementation of upstream BASECFM.solve_euler.

    The upstream version (modules/flow_matching.py) hardcodes the
    timestep tensor to length 2 when CFG is on, which only works for
    batch size 1. Here we expand ``t`` to ``(2B,)`` / ``(B,)`` so it
    matches the stacked / non-stacked forward.
    """
    B, T_total = mu.size(0), mu.size(1)
    z = torch.randn(
        [B, cfm.in_channels, T_total], device=mu.device,
    ) * temperature
    t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
    x = z
    t = t_span[0]
    prompt_len = prompt.size(-1)
    prompt_x = torch.zeros_like(x)
    prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
    x[..., :prompt_len] = 0
    if cfm.zero_prompt_speech_token:
        mu = mu.clone()
        mu[..., :prompt_len] = 0
    for step in range(1, len(t_span)):
        dt = t_span[step] - t_span[step - 1]
        if inference_cfg_rate > 0:
            stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
            stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
            stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
            stacked_x = torch.cat([x, x], dim=0)
            stacked_x_lens = torch.cat([x_lens, x_lens], dim=0)
            # CRITICAL: expand t to 2B, not 2 (upstream bug fix)
            stacked_t = t.expand(2 * B)
            stacked_dphi_dt = cfm.estimator(
                stacked_x, stacked_prompt_x, stacked_x_lens,
                stacked_t, stacked_style, stacked_mu,
            )
            dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
            dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
        else:
            t_b = t.expand(B)
            dphi_dt = cfm.estimator(x, prompt_x, x_lens, t_b, style, mu)
        x = x + dt * dphi_dt
        t = t + dt
        x[:, :, :prompt_len] = 0
    return x


def convert_voice_batch(
    svc,
    sources: Sequence[str | Path],
    target_path: str | Path,
    *,
    emotion_audio_path: Optional[str | Path] = None,
    emotion_embedding: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    diffusion_steps: int = 100,
    inference_cfg_rate: float = 0.7,
    length_adjust: float = 1.0,
) -> tuple[int, list[np.ndarray]]:
    """Run SeedVC on a batch of sources sharing one target/emotion.

    Returns ``(sample_rate, [audio_array_per_source])``. Each output
    array is trimmed to its true length (no padding zeros).
    """
    import librosa

    inference_module = svc.model
    mel_fn = svc.to_mel
    bigvgan_fn = svc.bigvgan_model
    sr = 22050
    hop_length = 256
    device = svc.device

    # ---------- target ref (shared across batch) ----------
    ref_np = librosa.load(str(target_path), sr=sr)[0]
    ref_audio = torch.tensor(ref_np[: sr * 25]).unsqueeze(0).float().to(device)
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    mel2 = mel_fn(ref_audio.float())                                 # [1, n_mels, T_ref]
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(device)
    S_ori = svc._process_whisper_features(ref_waves_16k, is_source=False)
    prompt_condition, _, _, _, _ = inference_module.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=None,
    )                                                                # [1, T_prompt, C]
    prompt_len_mel = mel2.size(2)

    # campplus speaker style
    feat2 = torchaudio.compliance.kaldi.fbank(
        ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style_speaker = svc.campplus_model(feat2.unsqueeze(0))            # [1, 192]

    # ---------- emotion blend (shared per call) ----------
    if beta > 0.0:
        if emotion_embedding is None:
            from ..analyzers.emotion_analyzer import EmotionAnalyzer
            analyzer = EmotionAnalyzer()
            emotion_embedding = analyzer.analyze(
                str(emotion_audio_path or target_path)
            ).embedding
        emotion_embedding = np.array(emotion_embedding, dtype=np.float32, copy=True)
        emotion_to_style = _resolve_proj(svc)
        emo_style = emotion_to_style(emotion_embedding, target_dim=192)
        emo_t = torch.from_numpy(emo_style).to(device).unsqueeze(0)   # [1, 192]
        spk_norm = float(style_speaker.norm().item()) + 1e-6
        emo_t = (emo_t * spk_norm).detach().clone()
        style_blend = (alpha * style_speaker + beta * emo_t).detach().clone()
    else:
        style_blend = (alpha * style_speaker).detach().clone()        # [1, 192]

    # ---------- per-source feature extraction ----------
    conds: list[torch.Tensor] = []           # each: [T_i, C]
    for src in sources:
        src_np = librosa.load(str(src), sr=sr)[0]
        src_audio = torch.tensor(src_np).unsqueeze(0).float().to(device)
        src_waves_16k = torchaudio.functional.resample(src_audio, sr, 16000)
        S_alt = svc._process_whisper_features(src_waves_16k, is_source=True)
        mel_i = mel_fn(src_audio.float())
        target_lengths = torch.LongTensor(
            [int(mel_i.size(2) * length_adjust)]).to(device)
        cond_i, _, _, _, _ = inference_module.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=None,
        )                                                            # [1, T_i, C]
        conds.append(cond_i.squeeze(0))                              # [T_i, C]

    B = len(conds)
    cond_lens = [c.size(0) for c in conds]
    max_T = max(cond_lens)
    n_feats = conds[0].size(-1)

    cond_pad = torch.zeros(B, max_T, n_feats, device=device, dtype=conds[0].dtype)
    for i, c in enumerate(conds):
        cond_pad[i, : cond_lens[i]] = c

    # build batched cat_condition [B, prompt_len + max_T, C]
    prompt_T = prompt_condition.size(1)
    prompt_b = prompt_condition.expand(B, -1, -1)
    cat_condition = torch.cat([prompt_b, cond_pad], dim=1)
    x_lens = torch.LongTensor([prompt_T + L for L in cond_lens]).to(device)

    mel2_b = mel2.expand(B, -1, -1).contiguous()
    style_b = style_blend.expand(B, -1).contiguous()

    # ---------- single batched cfm + bigvgan ----------
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        vc_target = _batched_solve_euler(
            inference_module.cfm,
            cat_condition, x_lens, mel2_b, style_b,
            n_timesteps=diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )                                                             # [B, n_mels, prompt_T_mel + ...]
        # strip prompt section, keep content mel only
        vc_content_mel = vc_target[:, :, prompt_len_mel:]            # [B, n_mels, T_max]
        vc_wave = bigvgan_fn(vc_content_mel.float()).squeeze(1)      # [B, T_audio]

    # ---------- slice to each sample's true length ----------
    # mel frame count per sample = cond_len (length_regulator output frames)
    out_arrays: list[np.ndarray] = []
    for i, L in enumerate(cond_lens):
        true_audio_len = L * hop_length
        wave_i = vc_wave[i, :true_audio_len].detach().cpu().float().numpy()
        out_arrays.append(np.asarray(wave_i, dtype=np.float32))

    return int(sr), out_arrays
