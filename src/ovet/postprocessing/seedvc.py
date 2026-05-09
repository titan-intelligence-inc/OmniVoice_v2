"""SeedVC integration helper.

Wraps Plachta/seed-vc's ``SeedVCWrapper`` with two patches the upstream
code needs to run on our pinned dependency stack:

1. **BigVGAN ``_from_pretrained`` kwargs**: huggingface_hub's interface
   evolved; the upstream BigVGAN class still requires ``proxies`` /
   ``resume_download`` as positional kwargs, but the new
   ``ModelHubMixin.from_pretrained`` no longer passes them. We patch
   the classmethod with default values.

2. **SeedVCWrapper device handling**: the wrapper expects a
   ``torch.device`` rather than a string for some autocast paths.

Use this module's ``load_seedvc()`` instead of importing
``seed_vc_wrapper.SeedVCWrapper`` directly.

Setup once per environment:
  * ``git clone --depth 1 https://github.com/Plachtaa/seed-vc.git upstream/seed-vc``
  * The HF checkpoints download lazily on first use (~2.5 GB into
    ./checkpoints/, gitignored).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional

import torch


_SEEDVC_DIR = Path("upstream/seed-vc")
_PATCHED = False


def _ensure_patches_applied() -> None:
    """Apply the upstream patches before SeedVC imports anything."""
    global _PATCHED
    if _PATCHED:
        return

    if str(_SEEDVC_DIR.resolve()) not in sys.path:
        sys.path.insert(0, str(_SEEDVC_DIR.resolve()))

    # Patch BigVGAN._from_pretrained — must happen before
    # SeedVCWrapper imports/loads BigVGAN.
    from modules.bigvgan import bigvgan as _bigvgan_mod    # type: ignore
    _orig = _bigvgan_mod.BigVGAN._from_pretrained.__func__

    def _patched(
        cls, *,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        proxies: Optional[dict] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token=None,
        map_location: str = "cpu",
        strict: bool = False,
        use_cuda_kernel: bool = False,
        **model_kwargs,
    ):
        return _orig(
            cls,
            model_id=model_id, revision=revision, cache_dir=cache_dir,
            force_download=force_download, proxies=proxies,
            resume_download=resume_download, local_files_only=local_files_only,
            token=token, map_location=map_location, strict=strict,
            use_cuda_kernel=use_cuda_kernel, **model_kwargs,
        )

    _bigvgan_mod.BigVGAN._from_pretrained = classmethod(_patched)
    _PATCHED = True


def load_seedvc(device: str | torch.device = "cuda"):
    """Load Plachta/Seed-VC with the patches applied.

    Returns the upstream ``SeedVCWrapper`` instance unchanged once the
    patches are in place — call ``svc.convert_voice(source, target,
    **kwargs)`` exactly as in the upstream README.

    The wrapper's ``stream_output=True`` mode yields
    ``(mp3_bytes, (sample_rate, audio_ndarray))`` per chunk; only the
    last yield holds the full audio. Use::

        last = None
        for item in svc.convert_voice(source, target, stream_output=True):
            last = item
        sr, audio = last[1]
    """
    _ensure_patches_applied()
    from seed_vc_wrapper import SeedVCWrapper                 # type: ignore

    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
    return SeedVCWrapper(device=device)


def convert_voice_full(
    svc,
    source: str | Path,
    target: str | Path,
    *,
    diffusion_steps: int = 100,
    inference_cfg_rate: float = 0.7,
    length_adjust: float = 1.0,
    f0_condition: bool = False,
    auto_f0_adjust: bool = True,
):
    """Run SeedVC and return ``(sample_rate, audio_ndarray)`` of the
    fully-converted utterance. Convenience over the streaming yield
    loop.
    """
    last = None
    for item in svc.convert_voice(
        source=str(source), target=str(target),
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        f0_condition=f0_condition, auto_f0_adjust=auto_f0_adjust,
        stream_output=True,
    ):
        last = item
    if last is None:
        raise RuntimeError("SeedVC produced no output")
    sr, audio = last[1]
    return int(sr), audio
