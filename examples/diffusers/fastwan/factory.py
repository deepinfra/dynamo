# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastWan-QAD-FP8 model factory (TAEHV tiny-decoder path).

``load_model`` is the SINGLE source of truth for how the FastWan-QAD
``VideoGenerator`` is constructed, used by both the in-process path
(``lib.backend``) and the pool path (``lib.pool`` resolves it via
``--model-factory fastwan.factory:load_model``). Sharing it keeps the
torch.compile cache keys byte-identical between warmup-bake and serving.

FastWan-QAD-FP8-1.3B is a Wan2.1-T2V-1.3B distillation. vs the LTX-2.3 factory:
  * single-stage pipeline -- no refine/upsampler;
  * FP8 e4m3 per-tensor linear quant, ALWAYS on (the QAD checkpoint's native
    format -- baked scale_weight params), not gated on enable_optimizations;
  * text-to-video only;
  * 3 denoise steps, no CFG (per-request);
  * **TAEHV tiny autoencoder for decode** instead of the full Wan VAE. This is
    what makes it the fast model: ~2.4s end-to-end vs ~5.8s with the full VAE
    (the DiT denoise is identical; only the decode differs). The generator is
    built with ``output_type="latent"`` so the heavy VAE is bypassed, and the
    latents are decoded by TAEHV (``taew2_1.pth``, baked into the image). This
    mirrors FastVideo's own ``fp8_wan2_1_1_3b.py --taehv-checkpoint`` recipe.

Heavy imports live inside the function so the dispatcher's argv-parse + dynamic
import path stays cheap for pool subprocesses.
"""

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

# TAEHV (Wan2.1 tiny autoencoder) checkpoint, baked into the image (Dockerfile).
TAEHV_CKPT = os.environ.get("FASTWAN_TAEHV_CKPT", "/opt/app/fastwan/taew2_1.pth")


def _load_taehv():
    import torch
    repo_dir = os.path.dirname(TAEHV_CKPT)
    if repo_dir and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from taehv import TAEHV
    return TAEHV(checkpoint_path=TAEHV_CKPT).to("cuda", torch.float16)


def _decode_with_taehv(taehv_model, latents):
    import torch
    with torch.no_grad():
        latents = latents.permute(0, 2, 1, 3, 4)
        latents = latents.to(device=next(taehv_model.parameters()).device,
                             dtype=next(taehv_model.parameters()).dtype)
        decoded = taehv_model.decode_video(latents, parallel=False, show_progress_bar=False)
        return [(f.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()
                for f in decoded[0]]


class _TaehvVideoGenerator:
    """Adapter exposing the FastVideo ``generate_video(**kwargs)`` contract that
    ``lib.backend``/``lib.pool`` call, but decoding via TAEHV. The underlying
    generator is built with ``output_type="latent"`` (VAE bypassed), so
    ``generate()`` returns latents in ``result.samples``; we TAEHV-decode them
    and write the mp4 at the requested fps. The served shape is the single baked
    832x480@81 (the model's native default), so width/height/num_frames are
    accepted but not re-plumbed into the latent request -- the model defaults
    already produce that shape (matches the warmup-baked compile graph)."""

    def __init__(self, gen: Any, taehv: Any) -> None:
        self._gen = gen
        self._taehv = taehv

    def generate_video(self, prompt: str, output_path: str | None = None,
                       fps: int = 16, num_inference_steps: int = 3,
                       guidance_scale: float = 1.0, seed: int | None = None,
                       negative_prompt: str | None = None,
                       save_video: bool = True, return_frames: bool = False,
                       **_ignored: Any) -> Any:
        import imageio
        sampling: dict[str, Any] = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        if seed is not None:
            sampling["seed"] = seed
        request: dict[str, Any] = {
            "prompt": prompt,
            "sampling": sampling,
            "output": {"save_video": False},
        }
        if negative_prompt:
            request["negative_prompt"] = negative_prompt
        result = self._gen.generate(request=request)
        frames = _decode_with_taehv(self._taehv, result.samples)
        if output_path:
            imageio.mimsave(output_path, frames, fps=fps, format="mp4")
        return result


def load_model(
    model_path: str,
    num_gpus: int,
    enable_optimizations: bool,
) -> Any:
    """Build the FastWan-QAD-FP8 generator (FP8 + TAEHV decode).

    ``enable_optimizations`` is accepted for the shared pool/warmup factory
    signature but intentionally unused (FP8 is always on; see module docstring).
    """
    from fastvideo import VideoGenerator
    from fastvideo.layers.quantization import get_quantization_config

    del enable_optimizations
    quant_kwargs: dict[str, Any] = {
        "transformer_quant": get_quantization_config("FP8")(granularity="tensor"),
    }
    logger.info("FastWan-QAD: FP8 e4m3 per-tensor linear quant (always on); "
                "attention_backend=%s",
                os.environ.get("FASTVIDEO_ATTENTION_BACKEND", "<default>"))

    gen = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=num_gpus,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        # output_type="latent" bypasses the full Wan VAE (TAEHV decodes instead),
        # so the heavy VAE can be CPU-offloaded to save GPU VRAM.
        vae_cpu_offload=True,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
        enable_torch_compile=True,
        output_type="latent",
        **quant_kwargs,
    )
    taehv = _load_taehv()
    logger.info("FastWan-QAD: TAEHV tiny decoder loaded from %s", TAEHV_CKPT)
    return _TaehvVideoGenerator(gen, taehv)
