# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastWan2.2-TI2V-5B-FullAttn model factory (optimized: tiny decoder + compile).

``load_model`` is the SINGLE source of truth for how this family's
``VideoGenerator`` is constructed, used by both the in-process path
(``lib.backend``) and the pool path (``lib.pool`` resolves it via
``--model-factory fastwan22_5b.factory:load_model``).

Config (Johan quality-approved 2026-08-03 via seed-locked split-screen A/B
vs the full-VAE recipe -- "equally good to the eye"; ~3x faster/cheaper):
  * bf16 -- the model's native precision; NO quantization;
  * **TAEHV tiny autoencoder (taew2_2_super) for decode** instead of the full
    Wan2.2 VAE: generator built with ``output_type="latent"`` and the heavy
    VAE CPU-offloaded; latents decoded by the wrapper below. B200 measured
    ~3.0s/clip vs 9.8s full-VAE;
  * torch.compile ON (quality-neutral, ~19% on the bench);
  * 3 denoise steps, no CFG (per-request, matching the DMD schedule);
  * ``pipeline_config`` is PINNED, see below.
"""

import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Wan2.2-VAE tiny decoder (48-ch; taew2_1 does NOT fit this model family),
# baked into the image (Dockerfile).
TAEHV_CKPT = os.environ.get(
    "FASTWAN22_5B_TAEHV_CKPT", "/opt/app/fastwan22_5b/taew2_2_super.pth")


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
        latents = latents.to(
            device=next(taehv_model.parameters()).device,
            dtype=next(taehv_model.parameters()).dtype,
        )
        decoded = taehv_model.decode_video(
            latents, parallel=False, show_progress_bar=False
        )
        return [
            (f.clamp(0, 1) * 255).byte().cpu().permute(1, 2, 0).numpy()
            for f in decoded[0]
        ]


class _TaehvVideoGenerator:
    """Adapter exposing the FastVideo ``generate_video(**kwargs)`` contract that
    ``lib.backend``/``lib.pool`` call, but decoding via TAEHV. width/height/
    num_frames ARE plumbed into the latent request (via SamplingParam) so each
    pool subprocess renders its actual shape -- without this, portrait
    704x1280 silently comes out landscape (the QAD bug)."""

    def __init__(self, gen: Any, taehv: Any) -> None:
        self._gen = gen
        self._taehv = taehv

    def generate_video(
        self,
        prompt: str,
        output_path: str | None = None,
        fps: int = 24,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        seed: int | None = None,
        negative_prompt: str | None = None,
        width: int | None = None,
        height: int | None = None,
        num_frames: int | None = None,
        save_video: bool = True,
        return_frames: bool = False,
        **_ignored: Any,
    ) -> Any:
        import imageio

        sampling: dict[str, Any] = {
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        if width is not None:
            sampling["width"] = width
        if height is not None:
            sampling["height"] = height
        if num_frames is not None:
            sampling["num_frames"] = num_frames
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
    """Build the FastWan2.2-TI2V-5B generator (bf16 + compile + TAEHV decode).

    ``enable_optimizations`` is accepted for the shared pool/warmup factory
    signature but intentionally unused (the optimized recipe is always on).
    """
    from fastvideo import VideoGenerator
    from fastvideo.configs.pipelines.wan import FastWan2_2_TI2V_5B_Config

    del enable_optimizations

    # The serving mount anonymizes the weights path to /data/default, which
    # defeats fastvideo's path-based preset resolution -- and this model's
    # model_index.json ``_class_name`` (WanDMDPipeline) then matches the
    # FastWan2.1 detector, silently selecting the 480p Wan2.1 config.
    # Pinning the config makes the selection explicit; the check makes any
    # future rename/refactor of the config class loud instead of silent.
    # (Empirically verified 2026-08-03: RESOLVED_CONFIG=FastWan2_2_TI2V_5B_Config
    # flow_shift=5.0 dmd=[1000, 757, 522] at /data/default.)
    pipeline_config = FastWan2_2_TI2V_5B_Config()
    if not getattr(pipeline_config, "dmd_denoising_steps", None):
        raise RuntimeError(
            "FastWan2_2_TI2V_5B_Config has no dmd_denoising_steps; refusing "
            "to serve a distill checkpoint with a non-distill config"
        )
    logger.info(
        "fastwan22_5b: pinned pipeline_config=%s flow_shift=%s dmd_steps=%s "
        "attention_backend=%s taehv_ckpt=%s",
        type(pipeline_config).__name__,
        getattr(pipeline_config, "flow_shift", None),
        getattr(pipeline_config, "dmd_denoising_steps", None),
        os.environ.get("FASTVIDEO_ATTENTION_BACKEND", "<default>"),
        TAEHV_CKPT,
    )

    gen = VideoGenerator.from_pretrained(
        model_path,
        num_gpus=num_gpus,
        pipeline_config=pipeline_config,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        # output_type="latent" bypasses the full Wan2.2 VAE (TAEHV decodes
        # instead), so the heavy VAE is CPU-offloaded to save GPU VRAM.
        vae_cpu_offload=True,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
        enable_torch_compile=True,
        output_type="latent",
    )
    taehv = _load_taehv()
    logger.info("fastwan22_5b: TAEHV tiny decoder loaded from %s", TAEHV_CKPT)
    return _TaehvVideoGenerator(gen, taehv)
