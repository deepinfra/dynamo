# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LTX-2.3 model factory.

``load_model`` is the SINGLE source of truth for how an LTX-2.3
VideoGenerator is constructed. It is used by both the legacy
in-process path (``lib.backend.GenericVideoBackend.initialize_model``)
and the pool path (``lib.pool._pool_worker_main`` resolves it
dynamically via the ``--model-factory ltx23.factory:load_model`` arg).
Sharing the factory across both paths is what keeps torch.compile
cache keys byte-identical between warmup and serving.

Heavy imports (torch, fastvideo, the LTX-2.3 config kwargs) live inside
the function body so the dispatcher's argv-parse + dynamic-import
path stays cheap for pool subprocesses.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def load_model(
    model_path: str,
    num_gpus: int,
    enable_optimizations: bool,
) -> Any:
    """
    Build an LTX-2.3 ``VideoGenerator`` for ``model_path``.

    Returns an object that exposes ``generate_video(**kwargs)`` with the
    FastVideo keyword-args contract (prompt, save_video, return_frames,
    output_path, width, height, num_frames, fps, num_inference_steps,
    guidance_scale, seed, negative_prompt). Generic pool/backend code
    invokes ``.generate_video(...)`` and doesn't know about LTX-2.3
    internals.
    """
    import torch
    from fastvideo import VideoGenerator
    from fastvideo.configs.pipelines.base import PipelineConfig

    from .config import fp4_kwargs, standard_kwargs

    pipeline_config = PipelineConfig.from_pretrained(model_path)

    if not enable_optimizations:
        optimization_kwargs = standard_kwargs()
    else:
        major, minor = torch.cuda.get_device_capability()
        if major < 10:
            logger.warning(
                "FP4 quantization is only supported on NVIDIA Blackwell GPUs "
                "(compute capability 10.0+). Detected compute capability: %d.%d. "
                "Continuing without FP4 optimizations.",
                major,
                minor,
            )
            optimization_kwargs = standard_kwargs()
        else:
            logger.info(
                "Using FP4 quantization for VideoGenerator model=%s",
                model_path,
            )
            try:
                # LTX-2.3 / post-#1288 FastVideo renamed FP4Config -> NVFP4Config
                # (module fp4_config -> nvfp4_config). Verified present at the
                # pinned target SHA (nvfp4_config.py:NVFP4Config).
                from fastvideo.layers.quantization.nvfp4_config import NVFP4Config
            except ImportError as exc:
                raise RuntimeError(
                    "FastVideo optimizations require "
                    "fastvideo.layers.quantization.nvfp4_config, but this "
                    "FastVideo build does not provide it. Re-run "
                    "worker.py without --enable-optimizations or install a "
                    "FastVideo version that includes nvfp4_config."
                ) from exc
            pipeline_config.dit_config.quant_config = NVFP4Config()
            # Compile-mode choice for the NVFP4 path (Phase-0 A/B on di-slc-47):
            #   default (robust): mode="default" via standard_kwargs() -- portable
            #     compile cache, fast bake, recompile-tolerant; aligned with the
            #     never-recompile goal. NVFP4 quant is independent (set above).
            #   LTX23_FP4_MAX_AUTOTUNE=1: fp4_kwargs() (fullgraph + max-autotune)
            #     -- ~70-min bake, more env-fragile, <1% gain at 1080p in LTX-2
            #     testing. Measure before adopting.
            if os.environ.get("LTX23_FP4_MAX_AUTOTUNE") == "1":
                logger.info("NVFP4 + max-autotune compile (LTX23_FP4_MAX_AUTOTUNE=1)")
                optimization_kwargs = fp4_kwargs()
            else:
                logger.info("NVFP4 + mode=default compile (robust path)")
                optimization_kwargs = standard_kwargs()

    # LTX-2.3 distilled is a two-stage pipeline: a fast low-res denoise pass
    # followed by a latent-upsample + refine pass (this is what buys the 1080p
    # quality). The refine stage is enabled by the LTX-2.3 pipeline config, but
    # our weights' model_index.json is a plain diffusers LTX2Pipeline index with
    # no fastvideo_refine_* keys, so FastVideo's auto-discovery (which looks for
    # a "spatial_upsampler" entry / fastvideo_refine_upsampler_path in the index)
    # can't find the upsampler and load_modules raises
    # "ltx2_refine_enabled is True but ltx2_refine_upsampler_path was not
    # provided." Point it at the bundled spatial_upscaler component (a Diffusers
    # dir: config.json + model.safetensors). No separate refine transformer ships
    # with this checkpoint, so the refine pass reuses the main DiT.
    extra_kwargs: dict[str, Any] = {}
    upsampler_path = os.path.join(model_path, "spatial_upscaler")
    if os.path.isdir(upsampler_path):
        extra_kwargs["ltx2_refine_upsampler_path"] = upsampler_path
    else:
        logger.warning(
            "LTX-2.3 refine upsampler dir not found at %s; refine stage will "
            "fail to initialize. Check the weights layout.",
            upsampler_path,
        )

    return VideoGenerator.from_pretrained(
        model_path,
        num_gpus=num_gpus,
        pipeline_config=pipeline_config,
        **optimization_kwargs,
        **extra_kwargs,
    )
