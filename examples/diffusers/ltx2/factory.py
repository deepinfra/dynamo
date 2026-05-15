# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LTX-2 model factory.

``load_model`` is the SINGLE source of truth for how an LTX-2
VideoGenerator is constructed. It is used by both the legacy
in-process path (``lib.backend.GenericVideoBackend.initialize_model``)
and the pool path (``lib.pool._pool_worker_main`` resolves it
dynamically via the ``--model-factory ltx2.factory:load_model`` arg).
Sharing the factory across both paths is what keeps torch.compile
cache keys byte-identical between warmup and serving.

Heavy imports (torch, fastvideo, the LTX2 config kwargs) live inside
the function body so the dispatcher's argv-parse + dynamic-import
path stays cheap for pool subprocesses.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_model(
    model_path: str,
    num_gpus: int,
    enable_optimizations: bool,
) -> Any:
    """
    Build an LTX-2 ``VideoGenerator`` for ``model_path``.

    Returns an object that exposes ``generate_video(**kwargs)`` with the
    FastVideo keyword-args contract (prompt, save_video, return_frames,
    output_path, width, height, num_frames, fps, num_inference_steps,
    guidance_scale, seed, negative_prompt). Generic pool/backend code
    invokes ``.generate_video(...)`` and doesn't know about LTX-2
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
                from fastvideo.layers.quantization.fp4_config import FP4Config
            except ImportError as exc:
                raise RuntimeError(
                    "FastVideo optimizations require "
                    "fastvideo.layers.quantization.fp4_config, but this "
                    "FastVideo build does not provide it. Re-run "
                    "worker.py without --enable-optimizations or install a "
                    "FastVideo version that includes fp4_config."
                ) from exc
            pipeline_config.dit_config.quant_config = FP4Config()
            optimization_kwargs = fp4_kwargs()

    return VideoGenerator.from_pretrained(
        model_path,
        num_gpus=num_gpus,
        pipeline_config=pipeline_config,
        **optimization_kwargs,
    )
