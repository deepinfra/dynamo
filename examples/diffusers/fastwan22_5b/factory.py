# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastWan2.2-TI2V-5B-FullAttn model factory (FastVideo recommended recipe).

``load_model`` is the SINGLE source of truth for how this family's
``VideoGenerator`` is constructed, used by both the in-process path
(``lib.backend``) and the pool path (``lib.pool`` resolves it via
``--model-factory fastwan22_5b.factory:load_model``).

vs the ``fastwan`` (QAD) factory this family was cloned from:
  * bf16 -- the model's native precision; NO quantization;
  * full Wan2.2 VAE decode -- FastVideo's recommended recipe for this model
    (no TAEHV tiny-decoder wrapper; that is a possible later, separately
    quality-gated optimization);
  * 3 denoise steps, no CFG (per-request, matching the DMD schedule);
  * ``pipeline_config`` is PINNED, see below.
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
    """Build the FastWan2.2-TI2V-5B generator (bf16, full VAE, pinned config).

    ``enable_optimizations`` is accepted for the shared pool/warmup factory
    signature but intentionally unused (the recommended recipe has no gated
    optimizations; torch.compile is off pending its own validated delta).
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
    pipeline_config = FastWan2_2_TI2V_5B_Config()
    if not getattr(pipeline_config, "dmd_denoising_steps", None):
        raise RuntimeError(
            "FastWan2_2_TI2V_5B_Config has no dmd_denoising_steps; refusing "
            "to serve a distill checkpoint with a non-distill config"
        )
    logger.info(
        "fastwan22_5b: pinned pipeline_config=%s flow_shift=%s dmd_steps=%s "
        "attention_backend=%s",
        type(pipeline_config).__name__,
        getattr(pipeline_config, "flow_shift", None),
        getattr(pipeline_config, "dmd_denoising_steps", None),
        os.environ.get("FASTVIDEO_ATTENTION_BACKEND", "<default>"),
    )

    return VideoGenerator.from_pretrained(
        model_path,
        num_gpus=num_gpus,
        pipeline_config=pipeline_config,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
        enable_torch_compile=False,
    )
