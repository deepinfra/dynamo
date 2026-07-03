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


def _assert_distilled_preset(yaml_path: str) -> None:
    """Fail boot if FastVideo would not resolve the DISTILLED preset.

    FastVideo picks the preset by string-matching the weights path (needs
    "ltx-2"+"distilled"); a miss silently serves the BASE preset ~2.4x slower.
    """
    import yaml as _yaml

    with open(yaml_path) as fh:
        model_path = _yaml.safe_load(fh)["generator"]["model_path"]

    from fastvideo.registry import _get_config_info

    info = _get_config_info(model_path, raise_on_missing=False)
    preset = getattr(info, "default_preset", None) if info else None
    if preset != "ltx2_distilled":
        raise RuntimeError(
            f"LTX23_PROFILE=speed but FastVideo resolved preset {preset!r} for "
            f"model_path={model_path!r}; the path must contain 'ltx-2' and "
            "'distilled' (base preset is ~2.4x slower). Fix the weights alias "
            "(Dockerfile symlink /models/ltx-2.3-distilled-diffusers)."
        )
    logger.info("LTX-2.3 preset check OK: %s -> ltx2_distilled", model_path)


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
    import torch._inductor.config as _inductor
    from fastvideo import VideoGenerator
    from fastvideo.configs.pipelines.base import PipelineConfig

    from .config import profile_kwargs, profile_uses_nvfp4

    # Profile selects which of the two FastVideo-mirrored recipes to build:
    #   LTX23_PROFILE=quality (default) -> bf16, mode=default, VAE compile on
    #                                      (mirrors basic_ltx2_3_distilled example)
    #   LTX23_PROFILE=speed             -> NVFP4, max-autotune, no VAE compile
    #                                      (mirrors streaming_demo.yaml, the 4.55s path)
    # See ltx23/config.py / ltx23/PROFILES.md. The denoise step count (8 vs 5)
    # lives in the shapes file / per-request num_inference_steps, not here.
    profile = os.environ.get("LTX23_PROFILE", "quality").strip().lower()

    # SPEED loads the validated recipe verbatim via from_file -- do NOT re-derive
    # compile/quant/inductor settings in code (hand-set knobs are where the
    # recipe drifted before). Attention backend comes from the env (TORCH_SDPA).
    if profile == "speed":
        yaml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "streaming_speed.yaml"
        )
        _assert_distilled_preset(yaml_path)
        logger.info(
            "LTX-2.3 profile=speed: VideoGenerator.from_file(%s) "
            "[exact 10s recipe, no re-derivation]",
            yaml_path,
        )
        return VideoGenerator.from_file(yaml_path)

    optimization_kwargs = profile_kwargs(profile)

    # Inductor knobs from FastVideo's LTX-2.3 reference (basic_ltx2_3_distilled).
    # shape_padding=False is MANDATORY on Blackwell: without it the refine path
    # hits a cuBLAS INVALID_VALUE crash inside pad_mm. The rest are their
    # autotune-friendliness flags. Must be set before VideoGenerator compiles.
    _inductor.shape_padding = False
    _inductor.conv_1x1_as_mm = True
    _inductor.coordinate_descent_tuning = True
    _inductor.coordinate_descent_check_all_directions = True
    _inductor.epilogue_fusion = False

    pipeline_config = PipelineConfig.from_pretrained(model_path)

    # Quant: SPEED profile uses NVFP4; QUALITY stays bf16 (quant_config=None).
    # enable_optimizations gates NVFP4 as a safety (a caller asking for no-opt, or
    # non-Blackwell hardware, falls back to bf16 even on the speed profile).
    want_nvfp4 = profile_uses_nvfp4(profile) and enable_optimizations
    if want_nvfp4:
        major, minor = torch.cuda.get_device_capability()
        if major < 10:
            logger.warning(
                "NVFP4 (speed profile) needs Blackwell (cc>=10.0); detected %d.%d. "
                "Falling back to bf16.",
                major,
                minor,
            )
        else:
            from fastvideo.layers.quantization.nvfp4_config import NVFP4Config

            pipeline_config.dit_config.quant_config = NVFP4Config()
            logger.info(
                "LTX-2.3 profile=speed: NVFP4 + %s",
                optimization_kwargs["torch_compile_kwargs"]["mode"],
            )
    if pipeline_config.dit_config.quant_config is None:
        logger.info(
            "LTX-2.3 profile=%s: bf16 + %s",
            profile,
            optimization_kwargs["torch_compile_kwargs"]["mode"],
        )

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
