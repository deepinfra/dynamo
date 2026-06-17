#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Canonical LTX-2.3 optimization_kwargs.

This is the SINGLE source of truth for the kwargs we pass to
VideoGenerator.from_pretrained for LTX-2.3-Distilled-Diffusers. Every
process that touches the compile cache (warmup, worker, benchmark)
imports from here so the torch.compile cache keys produced and
consumed are byte-identical.

Changing any flag here requires re-baking the runtime image's compile
cache (RUNBOOK § "Producing a ship image"). The hash that ends up in
the image tag depends on the warmup_shapes.json menu, but the cache
contents themselves depend on these kwargs -- so don't edit them
without a coordinated rebuild.
"""

from copy import deepcopy
from typing import Any

# THE recipe -- mirrors FastVideo's OWN LTX-2.3 distilled reference example
# (examples/inference/basic/basic_ltx2_3_distilled_i2v_typed.py @ FASTVIDEO_SHA),
# the path behind their "5s 1080p video in ~4.55s on a single B200" result (with
# NVFP4 added -- the example itself runs bf16). Do NOT substitute hand-picked
# values: reproduce their path, don't invent our own.
#
# IMPORTANT: the 2.3 recipe differs from the 2.0 Dreamverse streaming_demo.yaml.
# 2.3 uses 8 denoise + 3 refine steps (not 5 + 2) and mode="default" (their
# comment: "Inductor's default schedule matches max-autotune on this pipeline
# while saving ~7 min of cold compile"). Denoise step count lives in
# shapes.json (num_inference_steps=8); refine steps are here.
#
# 2.3 example mapping:
#   compile  -> enabled + text_encoder + VAE; inductor, fullgraph, mode=default, dynamic=false
#   offload  -> all False (DiT / text-encoder / VAE resident on GPU)
#   vae_tiling -> False
#   refine   -> 3 steps, gs 1.0, add_noise true, no LoRA
# Blackwell-mandatory Inductor knobs (shape_padding=False etc.) are set in
# factory.py before compile -- without shape_padding=False the refine path
# crashes cuBLAS (pad_mm INVALID_VALUE) on B200.
# NVFP4 quant is set on pipeline_config in factory.py (enable_optimizations).
# FLASH_ATTN via FASTVIDEO_ATTENTION_BACKEND in worker.py / warmup.py.
LTX23_STANDARD_KWARGS: dict[str, Any] = {
    "ltx2_refine_enabled": True,
    "ltx2_refine_lora_path": "",
    "ltx2_refine_num_inference_steps": 3,
    "ltx2_refine_guidance_scale": 1.0,
    "ltx2_refine_add_noise": True,
    "enable_torch_compile": True,
    "enable_torch_compile_text_encoder": True,
    "enable_torch_compile_vae": True,
    "torch_compile_kwargs": {
        "backend": "inductor",
        "fullgraph": True,
        "mode": "default",
        "dynamic": False,
    },
    "dit_cpu_offload": False,
    "vae_cpu_offload": False,
    "text_encoder_cpu_offload": False,
    "ltx2_vae_tiling": False,
}


# The full-stack-optimized recipe now lives in LTX23_STANDARD_KWARGS (it IS the
# FastVideo recipe), so there is no separate "fast" overlay anymore. Kept as an
# empty overlay for import compatibility with factory.py's env switch; both
# paths now resolve to the same recipe.
LTX23_FP4_KWARGS: dict[str, Any] = {}


def standard_kwargs() -> dict[str, Any]:
    """Return a fresh deep copy of the standard kwargs.

    Callers must not mutate the module-level dict; deep-copy each call so
    a downstream mutation can't poison the next caller.
    """
    return deepcopy(LTX23_STANDARD_KWARGS)


def fp4_kwargs() -> dict[str, Any]:
    """
    Return a fresh deep copy of the FP4-mode kwargs (standard merged with
    FP4 overrides). FP4 quantization itself (FP4Config) is configured on
    the pipeline_config separately, not here.
    """
    merged = deepcopy(LTX23_STANDARD_KWARGS)
    for key, value in LTX23_FP4_KWARGS.items():
        merged[key] = deepcopy(value)
    return merged
