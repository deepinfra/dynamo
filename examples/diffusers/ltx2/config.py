#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Canonical LTX-2 optimization_kwargs.

This is the SINGLE source of truth for the kwargs we pass to
VideoGenerator.from_pretrained for LTX-2-Distilled-Diffusers. Every
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

# Standard ship-path kwargs. Used by warmup, benchmark, and production
# worker (when no FP4 / max-autotune is requested).
#
# torch_compile_kwargs notes:
#   - mode="default" is what we ship. We tested mode="max-autotune-no-cudagraphs"
#     in 2.2.0-ltx2-c3266d71 (single-process bake, ~70 min) and found <1%
#     improvement at 1080p steady state vs default mode. The 70-min bake cost
#     wasn't justified. Document: 2026-04-29 benchmark on slc-111 B200 GPU 0,
#     gens 11-14 default=50.4/50.2/15.2/40.6s vs max-autotune=49.8/49.4/15.2/39.8s.
#     736×1280@121f saw a -20% improvement (18.9s -> 15.2s) -- worth re-investigating
#     if that shape becomes a hot path, but not enough to justify the bake cost
#     for the whole menu.
#   - fullgraph=False because LTX-2's pipeline has Python control flow
#     (refine on/off, audio branch) that dynamo can't trace as one graph.
#
# ltx2_vae_tiling=True is mandatory for >=241-frame 1080p shapes; without
# it the VAE decoder's intermediate tensor exceeds 2^31 elements and F.pad
# trips PyTorch's int32 indexing limit. (For shapes that don't need
# tiling, leaving it on costs ~1s of VAE decode time -- negligible.)
LTX2_STANDARD_KWARGS: dict[str, Any] = {
    "ltx2_refine_enabled": True,
    "ltx2_refine_lora_path": "",
    "ltx2_refine_num_inference_steps": 2,
    "ltx2_refine_guidance_scale": 1.0,
    "ltx2_refine_add_noise": True,
    "enable_torch_compile": True,
    "enable_torch_compile_text_encoder": True,
    "torch_compile_kwargs": {
        "backend": "inductor",
        "fullgraph": False,
        "mode": "default",
    },
    "dit_cpu_offload": False,
    "vae_cpu_offload": False,
    "text_encoder_cpu_offload": False,
    "ltx2_vae_tiling": True,
}


# FP4-specific overlays. Only applied when the caller explicitly requests
# FP4 quantization (via worker.py --enable-optimizations). Switching to
# this path requires a separate cache bake; the standard cache won't hit.
LTX2_FP4_KWARGS: dict[str, Any] = {
    "torch_compile_kwargs": {
        "backend": "inductor",
        "fullgraph": True,
        "mode": "max-autotune-no-cudagraphs",
    },
}


def standard_kwargs() -> dict[str, Any]:
    """Return a fresh deep copy of the standard kwargs.

    Callers must not mutate the module-level dict; deep-copy each call so
    a downstream mutation can't poison the next caller.
    """
    return deepcopy(LTX2_STANDARD_KWARGS)


def fp4_kwargs() -> dict[str, Any]:
    """
    Return a fresh deep copy of the FP4-mode kwargs (standard merged with
    FP4 overrides). FP4 quantization itself (FP4Config) is configured on
    the pipeline_config separately, not here.
    """
    merged = deepcopy(LTX2_STANDARD_KWARGS)
    for key, value in LTX2_FP4_KWARGS.items():
        merged[key] = deepcopy(value)
    return merged
