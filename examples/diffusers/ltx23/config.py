#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 generation profiles — TWO configs, each a FAITHFUL mirror of a specific
FastVideo recipe. Selected via the LTX23_PROFILE env var: "quality" (default) or
"speed". See ltx23/PROFILES.md for the full spec + FastVideo source mapping.

  QUALITY  — mirrors FastVideo's bf16 reference example
             examples/inference/basic/basic_ltx2_3_distilled_i2v_typed.py
             bf16 (no quant), 8 denoise + 3 refine, mode=default, VAE compile ON.
             Vivid/crisp; slower. This is the "looks good" path.

  SPEED    — mirrors FastVideo's DEPLOYED fast config (the 4.55s/1080p path)
             apps/dreamverse/serve_configs/streaming_demo.yaml
             NVFP4 quant, 5 denoise + 2 refine, max-autotune-no-cudagraphs,
             VAE compile OFF, pin_cpu_memory. Fast; NVFP4 desaturates ~40%
             (measured) -- that is FastVideo's own fast-path tradeoff, not a bug.

Each profile has THREE coupled parts. Keep them consistent per profile:
  1. QUANT          (bf16 vs NVFP4)            -> applied in factory.py from the profile
  2. DENOISE STEPS  (8 vs 5)                   -> in the shapes file / per-request num_inference_steps
                                                  (see *_DENOISE_STEPS below for the canonical values)
  3. LOAD-TIME KWARGS (refine steps, compile mode, offload, ...) -> the dicts below

Both profiles share: model FastVideo/LTX-2.3-Distilled-Diffusers, 1920x1088
landscape, 121 frames @ 24fps, guidance 1.0, negative "", refine upsampler =
<model>/spatial_upscaler, vae_tiling False, the Blackwell Inductor
knobs (factory.py: shape_padding=False etc.), LD_LIBRARY_PATH unset, cu128 env.

Attention backend is per-profile (see profile_attention_backend): both QUALITY
and SPEED use FLASH_ATTN -- FastVideo's SM100/SM103 FA4 kernels, matching their
actual streaming_demo speed recipe. SPEED previously shipped TORCH_SDPA as a
stand-in (~10.4s, quality OK'd by Johan) because FA4 (flash-attn-cute) didn't
build: the pinned XOR-op fork was stale against cutlass-dsl 4.5+ API moves
(cute.core.ThrMma, cute.make_fragment renamed/moved). FastVideo hit the same
break and fixed it 2026-07-06 (hao-ai-lab/FastVideo#1564): they dropped the fork
and now pin flash-attn-4 straight from Dao-AILab/flash-attention upstream at a
CuTe-DSL-4.6-compatible commit. Reproducing that exact fix (+ FastVideo's own
matching `[:2]` tuple-unpack fix in their flash_attn_cute.py wrapper) gets FA4
working end-to-end: ~7.3s warm, quality OK'd by Johan as on par or better than
TORCH_SDPA. Override with worker.py --attention-backend to revisit TORCH_SDPA.

Changing any of these requires a coordinated re-bake (the compile cache is keyed
on the kwargs + shape). Do NOT hand-edit values away from the FastVideo source.
"""

from copy import deepcopy
from typing import Any

# Canonical denoise step counts (these live in shapes.json / per-request, NOT in
# the kwargs below -- documented here so the profile is fully specified in one place).
QUALITY_DENOISE_STEPS = 8  # basic_ltx2_3_distilled example: num_inference_steps=8
SPEED_DENOISE_STEPS = 5  # streaming_demo.yaml default_request: num_inference_steps=5


# ── QUALITY ── mirrors basic_ltx2_3_distilled_i2v_typed.py ────────────────────
# quant: bf16 (factory leaves dit_config.quant_config = None for this profile).
LTX23_QUALITY_KWARGS: dict[str, Any] = {
    # refine: example -> 3 steps, gs 1.0, add_noise true, no LoRA
    "ltx2_refine_enabled": True,
    "ltx2_refine_lora_path": "",
    "ltx2_refine_num_inference_steps": 3,
    "ltx2_refine_guidance_scale": 1.0,
    "ltx2_refine_add_noise": True,
    # compile: example -> transformer + text_encoder + VAE, inductor, fullgraph,
    # mode=default, dynamic=false ("default ties max-autotune on 2.3, saves ~7min")
    "enable_torch_compile": True,
    "enable_torch_compile_text_encoder": True,
    "enable_torch_compile_vae": True,
    "torch_compile_kwargs": {
        "backend": "inductor",
        "fullgraph": True,
        "mode": "default",
        "dynamic": False,
    },
    # offload: example -> all on GPU
    "dit_cpu_offload": False,
    "vae_cpu_offload": False,
    "text_encoder_cpu_offload": False,
    "ltx2_vae_tiling": False,
}


# ── SPEED ── mirrors streaming_demo.yaml (FastVideo's 4.55s/1080p deploy) ──────
# quant: NVFP4 (factory sets dit_config.quant_config = NVFP4Config() for this profile).
LTX23_SPEED_KWARGS: dict[str, Any] = {
    # preset_overrides.refine -> 2 steps, gs 1.0, add_noise true
    "ltx2_refine_enabled": True,
    "ltx2_refine_lora_path": "",
    "ltx2_refine_num_inference_steps": 2,
    "ltx2_refine_guidance_scale": 1.0,
    "ltx2_refine_add_noise": True,
    # engine.compile -> transformer + text_encoder (NO vae_enabled),
    # max-autotune-no-cudagraphs, fullgraph, dynamic=false
    "enable_torch_compile": True,
    "enable_torch_compile_text_encoder": True,
    "enable_torch_compile_vae": False,
    "torch_compile_kwargs": {
        "backend": "inductor",
        "fullgraph": True,
        "mode": "max-autotune-no-cudagraphs",
        "dynamic": False,
    },
    # engine.offload -> all False, pin_cpu_memory true
    "dit_cpu_offload": False,
    "vae_cpu_offload": False,
    "text_encoder_cpu_offload": False,
    "pin_cpu_memory": True,
    "ltx2_vae_tiling": False,
}


_PROFILES = {"quality": LTX23_QUALITY_KWARGS, "speed": LTX23_SPEED_KWARGS}


def profile_kwargs(profile: str) -> dict[str, Any]:
    """Fresh deep copy of the load-time kwargs for the named profile.

    Callers must not mutate the module-level dicts; deep-copy each call so a
    downstream mutation can't poison the next caller.
    """
    key = (profile or "quality").strip().lower()
    if key not in _PROFILES:
        raise ValueError(
            f"unknown LTX23 profile {profile!r}; expected one of {sorted(_PROFILES)}"
        )
    return deepcopy(_PROFILES[key])


def profile_uses_nvfp4(profile: str) -> bool:
    """SPEED uses NVFP4; QUALITY is bf16."""
    return (profile or "quality").strip().lower() == "speed"


def profile_attention_backend(profile: str) -> str:
    """Attention backend per profile.

    Both QUALITY and SPEED -> FLASH_ATTN (FastVideo's SM100/SM103 FA4 kernels),
    matching FastVideo's actual recipes. Validated for SPEED 2026-07-08: ~7.3s
    warm (down from TORCH_SDPA's ~10.4s), quality OK'd by Johan as on par or
    better. Requires the flash-attn-4 install pinned to a CuTe-DSL-4.6-compatible
    commit (see Dockerfile.dreamverse) + the matching fastvideo wrapper patch
    (patches/flash-attn-cute-fa4-tuple-fix.patch) -- without both, FA4 fails to
    import/run and this setting would silently break the worker. Bake (warmup)
    and serve (worker) both derive the default from here so the compile cache
    always matches; override with worker.py --attention-backend to revisit
    TORCH_SDPA.
    """
    return "FLASH_ATTN"


def profile_default_optimizations(profile: str) -> bool:
    """Whether NVFP4 + torch.compile optimizations default ON for this profile.

    SPEED needs them ON (NVFP4 is the whole point); QUALITY is bf16, OFF. Coupling
    this to the profile means a SPEED ship image (LTX23_PROFILE=speed baked in)
    serves NVFP4 with no extra launch arg -- the recipe lives in the image, not a
    redis deploy-config flag. factory.load_model still hardware-gates NVFP4 to
    Blackwell, so a non-Blackwell host safely falls back to bf16 regardless.
    """
    return profile_uses_nvfp4(profile)
