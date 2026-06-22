# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ltx23.config — the TWO generation profiles (QUALITY, SPEED),
each a faithful mirror of a FastVideo recipe (see ltx23/PROFILES.md).

These flags determine the torch.compile cache key + output quality. The tests
pin each profile's invariants against its FastVideo source so a careless edit
fails CI ("no nasty surprises about what we changed").
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ltx23.config import (  # noqa: E402
    QUALITY_DENOISE_STEPS,
    SPEED_DENOISE_STEPS,
    profile_kwargs,
    profile_uses_nvfp4,
)


def test_quality_profile_mirrors_reference_example() -> None:
    """QUALITY = basic_ltx2_3_distilled example: bf16, 3 refine, mode=default, VAE compile."""
    k = profile_kwargs("quality")
    assert k["ltx2_refine_enabled"] is True
    assert k["ltx2_refine_num_inference_steps"] == 3
    assert k["ltx2_refine_guidance_scale"] == 1.0
    assert k["ltx2_refine_add_noise"] is True
    assert k["enable_torch_compile"] is True
    assert k["enable_torch_compile_text_encoder"] is True
    assert k["enable_torch_compile_vae"] is True
    assert k["torch_compile_kwargs"]["backend"] == "inductor"
    assert k["torch_compile_kwargs"]["fullgraph"] is True
    assert k["torch_compile_kwargs"]["mode"] == "default"
    assert k["torch_compile_kwargs"]["dynamic"] is False
    assert k["ltx2_vae_tiling"] is False
    assert profile_uses_nvfp4("quality") is False
    assert QUALITY_DENOISE_STEPS == 8


def test_speed_profile_mirrors_streaming_demo() -> None:
    """SPEED = streaming_demo.yaml: NVFP4, 2 refine, max-autotune, no VAE compile, pin_cpu_memory."""
    k = profile_kwargs("speed")
    assert k["ltx2_refine_num_inference_steps"] == 2
    assert k["enable_torch_compile"] is True
    assert k["enable_torch_compile_text_encoder"] is True
    assert (
        k["enable_torch_compile_vae"] is False
    )  # streaming compiles only DiT+text_encoder
    assert k["torch_compile_kwargs"]["mode"] == "max-autotune-no-cudagraphs"
    assert k["torch_compile_kwargs"]["fullgraph"] is True
    assert k["torch_compile_kwargs"]["dynamic"] is False
    assert k["pin_cpu_memory"] is True
    assert k["ltx2_vae_tiling"] is False
    assert profile_uses_nvfp4("speed") is True
    assert SPEED_DENOISE_STEPS == 5


def test_profile_kwargs_returns_fresh_copy() -> None:
    """Mutating a returned dict must not poison the module-level profile."""
    a = profile_kwargs("quality")
    a["torch_compile_kwargs"]["mode"] = "mangled"
    a["ltx2_refine_num_inference_steps"] = 99
    b = profile_kwargs("quality")
    assert b["torch_compile_kwargs"]["mode"] == "default"
    assert b["ltx2_refine_num_inference_steps"] == 3


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError):
        profile_kwargs("turbo")


def test_default_profile_is_quality() -> None:
    """No env / empty -> quality (the safe, vivid default)."""
    assert profile_kwargs(None)["torch_compile_kwargs"]["mode"] == "default"
    assert profile_uses_nvfp4(None) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
