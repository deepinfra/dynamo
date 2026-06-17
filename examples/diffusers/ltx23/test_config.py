# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ltx23.config: the canonical kwargs we pass to
VideoGenerator.from_pretrained.

These flags determine the torch.compile cache key. If they drift, the
shipped runtime image's compile cache will silently miss and customers
will see 15+ minute first-call latencies. The tests below pin the
ship-path invariants so a careless edit fails CI.
"""

from __future__ import annotations

import os
import sys

import pytest

# Put examples/diffusers/ on sys.path so ``ltx23.config`` resolves
# regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ltx23.config import (  # noqa: E402
    LTX23_FP4_KWARGS,
    LTX23_STANDARD_KWARGS,
    fp4_kwargs,
    standard_kwargs,
)


def test_standard_kwargs_returns_fresh_copy() -> None:
    """Mutating a returned dict must not poison the module-level constant."""
    a = standard_kwargs()
    a["torch_compile_kwargs"]["mode"] = "max-autotune"
    a["ltx2_vae_tiling"] = False
    b = standard_kwargs()
    assert b["torch_compile_kwargs"]["mode"] == "default"
    assert b["ltx2_vae_tiling"] is True


def test_standard_kwargs_invariants() -> None:
    """Pin the ship-path flags. Changing any of these requires a cache rebake."""
    k = standard_kwargs()
    assert k["enable_torch_compile"] is True
    assert k["enable_torch_compile_text_encoder"] is True
    assert k["torch_compile_kwargs"]["backend"] == "inductor"
    assert k["torch_compile_kwargs"]["fullgraph"] is False
    assert k["torch_compile_kwargs"]["mode"] == "default"
    assert k["ltx2_vae_tiling"] is True
    assert k["dit_cpu_offload"] is False
    assert k["vae_cpu_offload"] is False
    assert k["text_encoder_cpu_offload"] is False
    assert k["ltx2_refine_enabled"] is True
    assert k["ltx2_refine_num_inference_steps"] == 2
    assert k["ltx2_refine_guidance_scale"] == 1.0
    assert k["ltx2_refine_add_noise"] is True


def test_fp4_kwargs_overlays_standard() -> None:
    """FP4 path overrides only torch_compile_kwargs; the rest of the standard
    config (vae tiling, refine, offload flags) must survive."""
    f = fp4_kwargs()
    assert f["torch_compile_kwargs"]["fullgraph"] is True
    assert f["torch_compile_kwargs"]["mode"] == "max-autotune-no-cudagraphs"
    assert f["torch_compile_kwargs"]["backend"] == "inductor"
    # Non-overlaid flags carry through.
    assert f["ltx2_vae_tiling"] is True
    assert f["enable_torch_compile"] is True
    assert f["ltx2_refine_enabled"] is True


def test_fp4_kwargs_returns_fresh_copy() -> None:
    """Same defensive-copy guarantee as standard_kwargs."""
    a = fp4_kwargs()
    a["torch_compile_kwargs"]["mode"] = "default"
    a["ltx2_vae_tiling"] = False
    b = fp4_kwargs()
    assert b["torch_compile_kwargs"]["mode"] == "max-autotune-no-cudagraphs"
    assert b["ltx2_vae_tiling"] is True


def test_module_constants_not_mutated_by_calls() -> None:
    """After many calls + mutations, the module dicts must be untouched."""
    for _ in range(5):
        d = standard_kwargs()
        d.clear()
        d2 = fp4_kwargs()
        d2.clear()
    assert LTX23_STANDARD_KWARGS["torch_compile_kwargs"]["mode"] == "default"
    assert LTX23_STANDARD_KWARGS["ltx2_vae_tiling"] is True
    assert LTX23_FP4_KWARGS["torch_compile_kwargs"]["fullgraph"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
