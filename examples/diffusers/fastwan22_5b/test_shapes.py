# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for fastwan22_5b/shapes.json + the menu-hash algorithm.

The hash is the load-bearing identifier across the whole pipeline:

  - bake step embeds it in IMAGE_SHAPE_HASH (Dockerfile env)
  - lib/menu.py asserts it matches at boot (refuses to start on mismatch)
  - backend's i model-add admission check rejects images whose hash
    doesn't match the vendored menu

If the algorithm or the menu drifts, every layer downstream is wrong.
The canonical algorithm lives in examples/diffusers/lib/menu.py; the
backend's menu-hash admission check must agree with it.

This module reimplements the canonical hash inline so it can run in CI
without FastVideo / torch. A second cross-check test imports the live
implementation from lib.menu and skips if heavy deps aren't available
-- that's the canary against the in-container algorithm drifting.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES_JSON = os.path.join(HERE, "shapes.json")

# Pinned hash for the current 2-shape FastWan2.2-TI2V-5B v1 menu
# (landscape 832x480 + portrait 480x832, both @ 81 frames). Updating
# shapes.json -- adding, removing, or renaming a shape -- changes this
# value, requires a new image bake, and requires updating this fixture.
EXPECTED_HASH = "e0c0ed3c"
EXPECTED_SHAPE_COUNT = 2

# Per-shape activation budget. A shape whose width*height*frames is too
# large makes the decoder's intermediate tensor exceed 2^31 elements,
# tripping PyTorch's int32 indexing limit in F.pad. Any shape we add must
# stay below that bound (the current 480p shapes are far under it).
INT32_LIMIT = 2**31
ACTIVATION_BUDGET = INT32_LIMIT // 2


def _canonical_menu_hash(shapes_json_path: str) -> tuple[str, int]:
    """Reimplements ``lib.menu.compute_menu_hash`` inline so this test
    file has no FastVideo / torch dependency. Must stay byte-identical to
    the canonical algorithm in ``examples/diffusers/lib/menu.py::compute_menu_hash``
    (the same algorithm the bake step and the backend menu-hash admission
    check rely on).
    """
    with open(shapes_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    shapes = sorted(
        (int(s["width"]), int(s["height"]), int(s["num_frames"])) for s in cfg["shapes"]
    )
    canonical = json.dumps(shapes, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8], len(shapes)


def _load_shapes() -> dict:
    with open(SHAPES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def test_warmup_shapes_parses() -> None:
    cfg = _load_shapes()
    assert cfg["model"] == "/data/default"
    assert "shapes" in cfg
    assert isinstance(cfg["shapes"], list)


def test_warmup_shapes_required_keys() -> None:
    cfg = _load_shapes()
    for shape in cfg["shapes"]:
        for key in ("width", "height", "num_frames"):
            assert key in shape, f"shape {shape} missing {key}"
            assert isinstance(shape[key], int)
            assert shape[key] > 0


def test_warmup_shapes_dimensions_multiple_of_16() -> None:
    """Wan2.1 needs dims that are multiples of 16 (8x VAE spatial stride x
    2x2 DiT patch); non-multiples crash decode/patchify."""
    cfg = _load_shapes()
    for shape in cfg["shapes"]:
        assert shape["width"] % 16 == 0, f"width not /16: {shape}"
        assert shape["height"] % 16 == 0, f"height not /16: {shape}"


def test_warmup_shapes_within_activation_budget() -> None:
    """Reject any shape whose width*height*frames overflows the int32 limit
    that broke 1080p@241f. Coarse proxy for VAE decode safety."""
    cfg = _load_shapes()
    for shape in cfg["shapes"]:
        product = shape["width"] * shape["height"] * shape["num_frames"]
        assert product < ACTIVATION_BUDGET, (
            f"shape {shape} has product {product:,} >= "
            f"budget {ACTIVATION_BUDGET:,} -- VAE decode will trip int32 indexing"
        )


def test_canonical_menu_hash_is_deterministic() -> None:
    h1, n1 = _canonical_menu_hash(SHAPES_JSON)
    h2, n2 = _canonical_menu_hash(SHAPES_JSON)
    assert h1 == h2
    assert n1 == n2 == EXPECTED_SHAPE_COUNT


def test_canonical_menu_hash_matches_expected() -> None:
    """The vendored menu's hash must match what the runtime image was
    baked with. Update both this fixture and the image tag together."""
    actual, count = _canonical_menu_hash(SHAPES_JSON)
    assert actual == EXPECTED_HASH, (
        f"Menu hash drift: warmup_shapes.json hashes to {actual} but "
        f"EXPECTED_HASH={EXPECTED_HASH}. If the menu change is intentional, "
        f"update EXPECTED_HASH and rebake the runtime image."
    )
    assert count == EXPECTED_SHAPE_COUNT


def test_canonical_menu_hash_changes_when_menu_changes(tmp_path) -> None:
    """A menu edit must produce a different hash; otherwise the boot
    assertion can't catch drift."""
    cfg = _load_shapes()
    cfg["shapes"].append({"width": 64, "height": 64, "num_frames": 17})
    new_path = tmp_path / "shapes_modified.json"
    new_path.write_text(json.dumps(cfg))

    original_hash, _ = _canonical_menu_hash(SHAPES_JSON)
    modified_hash, modified_count = _canonical_menu_hash(str(new_path))
    assert modified_hash != original_hash
    assert modified_count == EXPECTED_SHAPE_COUNT + 1


def test_canonical_menu_hash_order_independent(tmp_path) -> None:
    """Shape order in the JSON must not affect the hash; reviewers can
    reorder for readability without breaking caches."""
    cfg = _load_shapes()
    cfg["shapes"] = list(reversed(cfg["shapes"]))
    reversed_path = tmp_path / "shapes_reversed.json"
    reversed_path.write_text(json.dumps(cfg))
    h1, _ = _canonical_menu_hash(SHAPES_JSON)
    h2, _ = _canonical_menu_hash(str(reversed_path))
    assert h1 == h2


def test_lib_compute_menu_hash_matches_canonical() -> None:
    """Cross-check: ``lib.menu.compute_menu_hash``'s live implementation
    must agree with the inline algorithm. Skipped if the package isn't
    importable in this env (e.g., CI runs without the package root on
    sys.path)."""
    # Put examples/diffusers/ on sys.path so ``lib.menu`` resolves.
    sys.path.insert(0, os.path.dirname(HERE))
    # Pre-declare so static analyzers (CodeQL) don't flag the call site
    # below as potentially-unbound. pytest.skip() raises so the call is
    # actually unreachable on the ImportError path, but the analyzer
    # doesn't model that.
    compute_menu_hash = None
    try:
        from lib.menu import compute_menu_hash  # type: ignore
    except ImportError as exc:
        pytest.skip(f"lib.menu not importable in this env: {exc}")
    canonical_hash, _ = _canonical_menu_hash(SHAPES_JSON)
    library_hash, _ = compute_menu_hash(SHAPES_JSON)
    assert canonical_hash == library_hash


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
