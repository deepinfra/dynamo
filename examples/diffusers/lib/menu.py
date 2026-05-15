# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shape-menu hash check.

The hash is the load-bearing identifier across the whole pipeline:
  - bake step embeds it in IMAGE_SHAPE_HASH (Dockerfile env)
  - worker.py asserts it matches at boot (refuses to start on mismatch)
  - backend's i model-add admission check rejects images whose hash
    doesn't match the vendored menu
  - backend/tests/test_ltx_shape_menu.py reimplements the algorithm to
    pin it against drift

If the algorithm changes here, the backend test MUST be updated to
match in the same change. The hash is the first 8 hex chars of
sha256(json.dumps(sorted(shape_tuples), separators=(',',':'))).
"""

import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)


def compute_menu_hash(shapes_json_path: str) -> tuple[str, int]:
    """
    Hash the canonical shape-tuple list from a shapes JSON file. Returns
    (hash, shape_count) so callers can log both without re-reading.
    """
    with open(shapes_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    shapes = sorted(
        (int(s["width"]), int(s["height"]), int(s["num_frames"])) for s in cfg["shapes"]
    )
    canonical = json.dumps(shapes, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:8], len(shapes)


def assert_shape_menu_hash_matches(default_shapes_path: str) -> None:
    """
    Refuse to start if the image's baked compile cache doesn't match the
    shape menu the worker is about to serve. The image's Dockerfile sets
    IMAGE_SHAPE_HASH at bake time; we recompute the same hash here from
    the shapes JSON and abort on mismatch.

    The shapes file is found via the ``WARMUP_SHAPES_JSON_PATH`` env var,
    falling back to ``default_shapes_path`` (which the caller supplies
    per-model -- e.g., for LTX-2: ``/opt/app/ltx2/shapes.json``).

    If IMAGE_SHAPE_HASH is unset (dev runs, or images built before this
    invariant landed), we log a warning and continue. Production images
    are expected to have it set.

    The procedure for resolving a mismatch is documented in the
    per-model RUNBOOK.
    """
    expected = os.environ.get("IMAGE_SHAPE_HASH")
    if not expected:
        logger.warning(
            "IMAGE_SHAPE_HASH is unset; skipping image/menu hash check. "
            "This is expected for dev runs and for images built before "
            "the bake step started writing IMAGE_SHAPE_HASH. Production "
            "images must set it -- see the RUNBOOK."
        )
        return

    shapes_json = os.environ.get("WARMUP_SHAPES_JSON_PATH", default_shapes_path)
    if not os.path.isfile(shapes_json):
        raise RuntimeError(
            f"shapes JSON not found at {shapes_json}. "
            f"Cannot verify the image/menu hash. Set "
            f"WARMUP_SHAPES_JSON_PATH to the correct location, or "
            f"check your image build."
        )

    actual, shape_count = compute_menu_hash(shapes_json)
    if actual != expected:
        raise RuntimeError(
            f"FATAL: image/menu shape-hash mismatch -- refusing to start.\n"
            f"\n"
            f"  Image baked with hash: {expected}  (env IMAGE_SHAPE_HASH, set at "
            f"bake time)\n"
            f"  Current menu hash:     {actual}  (recomputed from {shapes_json})\n"
            f"\n"
            f"This means the compile cache shipped in the image was produced "
            f"for a different shape menu than the one this worker is about to "
            f"serve. Customer requests for any shape in the gap would hit a "
            f"15+ minute torch.compile, violating our latency SLO.\n"
            f"\n"
            f"Either:\n"
            f"  A) Roll back the model config to the last image whose tag ends "
            f"in -{actual}, OR\n"
            f"  B) Build a new image for the current menu (~3 hours on B200).\n"
            f"\n"
            f"Procedures: see the per-model RUNBOOK.md"
        )

    logger.info(
        "image/menu shape-hash check OK (%s); %d shapes in menu",
        actual,
        shape_count,
    )
