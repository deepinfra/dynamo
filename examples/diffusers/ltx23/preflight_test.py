#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Standalone smoke-test of GenericVideoBackend.preflight() without spinning
up the Dynamo distributed runtime. Loads the model, runs the preflight
loop, exits. Logs per-shape timings and total wall time.

Used to verify that a candidate ship image's preflight will:
  - Find ltx23/shapes.json
  - Successfully warm every shape's in-memory cache
  - Complete in a reasonable wall-clock budget for pod startup
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

# Put examples/diffusers/ on sys.path so the lib / ltx23 packages
# resolve. This script lives at examples/diffusers/ltx23/preflight_test.py;
# parent dir = examples/diffusers/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="FastVideo/LTX-2.3-Distilled-Diffusers",
        help="HuggingFace model identifier",
    )
    p.add_argument(
        "--gpu-uuid",
        required=True,
        help="GPU UUID to pin to (sets CUDA_VISIBLE_DEVICES before torch import)",
    )
    return p.parse_args()


async def _amain(model: str) -> int:
    from lib.backend import GenericVideoBackend
    from ltx23.factory import load_model

    backend_args = argparse.Namespace(
        model=model,
        served_model_name=None,
        num_gpus=1,
        enable_optimizations=False,
        attention_backend="TORCH_SDPA",
    )

    backend = GenericVideoBackend(
        args=backend_args,
        model_factory_callable=load_model,
        model_factory_dotted="ltx23.factory:load_model",
        model_label="ltx2-3-distilled",
    )

    # Match production: default the shapes-JSON path so preflight can
    # find the menu without an explicit env var.
    if "WARMUP_SHAPES_JSON_PATH" not in os.environ:
        os.environ["WARMUP_SHAPES_JSON_PATH"] = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "shapes.json"
        )

    t_init = time.perf_counter()
    print("[preflight-test] initialize_model() ...", flush=True)
    await backend.initialize_model()
    print(
        "[preflight-test] initialize_model() done in %.1fs"
        % (time.perf_counter() - t_init),
        flush=True,
    )

    t_pre = time.perf_counter()
    print("[preflight-test] preflight() ...", flush=True)
    await backend.preflight()
    print(
        "[preflight-test] preflight() done in %.1fs" % (time.perf_counter() - t_pre),
        flush=True,
    )

    print(
        "[preflight-test] TOTAL boot-equivalent time: %.1fs"
        % (time.perf_counter() - t_init),
        flush=True,
    )
    return 0


def main() -> int:
    args = _parse_args()

    # Pin GPU before any torch import (the factory imports torch).
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid

    # Match worker.py's logging setup so preflight's log lines look the same.
    logging.basicConfig(
        level=(
            logging.DEBUG
            if os.environ.get("FASTVIDEO_LOG_LEVEL") == "DEBUG"
            else logging.INFO
        ),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    return asyncio.run(_amain(args.model))


if __name__ == "__main__":
    sys.exit(main())
