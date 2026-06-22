# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2.3 worker entry point for Dynamo (non-streaming).

Registers a FastVideo VideoGenerator as a Dynamo backend endpoint
compatible with the ``/v1/videos`` frontend endpoint. The endpoint
generates a full video clip from the request parameters and returns
it as a single response containing the complete MP4 file
base64-encoded in ``data[0].b64_json``.

Generation parameters (size, fps, num_frames, etc.) are taken from
the request body's ``nvext`` field, so the same worker instance can
serve requests with different resolutions and quality settings
without restarting.

One request at a time (asyncio.Lock — VideoGenerator is not
re-entrant).

Usage:
  python worker.py [--model MODEL] [--num-gpus N] [--enable-optimizations]
                   [--attention-backend ATTENTION_BACKEND]

This module is the LTX-2.3-specific entry. The top-level
``examples/diffusers/worker.py`` shim is what the production
deployment / pool-subprocess invocations actually execute; the shim
dispatches pool-worker invocations to ``lib.pool`` directly (to skip
heavy imports here) and otherwise calls :func:`main_cli` from this
module.
"""

import argparse
import asyncio
import logging
import os
import sys

import uvloop
from lib.backend import GenericVideoBackend
from lib.dynamo_wiring import get_worker_namespace, register_model
from lib.menu import assert_shape_menu_hash_matches
from lib.metrics import VIDEO_REGISTRY

from dynamo.common.utils.prometheus import register_engine_metrics_callback
from dynamo.runtime import DistributedRuntime

from .factory import load_model

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "FastVideo/LTX-2.3-Distilled-Diffusers"
# FastVideo's deployed LTX-2.3 1080p recipe uses FLASH_ATTN (their optimized
# SM100/SM103 kernels); streaming_demo launch sets FASTVIDEO_ATTENTION_BACKEND=
# FLASH_ATTN. TORCH_SDPA (the prior default, inherited from LTX-2) is the slow
# generic fallback and does NOT reproduce the ~4.5s/1080p result.
DEFAULT_ATTENTION_BACKEND = "FLASH_ATTN"
DEFAULT_MODEL_LABEL = "ltx2-3-distilled"

# Where the LTX-2.3 shapes JSON lives in the production image. Production
# bake-time IMAGE_SHAPE_HASH is computed against this path; if the
# image build moves it, update both ends in lockstep.
DEFAULT_SHAPES_JSON_PATH = os.path.join(os.path.dirname(__file__), "shapes.json")
PRODUCTION_SHAPES_JSON_PATH = "/opt/app/ltx23/shapes.json"


def _attention_backend_choices() -> tuple[str, ...]:
    # Lazy: FastVideo's enum is heavy. Only evaluate when the parent
    # parses CLI args; pool subprocesses don't reach _parse_args.
    from fastvideo.platforms.interface import AttentionBackendEnum

    return tuple(
        backend_name
        for backend_name in AttentionBackendEnum.__members__
        if backend_name != "NO_ATTENTION"
    )


def _parse_args() -> argparse.Namespace:
    choices = _attention_backend_choices()
    parser = argparse.ArgumentParser(
        description="FastVideo Worker for Dynamo (non-streaming)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Name advertised to the Dynamo discovery layer (default: same as --model)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        dest="num_gpus",
        help="Number of GPUs (default: 1)",
    )
    parser.add_argument(
        "--enable-optimizations",
        action="store_true",
        dest="enable_optimizations",
        help="Enable FP4 quantization (if available) and torch.compile",
    )
    parser.add_argument(
        "--attention-backend",
        choices=choices,
        default=DEFAULT_ATTENTION_BACKEND,
        dest="attention_backend",
        help=(
            "Attention backend to set via FASTVIDEO_ATTENTION_BACKEND "
            f"(choices: {', '.join(choices)}; "
            f"default: {DEFAULT_ATTENTION_BACKEND})"
        ),
    )
    return parser.parse_args()


async def _main(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    # Use Kubernetes discovery in-cluster and file discovery for local compose by default.
    discovery_backend = os.environ.get("DYN_DISCOVERY_BACKEND")
    if not discovery_backend:
        discovery_backend = (
            "kubernetes" if os.environ.get("KUBERNETES_SERVICE_HOST") else "file"
        )
    namespace_name = get_worker_namespace()
    logger.info("Using discovery backend: %s", discovery_backend)
    logger.info("Resolved worker namespace: %s", namespace_name)
    # Pass enable_nats=False explicitly: the bundled ai-dynamo-runtime 1.0.0
    # is from before upstream commit af0ff07 ("remove enable_nats usage")
    # and so its DistributedRuntime ctor still requires the 4th positional
    # arg to gate the NATS client. Omitting it defaults to True, which
    # makes the worker hard-fail at startup with "Failed to connect to
    # NATS: Connection refused" because the cluster runs ZMQ-only
    # (DYN_EVENT_PLANE=zmq, deepinfra has no NATS).
    runtime = DistributedRuntime(loop, discovery_backend, "tcp", False)

    component_name = "backend"
    endpoint_name = "generate"
    endpoint = runtime.endpoint(f"{namespace_name}.{component_name}.{endpoint_name}")
    logger.info(
        "Serving endpoint %s/%s/%s", namespace_name, component_name, endpoint_name
    )

    model_label = args.served_model_name or DEFAULT_MODEL_LABEL
    backend = GenericVideoBackend(
        args=args,
        model_factory_callable=load_model,
        model_factory_dotted="ltx23.factory:load_model",
        model_label=model_label,
    )
    await backend.initialize_model()
    # LTX-2.3: eager warm-on-boot is the DEFAULT. Unlike LTX-2's 10-shape menu
    # (~25 min, hence opt-in there), the LTX-2.3 v1 menu is tiny (2 shapes), so
    # warming every shape's pool subprocess at boot is cheap (~minutes). It runs
    # BEFORE we register/serve below, so the first real request per shape is
    # steady-state and traffic routing (register_model) is gated on it; existing
    # warm pods cover traffic during a new pod's boot (min-instances >= 1). The
    # startup-probe budget in backend kubernetes_utils_async.py is sized to cover
    # this. Opt out with LTX23_EAGER_WARM=0 for debugging only.
    if os.environ.get("LTX23_EAGER_WARM", "1") != "0":
        await backend.preflight()

    # Wire video_pool_* series into the Dynamo runtime's /metrics scrape.
    # Registers unconditionally: in legacy in-process mode (VIDEO_POOL_MODE=0)
    # the series stay zero-valued, which is harmless and means dashboards
    # don't break when toggling the env var.
    register_engine_metrics_callback(
        endpoint=endpoint,
        registry=VIDEO_REGISTRY,
        metric_prefix_filters=["video_"],
        namespace_name=namespace_name,
        component_name=component_name,
        endpoint_name=endpoint_name,
        model_name=backend.served_model_name,
    )

    # Pool metrics flow through dynamo's system-status server, which is
    # disabled by default. If DYN_SYSTEM_PORT isn't set to a non-negative
    # value, the /metrics endpoint never binds and our registered callback
    # sits dormant -- operators would see zero pool visibility in
    # production. Warn loudly so the misconfiguration is obvious in pod
    # startup logs.
    _system_port = os.environ.get("DYN_SYSTEM_PORT", "-1")
    try:
        _system_port_int = int(_system_port)
    except ValueError:
        _system_port_int = -1
    if _system_port_int < 0:
        logger.warning(
            "DYN_SYSTEM_PORT is not set (or is %r); pool metrics will NOT "
            "be exposed. Set DYN_SYSTEM_PORT to a port (e.g. 9090) and "
            "DYN_SYSTEM_HOST=0.0.0.0 to enable the /metrics endpoint. "
            "See examples/diffusers/ltx23/RUNBOOK.md § Metrics.",
            _system_port,
        )
    else:
        logger.info(
            "Pool metrics enabled on DYN_SYSTEM_PORT=%d",
            _system_port_int,
        )

    try:
        await asyncio.gather(
            endpoint.serve_endpoint(backend.create_video),  # type: ignore[arg-type]
            register_model(endpoint, backend.served_model_name, backend.model_name),
        )
    finally:
        if backend.pool is not None:
            logger.info("shutting down subprocess pool")
            await backend.pool.shutdown()


def main_cli() -> None:
    """
    LTX-2.3 worker entry. Called by the top-level ``worker.py`` shim
    when the invocation is NOT a pool-worker subprocess (pool-worker
    dispatch happens in the shim itself, before any of these heavy
    imports run).

    Contract with the shim: ``--pool-worker`` is already handled.
    Do not re-dispatch here.
    """
    args = _parse_args()
    logging.basicConfig(
        level=(
            logging.DEBUG
            if os.environ.get("FASTVIDEO_LOG_LEVEL") == "DEBUG"
            else logging.INFO
        ),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    # Default the shapes-JSON path for both the in-container production
    # location and a dev-run-from-source location. The pool subprocess
    # also inherits this via env, so they see the same shape file.
    if "WARMUP_SHAPES_JSON_PATH" not in os.environ:
        if os.path.isfile(PRODUCTION_SHAPES_JSON_PATH):
            os.environ["WARMUP_SHAPES_JSON_PATH"] = PRODUCTION_SHAPES_JSON_PATH
        else:
            os.environ["WARMUP_SHAPES_JSON_PATH"] = DEFAULT_SHAPES_JSON_PATH
    # Refuse to start if image cache and shape menu disagree. Runs before
    # any model load or networking so we fail fast with a clear message.
    assert_shape_menu_hash_matches(os.environ["WARMUP_SHAPES_JSON_PATH"])
    uvloop.install()
    asyncio.run(_main(args))


if __name__ == "__main__":
    # If invoked directly (e.g. `python3 -m ltx23.worker`), guard the
    # entry: pool-worker dispatch lives in the top-level shim. Direct
    # invocation with --pool-worker would skip that dispatch path.
    if "--pool-worker" in sys.argv:
        raise SystemExit(
            "ltx23.worker invoked directly with --pool-worker; "
            "use the top-level worker.py shim, which dispatches "
            "to lib.pool._pool_worker_dispatch_if_requested first."
        )
    main_cli()
