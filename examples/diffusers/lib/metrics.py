# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics for the generic video-pipeline subprocess pool.

All series are prefixed ``video_pool_`` and carry a ``model`` label whose
value is set per-pod (via ``--served-model-name`` or the model factory
choice). A dedicated ``CollectorRegistry`` keeps these series isolated
from the default prometheus_client registry.

These metrics flow through Dynamo's worker-side ``system_status_server``
``/metrics`` endpoint via ``register_engine_metrics_callback``, which
auto-injects ``dynamo_namespace`` / ``dynamo_component`` /
``dynamo_endpoint`` / ``model_name`` hierarchy labels alongside our
``model`` label.

Pool subprocesses do not have a Dynamo endpoint and therefore do not
register a callback -- all accounting happens in the parent. The
``route()`` FATAL branch is the single point that records ``cuda_fault``
so subprocess-side state changes are observed exactly once.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

VIDEO_REGISTRY = CollectorRegistry()

POOL_SIZE = Gauge(
    "video_pool_size",
    "Current number of live pool subprocesses",
    ["model"],
    registry=VIDEO_REGISTRY,
)
POOL_SPAWN_TOTAL = Counter(
    "video_pool_spawn_total",
    "Pool subprocess spawns",
    ["model", "shape_key"],
    registry=VIDEO_REGISTRY,
)
POOL_EVICTION_TOTAL = Counter(
    "video_pool_eviction_total",
    "Pool LRU evictions",
    ["model", "shape_key"],
    registry=VIDEO_REGISTRY,
)
POOL_REQUEST_TOTAL = Counter(
    "video_pool_request_total",
    "Pool routed requests by terminal status (clean-response path)",
    ["model", "shape_key", "status"],  # status: DONE, ERROR, FATAL
    registry=VIDEO_REGISTRY,
)
POOL_REQUEST_LATENCY = Histogram(
    "video_pool_request_latency_seconds",
    "End-to-end pool request latency (route entry to clean DONE response)",
    ["model", "shape_key"],
    buckets=(1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 1800),
    registry=VIDEO_REGISTRY,
)
POOL_COLD_SPAWN_SECONDS = Histogram(
    "video_pool_cold_spawn_seconds",
    "Time from subprocess fork to READY message",
    ["model", "shape_key"],
    buckets=(10, 25, 50, 100, 150, 200, 250, 300),
    registry=VIDEO_REGISTRY,
)
POOL_SUBPROCESS_FAILURE_TOTAL = Counter(
    "video_pool_subprocess_failure_total",
    "Subprocess failures by cause",
    ["model", "reason"],
    # reason: cuda_fault, spawn_timeout, spawn_eof, spawn_parse_error,
    #         gen_timeout, gen_eof, desync, parse_error, send_failed.
    # parent_disconnect is in the conceptual domain but is observed only
    # from the subprocess (which has no endpoint to emit from), so no
    # call site increments it today.
    registry=VIDEO_REGISTRY,
)
