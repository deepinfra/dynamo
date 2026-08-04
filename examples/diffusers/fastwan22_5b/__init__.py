# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastWan2.2-TI2V-5B video-pipeline integration.

Model-specific glue (factory, shape menu, warmup) lives here. Generic
infrastructure (pool, backend, metrics, models, menu-hash) lives in the
sibling ``lib`` package; shared operational docs live in ``ltx23/``
(RUNBOOK, CACHING, ARCHITECTURE) since the machinery is common.
"""
