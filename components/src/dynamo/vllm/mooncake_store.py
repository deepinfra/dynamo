# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Announce the Mooncake store of vLLM's ``MooncakeStoreConnector`` to the router.

With ``--router-shared-cache-type mooncake-store`` the frontend credits each worker
with the chunks its own store holds. Workers announcing the same store share them.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from dynamo.llm import ModelRuntimeConfig

VLLM_MOONCAKE_STORE_RUNTIME_KEY = "vllm_mooncake_store"
# Shared with the SGLang HiCache worker: the store master's KV-event endpoint
# (`mooncake_master --enable_kv_events --kv_events_bind_endpoint`), as the router reaches it.
MOONCAKE_KV_EVENTS_ENDPOINT_ENV = "DYN_MOONCAKE_KV_EVENTS_ENDPOINT"
MOONCAKE_STORE_CONNECTOR = "MooncakeStoreConnector"

logger = logging.getLogger(__name__)


def _mooncake_store_extra_config(kv_transfer_config: Any) -> Optional[dict]:
    """The connector's extra config, directly or inside a ``MultiConnector``."""
    if kv_transfer_config is None:
        return None
    extra = kv_transfer_config.kv_connector_extra_config or {}
    if kv_transfer_config.kv_connector == MOONCAKE_STORE_CONNECTOR:
        return extra
    if kv_transfer_config.kv_connector == "MultiConnector":
        for connector in extra.get("connectors", []):
            if connector.get("kv_connector") == MOONCAKE_STORE_CONNECTOR:
                return connector.get("kv_connector_extra_config") or {}
    return None


def publish_vllm_mooncake_store(
    runtime_config: ModelRuntimeConfig, vllm_config: Any
) -> bool:
    """Announce this worker's store when the connector and the endpoint are configured."""
    endpoint = os.getenv(MOONCAKE_KV_EVENTS_ENDPOINT_ENV)
    if not endpoint:
        return False
    extra = _mooncake_store_extra_config(vllm_config.kv_transfer_config)
    if extra is None:
        logger.warning(
            "%s is set but %s is not configured; not announcing a Mooncake store",
            MOONCAKE_KV_EVENTS_ENDPOINT_ENV,
            MOONCAKE_STORE_CONNECTOR,
        )
        return False
    # The connector's chunk group id is
    # vllm-mooncake-store:[{cache_prefix}@]{model}@{chunk hash hex}.
    cache_prefix = str(extra.get("cache_prefix") or "")
    model = vllm_config.model_config.model.rstrip("/").split("/")[-1]
    scope = f"{cache_prefix}@" if cache_prefix else ""
    runtime_config.set_engine_specific(
        VLLM_MOONCAKE_STORE_RUNTIME_KEY,
        json.dumps(
            {
                "kv_events_endpoint": endpoint,
                "group_prefix": f"vllm-mooncake-store:{scope}{model}@",
            }
        ),
    )
    return True
