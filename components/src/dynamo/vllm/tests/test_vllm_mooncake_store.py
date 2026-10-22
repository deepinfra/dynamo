# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.vllm.mooncake_store import (
    MOONCAKE_KV_EVENTS_ENDPOINT_ENV,
    VLLM_MOONCAKE_STORE_RUNTIME_KEY,
    publish_vllm_mooncake_store,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

ENDPOINT = "tcp://mooncake-master:5558"


def vllm_config(kv_connector, extra=None, model="/models/deepseek-ai/DeepSeek-V3.2/"):
    kv_transfer_config = (
        None
        if kv_connector is None
        else SimpleNamespace(
            kv_connector=kv_connector, kv_connector_extra_config=extra or {}
        )
    )
    return SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        model_config=SimpleNamespace(model=model),
    )


def announced(runtime_config):
    ((key, value),) = (
        c.args for c in runtime_config.set_engine_specific.call_args_list
    )
    assert key == VLLM_MOONCAKE_STORE_RUNTIME_KEY
    return json.loads(value)


@pytest.mark.parametrize(
    "config, group_prefix",
    [
        (
            vllm_config("MooncakeStoreConnector"),
            "vllm-mooncake-store:DeepSeek-V3.2@",
        ),
        (
            vllm_config("MooncakeStoreConnector", {"cache_prefix": "team-a"}),
            "vllm-mooncake-store:team-a@DeepSeek-V3.2@",
        ),
        (
            vllm_config(
                "MultiConnector",
                {
                    "connectors": [
                        {"kv_connector": "NixlConnector"},
                        {
                            "kv_connector": "MooncakeStoreConnector",
                            "kv_connector_extra_config": {"cache_prefix": "b"},
                        },
                    ]
                },
            ),
            "vllm-mooncake-store:b@DeepSeek-V3.2@",
        ),
    ],
)
def test_announces_the_connectors_store(monkeypatch, config, group_prefix):
    monkeypatch.setenv(MOONCAKE_KV_EVENTS_ENDPOINT_ENV, ENDPOINT)
    runtime_config = Mock()
    assert publish_vllm_mooncake_store(runtime_config, config)
    assert announced(runtime_config) == {
        "kv_events_endpoint": ENDPOINT,
        "group_prefix": group_prefix,
    }


@pytest.mark.parametrize(
    "endpoint, config",
    [
        (None, vllm_config("MooncakeStoreConnector")),
        ("", vllm_config("MooncakeStoreConnector")),
        (ENDPOINT, vllm_config(None)),
        (ENDPOINT, vllm_config("OffloadingConnector")),
        (ENDPOINT, vllm_config("MultiConnector", {"connectors": []})),
    ],
)
def test_announces_nothing_without_endpoint_and_connector(
    monkeypatch, endpoint, config
):
    if endpoint is None:
        monkeypatch.delenv(MOONCAKE_KV_EVENTS_ENDPOINT_ENV, raising=False)
    else:
        monkeypatch.setenv(MOONCAKE_KV_EVENTS_ENDPOINT_ENV, endpoint)
    runtime_config = Mock()
    assert not publish_vllm_mooncake_store(runtime_config, config)
    runtime_config.set_engine_specific.assert_not_called()
