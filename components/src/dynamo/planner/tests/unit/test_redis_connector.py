# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock, patch

import pytest

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.redis_connector import RedisConnector
from dynamo.planner.errors import EmptyTargetReplicasError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.fixture
def mock_redis_client():
    client = AsyncMock()
    client.hgetall = AsyncMock(return_value={})
    client.hset = AsyncMock()
    return client


@pytest.fixture
def connector(mock_redis_client):
    with patch(
        "dynamo.planner.connectors.redis_connector.redis_asyncio.from_url",
        return_value=mock_redis_client,
    ):
        return RedisConnector(
            "test-namespace",
            model_name="test-model",
            redis_url="redis://localhost:6379",
        )


def test_requires_model_name():
    with pytest.raises(ValueError, match="Model name is required"):
        RedisConnector(
            "test-namespace", model_name=None, redis_url="redis://localhost:6379"
        )


def test_requires_redis_url(monkeypatch):
    monkeypatch.delenv("DYNAMO_REDIS_URL", raising=False)
    with pytest.raises(ValueError, match="redis_url is required"):
        RedisConnector("test-namespace", model_name="test-model", redis_url=None)


def test_redis_url_falls_back_to_env(mock_redis_client, monkeypatch):
    monkeypatch.setenv("DYNAMO_REDIS_URL", "redis://from-env:6379")
    with patch(
        "dynamo.planner.connectors.redis_connector.redis_asyncio.from_url",
        return_value=mock_redis_client,
    ) as mock_from_url:
        RedisConnector("test-namespace", model_name="test-model")
        mock_from_url.assert_called_once_with(
            "redis://from-env:6379", decode_responses=True
        )


def test_key_uses_hash_tag_on_model_name(connector):
    assert connector._key == "mm:dynamo_planner_target:{test-model}"


@pytest.mark.asyncio
async def test_set_component_replicas_disagg_writes_both_roles(
    connector, mock_redis_client
):
    await connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=3
            ),
            TargetReplica(
                sub_component_type=SubComponentType.DECODE, desired_replicas=5
            ),
        ]
    )
    mock_redis_client.hset.assert_called_once()
    args, kwargs = mock_redis_client.hset.call_args
    assert args[0] == "mm:dynamo_planner_target:{test-model}"
    mapping = kwargs["mapping"]
    assert mapping["prefill"] == 3
    assert mapping["decode"] == 5
    assert "updated_at" in mapping


@pytest.mark.asyncio
async def test_set_component_replicas_single_role_omits_the_other(
    connector, mock_redis_client
):
    """A dedicated PrefillPlanner only ever sends its own role -- the write
    must not clobber whatever a sibling DecodePlanner last wrote for
    "decode" under the same model_name key."""
    await connector.set_component_replicas(
        [
            TargetReplica(
                sub_component_type=SubComponentType.PREFILL, desired_replicas=4
            )
        ]
    )
    mapping = mock_redis_client.hset.call_args.kwargs["mapping"]
    assert mapping["prefill"] == 4
    assert "decode" not in mapping


@pytest.mark.asyncio
async def test_set_component_replicas_empty_raises(connector):
    with pytest.raises(EmptyTargetReplicasError):
        await connector.set_component_replicas([])


@pytest.mark.asyncio
async def test_add_component_increments_from_current(connector, mock_redis_client):
    mock_redis_client.hgetall.return_value = {"prefill": "2", "decode": "1"}
    await connector.add_component(SubComponentType.PREFILL)
    mapping = mock_redis_client.hset.call_args.kwargs["mapping"]
    assert mapping["prefill"] == 3


@pytest.mark.asyncio
async def test_remove_component_floors_at_zero(connector, mock_redis_client):
    mock_redis_client.hgetall.return_value = {"prefill": "0", "decode": "0"}
    await connector.remove_component(SubComponentType.DECODE)
    mapping = mock_redis_client.hset.call_args.kwargs["mapping"]
    assert mapping["decode"] == 0


def test_get_worker_info_uses_defaults(connector):
    info = connector.get_worker_info(SubComponentType.PREFILL, backend="vllm")
    assert info.model_name == "test-model"


@pytest.mark.asyncio
async def test_validate_deployment_and_wait_are_no_ops(connector):
    await connector.validate_deployment()
    await connector.wait_for_deployment_ready()


@pytest.mark.asyncio
async def test_get_model_name_returns_constructor_value(connector):
    assert await connector.get_model_name() == "test-model"
