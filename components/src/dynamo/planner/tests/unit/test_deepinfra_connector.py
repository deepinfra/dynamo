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

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.deepinfra_connector import DeepInfraConnector
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
        "dynamo.planner.connectors.deepinfra_connector.redis_asyncio.from_url",
        return_value=mock_redis_client,
    ):
        return DeepInfraConnector(
            "test-namespace",
            model_name="test-model",
            redis_url="redis://localhost:6379",
        )


def test_requires_model_name():
    with pytest.raises(ValueError, match="Model name is required"):
        DeepInfraConnector(
            "test-namespace", model_name=None, redis_url="redis://localhost:6379"
        )


def test_requires_redis_url(monkeypatch):
    monkeypatch.delenv("DYN_REDIS_URL", raising=False)
    with pytest.raises(ValueError, match="redis_url is required"):
        DeepInfraConnector("test-namespace", model_name="test-model", redis_url=None)


def test_redis_url_falls_back_to_env(mock_redis_client, monkeypatch):
    monkeypatch.setenv("DYN_REDIS_URL", "redis://from-env:6379")
    with patch(
        "dynamo.planner.connectors.deepinfra_connector.redis_asyncio.from_url",
        return_value=mock_redis_client,
    ) as mock_from_url:
        DeepInfraConnector("test-namespace", model_name="test-model")
        mock_from_url.assert_called_once_with(
            "redis://from-env:6379", decode_responses=True
        )


def test_key_uses_hash_tag_on_namespace_and_model_name(connector):
    assert connector._key == "dynamo:planner:target:{test-namespace:test-model}"


def test_key_isolates_same_model_name_across_namespaces(mock_redis_client):
    with patch(
        "dynamo.planner.connectors.deepinfra_connector.redis_asyncio.from_url",
        return_value=mock_redis_client,
    ):
        a = DeepInfraConnector(
            "namespace-a", model_name="shared-model", redis_url="redis://localhost:6379"
        )
        b = DeepInfraConnector(
            "namespace-b", model_name="shared-model", redis_url="redis://localhost:6379"
        )
    assert a._key != b._key


def test_get_model_name_is_sync(connector):
    assert connector.get_model_name() == "test-model"


def test_model_name_case_is_preserved(mock_redis_client):
    """Unlike VirtualConnector/KubernetesConnector, this connector never
    matches model_name against MDC entries, so there's no reason to fold
    case -- and doing so risks merging two distinct, differently-cased
    model names onto the same Redis key."""
    with patch(
        "dynamo.planner.connectors.deepinfra_connector.redis_asyncio.from_url",
        return_value=mock_redis_client,
    ):
        connector = DeepInfraConnector(
            "test-namespace",
            model_name="Some-Mixed-Case-Model",
            redis_url="redis://localhost:6379",
        )
    assert connector.model_name == "Some-Mixed-Case-Model"
    assert connector.get_model_name() == "Some-Mixed-Case-Model"
    assert "{test-namespace:Some-Mixed-Case-Model}" in connector._key


def test_get_gpu_counts_returns_none_none(connector):
    assert connector.get_gpu_counts() == (None, None)


def test_get_worker_info_uses_defaults(connector):
    info = connector.get_worker_info(SubComponentType.PREFILL, backend="vllm")
    assert info.model_name == "test-model"


def test_get_worker_info_delegates_to_provider_when_wired(mock_redis_client):
    """When construct_environment wires in a worker_info_provider (a
    RuntimeFpmProvider), get_worker_info must delegate to it so the planner
    reads real worker capabilities (e.g. KV cache size) from MDC instead of
    defaults."""
    provider = SimpleNamespace(
        get_worker_info=lambda st, backend: SimpleNamespace(
            model_name="from-provider",
            total_kv_blocks=999,
            kv_cache_block_size=2,
        )
    )
    with patch(
        "dynamo.planner.connectors.deepinfra_connector.redis_asyncio.from_url",
        return_value=mock_redis_client,
    ):
        connector = DeepInfraConnector(
            "test-namespace",
            model_name="test-model",
            redis_url="redis://localhost:6379",
            worker_info_provider=provider,
        )
    info = connector.get_worker_info(SubComponentType.DECODE, backend="vllm")
    assert info.model_name == "from-provider"
    assert info.total_kv_blocks == 999
    assert info.kv_cache_block_size == 2


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
    assert args[0] == "dynamo:planner:target:{test-namespace:test-model}"
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
        [TargetReplica(sub_component_type=SubComponentType.PREFILL, desired_replicas=4)]
    )
    mapping = mock_redis_client.hset.call_args.kwargs["mapping"]
    assert mapping["prefill"] == 4
    assert "decode" not in mapping


@pytest.mark.asyncio
async def test_set_component_replicas_empty_raises(connector):
    with pytest.raises(EmptyTargetReplicasError):
        await connector.set_component_replicas([])


@pytest.mark.asyncio
async def test_set_component_replicas_negative_raises(connector, mock_redis_client):
    with pytest.raises(ValueError, match="must not be negative"):
        await connector.set_component_replicas(
            [
                TargetReplica(
                    sub_component_type=SubComponentType.PREFILL, desired_replicas=-1
                )
            ]
        )
    mock_redis_client.hset.assert_not_called()


class TestWriteTargetPower:
    """_write_target_power translates the decode worker count to target_power
    on the model:{name} hash using the per-model power_coefficient, and
    enrolls the name in the power_scaled_models set."""

    @pytest.mark.asyncio
    async def test_writes_target_power_from_decode_with_default_coefficient(
        self, connector, mock_redis_client
    ):
        # hgetall returns {} -> power_coefficient absent -> falls back to 1
        mock_redis_client.hgetall.return_value = {}
        await connector._write_target_power({"decode": 5})
        # target_power = 5 * 1 = 5, written via the pipeline
        pipe = mock_redis_client.pipeline.return_value
        pipe.hset.assert_called_once_with(
            "model:{test-model}", "target_power", 5
        )

    @pytest.mark.asyncio
    async def test_writes_target_power_using_power_coefficient(
        self, connector, mock_redis_client
    ):
        mock_redis_client.hgetall.return_value = {"power_coefficient": "2"}
        await connector._write_target_power({"decode": 5})
        # target_power = 5 * 2 = 10
        pipe = mock_redis_client.pipeline.return_value
        pipe.hset.assert_called_once_with(
            "model:{test-model}", "target_power", 10
        )

    @pytest.mark.asyncio
    async def test_no_decode_does_not_write_target_power(
        self, connector, mock_redis_client
    ):
        await connector._write_target_power({"prefill": 3})
        mock_redis_client.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_enrolls_in_power_scaled_set(
        self, connector, mock_redis_client
    ):
        mock_redis_client.hgetall.return_value = {}
        await connector._write_target_power({"decode": 5})
        pipe = mock_redis_client.pipeline.return_value
        pipe.sadd.assert_called_once_with("power_scaled_models", "test-model")


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


@pytest.mark.asyncio
async def test_validate_deployment_and_wait_are_no_ops(connector):
    await connector.validate_deployment()
    await connector.wait_for_deployment_ready()


@pytest.mark.asyncio
async def test_read_desired_counts_negative_raises(connector, mock_redis_client):
    mock_redis_client.hgetall.return_value = {"prefill": "-1", "decode": "0"}
    with pytest.raises(ValueError, match="'prefill'.*must not be negative"):
        await connector.add_component(SubComponentType.PREFILL)


@pytest.mark.asyncio
async def test_read_desired_counts_invalid_raises(connector, mock_redis_client):
    mock_redis_client.hgetall.return_value = {"prefill": "not-a-number", "decode": "0"}
    with pytest.raises(ValueError, match="'prefill'.*not a valid integer"):
        await connector.add_component(SubComponentType.PREFILL)


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_closes_the_client(self, connector, mock_redis_client):
        mock_redis_client.aclose = AsyncMock()
        await connector.shutdown()
        mock_redis_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, connector, mock_redis_client):
        mock_redis_client.aclose = AsyncMock()
        await connector.shutdown()
        await connector.shutdown()
        mock_redis_client.aclose.assert_awaited_once()


class TestGetActualWorkerCounts:
    """get_actual_worker_counts reads the model manager's committed_power back
    from the model:{name} hash and translates it to a worker count via the
    per-model power_coefficient. If the MM hasn't published committed_power
    yet -- or never does -- the field is absent and we fail closed as
    inactive/unstable, never a false "settled empty".
    """

    @pytest.mark.asyncio
    async def test_no_committed_power_defaults_to_unstable(
        self, connector, mock_redis_client
    ):
        mock_redis_client.hgetall.return_value = {}
        prefill, decode, stable = await connector.get_actual_worker_counts(
            prefill_component_name="prefill-worker",
            decode_component_name="decode-worker",
        )
        assert (prefill, decode, stable) == (0, 0, False)

    @pytest.mark.asyncio
    async def test_reads_committed_power_when_present(
        self, connector, mock_redis_client
    ):
        mock_redis_client.hgetall.return_value = {"committed_power": "5"}
        prefill, decode, stable = await connector.get_actual_worker_counts(
            prefill_component_name="prefill-worker",
            decode_component_name="decode-worker",
        )
        # coefficient defaults to 1 -> committed power 5 == 5 workers
        assert (prefill, decode, stable) == (5, 5, True)

    @pytest.mark.asyncio
    async def test_uses_power_coefficient_to_convert_to_count(
        self, connector, mock_redis_client
    ):
        mock_redis_client.hgetall.return_value = {
            "committed_power": "10",
            "power_coefficient": "2",
        }
        prefill, decode, stable = await connector.get_actual_worker_counts(
            prefill_component_name="prefill-worker",
            decode_component_name="decode-worker",
        )
        # 10 power units / 2 power-per-worker = 5 workers
        assert (prefill, decode, stable) == (5, 5, True)

    @pytest.mark.asyncio
    async def test_component_name_none_reports_zero(
        self, connector, mock_redis_client
    ):
        """Decode isn't required by this planner mode (name arg is None): it
        reports 0 and does not gate stability, even though committed_power is
        present. In agg mode prefill is not a separate role, so a None prefill
        also reports 0.
        """
        mock_redis_client.hgetall.return_value = {"committed_power": "5"}
        prefill, decode, stable = await connector.get_actual_worker_counts(
            prefill_component_name="prefill-worker",
            decode_component_name=None,
        )
        assert (prefill, decode, stable) == (5, 0, True)

    @pytest.mark.asyncio
    async def test_negative_committed_power_raises(
        self, connector, mock_redis_client
    ):
        mock_redis_client.hgetall.return_value = {"committed_power": "-2"}
        with pytest.raises(ValueError, match="'committed_power'.*must not be negative"):
            await connector.get_actual_worker_counts(
                prefill_component_name="prefill-worker",
                decode_component_name=None,
            )
