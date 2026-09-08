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

"""DeepInfra connector: publishes scaling decisions to Redis and reads back
observed worker state, instead of talking to Kubernetes directly.

Sibling to VirtualConnector for "hand the decision to a non-native
environment," but with no etcd/nats/DistributedRuntime dependency -- just a
Redis connection. Specialized to DeepInfra's model manager (not a generic
upstream connector).

The planner publishes its per-tick decision to its own deployment hash
``{prefix}:{dynamo_namespace}:{model_name}`` (the ``prefill``/``decode``
desired counts), and separately translates the decode target to a power unit
on DeepInfra's ``model:{model_name}`` hash (``target_power``), enrolling the
name in the ``power_scaled_models`` set. The model manager's power realizer
reads ``target_power`` and actuates workers, then writes the realized power
back as ``committed_power`` on the same hash. ``get_actual_worker_counts``
reads ``committed_power`` back and translates it to a worker count via the
per-model ``power_coefficient``.

If the model manager hasn't published ``committed_power`` yet -- before its
first write-back, or because it doesn't track this at all --
``get_actual_worker_counts`` reads inactive/unstable defaults rather than
raising or assuming a settled empty deployment.
"""

import logging
import os
import time

import redis.asyncio as redis_asyncio
from redis.asyncio.sentinel import Sentinel

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import (
    PlannerConnector,
    WorkerInfoProvider,
)
from dynamo.planner.errors import EmptyTargetReplicasError
from dynamo.planner.monitoring.worker_info import (
    WorkerInfo,
    build_worker_info_from_defaults,
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

REDIS_KEY_PREFIX = os.environ.get("DYN_REDIS_KEY_PREFIX", "dynamo:planner:target")

# DeepInfra model-catalog keys (mirror RedisModelCatalog in the backend).
# The planner writes its target as a power unit on the model-name's ModelInfo
# hash, and enrolls the name in the power-scaled set so the model manager's
# power realizer discovers and actuates on it. The model manager writes the
# realized power back as `committed_power` on the same hash, which this
# connector reads as the actual worker count. See set_target_power /
# set_committed_power in deepinfra/redis_models/redis_model_catalog.py.
MODEL_KEY = "model:{%s}"
POWER_SCALED = "power_scaled_models"
# Power <-> instance translation. The per-model coefficient is read from the
# model:{name} hash (`power_coefficient`) on each tick so config updates take
# effect without a planner restart; POWER_COEFFICIENT is the fallback when the
# field is absent. For now each worker is one power unit, so the translation
# is a no-op; revisit when a config carries power != 1.
POWER_COEFFICIENT = 1


def _parse_nonnegative_int(raw: dict[str, str], field: str) -> int:
    """Parse one hash field as a non-negative int, defaulting to 0 if absent.

    Raises ValueError naming the offending field on a negative or otherwise
    invalid value -- corrupted or hand-edited Redis data should fail loudly
    rather than silently propagate a nonsensical replica count.
    """
    value = raw.get(field, "0")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"redis field {field!r} is not a valid integer: {value!r}"
        ) from None
    if parsed < 0:
        raise ValueError(f"redis field {field!r} must not be negative, got {parsed}")
    return parsed


def _parse_nonnegative_float(raw: dict[str, str], field: str) -> float | None:
    """Parse one hash field as a non-negative float, or None if absent.

    Unlike the int variant, absence is meaningful here: a missing field means
    "no value published yet" (the caller decides how to treat that), not
    "zero". Raises ValueError on a negative or otherwise invalid value.
    """
    value = raw.get(field)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"redis field {field!r} is not a valid number: {value!r}"
        ) from None
    if parsed < 0:
        raise ValueError(f"redis field {field!r} must not be negative, got {parsed}")
    return parsed


class DeepInfraConnector(PlannerConnector):
    """Publishes scaling decisions to Redis; reads observed state back from it.

    Specialized to DeepInfra's model manager: the planner writes its target as
    a power unit on the model-name's ModelInfo hash (``target_power``) and
    enrolls the name in the ``power_scaled_models`` set; the model manager's
    power realizer reads that and actuates workers, then writes the realized
    power back as ``committed_power`` on the same hash, which this connector
    reads as the actual worker count. See set_target_power / set_committed_power
    in deepinfra/redis_models/redis_model_catalog.py.
    """

    def __init__(
        self,
        dynamo_namespace: str,
        model_name: str | None = None,
        redis_url: str | None = None,
        worker_info_provider: WorkerInfoProvider | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("Model name is required for redis connector")
        # Case-preserving: unlike VirtualConnector/KubernetesConnector, this
        # connector never matches model_name against MDC entries, so there's
        # no reason to fold case -- and doing so would risk merging two
        # distinct, differently-cased model names onto the same Redis key.
        self.model_name = model_name
        self.dynamo_namespace = dynamo_namespace
        # Optional source of runtime WorkerInfo/MDC, wired in by
        # construct_environment (a RuntimeFpmProvider). When present,
        # get_worker_info delegates to it so the planner reads real worker
        # capabilities (e.g. KV cache size) from MDC instead of defaults.
        self.worker_info_provider = worker_info_provider

        # Two ways to connect, selected by which env is set:
        #  - Sentinel (DYN_REDIS_SENTINELS): the client asks Sentinel for the
        #    current master on connect and re-asks after a failover, so it keeps
        #    working when the master moves. Use this for a self-run HA Redis.
        #  - Direct URL (DYN_REDIS_URL / the redis_url arg): one fixed address,
        #    for a single-node or externally-load-balanced Redis.
        # Sentinel wins if both are set.
        self._sentinel: Sentinel | None = None
        self._redis = self._build_client(redis_url)
        self._closed = False
        # Curly braces are a Redis Cluster hash tag: only the portion inside
        # them is hashed to pick a shard, so every key for one deployment
        # lands on the same node regardless of what prefix precedes it.
        # Namespace is part of the tag, not just the prefix, so two
        # deployments sharing a model_name under different namespaces never
        # collide on the same key.
        self._key = f"{REDIS_KEY_PREFIX}:{{{self.dynamo_namespace}:{self.model_name}}}"

    def _build_client(self, redis_url: str | None) -> "redis_asyncio.Redis":
        sentinels = os.environ.get("DYN_REDIS_SENTINELS")
        if sentinels:
            return self._build_sentinel_client(sentinels)
        redis_url = redis_url or os.environ.get("DYN_REDIS_URL")
        if not redis_url:
            raise ValueError(
                "redis connector needs either DYN_REDIS_SENTINELS (Sentinel) "
                "or DYN_REDIS_URL / the redis_url arg (direct)"
            )
        return redis_asyncio.from_url(redis_url, decode_responses=True)

    def _build_sentinel_client(self, sentinels: str) -> "redis_asyncio.Redis":
        master_name = os.environ.get("DYN_REDIS_MASTER_NAME")
        if not master_name:
            raise ValueError(
                "DYN_REDIS_MASTER_NAME is required when DYN_REDIS_SENTINELS is set"
            )
        nodes: list[tuple[str, int]] = []
        for entry in sentinels.split(","):
            entry = entry.strip()
            if not entry:
                continue
            host, sep, port = entry.rpartition(":")
            if not sep:
                raise ValueError(
                    f"DYN_REDIS_SENTINELS entry {entry!r} must be host:port"
                )
            nodes.append((host, int(port)))
        if not nodes:
            raise ValueError("DYN_REDIS_SENTINELS is set but empty")

        # Auth for the Sentinel connections themselves vs. the data (master)
        # connections are separate credentials -- keep them apart.
        sentinel_kwargs: dict[str, str] = {}
        s_user = os.environ.get("DYN_REDIS_SENTINEL_USERNAME")
        s_pass = os.environ.get("DYN_REDIS_SENTINEL_PASSWORD")
        if s_user:
            sentinel_kwargs["username"] = s_user
        if s_pass:
            sentinel_kwargs["password"] = s_pass

        connection_kwargs: dict[str, object] = {
            "decode_responses": True,
            "db": int(os.environ.get("DYN_REDIS_DB", "0")),
        }
        d_user = os.environ.get("DYN_REDIS_USERNAME")
        d_pass = os.environ.get("DYN_REDIS_PASSWORD")
        if d_user:
            connection_kwargs["username"] = d_user
        if d_pass:
            connection_kwargs["password"] = d_pass

        self._sentinel = Sentinel(
            nodes,
            min_other_sentinels=0,
            sentinel_kwargs=sentinel_kwargs or None,
            **connection_kwargs,
        )
        # master_for returns a client that resolves the current master through
        # Sentinel on each connection, so it survives failover without a restart.
        return self._sentinel.master_for(master_name)

    async def async_init(self) -> None:
        """Nothing to do -- the Redis client connects lazily on first command."""

    def get_worker_runtime_namespace(self, base_dynamo_namespace: str) -> str:
        # Required so the planner wires up the RuntimeFpmProvider (forward-pass
        # metrics subscription). Workers live under the same dynamo namespace.
        return base_dynamo_namespace

    async def _read_desired_counts(self) -> tuple[int, int]:
        raw = await self._redis.hgetall(self._key)
        return (
            _parse_nonnegative_int(raw, "prefill"),
            _parse_nonnegative_int(raw, "decode"),
        )

    async def _read_power_coefficient(self) -> float:
        """Read the per-model power coefficient from the model:{name} hash.

        The coefficient is the number of power units one worker consumes; the
        planner uses it to translate worker counts <-> power units. It is read
        on each tick (not cached at init) so a config update takes effect
        without a planner restart. Falls back to ``POWER_COEFFICIENT`` (1) when
        the field is absent -- e.g. before the backend starts writing it.
        """
        raw = await self._redis.hgetall(MODEL_KEY % self.model_name)
        coefficient = _parse_nonnegative_float(raw, "power_coefficient")
        return coefficient if coefficient is not None else float(POWER_COEFFICIENT)

    async def add_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ) -> None:
        """Add a component by increasing its published replica count by 1.

        Not exercised by the automatic tick loop (that goes through
        ``set_component_replicas`` below); implemented only for parity with
        the other connectors' manual/CLI use. Read-then-write, same as
        ``VirtualConnector``'s equivalent -- fine off the hot path.
        """
        current_p, current_d = await self._read_desired_counts()
        if sub_component_type == SubComponentType.PREFILL:
            await self._redis.hset(
                self._key,
                mapping={"prefill": current_p + 1, "updated_at": time.time()},
            )
        elif sub_component_type == SubComponentType.DECODE:
            await self._redis.hset(
                self._key,
                mapping={"decode": current_d + 1, "updated_at": time.time()},
            )

    async def remove_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ) -> None:
        """Remove a component by decreasing its published replica count by 1.

        See ``add_component`` -- not on the automatic scaling path.
        """
        current_p, current_d = await self._read_desired_counts()
        if sub_component_type == SubComponentType.PREFILL:
            await self._redis.hset(
                self._key,
                mapping={"prefill": max(0, current_p - 1), "updated_at": time.time()},
            )
        elif sub_component_type == SubComponentType.DECODE:
            await self._redis.hset(
                self._key,
                mapping={"decode": max(0, current_d - 1), "updated_at": time.time()},
            )

    async def set_component_replicas(
        self, target_replicas: list[TargetReplica], blocking: bool = True
    ) -> None:
        """Publish the tick's final decision to Redis.

        This is the method every planner mode actually calls. A single
        ``HSET`` covers only the roles present in ``target_replicas`` -- no
        read-modify-write, so two planners sharing this model_name's key
        (e.g. a dedicated PrefillPlanner and DecodePlanner) can't race each
        other clobbering the other's field.

        ``blocking`` is accepted for interface compatibility but ignored:
        there's no "wait for the external actuator to finish scaling"
        concept here. Our job ends when the write returns; the external
        control plane picks it up and actuates on its own schedule.
        """
        if not target_replicas:
            raise EmptyTargetReplicasError()

        mapping: dict[str, int | float] = {}
        for target_replica in target_replicas:
            if target_replica.desired_replicas < 0:
                raise ValueError(
                    f"desired_replicas must not be negative, got "
                    f"{target_replica.desired_replicas} for "
                    f"{target_replica.sub_component_type.value}"
                )
            if target_replica.sub_component_type == SubComponentType.PREFILL:
                mapping["prefill"] = target_replica.desired_replicas
            elif target_replica.sub_component_type == SubComponentType.DECODE:
                mapping["decode"] = target_replica.desired_replicas

        if not mapping:
            return
        mapping["updated_at"] = time.time()
        await self._redis.hset(self._key, mapping=mapping)
        await self._write_target_power(mapping)

    async def _write_target_power(self, mapping: dict[str, int | float]) -> None:
        """Publish the tick's target to DeepInfra's model-name power field.

        DeepInfra scales power-scaled names by ``ModelInfo.target_power`` (the
        model manager's power realizer reads it and actuates workers). The
        planner thinks in instance counts; translate to power here using the
        per-model ``power_coefficient`` (read from the model hash each tick),
        so the MM side only ever talks power.

        Mirrors ``RedisModelCatalog.set_target_power``: write ``target_power``
        into the ``model:{model_name}`` hash and enroll the name in the
        ``power_scaled_models`` set so the realizer discovers it.
        """
        # In agg mode the decode target is the aggregated worker count; in
        # disagg the decode target is the decode pool. Prefer decode as the
        # power target (the aggregated engine is the capacity unit).
        decode = mapping.get("decode")
        if decode is None:
            return
        coefficient = await self._read_power_coefficient()
        target_power = int(decode * coefficient)
        model_key = MODEL_KEY % self.model_name
        pipe = self._redis.pipeline()
        pipe.hset(model_key, "target_power", target_power)
        pipe.sadd(POWER_SCALED, self.model_name)
        await pipe.execute()

    async def validate_deployment(
        self,
        prefill_component_name: str | None = None,
        decode_component_name: str | None = None,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> None:
        """No external deployment for this connector to validate against."""

    async def wait_for_deployment_ready(self, include_planner: bool = True) -> None:
        """No external deployment to wait on."""

    def get_model_name(
        self, require_prefill: bool = True, require_decode: bool = True
    ) -> str:
        """Get the model name (as given at construction)."""
        del require_prefill, require_decode
        return self.model_name

    def get_gpu_counts(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> tuple[int | None, int | None]:
        """No live view of GPU shape -- same reasoning as ``VirtualConnector``:
        this connector's Redis hash carries replica counts only, never GPU
        shape. The planner falls back to its own ``prefill_engine_num_gpu``/
        ``decode_engine_num_gpu`` config for its sizing math.
        """
        del require_prefill, require_decode
        return None, None

    def get_worker_info(
        self,
        sub_component_type: SubComponentType,
        backend: str = "vllm",
    ) -> WorkerInfo:
        """Resolve worker capabilities from the runtime MDC when a provider is
        wired in, else fall back to defaults.

        ``construct_environment`` supplies a ``RuntimeFpmProvider`` (the same
        source used for forward-pass metrics), so the planner reads real worker
        capabilities -- e.g. KV cache size -- from MDC, exactly like
        ``VirtualConnector``. Without a provider (e.g. a bare connector with no
        runtime), fall back to defaults.
        """
        if self.worker_info_provider is not None:
            return self.worker_info_provider.get_worker_info(
                sub_component_type, backend
            )
        info = build_worker_info_from_defaults(backend, sub_component_type)
        info.model_name = self.model_name
        return info

    async def get_actual_worker_counts(
        self,
        prefill_component_name: str | None = None,
        decode_component_name: str | None = None,
    ) -> tuple[int, int, bool]:
        """Read observed state written back by the external actuator.

        The model manager writes the realized power as ``committed_power`` on
        the ``model:{model_name}`` hash (see ``set_committed_power`` in the
        backend). Committed power counts the pods the model manager has
        committed to -- BOOT/PENDING/ACTIVE -- not just the ones currently
        serving; that is the right figure for the planner to compare against
        its own target, since it is the capacity the control plane is actually
        standing up. Translate power back to a worker count here via the
        per-model ``power_coefficient`` (read from the same hash, so no extra
        round-trip).

        Per the ``PlannerConnector`` contract, a component whose name arg is
        ``None`` (not required by this planner mode) reports 0 and does not
        count against the returned ``stable`` flag -- only roles actually in
        play for this planner instance affect it.

        If the model manager hasn't published ``committed_power`` yet --
        before its first write-back, or because it doesn't track this at all
        -- the field is absent and we fail closed as "still converging,"
        never a false "settled empty."
        """
        raw = await self._redis.hgetall(MODEL_KEY % self.model_name)
        committed = _parse_nonnegative_float(raw, "committed_power")
        if committed is None:
            # No observed state yet: report inactive/unstable so the planner
            # keeps converging rather than assuming a settled deployment.
            return 0, 0, False

        # In agg mode the decode target is the aggregated worker count (the
        # capacity unit the planner sizes against), so the committed power
        # maps onto decode. Prefill is not a separate role in agg mode.
        coefficient = _parse_nonnegative_float(raw, "power_coefficient")
        coefficient = coefficient if coefficient is not None else float(POWER_COEFFICIENT)
        decode = int(committed / coefficient)
        prefill = 0 if prefill_component_name is None else decode
        return prefill, decode, True

    async def shutdown(self) -> None:
        """Release the Redis client. Idempotent -- safe to call more than
        once (e.g. if the environment's own shutdown and a caller's
        explicit cleanup both fire)."""
        if self._closed:
            return
        self._closed = True
        await self._redis.aclose()
        if self._sentinel is not None:
            for s in self._sentinel.sentinels:
                await s.aclose()
