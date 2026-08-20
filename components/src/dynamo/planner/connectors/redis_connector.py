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

# DEEPINFRA: this connector does not exist upstream. It lets the planner hand
# a scaling decision to an external control plane (ours) instead of scaling
# a Kubernetes DynamoGraphDeployment itself. Unlike VirtualConnector -- which
# is also built for "non-native environments" but requires a live
# etcd+nats-backed DistributedRuntime coordinator -- this talks to Redis
# directly, with no other runtime dependency.
#
# Deliberately named ``redis_connector.py`` rather than ``redis.py`` (every
# sibling module here is named after its target system, e.g. ``virtual.py``,
# ``kubernetes.py``) to avoid a same-named module shadowing the ``redis``
# package it imports.

import logging
import os
import time
from typing import Optional

import redis.asyncio as redis_asyncio

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.connectors.base import PlannerConnector
from dynamo.planner.errors import EmptyTargetReplicasError
from dynamo.planner.monitoring.worker_info import (
    WorkerInfo,
    build_worker_info_from_defaults,
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)

REDIS_KEY_PREFIX = os.environ.get(
    "DYNAMO_REDIS_KEY_PREFIX", "mm:dynamo_planner_target"
)


class RedisConnector(PlannerConnector):
    """Publishes scaling decisions to Redis instead of actuating them.

    Sibling to ``VirtualConnector`` in spirit ("hand the decision to a
    non-native environment"), but with no etcd/nats dependency: the decision
    is written to a single Redis hash, keyed by model name, and an external
    control plane is expected to read it and actuate on its own schedule.
    """

    def __init__(
        self,
        dynamo_namespace: str,
        model_name: Optional[str] = None,
        redis_url: Optional[str] = None,
    ) -> None:
        if not model_name:
            raise ValueError("Model name is required for redis connector")
        self.model_name = model_name.lower()  # normalize model name to lowercase (MDC)
        self.dynamo_namespace = dynamo_namespace

        redis_url = redis_url or os.environ.get("DYNAMO_REDIS_URL")
        if not redis_url:
            raise ValueError(
                "redis_url is required for redis connector "
                "(pass explicitly or set DYNAMO_REDIS_URL)"
            )
        self._redis = redis_asyncio.from_url(redis_url, decode_responses=True)
        # Curly-brace hash tag, matching the {model_name}-keyed convention
        # used elsewhere for this Redis instance (e.g. mm:dynamo_fe_applied:{%s}).
        self._key = f"{REDIS_KEY_PREFIX}:{{{self.model_name}}}"

    async def _read_counts(self) -> tuple[int, int]:
        raw = await self._redis.hgetall(self._key)
        return int(raw.get("prefill", 0)), int(raw.get("decode", 0))

    async def add_component(
        self, sub_component_type: SubComponentType, blocking: bool = True
    ) -> None:
        """Add a component by increasing its published replica count by 1.

        Not exercised by the automatic tick loop (that goes through
        ``set_component_replicas`` below); implemented only to satisfy the
        ``PlannerConnector`` ABC for manual/CLI parity with the other
        connectors. Read-then-write, same as ``VirtualConnector``'s
        equivalent -- fine off the hot path, at tick-interval cadence.
        """
        current_p, current_d = await self._read_counts()
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
        current_p, current_d = await self._read_counts()
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

        This is the method ``NativePlannerBase._apply_scaling_targets``
        actually calls for every mode (prefill/decode/agg/disagg alike).

        A single ``HSET`` covers only the roles present in ``target_replicas``
        -- no read-modify-write, so two planners sharing this model_name's
        key (e.g. a dedicated PrefillPlanner and DecodePlanner) can't race
        each other clobbering the other's field.

        ``blocking`` is accepted for interface compatibility but ignored:
        there's no "wait for the external actuator to finish scaling"
        concept here. Our job ends when the write returns; the external
        control plane (ours) picks it up and actuates on its own schedule.
        """
        if not target_replicas:
            raise EmptyTargetReplicasError()

        mapping: dict[str, str | float] = {}
        for target_replica in target_replicas:
            if target_replica.sub_component_type == SubComponentType.PREFILL:
                mapping["prefill"] = target_replica.desired_replicas
            elif target_replica.sub_component_type == SubComponentType.DECODE:
                mapping["decode"] = target_replica.desired_replicas

        if not mapping:
            return
        mapping["updated_at"] = time.time()
        await self._redis.hset(self._key, mapping=mapping)

    async def validate_deployment(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> None:
        """No external deployment for this connector to validate against."""
        pass

    async def wait_for_deployment_ready(self, include_planner: bool = True) -> None:
        """No external deployment to wait on."""
        pass

    def get_worker_info(
        self,
        sub_component_type: SubComponentType,
        backend: str = "vllm",
    ) -> WorkerInfo:
        """No live discovery source for this connector -- always defaults.

        VirtualConnector gets an MDC source wired in after construction
        (``NativePlannerBase`` special-cases it via ``set_mdc_subscribers``);
        that hook is VirtualConnector-specific, so this connector never
        receives it and always falls back to defaults.
        """
        info = build_worker_info_from_defaults(backend, sub_component_type)
        info.model_name = self.model_name
        return info

    async def get_model_name(
        self, require_prefill: bool = True, require_decode: bool = True
    ) -> str:
        """Get the model name (as given at construction)."""
        del require_prefill, require_decode
        return self.model_name
