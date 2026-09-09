# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import Optional

from dynamo.common.forward_pass_metrics import ForwardPassMetrics
from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.connectors.base import WorkerInfoProvider
from dynamo.planner.connectors.mdc import MdcEntry, select_entry, worker_info_from_mdc
from dynamo.planner.core.types import FpmObservations
from dynamo.planner.environment.interface import (
    DeploymentStateSource,
    RuntimeNamespaceSource,
)
from dynamo.planner.environment.metrics_provider.interface import FpmMetricsProvider
from dynamo.planner.environment.state import DeploymentState
from dynamo.planner.monitoring.worker_info import (
    WorkerInfo,
    build_worker_info_from_defaults,
)
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


class RuntimeFpmProvider(FpmMetricsProvider, WorkerInfoProvider):
    # DEEPINFRA: consecutive polls of wall_time==0 with no scheduled work after
    # which a (worker_id, dp_rank) is dropped from the observed engine set.
    _FPM_STALE_POLLS = 3

    def __init__(
        self,
        *,
        require_prefill: bool,
        require_decode: bool,
        backend: str,
        model_name: Optional[str],
        runtime: DistributedRuntime,
        state_source: Optional[DeploymentStateSource] = None,
        namespace_source: Optional[RuntimeNamespaceSource] = None,
    ) -> None:
        self.require_prefill = require_prefill
        self.require_decode = require_decode
        self.backend = backend
        self.model_name = model_name
        self.runtime = runtime
        self.state_source = state_source
        self.namespace_source = namespace_source
        self._prefill_fpm_sub = None
        self._decode_fpm_sub = None
        # DEEPINFRA: per-(label) idle-FPM tracker. The Rust FpmDirectPublisher
        # keeps an idle-heartbeat task that emits zeroed snapshots every ~5s
        # with an incrementing counter -- so bytes-based staleness can't tell a
        # drained worker from an active one. Track by FPM semantics instead:
        # any consecutive run of polls where wall_time==0 and no scheduled
        # work was observed counts as drained/idle. Maps
        # label -> {(wid, dp): consecutive_idle_polls}.
        self._fpm_stale_tracker: dict[str, dict[tuple[str, int], int]] = {}

    def bind_sources(
        self,
        *,
        state_source: DeploymentStateSource,
        namespace_source: RuntimeNamespaceSource,
    ) -> None:
        self.state_source = state_source
        self.namespace_source = namespace_source

    async def async_init(self, namespace: Optional[str] = None) -> None:
        del namespace
        await self._rebind_subscribers()

    async def refresh(self, state: DeploymentState) -> None:
        del state
        await self._rebind_missing_subscribers()

    def collect_fpm(self) -> FpmObservations:
        prefill_stats = None
        decode_stats = None

        if self._prefill_fpm_sub is not None:
            stats = self._decode_fpm_bytes(self._prefill_fpm_sub, "prefill")
            if stats:
                for (wid, dp), fpm in stats.items():
                    _log_fpm(wid, dp, fpm, "prefill")
                prefill_stats = stats

        if self._decode_fpm_sub is not None:
            stats = self._decode_fpm_bytes(self._decode_fpm_sub, "decode")
            if stats:
                for (wid, dp), fpm in stats.items():
                    _log_fpm(wid, dp, fpm, "decode")
                decode_stats = stats

        return FpmObservations(prefill=prefill_stats, decode=decode_stats)

    def get_worker_info(
        self,
        sub_component_type: SubComponentType,
        backend: str = "vllm",
    ) -> WorkerInfo:
        subscriber = (
            self._prefill_fpm_sub
            if sub_component_type == SubComponentType.PREFILL
            else self._decode_fpm_sub
        )
        entry = select_entry(
            _mdc_entries_from_subscriber(subscriber),
            sub_component_type,
        )
        if entry is not None:
            info = worker_info_from_mdc(
                entry,
                sub_component_type,
                backend=backend,
                model_name_fallback=lambda: self.model_name,
            )
        else:
            info = build_worker_info_from_defaults(backend, sub_component_type)
            info.model_name = self.model_name
        return info

    async def shutdown(self) -> None:
        self._shutdown_subscribers()

    async def _rebind_subscribers(self) -> None:
        self._shutdown_subscribers()
        if self.require_prefill:
            self._prefill_fpm_sub = await self._init_fpm_subscriber("prefill")
        if self.require_decode:
            self._decode_fpm_sub = await self._init_fpm_subscriber("decode")

    async def _rebind_missing_subscribers(self) -> None:
        if self.require_prefill and self._prefill_fpm_sub is None:
            self._prefill_fpm_sub = await self._init_fpm_subscriber("prefill")
        if self.require_decode and self._decode_fpm_sub is None:
            self._decode_fpm_sub = await self._init_fpm_subscriber("decode")

    async def _init_fpm_subscriber(self, component: str):
        # Avoid loading binding-heavy dynamo.llm until runtime FPM is configured.
        from dynamo.llm import FpmEventSubscriber

        if self.state_source is None or self.namespace_source is None:
            logger.warning(
                "Runtime FPM provider is missing state or namespace source, "
                "cannot create FPM subscriber for %s",
                component,
            )
            return None
        state = self.state_source.deployment_state()
        worker_info = (
            state.prefill.info if component == "prefill" else state.decode.info
        )
        if (
            worker_info is None
            or not worker_info.component_name
            or not worker_info.endpoint
        ):
            logger.warning(
                "WorkerInfo missing for %s, cannot create FPM subscriber", component
            )
            return None

        runtime_namespace = self.namespace_source.runtime_namespace()
        endpoint = self.runtime.endpoint(
            f"{runtime_namespace}.{worker_info.component_name}.{worker_info.endpoint}"
        )
        sub = FpmEventSubscriber(endpoint)
        sub.start_tracking()
        logger.info(
            "FPM tracker started for %s.%s.%s",
            runtime_namespace,
            worker_info.component_name,
            worker_info.endpoint,
        )
        return sub

    def _shutdown_subscribers(self) -> None:
        if self._prefill_fpm_sub is not None:
            self._prefill_fpm_sub.shutdown()
        if self._decode_fpm_sub is not None:
            self._decode_fpm_sub.shutdown()
        self._prefill_fpm_sub = None
        self._decode_fpm_sub = None

    def _decode_fpm_bytes(
        self, subscriber, label: str = ""
    ) -> dict[tuple[str, int], ForwardPassMetrics]:
        # Match the subscriber's lazy binding path; decoding is optional at startup.
        from dynamo.common.forward_pass_metrics import decode as decode_fpm

        if subscriber is None:
            return {}
        # DEEPINFRA: a worker emitting wall_time==0 with no scheduled work for N
        # consecutive polls is treated as drained/idle and dropped from the
        # observed set so it doesn't inflate the engine count past DGD. Drain
        # v9's Python emission gate produces exactly this shape (Rust heartbeats
        # keep firing zeroed snapshots while paused). Truly idle workers get
        # evicted too, which is fine -- they re-enter the moment a non-zero FPM
        # arrives, and an idle worker contributes nothing to the load estimate.
        raw = subscriber.get_recent_stats()
        tracker = self._fpm_stale_tracker.setdefault(label, {})
        result: dict[tuple[str, int], ForwardPassMetrics] = {}
        for key, raw_bytes in raw.items():
            fpm = decode_fpm(raw_bytes)
            if fpm is None:
                continue
            sched = fpm.scheduled_requests
            is_idle = (
                fpm.wall_time == 0.0
                and sched.num_prefill_requests == 0
                and sched.num_decode_requests == 0
            )
            stale = tracker.get(key, 0) + 1 if is_idle else 0
            tracker[key] = stale
            if stale >= self._FPM_STALE_POLLS:
                if stale == self._FPM_STALE_POLLS:
                    logger.warning(
                        f"Evicting idle FPM engine {key[0]}:dp{key[1]} ({label}): "
                        f"wall_time==0 with no scheduled work for {stale} polls "
                        f"(publisher paused or worker truly idle)"
                    )
                continue
            result[key] = fpm
        # forget keys the subscriber no longer reports (worker left discovery)
        for key in [k for k in tracker if k not in raw]:
            del tracker[key]
        return result


def _mdc_entries_from_subscriber(subscriber) -> list[MdcEntry]:
    if subscriber is None:
        return []
    try:
        cards = subscriber.get_model_cards()
    except RuntimeError:
        return []

    entries: list[MdcEntry] = []
    for worker_id, card_str in cards.items():
        try:
            card_json = json.loads(card_str)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed MDC card JSON for worker %s", worker_id)
            continue
        entries.append(MdcEntry(card_json=card_json, instance_id=worker_id))
    return entries


def _log_fpm(wid: str, dp: int, fpm: ForwardPassMetrics, label: str) -> None:
    sched = fpm.scheduled_requests
    queued = fpm.queued_requests
    logger.info(
        "FPM %s engine %s:dp%d: wall_time=%.4fs, "
        "sched(prefill_tok=%s, prefill_req=%s, decode_kv=%s, decode_req=%s), "
        "queued(prefill_tok=%s, decode_kv=%s)",
        label,
        wid,
        dp,
        fpm.wall_time,
        sched.sum_prefill_tokens,
        sched.num_prefill_requests,
        sched.sum_decode_kv_tokens,
        sched.num_decode_requests,
        queued.sum_prefill_tokens,
        queued.sum_decode_kv_tokens,
    )
