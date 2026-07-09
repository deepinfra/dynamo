# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# mypy: disable-error-code="attr-defined"

"""Throughput-based scaling logic (Prometheus traffic-driven, predictive).

Mixin consumed by ``PlannerScalingState``.  All methods access state via
``self._config``, ``self._capabilities``, and perf models.
"""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

from dynamo.planner.core.types import ScalingDecision, TrafficShape

logger = logging.getLogger(__name__)

# Erlang-C search stops here; a prescription this large means the inputs are
# broken, not that the fleet should be this large.
_ERLANG_MAX_N = 512


def erlang_c(n: int, offered_load: float) -> float:
    """P(wait > 0) for an M/M/n queue at ``offered_load`` = lambda * S erlangs.

    Computed via the numerically stable Erlang-B recursion
    ``B_k = a*B_{k-1} / (k + a*B_{k-1})``, then converted to Erlang C.
    Returns 1.0 when the queue is unstable (offered_load >= n).
    """
    if offered_load <= 0:
        return 0.0
    if offered_load >= n:
        return 1.0
    b = 1.0
    for k in range(1, n + 1):
        b = offered_load * b / (k + offered_load * b)
    rho = offered_load / n
    return b / (1.0 - rho + rho * b)


def erlang_c_min_servers(
    arrival_rps: float,
    service_s: float,
    wait_budget_s: float,
    variability: float,
    max_n: int = _ERLANG_MAX_N,
) -> Optional[int]:
    """Smallest n whose mean queue wait fits ``wait_budget_s``.

    Mean wait is the M/M/n Erlang-C wait scaled by the Allen-Cunneen
    correction ``variability`` = (Ca^2 + Cs^2) / 2 for non-Poisson arrivals
    and non-exponential service. Returns None when no n <= max_n satisfies
    the budget.
    """
    if arrival_rps <= 0 or service_s <= 0:
        return 1
    if wait_budget_s <= 0:
        return None
    offered = arrival_rps * service_s
    for n in range(max(1, math.ceil(offered)), max_n + 1):
        if n <= offered:
            continue
        wait = erlang_c(n, offered) * service_s / (n - offered) * variability
        if wait <= wait_budget_s:
            return n
    return None


class ThroughputScalingMixin:
    """Traffic-driven throughput-based scaling decisions."""

    # Scratch fields owned by PlannerScalingState, declared here for mypy
    _diag_predicted_num_req: Optional[float]
    _diag_predicted_isl: Optional[float]
    _diag_predicted_osl: Optional[float]
    _diag_predicted_kv_hit_rate: Optional[float]
    _diag_engine_rps_prefill: Optional[float]
    _diag_engine_rps_decode: Optional[float]
    _diag_throughput_reason: Optional[str]
    _diag_throughput_reason_prefill: Optional[str]
    _diag_throughput_reason_decode: Optional[str]
    _traffic_shape_provider: Optional[Callable[[], Optional[TrafficShape]]]
    _last_erlang_bound_p: int

    def _throughput_single(
        self,
        demand_rps: float,
        isl: float,
        osl: float,
        component: str,
        kv_hit_rate: Optional[float] = None,
    ) -> Optional[ScalingDecision]:
        desired = (
            self._compute_prefill_replicas(demand_rps, isl, osl, kv_hit_rate)
            if component == "prefill"
            else self._compute_decode_replicas(demand_rps, isl, osl)
        )
        if desired is None:
            return None

        if self._config.enable_load_scaling:
            if component == "prefill":
                self._throughput_lower_bound_p = desired
            else:
                self._throughput_lower_bound_d = desired
            logger.info(f"Throughput lower bound set to {desired} for {component}")
            self._diag_throughput_reason = "set_lower_bound"
            return None

        desired = self._apply_single_budget(desired, component)
        self._diag_throughput_reason = "scale"
        return (
            ScalingDecision(num_prefill=desired)
            if component == "prefill"
            else ScalingDecision(num_decode=desired)
        )

    def _throughput_disagg(
        self,
        demand_rps: float,
        isl: float,
        osl: float,
        kv_hit_rate: Optional[float] = None,
    ) -> Optional[ScalingDecision]:
        num_p = self._compute_prefill_replicas(demand_rps, isl, osl, kv_hit_rate)
        num_d = self._compute_decode_replicas(demand_rps, isl, osl)
        # _compute_* sets _diag_throughput_reason = "model_not_ready" when
        # the perf model cannot estimate yet. If one side is not ready, the other
        # side's computation was still valid but its decision is blocked,
        # so we label it "partner_not_ready" to keep per-component
        # diagnostics consistent with the aggregate reason.
        if num_p is None or num_d is None:
            self._diag_throughput_reason_prefill = (
                "model_not_ready" if num_p is None else "partner_not_ready"
            )
            self._diag_throughput_reason_decode = (
                "model_not_ready" if num_d is None else "partner_not_ready"
            )
            return None

        reason = "set_lower_bound" if self._config.enable_load_scaling else "scale"
        self._diag_throughput_reason_prefill = reason
        self._diag_throughput_reason_decode = reason

        if self._config.enable_load_scaling:
            self._throughput_lower_bound_p = num_p
            self._throughput_lower_bound_d = num_d
            logger.info(f"Throughput lower bounds set: prefill={num_p}, decode={num_d}")
            self._diag_throughput_reason = "set_lower_bound"
            return None

        num_p, num_d = self._apply_global_budget(num_p, num_d)
        self._diag_throughput_reason = "scale"
        return ScalingDecision(num_prefill=num_p, num_decode=num_d)

    def _throughput_agg(
        self,
        demand_rps: float,
        isl: float,
        osl: float,
        kv_hit_rate: Optional[float] = None,
    ) -> Optional[ScalingDecision]:
        d_caps = self._capabilities.decode
        max_tokens = d_caps.max_num_batched_tokens if d_caps else None
        if not max_tokens or max_tokens <= 0:
            logger.warning(
                "max_num_batched_tokens not available, skipping agg throughput"
            )
            self._diag_throughput_reason = "model_not_ready"
            return None

        capacity = self._agg_regression.find_engine_capacity_rps(
            isl=isl,
            osl=osl,
            ttft_sla_ms=self._config.ttft_ms,
            itl_sla_ms=self._config.itl_ms,
            kv_hit_rate=kv_hit_rate,
            accept_length=self._current_decode_accept_length(),
        )
        engine_rps = capacity.rps if capacity is not None else 0.0
        if engine_rps <= 0:
            logger.warning("Agg perf model not ready, skipping throughput scaling")
            self._diag_throughput_reason = "model_not_ready"
            return None
        actual_ttft = capacity.ttft_ms or 0.0
        actual_itl = capacity.itl_ms or 0.0
        if (
            not capacity.eligible
            or actual_ttft > self._config.ttft_ms
            or actual_itl > self._config.itl_ms
        ):
            logger.warning(
                f"Agg SLA not fully met: TTFT={actual_ttft:.1f}ms, ITL={actual_itl:.1f}ms"
            )

        self._diag_engine_rps_prefill = engine_rps
        self._diag_engine_rps_decode = engine_rps

        desired = max(math.ceil(demand_rps / engine_rps), self._config.min_endpoint)
        logger.info(
            f"Agg: {demand_rps:.2f} rps / {engine_rps:.2f} engine_rps = {desired} replicas"
        )

        if self._config.enable_load_scaling:
            self._throughput_lower_bound_d = desired
            logger.info(f"Agg throughput lower bound set to {desired}")
            self._diag_throughput_reason = "set_lower_bound"
            return None

        desired = self._apply_single_budget(desired, "decode")
        self._diag_throughput_reason = "scale"
        return ScalingDecision(num_decode=desired)

    def _compute_prefill_replicas(
        self,
        demand_rps: float,
        isl: float,
        osl: float,
        kv_hit_rate: Optional[float] = None,
    ) -> Optional[int]:
        capacity = self._prefill_regression.find_engine_capacity_rps(
            isl=isl,
            osl=osl,
            ttft_sla_ms=self._config.ttft_ms,
            kv_hit_rate=kv_hit_rate,
        )
        engine_rps = capacity.rps if capacity is not None else 0.0
        if engine_rps <= 0:
            logger.warning("Prefill perf model not ready, skipping throughput scaling")
            self._diag_throughput_reason = "model_not_ready"
            return None
        ttft_ms = capacity.ttft_ms or 0.0
        sla_floor = 1
        if not capacity.eligible or ttft_ms > self._config.ttft_ms:
            logger.warning(
                f"Prefill TTFT SLA not met: {ttft_ms:.1f}ms > {self._config.ttft_ms:.1f}ms"
            )
            # Latency-driven floor
            sla_floor = math.ceil(ttft_ms / self._config.ttft_ms)

        if self._config.prefill_sizing_mode == "erlang_c":
            return self._prefill_replicas_erlang(
                demand_rps, isl, kv_hit_rate, engine_rps
            )

        self._diag_engine_rps_prefill = engine_rps
        util_target = self._config.throughput_utilization_target
        effective_rps = engine_rps * util_target

        result = max(
            math.ceil(demand_rps / effective_rps), sla_floor, self._config.min_endpoint
        )
        logger.info(
            f"Prefill: {demand_rps:.2f} rps / (engine_rps={engine_rps:.2f} × "
            f"util_target={util_target:.2f} = {effective_rps:.2f}) = {result}, "
            f"est_ttft={ttft_ms:.1f}ms, isl_raw={isl:.1f}, "
            f"kv_hit_rate={kv_hit_rate or 0.0:.3f}"
        )
        return result

    def _prefill_replicas_erlang(
        self,
        demand_rps: float,
        isl: float,
        kv_hit_rate: Optional[float],
        aic_engine_rps: float,
    ) -> int:
        """DEEPINFRA: queueing-derived prefill sizing (Erlang-C/Allen-Cunneen).

        Smallest N whose predicted mean queue wait fits the TTFT wait budget
        ``ttft_sla - overhead - service``, with service time from the
        FPM-measured per-token slope (falling back to kappa-corrected AIC) and
        the variability correction (Ca^2 + Cs^2)/2 from measured traffic shape
        (falling back to configured constants). ``prefill_rho_ceiling`` is an
        operator guardrail applied on top.
        """
        cfg = self._config
        hit = min(max(kv_hit_rate or 0.0, 0.0), 0.99)
        eff_tokens = max(1.0, isl * (1.0 - hit))

        service_s = self._prefill_regression.measured_prefill_service_seconds(
            eff_tokens, cfg.prefill_service_overhead_s
        )
        service_source = "fpm_slope"
        if service_s is None or service_s <= 0:
            # AIC's batch-1 iteration time, corrected by the measured
            # realization factor (iteration time != sustainable rate).
            service_s = 1.0 / (aic_engine_rps * cfg.prefill_aic_service_kappa)
            service_source = "aic_kappa"

        cs2 = cfg.prefill_service_scv
        cs2_source = "config"
        shape: Optional[TrafficShape] = None
        provider = getattr(self, "_traffic_shape_provider", None)
        if cfg.prefill_measure_traffic_shape and provider is not None:
            try:
                shape = provider()
            except Exception as e:  # noqa: BLE001 - shape is best-effort
                logger.warning("Traffic shape provider failed: %s", e)
        if shape is not None and shape.isl_scv is not None:
            hit_scv = (
                shape.one_minus_hit_scv
                if shape.one_minus_hit_scv is not None
                else 0.5
            )
            # eff = isl * (1 - hit), independent marginals
            scv_eff = (1.0 + shape.isl_scv) * (1.0 + hit_scv) - 1.0
            # Fixed per-request overhead damps token variability:
            # S = c + tok/R  =>  Cs^2 = SCV(tok) * (E[tok/R] / E[S])^2
            token_share = max(0.0, service_s - cfg.prefill_service_overhead_s)
            damp = (token_share / service_s) ** 2 if service_s > 0 else 1.0
            cs2 = scv_eff * damp
            cs2_source = "measured"

        variability = (cfg.prefill_arrival_scv + cs2) / 2.0
        wait_budget_s = (
            cfg.ttft_ms / 1000.0 - cfg.prefill_ttft_overhead_ms / 1000.0 - service_s
        )
        offered = demand_rps * service_s
        n_rho_ceiling = max(1, math.ceil(offered / cfg.prefill_rho_ceiling))

        if wait_budget_s <= 0:
            # Even a lone request can't meet the SLA at this shape; replicas
            # can't fix service time. Provision to the utilization guardrail.
            logger.warning(
                "Erlang-C prefill: TTFT budget infeasible (service=%.0fms + "
                "overhead=%.0fms > sla=%.0fms); best-effort N=%d at "
                "rho_ceiling=%.2f",
                service_s * 1000, cfg.prefill_ttft_overhead_ms, cfg.ttft_ms,
                n_rho_ceiling, cfg.prefill_rho_ceiling,
            )
            self._diag_engine_rps_prefill = 1.0 / service_s
            return max(n_rho_ceiling, cfg.min_endpoint)

        n_queue = erlang_c_min_servers(
            demand_rps, service_s, wait_budget_s, variability
        )
        if n_queue is None:
            logger.warning(
                "Erlang-C prefill: no N<=%d meets wait budget %.0fms; "
                "falling back to rho ceiling N=%d",
                _ERLANG_MAX_N, wait_budget_s * 1000, n_rho_ceiling,
            )
            n_queue = n_rho_ceiling

        result = max(n_queue, n_rho_ceiling, cfg.min_endpoint)

        # Down-hysteresis (Schmitt trigger): near an integer boundary the
        # prescription flips with a few percent of input noise (observed
        # 2026-07-03 10-12Z: 16 confirmed 3->2->3 recommendation cycles in 2h
        # at offered~1.6). Only lower the prescription when the formula still
        # says "lower" with demand padded by prefill_down_demand_pad; raise
        # immediately as before.
        last_n = self._last_erlang_bound_p
        hysteresis_held = False
        if 0 < result < last_n:
            pad = cfg.prefill_down_demand_pad
            n_padded = erlang_c_min_servers(
                demand_rps * pad, service_s, wait_budget_s, variability
            )
            n_padded = max(
                n_padded if n_padded is not None else last_n,
                math.ceil(offered * pad / cfg.prefill_rho_ceiling),
                cfg.min_endpoint,
            )
            if n_padded >= last_n:
                result = last_n
                hysteresis_held = True
        self._last_erlang_bound_p = result

        self._diag_engine_rps_prefill = 1.0 / service_s
        logger.info(
            "Prefill[erlang_c]: %.2f rps, eff_tokens=%.0f (isl=%.1f hit=%.3f), "
            "S=%.1fms (%s), Ca2=%.1f Cs2=%.1f (%s), offered=%.2f, "
            "wait_budget=%.0fms -> N=%d (queue=%d rho_ceil=%d min=%d%s)",
            demand_rps, eff_tokens, isl, hit,
            service_s * 1000, service_source,
            cfg.prefill_arrival_scv, cs2, cs2_source, offered,
            wait_budget_s * 1000, result, n_queue, n_rho_ceiling,
            cfg.min_endpoint,
            " hysteresis_hold" if hysteresis_held else "",
        )
        return result

    def _compute_decode_replicas(
        self, demand_rps: float, isl: float, osl: float
    ) -> Optional[int]:
        accept_length = self._current_decode_accept_length()
        capacity = self._decode_regression.find_engine_capacity_rps(
            isl=isl,
            osl=osl,
            itl_sla_ms=self._config.itl_ms,
            accept_length=accept_length,
        )
        engine_rps = capacity.rps if capacity is not None else 0.0
        if engine_rps <= 0:
            logger.warning("Decode perf model not ready, skipping throughput scaling")
            self._diag_throughput_reason = "model_not_ready"
            return None
        itl_ms = capacity.itl_ms or 0.0
        if not capacity.eligible or itl_ms > self._config.itl_ms:
            logger.warning(
                f"Decode ITL SLA not met: {itl_ms:.1f}ms > {self._config.itl_ms:.1f}ms"
            )

        self._diag_engine_rps_decode = engine_rps

        result = max(math.ceil(demand_rps / engine_rps), self._config.min_endpoint)
        logger.info(
            f"Decode: {demand_rps:.2f} rps / {engine_rps:.2f} = {result}, "
            f"est_itl={itl_ms:.1f}ms, accept_length={accept_length:.2f}"
        )
        return result
