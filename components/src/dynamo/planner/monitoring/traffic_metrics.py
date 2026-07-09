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

import logging
import math
import time
import typing
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from prometheus_api_client import PrometheusConnect
from pydantic import BaseModel, ValidationError

from dynamo import prometheus_names
from dynamo.planner.core.types import TrafficShape
from dynamo.runtime.logging import configure_dynamo_logging


def _histogram_moments(
    bucket_sums: Dict[str, float],
    *,
    log_spaced: bool,
    calibrate_mean: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    """(mean, second moment, count) from cumulative histogram buckets.

    Each bucket's mass is placed at a representative point: the geometric mean
    of its edges for log-spaced buckets (frontend token histograms), the
    midpoint for linear ones (router hit-rate histogram); ``1.5 * lower`` for
    the +Inf bucket and ``upper / 2`` for the first. When ``calibrate_mean``
    (the exact ``_sum/_count`` mean) is given, points are rescaled so the
    histogram mean matches it — removing most binning bias where it matters.
    Returns None when the buckets are empty or malformed.
    """
    try:
        rows: List[Tuple[float, float]] = sorted(
            (math.inf if le in ("+Inf", "Inf", "inf") else float(le), v)
            for le, v in bucket_sums.items()
        )
    except ValueError:
        return None
    if not rows:
        return None

    masses: List[float] = []
    points: List[float] = []
    prev_le, prev_cum = 0.0, 0.0
    for le, cum in rows:
        mass = cum - prev_cum
        if mass > 0:
            if math.isinf(le):
                point = prev_le * 1.5
            elif prev_le <= 0:
                point = le / 2.0
            elif log_spaced:
                point = math.sqrt(prev_le * le)
            else:
                point = (prev_le + le) / 2.0
            masses.append(mass)
            points.append(point)
        prev_le, prev_cum = le, cum

    n = sum(masses)
    if n <= 0:
        return None
    mean = sum(m * p for m, p in zip(masses, points)) / n
    if calibrate_mean is not None and calibrate_mean > 0 and mean > 0:
        scale = calibrate_mean / mean
        points = [p * scale for p in points]
        mean = calibrate_mean
    m2 = sum(m * p * p for m, p in zip(masses, points)) / n
    return mean, m2, n


class _BearerTokenFileAuth:
    """Auth callable that re-reads a bearer token from disk on every request.

    Assigned to requests.Session.auth. Any callable that takes a PreparedRequest
    and returns it qualifies — no AuthBase subclass required.
    Useful for rotating tokens (Kubernetes projected ServiceAccount tokens).
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def __call__(self, request):  # type: ignore[override]
        with open(self._path) as f:
            token = f.read().strip()
        request.headers["Authorization"] = f"Bearer {token}"
        return request


configure_dynamo_logging()
logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    ttft: Optional[float] = None
    itl: Optional[float] = None
    num_req: Optional[float] = None
    isl: Optional[float] = None
    osl: Optional[float] = None
    request_duration: Optional[float] = None
    p_load: Optional[float] = None
    d_load: Optional[float] = None
    kv_hit_rate: Optional[float] = None
    accept_length: Optional[float] = None

    def normalize_idle_nans(self) -> list[str]:
        """Replace undefined averages only for a confirmed idle window."""
        if self.num_req != 0:
            return []

        normalized: list[str] = []
        for field_name in ("ttft", "itl", "isl", "osl", "request_duration"):
            value = getattr(self, field_name)
            if value is not None and math.isnan(value):
                setattr(self, field_name, 0.0)
                normalized.append(field_name)
        return normalized

    def is_valid(self) -> bool:
        """Check if all required metrics are valid (not None and not NaN)."""
        required = [
            self.ttft,
            self.itl,
            self.isl,
            self.osl,
            self.num_req,
            self.request_duration,
        ]
        return all(v is not None and not math.isnan(v) for v in required)


class FrontendMetric(BaseModel):
    container: typing.Optional[str] = None
    dynamo_namespace: typing.Optional[str] = None
    endpoint: typing.Optional[str] = None
    instance: typing.Optional[str] = None
    job: typing.Optional[str] = None
    model: typing.Optional[str] = None
    namespace: typing.Optional[str] = None
    pod: typing.Optional[str] = None


class FrontendMetricContainer(BaseModel):
    metric: FrontendMetric
    value: typing.Tuple[float, float]  # [timestamp, value]


class PrometheusAPIClient:
    def __init__(
        self,
        url: str,
        dynamo_namespace: str,
        metrics_source: str = "frontend",
        bearer_token: Optional[str] = None,
        bearer_token_file: Optional[str] = None,
        ssl_verify: bool = False,
        extra_query_params: Optional[Dict[str, str]] = None,
        ca_bundle: Optional[str] = None,
    ):
        self.prom = PrometheusConnect(url=url, disable_ssl=not ssl_verify)
        if bearer_token:
            self.prom._session.headers["Authorization"] = f"Bearer {bearer_token}"
        if bearer_token_file:
            self.prom._session.auth = _BearerTokenFileAuth(bearer_token_file)
        if extra_query_params:
            self.prom._session.params = dict(extra_query_params)
        if ca_bundle:
            self.prom._session.verify = ca_bundle
        self.dynamo_namespace = dynamo_namespace
        self.metrics_source = metrics_source  # "frontend" | "router"
        # DEEPINFRA: cached traffic-shape stats (second moments drift over
        # hours; sample coarsely instead of per tick).
        self._shape_cache: Optional[Tuple[float, "TrafficShape"]] = None

    def _frontend_metric_name(self, metric_name: str) -> str:
        if metric_name.startswith(prometheus_names.name_prefix.FRONTEND):
            return metric_name
        return f"{prometheus_names.name_prefix.FRONTEND}_{metric_name}"

    def _sum_frontend_metric(self, result, model_name: str) -> Optional[float]:
        if not result:
            return None

        metrics_containers = parse_frontend_metric_containers(result)
        total = 0.0
        matched = False
        for container in metrics_containers:
            # Frontend lowercases model names in Prometheus labels.
            if (
                container.metric.model
                and container.metric.model.lower() == model_name.lower()
                and container.metric.dynamo_namespace == self.dynamo_namespace
                and not math.isnan(container.value[1])
            ):
                matched = True
                total += container.value[1]
        return total if matched else None

    def _get_average_metric(
        self,
        full_metric_name: str,
        interval: str,
        operation_name: str,
        model_name: Optional[str] = None,
    ) -> float:
        """Query average histogram metric.

        When model_name is None (router source): queries aggregate metrics via
        sum(increase(metric_sum[interval])) / sum(increase(metric_count[interval])),
        filtered by dynamo_namespace. DYN_NAMESPACE uses dashes but Prometheus labels
        use underscores, so dashes are normalized before building the PromQL filter.

        When model_name is provided (frontend source): queries per-model metrics
        via increase(metric_sum)/increase(metric_count), filtered by model and
        dynamo_namespace labels. The dynamo_frontend_ prefix is prepended
        automatically if absent.

        Returns:
            Average metric value, or 0 if no data/error.
        """
        try:
            if model_name is None:
                # Router aggregate path: filter by dynamo_namespace so each pool
                # planner only reads its own LocalRouter's metrics.
                # dynamo_component_router_* metrics are registered via MetricsHierarchy
                # which auto-injects dynamo_namespace with underscores (e.g.
                # "darfeen_dynamo_cloud_gp_prefill_1"). DYN_NAMESPACE uses dashes, so
                # normalize before building the PromQL filter.
                ns = self.dynamo_namespace.replace("-", "_")
                ns_filter = f'{prometheus_names.labels.NAMESPACE}="{ns}"'
                query = (
                    f"sum(increase({full_metric_name}_sum{{{ns_filter}}}[{interval}])) / "
                    f"sum(increase({full_metric_name}_count{{{ns_filter}}}[{interval}]))"
                )
                result = self.prom.custom_query(query=query)
                if not result:
                    logger.warning(
                        f"No prometheus metric data available for {full_metric_name}, use 0 instead"
                    )
                    return 0
                value = float(result[0]["value"][1])
                return 0 if math.isnan(value) else value
            else:
                # Frontend per-model path: filter by model and dynamo_namespace labels
                if not full_metric_name.startswith(
                    prometheus_names.name_prefix.FRONTEND
                ):
                    full_metric_name = (
                        f"{prometheus_names.name_prefix.FRONTEND}_{full_metric_name}"
                    )
                query = f"increase({full_metric_name}_sum[{interval}])/increase({full_metric_name}_count[{interval}])"
                result = self.prom.custom_query(query=query)
                if not result:
                    logger.warning(
                        f"No prometheus metric data available for {full_metric_name}, use 0 instead"
                    )
                    return 0
                metrics_containers = parse_frontend_metric_containers(result)
                values = []
                for container in metrics_containers:
                    # Frontend lowercases model names for Prometheus labels so we need to do case-insensitive comparison
                    if (
                        container.metric.model
                        and container.metric.model.lower() == model_name.lower()
                        and container.metric.dynamo_namespace == self.dynamo_namespace
                    ):
                        values.append(container.value[1])
                if not values:
                    logger.warning(
                        f"No prometheus metric data available for {full_metric_name} with model {model_name} and dynamo namespace {self.dynamo_namespace}, use 0 instead"
                    )
                    return 0
                return sum(values) / len(values)
        except Exception as e:
            logger.error(f"Error getting {operation_name}: {e}")
            return 0

    def get_avg_inter_token_latency(self, interval: str, model_name: str):
        if self.metrics_source == "router":
            return self._get_average_metric(
                f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.INTER_TOKEN_LATENCY_SECONDS}",
                interval,
                "avg inter token latency",
            )
        return self._get_average_metric(
            prometheus_names.frontend_service.INTER_TOKEN_LATENCY_SECONDS,
            interval,
            "avg inter token latency",
            model_name,
        )

    def get_avg_time_to_first_token(self, interval: str, model_name: str):
        if self.metrics_source == "router":
            return self._get_average_metric(
                f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.TIME_TO_FIRST_TOKEN_SECONDS}",
                interval,
                "avg time to first token",
            )
        return self._get_average_metric(
            prometheus_names.frontend_service.TIME_TO_FIRST_TOKEN_SECONDS,
            interval,
            "avg time to first token",
            model_name,
        )

    def get_avg_request_duration(self, interval: str, model_name: str):
        if self.metrics_source == "router":
            # TODO: Replace work_handler.REQUEST_DURATION_SECONDS with
            #       prometheus_names.router.REQUEST_DURATION_SECONDS once
            #       RouterRequestMetrics in lib/llm/src/kv_router/metrics.rs
            #       registers dynamo_component_router_request_duration_seconds.
            #       Until then this queries a non-existent metric and returns 0,
            #       which causes throughput planning to see
            #       concurrency=0 (under-estimated), inflating replica recommendations.
            return self._get_average_metric(
                f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.work_handler.REQUEST_DURATION_SECONDS}",
                interval,
                "avg request duration",
            )
        return self._get_average_metric(
            prometheus_names.frontend_service.REQUEST_DURATION_SECONDS,
            interval,
            "avg request duration",
            model_name,
        )

    def get_avg_request_count(self, interval: str, model_name: str):
        if self.metrics_source == "router":
            try:
                router_req_total = f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.REQUESTS_TOTAL}"
                ns = self.dynamo_namespace.replace("-", "_")
                ns_filter = f'{prometheus_names.labels.NAMESPACE}="{ns}"'
                query = f"sum(increase({router_req_total}{{{ns_filter}}}[{interval}]))"
                result = self.prom.custom_query(query=query)
                if not result:
                    logger.warning(
                        f"No prometheus metric data available for "
                        f"{router_req_total}, use 0 instead"
                    )
                    return 0
                value = float(result[0]["value"][1])
                return 0 if math.isnan(value) else value
            except Exception as e:
                logger.error(f"Error getting avg request count: {e}")
                return 0
        # This function follows a different query pattern than the other metrics:
        # use frontend-started requests so throughput planning sees offered load,
        # not only completed responses.
        try:
            requests_started_metric = self._frontend_metric_name(
                prometheus_names.frontend_service.REQUESTS_STARTED_TOTAL
            )
            started_res = self.prom.custom_query(
                query=f"increase({requests_started_metric}[{interval}])"
            )
            started_count = self._sum_frontend_metric(started_res, model_name)
            if started_count is not None:
                return started_count

            logger.warning(
                f"No prometheus metric data available for {requests_started_metric} "
                f"with model {model_name} and dynamo namespace "
                f"{self.dynamo_namespace}; falling back to completed request count"
            )

            requests_total_metric = self._frontend_metric_name(
                prometheus_names.frontend_service.REQUESTS_TOTAL
            )
            completed_res = self.prom.custom_query(
                query=f"increase({requests_total_metric}[{interval}])"
            )
            completed_count = self._sum_frontend_metric(completed_res, model_name)
            return completed_count or 0
        except Exception as e:
            logger.error(f"Error getting avg request count: {e}")
            return 0

    # DEEPINFRA: ISL and OSL have heavy right tails (median ~500 tokens,
    # p99 ~8k-30k). The mean over the caller's throughput-adjustment window
    # (30s) is genuinely correct for Little's-law capacity sizing, but is
    # statistically unstable at that sample size — one window catches the
    # densest moment of a long-completion cluster and the mean spikes
    # (observed: 30s mean jumped from ~900 to 2046 in a single tick, while
    # the 5m mean rose only to 1101). Override these specific queries to a
    # 5m window so the planner sees a stable mean that still includes long
    # sequences' full contribution to total decode work. Reaction lag to
    # genuine distribution shifts becomes ~5min — acceptable because OSL
    # distribution character changes over hours, not seconds.
    _ISL_OSL_AVG_INTERVAL: str = "5m"

    def get_avg_input_sequence_tokens(self, interval: str, model_name: str):
        if self.metrics_source == "router":
            return self._get_average_metric(
                f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.INPUT_SEQUENCE_TOKENS}",
                self._ISL_OSL_AVG_INTERVAL,
                "avg input sequence tokens",
            )
        return self._get_average_metric(
            prometheus_names.frontend_service.INPUT_SEQUENCE_TOKENS,
            self._ISL_OSL_AVG_INTERVAL,
            "avg input sequence tokens",
            model_name,
        )

    def get_avg_output_sequence_tokens(self, interval: str, model_name: str):
        if self.metrics_source == "router":
            return self._get_average_metric(
                f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.OUTPUT_SEQUENCE_TOKENS}",
                self._ISL_OSL_AVG_INTERVAL,
                "avg output sequence tokens",
            )
        return self._get_average_metric(
            prometheus_names.frontend_service.OUTPUT_SEQUENCE_TOKENS,
            self._ISL_OSL_AVG_INTERVAL,
            "avg output sequence tokens",
            model_name,
        )

    def get_avg_kv_hit_rate(self, interval: str, model_name: str) -> Optional[float]:
        """Average predicted KV cache hit rate (0.0-1.0) from the router.

        The histogram lives on the router component, but it can be exposed on
        the frontend scrape endpoint when the frontend runs in KV router mode.
        Query the router component metric regardless of the traffic metrics
        source so deployments can keep frontend-sourced request/ISL/OSL metrics
        while still using router-sourced KV hit rate.

        Returns ``None`` (not ``0.0``) on missing data — Prometheus scrape
        gaps must not be confused with a real "no reuse" signal: the state
        machine treats a real ``0.0`` as a valid observation and would
        otherwise drag the predictor / sticky value down toward zero on
        every scrape failure. The caller's ``_clamp_kv_hit_rate(None)``
        falls back to no-discount behavior, which is the safe choice.
        """
        full_metric_name = (
            f"{prometheus_names.name_prefix.COMPONENT}_"
            f"{prometheus_names.router.KV_HIT_RATE}"
        )
        try:
            ns = self.dynamo_namespace.replace("-", "_")
            ns_filter = f'{prometheus_names.labels.NAMESPACE}="{ns}"'
            query = (
                f"sum(increase({full_metric_name}_sum{{{ns_filter}}}[{interval}])) / "
                f"sum(increase({full_metric_name}_count{{{ns_filter}}}[{interval}]))"
            )
            result = self.prom.custom_query(query=query)
            if not result:
                logger.info(
                    f"No prometheus data for {full_metric_name}, returning None"
                )
                return None
            value = float(result[0]["value"][1])
            return None if math.isnan(value) else value
        except Exception as e:
            logger.warning(f"Error getting avg kv hit rate: {e}")
            return None

    def scrape_gap_recent(
        self,
        model_name: str,
        lookback_s: int = 90,
        scrape_period_s: int = 15,
        min_healthy_fraction: float = 0.6,
    ) -> bool:
        """DEEPINFRA: True when the request counter is missing recent samples.

        A Prometheus outage poisons count-type inputs twice: an empty read
        during the gap, then one ``increase()`` window holding the whole
        gap's counter delta (the engine looks back past the window start for
        the previous point). Both poisoned reads occur while the lookback
        window is missing samples — so sample completeness, not value
        magnitude, is the reliable discriminator (a genuine demand surge has
        a full complement of samples and must never be suppressed).

        Checks ``count_over_time`` of the request counter over ``lookback_s``
        per series (frontend pods) and reports a gap when any series has
        fewer than ``min_healthy_fraction`` of the expected samples, when no
        series matches, or when the query itself fails (Prometheus down).
        """
        metric = self._frontend_metric_name(
            prometheus_names.frontend_service.REQUESTS_STARTED_TOTAL
        )
        expected = lookback_s / scrape_period_s
        threshold = expected * min_healthy_fraction
        try:
            result = self.prom.custom_query(
                query=f"count_over_time({metric}[{lookback_s}s])"
            )
        except Exception as e:  # noqa: BLE001 - treat query failure as a gap
            logger.warning("Scrape-gap check query failed (%s): assuming gap", e)
            return True
        counts = []
        for entry in result or []:
            labels = entry.get("metric", {})
            if labels.get("model", "").lower() != model_name.lower():
                continue
            if labels.get("dynamo_namespace") != self.dynamo_namespace:
                continue  # skip the duplicate service-job scrape
            try:
                counts.append(float(entry["value"][1]))
            except (KeyError, IndexError, ValueError):
                continue
        if not counts:
            logger.warning(
                "Scrape-gap check: no per-pod series for %s: assuming gap",
                model_name,
            )
            return True
        worst = min(counts)
        if worst < threshold:
            logger.warning(
                "Scrape-gap check: series has %.0f/%.0f expected samples in "
                "the last %ds — metrics gap in progress or just ended",
                worst, expected, lookback_s,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # DEEPINFRA: traffic-shape (second moment) measurement for Erlang-C
    # prefill sizing. Means come from _sum/_count ratios (scrape-duplication
    # safe); shapes come from histogram buckets, which must exclude the
    # duplicate service-job scrape — the per-pod scrape is identified by the
    # presence of the dynamo_namespace label (frontend metrics) or its
    # dash-form value (router metrics, stamped from pod labels).
    # ------------------------------------------------------------------

    def _filtered_bucket_sums(
        self,
        metric: str,
        interval: str,
        match: typing.Callable[[dict], bool],
    ) -> Dict[str, float]:
        """Per-``le`` sums of ``increase(metric[interval])`` over series
        accepted by ``match``. Returns {} on missing data."""
        result = self.prom.custom_query(query=f"increase({metric}[{interval}])")
        sums: Dict[str, float] = {}
        for entry in result or []:
            labels = entry.get("metric", {})
            if not match(labels):
                continue
            le = labels.get("le")
            if le is None:
                continue
            try:
                value = float(entry["value"][1])
            except (KeyError, IndexError, ValueError):
                continue
            if math.isnan(value):
                continue
            sums[le] = sums.get(le, 0.0) + value
        return sums

    def _filtered_scalar_sum(
        self,
        metric: str,
        interval: str,
        match: typing.Callable[[dict], bool],
    ) -> Optional[float]:
        result = self.prom.custom_query(query=f"increase({metric}[{interval}])")
        total, matched = 0.0, False
        for entry in result or []:
            if not match(entry.get("metric", {})):
                continue
            try:
                value = float(entry["value"][1])
            except (KeyError, IndexError, ValueError):
                continue
            if math.isnan(value):
                continue
            total += value
            matched = True
        return total if matched else None

    def get_traffic_shape(
        self,
        model_name: str,
        window: str = "1h",
        ttl_s: float = 300.0,
        min_samples: float = 2000.0,
    ) -> "TrafficShape":
        """Shape statistics (SCVs) of ISL and router kv-hit distributions.

        Cached for ``ttl_s``; individual fields are ``None`` when the source
        histogram is missing or has fewer than ``min_samples`` observations
        in ``window`` — callers fall back to configured defaults.
        """
        now = time.monotonic()
        if self._shape_cache is not None and now - self._shape_cache[0] < ttl_s:
            return self._shape_cache[1]

        shape = TrafficShape()
        try:
            shape.isl_scv, shape.isl_samples = self._measure_isl_scv(
                model_name, window, min_samples
            )
        except Exception as e:  # noqa: BLE001 - shape is best-effort
            logger.warning("Traffic shape: ISL SCV measurement failed: %s", e)
        try:
            (
                shape.one_minus_hit_scv,
                shape.hit_samples,
            ) = self._measure_one_minus_hit_scv(window, min_samples)
        except Exception as e:  # noqa: BLE001 - shape is best-effort
            logger.warning("Traffic shape: kv-hit SCV measurement failed: %s", e)

        self._shape_cache = (now, shape)
        logger.info(
            "Traffic shape refreshed: isl_scv=%s (n=%.0f) one_minus_hit_scv=%s "
            "(n=%.0f)",
            None if shape.isl_scv is None else f"{shape.isl_scv:.2f}",
            shape.isl_samples,
            None
            if shape.one_minus_hit_scv is None
            else f"{shape.one_minus_hit_scv:.2f}",
            shape.hit_samples,
        )
        return shape

    def _measure_isl_scv(
        self, model_name: str, window: str, min_samples: float
    ) -> Tuple[Optional[float], float]:
        metric = self._frontend_metric_name(
            prometheus_names.frontend_service.INPUT_SEQUENCE_TOKENS
        )

        def match(labels: dict) -> bool:
            # Per-pod scrape only: the duplicate service-job scrape carries no
            # dynamo_namespace label.
            return (
                labels.get("model", "").lower() == model_name.lower()
                and labels.get("dynamo_namespace") == self.dynamo_namespace
            )

        buckets = self._filtered_bucket_sums(f"{metric}_bucket", window, match)
        exact_sum = self._filtered_scalar_sum(f"{metric}_sum", window, match)
        exact_count = self._filtered_scalar_sum(f"{metric}_count", window, match)
        exact_mean = (
            exact_sum / exact_count if exact_sum and exact_count else None
        )
        moments = _histogram_moments(
            buckets, log_spaced=True, calibrate_mean=exact_mean
        )
        if moments is None or moments[2] < min_samples:
            return None, moments[2] if moments else 0.0
        mean, m2, n = moments
        if mean <= 0:
            return None, n
        return max(0.0, (m2 - mean * mean) / (mean * mean)), n

    def _measure_one_minus_hit_scv(
        self, window: str, min_samples: float
    ) -> Tuple[Optional[float], float]:
        metric = (
            f"{prometheus_names.name_prefix.COMPONENT}_"
            f"{prometheus_names.router.KV_HIT_RATE}"
        )
        ns_dash = self.dynamo_namespace
        ns_under = self.dynamo_namespace.replace("-", "_")

        def match_dash(labels: dict) -> bool:
            return labels.get("dynamo_namespace") == ns_dash

        def match_under(labels: dict) -> bool:
            return labels.get("dynamo_namespace") == ns_under

        # Prefer the per-pod scrape (dash-form namespace stamped from pod
        # labels); fall back to the native underscore form for deployments
        # without that relabeling.
        buckets = self._filtered_bucket_sums(f"{metric}_bucket", window, match_dash)
        match = match_dash
        if not buckets:
            buckets = self._filtered_bucket_sums(
                f"{metric}_bucket", window, match_under
            )
            match = match_under
        exact_sum = self._filtered_scalar_sum(f"{metric}_sum", window, match)
        exact_count = self._filtered_scalar_sum(f"{metric}_count", window, match)
        exact_mean = (
            exact_sum / exact_count if exact_sum is not None and exact_count else None
        )
        moments = _histogram_moments(
            buckets, log_spaced=False, calibrate_mean=exact_mean
        )
        if moments is None or moments[2] < min_samples:
            return None, moments[2] if moments else 0.0
        hit_mean, hit_m2, n = moments
        hit_var = max(0.0, hit_m2 - hit_mean * hit_mean)
        one_minus = 1.0 - hit_mean
        if one_minus < 0.05:
            # Near-total cache reuse: (1-hit) SCV is numerically unstable and
            # the service time is overhead-dominated anyway.
            return None, n
        return hit_var / (one_minus * one_minus), n

    @staticmethod
    def _quote_label_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _engine_metric_filter(
        self,
        component_name: Optional[str],
        model_name: Optional[str],
        namespace: Optional[str] = None,
        endpoint_name: Optional[str] = None,
    ) -> str:
        metric_namespace = namespace or self.dynamo_namespace
        metric_endpoint = endpoint_name or "generate"
        filters = [
            f'{prometheus_names.labels.NAMESPACE}="{self._quote_label_value(metric_namespace)}"',
            f'{prometheus_names.labels.ENDPOINT}="{self._quote_label_value(metric_endpoint)}"',
        ]
        if component_name:
            filters.append(
                f'{prometheus_names.labels.COMPONENT}="{self._quote_label_value(component_name)}"'
            )
        if model_name:
            filters.append(
                f'{prometheus_names.labels.MODEL}="{self._quote_label_value(model_name)}"'
            )
        return ",".join(filters)

    def _query_single_value(self, query: str, operation_name: str) -> Optional[float]:
        try:
            result = self.prom.custom_query(query=query)
            if not result:
                logger.info(f"No prometheus data for {operation_name}")
                return None
            value = float(result[0]["value"][1])
            return value if math.isfinite(value) else None
        except Exception as e:
            logger.warning(f"Error getting {operation_name}: {e}")
            return None

    def get_avg_spec_decode_accept_length(
        self,
        interval: str,
        backend: str,
        component_name: Optional[str],
        model_name: Optional[str],
        namespace: Optional[str] = None,
        endpoint_name: Optional[str] = None,
    ) -> Optional[float]:
        """Average spec-decode accept length from worker engine metrics.

        Returns tokens produced per decode forward, including the base token.
        Missing data returns ``None`` so callers can fall back to no discount.
        """
        selector = self._engine_metric_filter(
            component_name, model_name, namespace, endpoint_name
        )
        if backend == "vllm":
            accepted = (
                f"sum(rate(vllm:spec_decode_num_accepted_tokens_total"
                f"{{{selector}}}[{interval}]))"
            )
            drafts = (
                f"sum(rate(vllm:spec_decode_num_drafts_total"
                f"{{{selector}}}[{interval}]))"
            )
            return self._query_single_value(
                f"1 + ({accepted}) / ({drafts})",
                "vLLM spec decode accept length",
            )
        if backend == "sglang":
            return self._query_single_value(
                f"avg(avg_over_time(sglang:spec_accept_length"
                f"{{{selector}}}[{interval}]))",
                "SGLang spec decode accept length",
            )
        if backend == "trtllm":
            return self._query_single_value(
                f"avg(avg_over_time(trtllm_spec_decode_acceptance_length"
                f"{{{selector}}}[{interval}]))",
                "TRT-LLM spec decode accept length",
            )
        return None

    def get_total_kv_blocks(
        self,
        component_name: Optional[str] = None,
    ) -> Optional[int]:
        """DEEPINFRA: query the TRT-LLM kv_cache_max_blocks gauge from VM.

        TRT-LLM workers expose ``trtllm_kv_cache_max_blocks`` per worker. The
        worker doesn't currently include ``total_kv_blocks`` in its
        ``ModelRuntimeConfig`` at registration time (TRT-LLM lacks a sync
        startup-time accessor — see ``trtllm/workers/llm_worker.py``
        ``total_kv_blocks`` TODO), so the planner's
        ``WorkerInfo.max_kv_tokens`` is ``None`` and both the consolidation
        feasibility check and the KV-saturation scale-up trigger silently
        no-op. Pull the value directly from VM as a fallback so KV-pressure
        protection works on trtllm.

        Returns None if no series found, the query fails, or values vary
        across workers (which would indicate a mis-config worth surfacing
        rather than averaging silently).
        """
        try:
            # trtllm_* metrics use the raw namespace (with dashes) on the
            # dynamo_namespace label — unlike dynamo_component_* metrics
            # which normalize to underscores. Use the namespace as-is.
            filters = [f'dynamo_namespace="{self.dynamo_namespace}"']
            if component_name:
                filters.append(f'dynamo_component="{component_name}"')
            query = f"trtllm_kv_cache_max_blocks{{{','.join(filters)}}}"
            result = self.prom.custom_query(query=query)
            if not result:
                logger.info(
                    f"No prometheus data for trtllm_kv_cache_max_blocks "
                    f"(component={component_name}, ns={self.dynamo_namespace})"
                )
                return None
            values = sorted(
                int(float(r["value"][1]))
                for r in result
                if not math.isnan(float(r["value"][1]))
            )
            if not values:
                return None
            # Workers backed by the same model+config should report identical
            # max_blocks. If the spread is wide, log a warning but still
            # return the minimum (conservative — gives the saturation check
            # a tighter ceiling).
            if values[-1] - values[0] > max(1, values[0] // 100):
                logger.warning(
                    f"trtllm_kv_cache_max_blocks varies across workers "
                    f"(min={values[0]}, max={values[-1]}); using min"
                )
            return values[0]
        except Exception as e:
            logger.warning(f"Error getting total_kv_blocks: {e}")
            return None

    def warn_if_router_not_scraped(self) -> None:
        """Warn if Prometheus is not scraping any dynamo_component_router_* series.

        Called once at planner startup when throughput_metrics_source="router".
        Detects a missing or misconfigured PodMonitor early so the operator
        sees a clear warning rather than silent zero metrics.

        Uses absent() to check whether any dynamo_component_router_requests_total
        series exist for this namespace. MetricsHierarchy injects dynamo_namespace
        with underscores, so DYN_NAMESPACE dashes are normalized before the query.
        """
        try:
            metric = f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.REQUESTS_TOTAL}"
            ns = self.dynamo_namespace.replace("-", "_")
            ns_filter = f'{prometheus_names.labels.NAMESPACE}="{ns}"'
            result = self.prom.custom_query(query=f"absent({metric}{{{ns_filter}}})")
            if result:
                logger.warning(
                    f"[throughput_metrics_source=router] No '{metric}' series found "
                    f"for namespace '{ns}' in Prometheus. "
                    "Router metrics will read as zero until scraping is working. "
                    "Check: (1) PodMonitor 'dynamo-router' is installed in the operator namespace, "
                    "(2) LocalRouter pods have DYN_SYSTEM_PORT=9090, "
                    "(3) pods have label nvidia.com/metrics-enabled=true."
                )
        except Exception as e:
            logger.warning(f"Could not check router scraping status: {e}")


def parse_frontend_metric_containers(
    result: list[dict],
) -> list[FrontendMetricContainer]:
    metrics_containers: list[FrontendMetricContainer] = []
    for res in result:
        try:
            metrics_containers.append(FrontendMetricContainer.model_validate(res))
        except ValidationError as e:
            logger.error(f"Error parsing frontend metric container: {e}")
            continue
    return metrics_containers
