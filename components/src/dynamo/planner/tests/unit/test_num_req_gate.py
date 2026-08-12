# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics-gap guard: gate on scrape-sample completeness, not value magnitude.

Guards the 2026-07-08 00:23-00:25Z incident (VM pod reschedule -> num_req read
0 rps then 152 rps at true ~44) without ever suppressing a genuine demand
surge — a real 3x+ spike arrives with a full complement of scrape samples.
"""

import pytest

from dynamo import prometheus_names

# Local dev envs may carry an older prometheus_names than the repo bindings
# (which define REQUESTS_STARTED_TOTAL = "requests_started_total").
if not hasattr(prometheus_names.frontend_service, "REQUESTS_STARTED_TOTAL"):
    prometheus_names.frontend_service.REQUESTS_STARTED_TOTAL = "requests_started_total"

from dynamo.planner.monitoring.traffic_metrics import PrometheusAPIClient

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _client(query_result=None, raises=False):
    c = PrometheusAPIClient.__new__(PrometheusAPIClient)
    c.dynamo_namespace = "test--ns"
    c.metrics_source = "frontend"

    class _Prom:
        def custom_query(self, query):
            if raises:
                raise RuntimeError("prometheus down")
            return query_result

    c.prom = _Prom()
    return c


def _series(count, model="m/x", ns="test--ns"):
    metric = {"model": model}
    if ns is not None:
        metric["dynamo_namespace"] = ns
    return {"metric": metric, "value": [0, str(count)]}


def test_healthy_scrapes_no_gap():
    # 90s lookback at 15s scrape = 6 expected; both pods full
    c = _client([_series(6), _series(6)])
    assert c.scrape_gap_recent("m/x") is False


def test_one_lost_scrape_tolerated():
    c = _client([_series(5), _series(6)])
    assert c.scrape_gap_recent("m/x") is False  # 5 >= 0.6*6


def test_gap_detected_when_samples_missing():
    # mid-gap or just after: only 1-2 samples in the lookback
    c = _client([_series(2), _series(6)])
    assert c.scrape_gap_recent("m/x") is True


def test_gap_when_no_series():
    c = _client([])
    assert c.scrape_gap_recent("m/x") is True


def test_gap_when_query_fails():
    c = _client(raises=True)
    assert c.scrape_gap_recent("m/x") is True


def test_duplicate_service_scrape_ignored():
    # tgi service-job series has no dynamo_namespace label; its (sparse)
    # sample count must not trigger the gap verdict
    tgi = _series(1, ns=None)  # helper omits the label when ns is None
    c = _client([_series(6), _series(6), tgi])
    assert c.scrape_gap_recent("m/x") is False


def test_other_model_series_ignored():
    c = _client([_series(6), _series(1, model="other/model")])
    assert c.scrape_gap_recent("m/x") is False


def test_genuine_surge_is_not_gated():
    # value magnitude plays no role: full samples -> no gap, whatever the
    # counter delta says (this is the property the EWMA approach lacked)
    c = _client([_series(6), _series(6)])
    assert c.scrape_gap_recent("m/x") is False
