# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Erlang-C prefill sizing: solver, histogram moments, and the sizing path.

Reference values cross-checked against the 2026-07-02 gpt-oss-120b-disagg
analysis (simulation + 21-day backtest): at S=63.6ms, lambda=44 rps,
wait budget 236ms, variability (Ca2+Cs2)/2=6.4 the prescription is N=4.
"""

import math
from types import SimpleNamespace

import pytest

from dynamo.planner.core.throughput_scaling import (
    ThroughputScalingMixin,
    erlang_c,
    erlang_c_min_servers,
)
from dynamo.planner.core.types import TrafficShape
from dynamo.planner.monitoring.traffic_metrics import _histogram_moments

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

# ---------------------------------------------------------------------------
# erlang_c
# ---------------------------------------------------------------------------


def test_erlang_c_known_value():
    # C(4, 2.8) computed independently (stable recursion + closed form)
    assert erlang_c(4, 2.8) == pytest.approx(0.4287, abs=2e-3)


def test_erlang_c_bounds():
    assert erlang_c(4, 0.0) == 0.0
    assert erlang_c(4, 4.0) == 1.0  # unstable
    assert erlang_c(4, 5.0) == 1.0
    # more servers -> lower wait probability at fixed load
    assert erlang_c(8, 2.8) < erlang_c(5, 2.8) < erlang_c(4, 2.8)


def test_erlang_c_min_servers_reference_case():
    # gpt-oss-120b-disagg reference: 44 rps, S=63.6ms, t=236ms, corr=6.4
    n = erlang_c_min_servers(44.0, 0.0636, 0.236, 6.4)
    assert n == 4
    # tighter budget needs more servers
    assert erlang_c_min_servers(44.0, 0.0636, 0.030, 6.4) > 4
    # double demand: a=5.58, N=7 -> rho=0.80, wait ~= 0.17s <= budget
    assert erlang_c_min_servers(87.7, 0.0636, 0.236, 6.4) == 7


def test_erlang_c_min_servers_edge_cases():
    assert erlang_c_min_servers(0.0, 0.0636, 0.2, 6.4) == 1
    assert erlang_c_min_servers(44.0, 0.0636, 0.0, 6.4) is None
    assert erlang_c_min_servers(44.0, 0.0636, -1.0, 6.4) is None


# ---------------------------------------------------------------------------
# _histogram_moments
# ---------------------------------------------------------------------------


def test_histogram_moments_two_point():
    # all mass in (0,100] and (100,210]: points at 50 and sqrt(100*210)
    buckets = {"100": 10.0, "210": 20.0, "+Inf": 20.0}
    mean, m2, n = _histogram_moments(buckets, log_spaced=True)
    assert n == 20.0
    p2 = math.sqrt(100 * 210)
    assert mean == pytest.approx((10 * 50 + 10 * p2) / 20)
    assert m2 == pytest.approx((10 * 50**2 + 10 * p2**2) / 20)


def test_histogram_moments_calibration_scales_mean():
    buckets = {"100": 10.0, "210": 20.0, "+Inf": 20.0}
    mean, m2, n = _histogram_moments(buckets, log_spaced=True, calibrate_mean=200.0)
    assert mean == pytest.approx(200.0)
    # SCV is scale-invariant: matches the uncalibrated shape
    mean0, m20, _ = _histogram_moments(buckets, log_spaced=True)
    assert (m2 - mean**2) / mean**2 == pytest.approx(
        (m20 - mean0**2) / mean0**2
    )


def test_histogram_moments_empty_and_degenerate():
    assert _histogram_moments({}, log_spaced=True) is None
    assert _histogram_moments({"+Inf": 0.0}, log_spaced=True) is None
    assert _histogram_moments({"bogus": 1.0}, log_spaced=True) is None


def test_histogram_moments_linear_midpoints():
    # linear buckets (hit-rate style): mass 10 in (0,0.5], 10 in (0.5,1.0]
    buckets = {"0.5": 10.0, "1.0": 20.0, "+Inf": 20.0}
    mean, m2, n = _histogram_moments(buckets, log_spaced=False)
    assert mean == pytest.approx((0.25 + 0.75) / 2)


# ---------------------------------------------------------------------------
# _prefill_replicas_erlang
# ---------------------------------------------------------------------------


def _make_state(
    slope_service=None,
    shape=None,
    ttft_ms=400.0,
    overhead_ms=100.0,
    min_endpoint=1,
    measure_shape=True,
    down_pad=1.0,
):
    state = ThroughputScalingMixin()
    state._config = SimpleNamespace(
        ttft_ms=ttft_ms,
        min_endpoint=min_endpoint,
        prefill_sizing_mode="erlang_c",
        prefill_rho_ceiling=0.85,
        prefill_ttft_overhead_ms=overhead_ms,
        prefill_service_overhead_s=0.005,
        prefill_arrival_scv=6.0,
        prefill_service_scv=7.0,
        prefill_measure_traffic_shape=measure_shape,
        prefill_aic_service_kappa=0.67,
        prefill_down_demand_pad=down_pad,
    )
    state._prefill_regression = SimpleNamespace(
        measured_prefill_service_seconds=lambda eff, ovh: slope_service
    )
    state._traffic_shape_provider = (lambda: shape) if shape is not None else None
    state._last_erlang_bound_p = 0
    return state


def test_erlang_path_reference_prescription():
    # service fixed at the reference 63.6ms via the FPM-slope stub;
    # measured shape reproducing Cs2 ~= 6.8 after damping
    shape = TrafficShape(isl_scv=4.94, one_minus_hit_scv=0.47, isl_samples=1e6)
    state = _make_state(slope_service=0.0636, shape=shape)
    n = state._prefill_replicas_erlang(44.0, 5724.0, 0.484, aic_engine_rps=37.0)
    # direct computation with the same inputs
    scv_eff = (1 + 4.94) * (1 + 0.47) - 1
    damp = ((0.0636 - 0.005) / 0.0636) ** 2
    var = (6.0 + scv_eff * damp) / 2
    expected = erlang_c_min_servers(44.0, 0.0636, 0.4 - 0.1 - 0.0636, var)
    offered = 44.0 * 0.0636
    expected = max(expected, math.ceil(offered / 0.85), 1)
    assert n == expected
    assert state._diag_engine_rps_prefill == pytest.approx(1 / 0.0636)


def test_erlang_path_aic_kappa_fallback():
    state = _make_state(slope_service=None, shape=None, measure_shape=False)
    n = state._prefill_replicas_erlang(44.0, 5724.0, 0.484, aic_engine_rps=37.0)
    service = 1.0 / (37.0 * 0.67)  # ~40.3ms
    var = (6.0 + 7.0) / 2
    expected = erlang_c_min_servers(44.0, service, 0.4 - 0.1 - service, var)
    expected = max(expected, math.ceil(44.0 * service / 0.85), 1)
    assert n == expected


def test_erlang_path_infeasible_budget_uses_rho_ceiling():
    # service alone exceeds the SLA: no amount of replicas fixes latency
    state = _make_state(slope_service=0.5, shape=None, measure_shape=False)
    n = state._prefill_replicas_erlang(10.0, 40000.0, 0.0, aic_engine_rps=2.0)
    assert n == math.ceil(10.0 * 0.5 / 0.85)


def test_erlang_path_min_endpoint_floor():
    state = _make_state(
        slope_service=0.01, shape=None, min_endpoint=3, measure_shape=False
    )
    n = state._prefill_replicas_erlang(1.0, 500.0, 0.0, aic_engine_rps=100.0)
    assert n >= 3


def test_down_hysteresis_holds_boundary_dither():
    state = _make_state(
        slope_service=0.0636, shape=None, measure_shape=False, down_pad=1.25
    )
    # demand oscillating a few percent around an integer boundary
    n_high = state._prefill_replicas_erlang(46.0, 5724.0, 0.484, 37.0)
    dithered = [
        state._prefill_replicas_erlang(rps, 5724.0, 0.484, 37.0)
        for rps in (43.0, 46.5, 42.5, 45.5, 43.5)
    ]
    # small dips never lower the prescription
    assert all(n >= n_high for n in dithered)
    # a genuine demand collapse does lower it
    n_low = state._prefill_replicas_erlang(15.0, 5724.0, 0.484, 37.0)
    assert n_low < n_high


def test_down_hysteresis_disabled_with_pad_one():
    state = _make_state(
        slope_service=0.0636, shape=None, measure_shape=False, down_pad=1.0
    )
    n_high = state._prefill_replicas_erlang(46.0, 5724.0, 0.484, 37.0)
    n_dip = state._prefill_replicas_erlang(40.0, 5724.0, 0.484, 37.0)
    # pad=1.0: padded run equals the plain run, so any strictly lower
    # prescription passes through immediately
    assert n_dip <= n_high


def test_erlang_path_shape_provider_error_falls_back():
    state = _make_state(slope_service=0.0636)

    def boom():
        raise RuntimeError("prom down")

    state._traffic_shape_provider = boom
    n = state._prefill_replicas_erlang(44.0, 5724.0, 0.484, aic_engine_rps=37.0)
    # falls back to config Cs2=7.0
    var = (6.0 + 7.0) / 2
    expected = erlang_c_min_servers(44.0, 0.0636, 0.4 - 0.1 - 0.0636, var)
    expected = max(expected, math.ceil(44.0 * 0.0636 / 0.85), 1)
    assert n == expected
