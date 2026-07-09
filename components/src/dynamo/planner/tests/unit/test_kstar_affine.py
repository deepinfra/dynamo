# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed-form steady-state K* for affine ITL models (regression mode)."""

import math

import pytest

from dynamo.planner.core.load_scaling import _kstar_affine


def _itl(alpha, beta, k):
    return alpha + beta * k


def test_matches_analytic_fixed_point():
    # ITL(K) = 5ms + 10ns*K; consolidate 500k -> 625k naive
    alpha, beta = 0.005, 1e-8
    k_curr, k_naive = 500_000.0, 625_000.0
    k_star, q = _kstar_affine(
        k_curr, k_naive, _itl(alpha, beta, k_curr), _itl(alpha, beta, k_naive)
    )
    # verify it satisfies K* = k_naive * ITL(K*) / itl_curr
    assert k_star == pytest.approx(
        k_naive * _itl(alpha, beta, k_star) / _itl(alpha, beta, k_curr), rel=1e-9
    )
    assert 0 < q < 1
    assert k_star > k_naive  # feedback always inflates the naive projection


def test_no_feedback_returns_naive():
    # flat ITL (beta=0): steady state IS the naive projection
    k_star, q = _kstar_affine(500_000, 625_000, 0.01, 0.01)
    assert k_star == pytest.approx(625_000)
    assert q == 0.0


def test_divergence_detected():
    # strong feedback: q >= 1 -> no steady state
    alpha, beta = 0.001, 4e-8
    k_curr, k_naive = 500_000.0, 900_000.0
    itl_curr = _itl(alpha, beta, k_curr)  # 21ms
    q_expected = k_naive * beta / itl_curr  # ~1.71
    k_star, q = _kstar_affine(
        k_curr, k_naive, itl_curr, _itl(alpha, beta, k_naive)
    )
    assert math.isinf(k_star)
    assert q == pytest.approx(q_expected)


def test_closed_form_exceeds_truncated_iteration():
    # the 2-iter Banach truncation underestimates the limit when q > 0
    alpha, beta = 0.005, 1.2e-8
    k_curr, k_naive = 600_000.0, 750_000.0
    itl_curr = _itl(alpha, beta, k_curr)
    k1 = k_naive * _itl(alpha, beta, k_naive) / itl_curr
    k2 = k_naive * _itl(alpha, beta, k1) / itl_curr  # 2-iter estimate
    k_star, q = _kstar_affine(
        k_curr, k_naive, itl_curr, _itl(alpha, beta, k_naive)
    )
    assert k_star > k2 > k_naive


def test_degenerate_inputs_fall_back():
    assert _kstar_affine(500_000, 500_000, 0.01, 0.012) == (None, None)  # same K
    assert _kstar_affine(500_000, 625_000, 0.0, 0.012) == (None, None)  # zero itl
    assert _kstar_affine(500_000, 625_000, 0.01, 0.0) == (None, None)  # zero post
    assert _kstar_affine(500_000, 0, 0.01, 0.012) == (None, None)  # zero k_naive


def test_noise_negative_slope_clamped():
    # itl_naive slightly below itl_curr (probe noise): beta clamps to 0,
    # K* falls back to the naive projection instead of shrinking below it
    k_star, q = _kstar_affine(500_000, 625_000, 0.0100, 0.0099)
    assert k_star == pytest.approx(625_000)
    assert q == 0.0
