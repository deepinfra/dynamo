# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asymmetric confirmation-gate tests for LoadScalingMixin._confirm_proposal.

Covers the 2026-07-02 20:43Z breach regression: a rising throughput lower
bound interleaved with +1 load proposals must confirm upward within
``_scaling_confirmation_ticks_up`` agreeing ticks instead of being blocked by
full-buffer unanimity.
"""

from collections import deque
from types import SimpleNamespace

import pytest

from dynamo.planner.core.load_scaling import LoadScalingMixin


class _Gate(LoadScalingMixin):
    """Minimal host for the mixin: only the fields the gate touches."""

    def __init__(
        self,
        ticks: int = 6,
        ticks_up: int = 2,
        commit: int = 0,
        cooldown: int = 0,
    ):
        self._scaling_confirmation_ticks = ticks
        self._scaling_confirmation_ticks_up = ticks_up
        self._proposed_buffer_p = deque(maxlen=ticks)
        self._last_suggested_p = commit
        self._confirm_ticks = {}
        self._last_up_tick = {}
        self._config = SimpleNamespace(scale_down_cooldown_ticks=cooldown)

    def push(self, proposed: int, observed: int) -> int:
        return self._confirm_proposal(
            self._proposed_buffer_p, proposed, "_last_suggested_p", observed
        )


def test_scale_up_confirms_after_two_agreeing_ticks():
    g = _Gate(commit=5)
    assert g.push(6, observed=5) == 5  # single tick above: hold
    assert g.push(6, observed=5) == 6  # second agreeing tick: confirm
    assert g._last_suggested_p == 6


def test_scale_up_emits_min_of_agreeing_window():
    g = _Gate(commit=5)
    g.push(8, observed=5)
    # window is [8, 10]: both above commit, emit the min the ticks agree on
    assert g.push(10, observed=5) == 8
    # next agreeing pair ratchets the rest of the way
    assert g.push(10, observed=5) == 10


def test_scale_up_single_spike_does_not_confirm():
    g = _Gate(commit=5)
    assert g.push(9, observed=5) == 5
    assert g.push(5, observed=5) == 5  # dissent breaks the window
    assert g.push(9, observed=5) == 5  # [5, 9] mixed: still held
    assert g._last_suggested_p == 5


def test_breach_replay_mixed_stream_confirms_fast():
    """Replay of the 2026-07-02 stream shape that the symmetric gate held for
    ~6 minutes: +1 load proposals interleaved with a rising throughput bound.
    """
    g = _Gate(commit=5)
    stream = [6, 6, 6, 7, 6, 7, 8, 8, 10, 10]
    commits = [g.push(p, observed=5) for p in stream]
    # confirms to 6 on the second tick, and reaches >=8 within the stream
    assert commits[1] == 6
    assert max(commits) >= 8
    assert g._last_suggested_p >= 8


def test_scale_down_still_requires_full_unanimous_buffer():
    g = _Gate(ticks=6, commit=5)
    for _ in range(5):
        assert g.push(3, observed=5) == 5  # filling: hold
    assert g.push(4, observed=5) == 4  # 6th tick, all < 5: confirm
    assert g._last_suggested_p == 4


def test_scale_down_one_dissent_blocks():
    g = _Gate(ticks=6, commit=5)
    for p in (3, 3, 5, 3, 3, 3):
        assert g.push(p, observed=5) == 5
    assert g._last_suggested_p == 5


def test_scale_down_emits_max_of_buffer():
    g = _Gate(ticks=6, commit=5)
    for p in (3, 3, 3, 3, 3):
        g.push(p, observed=5)
    # last tick dips to 1; the buffer only unanimously justifies 3
    assert g.push(1, observed=5) == 3
    assert g._last_suggested_p == 3


def test_commit_latches_to_observed_on_first_tick():
    g = _Gate(commit=0)
    assert g.push(4, observed=4) == 4  # latched to observed, 4 == commit: hold
    assert g._last_suggested_p == 4


def test_disable_with_ticks_one():
    g = _Gate(ticks=1, ticks_up=1, commit=5)
    assert g.push(7, observed=5) == 7
    assert g.push(2, observed=7) == 2


def test_steady_state_holds():
    g = _Gate(commit=4)
    for _ in range(8):
        assert g.push(4, observed=4) == 4


def test_down_suppressed_during_cooldown_after_up():
    g = _Gate(ticks=3, cooldown=20, commit=3)
    # confirm up to 4 (2 agreeing ticks)
    g.push(4, observed=3)
    assert g.push(4, observed=3) == 4
    # unanimous down proposals within the cooldown window: held
    for _ in range(6):
        assert g.push(3, observed=4) == 4
    assert g._last_suggested_p == 4


def test_down_confirms_after_cooldown_expires():
    g = _Gate(ticks=3, cooldown=6, commit=3)
    g.push(4, observed=3)
    assert g.push(4, observed=3) == 4  # up at tick 2
    results = [g.push(3, observed=4) for _ in range(8)]
    # buffer full of 3s from tick 5; cooldown (6 ticks since tick 2) expires
    # at tick 8 -> the first down-confirm happens once both gates pass
    assert results[-1] == 3
    assert 4 in results  # held for at least part of the window


class _Floor(LoadScalingMixin):
    """Minimal host for reactive-floor decay bookkeeping."""

    def __init__(self, decay: int, floor: int = 0):
        self._config = SimpleNamespace(reactive_floor_decay_ticks=decay)
        self._reactive_floor_p = floor
        self._reactive_floor_bump_tick = 0
        self._load_tick_counter = 0


def test_reactive_floor_decays_one_step_per_period():
    f = _Floor(decay=10, floor=3)
    for _ in range(9):
        f._decay_reactive_floor()
    assert f._reactive_floor_p == 3  # not yet
    f._decay_reactive_floor()
    assert f._reactive_floor_p == 2  # one step after 10 quiet ticks
    for _ in range(10):
        f._decay_reactive_floor()
    assert f._reactive_floor_p == 1
    for _ in range(30):
        f._decay_reactive_floor()
    assert f._reactive_floor_p == 0  # bottoms out


def test_reactive_floor_bump_resets_decay_clock():
    f = _Floor(decay=10, floor=3)
    for _ in range(8):
        f._decay_reactive_floor()
    # a force-up refreshes the clock (as the force block does)
    f._reactive_floor_bump_tick = f._load_tick_counter
    for _ in range(9):
        f._decay_reactive_floor()
    assert f._reactive_floor_p == 3  # clock restarted, still holding


def test_reactive_floor_disabled_with_zero():
    f = _Floor(decay=0, floor=3)
    f._decay_reactive_floor()
    assert f._reactive_floor_p == 0


class _Trend(LoadScalingMixin):
    """Minimal host for the decode consolidation peak/trend pad."""

    def __init__(self, horizon: int = 360, pad_max: float = 2.0, peak_window: int = 360):
        from collections import deque as _deque

        self._config = SimpleNamespace(
            decode_consolidation_horizon_ticks=horizon,
            decode_consolidation_pad_max=pad_max,
            decode_consolidation_peak_window_ticks=peak_window,
        )
        self._decode_kv_history = _deque(maxlen=720)
        self._load_tick_counter = 0

    def feed(self, values):
        for v in values:
            self._load_tick_counter += 1
            self._record_decode_kv_observation(v)


def test_trend_pad_rising_demand():
    t = _Trend(horizon=360, peak_window=0)
    # +0.1%/tick growth: over 360 ticks -> ~36% projected
    t.feed([1_000_000 * (1 + 0.001 * i) for i in range(120)])
    pad = t._decode_consolidation_pad()
    assert 1.25 < pad < 1.55


def test_trend_pad_flat_and_falling():
    t = _Trend()
    t.feed([1_000_000] * 120)
    assert t._decode_consolidation_pad() == 1.0
    t2 = _Trend(peak_window=0)
    t2.feed([1_000_000 - 2_000 * i for i in range(120)])
    assert t2._decode_consolidation_pad() == 1.0  # falling clamps at 1.0


def test_trend_pad_short_history_and_disabled():
    t = _Trend()
    t.feed([1_000_000 * (1 + 0.01 * i) for i in range(10)])
    assert t._decode_consolidation_pad() == 1.0  # <24 samples
    t2 = _Trend(horizon=0, peak_window=0)
    t2.feed([1_000_000 * (1 + 0.01 * i) for i in range(120)])
    assert t2._decode_consolidation_pad() == 1.0  # both terms disabled


def test_trend_pad_clamped_at_max():
    t = _Trend(pad_max=2.0)
    t.feed([1_000_000 * (1 + 0.02 * i) for i in range(120)])  # steep ramp
    assert t._decode_consolidation_pad() == 2.0


def test_peak_pad_covers_wave():
    # wave: rose to 3M, drained back to 1M — the 19:00Z bounce shape.
    # instantaneous (and trend, which is now negative) would allow
    # consolidation; the peak term demands the wave still fits.
    t = _Trend(horizon=0, pad_max=5.0)
    t.feed([1_000_000] * 30)
    t.feed([1_000_000 + 100_000 * i for i in range(20)])  # up to 3M
    t.feed([3_000_000 - 100_000 * i for i in range(20)])  # back down
    t.feed([1_000_000] * 30)
    pad = t._decode_consolidation_pad()
    assert pad == pytest.approx(3.0, rel=0.05)


def test_peak_pad_expires_outside_window():
    t = _Trend(horizon=0, peak_window=50, pad_max=5.0)
    t.feed([3_000_000] * 10)   # old peak
    t.feed([1_000_000] * 100)  # peak now outside the 50-tick window
    assert t._decode_consolidation_pad() == 1.0


def test_peak_pad_one_during_pure_ramp_but_trend_covers():
    # during a monotone ramp peak==current -> peak term is 1.0;
    # the trend term must carry it
    t = _Trend(horizon=360, peak_window=360)
    t.feed([1_000_000 * (1 + 0.001 * i) for i in range(120)])
    assert t._decode_consolidation_pad() > 1.2


def test_zero_cooldown_preserves_old_down_behavior():
    g = _Gate(ticks=3, cooldown=0, commit=3)
    g.push(4, observed=3)
    assert g.push(4, observed=3) == 4
    for _ in range(2):
        g.push(3, observed=4)
    assert g.push(3, observed=4) == 3  # full buffer of 3s confirms immediately


# ---------------------------------------------------------------------------
# _anchor_no_change_proposal: "no change" must anchor to the commit, not the
# observed ready count (model-manager replace surges inflate observed).
# ---------------------------------------------------------------------------


def _anchor(final, observed, reason, commit):
    return LoadScalingMixin._anchor_no_change_proposal(final, observed, reason, commit)


def test_anchor_rewrites_no_change_to_commit_during_replace_surge():
    # MM replace: observed 8 (old+new side by side), commit 7, model says keep.
    assert _anchor(8, 8, "no_change", 7) == 7


def test_anchor_passes_directional_decisions_through():
    assert _anchor(9, 8, "scale_up", 7) == 9
    assert _anchor(6, 8, "scale_down", 7) == 6
    assert _anchor(8, 8, "scale_down_capped_by_throughput", 7) == 8


def test_anchor_noop_before_gate_latches():
    assert _anchor(8, 8, "no_change", 0) == 8


def test_anchor_noop_when_final_diverges_from_observed():
    # floor already lifted the value away from observed: not a bare "keep".
    assert _anchor(9, 8, "no_change", 7) == 9


def test_replace_surge_does_not_ratchet_commit_through_gate():
    # End-to-end through the confirmation gate: a replace surge lasting far
    # longer than the confirmation window must not move the commit when every
    # tick's decision is "no change" (anchored to commit before buffering).
    g = _Gate(ticks=6, ticks_up=2, commit=7)
    for _ in range(12):
        proposal = _anchor(8, 8, "no_change", g._last_suggested_p)
        assert g.push(proposal, observed=8) == 7
    assert g._last_suggested_p == 7


def test_unanchored_surge_would_have_ratcheted():
    # Documents the failure mode the anchor prevents: feeding observed
    # directly confirms a phantom scale-up within ticks_up ticks.
    g = _Gate(ticks=6, ticks_up=2, commit=7)
    g.push(8, observed=8)
    assert g.push(8, observed=8) == 8


# ---------------------------------------------------------------------------
# Consolidation spike guards: measured trend pad and the static ceiling
# reserve estimate the SAME next-peak/current ratio -> combine by max, not
# product.
# ---------------------------------------------------------------------------


def _reserve(threshold):
    return LoadScalingMixin._consolidation_reserve_pad(threshold)


def test_reserve_pad_from_ceiling_fraction():
    assert _reserve(0.5) == 2.0
    assert _reserve(0.8) == pytest.approx(1.25)


def test_reserve_pad_disabled_when_threshold_unset_or_invalid():
    assert _reserve(None) == 1.0
    assert _reserve(0.0) == 1.0


def test_guards_combine_by_max_not_product():
    # measured 1.8x wave, 2x unknown-spike prior: the guard is 2x, not 3.6x
    trend_pad = 1.8
    effective = max(trend_pad, _reserve(0.5))
    assert effective == 2.0
    # measured 3x wave dominates the prior
    assert max(3.0, _reserve(0.5)) == 3.0


# ---------------------------------------------------------------------------
# Stale-FPM tolerance: a wedged (ready but non-reporting) worker must not
# veto load decisions forever.
# ---------------------------------------------------------------------------


class _Tolerator(LoadScalingMixin):
    from dynamo.planner.core.state_machine import (
        PlannerScalingState as _PSS,
    )

    _reconcile_fpm_worker_count = staticmethod(
        _PSS.__dict__["_reconcile_fpm_worker_count"].__func__
    )

    def __init__(self, tolerance=3):
        self._fpm_mismatch_ticks = {}
        self._config = SimpleNamespace(fpm_mismatch_tolerance_ticks=tolerance)


def _stats(n_workers, dp=1):
    return {(f"w{i}", d): object() for i in range(n_workers) for d in range(dp)}


def test_tolerance_holds_then_proceeds_with_reporting_subset():
    t = _Tolerator(tolerance=3)
    # 10 of 11 ready workers report (one wedged): hold for 2 ticks...
    assert t._reconcile_fpm_or_tolerate(_stats(10), 11, "decode") is None
    assert t._reconcile_fpm_or_tolerate(_stats(10), 11, "decode") is None
    # ...then proceed with the 10 that report.
    assert t._reconcile_fpm_or_tolerate(_stats(10), 11, "decode") == 10


def test_tolerance_resets_when_coverage_recovers():
    t = _Tolerator(tolerance=3)
    t._reconcile_fpm_or_tolerate(_stats(10), 11, "decode")
    t._reconcile_fpm_or_tolerate(_stats(10), 11, "decode")
    # full coverage: pass-through and streak reset
    assert t._reconcile_fpm_or_tolerate(_stats(11), 11, "decode") == 11
    assert t._reconcile_fpm_or_tolerate(_stats(10), 11, "decode") is None


def test_tolerance_never_proceeds_with_zero_reporting():
    t = _Tolerator(tolerance=2)
    for _ in range(5):
        assert t._reconcile_fpm_or_tolerate({}, 11, "decode") is None


def test_tolerance_per_component_streaks_independent():
    t = _Tolerator(tolerance=2)
    assert t._reconcile_fpm_or_tolerate(_stats(2), 3, "prefill") is None
    assert t._reconcile_fpm_or_tolerate(_stats(10), 11, "decode") is None
    assert t._reconcile_fpm_or_tolerate(_stats(2), 3, "prefill") == 2
    assert t._reconcile_fpm_or_tolerate(_stats(10), 11, "decode") == 10
