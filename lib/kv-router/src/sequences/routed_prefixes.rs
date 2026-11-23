// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prompt prefixes this router replica set routed to the worker pool recently.
//!
//! When the pool's workers load each other's prefixes from a shared KV store,
//! a prefix any of them computed is cheap on all of them for as long as the
//! store keeps it. The per-worker overlap index cannot see that: it mirrors
//! each worker's own cache. This set answers the pool-wide question instead:
//! "how much of this prompt did some replica route to some worker within the
//! last `ttl`?". The scheduler uses the answer for the prefill load it books
//! and for the queue's uncached-token accounting, never for worker selection.
//!
//! It is fed by every prefill booking: local ones as they are admitted and
//! peers' as their `AddRequest` replica events arrive, which already carry the
//! request's sequence hashes. A sequence hash identifies the whole prefix up
//! to its block, so one hash per `stride` blocks is enough to find how deep a
//! new prompt's prefix reaches.

use std::collections::VecDeque;
use std::env;

use dynamo_tokens::SequenceHash;
use parking_lot::Mutex;
use rustc_hash::FxHashSet;
use tokio::time::{Duration, Instant};

/// Hashes live in time slices; a whole slice is dropped at once, so expiry
/// costs nothing per entry. An entry lives between `ttl * (SLICES - 1) / SLICES`
/// and `ttl`.
const SLICES: usize = 8;
const DEFAULT_STRIDE_BLOCKS: usize = 16;

pub struct RoutedPrefixes {
    slice_len: Duration,
    stride: usize,
    state: Mutex<State>,
}

struct State {
    /// Newest slice first.
    slices: VecDeque<FxHashSet<SequenceHash>>,
    newest_started: Instant,
}

impl RoutedPrefixes {
    /// `ttl`: how long a routed prefix counts as cached, at most what the
    /// shared store retains. `stride`: record one hash per this many blocks;
    /// matches are rounded down to a multiple of it.
    pub fn new(ttl: Duration, stride: usize, now: Instant) -> Self {
        Self {
            slice_len: (ttl / SLICES as u32).max(Duration::from_millis(1)),
            stride: stride.max(1),
            state: Mutex::new(State {
                slices: VecDeque::from([FxHashSet::default()]),
                newest_started: now,
            }),
        }
    }

    /// Enabled by `DYN_ROUTER_ROUTED_PREFIX_TTL_SECS` (> 0). The stride is
    /// `DYN_ROUTER_ROUTED_PREFIX_STRIDE_BLOCKS` (default 16).
    pub fn from_env(now: Instant) -> Option<Self> {
        let ttl_secs = parse_ttl_secs(env::var("DYN_ROUTER_ROUTED_PREFIX_TTL_SECS").ok().as_deref())?;
        let stride = env::var("DYN_ROUTER_ROUTED_PREFIX_STRIDE_BLOCKS")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|stride| *stride > 0)
            .unwrap_or(DEFAULT_STRIDE_BLOCKS);
        Some(Self::new(Duration::from_secs_f64(ttl_secs), stride, now))
    }

    fn rotate(&self, state: &mut State, now: Instant) {
        let mut elapsed = now.saturating_duration_since(state.newest_started);
        while elapsed >= self.slice_len {
            state.slices.push_front(FxHashSet::default());
            state.slices.truncate(SLICES);
            state.newest_started += self.slice_len;
            elapsed -= self.slice_len;
            if state.slices.iter().all(FxHashSet::is_empty) {
                // Idle for a whole ttl: nothing left to age, so catch up at once.
                state.newest_started = now;
                break;
            }
        }
    }

    /// Record that the prompt with these per-block sequence hashes was routed.
    /// Re-routing a prefix renews it.
    pub fn record(&self, sequence: &[SequenceHash], now: Instant) {
        if sequence.len() < self.stride {
            return;
        }
        let mut state = self.state.lock();
        self.rotate(&mut state, now);
        let newest = state.slices.front_mut().expect("at least one slice");
        for hash in sequence.iter().skip(self.stride - 1).step_by(self.stride) {
            newest.insert(*hash);
        }
    }

    /// How many leading blocks of this prompt were routed within the ttl,
    /// rounded down to the stride.
    pub fn matched_blocks(&self, sequence: &[SequenceHash], now: Instant) -> usize {
        let marks = sequence.len() / self.stride;
        if marks == 0 {
            return 0;
        }
        let mut state = self.state.lock();
        self.rotate(&mut state, now);
        let present = |mark: usize| {
            let hash = &sequence[mark * self.stride - 1];
            state.slices.iter().any(|slice| slice.contains(hash))
        };
        // A recorded prompt records every mark up to its length, and a deeper
        // mark is never renewed without the shallower ones, so presence is
        // monotone in depth: binary-search the deepest present mark.
        let (mut low, mut high) = (0, marks);
        while low < high {
            let mid = (low + high).div_ceil(2);
            if present(mid) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        low * self.stride
    }

    pub fn len(&self) -> usize {
        self.state.lock().slices.iter().map(FxHashSet::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn parse_ttl_secs(raw: Option<&str>) -> Option<f64> {
    let raw = raw?;
    match raw.trim().parse::<f64>() {
        Ok(value) if value.is_finite() && value > 0.0 => Some(value),
        Ok(value) if value == 0.0 => None,
        _ => {
            tracing::warn!(
                value = raw,
                "ignoring DYN_ROUTER_ROUTED_PREFIX_TTL_SECS: expected a positive number of seconds"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prompt(tag: u64, blocks: usize) -> Vec<SequenceHash> {
        (0..blocks as u64).map(|block| tag * 1_000_000 + block).collect()
    }

    #[test]
    fn finds_how_deep_a_routed_prefix_reaches() {
        let now = Instant::now();
        let routed = RoutedPrefixes::new(Duration::from_secs(80), 4, now);
        assert_eq!(routed.matched_blocks(&prompt(1, 40), now), 0);

        routed.record(&prompt(1, 40), now);
        // The same prompt, and a longer one that continues it, match to the
        // last recorded mark (40 blocks -> marks at 4, 8, ..., 40).
        assert_eq!(routed.matched_blocks(&prompt(1, 40), now), 40);
        assert_eq!(routed.matched_blocks(&prompt(1, 55), now), 40);
        // A shorter prefix of it matches fully, rounded down to the stride.
        assert_eq!(routed.matched_blocks(&prompt(1, 22), now), 20);
        // An unrelated prompt does not.
        assert_eq!(routed.matched_blocks(&prompt(2, 40), now), 0);

        // A prompt that shares 12 blocks and then diverges matches 12.
        let mut diverged = prompt(1, 12);
        diverged.extend(prompt(3, 30));
        assert_eq!(routed.matched_blocks(&diverged, now), 12);
    }

    #[test]
    fn prompts_shorter_than_the_stride_are_not_tracked() {
        let now = Instant::now();
        let routed = RoutedPrefixes::new(Duration::from_secs(80), 16, now);
        routed.record(&prompt(1, 15), now);
        assert!(routed.is_empty());
        assert_eq!(routed.matched_blocks(&prompt(1, 15), now), 0);
    }

    #[test]
    fn entries_expire_after_the_ttl_and_rerouting_renews_them() {
        let start = Instant::now();
        let ttl = Duration::from_secs(80);
        let routed = RoutedPrefixes::new(ttl, 4, start);
        routed.record(&prompt(1, 8), start);
        routed.record(&prompt(2, 8), start);

        // Still there shortly before the ttl (at least 7/8 of it is kept).
        let later = start + Duration::from_secs(60);
        assert_eq!(routed.matched_blocks(&prompt(1, 8), later), 8);
        // Prompt 1 is routed again; prompt 2 is not.
        routed.record(&prompt(1, 8), later);

        let after_ttl = start + ttl + Duration::from_secs(1);
        assert_eq!(routed.matched_blocks(&prompt(1, 8), after_ttl), 8);
        assert_eq!(routed.matched_blocks(&prompt(2, 8), after_ttl), 0);

        // Long idle: everything is gone and the set is empty again.
        let much_later = start + ttl * 10;
        assert_eq!(routed.matched_blocks(&prompt(1, 8), much_later), 0);
        assert!(routed.is_empty());
    }

    #[test]
    fn ttl_setting_is_parsed_strictly() {
        assert_eq!(parse_ttl_secs(None), None);
        assert_eq!(parse_ttl_secs(Some("0")), None);
        assert_eq!(parse_ttl_secs(Some("3000")), Some(3000.0));
        assert_eq!(parse_ttl_secs(Some(" 90.5 ")), Some(90.5));
        assert_eq!(parse_ttl_secs(Some("-1")), None);
        assert_eq!(parse_ttl_secs(Some("soon")), None);
    }
}
