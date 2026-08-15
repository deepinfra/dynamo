// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `--h24` counterfactual indexer: answers "how many prefix blocks of this
//! query were stored by ANY worker within the retention horizon", ignoring
//! evictions and worker identity entirely.
//!
//! Instead of a radix tree it keeps two flat maps:
//! - `attach`: engine sequence hash -> our chained prefix hash, so store
//!   events that name their parent by engine hash can keep chaining.
//! - `seen`: chained prefix hash -> (last_stored, last_touched) timestamps.
//!
//! The chain recurrence is the crate-wide [`compute_next_seq_hash`] (first
//! block's prefix hash equals its tokens hash), so `seen` keys are the same
//! `SequenceHash` values the rest of the codebase computes for a prefix.
//! Queries recompute the chain from the caller's tokens hashes and walk it to
//! the first miss; matched-block ages feed the retention-curve histograms.
//!
//! Matches are reported under a single synthetic worker ([`H24_WORKER_ID`])
//! so the HTTP response shape stays identical for existing callers.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use tokio_util::sync::CancellationToken;

use crate::protocols::{
    KvCacheStoreData, LocalBlockHash, OverlapScores, WorkerWithDpRank, compute_next_seq_hash,
};

use super::evictions::memory_usage_fraction;
use super::indexer::Indexer;
use super::metrics;
use super::registry::WorkerRegistry;

/// Synthetic worker id all h24 matches are reported under. Real worker ids
/// are xxh3 hashes of pod names, so 0 cannot collide with one.
pub const H24_WORKER_ID: u64 = 0;

/// How often the expiry sweep runs.
const SWEEP_INTERVAL: Duration = Duration::from_secs(600);

/// Memory fraction above which the sweep tightens the horizon to half.
const MEMORY_GUARD_THRESHOLD: f64 = 0.92;

/// Wall-clock seconds since the Unix epoch (u32 is fine until 2106).
fn unix_secs() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as u32)
        .unwrap_or(0)
}

#[derive(Clone, Copy)]
struct AttachEntry {
    prefix: u64,
    last_seen: u32,
}

#[derive(Clone, Copy)]
struct SeenEntry {
    last_stored: u32,
    last_touched: u32,
}

#[derive(Default)]
struct Inner {
    /// Engine sequence hash -> our chained prefix hash. Store-side plumbing
    /// only: lets a store that names its parent by engine hash keep chaining.
    attach: FxHashMap<u64, AttachEntry>,
    /// Chained prefix hash -> timestamps. One entry per unique prefix.
    seen: FxHashMap<u64, SeenEntry>,
}

pub struct H24Indexer {
    inner: Mutex<Inner>,
    parent_misses: AtomicU64,
    ignored_events: AtomicU64,
}

impl H24Indexer {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            parent_misses: AtomicU64::new(0),
            ignored_events: AtomicU64::new(0),
        }
    }

    /// Apply a store event: resolve the parent's prefix hash, chain through
    /// the event's blocks, and stamp `seen`/`attach`. A missing parent drops
    /// the whole event (every block chains off it) and is counted.
    pub fn apply_store(&self, data: &KvCacheStoreData) {
        self.apply_store_at(data, unix_secs());
    }

    fn apply_store_at(&self, data: &KvCacheStoreData, now: u32) {
        if data.blocks.is_empty() {
            return;
        }

        let mut inner = self.inner.lock();

        // None parent = sequence start: the first block's prefix hash equals
        // its tokens hash (compute_seq_hash_for_block's recurrence).
        let mut parent_prefix: Option<u64> = match data.parent_hash {
            None => None,
            Some(parent) => match inner.attach.get(&parent.0) {
                Some(entry) => Some(entry.prefix),
                None => {
                    self.parent_misses.fetch_add(1, Ordering::Relaxed);
                    metrics::h24_inc_parent_miss();
                    return;
                }
            },
        };

        for block in &data.blocks {
            let prefix = match parent_prefix {
                None => block.tokens_hash.0,
                Some(parent) => compute_next_seq_hash(parent, block.tokens_hash),
            };
            let entry = inner.seen.entry(prefix).or_insert(SeenEntry {
                last_stored: now,
                last_touched: now,
            });
            entry.last_stored = now;
            entry.last_touched = entry.last_touched.max(now);
            inner.attach.insert(
                block.block_hash.0,
                AttachEntry {
                    prefix,
                    last_seen: now,
                },
            );
            parent_prefix = Some(prefix);
        }
    }

    /// Longest-prefix match: rebuild the chain from the query's tokens hashes
    /// and walk `seen` to the first miss. Matched blocks bump `last_touched`
    /// and feed the age histograms; the score is reported under the synthetic
    /// worker.
    pub fn find_matches(&self, sequence: &[LocalBlockHash]) -> OverlapScores {
        self.find_matches_at(sequence, unix_secs())
    }

    fn find_matches_at(&self, sequence: &[LocalBlockHash], now: u32) -> OverlapScores {
        let mut scores = OverlapScores::new();
        if sequence.is_empty() {
            return scores;
        }

        let mut inner = self.inner.lock();
        let mut depth = 0u32;
        let mut prefix: Option<u64> = None;
        for hash in sequence {
            let next = match prefix {
                None => hash.0,
                Some(parent) => compute_next_seq_hash(parent, *hash),
            };
            let Some(entry) = inner.seen.get_mut(&next) else {
                break;
            };
            metrics::h24_observe_age("since_stored", now.saturating_sub(entry.last_stored));
            metrics::h24_observe_age("since_touched", now.saturating_sub(entry.last_touched));
            entry.last_touched = entry.last_touched.max(now);
            depth += 1;
            prefix = Some(next);
        }
        drop(inner);

        if depth > 0 {
            metrics::h24_add_matched_blocks(depth as usize);
            scores
                .scores
                .insert(WorkerWithDpRank::new(H24_WORKER_ID, 0), depth);
        }
        scores
    }

    pub fn note_ignored_event(&self) {
        self.ignored_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Drop entries not seen (stored or touched) since `cutoff_secs`.
    /// Returns (seen_dropped, attach_dropped).
    pub fn expire_older_than(&self, cutoff_secs: u32) -> (usize, usize) {
        let mut inner = self.inner.lock();
        let seen_before = inner.seen.len();
        let attach_before = inner.attach.len();
        inner
            .seen
            .retain(|_, e| e.last_stored.max(e.last_touched) >= cutoff_secs);
        inner.attach.retain(|_, e| e.last_seen >= cutoff_secs);
        (
            seen_before - inner.seen.len(),
            attach_before - inner.attach.len(),
        )
    }

    /// (seen entries, attach entries).
    pub fn sizes(&self) -> (usize, usize) {
        let inner = self.inner.lock();
        (inner.seen.len(), inner.attach.len())
    }

    pub fn parent_misses(&self) -> u64 {
        self.parent_misses.load(Ordering::Relaxed)
    }

    pub fn ignored_events(&self) -> u64 {
        self.ignored_events.load(Ordering::Relaxed)
    }
}

impl Default for H24Indexer {
    fn default() -> Self {
        Self::new()
    }
}

/// Background expiry sweep: every [`SWEEP_INTERVAL`], drop entries older than
/// `horizon_secs` from every h24 indexer in the registry, refresh the size
/// gauges, and tighten to half the horizon if memory is still above
/// [`MEMORY_GUARD_THRESHOLD`] afterwards.
pub fn spawn_expiry_loop(
    registry: Arc<WorkerRegistry>,
    horizon_secs: u64,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(SWEEP_INTERVAL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = ticker.tick() => {}
            }

            let now = unix_secs();
            let cutoff = now.saturating_sub(horizon_secs as u32);
            let mut dropped = (0usize, 0usize);
            let mut sizes = (0usize, 0usize);
            let indexers = registry.all_indexers_with_block_size();
            for (_, indexer, _) in &indexers {
                let Indexer::H24(h24) = indexer else {
                    continue;
                };
                let (seen_dropped, attach_dropped) = h24.expire_older_than(cutoff);
                dropped.0 += seen_dropped;
                dropped.1 += attach_dropped;
                let (seen, attach) = h24.sizes();
                sizes.0 += seen;
                sizes.1 += attach;
            }

            // Memory guard: horizon is the target, memory is the hard limit.
            let usage = memory_usage_fraction();
            if usage.is_some_and(|u| u >= MEMORY_GUARD_THRESHOLD) {
                let tight_cutoff = now.saturating_sub((horizon_secs / 2) as u32);
                let mut guard_dropped = 0usize;
                for (_, indexer, _) in &indexers {
                    if let Indexer::H24(h24) = indexer {
                        let (s, a) = h24.expire_older_than(tight_cutoff);
                        guard_dropped += s + a;
                    }
                }
                tracing::warn!(
                    usage = ?usage,
                    horizon_secs,
                    guard_dropped,
                    "h24 memory guard: usage above threshold, expired at half horizon"
                );
            }

            metrics::h24_set_sizes(sizes.0, sizes.1);
            tracing::info!(
                seen_blocks = sizes.0,
                attach_entries = sizes.1,
                seen_dropped = dropped.0,
                attach_dropped = dropped.1,
                horizon_secs,
                memory_fraction = usage.map(|u| format!("{u:.3}")),
                "h24 expiry sweep"
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{
        ExternalSequenceBlockHash, KvCacheStoredBlockData, compute_seq_hash_for_block,
    };

    /// Store `local_hashes` chained after `prefix_hashes`, with engine ext
    /// hashes derived from the crate chain recurrence offset by `ext_salt`
    /// (different salts model different hash regimes storing the same tokens).
    fn store_data(prefix_hashes: &[u64], local_hashes: &[u64], ext_salt: u64) -> KvCacheStoreData {
        let prefix: Vec<LocalBlockHash> = prefix_hashes.iter().copied().map(LocalBlockHash).collect();
        let full: Vec<LocalBlockHash> = prefix_hashes
            .iter()
            .chain(local_hashes.iter())
            .copied()
            .map(LocalBlockHash)
            .collect();
        let full_seq = compute_seq_hash_for_block(&full);
        let parent_hash = compute_seq_hash_for_block(&prefix)
            .last()
            .map(|h| ExternalSequenceBlockHash(h.wrapping_add(ext_salt)));
        let blocks = local_hashes
            .iter()
            .zip(full_seq[prefix_hashes.len()..].iter())
            .map(|(&local, &seq)| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(seq.wrapping_add(ext_salt)),
                tokens_hash: LocalBlockHash(local),
                mm_extra_info: None,
            })
            .collect();
        KvCacheStoreData {
            parent_hash,
            start_position: None,
            blocks,
        }
    }

    fn seq(hashes: &[u64]) -> Vec<LocalBlockHash> {
        hashes.iter().copied().map(LocalBlockHash).collect()
    }

    fn depth(scores: &OverlapScores) -> u32 {
        scores
            .scores
            .get(&WorkerWithDpRank::new(H24_WORKER_ID, 0))
            .copied()
            .unwrap_or(0)
    }

    #[test]
    fn store_then_query_matches_prefix() {
        let h24 = H24Indexer::new();
        h24.apply_store_at(&store_data(&[], &[1, 2, 3], 0), 100);

        assert_eq!(depth(&h24.find_matches_at(&seq(&[1, 2, 3]), 200)), 3);
        assert_eq!(depth(&h24.find_matches_at(&seq(&[1, 2]), 200)), 2);
        // Divergence after block 1 stops the walk.
        assert_eq!(depth(&h24.find_matches_at(&seq(&[1, 9, 3]), 200)), 1);
        // Different first block = no match at all.
        assert_eq!(depth(&h24.find_matches_at(&seq(&[9, 2, 3]), 200)), 0);
        assert_eq!(h24.sizes(), (3, 3));
    }

    #[test]
    fn chained_store_attaches_via_engine_hash() {
        let h24 = H24Indexer::new();
        h24.apply_store_at(&store_data(&[], &[1, 2], 0), 100);
        // Second event chains off ext hash of block 2.
        h24.apply_store_at(&store_data(&[1, 2], &[3], 0), 150);

        assert_eq!(depth(&h24.find_matches_at(&seq(&[1, 2, 3]), 200)), 3);
        assert_eq!(h24.parent_misses(), 0);
    }

    #[test]
    fn same_tokens_different_ext_regime_dedup() {
        let h24 = H24Indexer::new();
        // Two regimes (different engine ext hashes) store identical tokens.
        h24.apply_store_at(&store_data(&[], &[1, 2], 0), 100);
        h24.apply_store_at(&store_data(&[], &[1, 2], 777), 150);

        let (seen, attach) = h24.sizes();
        assert_eq!(seen, 2, "identical token chains must share seen entries");
        assert_eq!(attach, 4, "each regime keeps its own attach aliases");
        // Both regimes can chain further stores off their own ext hashes.
        h24.apply_store_at(&store_data(&[1, 2], &[3], 777), 160);
        assert_eq!(depth(&h24.find_matches_at(&seq(&[1, 2, 3]), 200)), 3);
        assert_eq!(h24.parent_misses(), 0);
    }

    #[test]
    fn missing_parent_drops_event_and_counts() {
        let h24 = H24Indexer::new();
        h24.apply_store_at(&store_data(&[1, 2], &[3], 0), 100);

        assert_eq!(h24.parent_misses(), 1);
        assert_eq!(h24.sizes(), (0, 0));
        assert_eq!(depth(&h24.find_matches_at(&seq(&[1, 2, 3]), 200)), 0);
    }

    #[test]
    fn restore_refreshes_timestamps_and_expiry_respects_touch() {
        let h24 = H24Indexer::new();
        h24.apply_store_at(&store_data(&[], &[1, 2], 0), 100);

        // A query at t=5000 touches block 1 only (divergent second block).
        let _ = h24.find_matches_at(&seq(&[1, 9]), 5000);

        // Expire everything not seen since t=1000: block 2's entry dies
        // (stored 100, never touched), block 1 survives via its touch.
        let (seen_dropped, attach_dropped) = h24.expire_older_than(1000);
        assert_eq!(seen_dropped, 1);
        assert_eq!(attach_dropped, 2, "attach expires on store recency only");
        assert_eq!(depth(&h24.find_matches_at(&seq(&[1, 2]), 6000)), 1);

        // Re-store refreshes: nothing left to expire at the same cutoff.
        h24.apply_store_at(&store_data(&[], &[1, 2], 0), 7000);
        assert_eq!(h24.expire_older_than(1000), (0, 0));
        assert_eq!(depth(&h24.find_matches_at(&seq(&[1, 2]), 8000)), 2);
    }

    #[test]
    fn empty_query_and_empty_store_are_noops() {
        let h24 = H24Indexer::new();
        h24.apply_store_at(
            &KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![],
            },
            100,
        );
        assert_eq!(h24.sizes(), (0, 0));
        assert!(h24.find_matches_at(&[], 100).scores.is_empty());
    }
}
