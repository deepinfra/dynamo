// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `--keep-evictions` support: park engine eviction events in a per-listener
//! buffer instead of applying them, then replay the aged ones into the tree
//! through the normal `Removed` apply path once the pod's memory crosses a
//! threshold. The radix tree itself is untouched — from its point of view
//! evictions simply arrive late.
//!
//! Correctness hinges on one rule: a buffered eviction is only replayed if it
//! is still the block's *latest* eviction. The map holds the latest eviction
//! time per hash; the FIFO queue entries each carry the time they were added,
//! and an entry whose time no longer matches the map is dead (the block was
//! re-stored, or re-evicted later — the newer queue entry handles it).

use std::collections::VecDeque;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use rustc_hash::FxHashMap;
use tokio_util::sync::CancellationToken;

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, RouterEvent,
    StorageTier,
};

use super::registry::WorkerRegistry;

/// How often the cleanup loop re-checks memory pressure.
const SWEEP_INTERVAL: Duration = Duration::from_secs(15);

/// Max hashes per synthesized `Removed` event, to bound message size on the
/// indexer's internal apply channels.
const REMOVE_CHUNK: usize = 4096;

/// Monotonic seconds since process start. Immune to wall-clock jumps, which
/// matters because entry age decides what gets removed from the tree.
pub(super) fn monotonic_secs() -> u64 {
    static START: LazyLock<Instant> = LazyLock::new(Instant::now);
    START.elapsed().as_secs()
}

/// Buffer of eviction events not yet applied to the tree, owned by one
/// listener (one `(worker_id, dp_rank)` event stream).
#[derive(Default)]
pub struct PendingEvictions {
    /// Latest eviction per sequence hash: when it happened and on which tier.
    pending: FxHashMap<u64, (u64, StorageTier)>,
    /// Arrival-ordered `(hash, at_secs, tier)`. Only pushed at the back and
    /// popped at the front, so timestamps are non-decreasing front to back.
    queue: VecDeque<(u64, u64, StorageTier)>,
}

impl PendingEvictions {
    /// Record an eviction batch instead of applying it.
    pub fn buffer(&mut self, hashes: &[ExternalSequenceBlockHash], tier: StorageTier) {
        self.buffer_at(hashes, tier, monotonic_secs());
    }

    /// Timestamp-explicit form of [`Self::buffer`]. `at` must be
    /// non-decreasing across calls (queue order is arrival order).
    fn buffer_at(&mut self, hashes: &[ExternalSequenceBlockHash], tier: StorageTier, at: u64) {
        for hash in hashes {
            self.pending.insert(hash.0, (at, tier));
            self.queue.push_back((hash.0, at, tier));
        }
    }

    /// Cancel pending evictions for blocks the engine just re-stored. Their
    /// stale queue entries stay behind and are skipped at drain time.
    pub fn cancel(&mut self, hashes: impl IntoIterator<Item = u64>) {
        for hash in hashes {
            self.pending.remove(&hash);
        }
    }

    /// Drop all state (used when a recovery snapshot replaces the worker's
    /// tree state wholesale).
    pub fn clear(&mut self) {
        self.pending.clear();
        self.queue.clear();
    }

    /// Number of evictions currently pending replay.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Raw queue length, including stale (cancelled/superseded) entries that
    /// still occupy memory until a drain pops them. `queue_len - len` is the
    /// dead weight from re-stored blocks.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Pop every queue entry with `at <= cutoff_secs` and return the hashes
    /// whose latest eviction that entry is, grouped by storage tier. Entries
    /// that were cancelled by a re-store, or superseded by a later eviction
    /// of the same hash, are skipped — not stopped at.
    pub fn drain_older_than(
        &mut self,
        cutoff_secs: u64,
    ) -> FxHashMap<StorageTier, Vec<ExternalSequenceBlockHash>> {
        let mut doomed: FxHashMap<StorageTier, Vec<ExternalSequenceBlockHash>> =
            FxHashMap::default();
        while let Some(&(hash, at, _)) = self.queue.front() {
            if at > cutoff_secs {
                break;
            }
            self.queue.pop_front();
            // Only replay if this entry is still the hash's latest eviction.
            // The tier stored in `pending` is authoritative (same write as
            // the timestamp), so read it from there.
            if let Some(&(pending_at, tier)) = self.pending.get(&hash)
                && pending_at == at
            {
                self.pending.remove(&hash);
                doomed
                    .entry(tier)
                    .or_default()
                    .push(ExternalSequenceBlockHash(hash));
            }
        }
        doomed
    }
}

/// Fraction of the memory limit currently in use, best-effort:
/// cgroup v2, then cgroup v1, then RSS against total host memory.
/// `None` if no source could be read (never expected on Linux).
pub fn memory_usage_fraction() -> Option<f64> {
    fn read_u64(path: &str) -> Option<u64> {
        std::fs::read_to_string(path).ok()?.trim().parse().ok()
    }
    // "max" in cgroup v2 (or the v1 sentinel ~2^63) means "no limit set";
    // fall through to the host total in that case.
    fn read_limit(path: &str) -> Option<u64> {
        let raw = std::fs::read_to_string(path).ok()?;
        let raw = raw.trim();
        if raw == "max" {
            return None;
        }
        let value: u64 = raw.parse().ok()?;
        (value < (1 << 62)).then_some(value)
    }

    let usage = read_u64("/sys/fs/cgroup/memory.current")
        .or_else(|| read_u64("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
        .or_else(vm_rss_bytes)?;
    let limit = read_limit("/sys/fs/cgroup/memory.max")
        .or_else(|| read_limit("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        .or_else(host_total_bytes)?;
    (limit > 0).then(|| usage as f64 / limit as f64)
}

/// Resident set size from `/proc/self/status` (`VmRSS`, reported in kB).
fn vm_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_kb_line(&status, "VmRSS:")
}

/// Total host memory from `/proc/meminfo` (`MemTotal`, reported in kB).
fn host_total_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_kb_line(&meminfo, "MemTotal:")
}

fn parse_kb_line(text: &str, key: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with(key))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Spawn the background sweep: every [`SWEEP_INTERVAL`], if memory usage is
/// at or above `memory_threshold` of the limit, drain evictions older than
/// `retention_secs` from every listener's buffer and replay them into that
/// listener's indexer as ordinary `Removed` events.
pub fn spawn_cleanup_loop(
    registry: Arc<WorkerRegistry>,
    retention_secs: u64,
    memory_threshold: f64,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(SWEEP_INTERVAL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        let mut tick_count: u64 = 0;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = ticker.tick() => {}
            }
            tick_count += 1;

            let usage = memory_usage_fraction();

            // Once a minute, log buffer/memory state unconditionally — this is
            // the only visibility into the buffers below the sweep threshold,
            // and the pod's log retention is far too short to rely on
            // startup-time lines.
            if tick_count % 4 == 1 {
                let (mut pending, mut queued) = (0usize, 0usize);
                let records = registry.listener_records();
                for (_, _, record) in &records {
                    let buffer = record.pending_evictions();
                    let buffer = buffer.lock();
                    pending += buffer.len();
                    queued += buffer.queue_len();
                }
                tracing::info!(
                    memory_fraction = usage.map(|u| format!("{u:.3}")),
                    pending_evictions = pending,
                    queued_entries = queued,
                    listeners = records.len(),
                    "keep-evictions status"
                );
            }

            let Some(usage) = usage else {
                tracing::warn!("keep-evictions sweep: could not read memory usage; skipping");
                continue;
            };
            if usage < memory_threshold {
                continue;
            }

            let cutoff = monotonic_secs().saturating_sub(retention_secs);
            let mut replayed = 0usize;
            let mut still_pending = 0usize;
            for (worker_id, dp_rank, record) in registry.listener_records() {
                // Drain under the lock, apply outside it: applies await on the
                // indexer channels and must not block the listener's event path.
                let pending_evictions = record.pending_evictions();
                let (doomed, remaining) = {
                    let mut buffer = pending_evictions.lock();
                    let doomed = buffer.drain_older_than(cutoff);
                    (doomed, buffer.len())
                };
                still_pending += remaining;
                let indexer = record.indexer();
                for (tier, hashes) in doomed {
                    replayed += hashes.len();
                    for chunk in hashes.chunks(REMOVE_CHUNK) {
                        let event = RouterEvent::with_storage_tier(
                            worker_id,
                            KvCacheEvent {
                                // Synthetic id: the standalone apply path does
                                // gap-tracking on ZMQ seq numbers, not on
                                // event_id, so 0 is safe here.
                                event_id: 0,
                                data: KvCacheEventData::Removed(KvCacheRemoveData {
                                    block_hashes: chunk.to_vec(),
                                }),
                                dp_rank,
                            },
                            tier,
                        );
                        indexer.apply_event_routed(event).await;
                    }
                }
            }
            tracing::info!(
                usage = format!("{:.1}%", usage * 100.0),
                threshold = format!("{:.1}%", memory_threshold * 100.0),
                replayed,
                still_pending,
                retention_secs,
                "keep-evictions sweep: replayed aged evictions into the tree"
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hashes(values: &[u64]) -> Vec<ExternalSequenceBlockHash> {
        values
            .iter()
            .copied()
            .map(ExternalSequenceBlockHash)
            .collect()
    }

    fn drained_flat(buffer: &mut PendingEvictions, cutoff: u64) -> Vec<u64> {
        let mut out: Vec<u64> = buffer
            .drain_older_than(cutoff)
            .into_values()
            .flatten()
            .map(|h| h.0)
            .collect();
        out.sort_unstable();
        out
    }

    #[test]
    fn drain_respects_cutoff() {
        let mut buffer = PendingEvictions::default();
        buffer.buffer_at(&hashes(&[1, 2]), StorageTier::Device, 100);
        buffer.buffer_at(&hashes(&[3]), StorageTier::Device, 200);

        // Entries newer than the cutoff stay.
        assert!(drained_flat(&mut buffer, 99).is_empty());
        assert_eq!(buffer.len(), 3);

        // Cutoff between the batches: only the older batch drains.
        assert_eq!(drained_flat(&mut buffer, 150), vec![1, 2]);
        assert_eq!(buffer.len(), 1);

        // Cutoff past everything: the rest drains; a second drain is a no-op.
        assert_eq!(drained_flat(&mut buffer, 500), vec![3]);
        assert!(buffer.is_empty());
        assert!(drained_flat(&mut buffer, 500).is_empty());
    }

    #[test]
    fn restore_cancels_pending_eviction() {
        let mut buffer = PendingEvictions::default();
        buffer.buffer(&hashes(&[1, 2, 3]), StorageTier::Device);
        buffer.cancel([2]);

        assert_eq!(buffer.len(), 2);
        let drained = drained_flat(&mut buffer, monotonic_secs() + 1);
        assert_eq!(drained, vec![1, 3], "re-stored block 2 must not be removed");
        // The stale queue entry for 2 must not stall the drain of 3 behind it.
        assert!(buffer.is_empty());
    }

    #[test]
    fn later_re_eviction_is_not_removed_early() {
        let mut buffer = PendingEvictions::default();
        // Evict at t=100, re-store, evict again at t=2000. A drain whose
        // cutoff covers only the first eviction must remove NOTHING: the
        // block's current eviction is too fresh, and the stale first entry
        // must not act on the map's newer state.
        buffer.buffer_at(&hashes(&[7]), StorageTier::Device, 100);
        buffer.cancel([7]);
        buffer.buffer_at(&hashes(&[7]), StorageTier::Device, 2000);

        assert!(
            drained_flat(&mut buffer, 1000).is_empty(),
            "the fresh re-eviction at t=2000 must not be removed by the stale t=100 entry"
        );
        assert_eq!(buffer.len(), 1);

        // Once the second eviction ages past the cutoff, it drains exactly once.
        assert_eq!(drained_flat(&mut buffer, 2000), vec![7]);
        assert!(buffer.is_empty());
        assert!(buffer.queue.is_empty(), "no leftover queue entries");
    }

    #[test]
    fn duplicate_eviction_without_restore_drains_once() {
        let mut buffer = PendingEvictions::default();
        // Same hash evicted twice with no store in between (shouldn't happen,
        // but must not double-remove): only the latest entry replays.
        buffer.buffer_at(&hashes(&[9]), StorageTier::Device, 100);
        buffer.buffer_at(&hashes(&[9]), StorageTier::Device, 200);

        assert_eq!(buffer.len(), 1);
        assert_eq!(drained_flat(&mut buffer, 500), vec![9]);
        assert!(buffer.queue.is_empty());
    }

    #[test]
    fn drain_groups_by_storage_tier() {
        let mut buffer = PendingEvictions::default();
        buffer.buffer(&hashes(&[1]), StorageTier::Device);
        buffer.buffer(&hashes(&[2]), StorageTier::HostPinned);

        let drained = buffer.drain_older_than(monotonic_secs() + 1);
        assert_eq!(drained.get(&StorageTier::Device).map(Vec::len), Some(1));
        assert_eq!(drained.get(&StorageTier::HostPinned).map(Vec::len), Some(1));
    }

    #[test]
    fn clear_drops_everything() {
        let mut buffer = PendingEvictions::default();
        buffer.buffer(&hashes(&[1, 2]), StorageTier::Device);
        buffer.clear();
        assert!(buffer.is_empty());
        assert!(drained_flat(&mut buffer, monotonic_secs() + 1).is_empty());
    }

    /// End-to-end shape of the sweep: a stored-then-evicted block stays
    /// queryable while parked, and disappears once the drained eviction is
    /// replayed through the normal `Removed` apply path.
    #[tokio::test]
    async fn drained_evictions_replay_into_the_tree() {
        use crate::protocols::{LocalBlockHash, WorkerWithDpRank, compute_seq_hash_for_block};
        use crate::standalone_indexer::indexer::{create_indexer, test_util::store_event};

        let worker = WorkerWithDpRank::new(7, 0);
        let indexer = create_indexer(4, 1);
        indexer
            .apply_event_routed(store_event(
                worker.worker_id,
                worker.dp_rank,
                0,
                &[],
                &[11, 12],
                StorageTier::Device,
            ))
            .await;
        let _ = indexer.dump_events().await; // flush in-flight applies

        let sequence = vec![LocalBlockHash(11), LocalBlockHash(12)];
        let scores = indexer.find_matches(sequence.clone()).await.unwrap();
        assert_eq!(scores.scores.get(&worker).copied(), Some(2));

        // The engine evicts the tail block; the buffer parks it, so the tree
        // still matches both blocks.
        let tail_seq_hash = ExternalSequenceBlockHash(compute_seq_hash_for_block(&sequence)[1]);
        let mut buffer = PendingEvictions::default();
        buffer.buffer_at(&[tail_seq_hash], StorageTier::Device, 100);
        let scores = indexer.find_matches(sequence.clone()).await.unwrap();
        assert_eq!(scores.scores.get(&worker).copied(), Some(2));

        // The sweep drains the aged entry and replays it as a Removed event —
        // exactly what spawn_cleanup_loop's body does.
        for (tier, hashes) in buffer.drain_older_than(1000) {
            let event = RouterEvent::with_storage_tier(
                worker.worker_id,
                KvCacheEvent {
                    event_id: 0,
                    data: KvCacheEventData::Removed(KvCacheRemoveData {
                        block_hashes: hashes,
                    }),
                    dp_rank: worker.dp_rank,
                },
                tier,
            );
            indexer.apply_event_routed(event).await;
        }
        let _ = indexer.dump_events().await;

        let scores = indexer.find_matches(sequence).await.unwrap();
        assert_eq!(
            scores.scores.get(&worker).copied().unwrap_or(0),
            1,
            "only the surviving head block should match after the replayed eviction"
        );
        assert!(buffer.is_empty());
    }

    #[test]
    fn memory_usage_fraction_reads_something_sane() {
        let usage = memory_usage_fraction().expect("some memory source must be readable");
        assert!(usage > 0.0 && usage < 10.0, "got {usage}");
    }
}
