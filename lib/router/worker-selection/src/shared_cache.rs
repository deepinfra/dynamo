// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block positions an external shared cache holds for a request, as the host looked them
//! up and the cache column reports them.

use std::ops::Range;
use std::sync::Arc;

use dynamo_router_protocol::worker::WorkerId;
use rustc_hash::FxHashMap;

/// Shared cache hit information, represented as sorted non-overlapping half-open ranges.
///
/// Ranges encode which block positions exist in the external shared KV cache pool.
/// Using ranges instead of `Vec<bool>` avoids iterating over potentially thousands
/// of blocks per worker. Typical shared cache patterns produce few contiguous regions,
/// making `hits_beyond` O(num_ranges) ~ O(1-5).
///
/// A cache whose stores only some workers can load from (for example one store
/// per node) answers per worker through `by_worker`; read a worker's view with
/// [`SharedCacheHits::for_worker`].
#[derive(Debug, Clone, Default)]
pub struct SharedCacheHits {
    /// Ranges of block positions that exist in the shared cache.
    /// Half-open ranges [start, end), sorted and non-overlapping.
    /// With `by_worker` set, these are the hits of every worker not listed there.
    pub ranges: Vec<Range<u32>>,
    /// Total number of hits (sum of range lengths).
    pub total_hits: u32,
    /// Per-worker hits of a worker-scoped cache. `None`: one pool every worker
    /// reaches, so every worker sees `ranges`. Entries are never scoped themselves.
    pub by_worker: Option<Arc<FxHashMap<WorkerId, SharedCacheHits>>>,
}

impl SharedCacheHits {
    /// Create from sorted, non-overlapping ranges.
    pub fn from_ranges(ranges: Vec<Range<u32>>) -> Self {
        let total_hits = ranges.iter().map(|r| r.end - r.start).sum();
        Self {
            ranges,
            total_hits,
            by_worker: None,
        }
    }

    /// Hits that differ per worker: a listed worker sees its own entry, every
    /// other worker sees `fallback`. Nested scopes in the entries are dropped.
    pub fn scoped(
        fallback: SharedCacheHits,
        by_worker: impl IntoIterator<Item = (WorkerId, SharedCacheHits)>,
    ) -> Self {
        let by_worker: FxHashMap<WorkerId, SharedCacheHits> = by_worker
            .into_iter()
            .map(|(worker_id, hits)| (worker_id, hits.unscoped()))
            .collect();
        Self {
            by_worker: Some(Arc::new(by_worker)),
            ..fallback.unscoped()
        }
    }

    /// Whether some workers see different hits than others.
    pub fn is_scoped(&self) -> bool {
        self.by_worker.is_some()
    }

    /// The hits `worker_id` can load from the shared cache.
    pub fn for_worker(&self, worker_id: WorkerId) -> &SharedCacheHits {
        self.by_worker
            .as_deref()
            .and_then(|by_worker| by_worker.get(&worker_id))
            .unwrap_or(self)
    }

    /// The view with the most hits, for summaries that are not about one worker.
    pub fn widest(&self) -> &SharedCacheHits {
        self.by_worker
            .as_deref()
            .into_iter()
            .flat_map(|by_worker| by_worker.values())
            .fold(self, |widest, hits| {
                if hits.total_hits > widest.total_hits {
                    hits
                } else {
                    widest
                }
            })
    }

    fn unscoped(self) -> Self {
        Self {
            by_worker: None,
            ..self
        }
    }

    /// Create from a boolean hit vector (convenience for tests and simple backends).
    /// Coalesces consecutive `true` entries into ranges.
    pub fn from_hits(hits: &[bool]) -> Self {
        let mut ranges = Vec::new();
        let mut i = 0;
        while i < hits.len() {
            if hits[i] {
                let start = i as u32;
                while i < hits.len() && hits[i] {
                    i += 1;
                }
                ranges.push(start..i as u32);
            } else {
                i += 1;
            }
        }
        Self::from_ranges(ranges)
    }

    /// Count hits at positions >= `from_position`.
    /// O(num_ranges), not O(num_blocks).
    pub fn hits_beyond(&self, from_position: u32) -> u32 {
        self.ranges
            .iter()
            .map(|r| {
                if r.end <= from_position {
                    0
                } else if r.start >= from_position {
                    r.end - r.start
                } else {
                    r.end - from_position
                }
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_cache_hits_from_hits() {
        // All hits contiguous
        let hits = SharedCacheHits::from_hits(&[true, true, true, true]);
        assert_eq!(hits.ranges, vec![0..4]);
        assert_eq!(hits.total_hits, 4);

        // Sparse hits
        let hits = SharedCacheHits::from_hits(&[true, false, true, true, false, true]);
        assert_eq!(hits.ranges, vec![0..1, 2..4, 5..6]);
        assert_eq!(hits.total_hits, 4);

        // No hits
        let hits = SharedCacheHits::from_hits(&[false, false, false]);
        assert!(hits.ranges.is_empty());
        assert_eq!(hits.total_hits, 0);

        // Empty
        let hits = SharedCacheHits::from_hits(&[]);
        assert!(hits.ranges.is_empty());
        assert_eq!(hits.total_hits, 0);
    }

    #[test]
    fn test_shared_cache_hits_beyond() {
        // Shared has [A, B, C, D] => range 0..4
        #[allow(clippy::single_range_in_vec_init)]
        let hits = SharedCacheHits::from_ranges(vec![0..4]);

        // Device has overlap=2 (positions 0,1 on device) => shared_beyond should count positions 2,3
        assert_eq!(hits.hits_beyond(2), 2);

        // Device has overlap=0 => all 4 shared hits count
        assert_eq!(hits.hits_beyond(0), 4);

        // Device has overlap=4 => nothing beyond
        assert_eq!(hits.hits_beyond(4), 0);

        // Device overlap exceeds range
        assert_eq!(hits.hits_beyond(10), 0);
    }

    #[test]
    fn test_shared_cache_hits_beyond_sparse() {
        // Ranges: [1..3, 5..8] => positions 1,2,5,6,7
        let hits = SharedCacheHits::from_ranges(vec![1..3, 5..8]);
        assert_eq!(hits.total_hits, 5);

        // from_position=0 => all 5 hits
        assert_eq!(hits.hits_beyond(0), 5);
        // from_position=2 => pos 2 (from first range) + 5,6,7 (from second) = 4
        assert_eq!(hits.hits_beyond(2), 4);
        // from_position=3 => only second range: 3 hits
        assert_eq!(hits.hits_beyond(3), 3);
        // from_position=6 => positions 6,7 from second range = 2
        assert_eq!(hits.hits_beyond(6), 2);
        // from_position=8 => nothing
        assert_eq!(hits.hits_beyond(8), 0);
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn test_shared_cache_hits_scoped_per_worker() {
        let pool = SharedCacheHits::from_ranges(vec![0..4]);
        assert!(!pool.is_scoped());
        // An unscoped pool answers every worker with the same hits.
        assert_eq!(pool.for_worker(1).total_hits, 4);
        assert_eq!(pool.for_worker(2).total_hits, 4);
        assert_eq!(pool.widest().total_hits, 4);

        let nested = SharedCacheHits::scoped(
            SharedCacheHits::default(),
            [(9, SharedCacheHits::from_ranges(vec![0..1]))],
        );
        let scoped = SharedCacheHits::scoped(
            SharedCacheHits::default(),
            [
                (1, SharedCacheHits::from_ranges(vec![0..6])),
                (2, SharedCacheHits::from_ranges(vec![0..2])),
                (3, nested),
            ],
        );
        assert!(scoped.is_scoped());
        assert_eq!(scoped.total_hits, 0);
        assert_eq!(scoped.for_worker(1).total_hits, 6);
        assert_eq!(scoped.for_worker(1).hits_beyond(4), 2);
        assert_eq!(scoped.for_worker(2).hits_beyond(0), 2);
        // Unlisted workers see the fallback.
        assert_eq!(scoped.for_worker(7).total_hits, 0);
        // Entries are flattened: a worker's view is never scoped again.
        assert!(!scoped.for_worker(3).is_scoped());
        assert_eq!(scoped.widest().ranges, vec![0..6]);
    }
}
