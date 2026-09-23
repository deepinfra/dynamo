// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block-size check on engine `BlockStored` events before they are indexed.
//!
//! The shared ZMQ normalizer silently drops every block whose size differs
//! from the configured one, so an indexer started with the wrong
//! `--block-size` stays empty while its event stream looks healthy. The
//! standalone indexer treats that as a fatal misconfiguration instead, except
//! for vLLM's partial-prefix entries (see [`StoredBlockSize::PartialPrefix`]).

use crate::protocols::WorkerWithDpRank;
use crate::zmq_wire::RawKvEvent;

/// Per-listener gate applied to every preprocessed engine event.
pub(super) struct BlockSizeGuard {
    configured_block_size: u32,
    partial_prefix_skips: u64,
}

impl BlockSizeGuard {
    pub(super) fn new(configured_block_size: u32) -> Self {
        Self {
            configured_block_size,
            partial_prefix_skips: 0,
        }
    }

    /// Whether `raw` should be indexed. Partial-prefix stores are dropped
    /// (first few logged). A block-size mismatch **exits the process**: every
    /// later store would be dropped the same way, and a crash loop is the
    /// only signal k8s surfaces for an indexer that looks alive but is empty.
    pub(super) fn admit(&mut self, raw: &RawKvEvent, worker: WorkerWithDpRank) -> bool {
        match classify_stored_block_size(raw, self.configured_block_size) {
            StoredBlockSize::Indexable => true,
            StoredBlockSize::PartialPrefix { event_block_size } => {
                self.partial_prefix_skips += 1;
                if self.partial_prefix_skips <= 3 {
                    tracing::warn!(
                        worker_id = worker.worker_id,
                        dp_rank = worker.dp_rank,
                        event_block_size,
                        configured_block_size = self.configured_block_size,
                        "Skipping sub-block BlockStored: a partial-prefix entry \
                         (prefix_match_unit < block_size), unless the indexer's \
                         --block-size is larger than the engine's -- then the \
                         index stays empty; check configured vs event size"
                    );
                }
                false
            }
            StoredBlockSize::Mismatch { event_block_size } => {
                tracing::error!(
                    worker_id = worker.worker_id,
                    dp_rank = worker.dp_rank,
                    event_block_size,
                    configured_block_size = self.configured_block_size,
                    "Block size mismatch: every stored block would be dropped and the \
                     index would stay empty. Fix the indexer's --block-size to match \
                     the engine's --block-size. Exiting."
                );
                std::process::exit(1);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StoredBlockSize {
    /// Not a store, an empty store, or a store of whole configured blocks.
    Indexable,
    /// A store whose `block_size` is smaller than the configured cache block:
    /// vLLM hashes prefixes every `prefix_match_unit` (hash_block_size)
    /// tokens, which can be finer than the cache block, and publishes the
    /// prompt tail inside a cache block as a sub-block store (32/64/96 tokens
    /// for a 128-token block). It carries nothing the index keys on.
    PartialPrefix { event_block_size: usize },
    /// A store of blocks larger than configured: the indexer's `--block-size`
    /// does not match the engine's, and every stored block would be dropped.
    Mismatch { event_block_size: usize },
}

pub(super) fn classify_stored_block_size(
    raw: &RawKvEvent,
    configured_block_size: u32,
) -> StoredBlockSize {
    let RawKvEvent::BlockStored {
        block_hashes,
        block_size,
        ..
    } = raw
    else {
        return StoredBlockSize::Indexable;
    };
    let event_block_size = *block_size;
    let configured = configured_block_size as usize;
    if block_hashes.is_empty() || event_block_size == configured {
        StoredBlockSize::Indexable
    } else if event_block_size > 0 && event_block_size < configured {
        StoredBlockSize::PartialPrefix { event_block_size }
    } else {
        StoredBlockSize::Mismatch { event_block_size }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zmq_wire::BlockHashValue;

    fn stored(block_size: usize, num_blocks: u64) -> RawKvEvent {
        RawKvEvent::BlockStored {
            block_hashes: (0..num_blocks).map(BlockHashValue::Unsigned).collect(),
            parent_block_hash: None,
            token_ids: vec![0; block_size * num_blocks as usize],
            block_size,
            medium: None,
            lora_name: None,
            cache_namespace: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality: None,
            ownership: None,
        }
    }

    #[test]
    fn whole_blocks_are_indexable() {
        assert_eq!(
            classify_stored_block_size(&stored(128, 2), 128),
            StoredBlockSize::Indexable
        );
    }

    #[test]
    fn sub_block_store_is_a_partial_prefix_entry() {
        for size in [32, 64, 96] {
            assert_eq!(
                classify_stored_block_size(&stored(size, 1), 128),
                StoredBlockSize::PartialPrefix {
                    event_block_size: size
                }
            );
        }
    }

    #[test]
    fn larger_engine_block_is_a_mismatch() {
        assert_eq!(
            classify_stored_block_size(&stored(256, 1), 128),
            StoredBlockSize::Mismatch {
                event_block_size: 256
            }
        );
    }

    #[test]
    fn empty_store_and_non_store_events_are_indexable() {
        assert_eq!(
            classify_stored_block_size(&stored(256, 0), 128),
            StoredBlockSize::Indexable
        );
        assert_eq!(
            classify_stored_block_size(&RawKvEvent::Ignored, 128),
            StoredBlockSize::Indexable
        );
    }
}
