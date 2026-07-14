// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::protocols::{
    BlockExtraInfo, BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
    KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData, Placement, PlacementEvent,
    StorageTier, WorkerWithDpRank, compute_block_hash_for_seq,
};

use super::types::{BlockHashValue, RawKvEvent};

/// Fatal conversion failures. These are configuration errors, not data
/// anomalies: every subsequent event would fail the same way, so callers
/// should escalate loudly instead of skipping the event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvertError {
    /// The engine publishes blocks of a different size than this indexer was
    /// configured for. Every stored block would be rejected, leaving the
    /// index permanently empty while the event stream looks alive.
    BlockSizeMismatch {
        event_block_size: usize,
        configured_block_size: u32,
    },
}

impl std::fmt::Display for ConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BlockSizeMismatch {
                event_block_size,
                configured_block_size,
            } => write!(
                f,
                "engine block size {event_block_size} != configured --block-size {configured_block_size}"
            ),
        }
    }
}

impl std::error::Error for ConvertError {}

/// Convert a raw event coming from the ZMQ channel into a placement-aware worker event.
pub fn convert_event(
    raw: RawKvEvent,
    event_id: u64,
    kv_block_size: u32,
    worker: WorkerWithDpRank,
    warning_count: &Arc<AtomicU32>,
) -> Result<Option<PlacementEvent>, ConvertError> {
    let storage_tier = match &raw {
        RawKvEvent::BlockStored { medium, .. } | RawKvEvent::BlockRemoved { medium, .. } => {
            StorageTier::from_kv_medium_or_default(medium.as_deref())
        }
        RawKvEvent::AllBlocksCleared => StorageTier::Device,
        RawKvEvent::Ignored => return Ok(None),
    };
    let dp_rank = worker.dp_rank;
    let event = match raw {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            lora_name,
            block_mm_infos,
            medium: _,
            is_eagle,
            group_idx: _,
            kv_cache_spec_kind: _,
            kv_cache_spec_sliding_window: _,
        } => {
            if !block_hashes.is_empty() && block_size != kv_block_size as usize {
                tracing::error!(
                    event_id,
                    worker_id = worker.worker_id,
                    dp_rank = worker.dp_rank,
                    event_block_size = block_size,
                    configured_block_size = kv_block_size,
                    "Block size mismatch: every stored block would be dropped and the \
                     index would stay empty. Fix the indexer's --block-size to match \
                     the engine's --block-size."
                );
                return Err(ConvertError::BlockSizeMismatch {
                    event_block_size: block_size,
                    configured_block_size: kv_block_size,
                });
            }

            // Reject self-referencing blocks: all block hashes (including parent) must be unique.
            {
                let mut seen = HashSet::with_capacity(block_hashes.len() + 1);
                if let Some(parent) = parent_block_hash {
                    seen.insert(parent.into_u64());
                }
                let has_duplicate = block_hashes.iter().any(|h| !seen.insert(h.into_u64()));
                if has_duplicate {
                    tracing::warn!(
                        event_id,
                        "Self-referencing block detected: duplicate hash in store event; dropping"
                    );
                    // Return an empty Removed instead of Cleared to avoid nuking
                    // the worker's entire index state. An empty Removed is a no-op
                    // in the radix tree (zero iterations, returns Ok(())).
                    return Ok(Some(PlacementEvent::new(
                        Placement::local_worker(worker.worker_id, worker.dp_rank, storage_tier),
                        KvCacheEvent {
                            event_id,
                            data: KvCacheEventData::Removed(KvCacheRemoveData {
                                block_hashes: vec![],
                            }),
                            dp_rank,
                        },
                    )));
                }
            }

            let num_block_tokens = vec![block_size as u64; block_hashes.len()];
            let block_hashes_u64: Vec<u64> = block_hashes
                .into_iter()
                .map(BlockHashValue::into_u64)
                .collect();
            let blocks = create_stored_blocks(
                kv_block_size,
                &token_ids,
                &num_block_tokens,
                &block_hashes_u64,
                lora_name.as_deref(),
                warning_count,
                block_mm_infos.as_deref(),
                is_eagle,
            );
            if blocks.len() < block_hashes_u64.len() {
                tracing::error!(
                    event_id,
                    worker_id = worker.worker_id,
                    dp_rank = worker.dp_rank,
                    expected = block_hashes_u64.len(),
                    published = blocks.len(),
                    token_ids_len = token_ids.len(),
                    "Stored blocks dropped during conversion; the index will miss them"
                );
            }
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_block_hash
                        .map(BlockHashValue::into_u64)
                        .map(ExternalSequenceBlockHash::from),
                    start_position: None,
                    blocks,
                }),
                dp_rank,
            }
        }
        RawKvEvent::BlockRemoved { block_hashes, .. } => {
            let hashes = block_hashes
                .into_iter()
                .map(BlockHashValue::into_u64)
                .map(ExternalSequenceBlockHash::from)
                .collect();
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes,
                }),
                dp_rank,
            }
        }
        RawKvEvent::AllBlocksCleared => KvCacheEvent {
            event_id,
            data: KvCacheEventData::Cleared,
            dp_rank,
        },
        RawKvEvent::Ignored => unreachable!("ignored events return before conversion"),
    };

    Ok(Some(PlacementEvent::new(
        Placement::local_worker(worker.worker_id, worker.dp_rank, storage_tier),
        event,
    )))
}

pub fn create_stored_block_from_parts(
    kv_block_size: u32,
    block_hash: u64,
    token_ids: &[u32],
    lora_name: Option<&str>,
    mm_extra_info: Option<BlockExtraInfo>,
    is_eagle: Option<bool>,
) -> KvCacheStoredBlockData {
    let block_mm_infos = mm_extra_info.as_ref().map(|info| vec![Some(info.clone())]);
    let tokens_hash = compute_block_hash_for_seq(
        token_ids,
        kv_block_size,
        BlockHashOptions {
            block_mm_infos: block_mm_infos.as_deref(),
            lora_name,
            is_eagle,
        },
    )[0];

    tracing::trace!(
        "Creating stored block: external_block_hash={}, tokens_hash={}, token_ids={:?}, kv_block_size={}, mm_extra_info={:?}",
        block_hash,
        tokens_hash.0,
        token_ids,
        kv_block_size,
        mm_extra_info
    );
    KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash::from(block_hash),
        tokens_hash,
        mm_extra_info,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn create_stored_blocks(
    kv_block_size: u32,
    token_ids: &[u32],
    num_block_tokens: &[u64],
    block_hashes: &[u64],
    lora_name: Option<&str>,
    warning_count: &Arc<AtomicU32>,
    block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
    is_eagle: Option<bool>,
) -> Vec<KvCacheStoredBlockData> {
    let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();

    let mut token_offset: usize = 0;
    let append = is_eagle.unwrap_or(false) as usize;

    for (block_idx, (num_tokens_it, block_hash_it)) in
        num_block_tokens.iter().zip(block_hashes.iter()).enumerate()
    {
        if *num_tokens_it != kv_block_size as u64 {
            if warning_count.fetch_add(1, Ordering::Relaxed) < 3 {
                tracing::warn!(
                    "Block not published. Block size must be {} tokens to be published. Block size is: {}",
                    kv_block_size,
                    *num_tokens_it
                );
            }
            break;
        }

        let end = token_offset + append + *num_tokens_it as usize;
        if end > token_ids.len() {
            if warning_count.fetch_add(1, Ordering::Relaxed) < 3 {
                tracing::warn!(
                    "Block not published. token_ids too short: need {}, got {}",
                    end,
                    token_ids.len()
                );
            }
            break;
        }

        let tokens = &token_ids[token_offset..end];
        let mm_extra_info = block_mm_infos
            .and_then(|infos| infos.get(block_idx))
            .and_then(|opt| opt.clone());

        blocks.push(create_stored_block_from_parts(
            kv_block_size,
            *block_hash_it,
            tokens,
            lora_name,
            mm_extra_info,
            is_eagle,
        ));
        token_offset += *num_tokens_it as usize;
    }

    blocks
}
