// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kernel-side hook for an external shared KV cache (Mooncake, HiCache). The scheduler owns
//! this seam; the indexer stage never calls it.

use async_trait::async_trait;

use std::sync::Arc;

use dynamo_kv_events::event::KvCacheStoreData;
use dynamo_kv_hashing::LocalBlockHash;
use dynamo_kv_indexer::{KvRouterError, SessionPrefixIndexer};
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_worker_selection::SharedCacheHits;

/// One request's lookup in a [`SharedKvCache`].
#[derive(Clone, Copy, Debug)]
pub struct SharedCacheQuery<'a> {
    pub tokens: &'a [u32],
    /// The router's per-block hashes of `tokens`, as the indexer uses them.
    pub block_hashes: &'a [LocalBlockHash],
    pub block_size: u32,
    /// Must be honored by implementations to avoid scoring shared-cache hits
    /// across isolated cache namespaces.
    pub cache_namespace: Option<&'a str>,
}

/// Trait for querying an external shared KV cache pool.
///
/// Implementations check which blocks/pages from a request's token sequence
/// exist in the shared cache. The returned `SharedCacheHits` describes which
/// block positions are available externally (and thus cheaper to prefill).
/// A cache whose stores only some workers can load from (for example one store
/// per node) answers per worker; see [`SharedCacheHits::scoped`].
#[async_trait]
pub trait SharedKvCache: Send + Sync {
    /// Query which blocks exist in the shared cache for the given token sequence.
    async fn check_blocks(
        &self,
        query: SharedCacheQuery<'_>,
    ) -> Result<SharedCacheHits, KvRouterError>;

    /// Called with every device-tier store a worker announces, before the
    /// indexer applies it. Caches that name blocks by the engine's own hashes
    /// learn them here.
    fn observe_stored(&self, _worker: WorkerWithDpRank, _data: &KvCacheStoreData) {}

    /// Hands the router's session prefix index to the cache so residency events that name
    /// a session can update it. The default ignores the index.
    fn attach_session_prefix_index(&self, _index: Arc<SessionPrefixIndexer>) {}
}
