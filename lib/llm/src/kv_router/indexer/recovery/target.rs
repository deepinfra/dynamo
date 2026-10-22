// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::SharedKvCache;
use dynamo_kv_router::protocols::{
    DpRank, KvCacheEventData, RouterEvent, WorkerId, WorkerWithDpRank,
};
use std::future::Future;
use std::sync::Arc;

use crate::discovery::PublisherId;
use crate::kv_router::Indexer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecoveryResetReason {
    Lifecycle,
    TreeDumpFailed,
    TargetFault,
}

/// Destination semantics required by worker-local KV recovery.
///
/// Ordinary events complete when the destination queue accepts them. Exact-rank
/// reset and replacement are completion barriers. Targets never provide recovery
/// state themselves; the source remains the worker's exact local indexer.
pub(crate) trait RecoveryTarget: Send + Sync + 'static {
    fn admit_event(
        &self,
        publisher_id: PublisherId,
        event: RouterEvent,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;

    fn replace_rank(
        &self,
        publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;

    fn reset_rank(
        &self,
        publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: DpRank,
        reason: RecoveryResetReason,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;

    /// Mark a source's initial recovery attempt complete when it did not install a tree dump.
    /// Targets that batch cold-start replacements use this to close the initial recovery wave.
    fn complete_initial_recovery(
        &self,
        _worker_id: WorkerId,
        _dp_rank: DpRank,
    ) -> impl Future<Output = ()> + Send {
        async {}
    }
}

#[derive(Clone)]
pub(crate) struct IndexerRecoveryTarget {
    indexer: Indexer,
    shared_cache: Option<Arc<dyn SharedKvCache>>,
}

impl IndexerRecoveryTarget {
    pub(crate) fn new(indexer: Indexer) -> Self {
        Self {
            indexer,
            shared_cache: None,
        }
    }

    /// Show `shared_cache` every device-tier store, live or replayed, before
    /// the indexer applies it.
    pub(crate) fn with_shared_cache(
        mut self,
        shared_cache: Option<Arc<dyn SharedKvCache>>,
    ) -> Self {
        self.shared_cache = shared_cache;
        self
    }
}

impl RecoveryTarget for IndexerRecoveryTarget {
    async fn admit_event(
        &self,
        _publisher_id: PublisherId,
        event: RouterEvent,
    ) -> anyhow::Result<()> {
        if let Some(shared_cache) = &self.shared_cache
            && let KvCacheEventData::Stored(data) = &event.event.data
            && event.targets_primary().unwrap_or(false)
        {
            shared_cache.observe_stored(
                WorkerWithDpRank::new(event.worker_id, event.event.dp_rank),
                data,
            );
        }
        self.indexer
            .try_apply_event(event)
            .await
            .map_err(Into::into)
    }

    async fn replace_rank(
        &self,
        publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> anyhow::Result<()> {
        self.reset_rank(
            publisher_id,
            worker_id,
            dp_rank,
            RecoveryResetReason::Lifecycle,
        )
        .await?;
        for event in events {
            self.admit_event(publisher_id, event).await?;
        }
        Ok(())
    }

    async fn reset_rank(
        &self,
        _publisher_id: PublisherId,
        worker_id: WorkerId,
        dp_rank: DpRank,
        _reason: RecoveryResetReason,
    ) -> anyhow::Result<()> {
        self.indexer
            .reset_worker_dp_rank_and_wait(worker_id, dp_rank)
            .await
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::{
        SharedCacheQuery,
        indexer::KvRouterError,
        protocols::{KvCacheStoreData, SharedCacheHits, StorageTier},
        services::indexer::backend::create_indexer,
    };

    use super::*;
    use crate::kv_router::indexer::test_util::store_event;

    #[derive(Default)]
    struct StoreObserver {
        stored: parking_lot::Mutex<Vec<(WorkerWithDpRank, usize)>>,
    }

    #[async_trait::async_trait]
    impl SharedKvCache for StoreObserver {
        async fn check_blocks(
            &self,
            _query: SharedCacheQuery<'_>,
        ) -> Result<SharedCacheHits, KvRouterError> {
            Ok(SharedCacheHits::default())
        }

        fn observe_stored(&self, worker: WorkerWithDpRank, data: &KvCacheStoreData) {
            self.stored.lock().push((worker, data.blocks.len()));
        }
    }

    #[tokio::test]
    async fn shared_cache_sees_device_stores_only() {
        let observer = Arc::new(StoreObserver::default());
        let target = IndexerRecoveryTarget::new(create_indexer(4, 1))
            .with_shared_cache(Some(observer.clone()));
        target
            .admit_event(0, store_event(7, 1, 1, &[], &[11, 12], StorageTier::Device))
            .await
            .unwrap();
        target
            .admit_event(0, store_event(7, 0, 2, &[], &[21], StorageTier::HostPinned))
            .await
            .unwrap();
        target
            .replace_rank(
                0,
                8,
                0,
                vec![store_event(8, 0, 1, &[], &[31], StorageTier::Device)],
            )
            .await
            .unwrap();
        assert_eq!(
            *observer.stored.lock(),
            vec![
                (WorkerWithDpRank::new(7, 1), 2),
                (WorkerWithDpRank::new(8, 0), 1),
            ]
        );
    }
}
