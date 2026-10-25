// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The frontend's KV index: the shared `dynamo_kv_router` [`Indexer`] built
//! from router configuration with runtime-backed metrics and, when configured,
//! a request-plane remote primary.

use std::{collections::HashSet, sync::Arc, time::Duration};

use anyhow::Result;
use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed, SessionPrefixIndexer,
    approx::PruneConfig,
    config::{ApproximateCachePolicyKind, KvRouterConfig},
    indexer::{
        ApproximateRetentionConfig, KvIndexer, KvIndexerInterface, KvIndexerMetrics,
        LowerTierIndexers, ThreadPoolIndexer,
    },
    protocols::WorkerId,
};

// Re-export tiered-match types so internal callers (`indexer::TieredMatchDetails`)
// keep working after these types moved to `dynamo-kv-router`.
pub(crate) use dynamo_kv_router::indexer::TieredMatchDetails;
#[allow(unused_imports)]
pub(crate) use dynamo_kv_router::indexer::WireTieredMatchDetails;
pub use dynamo_kv_router::services::indexer::backend::{Indexer, RemotePrimary, SideIndexer};
pub(crate) use dynamo_kv_router::services::indexer::recording::ApproximateRequestLease;
use dynamo_kv_router::services::indexer::session_updates::SessionUpdateSender;
use dynamo_runtime::{
    component::Component, metrics::MetricsHierarchy, traits::DistributedRuntimeProvider,
};
pub(crate) use ingress::{RuntimeIngress, RuntimeIngressArgs};
use tokio_util::sync::CancellationToken;

mod embedding_cache;
mod ingress;
mod recovery;
pub mod remote;

pub use self::embedding_cache::{
    EmbeddingCacheIndexer, preprocessed_multimodal_cache_keys, try_build_cache_indexer,
};
use self::remote::RemoteIndexer;
pub use self::remote::{ServedIndexerHandle, ServedIndexerMode, ensure_served_indexer_service};
#[cfg(feature = "ckf-diagnostics")]
pub(crate) use recovery::WorkerQueryHealthSnapshot;
pub(crate) use recovery::{
    DEFAULT_RECOVERY_ATTEMPT_TIMEOUT, KvEventSubscriptionHandle, RecoveryResetReason,
    RecoverySupervisor, RecoveryTarget, TargetFaultDisposition, start_target_subscriber,
};
pub(crate) use recovery::{
    KvStateAgentStatusCell, start_subscriber, start_worker_kv_query_endpoint,
    start_worker_kv_query_endpoint_with_status,
};
#[cfg(test)]
pub(crate) use recovery::{WorkerQueryClient, WorkerQueryTransport};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResolvedApproximatePrimaryPolicy {
    Ttl,
    Lru,
    TtlRemoteFallback,
}

fn resolve_approximate_primary_policy(
    config: &KvRouterConfig,
) -> Result<ResolvedApproximatePrimaryPolicy> {
    if config.use_kv_events
        && config.router_approximate_cache_policy == ApproximateCachePolicyKind::Lru
    {
        anyhow::bail!(
            "router_approximate_cache_policy=lru requires use_kv_events=false; the local side indexer is TTL-only"
        );
    }
    if config.use_kv_events
        || config.router_approximate_cache_policy == ApproximateCachePolicyKind::Ttl
    {
        return Ok(ResolvedApproximatePrimaryPolicy::Ttl);
    }
    if config.use_remote_indexer || config.serve_indexer {
        return Ok(ResolvedApproximatePrimaryPolicy::TtlRemoteFallback);
    }
    Ok(ResolvedApproximatePrimaryPolicy::Lru)
}

/// DEEPINFRA: periodically reconcile the kv-events radix tree against live
/// discovery and purge blocks for departed workers.
///
/// In kv-events ("exact") mode the primary tree is built without a prune TTL
/// (`ThreadPoolIndexer::new_with_metrics`), and — unlike the NATS recovery path
/// (`recovery::jetstream`'s `purge_then_snapshot`) — the ZMQ event plane has no
/// worker-departure reconciliation. A worker that restarts/scales/churns leaves
/// its entire block set in the tree forever, which leaks unbounded memory
/// (observed: frontend → ~500GB). This mirrors the NATS purge: list live
/// instances and drop tree workers no longer present. It only removes workers
/// that no longer exist in discovery, so it never evicts live cache state.
/// Indexers that support periodic discovery reconciliation. Both the
/// single-threaded `KvIndexer` (kv-events, `router_event_threads == 1`) and the
/// multi-threaded `ThreadPoolIndexer` (`router_event_threads > 1`) take a
/// kv-events code path with NO prune TTL, so both leak departed-worker block
/// sets without this reconciliation. The trait lets one loop drive either.
trait ReconcilableIndexer: Send + Sync + 'static {
    fn reconcile_workers(&self) -> impl std::future::Future<Output = Vec<WorkerId>> + Send;
    fn reconcile_total_blocks(&self) -> impl std::future::Future<Output = usize> + Send;
    fn reconcile_remove_worker(
        &self,
        worker: WorkerId,
    ) -> impl std::future::Future<Output = ()> + Send;
}

impl ReconcilableIndexer for ThreadPoolIndexer<ConcurrentRadixTreeCompressed> {
    async fn reconcile_workers(&self) -> Vec<WorkerId> {
        self.get_workers().await
    }
    async fn reconcile_total_blocks(&self) -> usize {
        self.total_blocks().await
    }
    async fn reconcile_remove_worker(&self, worker: WorkerId) {
        KvIndexerInterface::remove_worker(self, worker).await
    }
}

impl ReconcilableIndexer for KvIndexer {
    async fn reconcile_workers(&self) -> Vec<WorkerId> {
        self.get_workers().await
    }
    async fn reconcile_total_blocks(&self) -> usize {
        self.total_blocks().await
    }
    async fn reconcile_remove_worker(&self, worker: WorkerId) {
        KvIndexerInterface::remove_worker(self, worker).await
    }
}

fn spawn_zmq_worker_reconciliation<I: ReconcilableIndexer>(
    component: Component,
    indexer: Arc<I>,
) {
    const RECONCILE_INTERVAL: Duration = Duration::from_secs(60);
    let token = component.drt().primary_token();
    // Publishes the tree's tracked-block count each tick. In kv-events mode the
    // tree has no prune TTL, so this gauge is the primary signal for the
    // unbounded-growth leak this reconciliation guards against. `.ok()` so a
    // registration failure (e.g. duplicate) degrades to no gauge, not a panic.
    let block_gauge = component
        .metrics()
        .create_intgauge(
            "kv_indexer_tracked_blocks",
            "Number of (worker, block) entries currently tracked in the KV router radix tree",
            &[],
        )
        .map_err(|e| tracing::warn!("kv-indexer: failed to create tracked-blocks gauge: {e}"))
        .ok();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(RECONCILE_INTERVAL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = token.cancelled() => break,
                _ = ticker.tick() => {
                    if let Some(gauge) = &block_gauge {
                        gauge.set(indexer.reconcile_total_blocks().await as i64);
                    }
                    let live: HashSet<WorkerId> = match component.list_instances().await {
                        Ok(instances) => instances.iter().map(|i| i.instance_id).collect(),
                        Err(e) => {
                            tracing::warn!("kv-indexer worker reconciliation: list_instances failed: {e}");
                            continue;
                        }
                    };
                    for worker_id in workers_to_purge(&live, &indexer.reconcile_workers().await) {
                        tracing::info!(
                            worker_id,
                            "kv-indexer reconciliation: purging blocks for departed worker"
                        );
                        indexer.reconcile_remove_worker(worker_id).await;
                    }
                }
            }
        }
    });
}

/// Pure reconciliation step: given the live worker set from discovery and the
/// workers currently in the tree, return those that should be purged (present
/// in the tree but absent from discovery). An empty `live` set purges nothing,
/// guarding against a transient empty discovery result wiping the whole tree.
fn workers_to_purge(live: &HashSet<WorkerId>, tree_workers: &[WorkerId]) -> Vec<WorkerId> {
    if live.is_empty() {
        return Vec::new();
    }
    tree_workers
        .iter()
        .copied()
        .filter(|worker_id| !live.contains(worker_id))
        .collect()
}

#[cfg(test)]
mod reconciliation_tests {
    use super::{WorkerId, workers_to_purge};
    use std::collections::HashSet;

    fn live(ids: &[WorkerId]) -> HashSet<WorkerId> {
        ids.iter().copied().collect()
    }

    #[test]
    fn purges_departed_keeps_live() {
        let mut purge = workers_to_purge(&live(&[1, 2]), &[1, 2, 3, 4]);
        purge.sort_unstable();
        assert_eq!(purge, vec![3, 4]);
    }

    #[test]
    fn empty_discovery_purges_nothing() {
        // Transient empty discovery must not wipe the tree.
        assert!(workers_to_purge(&HashSet::new(), &[1, 2, 3]).is_empty());
    }

    #[test]
    fn all_workers_live_purges_nothing() {
        assert!(workers_to_purge(&live(&[1, 2, 3]), &[1, 2, 3]).is_empty());
    }

    #[test]
    fn empty_tree_purges_nothing() {
        assert!(workers_to_purge(&live(&[1, 2]), &[]).is_empty());
    }
}

/// Build the frontend router's index for `kv_router_config`.
pub(crate) async fn build(
    component: &Component,
    kv_router_config: &KvRouterConfig,
    block_size: u32,
    model_name: Option<&str>,
    cancellation_token: CancellationToken,
    session_prefix_index: Option<Arc<SessionPrefixIndexer>>,
) -> Result<Indexer> {
    let approximate_policy = resolve_approximate_primary_policy(kv_router_config)?;
    if approximate_policy == ResolvedApproximatePrimaryPolicy::TtlRemoteFallback {
        tracing::warn!(
            use_remote_indexer = kv_router_config.use_remote_indexer,
            serve_indexer = kv_router_config.serve_indexer,
            "Approximate LRU requires a router-local primary indexer; falling back to TTL"
        );
    }

    if kv_router_config.router_predicted_ttl_secs.is_some() && !kv_router_config.use_kv_events {
        anyhow::bail!(
            "router_predicted_ttl_secs requires use_kv_events=true; \
             do not combine a primary approximate indexer with a side approximate indexer"
        );
    }
    if kv_router_config.use_remote_indexer {
        let model_name = model_name
            .ok_or_else(|| {
                anyhow::anyhow!("model_name is required when use_remote_indexer is configured")
            })?
            .to_string();
        let indexer_component_name = component.name();
        tracing::info!(
            indexer_component = %indexer_component_name,
            model_name,
            "Using remote KV indexer"
        );
        let remote =
            RemoteIndexer::new(component, model_name, kv_router_config.use_kv_events).await?;
        let approx = predict_on_route_side_indexer(
            component,
            kv_router_config,
            block_size,
            cancellation_token.child_token(),
        );
        return Ok(Indexer::Remote {
            primary: Arc::new(remote),
            approx,
            primary_records_routing_decisions: !kv_router_config.use_kv_events,
        });
    }

    if !kv_router_config.use_kv_events {
        let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
        let prune_config = PruneConfig {
            ttl: Duration::from_secs_f64(kv_router_config.router_ttl_secs),
        };
        let retention = if approximate_policy == ResolvedApproximatePrimaryPolicy::Lru {
            tracing::info!(
                "Starting local primary approximate indexer with capacity-bounded LRU retention"
            );
            ApproximateRetentionConfig::Lru {
                fallback_ttl: prune_config,
            }
        } else {
            ApproximateRetentionConfig::Ttl(prune_config)
        };
        if kv_router_config.router_event_threads > 1 {
            let primary = Arc::new(
                ThreadPoolIndexer::new_with_metrics_and_approximate_retention(
                    ConcurrentRadixTreeCompressed::new(),
                    kv_router_config.router_event_threads as usize,
                    block_size,
                    Some(kv_indexer_metrics.clone()),
                    Some(retention),
                ),
            );
            let lower_tier = LowerTierIndexers::new_with_metrics(
                kv_router_config.router_event_threads as usize,
                block_size,
                Some(kv_indexer_metrics),
            );
            let session_updates = session_prefix_index.map(|index| {
                SessionUpdateSender::for_concurrent(index, Arc::clone(&primary), lower_tier.clone())
            });
            return Ok(Indexer::Concurrent {
                primary,
                lower_tier,
                approx: None,
                primary_records_routing_decisions: true,
                session_updates,
            });
        }

        let primary = KvIndexer::new_with_approximate_retention(
            cancellation_token.child_token(),
            block_size,
            kv_indexer_metrics.clone(),
            Some(retention),
        );
        let lower_tier =
            LowerTierIndexers::new_with_metrics(1, block_size, Some(kv_indexer_metrics));
        let session_updates = session_prefix_index.map(|index| {
            SessionUpdateSender::for_legacy(index, primary.clone(), lower_tier.clone())
        });
        return Ok(Indexer::Single {
            primary,
            lower_tier,
            approx: None,
            primary_records_routing_decisions: true,
            session_updates,
        });
    }

    let approx = predict_on_route_side_indexer(
        component,
        kv_router_config,
        block_size,
        cancellation_token.child_token(),
    );

    if kv_router_config.router_event_threads > 1 {
        let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
        let primary = Arc::new(ThreadPoolIndexer::new_with_metrics(
            ConcurrentRadixTreeCompressed::new(),
            kv_router_config.router_event_threads as usize,
            block_size,
            Some(kv_indexer_metrics.clone()),
        ));
        // DEEPINFRA: kv-events mode has no prune TTL and the ZMQ event plane
        // has no worker-departure purge, so churned/restarted workers leak
        // their block sets forever. Reconcile against discovery to drop them.
        spawn_zmq_worker_reconciliation(component.clone(), primary.clone());
        let lower_tier = LowerTierIndexers::new_with_metrics(
            kv_router_config.router_event_threads as usize,
            block_size,
            Some(kv_indexer_metrics),
        );
        let session_updates = session_prefix_index.map(|index| {
            SessionUpdateSender::for_concurrent(index, Arc::clone(&primary), lower_tier.clone())
        });
        return Ok(Indexer::Concurrent {
            primary,
            lower_tier,
            approx,
            primary_records_routing_decisions: false,
            session_updates,
        });
    }

    let kv_indexer_metrics = KvIndexerMetrics::from_component(component);
    let primary = KvIndexer::new_with_pruning(
        cancellation_token.child_token(),
        block_size,
        kv_indexer_metrics.clone(),
        None,
    );
    // DEEPINFRA: same leak as the threads>1 branch — kv-events mode has no
    // prune TTL and the ZMQ plane has no worker-departure purge, so churned
    // workers leak their block sets forever. (KvIndexer is cheap to clone —
    // it's just channel senders.)
    spawn_zmq_worker_reconciliation(component.clone(), Arc::new(primary.clone()));
    let lower_tier = LowerTierIndexers::new_with_metrics(1, block_size, Some(kv_indexer_metrics));
    let session_updates = session_prefix_index
        .map(|index| SessionUpdateSender::for_legacy(index, primary.clone(), lower_tier.clone()));
    Ok(Indexer::Single {
        primary,
        lower_tier,
        approx,
        primary_records_routing_decisions: false,
        session_updates,
    })
}

/// Predict-on-route side indexer for `router_predicted_ttl_secs`, with the
/// component's indexer metrics.
fn predict_on_route_side_indexer(
    component: &Component,
    kv_router_config: &KvRouterConfig,
    block_size: u32,
    cancellation_token: CancellationToken,
) -> Option<SideIndexer> {
    let ttl_secs = kv_router_config.router_predicted_ttl_secs?;
    tracing::info!(
        ttl_secs,
        "Starting predict-on-route side indexer (short-TTL approximate)"
    );
    Some(SideIndexer::new(
        Duration::from_secs_f64(ttl_secs),
        block_size,
        kv_router_config.router_event_threads as usize,
        KvIndexerMetrics::from_component(component),
        cancellation_token,
    ))
}

#[cfg(test)]
pub(super) mod test_util {
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash, RouterEvent, StorageTier,
        compute_seq_hash_for_block,
    };

    pub(crate) fn store_event(
        worker_id: u64,
        dp_rank: u32,
        event_id: u64,
        prefix_hashes: &[u64],
        local_hashes: &[u64],
        storage_tier: StorageTier,
    ) -> RouterEvent {
        let prefix_block_hashes: Vec<LocalBlockHash> =
            prefix_hashes.iter().copied().map(LocalBlockHash).collect();
        let parent_hash = compute_seq_hash_for_block(&prefix_block_hashes)
            .last()
            .copied()
            .map(ExternalSequenceBlockHash);

        let full_hashes: Vec<LocalBlockHash> = prefix_hashes
            .iter()
            .chain(local_hashes.iter())
            .copied()
            .map(LocalBlockHash)
            .collect();
        let full_sequence_hashes = compute_seq_hash_for_block(&full_hashes);
        let new_sequence_hashes = &full_sequence_hashes[prefix_hashes.len()..];
        let blocks = local_hashes
            .iter()
            .zip(new_sequence_hashes.iter())
            .map(|(&local_hash, &sequence_hash)| KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(sequence_hash),
                tokens_hash: LocalBlockHash(local_hash),
                mm_extra_info: None,
            })
            .collect();

        RouterEvent::with_storage_tier(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    start_position: None,
                    blocks,
                }),
                dp_rank,
            },
            storage_tier,
        )
    }
}
