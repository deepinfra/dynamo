// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared KV cache for vLLM's `MooncakeStoreConnector` (`shared_cache_type: mooncake-store`).
//!
//! The connector saves prompt chunks into a Mooncake store and loads the longest
//! stored prefix of later prompts, so a worker reuses what any worker sharing its
//! store computed. Which workers share a store is a deployment choice; one store
//! per node keeps loads on the node's own links. Each worker announces its store
//! in its runtime config ([`VLLM_MOONCAKE_STORE_RUNTIME_KEY`]). Workers that
//! announce the same store share its hits, and no other worker gets them.
//!
//! Two maps answer a lookup:
//! - **Chain.** The connector names a chunk by vLLM's chained block hash, which
//!   the router cannot compute from tokens. Device-tier `Stored` events carry it
//!   next to the router's block hash, so the cache learns
//!   `(parent block hash, router block hash) -> block hash` from them
//!   ([`SharedKvCache::observe_stored`]). Links outlive the GPU copy; the least
//!   recently used are dropped beyond twice the chunks of all stores.
//! - **Presence.** One subscriber per store follows its master's KV events
//!   (`mooncake_master --enable_kv_events`): `stored`/`removed` per object and
//!   replica medium, with the chunk's group id
//!   `vllm-mooncake-store:[{cache_prefix}@]{model}@{chunk hash hex}`. A chunk
//!   counts while any medium holds it. Events are the only source: after the
//!   subscriber (re)connects, the store's earlier chunks are unknown until
//!   stored again. A missed batch is only counted: forgetting a whole store
//!   over it would cost more hits than the few chunks it leaves wrong.
//!
//! A worker's hits are the leading chunks of the prompt its store holds, as the
//! connector loads one prefix. A chunk must be one router block (the connector's
//! default chunk size); other chunk sizes find no hits.

use std::collections::{HashMap, hash_map::Entry};
use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicUsize, Ordering},
};

use arc_swap::ArcSwap;
use async_trait::async_trait;
use dynamo_kv_router::{
    SharedCacheQuery, SharedKvCache,
    indexer::KvRouterError,
    protocols::{KvCacheStoreData, LocalBlockHash, SharedCacheHits, WorkerId, WorkerWithDpRank},
};
use dynamo_runtime::{component::Component, metrics::MetricsHierarchy};
use futures::StreamExt;
use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use prometheus::{IntCounter, IntCounterVec, IntGauge};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use super::{MOONCAKE_EVENT_RECONNECT_DELAY, parse_mooncake_event_batch};
use crate::{
    discovery::RuntimeConfigWatch,
    kv_router::metrics::RoutingOverheadMetrics,
    local_model::runtime_config::ModelRuntimeConfig,
    utils::zmq::{connect_sub_socket, multipart_message},
};

/// Runtime-config key under which a vLLM worker announces its Mooncake store.
pub const VLLM_MOONCAKE_STORE_RUNTIME_KEY: &str = "vllm_mooncake_store";

/// Bounds of the chain: at least this many links, at most this many.
const MIN_CHAIN_LINKS: usize = 1 << 18;
const MAX_CHAIN_LINKS: usize = 1 << 24;
/// A store with more chunks than this is forgotten and relearned (memory guard).
const MAX_STORE_CHUNKS: usize = 1 << 25;

/// The store a worker's `MooncakeStoreConnector` writes to and loads from.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StoreAnnouncement {
    /// ZMQ endpoint of the store master's KV events, as the router reaches it.
    pub kv_events_endpoint: String,
    /// Group id of the worker's chunks up to the chunk hash:
    /// `vllm-mooncake-store:[{cache_prefix}@]{model}@`.
    pub group_prefix: String,
}

/// One object event of a Mooncake master; other fields are ignored.
#[derive(Debug, Deserialize, Serialize)]
struct StoreEvent {
    event_type: String,
    #[serde(default)]
    group_id: Option<String>,
    #[serde(default)]
    medium: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Link {
    parent: Option<u64>,
    block: LocalBlockHash,
}

struct Store {
    announcement: StoreAnnouncement,
    state: RwLock<StoreState>,
}

#[derive(Default)]
struct StoreState {
    /// Chunk (low 64 bits of its hash, as vLLM's int block hashes) -> one bit per
    /// medium in `media` holding it.
    chunks: HashMap<u64, u8>,
    media: Vec<String>,
    sequence: Option<u64>,
}

impl StoreState {
    fn medium_bit(&mut self, medium: &str) -> u8 {
        let index = match self.media.iter().position(|known| known == medium) {
            Some(index) => index,
            None if self.media.len() < u8::BITS as usize => {
                self.media.push(medium.to_string());
                self.media.len() - 1
            }
            // More media than bits: the rest share the last one.
            None => u8::BITS as usize - 1,
        };
        1 << index
    }
}

struct Inner {
    chain: Mutex<LruCache<Link, u64>>,
    /// Chunks of all stores; sizes the chain bound.
    store_chunks: AtomicUsize,
    /// Each announcing worker's store, replaced whole when announcements change.
    workers: ArcSwap<HashMap<WorkerId, Arc<Store>>>,
    metrics: Option<Arc<MooncakeStoreMetrics>>,
    cancellation_token: CancellationToken,
}

/// Mooncake stores of one worker set; see the module docs.
#[derive(Clone)]
pub struct MooncakeStoreSharedKvCache {
    inner: Arc<Inner>,
}

impl MooncakeStoreSharedKvCache {
    pub fn new(
        metrics: Option<Arc<MooncakeStoreMetrics>>,
        cancellation_token: CancellationToken,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                chain: Mutex::new(LruCache::unbounded()),
                store_chunks: AtomicUsize::new(0),
                workers: ArcSwap::default(),
                metrics,
                cancellation_token,
            }),
        }
    }

    /// Follow the stores the workers in `runtime_configs` announce.
    pub fn start(&self, runtime_configs: RuntimeConfigWatch) {
        tokio::spawn(self.clone().supervise(runtime_configs));
    }

    pub fn shutdown(&self) {
        self.inner.cancellation_token.cancel();
    }

    async fn supervise(self, mut runtime_configs: RuntimeConfigWatch) {
        let cancellation_token = self.inner.cancellation_token.clone();
        let mut stores: HashMap<StoreAnnouncement, (Arc<Store>, CancellationToken)> =
            HashMap::new();
        loop {
            let announced = announcements(&runtime_configs.borrow_and_update());
            let mut workers = HashMap::with_capacity(announced.len());
            for (worker_id, announcement) in announced {
                let (store, _) = stores.entry(announcement.clone()).or_insert_with(|| {
                    let store = Arc::new(Store {
                        announcement,
                        state: RwLock::default(),
                    });
                    let token = cancellation_token.child_token();
                    tokio::spawn(self.clone().follow(store.clone(), token.clone()));
                    (store, token)
                });
                workers.insert(worker_id, store.clone());
            }
            stores.retain(|_, (store, token)| {
                let announced = workers.values().any(|other| Arc::ptr_eq(other, store));
                if !announced {
                    token.cancel();
                    self.forget(store);
                }
                announced
            });
            self.inner.workers.store(Arc::new(workers));
            if let Some(metrics) = &self.inner.metrics {
                metrics.stores.set(stores.len() as i64);
            }

            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                result = runtime_configs.changed() => if result.is_err() { break },
            }
        }
        for (store, token) in stores.into_values() {
            token.cancel();
            self.forget(&store);
        }
        self.inner.workers.store(Arc::default());
    }

    /// Track one store from its master's KV events until `cancellation_token` fires.
    async fn follow(self, store: Arc<Store>, cancellation_token: CancellationToken) {
        let endpoint = store.announcement.kv_events_endpoint.as_str();
        loop {
            // Whatever the store held before this connection is unknown again.
            self.reset(&store, "connect");
            match connect_sub_socket(endpoint, None).await {
                Ok(mut socket) => {
                    tracing::info!(%endpoint, "Following Mooncake store KV events");
                    loop {
                        tokio::select! {
                            _ = cancellation_token.cancelled() => break,
                            message = socket.next() => match message {
                                Some(Ok(frames)) => {
                                    match parse_mooncake_event_batch::<StoreEvent>(
                                        &multipart_message(frames),
                                    ) {
                                        Ok((sequence, events)) => self.apply(&store, sequence, events),
                                        Err(error) => {
                                            record_subscriber_error();
                                            tracing::warn!(%endpoint, %error, "Dropping invalid Mooncake KV event frame");
                                        }
                                    }
                                }
                                Some(Err(error)) => {
                                    record_subscriber_error();
                                    tracing::warn!(%endpoint, %error, "Mooncake store KV event stream failed; reconnecting");
                                    break;
                                }
                                None => {
                                    record_subscriber_error();
                                    tracing::warn!(%endpoint, "Mooncake store KV event stream ended; reconnecting");
                                    break;
                                }
                            },
                        }
                    }
                }
                Err(error) => {
                    record_subscriber_error();
                    tracing::warn!(%endpoint, %error, "Failed to connect to Mooncake store KV events; retrying");
                }
            }
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                _ = tokio::time::sleep(MOONCAKE_EVENT_RECONNECT_DELAY) => {}
            }
        }
        self.forget(&store);
    }

    fn apply(&self, store: &Store, sequence: u64, events: Vec<StoreEvent>) {
        let mut state = store.state.write();
        if let Some(previous) = state.sequence {
            if sequence == previous {
                return;
            }
            // The master numbers batches consecutively: anything else lost some.
            if sequence != previous.wrapping_add(1) {
                let missed = sequence.wrapping_sub(previous).wrapping_sub(1);
                tracing::debug!(
                    endpoint = %store.announcement.kv_events_endpoint,
                    previous,
                    sequence,
                    "Mooncake store KV event sequence gap"
                );
                if let Some(metrics) = &self.inner.metrics {
                    metrics.missed_batches.inc_by(missed);
                }
            }
        }
        state.sequence = Some(sequence);

        let before = state.chunks.len();
        let (mut applied, mut ignored) = (0, 0);
        for event in events {
            let Some(chunk) = event
                .group_id
                .as_deref()
                .and_then(|group_id| chunk_hash(group_id, &store.announcement.group_prefix))
            else {
                ignored += 1;
                continue;
            };
            let bit = state.medium_bit(event.medium.as_deref().unwrap_or_default());
            match event.event_type.as_str() {
                "stored" => *state.chunks.entry(chunk).or_default() |= bit,
                "removed" => {
                    if let Entry::Occupied(mut held) = state.chunks.entry(chunk) {
                        *held.get_mut() &= !bit;
                        if *held.get() == 0 {
                            held.remove();
                        }
                    }
                }
                _ => {
                    ignored += 1;
                    continue;
                }
            }
            applied += 1;
        }
        let after = state.chunks.len();
        self.add_chunks(after as isize - before as isize);
        if after > MAX_STORE_CHUNKS {
            tracing::warn!(
                endpoint = %store.announcement.kv_events_endpoint,
                chunks = after,
                "Mooncake store index exceeded its size limit; forgetting the store's chunks"
            );
            self.clear(&mut state, "size");
        }
        if let Some(metrics) = &self.inner.metrics {
            metrics
                .events
                .with_label_values(&["applied"])
                .inc_by(applied);
            metrics
                .events
                .with_label_values(&["ignored"])
                .inc_by(ignored);
        }
    }

    fn reset(&self, store: &Store, reason: &'static str) {
        let mut state = store.state.write();
        self.clear(&mut state, reason);
        state.sequence = None;
    }

    fn forget(&self, store: &Store) {
        let mut state = store.state.write();
        self.add_chunks(-(state.chunks.len() as isize));
        *state = StoreState::default();
    }

    fn clear(&self, state: &mut StoreState, reason: &'static str) {
        self.add_chunks(-(state.chunks.len() as isize));
        state.chunks = HashMap::new();
        if let Some(metrics) = &self.inner.metrics {
            metrics.resets.with_label_values(&[reason]).inc();
        }
    }

    fn add_chunks(&self, delta: isize) {
        let total = if delta >= 0 {
            self.inner
                .store_chunks
                .fetch_add(delta as usize, Ordering::Relaxed)
                + delta as usize
        } else {
            self.inner
                .store_chunks
                .fetch_sub(delta.unsigned_abs(), Ordering::Relaxed)
                - delta.unsigned_abs()
        };
        if let Some(metrics) = &self.inner.metrics {
            metrics.chunks.set(total as i64);
        }
    }

    fn chain_bound(&self) -> usize {
        self.inner
            .store_chunks
            .load(Ordering::Relaxed)
            .saturating_mul(2)
            .clamp(MIN_CHAIN_LINKS, MAX_CHAIN_LINKS)
    }
}

#[async_trait]
impl SharedKvCache for MooncakeStoreSharedKvCache {
    async fn check_blocks(
        &self,
        query: SharedCacheQuery<'_>,
    ) -> Result<SharedCacheHits, KvRouterError> {
        let workers = self.inner.workers.load();
        if workers.is_empty() || query.block_hashes.is_empty() {
            return Ok(SharedCacheHits::default());
        }
        let mut stores: Vec<&Arc<Store>> = Vec::new();
        for store in workers.values() {
            if !stores.iter().any(|known| Arc::ptr_eq(known, store)) {
                stores.push(store);
            }
        }

        // Each store's run of leading chunks; the walk ends once none extends.
        let mut runs = vec![0_u32; stores.len()];
        let stop = {
            let states: Vec<_> = stores.iter().map(|store| store.state.read()).collect();
            let mut chain = self.inner.chain.lock();
            let mut parent = None;
            let mut stop = "end";
            for (position, &block) in (0_u32..).zip(query.block_hashes) {
                let Some(&hash) = chain.get(&Link { parent, block }) else {
                    stop = "unlinked";
                    break;
                };
                let mut extended = false;
                for (run, state) in runs.iter_mut().zip(&states) {
                    if *run == position && state.chunks.contains_key(&hash) {
                        *run += 1;
                        extended = true;
                    }
                }
                if !extended {
                    stop = "absent";
                    break;
                }
                parent = Some(hash);
            }
            stop
        };
        if let Some(metrics) = &self.inner.metrics {
            metrics.lookup_stops.with_label_values(&[stop]).inc();
        }

        Ok(SharedCacheHits::scoped(
            SharedCacheHits::default(),
            workers.iter().map(|(&worker_id, store)| {
                let index = stores
                    .iter()
                    .position(|known| Arc::ptr_eq(known, store))
                    .expect("every worker's store is listed");
                (worker_id, leading_blocks(runs[index]))
            }),
        ))
    }

    fn observe_stored(&self, _worker: WorkerWithDpRank, data: &KvCacheStoreData) {
        if self.inner.workers.load().is_empty() {
            return;
        }
        let bound = self.chain_bound();
        let mut chain = self.inner.chain.lock();
        let mut parent = data.parent_hash.map(|hash| hash.0);
        for block in &data.blocks {
            chain.put(
                Link {
                    parent,
                    block: block.tokens_hash,
                },
                block.block_hash.0,
            );
            parent = Some(block.block_hash.0);
        }
        while chain.len() > bound {
            chain.pop_lru();
        }
        if let Some(metrics) = &self.inner.metrics {
            metrics.chain_links.set(chain.len() as i64);
        }
    }
}

/// The store each worker announces. A worker without one, or with an
/// unreadable one, gets no hits.
fn announcements(
    runtime_configs: &HashMap<WorkerId, ModelRuntimeConfig>,
) -> HashMap<WorkerId, StoreAnnouncement> {
    runtime_configs
        .iter()
        .filter_map(|(&worker_id, runtime_config)| {
            match runtime_config
                .get_engine_specific::<StoreAnnouncement>(VLLM_MOONCAKE_STORE_RUNTIME_KEY)
            {
                Ok(Some(announcement)) if !announcement.kv_events_endpoint.is_empty() => {
                    Some((worker_id, announcement))
                }
                Ok(_) => None,
                Err(error) => {
                    tracing::warn!(
                        worker_id,
                        runtime_key = VLLM_MOONCAKE_STORE_RUNTIME_KEY,
                        %error,
                        "Ignoring an unreadable Mooncake store announcement"
                    );
                    None
                }
            }
        })
        .collect()
}

/// The chunk a group id names under `group_prefix`: the low 64 bits of its hash,
/// matching vLLM's int block hashes in KV events.
fn chunk_hash(group_id: &str, group_prefix: &str) -> Option<u64> {
    let hex = group_id.strip_prefix(group_prefix)?;
    if hex.len() < 16 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(&hex[hex.len() - 16..], 16).ok()
}

#[allow(clippy::single_range_in_vec_init)]
fn leading_blocks(blocks: u32) -> SharedCacheHits {
    if blocks == 0 {
        SharedCacheHits::default()
    } else {
        SharedCacheHits::from_ranges(vec![0..blocks])
    }
}

fn record_subscriber_error() {
    if let Some(metrics) = RoutingOverheadMetrics::get() {
        metrics.inc_shared_cache_errors();
    }
}

/// Process-wide metrics of the Mooncake store shared cache.
pub struct MooncakeStoreMetrics {
    stores: IntGauge,
    chunks: IntGauge,
    chain_links: IntGauge,
    events: IntCounterVec,
    missed_batches: IntCounter,
    resets: IntCounterVec,
    lookup_stops: IntCounterVec,
}

static MOONCAKE_STORE_METRICS: OnceLock<Option<Arc<MooncakeStoreMetrics>>> = OnceLock::new();

impl MooncakeStoreMetrics {
    /// Registered once, on the first component that asks.
    pub fn from_component(component: &Component) -> Option<Arc<Self>> {
        MOONCAKE_STORE_METRICS
            .get_or_init(|| match Self::create(component) {
                Ok(metrics) => Some(Arc::new(metrics)),
                Err(error) => {
                    tracing::warn!(%error, "Failed to register Mooncake store metrics");
                    None
                }
            })
            .clone()
    }

    fn create(component: &Component) -> anyhow::Result<Self> {
        let metrics = component.metrics();
        Ok(Self {
            stores: metrics.create_intgauge(
                "router_mooncake_store_stores",
                "Mooncake stores the router follows",
                &[],
            )?,
            chunks: metrics.create_intgauge(
                "router_mooncake_store_chunks",
                "Chunks the followed Mooncake stores hold, as their KV events announced them",
                &[],
            )?,
            chain_links: metrics.create_intgauge(
                "router_mooncake_store_chain_links",
                "Learned links from router block hashes to vLLM block hashes",
                &[],
            )?,
            events: metrics.create_intcountervec(
                "router_mooncake_store_events_total",
                "Mooncake store KV events, by whether they changed a followed chunk",
                &["result"],
                &[],
            )?,
            missed_batches: metrics.create_intcounter(
                "router_mooncake_store_missed_batches_total",
                "Mooncake store KV event batches lost between master and router (sequence gaps)",
                &[],
            )?,
            resets: metrics.create_intcountervec(
                "router_mooncake_store_resets_total",
                "Times a store's chunks were forgotten (connect or size)",
                &["reason"],
                &[],
            )?,
            lookup_stops: metrics.create_intcountervec(
                "router_mooncake_store_lookup_stops_total",
                "Where lookups stopped: end of prompt, a block no store holds (absent) or a block with no learned vLLM hash (unlinked)",
                &["reason"],
                &[],
            )?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_kv_router::protocols::{ExternalSequenceBlockHash, KvCacheStoredBlockData};
    use tokio::sync::watch;

    use super::*;

    const PREFIX: &str = "vllm-mooncake-store:DeepSeek-V3.2@";

    fn group_id(hash: u64) -> String {
        // vLLM hashes are 32 bytes; the router keeps the low 64 bits.
        format!("{PREFIX}{:048x}{hash:016x}", 0xabcdef_u64)
    }

    fn announcement(endpoint: &str) -> StoreAnnouncement {
        StoreAnnouncement {
            kv_events_endpoint: endpoint.to_string(),
            group_prefix: PREFIX.to_string(),
        }
    }

    fn event(event_type: &str, hash: u64, medium: &str) -> StoreEvent {
        StoreEvent {
            event_type: event_type.to_string(),
            group_id: Some(group_id(hash)),
            medium: Some(medium.to_string()),
        }
    }

    /// Device-tier store of blocks with router hashes `1..=n` and vLLM hashes `101..`.
    fn stored(blocks: u64) -> KvCacheStoreData {
        KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: (1..=blocks)
                .map(|block| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(100 + block),
                    tokens_hash: LocalBlockHash(block),
                    mm_extra_info: None,
                })
                .collect(),
        }
    }

    fn router_hashes(blocks: u64) -> Vec<LocalBlockHash> {
        (1..=blocks).map(LocalBlockHash).collect()
    }

    /// A cache whose workers 1 and 2 share store `a` and worker 3 uses store `b`,
    /// without subscribers.
    fn cache_with_stores() -> (MooncakeStoreSharedKvCache, Arc<Store>, Arc<Store>) {
        let cache = MooncakeStoreSharedKvCache::new(None, CancellationToken::new());
        let store = |endpoint: &str| {
            Arc::new(Store {
                announcement: announcement(endpoint),
                state: RwLock::default(),
            })
        };
        let (a, b) = (store("tcp://a:1"), store("tcp://b:1"));
        cache.inner.workers.store(Arc::new(HashMap::from([
            (1, a.clone()),
            (2, a.clone()),
            (3, b.clone()),
        ])));
        (cache, a, b)
    }

    async fn hits(cache: &MooncakeStoreSharedKvCache, blocks: u64) -> HashMap<WorkerId, u32> {
        let block_hashes = router_hashes(blocks);
        let hits = cache
            .check_blocks(SharedCacheQuery {
                tokens: &[],
                block_hashes: &block_hashes,
                block_size: 64,
                cache_namespace: None,
            })
            .await
            .unwrap();
        [1, 2, 3, 4]
            .into_iter()
            .map(|worker| (worker, hits.for_worker(worker).total_hits))
            .collect()
    }

    #[test]
    fn chunk_hash_reads_the_low_64_bits_of_the_workers_chunks() {
        assert_eq!(chunk_hash(&group_id(0x1234), PREFIX), Some(0x1234));
        assert_eq!(
            chunk_hash(&format!("{PREFIX}{:016x}", u64::MAX), PREFIX),
            Some(u64::MAX)
        );
        // Another model, cache prefix or the SGLang layout on the same master.
        assert_eq!(
            chunk_hash(&group_id(7).replace("V3.2", "V3.1"), PREFIX),
            None
        );
        assert_eq!(
            chunk_hash("vllm-mooncake-store:team@DeepSeek-V3.2@00ff", PREFIX),
            None
        );
        // Not a full hash.
        assert_eq!(chunk_hash(&format!("{PREFIX}00ff"), PREFIX), None);
        assert_eq!(chunk_hash(&format!("{PREFIX}{:016x}zz", 1), PREFIX), None);
    }

    #[test]
    fn announcements_come_from_runtime_configs() {
        let mut announcing = ModelRuntimeConfig::new();
        announcing
            .set_engine_specific(VLLM_MOONCAKE_STORE_RUNTIME_KEY, announcement("tcp://a:1"))
            .unwrap();
        let mut empty_endpoint = ModelRuntimeConfig::new();
        empty_endpoint
            .set_engine_specific(VLLM_MOONCAKE_STORE_RUNTIME_KEY, announcement(""))
            .unwrap();
        let mut unreadable = ModelRuntimeConfig::new();
        unreadable
            .set_engine_specific(VLLM_MOONCAKE_STORE_RUNTIME_KEY, "tcp://a:1")
            .unwrap();
        let configs = HashMap::from([
            (1, announcing),
            (2, empty_endpoint),
            (3, unreadable),
            (4, ModelRuntimeConfig::new()),
        ]);
        assert_eq!(
            announcements(&configs),
            HashMap::from([(1, announcement("tcp://a:1"))])
        );
    }

    #[tokio::test]
    async fn workers_get_the_leading_chunks_of_their_own_store() {
        let (cache, a, b) = cache_with_stores();
        cache.observe_stored(WorkerWithDpRank::new(3, 0), &stored(6));
        cache.apply(
            &a,
            1,
            (101..=104)
                .map(|hash| event("stored", hash, "cpu"))
                .collect(),
        );
        // Store b holds blocks 1-2 and 4: its run stops at block 3.
        cache.apply(
            &b,
            1,
            [101, 102, 104]
                .map(|hash| event("stored", hash, "cpu"))
                .into(),
        );

        assert_eq!(
            hits(&cache, 6).await,
            HashMap::from([(1, 4), (2, 4), (3, 2), (4, 0)])
        );
        // The prompt's own length bounds the run.
        assert_eq!(
            hits(&cache, 3).await,
            HashMap::from([(1, 3), (2, 3), (3, 2), (4, 0)])
        );
    }

    #[tokio::test]
    async fn lookups_follow_the_learned_chain_only() {
        let (cache, a, _) = cache_with_stores();
        cache.apply(
            &a,
            1,
            (101..=104)
                .map(|hash| event("stored", hash, "cpu"))
                .collect(),
        );
        // Nothing learned yet: the router cannot name any chunk.
        assert_eq!(hits(&cache, 4).await[&1], 0);

        // A continuation learned on its own still needs its parent's link.
        cache.observe_stored(
            WorkerWithDpRank::new(1, 0),
            &KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(102)),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(103),
                    tokens_hash: LocalBlockHash(3),
                    mm_extra_info: None,
                }],
            },
        );
        assert_eq!(hits(&cache, 4).await[&1], 0);
        cache.observe_stored(WorkerWithDpRank::new(1, 0), &stored(2));
        assert_eq!(hits(&cache, 4).await[&1], 3);

        // Identical blocks under another parent are other chunks.
        let other_parent = KvCacheStoreData {
            parent_hash: Some(ExternalSequenceBlockHash(999)),
            ..stored(1)
        };
        cache.observe_stored(WorkerWithDpRank::new(1, 0), &other_parent);
        assert_eq!(hits(&cache, 4).await[&1], 3);
    }

    #[tokio::test]
    async fn a_chunk_counts_while_any_medium_holds_it() {
        let (cache, a, _) = cache_with_stores();
        cache.observe_stored(WorkerWithDpRank::new(1, 0), &stored(2));
        cache.apply(
            &a,
            1,
            vec![
                event("stored", 101, "cpu"),
                event("stored", 101, "disk"),
                event("stored", 102, "cpu"),
                // Repeated announcements change nothing.
                event("stored", 102, "cpu"),
            ],
        );
        assert_eq!(hits(&cache, 2).await[&1], 2);

        cache.apply(&a, 2, vec![event("removed", 101, "cpu")]);
        assert_eq!(hits(&cache, 2).await[&1], 2);
        cache.apply(&a, 3, vec![event("removed", 101, "disk")]);
        assert_eq!(hits(&cache, 2).await[&1], 0);
        assert_eq!(cache.inner.store_chunks.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn sequence_gaps_keep_the_store_and_duplicates_are_skipped() {
        let (cache, a, b) = cache_with_stores();
        cache.observe_stored(WorkerWithDpRank::new(1, 0), &stored(2));
        cache.apply(&a, 5, vec![event("stored", 101, "cpu")]);
        cache.apply(&b, 1, vec![event("stored", 101, "cpu")]);
        // A replayed batch is not applied twice.
        cache.apply(&a, 5, vec![event("removed", 101, "cpu")]);
        assert_eq!(hits(&cache, 2).await[&1], 1);

        // Batch 6 was lost: store a keeps what it had and takes batch 7.
        cache.apply(&a, 7, vec![event("stored", 102, "cpu")]);
        assert_eq!(
            hits(&cache, 2).await,
            HashMap::from([(1, 2), (2, 2), (3, 1), (4, 0)])
        );
        assert_eq!(cache.inner.store_chunks.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn foreign_and_unknown_events_are_ignored() {
        let (cache, a, _) = cache_with_stores();
        cache.apply(
            &a,
            1,
            vec![
                StoreEvent {
                    event_type: "stored".to_string(),
                    group_id: None,
                    medium: Some("cpu".to_string()),
                },
                StoreEvent {
                    event_type: "stored".to_string(),
                    group_id: Some(group_id(101).replace("V3.2", "V3.1")),
                    medium: Some("cpu".to_string()),
                },
                event("updated", 101, "cpu"),
            ],
        );
        assert!(a.state.read().chunks.is_empty());
    }

    #[test]
    fn the_chain_is_bounded_by_the_stores_size() {
        let (cache, a, _) = cache_with_stores();
        cache.observe_stored(
            WorkerWithDpRank::new(1, 0),
            &stored(MIN_CHAIN_LINKS as u64 + 10),
        );
        assert_eq!(cache.inner.chain.lock().len(), MIN_CHAIN_LINKS);
        // The oldest links went first.
        assert!(!cache.inner.chain.lock().contains(&Link {
            parent: None,
            block: LocalBlockHash(1),
        }));

        let chunks = MIN_CHAIN_LINKS as u64;
        cache.apply(
            &a,
            1,
            (1..=chunks)
                .map(|hash| event("stored", hash, "cpu"))
                .collect(),
        );
        assert_eq!(cache.chain_bound(), 2 * MIN_CHAIN_LINKS);
    }

    #[test]
    fn nothing_is_learned_without_an_announced_store() {
        let cache = MooncakeStoreSharedKvCache::new(None, CancellationToken::new());
        cache.observe_stored(WorkerWithDpRank::new(1, 0), &stored(4));
        assert_eq!(cache.inner.chain.lock().len(), 0);
    }

    #[tokio::test]
    async fn follows_the_announced_stores_master() {
        use crate::utils::zmq::{bind_pub_socket, send_multipart};
        use tmq::AsZmqSocket;

        let publisher = bind_pub_socket("tcp://127.0.0.1:*").await.unwrap();
        let endpoint = publisher
            .lock()
            .await
            .get_socket()
            .get_last_endpoint()
            .unwrap()
            .unwrap();
        let mut runtime_config = ModelRuntimeConfig::new();
        runtime_config
            .set_engine_specific(VLLM_MOONCAKE_STORE_RUNTIME_KEY, announcement(&endpoint))
            .unwrap();
        let (tx, runtime_configs) = watch::channel(HashMap::from([(1, runtime_config)]));
        let cache = MooncakeStoreSharedKvCache::new(None, CancellationToken::new());
        cache.start(runtime_configs);

        tokio::time::timeout(Duration::from_secs(5), async {
            let mut sequence = 0_u64;
            loop {
                // Learning starts once the announcement is seen.
                cache.observe_stored(WorkerWithDpRank::new(1, 0), &stored(2));
                sequence += 1;
                // The master sends named maps with more fields than the cache reads.
                let events = vec![serde_json::json!({
                    "event_type": "stored",
                    "object_key": "DeepSeek-V3.2@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@00",
                    "group_id": group_id(101),
                    "medium": "cpu",
                    "tenant_id": "default",
                    "backend_id": "ds32",
                })];
                let payload = rmp_serde::to_vec_named(&(0_i64, events, 0_u32)).unwrap();
                send_multipart(
                    &publisher,
                    vec![vec![], sequence.to_be_bytes().to_vec(), payload],
                )
                .await
                .unwrap();
                tokio::time::sleep(Duration::from_millis(10)).await;
                if hits(&cache, 2).await[&1] == 1 {
                    break;
                }
            }
        })
        .await
        .expect("the store's chunk becomes a hit");

        // The worker stops announcing: its store is no longer followed.
        tx.send(HashMap::new()).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while !cache.inner.workers.load().is_empty() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert_eq!(hits(&cache, 2).await[&1], 0);
        tokio::time::timeout(Duration::from_secs(5), async {
            while cache.inner.store_chunks.load(Ordering::Relaxed) != 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        cache.shutdown();
    }
}
