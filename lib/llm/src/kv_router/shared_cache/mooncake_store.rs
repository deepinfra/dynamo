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
//! - **Chain.** The connector names a chunk by vLLM's chained block hash. With
//!   the workers' hash scheme ([`BlockHashScheme`], announced or set with
//!   [`BLOCK_HASH_ENV`]) the cache computes it from the prompt's tokens.
//!   Otherwise, and as a check, it uses links learned from device-tier `Stored`
//!   events, which carry it next to the router's block hash:
//!   `(parent block hash, router block hash) -> block hash`
//!   ([`SharedKvCache::observe_stored`]). Learned links outlive the GPU copy; the
//!   least recently used are dropped beyond twice the chunks of all stores.
//! - **Presence.** One subscriber per store follows its master's KV events
//!   (`mooncake_master --enable_kv_events`): `stored`/`removed` per object and
//!   replica medium, with the chunk's group id
//!   `vllm-mooncake-store:[{cache_prefix}@]{model}@{chunk hash hex}`. A chunk
//!   counts while any medium holds it. After the subscriber (re)connects or
//!   misses a batch, it reads the master's snapshot (`GET /kv_events/snapshot`
//!   on its admin server) and applies the batches from the snapshot's sequence
//!   on, so a new router knows the whole store within seconds. Until a snapshot
//!   loads, or from a master that serves none, only chunks stored since the
//!   subscriber connected are known. A complete store is read again every 30
//!   minutes; what differs is what the master changed without an event. The
//!   frontend's admin API reports each store and reads its snapshot on request
//!   ([`MooncakeStoreSharedKvCache::status`], [`MooncakeStoreSharedKvCache::resync`]).
//!
//! The routers' ideal KV caches ([`MooncakeStoreSharedKvCache::feed_ideal_cache`])
//! are stamped with every chunk a store holds when they start, and with every
//! chunk stored after; removals never reach them.
//!
//! A worker's hits are the leading chunks of the prompt its store holds, as the
//! connector loads one prefix. A chunk must be one router block (the connector's
//! default chunk size); other chunk sizes find no hits.

use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::sync::{
    Arc, OnceLock, Weak,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::{Duration, Instant};

use arc_swap::{ArcSwap, ArcSwapOption};
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
use prometheus::{IntCounterVec, IntGauge};
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, mpsc, watch};
use tokio_util::sync::CancellationToken;

use super::{MOONCAKE_EVENT_RECONNECT_DELAY, parse_mooncake_event_batch};
use crate::kv_router::ideal_cache::IdealCache;
use crate::kv_router::vllm_block_hash::{BLOCK_HASH_ENV, BlockHashScheme, VllmBlockHasher};
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
/// `mooncake_master --metrics_port` default, where it serves its admin API.
const MOONCAKE_ADMIN_PORT: u16 = 9003;
const SNAPSHOT_PAGE_OBJECTS: usize = 65536;
/// Start of every worker's group prefix: `vllm-mooncake-store:[{cache_prefix}@]{model}@`.
const GROUP_ID_PREFIX: &str = "vllm-mooncake-store:";
const SNAPSHOT_PAGE_TIMEOUT: Duration = Duration::from_secs(60);
/// How long a new subscription may stay silent before the snapshot is read anyway.
const SUBSCRIBE_GRACE: Duration = Duration::from_secs(1);
const SNAPSHOT_RETRY_MIN: Duration = Duration::from_secs(1);
const SNAPSHOT_RETRY_MAX: Duration = Duration::from_secs(60);
/// Retry delay for a master that serves no snapshot (older masters).
const SNAPSHOT_UNSUPPORTED_RETRY: Duration = Duration::from_secs(600);
/// Events kept for replay while a snapshot loads; a load that outlasts them starts over.
const MAX_PENDING_EVENTS: usize = 1 << 22;
/// How often a complete store is read again. The difference it finds is what
/// the master changed without an event; it heals then.
const SNAPSHOT_RESYNC: Duration = Duration::from_secs(1800);

/// The store a worker's `MooncakeStoreConnector` writes to and loads from.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StoreAnnouncement {
    /// ZMQ endpoint of the store master's KV events, as the router reaches it.
    pub kv_events_endpoint: String,
    /// Group id of the worker's chunks up to the chunk hash:
    /// `vllm-mooncake-store:[{cache_prefix}@]{model}@`.
    pub group_prefix: String,
    /// The master's admin server (`mooncake_master --metrics_port`), as the router
    /// reaches it. Defaults to the KV events host at port 9003.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admin_url: Option<String>,
}

impl StoreAnnouncement {
    fn snapshot_url(&self) -> Option<String> {
        let admin_url = match &self.admin_url {
            Some(url) => url.trim_end_matches('/').to_string(),
            None => {
                let address = self.kv_events_endpoint.strip_prefix("tcp://")?;
                let (host, _) = address.rsplit_once(':')?;
                if host.is_empty() || host == "*" {
                    return None;
                }
                format!("http://{host}:{MOONCAKE_ADMIN_PORT}")
            }
        };
        Some(format!("{admin_url}/kv_events/snapshot"))
    }
}

/// What a worker announces under [`VLLM_MOONCAKE_STORE_RUNTIME_KEY`].
#[derive(Debug, Deserialize)]
struct WorkerAnnouncement {
    #[serde(flatten)]
    store: StoreAnnouncement,
    #[serde(default)]
    block_hash: Option<BlockHashScheme>,
}

/// One object event of a Mooncake master; other fields are ignored.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct StoreEvent {
    event_type: String,
    #[serde(default)]
    group_id: Option<String>,
    /// Used when the object has no group id (see [`object_chunk`]).
    #[serde(default)]
    object_key: Option<String>,
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
    /// A snapshot was installed: from then on a gap or reconnect keeps the
    /// chunks, stale until the next snapshot replaces them.
    snapshots: AtomicBool,
    /// The chunks are a snapshot plus every batch since.
    complete: AtomicBool,
    /// Asks the subscriber to read the snapshot now.
    resync: Notify,
    /// A snapshot read is in flight.
    loading: AtomicBool,
    /// The last snapshot read that ended.
    loads: watch::Sender<Option<LoadReport>>,
}

impl Store {
    fn new(announcement: StoreAnnouncement) -> Self {
        Self {
            announcement,
            state: RwLock::default(),
            snapshots: AtomicBool::new(false),
            complete: AtomicBool::new(false),
            resync: Notify::new(),
            loading: AtomicBool::new(false),
            loads: watch::Sender::new(None),
        }
    }

    fn status(&self, workers: Vec<WorkerId>) -> StoreStatus {
        let (chunks, sequence) = {
            let state = self.state.read();
            (state.chunks.len(), state.sequence)
        };
        let last_load = self.loads.borrow().as_ref().map(|report| LoadStatus {
            result: report.result,
            error: report.error.clone(),
            seconds_ago: report.ended.elapsed().as_secs_f64(),
            seconds: report.took.as_secs_f64(),
            installed: report.installed,
        });
        StoreStatus {
            kv_events_endpoint: self.announcement.kv_events_endpoint.clone(),
            snapshot_url: self.announcement.snapshot_url(),
            group_prefix: self.announcement.group_prefix.clone(),
            workers,
            complete: self.complete.load(Ordering::Relaxed),
            loading: self.loading.load(Ordering::Relaxed),
            chunks,
            sequence,
            last_load,
            resync: None,
        }
    }

    /// Asks for a snapshot read now; a read in flight answers it too. With
    /// `wait`, returns how the read ended, waiting through reads retried at once.
    async fn resync(&self, wait: Option<Duration>) -> ResyncResult {
        if self.announcement.snapshot_url().is_none() {
            return ResyncResult::NoSnapshotUrl;
        }
        let mut loads = self.loads.subscribe();
        let seen = loads
            .borrow_and_update()
            .as_ref()
            .map_or(0, |report| report.attempt);
        self.resync.notify_one();
        let Some(wait) = wait else {
            return ResyncResult::Requested;
        };
        let ended = async {
            let report = loads
                .wait_for(|report| {
                    report
                        .as_ref()
                        .is_some_and(|report| report.attempt > seen && !report.result.retried())
                })
                .await
                .ok()?;
            report.as_ref().map(|report| report.result)
        };
        match tokio::time::timeout(wait, ended).await {
            Ok(Some(LoadResult::Installed)) => ResyncResult::Installed,
            Ok(Some(LoadResult::Unsupported)) => ResyncResult::Unsupported,
            Ok(Some(_)) => ResyncResult::Failed,
            Ok(None) => ResyncResult::Stopped,
            Err(_) => ResyncResult::Timeout,
        }
    }
}

/// How a snapshot read ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadResult {
    Installed,
    /// A batch after the snapshot was lost; read again after a second.
    Stale,
    /// More events arrived meanwhile than are kept for replay; read again at once.
    Overrun,
    /// The master serves no snapshot; read again after ten minutes.
    Unsupported,
    Failed,
}

impl LoadResult {
    fn label(self) -> &'static str {
        match self {
            Self::Installed => "installed",
            Self::Stale => "stale",
            Self::Overrun => "overrun",
            Self::Unsupported => "unsupported",
            Self::Failed => "failed",
        }
    }

    /// The subscriber reads again right away.
    fn retried(self) -> bool {
        matches!(self, Self::Stale | Self::Overrun)
    }
}

/// What an installed snapshot held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Installed {
    pub objects: usize,
    pub chunks: usize,
    /// Batches received during the read, applied on top.
    pub replayed: usize,
    /// When the index was complete before: chunks it held that the snapshot
    /// lacks, and the reverse. What the events missed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stale: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub missing: Option<usize>,
}

#[derive(Debug, Clone)]
struct LoadReport {
    /// Reads ended since the store was announced.
    attempt: u64,
    result: LoadResult,
    error: Option<String>,
    ended: Instant,
    took: Duration,
    installed: Option<Installed>,
}

/// A store as the admin API reports it.
#[derive(Debug, Clone, Serialize)]
pub struct StoreStatus {
    pub kv_events_endpoint: String,
    pub snapshot_url: Option<String>,
    pub group_prefix: String,
    /// The workers that announce the store.
    pub workers: Vec<WorkerId>,
    /// The chunks are the master's snapshot plus every batch since.
    pub complete: bool,
    pub loading: bool,
    pub chunks: usize,
    /// The last batch applied.
    pub sequence: Option<u64>,
    pub last_load: Option<LoadStatus>,
    /// How the requested resync ended; only in resync replies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resync: Option<ResyncResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoadStatus {
    pub result: LoadResult,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub seconds_ago: f64,
    pub seconds: f64,
    #[serde(flatten)]
    pub installed: Option<Installed>,
}

/// How a requested resync of a store ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResyncResult {
    /// Asked for, not waited on.
    Requested,
    Installed,
    Unsupported,
    Failed,
    /// No read ended within the wait.
    Timeout,
    /// The store has no admin server to read from.
    NoSnapshotUrl,
    /// The store is no longer followed.
    Stopped,
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

    /// Applies object events; returns how many changed a chunk, how many were
    /// ignored, and the chunks stored.
    fn apply_events(
        &mut self,
        group_prefix: &str,
        events: Vec<StoreEvent>,
    ) -> (u64, u64, Vec<u64>) {
        let (mut applied, mut ignored, mut stored) = (0, 0, Vec::new());
        for event in events {
            let Some(chunk) = object_chunk(
                event.group_id.as_deref(),
                event.object_key.as_deref(),
                group_prefix,
            ) else {
                ignored += 1;
                continue;
            };
            let bit = self.medium_bit(event.medium.as_deref().unwrap_or_default());
            match event.event_type.as_str() {
                "stored" => {
                    *self.chunks.entry(chunk).or_default() |= bit;
                    stored.push(chunk);
                }
                "removed" => {
                    if let Entry::Occupied(mut held) = self.chunks.entry(chunk) {
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
        (applied, ignored, stored)
    }
}

/// How a batch met the store's sequence.
#[derive(Debug, PartialEq, Eq)]
enum Applied {
    Next,
    Duplicate,
    /// Some batches before it were lost.
    Gap,
}

/// Event batches received while a snapshot loads, for replay on top of it.
type PendingBatches = Vec<(u64, Vec<StoreEvent>)>;

/// A store's state as its master's snapshot reports it, not yet caught up.
struct Snapshot {
    state: StoreState,
    /// The first batch the snapshot does not reflect.
    next_sequence: u64,
    objects: usize,
}

#[derive(Debug, Deserialize)]
struct SnapshotPage {
    next_sequence: Option<u64>,
    next_shard: Option<u64>,
    /// Object key (nil: not asked for), group id, media.
    objects: Vec<(Option<String>, Option<String>, Vec<String>)>,
}

#[derive(Debug)]
enum SnapshotError {
    /// The master serves no snapshot: an older master, or KV events off.
    Unsupported(String),
    Failed(anyhow::Error),
}

struct Inner {
    chain: Mutex<LruCache<Link, u64>>,
    /// Computes vLLM chunk hashes when all announcing workers share one scheme.
    hasher: ArcSwapOption<VllmBlockHasher>,
    /// The scheme of workers that announce none.
    default_scheme: Option<BlockHashScheme>,
    /// Chunks of all stores; sizes the chain bound.
    store_chunks: AtomicUsize,
    /// Each announcing worker's store, replaced whole when announcements change.
    workers: ArcSwap<HashMap<WorkerId, Arc<Store>>>,
    http: reqwest::Client,
    metrics: Option<Arc<MooncakeStoreMetrics>>,
    /// The routers' trackers stamped with stored chunks; see [`Self::feed_ideal_cache`].
    ideal_caches: Mutex<Vec<Weak<IdealCache>>>,
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
        let default_scheme = std::env::var(BLOCK_HASH_ENV).ok().and_then(|value| {
            let scheme = BlockHashScheme::from_env(&value);
            if scheme.is_none() {
                tracing::warn!(env = BLOCK_HASH_ENV, %value, "Ignoring an unreadable block hash scheme");
            }
            scheme
        });
        Self::with_default_scheme(metrics, cancellation_token, default_scheme)
    }

    fn with_default_scheme(
        metrics: Option<Arc<MooncakeStoreMetrics>>,
        cancellation_token: CancellationToken,
        default_scheme: Option<BlockHashScheme>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                chain: Mutex::new(LruCache::unbounded()),
                hasher: ArcSwapOption::empty(),
                default_scheme,
                store_chunks: AtomicUsize::new(0),
                workers: ArcSwap::default(),
                http: reqwest::Client::builder()
                    .timeout(SNAPSHOT_PAGE_TIMEOUT)
                    .build()
                    .unwrap_or_default(),
                metrics,
                ideal_caches: Mutex::new(Vec::new()),
                cancellation_token,
            }),
        }
    }

    /// A cache whose workers announce these KV events endpoints, without subscribers.
    #[cfg(test)]
    pub(crate) fn with_unfollowed_stores(endpoints: &[(WorkerId, &str)]) -> Self {
        let cache = Self::with_default_scheme(None, CancellationToken::new(), None);
        let mut stores: HashMap<&str, Arc<Store>> = HashMap::new();
        let workers = endpoints
            .iter()
            .map(|&(worker, endpoint)| {
                let store = stores.entry(endpoint).or_insert_with(|| {
                    Arc::new(Store::new(StoreAnnouncement {
                        kv_events_endpoint: endpoint.to_string(),
                        group_prefix: "vllm-mooncake-store:test@".to_string(),
                        admin_url: None,
                    }))
                });
                (worker, store.clone())
            })
            .collect();
        cache.inner.workers.store(Arc::new(workers));
        cache
    }

    /// Follow the stores the workers in `runtime_configs` announce.
    pub fn start(&self, runtime_configs: RuntimeConfigWatch) {
        tokio::spawn(self.clone().supervise(runtime_configs));
    }

    /// Stamps a router's ideal KV cache with every chunk the stores hold, then
    /// with every chunk they store, live or from a snapshot. Removals skip it.
    pub fn feed_ideal_cache(&self, ideal_cache: &Arc<IdealCache>) {
        // Registered first, so a chunk stored meanwhile is stamped at least once.
        self.inner
            .ideal_caches
            .lock()
            .push(Arc::downgrade(ideal_cache));
        for (store, _) in self.stores() {
            let chunks = store.state.read().chunks.keys().copied().collect();
            ideal_cache.observe_stored_hashes(chunks);
        }
    }

    fn stamp_ideal_caches(&self, chunks: Vec<u64>) {
        let mut ideal_caches = self.inner.ideal_caches.lock();
        ideal_caches.retain(|ideal_cache| ideal_cache.strong_count() > 0);
        for ideal_cache in ideal_caches.iter().filter_map(Weak::upgrade) {
            ideal_cache.observe_stored_hashes(chunks.clone());
        }
    }

    pub fn shutdown(&self) {
        self.inner.cancellation_token.cancel();
    }

    /// The stores the workers announce, by KV events endpoint.
    pub fn status(&self) -> Vec<StoreStatus> {
        self.stores()
            .into_iter()
            .map(|(store, workers)| store.status(workers))
            .collect()
    }

    /// Reads every store's snapshot now, as after a gap; a read in flight
    /// counts. With `wait`, replies when each store's read ended or `wait` ran out.
    pub async fn resync(&self, wait: Option<Duration>) -> Vec<StoreStatus> {
        futures::future::join_all(
            self.stores()
                .into_iter()
                .map(|(store, workers)| async move {
                    let resync = store.resync(wait).await;
                    StoreStatus {
                        resync: Some(resync),
                        ..store.status(workers)
                    }
                }),
        )
        .await
    }

    /// Each announced store with the workers announcing it.
    fn stores(&self) -> Vec<(Arc<Store>, Vec<WorkerId>)> {
        let mut stores: Vec<(Arc<Store>, Vec<WorkerId>)> = Vec::new();
        for (&worker, store) in self.inner.workers.load().iter() {
            match stores
                .iter_mut()
                .find(|(known, _)| Arc::ptr_eq(known, store))
            {
                Some((_, workers)) => workers.push(worker),
                None => stores.push((store.clone(), vec![worker])),
            }
        }
        for (_, workers) in &mut stores {
            workers.sort_unstable();
        }
        stores.sort_by(|(a, _), (b, _)| {
            a.announcement
                .kv_events_endpoint
                .cmp(&b.announcement.kv_events_endpoint)
        });
        stores
    }

    async fn supervise(self, mut runtime_configs: RuntimeConfigWatch) {
        let cancellation_token = self.inner.cancellation_token.clone();
        let mut stores: HashMap<StoreAnnouncement, (Arc<Store>, CancellationToken)> =
            HashMap::new();
        let mut warned_schemes = false;
        loop {
            let announced = announcements(&runtime_configs.borrow_and_update());
            let mut workers = HashMap::with_capacity(announced.len());
            let mut schemes = HashSet::new();
            for (worker_id, announcement) in announced {
                let (store, _) = stores.entry(announcement.store.clone()).or_insert_with(|| {
                    let store = Arc::new(Store::new(announcement.store));
                    let token = cancellation_token.child_token();
                    tokio::spawn(self.clone().follow(store.clone(), token.clone()));
                    (store, token)
                });
                workers.insert(worker_id, store.clone());
                schemes.insert(
                    announcement
                        .block_hash
                        .or_else(|| self.inner.default_scheme.clone()),
                );
            }
            stores.retain(|_, (store, token)| {
                let announced = workers.values().any(|other| Arc::ptr_eq(other, store));
                if !announced {
                    token.cancel();
                    self.forget(store);
                }
                announced
            });
            // Hashes are computed only for the one scheme every worker shares.
            let hasher = match schemes.into_iter().collect::<Vec<_>>().as_slice() {
                [Some(scheme)] => scheme.hasher(),
                [] | [None] => None,
                _ => {
                    if !warned_schemes {
                        warned_schemes = true;
                        tracing::warn!(
                            "Workers announce different block hash schemes; using learned links only"
                        );
                    }
                    None
                }
            };
            self.inner.hasher.store(hasher.map(Arc::new));
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
        let snapshot_url = store.announcement.snapshot_url();
        // Snapshot reads report here, tagged with the load they belong to.
        let (loaded_tx, mut loaded) = mpsc::channel(1);
        let mut generation = 0_u64;
        let mut retry = SNAPSHOT_RETRY_MIN;
        'connect: loop {
            self.reset(&store, "connect");
            match connect_sub_socket(endpoint, None).await {
                Ok(mut socket) => {
                    tracing::info!(%endpoint, "Following Mooncake store KV events");
                    // The load in flight, the batches received since it began, and when.
                    let mut load: Option<(tokio::task::JoinHandle<()>, PendingBatches, Instant)> =
                        None;
                    let mut pending_events = 0;
                    // When to read the next snapshot. The first batch shows the
                    // subscription is live, and moves it up.
                    let mut next_load = snapshot_url
                        .as_ref()
                        .map(|_| tokio::time::Instant::now() + SUBSCRIBE_GRACE);
                    loop {
                        tokio::select! {
                            _ = cancellation_token.cancelled() => {
                                if let Some((task, _, _)) = load {
                                    task.abort();
                                }
                                break 'connect;
                            }
                            message = socket.next() => match message {
                                Some(Ok(frames)) => {
                                    let (sequence, events) = match parse_mooncake_event_batch::<StoreEvent>(
                                        &multipart_message(frames),
                                    ) {
                                        Ok(batch) => batch,
                                        Err(error) => {
                                            record_subscriber_error();
                                            tracing::warn!(%endpoint, %error, "Dropping invalid Mooncake KV event frame");
                                            continue;
                                        }
                                    };
                                    if let Some((task, pending, started)) = &mut load {
                                        pending_events += events.len();
                                        pending.push((sequence, events.clone()));
                                        if pending_events > MAX_PENDING_EVENTS {
                                            task.abort();
                                            let started = *started;
                                            load = None;
                                            self.finish_load(&store, LoadResult::Overrun, started, None, None);
                                            next_load = Some(tokio::time::Instant::now());
                                        }
                                    } else if next_load.is_some() && store.state.read().sequence.is_none() {
                                        next_load = Some(tokio::time::Instant::now());
                                    }
                                    if self.apply(&store, sequence, events) == Applied::Gap
                                        && load.is_none()
                                        && snapshot_url.is_some()
                                    {
                                        next_load = Some(tokio::time::Instant::now());
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
                            _ = store.resync.notified(), if snapshot_url.is_some() => {
                                // A read in flight answers the request too.
                                if load.is_none() {
                                    next_load = Some(tokio::time::Instant::now());
                                }
                            }
                            _ = sleep_until(next_load), if load.is_none() && next_load.is_some() => {
                                next_load = None;
                                generation += 1;
                                let (http, url, prefix, loaded_tx) = (
                                    self.inner.http.clone(),
                                    snapshot_url.clone().unwrap_or_default(),
                                    store.announcement.group_prefix.clone(),
                                    loaded_tx.clone(),
                                );
                                let task = tokio::spawn(async move {
                                    let result = read_snapshot(&http, &url, &prefix).await;
                                    let _ = loaded_tx.send((generation, result)).await;
                                });
                                load = Some((task, Vec::new(), Instant::now()));
                                store.loading.store(true, Ordering::Relaxed);
                                pending_events = 0;
                            }
                            Some((loaded_generation, result)) = loaded.recv() => {
                                if loaded_generation != generation {
                                    continue;
                                }
                                let Some((_, pending, started)) = load.take() else {
                                    continue;
                                };
                                let delay = match result {
                                    Ok(snapshot) => match self.install(&store, snapshot, pending) {
                                        Some(installed) => {
                                            self.finish_load(&store, LoadResult::Installed, started, None, Some(installed));
                                            retry = SNAPSHOT_RETRY_MIN;
                                            SNAPSHOT_RESYNC
                                        }
                                        None => {
                                            // A batch after the snapshot was lost: read it again.
                                            self.finish_load(&store, LoadResult::Stale, started, None, None);
                                            SNAPSHOT_RETRY_MIN
                                        }
                                    },
                                    Err(SnapshotError::Unsupported(reason)) => {
                                        tracing::info!(
                                            url = snapshot_url.as_deref().unwrap_or_default(),
                                            %reason,
                                            "Mooncake master serves no KV snapshot; following its events only"
                                        );
                                        self.finish_load(&store, LoadResult::Unsupported, started, Some(reason), None);
                                        SNAPSHOT_UNSUPPORTED_RETRY
                                    }
                                    Err(SnapshotError::Failed(error)) => {
                                        tracing::warn!(
                                            url = snapshot_url.as_deref().unwrap_or_default(),
                                            %error,
                                            retry_in = ?retry,
                                            "Failed to read the Mooncake store snapshot"
                                        );
                                        self.finish_load(&store, LoadResult::Failed, started, Some(format!("{error:#}")), None);
                                        let delay = retry;
                                        retry = (retry * 2).min(SNAPSHOT_RETRY_MAX);
                                        delay
                                    }
                                };
                                next_load = Some(tokio::time::Instant::now() + delay);
                            }
                        }
                    }
                    if let Some((task, _, _)) = load {
                        task.abort();
                        store.loading.store(false, Ordering::Relaxed);
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

    fn apply(&self, store: &Store, sequence: u64, events: Vec<StoreEvent>) -> Applied {
        let mut state = store.state.write();
        let mut applied_as = Applied::Next;
        if let Some(previous) = state.sequence {
            if sequence == previous {
                return Applied::Duplicate;
            }
            // The master numbers batches consecutively: anything else lost some.
            if sequence != previous.wrapping_add(1) {
                applied_as = Applied::Gap;
                self.set_complete(store, false);
                if store.snapshots.load(Ordering::Relaxed) {
                    tracing::warn!(
                        endpoint = %store.announcement.kv_events_endpoint,
                        previous,
                        sequence,
                        "Mooncake store KV event sequence gap; reloading the store's snapshot"
                    );
                } else {
                    tracing::warn!(
                        endpoint = %store.announcement.kv_events_endpoint,
                        previous,
                        sequence,
                        "Mooncake store KV event sequence gap; forgetting the store's chunks"
                    );
                    self.clear(&mut state, "gap");
                }
            }
        }
        state.sequence = Some(sequence);

        let before = state.chunks.len();
        let (applied, ignored, stored) =
            state.apply_events(&store.announcement.group_prefix, events);
        self.stamp_ideal_caches(stored);
        let after = state.chunks.len();
        self.add_chunks(after as isize - before as isize);
        self.check_size(store, &mut state);
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
        applied_as
    }

    /// Replaces the store's chunks with `snapshot` caught up on `pending`, the
    /// batches received since it was requested. None if one it needs is missing.
    fn install(
        &self,
        store: &Store,
        snapshot: Snapshot,
        pending: PendingBatches,
    ) -> Option<Installed> {
        let Snapshot {
            state: mut fresh,
            next_sequence,
            objects,
        } = snapshot;
        let mut expected = next_sequence;
        let mut replayed = 0;
        for (sequence, events) in pending {
            if sequence < expected {
                continue;
            }
            if sequence != expected {
                tracing::warn!(
                    endpoint = %store.announcement.kv_events_endpoint,
                    expected,
                    sequence,
                    "Mooncake store snapshot misses a batch; reading it again"
                );
                return None;
            }
            fresh.apply_events(&store.announcement.group_prefix, events);
            expected += 1;
            replayed += 1;
        }
        fresh.sequence = expected.checked_sub(1);
        // A complete index should equal the snapshot: count what differs.
        let drift = store.complete.load(Ordering::Relaxed).then(|| {
            let state = store.state.read();
            let stale = state
                .chunks
                .keys()
                .filter(|chunk| !fresh.chunks.contains_key(chunk))
                .count();
            let missing = fresh
                .chunks
                .keys()
                .filter(|chunk| !state.chunks.contains_key(chunk))
                .count();
            (stale, missing)
        });
        let mut state = store.state.write();
        // Chunks the index lacked are stores it missed: all of them at the first
        // load. The others were stamped when their events came; stamping them again
        // on every 30-minute read would keep them past their window.
        let unknown = fresh
            .chunks
            .keys()
            .filter(|chunk| !state.chunks.contains_key(chunk))
            .copied()
            .collect();
        self.stamp_ideal_caches(unknown);
        let (before, after) = (state.chunks.len(), fresh.chunks.len());
        *state = fresh;
        self.add_chunks(after as isize - before as isize);
        store.snapshots.store(true, Ordering::Relaxed);
        self.set_complete(store, true);
        self.check_size(store, &mut state);
        let (stale, missing) = drift.unwrap_or_default();
        if let Some(metrics) = &self.inner.metrics {
            metrics
                .resync_drift
                .with_label_values(&["stale"])
                .inc_by(stale as u64);
            metrics
                .resync_drift
                .with_label_values(&["missing"])
                .inc_by(missing as u64);
        }
        tracing::info!(
            endpoint = %store.announcement.kv_events_endpoint,
            objects,
            chunks = after,
            replayed,
            resync = drift.is_some(),
            stale,
            missing,
            "Loaded the Mooncake store's snapshot"
        );
        Some(Installed {
            objects,
            chunks: after,
            replayed,
            stale: drift.map(|(stale, _)| stale),
            missing: drift.map(|(_, missing)| missing),
        })
    }

    fn check_size(&self, store: &Store, state: &mut StoreState) {
        if state.chunks.len() > MAX_STORE_CHUNKS {
            tracing::warn!(
                endpoint = %store.announcement.kv_events_endpoint,
                chunks = state.chunks.len(),
                "Mooncake store index exceeded its size limit; forgetting the store's chunks"
            );
            self.clear(state, "size");
            self.set_complete(store, false);
        }
    }

    /// A new subscription: batches it missed may have changed the store. Chunks a
    /// snapshot loaded stay until the next one replaces them.
    fn reset(&self, store: &Store, reason: &'static str) {
        let mut state = store.state.write();
        if !store.snapshots.load(Ordering::Relaxed) {
            self.clear(&mut state, reason);
        }
        state.sequence = None;
        self.set_complete(store, false);
    }

    fn forget(&self, store: &Store) {
        let mut state = store.state.write();
        self.add_chunks(-(state.chunks.len() as isize));
        *state = StoreState::default();
        store.snapshots.store(false, Ordering::Relaxed);
        self.set_complete(store, false);
    }

    fn clear(&self, state: &mut StoreState, reason: &'static str) {
        self.add_chunks(-(state.chunks.len() as isize));
        state.chunks = HashMap::new();
        if let Some(metrics) = &self.inner.metrics {
            metrics.resets.with_label_values(&[reason]).inc();
        }
    }

    fn set_complete(&self, store: &Store, complete: bool) {
        if store.complete.swap(complete, Ordering::Relaxed) != complete
            && let Some(metrics) = &self.inner.metrics
        {
            if complete {
                metrics.loaded_stores.inc();
            } else {
                metrics.loaded_stores.dec();
            }
        }
    }

    /// Counts a snapshot read that ended and reports it to the admin API.
    fn finish_load(
        &self,
        store: &Store,
        result: LoadResult,
        started: Instant,
        error: Option<String>,
        installed: Option<Installed>,
    ) {
        store.loading.store(false, Ordering::Relaxed);
        if let Some(metrics) = &self.inner.metrics {
            metrics.snapshots.with_label_values(&[result.label()]).inc();
        }
        store.loads.send_modify(|last| {
            *last = Some(LoadReport {
                attempt: last.as_ref().map_or(0, |last| last.attempt) + 1,
                result,
                error,
                ended: Instant::now(),
                took: started.elapsed(),
                installed,
            })
        });
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

async fn sleep_until(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

/// Every page of a master's snapshot, from shard 0, into a store state.
async fn read_snapshot(
    http: &reqwest::Client,
    url: &str,
    group_prefix: &str,
) -> Result<Snapshot, SnapshotError> {
    let started = Instant::now();
    let mut state = StoreState::default();
    let (mut next_sequence, mut objects, mut pages) = (None, 0, 0);
    let mut shard = Some(0);
    while let Some(from) = shard {
        let response = http
            .get(url)
            .query(&[
                ("shard", from.to_string()),
                ("max_objects", SNAPSHOT_PAGE_OBJECTS.to_string()),
                // Keys stand in for a missing group id (see [`object_chunk`]).
                ("object_keys", "1".to_string()),
            ])
            .send()
            .await
            .map_err(|error| SnapshotError::Failed(error.into()))?;
        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND
            || status == reqwest::StatusCode::NOT_IMPLEMENTED
        {
            return Err(SnapshotError::Unsupported(format!("HTTP {status}")));
        }
        let body = response
            .error_for_status()
            .map_err(|error| SnapshotError::Failed(error.into()))?
            .bytes()
            .await
            .map_err(|error| SnapshotError::Failed(error.into()))?;
        let page: SnapshotPage =
            rmp_serde::from_slice(&body).map_err(|error| SnapshotError::Failed(error.into()))?;
        if pages == 0 {
            // Batches from this one on are not in the pages.
            next_sequence = Some(page.next_sequence.ok_or_else(|| {
                SnapshotError::Unsupported("the master publishes no KV events".to_string())
            })?);
        }
        pages += 1;
        objects += page.objects.len();
        for (object_key, group_id, media) in page.objects {
            let Some(chunk) =
                object_chunk(group_id.as_deref(), object_key.as_deref(), group_prefix)
            else {
                continue;
            };
            let mut bits = 0;
            for medium in &media {
                bits |= state.medium_bit(medium);
            }
            if bits != 0 {
                *state.chunks.entry(chunk).or_default() |= bits;
            }
        }
        shard = page.next_shard;
    }
    tracing::debug!(url, pages, objects, chunks = state.chunks.len(), elapsed = ?started.elapsed(), "Read a Mooncake store snapshot");
    Ok(Snapshot {
        state,
        next_sequence: next_sequence.unwrap_or_default(),
        objects,
    })
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
        // A cache namespace changes vLLM's hashes (extra keys): use learned links.
        let computed = match (
            &*self.inner.hasher.load(),
            query
                .cache_namespace
                .filter(|namespace| !namespace.is_empty()),
        ) {
            (Some(hasher), None) => hasher.chunk_hashes(
                query.tokens,
                query.block_size as usize,
                query.block_hashes.len(),
            ),
            _ => None,
        };

        // Each store's run of leading chunks; the walk ends once none extends.
        let mut runs = vec![0_u32; stores.len()];
        let mut verdicts = [0_u64; 3];
        let stop = {
            let states: Vec<_> = stores.iter().map(|store| store.state.read()).collect();
            let mut chain = self.inner.chain.lock();
            let mut computed = computed.as_deref();
            let mut parent = None;
            let mut stop = "end";
            for (position, &block) in (0_u32..).zip(query.block_hashes) {
                let learned = chain.get(&Link { parent, block }).copied();
                let hash = match (
                    computed.and_then(|hashes| hashes.get(position as usize)),
                    learned,
                ) {
                    (Some(&computed_hash), Some(learned_hash)) if computed_hash != learned_hash => {
                        // Not the workers' hash for this prompt: trust learned links from here.
                        verdicts[Verdict::Differ as usize] += 1;
                        computed = None;
                        learned_hash
                    }
                    (Some(&computed_hash), learned_hash) => {
                        let verdict = if learned_hash.is_some() {
                            Verdict::Agree
                        } else {
                            Verdict::Unverified
                        };
                        verdicts[verdict as usize] += 1;
                        computed_hash
                    }
                    (None, Some(learned_hash)) => learned_hash,
                    (None, None) => {
                        stop = "unlinked";
                        break;
                    }
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
            for verdict in [Verdict::Agree, Verdict::Unverified, Verdict::Differ] {
                let count = verdicts[verdict as usize];
                if count > 0 {
                    metrics
                        .computed_hashes
                        .with_label_values(&[verdict.label()])
                        .inc_by(count);
                }
            }
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

/// How a computed chunk hash compared with the learned link at its position.
#[derive(Clone, Copy)]
enum Verdict {
    Agree,
    Unverified,
    Differ,
}

impl Verdict {
    fn label(self) -> &'static str {
        match self {
            Verdict::Agree => "agree",
            Verdict::Unverified => "unverified",
            Verdict::Differ => "differ",
        }
    }
}

/// The store each worker announces. A worker without one, or with an
/// unreadable one, gets no hits.
fn announcements(
    runtime_configs: &HashMap<WorkerId, ModelRuntimeConfig>,
) -> HashMap<WorkerId, WorkerAnnouncement> {
    runtime_configs
        .iter()
        .filter_map(|(&worker_id, runtime_config)| {
            match runtime_config
                .get_engine_specific::<WorkerAnnouncement>(VLLM_MOONCAKE_STORE_RUNTIME_KEY)
            {
                Ok(Some(announcement)) if !announcement.store.kv_events_endpoint.is_empty() => {
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

/// The chunk an object belongs to: from its group id, or else from its key.
///
/// The connector derives an object's group id from the chunk hash that ends its key, so both name the same chunk.
/// An object re-registered from the SSD tier after a master restart (`ReRegisterOffloadedObjects`) has no group id,
/// only its key: without the key fallback a backfilled index misses every such chunk.
fn object_chunk(
    group_id: Option<&str>,
    object_key: Option<&str>,
    group_prefix: &str,
) -> Option<u64> {
    group_id
        .and_then(|group_id| chunk_hash(group_id, group_prefix))
        .or_else(|| object_key.and_then(|key| key_chunk_hash(key, group_prefix)))
}

/// The chunk hash of an object key `[{cache_prefix}@]{model}@tp_rank:…@{chunk hash hex}`, for the worker whose
/// group prefix is `vllm-mooncake-store:[{cache_prefix}@]{model}@`.
fn key_chunk_hash(object_key: &str, group_prefix: &str) -> Option<u64> {
    let scope = group_prefix.strip_prefix(GROUP_ID_PREFIX)?;
    if scope.is_empty() || !object_key.starts_with(scope) {
        return None;
    }
    let (_, hex) = object_key.rsplit_once('@')?;
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
    loaded_stores: IntGauge,
    chunks: IntGauge,
    chain_links: IntGauge,
    events: IntCounterVec,
    resets: IntCounterVec,
    snapshots: IntCounterVec,
    resync_drift: IntCounterVec,
    lookup_stops: IntCounterVec,
    computed_hashes: IntCounterVec,
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
            loaded_stores: metrics.create_intgauge(
                "router_mooncake_store_loaded_stores",
                "Followed stores whose chunks are a snapshot plus every KV event since; a router is warm when this equals the stores",
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
            resets: metrics.create_intcountervec(
                "router_mooncake_store_resets_total",
                "Times a store's chunks were forgotten (connect or gap without a snapshot, or size)",
                &["reason"],
                &[],
            )?,
            snapshots: metrics.create_intcountervec(
                "router_mooncake_store_snapshots_total",
                "Store snapshot loads: installed, stale (missed a batch, read again), overrun, failed or unsupported",
                &["result"],
                &[],
            )?,
            resync_drift: metrics.create_intcountervec(
                "router_mooncake_store_resync_drift_total",
                "Chunks a periodic snapshot found wrong in a complete index: stale (the master no longer holds it) or missing; the KV events missed these",
                &["kind"],
                &[],
            )?,
            lookup_stops: metrics.create_intcountervec(
                "router_mooncake_store_lookup_stops_total",
                "Where lookups stopped: end of prompt, a block no store holds (absent) or a block with no vLLM hash (unlinked)",
                &["reason"],
                &[],
            )?,
            computed_hashes: metrics.create_intcountervec(
                "router_mooncake_store_computed_hashes_total",
                "Computed vLLM block hashes a lookup used, by how they compared with the learned link: agree, unverified (none learned) or differ",
                &["result"],
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
            admin_url: None,
        }
    }

    fn event(event_type: &str, hash: u64, medium: &str) -> StoreEvent {
        StoreEvent {
            event_type: event_type.to_string(),
            group_id: Some(group_id(hash)),
            object_key: Some(object_key(hash)),
            medium: Some(medium.to_string()),
        }
    }

    /// The key the connector writes for `hash` under [`PREFIX`]'s model.
    fn object_key(hash: u64) -> String {
        format!(
            "DeepSeek-V3.2@tp_rank:0@pcp0@dcp0@pp_rank:0@group:0@{:048x}{hash:016x}",
            0xabcdef_u64
        )
    }

    /// An event for an object the master knows only by key, as after an SSD re-registration.
    fn keyed_event(event_type: &str, hash: u64, medium: &str) -> StoreEvent {
        StoreEvent {
            group_id: None,
            ..event(event_type, hash, medium)
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
        let cache =
            MooncakeStoreSharedKvCache::with_default_scheme(None, CancellationToken::new(), None);
        let store = |endpoint: &str| Arc::new(Store::new(announcement(endpoint)));
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
        let announced = announcements(&configs);
        assert_eq!(announced.len(), 1);
        assert_eq!(announced[&1].store, announcement("tcp://a:1"));
        assert_eq!(announced[&1].block_hash, None);
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
    async fn sequence_gaps_forget_the_store_and_duplicates_are_skipped() {
        let (cache, a, b) = cache_with_stores();
        cache.observe_stored(WorkerWithDpRank::new(1, 0), &stored(2));
        cache.apply(&a, 5, vec![event("stored", 101, "cpu")]);
        cache.apply(&b, 1, vec![event("stored", 101, "cpu")]);
        // A replayed batch is not applied twice.
        cache.apply(&a, 5, vec![event("removed", 101, "cpu")]);
        assert_eq!(hits(&cache, 2).await[&1], 1);

        // Batch 6 was lost: store a's chunks are unknown, store b is untouched.
        cache.apply(&a, 7, vec![event("stored", 102, "cpu")]);
        assert_eq!(
            hits(&cache, 2).await,
            HashMap::from([(1, 0), (2, 0), (3, 1), (4, 0)])
        );
        assert_eq!(cache.inner.store_chunks.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn key_chunk_hash_reads_the_chunk_that_ends_the_key() {
        assert_eq!(key_chunk_hash(&object_key(0x1234), PREFIX), Some(0x1234));
        assert_eq!(object_chunk(None, Some(&object_key(7)), PREFIX), Some(7));
        // The group id wins when both are there.
        assert_eq!(
            object_chunk(Some(&group_id(5)), Some(&object_key(7)), PREFIX),
            Some(5)
        );
        // Another model's key, a key without a hex hash, and no key at all.
        assert_eq!(
            key_chunk_hash(&object_key(1).replace("V3.2", "V3.1"), PREFIX),
            None
        );
        assert_eq!(
            key_chunk_hash("DeepSeek-V3.2@tp_rank:0@group:0@not-hex", PREFIX),
            None
        );
        assert_eq!(key_chunk_hash(&object_key(1), "no-group-prefix@"), None);
        assert_eq!(object_chunk(None, None, PREFIX), None);
    }

    #[test]
    fn objects_known_only_by_key_count() {
        let (cache, a, _) = cache_with_stores();
        cache.apply(
            &a,
            1,
            vec![
                keyed_event("stored", 101, "disk"),
                keyed_event("stored", 102, "disk"),
            ],
        );
        assert_eq!(a.state.read().chunks.len(), 2);
        cache.apply(&a, 2, vec![keyed_event("removed", 101, "disk")]);
        assert_eq!(a.state.read().chunks.len(), 1);
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
                    object_key: None,
                    medium: Some("cpu".to_string()),
                },
                StoreEvent {
                    event_type: "stored".to_string(),
                    group_id: Some(group_id(101).replace("V3.2", "V3.1")),
                    object_key: Some(object_key(101).replace("V3.2", "V3.1")),
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
        let cache =
            MooncakeStoreSharedKvCache::with_default_scheme(None, CancellationToken::new(), None);
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
        // A master without a snapshot: only events count.
        let events_only = StoreAnnouncement {
            admin_url: Some("http://127.0.0.1:1".to_string()),
            ..announcement(&endpoint)
        };
        runtime_config
            .set_engine_specific(VLLM_MOONCAKE_STORE_RUNTIME_KEY, events_only)
            .unwrap();
        let (tx, runtime_configs) = watch::channel(HashMap::from([(1, runtime_config)]));
        let cache =
            MooncakeStoreSharedKvCache::with_default_scheme(None, CancellationToken::new(), None);
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

    // CPython 3.12: `NONE_HASH` under PYTHONHASHSEED=0, i.e. sha256(pickle.dumps("0", protocol=5)).
    const NONE_HASH: &str = "1973e23848344dc43a988a9b478663803cfffe1243480253f9a3cf004b14aa7c";

    #[test]
    fn snapshot_urls_parse() {
        let derived = announcement("tcp://mooncake-master.ns.svc:5558");
        assert_eq!(
            derived.snapshot_url().as_deref(),
            Some("http://mooncake-master.ns.svc:9003/kv_events/snapshot")
        );
        let explicit = StoreAnnouncement {
            admin_url: Some("http://10.0.0.1:9903/".to_string()),
            ..derived
        };
        assert_eq!(
            explicit.snapshot_url().as_deref(),
            Some("http://10.0.0.1:9903/kv_events/snapshot")
        );
        assert_eq!(announcement("tcp://*:5558").snapshot_url(), None);
        assert_eq!(announcement("ipc:///tmp/events").snapshot_url(), None);
    }

    #[test]
    fn workers_announce_their_block_hash_scheme() {
        let mut runtime_config = ModelRuntimeConfig::new();
        runtime_config
            .set_engine_specific(
                VLLM_MOONCAKE_STORE_RUNTIME_KEY,
                serde_json::json!({
                    "kv_events_endpoint": "tcp://a:1",
                    "group_prefix": PREFIX,
                    "admin_url": "http://a:9003",
                    "block_hash": {"algo": "sha256", "none_hash": NONE_HASH},
                }),
            )
            .unwrap();
        let announced = announcements(&HashMap::from([(1, runtime_config)]));
        assert_eq!(
            announced[&1].store.admin_url.as_deref(),
            Some("http://a:9003")
        );
        assert_eq!(
            announced[&1]
                .block_hash
                .as_ref()
                .map(|scheme| scheme.algo.as_str()),
            Some("sha256")
        );
    }

    async fn token_hits(
        cache: &MooncakeStoreSharedKvCache,
        tokens: &[u32],
        cache_namespace: Option<&str>,
    ) -> HashMap<WorkerId, u32> {
        let block_hashes = router_hashes((tokens.len() / 64) as u64);
        let hits = cache
            .check_blocks(SharedCacheQuery {
                tokens,
                block_hashes: &block_hashes,
                block_size: 64,
                cache_namespace,
            })
            .await
            .unwrap();
        [1, 2, 3]
            .into_iter()
            .map(|worker| (worker, hits.for_worker(worker).total_hits))
            .collect()
    }

    #[tokio::test]
    async fn computed_hashes_find_chunks_no_link_was_learned_for() {
        let (cache, a, b) = cache_with_stores();
        let tokens: Vec<u32> = (0..200).collect();
        let hasher = BlockHashScheme::from_env(&format!("sha256:{NONE_HASH}"))
            .and_then(|scheme| scheme.hasher())
            .unwrap();
        let chunks = hasher.chunk_hashes(&tokens, 64, 3).unwrap();
        cache.apply(
            &a,
            1,
            chunks
                .iter()
                .map(|&hash| event("stored", hash, "cpu"))
                .collect(),
        );
        cache.apply(&b, 1, vec![event("stored", chunks[0], "disk")]);
        // No scheme: nothing links the router's blocks to the store's chunks.
        assert_eq!(token_hits(&cache, &tokens, None).await[&1], 0);

        cache.inner.hasher.store(Some(Arc::new(hasher)));
        assert_eq!(
            token_hits(&cache, &tokens, None).await,
            HashMap::from([(1, 3), (2, 3), (3, 1)])
        );
        // A cache namespace changes vLLM's hashes.
        assert_eq!(token_hits(&cache, &tokens, Some("lora-a")).await[&1], 0);

        // A learned link that disagrees wins, and the rest follows learned links.
        cache.observe_stored(
            WorkerWithDpRank::new(1, 0),
            &KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(chunks[0])),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(77),
                    tokens_hash: LocalBlockHash(2),
                    mm_extra_info: None,
                }],
            },
        );
        assert_eq!(token_hits(&cache, &tokens, None).await[&1], 1);
        cache.apply(&a, 2, vec![event("stored", 77, "cpu")]);
        assert_eq!(token_hits(&cache, &tokens, None).await[&1], 2);
    }

    #[test]
    fn a_snapshot_catches_up_from_its_sequence() {
        let (cache, a, _) = cache_with_stores();
        // Live events before the snapshot arrived.
        cache.apply(&a, 4, vec![event("stored", 900, "cpu")]);
        let snapshot = |chunks: &[u64]| {
            let mut state = StoreState::default();
            state.apply_events(
                PREFIX,
                chunks
                    .iter()
                    .map(|&hash| event("stored", hash, "cpu"))
                    .collect(),
            );
            Snapshot {
                state,
                next_sequence: 6,
                objects: chunks.len(),
            }
        };
        let pending = vec![
            // Already in the snapshot.
            (5, vec![event("removed", 101, "cpu")]),
            (
                6,
                vec![event("removed", 101, "cpu"), event("stored", 103, "disk")],
            ),
            (7, vec![event("stored", 104, "cpu")]),
        ];
        assert!(cache.install(&a, snapshot(&[101, 102]), pending).is_some());
        let state = a.state.read();
        assert_eq!(
            state.chunks.keys().copied().collect::<HashSet<_>>(),
            HashSet::from([102, 103, 104])
        );
        assert_eq!(state.sequence, Some(7));
        drop(state);
        assert!(a.complete.load(Ordering::Relaxed));
        assert_eq!(cache.inner.store_chunks.load(Ordering::Relaxed), 3);

        // The next batch continues the snapshot's sequence.
        assert_eq!(
            cache.apply(&a, 8, vec![event("stored", 105, "cpu")]),
            Applied::Next
        );
        // A gap now keeps the chunks: the next snapshot replaces them.
        assert_eq!(cache.apply(&a, 10, vec![]), Applied::Gap);
        assert_eq!(a.state.read().chunks.len(), 4);
        assert!(!a.complete.load(Ordering::Relaxed));

        // A batch the snapshot needs is missing: it is not installed.
        let gap = vec![(7, vec![event("stored", 104, "cpu")])];
        assert!(cache.install(&a, snapshot(&[101]), gap).is_none());
        assert_eq!(a.state.read().chunks.len(), 4);
    }

    #[test]
    fn ideal_caches_get_every_stored_chunk_and_no_removal() {
        let (cache, a, b) = cache_with_stores();
        cache.apply(
            &a,
            1,
            vec![event("stored", 1, "cpu"), event("stored", 2, "cpu")],
        );
        cache.apply(&b, 1, vec![event("stored", 3, "disk")]);
        let (ideal_cache, stamped) = IdealCache::unstarted();
        let stamped = || {
            let mut hashes = stamped();
            hashes.sort_unstable();
            hashes
        };

        // Registering replays what the stores hold.
        cache.feed_ideal_cache(&ideal_cache);
        assert_eq!(stamped(), vec![1, 2, 3]);
        // Then every store, of any medium; removals stamp nothing.
        cache.apply(
            &a,
            2,
            vec![
                event("removed", 1, "cpu"),
                event("stored", 1, "disk"),
                event("removed", 2, "cpu"),
            ],
        );
        assert_eq!(stamped(), vec![1]);
        // A snapshot stamps the chunks the index lacked, not those it knew.
        let mut state = StoreState::default();
        state.apply_events(
            PREFIX,
            Vec::from([1, 4, 5].map(|hash| event("stored", hash, "disk"))),
        );
        let snapshot = Snapshot {
            state,
            next_sequence: 3,
            objects: 3,
        };
        assert!(cache.install(&a, snapshot, vec![]).is_some());
        assert_eq!(stamped(), vec![4, 5]);

        // A router gone leaves the list.
        drop(ideal_cache);
        cache.apply(&a, 3, vec![event("stored", 6, "cpu")]);
        assert!(cache.inner.ideal_caches.lock().is_empty());
    }

    /// A master's admin server with the snapshot of `chunks`, in pages of two
    /// objects. The chunks can change without events.
    async fn serve_snapshot(
        next_sequence: Option<u64>,
        chunks: Vec<u64>,
    ) -> (String, Arc<AtomicUsize>, Arc<Mutex<Vec<u64>>>) {
        use axum::{Router, extract::Query, routing::get};

        let requests = Arc::new(AtomicUsize::new(0));
        let counter = requests.clone();
        let chunks = Arc::new(Mutex::new(chunks));
        let held = chunks.clone();
        let app = Router::new().route(
            "/kv_events/snapshot",
            get(move |Query(query): Query<HashMap<String, String>>| {
                let chunks = held.lock().clone();
                counter.fetch_add(1, Ordering::Relaxed);
                async move {
                    assert_eq!(query.get("object_keys").map(String::as_str), Some("1"));
                    let shard: usize = query["shard"].parse().unwrap();
                    let page: Vec<_> = chunks
                        .iter()
                        .skip(shard * 2)
                        .take(2)
                        // Odd hashes come without a group id, as SSD objects re-registered after a master
                        // restart do; their keys still place them.
                        .map(|&hash| {
                            let group = (hash % 2 == 0).then(|| group_id(hash));
                            (Some(object_key(hash)), group, vec!["cpu"])
                        })
                        .collect();
                    let next_shard = (shard * 2 + 2 < chunks.len()).then_some(shard + 1);
                    rmp_serde::to_vec_named(&serde_json::json!({
                        "next_sequence": next_sequence,
                        "next_shard": next_shard,
                        "objects": page,
                    }))
                    .unwrap()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, requests, chunks)
    }

    /// A cache following one store, worker 1's, whose master serves `admin_url`.
    async fn follow_store(admin_url: String) -> MooncakeStoreSharedKvCache {
        use crate::utils::zmq::bind_pub_socket;
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
            .set_engine_specific(
                VLLM_MOONCAKE_STORE_RUNTIME_KEY,
                StoreAnnouncement {
                    admin_url: Some(admin_url),
                    ..announcement(&endpoint)
                },
            )
            .unwrap();
        let (tx, runtime_configs) = watch::channel(HashMap::from([(1, runtime_config)]));
        let cache =
            MooncakeStoreSharedKvCache::with_default_scheme(None, CancellationToken::new(), None);
        cache.start(runtime_configs);
        // Keep the master's socket and the announcement until the test ends.
        tokio::spawn(async move {
            let _publisher = publisher;
            tx.closed().await;
        });
        cache
    }

    async fn first_load(cache: &MooncakeStoreSharedKvCache) -> StoreStatus {
        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let [store] = cache.status().as_slice()
                    && store.last_load.is_some()
                {
                    return store.clone();
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the subscriber reads the snapshot")
    }

    #[tokio::test]
    async fn a_resync_reads_the_snapshot_at_once() {
        let (admin_url, requests, chunks) = serve_snapshot(Some(1), vec![101, 102, 103]).await;
        let cache = follow_store(admin_url).await;
        let store = first_load(&cache).await;
        assert!(store.complete);
        assert_eq!(store.workers, vec![1]);
        assert_eq!((store.chunks, store.sequence), (3, Some(0)));
        let load = store.last_load.unwrap();
        assert_eq!(load.result, LoadResult::Installed);
        assert_eq!(
            load.installed,
            Some(Installed {
                objects: 3,
                chunks: 3,
                replayed: 0,
                stale: None,
                missing: None,
            })
        );

        // The master changed without events: a resync finds it now, not at the
        // next periodic read.
        *chunks.lock() = vec![101, 102, 104];
        let started = Instant::now();
        let replies = cache.resync(Some(Duration::from_secs(5))).await;
        assert!(started.elapsed() < Duration::from_secs(1));
        let [store] = replies.as_slice() else {
            panic!("one store: {replies:?}")
        };
        assert_eq!(store.resync, Some(ResyncResult::Installed));
        let installed = store.last_load.as_ref().unwrap().installed.unwrap();
        assert_eq!(
            (installed.chunks, installed.stale, installed.missing),
            (3, Some(1), Some(1))
        );
        let followed = cache.inner.workers.load()[&1].clone();
        assert_eq!(
            followed
                .state
                .read()
                .chunks
                .keys()
                .copied()
                .collect::<HashSet<_>>(),
            HashSet::from([101, 102, 104])
        );
        let reply = serde_json::to_value(store).unwrap();
        assert_eq!(reply["resync"], "installed");
        assert_eq!(reply["last_load"]["result"], "installed");
        assert_eq!(reply["last_load"]["stale"], 1);

        // Without a wait it only asks; the read follows.
        let before = requests.load(Ordering::Relaxed);
        let replies = cache.resync(None).await;
        assert_eq!(replies[0].resync, Some(ResyncResult::Requested));
        tokio::time::timeout(Duration::from_secs(5), async {
            while requests.load(Ordering::Relaxed) <= before {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the requested read happens");
        cache.shutdown();
    }

    #[tokio::test]
    async fn a_resync_reports_stores_it_cannot_read() {
        // An older master: the next read would be ten minutes away.
        let (admin_url, _, _) = serve_snapshot(None, vec![101]).await;
        let cache = follow_store(admin_url).await;
        let store = first_load(&cache).await;
        assert!(!store.complete);
        assert_eq!(store.last_load.unwrap().result, LoadResult::Unsupported);
        let started = Instant::now();
        let replies = cache.resync(Some(Duration::from_secs(5))).await;
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(replies[0].resync, Some(ResyncResult::Unsupported));
        cache.shutdown();

        // No admin server to read, and a store whose subscriber never reads.
        let cache = MooncakeStoreSharedKvCache::with_unfollowed_stores(&[
            (1, "tcp://a:1"),
            (2, "tcp://*:1"),
            (3, "tcp://a:1"),
        ]);
        let replies = cache.resync(Some(Duration::from_millis(50))).await;
        let results: Vec<_> = replies
            .iter()
            .map(|store| {
                (
                    store.kv_events_endpoint.as_str(),
                    store.workers.clone(),
                    store.resync,
                )
            })
            .collect();
        assert_eq!(
            results,
            vec![
                ("tcp://*:1", vec![2], Some(ResyncResult::NoSnapshotUrl)),
                ("tcp://a:1", vec![1, 3], Some(ResyncResult::Timeout)),
            ]
        );
    }

    #[tokio::test]
    async fn a_new_subscriber_loads_the_masters_snapshot() {
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
        // The store held chunks 101-105 before the router started; batch 1 is next.
        let (admin_url, requests, _) = serve_snapshot(Some(1), (101..=105).collect()).await;
        let mut runtime_config = ModelRuntimeConfig::new();
        runtime_config
            .set_engine_specific(
                VLLM_MOONCAKE_STORE_RUNTIME_KEY,
                StoreAnnouncement {
                    admin_url: Some(admin_url),
                    ..announcement(&endpoint)
                },
            )
            .unwrap();
        let (_tx, runtime_configs) = watch::channel(HashMap::from([(1, runtime_config)]));
        let cache =
            MooncakeStoreSharedKvCache::with_default_scheme(None, CancellationToken::new(), None);
        cache.start(runtime_configs);

        tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                cache.observe_stored(WorkerWithDpRank::new(1, 0), &stored(6));
                let store = cache.inner.workers.load().get(&1).cloned();
                if store.is_some_and(|store| store.complete.load(Ordering::Relaxed)) {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the snapshot loads");
        // Three pages; a quiet master is read after the subscribe grace.
        assert_eq!(requests.load(Ordering::Relaxed), 3);
        assert_eq!(hits(&cache, 6).await[&1], 5);

        // Batch 1 continues the snapshot.
        let payload =
            |events: serde_json::Value| rmp_serde::to_vec_named(&(0_i64, events, 0_u32)).unwrap();
        send_multipart(
            &publisher,
            vec![
                vec![],
                1_u64.to_be_bytes().to_vec(),
                payload(serde_json::json!([
                    {"event_type": "stored", "group_id": group_id(106), "medium": "cpu"},
                    {"event_type": "removed", "group_id": group_id(103), "medium": "cpu"},
                ])),
            ],
        )
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while hits(&cache, 6).await[&1] != 2 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("events apply on top of the snapshot");
        assert_eq!(cache.inner.store_chunks.load(Ordering::Relaxed), 5);

        // A lost batch: the subscriber reads the snapshot again, keeping its
        // chunks meanwhile.
        send_multipart(
            &publisher,
            vec![
                vec![],
                3_u64.to_be_bytes().to_vec(),
                payload(serde_json::json!([])),
            ],
        )
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while requests.load(Ordering::Relaxed) < 6 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("a gap reads the snapshot again");
        cache.shutdown();
    }

    #[tokio::test]
    async fn a_master_without_snapshots_is_followed_by_events() {
        let (cache, a, _) = cache_with_stores();
        let (admin_url, _, _) = serve_snapshot(None, vec![101]).await;
        let error = read_snapshot(
            &cache.inner.http,
            &format!("{admin_url}/kv_events/snapshot"),
            PREFIX,
        )
        .await
        .err()
        .unwrap();
        assert!(matches!(error, SnapshotError::Unsupported(_)));
        let missing = read_snapshot(&cache.inner.http, &format!("{admin_url}/nothing"), PREFIX)
            .await
            .err()
            .unwrap();
        assert!(matches!(missing, SnapshotError::Unsupported(_)));
        // Without a snapshot ever loaded, a gap forgets as before.
        cache.apply(&a, 1, vec![event("stored", 101, "cpu")]);
        assert_eq!(cache.apply(&a, 3, vec![]), Applied::Gap);
        assert!(a.state.read().chunks.is_empty());
    }

    /// Against a live master, while writers change it:
    /// `MOONCAKE_E2E_EVENTS=tcp://HOST:5558 MOONCAKE_E2E_ADMIN=http://HOST:9003
    /// MOONCAKE_E2E_PREFIX=vllm-mooncake-store:MODEL@ MOONCAKE_E2E_SECONDS=N cargo test -p dynamo-llm
    /// --lib follows_a_live_master -- --ignored --nocapture`. A resync halfway finds
    /// nothing the events missed. After N seconds, when the writers have stopped, the
    /// index must equal a fresh snapshot, and a resync must undo damage to it.
    #[tokio::test]
    #[ignore]
    async fn follows_a_live_master() {
        let env = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"));
        let announcement = StoreAnnouncement {
            kv_events_endpoint: env("MOONCAKE_E2E_EVENTS"),
            group_prefix: env("MOONCAKE_E2E_PREFIX"),
            admin_url: Some(env("MOONCAKE_E2E_ADMIN")),
        };
        let follow_for = Duration::from_secs(env("MOONCAKE_E2E_SECONDS").parse().unwrap());
        let cache =
            MooncakeStoreSharedKvCache::with_default_scheme(None, CancellationToken::new(), None);
        let store = Arc::new(Store::new(announcement.clone()));
        cache
            .inner
            .workers
            .store(Arc::new(HashMap::from([(1, store.clone())])));
        let token = CancellationToken::new();
        tokio::spawn(cache.clone().follow(store.clone(), token.clone()));

        let started = Instant::now();
        tokio::time::timeout(Duration::from_secs(120), async {
            while !store.complete.load(Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the snapshot loads");
        println!(
            "loaded in {:?}: {} chunks",
            started.elapsed(),
            store.state.read().chunks.len()
        );
        tokio::time::sleep(follow_for / 2).await;
        let resync = |what: &'static str| {
            let cache = cache.clone();
            async move {
                let started = Instant::now();
                let replies = cache.resync(Some(Duration::from_secs(120))).await;
                let installed = replies[0].last_load.as_ref().unwrap().installed.unwrap();
                println!(
                    "{what}: {:?} in {:?}, {installed:?}",
                    replies[0].resync,
                    started.elapsed()
                );
                assert_eq!(replies[0].resync, Some(ResyncResult::Installed));
                installed
            }
        };
        let installed = resync("resync while writing").await;
        assert_eq!((installed.stale, installed.missing), (Some(0), Some(0)));
        tokio::time::sleep(follow_for - follow_for / 2).await;

        let named = |state: &StoreState| -> HashMap<u64, Vec<String>> {
            state
                .chunks
                .iter()
                .map(|(&chunk, &bits)| {
                    let mut media: Vec<String> = (0..state.media.len())
                        .filter(|index| bits & (1 << index) != 0)
                        .map(|index| state.media[index].clone())
                        .collect();
                    media.sort();
                    (chunk, media)
                })
                .collect()
        };
        let fresh = read_snapshot(
            &cache.inner.http,
            &announcement.snapshot_url().unwrap(),
            &announcement.group_prefix,
        )
        .await
        .ok()
        .unwrap();
        let (index, truth) = (named(&store.state.read()), named(&fresh.state));
        let differ = index
            .keys()
            .chain(truth.keys())
            .filter(|chunk| index.get(chunk) != truth.get(chunk))
            .count();
        println!(
            "after {follow_for:?}: index {} chunks, snapshot {} chunks, {differ} differ, sequence {:?}, next batch {}",
            index.len(),
            truth.len(),
            store.state.read().sequence,
            fresh.next_sequence
        );
        assert!(
            store.complete.load(Ordering::Relaxed),
            "a gap left the index incomplete"
        );
        assert_eq!(differ, 0);

        // Damage the index as lost events would: a resync restores it.
        let removed = {
            let mut state = store.state.write();
            let real: Vec<u64> = state.chunks.keys().take(100).copied().collect();
            for chunk in &real {
                state.chunks.remove(chunk);
            }
            for fake in 0..1000 {
                state.chunks.insert(0xdead_0000_0000_0000 | fake, 1);
            }
            real.len()
        };
        cache.add_chunks(1000 - removed as isize);
        let installed = resync("resync after damage").await;
        assert_eq!(
            (installed.stale, installed.missing),
            (Some(1000), Some(removed))
        );
        assert_eq!(named(&store.state.read()), truth);
        token.cancel();
    }
}
