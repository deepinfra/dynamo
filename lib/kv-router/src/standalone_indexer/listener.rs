// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::indexer::WorkerKvQueryResponse;
use crate::protocols::{KvCacheEventData, RouterEvent, WorkerId, WorkerWithDpRank};
use crate::recovery::{CursorObservation, CursorState};
use crate::zmq_wire::{ZmqEventNormalizer, decode_event_batch};

use super::evictions::PendingEvictions;
use super::indexer::Indexer;
use super::registry::ListenerRecord;
use super::zmq::{MultipartMessage, SharedSocket, connect_sub_socket, recv_multipart};

const WATERMARK_UNSET: u64 = u64::MAX;

fn cursor_from_watermark(watermark: u64) -> CursorState {
    if watermark == WATERMARK_UNSET {
        CursorState::Initial
    } else {
        CursorState::Live(watermark)
    }
}

/// Emit a `kv_audit` line for a single KV event ingested from the engine,
/// before the `keep_evictions` measurement filter is applied, so the log
/// reflects what the engine actually published. `source`
/// distinguishes the live SUB stream (`"live"`) from events pulled in by
/// gap recovery (`"recover"`). No-op unless audit logging is enabled.
fn audit_log_event(router_event: &RouterEvent, seq: u64, source: &'static str) {
    if !super::logging_enabled() {
        return;
    }
    let ts_ms = super::now_unix_millis();
    let worker_id = router_event.worker_id;
    let dp_rank = router_event.event.dp_rank;
    let storage_tier = router_event.storage_tier;
    let event_id = router_event.event.event_id;
    match &router_event.event.data {
        KvCacheEventData::Stored(data) => {
            // `tokens_hash` is the local block hash (matches what `/query`
            // computes from token_ids); `block_hash` is the sequence hash that
            // chains blocks together. Log both so queries and events correlate.
            let token_block_hashes: Vec<u64> =
                data.blocks.iter().map(|b| b.tokens_hash.0).collect();
            let sequence_block_hashes: Vec<u64> =
                data.blocks.iter().map(|b| b.block_hash.0).collect();
            let parent_hash = data.parent_hash.map(|h| h.0);
            tracing::info!(
                target: "kv_audit",
                kind = "STORE",
                ts_ms,
                source,
                seq,
                worker_id,
                dp_rank,
                storage_tier = ?storage_tier,
                event_id,
                parent_hash = ?parent_hash,
                num_blocks = token_block_hashes.len(),
                token_block_hashes = ?token_block_hashes,
                sequence_block_hashes = ?sequence_block_hashes,
                "kv_audit STORE"
            );
        }
        KvCacheEventData::Removed(data) => {
            // Removal events carry only sequence (external) block hashes.
            let sequence_block_hashes: Vec<u64> = data.block_hashes.iter().map(|h| h.0).collect();
            tracing::info!(
                target: "kv_audit",
                kind = "EVICT",
                ts_ms,
                source,
                seq,
                worker_id,
                dp_rank,
                storage_tier = ?storage_tier,
                event_id,
                num_blocks = sequence_block_hashes.len(),
                sequence_block_hashes = ?sequence_block_hashes,
                "kv_audit EVICT"
            );
        }
        KvCacheEventData::Cleared => {
            tracing::info!(
                target: "kv_audit",
                kind = "CLEAR",
                ts_ms,
                source,
                seq,
                worker_id,
                dp_rank,
                storage_tier = ?storage_tier,
                event_id,
                "kv_audit CLEAR"
            );
        }
    }
}

struct ListenerLoop {
    worker_id: WorkerId,
    dp_rank: u32,
    indexer: Indexer,
    cancel: CancellationToken,
    live_socket: SharedSocket,
    /// Base URL of the worker's `GET /kv_recover` endpoint, queried on a
    /// detected gap. `None` when no recovery endpoint was configured for this
    /// worker, in which case gaps are logged and the dropped batches are lost.
    recover_endpoint: Option<String>,
    http_client: reqwest::Client,
    watermark: Arc<AtomicU64>,
    /// Shared with the listener's registry record; the keep-evictions sweep
    /// drains it through there. Untouched unless `--keep-evictions` is set.
    pending_evictions: Arc<Mutex<PendingEvictions>>,
    normalizer: ZmqEventNormalizer,
    messages_processed: u64,
}

impl ListenerLoop {
    #[expect(clippy::too_many_arguments)]
    fn new(
        worker_id: WorkerId,
        dp_rank: u32,
        block_size: u32,
        indexer: Indexer,
        cancel: CancellationToken,
        live_socket: SharedSocket,
        recover_endpoint: Option<String>,
        http_client: reqwest::Client,
        watermark: Arc<AtomicU64>,
        pending_evictions: Arc<Mutex<PendingEvictions>>,
    ) -> Self {
        Self {
            worker_id,
            dp_rank,
            indexer,
            cancel,
            live_socket,
            recover_endpoint,
            http_client,
            watermark,
            pending_evictions,
            normalizer: ZmqEventNormalizer::new(block_size),
            messages_processed: 0,
        }
    }

    /// `--keep-evictions` interception, applied to every event before it can
    /// reach the tree. STOREs cancel any pending eviction of their blocks and
    /// then apply as usual; EVICTs are parked in the buffer instead of
    /// applied; CLEARs are dropped (matching the old `--ignore-evictions`
    /// behavior). Returns `true` when the event must NOT be applied.
    fn keep_evictions_intercept(&self, event: &RouterEvent) -> bool {
        if !super::keep_evictions() {
            return false;
        }
        match &event.event.data {
            KvCacheEventData::Stored(data) => {
                self.pending_evictions
                    .lock()
                    .cancel(data.blocks.iter().map(|b| b.block_hash.0));
                false
            }
            KvCacheEventData::Removed(data) => {
                self.pending_evictions
                    .lock()
                    .buffer(&data.block_hashes, event.storage_tier);
                true
            }
            KvCacheEventData::Cleared => true,
        }
    }

    fn cursor(&self) -> CursorState {
        cursor_from_watermark(self.watermark.load(Ordering::Acquire))
    }

    /// Recover the events this consumer is missing for its `(worker_id,
    /// dp_rank)` by querying the worker's `GET /kv_recover` endpoint and
    /// applying the returned [`WorkerKvQueryResponse`] per the standalone
    /// recovery contract. Returns the number of events applied.
    ///
    /// `start_seq` is the first `event_id`/seq the consumer is missing;
    /// `end_seq` is the seq just observed live (the upper bound of the gap).
    /// `end_seq` is sent only so the worker can classify `TooNew`/
    /// `InvalidRange` — the authoritative resume point is always the response's
    /// `last_event_id`, never a count of events applied (recovery is
    /// at-least-once).
    async fn recover_gap(&mut self, start_seq: u64, end_seq: u64) -> u64 {
        tracing::info!(
            self.worker_id,
            self.dp_rank,
            start_seq,
            end_seq,
            "Requesting recovery from worker via /kv_recover"
        );

        let Some(recover_endpoint) = self.recover_endpoint.as_deref() else {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                gap_size = end_seq.saturating_sub(start_seq),
                "No recover endpoint configured; batches lost"
            );
            return 0;
        };

        let url = format!("{}/kv_recover", recover_endpoint.trim_end_matches('/'));
        let client = self.http_client.clone();
        let cancel = self.cancel.clone();
        let worker_id = self.worker_id;
        let dp_rank = self.dp_rank;

        let fetch = async move {
            let response = client
                .get(&url)
                .query(&[("start", start_seq), ("end", end_seq)])
                .send()
                .await?;
            if !response.status().is_success() {
                anyhow::bail!("kv_recover returned status {}", response.status());
            }
            let body: WorkerKvQueryResponse = response.json().await?;
            anyhow::Ok(body)
        };

        let response = tokio::select! {
            _ = cancel.cancelled() => {
                tracing::debug!(worker_id, dp_rank, "Recovery cancelled");
                return 0;
            }
            result = fetch => match result {
                Ok(body) => body,
                Err(error) => {
                    tracing::error!(worker_id, dp_rank, error = %error, "kv_recover request failed");
                    return 0;
                }
            }
        };

        self.apply_recover_response(response).await
    }

    /// Apply a [`WorkerKvQueryResponse`] to this listener's indexer and advance
    /// the watermark. See the recovery contract for per-variant semantics.
    async fn apply_recover_response(&mut self, response: WorkerKvQueryResponse) -> u64 {
        match response {
            WorkerKvQueryResponse::Events {
                events,
                last_event_id,
            } => {
                let applied = self.apply_recovered_events(events).await;
                // At-least-once: the watermark is driven by `last_event_id`, not
                // by counting events, so a few re-applied live batches stay safe.
                self.watermark.store(last_event_id, Ordering::Release);
                tracing::info!(
                    self.worker_id,
                    self.dp_rank,
                    applied,
                    last_event_id,
                    "Recovery complete (Events)"
                );
                applied
            }
            WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
            } => {
                // Drop stale state for just this logical KV unit before applying
                // the full snapshot. `remove_worker_dp_rank` (not
                // `remove_worker`) so a sibling dp_rank's state is left intact.
                // The dump carries synthetic 0-based event ids; the real resume
                // point is `last_event_id`.
                // Pending evictions refer to the state being replaced, so drop
                // them too — replaying them against the snapshot could remove
                // blocks the dump says are live.
                self.pending_evictions.lock().clear();
                self.indexer
                    .remove_worker_dp_rank(self.worker_id, self.dp_rank)
                    .await;
                let applied = self.apply_recovered_events(events).await;
                self.watermark.store(last_event_id, Ordering::Release);
                tracing::info!(
                    self.worker_id,
                    self.dp_rank,
                    applied,
                    last_event_id,
                    "Recovery complete (TreeDump)"
                );
                applied
            }
            WorkerKvQueryResponse::TooNew {
                requested_start,
                requested_end,
                newest_available,
            } => {
                // Consumer is ahead of the worker's buffer tail; nothing to do.
                tracing::debug!(
                    self.worker_id,
                    self.dp_rank,
                    ?requested_start,
                    ?requested_end,
                    newest_available,
                    "kv_recover returned TooNew; consumer is ahead, no-op"
                );
                0
            }
            WorkerKvQueryResponse::InvalidRange { start_id, end_id } => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    start_id,
                    end_id,
                    "kv_recover returned InvalidRange; no-op"
                );
                0
            }
            WorkerKvQueryResponse::Error(message) => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    %message,
                    "kv_recover returned Error; no-op"
                );
                0
            }
        }
    }

    /// Apply recovered [`RouterEvent`]s under this listener's
    /// consumer-assigned `(worker_id, dp_rank)`. The worker's self-reported
    /// `worker_id`/`dp_rank` are informational (the contract says to apply
    /// events under the worker that was queried), so they are rewritten before
    /// applying. Returns the count actually applied (after the
    /// `keep_evictions` measurement filter).
    async fn apply_recovered_events(&self, events: Vec<RouterEvent>) -> u64 {
        let mut applied = 0;
        for mut event in events {
            event.worker_id = self.worker_id;
            event.event.dp_rank = self.dp_rank;

            // Audit-log the recovered event before the measurement filter.
            audit_log_event(&event, event.event.event_id, "recover");

            // Feed-layer measurement filter (same as apply_live_batch).
            if self.keep_evictions_intercept(&event) {
                continue;
            }
            self.indexer.apply_event_routed(event).await;
            applied += 1;
        }
        applied
    }

    async fn handle_gap(&mut self, seq: u64) {
        match self.cursor().observe(seq) {
            CursorObservation::Initial { got } if got > 0 => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    expected = 0,
                    got,
                    "Gap detected: expected seq 0, got {got}"
                );
                self.recover_gap(0, got).await;
            }
            CursorObservation::Gap { expected, got } => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    expected,
                    got,
                    "Gap detected: expected seq {expected}, got {got}"
                );
                self.recover_gap(expected, got).await;
            }
            CursorObservation::Initial { .. }
            | CursorObservation::Contiguous { .. }
            | CursorObservation::Stale { .. }
            | CursorObservation::FreshAfterBarrier { .. } => {}
        }
    }

    async fn apply_live_batch(&mut self, seq: u64, payload: &[u8]) {
        let batch = match decode_event_batch(payload) {
            Ok(batch) => batch,
            Err(error) => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    "Failed to decode KvEventBatch: {error}"
                );
                return;
            }
        };

        let effective_dp_rank = batch
            .data_parallel_rank
            .map_or(self.dp_rank, |rank| rank.cast_unsigned());
        for raw_event in batch.events {
            let Some(placement_event) = self.normalizer.normalize(
                raw_event,
                seq,
                WorkerWithDpRank::new(self.worker_id, effective_dp_rank),
            ) else {
                continue;
            };
            let router_event = placement_event
                .into_router_event()
                .expect("local worker placement must convert to router event");
            // Audit-log the event as published by the engine, before the
            // measurement filter below can drop it.
            audit_log_event(&router_event, seq, "live");
            // Feed-layer measurement filter.
            if self.keep_evictions_intercept(&router_event) {
                continue;
            }
            self.indexer.apply_event_routed(router_event).await;
            self.messages_processed += 1;
        }
        self.watermark.store(seq, Ordering::Release);
    }

    async fn handle_message(&mut self, msg: MultipartMessage) {
        if msg.len() != 3 {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                "Unexpected ZMQ frame count: {}",
                msg.len()
            );
            return;
        }

        let seq_bytes = msg.get(1).expect("frame count checked above");
        if seq_bytes.len() != 8 {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                "Invalid sequence number length: {}",
                seq_bytes.len()
            );
            return;
        }

        let seq = u64::from_be_bytes(seq_bytes[..8].try_into().expect("length checked above"));
        self.handle_gap(seq).await;

        if matches!(self.cursor().observe(seq), CursorObservation::Stale { .. }) {
            return;
        }

        let payload = msg.get(2).expect("frame count checked above");
        self.apply_live_batch(seq, payload).await;
    }

    async fn run(mut self) -> Result<(), String> {
        loop {
            let msg = tokio::select! {
                biased;

                _ = self.cancel.cancelled() => {
                    tracing::info!(
                        self.worker_id,
                        self.dp_rank,
                        self.messages_processed,
                        "ZMQ listener exiting after cancellation"
                    );
                    return Ok(());
                }

                result = recv_multipart(&self.live_socket) => {
                    match result {
                        Ok(msg) => msg,
                        Err(error) => {
                            return Err(format!(
                                "ZMQ recv failed for worker {} dp_rank {}: {error}",
                                self.worker_id,
                                self.dp_rank,
                            ));
                        }
                    }
                }
            };

            self.handle_message(msg).await;
        }
    }
}

pub fn spawn_zmq_listener(
    worker_id: WorkerId,
    dp_rank: u32,
    record: Arc<ListenerRecord>,
    ready: watch::Receiver<bool>,
    generation: u64,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        if let Err(error) = run_listener(
            worker_id,
            dp_rank,
            record.clone(),
            ready,
            generation,
            cancel,
        )
        .await
        {
            tracing::error!(worker_id, dp_rank, error = %error, "ZMQ listener failed");
            record.try_mark_failed(generation, error);
        }
    });
}

async fn run_listener(
    worker_id: WorkerId,
    dp_rank: u32,
    record: Arc<ListenerRecord>,
    mut ready: watch::Receiver<bool>,
    generation: u64,
    cancel: CancellationToken,
) -> Result<(), String> {
    let endpoint = record.endpoint().to_string();
    let recover_endpoint = record.recover_endpoint().map(str::to_string);
    let block_size = record.block_size();
    let indexer = record.indexer();
    let watermark = record.watermark();
    let pending_evictions = record.pending_evictions();

    tracing::info!(worker_id, dp_rank, endpoint, "ZMQ listener starting");

    if cancel.is_cancelled() {
        return Ok(());
    }

    let socket = connect_sub_socket(&endpoint)
        .map_err(|e| format!("failed to connect ZMQ SUB socket to {endpoint}: {e}"))?;

    tokio::select! {
        _ = cancel.cancelled() => return Ok(()),
        result = ready.wait_for(|&value| value) => {
            result.map_err(|_| "ready channel closed before signaling".to_string())?;
        }
    }

    if !record.try_mark_active(generation) {
        tracing::debug!(
            worker_id,
            dp_rank,
            "Listener attempt is stale after readiness gate; exiting"
        );
        return Ok(());
    }

    tracing::info!(worker_id, dp_rank, "ZMQ listener ready, starting recv loop");

    let http_client = build_recover_client();
    if cancel.is_cancelled() || !record.is_current_attempt(generation) {
        return Ok(());
    }

    ListenerLoop::new(
        worker_id,
        dp_rank,
        block_size,
        indexer,
        cancel,
        socket,
        recover_endpoint,
        http_client,
        watermark,
        pending_evictions,
    )
    .run()
    .await
}

/// Build the HTTP client used for `/kv_recover` gap-recovery requests. The 10s
/// timeout bounds the whole request (connect + body read). Recovery is a
/// low-frequency, on-gap operation, so a fresh client per listener is fine.
fn build_recover_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_else(|error| {
            tracing::warn!(error = %error, "Failed to build recover HTTP client; using default");
            reqwest::Client::new()
        })
}

#[cfg(test)]
mod tests {
    use super::{ListenerLoop, WATERMARK_UNSET, cursor_from_watermark};
    use crate::indexer::WorkerKvQueryResponse;
    use crate::protocols::{LocalBlockHash, StorageTier, WorkerId, WorkerWithDpRank};
    use crate::recovery::CursorObservation;
    use crate::standalone_indexer::indexer::test_util::store_event;
    use crate::standalone_indexer::indexer::{Indexer, create_indexer};
    use crate::standalone_indexer::zmq::{
        SharedSocket, bind_pub_socket, connect_sub_socket, recv_multipart, send_multipart,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tokio_util::sync::CancellationToken;

    /// Build a `ListenerLoop` wired to a real (peerless) SUB socket so the
    /// HTTP-recovery apply path can be exercised without a publisher. Returns
    /// clones of the indexer and watermark for assertions. `recover_endpoint`
    /// is `None` because these tests drive `apply_recover_response` directly
    /// rather than the HTTP fetch.
    fn new_listener_loop(
        worker_id: WorkerId,
        dp_rank: u32,
        block_size: u32,
    ) -> (ListenerLoop, Indexer, Arc<AtomicU64>) {
        // connect() is lazy in ZMQ, so no peer needs to exist on this port.
        let endpoint = {
            let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("bind probe listener");
            let port = probe.local_addr().expect("probe local_addr").port();
            drop(probe);
            format!("tcp://127.0.0.1:{port}")
        };
        let live_socket = connect_sub_socket(&endpoint).expect("connect SUB socket");
        let indexer = create_indexer(block_size, 1);
        let watermark = Arc::new(AtomicU64::new(WATERMARK_UNSET));
        let listener = ListenerLoop::new(
            worker_id,
            dp_rank,
            block_size,
            indexer.clone(),
            CancellationToken::new(),
            live_socket,
            None,
            reqwest::Client::new(),
            watermark.clone(),
            std::sync::Arc::new(parking_lot::Mutex::new(
                crate::standalone_indexer::evictions::PendingEvictions::default(),
            )),
        );
        (listener, indexer, watermark)
    }

    /// Device-tier block count matched for `(worker_id, dp_rank)` over the
    /// given local-hash prefix. `dump_events` first acts as the FIFO barrier
    /// that flushes in-flight applies before the query lands.
    async fn matched_blocks(
        indexer: &Indexer,
        worker_id: WorkerId,
        dp_rank: u32,
        hashes: &[u64],
    ) -> u32 {
        let _ = indexer.dump_events().await;
        let sequence: Vec<LocalBlockHash> = hashes.iter().copied().map(LocalBlockHash).collect();
        let scores = indexer.find_matches(sequence).await.expect("find_matches");
        scores
            .scores
            .get(&WorkerWithDpRank::new(worker_id, dp_rank))
            .copied()
            .unwrap_or(0)
    }

    /// `Events` recovery applies every event under the consumer-assigned
    /// `(worker_id, dp_rank)` even though the worker self-reports a different
    /// identity, and advances the watermark to `last_event_id` (not a count).
    #[tokio::test]
    async fn recover_events_applies_and_rewrites_identity() {
        let (mut listener, indexer, watermark) = new_listener_loop(7, 0, 4);

        // Worker self-reports id 999 / dp_rank 3; the consumer must apply these
        // under (7, 0). Two chained device blocks: 11 then 12.
        let response = WorkerKvQueryResponse::Events {
            events: vec![
                store_event(999, 3, 0, &[], &[11], StorageTier::Device),
                store_event(999, 3, 1, &[11], &[12], StorageTier::Device),
            ],
            last_event_id: 5,
        };
        let applied = listener.apply_recover_response(response).await;
        assert_eq!(applied, 2, "both recovered events should be applied");

        assert_eq!(
            watermark.load(Ordering::Acquire),
            5,
            "watermark must track last_event_id, not the count of events"
        );
        assert_eq!(
            matched_blocks(&indexer, 7, 0, &[11, 12]).await,
            2,
            "recovered blocks must be queryable under the consumer-assigned (7, 0)"
        );
        assert_eq!(
            matched_blocks(&indexer, 999, 3, &[11, 12]).await,
            0,
            "the worker's self-reported (999, 3) identity must not be used"
        );
    }

    /// `TreeDump` recovery drops only this listener's `(worker_id, dp_rank)`
    /// state before applying the snapshot — a sibling dp_rank is left intact —
    /// and resumes from `last_event_id` rather than the synthetic dump ids.
    #[tokio::test]
    async fn recover_tree_dump_drops_target_dp_rank_only() {
        let (mut listener, indexer, watermark) = new_listener_loop(7, 0, 4);

        // Seed stale state: (7, 0) holds block 11; sibling (7, 1) holds block 99.
        indexer
            .apply_event_routed(store_event(7, 0, 0, &[], &[11], StorageTier::Device))
            .await;
        indexer
            .apply_event_routed(store_event(7, 1, 0, &[], &[99], StorageTier::Device))
            .await;
        let _ = indexer.dump_events().await;

        // Full snapshot for (7, 0): synthetic 0-based ids, foreign identity,
        // a single block 22. Real resume point is last_event_id = 10.
        let response = WorkerKvQueryResponse::TreeDump {
            events: vec![store_event(123, 9, 0, &[], &[22], StorageTier::Device)],
            last_event_id: 10,
        };
        let applied = listener.apply_recover_response(response).await;
        assert_eq!(applied, 1);

        assert_eq!(watermark.load(Ordering::Acquire), 10);
        assert_eq!(
            matched_blocks(&indexer, 7, 0, &[11]).await,
            0,
            "stale block 11 for (7, 0) must be dropped by remove_worker_dp_rank"
        );
        assert_eq!(
            matched_blocks(&indexer, 7, 0, &[22]).await,
            1,
            "the dumped block 22 must be applied under (7, 0)"
        );
        assert_eq!(
            matched_blocks(&indexer, 7, 1, &[99]).await,
            1,
            "sibling dp_rank (7, 1) must be untouched by the (7, 0) recovery"
        );
    }

    /// `TooNew` means the consumer is ahead of the worker's buffer tail: it is
    /// a no-op that leaves the watermark and indexer state unchanged.
    #[tokio::test]
    async fn recover_too_new_is_noop() {
        let (mut listener, indexer, watermark) = new_listener_loop(7, 0, 4);
        indexer
            .apply_event_routed(store_event(7, 0, 0, &[], &[11], StorageTier::Device))
            .await;
        let _ = indexer.dump_events().await;
        watermark.store(3, Ordering::Release);

        let applied = listener
            .apply_recover_response(WorkerKvQueryResponse::TooNew {
                requested_start: Some(4),
                requested_end: Some(9),
                newest_available: 3,
            })
            .await;

        assert_eq!(applied, 0, "TooNew applies nothing");
        assert_eq!(
            watermark.load(Ordering::Acquire),
            3,
            "TooNew must not move the watermark"
        );
        assert_eq!(
            matched_blocks(&indexer, 7, 0, &[11]).await,
            1,
            "TooNew must not disturb existing state"
        );
    }

    #[test]
    fn initial_gap_replays_from_zero_and_replayed_seq_becomes_stale() {
        let replay_start = match cursor_from_watermark(WATERMARK_UNSET).observe(5) {
            CursorObservation::Initial { got } if got > 0 => Some(0),
            CursorObservation::Gap { expected, .. } => Some(expected),
            _ => None,
        };
        assert_eq!(replay_start, Some(0));
        assert!(matches!(
            cursor_from_watermark(5).observe(5),
            CursorObservation::Stale {
                got: 5,
                last_applied: Some(5),
            }
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn zmq_buffers_messages_during_brief_delay() {
        let reserved_listener = reserve_open_port();
        let endpoint = format!(
            "tcp://127.0.0.1:{}",
            reserved_listener
                .local_addr()
                .expect("failed to read reserved listener address")
                .port()
        );
        drop(reserved_listener);
        let pub_socket = bind_pub_socket(&endpoint).unwrap();
        let mut sub_socket = connect_sub_socket(&endpoint).unwrap();

        let (tx, mut rx) = tokio::sync::mpsc::channel::<SharedSocket>(1);
        tokio::spawn(async move {
            let _ = recv_multipart(&sub_socket).await.unwrap();
            let _ = tx.send(sub_socket).await;
        });
        loop {
            send_multipart(&pub_socket, vec![b"probe".to_vec()])
                .await
                .unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            if let Ok(sub) = rx.try_recv() {
                sub_socket = sub;
                break;
            }
        }

        let num_messages = 10u64;

        for i in 0..num_messages {
            send_multipart(&pub_socket, vec![i.to_le_bytes().to_vec()])
                .await
                .unwrap();
        }

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        for i in 0u64..num_messages {
            let msg = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                recv_multipart(&sub_socket),
            )
            .await
            .expect("timed out waiting for ZMQ message")
            .unwrap();

            let payload = msg.first().unwrap();
            let received = u64::from_le_bytes(payload[..8].try_into().unwrap());
            assert_eq!(received, i, "message {i} arrived out of order");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn zmq_subscriber_connects_before_publisher_bind() {
        let reserved_listener = reserve_open_port();
        let endpoint = format!(
            "tcp://127.0.0.1:{}",
            reserved_listener
                .local_addr()
                .expect("failed to read reserved listener address")
                .port()
        );
        drop(reserved_listener);
        let sub_socket = connect_sub_socket(&endpoint).unwrap();

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let pub_socket = bind_pub_socket(&endpoint).unwrap();
        for _ in 0..5 {
            send_multipart(&pub_socket, vec![b"probe".to_vec()])
                .await
                .unwrap();
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        let msg = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            recv_multipart(&sub_socket),
        )
        .await
        .expect("timed out waiting for ZMQ message")
        .unwrap();

        assert_eq!(msg, vec![b"probe".to_vec()]);
    }

    fn reserve_open_port() -> std::net::TcpListener {
        std::net::TcpListener::bind("127.0.0.1:0").expect("failed to bind probe listener")
    }
}
