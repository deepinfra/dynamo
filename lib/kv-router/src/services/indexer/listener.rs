// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::protocols::{KvCacheEventData, RouterEvent, WorkerId, WorkerWithDpRank};
use crate::recovery::{CursorObservation, CursorState};
use crate::zmq_wire::{ZmqEventNormalizer, decode_event_batch};

use super::audit;
use super::backend::Indexer;
use super::block_size::BlockSizeGuard;
use super::kv_recover::plan_recovery;
use super::registry::{ListenerRecord, RecoverTarget, SharedPendingEvictions};
use crate::services::common::zmq::{
    MultipartMessage, SharedSocket, connect_dealer_socket, connect_sub_socket, recv_multipart,
    send_multipart,
};

const WATERMARK_UNSET: u64 = u64::MAX;

fn cursor_from_watermark(watermark: u64) -> CursorState {
    if watermark == WATERMARK_UNSET {
        CursorState::Initial
    } else {
        CursorState::Live(watermark)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReplayRecoveryFailureReason {
    Empty,
    StartedAfterRequested,
    NonContiguous,
    EndedBeforeTarget,
}

impl ReplayRecoveryFailureReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::StartedAfterRequested => "started_after_requested",
            Self::NonContiguous => "non_contiguous",
            Self::EndedBeforeTarget => "ended_before_target",
        }
    }
}

struct ReplayRecoveryProgress {
    start_seq: u64,
    end_seq: u64,
    first_replayed: Option<u64>,
    last_replayed: Option<u64>,
    expected_next: u64,
    replayed: u64,
    non_contiguous: bool,
}

impl ReplayRecoveryProgress {
    fn new(start_seq: u64, end_seq: u64) -> Self {
        Self {
            start_seq,
            end_seq,
            first_replayed: None,
            last_replayed: None,
            expected_next: start_seq,
            replayed: 0,
            non_contiguous: false,
        }
    }

    fn record_batch(&mut self, seq: u64) {
        if self.first_replayed.is_none() {
            self.first_replayed = Some(seq);
        }
        if seq != self.expected_next {
            self.non_contiguous = true;
        }
        self.expected_next = seq.saturating_add(1);
        self.last_replayed = Some(seq);
        self.replayed += 1;
    }

    fn failure_reason(&self) -> Option<ReplayRecoveryFailureReason> {
        if self.start_seq >= self.end_seq {
            return None;
        }
        if self.replayed == 0 {
            return Some(ReplayRecoveryFailureReason::Empty);
        }
        if self
            .first_replayed
            .is_some_and(|first| first > self.start_seq)
        {
            return Some(ReplayRecoveryFailureReason::StartedAfterRequested);
        }
        if self.non_contiguous {
            return Some(ReplayRecoveryFailureReason::NonContiguous);
        }
        if self
            .last_replayed
            .is_some_and(|last| last < self.end_seq - 1)
        {
            return Some(ReplayRecoveryFailureReason::EndedBeforeTarget);
        }
        None
    }

    fn replayed(&self) -> u64 {
        self.replayed
    }

    fn warn_if_incomplete(&self, worker_id: WorkerId, dp_rank: u32) {
        let Some(reason) = self.failure_reason() else {
            return;
        };

        let reason = reason.as_str();
        let start_seq = self.start_seq;
        let end_seq = self.end_seq;
        let replayed = self.replayed;
        let first_replayed_display = self
            .first_replayed
            .map(|seq| seq.to_string())
            .unwrap_or_else(|| "none".to_string());
        let last_replayed_display = self
            .last_replayed
            .map(|seq| seq.to_string())
            .unwrap_or_else(|| "none".to_string());
        tracing::warn!(
            worker_id,
            dp_rank,
            requested_start = start_seq,
            requested_end = end_seq,
            first_replayed = ?self.first_replayed,
            last_replayed = ?self.last_replayed,
            replayed,
            reason,
            "Replay incomplete: requested=[{start_seq},{end_seq}), first={}, last={}, replayed={replayed}, reason={reason}",
            first_replayed_display,
            last_replayed_display,
        );
    }
}

struct ListenerLoop {
    worker_id: WorkerId,
    dp_rank: u32,
    indexer: Indexer,
    cancel: CancellationToken,
    live_socket: SharedSocket,
    replay_socket: Option<SharedSocket>,
    recover: Option<RecoverTarget>,
    pending_evictions: Option<SharedPendingEvictions>,
    audit_log: bool,
    watermark: Arc<AtomicU64>,
    normalizer: ZmqEventNormalizer,
    block_size_guard: BlockSizeGuard,
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
        replay_socket: Option<SharedSocket>,
        recover: Option<RecoverTarget>,
        pending_evictions: Option<SharedPendingEvictions>,
        audit_log: bool,
        watermark: Arc<AtomicU64>,
    ) -> Self {
        Self {
            worker_id,
            dp_rank,
            indexer,
            cancel,
            live_socket,
            replay_socket,
            recover,
            pending_evictions,
            audit_log,
            watermark,
            normalizer: ZmqEventNormalizer::new(block_size),
            block_size_guard: BlockSizeGuard::new(block_size),
            messages_processed: 0,
        }
    }

    /// Audit-log `event` (as published) and apply the `--keep-evictions`
    /// filter. `true` when the event must not reach the tree.
    fn withhold_from_tree(&self, event: &RouterEvent, seq: u64, source: &'static str) -> bool {
        if self.audit_log {
            audit::log_event(event, seq, source);
        }
        self.park_kept_eviction(event)
    }

    /// `--keep-evictions`: a store cancels any parked eviction of its blocks
    /// and applies; an eviction is parked instead of applied; a clear is
    /// dropped. `true` when the event must not reach the tree.
    fn park_kept_eviction(&self, event: &RouterEvent) -> bool {
        let Some(pending) = self.pending_evictions.as_ref() else {
            return false;
        };
        match &event.event.data {
            KvCacheEventData::Stored(data) => {
                pending
                    .lock()
                    .cancel(data.blocks.iter().map(|b| b.block_hash.0));
                false
            }
            KvCacheEventData::Removed(data) => {
                pending
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

    async fn replay_gap(&mut self, start_seq: u64, end_seq: u64) -> Result<u64, String> {
        tracing::info!(
            self.worker_id,
            self.dp_rank,
            start_seq,
            end_seq,
            "Requesting replay from engine"
        );

        let Some(replay_socket) = self.replay_socket.as_ref() else {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                gap_size = end_seq.saturating_sub(start_seq),
                "No replay endpoint configured; batches lost"
            );
            return Ok(0);
        };

        let worker_id = self.worker_id;
        let dp_rank = self.dp_rank;
        let indexer = &self.indexer;
        let watermark = &self.watermark;

        let req_frames = vec![Vec::new(), start_seq.to_be_bytes().to_vec()];
        if let Err(error) = send_multipart(replay_socket, req_frames).await {
            tracing::error!(worker_id, dp_rank, error = %error, "Failed to send replay request");
            return Ok(0);
        }

        let mut replay_progress = ReplayRecoveryProgress::new(start_seq, end_seq);
        loop {
            let msg = tokio::select! {
                _ = self.cancel.cancelled() => break,
                result = recv_multipart(replay_socket) => {
                    match result {
                        Ok(msg) => msg,
                        Err(error) => {
                            tracing::error!(worker_id, dp_rank, error = %error, "Replay recv error");
                            break;
                        }
                    }
                }
            };
            // DEALER strips the ROUTER identity. vLLM includes the PUB topic;
            // SGLang sends only the delimiter, sequence number, and payload.
            let (seq_bytes, payload) = match msg.as_slice() {
                [_, seq, payload] | [_, _, seq, payload] => (seq, payload),
                _ => {
                    tracing::warn!(
                        worker_id,
                        dp_rank,
                        "Unexpected replay frame count: {}",
                        msg.len()
                    );
                    break;
                }
            };
            if payload.is_empty() {
                break;
            }

            if seq_bytes.len() != 8 {
                tracing::warn!(
                    worker_id,
                    dp_rank,
                    "Invalid replay seq length: {}",
                    seq_bytes.len()
                );
                break;
            }
            let seq = u64::from_be_bytes(seq_bytes[..8].try_into().expect("length checked above"));

            let Ok(batch) = decode_event_batch(payload) else {
                tracing::warn!(worker_id, dp_rank, seq, "Failed to decode replayed batch");
                continue;
            };

            let effective_dp_rank = batch
                .data_parallel_rank
                .map_or(dp_rank, |rank| rank.cast_unsigned());
            for raw_event in batch.events {
                let worker = WorkerWithDpRank::new(worker_id, effective_dp_rank);
                let Some(raw_event) = self.normalizer.preprocess(raw_event, worker) else {
                    continue;
                };
                if !self.block_size_guard.admit(&raw_event, worker) {
                    continue;
                }
                let Some(placement_event) = self
                    .normalizer
                    .normalize_preprocessed(raw_event, seq, worker)
                else {
                    continue;
                };
                let router_event = placement_event
                    .into_router_event()
                    .expect("local worker placement must convert to router event");
                if self.withhold_from_tree(&router_event, seq, "replay") {
                    continue;
                }
                indexer
                    .apply_event_routed(router_event)
                    .await
                    .map_err(|error| {
                        format!(
                            "failed to apply replayed event for worker {worker_id} dp_rank {dp_rank}: {error}"
                        )
                    })?;
            }
            watermark.store(seq, Ordering::Release);
            replay_progress.record_batch(seq);
        }

        replay_progress.warn_if_incomplete(worker_id, dp_rank);

        let replayed = replay_progress.replayed();
        tracing::info!(worker_id, dp_rank, replayed, "Replay complete");
        Ok(replayed)
    }

    /// Recover `[start_seq, end_seq)` over HTTP `/kv_recover` when the worker
    /// serves it, else over the ZMQ replay socket.
    async fn recover_gap(&mut self, start_seq: u64, end_seq: u64) -> Result<u64, String> {
        match self.recover.clone() {
            Some(target) => self.recover_over_http(&target, start_seq, end_seq).await,
            None => self.replay_gap(start_seq, end_seq).await,
        }
    }

    async fn recover_over_http(
        &mut self,
        target: &RecoverTarget,
        start_seq: u64,
        end_seq: u64,
    ) -> Result<u64, String> {
        let (worker_id, dp_rank) = (self.worker_id, self.dp_rank);
        tracing::info!(
            worker_id,
            dp_rank,
            start_seq,
            end_seq,
            "Requesting recovery from worker via /kv_recover"
        );
        let Some(response) = target
            .client
            .fetch(
                &target.endpoint,
                start_seq,
                end_seq,
                worker_id,
                dp_rank,
                &self.cancel,
            )
            .await
        else {
            return Ok(0);
        };

        let plan = plan_recovery(response, worker_id, dp_rank);
        if plan.reset_dp_rank {
            // Parked evictions describe the state being replaced; replaying
            // them onto the snapshot could remove blocks it says are live.
            if let Some(pending) = self.pending_evictions.as_ref() {
                pending.lock().clear();
            }
            self.indexer.remove_worker_dp_rank(worker_id, dp_rank).await;
        }
        let mut applied = 0u64;
        for event in plan.events {
            if self.withhold_from_tree(&event, event.event.event_id, "recover") {
                continue;
            }
            self.indexer
                .apply_event_routed(event)
                .await
                .map_err(|error| {
                    format!(
                        "failed to apply recovered event for worker {worker_id} dp_rank {dp_rank}: {error}"
                    )
                })?;
            applied += 1;
        }
        if let Some(last_event_id) = plan.resume_at {
            self.watermark.store(last_event_id, Ordering::Release);
        }
        tracing::info!(
            worker_id,
            dp_rank,
            applied,
            reset = plan.reset_dp_rank,
            resume_at = ?plan.resume_at,
            "Recovery via /kv_recover complete"
        );
        Ok(applied)
    }

    async fn handle_gap(&mut self, seq: u64) -> Result<(), String> {
        match self.cursor().observe(seq) {
            CursorObservation::Initial { got } if got > 0 => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    expected = 0,
                    got,
                    "Gap detected: expected seq 0, got {got}"
                );
                self.recover_gap(0, got).await?;
            }
            CursorObservation::Gap { expected, got } => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    expected,
                    got,
                    "Gap detected: expected seq {expected}, got {got}"
                );
                self.recover_gap(expected, got).await?;
            }
            CursorObservation::Initial { .. }
            | CursorObservation::Contiguous { .. }
            | CursorObservation::Stale { .. } => {}
        }
        Ok(())
    }

    async fn apply_live_batch(&mut self, seq: u64, payload: &[u8]) -> Result<(), String> {
        let batch = match decode_event_batch(payload) {
            Ok(batch) => batch,
            Err(error) => {
                tracing::warn!(
                    self.worker_id,
                    self.dp_rank,
                    "Failed to decode KvEventBatch: {error}"
                );
                return Ok(());
            }
        };

        let effective_dp_rank = batch
            .data_parallel_rank
            .map_or(self.dp_rank, |rank| rank.cast_unsigned());
        for raw_event in batch.events {
            let worker = WorkerWithDpRank::new(self.worker_id, effective_dp_rank);
            let Some(raw_event) = self.normalizer.preprocess(raw_event, worker) else {
                continue;
            };
            if !self.block_size_guard.admit(&raw_event, worker) {
                continue;
            }
            let Some(placement_event) = self
                .normalizer
                .normalize_preprocessed(raw_event, seq, worker)
            else {
                continue;
            };
            let router_event = placement_event
                .into_router_event()
                .expect("local worker placement must convert to router event");
            if self.withhold_from_tree(&router_event, seq, "live") {
                continue;
            }
            self.indexer
                .apply_event_routed(router_event)
                .await
                .map_err(|error| {
                    format!(
                        "failed to apply live event for worker {} dp_rank {}: {error}",
                        self.worker_id, self.dp_rank
                    )
                })?;
            self.messages_processed += 1;
        }
        self.watermark.store(seq, Ordering::Release);
        Ok(())
    }

    async fn handle_message(&mut self, msg: MultipartMessage) -> Result<(), String> {
        if msg.len() != 3 {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                "Unexpected ZMQ frame count: {}",
                msg.len()
            );
            return Ok(());
        }

        let seq_bytes = msg.get(1).expect("frame count checked above");
        if seq_bytes.len() != 8 {
            tracing::warn!(
                self.worker_id,
                self.dp_rank,
                "Invalid sequence number length: {}",
                seq_bytes.len()
            );
            return Ok(());
        }

        let seq = u64::from_be_bytes(seq_bytes[..8].try_into().expect("length checked above"));
        self.handle_gap(seq).await?;

        if matches!(self.cursor().observe(seq), CursorObservation::Stale { .. }) {
            return Ok(());
        }

        let payload = msg.get(2).expect("frame count checked above");
        self.apply_live_batch(seq, payload).await
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

            self.handle_message(msg).await?;
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
    let replay_endpoint = record.replay_endpoint().map(str::to_string);
    let block_size = record.block_size();
    let indexer = record.indexer();
    let watermark = record.watermark();

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

    let replay_socket =
        connect_replay_socket(worker_id, dp_rank, replay_endpoint.as_deref(), &cancel).await;
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
        replay_socket,
        record.recover_target(),
        record.pending_evictions(),
        record.audit_log(),
        watermark,
    )
    .run()
    .await
}

async fn connect_replay_socket(
    worker_id: WorkerId,
    dp_rank: u32,
    replay_endpoint: Option<&str>,
    cancel: &CancellationToken,
) -> Option<SharedSocket> {
    let endpoint = replay_endpoint?;

    if cancel.is_cancelled() {
        return None;
    }

    match connect_dealer_socket(endpoint) {
        Ok(socket) => {
            tracing::info!(
                worker_id,
                dp_rank,
                replay_endpoint = endpoint,
                "Replay socket connected"
            );
            Some(socket)
        }
        Err(error) => {
            tracing::error!(
                worker_id,
                dp_rank,
                error = %error,
                "Failed to connect replay socket to {endpoint}"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::compute_block_hash_for_seq;
    use crate::services::indexer::backend::create_indexer;
    use std::time::Duration;

    #[rstest::rstest]
    #[case::sglang(false)]
    #[case::vllm(true)]
    #[tokio::test]
    async fn replay_accepts_engine_framing(#[case] include_topic: bool) {
        let context = zmq::Context::new();
        let router = context.socket(zmq::ROUTER).unwrap();
        router.set_linger(0).unwrap();
        router.set_rcvtimeo(5000).unwrap();
        router.set_sndtimeo(5000).unwrap();
        router.bind("tcp://127.0.0.1:*").unwrap();
        let endpoint = router.get_last_endpoint().unwrap().unwrap();
        let replay_socket = connect_dealer_socket(&endpoint).unwrap();
        let publisher = context.socket(zmq::PUB).unwrap();
        publisher.bind("tcp://127.0.0.1:*").unwrap();
        let live_socket =
            connect_sub_socket(&publisher.get_last_endpoint().unwrap().unwrap()).unwrap();

        let server = tokio::task::spawn_blocking(move || {
            let request = router.recv_multipart(0).unwrap();
            assert_eq!(&request[1..], &[vec![], 0_u64.to_be_bytes().to_vec()]);
            for seq in 0_u64..=2 {
                let mut frames = vec![request[0].clone(), vec![]];
                if include_topic {
                    frames.push(b"kv-events".to_vec());
                }
                if seq == 2 {
                    frames.extend([u64::MAX.to_be_bytes().to_vec(), vec![]]);
                } else {
                    let event = (
                        "BlockStored",
                        vec![seq + 100],
                        if seq == 0 { None } else { Some(100_u64) },
                        vec![1_u32 + seq as u32; 4],
                        4_usize,
                        Option::<u64>::None,
                        "GPU",
                    );
                    let payload = rmp_serde::to_vec(&(0.0_f64, vec![event], Some(0_i32))).unwrap();
                    frames.extend([seq.to_be_bytes().to_vec(), payload]);
                }
                router.send_multipart(frames, 0).unwrap();
            }
            // Keep the socket open until the client consumes the queued replies.
            assert_eq!(router.recv_multipart(0).unwrap()[1], b"done");
        });
        let indexer = create_indexer(4, 1);
        let watermark = Arc::new(AtomicU64::new(WATERMARK_UNSET));
        let cancel = CancellationToken::new();
        let mut listener = ListenerLoop::new(
            1,
            0,
            4,
            indexer.clone(),
            cancel.clone(),
            live_socket,
            Some(replay_socket.clone()),
            None,
            None,
            false,
            watermark.clone(),
        );
        let replayed = tokio::time::timeout(Duration::from_secs(5), listener.replay_gap(0, 2))
            .await
            .expect("replay completion marker")
            .unwrap();
        send_multipart(&replay_socket, vec![b"done".to_vec()])
            .await
            .unwrap();
        server.await.unwrap();
        assert_eq!(replayed, 2);
        assert_eq!(watermark.load(Ordering::Acquire), 1);
        indexer.dump_events().await.expect("flush indexer");
        let hashes = compute_block_hash_for_seq(&[1, 1, 1, 1, 2, 2, 2, 2], 4, Default::default());
        let matches = indexer.find_matches(hashes).await.unwrap();
        assert_eq!(matches.scores.get(&WorkerWithDpRank::new(1, 0)), Some(&2));
        cancel.cancel();
    }
}

#[cfg(test)]
mod deepinfra_tests {
    use super::*;
    use crate::protocols::{
        BlockHashOptions, LocalBlockHash, StorageTier, compute_block_hash_for_seq,
    };
    use crate::services::indexer::backend::create_indexer;
    use crate::services::indexer::evictions::PendingEvictions;

    const BLOCK_SIZE: u32 = 4;

    fn listener(pending: Option<SharedPendingEvictions>) -> (ListenerLoop, Indexer) {
        // connect() is lazy, so no publisher is needed.
        let live_socket = connect_sub_socket("tcp://127.0.0.1:1").unwrap();
        let indexer = create_indexer(BLOCK_SIZE, 1);
        let listener = ListenerLoop::new(
            7,
            0,
            BLOCK_SIZE,
            indexer.clone(),
            CancellationToken::new(),
            live_socket,
            None,
            None,
            pending,
            false,
            Arc::new(AtomicU64::new(WATERMARK_UNSET)),
        );
        (listener, indexer)
    }

    /// A vLLM batch of `BlockStored` events, positional msgpack as on the wire.
    fn stored_batch(tokens: &[u32], block_size: usize, medium: &str) -> Vec<u8> {
        let hashes: Vec<u64> = (0..tokens.len() / block_size)
            .map(|i| 1000 + i as u64)
            .collect();
        let event = (
            "BlockStored",
            hashes,
            Option::<u64>::None,
            tokens.to_vec(),
            block_size,
            Option::<u64>::None,
            medium,
        );
        rmp_serde::to_vec(&(0.0_f64, vec![event], Some(0_i32))).unwrap()
    }

    fn removed_batch(hashes: &[u64]) -> Vec<u8> {
        let event = ("BlockRemoved", hashes.to_vec(), "GPU");
        rmp_serde::to_vec(&(0.0_f64, vec![event], Some(0_i32))).unwrap()
    }

    fn probe(tokens: &[u32]) -> Vec<LocalBlockHash> {
        compute_block_hash_for_seq(tokens, BLOCK_SIZE, BlockHashOptions::default())
    }

    /// vLLM's CPU offload connector publishes `medium: "CPU"`. The fork this
    /// replaces filed those on the device tree; they must land in HostPinned
    /// so `/query` reports them as `cpu`, not `gpu` (upstream #10368).
    #[tokio::test]
    async fn cpu_medium_events_are_indexed_on_the_host_pinned_tier() {
        let (mut listener, indexer) = listener(None);
        let tokens: Vec<u32> = (1..=8).collect();
        listener
            .apply_live_batch(0, &stored_batch(&tokens, 4, "CPU"))
            .await
            .unwrap();
        indexer.dump_events().await.unwrap();

        let tiered = indexer.find_tiered_matches(probe(&tokens)).await.unwrap();
        let worker = WorkerWithDpRank::new(7, 0);
        assert_eq!(tiered.device.overlap_scores.scores.get(&worker), None);
        assert_eq!(
            tiered
                .lower_tier
                .get(&StorageTier::HostPinned)
                .and_then(|m| m.hits.get(&worker))
                .copied(),
            Some(2)
        );
    }

    #[tokio::test]
    async fn partial_prefix_stores_are_skipped_not_fatal() {
        let (mut listener, indexer) = listener(None);
        let tokens: Vec<u32> = (1..=2).collect();
        listener
            .apply_live_batch(0, &stored_batch(&tokens, 2, "GPU"))
            .await
            .unwrap();
        assert!(indexer.dump_events().await.unwrap().is_empty());
    }

    /// `--keep-evictions`: an eviction is parked, not applied, so the block
    /// stays matchable; a re-store cancels the parked eviction.
    #[tokio::test]
    async fn kept_evictions_are_parked_and_cancelled_by_a_restore() {
        let pending: SharedPendingEvictions =
            Arc::new(parking_lot::Mutex::new(PendingEvictions::default()));
        let (mut listener, indexer) = listener(Some(pending.clone()));
        let tokens: Vec<u32> = (1..=4).collect();
        listener
            .apply_live_batch(0, &stored_batch(&tokens, 4, "GPU"))
            .await
            .unwrap();
        listener
            .apply_live_batch(1, &removed_batch(&[1000]))
            .await
            .unwrap();
        indexer.dump_events().await.unwrap();

        let scores = indexer.find_matches(probe(&tokens)).await.unwrap();
        assert_eq!(scores.scores.get(&WorkerWithDpRank::new(7, 0)), Some(&1));
        assert_eq!(pending.lock().len(), 1);

        listener
            .apply_live_batch(2, &stored_batch(&tokens, 4, "GPU"))
            .await
            .unwrap();
        assert_eq!(pending.lock().len(), 0);
    }

    /// A gap on a listener with a recover endpoint is filled from
    /// `/kv_recover`: a TreeDump replaces the rank's state and moves the
    /// watermark to its `last_event_id`.
    #[tokio::test]
    async fn gap_is_recovered_from_kv_recover_tree_dump() {
        use crate::services::indexer::backend::test_util::store_event;
        use crate::services::indexer::kv_recover::{KvRecoverClient, KvRecoverSettings};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let tokens: Vec<u32> = (1..=8).collect();
        let hashes: Vec<u64> = probe(&tokens).into_iter().map(|h| h.0).collect();
        let body = serde_json::json!({
            "TreeDump": {
                "events": [store_event(999, 3, 0, &[], &hashes, StorageTier::Device)],
                "last_event_id": 41
            }
        })
        .to_string();
        let server = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", server.local_addr().unwrap());
        tokio::spawn(async move {
            let (mut stream, _) = server.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf).await;
            let reply = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(reply.as_bytes()).await.unwrap();
        });

        let (mut listener, indexer) = listener(None);
        listener.recover = Some(RecoverTarget {
            endpoint,
            client: Arc::new(KvRecoverClient::new(KvRecoverSettings::default()).unwrap()),
        });
        listener.handle_gap(42).await.unwrap();
        indexer.dump_events().await.unwrap();

        assert_eq!(listener.watermark.load(Ordering::Acquire), 41);
        let scores = indexer.find_matches(probe(&tokens)).await.unwrap();
        assert_eq!(scores.scores.get(&WorkerWithDpRank::new(7, 0)), Some(&2));
    }
}
