// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::poll_fn;
use std::sync::Arc;
use std::task::Poll;

use rustc_hash::FxHashMap;
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use super::multi_worker::{
    ActiveSequencesMultiWorker, REPLICA_PEER_SILENT_REFRESHES, ReplicaWorkerPolicy,
    SequencePublisher, SequenceSubscriber,
};
use super::prompt_registry::WorkerLoadSnapshot;
use crate::protocols::{
    ActiveSequenceEvent, ActiveSequenceEventData, MAX_REPLICA_BATCH_DURATION,
    MAX_REPLICA_BATCH_EVENTS, WorkerWithDpRank,
};
use crate::scheduling::queue::SchedulerBookingDescriptor;

#[derive(Default)]
struct ReplicaBatchEffects {
    worker_loads: FxHashMap<WorkerWithDpRank, PendingLoadPublication>,
    wake_scheduler: bool,
    cleanup_prompt_trie: bool,
}

struct PendingLoadPublication {
    latest_load: WorkerLoadSnapshot,
    publish: bool,
}

impl ReplicaBatchEffects {
    fn record_worker_load(
        &mut self,
        worker: WorkerWithDpRank,
        load: WorkerLoadSnapshot,
        publish: bool,
    ) {
        self.worker_loads
            .entry(worker)
            .and_modify(|pending| {
                pending.latest_load = load;
                pending.publish |= publish;
            })
            .or_insert(PendingLoadPublication {
                latest_load: load,
                publish,
            });
    }
}

impl<P: SequencePublisher + 'static> ActiveSequencesMultiWorker<P> {
    /// Apply one decoded replica-sync batch and flush its deferred effects once.
    pub fn apply_replica_batch(&self, events: Vec<ActiveSequenceEvent>) {
        let mut effects = ReplicaBatchEffects::default();
        for event in events {
            self.apply_replica_event(event, &mut effects);
        }
        self.flush_replica_batch_effects(&mut effects);
    }

    /// Spawn a background task that subscribes to replica-sync events from peer routers
    /// and applies them to the local state.
    pub fn start_replica_sync<S: SequenceSubscriber + 'static>(
        self: &Arc<Self>,
        subscriber: S,
        cancel_token: CancellationToken,
    ) {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(error) = this.run_replica_sync(subscriber, cancel_token).await {
                tracing::error!("Error in active sequences events subscription: {error}");
            }
        });
    }

    pub(super) async fn run_replica_sync<S: SequenceSubscriber>(
        &self,
        mut subscriber: S,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<()> {
        let mut effects = ReplicaBatchEffects::default();

        loop {
            let result = tokio::select! {
                result = subscriber.next_event() => result,
                _ = cancel_token.cancelled() => {
                    tracing::debug!("Subscription task cancelled");
                    break;
                }
            };

            let Some(result) = result else {
                break;
            };
            let Ok(event) = result else {
                tracing::error!(
                    "Error receiving active sequence event: {}",
                    result.unwrap_err()
                );
                continue;
            };

            let batch_start = Instant::now();
            let mut batch_events = 0;
            let mut exit_after_flush = false;
            let mut yield_after_flush = false;
            let mut next_event = Some(event);

            loop {
                let event = next_event
                    .take()
                    .expect("replica batch event must be present");
                self.apply_replica_event(event, &mut effects);
                batch_events += 1;

                if cancel_token.is_cancelled() {
                    exit_after_flush = true;
                    break;
                }

                if batch_events >= MAX_REPLICA_BATCH_EVENTS
                    || batch_start.elapsed() >= MAX_REPLICA_BATCH_DURATION
                {
                    yield_after_flush = true;
                    break;
                }

                match poll_fn(|cx| Poll::Ready(subscriber.poll_next_event(cx))).await {
                    Poll::Ready(Some(Ok(event))) => next_event = Some(event),
                    Poll::Ready(Some(Err(error))) => {
                        tracing::error!("Error receiving active sequence event: {error}");
                        break;
                    }
                    Poll::Ready(None) => {
                        exit_after_flush = true;
                        break;
                    }
                    Poll::Pending => break,
                }
            }

            self.flush_replica_batch_effects(&mut effects);

            if exit_after_flush {
                break;
            }
            if yield_after_flush {
                tokio::task::yield_now().await;
            }
        }

        Ok(())
    }

    /// Re-announce this router's live bookings to its peers and drop the mirrored bookings
    /// of peers that have been silent for [`REPLICA_PEER_SILENT_REFRESHES`] intervals.
    ///
    /// A re-announcement is an `AddRequest` without blocks: a peer that already mirrors the
    /// booking refreshes its lease, and one that missed it books it again. Routers without
    /// this refresh ignore it as a duplicate.
    pub fn start_replica_refresh(
        self: &Arc<Self>,
        interval: Duration,
        cancel_token: CancellationToken,
    ) {
        if !self.replica_sync {
            return;
        }
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = cancel_token.cancelled() => break,
                    _ = ticker.tick() => {
                        this.refresh_replicas(Instant::now(), interval * REPLICA_PEER_SILENT_REFRESHES);
                    }
                }
            }
        });
    }

    /// One refresh round; see [`Self::start_replica_refresh`].
    pub fn refresh_replicas(&self, now: Instant, peer_silence: Duration) {
        if !self.replica_sync {
            return;
        }
        self.republish_local_bookings();
        self.purge_silent_peers(now, peer_silence);
    }

    fn republish_local_bookings(&self) {
        for local in self.request_index.local_bookings() {
            if self.request_index.booking_for(&local.request_id) != Some(local.booking) {
                continue;
            }
            self.enqueue_publish_event(ActiveSequenceEvent {
                request_id: local.request_id,
                worker: local.booking.worker,
                data: ActiveSequenceEventData::AddRequest {
                    token_sequence: None,
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                },
                router_id: self.router_id,
                lora_name: local.lora_name,
            });
        }
    }

    fn purge_silent_peers(&self, now: Instant, peer_silence: Duration) {
        let origins = self.request_index.remote_origins();
        let silent: Vec<(u64, Option<Duration>)> = {
            let mut last_seen = self.peer_last_seen.lock();
            last_seen.retain(|origin, seen| {
                origins.contains(origin) || now.saturating_duration_since(*seen) <= peer_silence
            });
            origins
                .into_iter()
                .filter_map(|origin| {
                    let silence = last_seen
                        .get(&origin)
                        .map(|seen| now.saturating_duration_since(*seen));
                    silence
                        .is_none_or(|silence| silence > peer_silence)
                        .then_some((origin, silence))
                })
                .collect()
        };
        if silent.is_empty() {
            return;
        }

        let mut effects = ReplicaBatchEffects::default();
        for (origin, silence) in silent {
            let bookings = self.request_index.remote_requests_from(origin);
            tracing::warn!(
                router_id = origin,
                worker_type = self.worker_type(),
                bookings = bookings.len(),
                silent_secs = silence.map(|silence| silence.as_secs()),
                "Peer router stopped refreshing its bookings; dropping its mirrored bookings"
            );
            for (request_id, booking) in bookings {
                self.apply_replica_lifecycle(
                    ActiveSequenceEvent {
                        request_id,
                        worker: booking.worker,
                        data: ActiveSequenceEventData::Free,
                        router_id: origin,
                        lora_name: None,
                    },
                    &mut effects,
                );
            }
            self.peer_last_seen.lock().remove(&origin);
        }
        self.flush_replica_batch_effects(&mut effects);
    }

    /// A refresh from the peer that owns this mirrored booking keeps its lease alive.
    fn touch_mirrored_booking(&self, request_id: &str, worker: WorkerWithDpRank, origin: u64) {
        let request_id = request_id.to_string();
        let Some(current) = self.request_index.booking_for(&request_id) else {
            return;
        };
        if current.worker != worker || self.request_index.origin_for(&request_id) != Some(origin) {
            return;
        }
        if let Some(observer) = self.replica_request_lease_observer() {
            observer.progressed(&SchedulerBookingDescriptor {
                request_id,
                worker,
                attempt_id: current.attempt_id,
            });
        }
    }

    fn apply_replica_event(&self, event: ActiveSequenceEvent, effects: &mut ReplicaBatchEffects) {
        if event.router_id == self.router_id {
            return;
        }
        let worker_origin = matches!(event.data, ActiveSequenceEventData::MarkPrefillCompleted);
        if !self.replica_sync && !worker_origin {
            return;
        }
        if !worker_origin {
            self.peer_last_seen
                .lock()
                .insert(event.router_id, Instant::now());
        }
        self.apply_replica_lifecycle(event, effects);
    }

    fn apply_replica_lifecycle(
        &self,
        event: ActiveSequenceEvent,
        effects: &mut ReplicaBatchEffects,
    ) {
        let ActiveSequenceEvent {
            request_id,
            worker: event_worker,
            data,
            router_id,
            lora_name,
        } = event;

        // ActiveSequenceEvent does not carry prompt-load decay timestamps yet.
        // Peer routers still approximate decay anchoring with local receive time.
        let decay_now = Instant::now();

        match data {
            ActiveSequenceEventData::AddRequest {
                token_sequence,
                track_prefill_tokens,
                expected_output_tokens,
                prefill_load_hint,
            } => {
                let Ok(attempt_id) = self.request_index.try_insert_request_from(
                    request_id.clone(),
                    event_worker,
                    lora_name,
                    Some(router_id),
                ) else {
                    self.touch_mirrored_booking(&request_id, event_worker, router_id);
                    return;
                };
                let booking = SchedulerBookingDescriptor {
                    request_id: request_id.clone(),
                    worker: event_worker,
                    attempt_id,
                };
                if self.replica_worker_policy == ReplicaWorkerPolicy::LazyRegister {
                    self.ensure_worker_registered(event_worker);
                }
                let table = self.workers.read();
                let Some(&idx) = table.index.get(&event_worker) else {
                    self.request_index.remove_request_if_booking(
                        &request_id,
                        event_worker,
                        attempt_id,
                    );
                    tracing::debug!(
                        worker = ?event_worker,
                        "Dropping replica AddRequest for unregistered worker"
                    );
                    return;
                };
                let (expired_request_ids, load) = {
                    let slot = &table.slots[idx];
                    let mut seq = slot.sequences.write();
                    let outcome = seq.add_request_with_prefill_tracking(
                        request_id,
                        token_sequence,
                        expected_output_tokens,
                        track_prefill_tokens,
                        prefill_load_hint,
                        decay_now,
                    );
                    let load = seq.worker_load_snapshot();
                    self.prompt_registry
                        .apply_membership_delta_and_load_without_cleanup(
                            event_worker,
                            outcome.membership_delta,
                            load,
                        );
                    (outcome.expired_request_ids, load)
                };
                drop(table);
                self.request_index
                    .remove_requests(expired_request_ids.iter());
                if let Some(observer) = self.replica_request_lease_observer() {
                    observer.admitted(booking);
                }
                effects.record_worker_load(event_worker, load, true);
                effects.cleanup_prompt_trie = true;
            }
            ActiveSequenceEventData::Free => {
                let Some(current) = self.request_index.booking_for(&request_id) else {
                    return;
                };
                if current.worker != event_worker {
                    return;
                }
                let booking = SchedulerBookingDescriptor {
                    request_id: request_id.clone(),
                    worker: current.worker,
                    attempt_id: current.attempt_id,
                };
                let table = self.workers.read();
                let Some(&idx) = table.index.get(&current.worker) else {
                    return;
                };
                let load = {
                    let slot = &table.slots[idx];
                    let mut seq = slot.sequences.write();
                    let Some(delta) = seq.free(&request_id, decay_now) else {
                        return;
                    };
                    let load = seq.worker_load_snapshot();
                    self.prompt_registry
                        .apply_membership_delta_and_load_without_cleanup(
                            current.worker,
                            delta,
                            load,
                        );
                    load
                };
                drop(table);
                self.request_index.remove_request_if_booking(
                    &request_id,
                    current.worker,
                    current.attempt_id,
                );
                if let Some(observer) = self.replica_request_lease_observer() {
                    observer.completed(&booking);
                }
                effects.record_worker_load(current.worker, load, true);
                effects.wake_scheduler = true;
                effects.cleanup_prompt_trie = true;
            }
            ActiveSequenceEventData::MarkPrefillCompleted => {
                let Some(current) = self.request_index.booking_for(&request_id) else {
                    return;
                };
                if current.worker != event_worker {
                    return;
                }
                let booking = SchedulerBookingDescriptor {
                    request_id: request_id.clone(),
                    worker: current.worker,
                    attempt_id: current.attempt_id,
                };
                let table = self.workers.read();
                let Some(&idx) = table.index.get(&current.worker) else {
                    return;
                };
                let load = {
                    let mut seq = table.slots[idx].sequences.write();
                    if !seq.mark_prefill_completed(&request_id, decay_now) {
                        return;
                    }
                    let load = seq.worker_load_snapshot();
                    self.prompt_registry
                        .replace_worker_load_state(current.worker, load);
                    load
                };
                drop(table);
                if let Some(observer) = self.replica_request_lease_observer() {
                    observer.progressed(&booking);
                }
                effects.record_worker_load(current.worker, load, false);
                effects.wake_scheduler = true;
            }
        }
    }

    fn flush_replica_batch_effects(&self, effects: &mut ReplicaBatchEffects) {
        let decay_now = Instant::now();
        let mut scheduler_loads = Vec::with_capacity(effects.worker_loads.len());
        for (worker, pending) in effects.worker_loads.drain() {
            if pending.publish {
                scheduler_loads.push(self.observe_worker_load_snapshot(
                    worker,
                    pending.latest_load,
                    decay_now,
                ));
            }
        }
        if !scheduler_loads.is_empty() {
            self.publisher.publish_scheduler_load_batch(scheduler_loads);
        }

        if std::mem::take(&mut effects.wake_scheduler) {
            self.notify_remote_state_update();
        }
        if std::mem::take(&mut effects.cleanup_prompt_trie) {
            self.prompt_registry.maybe_cleanup();
        }
    }
}
