// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dashmap::{DashMap, mapref::entry::Entry};
use rustc_hash::FxHashSet;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::time::{Duration, Instant};

use super::single::RequestId;
use crate::protocols::WorkerWithDpRank;
use crate::scheduling::AttemptId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RequestBooking {
    pub(super) worker: WorkerWithDpRank,
    pub(super) attempt_id: AttemptId,
}

/// The peer router that owns a mirrored booking and when it last announced it.
#[derive(Debug, Clone, Copy)]
struct MirrorOrigin {
    router_id: u64,
    announced_at: Instant,
}

/// A mirrored booking whose owner has not announced it for a while.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct UnannouncedMirror {
    pub(super) request_id: RequestId,
    pub(super) booking: RequestBooking,
    pub(super) origin: u64,
}

#[derive(Debug, Default)]
pub(super) struct RequestIndex {
    request_to_booking: DashMap<RequestId, RequestBooking>,
    request_to_lora: DashMap<RequestId, String>,
    /// Peer router that owns a mirrored booking. Bookings absent here are this router's own.
    request_to_origin: DashMap<RequestId, MirrorOrigin>,
    next_attempt_id: AtomicU64,
}

/// One of this router's own bookings, as replica sync re-announces it to peers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LocalBooking {
    pub(super) request_id: RequestId,
    pub(super) booking: RequestBooking,
    pub(super) lora_name: Option<String>,
}

impl RequestIndex {
    pub(super) fn try_insert_request(
        &self,
        request_id: RequestId,
        worker: WorkerWithDpRank,
        lora_name: Option<String>,
    ) -> Result<AttemptId, WorkerWithDpRank> {
        self.try_insert_request_from(request_id, worker, lora_name, None)
    }

    /// Insert a booking; `origin` names the peer router that owns a mirrored one.
    pub(super) fn try_insert_request_from(
        &self,
        request_id: RequestId,
        worker: WorkerWithDpRank,
        lora_name: Option<String>,
        origin: Option<u64>,
    ) -> Result<AttemptId, WorkerWithDpRank> {
        match self.request_to_booking.entry(request_id.clone()) {
            Entry::Occupied(entry) => Err(entry.get().worker),
            Entry::Vacant(entry) => {
                let attempt_id =
                    AttemptId::new(self.next_attempt_id.fetch_add(1, Ordering::Relaxed) + 1);
                if let Some(origin) = origin {
                    self.request_to_origin.insert(
                        request_id.clone(),
                        MirrorOrigin {
                            router_id: origin,
                            announced_at: Instant::now(),
                        },
                    );
                }
                entry.insert(RequestBooking { worker, attempt_id });
                if let Some(lora_name) = lora_name {
                    self.request_to_lora.insert(request_id, lora_name);
                }
                Ok(attempt_id)
            }
        }
    }

    #[cfg(test)]
    pub(super) fn set_request(
        &self,
        request_id: RequestId,
        worker: WorkerWithDpRank,
        lora_name: Option<String>,
    ) {
        let attempt_id = AttemptId::new(self.next_attempt_id.fetch_add(1, Ordering::Relaxed) + 1);
        self.request_to_booking
            .insert(request_id.clone(), RequestBooking { worker, attempt_id });
        if let Some(lora_name) = lora_name {
            self.request_to_lora.insert(request_id, lora_name);
        } else {
            self.request_to_lora.remove(&request_id);
        }
    }

    pub(super) fn worker_for(&self, request_id: &RequestId) -> Option<WorkerWithDpRank> {
        self.booking_for(request_id).map(|booking| booking.worker)
    }

    pub(super) fn booking_for(&self, request_id: &RequestId) -> Option<RequestBooking> {
        self.request_to_booking.get(request_id).map(|entry| *entry)
    }

    /// Record that `origin` announced its mirrored booking again. False when the booking is
    /// not a mirror of that router.
    pub(super) fn touch_origin(&self, request_id: &RequestId, origin: u64, now: Instant) -> bool {
        match self.request_to_origin.get_mut(request_id) {
            Some(mut entry) if entry.router_id == origin => {
                entry.announced_at = now;
                true
            }
            _ => false,
        }
    }

    /// Mirrored bookings whose owner last announced them more than `max_age` ago.
    pub(super) fn unannounced_mirrors(
        &self,
        now: Instant,
        max_age: Duration,
    ) -> Vec<UnannouncedMirror> {
        let stale: Vec<(RequestId, u64)> = self
            .request_to_origin
            .iter()
            .filter(|entry| now.saturating_duration_since(entry.announced_at) > max_age)
            .map(|entry| (entry.key().clone(), entry.router_id))
            .collect();
        stale
            .into_iter()
            .filter_map(|(request_id, origin)| {
                let booking = self.booking_for(&request_id)?;
                Some(UnannouncedMirror {
                    request_id,
                    booking,
                    origin,
                })
            })
            .collect()
    }

    /// This router's own bookings (those without a peer origin).
    pub(super) fn local_bookings(&self) -> Vec<LocalBooking> {
        self.request_to_booking
            .iter()
            .filter(|entry| !self.request_to_origin.contains_key(entry.key()))
            .map(|entry| LocalBooking {
                request_id: entry.key().clone(),
                booking: *entry.value(),
                lora_name: self.lora_for(entry.key()),
            })
            .collect()
    }

    pub(super) fn remote_origins(&self) -> FxHashSet<u64> {
        self.request_to_origin
            .iter()
            .map(|entry| entry.router_id)
            .collect()
    }

    pub(super) fn remote_requests_from(&self, origin: u64) -> Vec<(RequestId, RequestBooking)> {
        self.request_to_origin
            .iter()
            .filter(|entry| entry.router_id == origin)
            .filter_map(|entry| {
                let request_id = entry.key().clone();
                let booking = self.booking_for(&request_id)?;
                Some((request_id, booking))
            })
            .collect()
    }

    pub(super) fn lora_for(&self, request_id: &RequestId) -> Option<String> {
        self.request_to_lora
            .get(request_id)
            .map(|entry| entry.value().clone())
    }

    pub(super) fn remove_request(&self, request_id: &RequestId) -> Option<WorkerWithDpRank> {
        let worker = self
            .request_to_booking
            .remove(request_id)
            .map(|(_request_id, booking)| booking.worker);
        self.request_to_lora.remove(request_id);
        self.request_to_origin.remove(request_id);
        worker
    }

    /// Drop the mapping for `request_id` only if it still points at `worker`.
    /// Returns whether the entry was removed.
    pub(super) fn remove_request_if_worker(
        &self,
        request_id: &RequestId,
        worker: WorkerWithDpRank,
    ) -> bool {
        let removed = self
            .request_to_booking
            .remove_if(request_id, |_, booking| booking.worker == worker)
            .is_some();
        if removed {
            self.request_to_lora.remove(request_id);
            self.request_to_origin.remove(request_id);
        }
        removed
    }

    /// Drop the mapping only when it still belongs to the captured attempt.
    pub(super) fn remove_request_if_booking(
        &self,
        request_id: &RequestId,
        worker: WorkerWithDpRank,
        attempt_id: AttemptId,
    ) -> bool {
        let removed = self
            .request_to_booking
            .remove_if(request_id, |_, booking| {
                booking.worker == worker && booking.attempt_id == attempt_id
            })
            .is_some();
        if removed {
            self.request_to_lora.remove(request_id);
            self.request_to_origin.remove(request_id);
        }
        removed
    }

    pub(super) fn remove_requests<'a>(&self, request_ids: impl IntoIterator<Item = &'a RequestId>) {
        for request_id in request_ids {
            self.remove_request(request_id);
        }
    }

    pub(super) fn remove_worker_requests(&self, worker: WorkerWithDpRank) -> Vec<RequestId> {
        let request_ids: Vec<_> = self
            .request_to_booking
            .iter()
            .filter(|entry| entry.value().worker == worker)
            .map(|entry| entry.key().clone())
            .collect();
        self.remove_requests(request_ids.iter());
        request_ids
    }

    pub(super) fn active_lora_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for entry in self.request_to_lora.iter() {
            let lora_name = entry.value().clone();
            *counts.entry(lora_name).or_insert(0) += 1;
        }
        counts
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn is_empty(&self) -> bool {
        self.request_to_booking.is_empty()
            && self.request_to_lora.is_empty()
            && self.request_to_origin.is_empty()
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn worker_len(&self) -> usize {
        self.request_to_booking.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_insert_returns_existing_worker() {
        let index = RequestIndex::default();
        let worker = WorkerWithDpRank::new(1, 0);

        index
            .try_insert_request("req-1".to_string(), worker, Some("adapter".to_string()))
            .unwrap();
        assert_eq!(
            index.try_insert_request("req-1".to_string(), WorkerWithDpRank::new(2, 0), None),
            Err(worker)
        );
        assert_eq!(index.worker_for(&"req-1".to_string()), Some(worker));
        assert_eq!(
            index.lora_for(&"req-1".to_string()),
            Some("adapter".to_string())
        );
    }

    #[tokio::test(start_paused = true)]
    async fn unannounced_mirrors_are_the_ones_their_owner_stopped_announcing() {
        let index = RequestIndex::default();
        let worker = WorkerWithDpRank::new(1, 0);
        let max_age = Duration::from_secs(30);
        index
            .try_insert_request("local".to_string(), worker, None)
            .unwrap();
        for request_id in ["kept", "dropped"] {
            index
                .try_insert_request_from(request_id.to_string(), worker, None, Some(7))
                .unwrap();
        }

        tokio::time::advance(Duration::from_secs(20)).await;
        assert!(index.touch_origin(&"kept".to_string(), 7, Instant::now()));
        assert!(!index.touch_origin(&"kept".to_string(), 8, Instant::now()));
        assert!(!index.touch_origin(&"local".to_string(), 7, Instant::now()));
        tokio::time::advance(Duration::from_secs(15)).await;

        let stale = index.unannounced_mirrors(Instant::now(), max_age);
        assert_eq!(stale.len(), 1, "a local booking is never a stale mirror");
        assert_eq!(stale[0].request_id, "dropped");
        assert_eq!(stale[0].origin, 7);
        assert_eq!(stale[0].booking.worker, worker);
    }

    #[test]
    fn remove_request_is_idempotent() {
        let index = RequestIndex::default();
        let worker = WorkerWithDpRank::new(1, 0);
        let request_id = "req-1".to_string();

        index.set_request(request_id.clone(), worker, Some("adapter".to_string()));
        assert_eq!(index.remove_request(&request_id), Some(worker));
        assert_eq!(index.remove_request(&request_id), None);
        assert!(index.is_empty());
    }

    #[test]
    fn set_request_without_lora_clears_stale_lora_mapping() {
        let index = RequestIndex::default();
        let request_id = "req-1".to_string();

        index.set_request(
            request_id.clone(),
            WorkerWithDpRank::new(1, 0),
            Some("adapter".to_string()),
        );
        index.set_request(request_id.clone(), WorkerWithDpRank::new(2, 0), None);

        assert_eq!(
            index.worker_for(&request_id),
            Some(WorkerWithDpRank::new(2, 0))
        );
        assert_eq!(index.lora_for(&request_id), None);
    }

    #[test]
    fn remove_worker_requests_clears_both_maps() {
        let index = RequestIndex::default();
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);
        index.set_request("req-a".to_string(), worker_a, Some("adapter-a".to_string()));
        index.set_request("req-b".to_string(), worker_b, Some("adapter-b".to_string()));
        index.set_request("req-c".to_string(), worker_a, None);

        let mut removed = index.remove_worker_requests(worker_a);
        removed.sort();
        assert_eq!(removed, vec!["req-a".to_string(), "req-c".to_string()]);
        assert_eq!(index.worker_for(&"req-b".to_string()), Some(worker_b));
        assert_eq!(
            index.active_lora_counts(),
            HashMap::from([("adapter-b".to_string(), 1)])
        );
    }
}
