// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-worker gap recovery over HTTP: `GET <recover_endpoint>/kv_recover`.
//!
//! An alternative to the ZMQ DEALER replay socket for engines that serve their
//! local indexer over HTTP. On a sequence gap the listener fetches a
//! [`WorkerKvQueryResponse`] for `[start, end)` and applies it per
//! [`plan_recovery`]:
//!
//! - `Events` — apply all, resume at `last_event_id`.
//! - `TreeDump` — drop this `(worker, dp_rank)`'s state, apply all, resume at
//!   `last_event_id` (dump event ids are synthetic).
//! - anything else — no-op, the watermark stays where it was.
//!
//! Recovery is at-least-once: the resume point is always the response's
//! `last_event_id`, never a count of applied events. Events are applied under
//! the consumer-assigned identity; the worker's self-reported one is ignored.
//!
//! A large engine's TreeDump is tens of MB serialized inside the engine
//! process, and a freshly started indexer asks every engine at once, so
//! downloads share a process-wide gate, get a long total timeout, and are
//! retried with backoff before the gap is given up.

use std::time::Duration;

use anyhow::Context;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::indexer::WorkerKvQueryResponse;
use crate::protocols::{ResetScope, RouterEvent, WorkerId};

pub const DEFAULT_RECOVER_TIMEOUT_S: u64 = 120;
pub const DEFAULT_RECOVER_CONCURRENCY: usize = 8;

const RECOVER_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const RECOVER_ATTEMPTS: u32 = 3;
const RECOVER_BACKOFF_BASE: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Copy)]
pub struct KvRecoverSettings {
    /// Total timeout (connect + body) for one download.
    pub timeout_s: u64,
    /// Downloads allowed in flight across every listener of this process.
    pub concurrency: usize,
}

impl Default for KvRecoverSettings {
    fn default() -> Self {
        Self {
            timeout_s: DEFAULT_RECOVER_TIMEOUT_S,
            concurrency: DEFAULT_RECOVER_CONCURRENCY,
        }
    }
}

/// Shared by every listener of a registry: one HTTP client and one gate.
pub struct KvRecoverClient {
    http: reqwest::Client,
    gate: Semaphore,
    backoff_base: Duration,
}

impl KvRecoverClient {
    pub fn new(settings: KvRecoverSettings) -> anyhow::Result<Self> {
        Self::with_backoff(settings, RECOVER_BACKOFF_BASE)
    }

    fn with_backoff(settings: KvRecoverSettings, backoff_base: Duration) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(RECOVER_CONNECT_TIMEOUT)
            .timeout(Duration::from_secs(settings.timeout_s.max(1)))
            .build()
            .context("failed to build kv_recover HTTP client")?;
        Ok(Self {
            http,
            gate: Semaphore::new(settings.concurrency.max(1)),
            backoff_base,
        })
    }

    /// Download the worker's recovery response for `[start_seq, end_seq)`,
    /// retrying failed downloads. `None` when every attempt failed (logged as
    /// `kv_recover request failed; giving up`) or `cancel` fired.
    pub(super) async fn fetch(
        &self,
        recover_endpoint: &str,
        start_seq: u64,
        end_seq: u64,
        worker_id: WorkerId,
        dp_rank: u32,
        cancel: &CancellationToken,
    ) -> Option<WorkerKvQueryResponse> {
        let url = format!("{}/kv_recover", recover_endpoint.trim_end_matches('/'));
        for attempt in 0..RECOVER_ATTEMPTS {
            let result = tokio::select! {
                _ = cancel.cancelled() => return None,
                result = self.fetch_once(&url, start_seq, end_seq) => result,
            };
            let error = match result {
                Ok(response) => return Some(response),
                Err(error) => error,
            };
            if attempt + 1 == RECOVER_ATTEMPTS {
                tracing::error!(
                    worker_id,
                    dp_rank,
                    attempts = RECOVER_ATTEMPTS,
                    gap_size = end_seq.saturating_sub(start_seq),
                    error = %format!("{error:#}"),
                    "kv_recover request failed; giving up, batches lost"
                );
                return None;
            }
            let delay = self.backoff(attempt);
            tracing::warn!(
                worker_id,
                dp_rank,
                attempt = attempt + 1,
                retry_in_ms = delay.as_millis() as u64,
                error = %format!("{error:#}"),
                "kv_recover request failed; retrying"
            );
            tokio::select! {
                _ = cancel.cancelled() => return None,
                _ = tokio::time::sleep(delay) => {}
            }
        }
        None
    }

    async fn fetch_once(
        &self,
        url: &str,
        start_seq: u64,
        end_seq: u64,
    ) -> anyhow::Result<WorkerKvQueryResponse> {
        let _permit = self
            .gate
            .acquire()
            .await
            .expect("kv_recover gate is never closed");
        let response = self
            .http
            .get(url)
            .query(&[("start", start_seq), ("end", end_seq)])
            .send()
            .await?;
        anyhow::ensure!(
            response.status().is_success(),
            "kv_recover returned status {}",
            response.status()
        );
        Ok(response.json().await?)
    }

    /// 2 s, 4 s, 8 s, ... capped at 16x the base.
    fn backoff(&self, attempt: u32) -> Duration {
        self.backoff_base * (1u32 << attempt.min(4))
    }
}

/// What a listener must do with one recovery response.
#[derive(Debug, Default)]
pub(super) struct RecoveryPlan {
    /// Drop this `(worker, dp_rank)`'s indexed state before applying.
    pub reset_dp_rank: bool,
    /// Events to apply, already rewritten to the consumer-assigned identity.
    pub events: Vec<RouterEvent>,
    /// New watermark; `None` leaves it unchanged.
    pub resume_at: Option<u64>,
}

pub(super) fn plan_recovery(
    response: WorkerKvQueryResponse,
    worker_id: WorkerId,
    dp_rank: u32,
) -> RecoveryPlan {
    let reassign = |mut events: Vec<RouterEvent>| {
        for event in &mut events {
            event.worker_id = worker_id;
            event.event.dp_rank = dp_rank;
        }
        events
    };
    match response {
        WorkerKvQueryResponse::Events {
            events,
            last_event_id,
        } => RecoveryPlan {
            reset_dp_rank: false,
            events: reassign(events),
            resume_at: Some(last_event_id),
        },
        WorkerKvQueryResponse::TreeDump {
            events,
            last_event_id,
            reset_scope,
        } => {
            // A domain-scoped snapshot does not describe the whole rank, so it
            // must not wipe it; applying it on top stays at-least-once.
            let reset_dp_rank = matches!(reset_scope, ResetScope::All);
            if !reset_dp_rank {
                tracing::warn!(
                    worker_id,
                    dp_rank,
                    ?reset_scope,
                    "kv_recover TreeDump is domain-scoped; applying without a reset"
                );
            }
            RecoveryPlan {
                reset_dp_rank,
                events: reassign(events),
                resume_at: Some(last_event_id),
            }
        }
        WorkerKvQueryResponse::StateAgentRecovery { response, .. } => {
            plan_recovery(*response, worker_id, dp_rank)
        }
        WorkerKvQueryResponse::TooNew {
            newest_available, ..
        } => {
            tracing::debug!(
                worker_id,
                dp_rank,
                newest_available,
                "kv_recover returned TooNew; consumer is ahead, no-op"
            );
            RecoveryPlan::default()
        }
        other => {
            tracing::warn!(
                worker_id,
                dp_rank,
                response = ?other,
                "kv_recover returned no recoverable state; no-op"
            );
            RecoveryPlan::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{KvCacheEventData, StorageTier};
    use crate::services::indexer::backend::test_util::store_event;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A real `/kv_recover` Events response captured from a prod
    /// zai-org/GLM-5.3 engine (2026-09-23), trimmed to two events.
    const CAPTURED_EVENTS_RESPONSE: &str = r#"{"Events": {"events": [{"worker_id": 0, "storage_tier": "device", "event": {"event_id": 22636960, "data": {"removed": {"block_hashes": [18405780835976372970]}}, "dp_rank": 0}}, {"worker_id": 0, "storage_tier": "device", "event": {"event_id": 22636961, "data": {"stored": {"parent_hash": 270656378982115943, "start_position": null, "blocks": [{"block_hash": 2733357436102908004, "tokens_hash": 2825904518766229643, "mm_extra_info": null}, {"block_hash": 2451106649480522483, "tokens_hash": 10358785732907640880, "mm_extra_info": null}]}}, "dp_rank": 0}}], "last_event_id": 22637089}}"#;

    #[test]
    fn captured_engine_response_plans_events_under_consumer_identity() {
        let response: WorkerKvQueryResponse =
            serde_json::from_str(CAPTURED_EVENTS_RESPONSE).unwrap();
        let plan = plan_recovery(response, 7, 1);

        assert!(!plan.reset_dp_rank);
        assert_eq!(plan.resume_at, Some(22637089));
        assert_eq!(plan.events.len(), 2);
        assert!(
            plan.events
                .iter()
                .all(|e| e.worker_id == 7 && e.event.dp_rank == 1)
        );
        let KvCacheEventData::Stored(stored) = &plan.events[1].event.data else {
            panic!("second captured event is a store");
        };
        assert_eq!(stored.blocks.len(), 2);
    }

    #[test]
    fn legacy_tree_dump_resets_the_rank() {
        let response: WorkerKvQueryResponse = serde_json::from_value(serde_json::json!({
            "TreeDump": {
                "events": [store_event(999, 3, 0, &[], &[22], StorageTier::Device)],
                "last_event_id": 10
            }
        }))
        .unwrap();
        let plan = plan_recovery(response, 7, 0);

        assert!(plan.reset_dp_rank);
        assert_eq!(plan.resume_at, Some(10));
        assert_eq!(plan.events[0].worker_id, 7);
    }

    #[test]
    fn too_new_leaves_the_watermark_alone() {
        let plan = plan_recovery(
            WorkerKvQueryResponse::TooNew {
                requested_start: Some(4),
                requested_end: Some(9),
                newest_available: 3,
            },
            7,
            0,
        );
        assert!(!plan.reset_dp_rank);
        assert!(plan.events.is_empty());
        assert_eq!(plan.resume_at, None);
    }

    /// Serve `/kv_recover` on a local port: the first `failures` requests get
    /// a 500, the rest the captured Events body.
    async fn flaky_recover_server(failures: usize) -> (String, Arc<AtomicUsize>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(AtomicUsize::new(0));
        let seen = requests.clone();
        tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await.unwrap();
                let n = seen.fetch_add(1, Ordering::SeqCst);
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf).await;
                let (status, body) = if n < failures {
                    ("500 Internal Server Error", "boom")
                } else {
                    ("200 OK", CAPTURED_EVENTS_RESPONSE)
                };
                let reply = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(reply.as_bytes()).await;
            }
        });
        (endpoint, requests)
    }

    fn fast_client(timeout_s: u64) -> KvRecoverClient {
        KvRecoverClient::with_backoff(
            KvRecoverSettings {
                timeout_s,
                concurrency: 1,
            },
            Duration::from_millis(10),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn failed_downloads_are_retried_until_one_succeeds() {
        let (endpoint, requests) = flaky_recover_server(2).await;
        let response = fast_client(5)
            .fetch(&endpoint, 5, 9, 7, 0, &CancellationToken::new())
            .await;

        assert!(matches!(
            response,
            Some(WorkerKvQueryResponse::Events { .. })
        ));
        assert_eq!(requests.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn gap_is_given_up_after_the_last_attempt() {
        let (endpoint, requests) = flaky_recover_server(usize::MAX).await;
        let response = fast_client(5)
            .fetch(&endpoint, 5, 9, 7, 0, &CancellationToken::new())
            .await;

        assert!(response.is_none());
        assert_eq!(requests.load(Ordering::SeqCst), RECOVER_ATTEMPTS as usize);
    }

    #[tokio::test]
    async fn a_download_slower_than_the_timeout_fails_the_attempt() {
        // Accepts and never answers.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let _hold = tokio::spawn(async move {
            let mut open = Vec::new();
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                open.push(stream);
            }
        });

        let started = std::time::Instant::now();
        let response = fast_client(1)
            .fetch(&endpoint, 5, 9, 7, 0, &CancellationToken::new())
            .await;

        assert!(response.is_none());
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_secs(RECOVER_ATTEMPTS as u64)
                && elapsed < Duration::from_secs(10),
            "each attempt must end at the 1 s timeout, took {elapsed:?}"
        );
    }

    #[test]
    fn default_settings_match_the_documented_flags() {
        let settings = KvRecoverSettings::default();
        assert_eq!(settings.timeout_s, 120);
        assert_eq!(settings.concurrency, 8);
        let client = KvRecoverClient::new(settings).unwrap();
        assert_eq!(client.backoff(0), Duration::from_secs(2));
        assert_eq!(client.backoff(1), Duration::from_secs(4));
        assert_eq!(client.backoff(9), Duration::from_secs(32));
    }
}
