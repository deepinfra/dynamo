// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shares each router replica's queue usage with its peers.
//!
//! A policy class's queue limits are checked by every replica against its own
//! queue, yet all replicas queue against the same workers, so N replicas hold
//! up to N times the configured limit. With this sync on
//! (`DYN_ROUTER_QUEUE_USAGE_SYNC=1`, and replica sync enabled), each replica
//! publishes what it holds queued per class and adds the peers' fresh reports
//! when it checks a limit, so the limit bounds the whole replica set.
//!
//! Reports travel on their own endpoint-scoped topic rather than as a new
//! active-sequence event: a replica that predates this ignores the topic
//! instead of failing to decode a booking batch.

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_kv_router::queue::{ClassQueueStats, PeerQueueUsage};
use dynamo_kv_router::scheduling::PolicyQueueStats;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::{EventPublisher, EventSubscriber};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

pub const ROUTER_QUEUE_USAGE_SUBJECT: &str = "router_queue_usage";

/// How often a replica with a non-empty queue reports it. Well under
/// `PEER_QUEUE_USAGE_MAX_AGE`, so a few lost reports do not age a peer out.
const QUEUE_USAGE_REPORT_INTERVAL: Duration = Duration::from_millis(100);

/// Opt-in: turning it on makes every queue limit of every class global, which
/// changes how much an existing per-replica limit admits.
pub fn queue_usage_sync_enabled() -> bool {
    parse_enabled(std::env::var("DYN_ROUTER_QUEUE_USAGE_SYNC").ok().as_deref())
}

fn parse_enabled(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|value| value.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RouterQueueClassUsage {
    pub policy_class: String,
    pub requests: u64,
    pub raw_isl_tokens: u64,
    pub cached_tokens: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RouterQueueUsageEvent {
    pub router_id: u64,
    pub classes: Vec<RouterQueueClassUsage>,
}

/// This replica's report: one entry per class, including empty ones, so a
/// peer replaces the whole previous report.
fn local_usage_event(
    router_id: u64,
    class_names: &[String],
    stats_for: &dyn Fn(usize) -> Option<ClassQueueStats>,
) -> RouterQueueUsageEvent {
    let classes = class_names
        .iter()
        .enumerate()
        .map(|(class_index, name)| {
            let stats = stats_for(class_index).unwrap_or(ClassQueueStats {
                pending_count: 0,
                pending_isl_tokens: 0,
                pending_cached_tokens: 0,
            });
            RouterQueueClassUsage {
                policy_class: name.clone(),
                requests: stats.pending_count as u64,
                raw_isl_tokens: stats.pending_isl_tokens as u64,
                cached_tokens: stats.pending_cached_tokens as u64,
            }
        })
        .collect();
    RouterQueueUsageEvent { router_id, classes }
}

fn is_empty(event: &RouterQueueUsageEvent) -> bool {
    event.classes.iter().all(|class| class.requests == 0)
}

/// A peer's report laid out by the local class order. Classes are matched by
/// name; one the peer does not report, or that only the peer has, counts as
/// empty.
fn peer_usage_by_class(
    event: &RouterQueueUsageEvent,
    class_names: &[String],
) -> Vec<PolicyQueueStats> {
    class_names
        .iter()
        .map(|name| {
            event
                .classes
                .iter()
                .find(|class| &class.policy_class == name)
                .map(|class| {
                    let raw_isl_tokens = class.raw_isl_tokens as usize;
                    let cached_tokens = (class.cached_tokens as usize).min(raw_isl_tokens);
                    PolicyQueueStats {
                        requests: class.requests as usize,
                        raw_isl_tokens,
                        cached_tokens,
                        uncached_tokens: raw_isl_tokens - cached_tokens,
                    }
                })
                .unwrap_or_default()
        })
        .collect()
}

/// Start publishing this replica's queue usage and applying its peers'.
pub async fn start_queue_usage_sync(
    endpoint: &Endpoint,
    router_id: u64,
    worker_type: &'static str,
    class_names: Vec<String>,
    stats_for: Arc<dyn Fn(usize) -> Option<ClassQueueStats> + Send + Sync>,
    peer_usage: Arc<PeerQueueUsage>,
    cancellation_token: CancellationToken,
) -> Result<()> {
    let transport_kind = endpoint.drt().default_event_transport_kind();
    let publisher = EventPublisher::for_endpoint_with_transport(
        endpoint,
        ROUTER_QUEUE_USAGE_SUBJECT,
        transport_kind,
    )
    .await?;
    let mut subscriber = EventSubscriber::for_endpoint_with_transport(
        endpoint,
        ROUTER_QUEUE_USAGE_SUBJECT,
        transport_kind,
    )
    .await?
    .typed::<RouterQueueUsageEvent>();

    let publish_names = class_names.clone();
    let publish_cancel = cancellation_token.clone();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(QUEUE_USAGE_REPORT_INTERVAL);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // An empty queue is reported once, when it drains, so peers stop
        // counting it at once; after that an idle replica stays silent.
        let mut last_was_empty = true;
        loop {
            tokio::select! {
                _ = publish_cancel.cancelled() => break,
                _ = ticker.tick() => {}
            }
            let event = local_usage_event(router_id, &publish_names, stats_for.as_ref());
            let empty = is_empty(&event);
            if empty && last_was_empty {
                continue;
            }
            last_was_empty = empty;
            if let Err(error) = publisher.publish(&event).await {
                tracing::warn!(worker_type, %error, "Failed to publish router queue usage");
            }
        }
    });

    tokio::spawn(async move {
        loop {
            let received = tokio::select! {
                _ = cancellation_token.cancelled() => break,
                received = subscriber.next() => received,
            };
            let Some(received) = received else {
                break;
            };
            let event = match received {
                Ok((_, event)) => event,
                Err(error) => {
                    tracing::warn!(worker_type, %error, "Error receiving router queue usage");
                    continue;
                }
            };
            if event.router_id == router_id {
                continue;
            }
            peer_usage.report(
                event.router_id,
                peer_usage_by_class(&event, &class_names),
                Instant::now(),
            );
        }
    });

    tracing::info!(
        worker_type,
        router_id,
        "Router queue usage sync enabled: queue limits count peer replicas' queues"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        vec!["standard".to_string(), "no-queue".to_string()]
    }

    #[test]
    fn enabled_only_when_asked() {
        assert!(!parse_enabled(None));
        assert!(!parse_enabled(Some("0")));
        assert!(!parse_enabled(Some("")));
        assert!(parse_enabled(Some("1")));
        assert!(parse_enabled(Some(" True ")));
    }

    #[test]
    fn local_report_lists_every_class() {
        let event = local_usage_event(7, &names(), &|class_index| {
            (class_index == 1).then_some(ClassQueueStats {
                pending_count: 3,
                pending_isl_tokens: 300_000,
                pending_cached_tokens: 250_000,
            })
        });
        assert_eq!(event.router_id, 7);
        assert_eq!(event.classes.len(), 2);
        assert_eq!(event.classes[0].requests, 0);
        assert_eq!(event.classes[1].policy_class, "no-queue");
        assert_eq!(event.classes[1].raw_isl_tokens, 300_000);
        assert!(!is_empty(&event));
        assert!(is_empty(&local_usage_event(7, &names(), &|_| None)));
    }

    #[test]
    fn peer_report_is_matched_by_class_name() {
        // The peer lists its classes in another order, has one we lack, and
        // omits one we have.
        let event = RouterQueueUsageEvent {
            router_id: 9,
            classes: vec![
                RouterQueueClassUsage {
                    policy_class: "cold".to_string(),
                    requests: 5,
                    raw_isl_tokens: 900_000,
                    cached_tokens: 0,
                },
                RouterQueueClassUsage {
                    policy_class: "no-queue".to_string(),
                    requests: 2,
                    raw_isl_tokens: 200_000,
                    cached_tokens: 150_000,
                },
            ],
        };
        let usage = peer_usage_by_class(&event, &names());
        assert_eq!(usage[0], PolicyQueueStats::default());
        assert_eq!(usage[1].requests, 2);
        assert_eq!(usage[1].uncached_tokens, 50_000);
    }

    #[test]
    fn wire_format_round_trips() {
        let event = local_usage_event(1, &names(), &|_| None);
        let bytes = rmp_serde::to_vec_named(&event).unwrap();
        assert_eq!(rmp_serde::from_slice::<RouterQueueUsageEvent>(&bytes).unwrap(), event);
    }
}
