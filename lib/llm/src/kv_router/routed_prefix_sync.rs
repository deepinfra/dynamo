// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-side estimate of what a shared KV store holds for a prompt.
//!
//! Prefill workers that load each other's prefixes from a shared store serve
//! a prompt cheaply on any of them once any of them computed it. The router's
//! overlap index mirrors each worker's own cache and cannot see that, so it
//! books most cached continuations as cold.
//!
//! With `DYN_ROUTER_ROUTED_PREFIX_TTL_SECS` set, each router replica remembers
//! the prompt prefixes routed to the prefill pool within that time: its own
//! routing decisions, and its peers', which every replica publishes on the
//! endpoint-scoped topic `router_routed_prefixes`. A new request's matching
//! prefix then counts as cached for the prefill load the scheduler books and
//! for the queue's uncached-token accounting. Worker selection is unaffected.
//!
//! A prefix is identified by a hash of the prompt's token ids up to a mark,
//! one mark every `DYN_ROUTER_ROUTED_PREFIX_MARK_TOKENS` tokens (default
//! 1024), independent of the router's block hashing: the prefill router runs
//! with active-block tracking off and has no sequence hashes of its own.
//! Set the window no longer than the store keeps a block.

use std::sync::Arc;

use anyhow::Result;
use dynamo_kv_router::sequences::routed_prefixes::RoutedPrefixes;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::{EventPublisher, EventSubscriber};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use xxhash_rust::xxh3::Xxh3;

use crate::protocols::TokenIdType;

pub const ROUTER_ROUTED_PREFIXES_SUBJECT: &str = "router_routed_prefixes";

const DEFAULT_MARK_TOKENS: usize = 1024;
/// Routing never waits on the publish queue; when it is full the newest
/// announcement is dropped and peers learn the prefix on its next turn.
const PUBLISH_QUEUE_CAPACITY: usize = 4096;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RoutedPrefixEvent {
    pub router_id: u64,
    pub mark_tokens: u32,
    pub marks: Vec<u64>,
}

pub struct RoutedPrefixTracker {
    routed: RoutedPrefixes,
    mark_tokens: usize,
    router_id: u64,
    publish_tx: Option<mpsc::Sender<RoutedPrefixEvent>>,
}

fn mark_tokens_from_env() -> usize {
    std::env::var("DYN_ROUTER_ROUTED_PREFIX_MARK_TOKENS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|tokens| *tokens > 0)
        .unwrap_or(DEFAULT_MARK_TOKENS)
}

/// One hash per `mark_tokens` tokens; element `i` covers tokens
/// `[0, (i + 1) * mark_tokens)`. A trailing partial mark is not hashed.
pub fn prefix_marks(token_ids: &[TokenIdType], mark_tokens: usize) -> Vec<u64> {
    let mut hasher = Xxh3::new();
    token_ids
        .chunks_exact(mark_tokens)
        .map(|chunk| {
            for token in chunk {
                hasher.update(&token.to_le_bytes());
            }
            hasher.digest()
        })
        .collect()
}

impl RoutedPrefixTracker {
    fn new(
        ttl: std::time::Duration,
        mark_tokens: usize,
        router_id: u64,
        publish_tx: Option<mpsc::Sender<RoutedPrefixEvent>>,
    ) -> Self {
        Self {
            routed: RoutedPrefixes::new(ttl, 1, Instant::now()),
            mark_tokens,
            router_id,
            publish_tx,
        }
    }

    /// `None` unless `DYN_ROUTER_ROUTED_PREFIX_TTL_SECS` is set. Replicas
    /// share what they route only when `share_with_replicas` is true.
    pub async fn from_env(
        endpoint: &Endpoint,
        router_id: u64,
        share_with_replicas: bool,
        cancellation_token: CancellationToken,
    ) -> Result<Option<Arc<Self>>> {
        let Some(ttl) = RoutedPrefixes::ttl_from_env() else {
            return Ok(None);
        };
        let mark_tokens = mark_tokens_from_env();
        if !share_with_replicas {
            tracing::info!(?ttl, mark_tokens, "Routed-prefix tracking enabled (this replica only)");
            return Ok(Some(Arc::new(Self::new(ttl, mark_tokens, router_id, None))));
        }

        let transport_kind = endpoint.drt().default_event_transport_kind();
        let publisher = EventPublisher::for_endpoint_with_transport(
            endpoint,
            ROUTER_ROUTED_PREFIXES_SUBJECT,
            transport_kind,
        )
        .await?;
        let mut subscriber = EventSubscriber::for_endpoint_with_transport(
            endpoint,
            ROUTER_ROUTED_PREFIXES_SUBJECT,
            transport_kind,
        )
        .await?
        .typed::<RoutedPrefixEvent>();

        let (publish_tx, mut publish_rx) = mpsc::channel(PUBLISH_QUEUE_CAPACITY);
        let tracker = Arc::new(Self::new(ttl, mark_tokens, router_id, Some(publish_tx)));

        let publish_cancel = cancellation_token.clone();
        tokio::spawn(async move {
            loop {
                let event = tokio::select! {
                    _ = publish_cancel.cancelled() => break,
                    event = publish_rx.recv() => match event {
                        Some(event) => event,
                        None => break,
                    },
                };
                if let Err(error) = publisher.publish(&event).await {
                    tracing::warn!(%error, "Failed to publish routed prefix");
                }
            }
        });

        let receiver = Arc::clone(&tracker);
        tokio::spawn(async move {
            loop {
                let received = tokio::select! {
                    _ = cancellation_token.cancelled() => break,
                    received = subscriber.next() => received,
                };
                let Some(received) = received else {
                    break;
                };
                match received {
                    Ok((_, event)) => receiver.apply_peer_event(&event),
                    Err(error) => tracing::warn!(%error, "Error receiving routed prefix"),
                }
            }
        });

        tracing::info!(
            ?ttl,
            mark_tokens,
            router_id,
            "Routed-prefix tracking enabled: prefixes routed by any replica count as cached for booking"
        );
        Ok(Some(tracker))
    }

    pub fn marks(&self, token_ids: &[TokenIdType]) -> Vec<u64> {
        prefix_marks(token_ids, self.mark_tokens)
    }

    /// Leading prompt tokens routed to the pool within the window.
    pub fn cached_tokens(&self, marks: &[u64]) -> usize {
        self.routed.matched_blocks(marks, Instant::now()) * self.mark_tokens
    }

    /// Note that a prompt was routed to the pool, and tell the peers.
    pub fn record(&self, marks: Vec<u64>) {
        if marks.is_empty() {
            return;
        }
        self.routed.record(&marks, Instant::now());
        if let Some(publish_tx) = &self.publish_tx {
            let _ = publish_tx.try_send(RoutedPrefixEvent {
                router_id: self.router_id,
                mark_tokens: self.mark_tokens as u32,
                marks,
            });
        }
    }

    fn apply_peer_event(&self, event: &RoutedPrefixEvent) {
        // A peer hashing at another granularity produces marks that mean
        // something else; ignore them rather than match nonsense.
        if event.router_id == self.router_id || event.mark_tokens as usize != self.mark_tokens {
            return;
        }
        self.routed.record(&event.marks, Instant::now());
    }

    /// Hashes held; the set's memory is proportional to this.
    pub fn held_hashes(&self) -> usize {
        self.routed.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker(router_id: u64) -> RoutedPrefixTracker {
        RoutedPrefixTracker::new(std::time::Duration::from_secs(60), 4, router_id, None)
    }

    fn prompt(seed: u32, tokens: usize) -> Vec<TokenIdType> {
        (0..tokens as u32).map(|i| seed.wrapping_mul(31).wrapping_add(i)).collect()
    }

    #[test]
    fn marks_identify_prefixes() {
        let short = prefix_marks(&prompt(1, 10), 4);
        let long = prefix_marks(&prompt(1, 21), 4);
        assert_eq!(short.len(), 2);
        assert_eq!(long.len(), 5);
        // A longer prompt with the same start has the same leading marks.
        assert_eq!(short[..], long[..2]);
        // A different prompt differs from the first mark on.
        assert_ne!(prefix_marks(&prompt(2, 10), 4)[0], short[0]);
        // Same tokens in a later position do not match an earlier mark.
        assert_ne!(long[0], long[1]);
    }

    #[test]
    fn a_routed_prompt_makes_its_continuation_cached() {
        let tracker = tracker(1);
        let first_turn = prompt(1, 22);
        assert_eq!(tracker.cached_tokens(&tracker.marks(&first_turn)), 0);
        tracker.record(tracker.marks(&first_turn));

        let mut second_turn = first_turn.clone();
        second_turn.extend(prompt(9, 13));
        // 22 tokens routed -> 5 marks of 4 tokens = 20 tokens known.
        assert_eq!(tracker.cached_tokens(&tracker.marks(&second_turn)), 20);
        assert_eq!(tracker.cached_tokens(&tracker.marks(&prompt(2, 22))), 0);
        assert_eq!(tracker.held_hashes(), 5);
    }

    #[test]
    fn peers_teach_each_other_and_ignore_their_own_echo() {
        let (a, b) = (tracker(1), tracker(2));
        let turn = prompt(3, 16);
        let event = RoutedPrefixEvent {
            router_id: 1,
            mark_tokens: 4,
            marks: a.marks(&turn),
        };
        // B never routed it, but learns it from A's announcement.
        b.apply_peer_event(&event);
        assert_eq!(b.cached_tokens(&b.marks(&turn)), 16);
        // A's own echo does not count as a second source.
        a.apply_peer_event(&event);
        assert_eq!(a.held_hashes(), 0);
        // An announcement hashed at another granularity is ignored.
        let c = tracker(3);
        c.apply_peer_event(&RoutedPrefixEvent { mark_tokens: 8, ..event });
        assert_eq!(c.held_hashes(), 0);
    }

    #[test]
    fn wire_format_round_trips() {
        let event = RoutedPrefixEvent { router_id: 7, mark_tokens: 1024, marks: vec![1, 2, 3] };
        let bytes = rmp_serde::to_vec_named(&event).unwrap();
        assert_eq!(rmp_serde::from_slice::<RoutedPrefixEvent>(&bytes).unwrap(), event);
    }
}
