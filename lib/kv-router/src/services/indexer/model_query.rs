// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-level queries: one `/query` or `/query_by_hash` answered from every
//! routing group's tree of the model, merged into one response.
//!
//! The pod watcher keys each pod's tree by its `engine_hash`, so a model's
//! routing groups are hash regimes, not caller-visible partitions. A probe
//! hashes in one regime and scores 0 against the others, so the merged
//! response converges on the pods that can reuse the cache. A request that
//! names a `routing_group` reads only that tree (upstream behavior); the
//! legacy `tenant_id` is ignored.

use std::collections::HashMap;

use axum::http::StatusCode;

use crate::identity::RoutingPartitionId;
use crate::protocols::{LocalBlockHash, WorkerId};

use super::backend::Indexer;
use super::server::{ScoreResponse, build_score_response};

pub(super) struct ModelQueryOutcome {
    pub status: StatusCode,
    pub body: serde_json::Value,
}

/// Query every tree in `trees` (non-empty; callers 404 before calling) and
/// merge. `hashes_for(block_size)` supplies each tree's probe hashes. A tree
/// that fails is skipped while any tree answers; 500 only if all fail.
pub(super) async fn query_model(
    trees: Vec<(RoutingPartitionId, Indexer, u32)>,
    mut hashes_for: impl FnMut(u32) -> Vec<LocalBlockHash>,
    pod_names: &HashMap<WorkerId, String>,
) -> ModelQueryOutcome {
    let mut merged: Option<ScoreResponse> = None;
    let mut last_error: Option<String> = None;

    for (key, indexer, block_size) in trees {
        match indexer.find_tiered_matches(hashes_for(block_size)).await {
            Ok(tiered) => {
                let response = build_score_response(&tiered, block_size, pod_names);
                match merged.as_mut() {
                    Some(acc) => merge_score_responses(acc, response),
                    None => merged = Some(response),
                }
            }
            Err(error) => {
                tracing::warn!(
                    model_name = %key.model_name,
                    routing_group = %key.routing_group,
                    %error,
                    "Routing-group tree query failed; continuing fan-out"
                );
                last_error = Some(error.to_string());
            }
        }
    }

    let (status, body) = match merged {
        Some(response) => (StatusCode::OK, serde_json::json!(response)),
        None => (
            StatusCode::INTERNAL_SERVER_ERROR,
            serde_json::json!({
                "error": last_error.unwrap_or_else(|| "no trees queried".to_string())
            }),
        ),
    };
    ModelQueryOutcome { status, body }
}

/// Worker sets are disjoint across trees (a worker registers into exactly one
/// routing group), so `scores` and `instances` merge by union;
/// `frequencies[i]` counts workers holding block `i`, so it sums element-wise.
fn merge_score_responses(acc: &mut ScoreResponse, other: ScoreResponse) {
    acc.scores.extend(other.scores);
    acc.instances.extend(other.instances);
    if acc.frequencies.len() < other.frequencies.len() {
        acc.frequencies.resize(other.frequencies.len(), 0);
    }
    for (slot, count) in acc.frequencies.iter_mut().zip(other.frequencies) {
        *slot += count;
    }
}
