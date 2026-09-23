// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `--enable-logging`: one `kv_audit` line per query and per ingested engine
//! event, for offline analysis (filter with `RUST_LOG=kv_audit=info`).
//!
//! Event lines are written before the `--keep-evictions` filter, so they show
//! what the engine published. `source` is `live` (ZMQ SUB), `replay` (ZMQ
//! replay socket) or `recover` (`/kv_recover`).

use crate::protocols::{KvCacheEventData, LocalBlockHash, RouterEvent};

fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub(super) fn log_event(event: &RouterEvent, seq: u64, source: &'static str) {
    let ts_ms = now_unix_ms();
    let worker_id = event.worker_id;
    let dp_rank = event.event.dp_rank;
    let storage_tier = event.storage_tier;
    let event_id = event.event.event_id;
    match &event.event.data {
        KvCacheEventData::Stored(data) => {
            // tokens_hash is the local block hash (what /query computes);
            // block_hash is the engine's chained sequence hash.
            let token_block_hashes: Vec<u64> =
                data.blocks.iter().map(|b| b.tokens_hash.0).collect();
            let sequence_block_hashes: Vec<u64> =
                data.blocks.iter().map(|b| b.block_hash.0).collect();
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
                parent_hash = ?data.parent_hash.map(|h| h.0),
                num_blocks = token_block_hashes.len(),
                token_block_hashes = ?token_block_hashes,
                sequence_block_hashes = ?sequence_block_hashes,
                "kv_audit STORE"
            );
        }
        KvCacheEventData::Removed(data) => {
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

pub(super) fn log_query(
    model_name: &str,
    queried_groups: &[String],
    status: u16,
    block_hashes: &[LocalBlockHash],
    response: &serde_json::Value,
) {
    let block_hashes: Vec<u64> = block_hashes.iter().map(|h| h.0).collect();
    tracing::info!(
        target: "kv_audit",
        kind = "QUERY",
        ts_ms = now_unix_ms(),
        model_name,
        fanout_groups = ?queried_groups,
        status,
        num_blocks = block_hashes.len(),
        block_hashes = ?block_hashes,
        response = %response,
        "kv_audit QUERY"
    );
}
