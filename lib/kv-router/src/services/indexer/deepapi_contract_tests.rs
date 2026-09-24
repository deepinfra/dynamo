// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The HTTP contract DeepInfra's deepapi relies on (`query_kv_indexer`,
//! `kv_shards_from_indexer_response` in deepinfra/backend): it POSTs
//! `{"model_name", "tenant_id": "default", "block_hashes": [i64]}` to
//! `/query_by_hash` and reads `instances[*].{pod_name, longest_matched, gpu}`.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use tower::ServiceExt;

use crate::protocols::StorageTier;

use super::backend::Indexer;
use super::backend::test_util::store_event;
use super::registry::{ListenerExtras, WorkerRegistry};
use super::server::{AppState, create_router};

const BLOCK_SIZE: u32 = 4;

async fn register_pod(
    registry: &WorkerRegistry,
    instance_id: u64,
    port: u16,
    model_name: &str,
    routing_group: &str,
    pod_name: &str,
) {
    registry
        .register_with_extras(
            instance_id,
            format!("tcp://127.0.0.1:{port}"),
            0,
            model_name.to_string(),
            routing_group.to_string(),
            BLOCK_SIZE,
            None,
            ListenerExtras {
                recover_endpoint: None,
                pod_name: Some(pod_name.to_string()),
            },
        )
        .await
        .unwrap();
}

async fn store_blocks(indexer: &Indexer, worker_id: u64, blocks: &[u64]) {
    indexer
        .apply_event_routed(store_event(
            worker_id,
            0,
            1,
            &[],
            blocks,
            StorageTier::Device,
        ))
        .await
        .unwrap();
    indexer.dump_events().await.expect("flush indexer");
}

fn router(registry: Arc<WorkerRegistry>) -> axum::Router {
    create_router(Arc::new(AppState {
        registry,
        access_log_sink: None,
        #[cfg(feature = "metrics")]
        prom_registry: prometheus::Registry::new(),
    }))
}

async fn post_json(
    app: &axum::Router,
    uri: &str,
    body: serde_json::Value,
) -> (StatusCode, serde_json::Value) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn query_by_hash_names_each_matched_instance_by_pod() {
    let registry = Arc::new(WorkerRegistry::new(1));
    registry.signal_ready();
    register_pod(&registry, 7, 15700, "m", "default", "vllm-m-abcd").await;
    let indexer = registry
        .get_indexer(&crate::identity::RoutingPartitionId::new("m", "default"))
        .unwrap()
        .indexer
        .clone();
    store_blocks(&indexer, 7, &[11, 12]).await;

    let app = router(registry);
    let (status, body) = post_json(
        &app,
        "/query_by_hash",
        serde_json::json!({"model_name": "m", "tenant_id": "default", "block_hashes": [11, 12]}),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    let instance = &body["instances"]["7"];
    assert_eq!(instance["pod_name"], "vllm-m-abcd");
    assert_eq!(instance["longest_matched"], 2 * BLOCK_SIZE);
    assert_eq!(instance["gpu"], 2 * BLOCK_SIZE);

    let workers = app
        .oneshot(
            Request::builder()
                .uri("/workers")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(workers.into_body(), usize::MAX)
        .await
        .unwrap();
    let workers: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(workers[0]["pod_name"], "vllm-m-abcd");
}

fn indexer_of(registry: &WorkerRegistry, model_name: &str, routing_group: &str) -> Indexer {
    registry
        .get_indexer(&crate::identity::RoutingPartitionId::new(
            model_name,
            routing_group,
        ))
        .unwrap()
        .indexer
        .clone()
}

/// One model under two routing groups (two engine_hash regimes) plus an
/// unrelated model: deepapi's request names only the model (and
/// `tenant_id: "default"`, which matches no tree), and must be answered from
/// both groups of that model and nothing else.
#[tokio::test]
async fn query_by_hash_fans_out_across_routing_groups_of_the_model() {
    let registry = Arc::new(WorkerRegistry::new(1));
    registry.signal_ready();
    register_pod(&registry, 7, 15710, "m", "hash-a", "pod-a").await;
    register_pod(&registry, 8, 15711, "m", "hash-b", "pod-b").await;
    register_pod(&registry, 9, 15712, "other", "hash-a", "pod-c").await;
    store_blocks(&indexer_of(&registry, "m", "hash-a"), 7, &[11, 12]).await;
    store_blocks(&indexer_of(&registry, "m", "hash-b"), 8, &[11]).await;
    store_blocks(&indexer_of(&registry, "other", "hash-a"), 9, &[11, 12]).await;

    let app = router(registry);
    let (status, body) = post_json(
        &app,
        "/query_by_hash",
        serde_json::json!({"model_name": "m", "tenant_id": "default", "block_hashes": [11, 12]}),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    let instances = body["instances"].as_object().unwrap();
    assert_eq!(instances.len(), 2, "{body}");
    assert_eq!(instances["7"]["pod_name"], "pod-a");
    assert_eq!(instances["7"]["longest_matched"], 2 * BLOCK_SIZE);
    assert_eq!(instances["8"]["pod_name"], "pod-b");
    assert_eq!(instances["8"]["longest_matched"], BLOCK_SIZE);
    assert_eq!(body["scores"]["7"]["0"], 2 * BLOCK_SIZE);

    let (status, _) = post_json(
        &app,
        "/query_by_hash",
        serde_json::json!({"model_name": "nonexistent", "block_hashes": [11]}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// deepapi hashes prompts itself (`deepinfra/utils.kv_block_hashes`: XXH3-64
/// of little-endian u32 tokens, seed 1337, lora seed 1337 + xxh3(name)). These
/// vectors were produced by that function; `/query` must hash identically or
/// every deepapi probe misses.
#[test]
fn token_hashing_matches_deepapi_golden_vectors() {
    use crate::protocols::{BlockHashOptions, compute_block_hash_for_seq};
    let tokens: Vec<u32> = (1..=8).collect();
    let plain: Vec<u64> = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default())
        .into_iter()
        .map(|h| h.0)
        .collect();
    assert_eq!(plain, [14643705804678351452, 16777012769546811212]);

    let lora: Vec<u64> = compute_block_hash_for_seq(
        &tokens,
        4,
        BlockHashOptions {
            lora_name: Some("lora_7"),
            ..Default::default()
        },
    )
    .into_iter()
    .map(|h| h.0)
    .collect();
    assert_eq!(lora, [6090315765804930700, 3193399369686574168]);
}

#[tokio::test]
async fn query_by_tokens_fans_out_across_routing_groups() {
    use crate::protocols::{BlockHashOptions, compute_block_hash_for_seq};
    let registry = Arc::new(WorkerRegistry::new(1));
    registry.signal_ready();
    register_pod(&registry, 7, 15713, "m", "hash-a", "pod-a").await;
    register_pod(&registry, 8, 15714, "m", "hash-b", "pod-b").await;
    let token_ids: Vec<u32> = (0..8).collect();
    let hashes: Vec<u64> =
        compute_block_hash_for_seq(&token_ids, BLOCK_SIZE, BlockHashOptions::default())
            .into_iter()
            .map(|h| h.0)
            .collect();
    store_blocks(&indexer_of(&registry, "m", "hash-a"), 7, &hashes).await;
    store_blocks(&indexer_of(&registry, "m", "hash-b"), 8, &hashes[..1]).await;

    let (status, body) = post_json(
        &router(registry),
        "/query",
        serde_json::json!({"model_name": "m", "token_ids": token_ids}),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["instances"]["7"]["longest_matched"], 2 * BLOCK_SIZE);
    assert_eq!(body["instances"]["8"]["longest_matched"], BLOCK_SIZE);
}
