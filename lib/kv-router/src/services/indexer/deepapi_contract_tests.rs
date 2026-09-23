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
