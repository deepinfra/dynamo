// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Administration of the routers' Mooncake store indexes (`shared_cache_type: mooncake-store`).
//!
//! ### GET /router_policy/mooncake_store[?model=M]
//!
//! Each store an endpoint's router follows: whether its index is complete (the
//! master's snapshot plus every KV event since), its chunks, and the last
//! snapshot read. `complete` at the top is true when there is a store and every
//! store is complete: the router knows all store contents.
//!
//! ### POST /router_policy/mooncake_store/resync[?model=M][&wait=SECONDS]
//!
//! Reads every store's snapshot now, as after a missed event; a read in flight
//! counts. Without `wait` it replies 202 at once. With it, it replies when each
//! store's read ended (at most 600 s): 200 if every store installed one, else
//! 503. Each store's `resync` says how its read ended.

use std::{sync::Arc, time::Duration};

use axum::{
    Json, Router,
    extract::{Query, State, rejection::QueryRejection},
    http::{Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use dynamo_runtime::protocols::EndpointId;
use serde::{Deserialize, Serialize};

use super::RouteDoc;
use crate::discovery::ModelManager;
use crate::kv_router::shared_cache::{ResyncResult, StoreStatus};

const STATUS_PATH: &str = "/router_policy/mooncake_store";
const RESYNC_PATH: &str = "/router_policy/mooncake_store/resync";
const MAX_WAIT: Duration = Duration::from_secs(600);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StatusQuery {
    model: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResyncQuery {
    model: Option<String>,
    /// Seconds to wait for the reads to end.
    wait: Option<u64>,
}

#[derive(Serialize)]
struct EndpointStores {
    namespace: String,
    component: String,
    endpoint: String,
    models: Vec<String>,
    stores: Vec<StoreStatus>,
}

#[derive(Serialize)]
struct StoresResponse {
    complete: bool,
    endpoints: Vec<EndpointStores>,
}

impl StoresResponse {
    fn new(endpoints: Vec<EndpointStores>) -> Self {
        let mut stores = endpoints.iter().flat_map(|endpoint| &endpoint.stores);
        let complete = stores.clone().next().is_some() && stores.all(|store| store.complete);
        Self {
            complete,
            endpoints,
        }
    }
}

pub(crate) fn router(manager: Arc<ModelManager>) -> (Vec<RouteDoc>, Router) {
    (
        vec![
            RouteDoc::new(Method::GET, STATUS_PATH),
            RouteDoc::new(Method::POST, RESYNC_PATH),
        ],
        Router::new()
            .route(STATUS_PATH, get(status))
            .route(RESYNC_PATH, post(resync))
            .with_state(manager),
    )
}

fn error(status: StatusCode, message: impl ToString) -> Response {
    (
        status,
        Json(serde_json::json!({"error": message.to_string()})),
    )
        .into_response()
}

fn no_index(model: Option<&str>) -> Response {
    let message = match model {
        Some(model) => format!("no router of model {model} follows Mooncake stores"),
        None => "no router follows Mooncake stores".to_string(),
    };
    error(StatusCode::NOT_FOUND, message)
}

fn endpoint_stores(
    endpoint: EndpointId,
    models: Vec<String>,
    stores: Vec<StoreStatus>,
) -> EndpointStores {
    EndpointStores {
        namespace: endpoint.namespace,
        component: endpoint.component,
        endpoint: endpoint.name,
        models,
        stores,
    }
}

async fn status(
    State(manager): State<Arc<ModelManager>>,
    query: Result<Query<StatusQuery>, QueryRejection>,
) -> Response {
    let Query(query) = match query {
        Ok(query) => query,
        Err(rejection) => return error(rejection.status(), rejection.body_text()),
    };
    let indexes = manager.mooncake_store_indexes(query.model.as_deref());
    if indexes.is_empty() {
        return no_index(query.model.as_deref());
    }
    let endpoints = indexes
        .into_iter()
        .map(|(endpoint, models, cache)| endpoint_stores(endpoint, models, cache.status()))
        .collect();
    Json(StoresResponse::new(endpoints)).into_response()
}

async fn resync(
    State(manager): State<Arc<ModelManager>>,
    query: Result<Query<ResyncQuery>, QueryRejection>,
) -> Response {
    let Query(query) = match query {
        Ok(query) => query,
        Err(rejection) => return error(rejection.status(), rejection.body_text()),
    };
    let indexes = manager.mooncake_store_indexes(query.model.as_deref());
    if indexes.is_empty() {
        return no_index(query.model.as_deref());
    }
    let wait = query
        .wait
        .map(|seconds| Duration::from_secs(seconds).min(MAX_WAIT));
    tracing::info!(model = ?query.model, ?wait, "Reading Mooncake store snapshots on request");
    let endpoints: Vec<_> = futures::future::join_all(indexes.into_iter().map(
        |(endpoint, models, cache)| async move {
            endpoint_stores(endpoint, models, cache.resync(wait).await)
        },
    ))
    .await;
    let mut stores = endpoints.iter().flat_map(|endpoint| &endpoint.stores);
    let status = if wait.is_none() {
        StatusCode::ACCEPTED
    } else if stores.clone().next().is_some()
        && stores.all(|store| store.resync == Some(ResyncResult::Installed))
    {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(StoresResponse::new(endpoints))).into_response()
}
