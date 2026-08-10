// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes pod auto-discovery for the standalone indexer.
//!
//! Watches engine pods in a namespace (filtered by a label selector) and keeps
//! the [`WorkerRegistry`] in sync with them:
//!
//! - a pod that becomes **Ready** is registered (subscribe to its ZMQ KV-event
//!   port), and
//! - a pod that is **deleted** (or stops being Ready) is deregistered.
//!
//! This module is the *only* place that imports the Kubernetes client. It
//! depends on [`WorkerRegistry`] and calls nothing but
//! [`WorkerRegistry::register`] / [`WorkerRegistry::deregister`], so the
//! indexer core never learns about Kubernetes.
//!
//! ## How resilience is handled
//!
//! We use the high-level `kube::runtime::watcher` + `reflector` machinery. The
//! reflector maintains an in-memory `Store` of the current pods and the watcher
//! transparently reconnects and **re-lists on `410 Gone`** (the "I fell behind
//! the resourceVersion history" case) with exponential backoff. On every change
//! we recompute the desired set of Ready pods from the Store and reconcile it
//! against what we have registered — so a missed delete during a disconnect is
//! corrected on the next re-list rather than leaking a subscription.
//!
//! ## Required RBAC
//!
//! The indexer's ServiceAccount needs `get`, `list`, and `watch` on `pods` in
//! the target namespace:
//!
//! ```yaml
//! apiVersion: rbac.authorization.k8s.io/v1
//! kind: Role
//! metadata:
//!   name: kv-indexer-pod-watcher
//!   namespace: <namespace>
//! rules:
//!   - apiGroups: [""]
//!     resources: ["pods"]
//!     verbs: ["get", "list", "watch"]
//! ---
//! apiVersion: rbac.authorization.k8s.io/v1
//! kind: RoleBinding
//! metadata:
//!   name: kv-indexer-pod-watcher
//!   namespace: <namespace>
//! subjects:
//!   - kind: ServiceAccount
//!     name: <indexer-service-account>
//!     namespace: <namespace>
//! roleRef:
//!   kind: Role
//!   name: kv-indexer-pod-watcher
//!   apiGroup: rbac.authorization.k8s.io
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use k8s_openapi::api::core::v1::Pod;
use kube::runtime::{WatchStreamExt, reflector, watcher};
use kube::{Api, Client};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use xxhash_rust::xxh3;

use crate::protocols::WorkerId;

use super::KubeDiscoveryConfig;
use super::registry::WorkerRegistry;

/// Label the backend stamps on every engine pod with a hash of the exact
/// deployment (image, env, args). Pods with different values may hash KV
/// blocks differently, so each value gets its own tree via `tenant_id`.
const ENGINE_HASH_LABEL: &str = "engine_hash";

/// A pod we have an active subscription for, keyed by pod name.
struct Subscribed {
    instance_id: WorkerId,
    ip: String,
    tenant_id: String,
}

/// A Ready engine pod as extracted from the watch store.
#[derive(Debug, PartialEq)]
struct DiscoveredPod {
    name: String,
    ip: String,
    /// Value of the pod's `engine_hash` label, when present.
    engine_hash: Option<String>,
}

impl DiscoveredPod {
    /// The tenant this pod's tree lives under: its `engine_hash`, so
    /// hash-incompatible engine generations never share a tree during
    /// migrations. Falls back to the configured tenant for unlabeled pods.
    fn tenant_id(&self, fallback: &str) -> String {
        self.engine_hash
            .clone()
            .unwrap_or_else(|| fallback.to_string())
    }
}

/// Spawn the pod watcher as a detached background task.
///
/// Returns immediately; the watch loop runs until `cancel` fires. Any fatal
/// error (e.g. the Kubernetes client cannot be created) is logged and the task
/// exits — it never panics or blocks the caller.
pub fn spawn_pod_watcher(
    config: KubeDiscoveryConfig,
    registry: Arc<WorkerRegistry>,
    cancel: CancellationToken,
) {
    tokio::spawn(async move {
        tracing::info!(
            namespace = %config.namespace,
            label_selector = %config.label_selector,
            zmq_port = config.zmq_port,
            model_name = %config.model_name,
            block_size = config.block_size,
            "Starting Kubernetes pod watcher"
        );
        if let Err(error) = run_pod_watcher(config, registry, cancel).await {
            tracing::error!(error = %error, "Pod watcher exited with error");
        }
    });
}

async fn run_pod_watcher(
    config: KubeDiscoveryConfig,
    registry: Arc<WorkerRegistry>,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    // Reads in-cluster config (mounted ServiceAccount token) when running as a
    // pod, or the local kubeconfig otherwise.
    let client = Client::try_default().await?;
    let api: Api<Pod> = Api::namespaced(client, &config.namespace);
    let watcher_config = watcher::Config::default().labels(&config.label_selector);

    // The reflector keeps `reader` populated with the current pod set; the
    // watcher feeds it and handles reconnects / 410-relist with backoff.
    let (reader, writer) = reflector::store();
    let notify = Arc::new(Notify::new());

    // Drive the watch stream on its own task. Each change (or error) pokes the
    // reconcile loop below; the store itself is updated by the reflector.
    let stream_notify = notify.clone();
    let stream_cancel = cancel.clone();
    tokio::spawn(async move {
        let stream = reflector(writer, watcher(api, watcher_config))
            .default_backoff()
            .touched_objects()
            .for_each(move |event| {
                if let Err(error) = event {
                    tracing::warn!(error = %error, "Pod watch stream error (auto-retrying)");
                }
                stream_notify.notify_one();
                futures::future::ready(())
            });
        tokio::select! {
            _ = stream => {}
            _ = stream_cancel.cancelled() => {}
        }
    });

    let mut subscribed: HashMap<String, Subscribed> = HashMap::new();

    // Initial sync. `wait_until_ready` returns once the reflector has committed
    // its first full LIST of pods, so the Store is readable. We then reconcile
    // once — this is what subscribes to engines that already exist at startup
    // (and on every restart), rather than only reacting to later changes.
    tokio::select! {
        _ = cancel.cancelled() => {
            tracing::info!("Pod watcher shutting down before initial sync");
            return Ok(());
        }
        result = reader.wait_until_ready() => {
            result?;
        }
    }
    reconcile(&reader, &mut subscribed, &registry, &config).await;
    tracing::info!(engines = subscribed.len(), "Initial pod sync complete");

    // Steady state: reconcile on every subsequent change.
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                tracing::info!("Pod watcher shutting down");
                return Ok(());
            }
            _ = notify.notified() => {
                reconcile(&reader, &mut subscribed, &registry, &config).await;
            }
        }
    }
}

/// Compare the current set of Ready pods against our subscriptions and apply
/// the difference: register newly-ready pods, deregister vanished ones.
async fn reconcile(
    store: &reflector::Store<Pod>,
    subscribed: &mut HashMap<String, Subscribed>,
    registry: &WorkerRegistry,
    config: &KubeDiscoveryConfig,
) {
    // Desired state: every Ready, non-terminating pod, keyed by name.
    let mut desired: HashMap<String, DiscoveredPod> = HashMap::new();
    for pod in store.state() {
        if let Some(discovered) = ready_pod_endpoint(pod.as_ref()) {
            desired.insert(discovered.name.clone(), discovered);
        }
    }

    // Unsubscribe first: pods that are gone, no longer Ready, or whose
    // engine_hash changed (the same pod name must re-register under its new
    // tenant, and register() rejects a worker that switches keys in place).
    let vanished: Vec<String> = subscribed
        .iter()
        .filter(|(name, entry)| {
            desired
                .get(*name)
                .is_none_or(|pod| pod.tenant_id(&config.tenant_id) != entry.tenant_id)
        })
        .map(|(name, _)| name.clone())
        .collect();
    for name in vanished {
        let entry = subscribed
            .remove(&name)
            .expect("name came from subscribed keys");
        match registry
            .deregister(entry.instance_id, &config.model_name, &entry.tenant_id)
            .await
        {
            Ok(()) => tracing::info!(
                pod = %name,
                ip = %entry.ip,
                instance_id = entry.instance_id,
                tenant_id = %entry.tenant_id,
                "Unsubscribed from engine pod"
            ),
            // Already gone from the registry — an idempotent no-op.
            Err(error) => tracing::debug!(
                pod = %name,
                error = %error,
                "Deregister was a no-op"
            ),
        }
    }

    // Subscribe to pods that are newly Ready (or re-tenanted above).
    for (name, pod) in &desired {
        if subscribed.contains_key(name) {
            continue; // already subscribed — idempotent no-op
        }

        let instance_id = instance_id_for(name);
        let ip = &pod.ip;
        let tenant_id = pod.tenant_id(&config.tenant_id);
        let endpoint = format!("tcp://{ip}:{}", config.zmq_port);
        let recover_endpoint = config
            .recover_port
            .map(|port| format!("http://{ip}:{port}"));

        match registry
            .register(
                instance_id,
                endpoint,
                0, // dp_rank: single-rank engines
                config.model_name.clone(),
                tenant_id.clone(),
                config.block_size,
                recover_endpoint,
                Some(name.clone()),
            )
            .await
        {
            Ok(()) => {
                tracing::info!(
                    pod = %name,
                    ip = %ip,
                    instance_id,
                    tenant_id = %tenant_id,
                    "Subscribed to engine pod"
                );
                subscribed.insert(
                    name.clone(),
                    Subscribed {
                        instance_id,
                        ip: ip.clone(),
                        tenant_id,
                    },
                );
            }
            Err(error) => {
                // Not recorded, so it is retried on the next reconcile.
                tracing::warn!(
                    pod = %name,
                    ip = %ip,
                    error = %error,
                    "Failed to register engine pod; will retry"
                );
            }
        }
    }
}

/// Derive a stable `WorkerId` from the pod name so the same pod always maps to
/// the same registry entry across MODIFIED events.
fn instance_id_for(pod_name: &str) -> WorkerId {
    xxh3::xxh3_64(pod_name.as_bytes())
}

/// Return the pod's name, IP, and `engine_hash` label if it is Ready, has an
/// IP, and is not terminating; otherwise `None`.
fn ready_pod_endpoint(pod: &Pod) -> Option<DiscoveredPod> {
    // A pod with a deletion timestamp is shutting down — treat it as gone.
    if pod.metadata.deletion_timestamp.is_some() {
        return None;
    }
    let name = pod.metadata.name.clone()?;
    let status = pod.status.as_ref()?;
    let ip = status.pod_ip.clone()?;

    let ready = status.conditions.as_ref().is_some_and(|conditions| {
        conditions
            .iter()
            .any(|c| c.type_ == "Ready" && c.status == "True")
    });
    if !ready {
        return None;
    }

    let engine_hash = pod
        .metadata
        .labels
        .as_ref()
        .and_then(|labels| labels.get(ENGINE_HASH_LABEL))
        .cloned();

    Some(DiscoveredPod {
        name,
        ip,
        engine_hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use k8s_openapi::api::core::v1::{PodCondition, PodStatus};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::{ObjectMeta, Time};
    use k8s_openapi::chrono::Utc;

    fn pod(name: &str, ip: Option<&str>, ready: Option<&str>, terminating: bool) -> Pod {
        pod_with_labels(name, ip, ready, terminating, &[])
    }

    fn pod_with_labels(
        name: &str,
        ip: Option<&str>,
        ready: Option<&str>,
        terminating: bool,
        labels: &[(&str, &str)],
    ) -> Pod {
        Pod {
            metadata: ObjectMeta {
                name: Some(name.to_string()),
                deletion_timestamp: terminating.then(|| Time(Utc::now())),
                labels: (!labels.is_empty()).then(|| {
                    labels
                        .iter()
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .collect()
                }),
                ..Default::default()
            },
            status: Some(PodStatus {
                pod_ip: ip.map(String::from),
                conditions: ready.map(|status| {
                    vec![PodCondition {
                        type_: "Ready".to_string(),
                        status: status.to_string(),
                        ..Default::default()
                    }]
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn ready_pod_with_ip_is_subscribable() {
        let p = pod("engine-abc", Some("10.0.0.1"), Some("True"), false);
        assert_eq!(
            ready_pod_endpoint(&p),
            Some(DiscoveredPod {
                name: "engine-abc".to_string(),
                ip: "10.0.0.1".to_string(),
                engine_hash: None,
            })
        );
    }

    #[test]
    fn engine_hash_label_becomes_tenant() {
        let p = pod_with_labels(
            "engine-abc",
            Some("10.0.0.1"),
            Some("True"),
            false,
            &[("engine_hash", "975d72d83374a79b"), ("engine", "pytrtllm")],
        );
        let discovered = ready_pod_endpoint(&p).unwrap();
        assert_eq!(discovered.engine_hash.as_deref(), Some("975d72d83374a79b"));
        assert_eq!(discovered.tenant_id("default"), "975d72d83374a79b");
    }

    #[test]
    fn missing_engine_hash_falls_back_to_configured_tenant() {
        let p = pod_with_labels(
            "engine-abc",
            Some("10.0.0.1"),
            Some("True"),
            false,
            &[("engine", "pytrtllm")],
        );
        let discovered = ready_pod_endpoint(&p).unwrap();
        assert_eq!(discovered.engine_hash, None);
        assert_eq!(discovered.tenant_id("default"), "default");
    }

    #[test]
    fn not_ready_pod_is_skipped() {
        let p = pod("engine-abc", Some("10.0.0.1"), Some("False"), false);
        assert_eq!(ready_pod_endpoint(&p), None);
    }

    #[test]
    fn ready_pod_without_ip_is_skipped() {
        let p = pod("engine-abc", None, Some("True"), false);
        assert_eq!(ready_pod_endpoint(&p), None);
    }

    #[test]
    fn terminating_pod_is_skipped_even_if_ready() {
        let p = pod("engine-abc", Some("10.0.0.1"), Some("True"), true);
        assert_eq!(ready_pod_endpoint(&p), None);
    }

    #[test]
    fn instance_id_is_stable_and_name_specific() {
        assert_eq!(instance_id_for("engine-abc"), instance_id_for("engine-abc"));
        assert_ne!(instance_id_for("engine-abc"), instance_id_for("engine-xyz"));
    }
}
