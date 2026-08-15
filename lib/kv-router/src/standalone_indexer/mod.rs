// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone HTTP KV-cache indexer.
//!
//! Hosts an Axum HTTP server with `/register`, `/unregister`, `/query`,
//! `/query_by_hash`, and peer-discovery routes that workers / gateways can
//! call to drive cache-aware routing decisions. Each registered worker spawns
//! a ZMQ listener that ingests its KV events into a per-(model, tenant)
//! [`indexer::Indexer`].
//!
//! ## Multi-tier responses
//!
//! `/query` and `/query_by_hash` return both:
//! - the legacy flat `scores`/`frequencies`/`tree_sizes` (device-tier overlap),
//!   for backward compatibility, and
//! - a per-instance `instances` map keyed by `instance_id` with `gpu`, `cpu`,
//!   `disk`, per-`dp_rank` device counts, and `longest_matched`.
//!
//! The `instances` shape is intended to align with Mooncake's
//! "[RFC]: KV-Store Indexer API Standardization"
//! (<https://github.com/kvcache-ai/Mooncake/issues/1403>).
//! Tier counts are CUMULATIVE through each tier's walk — see the doc on the
//! response struct in [`server`] for the exact semantics.

pub mod evictions;
pub mod h24;
pub mod indexer;
pub mod listener;
pub mod metrics;
#[cfg(feature = "kube-discovery")]
pub mod pod_watcher;
pub mod recovery;
pub mod registry;
pub mod server;
mod zmq;

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::config::min_initial_workers_from_env;
use registry::WorkerRegistry;
use server::{AppState, create_router};

// Process-global measurement-mode filter flags.
//
// Set once at startup by `run_server` from `IndexerConfig` and then read
// in the hot listener path.  `OnceLock` is free after the first `get()`
// (it is a thin wrapper around `Option`) and suits flags that are written
// exactly once and read many times.
static KEEP_EVICTIONS: OnceLock<bool> = OnceLock::new();
static ENABLE_LOGGING: OnceLock<bool> = OnceLock::new();

/// Returns `true` when this indexer instance parks `Removed` events in the
/// per-listener pending-evictions buffer (and drops `Cleared`) instead of
/// applying them, approximating a tree with infinite memory capacity. Aged
/// entries are replayed under memory pressure — see [`evictions`].
pub(crate) fn keep_evictions() -> bool {
    *KEEP_EVICTIONS.get().unwrap_or(&false)
}

/// Returns `true` when verbose audit logging is enabled. When on, the query
/// endpoint logs each query (block hashes + full indexer response) and the
/// listener logs every store/evict/clear event ingested from the engine.
/// All audit lines use the `kv_audit` tracing target so they can be filtered.
pub(crate) fn logging_enabled() -> bool {
    *ENABLE_LOGGING.get().unwrap_or(&false)
}

/// Wall-clock timestamp in milliseconds since the Unix epoch, for audit logs.
/// Returns 0 if the clock is before the epoch (should never happen).
pub(crate) fn now_unix_millis() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Label key the backend stamps on every engine pod with the sanitized model
/// name. Stable across GPU configs and engine versions, unlike `engine_hash`,
/// which fragments per GPU config (B200 vs B300 replicas of the same model
/// get different hashes).
pub const MODEL_NAME_LABEL: &str = "di/model_name";

/// Sanitize a model name into the backend's `di/model_name` label value:
/// lowercase, then every char NOT in `[a-z0-9-]` becomes `-`. This MUST match
/// the backend's `re.sub('[^a-z0-9-]', '-', name.lower())` exactly — if the
/// two drift, the derived selector matches zero pods.
pub fn sanitize_model_name_label(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// Kubernetes label selector matching every engine pod of `model_name`,
/// across all engine_hash variants: `di/model_name=<sanitized>`.
pub fn model_name_label_selector(model_name: &str) -> String {
    format!(
        "{MODEL_NAME_LABEL}={}",
        sanitize_model_name_label(model_name)
    )
}

/// Configuration for Kubernetes pod auto-discovery.
///
/// Plain data (no `kube` types) so it can live in the always-compiled config
/// even when the `kube-discovery` feature is off; it is only *consumed* by the
/// feature-gated [`pod_watcher`] module.
#[derive(Debug, Clone)]
pub struct KubeDiscoveryConfig {
    /// Namespace to watch for engine pods.
    pub namespace: String,
    /// Label selector that picks out this model's engine pods. Either derived
    /// from the model name (`di/model_name=<sanitized>`, spans every
    /// engine_hash) or supplied raw, e.g. `engine_hash=d4b7a85131172ca6`.
    pub label_selector: String,
    /// ZMQ KV-event port the engines publish on (e.g. 5557).
    pub zmq_port: u16,
    /// Optional HTTP port serving `GET /kv_recover` on the engines, used for
    /// per-worker gap recovery. `http://<pod-ip>:<recover_port>` is the base
    /// URL the indexer queries on a detected gap.
    pub recover_port: Option<u16>,
    /// Model name discovered pods are registered under.
    pub model_name: String,
    /// Tenant id discovered pods are registered under.
    pub tenant_id: String,
    /// KV cache block size for discovered engines.
    pub block_size: u32,
}

pub struct IndexerConfig {
    pub block_size: Option<u32>,
    pub port: u16,
    pub threads: usize,
    pub workers: Option<String>,
    pub model_name: String,
    pub tenant_id: String,
    pub peers: Option<String>,
    /// When set, watch Kubernetes and auto-register/deregister engine pods.
    pub kube_discovery: Option<KubeDiscoveryConfig>,
    /// Park `Removed` events in a buffer (and drop `Cleared`) instead of
    /// applying them, approximating infinite memory for the "ideal ceiling"
    /// measurement mode. Entries older than `evict_retention_secs` are
    /// replayed into the tree once memory usage crosses
    /// `evict_memory_threshold` of the cgroup limit.
    pub keep_evictions: bool,
    /// Minimum age (seconds) a parked eviction must reach before the sweep
    /// may replay it. Only meaningful with `keep_evictions`.
    pub evict_retention_secs: u64,
    /// Fraction of the memory limit (0..1] above which the sweep replays
    /// aged evictions. Only meaningful with `keep_evictions`.
    pub evict_memory_threshold: f64,
    /// Emit verbose per-query and per-event audit logs on the `kv_audit`
    /// tracing target. See [`logging_enabled`].
    pub enable_logging: bool,
    /// Run every indexer as the flat h24 counterfactual backend: ignore
    /// evictions and worker identity, track per-block last-seen timestamps,
    /// and report matches under a single synthetic worker. Mutually
    /// exclusive with `keep_evictions`.
    pub h24: bool,
    /// Retention horizon for the h24 expiry sweep (seconds). Entries not
    /// stored or touched within this window are dropped. Only meaningful
    /// with `h24`.
    pub h24_horizon_secs: u64,
}

pub(super) fn validate_zmq_endpoint(endpoint: &str) -> anyhow::Result<()> {
    let (scheme, address) = endpoint
        .split_once("://")
        .ok_or_else(|| anyhow::anyhow!("invalid ZMQ endpoint `{endpoint}`: missing scheme"))?;

    if address.is_empty() {
        anyhow::bail!("invalid ZMQ endpoint `{endpoint}`: missing address");
    }

    match scheme {
        "tcp" => {
            let (host, port) = address.rsplit_once(':').ok_or_else(|| {
                anyhow::anyhow!("invalid ZMQ endpoint `{endpoint}`: missing TCP port")
            })?;
            if host.is_empty() {
                anyhow::bail!("invalid ZMQ endpoint `{endpoint}`: missing TCP host");
            }
            if host.starts_with('[') {
                if !host.ends_with(']') {
                    anyhow::bail!("invalid ZMQ endpoint `{endpoint}`: missing closing `]`");
                }
            } else if host.contains(':') {
                anyhow::bail!("invalid ZMQ endpoint `{endpoint}`: missing TCP port");
            }
            port.parse::<u16>().map_err(|error| {
                anyhow::anyhow!("invalid ZMQ endpoint `{endpoint}`: invalid TCP port: {error}")
            })?;
            Ok(())
        }
        "ipc" | "inproc" => Ok(()),
        other => Err(anyhow::anyhow!(
            "invalid ZMQ endpoint `{endpoint}`: unsupported scheme `{other}`"
        )),
    }
}

/// Validate a per-worker recovery endpoint: the base URL the indexer queries
/// at `<recover_endpoint>/kv_recover` on a detected gap. Must be an absolute
/// `http`/`https` URL with a host.
pub(super) fn validate_recover_endpoint(endpoint: &str) -> anyhow::Result<()> {
    let url = reqwest::Url::parse(endpoint)
        .map_err(|error| anyhow::anyhow!("invalid recover endpoint `{endpoint}`: {error}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        anyhow::bail!(
            "invalid recover endpoint `{endpoint}`: scheme must be http or https, got `{}`",
            url.scheme()
        );
    }
    if url.host().is_none() {
        anyhow::bail!("invalid recover endpoint `{endpoint}`: missing host");
    }
    Ok(())
}

pub(super) fn validate_listener_endpoints(
    endpoint: &str,
    recover_endpoint: Option<&str>,
) -> anyhow::Result<()> {
    validate_zmq_endpoint(endpoint)?;
    if let Some(recover_endpoint) = recover_endpoint {
        validate_recover_endpoint(recover_endpoint)?;
    }
    Ok(())
}

pub fn parse_workers(s: &str) -> anyhow::Result<Vec<(u64, u32, String)>> {
    let mut workers = Vec::new();

    for entry in s.split(',').filter(|entry| !entry.trim().is_empty()) {
        let (id_part, addr) = entry.split_once('=').ok_or_else(|| {
            anyhow::anyhow!("invalid worker entry `{entry}`; expected worker_id[:dp_rank]=endpoint")
        })?;
        let id_part = id_part.trim();
        let (instance_id, dp_rank) = if let Some((id_str, rank_str)) = id_part.split_once(':') {
            (
                id_str
                    .parse::<u64>()
                    .map_err(|error| anyhow::anyhow!("invalid worker id in `{entry}`: {error}"))?,
                rank_str
                    .parse::<u32>()
                    .map_err(|error| anyhow::anyhow!("invalid dp_rank in `{entry}`: {error}"))?,
            )
        } else {
            (
                id_part
                    .parse::<u64>()
                    .map_err(|error| anyhow::anyhow!("invalid worker id in `{entry}`: {error}"))?,
                0,
            )
        };

        let endpoint = addr.trim().to_string();
        validate_zmq_endpoint(&endpoint)?;
        workers.push((instance_id, dp_rank, endpoint));
    }

    Ok(workers)
}

pub async fn run_server(config: IndexerConfig) -> anyhow::Result<()> {
    // Set process-global filter flags before spawning any listeners.
    // `set` returns Err if already set; that should never happen since
    // run_server is called once, so we discard the result.
    let _ = KEEP_EVICTIONS.set(config.keep_evictions);
    let _ = ENABLE_LOGGING.set(config.enable_logging);
    if config.enable_logging {
        tracing::info!(
            target: "kv_audit",
            "kv_audit logging enabled: queries and store/evict/clear events will be logged"
        );
    }

    let cancel_token = CancellationToken::new();
    let shutdown_token = cancel_token.clone();
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        tracing::info!("Received shutdown signal");
        shutdown_token.cancel();
    });

    let peers: Vec<String> = config
        .peers
        .as_deref()
        .map(|s| {
            s.split(',')
                .filter(|p| !p.is_empty())
                .map(|p| p.trim().to_string())
                .collect()
        })
        .unwrap_or_default();

    tracing::info!(
        block_size = ?config.block_size,
        port = config.port,
        threads = config.threads,
        model_name = %config.model_name,
        tenant_id = %config.tenant_id,
        num_peers = peers.len(),
        "Starting standalone KV cache indexer (HTTP-only mode)"
    );

    let registry = Arc::new(WorkerRegistry::new(config.threads));

    if config.h24 {
        anyhow::ensure!(
            !config.keep_evictions,
            "--h24 and --keep-evictions are mutually exclusive"
        );
        anyhow::ensure!(
            config.h24_horizon_secs > 0,
            "--h24-horizon-secs must be positive"
        );
        registry.enable_h24();
        h24::spawn_expiry_loop(
            registry.clone(),
            config.h24_horizon_secs,
            cancel_token.clone(),
        );
        tracing::info!(
            horizon_secs = config.h24_horizon_secs,
            "h24 mode enabled: flat counterfactual indexer, evictions ignored"
        );
    }

    if config.keep_evictions {
        anyhow::ensure!(
            config.evict_memory_threshold > 0.0 && config.evict_memory_threshold <= 1.0,
            "--evict-memory-threshold must be in (0, 1], got {}",
            config.evict_memory_threshold
        );
        evictions::spawn_cleanup_loop(
            registry.clone(),
            config.evict_retention_secs,
            config.evict_memory_threshold,
            cancel_token.clone(),
        );
        tracing::info!(
            retention_secs = config.evict_retention_secs,
            memory_threshold = config.evict_memory_threshold,
            "keep-evictions enabled: parking eviction events, sweeping under memory pressure"
        );
    }

    run_common(&config, &registry, cancel_token).await
}

async fn wait_for_min_initial_workers(
    registry: &WorkerRegistry,
    cancel_token: &CancellationToken,
) -> anyhow::Result<()> {
    let min_initial_workers = min_initial_workers_from_env()?;
    if min_initial_workers == 0 {
        return Ok(());
    }

    loop {
        let registered_workers = registry.list().len();
        if registered_workers >= min_initial_workers {
            return Ok(());
        }

        tokio::select! {
            _ = cancel_token.cancelled() => {
                anyhow::bail!(
                    "shutdown triggered before {} indexer workers appeared",
                    min_initial_workers
                );
            }
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
        }
    }
}

async fn run_common(
    config: &IndexerConfig,
    registry: &Arc<WorkerRegistry>,
    cancel_token: CancellationToken,
) -> anyhow::Result<()> {
    if let Some(ref workers_str) = config.workers {
        let block_size = config.block_size.ok_or_else(|| {
            anyhow::anyhow!("--block-size is required when --workers is specified")
        })?;
        for (instance_id, dp_rank, endpoint) in parse_workers(workers_str)? {
            tracing::info!(instance_id, dp_rank, endpoint, "Registering initial worker");
            registry
                .register(
                    instance_id,
                    endpoint,
                    dp_rank,
                    config.model_name.clone(),
                    config.tenant_id.clone(),
                    block_size,
                    None,
                    None,
                )
                .await?;
        }
    }

    let peers: Vec<String> = config
        .peers
        .as_deref()
        .map(|s| {
            s.split(',')
                .filter(|p| !p.is_empty())
                .map(|p| p.trim().to_string())
                .collect()
        })
        .unwrap_or_default();

    if !peers.is_empty() {
        match recovery::recover_from_peers(&peers, registry).await {
            Ok(true) => tracing::info!("P2P recovery completed"),
            Ok(false) => tracing::warn!("no reachable peers, starting with empty state"),
            Err(e) => tracing::warn!(error = %e, "P2P recovery failed, starting with empty state"),
        }
        for peer in &peers {
            registry.register_peer(peer.clone());
        }
    }

    wait_for_min_initial_workers(registry, &cancel_token).await?;
    registry.signal_ready();

    // Optional Kubernetes pod auto-discovery. Runs on its own background task so
    // it never blocks the HTTP server; stops when `cancel_token` fires.
    if let Some(kube_config) = config.kube_discovery.clone() {
        #[cfg(feature = "kube-discovery")]
        pod_watcher::spawn_pod_watcher(kube_config, registry.clone(), cancel_token.clone());
        #[cfg(not(feature = "kube-discovery"))]
        {
            let _ = kube_config;
            tracing::warn!(
                "kube_discovery is configured but this binary was built without the \
                 `kube-discovery` feature; pod auto-discovery is disabled"
            );
        }
    }

    #[cfg(feature = "metrics")]
    let prom_registry = {
        let r = prometheus::Registry::new();
        metrics::register(&r).expect("failed to register indexer metrics");
        r
    };

    let state = Arc::new(AppState {
        registry: registry.clone(),
        #[cfg(feature = "metrics")]
        prom_registry,
    });

    let app = create_router(state);
    let listener = TcpListener::bind(("0.0.0.0", config.port)).await?;
    tracing::info!("HTTP server listening on 0.0.0.0:{}", config.port);
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            cancel_token.cancelled().await;
            tracing::info!("Received shutdown signal, stopping HTTP server");
        })
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sanitization contract with the backend, verbatim. The backend
    /// stamps pods with `re.sub('[^a-z0-9-]', '-', name.lower())`; if these
    /// exact outputs drift from that, the derived selector matches zero pods.
    #[test]
    fn test_model_name_label_selector_matches_backend_contract() {
        assert_eq!(
            model_name_label_selector("openai/gpt-oss-120b"),
            "di/model_name=openai-gpt-oss-120b"
        );
        assert_eq!(
            model_name_label_selector("meta-llama/Llama-3.3-70B"),
            "di/model_name=meta-llama-llama-3-3-70b"
        );
        assert_eq!(
            model_name_label_selector("Qwen/Qwen2.5-72B-Instruct"),
            "di/model_name=qwen-qwen2-5-72b-instruct"
        );
    }

    #[test]
    fn test_sanitize_model_name_label_replaces_every_disallowed_char() {
        // '/', '.', uppercase, '_', and non-ASCII all become '-'.
        assert_eq!(sanitize_model_name_label("A_b.c/Dé9-"), "a-b-c-d-9-");
        // Already-clean names pass through untouched.
        assert_eq!(sanitize_model_name_label("gpt-oss-20b"), "gpt-oss-20b");
    }

    #[test]
    fn test_parse_workers() {
        let input = "1=tcp://host:5557,2:1=tcp://host:5558";
        let result = parse_workers(input).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (1, 0, "tcp://host:5557".to_string()));
        assert_eq!(result[1], (2, 1, "tcp://host:5558".to_string()));
    }

    #[test]
    fn test_parse_workers_empty() {
        assert!(parse_workers("").unwrap().is_empty());
    }

    #[test]
    fn test_parse_workers_invalid_entry() {
        let error = parse_workers("1").unwrap_err().to_string();
        assert!(error.contains("invalid worker entry"));
    }

    #[test]
    fn test_validate_zmq_endpoint_allows_wildcard_tcp_bind() {
        validate_zmq_endpoint("tcp://*:5558").unwrap();
        validate_zmq_endpoint("tcp://127.0.0.1:0").unwrap();
        validate_zmq_endpoint("inproc://listener").unwrap();
        validate_zmq_endpoint("ipc:///tmp/dynamo.sock").unwrap();
    }

    #[test]
    fn test_validate_zmq_endpoint_rejects_invalid_values() {
        assert!(validate_zmq_endpoint("tcp://host").is_err());
        assert!(validate_zmq_endpoint("tcp://:5558").is_err());
        assert!(validate_zmq_endpoint("udp://host:5558").is_err());
        assert!(validate_zmq_endpoint("not-an-endpoint").is_err());
    }
}
