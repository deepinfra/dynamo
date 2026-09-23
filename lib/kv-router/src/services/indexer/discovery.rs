// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for Kubernetes pod auto-discovery.
//!
//! Plain data, always compiled, so the CLI can build it without the
//! `kube-discovery` feature; only the feature-gated `pod_watcher` consumes it.

/// Label the backend stamps on every engine pod with the sanitized model name.
/// Stable across GPU configs and engine versions, unlike `engine_hash`, which
/// fragments per GPU config (B200 vs B300 replicas of one model differ).
pub const MODEL_NAME_LABEL: &str = "di/model_name";

/// The backend's `di/model_name` label value for `name`: lowercase, then every
/// char not in `[a-z0-9-]` becomes `-`. Must match the backend's
/// `re.sub('[^a-z0-9-]', '-', name.lower())` exactly, or the selector matches
/// zero pods.
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

/// Selector matching every engine pod of `model_name`, across engine_hash
/// variants.
pub fn model_name_label_selector(model_name: &str) -> String {
    format!(
        "{MODEL_NAME_LABEL}={}",
        sanitize_model_name_label(model_name)
    )
}

#[derive(Debug, Clone)]
pub struct KubeDiscoveryConfig {
    /// Namespace to watch for engine pods.
    pub namespace: String,
    /// Selector picking out this model's engine pods: derived from the model
    /// name, or supplied raw (e.g. `engine_hash=d4b7a85131172ca6`).
    pub label_selector: String,
    /// KV-event ZMQ port; data-parallel rank `r` publishes on `zmq_port + r`.
    pub zmq_port: u16,
    /// Port serving `GET /kv_recover`; rank `r` serves on `recover_port + r`.
    pub recover_port: Option<u16>,
    /// Data-parallel ranks per engine pod, each registered as a listener of
    /// the pod's instance.
    pub dp_size: u32,
    /// Model name discovered pods are registered under.
    pub model_name: String,
    /// Routing group for pods without an `engine_hash` label.
    pub routing_group: String,
    /// KV cache block size of the discovered engines.
    pub block_size: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sanitization contract with the backend, verbatim.
    #[test]
    fn model_name_label_selector_matches_backend_contract() {
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
    fn sanitize_replaces_every_disallowed_char() {
        assert_eq!(sanitize_model_name_label("A_b.c/Dé9-"), "a-b-c-d-9-");
        assert_eq!(sanitize_model_name_label("gpt-oss-20b"), "gpt-oss-20b");
    }
}
