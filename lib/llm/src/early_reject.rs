// SPDX-FileCopyrightText: Copyright (c) 2026 DeepInfra
// SPDX-License-Identifier: Apache-2.0

//! Cheap repeat-rejection for prompts the router has just refused.
//!
//! A cold prompt is rejected by the router only after it has been templated and tokenized, which
//! costs hundreds of milliseconds of CPU for a prompt of a few hundred thousand tokens. Clients
//! that retry at once make the frontend pay that again for the same answer. This module remembers
//! a fingerprint of each request the router rejected for a class whose name ends in one of the
//! configured suffixes, and for a short time answers an identical request with the same rejection
//! before any preprocessing.
//!
//! The fingerprint covers the model, the messages and the tools, so only a byte-for-byte repeat of
//! a rejected request matches. An entry is never refreshed by an early rejection: once it expires,
//! the next repeat goes through the router again, so at most one full evaluation happens per
//! request per time-to-live.
//!
//! Disabled unless `DYN_ROUTER_EARLY_REJECT_TTL_MS` is set to a positive number.

use std::io::Write;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use dynamo_kv_router::scheduling::QueueRejection;
use serde::Serialize;
use xxhash_rust::xxh3::Xxh3;

const TTL_ENV: &str = "DYN_ROUTER_EARLY_REJECT_TTL_MS";
const SUFFIXES_ENV: &str = "DYN_ROUTER_EARLY_REJECT_CLASS_SUFFIXES";
const DEFAULT_SUFFIXES: &str = "-cold";
const MAX_ENTRIES: usize = 262_144;
const LOG_EVERY: u64 = 1_000;

pub struct EarlyReject {
    ttl: Duration,
    suffixes: Vec<String>,
    entries: DashMap<u64, (Instant, QueueRejection)>,
    hits: AtomicU64,
}

struct HashWriter<'a>(&'a mut Xxh3);

impl Write for HashWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.update(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl EarlyReject {
    pub fn new(ttl: Duration, suffixes: Vec<String>) -> Self {
        Self {
            ttl,
            suffixes,
            entries: DashMap::new(),
            hits: AtomicU64::new(0),
        }
    }

    fn from_env() -> Self {
        let ttl_ms = std::env::var(TTL_ENV)
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(0);
        let suffixes = std::env::var(SUFFIXES_ENV)
            .unwrap_or_else(|_| DEFAULT_SUFFIXES.to_string())
            .split(',')
            .map(|suffix| suffix.trim().to_string())
            .filter(|suffix| !suffix.is_empty())
            .collect();
        Self::new(Duration::from_millis(ttl_ms), suffixes)
    }

    pub fn enabled(&self) -> bool {
        !self.ttl.is_zero() && !self.suffixes.is_empty()
    }

    /// Fingerprint of a request: the model plus the serialized content (pass a tuple for
    /// several parts).
    pub fn fingerprint<T: Serialize + ?Sized>(model: &str, content: &T) -> Option<u64> {
        let mut hasher = Xxh3::new();
        hasher.update(model.as_bytes());
        hasher.update(&[0xff]);
        serde_json::to_writer(HashWriter(&mut hasher), content).ok()?;
        Some(hasher.digest())
    }

    /// The rejection to repeat for this fingerprint, if the router issued one within the
    /// time-to-live.
    pub fn check(&self, fingerprint: u64) -> Option<QueueRejection> {
        self.check_at(fingerprint, Instant::now())
    }

    fn check_at(&self, fingerprint: u64, now: Instant) -> Option<QueueRejection> {
        if !self.enabled() {
            return None;
        }
        let rejection = {
            let entry = self.entries.get(&fingerprint)?;
            let (recorded_at, rejection) = entry.value();
            (now.saturating_duration_since(*recorded_at) < self.ttl).then(|| rejection.clone())
        };
        if rejection.is_none() {
            self.entries.remove(&fingerprint);
            return None;
        }
        let hits = self.hits.fetch_add(1, Ordering::Relaxed) + 1;
        if hits % LOG_EVERY == 0 {
            tracing::info!(
                hits,
                entries = self.entries.len(),
                "early rejection: repeated requests answered before preprocessing"
            );
        }
        rejection
    }

    /// Remember a router rejection when its class is one this module covers.
    pub fn record(&self, fingerprint: u64, rejection: &QueueRejection) {
        self.record_at(fingerprint, rejection, Instant::now());
    }

    fn record_at(&self, fingerprint: u64, rejection: &QueueRejection, now: Instant) {
        if !self.enabled()
            || !self
                .suffixes
                .iter()
                .any(|suffix| rejection.policy_class.ends_with(suffix.as_str()))
        {
            return;
        }
        if self.entries.len() >= MAX_ENTRIES {
            let ttl = self.ttl;
            self.entries
                .retain(|_, (recorded_at, _)| now.saturating_duration_since(*recorded_at) < ttl);
            if self.entries.len() >= MAX_ENTRIES {
                return;
            }
        }
        self.entries.insert(fingerprint, (now, rejection.clone()));
    }

    /// Record the queue rejection found in an error chain, if there is one.
    pub fn record_error(&self, fingerprint: u64, err: &anyhow::Error) {
        if !self.enabled() {
            return;
        }
        if let Some(rejection) = err
            .chain()
            .find_map(|cause| cause.downcast_ref::<QueueRejection>())
        {
            self.record(fingerprint, rejection);
        }
    }

    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }
}

static GLOBAL: LazyLock<EarlyReject> = LazyLock::new(EarlyReject::from_env);

pub fn global() -> &'static EarlyReject {
    &GLOBAL
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::scheduling::QueueLimitKind;

    fn rejection(class: &str) -> QueueRejection {
        QueueRejection {
            policy_class: class.to_string(),
            limit_kind: QueueLimitKind::Requests,
            current: 4,
            limit: 4,
        }
    }

    fn enabled() -> EarlyReject {
        EarlyReject::new(Duration::from_millis(2_000), vec!["-cold".to_string()])
    }

    #[test]
    fn disabled_without_ttl() {
        let gate = EarlyReject::new(Duration::ZERO, vec!["-cold".to_string()]);
        gate.record(1, &rejection("no-queue-cold"));
        assert!(gate.check(1).is_none());
    }

    #[test]
    fn repeats_a_cold_rejection_until_it_expires() {
        let gate = enabled();
        let start = Instant::now();
        gate.record_at(7, &rejection("no-queue-cold"), start);

        let repeated = gate.check_at(7, start + Duration::from_millis(1_500));
        assert_eq!(repeated, Some(rejection("no-queue-cold")));
        assert_eq!(gate.hits(), 1);

        // An early rejection does not refresh the entry.
        assert!(gate.check_at(7, start + Duration::from_millis(2_000)).is_none());
        assert!(gate.check_at(7, start + Duration::from_millis(2_001)).is_none());
        assert_eq!(gate.hits(), 1);
    }

    #[test]
    fn ignores_rejections_of_other_classes() {
        let gate = enabled();
        gate.record(9, &rejection("no-queue"));
        gate.record(10, &rejection("no-queue-new"));
        assert!(gate.check(9).is_none());
        assert!(gate.check(10).is_none());
    }

    #[test]
    fn finds_the_rejection_in_an_error_chain() {
        let gate = enabled();
        let err = anyhow::Error::new(rejection("no-queue-cold")).context("remote prefill failed");
        gate.record_error(11, &err);
        assert!(gate.check(11).is_some());

        gate.record_error(12, &anyhow::anyhow!("some other failure"));
        assert!(gate.check(12).is_none());
    }

    #[test]
    fn fingerprint_changes_with_any_part() {
        let messages_a = serde_json::json!([{"role": "user", "content": "hello"}]);
        let messages_b = serde_json::json!([{"role": "user", "content": "hello!"}]);
        let tools = serde_json::json!([{"type": "function"}]);
        let none = serde_json::Value::Null;

        let base = EarlyReject::fingerprint("m", &(&messages_a, &none)).unwrap();
        assert_eq!(
            base,
            EarlyReject::fingerprint("m", &(&messages_a, &none)).unwrap()
        );
        assert_ne!(
            base,
            EarlyReject::fingerprint("m", &(&messages_b, &none)).unwrap()
        );
        assert_ne!(
            base,
            EarlyReject::fingerprint("m", &(&messages_a, &tools)).unwrap()
        );
        assert_ne!(
            base,
            EarlyReject::fingerprint("other", &(&messages_a, &none)).unwrap()
        );
    }
}
