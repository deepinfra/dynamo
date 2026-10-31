// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What-if KV cache: the prefix hit rate a global, unbounded cache would reach if it
//! kept every block for a window W after it was last stored or hit, for each W in
//! [`IDEAL_CACHE_WINDOWS_ENV`] (e.g. `10m,1h,6h,24h`). Off when unset.
//!
//! The KV index sources feed it: every block a worker stores, prompt or output,
//! from any frontend's traffic, is stamped with the current minute. Each routed
//! prompt (not a decode hop) then counts, for each W, its leading blocks stamped
//! under W minutes ago, and restamps them; it adds no block of its own. Blocks are
//! keyed by the worker's block hash, which the prompt side computes from its tokens
//! with the workers' hash scheme ([`BLOCK_HASH_ENV`]). Block hashes chain, so a
//! block is never newer than its parent, and one pass answers every window.
//! Per-minute counts give, for each window, the blocks such a cache would hold:
//! the capacity behind its hit rate. Each prefill or aggregated router has its
//! own tracker; both feeds apply in arrival order on its thread.
//!
//! Each entry is one u64 in an open-addressing table: the block's 48-bit
//! fingerprint and the 16-bit minute it was last stamped. A cursor removes entries
//! older than the longest window in small batches, one pass every 10 minutes. An
//! entry not yet removed reads as a miss, so trimming bounds memory, not accuracy.

use std::sync::{Arc, OnceLock, mpsc};
use std::time::{Duration, Instant};

use dynamo_kv_router::protocols::KvCacheStoreData;
use dynamo_runtime::{component::Component, metrics::MetricsHierarchy};
use prometheus::{IntCounter, IntCounterVec, IntGauge, IntGaugeVec};

use super::vllm_block_hash::{BLOCK_HASH_ENV, BlockHashScheme, VllmBlockHasher};

/// Windows to measure, comma-separated: `10m,1h,6h,24h`. Unset or empty: off.
pub const IDEAL_CACHE_WINDOWS_ENV: &str = "DYN_ROUTER_IDEAL_CACHE_WINDOWS";

/// Longest window accepted, in minutes.
const MAX_WINDOW_MINUTES: u16 = 48 * 60;
/// Minute counters kept; more than the longest window plus a trim pass.
const RING_MINUTES: usize = 4096;
/// How long one trim pass over the whole table takes.
const TRIM_PASS: Duration = Duration::from_secs(600);
const TICK: Duration = Duration::from_millis(250);
/// Inputs waiting for the tracker thread; beyond it they are dropped and counted.
const QUEUE_INPUTS: usize = 4096;
const MIN_SLOTS: usize = 1 << 16;

/// One routed prompt: its full blocks' worker block hashes.
struct Prompt {
    hashes: Vec<u64>,
    block_size: u32,
    tokens: usize,
}

enum Input {
    /// Hashes of blocks a worker stored.
    Stored(Vec<u64>),
    Prompt(Prompt),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Window {
    label: String,
    minutes: u16,
}

fn parse_windows(value: &str) -> Result<Vec<Window>, String> {
    let mut windows = Vec::new();
    for label in value
        .split(',')
        .map(str::trim)
        .filter(|label| !label.is_empty())
    {
        let (number, unit) = label.split_at(label.len() - 1);
        let number: u32 = number
            .parse()
            .map_err(|_| format!("window {label:?}: expected e.g. 10m or 6h"))?;
        let minutes = match unit {
            "m" => number,
            "h" => number.saturating_mul(60),
            _ => return Err(format!("window {label:?}: unit must be m or h")),
        };
        if minutes == 0 || minutes > MAX_WINDOW_MINUTES as u32 {
            return Err(format!(
                "window {label:?}: must be 1 minute to {} hours",
                MAX_WINDOW_MINUTES / 60
            ));
        }
        windows.push(Window {
            label: label.to_string(),
            minutes: minutes as u16,
        });
    }
    windows.sort_by_key(|window| window.minutes);
    windows.dedup_by_key(|window| window.minutes);
    Ok(windows)
}

/// Block fingerprint: the hash's top 48 bits, never 0 (an empty slot).
fn fingerprint(hash: u64) -> u64 {
    (hash >> 16).max(1)
}

/// Open addressing with linear probing over packed `fingerprint << 16 | minute`
/// entries; 0 marks an empty slot.
struct Table {
    slots: Vec<u64>,
    len: usize,
}

impl Table {
    fn new() -> Self {
        Self {
            slots: vec![0; MIN_SLOTS],
            len: 0,
        }
    }

    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    fn home(&self, fingerprint: u64) -> usize {
        fingerprint as usize & self.mask()
    }

    /// Sets `fingerprint`'s minute to `minute`; returns the minute it had.
    fn exchange(&mut self, fingerprint: u64, minute: u16) -> Option<u16> {
        if (self.len + 1) * 4 > self.slots.len() * 3 {
            self.grow();
        }
        let entry = fingerprint << 16 | minute as u64;
        let mask = self.mask();
        let mut slot = self.home(fingerprint);
        loop {
            let held = self.slots[slot];
            if held == 0 {
                self.slots[slot] = entry;
                self.len += 1;
                return None;
            }
            if held >> 16 == fingerprint {
                self.slots[slot] = entry;
                return Some(held as u16);
            }
            slot = (slot + 1) & mask;
        }
    }

    /// The minute `fingerprint` was last stamped, if it is held.
    fn find(&self, fingerprint: u64) -> Option<u16> {
        let mask = self.mask();
        let mut slot = self.home(fingerprint);
        loop {
            let held = self.slots[slot];
            if held == 0 {
                return None;
            }
            if held >> 16 == fingerprint {
                return Some(held as u16);
            }
            slot = (slot + 1) & mask;
        }
    }

    /// Sets `fingerprint`'s minute to `minute` if it is held; returns the minute it had.
    fn refresh(&mut self, fingerprint: u64, minute: u16) -> Option<u16> {
        let mask = self.mask();
        let mut slot = self.home(fingerprint);
        loop {
            let held = self.slots[slot];
            if held == 0 {
                return None;
            }
            if held >> 16 == fingerprint {
                self.slots[slot] = fingerprint << 16 | minute as u64;
                return Some(held as u16);
            }
            slot = (slot + 1) & mask;
        }
    }

    /// Empties `slot`, moving later entries of its probe run back so lookups still
    /// find them.
    fn remove_at(&mut self, slot: usize) {
        let mask = self.mask();
        let mut hole = slot;
        let mut next = (slot + 1) & mask;
        loop {
            let entry = self.slots[next];
            if entry == 0 {
                break;
            }
            // The entry may fill the hole unless its home lies after the hole.
            let home = self.home(entry >> 16);
            if next.wrapping_sub(home) & mask >= next.wrapping_sub(hole) & mask {
                self.slots[hole] = entry;
                hole = next;
            }
            next = (next + 1) & mask;
        }
        self.slots[hole] = 0;
        self.len -= 1;
    }

    fn grow(&mut self) {
        let doubled = vec![0; self.slots.len() * 2];
        let old = std::mem::replace(&mut self.slots, doubled);
        let mask = self.mask();
        for entry in old.into_iter().filter(|&entry| entry != 0) {
            let mut slot = self.home(entry >> 16);
            while self.slots[slot] != 0 {
                slot = (slot + 1) & mask;
            }
            self.slots[slot] = entry;
        }
    }
}

/// Everything the tracker thread owns, driven by an explicit minute so tests can
/// move the clock.
struct State {
    windows: Vec<Window>,
    table: Table,
    /// Blocks by the minute of their last use (`minute % RING_MINUTES`).
    last_use: Vec<u64>,
    minute: Option<u16>,
    cursor: usize,
    hit_tokens: Vec<u64>,
    prompt_tokens: u64,
}

impl State {
    fn new(windows: Vec<Window>) -> Self {
        let hit_tokens = vec![0; windows.len()];
        Self {
            windows,
            table: Table::new(),
            last_use: vec![0; RING_MINUTES],
            minute: None,
            cursor: 0,
            hit_tokens,
            prompt_tokens: 0,
        }
    }

    fn ring(minute: u16) -> usize {
        minute as usize % RING_MINUTES
    }

    /// Moves the clock to `minute`, clearing the counters it reuses.
    fn advance(&mut self, minute: u16) {
        let mut at = self.minute.unwrap_or(minute.wrapping_sub(1));
        while at != minute {
            at = at.wrapping_add(1);
            self.last_use[Self::ring(at)] = 0;
        }
        self.minute = Some(minute);
    }

    /// Takes a block out of the count of `minute`, unless that minute is so old its
    /// counter was reused.
    fn forget(&mut self, minute: u16, now: u16) {
        if (now.wrapping_sub(minute) as usize) < RING_MINUTES {
            let count = &mut self.last_use[Self::ring(minute)];
            *count = count.saturating_sub(1);
        }
    }

    /// Moves a block's count from the minute it had to `now`.
    fn restamped(&mut self, old: Option<u16>, now: u16) {
        if let Some(minute) = old {
            self.forget(minute, now);
        }
        self.last_use[Self::ring(now)] += 1;
    }

    /// Stamps blocks a worker stored.
    fn stamp(&mut self, hashes: &[u64], now: u16) {
        self.advance(now);
        for &hash in hashes {
            let old = self.table.exchange(fingerprint(hash), now);
            self.restamped(old, now);
        }
    }

    /// Counts a prompt's hit for each window: its leading blocks stamped under the
    /// window's minutes ago. Restamps those blocks; stores none.
    fn observe(&mut self, prompt: &Prompt, now: u16) {
        self.advance(now);
        let max_age = self.windows.last().map_or(0, |window| window.minutes);
        let mut hits = vec![0u64; self.windows.len()];
        let mut open = self.windows.len();
        for &hash in &prompt.hashes {
            let fingerprint = fingerprint(hash);
            let Some(age) = self
                .table
                .find(fingerprint)
                .map(|minute| now.wrapping_sub(minute))
                .filter(|&age| age < max_age)
            else {
                break;
            };
            // Windows are sorted, so the ones still hitting are a suffix.
            while age >= self.windows[self.windows.len() - open].minutes {
                open -= 1;
            }
            for hit in &mut hits[self.windows.len() - open..] {
                *hit += 1;
            }
            let old = self.table.refresh(fingerprint, now);
            self.restamped(old, now);
        }
        for (tokens, blocks) in self.hit_tokens.iter_mut().zip(hits) {
            *tokens += blocks * prompt.block_size as u64;
        }
        self.prompt_tokens += prompt.tokens as u64;
    }

    /// Moves the cursor past `slots` slots, from where the last call stopped,
    /// removing entries past the longest window.
    fn trim(&mut self, slots: usize, now: u16) {
        self.advance(now);
        let max_age = self.windows.last().map_or(0, |window| window.minutes);
        let mut passed = 0;
        while passed < slots {
            let slot = self.cursor & self.table.mask();
            let entry = self.table.slots[slot];
            if entry != 0 && now.wrapping_sub(entry as u16) >= max_age {
                self.forget(entry as u16, now);
                // The slot may now hold a moved entry: look at it again.
                self.table.remove_at(slot);
            } else {
                self.cursor = (slot + 1) & self.table.mask();
                passed += 1;
            }
        }
    }

    /// Blocks last used under each window's minutes ago.
    fn live_blocks(&self, now: u16) -> Vec<u64> {
        let mut live = Vec::with_capacity(self.windows.len());
        let (mut total, mut age) = (0, 0u16);
        for window in &self.windows {
            while age < window.minutes {
                total += self.last_use[Self::ring(now.wrapping_sub(age))];
                age += 1;
            }
            live.push(total);
        }
        live
    }
}

/// Registered once per process, like the router's other metrics; each router's
/// tracker adds its share.
struct Metrics {
    hit_tokens: IntCounterVec,
    prompt_tokens: IntCounter,
    live_blocks: IntGaugeVec,
    entries: IntGauge,
    table_bytes: IntGauge,
    tracked_seconds: IntGauge,
    dropped: IntCounterVec,
}

static METRICS: OnceLock<Option<Metrics>> = OnceLock::new();

impl Metrics {
    /// The process's metrics, registered on the first router to ask.
    fn get(component: &Component) -> Option<&'static Self> {
        METRICS
            .get_or_init(|| match Self::create(component) {
                Ok(metrics) => Some(metrics),
                Err(error) => {
                    tracing::warn!(%error, "Failed to register ideal KV cache metrics");
                    None
                }
            })
            .as_ref()
    }

    fn create(component: &Component) -> anyhow::Result<Self> {
        let metrics = component.metrics();
        let dropped = metrics.create_intcountervec(
            "router_ideal_cache_dropped_total",
            "Stored-block batches and prompts the ideal KV cache trackers could not keep up with",
            &["input"],
            &[],
        )?;
        for input in ["stored", "prompt"] {
            dropped.with_label_values(&[input]);
        }
        Ok(Self {
            hit_tokens: metrics.create_intcountervec(
                "router_ideal_cache_hit_tokens_total",
                "Prompt tokens a global, unbounded KV cache keeping every stored block for `window` after its last store or hit would hit",
                &["window"],
                &[],
            )?,
            prompt_tokens: metrics.create_intcounter(
                "router_ideal_cache_prompt_tokens_total",
                "Prompt tokens of the requests the ideal KV cache tracks",
                &[],
            )?,
            live_blocks: metrics.create_intgaugevec(
                "router_ideal_cache_live_blocks",
                "Blocks stored or hit within `window`: what the ideal KV cache for that window holds",
                &["window"],
                &[],
            )?,
            entries: metrics.create_intgauge(
                "router_ideal_cache_entries",
                "Blocks the ideal KV cache trackers hold, including some past the longest window",
                &[],
            )?,
            table_bytes: metrics.create_intgauge(
                "router_ideal_cache_table_bytes",
                "Memory of the ideal KV cache trackers' tables",
                &[],
            )?,
            tracked_seconds: metrics.create_intgauge(
                "router_ideal_cache_tracked_seconds",
                "Time the ideal KV cache has been tracking; a window is complete once it exceeds it",
                &[],
            )?,
            dropped,
        })
    }
}

/// A router's ideal KV cache; see the module docs. Its thread stops with the router.
pub struct IdealCache {
    inputs: mpsc::SyncSender<Input>,
    hasher: VllmBlockHasher,
    metrics: &'static Metrics,
}

impl IdealCache {
    /// The tracker of a router on `component`, when [`IDEAL_CACHE_WINDOWS_ENV`] and
    /// the workers' hash scheme ([`BLOCK_HASH_ENV`]) are set.
    pub fn from_env(component: &Component) -> Option<Arc<Self>> {
        let value = std::env::var(IDEAL_CACHE_WINDOWS_ENV).ok()?;
        let windows = match parse_windows(&value) {
            Ok(windows) if !windows.is_empty() => windows,
            Ok(_) => return None,
            Err(error) => {
                tracing::warn!(env = IDEAL_CACHE_WINDOWS_ENV, %value, %error, "Ideal KV cache off");
                return None;
            }
        };
        let Some(hasher) = std::env::var(BLOCK_HASH_ENV)
            .ok()
            .and_then(|scheme| BlockHashScheme::from_env(&scheme)?.hasher())
        else {
            tracing::warn!(
                env = BLOCK_HASH_ENV,
                "Ideal KV cache off: it needs the workers' block hash scheme"
            );
            return None;
        };
        let metrics = Metrics::get(component)?;
        for window in &windows {
            metrics
                .hit_tokens
                .with_label_values(&[window.label.as_str()]);
            metrics
                .live_blocks
                .with_label_values(&[window.label.as_str()]);
        }
        tracing::info!(
            windows = ?windows.iter().map(|window| &window.label).collect::<Vec<_>>(),
            "Tracking the ideal KV cache hit rate"
        );
        let (inputs, received) = mpsc::sync_channel(QUEUE_INPUTS);
        let spawned = std::thread::Builder::new()
            .name("ideal-kv-cache".to_string())
            .spawn(move || run(received, State::new(windows), metrics));
        if let Err(error) = spawned {
            tracing::warn!(%error, "Failed to start the ideal KV cache tracker");
            return None;
        }
        Some(Arc::new(Self {
            inputs,
            hasher,
            metrics,
        }))
    }

    /// Stamps the blocks a worker stored.
    pub fn observe_stored(&self, data: &KvCacheStoreData) {
        self.observe_stored_hashes(data.blocks.iter().map(|block| block.block_hash.0).collect());
    }

    /// Stamps blocks a lower tier stored, named by the workers' block hashes.
    pub fn observe_stored_hashes(&self, hashes: Vec<u64>) {
        if !hashes.is_empty() {
            self.send(Input::Stored(hashes));
        }
    }

    /// Counts a routed prompt. Its full blocks are hashed as the workers hash
    /// them; a prompt whose blocks cannot be is skipped.
    pub fn observe_prompt(&self, block_size: u32, tokens: &[u32]) {
        let blocks = tokens.len() / block_size as usize;
        let Some(hashes) = self
            .hasher
            .chunk_hashes(tokens, block_size as usize, blocks)
        else {
            return;
        };
        self.send(Input::Prompt(Prompt {
            hashes,
            block_size,
            tokens: tokens.len(),
        }));
    }

    fn send(&self, input: Input) {
        let label = match input {
            Input::Stored(_) => "stored",
            Input::Prompt(_) => "prompt",
        };
        if self.inputs.try_send(input).is_err() {
            self.metrics.dropped.with_label_values(&[label]).inc();
        }
    }
}

/// What one tracker has added to the shared metrics.
struct Reported {
    hit_tokens: Vec<u64>,
    prompt_tokens: u64,
    live_blocks: Vec<i64>,
    entries: i64,
    table_bytes: i64,
}

impl Reported {
    /// Brings the shared gauges from this tracker's last values to `live_blocks`,
    /// `entries` and `table_bytes`.
    fn set_gauges(
        &mut self,
        metrics: &Metrics,
        windows: &[Window],
        live_blocks: &[i64],
        entries: i64,
        table_bytes: i64,
    ) {
        for ((window, &live), reported) in windows
            .iter()
            .zip(live_blocks)
            .zip(self.live_blocks.iter_mut())
        {
            metrics
                .live_blocks
                .with_label_values(&[window.label.as_str()])
                .add(live - *reported);
            *reported = live;
        }
        metrics.entries.add(entries - self.entries);
        self.entries = entries;
        metrics.table_bytes.add(table_bytes - self.table_bytes);
        self.table_bytes = table_bytes;
    }
}

fn run(inputs: mpsc::Receiver<Input>, mut state: State, metrics: &'static Metrics) {
    let started = Instant::now();
    let minute = || (started.elapsed().as_secs() / 60) as u16;
    let mut reported = Reported {
        hit_tokens: vec![0; state.windows.len()],
        prompt_tokens: 0,
        live_blocks: vec![0; state.windows.len()],
        entries: 0,
        table_bytes: 0,
    };
    let mut next_tick = Instant::now();
    loop {
        match inputs.recv_timeout(next_tick.saturating_duration_since(Instant::now())) {
            Ok(Input::Stored(hashes)) => state.stamp(&hashes, minute()),
            Ok(Input::Prompt(prompt)) => state.observe(&prompt, minute()),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // The router is gone: take its share out of the gauges.
                let zeros = vec![0; state.windows.len()];
                reported.set_gauges(metrics, &state.windows, &zeros, 0, 0);
                return;
            }
        }
        if Instant::now() < next_tick {
            continue;
        }
        next_tick = Instant::now() + TICK;
        let now = minute();
        let batch = (state.table.slots.len() as u128 * TICK.as_millis() / TRIM_PASS.as_millis())
            .max(1) as usize;
        state.trim(batch, now);

        for (window, (tokens, reported)) in state
            .windows
            .iter()
            .zip(state.hit_tokens.iter().zip(reported.hit_tokens.iter_mut()))
        {
            metrics
                .hit_tokens
                .with_label_values(&[window.label.as_str()])
                .inc_by(tokens - *reported);
            *reported = *tokens;
        }
        metrics
            .prompt_tokens
            .inc_by(state.prompt_tokens - reported.prompt_tokens);
        reported.prompt_tokens = state.prompt_tokens;
        let live: Vec<i64> = state
            .live_blocks(now)
            .into_iter()
            .map(|blocks| blocks as i64)
            .collect();
        let table_bytes = (state.table.slots.len() * std::mem::size_of::<u64>()) as i64;
        reported.set_gauges(
            metrics,
            &state.windows,
            &live,
            state.table.len as i64,
            table_bytes,
        );
        metrics
            .tracked_seconds
            .set(started.elapsed().as_secs() as i64);
    }
}

#[cfg(test)]
impl IdealCache {
    /// A tracker without its thread, for other modules' tests, and a reader of the
    /// block hashes stamped since the last read.
    pub(crate) fn unstarted() -> (Arc<Self>, impl Fn() -> Vec<u64>) {
        static METRICS: OnceLock<Metrics> = OnceLock::new();
        let metrics = METRICS.get_or_init(|| {
            let opts = |name: &str| prometheus::Opts::new(name, name);
            Metrics {
                hit_tokens: IntCounterVec::new(opts("hit_tokens"), &["window"]).unwrap(),
                prompt_tokens: IntCounter::with_opts(opts("prompt_tokens")).unwrap(),
                live_blocks: IntGaugeVec::new(opts("live_blocks"), &["window"]).unwrap(),
                entries: IntGauge::with_opts(opts("entries")).unwrap(),
                table_bytes: IntGauge::with_opts(opts("table_bytes")).unwrap(),
                tracked_seconds: IntGauge::with_opts(opts("tracked_seconds")).unwrap(),
                dropped: IntCounterVec::new(opts("dropped"), &["input"]).unwrap(),
            }
        });
        let hasher = BlockHashScheme::from_env(&format!("sha256:{}", "0".repeat(64)))
            .and_then(|scheme| scheme.hasher())
            .unwrap();
        let (inputs, received) = mpsc::sync_channel(QUEUE_INPUTS);
        let stamped = move || {
            received
                .try_iter()
                .flat_map(|input| match input {
                    Input::Stored(hashes) => hashes,
                    Input::Prompt(_) => Vec::new(),
                })
                .collect()
        };
        let cache = Self {
            inputs,
            hasher,
            metrics,
        };
        (Arc::new(cache), stamped)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn windows(value: &str) -> Vec<Window> {
        parse_windows(value).unwrap()
    }

    fn prompt(hashes: &[u64]) -> Prompt {
        Prompt {
            hashes: hashes.to_vec(),
            block_size: 64,
            tokens: hashes.len() * 64 + 10,
        }
    }

    /// Hashes whose fingerprints are distinct and spread.
    fn hash(n: u64) -> u64 {
        n.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 << 63
    }

    #[test]
    fn windows_parse_sorted_and_bounded() {
        assert_eq!(
            windows(" 24h, 10m,1h,10m ")
                .iter()
                .map(|window| (window.label.as_str(), window.minutes))
                .collect::<Vec<_>>(),
            vec![("10m", 10), ("1h", 60), ("24h", 1440)]
        );
        assert!(windows("").is_empty());
        for bad in ["0m", "49h", "10s", "h", "1x"] {
            assert!(parse_windows(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn prompts_hit_stored_blocks_within_each_window() {
        let mut state = State::new(windows("10m,1h,24h"));
        let (a, b, c, d) = (hash(1), hash(2), hash(3), hash(4));
        state.stamp(&[a, b, c], 0);
        // At 5: a and b were stored 5 minutes ago; d never was, so the hit ends there.
        state.observe(&prompt(&[a, b, d]), 5);
        assert_eq!(state.hit_tokens, vec![128, 128, 128]);
        // At 30: the hit at 5 restamped a and b (age 25); c is from 0 (age 30).
        state.observe(&prompt(&[a, b, c]), 30);
        assert_eq!(state.hit_tokens, vec![128, 128 + 192, 128 + 192]);
        // At 100 (age 70): only the 24h window still hits.
        state.observe(&prompt(&[a, b]), 100);
        assert_eq!(state.hit_tokens, vec![128, 320, 320 + 128]);
        assert_eq!(state.prompt_tokens, (3 + 3 + 2) * 64 + 30);
    }

    #[test]
    fn prompts_never_store_blocks() {
        let mut state = State::new(windows("10m"));
        state.observe(&prompt(&[hash(1), hash(2)]), 0);
        state.observe(&prompt(&[hash(1), hash(2)]), 1);
        assert_eq!(state.hit_tokens, vec![0]);
        assert_eq!(state.table.len, 0);
        // Once a worker stores them, they hit. A reply's blocks come the same way.
        state.stamp(&[hash(1), hash(2), hash(3)], 2);
        state.observe(&prompt(&[hash(1), hash(2), hash(3), hash(4)]), 3);
        assert_eq!(state.hit_tokens, vec![192]);
    }

    #[test]
    fn live_blocks_count_the_last_stamp_per_window() {
        let mut state = State::new(windows("10m,1h,24h"));
        state.stamp(&[hash(1), hash(2), hash(3)], 0);
        state.stamp(&[hash(1), hash(4)], 30);
        // At 35: 1 and 4 were stamped 5 min ago, 2 and 3 35 min ago.
        assert_eq!(state.live_blocks(35), vec![2, 4, 4]);
        // A hit moves a block's stamp, it does not add one.
        state.observe(&prompt(&[hash(1), hash(2)]), 40);
        assert_eq!(state.live_blocks(40), vec![2, 4, 4]);
        assert_eq!(state.live_blocks(100), vec![0, 0, 4]);
    }

    #[test]
    fn trimming_frees_memory_without_changing_answers() {
        let old: Vec<u64> = (0..1000).map(hash).collect();
        // Past the longest window but untrimmed: a miss all the same.
        let mut untrimmed = State::new(windows("10m,1h"));
        untrimmed.stamp(&old, 0);
        untrimmed.observe(&prompt(&old[..10]), 61);
        assert_eq!(untrimmed.hit_tokens, vec![0, 0]);

        let mut state = State::new(windows("10m,1h"));
        state.stamp(&old, 0);
        state.stamp(&[hash(5000)], 50);
        state.trim(state.table.slots.len(), 61);
        assert_eq!(state.table.len, 1);
        assert_eq!(state.live_blocks(61), vec![0, 1]);
        state.observe(&prompt(&old[..10]), 61);
        assert_eq!(state.hit_tokens, vec![0, 0]);
        state.observe(&prompt(&[hash(5000)]), 62);
        assert_eq!(state.hit_tokens, vec![0, 64]);
    }

    #[test]
    fn the_table_matches_a_map_through_growth_and_removals() {
        let mut table = Table::new();
        let mut model: HashMap<u64, u16> = HashMap::new();
        // Small fingerprints crowd the low slots: long probe runs, wrap-around.
        let fingerprints: Vec<u64> = (1..=200_000u64)
            .map(|n| if n % 3 == 0 { n } else { hash(n) >> 16 })
            .collect();
        for (n, &fp) in fingerprints.iter().enumerate() {
            let minute = (n % 1000) as u16;
            assert_eq!(table.exchange(fp, minute), model.insert(fp, minute));
        }
        assert_eq!(table.len, model.len());
        // Remove every entry with an odd minute in one forward pass, as trimming does.
        let mut slot = 0;
        while slot < table.slots.len() {
            let entry = table.slots[slot];
            if entry != 0 && entry as u16 % 2 == 1 {
                model.remove(&(entry >> 16));
                table.remove_at(slot);
            } else {
                slot += 1;
            }
        }
        assert_eq!(table.len, model.len());
        for &fp in &fingerprints {
            let expected = model.get(&fp).copied();
            assert_eq!(table.find(fp), expected, "fingerprint {fp}");
            assert_eq!(table.refresh(fp, 7), expected, "fingerprint {fp}");
            if expected.is_some() {
                model.insert(fp, 7);
            }
        }
        assert_eq!(table.len, model.len());
    }

    #[test]
    fn the_clock_wraps_without_losing_ages() {
        let mut state = State::new(windows("10m"));
        state.stamp(&[hash(1)], u16::MAX - 2);
        state.observe(&prompt(&[hash(1)]), 3);
        assert_eq!(state.hit_tokens, vec![64]);
        assert_eq!(state.live_blocks(3), vec![1]);
    }
}
