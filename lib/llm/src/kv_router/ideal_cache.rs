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
//! own tracker; its inputs apply in arrival order on its thread.
//!
//! [`IDEAL_CACHE_FEEDS_ENV`] can add a second stamp per block, independent of the
//! first: the last minute a routed prompt held it, this router's or a peer's. It
//! does not depend on the KV index sources, so when their events go missing or
//! misreport, the two readings part. Each routed prompt counts its hit against it
//! the same way, then stamps all its blocks. The peers' prompts come with replica
//! sync: a router keeping the prompt stamp sends its prompts' block hashes in its
//! `AddRequest`, and its peers stamp them without counting a hit. A router that
//! does not keep it sends none, so the prompt stamp covers only the prompts of the
//! routers that keep it. With both stamps, a third reading takes the newer one: a
//! cache both feeds fill. The readings are reported under `feed="stored"`,
//! `feed="prompt"` and `feed="union"`, with one shared prompt-token count.
//!
//! Each entry is one u64 in an open-addressing table: the block's 35-bit
//! fingerprint, the 16-bit minute of its newer stamp, and the signed 13-bit gap to
//! the other one, or a mark that it has none. Neither stamp is taken to be the
//! newer one, and each changes only with its own feed, so the readings never
//! affect each other. A cursor removes entries whose stamps are all older than the
//! longest window in small batches, one pass every 10 minutes; a stamp that old is
//! otherwise dropped when its entry is next written. An entry or stamp not yet
//! removed reads as a miss, so trimming bounds memory, not accuracy.

use std::sync::{Arc, OnceLock, mpsc};
use std::time::{Duration, Instant};

use dynamo_kv_router::protocols::KvCacheStoreData;
use dynamo_kv_router::sequences::ReplicaPromptObserver;
use dynamo_runtime::{component::Component, metrics::MetricsHierarchy};
use prometheus::{IntCounter, IntCounterVec, IntGauge, IntGaugeVec};

use super::vllm_block_hash::{BLOCK_HASH_ENV, BlockHashScheme, VllmBlockHasher};

/// Windows to measure, comma-separated: `10m,1h,6h,24h`. Unset or empty: off.
pub const IDEAL_CACHE_WINDOWS_ENV: &str = "DYN_ROUTER_IDEAL_CACHE_WINDOWS";
/// Stamps to keep per block, comma-separated: `stored` (the KV index sources) and
/// `prompt` (every routed prompt, this router's and its peers'). Unset or empty:
/// `stored`.
pub const IDEAL_CACHE_FEEDS_ENV: &str = "DYN_ROUTER_IDEAL_CACHE_FEEDS";

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

/// An entry's stamps: the minute of the newer one above the signed gap to the other.
const GAP_BITS: u32 = 13;
const STAMP_BITS: u32 = 16 + GAP_BITS;
const STAMP_MASK: u64 = (1 << STAMP_BITS) - 1;
const GAP_MASK: u64 = (1 << GAP_BITS) - 1;
/// A gap is kept as `gap + GAP_BIAS`; the two codes it never reaches mark a block
/// without that stamp.
const GAP_BIAS: i32 = 1 << (GAP_BITS - 1);
const NO_STORED: u64 = 0;
const NO_PROMPT: u64 = GAP_MASK;
// Two stamps both under the longest window old are less than that apart.
const _: () = assert!((MAX_WINDOW_MINUTES as i32) < GAP_BIAS);

/// One routed prompt: its full blocks' worker block hashes.
#[derive(Clone)]
struct Prompt {
    hashes: Vec<u64>,
    block_size: u32,
    tokens: usize,
}

#[derive(Clone)]
enum Input {
    /// Hashes of blocks a worker stored.
    Stored(Vec<u64>),
    Prompt(Prompt),
    /// A peer router's prompt, for the prompt stamp.
    PeerPrompt(Vec<u64>),
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

/// What stamps a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Feed {
    /// A worker stored it, or a prompt hit it while it held this stamp.
    Stored,
    /// A routed prompt held it, this router's or a peer's.
    Prompt,
}

/// The stamps a tracker keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Feeds {
    stored: bool,
    prompt: bool,
}

impl Feeds {
    fn keeps(self, feed: Feed) -> bool {
        match feed {
            Feed::Stored => self.stored,
            Feed::Prompt => self.prompt,
        }
    }

    /// Each stamp kept, and with both, the two together.
    fn readings(self) -> Vec<Reading> {
        let mut readings = Vec::new();
        if self.stored {
            readings.push(Reading::Stored);
        }
        if self.prompt {
            readings.push(Reading::Prompt);
        }
        if self.stored && self.prompt {
            readings.push(Reading::Union);
        }
        readings
    }
}

fn parse_feeds(value: &str) -> Result<Feeds, String> {
    let mut feeds = Feeds {
        stored: false,
        prompt: false,
    };
    for label in value
        .split(',')
        .map(str::trim)
        .filter(|label| !label.is_empty())
    {
        match label {
            "stored" => feeds.stored = true,
            "prompt" => feeds.prompt = true,
            _ => return Err(format!("feed {label:?}: expected stored or prompt")),
        }
    }
    if !feeds.prompt {
        feeds.stored = true;
    }
    Ok(feeds)
}

/// The stamp a hit is counted against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reading {
    Stored,
    Prompt,
    /// The newer of the two: a cache both feeds fill.
    Union,
}

impl Reading {
    fn label(self) -> &'static str {
        match self {
            Reading::Stored => "stored",
            Reading::Prompt => "prompt",
            Reading::Union => "union",
        }
    }
}

/// The minute each feed last stamped a block, if it has.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Stamps {
    stored: Option<u16>,
    prompt: Option<u16>,
}

impl Stamps {
    fn unpack(bits: u64) -> Self {
        let newer = (bits >> GAP_BITS) as u16;
        match bits & GAP_MASK {
            NO_STORED => Self {
                stored: None,
                prompt: Some(newer),
            },
            NO_PROMPT => Self {
                stored: Some(newer),
                prompt: None,
            },
            code => {
                // The prompt stamp's minute less the stored one's.
                let gap = (code as i32 - GAP_BIAS) as i16;
                if gap >= 0 {
                    Self {
                        stored: Some(newer.wrapping_sub(gap as u16)),
                        prompt: Some(newer),
                    }
                } else {
                    Self {
                        stored: Some(newer),
                        prompt: Some(newer.wrapping_add_signed(gap)),
                    }
                }
            }
        }
    }

    /// Packs stamps less than the longest window apart; a held block has one.
    fn pack(self) -> u64 {
        let (newer, code) = match (self.stored, self.prompt) {
            (Some(stored), Some(prompt)) => {
                let gap = prompt.wrapping_sub(stored) as i16;
                debug_assert!(gap.unsigned_abs() < MAX_WINDOW_MINUTES, "{self:?}");
                let newer = if gap >= 0 { prompt } else { stored };
                (newer, (i32::from(gap) + GAP_BIAS) as u64)
            }
            (Some(stored), None) => (stored, NO_PROMPT),
            (None, Some(prompt)) => (prompt, NO_STORED),
            (None, None) => unreachable!("a held block has a stamp"),
        };
        (newer as u64) << GAP_BITS | code
    }

    fn set(mut self, feed: Feed, now: u16) -> Self {
        match feed {
            Feed::Stored => self.stored = Some(now),
            Feed::Prompt => self.prompt = Some(now),
        }
        self
    }

    /// Drops the stamps at least `max_age` minutes old: misses in every window.
    fn live(self, now: u16, max_age: u16) -> Self {
        let live =
            |minute: Option<u16>| minute.filter(|&minute| now.wrapping_sub(minute) < max_age);
        Self {
            stored: live(self.stored),
            prompt: live(self.prompt),
        }
    }

    /// The minute `reading` counts the block from.
    fn minute(self, reading: Reading, now: u16) -> Option<u16> {
        match reading {
            Reading::Stored => self.stored,
            Reading::Prompt => self.prompt,
            Reading::Union => match (self.stored, self.prompt) {
                (Some(stored), Some(prompt)) => {
                    Some(if now.wrapping_sub(stored) < now.wrapping_sub(prompt) {
                        stored
                    } else {
                        prompt
                    })
                }
                (stored, prompt) => stored.or(prompt),
            },
        }
    }
}

/// Block fingerprint: the hash's top 35 bits, never 0 (an empty slot).
fn fingerprint(hash: u64) -> u64 {
    (hash >> STAMP_BITS).max(1)
}

/// Open addressing with linear probing over packed `fingerprint << STAMP_BITS |
/// stamps` entries; 0 marks an empty slot.
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

    /// The packed stamps of `fingerprint`, if it is held.
    fn get(&self, fingerprint: u64) -> Option<u64> {
        let mask = self.mask();
        let mut slot = self.home(fingerprint);
        loop {
            let held = self.slots[slot];
            if held == 0 {
                return None;
            }
            if held >> STAMP_BITS == fingerprint {
                return Some(held & STAMP_MASK);
            }
            slot = (slot + 1) & mask;
        }
    }

    /// Sets `fingerprint`'s packed stamps, adding it if it is not held.
    fn set(&mut self, fingerprint: u64, stamps: u64) {
        if (self.len + 1) * 4 > self.slots.len() * 3 {
            self.grow();
        }
        let entry = fingerprint << STAMP_BITS | stamps;
        let mask = self.mask();
        let mut slot = self.home(fingerprint);
        loop {
            let held = self.slots[slot];
            if held == 0 {
                self.slots[slot] = entry;
                self.len += 1;
                return;
            }
            if held >> STAMP_BITS == fingerprint {
                self.slots[slot] = entry;
                return;
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
            let home = self.home(entry >> STAMP_BITS);
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
            let mut slot = self.home(entry >> STAMP_BITS);
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
    feeds: Feeds,
    readings: Vec<Reading>,
    windows: Vec<Window>,
    table: Table,
    /// Per reading: blocks by the minute it counts them from (`minute % RING_MINUTES`).
    last_use: Vec<Vec<u64>>,
    minute: Option<u16>,
    cursor: usize,
    /// Per reading and window.
    hit_tokens: Vec<Vec<u64>>,
    prompt_tokens: u64,
}

impl State {
    fn new(feeds: Feeds, windows: Vec<Window>) -> Self {
        let readings = feeds.readings();
        Self {
            feeds,
            last_use: vec![vec![0; RING_MINUTES]; readings.len()],
            hit_tokens: vec![vec![0; windows.len()]; readings.len()],
            readings,
            windows,
            table: Table::new(),
            minute: None,
            cursor: 0,
            prompt_tokens: 0,
        }
    }

    fn ring(minute: u16) -> usize {
        minute as usize % RING_MINUTES
    }

    fn max_age(&self) -> u16 {
        self.windows.last().map_or(0, |window| window.minutes)
    }

    fn stamps(&self, fingerprint: u64) -> Stamps {
        self.table
            .get(fingerprint)
            .map_or_else(Stamps::default, Stamps::unpack)
    }

    /// Moves the clock to `minute`, clearing the counters it reuses.
    fn advance(&mut self, minute: u16) {
        let mut at = self.minute.unwrap_or(minute.wrapping_sub(1));
        while at != minute {
            at = at.wrapping_add(1);
            for last_use in &mut self.last_use {
                last_use[Self::ring(at)] = 0;
            }
        }
        self.minute = Some(minute);
    }

    /// Moves a block's count, for each reading, from the minute its `old` stamps
    /// give to the one its `new` stamps give. An old minute whose counter was
    /// reused is not taken out.
    fn recount(&mut self, old: Stamps, new: Stamps, now: u16) {
        for (&reading, last_use) in self.readings.iter().zip(&mut self.last_use) {
            let (was, is) = (old.minute(reading, now), new.minute(reading, now));
            if was == is {
                continue;
            }
            if let Some(minute) = was
                && (now.wrapping_sub(minute) as usize) < RING_MINUTES
            {
                let count = &mut last_use[Self::ring(minute)];
                *count = count.saturating_sub(1);
            }
            if let Some(minute) = is {
                last_use[Self::ring(minute)] += 1;
            }
        }
    }

    fn write(&mut self, fingerprint: u64, old: Stamps, new: Stamps, now: u16) {
        self.recount(old, new, now);
        self.table.set(fingerprint, new.pack());
    }

    fn apply(&mut self, input: Input, now: u16) {
        match input {
            Input::Stored(hashes) => self.stamp(Feed::Stored, &hashes, now),
            Input::Prompt(prompt) => self.observe(&prompt, now),
            Input::PeerPrompt(hashes) => self.stamp(Feed::Prompt, &hashes, now),
        }
    }

    /// Stamps blocks with `feed`, if this tracker keeps it.
    fn stamp(&mut self, feed: Feed, hashes: &[u64], now: u16) {
        self.advance(now);
        if !self.feeds.keeps(feed) {
            return;
        }
        let max_age = self.max_age();
        for &hash in hashes {
            let fingerprint = fingerprint(hash);
            let old = self.stamps(fingerprint);
            self.write(fingerprint, old, old.set(feed, now).live(now, max_age), now);
        }
    }

    /// Counts a prompt's hit for each reading and window: its leading blocks the
    /// reading has stamped under the window's minutes ago. Restamps the stored hits;
    /// with the prompt stamp, stamps every block.
    fn observe(&mut self, prompt: &Prompt, now: u16) {
        self.advance(now);
        let max_age = self.max_age();
        let windows = self.windows.len();
        let stored = self
            .readings
            .iter()
            .position(|&reading| reading == Reading::Stored);
        // Per reading: the windows still hitting, a suffix as windows are sorted.
        let mut open = vec![windows; self.readings.len()];
        let mut hits = vec![vec![0u64; windows]; self.readings.len()];
        for &hash in &prompt.hashes {
            if !self.feeds.prompt && open.iter().all(|&open| open == 0) {
                break;
            }
            let fingerprint = fingerprint(hash);
            let old = self.stamps(fingerprint);
            for (i, &reading) in self.readings.iter().enumerate() {
                let age = old
                    .minute(reading, now)
                    .map_or(u16::MAX, |minute| now.wrapping_sub(minute));
                while open[i] > 0 && age >= self.windows[windows - open[i]].minutes {
                    open[i] -= 1;
                }
                for hit in &mut hits[i][windows - open[i]..] {
                    *hit += 1;
                }
            }
            let mut new = old;
            if stored.is_some_and(|i| open[i] > 0) {
                new.stored = Some(now);
            }
            if self.feeds.prompt {
                new.prompt = Some(now);
            }
            if new != old {
                self.write(fingerprint, old, new.live(now, max_age), now);
            }
        }
        for (tokens, hits) in self.hit_tokens.iter_mut().zip(hits) {
            for (tokens, blocks) in tokens.iter_mut().zip(hits) {
                *tokens += blocks * prompt.block_size as u64;
            }
        }
        self.prompt_tokens += prompt.tokens as u64;
    }

    /// Moves the cursor past `slots` slots, from where the last call stopped,
    /// removing entries whose stamps are all past the longest window.
    fn trim(&mut self, slots: usize, now: u16) {
        self.advance(now);
        let max_age = self.max_age();
        let mut passed = 0;
        while passed < slots {
            let slot = self.cursor & self.table.mask();
            let entry = self.table.slots[slot];
            let stamps = Stamps::unpack(entry & STAMP_MASK);
            if entry != 0 && stamps.live(now, max_age) == Stamps::default() {
                self.recount(stamps, Stamps::default(), now);
                // The slot may now hold a moved entry: look at it again.
                self.table.remove_at(slot);
            } else {
                self.cursor = (slot + 1) & self.table.mask();
                passed += 1;
            }
        }
    }

    /// Per reading: blocks it counts from under each window's minutes ago.
    fn live_blocks(&self, now: u16) -> Vec<Vec<u64>> {
        self.last_use
            .iter()
            .map(|last_use| {
                let mut live = Vec::with_capacity(self.windows.len());
                let (mut total, mut age) = (0, 0u16);
                for window in &self.windows {
                    while age < window.minutes {
                        total += last_use[Self::ring(now.wrapping_sub(age))];
                        age += 1;
                    }
                    live.push(total);
                }
                live
            })
            .collect()
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
    peer_prompts: IntCounter,
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
        for input in ["stored", "prompt", "peer"] {
            dropped.with_label_values(&[input]);
        }
        Ok(Self {
            hit_tokens: metrics.create_intcountervec(
                "router_ideal_cache_hit_tokens_total",
                "Prompt tokens a global, unbounded KV cache would hit if it kept every block for `window` after its last `feed` stamp: stored (stored, or hit while stored), prompt (held by a routed prompt), or union (either)",
                &["feed", "window"],
                &[],
            )?,
            prompt_tokens: metrics.create_intcounter(
                "router_ideal_cache_prompt_tokens_total",
                "Prompt tokens of the requests the ideal KV cache tracks",
                &[],
            )?,
            live_blocks: metrics.create_intgaugevec(
                "router_ideal_cache_live_blocks",
                "Blocks with a `feed` stamp within `window`: what the ideal KV cache for that window holds",
                &["feed", "window"],
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
            peer_prompts: metrics.create_intcounter(
                "router_ideal_cache_peer_prompts_total",
                "Peer routers' prompts the prompt stamps received through replica sync",
                &[],
            )?,
        })
    }
}

/// A router's ideal KV cache; see the module docs. Its thread stops with the router.
pub struct IdealCache {
    inputs: mpsc::SyncSender<Input>,
    hasher: VllmBlockHasher,
    metrics: &'static Metrics,
    /// Whether it keeps the prompt stamp, which takes the peers' prompts.
    prompt_feed: bool,
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
        let value = std::env::var(IDEAL_CACHE_FEEDS_ENV).unwrap_or_default();
        let feeds = match parse_feeds(&value) {
            Ok(feeds) => feeds,
            Err(error) => {
                tracing::warn!(env = IDEAL_CACHE_FEEDS_ENV, %value, %error, "Ideal KV cache off");
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
        let readings = feeds.readings();
        for reading in &readings {
            for window in &windows {
                let labels = [reading.label(), window.label.as_str()];
                metrics.hit_tokens.with_label_values(&labels);
                metrics.live_blocks.with_label_values(&labels);
            }
        }
        tracing::info!(
            windows = ?windows.iter().map(|window| &window.label).collect::<Vec<_>>(),
            readings = ?readings.iter().map(|reading| reading.label()).collect::<Vec<_>>(),
            "Tracking the ideal KV cache hit rate"
        );
        let state = State::new(feeds, windows);
        let (inputs, received) = mpsc::sync_channel(QUEUE_INPUTS);
        let spawned = std::thread::Builder::new()
            .name("ideal-kv-cache".to_string())
            .spawn(move || run(received, state, metrics));
        if let Err(error) = spawned {
            tracing::warn!(%error, "Failed to start the ideal KV cache tracker");
            return None;
        }
        Some(Arc::new(Self {
            inputs,
            hasher,
            metrics,
            prompt_feed: feeds.prompt,
        }))
    }

    /// Whether the peers' prompts are wanted: this router keeps the prompt stamp,
    /// and sends its own prompts' block hashes to its peers.
    pub fn shares_prompts(&self) -> bool {
        self.prompt_feed
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

    /// A prompt's full blocks, hashed as the workers hash them; `None` when they
    /// cannot be.
    pub fn prompt_hashes(&self, block_size: u32, tokens: &[u32]) -> Option<Vec<u64>> {
        let blocks = tokens.len() / block_size as usize;
        self.hasher
            .chunk_hashes(tokens, block_size as usize, blocks)
    }

    /// Counts a routed prompt of `tokens` tokens by its [`Self::prompt_hashes`].
    pub fn observe_prompt(&self, block_size: u32, tokens: usize, hashes: Vec<u64>) {
        self.send(Input::Prompt(Prompt {
            hashes,
            block_size,
            tokens,
        }));
    }

    fn send(&self, input: Input) {
        let label = match input {
            Input::Stored(_) => "stored",
            Input::Prompt(_) => "prompt",
            Input::PeerPrompt(_) => "peer",
        };
        if self.inputs.try_send(input).is_err() {
            self.metrics.dropped.with_label_values(&[label]).inc();
        }
    }
}

impl ReplicaPromptObserver for IdealCache {
    /// Stamps a peer router's prompt with the prompt stamp.
    fn peer_prompt(&self, block_hashes: Vec<u64>) {
        if self.prompt_feed && !block_hashes.is_empty() {
            self.metrics.peer_prompts.inc();
            self.send(Input::PeerPrompt(block_hashes));
        }
    }
}

/// What one tracker has added to the shared metrics.
struct Reported {
    /// Per reading and window.
    hit_tokens: Vec<Vec<u64>>,
    prompt_tokens: u64,
    /// Per reading and window.
    live_blocks: Vec<Vec<i64>>,
    entries: i64,
    table_bytes: i64,
}

impl Reported {
    fn new(state: &State) -> Self {
        Self {
            hit_tokens: vec![vec![0; state.windows.len()]; state.readings.len()],
            prompt_tokens: 0,
            live_blocks: vec![vec![0; state.windows.len()]; state.readings.len()],
            entries: 0,
            table_bytes: 0,
        }
    }

    /// Adds the hits and prompt tokens `state` counted since the last call.
    fn add_counts(&mut self, metrics: &Metrics, state: &State) {
        for ((reading, tokens), reported) in state
            .readings
            .iter()
            .zip(&state.hit_tokens)
            .zip(&mut self.hit_tokens)
        {
            for ((window, &tokens), reported) in
                state.windows.iter().zip(tokens).zip(reported.iter_mut())
            {
                metrics
                    .hit_tokens
                    .with_label_values(&[reading.label(), window.label.as_str()])
                    .inc_by(tokens - *reported);
                *reported = tokens;
            }
        }
        metrics
            .prompt_tokens
            .inc_by(state.prompt_tokens - self.prompt_tokens);
        self.prompt_tokens = state.prompt_tokens;
    }

    /// Brings the shared gauges from this tracker's last values to `live_blocks`,
    /// `entries` and `table_bytes`.
    fn set_gauges(
        &mut self,
        metrics: &Metrics,
        state: &State,
        live_blocks: &[Vec<i64>],
        entries: i64,
        table_bytes: i64,
    ) {
        for ((reading, live), reported) in state
            .readings
            .iter()
            .zip(live_blocks)
            .zip(&mut self.live_blocks)
        {
            for ((window, &live), reported) in
                state.windows.iter().zip(live).zip(reported.iter_mut())
            {
                metrics
                    .live_blocks
                    .with_label_values(&[reading.label(), window.label.as_str()])
                    .add(live - *reported);
                *reported = live;
            }
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
    let mut reported = Reported::new(&state);
    let mut next_tick = Instant::now();
    loop {
        match inputs.recv_timeout(next_tick.saturating_duration_since(Instant::now())) {
            Ok(input) => state.apply(input, minute()),
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // The router is gone: take its share out of the gauges.
                let zeros = vec![vec![0; state.windows.len()]; state.readings.len()];
                reported.set_gauges(metrics, &state, &zeros, 0, 0);
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
        reported.add_counts(metrics, &state);
        let live: Vec<Vec<i64>> = state
            .live_blocks(now)
            .into_iter()
            .map(|live| live.into_iter().map(|blocks| blocks as i64).collect())
            .collect();
        let table_bytes = (state.table.slots.len() * std::mem::size_of::<u64>()) as i64;
        reported.set_gauges(metrics, &state, &live, state.table.len as i64, table_bytes);
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
                hit_tokens: IntCounterVec::new(opts("hit_tokens"), &["feed", "window"]).unwrap(),
                prompt_tokens: IntCounter::with_opts(opts("prompt_tokens")).unwrap(),
                live_blocks: IntGaugeVec::new(opts("live_blocks"), &["feed", "window"]).unwrap(),
                entries: IntGauge::with_opts(opts("entries")).unwrap(),
                table_bytes: IntGauge::with_opts(opts("table_bytes")).unwrap(),
                tracked_seconds: IntGauge::with_opts(opts("tracked_seconds")).unwrap(),
                dropped: IntCounterVec::new(opts("dropped"), &["input"]).unwrap(),
                peer_prompts: IntCounter::with_opts(opts("peer_prompts")).unwrap(),
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
                    Input::Prompt(_) | Input::PeerPrompt(_) => Vec::new(),
                })
                .collect()
        };
        let cache = Self {
            inputs,
            hasher,
            metrics,
            prompt_feed: false,
        };
        (Arc::new(cache), stamped)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    const STORED: Feeds = Feeds {
        stored: true,
        prompt: false,
    };
    const PROMPT: Feeds = Feeds {
        stored: false,
        prompt: true,
    };
    const BOTH: Feeds = Feeds {
        stored: true,
        prompt: true,
    };

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
    fn feeds_parse_with_stored_by_default() {
        assert_eq!(parse_feeds("").unwrap(), STORED);
        assert_eq!(parse_feeds(" prompt,stored, prompt ").unwrap(), BOTH);
        assert_eq!(parse_feeds("prompt").unwrap(), PROMPT);
        assert!(parse_feeds("stored,gpu").is_err());
        assert_eq!(
            BOTH.readings(),
            vec![Reading::Stored, Reading::Prompt, Reading::Union]
        );
    }

    #[test]
    fn stamps_round_trip_whichever_is_newer() {
        let far = MAX_WINDOW_MINUTES - 1;
        let at = |minute: u16, back: u16| Some(minute.wrapping_sub(back));
        for now in [far, 1000, u16::MAX, 2] {
            for (stored, prompt) in [
                (at(now, 0), None),
                (None, at(now, 0)),
                (at(now, 0), at(now, 0)),
                (at(now, 0), at(now, far)),
                (at(now, far), at(now, 0)),
                (at(now, 3), at(now, 7)),
                (at(now, 7), at(now, 3)),
            ] {
                let stamps = Stamps { stored, prompt };
                let packed = stamps.pack();
                assert!(packed <= STAMP_MASK, "{stamps:?}");
                assert_eq!(Stamps::unpack(packed), stamps);
            }
        }
    }

    #[test]
    fn prompts_hit_stored_blocks_within_each_window() {
        let mut state = State::new(STORED, windows("10m,1h,24h"));
        let (a, b, c, d) = (hash(1), hash(2), hash(3), hash(4));
        state.stamp(Feed::Stored, &[a, b, c], 0);
        // At 5: a and b were stored 5 minutes ago; d never was, so the hit ends there.
        state.observe(&prompt(&[a, b, d]), 5);
        assert_eq!(state.hit_tokens[0], vec![128, 128, 128]);
        // At 30: the hit at 5 restamped a and b (age 25); c is from 0 (age 30).
        state.observe(&prompt(&[a, b, c]), 30);
        assert_eq!(state.hit_tokens[0], vec![128, 128 + 192, 128 + 192]);
        // At 100 (age 70): only the 24h window still hits.
        state.observe(&prompt(&[a, b]), 100);
        assert_eq!(state.hit_tokens[0], vec![128, 320, 320 + 128]);
        assert_eq!(state.prompt_tokens, (3 + 3 + 2) * 64 + 30);
    }

    #[test]
    fn prompts_never_store_blocks() {
        let mut state = State::new(STORED, windows("10m"));
        state.observe(&prompt(&[hash(1), hash(2)]), 0);
        state.observe(&prompt(&[hash(1), hash(2)]), 1);
        assert_eq!(state.hit_tokens[0], vec![0]);
        assert_eq!(state.table.len, 0);
        // Once a worker stores them, they hit. A reply's blocks come the same way.
        state.stamp(Feed::Stored, &[hash(1), hash(2), hash(3)], 2);
        state.observe(&prompt(&[hash(1), hash(2), hash(3), hash(4)]), 3);
        assert_eq!(state.hit_tokens[0], vec![192]);
    }

    #[test]
    fn the_prompt_stamp_holds_what_prompts_use() {
        let mut state = State::new(PROMPT, windows("10m,1h"));
        let (a, b, c, d) = (hash(1), hash(2), hash(3), hash(4));
        // A worker's stored blocks do not reach it.
        state.stamp(Feed::Stored, &[a, b, c, d], 0);
        state.observe(&prompt(&[a, b]), 0);
        assert_eq!(state.hit_tokens[0], vec![0, 0]);
        assert_eq!(state.table.len, 2);
        // At 5: a and b hit, c is new and stamped from here on.
        state.observe(&prompt(&[a, b, c]), 5);
        assert_eq!(state.hit_tokens[0], vec![128, 128]);
        assert_eq!(state.live_blocks(5)[0], vec![3, 3]);
        // At 30 (age 25): only the 1h window hits; d joins the table.
        state.observe(&prompt(&[a, b, c, d]), 30);
        assert_eq!(state.hit_tokens[0], vec![128, 128 + 192]);
        assert_eq!(state.live_blocks(30)[0], vec![4, 4]);
        // A miss stops the hit: x is new, b after it is not counted, x and b are stamped.
        let x = hash(9);
        state.observe(&prompt(&[a, x, b]), 31);
        assert_eq!(state.hit_tokens[0], vec![128 + 64, 128 + 192 + 64]);
        assert_eq!(state.table.len, 5);
    }

    #[test]
    fn either_stamp_may_be_the_newer_one() {
        let mut state = State::new(BOTH, windows("10m"));
        let hits = |state: &State| -> Vec<u64> {
            state.hit_tokens.iter().map(|tokens| tokens[0]).collect()
        };
        let (a, b, c, d) = (hash(1), hash(2), hash(3), hash(4));
        // A worker's blocks, then a peer's prompt over them and one more.
        state.apply(Input::Stored(vec![a, b]), 0);
        state.apply(Input::PeerPrompt(vec![a, b, c]), 0);
        // Readings: stored, prompt, union.
        state.apply(Input::Prompt(prompt(&[a, b, c, d])), 1);
        assert_eq!(hits(&state), vec![128, 192, 192]);
        // The prompt stamped d; nothing stored c or d.
        state.apply(Input::Prompt(prompt(&[a, b, c, d])), 2);
        assert_eq!(hits(&state), vec![256, 448, 448]);
        // A block stored and never prompted hits the stored and union readings only.
        let e = hash(5);
        state.apply(Input::Stored(vec![e]), 3);
        state.apply(Input::Prompt(prompt(&[e])), 4);
        assert_eq!(hits(&state), vec![320, 448, 512]);
        // d is stored after its prompt, a is prompted after its store.
        state.apply(Input::Stored(vec![d]), 5);
        state.apply(Input::PeerPrompt(vec![a]), 5);
        assert_eq!(
            state.stamps(fingerprint(d)),
            Stamps {
                stored: Some(5),
                prompt: Some(2)
            }
        );
        assert_eq!(
            state.stamps(fingerprint(a)),
            Stamps {
                stored: Some(2),
                prompt: Some(5)
            }
        );
        // Stored: a and b (c never was). Prompt and union: all four.
        state.apply(Input::Prompt(prompt(&[a, b, c, d])), 6);
        assert_eq!(hits(&state), vec![448, 704, 768]);
    }

    #[test]
    fn an_entry_lives_while_either_stamp_does() {
        let mut state = State::new(BOTH, windows("10m,1h"));
        let (a, b) = (hash(1), hash(2));
        state.apply(Input::Stored(vec![a, b]), 0);
        state.apply(Input::PeerPrompt(vec![a]), 50);
        // At 70: b and a's stored stamp are past the hour, a's prompt stamp is not.
        state.trim(state.table.slots.len(), 70);
        assert_eq!(state.table.len, 1);
        assert_eq!(
            state.stamps(fingerprint(a)),
            Stamps {
                stored: Some(0),
                prompt: Some(50)
            }
        );
        assert_eq!(
            state.live_blocks(70),
            vec![vec![0, 0], vec![0, 1], vec![0, 1]]
        );
        // Writing a drops its stale stored stamp.
        state.apply(Input::PeerPrompt(vec![a]), 75);
        assert_eq!(
            state.stamps(fingerprint(a)),
            Stamps {
                stored: None,
                prompt: Some(75)
            }
        );
        state.trim(state.table.slots.len(), 135);
        assert_eq!(state.table.len, 0);
        assert_eq!(state.live_blocks(135), vec![vec![0, 0]; 3]);
    }

    /// xorshift64*, for a reproducible input stream.
    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) % n
        }
    }

    /// Each reading as plain maps of block hash to stamp minute, never trimmed.
    struct Model {
        stored: HashMap<u64, u16>,
        prompt: HashMap<u64, u16>,
        /// One table every input stamps, as the prompt feed had before the stamps split.
        union: HashMap<u64, u16>,
        /// Per reading (stored, prompt, union) and window.
        hits: [Vec<u64>; 3],
        /// Prompts whose stored hit was longer than their prompt hit, and shorter.
        stored_ahead: usize,
        prompt_ahead: usize,
    }

    impl Model {
        fn new(windows: usize) -> Self {
            Self {
                stored: HashMap::new(),
                prompt: HashMap::new(),
                union: HashMap::new(),
                hits: std::array::from_fn(|_| vec![0; windows]),
                stored_ahead: 0,
                prompt_ahead: 0,
            }
        }

        /// For each window, the leading `hashes` stamped under its minutes ago.
        fn prefix(
            stamps: &HashMap<u64, u16>,
            windows: &[Window],
            hashes: &[u64],
            now: u16,
        ) -> Vec<usize> {
            windows
                .iter()
                .map(|window| {
                    hashes
                        .iter()
                        .take_while(|hash| {
                            stamps
                                .get(hash)
                                .is_some_and(|&minute| now.wrapping_sub(minute) < window.minutes)
                        })
                        .count()
                })
                .collect()
        }

        fn live(stamps: &HashMap<u64, u16>, windows: &[Window], now: u16) -> Vec<u64> {
            let mut live = vec![0; windows.len()];
            for &minute in stamps.values() {
                for (live, window) in live.iter_mut().zip(windows) {
                    *live += u64::from(now.wrapping_sub(minute) < window.minutes);
                }
            }
            live
        }

        fn apply(&mut self, input: &Input, windows: &[Window], now: u16) {
            match input {
                Input::Stored(hashes) => {
                    for &hash in hashes {
                        self.stored.insert(hash, now);
                        self.union.insert(hash, now);
                    }
                }
                Input::PeerPrompt(hashes) => {
                    for &hash in hashes {
                        self.prompt.insert(hash, now);
                        self.union.insert(hash, now);
                    }
                }
                Input::Prompt(prompt) => {
                    let prefixes = [&self.stored, &self.prompt, &self.union]
                        .map(|stamps| Self::prefix(stamps, windows, &prompt.hashes, now));
                    for (hits, prefix) in self.hits.iter_mut().zip(&prefixes) {
                        for (hit, &blocks) in hits.iter_mut().zip(prefix) {
                            *hit += blocks as u64 * prompt.block_size as u64;
                        }
                    }
                    let (held, prompted) =
                        (*prefixes[0].last().unwrap(), *prefixes[1].last().unwrap());
                    self.stored_ahead += usize::from(held > prompted);
                    self.prompt_ahead += usize::from(held < prompted);
                    // Stored stamps keep their hits; the others take every block.
                    for &hash in &prompt.hashes[..held] {
                        self.stored.insert(hash, now);
                    }
                    for &hash in &prompt.hashes {
                        self.prompt.insert(hash, now);
                        self.union.insert(hash, now);
                    }
                }
            }
        }

        fn reading(&self, reading: Reading) -> (&HashMap<u64, u16>, &Vec<u64>) {
            match reading {
                Reading::Stored => (&self.stored, &self.hits[0]),
                Reading::Prompt => (&self.prompt, &self.hits[1]),
                Reading::Union => (&self.union, &self.hits[2]),
            }
        }
    }

    #[test]
    fn every_reading_matches_its_model_through_trims_and_the_clock_wrap() {
        let windows = windows("5m,30m,1h");
        let mut states = [STORED, PROMPT, BOTH].map(|feeds| State::new(feeds, windows.clone()));
        let mut model = Model::new(windows.len());
        let mut rng = Rng(0x5EED_1DEA);
        // Documents of chained blocks: prompts and stores take runs of them.
        let document = |n: u64, len: u64| -> Vec<u64> {
            (0..len).map(|block| hash(1 + n * 1000 + block)).collect()
        };
        let mut fresh = 1 << 40;
        let mut now = u16::MAX - 3000;
        for step in 0..5000 {
            let elapsed = if rng.below(10) == 0 {
                rng.below(90)
            } else {
                rng.below(4)
            };
            now = now.wrapping_add(elapsed as u16);
            let mut hashes = document(rng.below(20), 1 + rng.below(12));
            let input = match rng.below(3) {
                // A worker stores a document's blocks: a prompt's from its start, or a
                // reply's further on.
                0 => {
                    let start = match rng.below(2) {
                        0 => 0,
                        _ => rng.below(hashes.len() as u64) as usize,
                    };
                    Input::Stored(hashes.split_off(start))
                }
                kind => {
                    for _ in 0..rng.below(3) {
                        fresh += 1;
                        hashes.push(hash(fresh));
                    }
                    if kind == 1 {
                        Input::Prompt(prompt(&hashes))
                    } else {
                        Input::PeerPrompt(hashes)
                    }
                }
            };
            model.apply(&input, &windows, now);
            for state in &mut states {
                state.apply(input.clone(), now);
                if rng.below(10) == 0 {
                    let slots = rng.below(state.table.slots.len() as u64) as usize;
                    state.trim(slots, now);
                }
            }
            let expected = [Reading::Stored, Reading::Prompt, Reading::Union].map(|reading| {
                let (stamps, hits) = model.reading(reading);
                (reading, hits.clone(), Model::live(stamps, &windows, now))
            });
            for state in &states {
                let live = state.live_blocks(now);
                for (i, &reading) in state.readings.iter().enumerate() {
                    let (_, hits, expected_live) = expected
                        .iter()
                        .find(|(expected, ..)| *expected == reading)
                        .unwrap();
                    let what = format!("step {step}, {reading:?} of {:?}", state.feeds);
                    assert_eq!(&state.hit_tokens[i], hits, "{what}: hits");
                    assert_eq!(&live[i], expected_live, "{what}: live");
                }
                assert_eq!(state.prompt_tokens, states[0].prompt_tokens);
            }
        }
        // Both orders of the two stamps came up, and every reading hit.
        assert!(model.stored_ahead > 50 && model.prompt_ahead > 50);
        assert!(model.hits.iter().all(|hits| hits[0] > 0));
    }

    #[test]
    fn live_blocks_count_the_last_stamp_per_window() {
        let mut state = State::new(STORED, windows("10m,1h,24h"));
        state.stamp(Feed::Stored, &[hash(1), hash(2), hash(3)], 0);
        state.stamp(Feed::Stored, &[hash(1), hash(4)], 30);
        // At 35: 1 and 4 were stamped 5 min ago, 2 and 3 35 min ago.
        assert_eq!(state.live_blocks(35)[0], vec![2, 4, 4]);
        // A hit moves a block's stamp, it does not add one.
        state.observe(&prompt(&[hash(1), hash(2)]), 40);
        assert_eq!(state.live_blocks(40)[0], vec![2, 4, 4]);
        assert_eq!(state.live_blocks(100)[0], vec![0, 0, 4]);
    }

    #[test]
    fn trimming_frees_memory_without_changing_answers() {
        let old: Vec<u64> = (0..1000).map(hash).collect();
        // Past the longest window but untrimmed: a miss all the same.
        let mut untrimmed = State::new(STORED, windows("10m,1h"));
        untrimmed.stamp(Feed::Stored, &old, 0);
        untrimmed.observe(&prompt(&old[..10]), 61);
        assert_eq!(untrimmed.hit_tokens[0], vec![0, 0]);

        let mut state = State::new(STORED, windows("10m,1h"));
        state.stamp(Feed::Stored, &old, 0);
        state.stamp(Feed::Stored, &[hash(5000)], 50);
        state.trim(state.table.slots.len(), 61);
        assert_eq!(state.table.len, 1);
        assert_eq!(state.live_blocks(61)[0], vec![0, 1]);
        state.observe(&prompt(&old[..10]), 61);
        assert_eq!(state.hit_tokens[0], vec![0, 0]);
        state.observe(&prompt(&[hash(5000)]), 62);
        assert_eq!(state.hit_tokens[0], vec![0, 64]);
    }

    #[test]
    fn the_table_matches_a_map_through_growth_and_removals() {
        let mut table = Table::new();
        let mut model: HashMap<u64, u64> = HashMap::new();
        // Small fingerprints crowd the low slots: long probe runs, wrap-around.
        let fingerprints: Vec<u64> = (1..=200_000u64)
            .map(|n| if n % 3 == 0 { n } else { hash(n) >> STAMP_BITS })
            .collect();
        for (n, &fp) in fingerprints.iter().enumerate() {
            let stamps = (n % 1000) as u64;
            assert_eq!(table.get(fp), model.insert(fp, stamps));
            table.set(fp, stamps);
        }
        assert_eq!(table.len, model.len());
        // Remove every entry with odd stamps in one forward pass, as trimming does.
        let mut slot = 0;
        while slot < table.slots.len() {
            let entry = table.slots[slot];
            if entry != 0 && entry % 2 == 1 {
                model.remove(&(entry >> STAMP_BITS));
                table.remove_at(slot);
            } else {
                slot += 1;
            }
        }
        assert_eq!(table.len, model.len());
        for &fp in &fingerprints {
            let expected = model.get(&fp).copied();
            assert_eq!(table.get(fp), expected, "fingerprint {fp}");
            if expected.is_some() {
                table.set(fp, 7);
                model.insert(fp, 7);
            }
        }
        assert_eq!(table.len, model.len());
        for (&fp, &stamps) in &model {
            assert_eq!(table.get(fp), Some(stamps));
        }
    }

    #[test]
    fn the_clock_wraps_without_losing_ages() {
        let mut state = State::new(STORED, windows("10m"));
        state.stamp(Feed::Stored, &[hash(1)], u16::MAX - 2);
        state.observe(&prompt(&[hash(1)]), 3);
        assert_eq!(state.hit_tokens[0], vec![64]);
        assert_eq!(state.live_blocks(3)[0], vec![1]);
    }
}
