// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in prefix-hit oracle for hybrid-attention engines (`--hybrid-swa-hits`).
//!
//! The radix tree tracks only the main full-attention group with set
//! semantics. On models that interleave sliding-window (SWA) groups with full
//! attention — Step-5-Preview: four SWA-512 groups plus one full-attention
//! group, all at block size 64, with MTP run as EAGLE — that over- and
//! under-predicts what the engine actually serves:
//!
//! - SWA groups keep cached blocks only at sparse checkpoints, so the engine's
//!   hit stops at the last position where every SWA group still holds the
//!   blocks its window needs, often well short of the full-attention chain.
//! - When the engine's hit is shorter than the chain it recomputes and
//!   re-stores the same hashes. A set-semantics index then forgets a block on
//!   the first `Removed` even though another copy is still cached.
//! - EAGLE/MTP drops the last matched block, and the engine never reuses the
//!   final prompt token's block (`(n - 1) / block_size` cap).
//!
//! This oracle consumes the same event stream, keeps per-copy refcounts for
//! every group, and predicts the engine's hit:
//! `k = max{k <= min(chain, (n-1)/bs) : every SWA group holds blocks [k-need, k)}`,
//! then subtracts the MTP block. Engine block hashes are shared across groups
//! (same hash for the same 64-token position), which is what makes the SWA
//! lookup by full-attention hash valid.
//!
//! Default-off: with the flag unset nothing here is constructed and the
//! listener / query paths are unchanged.

use std::sync::OnceLock;

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, LocalBlockHash, RouterEvent, WorkerWithDpRank,
};

/// Opt-in settings for the hybrid oracle.
#[derive(Debug, Clone, Copy)]
pub struct HybridConfig {
    /// Block size of the engine's KV groups (tokens per block).
    pub block_size: u32,
    /// Blocks dropped from the predicted hit for EAGLE/MTP speculative
    /// decoding (1 for Step-5's MTP; 0 for engines without it).
    pub mtp_drop_blocks: u32,
    /// Override for the contiguous SWA blocks a hit needs. `None` derives it
    /// per group from the sliding window: `ceil((window - 1) / bs) + mtp_drop`.
    pub swa_need_blocks: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct FaEntry {
    parent: Option<u64>,
    tokens: LocalBlockHash,
    copies: u32,
}

#[derive(Default)]
struct WorkerState {
    /// Full-attention chain: (parent engine hash, local tokens hash) -> engine hash.
    fa_child: FxHashMap<(Option<u64>, u64), u64>,
    /// Full-attention live blocks with per-copy refcount.
    fa_live: FxHashMap<u64, FaEntry>,
    /// Per SWA group: engine hash -> copies held.
    swa_live: FxHashMap<u32, FxHashMap<u64, u32>>,
    /// Per SWA group: sliding window in tokens (learned from events).
    swa_window: FxHashMap<u32, u32>,
}

impl WorkerState {
    fn fa_store(&mut self, parent: Option<u64>, blocks: &[(u64, LocalBlockHash)]) {
        let mut parent = parent;
        for &(hash, tokens) in blocks {
            match self.fa_live.get_mut(&hash) {
                Some(entry) => entry.copies += 1,
                None => {
                    self.fa_live.insert(
                        hash,
                        FaEntry {
                            parent,
                            tokens,
                            copies: 1,
                        },
                    );
                }
            }
            self.fa_child.insert((parent, tokens.0), hash);
            parent = Some(hash);
        }
    }

    /// Returns the hashes whose LAST copy was removed (the tree may drop them).
    fn fa_remove(&mut self, hashes: &[u64]) -> Vec<u64> {
        let mut gone = Vec::with_capacity(hashes.len());
        for &hash in hashes {
            let Some(entry) = self.fa_live.get_mut(&hash) else {
                // Unknown to us: let the tree decide (it treats it as a no-op).
                gone.push(hash);
                continue;
            };
            if entry.copies > 1 {
                entry.copies -= 1;
                continue;
            }
            let entry = self.fa_live.remove(&hash).expect("present");
            if self.fa_child.get(&(entry.parent, entry.tokens.0)) == Some(&hash) {
                self.fa_child.remove(&(entry.parent, entry.tokens.0));
            }
            gone.push(hash);
        }
        gone
    }

    fn swa_store(&mut self, group: u32, window: Option<u32>, hashes: &[u64]) {
        if let Some(window) = window {
            self.swa_window.insert(group, window);
        }
        let live = self.swa_live.entry(group).or_default();
        for &hash in hashes {
            *live.entry(hash).or_insert(0) += 1;
        }
    }

    fn swa_remove(&mut self, group: u32, hashes: &[u64]) {
        let Some(live) = self.swa_live.get_mut(&group) else {
            return;
        };
        for &hash in hashes {
            if let Some(copies) = live.get_mut(&hash) {
                if *copies > 1 {
                    *copies -= 1;
                } else {
                    live.remove(&hash);
                }
            }
        }
    }

    fn clear(&mut self) {
        *self = WorkerState::default();
    }

    /// Predicted engine hit in blocks for a probe of `num_tokens` tokens whose
    /// per-block local hashes are `local`.
    fn predict(&self, local: &[LocalBlockHash], num_tokens: usize, cfg: &HybridConfig) -> u32 {
        let bs = cfg.block_size as usize;
        let mut chain: Vec<u64> = Vec::with_capacity(local.len());
        let mut parent = None;
        for tokens in local {
            let Some(&hash) = self.fa_child.get(&(parent, tokens.0)) else {
                break;
            };
            if !self.fa_live.contains_key(&hash) {
                break;
            }
            chain.push(hash);
            parent = Some(hash);
        }
        let reusable = if num_tokens == 0 {
            0
        } else {
            (num_tokens - 1) / bs
        };
        let cap = chain.len().min(reusable);

        let mut k = cap;
        'search: while k > 0 {
            for (group, live) in &self.swa_live {
                let need = cfg.swa_need_blocks.unwrap_or_else(|| {
                    let window = self.swa_window.get(group).copied().unwrap_or(0);
                    window.saturating_sub(1).div_ceil(cfg.block_size) + cfg.mtp_drop_blocks
                }) as usize;
                let lo = k.saturating_sub(need);
                if !chain[lo..k].iter().all(|hash| live.contains_key(hash)) {
                    k -= 1;
                    continue 'search;
                }
            }
            break;
        }
        (k as u32).saturating_sub(cfg.mtp_drop_blocks)
    }
}

/// Process-global oracle, keyed by worker.
pub struct HybridOracle {
    cfg: HybridConfig,
    workers: RwLock<FxHashMap<WorkerWithDpRank, WorkerState>>,
}

static ORACLE: OnceLock<HybridOracle> = OnceLock::new();

/// Install the oracle (once, at startup). No-op if already installed.
pub fn install(cfg: HybridConfig) {
    let _ = ORACLE.set(HybridOracle::new(cfg));
}

/// The installed oracle, or `None` when `--hybrid-swa-hits` is off.
pub fn oracle() -> Option<&'static HybridOracle> {
    ORACLE.get()
}

impl HybridOracle {
    pub fn new(cfg: HybridConfig) -> Self {
        Self {
            cfg,
            workers: RwLock::new(FxHashMap::default()),
        }
    }

    pub fn config(&self) -> HybridConfig {
        self.cfg
    }

    /// Feed a main-attention event. For `Removed`, returns the event with
    /// still-held copies stripped, so the radix tree only forgets a block when
    /// its last copy is gone. Other events are returned unchanged.
    pub fn apply_main(&self, mut event: RouterEvent) -> Option<RouterEvent> {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let mut workers = self.workers.write();
        let state = workers.entry(worker).or_default();
        match &mut event.event.data {
            KvCacheEventData::Stored(data) => {
                let blocks: Vec<(u64, LocalBlockHash)> = data
                    .blocks
                    .iter()
                    .map(|b| (b.block_hash.0, b.tokens_hash))
                    .collect();
                state.fa_store(data.parent_hash.map(|h| h.0), &blocks);
            }
            KvCacheEventData::Removed(data) => {
                let hashes: Vec<u64> = data.block_hashes.iter().map(|h| h.0).collect();
                let gone = state.fa_remove(&hashes);
                if gone.is_empty() {
                    return None;
                }
                data.block_hashes = gone.into_iter().map(ExternalSequenceBlockHash).collect();
            }
            KvCacheEventData::Cleared => state.clear(),
        }
        Some(event)
    }

    /// Feed a sliding-window group event that the main path filters out.
    pub fn apply_swa(
        &self,
        worker: WorkerWithDpRank,
        group: u32,
        window: Option<u32>,
        stored: bool,
        hashes: &[u64],
    ) {
        let mut workers = self.workers.write();
        let state = workers.entry(worker).or_default();
        if stored {
            state.swa_store(group, window, hashes);
        } else {
            state.swa_remove(group, hashes);
        }
    }

    pub fn clear(&self, worker: WorkerWithDpRank) {
        if let Some(state) = self.workers.write().get_mut(&worker) {
            state.clear();
        }
    }

    pub fn remove_worker(&self, worker_id: u64) {
        self.workers.write().retain(|w, _| w.worker_id != worker_id);
    }

    /// Predicted engine hit, in blocks, for `worker`.
    pub fn predict(
        &self,
        worker: WorkerWithDpRank,
        local: &[LocalBlockHash],
        num_tokens: usize,
    ) -> Option<u32> {
        let workers = self.workers.read();
        workers
            .get(&worker)
            .map(|state| state.predict(local, num_tokens, &self.cfg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BS: u32 = 64;

    fn cfg() -> HybridConfig {
        HybridConfig {
            block_size: BS,
            mtp_drop_blocks: 1,
            swa_need_blocks: None,
        }
    }

    fn lh(i: u64) -> LocalBlockHash {
        LocalBlockHash(1000 + i)
    }

    /// Engine hash for chain position `i` of conversation `conv`.
    fn eh(conv: u64, i: u64) -> u64 {
        conv * 1_000_000 + i
    }

    fn store_chain(s: &mut WorkerState, conv: u64, n: u64) {
        let blocks: Vec<(u64, LocalBlockHash)> = (0..n).map(|i| (eh(conv, i), lh(i))).collect();
        s.fa_store(None, &blocks);
    }

    fn store_swa(s: &mut WorkerState, conv: u64, range: std::ops::Range<u64>) {
        let hashes: Vec<u64> = range.map(|i| eh(conv, i)).collect();
        for g in 0..4 {
            s.swa_store(g, Some(512), &hashes);
        }
    }

    fn probe(n_blocks: u64) -> (Vec<LocalBlockHash>, usize) {
        (
            (0..n_blocks).map(lh).collect(),
            (n_blocks as usize) * BS as usize,
        )
    }

    #[test]
    fn need_is_window_blocks_plus_mtp() {
        // ceil(511 / 64) + 1 = 9, the reference model's SWA_NEED.
        assert_eq!(511u32.div_ceil(BS) + 1, 9);
    }

    #[test]
    fn full_swa_coverage_predicts_chain_minus_mtp() {
        let mut s = WorkerState::default();
        store_chain(&mut s, 1, 40);
        store_swa(&mut s, 1, 0..40);
        let (local, n) = probe(40);
        // (n-1)/bs caps at 39, minus the MTP block = 38.
        assert_eq!(s.predict(&local, n, &cfg()), 38);
    }

    #[test]
    fn swa_junction_without_checkpoint_predicts_zero() {
        // DEEP-309 (b): shared scaffold, first reuse of a new junction. The
        // full-attention chain covers the scaffold, but SWA groups hold only
        // the original prompt's tail checkpoint, not the junction blocks.
        let mut s = WorkerState::default();
        store_chain(&mut s, 1, 60);
        store_swa(&mut s, 1, 51..60); // checkpoint at the original end only
        let (local, n) = probe(30); // diverges at block 30
        assert_eq!(s.predict(&local, n, &cfg()), 0);
    }

    #[test]
    fn continuation_stops_at_previous_prompt_checkpoint() {
        // DEEP-309 (a): turn N+1 extends turn N. The chain also covers turn N's
        // completion, but SWA was checkpointed only at turn N's prompt end.
        let mut s = WorkerState::default();
        store_chain(&mut s, 1, 50); // prompt(40) + completion(10)
        store_swa(&mut s, 1, 31..40); // window ending at the prompt end
        let (local, n) = probe(60);
        // Largest k with SWA holding [k-9, k) is 40, minus MTP = 39.
        assert_eq!(s.predict(&local, n, &cfg()), 39);
    }

    #[test]
    fn duplicate_copy_survives_first_removal() {
        // DEEP-309 eviction: the engine re-stored blocks it recomputed; evicting
        // the older copy must not forget a block the engine still holds.
        let mut s = WorkerState::default();
        store_chain(&mut s, 1, 20);
        store_chain(&mut s, 1, 20); // duplicate copies
        store_swa(&mut s, 1, 0..20);
        let gone = s.fa_remove(&[eh(1, 10)]);
        assert!(
            gone.is_empty(),
            "first removal of a duplicated block is held back"
        );
        let (local, n) = probe(20);
        assert_eq!(s.predict(&local, n, &cfg()), 18);
        let gone = s.fa_remove(&[eh(1, 10)]);
        assert_eq!(gone, vec![eh(1, 10)]);
        // Last copy gone: the chain now breaks at block 10 → k = 10, minus MTP.
        assert_eq!(s.predict(&local, n, &cfg()), 9);
    }

    #[test]
    fn no_swa_groups_behaves_like_capped_chain() {
        let mut s = WorkerState::default();
        store_chain(&mut s, 1, 10);
        let (local, n) = probe(10);
        assert_eq!(s.predict(&local, n, &cfg()), 8);
    }

    #[test]
    fn apply_main_holds_back_removal_of_duplicated_block() {
        use crate::protocols::{
            KvCacheEvent, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData, StorageTier,
        };
        let oracle = HybridOracle::new(cfg());
        let event = |event_id, data| RouterEvent {
            worker_id: 7,
            storage_tier: StorageTier::Device,
            event: KvCacheEvent {
                event_id,
                dp_rank: 0,
                data,
            },
        };
        let store = || {
            event(
                0,
                KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(42),
                        tokens_hash: lh(0),
                        mm_extra_info: None,
                    }],
                }),
            )
        };
        let remove = || {
            event(
                1,
                KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(42)],
                }),
            )
        };
        assert!(oracle.apply_main(store()).is_some());
        assert!(oracle.apply_main(store()).is_some());
        assert!(
            oracle.apply_main(remove()).is_none(),
            "one copy is still held"
        );
        let passed = oracle
            .apply_main(remove())
            .expect("last copy reaches the tree");
        match passed.event.data {
            KvCacheEventData::Removed(d) => {
                assert_eq!(d.block_hashes, vec![ExternalSequenceBlockHash(42)])
            }
            _ => panic!("expected Removed"),
        }
    }
}
