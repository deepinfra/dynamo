// SPDX-FileCopyrightText: Copyright (c) 2026 DeepInfra
// SPDX-License-Identifier: Apache-2.0

//! DeepInfra: make vLLM's per-copy stores and per-hash removes balance in the publisher.
//!
//! vLLM's block pool emits a `BlockStored` for every newly filled physical block, including a
//! second copy of a hash the same KV cache group already holds (two requests computing the same
//! prefix), but it announces `BlockRemoved` once per (group, hash), when the last copy is gone
//! (`BlockPool._emit_block_removed_events`). The publisher's refcounted dedup counts every
//! stored hash and forwards a remove only when the count drops to zero, so a hash that ever had
//! a duplicate copy never reaches zero: its remove is swallowed and the router keeps the block
//! forever. On MiMo-V2.6-Flash (two main-attention groups) ~7% of stores are such copies and the
//! frontend's KV index grew ~2M stale blocks an hour (27M tracked vs ~193k held on the GPUs).
//!
//! With `DI_KV_GROUP_REFCOUNT=1` this counts stores per (dp rank, tier, group, hash) and lists a
//! removed hash once per counted store, so the publisher's refcount reaches zero exactly when
//! every copy in every group is gone. Store events pass unchanged.

use rustc_hash::FxHashMap;

use super::types::RawKvEvent;
use crate::residency::StorageTier;
use dynamo_router_protocol::worker::DpRank;

pub(super) const GROUP_REFCOUNT_ENV: &str = "DI_KV_GROUP_REFCOUNT";

#[derive(Debug, Clone, Default)]
pub(super) struct GroupRefState {
    /// (dp rank, tier, group, hash) -> stores seen since the hash was last removed.
    stores: FxHashMap<(DpRank, StorageTier, u32, u64), u32>,
}

impl GroupRefState {
    pub(super) fn from_env() -> Option<Self> {
        match std::env::var(GROUP_REFCOUNT_ENV).ok().as_deref() {
            Some("1" | "true") => {
                tracing::info!("{GROUP_REFCOUNT_ENV}: balancing per-copy KV stores with per-hash removes");
                Some(Self::default())
            }
            _ => None,
        }
    }

    pub(super) fn apply(&mut self, raw: &mut RawKvEvent, dp_rank: DpRank, group: u32) {
        let tier = raw
            .medium()
            .and_then(StorageTier::from_kv_medium)
            .unwrap_or(StorageTier::Device);
        match raw {
            RawKvEvent::BlockStored { block_hashes, .. } => {
                for hash in block_hashes.iter() {
                    *self
                        .stores
                        .entry((dp_rank, tier, group, hash.into_u64()))
                        .or_insert(0) += 1;
                }
            }
            RawKvEvent::BlockRemoved { block_hashes, .. } => {
                let mut expanded = Vec::with_capacity(block_hashes.len());
                for hash in block_hashes.iter() {
                    // Unknown (stored before this listener started): pass once, as before.
                    let copies = self
                        .stores
                        .remove(&(dp_rank, tier, group, hash.into_u64()))
                        .unwrap_or(1)
                        .max(1);
                    for _ in 0..copies {
                        expanded.push(*hash);
                    }
                }
                *block_hashes = expanded;
            }
            RawKvEvent::AllBlocksCleared { .. } => {
                self.stores.retain(|(rank, ..), _| *rank != dp_rank);
            }
            RawKvEvent::Ignored => {}
        }
    }

    #[cfg(test)]
    pub(super) fn tracked(&self) -> usize {
        self.stores.len()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet, VecDeque};

    use super::*;
    use crate::zmq_wire::types::BlockHashValue;

    fn stored(hashes: &[u64], group: u32) -> RawKvEvent {
        RawKvEvent::BlockStored {
            block_hashes: hashes.iter().copied().map(BlockHashValue::Unsigned).collect(),
            parent_block_hash: None,
            token_ids: vec![],
            block_size: 16,
            medium: Some("GPU".to_string()),
            lora_name: None,
            cache_namespace: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: Some(group),
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality: None,
            ownership: None,
            session_id: None,
        }
    }

    fn removed(hashes: &[u64], group: u32) -> RawKvEvent {
        RawKvEvent::BlockRemoved {
            block_hashes: hashes.iter().copied().map(BlockHashValue::Unsigned).collect(),
            medium: Some("GPU".to_string()),
            group_idx: Some(group),
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality: None,
            ownership: None,
        }
    }

    fn hashes(event: &RawKvEvent) -> Vec<u64> {
        match event {
            RawKvEvent::BlockStored { block_hashes, .. }
            | RawKvEvent::BlockRemoved { block_hashes, .. } => {
                block_hashes.iter().map(|h| h.into_u64()).collect()
            }
            _ => vec![],
        }
    }

    /// The publisher's refcounted dedup (lib/llm kv_router/publisher/dedup.rs, RefCounted): every
    /// stored hash increments, a removed hash passes only when its count drops to zero, an
    /// untracked hash passes. Returns the hashes forwarded to the router.
    #[derive(Default)]
    struct PublisherRefcount(HashMap<u64, usize>);

    impl PublisherRefcount {
        fn store(&mut self, hs: &[u64]) {
            for h in hs {
                *self.0.entry(*h).or_insert(0) += 1;
            }
        }
        fn remove(&mut self, hs: &[u64]) -> Vec<u64> {
            hs.iter()
                .copied()
                .filter(|h| match self.0.get_mut(h) {
                    Some(c) => {
                        *c -= 1;
                        if *c == 0 {
                            self.0.remove(h);
                            true
                        } else {
                            false
                        }
                    }
                    None => true,
                })
                .collect()
        }
    }

    #[test]
    fn duplicate_copy_remove_is_expanded() {
        let mut refs = GroupRefState::default();
        let mut s = stored(&[7, 8], 0);
        refs.apply(&mut s, 0, 0);
        let mut s2 = stored(&[7], 0); // second physical copy of 7 in group 0
        refs.apply(&mut s2, 0, 0);
        assert_eq!(hashes(&s2), vec![7], "stores pass unchanged");
        let mut r = removed(&[7], 0); // vLLM: one remove when the last copy is gone
        refs.apply(&mut r, 0, 0);
        assert_eq!(hashes(&r), vec![7, 7]);
        let mut r8 = removed(&[8], 0);
        refs.apply(&mut r8, 0, 0);
        assert_eq!(hashes(&r8), vec![8]);
        assert_eq!(refs.tracked(), 0);
    }

    #[test]
    fn unknown_remove_passes_once_and_ranks_groups_are_separate() {
        let mut refs = GroupRefState::default();
        let mut r = removed(&[9], 0);
        refs.apply(&mut r, 0, 0);
        assert_eq!(hashes(&r), vec![9]);
        let mut a = stored(&[5], 0);
        refs.apply(&mut a, 0, 0);
        let mut b = stored(&[5], 1);
        refs.apply(&mut b, 0, 1);
        let mut c = stored(&[5], 0);
        refs.apply(&mut c, 1, 0); // other dp rank
        let mut r0 = removed(&[5], 0);
        refs.apply(&mut r0, 0, 0);
        assert_eq!(hashes(&r0), vec![5]);
        assert_eq!(refs.tracked(), 2);
        let mut cleared = RawKvEvent::AllBlocksCleared { ownership: None };
        refs.apply(&mut cleared, 0, 0);
        assert_eq!(refs.tracked(), 1, "clear drops only that dp rank");
    }

    /// vLLM block-pool semantics on two main-attention groups (MiMo-V2.6): each group stores
    /// every newly filled copy, evicts copies LRU, and announces a remove only when a hash's
    /// last copy in that group is gone. Returns (router-tracked, stale) after the run.
    fn simulate(fix: bool) -> (usize, usize) {
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut refs = GroupRefState::default();
        let mut publisher = PublisherRefcount::default();
        let mut router: HashSet<u64> = HashSet::new();
        let cap = [600usize, 580];
        let mut copies: [HashMap<u64, usize>; 2] = [HashMap::new(), HashMap::new()];
        let mut lru: [VecDeque<u64>; 2] = Default::default();
        for _ in 0..20_000 {
            let h = next() % 3_000;
            for g in 0..2u32 {
                let gi = g as usize;
                *copies[gi].entry(h).or_insert(0) += 1;
                lru[gi].push_back(h);
                let mut ev = stored(&[h], g);
                if fix {
                    refs.apply(&mut ev, 0, g);
                }
                publisher.store(&hashes(&ev));
                router.extend(hashes(&ev));
                while lru[gi].len() > cap[gi] {
                    let victim = lru[gi].pop_front().unwrap();
                    let c = copies[gi].get_mut(&victim).unwrap();
                    *c -= 1;
                    if *c == 0 {
                        copies[gi].remove(&victim);
                        let mut rm = removed(&[victim], g);
                        if fix {
                            refs.apply(&mut rm, 0, g);
                        }
                        for fwd in publisher.remove(&hashes(&rm)) {
                            router.remove(&fwd);
                        }
                    }
                }
            }
        }
        let held: HashSet<u64> = copies.iter().flat_map(|m| m.keys().copied()).collect();
        assert!(held.is_subset(&router), "the router must never lose a held block");
        (router.len(), router.difference(&held).count())
    }

    #[test]
    fn vllm_semantics_leak_without_fix_and_balance_with_it() {
        let (tracked, stale) = simulate(false);
        assert!(stale > tracked / 2, "expected the leak without the fix: {stale} of {tracked}");
        let (tracked, stale) = simulate(true);
        assert_eq!(stale, 0, "fix leaves {stale} stale of {tracked}");
    }
}
