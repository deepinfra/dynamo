// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-listener repair for device stores whose parent is only known in the
//! host (CPU) tier.
//!
//! With CPU offload, a pod's device-tier chain can reference a parent block the
//! device index never saw stored there (seeded from a TreeDump, reloaded from
//! the CPU tier, ...). The device radix tree rejects such a store with
//! `ParentBlockNotFound` and every later block of that sequence is rejected
//! too. The engine can only cache a device block whose parent is resident on
//! the device, so a device store is proof its ancestor chain is there: this
//! bridge re-publishes that chain from the host tier as a device store first.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, RouterEvent, StorageTier,
};

struct HostBlock {
    parent: Option<ExternalSequenceBlockHash>,
    block: KvCacheStoredBlockData,
}

/// Tracks one `(worker_id, dp_rank)`'s device and host block hashes, in event
/// order, so it can synthesize the missing device ancestors of a store.
#[derive(Default)]
pub struct TierBridge {
    device: FxHashSet<ExternalSequenceBlockHash>,
    host: FxHashMap<ExternalSequenceBlockHash, HostBlock>,
}

impl TierBridge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Forget everything, e.g. before a TreeDump replaces this worker's state.
    pub fn reset(&mut self) {
        self.device.clear();
        self.host.clear();
    }

    /// Record `event` and return a device store to apply BEFORE it, when
    /// `event` is a device store whose parent is only known in the host tier.
    pub fn observe(&mut self, event: &RouterEvent) -> Option<RouterEvent> {
        let promotion = match (&event.event.data, event.storage_tier) {
            (KvCacheEventData::Stored(store), StorageTier::Device) => store
                .parent_hash
                .and_then(|parent| self.promote_chain(parent))
                .map(|(anchor, blocks)| promoted_store(event, anchor, blocks)),
            _ => None,
        };
        if let Some(promoted) = &promotion {
            self.record(promoted);
        }
        self.record(event);
        promotion
    }

    fn record(&mut self, event: &RouterEvent) {
        match (&event.event.data, event.storage_tier) {
            (KvCacheEventData::Stored(store), StorageTier::Device) => {
                self.device
                    .extend(store.blocks.iter().map(|b| b.block_hash));
            }
            (KvCacheEventData::Stored(store), StorageTier::HostPinned) => {
                let mut parent = store.parent_hash;
                for block in &store.blocks {
                    self.host.insert(
                        block.block_hash,
                        HostBlock {
                            parent,
                            block: block.clone(),
                        },
                    );
                    parent = Some(block.block_hash);
                }
            }
            (KvCacheEventData::Removed(removed), StorageTier::Device) => {
                for hash in &removed.block_hashes {
                    self.device.remove(hash);
                }
            }
            (KvCacheEventData::Removed(removed), StorageTier::HostPinned) => {
                for hash in &removed.block_hashes {
                    self.host.remove(hash);
                }
            }
            (KvCacheEventData::Cleared, StorageTier::Device) => self.device.clear(),
            (KvCacheEventData::Cleared, StorageTier::HostPinned) => self.host.clear(),
            _ => {}
        }
    }

    /// Walk from `parent` up through host-only blocks until reaching a device
    /// block or the root. Returns the anchor and the chain root-first, or
    /// `None` when `parent` is already on the device or the chain is broken.
    fn promote_chain(
        &self,
        parent: ExternalSequenceBlockHash,
    ) -> Option<(
        Option<ExternalSequenceBlockHash>,
        Vec<KvCacheStoredBlockData>,
    )> {
        if self.device.contains(&parent) {
            return None;
        }
        let mut chain = Vec::new();
        let mut cursor = Some(parent);
        while let Some(hash) = cursor {
            if self.device.contains(&hash) {
                break;
            }
            let host_block = self.host.get(&hash)?;
            chain.push(host_block.block.clone());
            if chain.len() > self.host.len() {
                return None; // cycle guard
            }
            cursor = host_block.parent;
        }
        chain.reverse();
        Some((cursor, chain))
    }
}

fn promoted_store(
    source: &RouterEvent,
    anchor: Option<ExternalSequenceBlockHash>,
    blocks: Vec<KvCacheStoredBlockData>,
) -> RouterEvent {
    RouterEvent::with_storage_tier(
        source.worker_id,
        KvCacheEvent {
            event_id: source.event.event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: anchor,
                start_position: None,
                blocks,
            }),
            dp_rank: source.event.dp_rank,
        },
        StorageTier::Device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::{KvCacheRemoveData, LocalBlockHash};

    fn block(hash: u64) -> KvCacheStoredBlockData {
        KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(hash),
            tokens_hash: LocalBlockHash(hash + 1000),
            mm_extra_info: None,
        }
    }

    fn store(tier: StorageTier, parent: Option<u64>, hashes: &[u64]) -> RouterEvent {
        RouterEvent::with_storage_tier(
            7,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent.map(ExternalSequenceBlockHash),
                    start_position: None,
                    blocks: hashes.iter().copied().map(block).collect(),
                }),
                dp_rank: 0,
            },
            tier,
        )
    }

    fn remove(tier: StorageTier, hashes: &[u64]) -> RouterEvent {
        RouterEvent::with_storage_tier(
            7,
            KvCacheEvent {
                event_id: 2,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes
                        .iter()
                        .copied()
                        .map(ExternalSequenceBlockHash)
                        .collect(),
                }),
                dp_rank: 0,
            },
            tier,
        )
    }

    fn stored_hashes(event: &RouterEvent) -> (Option<u64>, Vec<u64>) {
        match &event.event.data {
            KvCacheEventData::Stored(s) => (
                s.parent_hash.map(|h| h.0),
                s.blocks.iter().map(|b| b.block_hash.0).collect(),
            ),
            other => panic!("expected store, got {other:?}"),
        }
    }

    #[test]
    fn device_store_with_host_only_parent_promotes_chain_to_device_anchor() {
        let mut bridge = TierBridge::new();
        assert!(
            bridge
                .observe(&store(StorageTier::Device, None, &[1]))
                .is_none()
        );
        assert!(
            bridge
                .observe(&store(StorageTier::HostPinned, Some(1), &[2, 3]))
                .is_none()
        );

        let promoted = bridge
            .observe(&store(StorageTier::Device, Some(3), &[4]))
            .expect("parent 3 is host-only");
        assert_eq!(promoted.storage_tier, StorageTier::Device);
        assert_eq!(stored_hashes(&promoted), (Some(1), vec![2, 3]));
        // Now on the device: a second child needs nothing.
        assert!(
            bridge
                .observe(&store(StorageTier::Device, Some(4), &[5]))
                .is_none()
        );
    }

    #[test]
    fn host_chain_reaching_the_root_is_promoted_from_the_root() {
        let mut bridge = TierBridge::new();
        bridge.observe(&store(StorageTier::HostPinned, None, &[1, 2]));
        let promoted = bridge
            .observe(&store(StorageTier::Device, Some(2), &[3]))
            .expect("chain is host-only up to the root");
        assert_eq!(stored_hashes(&promoted), (None, vec![1, 2]));
    }

    #[test]
    fn device_parent_or_unknown_parent_needs_no_promotion() {
        let mut bridge = TierBridge::new();
        bridge.observe(&store(StorageTier::Device, None, &[1]));
        bridge.observe(&store(StorageTier::HostPinned, Some(1), &[2]));
        assert!(
            bridge
                .observe(&store(StorageTier::Device, Some(1), &[9]))
                .is_none()
        );
        assert!(
            bridge
                .observe(&store(StorageTier::Device, Some(42), &[10]))
                .is_none()
        );
    }

    #[test]
    fn broken_host_chain_is_not_promoted() {
        let mut bridge = TierBridge::new();
        bridge.observe(&store(StorageTier::HostPinned, Some(99), &[2]));
        assert!(
            bridge
                .observe(&store(StorageTier::Device, Some(2), &[3]))
                .is_none()
        );
    }

    #[test]
    fn removals_and_reset_are_tracked_per_tier() {
        let mut bridge = TierBridge::new();
        bridge.observe(&store(StorageTier::Device, None, &[1]));
        bridge.observe(&store(StorageTier::HostPinned, Some(1), &[2]));
        bridge.observe(&remove(StorageTier::Device, &[1]));
        // 1 left the device but is not in the host tier: chain is broken.
        assert!(
            bridge
                .observe(&store(StorageTier::Device, Some(2), &[3]))
                .is_none()
        );

        bridge.reset();
        bridge.observe(&store(StorageTier::HostPinned, None, &[5]));
        bridge.observe(&remove(StorageTier::HostPinned, &[5]));
        assert!(
            bridge
                .observe(&store(StorageTier::Device, Some(5), &[6]))
                .is_none()
        );
    }
}
